"""Fused FP32 attention logits: optional LayerNorm -> activation -> dropout -> dot.

Forward is Triton; backward recomputes with differentiable PyTorch operations,
including second derivatives. This is NOT a fused-backward implementation.
The module adapter falls back to the original PyTorch path for non-FP32/AMP.
Run: python fused_attention_alpha.py  (requires CUDA, torch, triton).
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X, W, G, B, SEED, MASK, Y,
             H: tl.constexpr, C: tl.constexpr,
             LN: tl.constexpr, HAS_G: tl.constexpr, HAS_B: tl.constexpr,
             EPS: tl.constexpr, SLOPE: tl.constexpr, SILU: tl.constexpr,
             P: tl.constexpr, SAVE_MASK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    valid = c < C
    off = row * C + c
    x = tl.load(X + off, valid, other=0)
    z = x
    if LN:
        mean = tl.sum(x, 0) / C
        centered = tl.where(valid, x - mean, 0.0)
        var = tl.sum(centered * centered, 0) / C
        z = (x - mean) * tl.rsqrt(var + EPS)
        if HAS_G:
            z = z * tl.load(G + c, valid, other=1)
        if HAS_B:
            z = z + tl.load(B + c, valid, other=0)
    s = tl.sigmoid(z)
    if SILU:
        a = z * s
    else:
        # Preserve the supplied expression's operation order as far as possible.
        a = ((1.0 + SLOPE) / 2.0) * z + ((1.0 - SLOPE) / 2.0) * z * (2.0 * s - 1.0)
    if P > 0.0:
        if P == 1.0:
            keep = tl.full((BLOCK,), False, tl.int1)
            a = a * 0.0
        else:
            seed = tl.load(SEED).to(tl.uint32)
            keep = tl.rand(seed, off.to(tl.uint32)) >= P
            a = a * tl.where(keep, 1.0 / (1.0 - P), 0.0)
        if SAVE_MASK:
            tl.store(MASK + off, keep, valid)
    w = tl.load(W + (row % H) * C + c, valid, other=0)
    result = tl.sum(tl.where(valid, a * w, 0.0), 0)
    tl.store(Y + row, result)


def _reference(x, w, gamma, beta, ln, eps, silu, slope, p, mask):
    z = x
    if ln:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        z = centered * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)
        if gamma is not None:
            z = z * gamma
        if beta is not None:
            z = z + beta
    s = torch.sigmoid(z)
    a = z * s if silu else ((1 + slope) / 2) * z + ((1 - slope) / 2) * z * (2 * s - 1)
    if p > 0:
        a = a * (mask.to(x.dtype) / (1 - p) if p < 1 else 0.0)
    return (a * w).sum(-1)


class _FusedAlpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, gamma, beta, ln, eps, silu, slope, p, seed):
        n, h, c = x.shape
        y = torch.empty((n, h), device=x.device, dtype=x.dtype)
        save_mask = p > 0 and any(ctx.needs_input_grad[:4])
        mask = torch.empty(x.shape if save_mask else (0,), device=x.device, dtype=torch.bool)
        if n:
            with torch.cuda.device(x.device):
                _forward[(n * h,)](
                    x, w, gamma if gamma is not None else x,
                    beta if beta is not None else x,
                    seed if seed is not None else x, mask, y,
                    h, c, ln, gamma is not None, beta is not None,
                    eps, slope, silu, p, save_mask, triton.next_power_of_2(c),
                    num_warps=4 if c <= 1024 else 8, enable_fp_fusion=False)
        ctx.save_for_backward(x, w, gamma, beta, mask)
        ctx.config = ln, eps, silu, slope, p
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w, gamma, beta, mask = ctx.saved_tensors
        higher_order = torch.is_grad_enabled()
        inputs = (x, w, gamma, beta)
        required = [i for i, t in enumerate(inputs) if t is not None and ctx.needs_input_grad[i]]
        with torch.enable_grad():
            # Use saved original inputs, without detach, so double backward can
            # differentiate through x, dot weights, and LayerNorm parameters.
            y = _reference(x, w, gamma, beta, *ctx.config, mask)
            grads = torch.autograd.grad(y, [inputs[i] for i in required], grad_y,
                                        create_graph=higher_order)
        result = [None] * 10
        for i, g in zip(required, grads):
            result[i] = g
        return tuple(result)


def fused_attention_alpha(x, alpha_dot, norm_weight=None, norm_bias=None, *,
                          use_layer_norm=True, eps=1e-5, activation='silu',
                          negative_slope=0.2, dropout_p=0.0, training=True,
                          seed=None):
    """FP32 CUDA [E,H,C] -> [E,H].

    activation is independent of training: eval disables dropout, not SiLU.
    Seed, if supplied, must be an immutable scalar CUDA int64/int32 tensor.
    Default dropout uses one additional torch RNG launch for the device seed.
    A boolean [E,H,C] mask is saved only for active dropout with gradients.
    Noncontiguous tensors are copied; account for that cost in benchmarks.
    """
    if x.ndim != 3 or not x.is_cuda or x.dtype != torch.float32:
        raise ValueError('fast path requires FP32 CUDA x with shape [E,H,C]')
    n, h, c = x.shape
    if not 1 <= c <= 8192 or h < 1 or x.numel() > 2**32:
        raise ValueError('requires H >= 1, 1 <= C <= 8192, numel <= 2**32')
    for name, t, shape in [('alpha_dot', alpha_dot, (h, c)),
                            ('norm_weight', norm_weight, (c,)),
                            ('norm_bias', norm_bias, (c,))]:
        if t is not None and (tuple(t.shape) != shape or t.device != x.device or t.dtype != x.dtype):
            raise ValueError(f'{name} must have shape {shape}, same dtype/device as x')
    if activation not in ('silu', 'smooth_leaky_relu'):
        raise ValueError('unsupported activation')
    p = float(dropout_p)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('dropout_p must be in [0,1]')
    if not math.isfinite(eps) or eps <= 0 or not math.isfinite(negative_slope):
        raise ValueError('eps must be positive; negative_slope must be finite')
    if not use_layer_norm and (norm_weight is not None or norm_bias is not None):
        raise ValueError('norm parameters supplied while LayerNorm is disabled')
    p = p if training else 0.0
    if 0 < p < 1:
        if seed is None:
            seed = torch.randint(0, 2**31, (), device=x.device, dtype=torch.int64)
        if seed.device != x.device or seed.numel() != 1 or seed.dtype not in (torch.int32, torch.int64):
            raise ValueError('seed must be a one-element CUDA integer tensor on x.device')
    return _FusedAlpha.apply(x.contiguous(), alpha_dot.contiguous(),
                            norm_weight.contiguous() if norm_weight is not None else None,
                            norm_bias.contiguous() if norm_bias is not None else None,
                            use_layer_norm, float(eps), activation == 'silu',
                            float(negative_slope), p, seed)


def fused_atten_alpha(x, alpha_dot, alpha_norm, alpha_act, alpha_dropout):
    """Drop-in adapter; x must already be viewed as [E,H,C].

    Reads the ACTUAL modules, including dropout.training, norm.eps, and slope.
    Uses original PyTorch for AMP, non-FP32, unsupported shapes or modules.
    """
    is_ln = isinstance(alpha_norm, torch.nn.LayerNorm)
    is_silu = isinstance(alpha_act, torch.nn.SiLU) and not alpha_act.inplace
    is_smooth = type(alpha_act).__name__ == 'SmoothLeakyReLU' and hasattr(alpha_act, 'alpha')
    norm_ok = isinstance(alpha_norm, torch.nn.Identity) or (
        is_ln and tuple(alpha_norm.normalized_shape) == (x.shape[-1],))
    drop_ok = isinstance(alpha_dropout, (torch.nn.Dropout, torch.nn.Identity))
    params = [alpha_dot] + ([alpha_norm.weight, alpha_norm.bias] if is_ln else [])
    supported = (x.is_cuda and x.dtype == torch.float32 and not torch.is_autocast_enabled()
                 and x.ndim == 3 and 0 < x.shape[1] and 1 <= x.shape[-1] <= 8192
                 and x.numel() <= 2**32 and norm_ok and drop_ok and (is_silu or is_smooth)
                 and all(t is None or t.dtype == x.dtype for t in params))
    if not supported:
        z = alpha_dropout(alpha_act(alpha_norm(x)))
        return torch.einsum('bik,ik->bi', z, alpha_dot)
    return fused_attention_alpha(
        x, alpha_dot, alpha_norm.weight if is_ln else None,
        alpha_norm.bias if is_ln else None, use_layer_norm=is_ln,
        eps=alpha_norm.eps if is_ln else 1e-5,
        activation='silu' if is_silu else 'smooth_leaky_relu',
        negative_slope=float(alpha_act.alpha) if is_smooth else 0.2,
        dropout_p=alpha_dropout.p if isinstance(alpha_dropout, torch.nn.Dropout) else 0.0,
        training=alpha_dropout.training)


def self_test():
    if not torch.cuda.is_available():
        raise RuntimeError('Tests require CUDA')
    for ln in (False, True):
        for activation, p in (('silu', 0.0), ('silu', 0.25), ('silu', 1.0), ('smooth_leaky_relu', 0.0)):
            for c in (1, 13, 64):
                x = torch.randn(9, 3, c, device='cuda', requires_grad=True)
                w = torch.randn(3, c, device='cuda', requires_grad=True)
                g = torch.randn(c, device='cuda', requires_grad=True) if ln else None
                b = torch.randn(c, device='cuda', requires_grad=True) if ln else None
                seed = torch.tensor(123, device='cuda', dtype=torch.int64)
                # Capture the kernel's mask to compare identical dropout samples.
                with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
                    y = fused_attention_alpha(x, w, g, b, use_layer_norm=ln,
                                              activation=activation, dropout_p=p, seed=seed)
                mask = y.grad_fn.saved_tensors[-1]
                ref = _reference(x, w, g, b, ln, 1e-5, activation == 'silu', 0.2, p, mask)
                torch.testing.assert_close(y, ref, atol=3e-4, rtol=3e-4)
                inputs = [t for t in (x, w, g, b) if t is not None]
                upstream = torch.randn_like(y, requires_grad=True)
                actual = torch.autograd.grad(y, inputs, upstream, create_graph=True)
                expected = torch.autograd.grad(ref, inputs, upstream, create_graph=True)
                vectors = [torch.randn_like(t) for t in actual]
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e)
                a2 = torch.autograd.grad(sum((a*v).sum() for a,v in zip(actual,vectors)), inputs + [upstream], allow_unused=True)
                e2 = torch.autograd.grad(sum((e*v).sum() for e,v in zip(expected,vectors)), inputs + [upstream], allow_unused=True)
                for a, e in zip(a2, e2):
                    if a is None or e is None:
                        assert a is None and e is None
                    else:
                        torch.testing.assert_close(a, e)
    print('CUDA forward, backward and double-backward checks passed.')


if __name__ == '__main__':
    self_test()
