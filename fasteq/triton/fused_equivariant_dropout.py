"""Fused equivariant dropout. Run this file on a CUDA/Triton host to test.

Forward and backward each use one Triton kernel, plus one torch RNG launch
per forward to obtain a device seed (unless a seed tensor is supplied).
Random masks are distributionally, not bitwise, equivalent to torch Dropout.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _dropout_kernel(X, INDEX, SEED, Y, TOTAL: tl.constexpr,
                    K: tl.constexpr, C: tl.constexpr, L: tl.constexpr,
                    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                    P: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < TOTAL
    c = i % C
    k = (i // C) % K
    n = i // (K * C)
    if P == 1.0:
        result = tl.full((BLOCK,), 0, tl.float32)
    else:
        ell = tl.load(INDEX + k, valid, other=0).to(tl.int64)
        # Identical offset for all components sharing one mask.
        offset = ((n * L + ell) * C + c).to(tl.uint32)
        seed = tl.load(SEED).to(tl.uint32)
        keep = tl.rand(seed, offset) >= P
        value = tl.load(X + n * S0 + k * S1 + c * S2, valid, other=0)
        # Match the reference's mask rounding before multiplying x.
        scale = tl.full((BLOCK,), 1.0 / (1.0 - P), tl.float64
                        if X.dtype.element_ty == tl.float64 else tl.float32)
        mask = tl.where(keep, scale, 0.0).to(X.dtype.element_ty)
        if X.dtype.element_ty == tl.float64:
            result = value * mask
        else:
            result = value.to(tl.float32) * mask.to(tl.float32)
    tl.store(Y + i, result, valid)


def _launch(x, index, seed, levels, p):
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.numel():
        with torch.cuda.device(x.device):
            _dropout_kernel[(triton.cdiv(x.numel(), 256),)](
                x, index, seed, out, x.numel(), x.shape[1], x.shape[2],
                levels, *x.stride(), p, 256, num_warps=4)
    return out


class _ApplyMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index, seed, levels, p):
        ctx.save_for_backward(index, seed)
        ctx.levels, ctx.p = levels, p
        return _launch(x, index, seed, levels, p)

    @staticmethod
    def backward(ctx, grad_out):
        index, seed = ctx.saved_tensors
        # Reapply the same linear map. Autograd records this for double backward.
        gx = _ApplyMask.apply(grad_out, index, seed, ctx.levels, ctx.p)
        return gx, None, None, None, None


def fused_equivariant_dropout(x, expand_index, lmax, drop_prob=0.0,
                               training=True, *, seed=None):
    """x: [N,K,C]; index: [K] contiguous int32/int64 on x's device.

    index entries must be in [0,lmax]. This is guaranteed by the module below.
    Optional seed: immutable contiguous one-element int64/int32 CUDA tensor.
    Supplying seed avoids the additional RNG launch; reuse gives identical masks.
    Do not mutate seed or expand_index before backward completes.
    """
    p = float(drop_prob)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError('drop_prob must be in [0,1]')
    if not training or p == 0.0:
        return x
    if x.ndim != 3 or not x.is_cuda:
        raise ValueError('x must be a CUDA tensor of shape [N,K,C]')
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError('unsupported x dtype')
    if (expand_index.device != x.device or expand_index.ndim != 1
            or expand_index.numel() != x.shape[1]
            or not expand_index.is_contiguous()
            or expand_index.dtype not in (torch.int32, torch.int64)):
        raise ValueError('expand_index must be contiguous [K] integer indices on x.device')
    levels = int(lmax) + 1
    if levels <= 0 or x.shape[0] * levels * x.shape[2] > 2**32:
        raise ValueError('invalid lmax or random index range exceeds uint32')
    if seed is None:
        # GPU RNG participates in torch.manual_seed/checkpoint RNG restoration.
        # No .item(), host synchronization, or CPU RNG.
        seed = (torch.zeros((), device=x.device, dtype=torch.int64) if p == 1.0
                else torch.randint(0, 2**31, (), device=x.device, dtype=torch.int64))
    if (seed.device != x.device or seed.numel() != 1
            or seed.dtype not in (torch.int32, torch.int64)
            or not seed.is_contiguous()):
        raise ValueError('seed must be a one-element integer tensor on x.device')
    return _ApplyMask.apply(x, expand_index, seed, levels, p)


class EquivariantDropout(torch.nn.Module):
    def __init__(self, lmax, mmax, drop_prob, use_m_primary=False, *,
                 correct_m_primary=False):
        super().__init__()
        if not (isinstance(lmax, int) and isinstance(mmax, int)
                and 0 <= mmax <= lmax):
            raise ValueError('require integer 0 <= mmax <= lmax')
        if not math.isfinite(float(drop_prob)) or not 0 <= drop_prob <= 1:
            raise ValueError('drop_prob must be in [0,1]')
        self.lmax, self.mmax = lmax, mmax
        self.drop_prob = float(drop_prob)
        self.use_m_primary = use_m_primary
        self.correct_m_primary = correct_m_primary
        indices = []
        if not use_m_primary:
            for ell in range(lmax + 1):
                indices.extend([ell] * (2 * min(ell, mmax) + 1))
        else:
            for m in range(mmax + 1):
                block = list(range(m, lmax + 1)) if correct_m_primary else list(range(lmax + 1 - m))
                indices.extend(block)
                if m > 0:
                    indices.extend(block)
        self.register_buffer('expand_index', torch.tensor(indices, dtype=torch.long))

    def forward(self, x, *, seed=None):
        return fused_equivariant_dropout(x, self.expand_index, self.lmax,
                                         self.drop_prob, self.training, seed=seed)

    def extra_repr(self):
        return (f'lmax={self.lmax}, mmax={self.mmax}, drop_prob={self.drop_prob}, '
                f'use_m_primary={self.use_m_primary}, '
                f'correct_m_primary={self.correct_m_primary}')


def self_test():
    """Tests mask grouping, strided inputs, gradients, double backward, endpoints.

    Reference uses the SAME sampled mask; torch Dropout has a different RNG.
    These tests require a CUDA GPU and are not a performance benchmark.
    """
    assert torch.cuda.is_available(), 'CUDA GPU required'
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        for primary, corrected in ((False, False), (True, False), (True, True)):
            mod = EquivariantDropout(3, 2, 0.3, primary,
                                     correct_m_primary=corrected).cuda()
            k = mod.expand_index.numel()
            x = torch.randn(7, k, 22, device='cuda', dtype=dtype)[:, :, ::2].requires_grad_()
            seed = torch.tensor(12345, device='cuda', dtype=torch.int64)
            mask = mod(torch.ones_like(x), seed=seed).detach()
            for ell in range(4):
                group = mask[:, mod.expand_index == ell, :]
                if group.shape[1]:
                    assert torch.equal(group, group[:, :1, :].expand_as(group))
            y = mod(x, seed=seed)
            torch.testing.assert_close(y, x * mask, rtol=0, atol=0)
            g = torch.randn_like(y, requires_grad=True)
            gx, = torch.autograd.grad(y, x, g, create_graph=True)
            torch.testing.assert_close(gx, g * mask, rtol=0, atol=0)
            h = torch.randn_like(gx)
            gg, = torch.autograd.grad(gx, g, h)
            torch.testing.assert_close(gg, h * mask, rtol=0, atol=0)
            torch.manual_seed(42)
            a = mod(x)
            torch.manual_seed(42)
            torch.testing.assert_close(a, mod(x), rtol=0, atol=0)
            mod.eval()
            assert mod(x) is x
            mod.train()
            mod.drop_prob = 1.0
            assert torch.count_nonzero(mod(x)) == 0
            mod.drop_prob = 0.0
            assert mod(x) is x
    print('All CUDA correctness tests passed.')


if __name__ == '__main__':
    self_test()
