"""SO(2), m > 0: fused complex GEMM, without a materialized block weight.

x: [E, 2, I], weight: [2*O, I] = cat([W1,W2], 0).
y_real = x_real @ W1.T - x_imag @ W2.T
y_imag = x_real @ W2.T + x_imag @ W1.T

FP16/BF16/FP32 CUDA tensors; FP32 accumulation. NVIDIA default: tf32x3.
First backward uses Triton, including split-K weight-gradient reduction.
create_graph=True uses differentiable PyTorch backward to support higher orders.
No weight cache. Autocast is handled by differentiable input casts.
GPU compilation, numerical validation and speed are NOT verified by the author
in the generation environment. Run this file on your GPU to validate/benchmark.

Usage inside your existing class (keep its __init__ and fc unchanged):
    y = so2m_linear(x_m, self.fc.weight)
    return y.reshape(x_m.shape[0], 2, self.num_l,
                     self.m_output_channels).unbind(1)

Run: python so2m_linear_triton.py --edges 7506 --in-size 384 --out-size 384
"""
import argparse
import torch
import triton
import triton.language as tl


def _configs():
    return [triton.Config({'BM': m, 'BN': n, 'BK': k}, num_warps=w,
                         num_stages=s)
            for m, n, k, w, s in [(32, 64, 32, 4, 3), (64, 32, 32, 4, 3),
                                  (32, 64, 64, 4, 3), (64, 64, 32, 4, 3),
                                  (32, 128, 32, 4, 3)]]


@triton.autotune(configs=_configs(), key=['M', 'N', 'K', 'MODE', 'SPLIT', 'PREC'])
@triton.jit
def _complex_mm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                AM: tl.constexpr, AK: tl.constexpr, AC: tl.constexpr,
                BKSTR: tl.constexpr, BNSTR: tl.constexpr, BC: tl.constexpr,
                CM: tl.constexpr, CN: tl.constexpr, CC: tl.constexpr,
                MODE: tl.constexpr, SPLIT: tl.constexpr, PREC: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # MODE=0: forward; MODE=1: conjugated product used by dx and dw.
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    CHUNK: tl.constexpr = tl.cdiv(K, SPLIT * BK) * BK
    rk = part * CHUNK + tl.arange(0, BK)
    c0 = tl.zeros((BM, BN), tl.float32)
    c1 = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(CHUNK, BK)):
        kk = rk + step * BK
        ap = A + rm[:, None] * AM + kk[None, :] * AK
        bp = B + kk[:, None] * BKSTR + rn[None, :] * BNSTR
        ma = (rm[:, None] < M) & (kk[None, :] < K)
        mb = (kk[:, None] < K) & (rn[None, :] < N)
        a0 = tl.load(ap, ma, other=0)
        a1 = tl.load(ap + AC, ma, other=0)
        b0 = tl.load(bp, mb, other=0)
        b1 = tl.load(bp + BC, mb, other=0)
        c0 = tl.dot(a0, b0, c0, input_precision=PREC)
        if MODE == 0:
            c0 = tl.dot(a1, -b1, c0, input_precision=PREC)
            c1 = tl.dot(a0, b1, c1, input_precision=PREC)
        else:
            c0 = tl.dot(a1, b1, c0, input_precision=PREC)
            c1 = tl.dot(a0, -b1, c1, input_precision=PREC)
        c1 = tl.dot(a1, b0, c1, input_precision=PREC)
    cp = C + part * (2 * M * N) + rm[:, None] * CM + rn[None, :] * CN
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(cp, c0, mask)
    tl.store(cp + CC, c1, mask)


@triton.jit
def _sum_parts(T, W, SIZE: tl.constexpr, PARTS: tl.constexpr,
               R: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = tl.arange(0, R)
    v = tl.load(T + r[:, None] * SIZE + i[None, :],
                (r[:, None] < PARTS) & (i[None, :] < SIZE), other=0)
    tl.store(W + i, tl.sum(v, axis=0), i < SIZE)


def _mm(a, b, out, m, n, k, astrides, bstrides, cstrides,
        mode, precision, split=1):
    grid = lambda meta: (triton.cdiv(m, meta['BM']),
                         triton.cdiv(n, meta['BN']), split)
    with torch.cuda.device(a.device):
        _complex_mm[grid](a, b, out, m, n, k, *astrides, *bstrides, *cstrides,
                          mode, split, precision)


def reference(x, weight):
    """Differentiable block-matrix reference matching the supplied code."""
    w1, w2 = weight.chunk(2, 0)
    block = torch.cat((torch.cat((w1, -w2), 1),
                       torch.cat((w2, w1), 1)), 0)
    return (x.flatten(1) @ block.T).reshape(x.shape[0], 2, w1.shape[0])


class _SO2Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, precision):
        e, _, i = x.shape
        o = w.shape[0] // 2
        y = torch.empty((e, 2, o), device=x.device, dtype=x.dtype)
        if e:
            _mm(x, w, y, e, o, i,
                (x.stride(0), x.stride(2), x.stride(1)),
                (w.stride(1), w.stride(0), o*w.stride(0)),
                (2*o, 1, o), 0, precision)
        ctx.save_for_backward(x, w)
        ctx.precision = precision
        return y

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        e, _, i = x.shape
        o = w.shape[0] // 2
        nx, nw = ctx.needs_input_grad[:2]
        dx = dw = None
        # Building the derivative graph must not run opaque Triton backward.
        if torch.is_grad_enabled():
            with torch.autocast(device_type='cuda', enabled=False):
                gr, gi = g.unbind(1)
                xr, xi = x.unbind(1)
                w1, w2 = w.chunk(2, 0)
                if nx:
                    dx = torch.stack((gr @ w1 + gi @ w2,
                                      gi @ w1 - gr @ w2), 1)
                if nw:
                    dw = torch.cat((gr.T @ xr + gi.T @ xi,
                                    gi.T @ xr - gr.T @ xi), 0)
            return dx, dw, None
        if not e:
            return (torch.zeros_like(x) if nx else None,
                    torch.zeros_like(w) if nw else None, None)
        if nx:
            dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
            _mm(g, w, dx, e, i, o,
                (g.stride(0), g.stride(2), g.stride(1)),
                (w.stride(0), w.stride(1), o*w.stride(0)),
                (2*i, 1, i), 1, ctx.precision)
        if nw:
            parts = min(32, triton.cdiv(e, 1024))
            dw = torch.empty(w.shape, device=w.device, dtype=w.dtype)
            tmp = (torch.empty((parts, 2*o, i), device=w.device,
                               dtype=torch.float32) if parts > 1 else dw)
            _mm(g, x, tmp, o, i, e,
                (g.stride(2), g.stride(0), g.stride(1)),
                (x.stride(0), x.stride(2), x.stride(1)),
                (i, 1, o*i), 1, ctx.precision, parts)
            if parts > 1:
                with torch.cuda.device(w.device):
                    _sum_parts[(triton.cdiv(2*o*i, 128),)](
                        tmp, dw, 2*o*i, parts, triton.next_power_of_2(parts),
                        128, num_warps=4)
        return dx, dw, None


def so2m_linear(x_m, weight, *, precision=None):
    """Return [E,2,O]. Supports strided inputs and live trainable weights.

    precision: 'tf32x3' (NVIDIA default), 'ieee' (AMD default), or 'tf32'.
    No FP64 support. Autotuning occurs at first use per signature; warm up
    outside CUDA graph capture and before timing. Higher-order backward uses
    PyTorch, not Triton. No bitwise equivalence or speedup guarantee.
    """
    if x_m.ndim != 3 or x_m.shape[1] != 2:
        raise ValueError('x_m must have shape [E, 2, I]')
    if weight.ndim != 2 or weight.shape[0] % 2 or weight.shape[1] != x_m.shape[2]:
        raise ValueError('weight must have shape [2*O, I]')
    if weight.shape[0] == 0 or weight.shape[1] == 0:
        raise ValueError('I and O must be positive')
    if not x_m.is_cuda or x_m.device != weight.device:
        raise ValueError('x_m and weight must be on the same CUDA/HIP device')
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if x_m.dtype not in supported or weight.dtype not in supported:
        raise TypeError('only float16, bfloat16, float32 are supported')
    if torch.is_autocast_enabled('cuda'):
        dtype = torch.get_autocast_dtype('cuda')
        x_m, weight = x_m.to(dtype), weight.to(dtype)
    if x_m.dtype != weight.dtype:
        raise TypeError('outside autocast, x_m and weight must have the same dtype')
    if precision is None:
        precision = 'ieee' if torch.version.hip else 'tf32x3'
    if precision not in ('ieee', 'tf32x3', 'tf32'):
        raise ValueError('precision must be ieee, tf32x3 or tf32')
    if torch.version.hip and precision != 'ieee':
        raise ValueError('this implementation exposes only ieee precision on HIP')
    return _SO2Linear.apply(x_m, weight, precision)


def _check(e, i, o, dtype, precision, strided=False):
    # Test against a float64 block-GEMM oracle, using quantized input values.
    x = torch.randn(e, 2, i*(2 if strided else 1), device='cuda', dtype=dtype)
    if strided:
        x = x[..., ::2]
    x = x.detach().requires_grad_()
    w = (torch.randn(2*o, i, device='cuda', dtype=dtype) / i**0.5).requires_grad_()
    g = torch.randn(e, 2, o, device='cuda', dtype=dtype)
    y = so2m_linear(x, w, precision=precision)
    dx, dw = torch.autograd.grad(y, (x, w), g)
    xx, ww = x.detach().double().requires_grad_(), w.detach().double().requires_grad_()
    yy = reference(xx, ww)
    rx, rw = torch.autograd.grad(yy, (xx, ww), g.double())
    atol, rtol = {torch.float32: (3e-4, 3e-4),
                  torch.float16: (3e-3, 3e-3),
                  torch.bfloat16: (3e-2, 3e-2)}[dtype]
    for label, got, ref in [('y', y, yy), ('dx', dx, rx), ('dw', dw, rw)]:
        ref = ref.to(dtype)
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
        err = (got.float()-ref.float()).abs().max().item() if got.numel() else 0.
        print(f'  {label}: max_abs={err:.6g}')
    # Mixed derivative / Hessian-vector product including dependence on y.
    if dtype == torch.float32 and e:
        probes = (torch.randn_like(x), torch.randn_like(w))
        def hvp(fn, a, b):
            z = fn(a, b)
            first = torch.autograd.grad(z.square().sum()/2, (a, b), create_graph=True)
            return torch.autograd.grad(sum((v*p).sum() for v, p in zip(first, probes)), (a, b))
        h = hvp(lambda a, b: so2m_linear(a, b, precision=precision), x, w)
        r = hvp(reference, x.detach().requires_grad_(), w.detach().requires_grad_())
        for got, ref in zip(h, r):
            torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-3)
    print(f'PASS E={e} I={i} O={o} {dtype} strided={strided}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edges', type=int, default=7506)
    p.add_argument('--in-size', type=int, default=384)
    p.add_argument('--out-size', type=int, default=384)
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    p.add_argument('--precision', choices=['ieee', 'tf32x3'], default=None)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA/HIP GPU and Triton are required')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    for shape in [(1, 1, 1), (37, 45, 29), (0, 16, 32), (2053, 33, 47)]:
        _check(*shape, dtype, args.precision, strided=True)
    e, i, o = args.edges, args.in_size, args.out_size
    _check(e, i, o, dtype, args.precision)
    x = torch.randn(e, 2, i, device='cuda', dtype=dtype, requires_grad=True)
    w = torch.randn(2*o, i, device='cuda', dtype=dtype, requires_grad=True)
    g = torch.randn(e, 2, o, device='cuda', dtype=dtype)
    with torch.no_grad():
        w1, w2 = w.chunk(2, 0)
        block = torch.cat((torch.cat((w1, -w2), 1), torch.cat((w2, w1), 1)), 0)
    fused = lambda: so2m_linear(x, w, precision=args.precision)
    cached = lambda: (x.flatten(1) @ block.T).reshape(e, 2, o)
    train_ref = lambda: reference(x, w)
    # All first-use compilation and autotuning outside measurement.
    for fn in (fused, cached, train_ref):
        fn()
    def backward(fn):
        return torch.autograd.grad(fn(), (x, w), g)
    backward(fused)
    backward(train_ref)
    import triton.testing
    with torch.no_grad():
        ft = triton.testing.do_bench(fused)
        bt = triton.testing.do_bench(cached)
    print(f'forward: Triton={ft:.4f} ms, cached block GEMM={bt:.4f} ms, speedup={bt/ft:.3f}x')
    tf = triton.testing.do_bench(lambda: backward(fused))
    tb = triton.testing.do_bench(lambda: backward(train_ref))
    print(f'forward+backward: Triton={tf:.4f} ms, differentiable block={tb:.4f} ms, speedup={tb/tf:.3f}x')
    print('FP32 block baseline uses TF32 disabled; choose --precision ieee for stricter comparison.')


if __name__ == '__main__':
    main()
