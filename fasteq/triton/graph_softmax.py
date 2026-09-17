"""
In experimental/models/equiformer_v3/transformer_block.py:
replace :
self.attn_softmax = GraphSoftmax(
            eps=self.eps,
            exp_dropout=attn_mask_rate, 
            softcap=self.softcap
        )
to:
self.attn_softmax = FusedGraphSoftmax(
            eps=self.eps,
            exp_dropout=attn_mask_rate, 
            softcap=self.softcap
        )
"""
import argparse
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes


# Tests are defined before the implementation; run on a CUDA machine.
def check():
    torch.manual_seed(41)
    device = 'cuda'
    for E, N, H in [(0, 5, 8), (73, 17, 3), (7506, 210, 8)]:
        index = torch.randint(max(1, N - 2), (E,), device=device)
        csr = prepare_graph_softmax(index=index, num_nodes=N)
        for cap in [None, 3.0]:
            for eps in [1e-16, 0.1]:
                for rshape in [None, (), (E, 1), (1, H), (E, H)]:
                    x = torch.randn(E, H, device=device, requires_grad=True)
                    r = None if rshape is None else torch.rand(rshape, device=device, requires_grad=True)
                    if r is not None and r.numel():
                        with torch.no_grad():
                            r.reshape(-1)[0] = 0  # rescale gradient at zero must survive
                    g = torch.randn_like(x)
                    for p in [0.0, 1.0]:
                        a = fused_graph_softmax(x, index, num_nodes=N, csr=csr,
                                               softcap=cap, eps=eps, exp_rescale=r,
                                               exp_dropout=p, training=True)
                        b = reference_graph_softmax(x, index, num_nodes=N,
                                                    softcap=cap, eps=eps, exp_rescale=r,
                                                    exp_dropout=p, training=True)
                        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                        args = (x,) if r is None else (x, r)
                        ga = torch.autograd.grad(a, args, g)
                        gb = torch.autograd.grad(b, args, g)
                        if (r is not None and (r.ndim < 2 or r.shape[0] == 1)
                                and eps == 1e-16 and p == 0. and bool((r != 0).any())):
                            # Proven cancellation boundary only: native FP32
                            # dr can fluctuate beyond tolerance around zero.
                            with torch.no_grad():
                                xd = x.double()
                                if cap is not None:
                                    xd = cap * torch.tanh(xd / cap)
                                mx = scatter(xd, index, 0, dim_size=N, reduce='max')
                                u = (xd - mx[index]).exp()
                                den = scatter(u * r.double(), index, 0, dim_size=N, reduce='sum') + eps
                                gu = scatter(g.double() * u, index, 0, dim_size=N, reduce='sum')
                                expected_r = ((gu / den) * eps / den).sum_to_size(r.shape).to(r.dtype)
                            gb = (gb[0], torch.where(r != 0, expected_r, gb[1]))
                        for aa, bb in zip(ga, gb):
                            torch.testing.assert_close(aa, bb, atol=3e-5, rtol=3e-4)
        if E:
            x = torch.randn(E, H, device=device, requires_grad=True)
            # ptr route and preserved original edge order for index route
            xs = x[csr.order]
            cp = prepare_graph_softmax(ptr=csr.ptr)
            a = fused_graph_softmax(xs, ptr=csr.ptr, csr=cp)
            b = reference_graph_softmax(xs, ptr=csr.ptr)
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
            # Stochastic dropout: check exact replay with an explicit seed.
            seed = torch.tensor(12345, device=device, dtype=torch.int64)
            a = fused_graph_softmax(x, index, num_nodes=N, csr=csr,
                                   exp_dropout=.3, training=True, seed=seed)
            a2 = fused_graph_softmax(x, index, num_nodes=N, csr=csr,
                                    exp_dropout=.3, training=True, seed=seed)
            torch.testing.assert_close(a, a2, atol=0, rtol=0)
            # Recreate the same realized mask through equal logits.
            uniform = fused_graph_softmax(torch.zeros_like(x), index, num_nodes=N,
                         csr=csr, exp_dropout=.3, training=True, seed=seed)
            mask = (uniform != 0).float() / .7
            b = reference_graph_softmax(x, index, num_nodes=N, exp_rescale=mask)
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
            ga, = torch.autograd.grad(a, x, torch.ones_like(a))
            gb, = torch.autograd.grad(b, x, torch.ones_like(b))
            torch.testing.assert_close(ga, gb, atol=3e-6, rtol=3e-5)
    print('FP32 regular checks and analytic shared-rescale cancellation checks passed')


@dataclass(frozen=True)
class GraphCSR:
    ptr: torch.Tensor
    order: torch.Tensor | None
    edges: int
    nodes: int
    max_degree: int
    source: torch.Tensor
    source_version: int
    # For physically sorted features, preserve original edge IDs for dropout.
    rng_order: torch.Tensor | None = None


def _version(t):
    try:
        return t._version
    except RuntimeError:  # inference-mode tensors have no version counter
        return -1


@torch.no_grad()
def prepare_graph_softmax(index=None, ptr=None, num_nodes=None):
    """One-time construction/validation, including GPU->CPU synchronizations.
    ptr takes precedence, as in PyG. For index, order maps CSR position to
    ORIGINAL edge position; logits and outputs are never explicitly sorted.
    """
    source = ptr if ptr is not None else index
    if source is None or source.ndim != 1 or source.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected 1D int32/int64 index or ptr')
    if ptr is not None:
        if ptr.numel() < 1 or int(ptr[0]) != 0:
            raise ValueError('ptr must start at zero')
        count = ptr[1:] - ptr[:-1]
        if bool((count < 0).any()):
            raise ValueError('ptr must be nondecreasing')
        E, N = int(ptr[-1]), ptr.numel() - 1
        order = None
        offsets = ptr.contiguous()
    else:
        N = int(maybe_num_nodes(index, num_nodes))
        if N < 0 or (index.numel() and (int(index.min()) < 0 or int(index.max()) >= N)):
            raise ValueError('index outside [0, num_nodes)')
        E = index.numel()
        order = torch.argsort(index, stable=True)
        count = torch.bincount(index.long(), minlength=N)
        offsets = torch.cat((count.new_zeros(1), count.cumsum(0)))
    degree = int(count.max()) if count.numel() else 0
    return GraphCSR(offsets, order, E, N, degree, source, _version(source))


def reference_graph_softmax(src, index=None, ptr=None, num_nodes=None, dim=0,
                            exp_rescale=None, eps=1e-16, exp_dropout=0.,
                            softcap=None, training=False):
    if softcap is not None:
        src = softcap * torch.tanh(src / softcap)
    if ptr is not None:
        dim = dim + src.ndim if dim < 0 else dim
        count = ptr[1:] - ptr[:-1]
        p = ptr.view([1] * dim + [-1])
        mx = segment(src.detach(), p, reduce='max').repeat_interleave(count, dim=dim)
        out = (src - mx).exp()
        if exp_rescale is not None:
            out = out * exp_rescale
        out = torch.nn.functional.dropout(out, exp_dropout, training)
        den = (segment(out, p, reduce='sum') + eps).repeat_interleave(count, dim=dim)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        mx = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
        out = (src - mx.index_select(dim, index)).exp()
        if exp_rescale is not None:
            out = out * exp_rescale
        out = torch.nn.functional.dropout(out, exp_dropout, training)
        den = scatter(out, index, dim, dim_size=N, reduce='sum') + eps
        den = den.index_select(dim, index)
    else:
        raise NotImplementedError('index or ptr required')
    return out / den


@triton.jit
def _fwd(X, R, PTR, ORDER, Y, U, SEED,
         H: tl.constexpr, RS0: tl.constexpr, RS1: tl.constexpr,
         INDIRECT: tl.constexpr, RESCALE: tl.constexpr, SAVE_U: tl.constexpr,
         REMAP_RNG: tl.constexpr,
         CAP: tl.constexpr, EPS: tl.constexpr, P: tl.constexpr,
         B: tl.constexpr, C: tl.constexpr):
    n = tl.program_id(0)
    hs = tl.program_id(1) * C + tl.arange(0, C)
    start = tl.load(PTR + n)
    end = tl.load(PTR + n + 1)
    pos = start + tl.arange(0, B)
    if INDIRECT:
        edge = tl.load(ORDER + pos, pos < end, other=0)
    else:
        edge = pos
    off = edge.to(tl.int64)[:, None] * H + hs[None, :]
    valid = (pos[:, None] < end) & (hs[None, :] < H)
    x = tl.load(X + off, valid, other=0)
    if CAP > 0:
        x = CAP * (2.0 * tl.sigmoid(2.0 * x / CAP) - 1.0)
    x = tl.where(valid, x, -float('inf'))
    mx = tl.max(x, axis=0)
    mx = tl.where(start < end, mx, 0.0)
    z = tl.exp(x - mx[None, :])
    z = tl.where(valid, z, 0.0)
    if RESCALE:
        r = tl.load(R + edge[:, None] * RS0 + hs[None, :] * RS1, valid, other=0)
    else:
        r = 1.0
    if P >= 1.0:
        d = tl.full((B, C), 0, tl.float32)
    elif P > 0:
        seed = tl.load(SEED)
        rng_off = off
        if REMAP_RNG:
            original = tl.load(ORDER + pos, pos < end, other=0).to(tl.int64)
            rng_off = original[:, None] * H + hs[None, :]
        d = (tl.rand(seed, rng_off.to(tl.uint32)) >= P).to(tl.float32) / (1.0 - P)
    else:
        d = 1.0
    # Match the requested operation order: exp -> rescale -> dropout -> sum.
    a = (z * r) * d
    den = tl.sum(a, axis=0) + EPS
    y = a / den[None, :]
    tl.store(Y + off, y, valid)
    if SAVE_U:
        # Save before normalization: rounded y/q cannot recover tiny eps terms.
        # This also preserves nonzero rescale derivatives when r is zero.
        tl.store(U + off, z * d, valid)


@triton.jit
def _bwd(X, R, U, G, PTR, ORDER, DX, DR,
         H: tl.constexpr, RS0: tl.constexpr, RS1: tl.constexpr,
         INDIRECT: tl.constexpr, RESCALE: tl.constexpr,
         NEED_X: tl.constexpr, NEED_R: tl.constexpr, SHARED_EDGES: tl.constexpr,
         CAP: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    n = tl.program_id(0)
    hs = tl.program_id(1) * C + tl.arange(0, C)
    start = tl.load(PTR + n)
    end = tl.load(PTR + n + 1)
    pos = start + tl.arange(0, B)
    if INDIRECT:
        edge = tl.load(ORDER + pos, pos < end, other=0)
    else:
        edge = pos
    off = edge.to(tl.int64)[:, None] * H + hs[None, :]
    valid = (pos[:, None] < end) & (hs[None, :] < H)
    u = tl.load(U + off, valid, other=0).to(tl.float64)
    g = tl.load(G + off, valid, other=0).to(tl.float64)
    if RESCALE:
        r = tl.load(R + edge.to(tl.int64)[:, None] * RS0 + hs[None, :] * RS1,
                    valid, other=0).to(tl.float64)
        a = u * r
    else:
        a = u
    total = tl.sum(a, axis=0)
    den = total + EPS

    # Center upstream gradients around the largest-weight edge. For a single
    # active edge, or constant upstream gradients, the cancelling terms are
    # exactly zero; the small eps contribution survives independently.
    magnitude = tl.abs(a)
    largest = tl.max(magnitude, axis=0)
    locations = tl.arange(0, B)
    pivot_index = tl.min(tl.where(valid & (magnitude == largest[None, :]),
                                  locations[:, None], B), axis=0)
    pivot = tl.sum(tl.where(locations[:, None] == pivot_index[None, :], g, 0.0), axis=0)
    centered = g - pivot[None, :]
    centered_sum = tl.sum(a * centered, axis=0)
    numerator = centered * total[None, :] - centered_sum[None, :] + g * EPS
    # Divide twice instead of squaring the denominator (avoids overflow).
    da = (numerator / den[None, :]) / den[None, :]
    if NEED_X:
        dx = a * da
        if CAP > 0:
            x = tl.load(X + off, valid, other=0)
            v = 2.0 * tl.sigmoid(2.0 * x / CAP) - 1.0
            dx = dx * (1.0 - v * v).to(tl.float64)
        tl.store(DX + off, dx, valid)
    if NEED_R:
        if SHARED_EDGES:
            # d/dr sum_i g_i * (r*u_i)/(r*sum_i u_i + eps)
            # = eps * sum_i(g_i*u_i) / den**2, including r == 0.
            dr = ((tl.sum(g * u, axis=0) / den) * EPS) / den
            tl.store(DR + n.to(tl.int64) * H + hs, dr, hs < H)
        else:
            tl.store(DR + off, u * da, valid)


@triton.jit
def _sum_rows(PART, OUT, ROWS: tl.constexpr, H: tl.constexpr,
              BK: tl.constexpr, BH: tl.constexpr):
    chunk = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    row = chunk * BK + tl.arange(0, BK)
    v = tl.load(PART + row[:, None] * H + h[None, :],
                (row[:, None] < ROWS) & (h[None, :] < H), other=0).to(tl.float64)
    tl.store(OUT + chunk * H + h, tl.sum(v, axis=0), h < H)


@triton.jit
def _sum_heads(PART, OUT, ROWS: tl.constexpr, H: tl.constexpr,
               BR: tl.constexpr, BH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    h0 = tl.arange(0, BH)
    total = tl.full((BR,), 0, tl.float64)
    for i in range(tl.cdiv(H, BH)):
        h = i * BH + h0
        v = tl.load(PART + row[:, None] * H + h[None, :],
                    (row[:, None] < ROWS) & (h[None, :] < H), other=0).to(tl.float64)
        total += tl.sum(v, axis=1)
    tl.store(OUT + row, total, row < ROWS)


def _reduce_rescale(partial, out, shared_edges, shared_heads):
    rows, heads = partial.shape
    if shared_edges:
        # Fixed trees; keep partial sums in FP64 until the final FP32 store.
        while True:
            blocks = triton.cdiv(rows, 256)
            dest = (out if blocks == 1 and not shared_heads else
                    torch.empty((blocks, heads), device=out.device, dtype=torch.float64))
            _sum_rows[(blocks, triton.cdiv(heads, 32))](
                partial, dest, rows, heads, 256, 32, num_warps=4, enable_fp_fusion=False)
            partial, rows = dest, blocks
            if blocks == 1:
                break
    if shared_heads:
        _sum_heads[(triton.cdiv(rows, 16),)](
            partial, out, rows, heads, 16, triton.next_power_of_2(min(heads, 256)),
            num_warps=4, enable_fp_fusion=False)


class _GraphSoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, r, csr, cap, eps, p, seed, block_heads, warps):
        H = x.shape[1]
        B = triton.next_power_of_2(max(1, csr.max_degree))
        C = block_heads
        y = torch.empty_like(x)
        need_x, need_r = ctx.needs_input_grad[:2]
        need_r = r is not None and need_r
        u = torch.empty_like(x) if need_x or need_r else x.new_empty(0)
        rr = r.expand_as(x) if r is not None else x
        order = (csr.order if csr.order is not None else
                 csr.rng_order if csr.rng_order is not None else csr.ptr)
        with torch.cuda.device(x.device):
            _fwd[(csr.nodes, triton.cdiv(H, C))](
                x, rr, csr.ptr, order, y, u, seed, H, *rr.stride(),
                csr.order is not None, r is not None, need_x or need_r,
                csr.rng_order is not None, cap, eps, p,
                B, C, num_warps=warps, enable_fp_fusion=False)
        ctx.save_for_backward(x, r, u, csr.ptr, order)
        ctx.meta = H, B, C, warps, cap, eps, csr.order is not None, need_x, need_r
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, r, u, ptr, order = ctx.saved_tensors
        H, B, C, warps, cap, eps, indirect, need_x, need_r = ctx.meta
        # Infer broadcasting from the input shape, never from zero strides:
        # an expanded [E,H] input still needs distinct elementwise gradients.
        shared_edges = need_r and (r.ndim < 2 or r.shape[0] == 1)
        shared_heads = need_r and (r.ndim == 0 or r.shape[-1] == 1)
        dx = torch.empty_like(x) if need_x else None
        dr = torch.empty(r.shape, device=x.device, dtype=x.dtype) if need_r else None
        partial = dr
        if need_r and (shared_edges or shared_heads):
            rows = ptr.numel() - 1 if shared_edges else x.shape[0]
            partial = torch.empty((rows, H), device=x.device, dtype=torch.float64)
        rr = r.expand_as(x) if r is not None else x
        with torch.cuda.device(x.device):
            _bwd[(ptr.numel() - 1, triton.cdiv(H, C))](
                x, rr, u, grad.contiguous(), ptr, order,
                dx if need_x else x, partial if need_r else x,
                H, *rr.stride(), indirect, r is not None, need_x, need_r, shared_edges,
                cap, eps, B, C, num_warps=warps, enable_fp_fusion=False)
            if need_r and (shared_edges or shared_heads):
                _reduce_rescale(partial, dr, shared_edges, shared_heads)
        return dx, dr, None, None, None, None, None, None, None


def fused_graph_softmax(src, index=None, ptr=None, num_nodes=None, dim=0,
                        exp_rescale=None, *, eps=1e-16, exp_dropout=0.,
                        softcap=None, training=False, csr=None,
                        block_heads=8, num_warps=4, force_torch=False, seed=None):
    """Original order is preserved. CSR is reusable only for identical topology.

    block_heads: power of 2, try 1/2/4/8. num_warps: try 4/8 (2 on NVIDIA).
    seed: optional one-element GPU int64 tensor. Only used with 0<p<1.
    Omit seed to obtain fresh randomness from the PyTorch device RNG.
    Dropout has PyTorch semantics but does not reproduce its exact RNG mask.
    Explicit fixed seeds also freeze dropout under CUDA Graph replay.
    Broadcast rescale supports [], [H], [1,H], [E,1], [E,H].
    Large-degree graphs fall back rather than truncating neighbors.

    First-order backward saves unnormalized exponentials and uses centered
    FP64 arithmetic with fixed-tree rescale reductions. Shared-edge rescale
    gradients retain the eps term analytically, including at r=0. The forward
    remains FP32. This avoids cancellation artifacts of native FP32 backward
    in the shared-rescale and single-neighbor/small-rescale limits.
    """
    if not 0 <= exp_dropout <= 1:
        raise ValueError('exp_dropout must be in [0,1]')
    if softcap is not None and softcap <= 0:
        raise ValueError('softcap must be positive')
    if eps <= 0:
        raise ValueError('eps must be positive')
    if block_heads not in (1, 2, 4, 8, 16, 32) or num_warps not in (1, 2, 4, 8):
        raise ValueError('Unsupported block_heads / num_warps')
    fallback = lambda: reference_graph_softmax(
        src, index, ptr, num_nodes, dim, exp_rescale, eps, exp_dropout, softcap, training)
    fast = (not force_torch and src.is_cuda and src.dtype == torch.float32
            and src.ndim == 2 and dim in (0, -2) and src.is_contiguous()
            and src.numel() > 0)
    if exp_rescale is not None:
        fast = fast and (exp_rescale.device == src.device
                        and exp_rescale.dtype == src.dtype and exp_rescale.ndim <= 2)
    if not fast:
        return fallback()
    source = ptr if ptr is not None else index
    if source is None:
        raise ValueError('index or ptr required')
    if csr is None:
        csr = prepare_graph_softmax(index, ptr, num_nodes)
    if (csr.source is not source or csr.source_version != _version(source)
            or csr.edges != src.shape[0] or csr.ptr.device != src.device
            or (ptr is None and num_nodes is not None and csr.nodes != num_nodes)):
        raise ValueError('CSR mismatch: rebuild it for the current graph/index object')
    B = triton.next_power_of_2(max(1, csr.max_degree))
    if B > 2048 or B * block_heads > 16384:
        return fallback()
    if exp_rescale is not None:
        exp_rescale.expand_as(src)  # validate broadcasting before launching
    p = float(exp_dropout) if training else 0.0
    if 0 < p < 1:
        if seed is None:
            seed = torch.randint(0, 2**31, (), device=src.device, dtype=torch.int64)
        if seed.device != src.device or seed.dtype != torch.int64 or seed.numel() != 1:
            raise ValueError('seed must be a one-element int64 tensor on src.device')
    else:
        seed = src  # unused pointer; no RNG allocation in eval or p=0/1
    return _GraphSoftmaxFn.apply(src, exp_rescale, csr,
                                float(softcap or 0), float(eps), p, seed,
                                block_heads, num_warps)


class FusedGraphSoftmax(torch.nn.Module):
    """Drop-in call signature; pass csr=... to avoid per-call preprocessing.
    No implicit cache: tensor-pointer caches can silently reuse stale graphs.
    """
    def __init__(self, eps=1e-16, exp_dropout=0., softcap=None,
                 block_heads=8, num_warps=4, force_torch=False):
        super().__init__()
        self.eps, self.exp_dropout, self.softcap = eps, exp_dropout, softcap
        self.block_heads, self.num_warps = block_heads, num_warps
        self.force_torch = force_torch

    def forward(self, src, index=None, ptr=None, num_nodes=None, dim=0,
                exp_rescale=None, *, csr=None):
        return fused_graph_softmax(
            src, index, ptr, num_nodes, dim, exp_rescale, eps=self.eps,
            exp_dropout=self.exp_dropout, softcap=self.softcap,
            training=self.training, csr=csr, block_heads=self.block_heads,
            num_warps=self.num_warps, force_torch=self.force_torch)

    def extra_repr(self):
        return f'eps={self.eps}, exp_dropout={self.exp_dropout}, softcap={self.softcap}'


def benchmark():
    import time
    import triton.testing
    torch.manual_seed(7)
    E, N, H = 7506, 210, 8
    index = torch.randint(N, (E,), device='cuda')
    x = torch.randn(E, H, device='cuda')
    csr = prepare_graph_softmax(index=index, num_nodes=N)
    ref = lambda: reference_graph_softmax(x, index, num_nodes=N)
    print(f'Synthetic graph: E={E}, N={N}, H={H}, max_degree={csr.max_degree}')
    print('FP32 eval p=0; reference and fast path receive the same unsorted data.')
    print(f'PyG GPU time: {1000 * triton.testing.do_bench(ref):.3f} us')
    results = []
    for heads in [1, 2, 4, 8]:
        for warps in [4, 8]:
            fn = lambda: fused_graph_softmax(x, index, num_nodes=N, csr=csr,
                                            block_heads=heads, num_warps=warps)
            torch.testing.assert_close(fn(), ref(), atol=3e-6, rtol=3e-5)
            us = 1000 * triton.testing.do_bench(fn)
            results.append((us, heads, warps))
            print(f'CSR reused, heads={heads}, warps={warps}: {us:.3f} us')
    _, heads, warps = min(results)
    # Host sync inside CSR construction: use wall clock, not GPU-only events.
    def wall(fn, repeats=30):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1e6 / repeats
    fast = lambda: fused_graph_softmax(x, index, num_nodes=N, csr=csr,
                                       block_heads=heads, num_warps=warps)
    rebuild = lambda: fused_graph_softmax(x, index, num_nodes=N,
                                          block_heads=heads, num_warps=warps)
    print(f'Wall PyG: {wall(ref):.3f} us')
    print(f'Wall cached CSR: {wall(fast):.3f} us')
    print(f'Wall rebuild CSR + softmax: {wall(rebuild):.3f} us')
    x.requires_grad_(True)
    g = torch.randn_like(x)
    fb = lambda: torch.autograd.grad(fast(), x, g)
    rb = lambda: torch.autograd.grad(ref(), x, g)
    print(f'PyG forward+backward GPU: {1000 * triton.testing.do_bench(rb):.3f} us')
    print(f'Triton forward+backward GPU: {1000 * triton.testing.do_bench(fb):.3f} us')
    print('Use your real index/degree distribution for final tuning decisions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--bench', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required')
    if args.check:
        check()
    if args.bench:
        benchmark()
    if not (args.check or args.bench):
        parser.print_help()
