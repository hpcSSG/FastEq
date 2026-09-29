"""GraphSoftmax fusion for changing graphs, without sorting or CSR construction.

Edges stay in their input order. Node reductions use scatter atomics, so sums
are tolerance-correct rather than bitwise deterministic. Pass num_nodes to
avoid a host synchronization to infer the output size from index.max().
"""
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from torch_geometric.utils.num_nodes import maybe_num_nodes

from .graph_softmax import _reduce_rescale, reference_graph_softmax


@triton.jit
def _init(MAX, SUM, PIVOT, N: tl.constexpr, H: tl.constexpr, E: tl.constexpr,
          SAVE: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    tl.store(MAX + i, -float('inf'), i < N * H)
    tl.store(SUM + i, 0., i < N * H)
    if SAVE:
        tl.store(PIVOT + i, E, i < N)


@triton.jit
def _max(X, INDEX, MAX, PIVOT, E: tl.constexpr, N: tl.constexpr,
         H: tl.constexpr, CAP: tl.constexpr, SAVE: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    edge, head = i // H, i % H
    node = tl.load(INDEX + edge, i < E * H, other=0).to(tl.int64)
    valid = (i < E * H) & (node >= 0) & (node < N)
    tl.device_assert((i >= E * H) | ((node >= 0) & (node < N)), 'index out of range')
    x = tl.load(X + i, valid, other=0)
    if CAP > 0:
        x = CAP * (2. * tl.sigmoid(2. * x / CAP) - 1.)
    tl.atomic_max(MAX + node * H + head, x, valid, sem='relaxed')
    if SAVE:
        # An integer pivot per node is sufficient for centered derivatives.
        # It is not an edge permutation or a neighbor list.
        tl.atomic_min(PIVOT + node, edge, valid & (head == 0), sem='relaxed')


@triton.jit
def _exp_sum(X, R, INDEX, MAX, SUM, A, U, SEED,
             E: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
             RS0: tl.constexpr, RS1: tl.constexpr, RESCALE: tl.constexpr,
             SAVE: tl.constexpr, CAP: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    edge, head = i // H, i % H
    node = tl.load(INDEX + edge, i < E * H, other=0).to(tl.int64)
    valid = (i < E * H) & (node >= 0) & (node < N)
    x = tl.load(X + i, valid, other=0)
    if CAP > 0:
        x = CAP * (2. * tl.sigmoid(2. * x / CAP) - 1.)
    maximum = tl.load(MAX + node * H + head, valid, other=0)
    z = tl.exp(x - maximum)
    if RESCALE:
        r = tl.load(R + edge * RS0 + head * RS1, valid, other=0)
    else:
        r = 1.
    if P >= 1.:
        d = tl.full((B,), 0., tl.float32)
    elif P > 0.:
        d = (tl.rand(tl.load(SEED), i.to(tl.uint32)) >= P).to(tl.float32) / (1. - P)
    else:
        d = 1.
    a = (z * r) * d
    tl.store(A + i, a, valid)
    tl.atomic_add(SUM + node * H + head, a, valid, sem='relaxed')
    if SAVE:
        tl.store(U + i, z * d, valid)


@triton.jit
def _normalize(A, INDEX, SUM, E: tl.constexpr, N: tl.constexpr,
               H: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    node = tl.load(INDEX + i // H, i < E * H, other=0).to(tl.int64)
    valid = (i < E * H) & (node >= 0) & (node < N)
    a = tl.load(A + i, valid, other=0)
    den = tl.load(SUM + node * H + i % H, valid, other=0) + EPS
    tl.store(A + i, a / den, valid)


@triton.jit
def _backward_reduce(U, R, G, INDEX, PIVOT, STATS,
                     E: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
                     RS0: tl.constexpr, RS1: tl.constexpr,
                     RESCALE: tl.constexpr, SHARED_R: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    edge, head = i // H, i % H
    node = tl.load(INDEX + edge, i < E * H, other=0).to(tl.int64)
    valid = (i < E * H) & (node >= 0) & (node < N)
    pivot_edge = tl.load(PIVOT + node, valid, other=0)
    pivot = tl.load(G + pivot_edge * H + head, valid, other=0).to(tl.float64)
    u = tl.load(U + i, valid, other=0).to(tl.float64)
    g = tl.load(G + i, valid, other=0).to(tl.float64)
    if RESCALE:
        r = tl.load(R + edge * RS0 + head * RS1, valid, other=0).to(tl.float64)
        a = u * r
    else:
        a = u
    off = node * H + head
    # FP64 limits order-dependent scatter error and preserves cancellation
    # boundaries. No CUDA/HIP-specific arithmetic or assembly is used.
    tl.atomic_add(STATS + off, a, valid, sem='relaxed')
    tl.atomic_add(STATS + N * H + off, a * (g - pivot), valid, sem='relaxed')
    if SHARED_R:
        tl.atomic_add(STATS + 2 * N * H + off, g * u, valid, sem='relaxed')


@triton.jit
def _backward_edges(X, R, U, G, INDEX, PIVOT, STATS, DX, DR,
                    E: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
                    RS0: tl.constexpr, RS1: tl.constexpr,
                    RESCALE: tl.constexpr, NEED_X: tl.constexpr, EDGE_R: tl.constexpr,
                    CAP: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    edge, head = i // H, i % H
    node = tl.load(INDEX + edge, i < E * H, other=0).to(tl.int64)
    valid = (i < E * H) & (node >= 0) & (node < N)
    pivot_edge = tl.load(PIVOT + node, valid, other=0)
    pivot = tl.load(G + pivot_edge * H + head, valid, other=0).to(tl.float64)
    g = tl.load(G + i, valid, other=0).to(tl.float64)
    u = tl.load(U + i, valid, other=0).to(tl.float64)
    off = node * H + head
    total = tl.load(STATS + off, valid, other=0)
    centered_sum = tl.load(STATS + N * H + off, valid, other=0)
    den = total + EPS
    da = (((g - pivot) * total - centered_sum + g * EPS) / den) / den
    if NEED_X:
        if RESCALE:
            r = tl.load(R + edge * RS0 + head * RS1, valid, other=0).to(tl.float64)
        else:
            r = 1.
        dx = (u * r) * da
        if CAP > 0:
            x = tl.load(X + i, valid, other=0)
            t = 2. * tl.sigmoid(2. * x / CAP) - 1.
            dx = dx * (1. - t * t).to(tl.float64)
        tl.store(DX + i, dx, valid)
    if EDGE_R:
        tl.store(DR + i, u * da, valid)


@triton.jit
def _shared_rescale(STATS, DR, N: tl.constexpr, H: tl.constexpr,
                    EPS: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    total = tl.load(STATS + i, i < N * H, other=0)
    gu = tl.load(STATS + 2 * N * H + i, i < N * H, other=0)
    den = total + EPS
    tl.store(DR + i, ((gu / den) * EPS) / den, i < N * H)


class _GraphSoftmaxIndexFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, r, index, nodes, cap, eps, p, seed):
        e, h = x.shape
        need_x, need_r = ctx.needs_input_grad[:2]
        need_r = r is not None and need_r
        save = need_x or need_r
        y = torch.empty_like(x)
        u = torch.empty_like(x) if save else x.new_empty(0)
        maximum = x.new_empty((nodes, h))
        sums = torch.empty_like(maximum)
        pivot = torch.empty((nodes if save else 0,), device=x.device, dtype=torch.int64)
        rr = r.expand_as(x) if r is not None else x
        with torch.cuda.device(x.device):
            _init[(triton.cdiv(nodes * h, 256),)](
                maximum, sums, pivot, nodes, h, e, save, 256)
            grid = (triton.cdiv(e * h, 256),)
            _max[grid](x, index, maximum, pivot, e, nodes, h, cap, save, 256,
                       enable_fp_fusion=False)
            _exp_sum[grid](x, rr, index, maximum, sums, y, u, seed,
                           e, nodes, h, *rr.stride(), r is not None, save, cap, p,
                           256, enable_fp_fusion=False)
            _normalize[grid](y, index, sums, e, nodes, h, eps, 256)
        ctx.save_for_backward(x, r, u, index, pivot)
        ctx.meta = nodes, cap, eps, need_x, need_r
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, r, u, index, pivot = ctx.saved_tensors
        nodes, cap, eps, need_x, need_r = ctx.meta
        e, h = x.shape
        shared_edges = need_r and (r.ndim < 2 or r.shape[0] == 1)
        shared_heads = need_r and (r.ndim == 0 or r.shape[-1] == 1)
        dx = torch.empty_like(x) if need_x else None
        dr = torch.empty(r.shape, dtype=x.dtype, device=x.device) if need_r else None
        partial = dr
        if need_r and (shared_edges or shared_heads):
            partial = torch.empty((nodes if shared_edges else e, h),
                                  dtype=torch.float64, device=x.device)
        rr = r.expand_as(x) if r is not None else x
        g = grad.contiguous()
        with torch.cuda.device(x.device):
            stats = torch.zeros((3 if shared_edges else 2, nodes, h),
                                dtype=torch.float64, device=x.device)
            grid = (triton.cdiv(e * h, 256),)
            _backward_reduce[grid](u, rr, g, index, pivot, stats, e, nodes, h,
                                   *rr.stride(), r is not None, shared_edges, 256,
                                   enable_fp_fusion=False)
            if need_x or (need_r and not shared_edges):
                _backward_edges[grid](x, rr, u, g, index, pivot, stats,
                    dx if need_x else x, partial if need_r else x,
                    e, nodes, h, *rr.stride(), r is not None, need_x,
                    need_r and not shared_edges, cap, eps, 256, enable_fp_fusion=False)
            if shared_edges:
                _shared_rescale[(triton.cdiv(nodes * h, 256),)](
                    stats, partial, nodes, h, eps, 256, enable_fp_fusion=False)
            if need_r and (shared_edges or shared_heads):
                _reduce_rescale(partial, dr, shared_edges, shared_heads)
        return dx, dr, None, None, None, None, None, None


def fused_graph_softmax_index(src, index, num_nodes=None, dim=0, exp_rescale=None,
                              *, eps=1e-16, exp_dropout=0., softcap=None,
                              training=False, seed=None, force_torch=False):
    """Fuse GraphSoftmax on raw destination indices; no CSR/cache/permutation.

    The FP32 fast path accepts [E,H], int32/int64 indices and the same broadcast
    rescale shapes as fused_graph_softmax. Indices must be in [0,num_nodes).
    Pass num_nodes to avoid size inference synchronizing with the host. Every
    call reads the current index values, including in-place topology updates.
    Backward is first-order only; FP64 scatter sums are not bitwise ordered.
    Dropout uses original edge IDs and the same seed mapping as the CSR route.
    Unsupported shapes/dtypes and empty inputs use the native index expression.
    """
    if not 0 <= exp_dropout <= 1:
        raise ValueError('exp_dropout must be in [0,1]')
    if softcap is not None and softcap <= 0:
        raise ValueError('softcap must be positive')
    if eps <= 0:
        raise ValueError('eps must be positive')
    if index is None or index.ndim != 1 or index.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected 1D int32/int64 destination indices')
    if index.device != src.device:
        raise ValueError('index and src must be on the same device')
    if src.ndim and index.numel() != src.shape[dim]:
        raise ValueError('index length must match the edge dimension')
    fast = (not force_torch and src.is_cuda and src.dtype == torch.float32
            and src.ndim == 2 and dim in (0, -2) and src.numel() > 0)
    if exp_rescale is not None:
        fast = fast and (exp_rescale.device == src.device
                        and exp_rescale.dtype == src.dtype and exp_rescale.ndim <= 2)
    if not fast:
        return reference_graph_softmax(src, index=index, num_nodes=num_nodes, dim=dim,
            exp_rescale=exp_rescale, eps=eps, exp_dropout=exp_dropout,
            softcap=softcap, training=training)
    nodes = int(maybe_num_nodes(index, num_nodes))
    if nodes <= 0:
        raise ValueError('nonempty indices require positive num_nodes')
    if exp_rescale is not None:
        exp_rescale.expand_as(src)
    p = float(exp_dropout) if training else 0.
    if 0 < p < 1:
        if seed is None:
            seed = torch.randint(0, 2**31, (), device=src.device, dtype=torch.int64)
        if seed.device != src.device or seed.dtype != torch.int64 or seed.numel() != 1:
            raise ValueError('seed must be a one-element int64 tensor on src.device')
    else:
        seed = src
    return _GraphSoftmaxIndexFn.apply(src.contiguous(), exp_rescale, index.contiguous(),
        nodes, float(softcap or 0), float(eps), p, seed)


class FusedGraphSoftmaxIndex(torch.nn.Module):
    """Index-only alternative for graphs whose topology changes each step."""
    def __init__(self, eps=1e-16, exp_dropout=0., softcap=None, force_torch=False):
        super().__init__()
        self.eps, self.exp_dropout, self.softcap = eps, exp_dropout, softcap
        self.force_torch = force_torch

    def forward(self, src, index, num_nodes=None, dim=0, exp_rescale=None, *, seed=None):
        return fused_graph_softmax_index(src, index, num_nodes, dim, exp_rescale,
            eps=self.eps, exp_dropout=self.exp_dropout, softcap=self.softcap,
            training=self.training, force_torch=self.force_torch, seed=seed)
