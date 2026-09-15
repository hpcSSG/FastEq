"""Shared Triton normalization with framework-native source reductions.

Forward statistics/affine and backward dX/parameter partials use shared Triton
kernels. Native-source reduction axes use Torch without backend dispatch here.
"""
from dataclasses import dataclass
from typing import NamedTuple
import math
import torch
import triton
import triton.language as tl

@dataclass(frozen=True)
class EquivariantNormSpec:
    lmax: int
    channels: int
    stats_weights: tuple[tuple[float, ...], ...]
    output_group: tuple[int, ...]
    center_scalar: bool = True
    eps: float = 1e-5

    def __post_init__(self):
        if type(self.lmax) is not int or self.lmax < 0:
            raise ValueError('lmax must be a nonnegative integer')
        if type(self.channels) is not int or self.channels < 1:
            raise ValueError('channels must be a positive integer')
        if type(self.center_scalar) is not bool:
            raise ValueError('center_scalar must be boolean')
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('eps must be finite and positive')
        weights = tuple(tuple(float(v) for v in row) for row in self.stats_weights)
        groups = tuple(self.output_group)
        if not weights or any(len(row) != self.lmax + 1 for row in weights):
            raise ValueError('stats_weights must have shape [G, lmax+1], G > 0')
        if any(not math.isfinite(v) or v < 0 for row in weights for v in row):
            raise ValueError('statistic weights must be finite and nonnegative')
        if any(not any(v > 0 for v in row) for row in weights):
            raise ValueError('each statistic must have a nonzero source')
        if len(groups) != self.lmax + 1 or any(
            type(g) is not int or not 0 <= g < len(weights) for g in groups
        ):
            raise ValueError('output_group must select one valid group per degree')
        if set(groups) != set(range(len(weights))):
            raise ValueError('every statistic must be used by an output degree')
        object.__setattr__(self, 'stats_weights', weights)
        object.__setattr__(self, 'output_group', groups)

    @property
    def components(self):
        return (self.lmax + 1) ** 2

    @property
    def num_stats(self):
        return len(self.stats_weights)

    @classmethod
    def from_preset(cls, *, lmax, channels, grouping='per_degree',
                    weighting='degree_balanced', center_scalar=True, eps=1e-5):
        if type(lmax) is not int or lmax < 0:
            raise ValueError('lmax must be a nonnegative integer')
        if grouping == 'per_degree':
            sources = [(l,) for l in range(lmax + 1)]
        elif grouping == 'scalar_high':
            sources = [(0,)] + ([tuple(range(1, lmax + 1))] if lmax else [])
        else:
            raise ValueError('unknown grouping preset')
        if weighting not in ('component', 'degree_balanced', 'norm'):
            raise ValueError('unknown weighting preset')
        weights, output_group = [], [0] * (lmax + 1)
        for g, degrees in enumerate(sources):
            row = [0.] * (lmax + 1)
            for l in degrees:
                if weighting == 'degree_balanced':
                    row[l] = 1. / (len(degrees) * (2 * l + 1))
                elif weighting == 'component':
                    row[l] = 1. / sum(2 * j + 1 for j in degrees)
                else:
                    row[l] = 1.
                output_group[l] = g
            weights.append(tuple(row))
        return cls(lmax, channels, tuple(weights), tuple(output_group), center_scalar, eps)

class NormResult(NamedTuple):
    output: torch.Tensor
    mean: torch.Tensor | None   # [N]; absent if scalar centering is disabled
    moments: torch.Tensor | None  # [N,G]; omitted unless requested
    rstd: torch.Tensor          # [N,G]

@triton.jit
def _moment(square, w, coefficient, C: tl.constexpr, ORDER: tl.constexpr,
            UNIFORM: tl.constexpr):
    if ORDER == 'channels_first':
        q = tl.sum(square, 1) / C
        if UNIFORM:
            # Source unbalanced component/norm path reduces before scaling.
            return tl.sum(q, 0) * coefficient
        if q.shape[0] > 64:
            # Bound compile size for large generic specifications.
            return tl.sum(q * w, 0)
        # Two ordered FMA streams preserve weighted source statistics without
        # backend-specific math libraries or inline assembly.
        a = tl.full((), 0., tl.float32)
        b = tl.full((), 0., tl.float32)
        k = tl.arange(0, q.shape[0])
        for i in tl.static_range(q.shape[0] // 2):
            ai = tl.sum(tl.where(k == i, q, 0.), 0)
            aw = tl.sum(tl.where(k == i, w, 0.), 0)
            bi = tl.sum(tl.where(k == i + q.shape[0] // 2, q, 0.), 0)
            bw = tl.sum(tl.where(k == i + q.shape[0] // 2, w, 0.), 0)
            a = tl.fma(ai, aw, a)
            b = tl.fma(bi, bw, b)
        return a + b
    else:
        if UNIFORM:
            q = tl.sum(square, 0) * coefficient
        else:
            q = tl.sum(square * w[:, None], 0)
        return tl.sum(q, 0) / C

@triton.jit
def _affine(z, r, k, c, valid, DEGREE, WEIGHT0, WEIGHTH, BIAS,
            HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr,
            WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr,
            BS: tl.constexpr, K: tl.constexpr):
    if HAS_WEIGHT:
        degree = tl.load(DEGREE + k, k < K, 0)
        if SPLIT:
            scalar_weight = tl.load(WEIGHT0 + c * WS0 + tl.zeros_like(k), valid & (k == 0), 0.)
            higher_weight = tl.load(WEIGHTH + (degree - 1) * WSL + c * WSC,
                                    valid & (k > 0), 0.)
            gamma = tl.where(k == 0, scalar_weight, higher_weight)
        else:
            gamma = tl.load(WEIGHT0 + degree * WSL + c * WSC, valid, 0.)
        y = z * (r * gamma)
    else:
        y = z * r
    if HAS_BIAS:
        beta = tl.load(BIAS + c * BS + tl.zeros_like(k), valid & (k == 0), 0.)
        y = tl.where(k == 0, y + beta, y)
    return y

@triton.jit
def _gamma(k, c, valid, DEGREE, W0, WH, K: tl.constexpr,
           HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr,
           WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr):
    if HAS_WEIGHT:
        degree = tl.load(DEGREE + k, k < K, 0)
        if SPLIT:
            scalar = tl.load(W0 + c * WS0 + tl.zeros_like(k), valid & (k == 0), 0.)
            higher = tl.load(WH + (degree - 1) * WSL + c * WSC,
                             valid & (k > 0), 0.)
            return tl.where(k == 0, scalar, higher)
        return tl.load(W0 + degree * WSL + c * WSC, valid, 0.)
    return tl.full(valid.shape, 1., tl.float32)

@triton.jit
def _fused(X, Y, MU, MOMENTS, RSTD, COEFFICIENT, DEGREE,
           WEIGHT0, WEIGHTH, BIAS, K: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
           SN: tl.constexpr, SK: tl.constexpr, CENTER: tl.constexpr,
           EPS: tl.constexpr, ORDER: tl.constexpr, UNIFORM: tl.constexpr,
           HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr,
           WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr, BS: tl.constexpr,
           SAVE: tl.constexpr, SAVE_MOMENTS: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
           GROUP_BOUNDS):
    n = tl.program_id(0)
    k = tl.arange(0, BK)
    c = tl.arange(0, BC)
    valid = (k[:, None] < K) & (c[None, :] < C)
    x = tl.load(X + n * SN + k[:, None] * SK + c[None, :], valid, 0.)
    if CENTER:
        scalar = tl.sum(tl.where(k[:, None] == 0, x, 0.), 0)
        mu = tl.sum(scalar, 0) / C
        z = tl.where(valid, x - tl.where(k[:, None] == 0, mu, 0.), 0.)
    else:
        z = x
    w = tl.load(COEFFICIENT + k, k < K, 0.)
    square = z * z
    row_rstd = tl.full((BK,), 0., tl.float32)
    for g in tl.static_range(G):
        start = tl.load(GROUP_BOUNDS + g)
        end = tl.load(GROUP_BOUNDS + g + 1)
        member = (k >= start) & (k < end)
        # Select absent sources before reduction: 0 * NaN is not isolation.
        group_square = tl.where(member[:, None], square, 0.)
        group_w = tl.where(member, w, 0.)
        coefficient = tl.sum(tl.where(k == start, w, 0.), 0)
        v = _moment(group_square, group_w, coefficient, C, ORDER, UNIFORM)
        r = tl.rsqrt(v + EPS)
        row_rstd = tl.where(member, r, row_rstd)
        if SAVE:
            if SAVE_MOMENTS:
                tl.store(MOMENTS + n * G + g, v)
            tl.store(RSTD + n * G + g, r)
    y = _affine(z, row_rstd[:, None], k[:, None], c[None, :], valid, DEGREE, WEIGHT0,
                WEIGHTH, BIAS, HAS_WEIGHT, SPLIT, HAS_BIAS, WS0, WSL, WSC, BS, K)
    tl.store(Y + n * K * C + k[:, None] * C + c[None, :], y, valid)
    if SAVE:
        if CENTER:
            tl.store(MU + n, mu)

@triton.jit
def _component_sum(value, k, start, end, SEQUENTIAL: tl.constexpr):
    if SEQUENTIAL:
        # Native affine broadcasting reduces each degree in component order.
        # Keep this local order for cancellation; atom reduction stays a tree.
        total = tl.full((value.shape[1],), 0., tl.float32)
        for i in range(start, end):
            total += tl.sum(tl.where(k == i, value, 0.), 0)
        return total
    return tl.sum(tl.where((k >= start) & (k < end), value, 0.), 0)


@triton.jit
def _portable_backward(X, DY, MU, RSTD, DR, DX, PW, PB, COEF, DEGREE, W0, WH, BOUNDS,
                       K: tl.constexpr, C: tl.constexpr, G: tl.constexpr, L: tl.constexpr,
                       SN: tl.constexpr, SK: tl.constexpr, DN: tl.constexpr, DK: tl.constexpr, DC: tl.constexpr,
                       CENTER: tl.constexpr, HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr,
                       WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr,
                       NX: tl.constexpr, NW: tl.constexpr, NB: tl.constexpr,
                       EXPANDED: tl.constexpr, NATIVE_REDUCE: tl.constexpr, PRECOMPUTED_DR: tl.constexpr, SOURCE_ORDER: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr):
    n = tl.program_id(0)
    k = tl.arange(0, BK)[:, None]
    c = tl.arange(0, BC)[None, :]
    valid = (k < K) & (c < C)
    z = tl.load(X + n * SN + k * SK + c, valid, 0.)
    if CENTER:
        mu = tl.load(MU + n)
        z = tl.where(valid, z - tl.where(k == 0, mu, 0.), 0.)
    dy = tl.load(DY + n * DN + k * DK + c * DC, valid, 0.)
    row_r = tl.full((BK, 1), 0., tl.float32)
    for g in tl.static_range(G):
        a = tl.load(BOUNDS + g)
        b = tl.load(BOUNDS + g + 1)
        member = (k >= a) & (k < b)
        r = tl.load(RSTD + n * G + g)
        row_r = tl.where(member, r, row_r)
    # Source operation order: d(scale)=dY*z; broadcast reduces m before r.
    product = dy * z
    if NX:
        gamma = _gamma(k, c, valid, DEGREE, W0, WH, K, HAS_WEIGHT, SPLIT, WS0, WSL, WSC)
        direct = dy * (row_r * gamma)
        row_dv = tl.full((BK, 1), 0., tl.float32)
        for g in tl.static_range(G):
            a = tl.load(BOUNDS + g)
            b = tl.load(BOUNDS + g + 1)
            member = (k >= a) & (k < b)
            if PRECOMPUTED_DR:
                dr = tl.load(DR + n * G + g)
            else:
                if EXPANDED:
                    dr = tl.sum(tl.reshape(tl.where(member & valid, product * gamma, 0.), (BK * BC,)), 0)
                else:
                    dr = tl.full((), 0., tl.float32)
                    for l in tl.static_range(L, -1, -1):
                        lm = (k >= l*l) & (k < (l+1)*(l+1)) & member
                        ds = _component_sum(tl.where(member & valid, product, 0.), k, l*l, (l+1)*(l+1), SOURCE_ORDER)
                        weight = tl.sum(tl.where(k == l*l, gamma, 0.), 0)
                        if HAS_WEIGHT:
                            dr += tl.sum(ds * weight, 0)
                        else:
                            dr += tl.sum(tl.reshape(tl.where(lm & valid, product, 0.), (BK * BC,)), 0)
            r = tl.load(RSTD + n * G + g)
            # Multiply dr first: avoids overflowing r**3 for tiny eps and zero dr.
            dv = (((dr * -0.5) * r) * r) * r
            row_dv = tl.where(member, dv, row_dv)
        coeff = tl.load(COEF + k, k < K, 0.)
        u = tl.where(valid, direct + ((row_dv * coeff) / C) * (2. * z), 0.)
        if CENTER:
            scalar = tl.sum(tl.where(k == 0, u, 0.), 0)
            mean_u = tl.sum(scalar, 0) / C
            u = tl.where(valid, u - tl.where(k == 0, mean_u, 0.), 0.)
        tl.store(DX + n*K*C + k*C + c, u, valid)
    if NW and (EXPANDED and NATIVE_REDUCE):
        tl.store(PW + n*K*C + k*C + c, product * row_r, valid)
    elif NW:
        for l in tl.static_range(L + 1):
            member = (k >= l*l) & (k < (l+1)*(l+1))
            if EXPANDED:
                partial = _component_sum(tl.where(valid, product * row_r, 0.), k, l*l, (l+1)*(l+1), SOURCE_ORDER)
            else:
                partial = _component_sum(tl.where(valid, product, 0.), k, l*l, (l+1)*(l+1), SOURCE_ORDER)
                partial *= tl.sum(tl.where(k == l*l, row_r, 0.), 0)
            tl.store(PW + (n*(L+1)+l)*C + tl.arange(0, BC), partial, tl.arange(0, BC) < C)
    if NB:
        partial = tl.sum(tl.where(k == 0, dy, 0.), 0)
        tl.store(PB + n*C + tl.arange(0, BC), partial, tl.arange(0, BC) < C)


@triton.jit
def _tree_reduce_rows(X, Y, N: tl.constexpr, W: tl.constexpr,
                      FP64: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr):
    block = tl.program_id(0)
    row = block * BN + tl.arange(0, BN)
    col = tl.program_id(1) * BC + tl.arange(0, BC)
    value = tl.load(X + row[:, None]*W + col[None, :],
                    (row[:, None] < N) & (col[None, :] < W), 0.)
    if FP64:
        value = value.to(tl.float64)
    total = tl.sum(value, 0)
    tl.store(Y + block*W + col, total, col < W)


def _reduce_rows(value, width, accumulation):
    """Fixed 256-row tree; BC=32 is a channel tile, not the device warp width."""
    n = value.shape[0]
    if not n:
        return torch.zeros(width, dtype=value.dtype, device=value.device)
    if n == 1:
        return value.reshape(width)
    dtype = torch.float64 if accumulation == 'fp64' else torch.float32
    while n > 1:
        blocks = triton.cdiv(n, 256)
        output = torch.empty((blocks, width), dtype=dtype, device=value.device)
        _tree_reduce_rows[(blocks, triton.cdiv(width, 32))](
            value, output, n, width, accumulation == 'fp64', 256, 32,
            num_warps=4, enable_fp_fusion=False)
        value, n = output, blocks
    return value.reshape(width).to(torch.float32)


def _sum_n(partials, width, groups=1, group_stride=0, **_):
    # The same public Torch reduction API handles the active GPU backend.
    view = partials.as_strided((partials.shape[0], groups, width),
                              (partials.stride(0), group_stride, 1))
    if groups == 1:
        return view[:, 0].sum(0, keepdim=True)
    return torch.stack([view[:, g].sum(0) for g in range(groups)])

@triton.jit
def _write_parameters(V, S, B, DW0, DWH, DB, C: tl.constexpr,
                       EXPANDED: tl.constexpr, SPLIT: tl.constexpr,
                       W0_GRAD: tl.constexpr, WH_GRAD: tl.constexpr,
                       BIAS_GRAD: tl.constexpr, BC: tl.constexpr):
    l = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    weight_grad = (W0_GRAD & ((l == 0) | (not SPLIT))) | (WH_GRAD & (l > 0))
    if weight_grad:
        if EXPANDED:
            if SPLIT and l == 0:
                value = tl.load(S + c, c < C, 0.)
            else:
                value = tl.full((BC,), 0., tl.float32)
                for k in range(l * l, (l + 1) * (l + 1)):
                    value += tl.load(V + (k - (1 if SPLIT else 0)) * C + c, c < C, 0.)
        else:
            value = tl.load(V + l * C + c, c < C, 0.)
        if SPLIT:
            if l == 0:
                tl.store(DW0 + c, value, c < C)
            else:
                tl.store(DWH + (l - 1) * C + c, value, c < C)
        else:
            tl.store(DW0 + l * C + c, value, c < C)
    if BIAS_GRAD:
        if l == 0:
            tl.store(DB + c, tl.load(B + c, c < C, 0.), c < C)

def reduce_parameters(op, pw, pb, dw0, dwh, db, split, needs, placeholder, dy):
    _, nw0, nwh, nb = needs
    c, lmax = op.spec.channels, op.spec.lmax
    expanded = op.parameter_order == 'expanded'
    value = scalar = bias = placeholder
    if nw0 or nwh:
        if expanded:
            if split:
                if nw0:
                    scalar = _sum_n(pw[:, 0], c)
                if nwh:
                    value = _sum_n(pw[:, 1:], (op.spec.components - 1) * c)
            else:
                value = _sum_n(pw, op.spec.components * c)
        else:
            value = _sum_n(pw, c, lmax + 1, c)
    if nb:
        bias = _sum_n(pb, c, logical_stride=dy.stride(0), logical_ptr=dy.data_ptr(),
                      logical_c_stride=dy.stride(2))
    _write_parameters[(lmax + 1, triton.cdiv(c, 32))](
        value, scalar, bias, dw0 if nw0 else placeholder, dwh if nwh else placeholder,
        db if nb else placeholder, c, expanded, split, nw0, nwh, nb, 32,
        num_warps=4, enable_fp_fusion=False)


def _source_dr(op, x, mean, w0, wh, dy):
    """Source tensor reductions through a backend-independent framework API."""
    n = x.shape[0]
    z = torch.cat((x[:, :1] - mean[:, None, None], x[:, 1:]), 1) if op.spec.center_scalar else x
    result = []
    split = wh is not None
    for a, b in zip(op.group_bounds, op.group_bounds[1:]):
        if op.parameter_order == 'expanded':
            p = dy[:, a:b] * z[:, a:b]
            if w0 is not None:
                weight = torch.cat((w0[None], wh), 0) if split else w0
                p = p * weight.index_select(0, op.degrees[a:b].long())[None]
            dr = p.sum((1, 2))
        else:
            dr = torch.zeros(n, device=x.device, dtype=x.dtype)
            for l in range(op.spec.lmax, -1, -1):
                if a <= l * l < b:
                    p = dy[:, l*l:(l+1)**2] * z[:, l*l:(l+1)**2]
                    if w0 is not None:
                        weight = (w0 if l == 0 else wh[l-1]) if split else w0[l]
                        value = (p.sum(1) * weight).sum(1)
                    else:
                        value = p.sum((1, 2))
                    dr = dr + value
        result.append(dr)
    return torch.stack(result, 1)


def _backward(op, x, mean, rstd, weight0, weight_high, dy, needs, *, return_partials=False):
    with torch.cuda.device(x.device):
        return _backward_on_device(op, x, mean, rstd, weight0, weight_high, dy,
                                   needs, return_partials=return_partials)


def _backward_on_device(op, x, mean, rstd, weight0, weight_high, dy, needs, *, return_partials=False):
    if dy.layout != torch.strided or dy.shape != x.shape or dy.dtype != x.dtype or dy.device != x.device:
        raise ValueError('output gradient must match input shape, device and dtype')
    if any(s < 0 for s in dy.stride()) or sum(max(d-1, 0)*s for d,s in zip(dy.shape,dy.stride())) > 2**31-1:
        raise NotImplementedError('output gradient requires nonnegative 32-bit strides')
    n,k,c=x.shape; s=op.spec
    nx,nw0,nwh,nb=needs; nwh=nwh and s.lmax>0; nw=nw0 or nwh
    split=weight_high is not None
    def empty(shape):return torch.empty(shape,device=x.device,dtype=x.dtype)
    dx=empty(x.shape) if nx else None
    native=(op.parameter_reduction == 'native' and op._from_reference)
    pw=empty((n,k if native and op.parameter_order=='expanded' else s.lmax+1,c)) if nw else None
    pb=empty((n,c)) if nb else None
    def ptr(v):return x if v is None else v
    dr = _source_dr(op, x, mean, weight0, weight_high, dy) if native and nx and n else None
    if n:
        _portable_backward[(n,)](x,dy,ptr(mean),rstd,ptr(dr),ptr(dx),ptr(pw),ptr(pb),op.coefficients,
            op.degrees,ptr(weight0),ptr(weight_high),op.bounds,
            k,c,s.num_stats,s.lmax,x.stride(0),x.stride(1),*dy.stride(),s.center_scalar,
            weight0 is not None,split,weight0.stride(0) if split else 0,
            weight_high.stride(0) if split else weight0.stride(0) if weight0 is not None else 0,
            weight_high.stride(1) if split else weight0.stride(1) if weight0 is not None else 0,
            nx,nw,nb,op.parameter_order=='expanded',native,dr is not None,op.parameter_order!='accurate',op.atom_block_k,op.block_c,
            num_warps=4,enable_fp_fusion=False)
    if native:
        dw0=empty(weight0.shape) if nw0 else None
        dwh=empty(weight_high.shape) if nwh else None
        db=empty((c,)) if nb else None
        reduce_parameters(op,pw,pb,dw0,dwh,db,split,(nx,nw0,nwh,nb),x,dy)
    else:
        dw=_reduce_rows(pw,(s.lmax+1)*c,op.accumulation).view(s.lmax+1,c) if nw else None
        dw0=(dw[0] if split else dw) if nw0 else None
        dwh=dw[1:] if nwh else None
        db=_reduce_rows(pb,c,op.accumulation) if nb else None
    result=(dx,dw0,dwh,db)
    return (result,dict(weight=pw,bias=pb)) if return_partials else result

class _EquivariantNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight0, weight_high, bias, op, save_stats):
        weight = (weight0, weight_high) if weight_high is not None else weight0
        result = op._run(x, weight, bias, save_stats, training=True)
        ctx.op = op
        ctx.save_for_backward(x, result.mean, result.rstd, weight0, weight_high)
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(*(value for value in result[1:] if value is not None))
        return tuple(result)

    @staticmethod
    def backward(ctx, dy, _dmean, _dmoments, _drstd):
        if torch.is_grad_enabled():
            raise RuntimeError('equivariant norm supports first-order gradients only; '
                               'create_graph=True and double backward are not supported')
        if dy is None:
            return (None,) * 6
        x, mean, rstd, weight0, weight_high = ctx.saved_tensors
        gradients = _backward(ctx.op, x, mean, rstd, weight0, weight_high,
                              dy, ctx.needs_input_grad[:4])
        return (*gradients, None, None)

class TritonEquivariantNorm(torch.nn.Module):
    """Execution plan; affine parameters belong to the caller.

    GPU scope: FP32 forward/backward, contiguous NKC / transposed KNC inputs,
    positive-stride packed or split weights. Each statistic must describe one
    contiguous, disjoint range of full degrees and scale exactly that range.
    Broader/overlapping specs remain valid math specs but are rejected here.
    """
    def __init__(self, spec, *, reduction_order='channels_first',
                 device='cuda', parameter_order='accurate', accumulation='fp64'):
        super().__init__()
        if not isinstance(spec, EquivariantNormSpec):
            raise TypeError('spec must be EquivariantNormSpec')
        if reduction_order not in ('components_first', 'channels_first'):
            raise ValueError('unknown reduction order')
        bounds, coefficients, degrees = [0], [], []
        uniform = True
        for g, row in enumerate(spec.stats_weights):
            source = [l for l, a in enumerate(row) if a > 0]
            target = [l for l, h in enumerate(spec.output_group) if h == g]
            if (source != target or source != list(range(source[0], source[-1] + 1))
                    or source[0] ** 2 != bounds[-1]):
                raise NotImplementedError('GPU groups must be ordered contiguous disjoint '
                                          'degree ranges, with identical source/output support')
            bounds.append((source[-1] + 1) ** 2)
            uniform &= len({row[l] for l in source}) == 1
            for l in source:
                coefficients.extend([row[l]] * (2*l + 1))
                degrees.extend([l] * (2*l + 1))
        if bounds[-1] != spec.components:
            raise NotImplementedError('groups must cover all degrees')
        self.spec = spec
        if parameter_order not in ('accurate', 'broadcast', 'expanded'):
            raise ValueError('unknown parameter reduction order')
        if accumulation not in ('fp32', 'fp64'):
            raise ValueError('accumulation must be fp32 or fp64')
        self.accumulation = accumulation
        self.parameter_reduction = 'stable'
        with torch.cuda.device(device):
            self.warp_size = triton.runtime.driver.active.get_current_target().warp_size
        self.parameter_order = parameter_order
        self._from_reference = False
        self.reduction_order = reduction_order
        self.uniform = uniform
        self.group_bounds = tuple(bounds)
        self.atom_block_k = triton.next_power_of_2(spec.components)
        self.block_c = triton.next_power_of_2(spec.channels)
        if self.atom_block_k * self.block_c > 65536:
            raise NotImplementedError('execution tile exceeds 65536 padded elements')
        for name, values, dtype in [('bounds', bounds, torch.int32), ('coefficients', coefficients, torch.float32),
                                    ('degrees', degrees, torch.int32)]:
            self.register_buffer(name, torch.tensor(values, dtype=dtype, device=device),
                                 persistent=False)

    def _check(self, x, weight, bias):
        s = self.spec
        if x.ndim != 3 or tuple(x.shape[1:]) != (s.components, s.channels):
            raise ValueError('expected [N,(lmax+1)^2,channels]')
        if x.device.type != 'cuda' or x.dtype != torch.float32:
            raise TypeError('Triton backend supports GPU FP32 only')
        if x.device != self.coefficients.device or self.coefficients.dtype != torch.float32:
            raise ValueError('move the execution plan to the input device; keep it FP32')
        if not (x.is_contiguous() or x.stride() == (s.channels, x.shape[0]*s.channels, 1)):
            raise ValueError('supported input storage: NKC and KNC')
        if x.numel() > 2**31 - 1:
            raise NotImplementedError('this backend uses 32-bit tensor indexing')
        def check(t, shape):
            if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape:
                raise ValueError(f'expected parameter shape {shape}')
            if t.device != x.device or t.dtype != x.dtype:
                raise ValueError('parameter device and dtype must match input')
            if any(a <= 0 for a in t.stride()):
                raise ValueError('parameter strides must be positive')
            if sum(max(a - 1, 0) * b for a, b in zip(t.shape, t.stride())) > 2**31 - 1:
                raise NotImplementedError('parameter offsets exceed 32-bit indexing')

        split = isinstance(weight, tuple)
        if weight is not None:
            if split:
                if len(weight) != 2:
                    raise ValueError('split weight must be (scalar, higher)')
                check(weight[0], (s.channels,))
                check(weight[1], (s.lmax, s.channels))
            else:
                check(weight, (s.lmax+1, s.channels))
        if bias is not None:
            check(bias, (s.channels,))

    def _run(self, x, weight, bias, save, training=False):
        with torch.cuda.device(x.device):
            self.warp_size = triton.runtime.driver.active.get_current_target().warp_size
            return self._run_on_device(x, weight, bias, save, training)

    def _run_on_device(self, x, weight, bias, save, training=False):
        n,k,c=x.shape; s=self.spec
        y=torch.empty(x.shape,device=x.device,dtype=x.dtype)
        need=save or training
        mu=torch.empty(n,device=x.device,dtype=x.dtype) if need and s.center_scalar else None
        moments=torch.empty((n,s.num_stats),device=x.device,dtype=x.dtype) if save else None
        rstd=torch.empty((n,s.num_stats),device=x.device,dtype=x.dtype) if need else None
        if n:
            split=isinstance(weight,tuple);w0,wh=weight if split else (weight,None)
            def ptr(v):return y if v is None else v
            _fused[(n,)](x,y,ptr(mu),ptr(moments),ptr(rstd),self.coefficients,self.degrees,
                ptr(w0),ptr(wh),ptr(bias),K=k,C=c,G=s.num_stats,SN=x.stride(0),SK=x.stride(1),
                CENTER=s.center_scalar,EPS=s.eps,ORDER=self.reduction_order,UNIFORM=self.uniform,
                HAS_WEIGHT=w0 is not None,SPLIT=split,HAS_BIAS=bias is not None,
                WS0=w0.stride(0) if split else 0,
                WSL=wh.stride(0) if split else w0.stride(0) if w0 is not None else 0,
                WSC=wh.stride(1) if split else w0.stride(1) if w0 is not None else 0,
                BS=bias.stride(0) if bias is not None else 0,SAVE=need,SAVE_MOMENTS=save,
                BK=self.atom_block_k,BC=self.block_c,GROUP_BOUNDS=self.bounds,
                num_warps=4,enable_fp_fusion=False)
        return NormResult(y,mu,moments,rstd) if need else y

    def _dispatch(self, x, weight, bias, save):
        self._check(x, weight, bias)
        weight0, weight_high = weight if isinstance(weight, tuple) else (weight, None)
        tensors = (x, weight0, weight_high, bias)
        if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
            result = NormResult(*_EquivariantNormFunction.apply(*tensors, self, save))
            return result if save else result.output
        return self._run(x, weight, bias, save)

    def forward(self, x, *, weight=None, bias=None):
        return self._dispatch(x, weight, bias, False)

    def forward_with_stats(self, x, *, weight=None, bias=None):
        """Return differentiable output and detached diagnostic statistics."""
        return self._dispatch(x, weight, bias, True)

class _SourceAdapter(torch.nn.Module):
    """Keep source parameter ownership/state keys; never call its forward.

    The registered source means .to() and parameter replacement stay live.
    The adapter's state_dict is namespaced under source.; source.state_dict()
    retains the original class keys. Source parameters receive Triton gradients.
    """
    def __init__(self, source, op):
        super().__init__()
        self.source, self.op = source, op
        self.spec = op.spec

    def _params(self):
        if not self.source.affine:
            return None, None
        if hasattr(self.source, 'norm_l0'):
            return (self.source.norm_l0.weight, self.source.affine_weight), self.source.norm_l0.bias
        return self.source.affine_weight, getattr(self.source, 'affine_bias', None)

    def forward(self, x):
        weight, bias = self._params()
        return self.op(x, weight=weight, bias=bias)

    def forward_with_stats(self, x):
        weight, bias = self._params()
        return self.op.forward_with_stats(x, weight=weight, bias=bias)

def from_reference(source, *, device=None, accumulation="fp64", parameter_reduction="native"):
    """Adapt source equations with the same implementation on every backend.

    Native reductions use Torch APIs; stable reductions use a fixed Triton tree.
    Neither policy selects kernels by CUDA/HIP identity. Native is the default
    for agreement with original Torch FP32 forward and first-order autograd.
    """
    if parameter_reduction not in ('native', 'stable'):
        raise ValueError('parameter_reduction must be native or stable')
    name = type(source).__name__
    if name in ('EquivariantLayerNorm', 'EquivariantLayerNormArray'):
        grouping, order, center = 'per_degree', 'components_first', True
    elif name in ('EquivariantSeparableLayerNorm', 'EquivariantLayerNormArraySphericalHarmonics'):
        grouping, order, center = 'scalar_high', 'components_first', True
        # V3 separable actually reduces channels before weighted components.
        if name == 'EquivariantSeparableLayerNorm':
            order = 'channels_first'
    else:
        raise NotImplementedError(f'no verified source adapter for {name}')
    weighting = ('degree_balanced' if source.normalization == 'component'
                 and getattr(source, 'std_balance_degrees', False) else source.normalization)
    spec = EquivariantNormSpec.from_preset(lmax=source.lmax, channels=source.num_channels,
        grouping=grouping, weighting=weighting, center_scalar=center, eps=source.eps)
    if weighting == 'degree_balanced':
        weights = [list(row) for row in spec.stats_weights]
        cached = source.balance_degree_weight.detach().reshape(-1).cpu()
        for l in range(1, source.lmax+1):
            weights[spec.output_group[l]][l] = float(cached[l*l-1])
        spec = EquivariantNormSpec(spec.lmax, spec.channels, tuple(map(tuple, weights)),
                                   spec.output_group, spec.center_scalar, spec.eps)
    if device is None:
        tensors = list(source.parameters()) + list(source.buffers())
        device = tensors[0].device if tensors else torch.device('cuda')
    parameter_order = ('expanded' if name == 'EquivariantSeparableLayerNorm'
                       else 'broadcast')
    op = TritonEquivariantNorm(spec, reduction_order=order, device=device,
                              parameter_order=parameter_order, accumulation=accumulation)
    op.parameter_reduction = parameter_reduction
    op._from_reference = True
    return _SourceAdapter(source, op)
