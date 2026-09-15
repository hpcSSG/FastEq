"""First-order FP32 gradients for disjoint equivariant normalization groups.

For a component in group g, a is its statistic coefficient, z is the
scalar-centered input, r = rsqrt(moment + eps), and h = dY * gamma:

    zhat = z * r
    Q[g] = sum_group(h * zhat)          # NOT weighted by a
    u = r * (h - (a / C) * zhat * Q[g])
    dX = u, except dX_scalar = u_scalar - mean_c(u_scalar)

The normalized form avoids explicitly forming r**3. Parameter gradients are
sum_(n,m)(dY * zhat) for gamma and sum_n(dY_scalar) for beta. Reductions use
deterministic partial buffers rather than floating-point atomic additions.
"""

import torch
import triton
import triton.language as tl

from ._torch_norm_reduce import reduce_parameters
from ._torch_norm_order import m_config, configured_m_sum
from ._torch_norm_dot import source_r_gradient


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
def _sum_group(value, ORDER: tl.constexpr):
    if ORDER == 'channels_first':
        return tl.sum(tl.sum(value, 1), 0)
    return tl.sum(tl.sum(value, 0), 0)


@triton.jit
def _input_gradient(h, zhat, r, q, w, valid, k, C: tl.constexpr,
                    CENTER: tl.constexpr):
    u = tl.where(valid, r * (h - (w / C) * zhat * q), 0.)
    if CENTER:
        scalar = tl.sum(tl.where(k == 0, u, 0.), 0)
        mean_u = tl.sum(scalar, 0) / C
        u = tl.where(valid, u - tl.where(k == 0, mean_u, 0.), 0.)
    return u


@triton.jit
def _group_dot(X, DY, MU, RSTD, Q, BOUNDS, DEGREE, W0, WH,
               K: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
               SN: tl.constexpr, SK: tl.constexpr,
               DN: tl.constexpr, DK: tl.constexpr, DC: tl.constexpr,
               CENTER: tl.constexpr, HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr,
               WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr,
               ORDER: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
               PARAM_ORDER: tl.constexpr, L: tl.constexpr, GROUP_BOUNDS: tl.constexpr,
               M_CONFIG: tl.constexpr, C_CONFIG: tl.constexpr, Q_CONFIG: tl.constexpr,
               DEGREE_CONFIG: tl.constexpr):
    n, g = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(BOUNDS + g), tl.load(BOUNDS + g + 1)
    k = start + tl.arange(0, BK)[:, None]
    c = tl.arange(0, BC)[None, :]
    valid = (k < end) & (c < C)
    z = tl.load(X + n * SN + k * SK + c, valid, 0.)
    if CENTER:
        mu = tl.load(MU + n)
        z = tl.where(valid, z - tl.where(k == 0, mu, 0.), 0.)
    dy = tl.load(DY + n * DN + k * DK + c * DC, valid, 0.)
    gamma = _gamma(k, c, valid, DEGREE, W0, WH, K, HAS_WEIGHT, SPLIT, WS0, WSL, WSC)
    r = tl.load(RSTD + n * G + g)
    if PARAM_ORDER == 'accurate':
        q = _sum_group(tl.where(valid, (dy * gamma) * (z * r), 0.), ORDER)
    else:
        q = tl.full((), 0., tl.float32)
        for group in tl.static_range(G):
            if g == group:
                q = r * source_r_gradient(z, dy, gamma, n, GROUP_BOUNDS[group],
                    GROUP_BOUNDS[group], GROUP_BOUNDS[group + 1], C, L, PARAM_ORDER,
                    M_CONFIG, C_CONFIG, ((Q_CONFIG >> (12 * group)) & 4095), HAS_WEIGHT, DEGREE_CONFIG)
    tl.store(Q + n * G + g, q)


@triton.jit
def _input_v0(X, DY, MU, RSTD, Q, DX, BOUNDS, COEFFICIENT, DEGREE, W0, WH,
              K: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
              SN: tl.constexpr, SK: tl.constexpr,
              DN: tl.constexpr, DK: tl.constexpr, DC: tl.constexpr,
              CENTER: tl.constexpr, HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr,
              WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr,
              BK: tl.constexpr, BC: tl.constexpr):
    n, g = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(BOUNDS + g), tl.load(BOUNDS + g + 1)
    k = start + tl.arange(0, BK)[:, None]
    c = tl.arange(0, BC)[None, :]
    valid = (k < end) & (c < C)
    z = tl.load(X + n * SN + k * SK + c, valid, 0.)
    if CENTER:
        mu = tl.load(MU + n)
        z = tl.where(valid, z - tl.where(k == 0, mu, 0.), 0.)
    dy = tl.load(DY + n * DN + k * DK + c * DC, valid, 0.)
    gamma = _gamma(k, c, valid, DEGREE, W0, WH, K, HAS_WEIGHT, SPLIT, WS0, WSL, WSC)
    r, q = tl.load(RSTD + n * G + g), tl.load(Q + n * G + g)
    w = tl.load(COEFFICIENT + k, k < end, 0.)
    dx = _input_gradient(dy * gamma, z * r, r, q, w, valid, k, C, CENTER)
    tl.store(DX + n * K * C + k * C + c, dx, valid)


@triton.jit
def _parameter_partial(X, DY, MU, RSTD, PW, PB, GROUP,
                       K: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
                       L: tl.constexpr, SN: tl.constexpr, SK: tl.constexpr,
                       DN: tl.constexpr, DK: tl.constexpr, DC: tl.constexpr,
                       CENTER: tl.constexpr, WEIGHT_GRAD: tl.constexpr,
                       BIAS_GRAD: tl.constexpr, PARAM_ORDER: tl.constexpr, M_CONFIG: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
    n, l = tl.program_id(0), tl.program_id(1)
    m = tl.arange(0, BM)[:, None]
    k = l * l + m
    c = tl.arange(0, BC)[None, :]
    valid = (m < 2 * l + 1) & (c < C)
    dy = tl.load(DY + n * DN + k * DK + c * DC, valid, 0.)
    if WEIGHT_GRAD:
        z = tl.load(X + n * SN + k * SK + c, valid, 0.)
        if CENTER:
            mu = tl.load(MU + n)
            z = tl.where(valid, z - tl.where(k == 0, mu, 0.), 0.)
        g = tl.load(GROUP + l * l)
        r = tl.load(RSTD + n * G + g)
        if PARAM_ORDER == 'expanded':
            tl.store(PW + n * K * C + k * C + c, (dy * z) * r, valid)
        else:
            if PARAM_ORDER == 'broadcast':
                partial = tl.full((BC,), 0., tl.float32)
                for degree in tl.static_range(L + 1):
                    if l == degree:
                        partial = configured_m_sum(tl.where(valid, dy * z, 0.), n, 0, 2 * degree + 1, ((M_CONFIG >> (12 * degree)) & 4095)) * r
            else:
                partial = tl.sum(tl.where(valid, dy * (z * r), 0.), 0)
            tl.store(PW + (n * (L + 1) + l) * C + tl.arange(0, BC), partial,
                     tl.arange(0, BC) < C)
    if BIAS_GRAD:
        if l == 0:
            partial = tl.sum(tl.where(m == 0, dy, 0.), 0)
            tl.store(PB + n * C + tl.arange(0, BC), partial, tl.arange(0, BC) < C)


@triton.jit
def _fused_backward(X, DY, MU, RSTD, DX, PW, PB, COEFFICIENT, DEGREE, W0, WH,
                    K: tl.constexpr, C: tl.constexpr, G: tl.constexpr, L: tl.constexpr,
                    SN: tl.constexpr, SK: tl.constexpr,
                    DN: tl.constexpr, DK: tl.constexpr, DC: tl.constexpr,
                    CENTER: tl.constexpr, HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr,
                    WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr,
                    INPUT_GRAD: tl.constexpr, WEIGHT_GRAD: tl.constexpr,
                    BIAS_GRAD: tl.constexpr, ORDER: tl.constexpr,
                    GROUP_BOUNDS: tl.constexpr, PARAM_ORDER: tl.constexpr, M_CONFIG: tl.constexpr,
                    C_CONFIG: tl.constexpr, Q_CONFIG: tl.constexpr, DEGREE_CONFIG: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr):
    n = tl.program_id(0)
    k = tl.arange(0, BK)[:, None]
    c = tl.arange(0, BC)[None, :]
    valid = (k < K) & (c < C)
    dy = tl.load(DY + n * DN + k * DK + c * DC, valid, 0.)
    if INPUT_GRAD or WEIGHT_GRAD:
        z = tl.load(X + n * SN + k * SK + c, valid, 0.)
        if CENTER:
            mu = tl.load(MU + n)
            z = tl.where(valid, z - tl.where(k == 0, mu, 0.), 0.)
        row_rstd = tl.full((BK, 1), 0., tl.float32)
        for g in tl.static_range(G):
            member = (k >= GROUP_BOUNDS[g]) & (k < GROUP_BOUNDS[g + 1])
            r = tl.load(RSTD + n * G + g)
            row_rstd = tl.where(member, r, row_rstd)
        zhat = tl.where(valid, z * row_rstd, 0.)
    if INPUT_GRAD:
        gamma = _gamma(k, c, valid, DEGREE, W0, WH, K, HAS_WEIGHT, SPLIT, WS0, WSL, WSC)
        h = dy * gamma
        product = h * zhat
        row_q = tl.full((BK, 1), 0., tl.float32)
        for g in tl.static_range(G):
            member = (k >= GROUP_BOUNDS[g]) & (k < GROUP_BOUNDS[g + 1])
            if PARAM_ORDER == 'accurate':
                q = _sum_group(tl.where(member & valid, product, 0.), ORDER)
            else:
                r = tl.load(RSTD + n * G + g)
                q = r * source_r_gradient(z, dy, gamma, n, 0,
                    GROUP_BOUNDS[g], GROUP_BOUNDS[g + 1], C, L, PARAM_ORDER,
                    M_CONFIG, C_CONFIG, ((Q_CONFIG >> (12 * g)) & 4095), HAS_WEIGHT, DEGREE_CONFIG)
            row_q = tl.where(member, q, row_q)
        w = tl.load(COEFFICIENT + k, k < K, 0.)
        dx = _input_gradient(h, zhat, row_rstd, row_q, w, valid, k, C, CENTER)
        tl.store(DX + n * K * C + k * C + c, dx, valid)
    if WEIGHT_GRAD:
        if PARAM_ORDER == 'expanded':
            tl.store(PW + n * K * C + k * C + c, (dy * z) * row_rstd, valid)
        else:
            product = dy * z if PARAM_ORDER == 'broadcast' else dy * zhat
            # Source broadcast: reduce m before multiplying the scale.
            for l in tl.static_range(L + 1):
                member = (k >= l * l) & (k < (l + 1) * (l + 1))
                if PARAM_ORDER == 'broadcast':
                    partial = configured_m_sum(tl.where(valid, product, 0.), n, l * l, 2 * l + 1, ((M_CONFIG >> (12 * l)) & 4095))
                    partial *= tl.sum(tl.where(k == l * l, row_rstd, 0.), 0)
                else:
                    partial = tl.sum(tl.where(member & valid, product, 0.), 0)
                tl.store(PW + (n * (L + 1) + l) * C + tl.arange(0, BC), partial,
                         tl.arange(0, BC) < C)
    if BIAS_GRAD:
        partial = tl.sum(tl.where(k == 0, dy, 0.), 0)
        tl.store(PB + n * C + tl.arange(0, BC), partial, tl.arange(0, BC) < C)


@triton.jit
def _parameter_reduce(PW, PB, DW0, DWH, DB, N: tl.constexpr, L: tl.constexpr,
                      C: tl.constexpr, SPLIT: tl.constexpr,
                      W0_GRAD: tl.constexpr, WH_GRAD: tl.constexpr,
                      BIAS_GRAD: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr):
    l = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    rows = tl.arange(0, BN)
    weight_grad = (W0_GRAD & ((l == 0) | (not SPLIT))) | (WH_GRAD & (l > 0))
    # Each lane accumulates ceil(N / BN) atom contributions. FP32 loses small
    # terms when large positive and negative partials cancel at large N.
    # Widen only the accumulator; partial buffers and returned grads stay FP32.
    if weight_grad:
        acc = tl.full((BN, BC), 0., tl.float64)
        for start in range(0, N, BN):
            n = start + rows
            value = tl.load(PW + (n[:, None] * (L + 1) + l) * C + c[None, :],
                            (n[:, None] < N) & (c[None, :] < C), 0.)
            acc += value
        total = tl.sum(acc, 0)
        if SPLIT:
            if l == 0:
                tl.store(DW0 + c, total, c < C)
            else:
                tl.store(DWH + (l - 1) * C + c, total, c < C)
        else:
            tl.store(DW0 + l * C + c, total, c < C)
    if BIAS_GRAD:
        if l == 0:
            acc = tl.full((BN, BC), 0., tl.float64)
            for start in range(0, N, BN):
                n = start + rows
                value = tl.load(PB + n[:, None] * C + c[None, :],
                                (n[:, None] < N) & (c[None, :] < C), 0.)
                acc += value
            tl.store(DB + c, tl.sum(acc, 0), c < C)


def backward(op, x, mean, rstd, weight0, weight_high, dy, needs):
    """Allocate gradients and dispatch kernels; no PyTorch gradient arithmetic."""
    if dy.layout != torch.strided or dy.shape != x.shape:
        raise ValueError('expected a strided output gradient with the output shape')
    if dy.dtype != x.dtype or dy.device != x.device:
        raise ValueError('output gradient must match the input device and dtype')
    if any(s < 0 for s in dy.stride()) or sum(
            max(d - 1, 0) * s for d, s in zip(dy.shape, dy.stride())) > 2**31 - 1:
        raise NotImplementedError('output gradient requires nonnegative 32-bit strides')
    s = op.spec
    n, k, c = x.shape
    if n and op.parameter_order != 'accurate' and torch.version.hip is not None:
        raise NotImplementedError('source-ordered backward currently implements the PyTorch CUDA '
                                  'reduction schedule; ROCm requires its own validated schedule')
    split = weight_high is not None
    nx, nw0, nwh, nb = needs
    # The empty higher-degree parameter of an lmax=0 source is unused.
    nwh = nwh and s.lmax > 0
    nw = nw0 or nwh

    def empty(shape):
        return torch.empty(shape, device=x.device, dtype=x.dtype)

    dx = empty(x.shape) if nx else None
    dw0 = empty(weight0.shape) if nw0 else None
    dwh = empty(weight_high.shape) if nwh else None
    db = empty((c,)) if nb else None
    pw = empty((n, k if op.parameter_order == 'expanded' else s.lmax + 1, c)) if nw else None
    pb = empty((n, c)) if nb else None
    # These placeholders are eliminated by constexpr flags before dereferencing.
    def ptr(value):
        return x if value is None else value
    common = dict(K=k, C=c, G=s.num_stats, SN=x.stride(0), SK=x.stride(1),
                  DN=dy.stride(0), DK=dy.stride(1), DC=dy.stride(2),
                  CENTER=s.center_scalar, BC=op.block_c)
    affine = dict(HAS_WEIGHT=weight0 is not None, SPLIT=split,
                  WS0=weight0.stride(0) if split else 0,
                  WSL=weight_high.stride(0) if split else weight0.stride(0) if weight0 is not None else 0,
                  WSC=weight_high.stride(1) if split else weight0.stride(1) if weight0 is not None else 0)
    def config_code(cfg):
        return (int(cfg['fastest']) | (int(cfg['vectorize']) << 1)
                | ((cfg['block_width'].bit_length() - 1) << 2)
                | ((cfg['reduction_height'].bit_length() - 1) << 6))
    m_configs = c_config = q_configs = degree_configs = 0
    if n and op.parameter_order == 'broadcast':
        m_configs = sum(config_code(m_config(n, 2*l + 1, c)) << (12*l) for l in range(s.lmax + 1))
    if n and op.parameter_order != 'accurate' and nx:
        c_config = config_code(m_config(n, c, 1))
        if op.parameter_order == 'expanded':
            q_configs = sum(config_code(m_config(n, (b-a)*c, 1)) << (12*g)
                            for g, (a, b) in enumerate(zip(op.group_bounds, op.group_bounds[1:])))
        elif weight0 is None:
            degree_configs = sum(config_code(m_config(n, (2*l+1)*c, 1)) << (12*l)
                                 for l in range(s.lmax + 1))
    launch = dict(num_warps=4, enable_fp_fusion=False)
    with torch.cuda.device(x.device):
        if n:
            if op.version == 'v0':
                if nx:
                    q = empty((n, s.num_stats))
                    _group_dot[(n, s.num_stats)](x, dy, ptr(mean), rstd, q, op.bounds,
                        op.degrees, ptr(weight0), ptr(weight_high), **common, **affine,
                        ORDER=op.reduction_order, BK=op.block_k, PARAM_ORDER=op.parameter_order,
                        L=s.lmax, GROUP_BOUNDS=op.group_bounds, M_CONFIG=m_configs,
                        C_CONFIG=c_config, Q_CONFIG=q_configs, DEGREE_CONFIG=degree_configs, **launch)
                    _input_v0[(n, s.num_stats)](x, dy, ptr(mean), rstd, q, dx, op.bounds,
                        op.coefficients, op.degrees, ptr(weight0), ptr(weight_high),
                        **common, **affine, BK=op.block_k, **launch)
                if nw or nb:
                    _parameter_partial[(n, s.lmax + 1)](x, dy, ptr(mean), rstd, ptr(pw),
                        ptr(pb), op.groups, **common, L=s.lmax, WEIGHT_GRAD=nw, BIAS_GRAD=nb,
                        PARAM_ORDER=op.parameter_order, M_CONFIG=m_configs, BM=triton.next_power_of_2(2 * s.lmax + 1), **launch)
            elif nx or nw or nb:
                _fused_backward[(n,)](x, dy, ptr(mean), rstd, ptr(dx), ptr(pw), ptr(pb),
                    op.coefficients, op.degrees, ptr(weight0), ptr(weight_high),
                    **common, **affine, L=s.lmax, INPUT_GRAD=nx, WEIGHT_GRAD=nw,
                    BIAS_GRAD=nb, ORDER=op.reduction_order, GROUP_BOUNDS=op.group_bounds,
                    PARAM_ORDER=op.parameter_order, M_CONFIG=m_configs,
                    C_CONFIG=c_config, Q_CONFIG=q_configs, DEGREE_CONFIG=degree_configs, BK=op.atom_block_k, **launch)
        if n and (nw or nb) and op.parameter_order != 'accurate':
            reduce_parameters(op, pw, pb, dw0, dwh, db, split, (nx, nw0, nwh, nb), x, dy)
        elif nw or nb:
            _parameter_reduce[(s.lmax + 1, triton.cdiv(c, 32))](ptr(pw), ptr(pb),
                ptr(dw0), ptr(dwh), ptr(db), n, s.lmax, c, split, nw0, nwh, nb,
                BN=32, BC=32, **launch)
    return dx, dw0, dwh, db
