"""FP32 attention logits with Triton forward and first-order backward."""
import math
import torch
import triton
import triton.language as tl


# Accumulator count is a numerical schedule, not a hardware warp width.
# The gfx936 Torch FP32 contraction needs a sequential chain in the reproduced
# cancellation cases. All targets still execute the same Triton kernel.
_WEIGHT_GRAD_ACCUMULATORS = {"gfx936": 1}


def _row_grid(programs):
    # Bound every launch dimension independently of a device's warp width.
    x = min(programs, 65535)
    return (x, triton.cdiv(programs, x))


@triton.jit
def _forward(X, W, G, B, SEED, MASK, Y, U, A, RS,
             R: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
             LN: tl.constexpr, HAS_G: tl.constexpr, HAS_B: tl.constexpr,
             EPS: tl.constexpr, SLOPE: tl.constexpr, SILU: tl.constexpr,
             P: tl.constexpr, SAVE_U: tl.constexpr, SAVE_A: tl.constexpr,
             SAVE_RS: tl.constexpr, SAVE_MASK: tl.constexpr,
             BC: tl.constexpr, BR: tl.constexpr):
    pid = (tl.program_id(0).to(tl.int64)
           + tl.program_id(1).to(tl.int64) * tl.num_programs(0))
    row = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, BC)
    valid = (c[None, :] < C) & (row[:, None] < R)
    off = row[:, None] * C + c[None, :]
    x = tl.load(X + off, valid, other=0)
    u = x
    rs = tl.full((BR,), 1, tl.float32)
    if LN:
        mean = tl.sum(x, 1) / C
        centered = tl.where(c[None, :] < C, x - mean[:, None], 0.0)
        var = tl.sum(centered * centered, 1) / C
        rs = tl.rsqrt(var + EPS)
        u = (x - mean[:, None]) * rs[:, None]
    z = u
    if LN and HAS_G:
        z = z * tl.load(G + c, c < C, other=1)[None, :]
    if LN and HAS_B:
        z = z + tl.load(B + c, c < C, other=0)[None, :]
    s = tl.sigmoid(z)
    if SILU:
        a = z * s
    else:
        a = ((1.0 + SLOPE) / 2.0) * z + ((1.0 - SLOPE) / 2.0) * z * (2.0 * s - 1.0)
    if P > 0.0:
        if P == 1.0:
            keep = tl.full((BR, BC), False, tl.int1)
            a = a * 0.0
        else:
            seed = tl.load(SEED).to(tl.uint32)
            keep = tl.rand(seed, off.to(tl.uint32)) >= P
            a = a * tl.where(keep, 1.0 / (1.0 - P), 0.0)
        if SAVE_MASK:
            tl.store(MASK + off, keep, valid)
    w = tl.load(W + (row[:, None] % H) * C + c[None, :], c[None, :] < C, other=0)
    result = tl.sum(tl.where(c[None, :] < C, a * w, 0.0), 1)
    tl.store(Y + row, result, row < R)
    if SAVE_U:
        tl.store(U + off, x - mean[:, None], valid)
    if SAVE_A:
        tl.store(A + off, a, valid)
    if SAVE_RS:
        tl.store(RS + row, rs, row < R)


@triton.jit
def _select_row(values, index: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    # Static binary slicing works on Triton 3.1 as well as newer compilers.
    # Unlike masked sums it does not turn every extraction into a reduction.
    tl.static_assert(ROWS <= 64 and (ROWS & (ROWS - 1)) == 0)
    result = values
    for shift in tl.static_range(6):
        if ROWS > (1 << shift):
            paired = tl.permute(result.reshape((ROWS >> (shift + 1), 2, COLS)), (0, 2, 1))
            even, odd = tl.split(paired)
            if (index & (1 << shift)) == 0:
                result = even
            else:
                result = odd
    return result.reshape((COLS,))


@triton.jit
def _backward_rows(DY, W, G, B, U, MASK, RS, DX, PG, PB,
                   R: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
                   LN: tl.constexpr, HAS_G: tl.constexpr, HAS_B: tl.constexpr,
                   SILU: tl.constexpr, SLOPE: tl.constexpr, P: tl.constexpr,
                   NEED_X: tl.constexpr, NEED_G: tl.constexpr, NEED_B: tl.constexpr,
                   BC: tl.constexpr, BR: tl.constexpr):
    pid = (tl.program_id(0).to(tl.int64)
           + tl.program_id(1).to(tl.int64) * tl.num_programs(0))
    row = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, BC)
    valid = (row[:, None] < R) & (c[None, :] < C)
    off = row[:, None] * C + c[None, :]
    centered = tl.load(U + off, valid, other=0)
    u = centered
    if LN:
        rs = tl.load(RS + row, row < R, other=0)
        u = centered * rs[:, None]
    z = u
    if LN and HAS_G:
        g = tl.load(G + c, c < C, other=0)
        z = z * g[None, :]
    if LN and HAS_B:
        z = z + tl.load(B + c, c < C, other=0)[None, :]
    w = tl.load(W + (row[:, None] % H) * C + c[None, :], c[None, :] < C, other=0)
    dy = tl.load(DY + row, row < R, other=0)
    da = dy[:, None] * w
    if P > 0.0:
        if P == 1.0:
            da = da * 0.0
        else:
            keep = tl.load(MASK + off, valid, other=0)
            da = da * tl.where(keep, 1.0 / (1.0 - P), 0.0)
    s = tl.sigmoid(z)
    if SILU:
        dz = (da * s) * (1.0 + z * (1.0 - s))
    else:
        k1 = (1.0 + SLOPE) / 2.0
        k2 = (1.0 - SLOPE) / 2.0
        ds = (da * (k2 * z)) * 2.0
        dz = (ds * (1.0 - s)) * s + (da * (2.0 * s - 1.0)) * k2 + da * k1
    dz = tl.where(valid, dz, 0.0)
    if NEED_G or NEED_B:
        local_g = tl.full((BC,), 0, tl.float32)
        local_b = tl.full((BC,), 0, tl.float32)
        for i in tl.static_range(BR):
            gi = _select_row(dz, i, BR, BC)
            if NEED_G:
                ci = _select_row(centered, i, BR, BC)
                ri = tl.sum(_select_row(rs[:, None], i, BR, 1), 0)
                local_g = tl.fma(gi * ci, ri, local_g)
            if NEED_B:
                local_b += gi
        if NEED_G:
            tl.store(PG + pid * C + c, local_g, c < C)
        if NEED_B:
            tl.store(PB + pid * C + c, local_b, c < C)
    if NEED_X:
        dx = dz
        if LN:
            rs = tl.load(RS + row, row < R, other=0)
            if HAS_G:
                dx = dx * g[None, :]
            mean_dx = tl.sum(dx, 1) / C
            mean_dxu = tl.sum(dx * u, 1) / C
            dx = (dx - mean_dx[:, None] - u * mean_dxu[:, None]) * rs[:, None]
        tl.store(DX + off, dx, valid)


@triton.jit
def _load_weight_pair(A, DY, edge, head, c, E: tl.constexpr,
                      H: tl.constexpr, C: tl.constexpr):
    valid = (edge < E) & (head < H)
    a = tl.load(A + (edge * H + head) * C + c, valid, other=0)
    dy = tl.load(DY + edge * H + head, valid, other=0)
    return a, dy


@triton.jit
def _weight_gradient(A, DY, DW, E: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
                     LANES: tl.constexpr):
    pid = (tl.program_id(0).to(tl.int64)
           + tl.program_id(1).to(tl.int64) * tl.num_programs(0))
    head, c = pid // C, pid % C
    # A fixed number of independent FP32 chains, selected by the host schedule.
    lane = tl.arange(0, LANES)
    acc = tl.full((LANES,), 0, tl.float32)
    for start in range(tl.cdiv(E, 8 * LANES)):
        edge = start.to(tl.int64) * (8 * LANES) + lane
        # Keep each prefetched round in its own lane vector. Extracting rows
        # from a [8, LANES] tile introduces repeated LDS transfers on HCU 3.1.
        # Explicit vectors preserve the exact per-lane FP32 FMA sequence.
        a0, g0 = _load_weight_pair(A, DY, edge, head, c, E, H, C)
        a1, g1 = _load_weight_pair(A, DY, edge + LANES, head, c, E, H, C)
        a2, g2 = _load_weight_pair(A, DY, edge + 2 * LANES, head, c, E, H, C)
        a3, g3 = _load_weight_pair(A, DY, edge + 3 * LANES, head, c, E, H, C)
        a4, g4 = _load_weight_pair(A, DY, edge + 4 * LANES, head, c, E, H, C)
        a5, g5 = _load_weight_pair(A, DY, edge + 5 * LANES, head, c, E, H, C)
        a6, g6 = _load_weight_pair(A, DY, edge + 6 * LANES, head, c, E, H, C)
        a7, g7 = _load_weight_pair(A, DY, edge + 7 * LANES, head, c, E, H, C)
        acc = tl.fma(a0, g0, acc)
        acc = tl.fma(a1, g1, acc)
        acc = tl.fma(a2, g2, acc)
        acc = tl.fma(a3, g3, acc)
        acc = tl.fma(a4, g4, acc)
        acc = tl.fma(a5, g5, acc)
        acc = tl.fma(a6, g6, acc)
        acc = tl.fma(a7, g7, acc)
    tl.store(DW + head * C + c, tl.sum(acc, 0), head < H)


@triton.jit
def _reduce_chunks(PART, OUT, K: tl.constexpr, D: tl.constexpr,
                   BK: tl.constexpr, BD: tl.constexpr):
    pid = (tl.program_id(0).to(tl.int64)
           + tl.program_id(1).to(tl.int64) * tl.num_programs(0))
    tiles = tl.cdiv(D, BD)
    chunk = pid // tiles
    d = (pid % tiles) * BD + tl.arange(0, BD)
    k = chunk * BK + tl.arange(0, BK)
    values = tl.load(PART + k[:, None] * D + d[None, :],
                     (k[:, None] < K) & (d[None, :] < D), other=0)
    tl.store(OUT + chunk * D + d, tl.sum(values.to(tl.float64), 0),
             (d < D) & (chunk < tl.maximum(1, tl.cdiv(K, BK))))


def _sum_partials(partial, out):
    count, width = partial.shape
    # Widen only the cross-tile sum: FP32 partials can nearly cancel.
    # Each level has a fixed tree; no floating-point atomics or device branches.
    while True:
        blocks = max(1, triton.cdiv(count, 256))
        target = out if blocks == 1 else torch.empty((blocks, width), device=out.device, dtype=torch.float64)
        _reduce_chunks[_row_grid(triton.cdiv(width, 32) * blocks)](
            partial, target, count, width, 256, 32, num_warps=4, enable_fp_fusion=False)
        if blocks == 1:
            return
        partial, count = target, blocks


class _FusedAlpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, gamma, beta, ln, eps, silu, slope, p, seed, needs_backward):
        n, h, c = x.shape
        rows = n * h
        need_x, need_w, need_g, need_b = (ctx.needs_input_grad[:4]
                                        if needs_backward else (False,) * 4)
        need_u = need_x or need_g or need_b
        y = torch.empty((n, h), device=x.device, dtype=x.dtype)
        save_mask = p > 0 and needs_backward and any(ctx.needs_input_grad[:4])
        mask = torch.empty(x.shape if save_mask else (0,), device=x.device, dtype=torch.bool)
        u = torch.empty_like(x) if ln and need_u else x
        a = torch.empty_like(x) if need_w else x
        rs = torch.empty((rows,), device=x.device, dtype=x.dtype) if ln and need_u else x
        br = 32 if c <= 128 else (4 if c <= 1024 else 1)
        if rows:
            with torch.cuda.device(x.device):
                _forward[_row_grid(triton.cdiv(rows, br))](
                    x, w, gamma if gamma is not None else x, beta if beta is not None else x,
                    seed if seed is not None else x, mask, y, u, a, rs,
                    rows, h, c, ln, gamma is not None, beta is not None,
                    eps, slope, silu, p, ln and need_u, need_w, ln and need_u,
                    save_mask, triton.next_power_of_2(c), br,
                    num_warps=4 if c <= 1024 else 8, enable_fp_fusion=False)
        if needs_backward:
            ctx.save_for_backward(w, gamma, beta, u, a, rs, mask)
        ctx.config = x.shape, ln, silu, slope, p, br
        return y

    @staticmethod
    def backward(ctx, grad_y):
        if torch.is_grad_enabled():
            raise NotImplementedError("AttentionAlpha Triton backward supports first-order gradients only")
        w, gamma, beta, u, a, rs, mask = ctx.saved_tensors
        (n, h, c), ln, silu, slope, p, br = ctx.config
        rows = n * h
        need_x, need_w, need_g, need_b = ctx.needs_input_grad[:4]
        dy = grad_y.contiguous()
        dx = torch.empty_like(u) if need_x else None
        dw = torch.empty_like(w) if need_w else None
        dg = torch.empty((c,), device=w.device, dtype=w.dtype) if need_g else None
        db = torch.empty((c,), device=w.device, dtype=w.dtype) if need_b else None
        programs = triton.cdiv(rows, br)
        # Masked programs in the rectangular grid write zero partials.
        padded = math.prod(_row_grid(programs)) if programs else 0
        pg = torch.empty((padded, c), device=w.device, dtype=w.dtype) if need_g else u
        pb = torch.empty((padded, c), device=w.device, dtype=w.dtype) if need_b else u
        with torch.cuda.device(w.device):
            if rows and (need_x or need_g or need_b):
                _backward_rows[_row_grid(programs)](
                    dy, w, gamma if gamma is not None else u, beta if beta is not None else u,
                    u, mask, rs, dx if dx is not None else u, pg, pb,
                    rows, h, c, ln, gamma is not None, beta is not None,
                    silu, slope, p, need_x, need_g, need_b, triton.next_power_of_2(c), br,
                    num_warps=4 if c <= 1024 else 8, enable_fp_fusion=False)
            if need_w:
                target = triton.runtime.driver.active.get_current_target()
                lanes = _WEIGHT_GRAD_ACCUMULATORS.get(target.arch, target.warp_size)
                # The lane vectors fit within one target warp/wavefront.
                _weight_gradient[_row_grid(h * c)](
                    a, dy, dw, n, h, c, lanes, num_warps=1, enable_fp_fusion=False)
            if need_g:
                _sum_partials(pg, dg)
            if need_b:
                _sum_partials(pb, db)
        return dx, dw, dg, db, None, None, None, None, None, None, None


def fused_attention_alpha(x, alpha_dot, norm_weight=None, norm_bias=None, *,
                          use_layer_norm=True, eps=1e-5, activation='silu',
                          negative_slope=0.2, dropout_p=0.0, training=True,
                          seed=None):
    """FP32 CUDA/HIP [E,H,C] -> [E,H], with a Triton first-order backward.

    Double backward is unsupported and rejected; no Torch recomputation is used.

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
                            float(negative_slope), p, seed, torch.is_grad_enabled())


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
