import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


# ============================================================================
# Optimized fused S2 activation
#
# separable=False:
#   X [N,J,C]
#       -> T @ X                (to_grid, tl.dot)
#       -> SiLU
#       -> FM @ grid            (from_grid, tl.dot)
#   Y [N,J,C]
#
# separable=True:
#   X [N,J,2C]
#       -> T @ X[..., :C]       (tl.dot)
#       -> T @ X[..., C:]       (tl.dot)
#       -> u * v
#       -> optional fused grid dropout
#       -> FM @ grid            (tl.dot)
#   Y [N,J,C]
#
# Matrix layout:
#   T  = to_grid_mat   [A, J]
#   FM = from_grid_mat [J, A]
#
# Performance strategy:
#   * one Triton program owns one (n, C-tile) and all J rows;
#   * J is padded to BLOCK_J (16/32/64) and evaluated by tl.dot;
#   * A is tiled (16/32/64), so the full grid dimension is never held live;
#   * X1/X2 are loaded once per program and reused for every A tile;
#   * no x_grid tensor and no torch.chunk tensor are materialized;
#   * backward uses the same A-tiled tl.dot structure and needs no atomics.
#
# This fast path is intended primarily for the small-J SO(3) layouts used by
# Equiformer-style models, e.g. J=19 for lmax=4,mmax=2.
# ============================================================================


def _dot_configs():
    # BLOCK_A/BLOCK_C are deliberately >=16 so every tl.dot dimension satisfies
    # MMA/Tensor-Core-friendly tile constraints.  32x32 is the safest default;
    # autotune can choose wider C tiles when channel count/occupancy allow it.
    return [
        triton.Config({"BLOCK_A": 16, "BLOCK_C": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_A": 32, "BLOCK_C": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_A": 16, "BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_A": 32, "BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_A": 64, "BLOCK_C": 32}, num_warps=8, num_stages=2),
    ]


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _silu_grad(x):
    sig = tl.sigmoid(x)
    return sig + x * sig * (1.0 - sig)


@triton.jit
def _dropout_scale_tile(
    n,
    offs_a,
    offs_c,
    A: tl.constexpr,
    C: tl.constexpr,
    drop_p,
    seed,
):
    # Deterministic mapping (n,a,c) -> RNG offset, so backward reconstructs
    # exactly the same mask without storing an [N,A,C] mask tensor.
    offs = (
        n * (A * C)
        + offs_a[:, None] * C
        + offs_c[None, :]
    ).to(tl.uint32)
    rnd = tl.rand(seed, offs)
    keep = rnd > drop_p
    return keep.to(tl.float32) / (1.0 - drop_p)


@triton.autotune(
    configs=_dot_configs(),
    key=["J", "A", "C", "BLOCK_J", "SEPARABLE", "USE_DROPOUT"],
)
@triton.jit(do_not_specialize=["N"])
def _s2_dot_fwd_kernel(
    X,
    T,
    FM,
    Y,
    N,
    drop_p,
    seed,
    J: tl.constexpr,
    A: tl.constexpr,
    C: tl.constexpr,
    BLOCK_J: tl.constexpr,
    SEPARABLE: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1)

    offs_j = tl.arange(0, BLOCK_J)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_j = offs_j < J
    mask_c = offs_c < C
    mask_jc = mask_j[:, None] & mask_c[None, :]

    # ------------------------------------------------------------------
    # Load X once.  Logical chunk is encoded only by address offset.
    # x1/x2: [BLOCK_J, BLOCK_C]
    # ------------------------------------------------------------------
    if SEPARABLE:
        stride_xj = 2 * C
        x_base = X + n * J * stride_xj

        x1 = tl.load(
            x_base + offs_j[:, None] * stride_xj + offs_c[None, :],
            mask=mask_jc,
            other=0.0,
        )
        x2 = tl.load(
            x_base + offs_j[:, None] * stride_xj + C + offs_c[None, :],
            mask=mask_jc,
            other=0.0,
        )
    else:
        stride_xj = C
        x_base = X + n * J * stride_xj
        x1 = tl.load(
            x_base + offs_j[:, None] * stride_xj + offs_c[None, :],
            mask=mask_jc,
            other=0.0,
        )

    # One accumulator holds all output J rows for this channel tile.
    y = tl.zeros((BLOCK_J, BLOCK_C), dtype=tl.float32)

    # ------------------------------------------------------------------
    # A-tiled pipeline:
    #   u/v = T[a,:] @ X
    #   grid = activation(u[,v])
    #   y += FM[:,a] @ grid
    # ------------------------------------------------------------------
    for a0 in tl.static_range(0, A, BLOCK_A):
        offs_a = a0 + tl.arange(0, BLOCK_A)
        mask_a = offs_a < A

        # T tile: [BLOCK_A, BLOCK_J]
        t = tl.load(
            T + offs_a[:, None] * J + offs_j[None, :],
            mask=mask_a[:, None] & mask_j[None, :],
            other=0.0,
        )

        # to_grid using dot/MMA instead of a Python/static J-loop of FMAs.
        u = tl.dot(t, x1)

        if SEPARABLE:
            v = tl.dot(t, x2)
            grid_val = u * v

            if USE_DROPOUT:
                dscale = _dropout_scale_tile(
                    n,
                    offs_a,
                    offs_c,
                    A=A,
                    C=C,
                    drop_p=drop_p,
                    seed=seed,
                )
                grid_val *= dscale
        else:
            grid_val = _silu(u)

        # FM tile: [BLOCK_J, BLOCK_A]
        fm = tl.load(
            FM + offs_j[:, None] * A + offs_a[None, :],
            mask=mask_j[:, None] & mask_a[None, :],
            other=0.0,
        )

        # from_grid and accumulate across A tiles.
        y += tl.dot(fm, grid_val)

    tl.store(
        Y + n * J * C + offs_j[:, None] * C + offs_c[None, :],
        y,
        mask=mask_jc,
    )


@triton.autotune(
    configs=_dot_configs(),
    key=["J", "A", "C", "BLOCK_J", "SEPARABLE", "USE_DROPOUT"],
)
@triton.jit(do_not_specialize=["N"])
def _s2_dot_bwd_kernel(
    X,
    T,
    FM,
    DY,
    DX,
    N,
    drop_p,
    seed,
    J: tl.constexpr,
    A: tl.constexpr,
    C: tl.constexpr,
    BLOCK_J: tl.constexpr,
    SEPARABLE: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1)

    offs_j = tl.arange(0, BLOCK_J)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_j = offs_j < J
    mask_c = offs_c < C
    mask_jc = mask_j[:, None] & mask_c[None, :]

    # dY [BLOCK_J, BLOCK_C] is reused for every A tile.
    dy = tl.load(
        DY + n * J * C + offs_j[:, None] * C + offs_c[None, :],
        mask=mask_jc,
        other=0.0,
    )

    # X is also reused for every A tile.
    if SEPARABLE:
        stride_xj = 2 * C
        x_base = X + n * J * stride_xj
        x1 = tl.load(
            x_base + offs_j[:, None] * stride_xj + offs_c[None, :],
            mask=mask_jc,
            other=0.0,
        )
        x2 = tl.load(
            x_base + offs_j[:, None] * stride_xj + C + offs_c[None, :],
            mask=mask_jc,
            other=0.0,
        )
        dx1_acc = tl.zeros((BLOCK_J, BLOCK_C), dtype=tl.float32)
        dx2_acc = tl.zeros((BLOCK_J, BLOCK_C), dtype=tl.float32)
    else:
        stride_xj = C
        x_base = X + n * J * stride_xj
        x1 = tl.load(
            x_base + offs_j[:, None] * stride_xj + offs_c[None, :],
            mask=mask_jc,
            other=0.0,
        )
        dx1_acc = tl.zeros((BLOCK_J, BLOCK_C), dtype=tl.float32)

    # ------------------------------------------------------------------
    # For each A tile:
    #   r      = FM_tile^T @ dY
    #   u/v    = T_tile @ X
    #   du/dv  = local activation backward
    #   dX    += T_tile^T @ du/dv
    # ------------------------------------------------------------------
    for a0 in tl.static_range(0, A, BLOCK_A):
        offs_a = a0 + tl.arange(0, BLOCK_A)
        mask_a = offs_a < A

        # Load FM^T directly as [BA, BJ], avoiding an explicit layout transpose.
        fm_t = tl.load(
            FM + offs_j[None, :] * A + offs_a[:, None],
            mask=mask_a[:, None] & mask_j[None, :],
            other=0.0,
        )

        # r = FM^T @ dY : [BA, BC]
        r = tl.dot(fm_t, dy)

        # T [BA, J]
        t = tl.load(
            T + offs_a[:, None] * J + offs_j[None, :],
            mask=mask_a[:, None] & mask_j[None, :],
            other=0.0,
        )

        u = tl.dot(t, x1)

        if SEPARABLE:
            v = tl.dot(t, x2)

            if USE_DROPOUT:
                dscale = _dropout_scale_tile(
                    n,
                    offs_a,
                    offs_c,
                    A=A,
                    C=C,
                    drop_p=drop_p,
                    seed=seed,
                )
                r *= dscale

            du = r * v
            dv = r * u

            # Load T^T directly as [BJ, BA] for dX accumulation.
            t_t = tl.load(
                T + offs_a[None, :] * J + offs_j[:, None],
                mask=mask_j[:, None] & mask_a[None, :],
                other=0.0,
            )
            dx1_acc += tl.dot(t_t, du)
            dx2_acc += tl.dot(t_t, dv)
        else:
            du = r * _silu_grad(u)
            t_t = tl.load(
                T + offs_a[None, :] * J + offs_j[:, None],
                mask=mask_j[:, None] & mask_a[None, :],
                other=0.0,
            )
            dx1_acc += tl.dot(t_t, du)

    # ------------------------------------------------------------------
    # Store dX.  No atomic needed: this program owns all J rows for this
    # (n, C-tile), and has already reduced over all A tiles locally.
    # ------------------------------------------------------------------
    if SEPARABLE:
        dx_base = DX + n * J * (2 * C)
        tl.store(
            dx_base + offs_j[:, None] * (2 * C) + offs_c[None, :],
            dx1_acc,
            mask=mask_jc,
        )
        tl.store(
            dx_base + offs_j[:, None] * (2 * C) + C + offs_c[None, :],
            dx2_acc,
            mask=mask_jc,
        )
    else:
        tl.store(
            DX + n * J * C + offs_j[:, None] * C + offs_c[None, :],
            dx1_acc,
            mask=mask_jc,
        )


def _choose_block_j(J: int) -> int:
    # tl.dot requires a reasonably sized K/M dimension.  Padding J=19 -> 32 is
    # intentional.  The current optimized path targets J<=64, which covers the
    # common lmax/mmax settings for the intended Equiformer use case.
    if J <= 16:
        return 16
    if J <= 32:
        return 32
    if J <= 64:
        return 64
    raise ValueError(
        f"Optimized fused S2 tl.dot path currently supports J<=64, got J={J}. "
        "Use a separate large-J kernel/fallback for this configuration."
    )


def _expected_j(lmax: int, mmax):
    if mmax is None:
        return (lmax + 1) ** 2
    if not isinstance(mmax, int) or not (0 <= mmax <= lmax):
        raise ValueError(f"mmax must satisfy 0 <= mmax <= lmax; got {mmax}.")
    return sum(2 * min(l, mmax) + 1 for l in range(lmax + 1))


def _validate(
    inputs,
    to_grid_mat,
    from_grid_mat,
    separable,
    dropout_p,
    lmax,
    mmax,
):
    tensors = (inputs, to_grid_mat, from_grid_mat)

    if not all(torch.is_tensor(t) for t in tensors):
        raise TypeError("inputs/to_grid_mat/from_grid_mat must be tensors.")
    if not all(t.is_cuda for t in tensors):
        raise ValueError("inputs/to_grid_mat/from_grid_mat must all be CUDA tensors.")
    if not all(t.device == inputs.device for t in tensors):
        raise ValueError("All tensors must be on the same CUDA device.")
    if not all(t.dtype == torch.float32 for t in tensors):
        raise TypeError("This optimized implementation currently supports float32 only.")

    if to_grid_mat.requires_grad or from_grid_mat.requires_grad:
        raise ValueError("Gradients for to_grid_mat/from_grid_mat are not implemented.")

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape [N,J,C] or [N,J,2C].")
    if to_grid_mat.ndim != 2 or from_grid_mat.ndim != 2:
        raise ValueError("Projection matrices must both be rank-2.")

    N, J, cin = inputs.shape
    A, jt = to_grid_mat.shape
    jf, af = from_grid_mat.shape

    if jt != J:
        raise ValueError(
            f"to_grid_mat must have shape [A,{J}], got {tuple(to_grid_mat.shape)}."
        )
    if (jf, af) != (J, A):
        raise ValueError(
            f"from_grid_mat must have shape [{J},{A}], got {tuple(from_grid_mat.shape)}."
        )
    if N < 0 or J <= 0 or A <= 0 or cin <= 0:
        raise ValueError("J, A and channel dimensions must be positive.")
    if separable and cin % 2 != 0:
        raise ValueError(
            f"separable=True requires even input channels [N,J,2C], got cin={cin}."
        )
    if not (0.0 <= dropout_p < 1.0):
        raise ValueError(f"dropout_p must be in [0,1), got {dropout_p}.")

    if lmax is not None:
        if not isinstance(lmax, int) or lmax < 0:
            raise ValueError("lmax must be a nonnegative integer.")
        expected = _expected_j(lmax, mmax)
        if J != expected:
            raise ValueError(
                f"lmax={lmax}, mmax={mmax} requires J={expected}, got J={J}."
            )

    C = cin // 2 if separable else cin
    block_j = _choose_block_j(J)
    return N, J, C, A, block_j


class _FusedS2DotActivationFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        to_grid_mat,
        from_grid_mat,
        separable,
        dropout_p,
        training,
        seed,
        block_j,
    ):
        separable = bool(separable)
        dropout_p = float(dropout_p)
        training = bool(training)
        seed = int(seed)
        block_j = int(block_j)

        N, J, cin = inputs.shape
        C = cin // 2 if separable else cin
        A = to_grid_mat.shape[0]
        use_dropout = separable and training and dropout_p > 0.0

        output = torch.empty((N, J, C), device=inputs.device, dtype=inputs.dtype)

        ctx.save_for_backward(inputs, to_grid_mat, from_grid_mat)
        ctx.separable = separable
        ctx.dropout_p = dropout_p
        ctx.use_dropout = use_dropout
        ctx.seed = seed
        ctx.block_j = block_j

        if N > 0:
            grid = lambda meta: (N, triton.cdiv(C, meta["BLOCK_C"]))
            with torch.cuda.device(inputs.device):
                _s2_dot_fwd_kernel[grid](
                    inputs,
                    to_grid_mat,
                    from_grid_mat,
                    output,
                    N,
                    dropout_p,
                    seed,
                    J=J,
                    A=A,
                    C=C,
                    BLOCK_J=block_j,
                    SEPARABLE=separable,
                    USE_DROPOUT=use_dropout,
                )

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        inputs, to_grid_mat, from_grid_mat = ctx.saved_tensors

        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None, None, None

        N, J, cin = inputs.shape
        C = cin // 2 if ctx.separable else cin
        A = to_grid_mat.shape[0]

        grad_output = grad_output.contiguous()
        dx = torch.empty_like(inputs)

        if N > 0:
            grid = lambda meta: (N, triton.cdiv(C, meta["BLOCK_C"]))
            with torch.cuda.device(inputs.device):
                _s2_dot_bwd_kernel[grid](
                    inputs,
                    to_grid_mat,
                    from_grid_mat,
                    grad_output,
                    dx,
                    N,
                    ctx.dropout_p,
                    ctx.seed,
                    J=J,
                    A=A,
                    C=C,
                    BLOCK_J=ctx.block_j,
                    SEPARABLE=ctx.separable,
                    USE_DROPOUT=ctx.use_dropout,
                )

        # inputs, to_grid_mat, from_grid_mat, separable,
        # dropout_p, training, seed, block_j
        return dx, None, None, None, None, None, None, None


def fused_s2_activation(
    inputs,
    to_grid_mat,
    from_grid_mat,
    *,
    separable=False,
    dropout_p=0.0,
    training=False,
    seed=None,
    lmax=None,
    mmax=None,
):
    """Optimized fused S2 activation using A-tiled tl.dot kernels.

    Parameters
    ----------
    inputs:
        separable=False: [N,J,C]
        separable=True : [N,J,2C]
    to_grid_mat:
        [A,J]
    from_grid_mat:
        [J,A]
    separable:
        False -> to_grid + SiLU + from_grid
        True  -> dual to_grid + product + optional dropout + from_grid
    dropout_p/training:
        Applied only when separable=True.
    seed:
        Optional deterministic dropout seed.
    lmax/mmax:
        Optional shape validation only.  For m-truncated layouts pass both;
        e.g. lmax=4,mmax=2 gives J=19.  They are not needed by the kernel.

    Notes
    -----
    * float32 CUDA only in this implementation.
    * projection matrices are constants; gradients for them are not implemented.
    * first-order backward is implemented; double backward is disabled.
    * optimized fast path currently targets J<=64.
    """
    separable = bool(separable)
    dropout_p = float(dropout_p)
    training = bool(training)

    # These .contiguous() calls are no-ops when tensors already use the expected
    # layout.  For benchmarking, ensure SO3Grid matrices are stored contiguous so
    # hidden copies do not pollute latency.
    x = inputs.contiguous()
    t = to_grid_mat.contiguous()
    fm = from_grid_mat.contiguous()

    _, _, _, _, block_j = _validate(
        x,
        t,
        fm,
        separable,
        dropout_p,
        lmax,
        mmax,
    )

    if seed is None:
        if separable and training and dropout_p > 0.0:
            seed = int(torch.randint(0, 2**31 - 1, (), device="cpu").item())
        else:
            seed = 0

    return _FusedS2DotActivationFn.apply(
        x,
        t,
        fm,
        separable,
        dropout_p,
        training,
        int(seed),
        block_j,
    )


# ============================================================================
# PyTorch reference implementation
# ============================================================================


def s2_activation_reference(
    inputs,
    to_grid_mat,
    from_grid_mat,
    *,
    separable=False,
    dropout_p=0.0,
    training=False,
):
    x_grid = torch.einsum("aj,njc->nac", to_grid_mat, inputs)

    if separable:
        x1, x2 = torch.chunk(x_grid, chunks=2, dim=-1)
        x_grid = x1 * x2
        x_grid = torch.nn.functional.dropout(
            x_grid,
            p=dropout_p,
            training=training,
        )
    else:
        x_grid = torch.nn.functional.silu(x_grid)

    return torch.einsum("ja,nac->njc", from_grid_mat, x_grid)


# ============================================================================
# Minimal benchmark/correctness helper
# ============================================================================


def benchmark_once(
    inputs,
    to_grid_mat,
    from_grid_mat,
    *,
    separable=True,
    dropout_p=0.0,
    training=False,
    warmup=50,
    rep=200,
):
    """Return median latency (ms) for fused and PyTorch-reference forward.

    Do not benchmark training dropout here if exact numerical comparison is needed,
    because PyTorch and Triton intentionally use different RNG streams.
    """
    if not inputs.is_cuda:
        raise ValueError("benchmark_once requires CUDA tensors")

    def fused_call():
        return fused_s2_activation(
            inputs,
            to_grid_mat,
            from_grid_mat,
            separable=separable,
            dropout_p=dropout_p,
            training=training,
            lmax=None,
        )

    def ref_call():
        return s2_activation_reference(
            inputs,
            to_grid_mat,
            from_grid_mat,
            separable=separable,
            dropout_p=dropout_p,
            training=training,
        )

    # triton.testing.do_bench excludes the one-time kernel launch timing itself,
    # but call both paths before measurement to force lazy initialization/JIT.
    for _ in range(3):
        fused_call()
        ref_call()
    torch.cuda.synchronize()

    fused_ms = triton.testing.do_bench(fused_call, warmup=warmup, rep=rep)
    ref_ms = triton.testing.do_bench(ref_call, warmup=warmup, rep=rep)
    return {
        "fused_ms": float(fused_ms),
        "reference_ms": float(ref_ms),
        "speedup": float(ref_ms / fused_ms),
    }
