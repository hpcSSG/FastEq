import torch
import os, math, time
from typing import List


import triton
import triton.language as tl


# -------------------------
# Triton fused kernel (no materialize)
# out[b, k, w] = alpha * val[k] * sum_u x[b, i[k], u] * w_table[p[k], klocal[k], vstar[b],u, w]
# -------------------------

@triton.jit
def fused_onehot_wpuvw_kernel(
    x_ptr,            # *fp32/fp64, [B, I, U]
    w_ptr,            # *fp32/fp64, [P, U, V, W]
    vstar_ptr,        # *int32,     [B]
    p_for_k_ptr,      # *int32,     [K]
    i_for_k_ptr,      # *int32,     [K]  (-1 => empty)
    val_for_k_ptr,    # *fp32/fp64, [K]
    out_ptr,          # *fp32/fp64, [B, K, W]
    B: tl.constexpr, I: tl.constexpr, K: tl.constexpr, U: tl.constexpr,
    P: tl.constexpr, V: tl.constexpr, W: tl.constexpr,
    alpha: tl.constexpr,
    BK: tl.constexpr, BW: tl.constexpr, BU: tl.constexpr,
    ACC_DTYPE: tl.constexpr,          # tl.float32 or tl.float64
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_w = tl.program_id(2)

    k_ids = pid_k * BK + tl.arange(0, BK)
    w_ids = pid_w * BW + tl.arange(0, BW)

    mk = k_ids < K
    mw = w_ids < W

    # scalar v = vstar[b]
    v = tl.load(vstar_ptr + pid_b).to(tl.int32)

    # gather per-k metadata
    p_ids = tl.load(p_for_k_ptr + k_ids, mask=mk, other=0).to(tl.int32)   # [BK]
    i_ids = tl.load(i_for_k_ptr + k_ids, mask=mk, other=-1).to(tl.int32)  # [BK]
    mk2 = mk & (i_ids >= 0)

    # val_for_k (scale per k)
    val = tl.load(val_for_k_ptr + k_ids, mask=mk, other=0.0).to(ACC_DTYPE)  # [BK]

    # accumulator
    acc = tl.zeros((BK, BW), dtype=ACC_DTYPE)

    # reduction over U in chunks of BU
    for u0 in range(0, U, BU):
        u_ids = u0 + tl.arange(0, BU)
        mu = u_ids < U

        # x: [BK, BU]  x[b, i_ids[row], u]
        # offset = ((b*I + i)*U + u)
        x_off = ((pid_b * I + i_ids)[:, None] * U + u_ids[None, :])
        x_val = tl.load(
            x_ptr + x_off,
            mask=mk2[:, None] & mu[None, :],
            other=0.0
        ).to(ACC_DTYPE)

        # w: [BK, BU, BW]  w[p, u, v, w]
        # offset = (((p*U + u)*V + v)*W + w)
        w_off = (((p_ids[:, None, None] * U + u_ids[None, :, None]) * V + v) * W + w_ids[None, None, :])
        w_val = tl.load(
            w_ptr + w_off,
            mask=mk2[:, None, None] & mu[None, :, None] & mw[None, None, :],
            other=0.0
        ).to(ACC_DTYPE)

        # acc[k,w] += sum_u x[k,u] * w[k,u,w]
        acc += tl.sum(x_val[:, :, None] * w_val, axis=1)

    # scale
    val = val.to(ACC_DTYPE)
    acc *= val[:, None] * alpha

    # store
    out_off = (pid_b * K + k_ids)[:, None] * W + w_ids[None, :]
    tl.store(out_ptr + out_off, acc, mask=mk[:, None] & mw[None, :])


@torch.no_grad()
def triton_fused_fctp_fwd(
    x_biu,            # [B, I, U] fp32/fp64
    vstar,            # [B] int32
    w_puvw,           # [P, U, V, W] same dtype as x
    p_for_k,          # [K] int32
    i_for_k,          # [K] int32
    val_for_k,        # [K] same dtype as x (recommended)
    alpha,            # python float or 0-d tensor
    K_total: int,
    BK=8, BW=64, BU=32,
    num_warps=4,
):
    assert x_biu.is_cuda, "Triton kernel expects CUDA/ROCm tensor"
    assert x_biu.dtype in (torch.float32, torch.float64), "Only fp32/fp64 supported here"
    assert w_puvw.dtype == x_biu.dtype
    assert val_for_k.dtype == x_biu.dtype

    B, I, U = x_biu.shape
    P, U2, V, W = w_puvw.shape
    assert U == U2
    assert p_for_k.numel() == K_total, (
        f"p_for_k must contain one entry per output k: "
        f"got {p_for_k.numel()}, expected {K_total}"
    )
    assert i_for_k.numel() == K_total, (
        f"i_for_k must contain one entry per output k: "
        f"got {i_for_k.numel()}, expected {K_total}"
    )
    assert val_for_k.numel() == K_total, (
        f"val_for_k must contain one entry per output k: "
        f"got {val_for_k.numel()}, expected {K_total}"
    )

    #alpha = alpha.reshape(-1)[0].item()

    out = torch.empty((B, K_total, W), device=x_biu.device, dtype=x_biu.dtype)

    # pick accumulator dtype
    ACC_DTYPE = tl.float64 if x_biu.dtype == torch.float64 else tl.float32

    grid = (B, triton.cdiv(K_total, BK), triton.cdiv(W, BW))

    fused_onehot_wpuvw_kernel[grid](
        x_biu, w_puvw, vstar,
        p_for_k, i_for_k, val_for_k,
        out,
        B=B, I=I, K=K_total, U=U, P=P, V=V, W=W,
        alpha=alpha,
        BK=BK, BW=BW, BU=BU,
        ACC_DTYPE=ACC_DTYPE,
        num_warps=num_warps,
    )
    return out.view(B, -1)



@triton.jit
def fused_onehot_wpuvw_bwd_dx_noatomic_kernel(
    grad_out_ptr,     # *fp32/fp64, [B, K, W]
    w_ptr,            # *fp32/fp64, [P, U, V, W]
    vstar_ptr,        # *int32, [B]
    p_for_k_ptr,      # *int32, [K]
    i_for_k_ptr,      # *int32, [K]   (-1 => empty)
    val_for_k_ptr,    # *fp32/fp64, [K]
    grad_x_ptr,       # *fp32/fp64, [B, I, U]  (direct store, no atomic)
    B: tl.constexpr, I: tl.constexpr, K: tl.constexpr, U: tl.constexpr,
    P: tl.constexpr, V: tl.constexpr, W: tl.constexpr,
    alpha: tl.constexpr,
    BK: tl.constexpr, BU: tl.constexpr, BW: tl.constexpr,
    ACC_DTYPE: tl.constexpr,          # tl.float32 or tl.float64
):
    pid_b = tl.program_id(0)   # batch
    pid_k = tl.program_id(1)   # k tile
    pid_u = tl.program_id(2)   # u tile

    k_ids = pid_k * BK + tl.arange(0, BK)          # [BK]
    u_ids = pid_u * BU + tl.arange(0, BU)          # [BU]

    mk = k_ids < K
    mu = u_ids < U

    # scalar v
    v = tl.load(vstar_ptr + pid_b).to(tl.int32)

    # per-k metadata
    p_ids = tl.load(p_for_k_ptr + k_ids, mask=mk, other=0).to(tl.int32)       # [BK]
    i_ids = tl.load(i_for_k_ptr + k_ids, mask=mk, other=-1).to(tl.int32)      # [BK]
    mk2 = mk & (i_ids >= 0)

    val = tl.load(val_for_k_ptr + k_ids, mask=mk, other=0.0).to(ACC_DTYPE)    # [BK]

    # acc[k,u] = sum_w go[k,w] * w[p,u,v,w]
    acc = tl.zeros((BK, BU), dtype=ACC_DTYPE)

    # iterate W in BW chunks
    for w0 in range(0, W, BW):
        w_ids = w0 + tl.arange(0, BW)
        mw = w_ids < W

        # go: [BK, BW]
        go_off = (pid_b * K + k_ids)[:, None] * W + w_ids[None, :]
        go = tl.load(
            grad_out_ptr + go_off,
            mask=mk2[:, None] & mw[None, :],
            other=0.0
        ).to(ACC_DTYPE)

        # w: [BK, BU, BW] for given v
        w_off = (((p_ids[:, None, None] * U + u_ids[None, :, None]) * V + v) * W
                 + w_ids[None, None, :])
        ww = tl.load(
            w_ptr + w_off,
            mask=mk2[:, None, None] & mu[None, :, None] & mw[None, None, :],
            other=0.0
        ).to(ACC_DTYPE)

        acc += tl.sum(go[:, None, :] * ww, axis=2)   # [BK,BU]

    # scale
    acc *= (val[:, None] * tl.full((), alpha, ACC_DTYPE))

    # direct store (no atomic) because i_for_k is one-to-one
    gx_off = ((pid_b * I + i_ids)[:, None] * U + u_ids[None, :])
    tl.store(
        grad_x_ptr + gx_off,
        acc,
        mask=mk2[:, None] & mu[None, :]
    )


@torch.no_grad()
def triton_fused_fctp_bwd(
    grad_out: torch.Tensor,    # [B,K,W] fp32/fp64
    w: torch.Tensor,           # [P,U,V,W] fp32/fp64
    vstar: torch.Tensor,       # [B] int32
    p_for_k: torch.Tensor,     # [K] int32
    i_for_k: torch.Tensor,     # [K] int32, one-to-one mapping
    val_for_k: torch.Tensor,   # [K] fp32/fp64
    I: int,
    alpha,
    can_use_empty_grad_x: bool,
    BU=32, BW=32, BK=8,
    num_warps=4,
):
    assert grad_out.is_cuda and w.is_cuda
    assert grad_out.dtype in (torch.float32, torch.float64)
    assert w.dtype == grad_out.dtype
    assert val_for_k.dtype == grad_out.dtype
    assert vstar.dtype == torch.int32
    assert p_for_k.dtype == torch.int32 and i_for_k.dtype == torch.int32

    B, K, W_ = grad_out.shape
    P, U, V, W = w.shape
    assert W_ == W
    assert p_for_k.numel() == K, (
        f"p_for_k must contain one entry per output k: "
        f"got {p_for_k.numel()}, expected {K}"
    )
    assert i_for_k.numel() == K, (
        f"i_for_k must contain one entry per output k: "
        f"got {i_for_k.numel()}, expected {K}"
    )
    assert val_for_k.numel() == K, (
        f"val_for_k must contain one entry per output k: "
        f"got {val_for_k.numel()}, expected {K}"
    )

    if can_use_empty_grad_x:
        grad_x = torch.empty((B, I, U), device=grad_out.device, dtype=grad_out.dtype)
    else:
        grad_x = torch.zeros((B, I, U), device=grad_out.device, dtype=grad_out.dtype)

    ACC_DTYPE = tl.float64 if grad_out.dtype == torch.float64 else tl.float32    

    grid = (B, triton.cdiv(K, BK), triton.cdiv(U, BU))
    fused_onehot_wpuvw_bwd_dx_noatomic_kernel[grid](
        grad_out, w, vstar, p_for_k, i_for_k, val_for_k, grad_x,
        B=B, I=I, K=K, U=U, P=P, V=V, W=W,
        alpha=alpha,
        BK=BK, BU=BU, BW=BW,
        ACC_DTYPE=ACC_DTYPE,
        num_warps=num_warps,
    )
    return grad_x.view(B, -1)


@triton.jit
def fused_onehot_wpuvw_bwd_dw_kernel(
    grad_out_ptr,     # *fp32/fp64, [B,K,W]
    x_ptr,            # *fp32/fp64, [B,I,U]
    vstar_ptr,        # *int32, [B]
    p_for_k_ptr,      # *int32, [K]
    i_for_k_ptr,      # *int32, [K]
    val_for_k_ptr,    # *fp32/fp64, [K]
    grad_w_ptr,       # *fp32/fp64, [P,U,V,W]
    B: tl.constexpr, I: tl.constexpr, K: tl.constexpr, U: tl.constexpr,
    P: tl.constexpr, V: tl.constexpr, W: tl.constexpr,
    alpha: tl.constexpr,
    BU: tl.constexpr, BW: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    """Fuse the outer product and index_add reduction used by grad_w.

    Each program handles one (batch,k) pair and directly scatters its [BU,BW]
    outer-product tile into grad_w.  atomic_add performs the former index_add
    reduction without materializing a [B,K,U,W] contribution tensor.
    """
    pid_r = tl.program_id(0)
    pid_u = tl.program_id(1)
    pid_w = tl.program_id(2)

    b = pid_r // K
    k = pid_r - b * K
    u_ids = pid_u * BU + tl.arange(0, BU)
    w_ids = pid_w * BW + tl.arange(0, BW)
    mu = u_ids < U
    mw = w_ids < W

    p = tl.load(p_for_k_ptr + k).to(tl.int32)
    i = tl.load(i_for_k_ptr + k).to(tl.int32)
    v = tl.load(vstar_ptr + b).to(tl.int32)
    val = tl.load(val_for_k_ptr + k).to(ACC_DTYPE)
    valid = (i >= 0) & (p >= 0) & (p < P) & (v >= 0) & (v < V)

    x_off = (b * I + i) * U + u_ids
    x_val = tl.load(
        x_ptr + x_off,
        mask=valid & mu,
        other=0.0,
    ).to(ACC_DTYPE)

    go_off = (b * K + k) * W + w_ids
    go = tl.load(
        grad_out_ptr + go_off,
        mask=valid & mw,
        other=0.0,
    ).to(ACC_DTYPE)

    contribution = (
        x_val[:, None]
        * go[None, :]
        * val
        * tl.full((), alpha, ACC_DTYPE)
    )
    out_off = (((p * U + u_ids)[:, None] * V + v) * W
               + w_ids[None, :])
    tl.atomic_add(
        grad_w_ptr + out_off,
        contribution,
        mask=valid & mu[:, None] & mw[None, :],
    )


@torch.no_grad()
def triton_fused_fctp_bwd_w(
    grad_out: torch.Tensor,  # [B,K,W]
    x: torch.Tensor,         # [B,I,U]
    vstar: torch.Tensor,
    p_for_k: torch.Tensor,
    i_for_k: torch.Tensor,
    val_for_k: torch.Tensor,
    P: int,
    V: int,
    alpha,
    BU=16,
    BW=16,
    num_warps=4,
):
    """Triton grad_w with fused contraction and sparse index reduction."""
    if grad_out.ndim != 3 or x.ndim != 3:
        raise ValueError("expected grad_out [B,K,W] and x [B,I,U]")
    if not grad_out.is_cuda or not x.is_cuda:
        raise ValueError("Triton grad_w expects CUDA/ROCm tensors")
    if grad_out.dtype not in (torch.float32, torch.float64):
        raise TypeError("Triton grad_w supports fp32/fp64 only")
    if x.dtype != grad_out.dtype or val_for_k.dtype != grad_out.dtype:
        raise TypeError("x, grad_out, and val_for_k must have the same dtype")

    B, K, W = grad_out.shape
    B_x, I, U = x.shape
    if B_x != B:
        raise ValueError(f"x batch {B_x} does not match grad_out batch {B}")
    if p_for_k.numel() != K or i_for_k.numel() != K:
        raise ValueError("p_for_k and i_for_k must contain K entries")
    if val_for_k.numel() != K or vstar.numel() != B:
        raise ValueError("val_for_k must contain K entries and vstar B entries")

    # Required because multiple (batch,k) programs atomically accumulate into
    # the same (path,u,v,w) output entries.
    grad_w = torch.zeros((P, U, V, W), device=x.device, dtype=x.dtype)
    acc_dtype = tl.float64 if x.dtype == torch.float64 else tl.float32
    grid = (B * K, triton.cdiv(U, BU), triton.cdiv(W, BW))
    fused_onehot_wpuvw_bwd_dw_kernel[grid](
        grad_out,
        x,
        vstar,
        p_for_k,
        i_for_k,
        val_for_k,
        grad_w,
        B=B,
        I=I,
        K=K,
        U=U,
        P=P,
        V=V,
        W=W,
        alpha=alpha,
        BU=BU,
        BW=BW,
        ACC_DTYPE=acc_dtype,
        num_warps=num_warps,
    )
    return grad_w


@torch.no_grad()
def pad_p_for_k(
    p_for_k: torch.Tensor,
    i_for_k: torch.Tensor,
    K_total: int,
    path_num: int,
) -> torch.Tensor:
    """Pad ``p_for_k`` to one entry per output component.

    Missing entries are safe to fill with path 0 only when their corresponding
    ``i_for_k`` entries are negative, because the Triton kernel masks those
    output components before loading ``x`` and ``w``.
    """
    if p_for_k.ndim != 1 or i_for_k.ndim != 1:
        raise ValueError("p_for_k and i_for_k must be one-dimensional tensors")
    if p_for_k.dtype != torch.int32 or i_for_k.dtype != torch.int32:
        raise TypeError("p_for_k and i_for_k must have dtype torch.int32")
    if p_for_k.device != i_for_k.device:
        raise ValueError("p_for_k and i_for_k must be on the same device")
    if i_for_k.numel() != K_total:
        raise ValueError(
            f"i_for_k has {i_for_k.numel()} entries, expected K_total={K_total}"
        )
    if p_for_k.numel() > K_total:
        raise ValueError(
            f"p_for_k has {p_for_k.numel()} entries, exceeding K_total={K_total}"
        )
    if path_num <= 0:
        raise ValueError(f"path_num must be positive, got {path_num}")

    if p_for_k.numel() > 0:
        invalid_path = (p_for_k < 0) | (p_for_k >= path_num)
        if torch.any(invalid_path).item():
            raise ValueError(
                f"p_for_k contains a path index outside [0, {path_num})"
            )

    if p_for_k.numel() == K_total:
        return p_for_k.contiguous()

    first_missing = p_for_k.numel()
    if torch.any(i_for_k[first_missing:] >= 0).item():
        raise ValueError(
            "p_for_k is missing entries for valid output components; "
            "the path mapping must be rebuilt instead of zero-padded"
        )

    padded = torch.zeros(
        K_total,
        device=p_for_k.device,
        dtype=torch.int32,
    )
    padded[:first_missing] = p_for_k
    return padded


class _FCTPGradXWithDoubleBackward(torch.autograd.Function):
    """First backward for x, with an explicit fused second backward."""

    @staticmethod
    def forward(
        ctx,
        grad_out,
        w,
        vstar,
        p_for_k,
        i_for_k,
        val_for_k,
        I,
        alpha,
        can_use_empty_grad_x,
    ):
        B, K, W = grad_out.shape
        P, U, V, W_weight = w.shape
        if W_weight != W:
            raise ValueError("grad_out and w have different W dimensions")

        ctx.save_for_backward(
            grad_out, w, vstar, p_for_k, i_for_k, val_for_k
        )
        ctx.I = I
        ctx.alpha = alpha
        ctx.can_use_empty_grad_x = can_use_empty_grad_x

        grad_x = triton_fused_fctp_bwd(
            grad_out,
            w,
            vstar,
            p_for_k,
            i_for_k,
            val_for_k,
            I,
            alpha,
            can_use_empty_grad_x,
            BK=8,
            BW=32,
            BU=32,
            num_warps=4,
        )
        return grad_x.view(B, I, U)

    @staticmethod
    def backward(ctx, grad_grad_x):
        grad_out, w, vstar, p_for_k, i_for_k, val_for_k = ctx.saved_tensors
        if grad_grad_x is None:
            return (None,) * 9

        grad_grad_x = grad_grad_x.contiguous()
        B, K, W = grad_out.shape
        P, U, V, _ = w.shape

        # d(grad_x)/d(grad_out): same contraction as FCTP forward, with
        # grad_grad_x replacing x.
        grad_grad_out = None
        if ctx.needs_input_grad[0]:
            grad_grad_out = triton_fused_fctp_fwd(
                grad_grad_x,
                vstar,
                w,
                p_for_k,
                i_for_k,
                val_for_k,
                ctx.alpha,
                K,
                BK=8,
                BW=64,
                BU=32,
                num_warps=4,
            ).view_as(grad_out)

        # d(grad_x)/d(w): same fused outer-product/reduction primitive used to
        # generate grad_w, with grad_grad_x replacing x.
        grad_w = None
        if ctx.needs_input_grad[1]:
            grad_w = triton_fused_fctp_bwd_w(
                grad_out,
                grad_grad_x,
                vstar,
                p_for_k,
                i_for_k,
                val_for_k,
                P,
                V,
                ctx.alpha,
            )

        return grad_grad_out, grad_w, None, None, None, None, None, None, None


class _FCTPGradWWithDoubleBackward(torch.autograd.Function):
    """First backward for w, with an explicit fused second backward."""

    @staticmethod
    def forward(
        ctx,
        grad_out,
        x,
        vstar,
        p_for_k,
        i_for_k,
        val_for_k,
        P,
        V,
        alpha,
        can_use_empty_grad_x,
    ):
        ctx.save_for_backward(
            grad_out, x, vstar, p_for_k, i_for_k, val_for_k
        )
        ctx.P = P
        ctx.V = V
        ctx.alpha = alpha
        ctx.can_use_empty_grad_x = can_use_empty_grad_x

        return triton_fused_fctp_bwd_w(
            grad_out,
            x,
            vstar,
            p_for_k,
            i_for_k,
            val_for_k,
            P,
            V,
            alpha,
        )

    @staticmethod
    def backward(ctx, grad_grad_w):
        grad_out, x, vstar, p_for_k, i_for_k, val_for_k = ctx.saved_tensors
        if grad_grad_w is None:
            return (None,) * 10

        grad_grad_w = grad_grad_w.contiguous()
        B, K, W = grad_out.shape
        _, I, U = x.shape

        # d(grad_w)/d(grad_out): FCTP forward with grad_grad_w as its weight.
        grad_grad_out = None
        if ctx.needs_input_grad[0]:
            grad_grad_out = triton_fused_fctp_fwd(
                x,
                vstar,
                grad_grad_w,
                p_for_k,
                i_for_k,
                val_for_k,
                ctx.alpha,
                K,
                BK=8,
                BW=64,
                BU=32,
                num_warps=4,
            ).view_as(grad_out)

        # d(grad_w)/d(x): the existing fused grad_x contraction with
        # grad_grad_w replacing w.
        grad_x = None
        if ctx.needs_input_grad[1]:
            grad_x = triton_fused_fctp_bwd(
                grad_out,
                grad_grad_w,
                vstar,
                p_for_k,
                i_for_k,
                val_for_k,
                I,
                ctx.alpha,
                ctx.can_use_empty_grad_x,
                BK=8,
                BW=32,
                BU=32,
                num_warps=4,
            ).view_as(x)

        return grad_grad_out, grad_x, None, None, None, None, None, None, None, None


class FastFullyConnectedTensorProductPathFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, y, meta):

        # Save the original Function inputs, not no-grad views created below.
        # Saving the actual inputs is required for reliable double backward.
        w_input, x_input, y_input = w, x, y

        cg_indices = meta["cg_indices"]
        cg_values = meta["cg_values"]
        cg_i_all = meta["cg_i_all"]
        cg_j_all = meta["cg_j_all"]
        cg_k_all = meta["cg_k_all"]
        cg_val_all = meta["cg_val_all"]
        nnz_per_path = meta["nnz_per_path"]
        K_per_path = meta["K_per_path"]
        path_offset = meta["path_offset"]
        i_for_k = meta["i_for_k"]
        val_for_k = meta["val_for_k"]
        p_for_k = meta["p_for_k"]
        cg_val = meta["cg_val_0"]
        path_num = meta["P"]
        nnz0 = meta["nnz0"]
        U, V, W, K_total, I_total = meta["U"], meta["V"], meta["W"], meta["K_total"], meta["I_total"]

        # Triton loads p_for_k for every k in [0, K_total).  Some descriptors
        # omit entries for output components with i_for_k == -1.  Materialize
        # those masked entries so the kernel never performs an out-of-bounds
        # metadata load.  Keep the padded tensor in ctx.meta for backward too.
        p_for_k = pad_p_for_k(p_for_k, i_for_k, K_total, path_num)
        meta = dict(meta)
        meta["p_for_k"] = p_for_k
        
        B = x.shape[0]

        #torch.cuda.synchronize()
        #start_time = time.perf_counter() * 1000

        if path_num == 1 and nnz0 == 1 and I_total == 1 and K_total == 1:
            # use torch is better when open MPS
            # ====================== torch Implementation ======================
            vstar = torch.argmax(y, dim=1)
            ctx.vstar = vstar.to(torch.int32).contiguous()
            w = w.view(U, V, W)
            w_selected = w[:, vstar, :].permute(1, 0, 2).contiguous()

            # out[b, w] = sum_u a[b, u] * w_selected[b, u, w]
            out = torch.einsum("bu,buw->bw", x, w_selected)
            output = out * cg_val

        else:
            #====================== Triton Implementation ====================== 
            x = x.view(B, I_total, U)
            y = y.view(B, V)
            w = w.view(path_num, U, V, W)
            vstar = torch.argmax(y, dim=1).to(torch.int32).contiguous()
            ctx.vstar = vstar
            
            output = triton_fused_fctp_fwd(x, vstar, w, p_for_k, i_for_k, val_for_k, cg_val, K_total,
                                                        BK=8, BW=64, BU=32, num_warps=4)
        
        #torch.cuda.synchronize()
        #end_time = time.perf_counter() * 1000
        #execution_time_ms = end_time - start_time

        ctx.save_for_backward(w_input, x_input, y_input)
        ctx.meta = meta

        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000


        w, x, y = ctx.saved_tensors

        # ``ctx.needs_input_grad`` records which original forward inputs need
        # gradients.  When grad mode is enabled inside backward, the caller is
        # using create_graph=True and is therefore constructing a graph for a
        # subsequent double backward.  Print the actual runtime requirements
        # here instead of inferring them only from the model configuration.
        needs_w_grad, needs_x_grad, needs_y_grad, _ = ctx.needs_input_grad

        if not needs_x_grad and not needs_w_grad:
            return None, None, None, None

        meta = ctx.meta
        cg_values = meta["cg_values"]
        cg_i_all = meta["cg_i_all"]
        cg_j_all = meta["cg_j_all"]
        cg_k_all = meta["cg_k_all"]
        cg_val_all = meta["cg_val_all"]
        nnz_per_path = meta["nnz_per_path"]
        K_per_path = meta["K_per_path"]
        path_offset = meta["path_offset"]
        i_for_k = meta["i_for_k"]
        val_for_k = meta["val_for_k"]
        p_for_k = meta["p_for_k"]
        U, V, W, K_total, I_total = meta["U"], meta["V"], meta["W"], meta["K_total"], meta["I_total"]
        cg_val = meta["cg_val_0"]
        B = x.shape[0]
        path_num = meta["P"]
        nnz0 = meta["nnz0"]
        can_use_empty_grad_x = meta["can_use_empty_grad_x"]

        grad_out_3d = grad_out.view(B, K_total, W)
        w_4d = w.view(path_num, U, V, W)
        x_3d = x.view(B, I_total, U)

        # The single-path torch forward multiplies only by cg_val.  The Triton
        # forward multiplies by both val_for_k and cg_val.  Select the matching
        # per-k scale here so both backward branches exactly mirror forward.
        is_single_torch_path = (
            path_num == 1 and nnz0 == 1 and I_total == 1 and K_total == 1
        )
        backward_val_for_k = (
            torch.ones_like(val_for_k) if is_single_torch_path else val_for_k
        )

        grad_x = None
        grad_w = None

        if torch.is_grad_enabled():
            # create_graph=True: create two explicit first-backward nodes.
            # Their custom backward methods invoke fused Triton primitives for
            # the two mixed second-derivative paths, without materializing the
            # einsum result consumed by index_add.
            if needs_x_grad:
                grad_x = _FCTPGradXWithDoubleBackward.apply(
                    grad_out_3d,
                    w_4d,
                    ctx.vstar,
                    p_for_k,
                    i_for_k,
                    backward_val_for_k,
                    I_total,
                    cg_val,
                    can_use_empty_grad_x,
                ).reshape_as(x)
            if needs_w_grad:
                grad_w = _FCTPGradWWithDoubleBackward.apply(
                    grad_out_3d,
                    x_3d,
                    ctx.vstar,
                    p_for_k,
                    i_for_k,
                    backward_val_for_k,
                    path_num,
                    V,
                    cg_val,
                    can_use_empty_grad_x,
                ).reshape_as(w)
        else:
            # Standard first backward: retain the fast existing grad_x path.
            if needs_x_grad:
                if is_single_torch_path:
                    vstar = ctx.vstar.to(torch.long)
                    w_single = w_4d.view(U, V, W)
                    go_single = grad_out_3d.view(B, W)
                    w_selected = w_single[:, vstar, :].permute(1, 0, 2)
                    grad_x = (
                        torch.einsum("bw,buw->bu", go_single, w_selected)
                        * cg_val
                    ).reshape_as(x)
                else:
                    grad_x = triton_fused_fctp_bwd(
                        grad_out_3d,
                        w_4d,
                        ctx.vstar,
                        p_for_k,
                        i_for_k,
                        val_for_k,
                        I_total,
                        cg_val,
                        can_use_empty_grad_x,
                        BK=8,
                        BW=32,
                        BU=32,
                        num_warps=4,
                    ).reshape_as(x)

            # Fuse the former einsum + index_add grad_w path into one Triton
            # reduction kernel.  Each program exclusively owns an output tile,
            # avoiding both the large contribution tensor and atomics.
            if needs_w_grad:
                grad_w = triton_fused_fctp_bwd_w(
                    grad_out_3d,
                    x_3d,
                    ctx.vstar,
                    p_for_k,
                    i_for_k,
                    backward_val_for_k,
                    path_num,
                    V,
                    cg_val,
                ).reshape_as(w)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        if torch.is_grad_enabled():
            print(f"<< fasteq fctp double backward cost {execution_time_ms:.3f} ms >>")
        else:
            print(f"<< fasteq fctp backward cost: {execution_time_ms:.3f} ms >>")

        return grad_w, grad_x, None, None  # y uses argmax; meta is non-Tensor

def fast_fctp(w, x, y, meta):
    return FastFullyConnectedTensorProductPathFused.apply(w, x, y, meta)
