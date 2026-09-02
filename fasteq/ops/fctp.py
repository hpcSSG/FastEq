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


def differentiable_fctp_bwd_x(
    grad_out: torch.Tensor,
    w: torch.Tensor,
    vstar: torch.Tensor,
    p_for_k: torch.Tensor,
    i_for_k: torch.Tensor,
    val_for_k: torch.Tensor,
    I: int,
    alpha,
) -> torch.Tensor:
    """Compute ``grad_x`` with differentiable PyTorch operations.

    This implements the same one-hot-y contraction as the Triton backward, but
    deliberately remains in PyTorch's autograd graph.  It is used when the
    caller requests ``create_graph=True`` so a subsequent double backward can
    differentiate ``grad_x`` with respect to ``grad_out`` and ``w``.
    """
    if grad_out.ndim != 3 or w.ndim != 4:
        raise ValueError("expected grad_out [B,K,W] and w [P,U,V,W]")

    B, K, W_out = grad_out.shape
    P, U, V, W = w.shape
    if W_out != W:
        raise ValueError(f"grad_out W={W_out} does not match weight W={W}")
    if p_for_k.numel() != K or i_for_k.numel() != K:
        raise ValueError("p_for_k and i_for_k must contain K entries")
    if val_for_k.numel() != K:
        raise ValueError("val_for_k must contain K entries")

    valid_k = torch.nonzero(i_for_k >= 0, as_tuple=False).flatten()
    if valid_k.numel() == 0:
        # Keep a zero-valued dependency on grad_out and w.  Without it, the
        # returned tensor would not require grad and double backward would fail
        # for descriptors whose output components are all structurally zero.
        zero = grad_out.sum() * 0 + w.sum() * 0
        return torch.zeros(
            (B, I * U), device=grad_out.device, dtype=grad_out.dtype
        ) + zero

    p_idx = p_for_k.index_select(0, valid_k).to(torch.long)
    i_idx = i_for_k.index_select(0, valid_k).to(torch.long)
    scale = val_for_k.index_select(0, valid_k) * alpha

    if torch.any((p_idx < 0) | (p_idx >= P)).item():
        raise ValueError("p_for_k contains an out-of-range path index")
    if torch.any((i_idx < 0) | (i_idx >= I)).item():
        raise ValueError("i_for_k contains an out-of-range input index")
    if vstar.numel() != B:
        raise ValueError(f"vstar has {vstar.numel()} entries, expected B={B}")

    # Arrange the weight as [P*V,U,W].  For every (batch, valid-k), select
    # weight[p_for_k[k], :, vstar[b], :].
    w_pvuw = w.permute(0, 2, 1, 3).reshape(P * V, U, W)
    pv_idx = (
        p_idx.unsqueeze(0) * V + vstar.to(torch.long).unsqueeze(1)
    ).reshape(-1)
    w_selected = w_pvuw.index_select(0, pv_idx).view(
        B, valid_k.numel(), U, W
    )

    go = grad_out.index_select(1, valid_k)
    contribution = torch.einsum(
        "bqw,bquw,q->bqu", go, w_selected, scale
    )

    # Multiple output k values may contribute to the same input i.  index_add
    # performs the required reduction and, unlike the direct-store Triton fast
    # path, remains correct for duplicate i_idx values.
    grad_x = torch.zeros(
        (B, I, U), device=grad_out.device, dtype=grad_out.dtype
    )
    grad_x = grad_x.index_add(1, i_idx, contribution)
    return grad_x.reshape(B, I * U)


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
        #print(f"triton_fused_fctp_fwd output shape: {output.shape}")

        ctx.save_for_backward(w_input, x_input, y_input)
        ctx.meta = meta

        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000


        w, x, y = ctx.saved_tensors

        if not x.requires_grad:
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

        grad_out = grad_out.view(B, K_total, W)
        w = w.view(path_num, U, V, W)

        if path_num == 1 and nnz0 == 1 and I_total == 1 and K_total == 1:
            #================ Path = 1, torch ===============
            # vstar: [B]
            vstar = torch.argmax(y, dim=1)

            w = w.view(U, V, W)
            grad_out = grad_out.view(B, -1)
            w_selected = w[:, vstar, :].permute(1, 0, 2).contiguous()

            # grad_x[b, u] = cg_val * sum_w grad_out[b, w] * w_selected[b, u, w]
            grad_x = torch.einsum("bw,buw->bu", grad_out, w_selected)
            grad_x = grad_x * cg_val

        else:
            if torch.is_grad_enabled():
                # create_graph=True: keep the first backward differentiable so
                # PyTorch can construct and execute the double-backward graph.
                grad_x = differentiable_fctp_bwd_x(
                    grad_out,
                    w,
                    ctx.vstar,
                    p_for_k,
                    i_for_k,
                    val_for_k,
                    I_total,
                    cg_val,
                )
            else:
                # Standard first backward: retain the faster Triton kernel.
                grad_x = triton_fused_fctp_bwd(
                    grad_out,
                    w,
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
                )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq fctp backward cost: {execution_time_ms:.3f} ms >>")

        return None, grad_x, None, None  # None for w, y, meta gradients

def fast_fctp(w, x, y, meta):
    return FastFullyConnectedTensorProductPathFused.apply(w, x, y, meta)
