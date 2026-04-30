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

    alpha = alpha.reshape(-1)[0].item()

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

    if can_use_empty_grad_x:
        grad_x = torch.empty((B, I, U), device=grad_out.device, dtype=grad_out.dtype)
    else:
        grad_x = torch.zeros((B, I, U), device=grad_out.device, dtype=grad_out.dtype)

    ACC_DTYPE = tl.float64 if grad_out.dtype == torch.float64 else tl.float32

    alpha = alpha.reshape(-1)[0].item()

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


class FastFullyConnectedTensorProductPathFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, y, meta):

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
        
        B = x.shape[0]
        
        #====================== Triton Implementation ====================== 
        x = x.view(B, I_total, U)
        y = y.view(B, V)
        w = w.view(path_num, U, V, W)
        vstar = torch.argmax(y, dim=1).to(torch.int32).contiguous()
        ctx.vstar = vstar
        
        output = triton_fused_fctp_fwd(x, vstar, w, p_for_k, i_for_k, val_for_k, cg_val, K_total,
                                                        BK=8, BW=64, BU=32, num_warps=4)
        
        """
        output = torch.ops.fctp_fused_multipath_fwd.forward(w, x, y, vstar,
            cg_i_all, cg_j_all, cg_k_all, cg_val_all,
            nnz_per_path, K_per_path, path_offset, U, V, W, K_total)
        """

        ctx.save_for_backward(w, x, y)
        ctx.meta = meta

        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        """ torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000 """


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

        
        #====================== Triton Implementation ====================== 
        grad_x = triton_fused_fctp_bwd(grad_out, w, ctx.vstar, p_for_k, i_for_k, val_for_k, I_total, cg_val, can_use_empty_grad_x,
                                                        BK=8, BW=32, BU=32, num_warps=4)
        
        """ grad_x = torch.ops.fctp_fused_multipath_bwd.backward(grad_out, w, x, y, ctx.vstar, 
            cg_i_all, cg_j_all, cg_k_all, cg_val_all,
            nnz_per_path, K_per_path, path_offset, U, V, W, K_total) """
        

        return None, grad_x, None, None  # None for w, y, meta gradients

def fast_fctp(w, x, y, meta):
    return FastFullyConnectedTensorProductPathFused.apply(w, x, y, meta)
