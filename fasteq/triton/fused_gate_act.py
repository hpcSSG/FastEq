from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Triton kernels
# -----------------------------------------------------------------------------

@triton.jit
def _gate_activation_fwd_kernel(
    X,                  # [N, M, C]
    G,                  # [N, L * C]
    EXPAND_INDEX,       # [M - 1], value in [0, L-1]
    Y,                  # [N, M, C]
    N: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    stride_xn,
    stride_xm,
    stride_xc,
    stride_gn,
    stride_gc,
    stride_yn,
    stride_ym,
    stride_yc,
    BLOCK_C: tl.constexpr,
):
    """
    Forward:
      m == 0:
        y = SiLU(x)
      m > 0:
        l_idx = EXPAND_INDEX[m - 1]
        y = x * sigmoid(g[n, l_idx, c])

    One Triton program handles one (n, m, channel tile).
    """
    n = tl.program_id(0)
    m = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    x_ptrs = X + n * stride_xn + m * stride_xm + offs_c * stride_xc
    x = tl.load(x_ptrs, mask=mask_c, other=0.0).to(tl.float32)

    # Safe lookup address for m == 0; the loaded value is ignored there.
    vector_idx = tl.maximum(m - 1, 0)
    l_idx = tl.load(
        EXPAND_INDEX + vector_idx,
        mask=m > 0,
        other=0,
    ).to(tl.int32)

    g_ptrs = G + n * stride_gn + (l_idx * C + offs_c) * stride_gc
    g = tl.load(
        g_ptrs,
        mask=mask_c & (m > 0),
        other=0.0,
    ).to(tl.float32)

    # scalar branch: SiLU(x) = x * sigmoid(x)
    sx = tl.sigmoid(x)
    y_scalar = x * sx

    # vector/tensor branch
    sg = tl.sigmoid(g)
    y_vector = x * sg

    y = tl.where(m == 0, y_scalar, y_vector)

    y_ptrs = Y + n * stride_yn + m * stride_ym + offs_c * stride_yc
    tl.store(y_ptrs, y, mask=mask_c)


@triton.jit
def _gate_activation_bwd_kernel(
    GRAD_Y,             # [N, M, C]
    X,                  # [N, M, C]
    G,                  # [N, L * C]
    STARTS,             # [L], input component start index, starts from 1
    COUNTS,             # [L], number of m-components for each l
    GRAD_X,             # [N, M, C]
    GRAD_G,             # [N, L * C]
    C: tl.constexpr,
    stride_dyn,
    stride_dym,
    stride_dyc,
    stride_xn,
    stride_xm,
    stride_xc,
    stride_gn,
    stride_gc,
    stride_dxn,
    stride_dxm,
    stride_dxc,
    stride_dgn,
    stride_dgc,
    LMAX: tl.constexpr,
    MAX_COUNT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Backward without atomics.

    Each program owns one (n, channel tile), so:
      1. scalar grad_x is written once;
      2. every vector/tensor grad_x is written once;
      3. grad_gate for each l is reduced locally over all m belonging to l.

    Derivatives:
      SiLU(x) = x * sigmoid(x)
      dSiLU/dx = s + x*s*(1-s)

      y = x * sigmoid(g)
      dy/dx = sigmoid(g)
      dy/dg = x * sigmoid(g) * (1 - sigmoid(g))
    """
    n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # -------------------------------------------------------------------------
    # m = 0 scalar path
    # -------------------------------------------------------------------------
    x0_ptrs = X + n * stride_xn + offs_c * stride_xc
    dy0_ptrs = GRAD_Y + n * stride_dyn + offs_c * stride_dyc

    x0 = tl.load(x0_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    dy0 = tl.load(dy0_ptrs, mask=mask_c, other=0.0).to(tl.float32)

    sx = tl.sigmoid(x0)
    dsilu = sx + x0 * sx * (1.0 - sx)
    dx0 = dy0 * dsilu

    dx0_ptrs = GRAD_X + n * stride_dxn + offs_c * stride_dxc
    tl.store(dx0_ptrs, dx0, mask=mask_c)

    # -------------------------------------------------------------------------
    # l >= 1 gated paths
    # -------------------------------------------------------------------------
    for l_idx in tl.static_range(0, LMAX):
        start_m = tl.load(STARTS + l_idx).to(tl.int32)
        count_m = tl.load(COUNTS + l_idx).to(tl.int32)

        g_ptrs = G + n * stride_gn + (l_idx * C + offs_c) * stride_gc
        g = tl.load(g_ptrs, mask=mask_c, other=0.0).to(tl.float32)

        sg = tl.sigmoid(g)
        dsg = sg * (1.0 - sg)

        dg_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for j in tl.static_range(0, MAX_COUNT):
            valid_m = j < count_m
            mask = mask_c & valid_m

            m = start_m + j

            x_ptrs = X + n * stride_xn + m * stride_xm + offs_c * stride_xc
            dy_ptrs = (
                GRAD_Y
                + n * stride_dyn
                + m * stride_dym
                + offs_c * stride_dyc
            )

            x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

            # dL/dx = dL/dy * sigmoid(g)
            dx = dy * sg

            dx_ptrs = (
                GRAD_X
                + n * stride_dxn
                + m * stride_dxm
                + offs_c * stride_dxc
            )
            tl.store(dx_ptrs, dx, mask=mask)

            # dL/dg accumulates all m-components sharing this l gate.
            dg_term = dy * x * dsg
            dg_acc += tl.where(valid_m, dg_term, 0.0)

        dg_ptrs = (
            GRAD_G
            + n * stride_dgn
            + (l_idx * C + offs_c) * stride_dgc
        )
        tl.store(dg_ptrs, dg_acc, mask=mask_c)


# -----------------------------------------------------------------------------
# Autograd wrapper
# -----------------------------------------------------------------------------

class _FusedGateActivationFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gating_scalars: torch.Tensor,   # [N, L*C]
        input_tensors: torch.Tensor,    # [N, M, C]
        expand_index: torch.Tensor,     # [M-1], int32
        starts: torch.Tensor,           # [L], int32
        counts: torch.Tensor,           # [L], int32
        lmax: int,
        max_count: int,
    ) -> torch.Tensor:
        if not gating_scalars.is_cuda or not input_tensors.is_cuda:
            raise RuntimeError("Fused GateActivation Triton path requires CUDA tensors.")

        if gating_scalars.ndim != 2:
            raise ValueError(
                f"gating_scalars must be [N, L*C], got {tuple(gating_scalars.shape)}"
            )
        if input_tensors.ndim != 3:
            raise ValueError(
                f"input_tensors must be [N, M, C], got {tuple(input_tensors.shape)}"
            )

        N, M, C = input_tensors.shape

        if gating_scalars.shape[0] != N:
            raise ValueError("gating_scalars and input_tensors must have the same N.")
        if gating_scalars.shape[1] != lmax * C:
            raise ValueError(
                f"Expected gating_scalars.shape[1] == lmax*C == {lmax*C}, "
                f"got {gating_scalars.shape[1]}"
            )
        if expand_index.numel() != M - 1:
            raise ValueError(
                f"Expected M-1 == len(expand_index) == {expand_index.numel()}, "
                f"got M={M}"
            )

        if expand_index.dtype != torch.int32:
            expand_index = expand_index.to(dtype=torch.int32)
        if starts.dtype != torch.int32:
            starts = starts.to(dtype=torch.int32)
        if counts.dtype != torch.int32:
            counts = counts.to(dtype=torch.int32)

        # Metadata must be on the same CUDA device.
        device = input_tensors.device
        if expand_index.device != device:
            expand_index = expand_index.to(device)
        if starts.device != device:
            starts = starts.to(device)
        if counts.device != device:
            counts = counts.to(device)

        y = torch.empty_like(input_tensors)

        block_c = min(256, triton.next_power_of_2(C))
        grid = (N, M, triton.cdiv(C, block_c))

        _gate_activation_fwd_kernel[grid](
            input_tensors,
            gating_scalars,
            expand_index,
            y,
            N=N,
            M=M,
            C=C,
            stride_xn=input_tensors.stride(0),
            stride_xm=input_tensors.stride(1),
            stride_xc=input_tensors.stride(2),
            stride_gn=gating_scalars.stride(0),
            stride_gc=gating_scalars.stride(1),
            stride_yn=y.stride(0),
            stride_ym=y.stride(1),
            stride_yc=y.stride(2),
            BLOCK_C=block_c,
            num_warps=4,
        )

        ctx.save_for_backward(input_tensors, gating_scalars, starts, counts)
        ctx.lmax = int(lmax)
        ctx.max_count = int(max_count)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        input_tensors, gating_scalars, starts, counts = ctx.saved_tensors

        grad_x = torch.empty_like(input_tensors)
        grad_g = torch.empty_like(gating_scalars)

        N, _, C = input_tensors.shape
        block_c = min(256, triton.next_power_of_2(C))
        grid = (N, triton.cdiv(C, block_c))

        _gate_activation_bwd_kernel[grid](
            grad_y,
            input_tensors,
            gating_scalars,
            starts,
            counts,
            grad_x,
            grad_g,
            C=C,
            stride_dyn=grad_y.stride(0),
            stride_dym=grad_y.stride(1),
            stride_dyc=grad_y.stride(2),
            stride_xn=input_tensors.stride(0),
            stride_xm=input_tensors.stride(1),
            stride_xc=input_tensors.stride(2),
            stride_gn=gating_scalars.stride(0),
            stride_gc=gating_scalars.stride(1),
            stride_dxn=grad_x.stride(0),
            stride_dxm=grad_x.stride(1),
            stride_dxc=grad_x.stride(2),
            stride_dgn=grad_g.stride(0),
            stride_dgc=grad_g.stride(1),
            LMAX=ctx.lmax,
            MAX_COUNT=ctx.max_count,
            BLOCK_C=block_c,
            num_warps=4,
        )

        # forward args:
        # gating_scalars, input_tensors, expand_index, starts, counts, lmax, max_count
        return grad_g, grad_x, None, None, None, None, None


def fused_gate_activation(
    gating_scalars: torch.Tensor,
    input_tensors: torch.Tensor,
    expand_index: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    lmax: int,
    max_count: int,
) -> torch.Tensor:
    return _FusedGateActivationFn.apply(
        gating_scalars,
        input_tensors,
        expand_index,
        starts,
        counts,
        lmax,
        max_count,
    )


# -----------------------------------------------------------------------------
# Drop-in module
# -----------------------------------------------------------------------------

class fused_gate_activation(nn.Module):
    """
    Fused replacement for the original GateActivation.

    Forward semantics:
      input_tensors[:, 0, :] = SiLU(input_tensors[:, 0, :])

      for l >= 1:
        input_tensors[:, components_of_l, :]
          *= sigmoid(gating_scalars[:, l-1, :])

    No expanded gate tensor is materialized.

    Notes:
      - CUDA/Triton path supports first-order backward.
      - This implementation does NOT provide double backward.
      - For mmax == lmax, M == (lmax + 1)^2.
      - For mmax < lmax, M is truncated consistently with the original
        expand_index construction.
    """

    def __init__(self, lmax: int, mmax: int, num_channels: int) -> None:
        super().__init__()

        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.num_channels = int(num_channels)

        expand_index = []
        starts = []
        counts = []

        component_cursor = 1  # m=0 / scalar occupies component 0
        max_count = 0

        for lval in range(1, self.lmax + 1):
            count = min(2 * lval + 1, 2 * self.mmax + 1)

            starts.append(component_cursor)
            counts.append(count)
            expand_index.extend([lval - 1] * count)

            component_cursor += count
            max_count = max(max_count, count)

        self.num_components = component_cursor
        self.max_count = max(1, max_count)

        self.register_buffer(
            "expand_index",
            torch.tensor(expand_index, dtype=torch.int32),
        )
        self.register_buffer(
            "_gate_starts",
            torch.tensor(starts, dtype=torch.int32),
        )
        self.register_buffer(
            "_gate_counts",
            torch.tensor(counts, dtype=torch.int32),
        )