"""Fused edge gather, source/target merge, radial modulation and SO(3) rotate.

In equiformer_v3/experimental/models/equiformer_v3/transformer_block.py:
class EquivariantGraphAttention:

    ...
    def forward():
        ...
        # Fused this entire block: 
        ===============================Begin==========================================
        # Merge source/target node features
        x = x.to(x_edge_weight.dtype)
        x_source = torch.index_select(x, index=edge_index[0], dim=0)
        x_target = torch.index_select(x, index=edge_index[1], dim=0)
        if not self.use_add_merge:
            # Concat    
            x_message = torch.cat((x_source, x_target), dim=2)
            if self.use_rad_l_parametrization:
                x_message = x_message * x_edge_weight
                x_message = self.so3_rotation.rotate(x_message)
            else:
                x_message = self.so3_rotation.rotate(x_message)
                x_message = x_message * x_edge_weight
        elif self.use_add_merge:
            # Add
            x_edge_weight_source = x_edge_weight.narrow(2, 0, self.num_in_channels)
            x_edge_weight_target = x_edge_weight.narrow(2, self.num_in_channels, self.num_in_channels)
            x_source = x_source * x_edge_weight_source
            x_target = x_target * x_edge_weight_target
            x_message = x_source + x_target
            x_message = self.so3_rotation.rotate(x_message)
        ===============================End==========================================
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from math import isqrt


# Compile-time modes. Keep them as kernel arguments rather than accessing Python
# globals inside @triton.jit code.
MODE_CONCAT_PRE = 0
MODE_CONCAT_POST = 1
MODE_ADD_PRE = 2


@triton.jit
def _m_primary_row_to_l(
    row,
    LMAX: tl.constexpr,
    MMAX: tl.constexpr,
):
    """Map an m-primary output row to its angular degree l.

    Row order is: m=0 for l=0..LMAX, followed by the two signs of
    |m|=1, then the two signs of |m|=2, etc. The sign order is irrelevant
    because both sign groups have the same l sequence.
    """
    ell = row
    cursor = LMAX + 1
    for abs_m in tl.static_range(1, MMAX + 1):
        group_size = LMAX - abs_m + 1
        in_first = (row >= cursor) & (row < cursor + group_size)
        ell = tl.where(in_first, abs_m + row - cursor, ell)
        cursor += group_size
        in_second = (row >= cursor) & (row < cursor + group_size)
        ell = tl.where(in_second, abs_m + row - cursor, ell)
        cursor += group_size
    return ell


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_C": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_C": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 1, "BLOCK_C": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_C": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_C": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_C": 64}, num_warps=8),
    ],
    key=["M", "K", "C", "C_OUT", "W_L", "MODE", "LMAX", "MMAX"],
)
@triton.jit(do_not_specialize=["E"])
def _sparse_fused_gather_merge_radial_rotate_kernel(
    X,
    EDGE_INDEX,
    RADIAL,
    ROTATION,
    OUT,
    E,
    M: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    C_OUT: tl.constexpr,
    W_L: tl.constexpr,
    MODE: tl.constexpr,
    LMAX: tl.constexpr,
    MMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64)
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_m = rows < M
    mask_c = cols < C_OUT

    src = tl.load(EDGE_INDEX + edge).to(tl.int64)
    dst = tl.load(EDGE_INDEX + E + edge).to(tl.int64)

    # For a row belonging to degree l, only input columns
    # [l*l, l*l + 2*l + 1) are structurally nonzero.
    ell = _m_primary_row_to_l(rows, LMAX, MMAX)
    k_start = ell * ell
    k_count = 2 * ell + 1
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for path in tl.static_range(0,  2 * LMAX + 1):
        valid_mk = mask_m & (path < k_count)
        k = k_start + path
        r = tl.load(
            ROTATION + edge * (M * K) + rows * K + k,
            mask=valid_mk,
            other=0.0,
        ).to(tl.float32)

        if MODE == 0:  # concat, radial before rotate
            use_src = cols < C
            node = tl.where(use_src, src, dst)
            node_c = tl.where(use_src, cols, cols - C)
            xv = tl.load(
                X
                + node[None, :] * (K * C)
                + k[:, None] * C
                + node_c[None, :],
                mask=valid_mk[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            wk = tl.zeros((BLOCK_M,), tl.int32) if W_L == 1 else k
            w = tl.load(
                RADIAL
                + edge * (W_L * C_OUT)
                + wk[:, None] * C_OUT
                + cols[None, :],
                mask=valid_mk[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            message = xv * w

        elif MODE == 1:  # concat, radial after rotate
            use_src = cols < C
            node = tl.where(use_src, src, dst)
            node_c = tl.where(use_src, cols, cols - C)
            message = tl.load(
                X
                + node[None, :] * (K * C)
                + k[:, None] * C
                + node_c[None, :],
                mask=valid_mk[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)

        else:  # add, radial before rotate
            xs = tl.load(
                X + src * (K * C) + k[:, None] * C + cols[None, :],
                mask=valid_mk[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            xt = tl.load(
                X + dst * (K * C) + k[:, None] * C + cols[None, :],
                mask=valid_mk[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            wk = tl.zeros((BLOCK_M,), tl.int32) if W_L == 1 else k
            radial_base = (
                RADIAL
                + edge * (W_L * (2 * C))
                + wk[:, None] * (2 * C)
            )
            ws = tl.load(
                radial_base + cols[None, :],
                mask=valid_mk[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            wt = tl.load(
                radial_base + C + cols[None, :],
                mask=valid_mk[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
            message = xs * ws + xt * wt

        acc += r[:, None] * message

    if MODE == 1:
        wm = tl.zeros((BLOCK_M,), tl.int32) if W_L == 1 else rows
        w = tl.load(
            RADIAL
            + edge * (W_L * C_OUT)
            + wm[:, None] * C_OUT
            + cols[None, :],
            mask=mask_m[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)
        acc *= w

    tl.store(
        OUT
        + edge * (M * C_OUT)
        + rows[:, None] * C_OUT
        + cols[None, :],
        acc,
        mask=mask_m[:, None] & mask_c[None, :],
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_C": 32}, num_warps=1),
        triton.Config({"BLOCK_M": 2, "BLOCK_C": 32}, num_warps=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_C": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_C": 32}, num_warps=4),
    ],
    # E is deliberately absent: changing the number of edges reuses tuning.
    key=["M", "K", "C", "C_OUT", "W_L", "MODE"],
)
@triton.jit(do_not_specialize=["E"])
def _fused_gather_merge_radial_rotate_kernel(
    X,             # [V, K, C]
    EDGE_INDEX,    # [2, E]
    RADIAL,        # layouts documented above
    ROTATION,      # [E, M, K]
    OUT,           # [E, M, C_OUT]
    E,
    M: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    C_OUT: tl.constexpr,
    W_L: tl.constexpr,       # radial.shape[1], either 1, K, or M
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64)
    offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_m = offs_m < M
    mask_c = offs_c < C_OUT

    # PyTorch source:
    #   x_source = torch.index_select(x, index=edge_index[0], dim=0)
    #   x_target = torch.index_select(x, index=edge_index[1], dim=0)
    src = tl.load(EDGE_INDEX + edge).to(tl.int64)
    dst = tl.load(EDGE_INDEX + E + edge).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # PyTorch source:
    #   x_message = self.so3_rotation.rotate(x_message)
    # Equivalent contraction:
    #   out[e, m, c] = sum_k rotation[e, m, k] * message[e, k, c]
    # K is constexpr, so the reduction is completely unrolled.
    for k in tl.static_range(0, K):
        r = tl.load(
            ROTATION + edge * (M * K) + offs_m * K + k,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)

        if MODE == 0:  # MODE_CONCAT_PRE
            # PyTorch source:
            #   x_message = torch.cat((x_source, x_target), dim=2)
            #   x_message = x_message * x_edge_weight
            use_src = offs_c < C
            node = tl.where(use_src, src, dst)
            node_c = tl.where(use_src, offs_c, offs_c - C)

            xv = tl.load(
                X + node * (K * C) + k * C + node_c,
                mask=mask_c,
                other=0.0,
            ).to(tl.float32)

            wk = 0 if W_L == 1 else k
            w = tl.load(
                RADIAL + edge * (W_L * C_OUT) + wk * C_OUT + offs_c,
                mask=mask_c,
                other=0.0,
            ).to(tl.float32)
            message = xv * w

        elif MODE == 1:  # MODE_CONCAT_POST
            # PyTorch source before rotation:
            #   x_message = torch.cat((x_source, x_target), dim=2)
            # Radial modulation is applied after the K reduction below.
            use_src = offs_c < C
            node = tl.where(use_src, src, dst)
            node_c = tl.where(use_src, offs_c, offs_c - C)
            message = tl.load(
                X + node * (K * C) + k * C + node_c,
                mask=mask_c,
                other=0.0,
            ).to(tl.float32)

        else:  # MODE_ADD_PRE
            # PyTorch source:
            #   weight_source = x_edge_weight[..., :C]
            #   weight_target = x_edge_weight[..., C:2*C]
            #   x_message = x_source * weight_source
            #             + x_target * weight_target
            xs = tl.load(
                X + src * (K * C) + k * C + offs_c,
                mask=mask_c,
                other=0.0,
            ).to(tl.float32)
            xt = tl.load(
                X + dst * (K * C) + k * C + offs_c,
                mask=mask_c,
                other=0.0,
            ).to(tl.float32)

            wk = 0 if W_L == 1 else k
            radial_base = RADIAL + edge * (W_L * (2 * C)) + wk * (2 * C)
            ws = tl.load(
                radial_base + offs_c,
                mask=mask_c,
                other=0.0,
            ).to(tl.float32)
            wt = tl.load(
                radial_base + C + offs_c,
                mask=mask_c,
                other=0.0,
            ).to(tl.float32)
            message = xs * ws + xt * wt

        acc += r[:, None] * message[None, :]

    if MODE == 1:  # MODE_CONCAT_POST
        # PyTorch source:
        #   x_message = self.so3_rotation.rotate(x_message)
        #   x_message = x_message * x_edge_weight
        wm = tl.zeros((BLOCK_M,), dtype=tl.int32) if W_L == 1 else offs_m
        w = tl.load(
            RADIAL
            + edge * (W_L * C_OUT)
            + wm[:, None] * C_OUT
            + offs_c[None, :],
            mask=mask_m[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)
        acc *= w

    tl.store(
        OUT
        + edge * (M * C_OUT)
        + offs_m[:, None] * C_OUT
        + offs_c[None, :],
        acc,
        mask=mask_m[:, None] & mask_c[None, :],
    )


# Backward for MODE_CONCAT_PRE and MODE_ADD_PRE. Each program owns one
# (edge, k, channel tile). Gradients to x use atomics because many edges can
# reference the same node. A broadcast radial dimension (W_L == 1) also needs
# atomics to reduce over k.
@triton.jit(do_not_specialize=["E"])
def _backward_pre_x_radial_kernel(
    X,
    EDGE_INDEX,
    RADIAL,
    ROTATION,
    DY,
    DX,
    DRADIAL,
    E,
    M: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    C_OUT: tl.constexpr,
    W_L: tl.constexpr,
    MODE: tl.constexpr,
    LMAX: tl.constexpr,
    MMAX: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DRADIAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1)
    offs_c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C_OUT

    src = tl.load(EDGE_INDEX + edge).to(tl.int64)
    dst = tl.load(EDGE_INDEX + E + edge).to(tl.int64)

    # k belongs to exactly one l block. In m-primary layout that l appears in
    # at most 1 + 2*MMAX output rows, rather than all M rows.
    ell = 0
    for l_test in tl.static_range(1, LMAX + 1):
        ell += (k >= l_test * l_test).to(tl.int32)

    # dmessage[e,k,c] = sum_{m in same l} rotation[e,m,k] * dy[e,m,c]
    dmessage = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for slot in tl.static_range(0, 2 * MMAX + 1):
        if slot == 0:
            m = ell
            valid_m = True
        else:
            abs_m = (slot + 1) // 2
            group_size = LMAX - abs_m + 1
            previous = (
                (abs_m - 1) * (LMAX + 1)
                - (abs_m - 1) * abs_m // 2
            )
            first_base = LMAX + 1 + 2 * previous
            second_offset = group_size if slot % 2 == 0 else 0
            m = first_base + second_offset + ell - abs_m
            valid_m = ell >= abs_m

        r = tl.load(
            ROTATION + edge * (M * K) + m * K + k,
            mask=valid_m,
            other=0.0,
        ).to(tl.float32)
        dy = tl.load(
            DY + edge * (M * C_OUT) + m * C_OUT + offs_c,
            mask=mask_c & valid_m,
            other=0.0,
        ).to(tl.float32)
        dmessage += r * dy

    wk = 0 if W_L == 1 else k

    if MODE == 0:  # MODE_CONCAT_PRE
        use_src = offs_c < C
        node = tl.where(use_src, src, dst)
        node_c = tl.where(use_src, offs_c, offs_c - C)
        xv = tl.load(
            X + node * (K * C) + k * C + node_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        w_ptr = RADIAL + edge * (W_L * C_OUT) + wk * C_OUT + offs_c
        w = tl.load(w_ptr, mask=mask_c, other=0.0).to(tl.float32)

        if NEED_DX:
            tl.atomic_add(
                DX + node * (K * C) + k * C + node_c,
                dmessage * w,
                mask=mask_c,
            )
        if NEED_DRADIAL:
            dw = dmessage * xv
            dw_ptr = DRADIAL + edge * (W_L * C_OUT) + wk * C_OUT + offs_c
            if W_L == 1:
                tl.atomic_add(dw_ptr, dw, mask=mask_c)
            else:
                tl.store(dw_ptr, dw, mask=mask_c)

    else:  # MODE_ADD_PRE
        xs = tl.load(
            X + src * (K * C) + k * C + offs_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        xt = tl.load(
            X + dst * (K * C) + k * C + offs_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        w_base = RADIAL + edge * (W_L * (2 * C)) + wk * (2 * C)
        ws = tl.load(w_base + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        wt = tl.load(w_base + C + offs_c, mask=mask_c, other=0.0).to(tl.float32)

        if NEED_DX:
            tl.atomic_add(
                DX + src * (K * C) + k * C + offs_c,
                dmessage * ws,
                mask=mask_c,
            )
            tl.atomic_add(
                DX + dst * (K * C) + k * C + offs_c,
                dmessage * wt,
                mask=mask_c,
            )
        if NEED_DRADIAL:
            dws = dmessage * xs
            dwt = dmessage * xt
            dws_ptr = DRADIAL + edge * (W_L * (2 * C)) + wk * (2 * C) + offs_c
            dwt_ptr = dws_ptr + C
            if W_L == 1:
                tl.atomic_add(dws_ptr, dws, mask=mask_c)
                tl.atomic_add(dwt_ptr, dwt, mask=mask_c)
            else:
                tl.store(dws_ptr, dws, mask=mask_c)
                tl.store(dwt_ptr, dwt, mask=mask_c)


# x gradient for MODE_CONCAT_POST. The post-rotation radial weight becomes part
# of dz before the transpose rotation is applied.
@triton.jit(do_not_specialize=["E"])
def _backward_post_x_kernel(
    EDGE_INDEX,
    RADIAL,
    ROTATION,
    DY,
    DX,
    E,
    M: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    C_OUT: tl.constexpr,
    W_L: tl.constexpr,
    LMAX: tl.constexpr,
    MMAX: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1)
    offs_c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C_OUT

    src = tl.load(EDGE_INDEX + edge).to(tl.int64)
    dst = tl.load(EDGE_INDEX + E + edge).to(tl.int64)
    use_src = offs_c < C
    node = tl.where(use_src, src, dst)
    node_c = tl.where(use_src, offs_c, offs_c - C)

    ell = 0
    for l_test in tl.static_range(1, LMAX + 1):
        ell += (k >= l_test * l_test).to(tl.int32)

    dx = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for slot in tl.static_range(0, 2 * MMAX + 1):
        if slot == 0:
            m = ell
            valid_m = True
        else:
            abs_m = (slot + 1) // 2
            group_size = LMAX - abs_m + 1
            previous = (
                (abs_m - 1) * (LMAX + 1)
                - (abs_m - 1) * abs_m // 2
            )
            first_base = LMAX + 1 + 2 * previous
            second_offset = group_size if slot % 2 == 0 else 0
            m = first_base + second_offset + ell - abs_m
            valid_m = ell >= abs_m

        r = tl.load(
            ROTATION + edge * (M * K) + m * K + k,
            mask=valid_m,
            other=0.0,
        ).to(tl.float32)
        dy = tl.load(
            DY + edge * (M * C_OUT) + m * C_OUT + offs_c,
            mask=mask_c & valid_m,
            other=0.0,
        ).to(tl.float32)
        wm = 0 if W_L == 1 else m
        w = tl.load(
            RADIAL + edge * (W_L * C_OUT) + wm * C_OUT + offs_c,
            mask=mask_c & valid_m,
            other=0.0,
        ).to(tl.float32)
        dx += r * dy * w

    tl.atomic_add(
        DX + node * (K * C) + k * C + node_c,
        dx,
        mask=mask_c,
    )


# Radial gradient for MODE_CONCAT_POST:
#   z = rotation @ concat(x_src, x_dst)
#   dradial = dy * z
@triton.jit(do_not_specialize=["E"])
def _backward_post_radial_kernel(
    X,
    EDGE_INDEX,
    ROTATION,
    DY,
    DRADIAL,
    E,
    M: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    C_OUT: tl.constexpr,
    W_L: tl.constexpr,
    LMAX: tl.constexpr,
    MMAX: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64)
    m = tl.program_id(1)
    offs_c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C_OUT

    src = tl.load(EDGE_INDEX + edge).to(tl.int64)
    dst = tl.load(EDGE_INDEX + E + edge).to(tl.int64)
    use_src = offs_c < C
    node = tl.where(use_src, src, dst)
    node_c = tl.where(use_src, offs_c, offs_c - C)

    ell = _m_primary_row_to_l(m, LMAX, MMAX)
    k_start = ell * ell
    k_count = 2 * ell + 1
    z = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for path in tl.static_range(0, 2 * LMAX + 1):
        valid_k = path < k_count
        k = k_start + path
        r = tl.load(
            ROTATION + edge * (M * K) + m * K + k,
            mask=valid_k,
            other=0.0,
        ).to(tl.float32)
        xv = tl.load(
            X + node * (K * C) + k * C + node_c,
            mask=mask_c & valid_k,
            other=0.0,
        ).to(tl.float32)
        z += tl.where(valid_k, r * xv, 0.0)

    dy = tl.load(
        DY + edge * (M * C_OUT) + m * C_OUT + offs_c,
        mask=mask_c,
        other=0.0,
    ).to(tl.float32)
    wm = 0 if W_L == 1 else m
    dw_ptr = DRADIAL + edge * (W_L * C_OUT) + wm * C_OUT + offs_c
    if W_L == 1:
        tl.atomic_add(dw_ptr, dy * z, mask=mask_c)
    else:
        tl.store(dw_ptr, dy * z, mask=mask_c)


# Rotation gradient. Each program owns one (edge,m,k), so no atomics are needed.
@triton.jit(do_not_specialize=["E"])
def _backward_rotation_kernel(
    X,
    EDGE_INDEX,
    RADIAL,
    DY,
    DROTATION,
    E,
    M: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    C_OUT: tl.constexpr,
    W_L: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64)
    m = tl.program_id(1)
    k = tl.program_id(2)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C_OUT

    src = tl.load(EDGE_INDEX + edge).to(tl.int64)
    dst = tl.load(EDGE_INDEX + E + edge).to(tl.int64)
    dy = tl.load(
        DY + edge * (M * C_OUT) + m * C_OUT + offs_c,
        mask=mask_c,
        other=0.0,
    ).to(tl.float32)

    if MODE == 0:  # MODE_CONCAT_PRE
        use_src = offs_c < C
        node = tl.where(use_src, src, dst)
        node_c = tl.where(use_src, offs_c, offs_c - C)
        xv = tl.load(
            X + node * (K * C) + k * C + node_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        wk = 0 if W_L == 1 else k
        w = tl.load(
            RADIAL + edge * (W_L * C_OUT) + wk * C_OUT + offs_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        contribution = dy * xv * w

    elif MODE == 1:  # MODE_CONCAT_POST
        use_src = offs_c < C
        node = tl.where(use_src, src, dst)
        node_c = tl.where(use_src, offs_c, offs_c - C)
        xv = tl.load(
            X + node * (K * C) + k * C + node_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        wm = 0 if W_L == 1 else m
        w = tl.load(
            RADIAL + edge * (W_L * C_OUT) + wm * C_OUT + offs_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        contribution = dy * w * xv

    else:  # MODE_ADD_PRE
        xs = tl.load(
            X + src * (K * C) + k * C + offs_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        xt = tl.load(
            X + dst * (K * C) + k * C + offs_c,
            mask=mask_c,
            other=0.0,
        ).to(tl.float32)
        wk = 0 if W_L == 1 else k
        w_base = RADIAL + edge * (W_L * (2 * C)) + wk * (2 * C)
        ws = tl.load(w_base + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        wt = tl.load(w_base + C + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        contribution = dy * (xs * ws + xt * wt)

    dr = tl.sum(contribution, axis=0)
    tl.store(DROTATION + edge * (M * K) + m * K + k, dr)


def _infer_m_primary_layout(M: int, K: int) -> tuple[int, int]:
    """Infer lmax/mmax from K=(lmax+1)^2 and m-primary output size."""
    side = isqrt(K)
    if side * side != K:
        raise ValueError(f"sparse Wigner path requires square K, got K={K}")
    lmax = side - 1
    for mmax in range(lmax + 1):
        expected_m = (lmax + 1) + 2 * sum(
            lmax - abs_m + 1 for abs_m in range(1, mmax + 1)
        )
        if expected_m == M:
            return lmax, mmax
    raise ValueError(
        f"cannot infer m-primary layout from M={M}, K={K}; "
        "use the dense fallback for this coefficient ordering"
    )


def _validate_inputs(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    x_edge_weight: torch.Tensor,
    rotation: torch.Tensor,
    use_add_merge: bool,
    use_rad_l_parametrization: bool,
) -> tuple[int, int, int, int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [V,K,C], got {tuple(x.shape)}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if rotation.ndim != 3:
        raise ValueError("rotation must have shape [E,M,K]")
    if x_edge_weight.ndim != 3:
        raise ValueError("x_edge_weight must have shape [E,L,Cw]")

    if not all(t.is_cuda for t in (x, edge_index, x_edge_weight, rotation)):
        raise ValueError("all inputs must be CUDA tensors")
    if not (x.device == edge_index.device == x_edge_weight.device == rotation.device):
        raise ValueError("all inputs must be on the same CUDA device")
    if x.dtype != torch.float32 or x_edge_weight.dtype != torch.float32:
        raise TypeError("this implementation currently requires float32 x and radial weights")
    if rotation.dtype != torch.float32:
        raise TypeError("this implementation currently requires float32 rotation matrices")
    if edge_index.dtype != torch.int64:
        raise TypeError("edge_index must use torch.int64")

    V, K, C = x.shape
    E = edge_index.shape[1]
    Er, M, Kr = rotation.shape
    Ew, W_L, Cw = x_edge_weight.shape
    if Er != E or Ew != E or Kr != K:
        raise ValueError(
            f"incompatible E/K: x={tuple(x.shape)}, edge_index={tuple(edge_index.shape)}, "
            f"radial={tuple(x_edge_weight.shape)}, rotation={tuple(rotation.shape)}"
        )
    if V == 0 or K == 0 or M == 0 or C == 0:
        raise ValueError("V, K, M and C must be positive")
    if use_add_merge:
        # In the supplied model, add merge always applies radial weights before
        # rotation; use_rad_l_parametrization does not alter this branch.
        mode = MODE_ADD_PRE
        c_out = C
        expected_l = (1, K)
        expected_cw = 2 * C
    elif use_rad_l_parametrization:
        mode = MODE_CONCAT_PRE
        c_out = 2 * C
        expected_l = (1, K)
        expected_cw = 2 * C
    else:
        mode = MODE_CONCAT_POST
        c_out = 2 * C
        expected_l = (1, M)
        expected_cw = 2 * C

    if W_L not in expected_l or Cw != expected_cw:
        raise ValueError(
            f"expected radial [E,L,{expected_cw}] with L in {expected_l}, "
            f"got {tuple(x_edge_weight.shape)}"
        )
    _infer_m_primary_layout(M, K)
    return E, M, K, C, c_out, mode


def _launch_forward(x, edge_index, radial, rotation, mode, c_out):
    E = edge_index.shape[1]
    _, K, C = x.shape
    _, M, _ = rotation.shape
    lmax, mmax = _infer_m_primary_layout(M, K)
    out = torch.empty((E, M, c_out), device=x.device, dtype=radial.dtype)
    if E == 0:
        return out

    grid = lambda meta: (
        E,
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(c_out, meta["BLOCK_C"]),
    )
    with torch.cuda.device(x.device):
        _sparse_fused_gather_merge_radial_rotate_kernel[grid](
            x,
            edge_index,
            radial,
            rotation,
            out,
            E=E,
            M=M,
            K=K,
            C=C,
            C_OUT=c_out,
            W_L=radial.shape[1],
            MODE=mode,
            LMAX=lmax,
            MMAX=mmax,
        )
    return out


class _FusedGatherMergeRadialRotateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, edge_index, radial, rotation, mode, c_out):
        out = _launch_forward(x, edge_index, radial, rotation, mode, c_out)
        ctx.save_for_backward(x, edge_index, radial, rotation)
        ctx.mode = mode
        ctx.c_out = c_out
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x, edge_index, radial, rotation = ctx.saved_tensors
        mode = ctx.mode
        c_out = ctx.c_out
        need_dx = ctx.needs_input_grad[0]
        need_dradial = ctx.needs_input_grad[2]
        need_drotation = ctx.needs_input_grad[3]

        V, K, C = x.shape
        E = edge_index.shape[1]
        _, M, _ = rotation.shape
        lmax, mmax = _infer_m_primary_layout(M, K)
        W_L = radial.shape[1]
        dy = dy.contiguous()

        # x is shared by edges, so its kernel uses atomic_add into a zero buffer.
        dx = torch.zeros_like(x) if need_dx else None
        # Broadcast radial layouts require reduction and therefore a zero buffer.
        # Non-broadcast layouts are uniquely written, but zeros are also safe and
        # make the empty-E path straightforward.
        dradial = torch.zeros_like(radial) if need_dradial else None
        drotation = torch.empty_like(rotation) if need_drotation else None

        if E:
            block_c = 32 if c_out <= 64 else 64
            x_ptr = dx if need_dx else x
            radial_ptr = dradial if need_dradial else radial

            with torch.cuda.device(x.device):
                if mode == MODE_CONCAT_POST:
                    if need_dx:
                        grid_x = (E, K, triton.cdiv(c_out, block_c))
                        _backward_post_x_kernel[grid_x](
                            edge_index,
                            radial,
                            rotation,
                            dy,
                            x_ptr,
                            E=E,
                            M=M,
                            K=K,
                            C=C,
                            C_OUT=c_out,
                            W_L=W_L,
                            LMAX=lmax,
                            MMAX=mmax,
                            BLOCK_C=block_c,
                            num_warps=4,
                        )
                    if need_dradial:
                        grid_w = (E, M, triton.cdiv(c_out, block_c))
                        _backward_post_radial_kernel[grid_w](
                            x,
                            edge_index,
                            rotation,
                            dy,
                            radial_ptr,
                            E=E,
                            M=M,
                            K=K,
                            C=C,
                            C_OUT=c_out,
                            W_L=W_L,
                            LMAX=lmax,
                            MMAX=mmax,
                            BLOCK_C=block_c,
                            num_warps=4,
                        )
                elif need_dx or need_dradial:
                    grid_xw = (E, K, triton.cdiv(c_out, block_c))
                    _backward_pre_x_radial_kernel[grid_xw](
                        x,
                        edge_index,
                        radial,
                        rotation,
                        dy,
                        x_ptr,
                        radial_ptr,
                        E=E,
                        M=M,
                        K=K,
                        C=C,
                        C_OUT=c_out,
                        W_L=W_L,
                        MODE=mode,
                        LMAX=lmax,
                        MMAX=mmax,
                        NEED_DX=need_dx,
                        NEED_DRADIAL=need_dradial,
                        BLOCK_C=block_c,
                        num_warps=4,
                    )

                if need_drotation:
                    # One program reduces all output channels for one [e,m,k].
                    reduce_c = triton.next_power_of_2(c_out)
                    grid_r = (E, M, K)
                    _backward_rotation_kernel[grid_r](
                        x,
                        edge_index,
                        radial,
                        dy,
                        drotation,
                        E=E,
                        M=M,
                        K=K,
                        C=C,
                        C_OUT=c_out,
                        W_L=W_L,
                        MODE=mode,
                        BLOCK_C=reduce_c,
                        num_warps=8 if reduce_c >= 256 else 4,
                    )

        # forward inputs: x, edge_index, radial, rotation, mode, c_out
        return dx, None, dradial, drotation, None, None


def fused_gather_merge_radial_rotate(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    x_edge_weight: torch.Tensor,
    rotation: torch.Tensor,
    *,
    use_add_merge: bool,
    use_rad_l_parametrization: bool,
) -> torch.Tensor:
    """Fused forward with first-order gradients for x, radial, and rotation.

    ``rotation`` must be the actual dense matrix used by
    ``self.so3_rotation.rotate``, with layout ``[E,M,K]``. Its output rows must
    use the verified m-primary ordering. Double backward is unsupported.
    """
    _, _, _, _, c_out, mode = _validate_inputs(
        x,
        edge_index,
        x_edge_weight,
        rotation,
        use_add_merge,
        use_rad_l_parametrization,
    )

    # Contiguous operations remain visible to autograd, so gradients are mapped
    # back to the original views/layouts correctly.
    return _FusedGatherMergeRadialRotateFunction.apply(
        x.contiguous(),
        edge_index.contiguous(),
        x_edge_weight.contiguous(),
        rotation.contiguous(),
        mode,
        c_out,
    )


def torch_gather_merge_radial_rotate(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    x_edge_weight: torch.Tensor,
    rotation: torch.Tensor,
    *,
    use_add_merge: bool,
    use_rad_l_parametrization: bool,
) -> torch.Tensor:
    """Literal Torch reference corresponding to the model source."""
    x = x.to(x_edge_weight.dtype)
    x_source = torch.index_select(x, index=edge_index[0], dim=0)
    x_target = torch.index_select(x, index=edge_index[1], dim=0)

    if not use_add_merge:
        x_message = torch.cat((x_source, x_target), dim=2)
        if use_rad_l_parametrization:
            x_message = x_message * x_edge_weight
            x_message = torch.bmm(rotation, x_message)
        else:
            x_message = torch.bmm(rotation, x_message)
            x_message = x_message * x_edge_weight
    else:
        c = x.shape[2]
        weight_source = x_edge_weight.narrow(2, 0, c)
        weight_target = x_edge_weight.narrow(2, c, c)
        x_source = x_source * weight_source
        x_target = x_target * weight_target
        x_message = x_source + x_target
        x_message = torch.bmm(rotation, x_message)

    return x_message