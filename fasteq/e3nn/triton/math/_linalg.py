"""Block diagonal concatenation in one GPU launch."""

import torch
import triton
import triton.language as tl

from .._common import check_gpu, grid


@triton.jit
def _direct_sum_kernel(MATS, OUT, B: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
                       HEIGHTS: tl.constexpr, WIDTHS: tl.constexpr, R0: tl.constexpr,
                       C0: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = (idx // COLS) % ROWS
    col = idx % COLS
    batch = idx // (ROWS * COLS)
    val = tl.full((BLOCK,), 0, tl.float64 if OUT.dtype.element_ty == tl.float64 else tl.float32)
    for j in tl.static_range(len(MATS)):
        valid = (batch < B) & (row >= R0[j]) & (row < R0[j] + HEIGHTS[j]) & (col >= C0[j]) & (col < C0[j] + WIDTHS[j])
        v = tl.load(MATS[j] + batch * HEIGHTS[j] * WIDTHS[j]
                    + (row - R0[j]) * WIDTHS[j] + (col - C0[j]), valid, other=0)
        val = val + v
    tl.store(OUT + idx, val, idx < B * ROWS * COLS)


def direct_sum(*matrices):
    if not matrices:
        raise ValueError("at least one matrix required")
    check_gpu(*matrices)
    front = matrices[0].shape[:-2]
    if any(m.shape[:-2] != front or m.dtype != matrices[0].dtype or m.device != matrices[0].device for m in matrices):
        raise ValueError("matrix batch shapes, dtypes, and devices must match")
    heights = tuple(m.shape[-2] for m in matrices)
    widths = tuple(m.shape[-1] for m in matrices)
    r0, c0 = [], []
    for h, w in zip(heights, widths):
        r0.append(sum(heights[:len(r0)]))
        c0.append(sum(widths[:len(c0)]))
    rows, cols = sum(heights), sum(widths)
    out = matrices[0].new_empty((*front, rows, cols))
    batch = matrices[0].numel() // (heights[0] * widths[0])
    _direct_sum_kernel[grid(out.numel())](tuple(m.contiguous() for m in matrices), out, batch,
                                           rows, cols, heights, widths, tuple(r0), tuple(c0), 256)
    return out
