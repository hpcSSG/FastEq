"""Block diagonal assembly and sequential Gram--Schmidt on the GPU.

Orthonormalization uses one launch per candidate and one final host read of
the rank. CPU, gradient-bearing, unsupported dtype and oversized inputs use
the native e3nn implementation. These wrappers are not TorchScript functions.
"""

import math
import torch
import triton
import triton.language as tl

from .._common import check_gpu, grid
from e3nn.math._linalg import (
    _conditional_script,
    orthonormalize as _reference_orthonormalize,
    complete_basis as _reference_complete_basis,
)


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


@triton.jit
def _normalize_rows_kernel(X, Q, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + d, d < D, other=0)
    norm = tl.sqrt(tl.sum(x * x, 0))
    # Native complete_basis normalizes each input independently. In particular
    # it does not orthogonalize the supplied base or discard zero base vectors.
    tl.store(Q + row * D + d, x / norm, d < D)


@triton.jit
def _gs_candidate_kernel(X, Q, C, COUNT, ROW, D: tl.constexpr,
                         M: tl.constexpr, EPS: tl.constexpr,
                         WITH_MATRIX: tl.constexpr,
                         BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr):
    d = tl.arange(0, BLOCK_D)
    if WITH_MATRIX:
        x = tl.load(X + ROW * D + d, d < D, other=0)
        k = tl.arange(0, BLOCK_M)
        cx = (k == ROW).to(x.dtype)
    else:
        # Candidate ROW of the identity, without allocating torch.eye.
        x = (d == ROW).to(Q.dtype.element_ty)

    rank = tl.load(COUNT)
    # Modified Gram--Schmidt: use the UPDATED x in each dot product.
    # A simultaneous projection onto all rows would change native semantics.
    for j in range(rank):
        y = tl.load(Q + j * D + d, d < D, other=0)
        projection = tl.sum(x * y, 0)
        x = x - projection * y
        if WITH_MATRIX:
            cy = tl.load(C + j * M + k, k < M, other=0)
            cx = cx - projection * cy

    norm = tl.sqrt(tl.sum(x * x, 0))
    if norm > 2 * EPS:
        inverse_norm = 1.0 / norm
        x = x * inverse_norm
        x = tl.where(tl.abs(x) < EPS, 0., x)
        first = tl.min(tl.where((d < D) & (x != 0), d, BLOCK_D), 0)
        pivot = tl.sum(tl.where(d == first, x, 0.), 0)
        sign = tl.where(pivot > 0, 1., -1.)
        tl.store(Q + rank * D + d, sign * x, d < D)
        if WITH_MATRIX:
            cx = cx * inverse_norm
            cx = tl.where(tl.abs(cx) < EPS, 0., cx)
            tl.store(C + rank * M + k, sign * cx, k < M)
        tl.store(COUNT, rank + 1)


def _can_use_triton(x, eps):
    # Bound the single-program vectors; large problems need tiled reductions.
    # Restrict eps so zeroing cannot erase every component of a unit vector.
    return (
        x.device.type == "cuda"
        and x.dtype in (torch.float32, torch.float64)
        and not x.requires_grad
        and x.ndim == 2
        and max(x.shape) <= 4096
        and isinstance(eps, (float, int))
        and math.isfinite(eps)
        and 0 <= eps < 1 / math.sqrt(max(1, x.shape[1]))
    )


def orthonormalize(original: torch.Tensor, eps: float = 1e-9):
    """Return ``(final, matrix)`` with native ordering, cutoff and pivot sign.

    The supported CUDA FP32/FP64 no-grad path performs all Gram--Schmidt
    arithmetic in Triton. Output shapes are data-dependent, requiring one
    device-to-host rank read. Numerical rank close to the cutoff can depend on
    reduction rounding, as it also does across native backends.
    """
    if not _can_use_triton(original, eps):
        return _reference_orthonormalize(original, eps)
    rows, dim = original.shape
    if rows == 0 or dim == 0:
        return original.new_empty((0, dim)), original.new_empty((0, rows))
    original = original.contiguous()
    final = original.new_empty((rows, dim))
    matrix = original.new_empty((rows, rows))
    count = torch.zeros((), dtype=torch.int32, device=original.device)
    bd = max(32, triton.next_power_of_2(dim))
    bm = max(32, triton.next_power_of_2(rows))
    for i in range(rows):
        _gs_candidate_kernel[(1,)](
            original, final, matrix, count, i, dim, rows, eps, True, bd, bm,
            num_warps=4, enable_fp_fusion=False,
        )
    rank = int(count.item())
    return final[:rank].clone(), matrix[:rank].clone()


def complete_basis(vecs: torch.Tensor, eps: float = 1e-9):
    """Return additional basis vectors, retaining native input assumptions.

    Supplied vectors are normalized individually, not orthogonalized. For an
    orthogonal input base the result completes that base. A nonorthogonal or
    zero input follows native behavior rather than silently repairing it.
    """
    if not _can_use_triton(vecs, eps):
        return _reference_complete_basis(vecs, eps)
    rows, dim = vecs.shape
    if dim == 0:
        return vecs.new_empty((0, 0))
    vecs = vecs.contiguous()
    base = vecs.new_empty((rows + dim, dim))
    count = torch.full((), rows, dtype=torch.int32, device=vecs.device)
    bd = max(32, triton.next_power_of_2(dim))
    if rows:
        _normalize_rows_kernel[(rows,)](vecs, base, dim, bd, num_warps=4, enable_fp_fusion=False)
    for i in range(dim):
        # C is unused in this specialization. Reuse the same projection,
        # normalization and acceptance logic as orthonormalize.
        _gs_candidate_kernel[(1,)](
            vecs, base, base, count, i, dim, rows, eps, False, bd, 32,
            num_warps=4, enable_fp_fusion=False,
        )
    rank = int(count.item())
    return base[rows:rank].clone()
