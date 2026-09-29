"""GPU construction of the natural permutation representation."""

import torch
import triton
import triton.language as tl

from .._common import grid


@triton.jit
def _natural_kernel(OUT, N: tl.constexpr, INV: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // N
    col = i % N
    value = tl.full((BLOCK,), 0., tl.float32)
    for j in tl.static_range(N):
        value = tl.where((row == j) & (col == INV[j]), 1., value)
    tl.store(OUT + i, value, i < N * N)


def natural_representation(p, dtype=None, device=None) -> torch.Tensor:
    n = len(p)
    if n == 0 or sorted(p) != list(range(n)):
        raise ValueError("p must be a nonempty permutation")
    inverse = tuple(p.index(i) for i in range(n))
    out = torch.empty((n, n), dtype=dtype, device=device)
    if not out.is_cuda:
        raise ValueError("Triton implementation requires GPU output")
    _natural_kernel[grid(n * n)](out, n, inverse, 256)
    return out
