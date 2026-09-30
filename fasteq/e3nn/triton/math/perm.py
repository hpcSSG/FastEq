"""GPU construction of natural and standard permutation representations."""

import torch
import triton
import triton.language as tl

from .._common import grid
from ._linalg import complete_basis
from e3nn.math.perm import (
    is_perm, identity, compose, inverse, rand, from_int, to_int,
    group, germinate, is_group, to_cycles, sign,
    standard_representation as _reference_standard_representation,
)


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
    if sorted(p) != list(range(n)):
        raise ValueError("p must be a permutation")
    inverse = tuple(p.index(i) for i in range(n))
    out = torch.empty((n, n), dtype=dtype, device=device)
    if not out.is_cuda:
        raise ValueError("Triton implementation requires GPU output")
    if n:
        _natural_kernel[grid(n * n)](out, n, inverse, 256)
    return out


@triton.jit
def _standard_kernel(A, P, OUT, N: tl.constexpr, R: tl.constexpr,
                     BLOCK_OUT: tl.constexpr, BLOCK_N: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    row = idx // R
    col = idx % R
    k = tl.arange(0, BLOCK_N)
    pk = tl.load(P + k, k < N, other=0)
    valid = (idx[:, None] < R * R) & (k[None, :] < N)
    # P_native[a, inverse(p)[a]] = 1, so (A @ P_native)[i,k]
    # equals A[i,p[k]]. Gather and reduction avoid both P_native and A @ P.
    left = tl.load(A + row[:, None] * N + pk[None, :], valid, other=0.)
    right = tl.load(A + col[:, None] * N + k[None, :], valid, other=0.)
    value = tl.sum(left * right, 1)
    tl.store(OUT + idx, value, idx < R * R)


def standard_representation(p, dtype=None, device=None) -> torch.Tensor:
    """Standard Sn representation via Triton complete_basis and projection.

    CUDA FP32/FP64 for 1 <= len(p) <= 4096 uses the Triton path. The basis
    cutoff, row order and sign conventions remain those of native e3nn.
    Host/dtype/size fallback retains the original implementation.
    """
    n = len(p)
    if n == 0 or n > 4096 or not is_perm(p):
        return _reference_standard_representation(p, dtype=dtype, device=device)
    base = torch.ones((1, n), dtype=dtype, device=device)
    if base.device.type != "cuda" or base.dtype not in (torch.float32, torch.float64):
        return _reference_standard_representation(p, dtype=dtype, device=device)
    A = complete_basis(base, eps=0.1 / n)
    r = A.shape[0]
    out = A.new_empty((r, r))
    if r:
        permutation = torch.tensor(p, dtype=torch.int64, device=A.device)
        _standard_kernel[grid(out.numel(), 16)](
            A, permutation, out, n, r, 16, triton.next_power_of_2(n),
            num_warps=4, enable_fp_fusion=False,
        )
    return out
