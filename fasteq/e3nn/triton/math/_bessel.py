"""Bessel radial basis, with the upstream zero limit."""

import math

import torch
import triton
import triton.language as tl

from .._common import check_gpu, grid


@triton.jit
def _bessel_kernel(X, Y, N: tl.constexpr, K: tl.constexpr, XMAX: tl.constexpr,
                   SCALE: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + idx // K, idx < N * K, other=0)
    k = idx % K + 1
    safe = tl.where(x == 0, 1.0, x)
    v = tl.where(x == 0, k * math.pi / XMAX, tl.sin(k * math.pi / XMAX * safe) / safe)
    tl.store(Y + idx, SCALE * v, idx < N * K)


def bessel(x: torch.Tensor, n: int, x_max: float = 1.0) -> torch.Tensor:
    check_gpu(x)
    assert isinstance(n, int)
    if n < 1 or x_max <= 0:
        raise ValueError("n and x_max must be positive")
    x = x.contiguous()
    out = torch.empty((*x.shape, n), dtype=x.dtype, device=x.device)
    _bessel_kernel[grid(out.numel())](x, out, x.numel(), n, x_max,
                                     math.sqrt(2.0 / x_max), 256)
    return out
