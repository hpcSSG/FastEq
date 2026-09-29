"""Single launch for all six soft one-hot basis branches."""

import math

import torch
import triton
import triton.language as tl

from .._common import check_gpu, grid


@triton.jit
def _basis_kernel(X, Y, N: tl.constexpr, K: tl.constexpr, START: tl.constexpr,
                  END: tl.constexpr, BASIS: tl.constexpr, CUTOFF: tl.constexpr,
                  FOURIER_SCALE: tl.constexpr, BESSEL_SCALE: tl.constexpr,
                  BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i // K, i < N * K, other=0)
    k = i % K
    if CUTOFF:
        step = (END - START) / (K + 1)
        center = START + (k + 1) * step
    else:
        step = (END - START) / (K - 1)
        center = START + k * step
    diff = (x - center) / step
    if BASIS == 0:
        v = tl.exp(-diff * diff) / 1.12
    elif BASIS == 1:
        v = tl.where((diff < 1) & (-1 < diff), tl.cos(math.pi / 2 * diff), 0.)
    elif BASIS == 2:
        a = diff + 1.
        b = 1. - diff
        sa = tl.where(a > 0, a, 1.)
        sb = tl.where(b > 0, b, 1.)
        v = 1.14136 * math.e**2 * tl.where(a > 0, tl.exp(-1. / sa), 0.) * tl.where(b > 0, tl.exp(-1. / sb), 0.)
    elif BASIS == 3:
        t = (x - START) / (END - START)
        if CUTOFF:
            v = tl.where((t > 0) & (t < 1), tl.sin(math.pi * (k + 1) * t), 0.)
        else:
            v = tl.cos(math.pi * k * t)
        v = v * FOURIER_SCALE
    else:
        z = x - START
        c = END - START
        # Match upstream's 0/0 behavior for the bessel branch.
        v = BESSEL_SCALE * tl.sin((k + 1) * math.pi * z / c) / z
        if CUTOFF:
            v = v * ((z / c) < 1) * (z > 0)
    tl.store(Y + i, v, i < N * K)


def soft_one_hot_linspace(x: torch.Tensor, start, end, number, basis=None, cutoff=None) -> torch.Tensor:
    check_gpu(x)
    if cutoff not in (True, False):
        raise ValueError("cutoff must be specified")
    kinds = {"gaussian": 0, "cosine": 1, "smooth_finite": 2, "fourier": 3, "bessel": 4}
    if basis not in kinds:
        raise ValueError(f'basis="{basis}" is not a valid entry')
    if number < 2 or end == start:
        raise ValueError("number must be at least 2 and interval nonzero")
    x = x.contiguous()
    out = torch.empty((*x.shape, number), dtype=x.dtype, device=x.device)
    fourier_scale = 1.0 / math.sqrt(0.25 + number / 2) if basis == "fourier" else 1.0
    bessel_scale = math.sqrt(2.0 / (end - start)) if basis == "bessel" else 1.0
    _basis_kernel[grid(out.numel())](x, out, x.numel(), number, start, end,
        kinds[basis], cutoff, fourier_scale, bessel_scale, 256)
    return out
