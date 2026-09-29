"""Torch sampling and arbitrary activation, Triton power and mean reduction."""

import torch
import triton
import triton.language as tl

from e3nn.math._normalize_activation import normalize2mom, moment as _reference_moment
from e3nn.util.default_type import explicit_default_types


@triton.jit
def _moment_partial(X, TMP, N: tl.constexpr, POWER: tl.constexpr,
                    BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N
    x = tl.load(X + offs, valid, other=0)
    v = tl.full((BLOCK,), 1., x.dtype)
    for _ in tl.static_range(POWER):
        v *= x
    tl.store(TMP + tl.program_id(0), tl.sum(tl.where(valid, v, 0.), 0))


@triton.jit
def _moment_finish(TMP, OUT, PARTS: tl.constexpr,
                   N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    v = tl.load(TMP + i, i < PARTS, other=0)
    tl.store(OUT, tl.sum(v, 0) / N)


def moment(f, n, dtype=None, device=None):
    dtype, device = explicit_default_types(dtype, device)
    if torch.device(device).type != "cuda" or not isinstance(n, int) or n < 0 or n > 8:
        return _reference_moment(f, n, dtype=dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(0)
    z = torch.randn(1_000_000, generator=generator,
                    dtype=torch.float64, device=device).to(dtype=dtype, device=device)
    values = f(z)
    if (not values.is_cuda or values.numel() == 0
            or values.dtype not in (torch.float32, torch.float64) or values.requires_grad):
        return values.pow(n).mean()
    values = values.contiguous()
    block = 1024
    parts = triton.cdiv(values.numel(), block)
    tmp = values.new_empty((parts,))
    result = values.new_empty(())
    _moment_partial[(parts,)](values, tmp, values.numel(), n, block)
    _moment_finish[(1,)](tmp, result, parts, values.numel(), triton.next_power_of_2(parts))
    return result


__all__ = ["moment", "normalize2mom"]
