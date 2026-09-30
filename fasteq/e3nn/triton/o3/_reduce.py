"""Permutation orbits in Python; projection-matrix assembly in Triton."""

import itertools

import torch
import triton
import triton.language as tl

from e3nn.math._reduce import (
    germinate_formulas,
    reduce_permutation as _reference_reduce_permutation,
)


@triton.jit
def _zero_projection_kernel(Q, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Q + i, 0., i < N)


@triton.jit
def _write_projection_kernel(INDICES, VALUES, Q, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = tl.load(INDICES + i, i < N, other=0)
    value = tl.load(VALUES + i, i < N, other=0.)
    tl.store(Q + offset, value, i < N)


def reduce_permutation(f0, formulas, dtype=None, device=None, **dims):
    """Return the native permutation basis ``(Q, ret)``.

    CUDA FP32/FP64 uses Triton for zeroing and writing Q. Dimension checks,
    orbit enumeration, sorting, sign conventions and ret remain in Python.
    Other devices/dtypes retain the original implementation. As in native
    e3nn, formulas should be the closed signed group from germinate_formulas.
    """
    # Resolve torch's default device/dtype, including set_default_device.
    prototype = torch.empty((), dtype=dtype, device=device)
    if prototype.device.type != "cuda" or prototype.dtype not in (torch.float32, torch.float64):
        return _reference_reduce_permutation(f0, formulas, dtype=dtype, device=device, **dims)

    for _s, p in formulas:
        f = "".join(f0[i] for i in p)
        for i, j in zip(f0, f):
            if i in dims and j in dims and dims[i] != dims[j]:
                raise RuntimeError(f"dimension of {i} and {j} should be the same")
            if i in dims:
                dims[j] = dims[i]
            if j in dims:
                dims[i] = dims[j]
    for i in f0:
        if i not in dims:
            raise RuntimeError(f"index {i} has no dimension associated to it")
    dimensions = [dims[i] for i in f0]

    full_base = list(itertools.product(*(range(d) for d in dimensions)))
    base = set()
    for x in full_base:
        xs = {(s, tuple(x[i] for i in p)) for s, p in formulas}
        if (-1, x) not in xs:
            base.add(frozenset({frozenset(xs), frozenset({(-s, x) for s, x in xs})}))
    base = sorted([sorted([sorted(xs) for xs in x]) for x in base])

    ret = []
    # Deduplicate flattened destinations with last-write semantics, matching
    # native sequential assignments even if a caller supplies conflicting signs.
    entries = {}
    width = len(full_base)
    for i, orbit in enumerate(base):
        orbit = max(orbit, key=lambda xs: sum(s for s, x in xs))
        ret.append(orbit)
        for s, e in orbit:
            j = 0
            for k, d in zip(e, dimensions):
                j = j * d + k
            entries[i * width + j] = s / len(orbit) ** 0.5

    Q = prototype.new_empty((len(base), width))
    if Q.numel():
        _zero_projection_kernel[(triton.cdiv(Q.numel(), 256),)](Q, Q.numel(), 256)
    if entries:
        indices = torch.tensor(list(entries), dtype=torch.int64, device=Q.device)
        values = torch.tensor(list(entries.values()), dtype=Q.dtype, device=Q.device)
        _write_projection_kernel[(triton.cdiv(len(entries), 256),)](
            indices, values, Q, len(entries), 256,
        )
    return Q.reshape(len(base), *dimensions), ret


__all__ = ["germinate_formulas", "reduce_permutation"]
