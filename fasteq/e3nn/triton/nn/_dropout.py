"""Torch Bernoulli samples, Triton broadcast over irrep components."""

import torch
import triton
import triton.language as tl

from e3nn.nn._dropout import Dropout as _ReferenceDropout
from .._common import grid


@triton.jit
def _dropout_kernel(X, MASK, OUT, N: tl.constexpr, WIDTH: tl.constexpr,
                    SAMPLES: tl.constexpr, GROUPS: tl.constexpr,
                    STARTS: tl.constexpr, MULTS: tl.constexpr,
                    DIMS: tl.constexpr, IR_OFFSETS: tl.constexpr,
                    BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = idx % WIDTH
    batch = idx // (SAMPLES * WIDTH)
    ir = tl.full((BLOCK,), 0, tl.int32)
    for j in tl.static_range(len(STARTS)):
        valid = (channel >= STARTS[j]) & (channel < STARTS[j] + MULTS[j] * DIMS[j])
        ir = tl.where(valid, IR_OFFSETS[j] + (channel - STARTS[j]) // DIMS[j], ir)
    x = tl.load(X + idx, idx < N, other=0)
    mask = tl.load(MASK + batch * GROUPS + ir, idx < N, other=0)
    tl.store(OUT + idx, x * mask, idx < N)


class Dropout(_ReferenceDropout):
    def forward(self, x):
        if (not self.training or not x.is_cuda or x.requires_grad
                or x.dtype not in (torch.float32, torch.float64)
                or x.ndim < 2 or not 0 < self.p < 1
                or x.shape[-1] != self.irreps.dim):
            return super().forward(x)
        batch = x.shape[0]
        noises = []
        starts, mults, dims, ir_offsets = [], [], [], []
        c = g = 0
        for mul, (l, _) in self.irreps:
            # Keep the original number and order of Bernoulli calls.
            noise = x.new_empty(batch, mul)
            noise.bernoulli_(1 - self.p).div_(1 - self.p)
            noises.append(noise)
            d = 2 * l + 1
            starts.append(c)
            mults.append(mul)
            dims.append(d)
            ir_offsets.append(g)
            c += mul * d
            g += mul
        if not noises:
            return super().forward(x)
        mask = torch.cat(noises, dim=1)
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        if out.numel():
            _dropout_kernel[grid(out.numel())](
                x.contiguous(), mask, out, out.numel(), c,
                x.numel() // (batch * c), g, tuple(starts), tuple(mults),
                tuple(dims), tuple(ir_offsets), 256,
            )
        return out


__all__ = ["Dropout"]
