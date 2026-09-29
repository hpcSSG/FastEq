"""Gather irreps into the requested output ordering."""

import torch
import triton
import triton.language as tl

from e3nn.nn._extract import Extract as _ReferenceExtract, ExtractIr as _ReferenceExtractIr
from .._common import grid


@triton.jit
def _extract_kernel(X, Y, N: tl.constexpr, DIN: tl.constexpr,
                    DOUT: tl.constexpr, STARTS: tl.constexpr,
                    STOPS: tl.constexpr, SOURCES: tl.constexpr,
                    BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    column = idx % DOUT
    src = tl.full((BLOCK,), 0, tl.int32)
    for j in tl.static_range(len(STARTS)):
        src = tl.where((column >= STARTS[j]) & (column < STOPS[j]),
                       SOURCES[j] + column - STARTS[j], src)
    x = tl.load(X + idx // DOUT * DIN + src, idx < N, other=0)
    tl.store(Y + idx, x, idx < N)


class Extract(_ReferenceExtract):
    def __init__(self, irreps_in, irreps_outs, instructions, squeeze_out: bool = False) -> None:
        super().__init__(irreps_in, irreps_outs, instructions, squeeze_out)
        self._squeeze_out = squeeze_out

    def forward(self, x: torch.Tensor):
        if (not x.is_cuda or x.requires_grad or x.dtype not in (torch.float32, torch.float64)
                or x.shape[-1] != self.irreps_in.dim):
            return super().forward(x)
        source = x.contiguous()
        outputs = []
        source_starts = tuple(s.start for s in self.irreps_in.slices())
        for irreps_out, instructions in zip(self.irreps_outs, self.instructions):
            dim = irreps_out.dim
            out = x.new_empty((*x.shape[:-1], dim))
            if out.numel():
                slices = irreps_out.slices()
                starts = tuple(s.start for s in slices)
                stops = tuple(s.stop for s in slices)
                sources = tuple(source_starts[i] for i in instructions)
                _extract_kernel[grid(out.numel())](
                    source, out, out.numel(), self.irreps_in.dim,
                    dim, starts, stops, sources, 256,
                )
            outputs.append(out)
        if len(outputs) == 1 and self._squeeze_out:
            return outputs[0]
        return tuple(outputs)


class ExtractIr(Extract):
    def __init__(self, irreps_in, ir) -> None:
        # The upstream constructor also computes the selected irreps and instruction list.
        from e3nn.o3._irreps import Irrep, Irreps

        ir = Irrep(ir)
        irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps([mul_ir for mul_ir in irreps_in if mul_ir.ir == ir])
        instructions = [tuple(i for i, mul_ir in enumerate(irreps_in) if mul_ir.ir == ir)]
        super().__init__(irreps_in, [self.irreps_out], instructions, squeeze_out=True)


__all__ = ["Extract", "ExtractIr"]
