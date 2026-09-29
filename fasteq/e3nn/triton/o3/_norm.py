"""Triton inference norm of each irrep; e3nn handles gradient paths."""

import torch
import triton
import triton.language as tl

from e3nn.o3._norm import Norm as _ReferenceNorm


@triton.jit
def _norm_kernel(X, OUT, INPUT_DIM: tl.constexpr, OUTPUT_DIM: tl.constexpr,
                 STARTS: tl.constexpr, MULTS: tl.constexpr, DIMS: tl.constexpr,
                 SQUARED: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0)
    row, channel = idx // OUTPUT_DIM, idx % OUTPUT_DIM
    k = tl.arange(0, BLOCK)
    value = tl.full((), 0, tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32)
    offset = 0
    for j in tl.static_range(len(DIMS)):
        if (channel >= offset) & (channel < offset + MULTS[j]):
            component = tl.load(X + row * INPUT_DIM + STARTS[j]
                                + (channel - offset) * DIMS[j] + k,
                                k < DIMS[j], other=0)
            value = tl.sum(component * component, 0)
        offset += MULTS[j]
    if not SQUARED:
        value = tl.sqrt(tl.maximum(value, 0.))
    tl.store(OUT + idx, value)


class Norm(_ReferenceNorm):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.requires_grad or not features.is_cuda or features.dtype not in (torch.float32, torch.float64):
            return super().forward(features)
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError("incompatible input dimension")
        starts, mults, dims = [], [], []
        offset = 0
        for mul, ir in self.irreps_in:
            starts.append(offset)
            mults.append(mul)
            dims.append(ir.dim)
            offset += mul * ir.dim
        out = features.new_empty((*features.shape[:-1], self.irreps_out.dim))
        if out.numel():
            _norm_kernel[(out.numel(),)](
                features.contiguous(), out, self.irreps_in.dim, self.irreps_out.dim,
                tuple(starts), tuple(mults), tuple(dims), self.squared,
                triton.next_power_of_2(max(dims)),
            )
        return out
