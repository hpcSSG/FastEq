"""Compose Triton extraction, scalar activations and gated multiplication."""

import torch
import triton
import triton.language as tl

from e3nn.nn._gate import Gate as _ReferenceGate, _Sortcut as _ReferenceSortcut
from ._activation import Activation
from ._extract import Extract
from .._common import grid


@triton.jit
def _gate_mul_kernel(SCALARS, GATES, GATED, OUT, N: tl.constexpr,
                     SCALAR_DIM: tl.constexpr, GATED_DIM: tl.constexpr,
                     GATE_DIM: tl.constexpr,
                     STARTS: tl.constexpr, MULTS: tl.constexpr,
                     DIMS: tl.constexpr, IR_OFFS: tl.constexpr,
                     BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width = SCALAR_DIM + GATED_DIM
    col = idx % width
    row = idx // width
    gcol = col - SCALAR_DIM
    scale_idx = tl.full((BLOCK,), 0, tl.int32)
    for j in tl.static_range(len(STARTS)):
        valid = (gcol >= STARTS[j]) & (gcol < STARTS[j] + MULTS[j] * DIMS[j])
        scale_idx = tl.where(valid, IR_OFFS[j] + (gcol - STARTS[j]) // DIMS[j], scale_idx)
    scalar = tl.load(SCALARS + row * SCALAR_DIM + col,
                     (idx < N) & (col < SCALAR_DIM), other=0)
    value = tl.load(GATED + row * GATED_DIM + gcol,
                    (idx < N) & (col >= SCALAR_DIM), other=0)
    gate = tl.load(GATES + row * GATE_DIM + scale_idx,
                   (idx < N) & (col >= SCALAR_DIM), other=0)
    tl.store(OUT + idx, tl.where(col < SCALAR_DIM, scalar, value * gate), idx < N)


class _Sortcut(_ReferenceSortcut):
    def __init__(self, *irreps_outs) -> None:
        super().__init__(*irreps_outs)
        old = self.cut
        self.cut = Extract(old.irreps_in, old.irreps_outs, old.instructions)


class Gate(_ReferenceGate):
    def __init__(self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated) -> None:
        super().__init__(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)
        self.sc = _Sortcut(irreps_scalars, irreps_gates, irreps_gated)
        self.act_scalars = Activation(irreps_scalars, act_scalars)
        self.act_gates = Activation(irreps_gates, act_gates)

    def forward(self, features):
        if (not features.is_cuda or features.requires_grad
                or (torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()))
                or features.dtype not in (torch.float32, torch.float64)
                or not self.irreps_gates.num_irreps):
            return super().forward(features)
        scalars, gates, gated = self.sc(features)
        scalars = self.act_scalars(scalars)
        gates = self.act_gates(gates)
        starts, mults, dims, ir_offs = [], [], [], []
        c = ir = 0
        for mul, rep in self.irreps_gated:
            starts.append(c)
            mults.append(mul)
            dims.append(rep.dim)
            ir_offs.append(ir)
            c += mul * rep.dim
            ir += mul
        out = features.new_empty((*features.shape[:-1], scalars.shape[-1] + c))
        if out.numel():
            _gate_mul_kernel[grid(out.numel())](
                scalars.contiguous(), gates.contiguous(), gated.contiguous(), out,
                out.numel(), scalars.shape[-1], c, ir, tuple(starts), tuple(mults),
                tuple(dims), tuple(ir_offs), 256,
            )
        return out


__all__ = ["_Sortcut", "Gate"]
