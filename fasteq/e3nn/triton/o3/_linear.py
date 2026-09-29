"""Triton for simple no-grad Linear paths, original e3nn for general forms."""

import torch
import triton
import triton.language as tl

from e3nn.o3._linear import Instruction, LinearSlices, Linear as _ReferenceLinear, _codegen_linear


@triton.jit
def _single_linear_kernel(X, W, Y, DIN: tl.constexpr, DOUT: tl.constexpr,
                          MUL_IN: tl.constexpr, MUL_OUT: tl.constexpr,
                          IR_DIM: tl.constexpr, SCALE: tl.constexpr,
                          K: tl.constexpr):
    idx = tl.program_id(0)
    row, out_channel = idx // DOUT, idx % DOUT
    out_mul, component = out_channel // IR_DIM, out_channel % IR_DIM
    u = tl.arange(0, K)
    x = tl.load(X + row * DIN + u * IR_DIM + component, u < MUL_IN, other=0)
    w = tl.load(W + u * MUL_OUT + out_mul, u < MUL_IN, other=0)
    tl.store(Y + idx, tl.sum(x * w, 0) * SCALE)


class Linear(_ReferenceLinear):
    def forward(self, features, weight=None, bias=None):
        given_weight = weight
        candidate_weight = self.weight if weight is None else weight
        simple = (
            features.is_cuda and features.dtype in (torch.float32, torch.float64)
            and len(self.irreps_in) == len(self.irreps_out) == 1
            and len(self.instructions) == 1 and self.instructions[0].i_in == 0
            and self.instructions[0].i_out == 0 and self.bias_numel == 0
            and self.shared_weights and candidate_weight.numel() == self.weight_numel
            and self.weight_numel > 0 and features.shape[-1] == self.irreps_in.dim
            and candidate_weight.is_cuda and candidate_weight.dtype == features.dtype
            and not (torch.is_grad_enabled() and (features.requires_grad or candidate_weight.requires_grad))
        )
        if not simple:
            return super().forward(features, given_weight, bias)
        mul_in, ir_in = self.irreps_in[0]
        mul_out, ir_out = self.irreps_out[0]
        if ir_in != ir_out or self.weight_numel != mul_in * mul_out:
            return super().forward(features, given_weight, bias)
        out = features.new_empty((*features.shape[:-1], self.irreps_out.dim))
        if out.numel():
            _single_linear_kernel[(out.numel(),)](
                features.contiguous(), candidate_weight.contiguous(), out,
                self.irreps_in.dim, self.irreps_out.dim, mul_in, mul_out,
                ir_in.dim, self.instructions[0].path_weight,
                triton.next_power_of_2(mul_in),
            )
        return out


__all__ = ["Instruction", "LinearSlices", "Linear", "_codegen_linear"]
