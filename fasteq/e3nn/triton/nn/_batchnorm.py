"""Inference normalization in one pass; original e3nn handles training."""

import torch
import triton
import triton.language as tl

from e3nn.nn._batchnorm import BatchNorm as _ReferenceBatchNorm
from .._common import grid


@triton.jit
def _batchnorm_eval_kernel(X, RM, RV, W, B, OUT, N: tl.constexpr,
                           WIDTH: tl.constexpr, STARTS: tl.constexpr,
                           MULTS: tl.constexpr, DIMS: tl.constexpr,
                           SCALAR_OFFS: tl.constexpr, WEIGHT_OFFS: tl.constexpr,
                           AFFINE: tl.constexpr, HAS_BIAS: tl.constexpr,
                           EPS: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = idx % WIDTH
    value = tl.load(X + idx, idx < N, other=0)
    result = value
    for j in tl.static_range(len(STARTS)):
        local = channel - STARTS[j]
        group = local // DIMS[j]
        valid = (idx < N) & (local >= 0) & (local < MULTS[j] * DIMS[j])
        var = tl.load(RV + WEIGHT_OFFS[j] + group, valid, other=1)
        if DIMS[j] == 1:
            mean = tl.load(RM + SCALAR_OFFS[j] + group, valid, other=0)
            v = value - mean
        else:
            v = value
        scale = tl.rsqrt(var + EPS)
        if AFFINE:
            scale *= tl.load(W + WEIGHT_OFFS[j] + group, valid, other=1)
        v *= scale
        if HAS_BIAS and DIMS[j] == 1:
            v += tl.load(B + SCALAR_OFFS[j] + group, valid, other=0)
        result = tl.where(valid, v, result)
    tl.store(OUT + idx, result, idx < N)


class BatchNorm(_ReferenceBatchNorm):
    def forward(self, input):
        if (self.training or self.instance or input.requires_grad or not input.is_cuda
                or (torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()))
                or input.dtype not in (torch.float32, torch.float64)
                or input.shape[-1] != self.irreps.dim):
            return super().forward(input)
        starts, muls, dims, scalar_offs, weight_offs = [], [], [], [], []
        c = s = w = 0
        for mul, ir in self.irreps:
            d = ir.dim
            starts.append(c)
            muls.append(mul)
            dims.append(d)
            scalar_offs.append(s)
            weight_offs.append(w)
            c += mul * d
            w += mul
            if ir.is_scalar():
                s += mul
        out = torch.empty_like(input, memory_format=torch.contiguous_format)
        if out.numel():
            weight = self.weight if self.affine else self.running_var
            bias = self.bias if self.affine and self.include_bias else self.running_mean
            _batchnorm_eval_kernel[grid(out.numel())](
                input.contiguous(), self.running_mean, self.running_var,
                weight, bias, out, out.numel(), c, tuple(starts), tuple(muls),
                tuple(dims), tuple(scalar_offs), tuple(weight_offs),
                self.affine, self.affine and self.include_bias, self.eps, 256,
            )
        return out


__all__ = ["BatchNorm"]
