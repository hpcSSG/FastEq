"""Fused scalar activation and second-moment scaling on CUDA."""

import torch
import triton
import triton.language as tl

from e3nn.nn._activation import Activation as _ReferenceActivation
from .._common import grid


def _activation_id(f):
    if f in (torch.abs,):
        return 1
    if f in (torch.tanh,):
        return 2
    if f in (torch.sigmoid,):
        return 3
    if f in (torch.relu, torch.nn.functional.relu):
        return 4
    if f in (torch.nn.functional.silu,):
        return 5
    if isinstance(f, torch.nn.Tanh):
        return 2
    if isinstance(f, torch.nn.Sigmoid):
        return 3
    if isinstance(f, torch.nn.ReLU):
        return 4
    if isinstance(f, torch.nn.SiLU):
        return 5
    return None


@triton.jit
def _activate_kernel(X, Y, N: tl.constexpr, WIDTH: tl.constexpr,
                     ENDS: tl.constexpr, ACTS: tl.constexpr,
                     SCALES: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = idx % WIDTH
    x = tl.load(X + idx, idx < N, other=0)
    value = x
    start = 0
    for j in tl.static_range(len(ENDS)):
        if ACTS[j] == 1:
            v = tl.abs(x)
        elif ACTS[j] == 2:
            v = 2. / (1. + tl.exp(-2. * x)) - 1.
        elif ACTS[j] == 3:
            v = 1. / (1. + tl.exp(-x))
        elif ACTS[j] == 4:
            v = tl.maximum(x, 0.)
        elif ACTS[j] == 5:
            v = x / (1. + tl.exp(-x))
        else:
            v = x
        value = tl.where((c >= start) & (c < ENDS[j]), v * SCALES[j], value)
        start = ENDS[j]
    tl.store(Y + idx, value, idx < N)


class Activation(_ReferenceActivation):
    def forward(self, features, dim: int = -1):
        if (not features.is_cuda or features.requires_grad
                or features.dtype not in (torch.float32, torch.float64)
                or (torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()))
                or dim % features.ndim != features.ndim - 1
                or not any(a is not None for a in self.acts)
                or features.shape[-1] != self.irreps_in.dim):
            return super().forward(features, dim)
        ends, acts, scales = [], [], []
        offset = 0
        for (mul, (l, _)), act in zip(self.irreps_in, self.acts):
            offset += mul * (2 * l + 1)
            ends.append(offset)
            if act is None:
                acts.append(0)
                scales.append(1.)
                continue
            kind = _activation_id(act.f)
            if kind is None:
                return super().forward(features, dim)
            acts.append(kind)
            scales.append(1. if act._is_id else act.cst)
        output = torch.empty_like(features, memory_format=torch.contiguous_format)
        if output.numel():
            _activate_kernel[grid(output.numel())](
                features.contiguous(), output, output.numel(), offset,
                tuple(ends), tuple(acts), tuple(scales), 256,
            )
        return output


__all__ = ["Activation"]
