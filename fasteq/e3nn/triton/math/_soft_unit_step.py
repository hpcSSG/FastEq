"""Smooth step, including its custom backward."""

import torch
import triton
import triton.language as tl

from .._common import check_gpu, grid


@triton.jit
def _step_kernel(X, DY, Y, N: tl.constexpr, BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    safe = tl.where(x > 0, x, 1.0)
    value = tl.where(x > 0, tl.exp(-1.0 / safe), 0.0)
    if BACKWARD:
        dy = tl.load(DY + i, i < N, other=0)
        value = value / (safe * safe) * dy
    tl.store(Y + i, value, i < N)


class _SoftUnitStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        check_gpu(x)
        ctx.save_for_backward(x)
        y = torch.empty_like(x, memory_format=torch.contiguous_format)
        _step_kernel[grid(x.numel())](x.contiguous(), x, y, x.numel(), False, 256)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        y = torch.empty_like(x, memory_format=torch.contiguous_format)
        _step_kernel[grid(x.numel())](x.contiguous(), dy.contiguous(), y, x.numel(), True, 256)
        return y


def soft_unit_step(x):
    return _SoftUnitStep.apply(x)
