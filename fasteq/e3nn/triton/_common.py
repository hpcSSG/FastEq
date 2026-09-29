"""Shared launch and input validation helpers."""

import torch
import triton


def check_gpu(*tensors):
    for x in tensors:
        if not x.is_cuda:
            raise ValueError("Triton implementation requires GPU tensors")
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("Triton implementation supports float32 and float64")


def grid(n, block=256):
    return (triton.cdiv(n, block),)
