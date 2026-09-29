"""SO(3) grid contractions; one launch per direction."""

import math

import torch
import triton
import triton.language as tl

from e3nn.o3._so3grid import SO3Grid as _SO3Grid
from e3nn.o3._so3grid import flat_wigner
from .._common import check_gpu


@triton.jit
def _to_kernel(X, D, Y, DIM: tl.constexpr, POINTS: tl.constexpr, COUNT: tl.constexpr,
               SCALE: tl.constexpr, K: tl.constexpr):
    p = tl.program_id(0)
    k = tl.arange(0, K)
    x = tl.load(X + (p // POINTS) * DIM + k, k < DIM, other=0)
    d = tl.load(D + (p % POINTS) * DIM + k, k < DIM, other=0)
    tl.store(Y + p, tl.sum(x * d, 0) / SCALE, p < COUNT * POINTS)


@triton.jit
def _from_kernel(X, D, QW, Y, DIM: tl.constexpr, POINTS: tl.constexpr,
                 NB: tl.constexpr, NA: tl.constexpr, COUNT: tl.constexpr,
                 SCALE: tl.constexpr, K: tl.constexpr):
    p = tl.program_id(0)
    k = tl.arange(0, K)
    b = (k // NA) % NB
    x = tl.load(X + (p // DIM) * POINTS + k, k < POINTS, other=0)
    d = tl.load(D + k * DIM + p % DIM, k < POINTS, other=0)
    w = tl.load(QW + b, k < POINTS, other=0)
    tl.store(Y + p, tl.sum(x * d * w, 0) * SCALE, p < COUNT * DIM)


class SO3Grid(_SO3Grid):
    def to_grid(self, features) -> torch.Tensor:
        check_gpu(features, self.D)
        dim, points = self.D.shape[-1], self.D.numel() // self.D.shape[-1]
        if features.shape[-1] != dim:
            raise ValueError("incompatible feature dimension")
        batch = features.numel() // dim
        out = features.new_empty((*features.shape[:-1], self.res_alpha, self.res_beta, self.res_gamma))
        _to_kernel[(batch * points,)](features.contiguous(), self.D.contiguous(), out,
                                      dim, points, batch, math.sqrt(dim), triton.next_power_of_2(dim))
        return out

    def from_grid(self, features) -> torch.Tensor:
        check_gpu(features, self.D, self.qw)
        dim, points = self.D.shape[-1], self.D.numel() // self.D.shape[-1]
        if tuple(features.shape[-3:]) != (self.res_alpha, self.res_beta, self.res_gamma):
            raise ValueError("incompatible grid dimensions")
        batch = features.numel() // points
        out = features.new_empty((*features.shape[:-3], dim))
        _from_kernel[(batch * dim,)](features.contiguous(), self.D.contiguous(), self.qw.contiguous(), out,
                                     dim, points, self.res_beta, self.res_alpha, batch, math.sqrt(dim),
                                     triton.next_power_of_2(points))
        return out
