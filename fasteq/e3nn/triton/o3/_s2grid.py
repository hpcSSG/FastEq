"""Single launch grid generation and lookup-matrix construction."""

import math

import torch
import triton
import triton.language as tl

from e3nn.o3._s2grid import ToS2Grid as _ToS2Grid, FromS2Grid as _FromS2Grid
from .._common import check_gpu, grid


@triton.jit
def _quadrature_kernel(W, B: tl.constexpr, K: tl.constexpr):
    j = tl.program_id(0)
    k = tl.arange(0, K)
    t = math.pi * (2. * j + 1.) / (4. * B)
    v = tl.where(k < B, tl.sin((2. * k + 1.) * t) / (2. * k + 1.), 0.)
    weight = (2. / B) * tl.sin(t) * tl.sum(v, 0) / (2. * (2 * B)**2)
    tl.store(W + j, weight)


def _quadrature_weights(b, dtype=None, device=None):
    w = torch.empty((2 * b,), dtype=dtype, device=device)
    check_gpu(w)
    _quadrature_kernel[(2 * b,)](w, b, triton.next_power_of_2(b))
    return w


@triton.jit
def _grid_kernel(BETA, ALPHA, NB: tl.constexpr, NA: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(BETA + i, (i + 0.5) / NB * math.pi, i < NB)
    tl.store(ALPHA + i, i / NA * (2. * math.pi), i < NA)


def s2_grid(res_beta, res_alpha, dtype=None, device=None):
    beta = torch.empty(res_beta, dtype=dtype, device=device)
    alpha = torch.empty(res_alpha, dtype=dtype, device=device)
    check_gpu(beta, alpha)
    _grid_kernel[grid(max(res_beta, res_alpha))](beta, alpha, res_beta, res_alpha, 256)
    return beta, alpha


@triton.jit
def _expand_kernel(M, LS: tl.constexpr, LMAX: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = i // ((2 * LMAX + 1) * DIM)
    m = (i // DIM) % (2 * LMAX + 1)
    k = i % DIM
    v = tl.full((BLOCK,), 0., tl.float32)
    offset = 0
    for li in tl.static_range(len(LS)):
        l = LS[li]
        v = tl.where((j == li) & (m >= LMAX - l) & (m <= LMAX + l) &
                     (k == offset + m - (LMAX - l)), 1., v)
        offset += 2 * l + 1
    tl.store(M + i, v, i < len(LS) * (2 * LMAX + 1) * DIM)


def _expand_matrix(ls, like=None, dtype=None, device=None):
    ls = tuple(ls)
    if like is not None:
        dtype = like.dtype if dtype is None else dtype
        device = like.device if device is None else device
    lmax, dim = max(ls), sum(2 * l + 1 for l in ls)
    m = torch.empty((len(ls), 2 * lmax + 1, dim), dtype=dtype, device=device)
    check_gpu(m)
    _expand_kernel[grid(m.numel())](m, ls, lmax, dim, 256)
    return m


@triton.jit
def _xyz_grid_kernel(BETA, ALPHA, OUT, NB: tl.constexpr, NA: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(BETA + i // NA, i < NB * NA, other=0)
    a = tl.load(ALPHA + i % NA, i < NB * NA, other=0)
    tl.store(OUT + 3 * i, tl.sin(b) * tl.sin(a), i < NB * NA)
    tl.store(OUT + 3 * i + 1, tl.cos(b), i < NB * NA)
    tl.store(OUT + 3 * i + 2, tl.sin(b) * tl.cos(a), i < NB * NA)


def _grid_property(self):
    check_gpu(self.betas, self.alphas)
    out = self.betas.new_empty((self.res_beta, self.res_alpha, 3))
    _xyz_grid_kernel[grid(self.res_beta * self.res_alpha)](self.betas.contiguous(),
        self.alphas.contiguous(), out, self.res_beta, self.res_alpha, 256)
    return out


class ToS2Grid(_ToS2Grid):
    @property
    def grid(self) -> torch.Tensor:
        return _grid_property(self)


class FromS2Grid(_FromS2Grid):
    @property
    def grid(self) -> torch.Tensor:
        return _grid_property(self)
