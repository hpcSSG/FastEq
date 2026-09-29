"""Angle-based real spherical harmonics with precomputed Legendre terms."""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from e3nn.o3._angular_spherical_harmonics import (
    SphericalHarmonicsAlphaBeta as _SphericalHarmonicsAlphaBeta,
    _poly_legendre,
)
from .._common import check_gpu, grid


@triton.jit
def _alpha_kernel(A, OUT, N: tl.constexpr, L: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i // (2 * L + 1), i < N * (2 * L + 1), other=0)
    m = i % (2 * L + 1) - L
    value = tl.where(m < 0, 1.4142135623730951 * tl.sin(-m * a),
                     tl.where(m > 0, 1.4142135623730951 * tl.cos(m * a), 1.))
    tl.store(OUT + i, value, i < N * (2 * L + 1))


def spherical_harmonics_alpha(l: int, alpha: torch.Tensor) -> torch.Tensor:
    check_gpu(alpha)
    if l < 0:
        raise ValueError("l must be nonnegative")
    a = alpha.contiguous()
    out = alpha.new_empty((*alpha.shape, 2 * l + 1))
    _alpha_kernel[grid(out.numel())](a, out, a.numel(), l, 256)
    return out


@triton.jit
def _mul_kernel(XM, XLM, OUT, N: tl.constexpr, WM: tl.constexpr, WO: tl.constexpr,
                LS: tl.constexpr, MULS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = i // WO
    channel = i % WO
    xm_index = tl.full((BLOCK,), 0, tl.int32)
    offset = 0
    for j in tl.static_range(len(LS)):
        length = MULS[j] * (2 * LS[j] + 1)
        m = (channel - offset) % (2 * LS[j] + 1) - LS[j]
        xm_index = tl.where((channel >= offset) & (channel < offset + length), WM // 2 + m, xm_index)
        offset += length
    xm = tl.load(XM + b * WM + xm_index, i < N * WO, other=0)
    xlm = tl.load(XLM + i, i < N * WO, other=0)
    tl.store(OUT + i, xm * xlm, i < N * WO)


def _mul_m_lm(mul_l, x_m: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:
    check_gpu(x_m, x_lm)
    mul_l = tuple((int(mul), int(l)) for mul, l in mul_l)
    width = sum(mul * (2 * l + 1) for mul, l in mul_l)
    if x_lm.shape[-1] != width or x_m.shape[:-1] != x_lm.shape[:-1]:
        raise ValueError("incompatible spherical harmonic shapes")
    out = torch.empty_like(x_lm, memory_format=torch.contiguous_format)
    n = x_lm.numel() // width
    _mul_kernel[grid(out.numel())](x_m.contiguous(), x_lm.contiguous(), out, n,
        x_m.shape[-1], width, tuple(l for _, l in mul_l), tuple(m for m, _ in mul_l), 256)
    return out


@triton.jit
def _fused_kernel(A, B, C, ZP, YP, ORD, DEG, OUT, WIDTH: tl.constexpr,
                  TERMS: tl.constexpr, K: tl.constexpr, NORMALIZATION: tl.constexpr):
    idx = tl.program_id(0)
    row, col = idx // WIDTH, idx % WIDTH
    k = tl.arange(0, K)
    a = tl.load(A + row)
    b = tl.load(B + row)
    z, y = tl.cos(b), tl.sin(b)
    coef = tl.load(C + col * TERMS + k, k < TERMS, other=0)
    zp = tl.load(ZP + col * TERMS + k, k < TERMS, other=0)
    yp = tl.load(YP + col * TERMS + k, k < TERMS, other=0)
    p = tl.sum(coef * libdevice.pow(z, zp.to(z.dtype)) * libdevice.pow(y, yp.to(y.dtype)), 0)
    m = tl.load(ORD + col)
    l = tl.load(DEG + col)
    angular = tl.where(m < 0, 1.4142135623730951 * tl.sin(-m * a),
                       tl.where(m > 0, 1.4142135623730951 * tl.cos(m * a), 1.))
    value = p * angular
    if NORMALIZATION == 1:
        value = value * 3.544907701811032 / tl.sqrt(2. * l + 1.)
    elif NORMALIZATION == 2:
        value = value * 3.544907701811032
    tl.store(OUT + idx, value)


class SphericalHarmonicsAlphaBeta(_SphericalHarmonicsAlphaBeta):
    def __init__(self, l, normalization: str = "integral") -> None:
        super().__init__(l, normalization)
        if normalization not in ("integral", "norm", "component"):
            raise ValueError("unknown normalization")
        entries, orders, degrees = [], [], []
        for degree in self._ls_list:
            for m in range(-degree, degree + 1):
                entries.append(_poly_legendre(degree, abs(m)))
                orders.append(m)
                degrees.append(degree)
        terms = max(len(e) for e in entries)
        def padded(fn):
            return torch.tensor([[fn(k, v) for k, v in list(e.items())] + [0] * (terms - len(e)) for e in entries])
        self.register_buffer("_coef", padded(lambda k, v: float(v)))
        self.register_buffer("_zpower", padded(lambda k, v: k[0]).to(torch.int32))
        self.register_buffer("_ypower", padded(lambda k, v: k[1]).to(torch.int32))
        self.register_buffer("_order", torch.tensor(orders, dtype=torch.int32))
        self.register_buffer("_degree", torch.tensor(degrees, dtype=torch.int32))
        self._terms = terms
        self._gpu_coeff_cache = {}

    def _apply(self, fn):
        self._gpu_coeff_cache.clear()
        return super()._apply(fn)

    def forward(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        check_gpu(alpha, beta)
        if alpha.shape != beta.shape:
            raise ValueError("alpha and beta must have matching shapes")
        width = self._order.numel()
        out = alpha.new_empty((*alpha.shape, width))
        code = {"integral": 0, "norm": 1, "component": 2}[self.normalization]
        cache_key = (alpha.device, alpha.dtype)
        coeffs = self._gpu_coeff_cache.get(cache_key)
        if coeffs is None:
            coeffs = (self._coef.to(device=alpha.device, dtype=alpha.dtype).contiguous(),
                      self._zpower.to(alpha.device), self._ypower.to(alpha.device),
                      self._order.to(alpha.device), self._degree.to(alpha.device))
            self._gpu_coeff_cache[cache_key] = coeffs
        _fused_kernel[(out.numel(),)](alpha.contiguous(), beta.contiguous(),
            *coeffs, out,
            width, self._terms, triton.next_power_of_2(self._terms), code)
        return out
