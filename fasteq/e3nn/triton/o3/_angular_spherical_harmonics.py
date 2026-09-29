"""Angle-based real spherical harmonics with precomputed Legendre terms."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from sympy import Integer, Poly, diff, factorial, pi, sqrt, symbols

from e3nn import o3, get_optimization_defaults
from .._common import check_gpu, grid


def _conditional_script(fn):
    return torch.jit.script(fn) if get_optimization_defaults()["jit_mode"] == "script" else fn


def _sympy_legendre(l: int, m: int):
    """The polynomial used by e3nn, with P(l, -m) = P(l, m)."""
    l, m = Integer(l), Integer(abs(m))
    z, y = symbols("z y", real=True)
    expression = y**m * diff((z**2 - 1) ** l, z, l + m) / (2**l * factorial(l))
    return expression * sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))


def _poly_legendre(l: int, m: int):
    z, y = symbols("z y", real=True)
    return Poly(_sympy_legendre(l, m), domain="R", gens=(z, y)).as_dict()


def _coefficients(ls):
    entries, orders, degrees = [], [], []
    for degree in ls:
        for m in range(-degree, degree + 1):
            entries.append(_poly_legendre(degree, abs(m)))
            orders.append(m)
            degrees.append(degree)
    terms = max(map(len, entries))
    def padded(fn):
        return torch.tensor([
            [fn(k, v) for k, v in e.items()] + [0] * (terms - len(e))
            for e in entries
        ])
    return (padded(lambda k, v: float(v)).to(torch.float64),
            padded(lambda k, v: k[0]).to(torch.int32),
            padded(lambda k, v: k[1]).to(torch.int32),
            torch.tensor(orders, dtype=torch.int32),
            torch.tensor(degrees, dtype=torch.int32), terms)


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
    coef = tl.load(C + col * TERMS + k, k < TERMS, other=0).to(b.dtype)
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


@triton.jit
def _legendre_kernel(Z, Y, C, ZP, YP, OUT, N: tl.constexpr,
                     WIDTH: tl.constexpr, TERMS: tl.constexpr, K: tl.constexpr):
    idx = tl.program_id(0)
    row, col = idx // WIDTH, idx % WIDTH
    k = tl.arange(0, K)
    z = tl.load(Z + row)
    y = tl.load(Y + row)
    coef = tl.load(C + col * TERMS + k, k < TERMS, other=0).to(z.dtype)
    zp = tl.load(ZP + col * TERMS + k, k < TERMS, other=0)
    yp = tl.load(YP + col * TERMS + k, k < TERMS, other=0)
    value = tl.sum(coef * libdevice.pow(z, zp.to(z.dtype))
                   * libdevice.pow(y, yp.to(y.dtype)), 0)
    tl.store(OUT + idx, value)


class Legendre(torch.nn.Module):
    """Associated Legendre polynomials, with symbolic work done at initialization."""

    def __init__(self, ls) -> None:
        super().__init__()
        ls = tuple(int(degree) for degree in ls)
        if not ls or any(degree < 0 for degree in ls):
            raise ValueError("ls must contain nonnegative degrees")
        coef, zp, yp, _, _, self._terms = _coefficients(ls)
        self.register_buffer("_coef", coef)
        self.register_buffer("_zpower", zp)
        self.register_buffer("_ypower", yp)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        check_gpu(z, y)
        if z.shape != y.shape or z.dtype != y.dtype or z.device != y.device:
            raise ValueError("z and y must have matching shape, dtype, and device")
        if z.dtype not in (torch.float32, torch.float64):
            raise TypeError("Legendre supports float32 and float64")
        width = self._coef.shape[0]
        out = z.new_empty((*z.shape, width))
        if out.numel():
            _legendre_kernel[(out.numel(),)](
                z.contiguous(), y.contiguous(),
                self._coef.to(device=z.device), self._zpower.to(device=z.device),
                self._ypower.to(device=z.device), out,
                z.numel(), width, self._terms, triton.next_power_of_2(self._terms),
            )
        return out


class SphericalHarmonicsAlphaBeta(torch.nn.Module):
    """Fused angle based spherical harmonics, API compatible with e3nn."""

    def __init__(self, l, normalization: str = "integral") -> None:
        super().__init__()
        if normalization not in ("integral", "norm", "component"):
            raise ValueError("unknown normalization")
        if isinstance(l, o3.Irreps):
            ls = [degree for mul, (degree, _) in l for _ in range(mul)]
        elif isinstance(l, int):
            ls = [l]
        else:
            ls = list(l)
        if not ls or any(not isinstance(degree, int) or degree < 0 for degree in ls):
            raise ValueError("l must contain nonnegative integer degrees")
        self._ls_list = ls
        self._lmax = max(ls)
        self.normalization = normalization
        coef, zp, yp, order, degree, self._terms = _coefficients(ls)
        self.register_buffer("_coef", coef)
        self.register_buffer("_zpower", zp)
        self.register_buffer("_ypower", yp)
        self.register_buffer("_order", order)
        self.register_buffer("_degree", degree)

    def forward(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        check_gpu(alpha, beta)
        if alpha.shape != beta.shape or alpha.dtype != beta.dtype or alpha.device != beta.device:
            raise ValueError("alpha and beta must have matching shape, dtype, and device")
        if alpha.dtype not in (torch.float32, torch.float64):
            raise TypeError("SphericalHarmonicsAlphaBeta supports float32 and float64")
        width = self._order.numel()
        out = alpha.new_empty((*alpha.shape, width))
        code = {"integral": 0, "norm": 1, "component": 2}[self.normalization]
        if out.numel():
            coeffs = (self._coef.to(alpha.device), self._zpower.to(alpha.device),
                      self._ypower.to(alpha.device), self._order.to(alpha.device),
                      self._degree.to(alpha.device))
            _fused_kernel[(out.numel(),)](alpha.contiguous(), beta.contiguous(),
                *coeffs, out, width, self._terms, triton.next_power_of_2(self._terms), code)
        return out


def spherical_harmonics_alpha_beta(l, alpha: torch.Tensor, beta: torch.Tensor,
                                   *, normalization: str = "integral") -> torch.Tensor:
    return SphericalHarmonicsAlphaBeta(l, normalization)(alpha, beta)
