"""S2 grid construction and forward transforms using Triton kernels."""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ._angular_spherical_harmonics import Legendre, spherical_harmonics_alpha
from ._rotation import angles_to_xyz
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


def spherical_harmonics_s2_grid(lmax, res_beta, res_alpha, dtype=None, device=None):
    """Evaluate the angular and Legendre factors on the S2 grid."""
    betas, alphas = s2_grid(res_beta, res_alpha, dtype=dtype, device=device)
    # For the standard grid, sin(beta) is nonnegative; abs matches e3nn.
    shb = Legendre(range(lmax + 1)).to(betas.device)(betas.cos(), betas.sin().abs())
    sha = spherical_harmonics_alpha(lmax, alphas)
    return betas, alphas, shb, sha


def _complete_lmax_res(lmax, res_beta, res_alpha):
    if res_beta is None:
        if lmax is not None:
            res_beta = 2 * (lmax + 1)
        elif res_alpha is not None:
            res_beta = 2 * ((res_alpha + 1) // 2)
        else:
            raise ValueError("All the entries are None")
    if res_alpha is None:
        if lmax is not None:
            res_alpha = max(2 * lmax + 1, res_beta - 1)
        else:
            res_alpha = res_beta - 1
    if lmax is None:
        lmax = min(res_beta // 2 - 1, (res_alpha - 1) // 2)
    if res_beta % 2 or lmax + 1 > res_beta // 2:
        raise ValueError("res_beta must be even and support lmax")
    return lmax, res_beta, res_alpha


@triton.jit
def _rfft_dft_kernel(X, OUT, RES: tl.constexpr, L: tl.constexpr,
                     K: tl.constexpr):
    idx = tl.program_id(0)
    width = 2 * L + 1
    row, m = idx // width, idx % width - L
    t = tl.arange(0, K)
    x = tl.load(X + row * RES + t, t < RES, other=0)
    angle = t.to(x.dtype) * (6.283185307179586 * tl.abs(m) / RES)
    if m < 0:
        v = 1.4142135623730951 * tl.sum(x * libdevice.sin(angle), 0)
    elif m > 0:
        v = 1.4142135623730951 * tl.sum(x * libdevice.cos(angle), 0)
    else:
        v = tl.sum(x, 0)
    tl.store(OUT + idx, v)


def rfft(x: torch.Tensor, l: int) -> torch.Tensor:
    """Real Fourier coefficients in e3nn's [-l, ..., l] layout."""
    check_gpu(x)
    res = x.shape[-1]
    if l < 0 or l > res // 2:
        raise ValueError("l must be in [0, res // 2]")
    out = x.new_empty((*x.shape[:-1], 2 * l + 1))
    if out.numel():
        _rfft_dft_kernel[(out.numel(),)](
            x.contiguous(), out, res, l, triton.next_power_of_2(res)
        )
    return out


@triton.jit
def _irfft_dft_kernel(X, OUT, RES: tl.constexpr, L: tl.constexpr,
                      K: tl.constexpr):
    idx = tl.program_id(0)
    row, a = idx // RES, idx % RES
    k = tl.arange(0, K) + 1
    mask = k <= L
    xp = tl.load(X + row * (2 * L + 1) + L + k, mask, other=0)
    xn = tl.load(X + row * (2 * L + 1) + L - k, mask, other=0)
    angle = k.to(xp.dtype) * (6.283185307179586 * a / RES)
    value = tl.load(X + row * (2 * L + 1) + L)
    value += 1.4142135623730951 * tl.sum(
        xp * libdevice.cos(angle) + xn * libdevice.sin(angle), 0
    )
    tl.store(OUT + idx, value)


def irfft(x: torch.Tensor, res: int) -> torch.Tensor:
    """Inverse of rfft, scaled like e3nn's unnormalized torch.fft.irfft."""
    check_gpu(x)
    sm = x.shape[-1]
    if res % 2 != 1 or sm % 2 != 1 or res < sm:
        raise ValueError("res and coefficient width must be odd, with res >= width")
    l = sm // 2
    out = x.new_empty((*x.shape[:-1], res))
    if out.numel():
        _irfft_dft_kernel[(out.numel(),)](
            x.contiguous(), out, res, l, triton.next_power_of_2(max(l, 1))
        )
    return out


@triton.jit
def _prepare_shb_kernel(LEG, DEG, ORDER, SCALE, QW, OUT,
                        NB: tl.constexpr, DIM: tl.constexpr, M: tl.constexpr,
                        USE_QW: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < M * NB * DIM
    i = idx % DIM
    b = (idx // DIM) % NB
    m = idx // (NB * DIM)
    degree = tl.load(DEG + i, valid, other=0)
    order = tl.load(ORDER + i, valid, other=0)
    v = tl.load(LEG + b * DIM + i, valid, other=0)
    v *= tl.load(SCALE + degree, valid, other=0)
    if USE_QW:
        v *= tl.load(QW + b, valid, other=0)
    tl.store(OUT + idx, tl.where(m == order + M // 2, v, 0.), valid)


def _basis_metadata(lmax, device):
    degrees = [l for l in range(lmax + 1) for _ in range(2 * l + 1)]
    orders = [m for l in range(lmax + 1) for m in range(-l, l + 1)]
    return (torch.tensor(degrees, dtype=torch.int32, device=device),
            torch.tensor(orders, dtype=torch.int32, device=device))


def _prepare_shb(leg, degrees, orders, scale, qw=None):
    nb, dim = leg.shape
    m = 2 * (int(scale.numel()) - 1) + 1
    out = leg.new_empty((m, nb, dim))
    if out.numel():
        _prepare_shb_kernel[grid(out.numel())](
            leg.contiguous(), degrees, orders, scale.contiguous(),
            qw if qw is not None else scale, out,
            nb, dim, m, qw is not None, 256,
        )
    return out


@triton.jit
def _to_beta_kernel(X, SHB, OUT, NB: tl.constexpr, M: tl.constexpr,
                    DIM: tl.constexpr, K: tl.constexpr):
    idx = tl.program_id(0)
    row, b, m = idx // (NB * M), (idx // M) % NB, idx % M
    i = tl.arange(0, K)
    x = tl.load(X + row * DIM + i, i < DIM, other=0)
    shb = tl.load(SHB + (m * NB + b) * DIM + i, i < DIM, other=0)
    tl.store(OUT + idx, tl.sum(x * shb, 0))


@triton.jit
def _from_beta_kernel(X, SHB, ORDER, OUT, NB: tl.constexpr,
                      M: tl.constexpr, DIM: tl.constexpr, K: tl.constexpr):
    idx = tl.program_id(0)
    row, i = idx // DIM, idx % DIM
    b = tl.arange(0, K)
    m = tl.load(ORDER + i) + M // 2
    x = tl.load(X + (row * NB + b) * M + m, b < NB, other=0)
    shb = tl.load(SHB + (m * NB + b) * DIM + i, b < NB, other=0)
    tl.store(OUT + idx, tl.sum(x * shb, 0))


@triton.jit
def _alpha_project_kernel(X, SHA, OUT, NB: tl.constexpr,
                          NA: tl.constexpr, M: tl.constexpr, K: tl.constexpr):
    idx = tl.program_id(0)
    row, b, m = idx // (NB * M), (idx // M) % NB, idx % M
    a = tl.arange(0, K)
    x = tl.load(X + (row * NB + b) * NA + a, a < NA, other=0)
    sha = tl.load(SHA + a * M + m, a < NA, other=0)
    tl.store(OUT + idx, tl.sum(x * sha, 0))


@triton.jit
def _alpha_synth_kernel(X, SHA, OUT, NB: tl.constexpr,
                        NA: tl.constexpr, M: tl.constexpr, K: tl.constexpr):
    idx = tl.program_id(0)
    row, b, a = idx // (NB * NA), (idx // NA) % NB, idx % NA
    m = tl.arange(0, K)
    x = tl.load(X + (row * NB + b) * M + m, m < M, other=0)
    sha = tl.load(SHA + a * M + m, m < M, other=0)
    tl.store(OUT + idx, tl.sum(x * sha, 0))


class _S2GridBase(torch.nn.Module):
    def _initialize(self, lmax, res, normalization, lmax_in, inverse, dtype, device):
        if not (normalization in ("norm", "component", "integral")
                or torch.is_tensor(normalization)):
            raise ValueError("unknown normalization")
        if isinstance(res, int) or res is None:
            lmax, nb, na = _complete_lmax_res(lmax, res, None)
        else:
            lmax, nb, na = _complete_lmax_res(lmax, *res)
        if lmax_in is None:
            lmax_in = lmax
        betas, alphas, leg, sha = spherical_harmonics_s2_grid(
            lmax, nb, na, dtype=dtype, device=device
        )
        degrees, orders = _basis_metadata(lmax, betas.device)
        if torch.is_tensor(normalization):
            scale = normalization.to(dtype=betas.dtype, device=betas.device)
        elif inverse:
            factor = math.sqrt(4 * math.pi)
            if normalization == "component":
                scale = betas.new_tensor([factor * math.sqrt(2 * l + 1) * math.sqrt(lmax_in + 1)
                                           for l in range(lmax + 1)])
            elif normalization == "norm":
                scale = betas.new_full((lmax + 1,), factor * math.sqrt(lmax_in + 1))
            else:
                scale = betas.new_full((lmax + 1,), 4 * math.pi)
        else:
            factor = math.sqrt(4 * math.pi) / math.sqrt(lmax + 1)
            if normalization == "component":
                scale = betas.new_tensor([factor / math.sqrt(2 * l + 1)
                                           for l in range(lmax + 1)])
            elif normalization == "norm":
                scale = betas.new_full((lmax + 1,), factor)
            else:
                scale = betas.new_ones(lmax + 1)
        qw = None
        if inverse:
            qw = _quadrature_weights(nb // 2, dtype=betas.dtype, device=betas.device)
            qw = qw * (nb**2 / na)
        shb = _prepare_shb(leg, degrees, orders, scale, qw)
        self.lmax, self.res_beta, self.res_alpha = lmax, nb, na
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)
        self.register_buffer("sha", sha)
        self.register_buffer("shb", shb)
        self.register_buffer("_order", orders, persistent=False)

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax} res={self.res_beta}x{self.res_alpha} (beta x alpha))"

    @property
    def grid(self):
        return angles_to_xyz(self.alphas[None, :], self.betas[:, None])


class ToS2Grid(_S2GridBase):
    def __init__(self, lmax=None, res=None, normalization="component", dtype=None, device=None):
        super().__init__()
        self._initialize(lmax, res, normalization, None, False, dtype, device)

    @property
    def grid(self):
        return super().grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_gpu(x)
        dim = (self.lmax + 1)**2
        if x.shape[-1] != dim:
            raise ValueError(f"expected last dimension {dim}")
        shape = x.shape[:-1]
        batch = x.numel() // dim
        nb, na, m = self.res_beta, self.res_alpha, 2 * self.lmax + 1
        tmp = x.new_empty((batch, nb, m))
        if tmp.numel():
            _to_beta_kernel[(tmp.numel(),)](x.contiguous(), self.shb, tmp,
                nb, m, dim, triton.next_power_of_2(dim))
        if na >= m and na % 2:
            return irfft(tmp, na).reshape(*shape, nb, na)
        out = x.new_empty((batch, nb, na))
        if out.numel():
            _alpha_synth_kernel[(out.numel(),)](tmp, self.sha, out,
                nb, na, m, triton.next_power_of_2(m))
        return out.reshape(*shape, nb, na)

    def _make_tracing_inputs(self, n: int):
        return [{"forward": (torch.randn((self.lmax + 1)**2),)} for _ in range(n)]


class FromS2Grid(_S2GridBase):
    def __init__(self, res=None, lmax=None, normalization="component",
                 lmax_in=None, dtype=None, device=None):
        super().__init__()
        self._initialize(lmax, res, normalization, lmax_in, True, dtype, device)

    @property
    def grid(self):
        return super().grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_gpu(x)
        nb, na, m = self.res_beta, self.res_alpha, 2 * self.lmax + 1
        if x.shape[-2:] != (nb, na):
            raise ValueError(f"expected last dimensions {(nb, na)}")
        shape = x.shape[:-2]
        batch = x.numel() // (nb * na)
        if na >= m and na % 2:
            tmp = rfft(x.reshape(batch, nb, na), self.lmax)
        else:
            tmp = x.new_empty((batch, nb, m))
            if tmp.numel():
                _alpha_project_kernel[(tmp.numel(),)](x.contiguous(), self.sha, tmp,
                    nb, na, m, triton.next_power_of_2(na))
        dim = (self.lmax + 1)**2
        out = x.new_empty((batch, dim))
        if out.numel():
            _from_beta_kernel[(out.numel(),)](tmp, self.shb, self._order, out,
                nb, m, dim, triton.next_power_of_2(nb))
        return out.reshape(*shape, dim)

    def _make_tracing_inputs(self, n: int):
        return [{"forward": (torch.randn(self.res_beta, self.res_alpha),)} for _ in range(n)]
