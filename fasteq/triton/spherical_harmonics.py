"""FastEq spherical harmonics: fused Triton forward/backward/double backward.

API (same parameters as the supplied e3nn implementation):
    SphericalHarmonics(irreps_out, normalize, normalization="integral", irreps_in=None)
    spherical_harmonics(l, x, normalize, normalization="integral")

Usage:
    from fasteq_spherical_harmonics import SphericalHarmonics, spherical_harmonics
    sh = SphericalHarmonics([0, 1, 2, 3], True, "component").cuda()
    y = sh(x)  # x: (..., 3)
    gx = torch.autograd.grad(y, x, grad_out, create_graph=True)[0]
    dx, dgrad_out = torch.autograd.grad(gx, (x, grad_out), grad_grad_x)

    python fasteq_spherical_harmonics.py --test
    python fasteq_spherical_harmonics.py --test --full  # includes l=0..12
    python fasteq_spherical_harmonics.py --dump-kernels ./sh_kernels --lmax 3

Implementation:
  * The polynomial coefficients, ordering and recurrences are embedded from
    the supplied source, independent of the installed e3nn SH implementation.
  * Code generation analytically differentiates the recurrence DAG. Each GPU
    lane owns one input vector; contractions are local, without atomics or a
    global Jacobian/Hessian buffer. Three separate kernels are generated.
  * Double backward returns both sum_c grad_out[c] * Hessian(Y_c) @ v and J @ v.
    ctx.needs_input_grad prunes either unused double-backward output.
  * Input normalization and its first/second derivatives are fused on GPU.
    r < 1e-12 uses x/eps, including its finite derivatives at x=0.
    At r==eps use the active clamp branch, like clamp_min backward; a classical
    Hessian does not exist exactly at that boundary. PyTorch norm double
    backward can yield NaN at the origin; the GPU path uses the finite
    derivative of the locally linear x/eps map instead.
  * FP32 and FP64 compute in their native precision; FP16/BF16 compute in FP32
    and cast outputs back (not bitwise equivalent to low-precision e3nn).
    enable_fp_fusion=False limits FMA-related differences, not all roundoff.
  * CPU and TorchScript use the embedded PyTorch reference; CUDA/ROCm eager
    uses Triton. Third-order AD on the Triton path is intentionally unsupported.
    torch.compile fullgraph / torch.func transforms are not promised.
  * High l may have large compilation/register cost. No GPU speedup claim.

Requirements: torch, e3nn; triton for GPU eager execution.
Validation on the delivery host: generated arithmetic checked with NumPy,
including 36 configurations and finite-difference derivatives (maximum
normalized error 9.2e-8); GPU execution not available there.
Official Triton JIT reference: https://triton-lang.org/main/python-api/generated/triton.jit.html
"""
import ast
import functools
import hashlib
import linecache
import math
from pathlib import Path
from typing import Union, List, Any

import torch
from torch.autograd.function import once_differentiable
from e3nn.o3._irreps import Irreps
from e3nn.util.jit import compile_mode

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

_POLYNOMIALS = (
    ('sh_0_0', 'torch.ones_like(x)'),
    ('sh_1_0', 'math.sqrt(3) * x'),
    ('sh_1_1', 'math.sqrt(3) * y'),
    ('sh_1_2', 'math.sqrt(3) * z'),
    ('sh_2_0', 'math.sqrt(15) * x * z'),
    ('sh_2_1', 'math.sqrt(15) * x * y'),
    ('y2', 'y.pow(2)'),
    ('x2z2', 'x.pow(2) + z.pow(2)'),
    ('sh_2_2', 'math.sqrt(5) * (y2 - 1 / 2 * x2z2)'),
    ('sh_2_3', 'math.sqrt(15) * y * z'),
    ('sh_2_4', '1 / 2 * math.sqrt(15) * (z.pow(2) - x.pow(2))'),
    ('sh_3_0', '1 / 6 * math.sqrt(42) * (sh_2_0 * z + sh_2_4 * x)'),
    ('sh_3_1', 'math.sqrt(7) * sh_2_0 * y'),
    ('sh_3_2', '1 / 8 * math.sqrt(168) * (4.0 * y2 - x2z2) * x'),
    ('sh_3_3', '1 / 2 * math.sqrt(7) * y * (2.0 * y2 - 3.0 * x2z2)'),
    ('sh_3_4', '1 / 8 * math.sqrt(168) * z * (4.0 * y2 - x2z2)'),
    ('sh_3_5', 'math.sqrt(7) * sh_2_4 * y'),
    ('sh_3_6', '1 / 6 * math.sqrt(42) * (sh_2_4 * z - sh_2_0 * x)'),
    ('sh_4_0', '3 / 4 * math.sqrt(2) * (sh_3_0 * z + sh_3_6 * x)'),
    ('sh_4_1', '3 / 4 * sh_3_0 * y + 3 / 8 * math.sqrt(6) * sh_3_1 * z + 3 / 8 * math.sqrt(6) * sh_3_5 * x'),
    ('sh_4_2', '-3 / 56 * math.sqrt(14) * sh_3_0 * z + 3 / 14 * math.sqrt(21) * sh_3_1 * y + 3 / 56 * math.sqrt(210) * sh_3_2 * z + 3 / 56 * math.sqrt(210) * sh_3_4 * x + 3 / 56 * math.sqrt(14) * sh_3_6 * x'),
    ('sh_4_3', '-3 / 56 * math.sqrt(42) * sh_3_1 * z + 3 / 28 * math.sqrt(105) * sh_3_2 * y + 3 / 28 * math.sqrt(70) * sh_3_3 * x + 3 / 56 * math.sqrt(42) * sh_3_5 * x'),
    ('sh_4_4', '-3 / 28 * math.sqrt(42) * sh_3_2 * x + 3 / 7 * math.sqrt(7) * sh_3_3 * y - 3 / 28 * math.sqrt(42) * sh_3_4 * z'),
    ('sh_4_5', '-3 / 56 * math.sqrt(42) * sh_3_1 * x + 3 / 28 * math.sqrt(70) * sh_3_3 * z + 3 / 28 * math.sqrt(105) * sh_3_4 * y - 3 / 56 * math.sqrt(42) * sh_3_5 * z'),
    ('sh_4_6', '-3 / 56 * math.sqrt(14) * sh_3_0 * x - 3 / 56 * math.sqrt(210) * sh_3_2 * x + 3 / 56 * math.sqrt(210) * sh_3_4 * z + 3 / 14 * math.sqrt(21) * sh_3_5 * y - 3 / 56 * math.sqrt(14) * sh_3_6 * z'),
    ('sh_4_7', '-3 / 8 * math.sqrt(6) * sh_3_1 * x + 3 / 8 * math.sqrt(6) * sh_3_5 * z + 3 / 4 * sh_3_6 * y'),
    ('sh_4_8', '3 / 4 * math.sqrt(2) * (-sh_3_0 * x + sh_3_6 * z)'),
    ('sh_5_0', '1 / 10 * math.sqrt(110) * (sh_4_0 * z + sh_4_8 * x)'),
    ('sh_5_1', '1 / 5 * math.sqrt(11) * sh_4_0 * y + 1 / 5 * math.sqrt(22) * sh_4_1 * z + 1 / 5 * math.sqrt(22) * sh_4_7 * x'),
    ('sh_5_2', '-1 / 30 * math.sqrt(22) * sh_4_0 * z + 4 / 15 * math.sqrt(11) * sh_4_1 * y + 1 / 15 * math.sqrt(154) * sh_4_2 * z + 1 / 15 * math.sqrt(154) * sh_4_6 * x + 1 / 30 * math.sqrt(22) * sh_4_8 * x'),
    ('sh_5_3', '-1 / 30 * math.sqrt(66) * sh_4_1 * z + 1 / 15 * math.sqrt(231) * sh_4_2 * y + 1 / 30 * math.sqrt(462) * sh_4_3 * z + 1 / 30 * math.sqrt(462) * sh_4_5 * x + 1 / 30 * math.sqrt(66) * sh_4_7 * x'),
    ('sh_5_4', '-1 / 15 * math.sqrt(33) * sh_4_2 * z + 2 / 15 * math.sqrt(66) * sh_4_3 * y + 1 / 15 * math.sqrt(165) * sh_4_4 * x + 1 / 15 * math.sqrt(33) * sh_4_6 * x'),
    ('sh_5_5', '-1 / 15 * math.sqrt(110) * sh_4_3 * x + 1 / 3 * math.sqrt(11) * sh_4_4 * y - 1 / 15 * math.sqrt(110) * sh_4_5 * z'),
    ('sh_5_6', '-1 / 15 * math.sqrt(33) * sh_4_2 * x + 1 / 15 * math.sqrt(165) * sh_4_4 * z + 2 / 15 * math.sqrt(66) * sh_4_5 * y - 1 / 15 * math.sqrt(33) * sh_4_6 * z'),
    ('sh_5_7', '-1 / 30 * math.sqrt(66) * sh_4_1 * x - 1 / 30 * math.sqrt(462) * sh_4_3 * x + 1 / 30 * math.sqrt(462) * sh_4_5 * z + 1 / 15 * math.sqrt(231) * sh_4_6 * y - 1 / 30 * math.sqrt(66) * sh_4_7 * z'),
    ('sh_5_8', '-1 / 30 * math.sqrt(22) * sh_4_0 * x - 1 / 15 * math.sqrt(154) * sh_4_2 * x + 1 / 15 * math.sqrt(154) * sh_4_6 * z + 4 / 15 * math.sqrt(11) * sh_4_7 * y - 1 / 30 * math.sqrt(22) * sh_4_8 * z'),
    ('sh_5_9', '-1 / 5 * math.sqrt(22) * sh_4_1 * x + 1 / 5 * math.sqrt(22) * sh_4_7 * z + 1 / 5 * math.sqrt(11) * sh_4_8 * y'),
    ('sh_5_10', '1 / 10 * math.sqrt(110) * (-sh_4_0 * x + sh_4_8 * z)'),
    ('sh_6_0', '1 / 6 * math.sqrt(39) * (sh_5_0 * z + sh_5_10 * x)'),
    ('sh_6_1', '1 / 6 * math.sqrt(13) * sh_5_0 * y + 1 / 12 * math.sqrt(130) * sh_5_1 * z + 1 / 12 * math.sqrt(130) * sh_5_9 * x'),
    ('sh_6_2', '-1 / 132 * math.sqrt(286) * sh_5_0 * z + 1 / 33 * math.sqrt(715) * sh_5_1 * y + 1 / 132 * math.sqrt(286) * sh_5_10 * x + 1 / 44 * math.sqrt(1430) * sh_5_2 * z + 1 / 44 * math.sqrt(1430) * sh_5_8 * x'),
    ('sh_6_3', '-1 / 132 * math.sqrt(858) * sh_5_1 * z + 1 / 22 * math.sqrt(429) * sh_5_2 * y + 1 / 22 * math.sqrt(286) * sh_5_3 * z + 1 / 22 * math.sqrt(286) * sh_5_7 * x + 1 / 132 * math.sqrt(858) * sh_5_9 * x'),
    ('sh_6_4', '-1 / 66 * math.sqrt(429) * sh_5_2 * z + 2 / 33 * math.sqrt(286) * sh_5_3 * y + 1 / 66 * math.sqrt(2002) * sh_5_4 * z + 1 / 66 * math.sqrt(2002) * sh_5_6 * x + 1 / 66 * math.sqrt(429) * sh_5_8 * x'),
    ('sh_6_5', '-1 / 66 * math.sqrt(715) * sh_5_3 * z + 1 / 66 * math.sqrt(5005) * sh_5_4 * y + 1 / 66 * math.sqrt(3003) * sh_5_5 * x + 1 / 66 * math.sqrt(715) * sh_5_7 * x'),
    ('sh_6_6', '-1 / 66 * math.sqrt(2145) * sh_5_4 * x + 1 / 11 * math.sqrt(143) * sh_5_5 * y - 1 / 66 * math.sqrt(2145) * sh_5_6 * z'),
    ('sh_6_7', '-1 / 66 * math.sqrt(715) * sh_5_3 * x + 1 / 66 * math.sqrt(3003) * sh_5_5 * z + 1 / 66 * math.sqrt(5005) * sh_5_6 * y - 1 / 66 * math.sqrt(715) * sh_5_7 * z'),
    ('sh_6_8', '-1 / 66 * math.sqrt(429) * sh_5_2 * x - 1 / 66 * math.sqrt(2002) * sh_5_4 * x + 1 / 66 * math.sqrt(2002) * sh_5_6 * z + 2 / 33 * math.sqrt(286) * sh_5_7 * y - 1 / 66 * math.sqrt(429) * sh_5_8 * z'),
    ('sh_6_9', '-1 / 132 * math.sqrt(858) * sh_5_1 * x - 1 / 22 * math.sqrt(286) * sh_5_3 * x + 1 / 22 * math.sqrt(286) * sh_5_7 * z + 1 / 22 * math.sqrt(429) * sh_5_8 * y - 1 / 132 * math.sqrt(858) * sh_5_9 * z'),
    ('sh_6_10', '-1 / 132 * math.sqrt(286) * sh_5_0 * x - 1 / 132 * math.sqrt(286) * sh_5_10 * z - 1 / 44 * math.sqrt(1430) * sh_5_2 * x + 1 / 44 * math.sqrt(1430) * sh_5_8 * z + 1 / 33 * math.sqrt(715) * sh_5_9 * y'),
    ('sh_6_11', '-1 / 12 * math.sqrt(130) * sh_5_1 * x + 1 / 6 * math.sqrt(13) * sh_5_10 * y + 1 / 12 * math.sqrt(130) * sh_5_9 * z'),
    ('sh_6_12', '1 / 6 * math.sqrt(39) * (-sh_5_0 * x + sh_5_10 * z)'),
    ('sh_7_0', '1 / 14 * math.sqrt(210) * (sh_6_0 * z + sh_6_12 * x)'),
    ('sh_7_1', '1 / 7 * math.sqrt(15) * sh_6_0 * y + 3 / 7 * math.sqrt(5) * sh_6_1 * z + 3 / 7 * math.sqrt(5) * sh_6_11 * x'),
    ('sh_7_2', '-1 / 182 * math.sqrt(390) * sh_6_0 * z + 6 / 91 * math.sqrt(130) * sh_6_1 * y + 3 / 91 * math.sqrt(715) * sh_6_10 * x + 1 / 182 * math.sqrt(390) * sh_6_12 * x + 3 / 91 * math.sqrt(715) * sh_6_2 * z'),
    ('sh_7_3', '-3 / 182 * math.sqrt(130) * sh_6_1 * z + 3 / 182 * math.sqrt(130) * sh_6_11 * x + 3 / 91 * math.sqrt(715) * sh_6_2 * y + 5 / 182 * math.sqrt(858) * sh_6_3 * z + 5 / 182 * math.sqrt(858) * sh_6_9 * x'),
    ('sh_7_4', '3 / 91 * math.sqrt(65) * sh_6_10 * x - 3 / 91 * math.sqrt(65) * sh_6_2 * z + 10 / 91 * math.sqrt(78) * sh_6_3 * y + 15 / 182 * math.sqrt(78) * sh_6_4 * z + 15 / 182 * math.sqrt(78) * sh_6_8 * x'),
    ('sh_7_5', '-5 / 91 * math.sqrt(39) * sh_6_3 * z + 15 / 91 * math.sqrt(39) * sh_6_4 * y + 3 / 91 * math.sqrt(390) * sh_6_5 * z + 3 / 91 * math.sqrt(390) * sh_6_7 * x + 5 / 91 * math.sqrt(39) * sh_6_9 * x'),
    ('sh_7_6', '-15 / 182 * math.sqrt(26) * sh_6_4 * z + 12 / 91 * math.sqrt(65) * sh_6_5 * y + 2 / 91 * math.sqrt(1365) * sh_6_6 * x + 15 / 182 * math.sqrt(26) * sh_6_8 * x'),
    ('sh_7_7', '-3 / 91 * math.sqrt(455) * sh_6_5 * x + 1 / 13 * math.sqrt(195) * sh_6_6 * y - 3 / 91 * math.sqrt(455) * sh_6_7 * z'),
    ('sh_7_8', '-15 / 182 * math.sqrt(26) * sh_6_4 * x + 2 / 91 * math.sqrt(1365) * sh_6_6 * z + 12 / 91 * math.sqrt(65) * sh_6_7 * y - 15 / 182 * math.sqrt(26) * sh_6_8 * z'),
    ('sh_7_9', '-5 / 91 * math.sqrt(39) * sh_6_3 * x - 3 / 91 * math.sqrt(390) * sh_6_5 * x + 3 / 91 * math.sqrt(390) * sh_6_7 * z + 15 / 91 * math.sqrt(39) * sh_6_8 * y - 5 / 91 * math.sqrt(39) * sh_6_9 * z'),
    ('sh_7_10', '-3 / 91 * math.sqrt(65) * sh_6_10 * z - 3 / 91 * math.sqrt(65) * sh_6_2 * x - 15 / 182 * math.sqrt(78) * sh_6_4 * x + 15 / 182 * math.sqrt(78) * sh_6_8 * z + 10 / 91 * math.sqrt(78) * sh_6_9 * y'),
    ('sh_7_11', '-3 / 182 * math.sqrt(130) * sh_6_1 * x + 3 / 91 * math.sqrt(715) * sh_6_10 * y - 3 / 182 * math.sqrt(130) * sh_6_11 * z - 5 / 182 * math.sqrt(858) * sh_6_3 * x + 5 / 182 * math.sqrt(858) * sh_6_9 * z'),
    ('sh_7_12', '-1 / 182 * math.sqrt(390) * sh_6_0 * x + 3 / 91 * math.sqrt(715) * sh_6_10 * z + 6 / 91 * math.sqrt(130) * sh_6_11 * y - 1 / 182 * math.sqrt(390) * sh_6_12 * z - 3 / 91 * math.sqrt(715) * sh_6_2 * x'),
    ('sh_7_13', '-3 / 7 * math.sqrt(5) * sh_6_1 * x + 3 / 7 * math.sqrt(5) * sh_6_11 * z + 1 / 7 * math.sqrt(15) * sh_6_12 * y'),
    ('sh_7_14', '1 / 14 * math.sqrt(210) * (-sh_6_0 * x + sh_6_12 * z)'),
    ('sh_8_0', '1 / 4 * math.sqrt(17) * (sh_7_0 * z + sh_7_14 * x)'),
    ('sh_8_1', '1 / 8 * math.sqrt(17) * sh_7_0 * y + 1 / 16 * math.sqrt(238) * sh_7_1 * z + 1 / 16 * math.sqrt(238) * sh_7_13 * x'),
    ('sh_8_2', '-1 / 240 * math.sqrt(510) * sh_7_0 * z + 1 / 60 * math.sqrt(1785) * sh_7_1 * y + 1 / 240 * math.sqrt(46410) * sh_7_12 * x + 1 / 240 * math.sqrt(510) * sh_7_14 * x + 1 / 240 * math.sqrt(46410) * sh_7_2 * z'),
    ('sh_8_3', '1 / 80 * math.sqrt(2) * (-math.sqrt(85) * sh_7_1 * z + math.sqrt(2210) * sh_7_11 * x + math.sqrt(85) * sh_7_13 * x + math.sqrt(2210) * sh_7_2 * y + math.sqrt(2210) * sh_7_3 * z)'),
    ('sh_8_4', '1 / 40 * math.sqrt(935) * sh_7_10 * x + 1 / 40 * math.sqrt(85) * sh_7_12 * x - 1 / 40 * math.sqrt(85) * sh_7_2 * z + 1 / 10 * math.sqrt(85) * sh_7_3 * y + 1 / 40 * math.sqrt(935) * sh_7_4 * z'),
    ('sh_8_5', '1 / 48 * math.sqrt(2) * (math.sqrt(102) * sh_7_11 * x - math.sqrt(102) * sh_7_3 * z + math.sqrt(1122) * sh_7_4 * y + math.sqrt(561) * sh_7_5 * z + math.sqrt(561) * sh_7_9 * x)'),
    ('sh_8_6', '1 / 16 * math.sqrt(34) * sh_7_10 * x - 1 / 16 * math.sqrt(34) * sh_7_4 * z + 1 / 4 * math.sqrt(17) * sh_7_5 * y + 1 / 16 * math.sqrt(102) * sh_7_6 * z + 1 / 16 * math.sqrt(102) * sh_7_8 * x'),
    ('sh_8_7', '-1 / 80 * math.sqrt(1190) * sh_7_5 * z + 1 / 40 * math.sqrt(1785) * sh_7_6 * y + 1 / 20 * math.sqrt(255) * sh_7_7 * x + 1 / 80 * math.sqrt(1190) * sh_7_9 * x'),
    ('sh_8_8', '-1 / 60 * math.sqrt(1785) * sh_7_6 * x + 1 / 15 * math.sqrt(255) * sh_7_7 * y - 1 / 60 * math.sqrt(1785) * sh_7_8 * z'),
    ('sh_8_9', '-1 / 80 * math.sqrt(1190) * sh_7_5 * x + 1 / 20 * math.sqrt(255) * sh_7_7 * z + 1 / 40 * math.sqrt(1785) * sh_7_8 * y - 1 / 80 * math.sqrt(1190) * sh_7_9 * z'),
    ('sh_8_10', '-1 / 16 * math.sqrt(34) * sh_7_10 * z - 1 / 16 * math.sqrt(34) * sh_7_4 * x - 1 / 16 * math.sqrt(102) * sh_7_6 * x + 1 / 16 * math.sqrt(102) * sh_7_8 * z + 1 / 4 * math.sqrt(17) * sh_7_9 * y'),
    ('sh_8_11', '1 / 48 * math.sqrt(2) * (math.sqrt(1122) * sh_7_10 * y - math.sqrt(102) * sh_7_11 * z - math.sqrt(102) * sh_7_3 * x - math.sqrt(561) * sh_7_5 * x + math.sqrt(561) * sh_7_9 * z)'),
    ('sh_8_12', '1 / 40 * math.sqrt(935) * sh_7_10 * z + 1 / 10 * math.sqrt(85) * sh_7_11 * y - 1 / 40 * math.sqrt(85) * sh_7_12 * z - 1 / 40 * math.sqrt(85) * sh_7_2 * x - 1 / 40 * math.sqrt(935) * sh_7_4 * x'),
    ('sh_8_13', '1 / 80 * math.sqrt(2) * (-math.sqrt(85) * sh_7_1 * x + math.sqrt(2210) * sh_7_11 * z + math.sqrt(2210) * sh_7_12 * y - math.sqrt(85) * sh_7_13 * z - math.sqrt(2210) * sh_7_3 * x)'),
    ('sh_8_14', '-1 / 240 * math.sqrt(510) * sh_7_0 * x + 1 / 240 * math.sqrt(46410) * sh_7_12 * z + 1 / 60 * math.sqrt(1785) * sh_7_13 * y - 1 / 240 * math.sqrt(510) * sh_7_14 * z - 1 / 240 * math.sqrt(46410) * sh_7_2 * x'),
    ('sh_8_15', '-1 / 16 * math.sqrt(238) * sh_7_1 * x + 1 / 16 * math.sqrt(238) * sh_7_13 * z + 1 / 8 * math.sqrt(17) * sh_7_14 * y'),
    ('sh_8_16', '1 / 4 * math.sqrt(17) * (-sh_7_0 * x + sh_7_14 * z)'),
    ('sh_9_0', '1 / 6 * math.sqrt(38) * (sh_8_0 * z + sh_8_16 * x)'),
    ('sh_9_1', '1 / 9 * math.sqrt(19) * (sh_8_0 * y + 2 * sh_8_1 * z + 2 * sh_8_15 * x)'),
    ('sh_9_2', '-1 / 306 * math.sqrt(646) * sh_8_0 * z + 4 / 153 * math.sqrt(646) * sh_8_1 * y + 2 / 153 * math.sqrt(4845) * sh_8_14 * x + 1 / 306 * math.sqrt(646) * sh_8_16 * x + 2 / 153 * math.sqrt(4845) * sh_8_2 * z'),
    ('sh_9_3', '-1 / 306 * math.sqrt(1938) * sh_8_1 * z + 1 / 306 * math.sqrt(67830) * sh_8_13 * x + 1 / 306 * math.sqrt(1938) * sh_8_15 * x + 1 / 51 * math.sqrt(1615) * sh_8_2 * y + 1 / 306 * math.sqrt(67830) * sh_8_3 * z'),
    ('sh_9_4', '1 / 306 * math.sqrt(58786) * sh_8_12 * x + 1 / 153 * math.sqrt(969) * sh_8_14 * x - 1 / 153 * math.sqrt(969) * sh_8_2 * z + 2 / 153 * math.sqrt(4522) * sh_8_3 * y + 1 / 306 * math.sqrt(58786) * sh_8_4 * z'),
    ('sh_9_5', '1 / 153 * math.sqrt(12597) * sh_8_11 * x + 1 / 153 * math.sqrt(1615) * sh_8_13 * x - 1 / 153 * math.sqrt(1615) * sh_8_3 * z + 1 / 153 * math.sqrt(20995) * sh_8_4 * y + 1 / 153 * math.sqrt(12597) * sh_8_5 * z'),
    ('sh_9_6', '1 / 153 * math.sqrt(10659) * sh_8_10 * x + 1 / 306 * math.sqrt(9690) * sh_8_12 * x - 1 / 306 * math.sqrt(9690) * sh_8_4 * z + 2 / 51 * math.sqrt(646) * sh_8_5 * y + 1 / 153 * math.sqrt(10659) * sh_8_6 * z'),
    ('sh_9_7', '1 / 306 * math.sqrt(13566) * sh_8_11 * x - 1 / 306 * math.sqrt(13566) * sh_8_5 * z + 1 / 153 * math.sqrt(24871) * sh_8_6 * y + 1 / 306 * math.sqrt(35530) * sh_8_7 * z + 1 / 306 * math.sqrt(35530) * sh_8_9 * x'),
    ('sh_9_8', '1 / 153 * math.sqrt(4522) * sh_8_10 * x - 1 / 153 * math.sqrt(4522) * sh_8_6 * z + 4 / 153 * math.sqrt(1615) * sh_8_7 * y + 1 / 51 * math.sqrt(1615) * sh_8_8 * x'),
    ('sh_9_9', '1 / 51 * math.sqrt(323) * (-2 * sh_8_7 * x + 3 * sh_8_8 * y - 2 * sh_8_9 * z)'),
    ('sh_9_10', '-1 / 153 * math.sqrt(4522) * sh_8_10 * z - 1 / 153 * math.sqrt(4522) * sh_8_6 * x + 1 / 51 * math.sqrt(1615) * sh_8_8 * z + 4 / 153 * math.sqrt(1615) * sh_8_9 * y'),
    ('sh_9_11', '1 / 153 * math.sqrt(24871) * sh_8_10 * y - 1 / 306 * math.sqrt(13566) * sh_8_11 * z - 1 / 306 * math.sqrt(13566) * sh_8_5 * x - 1 / 306 * math.sqrt(35530) * sh_8_7 * x + 1 / 306 * math.sqrt(35530) * sh_8_9 * z'),
    ('sh_9_12', '1 / 153 * math.sqrt(10659) * sh_8_10 * z + 2 / 51 * math.sqrt(646) * sh_8_11 * y - 1 / 306 * math.sqrt(9690) * sh_8_12 * z - 1 / 306 * math.sqrt(9690) * sh_8_4 * x - 1 / 153 * math.sqrt(10659) * sh_8_6 * x'),
    ('sh_9_13', '1 / 153 * math.sqrt(12597) * sh_8_11 * z + 1 / 153 * math.sqrt(20995) * sh_8_12 * y - 1 / 153 * math.sqrt(1615) * sh_8_13 * z - 1 / 153 * math.sqrt(1615) * sh_8_3 * x - 1 / 153 * math.sqrt(12597) * sh_8_5 * x'),
    ('sh_9_14', '1 / 306 * math.sqrt(58786) * sh_8_12 * z + 2 / 153 * math.sqrt(4522) * sh_8_13 * y - 1 / 153 * math.sqrt(969) * sh_8_14 * z - 1 / 153 * math.sqrt(969) * sh_8_2 * x - 1 / 306 * math.sqrt(58786) * sh_8_4 * x'),
    ('sh_9_15', '-1 / 306 * math.sqrt(1938) * sh_8_1 * x + 1 / 306 * math.sqrt(67830) * sh_8_13 * z + 1 / 51 * math.sqrt(1615) * sh_8_14 * y - 1 / 306 * math.sqrt(1938) * sh_8_15 * z - 1 / 306 * math.sqrt(67830) * sh_8_3 * x'),
    ('sh_9_16', '-1 / 306 * math.sqrt(646) * sh_8_0 * x + 2 / 153 * math.sqrt(4845) * sh_8_14 * z + 4 / 153 * math.sqrt(646) * sh_8_15 * y - 1 / 306 * math.sqrt(646) * sh_8_16 * z - 2 / 153 * math.sqrt(4845) * sh_8_2 * x'),
    ('sh_9_17', '1 / 9 * math.sqrt(19) * (-2 * sh_8_1 * x + 2 * sh_8_15 * z + sh_8_16 * y)'),
    ('sh_9_18', '1 / 6 * math.sqrt(38) * (-sh_8_0 * x + sh_8_16 * z)'),
    ('sh_10_0', '1 / 10 * math.sqrt(105) * (sh_9_0 * z + sh_9_18 * x)'),
    ('sh_10_1', '1 / 10 * math.sqrt(21) * sh_9_0 * y + 3 / 20 * math.sqrt(42) * sh_9_1 * z + 3 / 20 * math.sqrt(42) * sh_9_17 * x'),
    ('sh_10_2', '-1 / 380 * math.sqrt(798) * sh_9_0 * z + 3 / 95 * math.sqrt(399) * sh_9_1 * y + 3 / 380 * math.sqrt(13566) * sh_9_16 * x + 1 / 380 * math.sqrt(798) * sh_9_18 * x + 3 / 380 * math.sqrt(13566) * sh_9_2 * z'),
    ('sh_10_3', '-3 / 380 * math.sqrt(266) * sh_9_1 * z + 1 / 95 * math.sqrt(6783) * sh_9_15 * x + 3 / 380 * math.sqrt(266) * sh_9_17 * x + 3 / 190 * math.sqrt(2261) * sh_9_2 * y + 1 / 95 * math.sqrt(6783) * sh_9_3 * z'),
    ('sh_10_4', '3 / 95 * math.sqrt(665) * sh_9_14 * x + 3 / 190 * math.sqrt(133) * sh_9_16 * x - 3 / 190 * math.sqrt(133) * sh_9_2 * z + 4 / 95 * math.sqrt(399) * sh_9_3 * y + 3 / 95 * math.sqrt(665) * sh_9_4 * z'),
    ('sh_10_5', '21 / 380 * math.sqrt(190) * sh_9_13 * x + 1 / 190 * math.sqrt(1995) * sh_9_15 * x - 1 / 190 * math.sqrt(1995) * sh_9_3 * z + 3 / 38 * math.sqrt(133) * sh_9_4 * y + 21 / 380 * math.sqrt(190) * sh_9_5 * z'),
    ('sh_10_6', '7 / 380 * math.sqrt(1482) * sh_9_12 * x + 3 / 380 * math.sqrt(1330) * sh_9_14 * x - 3 / 380 * math.sqrt(1330) * sh_9_4 * z + 21 / 95 * math.sqrt(19) * sh_9_5 * y + 7 / 380 * math.sqrt(1482) * sh_9_6 * z'),
    ('sh_10_7', '3 / 190 * math.sqrt(1729) * sh_9_11 * x + 21 / 380 * math.sqrt(38) * sh_9_13 * x - 21 / 380 * math.sqrt(38) * sh_9_5 * z + 7 / 190 * math.sqrt(741) * sh_9_6 * y + 3 / 190 * math.sqrt(1729) * sh_9_7 * z'),
    ('sh_10_8', '3 / 190 * math.sqrt(1463) * sh_9_10 * x + 7 / 190 * math.sqrt(114) * sh_9_12 * x - 7 / 190 * math.sqrt(114) * sh_9_6 * z + 6 / 95 * math.sqrt(266) * sh_9_7 * y + 3 / 190 * math.sqrt(1463) * sh_9_8 * z'),
    ('sh_10_9', '3 / 190 * math.sqrt(798) * sh_9_11 * x - 3 / 190 * math.sqrt(798) * sh_9_7 * z + 3 / 190 * math.sqrt(4389) * sh_9_8 * y + 1 / 190 * math.sqrt(21945) * sh_9_9 * x'),
    ('sh_10_10', '-3 / 190 * math.sqrt(1995) * sh_9_10 * z - 3 / 190 * math.sqrt(1995) * sh_9_8 * x + 1 / 19 * math.sqrt(399) * sh_9_9 * y'),
    ('sh_10_11', '3 / 190 * math.sqrt(4389) * sh_9_10 * y - 3 / 190 * math.sqrt(798) * sh_9_11 * z - 3 / 190 * math.sqrt(798) * sh_9_7 * x + 1 / 190 * math.sqrt(21945) * sh_9_9 * z'),
    ('sh_10_12', '3 / 190 * math.sqrt(1463) * sh_9_10 * z + 6 / 95 * math.sqrt(266) * sh_9_11 * y - 7 / 190 * math.sqrt(114) * sh_9_12 * z - 7 / 190 * math.sqrt(114) * sh_9_6 * x - 3 / 190 * math.sqrt(1463) * sh_9_8 * x'),
    ('sh_10_13', '3 / 190 * math.sqrt(1729) * sh_9_11 * z + 7 / 190 * math.sqrt(741) * sh_9_12 * y - 21 / 380 * math.sqrt(38) * sh_9_13 * z - 21 / 380 * math.sqrt(38) * sh_9_5 * x - 3 / 190 * math.sqrt(1729) * sh_9_7 * x'),
    ('sh_10_14', '7 / 380 * math.sqrt(1482) * sh_9_12 * z + 21 / 95 * math.sqrt(19) * sh_9_13 * y - 3 / 380 * math.sqrt(1330) * sh_9_14 * z - 3 / 380 * math.sqrt(1330) * sh_9_4 * x - 7 / 380 * math.sqrt(1482) * sh_9_6 * x'),
    ('sh_10_15', '21 / 380 * math.sqrt(190) * sh_9_13 * z + 3 / 38 * math.sqrt(133) * sh_9_14 * y - 1 / 190 * math.sqrt(1995) * sh_9_15 * z - 1 / 190 * math.sqrt(1995) * sh_9_3 * x - 21 / 380 * math.sqrt(190) * sh_9_5 * x'),
    ('sh_10_16', '3 / 95 * math.sqrt(665) * sh_9_14 * z + 4 / 95 * math.sqrt(399) * sh_9_15 * y - 3 / 190 * math.sqrt(133) * sh_9_16 * z - 3 / 190 * math.sqrt(133) * sh_9_2 * x - 3 / 95 * math.sqrt(665) * sh_9_4 * x'),
    ('sh_10_17', '-3 / 380 * math.sqrt(266) * sh_9_1 * x + 1 / 95 * math.sqrt(6783) * sh_9_15 * z + 3 / 190 * math.sqrt(2261) * sh_9_16 * y - 3 / 380 * math.sqrt(266) * sh_9_17 * z - 1 / 95 * math.sqrt(6783) * sh_9_3 * x'),
    ('sh_10_18', '-1 / 380 * math.sqrt(798) * sh_9_0 * x + 3 / 380 * math.sqrt(13566) * sh_9_16 * z + 3 / 95 * math.sqrt(399) * sh_9_17 * y - 1 / 380 * math.sqrt(798) * sh_9_18 * z - 3 / 380 * math.sqrt(13566) * sh_9_2 * x'),
    ('sh_10_19', '-3 / 20 * math.sqrt(42) * sh_9_1 * x + 3 / 20 * math.sqrt(42) * sh_9_17 * z + 1 / 10 * math.sqrt(21) * sh_9_18 * y'),
    ('sh_10_20', '1 / 10 * math.sqrt(105) * (-sh_9_0 * x + sh_9_18 * z)'),
    ('sh_11_0', '1 / 22 * math.sqrt(506) * (sh_10_0 * z + sh_10_20 * x)'),
    ('sh_11_1', '1 / 11 * math.sqrt(23) * sh_10_0 * y + 1 / 11 * math.sqrt(115) * sh_10_1 * z + 1 / 11 * math.sqrt(115) * sh_10_19 * x'),
    ('sh_11_2', '-1 / 462 * math.sqrt(966) * sh_10_0 * z + 2 / 231 * math.sqrt(4830) * sh_10_1 * y + 1 / 231 * math.sqrt(45885) * sh_10_18 * x + 1 / 231 * math.sqrt(45885) * sh_10_2 * z + 1 / 462 * math.sqrt(966) * sh_10_20 * x'),
    ('sh_11_3', '-1 / 154 * math.sqrt(322) * sh_10_1 * z + 1 / 154 * math.sqrt(18354) * sh_10_17 * x + 1 / 154 * math.sqrt(322) * sh_10_19 * x + 1 / 77 * math.sqrt(3059) * sh_10_2 * y + 1 / 154 * math.sqrt(18354) * sh_10_3 * z'),
    ('sh_11_4', '1 / 154 * math.sqrt(16422) * sh_10_16 * x + 1 / 77 * math.sqrt(161) * sh_10_18 * x - 1 / 77 * math.sqrt(161) * sh_10_2 * z + 2 / 77 * math.sqrt(966) * sh_10_3 * y + 1 / 154 * math.sqrt(16422) * sh_10_4 * z'),
    ('sh_11_5', '2 / 231 * math.sqrt(8211) * sh_10_15 * x + 1 / 231 * math.sqrt(2415) * sh_10_17 * x - 1 / 231 * math.sqrt(2415) * sh_10_3 * z + 1 / 231 * math.sqrt(41055) * sh_10_4 * y + 2 / 231 * math.sqrt(8211) * sh_10_5 * z'),
    ('sh_11_6', '2 / 77 * math.sqrt(805) * sh_10_14 * x + 1 / 154 * math.sqrt(1610) * sh_10_16 * x - 1 / 154 * math.sqrt(1610) * sh_10_4 * z + 4 / 77 * math.sqrt(322) * sh_10_5 * y + 2 / 77 * math.sqrt(805) * sh_10_6 * z'),
    ('sh_11_7', '1 / 22 * math.sqrt(230) * sh_10_13 * x + 1 / 22 * math.sqrt(46) * sh_10_15 * x - 1 / 22 * math.sqrt(46) * sh_10_5 * z + 1 / 11 * math.sqrt(115) * sh_10_6 * y + 1 / 22 * math.sqrt(230) * sh_10_7 * z'),
    ('sh_11_8', '1 / 66 * math.sqrt(1794) * sh_10_12 * x + 1 / 33 * math.sqrt(138) * sh_10_14 * x - 1 / 33 * math.sqrt(138) * sh_10_6 * z + 4 / 33 * math.sqrt(69) * sh_10_7 * y + 1 / 66 * math.sqrt(1794) * sh_10_8 * z'),
    ('sh_11_9', '1 / 77 * math.sqrt(2093) * sh_10_11 * x + 1 / 77 * math.sqrt(966) * sh_10_13 * x - 1 / 77 * math.sqrt(966) * sh_10_7 * z + 1 / 77 * math.sqrt(6279) * sh_10_8 * y + 1 / 77 * math.sqrt(2093) * sh_10_9 * z'),
    ('sh_11_10', '1 / 77 * math.sqrt(3542) * sh_10_10 * x + 1 / 154 * math.sqrt(4830) * sh_10_12 * x - 1 / 154 * math.sqrt(4830) * sh_10_8 * z + 2 / 77 * math.sqrt(1610) * sh_10_9 * y'),
    ('sh_11_11', '1 / 21 * math.sqrt(483) * sh_10_10 * y - 1 / 231 * math.sqrt(26565) * sh_10_11 * z - 1 / 231 * math.sqrt(26565) * sh_10_9 * x'),
    ('sh_11_12', '1 / 77 * math.sqrt(3542) * sh_10_10 * z + 2 / 77 * math.sqrt(1610) * sh_10_11 * y - 1 / 154 * math.sqrt(4830) * sh_10_12 * z - 1 / 154 * math.sqrt(4830) * sh_10_8 * x'),
    ('sh_11_13', '1 / 77 * math.sqrt(2093) * sh_10_11 * z + 1 / 77 * math.sqrt(6279) * sh_10_12 * y - 1 / 77 * math.sqrt(966) * sh_10_13 * z - 1 / 77 * math.sqrt(966) * sh_10_7 * x - 1 / 77 * math.sqrt(2093) * sh_10_9 * x'),
    ('sh_11_14', '1 / 66 * math.sqrt(1794) * sh_10_12 * z + 4 / 33 * math.sqrt(69) * sh_10_13 * y - 1 / 33 * math.sqrt(138) * sh_10_14 * z - 1 / 33 * math.sqrt(138) * sh_10_6 * x - 1 / 66 * math.sqrt(1794) * sh_10_8 * x'),
    ('sh_11_15', '1 / 22 * math.sqrt(230) * sh_10_13 * z + 1 / 11 * math.sqrt(115) * sh_10_14 * y - 1 / 22 * math.sqrt(46) * sh_10_15 * z - 1 / 22 * math.sqrt(46) * sh_10_5 * x - 1 / 22 * math.sqrt(230) * sh_10_7 * x'),
    ('sh_11_16', '2 / 77 * math.sqrt(805) * sh_10_14 * z + 4 / 77 * math.sqrt(322) * sh_10_15 * y - 1 / 154 * math.sqrt(1610) * sh_10_16 * z - 1 / 154 * math.sqrt(1610) * sh_10_4 * x - 2 / 77 * math.sqrt(805) * sh_10_6 * x'),
    ('sh_11_17', '2 / 231 * math.sqrt(8211) * sh_10_15 * z + 1 / 231 * math.sqrt(41055) * sh_10_16 * y - 1 / 231 * math.sqrt(2415) * sh_10_17 * z - 1 / 231 * math.sqrt(2415) * sh_10_3 * x - 2 / 231 * math.sqrt(8211) * sh_10_5 * x'),
    ('sh_11_18', '1 / 154 * math.sqrt(16422) * sh_10_16 * z + 2 / 77 * math.sqrt(966) * sh_10_17 * y - 1 / 77 * math.sqrt(161) * sh_10_18 * z - 1 / 77 * math.sqrt(161) * sh_10_2 * x - 1 / 154 * math.sqrt(16422) * sh_10_4 * x'),
    ('sh_11_19', '-1 / 154 * math.sqrt(322) * sh_10_1 * x + 1 / 154 * math.sqrt(18354) * sh_10_17 * z + 1 / 77 * math.sqrt(3059) * sh_10_18 * y - 1 / 154 * math.sqrt(322) * sh_10_19 * z - 1 / 154 * math.sqrt(18354) * sh_10_3 * x'),
    ('sh_11_20', '-1 / 462 * math.sqrt(966) * sh_10_0 * x + 1 / 231 * math.sqrt(45885) * sh_10_18 * z + 2 / 231 * math.sqrt(4830) * sh_10_19 * y - 1 / 231 * math.sqrt(45885) * sh_10_2 * x - 1 / 462 * math.sqrt(966) * sh_10_20 * z'),
    ('sh_11_21', '-1 / 11 * math.sqrt(115) * sh_10_1 * x + 1 / 11 * math.sqrt(115) * sh_10_19 * z + 1 / 11 * math.sqrt(23) * sh_10_20 * y'),
    ('sh_11_22', '1 / 22 * math.sqrt(506) * (-sh_10_0 * x + sh_10_20 * z)'),
    ('sh_12_0', '5 / 12 * math.sqrt(6) * (sh_11_0 * z + sh_11_22 * x)'),
    ('sh_12_1', '5 / 12 * sh_11_0 * y + 5 / 24 * math.sqrt(22) * sh_11_1 * z + 5 / 24 * math.sqrt(22) * sh_11_21 * x'),
    ('sh_12_2', '-5 / 552 * math.sqrt(46) * sh_11_0 * z + 5 / 138 * math.sqrt(253) * sh_11_1 * y + 5 / 552 * math.sqrt(10626) * sh_11_2 * z + 5 / 552 * math.sqrt(10626) * sh_11_20 * x + 5 / 552 * math.sqrt(46) * sh_11_22 * x'),
    ('sh_12_3', '-5 / 552 * math.sqrt(138) * sh_11_1 * z + 5 / 276 * math.sqrt(2415) * sh_11_19 * x + 5 / 92 * math.sqrt(161) * sh_11_2 * y + 5 / 552 * math.sqrt(138) * sh_11_21 * x + 5 / 276 * math.sqrt(2415) * sh_11_3 * z'),
    ('sh_12_4', '5 / 276 * math.sqrt(2185) * sh_11_18 * x - 5 / 276 * math.sqrt(69) * sh_11_2 * z + 5 / 276 * math.sqrt(69) * sh_11_20 * x + 5 / 69 * math.sqrt(115) * sh_11_3 * y + 5 / 276 * math.sqrt(2185) * sh_11_4 * z'),
    ('sh_12_5', '5 / 184 * math.sqrt(874) * sh_11_17 * x + 5 / 276 * math.sqrt(115) * sh_11_19 * x - 5 / 276 * math.sqrt(115) * sh_11_3 * z + 5 / 276 * math.sqrt(2185) * sh_11_4 * y + 5 / 184 * math.sqrt(874) * sh_11_5 * z'),
    ('sh_12_6', '5 / 552 * math.sqrt(3) * (math.sqrt(2346) * sh_11_16 * x + math.sqrt(230) * sh_11_18 * x - math.sqrt(230) * sh_11_4 * z + 12 * math.sqrt(23) * sh_11_5 * y + math.sqrt(2346) * sh_11_6 * z)'),
    ('sh_12_7', '5 / 138 * math.sqrt(391) * sh_11_15 * x + 5 / 552 * math.sqrt(966) * sh_11_17 * x - 5 / 552 * math.sqrt(966) * sh_11_5 * z + 5 / 276 * math.sqrt(2737) * sh_11_6 * y + 5 / 138 * math.sqrt(391) * sh_11_7 * z'),
    ('sh_12_8', '5 / 138 * math.sqrt(345) * sh_11_14 * x + 5 / 276 * math.sqrt(322) * sh_11_16 * x - 5 / 276 * math.sqrt(322) * sh_11_6 * z + 10 / 69 * math.sqrt(46) * sh_11_7 * y + 5 / 138 * math.sqrt(345) * sh_11_8 * z'),
    ('sh_12_9', '5 / 552 * math.sqrt(4830) * sh_11_13 * x + 5 / 92 * math.sqrt(46) * sh_11_15 * x - 5 / 92 * math.sqrt(46) * sh_11_7 * z + 5 / 92 * math.sqrt(345) * sh_11_8 * y + 5 / 552 * math.sqrt(4830) * sh_11_9 * z'),
    ('sh_12_10', '5 / 552 * math.sqrt(4186) * sh_11_10 * z + 5 / 552 * math.sqrt(4186) * sh_11_12 * x + 5 / 184 * math.sqrt(230) * sh_11_14 * x - 5 / 184 * math.sqrt(230) * sh_11_8 * z + 5 / 138 * math.sqrt(805) * sh_11_9 * y'),
    ('sh_12_11', '5 / 276 * math.sqrt(3289) * sh_11_10 * y + 5 / 276 * math.sqrt(1794) * sh_11_11 * x + 5 / 552 * math.sqrt(2530) * sh_11_13 * x - 5 / 552 * math.sqrt(2530) * sh_11_9 * z'),
    ('sh_12_12', '-5 / 276 * math.sqrt(1518) * sh_11_10 * x + 5 / 23 * math.sqrt(23) * sh_11_11 * y - 5 / 276 * math.sqrt(1518) * sh_11_12 * z'),
    ('sh_12_13', '5 / 276 * math.sqrt(1794) * sh_11_11 * z + 5 / 276 * math.sqrt(3289) * sh_11_12 * y - 5 / 552 * math.sqrt(2530) * sh_11_13 * z - 5 / 552 * math.sqrt(2530) * sh_11_9 * x'),
    ('sh_12_14', '-5 / 552 * math.sqrt(4186) * sh_11_10 * x + 5 / 552 * math.sqrt(4186) * sh_11_12 * z + 5 / 138 * math.sqrt(805) * sh_11_13 * y - 5 / 184 * math.sqrt(230) * sh_11_14 * z - 5 / 184 * math.sqrt(230) * sh_11_8 * x'),
    ('sh_12_15', '5 / 552 * math.sqrt(4830) * sh_11_13 * z + 5 / 92 * math.sqrt(345) * sh_11_14 * y - 5 / 92 * math.sqrt(46) * sh_11_15 * z - 5 / 92 * math.sqrt(46) * sh_11_7 * x - 5 / 552 * math.sqrt(4830) * sh_11_9 * x'),
    ('sh_12_16', '5 / 138 * math.sqrt(345) * sh_11_14 * z + 10 / 69 * math.sqrt(46) * sh_11_15 * y - 5 / 276 * math.sqrt(322) * sh_11_16 * z - 5 / 276 * math.sqrt(322) * sh_11_6 * x - 5 / 138 * math.sqrt(345) * sh_11_8 * x'),
    ('sh_12_17', '5 / 138 * math.sqrt(391) * sh_11_15 * z + 5 / 276 * math.sqrt(2737) * sh_11_16 * y - 5 / 552 * math.sqrt(966) * sh_11_17 * z - 5 / 552 * math.sqrt(966) * sh_11_5 * x - 5 / 138 * math.sqrt(391) * sh_11_7 * x'),
    ('sh_12_18', '5 / 552 * math.sqrt(3) * (math.sqrt(2346) * sh_11_16 * z + 12 * math.sqrt(23) * sh_11_17 * y - math.sqrt(230) * sh_11_18 * z - math.sqrt(230) * sh_11_4 * x - math.sqrt(2346) * sh_11_6 * x)'),
    ('sh_12_19', '5 / 184 * math.sqrt(874) * sh_11_17 * z + 5 / 276 * math.sqrt(2185) * sh_11_18 * y - 5 / 276 * math.sqrt(115) * sh_11_19 * z - 5 / 276 * math.sqrt(115) * sh_11_3 * x - 5 / 184 * math.sqrt(874) * sh_11_5 * x'),
    ('sh_12_20', '5 / 276 * math.sqrt(2185) * sh_11_18 * z + 5 / 69 * math.sqrt(115) * sh_11_19 * y - 5 / 276 * math.sqrt(69) * sh_11_2 * x - 5 / 276 * math.sqrt(69) * sh_11_20 * z - 5 / 276 * math.sqrt(2185) * sh_11_4 * x'),
    ('sh_12_21', '-5 / 552 * math.sqrt(138) * sh_11_1 * x + 5 / 276 * math.sqrt(2415) * sh_11_19 * z + 5 / 92 * math.sqrt(161) * sh_11_20 * y - 5 / 552 * math.sqrt(138) * sh_11_21 * z - 5 / 276 * math.sqrt(2415) * sh_11_3 * x'),
    ('sh_12_22', '-5 / 552 * math.sqrt(46) * sh_11_0 * x - 5 / 552 * math.sqrt(10626) * sh_11_2 * x + 5 / 552 * math.sqrt(10626) * sh_11_20 * z + 5 / 138 * math.sqrt(253) * sh_11_21 * y - 5 / 552 * math.sqrt(46) * sh_11_22 * z'),
    ('sh_12_23', '-5 / 24 * math.sqrt(22) * sh_11_1 * x + 5 / 24 * math.sqrt(22) * sh_11_21 * z + 5 / 12 * sh_11_22 * y'),
    ('sh_12_24', '5 / 12 * math.sqrt(6) * (-sh_11_0 * x + sh_11_22 * z)'),
)

class _Emitter:
    """Scalar expression DAG with constant folding and common subexpressions."""
    def __init__(self):
        self.lines, self.cache = [], {}

    def op(self, op, a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return {'+': lambda: a+b, '-': lambda: a-b,
                    '*': lambda: a*b, '/': lambda: a/b}[op]()
        if op == '+':
            if a == 0: return b
            if b == 0: return a
        if op == '-' and b == 0: return a
        if op == '*':
            if a == 0 or b == 0: return 0.0
            if a == 1: return b
            if b == 1: return a
        if op == '/' and b == 1: return a
        key = (op, a, b)
        if key not in self.cache:
            name = f'q{len(self.lines)}'
            self.lines.append(f'{name} = {a} {op} {b}')
            self.cache[key] = name
        return self.cache[key]

    def add(self, *args):
        value = 0.0
        for a in args: value = self.op('+', value, a)
        return value

    def mul(self, a, b): return self.op('*', a, b)


class _Jet:
    # v, gradient[3], directional derivative, directional gradient[3].
    def __init__(self, e, order, value, d=None, t=0.0, h=None):
        self.e, self.order, self.v = e, order, value
        self.d = d if d is not None else [0.0]*3
        self.t = t
        self.h = h if h is not None else [0.0]*3

    def binary(self, op, b):
        e, order = self.e, self.order
        v = e.op(op, self.v, b.v)
        d, t, h = [0.0]*3, 0.0, [0.0]*3
        if op in ('+', '-'):
            if order:
                d = [e.op(op, a, c) for a,c in zip(self.d,b.d)]
            if order == 2:
                t = e.op(op, self.t, b.t)
                h = [e.op(op,a,c) for a,c in zip(self.h,b.h)]
        elif op == '*':
            if order:
                d = [e.add(e.mul(a,b.v),e.mul(self.v,c)) for a,c in zip(self.d,b.d)]
            if order == 2:
                t = e.add(e.mul(self.t,b.v),e.mul(self.v,b.t))
                h = [e.add(e.mul(self.h[i],b.v),e.mul(self.d[i],b.t),
                           e.mul(self.t,b.d[i]),e.mul(self.v,b.h[i])) for i in range(3)]
        elif op == '/':
            if any(b.d) or b.t != 0 or any(b.h):
                raise ValueError('only constant polynomial denominators are supported')
            if order: d = [e.op('/',a,b.v) for a in self.d]
            if order == 2:
                t = e.op('/',self.t,b.v)
                h = [e.op('/',a,b.v) for a in self.h]
        else:
            raise ValueError(op)
        return _Jet(e,order,v,d,t,h)


def _parse_expr(node, env, e, order):
    const = lambda v: _Jet(e, order, v)
    if isinstance(node, ast.Constant): return const(node.value)
    if isinstance(node, ast.Name): return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return const(-1.0).binary('*',_parse_expr(node.operand,env,e,order))
    if isinstance(node, ast.BinOp):
        a = _parse_expr(node.left,env,e,order)
        b = _parse_expr(node.right,env,e,order)
        ops = {ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}
        return a.binary(ops[type(node.op)],b)
    if isinstance(node, ast.Call):
        func = ast.unparse(node.func)
        if func == 'math.sqrt':
            arg = _parse_expr(node.args[0],env,e,order).v
            if not isinstance(arg,(int,float)): raise ValueError('nonconstant sqrt')
            return const(math.sqrt(arg))
        if func == 'torch.ones_like': return const(1.0)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='pow':
            a = _parse_expr(node.func.value,env,e,order)
            n = ast.literal_eval(node.args[0])
            if not isinstance(n,int) or n<0: raise ValueError('unsupported power')
            out = const(1.0)
            for _ in range(n): out = out.binary('*',a)
            return out
    raise ValueError('unsupported expression: '+ast.dump(node))


@functools.lru_cache(None)
def _required_assignments(ls):
    table = {name:ast.parse(expr,mode='eval').body for name,expr in _POLYNOMIALS}
    needed = set()
    def visit(name):
        if name in needed or name not in table: return
        needed.add(name)
        for node in ast.walk(table[name]):
            if isinstance(node,ast.Name): visit(node.id)
    for l in ls:
        for m in range(2*l+1): visit(f'sh_{l}_{m}')
    return tuple((name,table[name]) for name,_ in _POLYNOMIALS if name in needed)


@functools.lru_cache(None)
def _kernel_source(ls, normalize, normalization, mode, need_x=True, need_g=True):
    """Generate straight-line Triton; no runtime eval, AD, or coefficient loads."""
    order = {'forward':0,'backward':1,'double_backward':2}[mode]
    if order == 2 and not need_x: order = 1
    dim = sum(2*l+1 for l in ls)
    sig = 'X, G, V, O, DX, DG, N, FP64: tl.constexpr, BLOCK: tl.constexpr'
    lines = [f'def spherical_harmonics_{mode}_kernel({sig}):',
             '    row = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)',
             '    mask = row < N',
             '    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32']
    body = []
    for i in range(3):
        body.append(f'r{i} = tl.load(X + row * 3 + {i}, mask, other=0).to(dtype)')
        if mode == 'double_backward':
            body.append(f'v{i} = tl.load(V + row * 3 + {i}, mask, other=0).to(dtype)')
    if normalize:
        body += ['radius = tl.sqrt(r0*r0 + r1*r1 + r2*r2)',
                 'denom = tl.maximum(radius, 1.0e-12)',
                 'inv = 1.0 / denom',
                 'active = radius >= 1.0e-12']
        for i in range(3): body.append(f'u{i} = r{i} / denom')
        if order:
            for a in range(3):
                for i in range(3):
                    body.append(f'j{a}{i} = ({float(a==i)} - tl.where(active, u{a}*u{i}, 0.0)) * inv')
        if order == 2:
            body.append('uv = u0*v0 + u1*v1 + u2*v2')
            for a in range(3):
                body.append(f't{a} = (v{a} - tl.where(active, u{a}*uv, 0.0)) * inv')
                for i in range(3):
                    body.append(f'h{a}{i} = tl.where(active, (-v{a}*u{i} - u{a}*v{i} - {float(a==i)}*uv + 3.0*u{a}*u{i}*uv) * inv * inv, 0.0)')
    e = _Emitter()
    env = {}
    for a,name in enumerate(('x','y','z')):
        d = [f'j{a}{i}' if normalize else float(a==i) for i in range(3)]
        h = [f'h{a}{i}' if normalize else 0.0 for i in range(3)]
        t = f't{a}' if normalize else f'v{a}'
        env[name] = _Jet(e,order,f'u{a}' if normalize else f'r{a}',d,t,h)
    accum = [0.0]*3
    cursor, emitted = 0, 0
    # Emit stores immediately after a requested polynomial is available.
    # Repeated degrees have distinct offsets; their VJP contributions sum.
    offsets = {}
    for l in ls:
        for m in range(2*l+1):
            offsets.setdefault(f'sh_{l}_{m}',[]).append((cursor,l))
            cursor += 1
    for name,node in _required_assignments(ls):
        jet = env[name] = _parse_expr(node,env,e,order)
        body.extend(e.lines[emitted:]); emitted = len(e.lines)
        for col,l in offsets.get(name,[]):
            divisor = math.sqrt(4*math.pi) if normalization=='integral' else math.sqrt(2*l+1) if normalization=='norm' else 1.0
            if mode == 'forward':
                body.append(f'tl.store(O + row * {dim} + {col}, {jet.v} / {divisor}, mask)')
            elif mode == 'backward':
                body.append(f'g{col} = tl.load(G + row * {dim} + {col}, mask, other=0).to(dtype) / {divisor}')
                accum = [e.add(accum[i],e.mul(f'g{col}',jet.d[i])) for i in range(3)]
            else:
                if need_x:
                    body.append(f'g{col} = tl.load(G + row * {dim} + {col}, mask, other=0).to(dtype) / {divisor}')
                    accum = [e.add(accum[i],e.mul(f'g{col}',jet.h[i])) for i in range(3)]
                if need_g:
                    tangent = e.add(*(e.mul(jet.d[i],f'v{i}') for i in range(3)))
                    body.extend(e.lines[emitted:]); emitted = len(e.lines)
                    body.append(f'tl.store(DG + row * {dim} + {col}, {tangent} / {divisor}, mask)')
            body.extend(e.lines[emitted:]); emitted = len(e.lines)
    if mode=='backward' or (mode=='double_backward' and need_x):
        for i in range(3): body.append(f'tl.store(DX + row * 3 + {i}, {accum[i]}, mask)')
    lines.extend('    '+line for line in body)
    return '\n'.join(lines)+'\n'


@functools.lru_cache(None)
def _get_kernel(ls, normalize, normalization, mode, need_x=True, need_g=True):
    if triton is None: raise ImportError('Install triton to use GPU spherical harmonics')
    source = _kernel_source(ls,normalize,normalization,mode,need_x,need_g)
    digest = hashlib.sha256(source.encode()).hexdigest()
    filename = f'<fasteq_sh_{digest}>'
    # inspect.getsource used by triton.jit requires source in linecache.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {'tl':tl,'__name__':__name__}
    exec(compile(source,filename,'exec'),namespace)
    return triton.jit(namespace[f'spherical_harmonics_{mode}_kernel'])


def _launch(x, g, v, out, dx, dg, ls, normalize, normalization, mode,
            need_x=True, need_g=True):
    n = x.numel()//3
    if not n: return
    kernel = _get_kernel(ls,normalize,normalization,mode,need_x,need_g)
    # Tensors supplied to these entry points are contiguous, including G/V.
    block = 64 if max(ls) >= 6 else 128
    with torch.cuda.device(x.device):
        kernel[(triton.cdiv(n,block),)](
            x,g,v,out,dx,dg,n,x.dtype==torch.float64,block,
            num_warps=4,enable_fp_fusion=False)


class _SHBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grad_out, ls, normalize, normalization):
        ctx.meta = (ls,normalize,normalization)
        ctx.save_for_backward(x,grad_out)
        gx = torch.empty_like(x)
        g = grad_out.contiguous()
        _launch(x,g,x,gx,gx,gx,*ctx.meta,'backward')
        return gx

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grad_x):
        x,g = ctx.saved_tensors
        need_x,need_g = ctx.needs_input_grad[:2]
        dx = torch.empty_like(x) if need_x else None
        dg = torch.empty_like(g, memory_format=torch.contiguous_format) if need_g else None
        if need_x or need_g:
            _launch(x,g.contiguous(),grad_grad_x.contiguous(),x,
                    dx if need_x else x, dg if need_g else x,
                    *ctx.meta,'double_backward',need_x,need_g)
        # Order follows forward inputs (x, grad_out, ls, normalize, normalization).
        return dx,dg,None,None,None


class _SH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ls, normalize, normalization):
        ctx.meta = (ls,normalize,normalization)
        ctx.save_for_backward(x)
        y = x.new_empty(x.shape[:-1]+(sum(2*l+1 for l in ls),))
        _launch(x,x,x,y,x,x,*ctx.meta,'forward')
        return y

    @staticmethod
    def backward(ctx, grad_out):
        if not ctx.needs_input_grad[0]: return None,None,None,None
        (x,) = ctx.saved_tensors
        gx = _SHBackward.apply(x,grad_out,*ctx.meta)
        return gx,None,None,None


@compile_mode('script')
class SphericalHarmonics(torch.nn.Module):
    normalize: bool
    normalization: str
    _ls_list: List[int]
    _lmax: int
    _is_range_lmax: bool
    _prof_str: str

    def __init__(self, irreps_out: Union[int, List[int], str, Irreps], normalize: bool, normalization: str='integral', irreps_in: Any=None) -> None:
        super().__init__()
        self.normalize = normalize
        self.normalization = normalization
        assert normalization in ['integral', 'component', 'norm']
        if isinstance(irreps_out, str):
            irreps_out = Irreps(irreps_out)
        if isinstance(irreps_out, Irreps) and irreps_in is None:
            for mul, (l, p) in irreps_out:
                if l % 2 == 1 and p == 1:
                    irreps_in = Irreps('1e')
        if irreps_in is None:
            irreps_in = Irreps('1o')
        irreps_in = Irreps(irreps_in)
        if irreps_in not in (Irreps('1x1o'), Irreps('1x1e')):
            raise ValueError(f'irreps_in for SphericalHarmonics must be either a vector (`1x1o`) or a pseudovector (`1x1e`), not `{irreps_in}`')
        self.irreps_in = irreps_in
        input_p = irreps_in[0].ir.p
        if isinstance(irreps_out, Irreps):
            ls = []
            for mul, (l, p) in irreps_out:
                if p != input_p ** l:
                    raise ValueError(f'irreps_out `{irreps_out}` passed to SphericalHarmonics asked for an output of l = {l} with parity p = {p}, which is inconsistent with the input parity {input_p} — the output parity should have been p = {input_p ** l}')
                ls.extend([l] * mul)
        elif isinstance(irreps_out, int):
            ls = [irreps_out]
        else:
            ls = list(irreps_out)
        if not ls or any((type(l) is not int or l < 0 for l in ls)):
            raise ValueError('degrees must be a nonempty sequence of nonnegative integers')
        irreps_out = Irreps([(1, (l, input_p ** l)) for l in ls]).simplify()
        self.irreps_out = irreps_out
        self._ls_list = ls
        self._lmax = max(ls)
        self._is_range_lmax = ls == list(range(max(ls) + 1))
        self._prof_str = f'spherical_harmonics({ls})'
        _lmax = 12
        if self._lmax > _lmax:
            raise NotImplementedError(f'spherical_harmonics maximum l implemented is {_lmax}, send us an email to ask for more')

    def _reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=-1)
        sh = _spherical_harmonics(self._lmax,x[...,0],x[...,1],x[...,2])
        if not self._is_range_lmax:
            sh = torch.cat([sh[...,l*l:(l+1)*(l+1)] for l in self._ls_list],dim=-1)
        if self.normalization == 'integral':
            sh = sh / math.sqrt(4*math.pi)
        elif self.normalization == 'norm':
            scales = torch.cat([math.sqrt(2*l+1)*torch.ones(2*l+1,dtype=sh.dtype,device=sh.device) for l in self._ls_list])
            sh = sh / scales
        return sh

    @torch.jit.unused
    def _eager_forward(self, x):
        if x.ndim < 1 or x.shape[-1]!=3:
            raise ValueError(f'expected input shape (..., 3), got {tuple(x.shape)}')
        if not x.is_floating_point(): raise TypeError('expected floating point input')
        if x.device.type != 'cuda': return self._reference_forward(x)
        if x.dtype not in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
            raise TypeError(f'unsupported dtype: {x.dtype}')
        # Outside Function.forward: preserve AD connectivity for noncontiguous x.
        return _SH.apply(x.contiguous(),tuple(self._ls_list),self.normalize,self.normalization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            return self._reference_forward(x)
        else:
            return self._eager_forward(x)


def spherical_harmonics(l: Union[int,List[int],str,Irreps],x: torch.Tensor,
                        normalize: bool,normalization: str='integral'):
    return SphericalHarmonics(l,normalize,normalization)(x)


# Exact embedded polynomial reference from the user attachment.
def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return torch.stack([sh_0_0], dim=-1)
    sh_1_0 = math.sqrt(3) * x
    sh_1_1 = math.sqrt(3) * y
    sh_1_2 = math.sqrt(3) * z
    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)
    sh_2_0 = math.sqrt(15) * x * z
    sh_2_1 = math.sqrt(15) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = math.sqrt(5) * (y2 - 1 / 2 * x2z2)
    sh_2_3 = math.sqrt(15) * y * z
    sh_2_4 = 1 / 2 * math.sqrt(15) * (z.pow(2) - x.pow(2))
    if lmax == 2:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)
    sh_3_0 = 1 / 6 * math.sqrt(42) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = math.sqrt(7) * sh_2_0 * y
    sh_3_2 = 1 / 8 * math.sqrt(168) * (4.0 * y2 - x2z2) * x
    sh_3_3 = 1 / 2 * math.sqrt(7) * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = 1 / 8 * math.sqrt(168) * z * (4.0 * y2 - x2z2)
    sh_3_5 = math.sqrt(7) * sh_2_4 * y
    sh_3_6 = 1 / 6 * math.sqrt(42) * (sh_2_4 * z - sh_2_0 * x)
    if lmax == 3:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4, sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6], dim=-1)
    sh_4_0 = 3 / 4 * math.sqrt(2) * (sh_3_0 * z + sh_3_6 * x)
    sh_4_1 = 3 / 4 * sh_3_0 * y + 3 / 8 * math.sqrt(6) * sh_3_1 * z + 3 / 8 * math.sqrt(6) * sh_3_5 * x
    sh_4_2 = -3 / 56 * math.sqrt(14) * sh_3_0 * z + 3 / 14 * math.sqrt(21) * sh_3_1 * y + 3 / 56 * math.sqrt(210) * sh_3_2 * z + 3 / 56 * math.sqrt(210) * sh_3_4 * x + 3 / 56 * math.sqrt(14) * sh_3_6 * x
    sh_4_3 = -3 / 56 * math.sqrt(42) * sh_3_1 * z + 3 / 28 * math.sqrt(105) * sh_3_2 * y + 3 / 28 * math.sqrt(70) * sh_3_3 * x + 3 / 56 * math.sqrt(42) * sh_3_5 * x
    sh_4_4 = -3 / 28 * math.sqrt(42) * sh_3_2 * x + 3 / 7 * math.sqrt(7) * sh_3_3 * y - 3 / 28 * math.sqrt(42) * sh_3_4 * z
    sh_4_5 = -3 / 56 * math.sqrt(42) * sh_3_1 * x + 3 / 28 * math.sqrt(70) * sh_3_3 * z + 3 / 28 * math.sqrt(105) * sh_3_4 * y - 3 / 56 * math.sqrt(42) * sh_3_5 * z
    sh_4_6 = -3 / 56 * math.sqrt(14) * sh_3_0 * x - 3 / 56 * math.sqrt(210) * sh_3_2 * x + 3 / 56 * math.sqrt(210) * sh_3_4 * z + 3 / 14 * math.sqrt(21) * sh_3_5 * y - 3 / 56 * math.sqrt(14) * sh_3_6 * z
    sh_4_7 = -3 / 8 * math.sqrt(6) * sh_3_1 * x + 3 / 8 * math.sqrt(6) * sh_3_5 * z + 3 / 4 * sh_3_6 * y
    sh_4_8 = 3 / 4 * math.sqrt(2) * (-sh_3_0 * x + sh_3_6 * z)
    if lmax == 4:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4, sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6, sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8], dim=-1)
    sh_5_0 = 1 / 10 * math.sqrt(110) * (sh_4_0 * z + sh_4_8 * x)
    sh_5_1 = 1 / 5 * math.sqrt(11) * sh_4_0 * y + 1 / 5 * math.sqrt(22) * sh_4_1 * z + 1 / 5 * math.sqrt(22) * sh_4_7 * x
    sh_5_2 = -1 / 30 * math.sqrt(22) * sh_4_0 * z + 4 / 15 * math.sqrt(11) * sh_4_1 * y + 1 / 15 * math.sqrt(154) * sh_4_2 * z + 1 / 15 * math.sqrt(154) * sh_4_6 * x + 1 / 30 * math.sqrt(22) * sh_4_8 * x
    sh_5_3 = -1 / 30 * math.sqrt(66) * sh_4_1 * z + 1 / 15 * math.sqrt(231) * sh_4_2 * y + 1 / 30 * math.sqrt(462) * sh_4_3 * z + 1 / 30 * math.sqrt(462) * sh_4_5 * x + 1 / 30 * math.sqrt(66) * sh_4_7 * x
    sh_5_4 = -1 / 15 * math.sqrt(33) * sh_4_2 * z + 2 / 15 * math.sqrt(66) * sh_4_3 * y + 1 / 15 * math.sqrt(165) * sh_4_4 * x + 1 / 15 * math.sqrt(33) * sh_4_6 * x
    sh_5_5 = -1 / 15 * math.sqrt(110) * sh_4_3 * x + 1 / 3 * math.sqrt(11) * sh_4_4 * y - 1 / 15 * math.sqrt(110) * sh_4_5 * z
    sh_5_6 = -1 / 15 * math.sqrt(33) * sh_4_2 * x + 1 / 15 * math.sqrt(165) * sh_4_4 * z + 2 / 15 * math.sqrt(66) * sh_4_5 * y - 1 / 15 * math.sqrt(33) * sh_4_6 * z
    sh_5_7 = -1 / 30 * math.sqrt(66) * sh_4_1 * x - 1 / 30 * math.sqrt(462) * sh_4_3 * x + 1 / 30 * math.sqrt(462) * sh_4_5 * z + 1 / 15 * math.sqrt(231) * sh_4_6 * y - 1 / 30 * math.sqrt(66) * sh_4_7 * z
    sh_5_8 = -1 / 30 * math.sqrt(22) * sh_4_0 * x - 1 / 15 * math.sqrt(154) * sh_4_2 * x + 1 / 15 * math.sqrt(154) * sh_4_6 * z + 4 / 15 * math.sqrt(11) * sh_4_7 * y - 1 / 30 * math.sqrt(22) * sh_4_8 * z
    sh_5_9 = -1 / 5 * math.sqrt(22) * sh_4_1 * x + 1 / 5 * math.sqrt(22) * sh_4_7 * z + 1 / 5 * math.sqrt(11) * sh_4_8 * y
    sh_5_10 = 1 / 10 * math.sqrt(110) * (-sh_4_0 * x + sh_4_8 * z)
    if lmax == 5:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4, sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6, sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8, sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10], dim=-1)
    sh_6_0 = 1 / 6 * math.sqrt(39) * (sh_5_0 * z + sh_5_10 * x)
    sh_6_1 = 1 / 6 * math.sqrt(13) * sh_5_0 * y + 1 / 12 * math.sqrt(130) * sh_5_1 * z + 1 / 12 * math.sqrt(130) * sh_5_9 * x
    sh_6_2 = -1 / 132 * math.sqrt(286) * sh_5_0 * z + 1 / 33 * math.sqrt(715) * sh_5_1 * y + 1 / 132 * math.sqrt(286) * sh_5_10 * x + 1 / 44 * math.sqrt(1430) * sh_5_2 * z + 1 / 44 * math.sqrt(1430) * sh_5_8 * x
    sh_6_3 = -1 / 132 * math.sqrt(858) * sh_5_1 * z + 1 / 22 * math.sqrt(429) * sh_5_2 * y + 1 / 22 * math.sqrt(286) * sh_5_3 * z + 1 / 22 * math.sqrt(286) * sh_5_7 * x + 1 / 132 * math.sqrt(858) * sh_5_9 * x
    sh_6_4 = -1 / 66 * math.sqrt(429) * sh_5_2 * z + 2 / 33 * math.sqrt(286) * sh_5_3 * y + 1 / 66 * math.sqrt(2002) * sh_5_4 * z + 1 / 66 * math.sqrt(2002) * sh_5_6 * x + 1 / 66 * math.sqrt(429) * sh_5_8 * x
    sh_6_5 = -1 / 66 * math.sqrt(715) * sh_5_3 * z + 1 / 66 * math.sqrt(5005) * sh_5_4 * y + 1 / 66 * math.sqrt(3003) * sh_5_5 * x + 1 / 66 * math.sqrt(715) * sh_5_7 * x
    sh_6_6 = -1 / 66 * math.sqrt(2145) * sh_5_4 * x + 1 / 11 * math.sqrt(143) * sh_5_5 * y - 1 / 66 * math.sqrt(2145) * sh_5_6 * z
    sh_6_7 = -1 / 66 * math.sqrt(715) * sh_5_3 * x + 1 / 66 * math.sqrt(3003) * sh_5_5 * z + 1 / 66 * math.sqrt(5005) * sh_5_6 * y - 1 / 66 * math.sqrt(715) * sh_5_7 * z
    sh_6_8 = -1 / 66 * math.sqrt(429) * sh_5_2 * x - 1 / 66 * math.sqrt(2002) * sh_5_4 * x + 1 / 66 * math.sqrt(2002) * sh_5_6 * z + 2 / 33 * math.sqrt(286) * sh_5_7 * y - 1 / 66 * math.sqrt(429) * sh_5_8 * z
    sh_6_9 = -1 / 132 * math.sqrt(858) * sh_5_1 * x - 1 / 22 * math.sqrt(286) * sh_5_3 * x + 1 / 22 * math.sqrt(286) * sh_5_7 * z + 1 / 22 * math.sqrt(429) * sh_5_8 * y - 1 / 132 * math.sqrt(858) * sh_5_9 * z
    sh_6_10 = -1 / 132 * math.sqrt(286) * sh_5_0 * x - 1 / 132 * math.sqrt(286) * sh_5_10 * z - 1 / 44 * math.sqrt(1430) * sh_5_2 * x + 1 / 44 * math.sqrt(1430) * sh_5_8 * z + 1 / 33 * math.sqrt(715) * sh_5_9 * y
    sh_6_11 = -1 / 12 * math.sqrt(130) * sh_5_1 * x + 1 / 6 * math.sqrt(13) * sh_5_10 * y + 1 / 12 * math.sqrt(130) * sh_5_9 * z
    sh_6_12 = 1 / 6 * math.sqrt(39) * (-sh_5_0 * x + sh_5_10 * z)
    if lmax == 6:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4, sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6, sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8, sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10, sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12], dim=-1)
    sh_7_0 = 1 / 14 * math.sqrt(210) * (sh_6_0 * z + sh_6_12 * x)
    sh_7_1 = 1 / 7 * math.sqrt(15) * sh_6_0 * y + 3 / 7 * math.sqrt(5) * sh_6_1 * z + 3 / 7 * math.sqrt(5) * sh_6_11 * x
    sh_7_2 = -1 / 182 * math.sqrt(390) * sh_6_0 * z + 6 / 91 * math.sqrt(130) * sh_6_1 * y + 3 / 91 * math.sqrt(715) * sh_6_10 * x + 1 / 182 * math.sqrt(390) * sh_6_12 * x + 3 / 91 * math.sqrt(715) * sh_6_2 * z
    sh_7_3 = -3 / 182 * math.sqrt(130) * sh_6_1 * z + 3 / 182 * math.sqrt(130) * sh_6_11 * x + 3 / 91 * math.sqrt(715) * sh_6_2 * y + 5 / 182 * math.sqrt(858) * sh_6_3 * z + 5 / 182 * math.sqrt(858) * sh_6_9 * x
    sh_7_4 = 3 / 91 * math.sqrt(65) * sh_6_10 * x - 3 / 91 * math.sqrt(65) * sh_6_2 * z + 10 / 91 * math.sqrt(78) * sh_6_3 * y + 15 / 182 * math.sqrt(78) * sh_6_4 * z + 15 / 182 * math.sqrt(78) * sh_6_8 * x
    sh_7_5 = -5 / 91 * math.sqrt(39) * sh_6_3 * z + 15 / 91 * math.sqrt(39) * sh_6_4 * y + 3 / 91 * math.sqrt(390) * sh_6_5 * z + 3 / 91 * math.sqrt(390) * sh_6_7 * x + 5 / 91 * math.sqrt(39) * sh_6_9 * x
    sh_7_6 = -15 / 182 * math.sqrt(26) * sh_6_4 * z + 12 / 91 * math.sqrt(65) * sh_6_5 * y + 2 / 91 * math.sqrt(1365) * sh_6_6 * x + 15 / 182 * math.sqrt(26) * sh_6_8 * x
    sh_7_7 = -3 / 91 * math.sqrt(455) * sh_6_5 * x + 1 / 13 * math.sqrt(195) * sh_6_6 * y - 3 / 91 * math.sqrt(455) * sh_6_7 * z
    sh_7_8 = -15 / 182 * math.sqrt(26) * sh_6_4 * x + 2 / 91 * math.sqrt(1365) * sh_6_6 * z + 12 / 91 * math.sqrt(65) * sh_6_7 * y - 15 / 182 * math.sqrt(26) * sh_6_8 * z
    sh_7_9 = -5 / 91 * math.sqrt(39) * sh_6_3 * x - 3 / 91 * math.sqrt(390) * sh_6_5 * x + 3 / 91 * math.sqrt(390) * sh_6_7 * z + 15 / 91 * math.sqrt(39) * sh_6_8 * y - 5 / 91 * math.sqrt(39) * sh_6_9 * z
    sh_7_10 = -3 / 91 * math.sqrt(65) * sh_6_10 * z - 3 / 91 * math.sqrt(65) * sh_6_2 * x - 15 / 182 * math.sqrt(78) * sh_6_4 * x + 15 / 182 * math.sqrt(78) * sh_6_8 * z + 10 / 91 * math.sqrt(78) * sh_6_9 * y
    sh_7_11 = -3 / 182 * math.sqrt(130) * sh_6_1 * x + 3 / 91 * math.sqrt(715) * sh_6_10 * y - 3 / 182 * math.sqrt(130) * sh_6_11 * z - 5 / 182 * math.sqrt(858) * sh_6_3 * x + 5 / 182 * math.sqrt(858) * sh_6_9 * z
    sh_7_12 = -1 / 182 * math.sqrt(390) * sh_6_0 * x + 3 / 91 * math.sqrt(715) * sh_6_10 * z + 6 / 91 * math.sqrt(130) * sh_6_11 * y - 1 / 182 * math.sqrt(390) * sh_6_12 * z - 3 / 91 * math.sqrt(715) * sh_6_2 * x
    sh_7_13 = -3 / 7 * math.sqrt(5) * sh_6_1 * x + 3 / 7 * math.sqrt(5) * sh_6_11 * z + 1 / 7 * math.sqrt(15) * sh_6_12 * y
    sh_7_14 = 1 / 14 * math.sqrt(210) * (-sh_6_0 * x + sh_6_12 * z)
    if lmax == 7:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4, sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6, sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8, sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10, sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12, sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14], dim=-1)
    sh_8_0 = 1 / 4 * math.sqrt(17) * (sh_7_0 * z + sh_7_14 * x)
    sh_8_1 = 1 / 8 * math.sqrt(17) * sh_7_0 * y + 1 / 16 * math.sqrt(238) * sh_7_1 * z + 1 / 16 * math.sqrt(238) * sh_7_13 * x
    sh_8_2 = -1 / 240 * math.sqrt(510) * sh_7_0 * z + 1 / 60 * math.sqrt(1785) * sh_7_1 * y + 1 / 240 * math.sqrt(46410) * sh_7_12 * x + 1 / 240 * math.sqrt(510) * sh_7_14 * x + 1 / 240 * math.sqrt(46410) * sh_7_2 * z
    sh_8_3 = 1 / 80 * math.sqrt(2) * (-math.sqrt(85) * sh_7_1 * z + math.sqrt(2210) * sh_7_11 * x + math.sqrt(85) * sh_7_13 * x + math.sqrt(2210) * sh_7_2 * y + math.sqrt(2210) * sh_7_3 * z)
    sh_8_4 = 1 / 40 * math.sqrt(935) * sh_7_10 * x + 1 / 40 * math.sqrt(85) * sh_7_12 * x - 1 / 40 * math.sqrt(85) * sh_7_2 * z + 1 / 10 * math.sqrt(85) * sh_7_3 * y + 1 / 40 * math.sqrt(935) * sh_7_4 * z
    sh_8_5 = 1 / 48 * math.sqrt(2) * (math.sqrt(102) * sh_7_11 * x - math.sqrt(102) * sh_7_3 * z + math.sqrt(1122) * sh_7_4 * y + math.sqrt(561) * sh_7_5 * z + math.sqrt(561) * sh_7_9 * x)
    sh_8_6 = 1 / 16 * math.sqrt(34) * sh_7_10 * x - 1 / 16 * math.sqrt(34) * sh_7_4 * z + 1 / 4 * math.sqrt(17) * sh_7_5 * y + 1 / 16 * math.sqrt(102) * sh_7_6 * z + 1 / 16 * math.sqrt(102) * sh_7_8 * x
    sh_8_7 = -1 / 80 * math.sqrt(1190) * sh_7_5 * z + 1 / 40 * math.sqrt(1785) * sh_7_6 * y + 1 / 20 * math.sqrt(255) * sh_7_7 * x + 1 / 80 * math.sqrt(1190) * sh_7_9 * x
    sh_8_8 = -1 / 60 * math.sqrt(1785) * sh_7_6 * x + 1 / 15 * math.sqrt(255) * sh_7_7 * y - 1 / 60 * math.sqrt(1785) * sh_7_8 * z
    sh_8_9 = -1 / 80 * math.sqrt(1190) * sh_7_5 * x + 1 / 20 * math.sqrt(255) * sh_7_7 * z + 1 / 40 * math.sqrt(1785) * sh_7_8 * y - 1 / 80 * math.sqrt(1190) * sh_7_9 * z
    sh_8_10 = -1 / 16 * math.sqrt(34) * sh_7_10 * z - 1 / 16 * math.sqrt(34) * sh_7_4 * x - 1 / 16 * math.sqrt(102) * sh_7_6 * x + 1 / 16 * math.sqrt(102) * sh_7_8 * z + 1 / 4 * math.sqrt(17) * sh_7_9 * y
    sh_8_11 = 1 / 48 * math.sqrt(2) * (math.sqrt(1122) * sh_7_10 * y - math.sqrt(102) * sh_7_11 * z - math.sqrt(102) * sh_7_3 * x - math.sqrt(561) * sh_7_5 * x + math.sqrt(561) * sh_7_9 * z)
    sh_8_12 = 1 / 40 * math.sqrt(935) * sh_7_10 * z + 1 / 10 * math.sqrt(85) * sh_7_11 * y - 1 / 40 * math.sqrt(85) * sh_7_12 * z - 1 / 40 * math.sqrt(85) * sh_7_2 * x - 1 / 40 * math.sqrt(935) * sh_7_4 * x
    sh_8_13 = 1 / 80 * math.sqrt(2) * (-math.sqrt(85) * sh_7_1 * x + math.sqrt(2210) * sh_7_11 * z + math.sqrt(2210) * sh_7_12 * y - math.sqrt(85) * sh_7_13 * z - math.sqrt(2210) * sh_7_3 * x)
    sh_8_14 = -1 / 240 * math.sqrt(510) * sh_7_0 * x + 1 / 240 * math.sqrt(46410) * sh_7_12 * z + 1 / 60 * math.sqrt(1785) * sh_7_13 * y - 1 / 240 * math.sqrt(510) * sh_7_14 * z - 1 / 240 * math.sqrt(46410) * sh_7_2 * x
    sh_8_15 = -1 / 16 * math.sqrt(238) * sh_7_1 * x + 1 / 16 * math.sqrt(238) * sh_7_13 * z + 1 / 8 * math.sqrt(17) * sh_7_14 * y
    sh_8_16 = 1 / 4 * math.sqrt(17) * (-sh_7_0 * x + sh_7_14 * z)
    if lmax == 8:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4, sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6, sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8, sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10, sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12, sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14, sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16], dim=-1)
    sh_9_0 = 1 / 6 * math.sqrt(38) * (sh_8_0 * z + sh_8_16 * x)
    sh_9_1 = 1 / 9 * math.sqrt(19) * (sh_8_0 * y + 2 * sh_8_1 * z + 2 * sh_8_15 * x)
    sh_9_2 = -1 / 306 * math.sqrt(646) * sh_8_0 * z + 4 / 153 * math.sqrt(646) * sh_8_1 * y + 2 / 153 * math.sqrt(4845) * sh_8_14 * x + 1 / 306 * math.sqrt(646) * sh_8_16 * x + 2 / 153 * math.sqrt(4845) * sh_8_2 * z
    sh_9_3 = -1 / 306 * math.sqrt(1938) * sh_8_1 * z + 1 / 306 * math.sqrt(67830) * sh_8_13 * x + 1 / 306 * math.sqrt(1938) * sh_8_15 * x + 1 / 51 * math.sqrt(1615) * sh_8_2 * y + 1 / 306 * math.sqrt(67830) * sh_8_3 * z
    sh_9_4 = 1 / 306 * math.sqrt(58786) * sh_8_12 * x + 1 / 153 * math.sqrt(969) * sh_8_14 * x - 1 / 153 * math.sqrt(969) * sh_8_2 * z + 2 / 153 * math.sqrt(4522) * sh_8_3 * y + 1 / 306 * math.sqrt(58786) * sh_8_4 * z
    sh_9_5 = 1 / 153 * math.sqrt(12597) * sh_8_11 * x + 1 / 153 * math.sqrt(1615) * sh_8_13 * x - 1 / 153 * math.sqrt(1615) * sh_8_3 * z + 1 / 153 * math.sqrt(20995) * sh_8_4 * y + 1 / 153 * math.sqrt(12597) * sh_8_5 * z
    sh_9_6 = 1 / 153 * math.sqrt(10659) * sh_8_10 * x + 1 / 306 * math.sqrt(9690) * sh_8_12 * x - 1 / 306 * math.sqrt(9690) * sh_8_4 * z + 2 / 51 * math.sqrt(646) * sh_8_5 * y + 1 / 153 * math.sqrt(10659) * sh_8_6 * z
    sh_9_7 = 1 / 306 * math.sqrt(13566) * sh_8_11 * x - 1 / 306 * math.sqrt(13566) * sh_8_5 * z + 1 / 153 * math.sqrt(24871) * sh_8_6 * y + 1 / 306 * math.sqrt(35530) * sh_8_7 * z + 1 / 306 * math.sqrt(35530) * sh_8_9 * x
    sh_9_8 = 1 / 153 * math.sqrt(4522) * sh_8_10 * x - 1 / 153 * math.sqrt(4522) * sh_8_6 * z + 4 / 153 * math.sqrt(1615) * sh_8_7 * y + 1 / 51 * math.sqrt(1615) * sh_8_8 * x
    sh_9_9 = 1 / 51 * math.sqrt(323) * (-2 * sh_8_7 * x + 3 * sh_8_8 * y - 2 * sh_8_9 * z)
    sh_9_10 = -1 / 153 * math.sqrt(4522) * sh_8_10 * z - 1 / 153 * math.sqrt(4522) * sh_8_6 * x + 1 / 51 * math.sqrt(1615) * sh_8_8 * z + 4 / 153 * math.sqrt(1615) * sh_8_9 * y
    sh_9_11 = 1 / 153 * math.sqrt(24871) * sh_8_10 * y - 1 / 306 * math.sqrt(13566) * sh_8_11 * z - 1 / 306 * math.sqrt(13566) * sh_8_5 * x - 1 / 306 * math.sqrt(35530) * sh_8_7 * x + 1 / 306 * math.sqrt(35530) * sh_8_9 * z
    sh_9_12 = 1 / 153 * math.sqrt(10659) * sh_8_10 * z + 2 / 51 * math.sqrt(646) * sh_8_11 * y - 1 / 306 * math.sqrt(9690) * sh_8_12 * z - 1 / 306 * math.sqrt(9690) * sh_8_4 * x - 1 / 153 * math.sqrt(10659) * sh_8_6 * x
    sh_9_13 = 1 / 153 * math.sqrt(12597) * sh_8_11 * z + 1 / 153 * math.sqrt(20995) * sh_8_12 * y - 1 / 153 * math.sqrt(1615) * sh_8_13 * z - 1 / 153 * math.sqrt(1615) * sh_8_3 * x - 1 / 153 * math.sqrt(12597) * sh_8_5 * x
    sh_9_14 = 1 / 306 * math.sqrt(58786) * sh_8_12 * z + 2 / 153 * math.sqrt(4522) * sh_8_13 * y - 1 / 153 * math.sqrt(969) * sh_8_14 * z - 1 / 153 * math.sqrt(969) * sh_8_2 * x - 1 / 306 * math.sqrt(58786) * sh_8_4 * x
    sh_9_15 = -1 / 306 * math.sqrt(1938) * sh_8_1 * x + 1 / 306 * math.sqrt(67830) * sh_8_13 * z + 1 / 51 * math.sqrt(1615) * sh_8_14 * y - 1 / 306 * math.sqrt(1938) * sh_8_15 * z - 1 / 306 * math.sqrt(67830) * sh_8_3 * x
    sh_9_16 = -1 / 306 * math.sqrt(646) * sh_8_0 * x + 2 / 153 * math.sqrt(4845) * sh_8_14 * z + 4 / 153 * math.sqrt(646) * sh_8_15 * y - 1 / 306 * math.sqrt(646) * sh_8_16 * z - 2 / 153 * math.sqrt(4845) * sh_8_2 * x
    sh_9_17 = 1 / 9 * math.sqrt(19) * (-2 * sh_8_1 * x + 2 * sh_8_15 * z + sh_8_16 * y)
    sh_9_18 = 1 / 6 * math.sqrt(38) * (-sh_8_0 * x + sh_8_16 * z)
    if lmax == 9:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4, sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6, sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8, sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10, sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12, sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14, sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16, sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18], dim=-1)
    sh_10_0 = 1 / 10 * math.sqrt(105) * (sh_9_0 * z + sh_9_18 * x)
    sh_10_1 = 1 / 10 * math.sqrt(21) * sh_9_0 * y + 3 / 20 * math.sqrt(42) * sh_9_1 * z + 3 / 20 * math.sqrt(42) * sh_9_17 * x
    sh_10_2 = -1 / 380 * math.sqrt(798) * sh_9_0 * z + 3 / 95 * math.sqrt(399) * sh_9_1 * y + 3 / 380 * math.sqrt(13566) * sh_9_16 * x + 1 / 380 * math.sqrt(798) * sh_9_18 * x + 3 / 380 * math.sqrt(13566) * sh_9_2 * z
    sh_10_3 = -3 / 380 * math.sqrt(266) * sh_9_1 * z + 1 / 95 * math.sqrt(6783) * sh_9_15 * x + 3 / 380 * math.sqrt(266) * sh_9_17 * x + 3 / 190 * math.sqrt(2261) * sh_9_2 * y + 1 / 95 * math.sqrt(6783) * sh_9_3 * z
    sh_10_4 = 3 / 95 * math.sqrt(665) * sh_9_14 * x + 3 / 190 * math.sqrt(133) * sh_9_16 * x - 3 / 190 * math.sqrt(133) * sh_9_2 * z + 4 / 95 * math.sqrt(399) * sh_9_3 * y + 3 / 95 * math.sqrt(665) * sh_9_4 * z
    sh_10_5 = 21 / 380 * math.sqrt(190) * sh_9_13 * x + 1 / 190 * math.sqrt(1995) * sh_9_15 * x - 1 / 190 * math.sqrt(1995) * sh_9_3 * z + 3 / 38 * math.sqrt(133) * sh_9_4 * y + 21 / 380 * math.sqrt(190) * sh_9_5 * z
    sh_10_6 = 7 / 380 * math.sqrt(1482) * sh_9_12 * x + 3 / 380 * math.sqrt(1330) * sh_9_14 * x - 3 / 380 * math.sqrt(1330) * sh_9_4 * z + 21 / 95 * math.sqrt(19) * sh_9_5 * y + 7 / 380 * math.sqrt(1482) * sh_9_6 * z
    sh_10_7 = 3 / 190 * math.sqrt(1729) * sh_9_11 * x + 21 / 380 * math.sqrt(38) * sh_9_13 * x - 21 / 380 * math.sqrt(38) * sh_9_5 * z + 7 / 190 * math.sqrt(741) * sh_9_6 * y + 3 / 190 * math.sqrt(1729) * sh_9_7 * z
    sh_10_8 = 3 / 190 * math.sqrt(1463) * sh_9_10 * x + 7 / 190 * math.sqrt(114) * sh_9_12 * x - 7 / 190 * math.sqrt(114) * sh_9_6 * z + 6 / 95 * math.sqrt(266) * sh_9_7 * y + 3 / 190 * math.sqrt(1463) * sh_9_8 * z
    sh_10_9 = 3 / 190 * math.sqrt(798) * sh_9_11 * x - 3 / 190 * math.sqrt(798) * sh_9_7 * z + 3 / 190 * math.sqrt(4389) * sh_9_8 * y + 1 / 190 * math.sqrt(21945) * sh_9_9 * x
    sh_10_10 = -3 / 190 * math.sqrt(1995) * sh_9_10 * z - 3 / 190 * math.sqrt(1995) * sh_9_8 * x + 1 / 19 * math.sqrt(399) * sh_9_9 * y
    sh_10_11 = 3 / 190 * math.sqrt(4389) * sh_9_10 * y - 3 / 190 * math.sqrt(798) * sh_9_11 * z - 3 / 190 * math.sqrt(798) * sh_9_7 * x + 1 / 190 * math.sqrt(21945) * sh_9_9 * z
    sh_10_12 = 3 / 190 * math.sqrt(1463) * sh_9_10 * z + 6 / 95 * math.sqrt(266) * sh_9_11 * y - 7 / 190 * math.sqrt(114) * sh_9_12 * z - 7 / 190 * math.sqrt(114) * sh_9_6 * x - 3 / 190 * math.sqrt(1463) * sh_9_8 * x
    sh_10_13 = 3 / 190 * math.sqrt(1729) * sh_9_11 * z + 7 / 190 * math.sqrt(741) * sh_9_12 * y - 21 / 380 * math.sqrt(38) * sh_9_13 * z - 21 / 380 * math.sqrt(38) * sh_9_5 * x - 3 / 190 * math.sqrt(1729) * sh_9_7 * x
    sh_10_14 = 7 / 380 * math.sqrt(1482) * sh_9_12 * z + 21 / 95 * math.sqrt(19) * sh_9_13 * y - 3 / 380 * math.sqrt(1330) * sh_9_14 * z - 3 / 380 * math.sqrt(1330) * sh_9_4 * x - 7 / 380 * math.sqrt(1482) * sh_9_6 * x
    sh_10_15 = 21 / 380 * math.sqrt(190) * sh_9_13 * z + 3 / 38 * math.sqrt(133) * sh_9_14 * y - 1 / 190 * math.sqrt(1995) * sh_9_15 * z - 1 / 190 * math.sqrt(1995) * sh_9_3 * x - 21 / 380 * math.sqrt(190) * sh_9_5 * x
    sh_10_16 = 3 / 95 * math.sqrt(665) * sh_9_14 * z + 4 / 95 * math.sqrt(399) * sh_9_15 * y - 3 / 190 * math.sqrt(133) * sh_9_16 * z - 3 / 190 * math.sqrt(133) * sh_9_2 * x - 3 / 95 * math.sqrt(665) * sh_9_4 * x
    sh_10_17 = -3 / 380 * math.sqrt(266) * sh_9_1 * x + 1 / 95 * math.sqrt(6783) * sh_9_15 * z + 3 / 190 * math.sqrt(2261) * sh_9_16 * y - 3 / 380 * math.sqrt(266) * sh_9_17 * z - 1 / 95 * math.sqrt(6783) * sh_9_3 * x
    sh_10_18 = -1 / 380 * math.sqrt(798) * sh_9_0 * x + 3 / 380 * math.sqrt(13566) * sh_9_16 * z + 3 / 95 * math.sqrt(399) * sh_9_17 * y - 1 / 380 * math.sqrt(798) * sh_9_18 * z - 3 / 380 * math.sqrt(13566) * sh_9_2 * x
    sh_10_19 = -3 / 20 * math.sqrt(42) * sh_9_1 * x + 3 / 20 * math.sqrt(42) * sh_9_17 * z + 1 / 10 * math.sqrt(21) * sh_9_18 * y
    sh_10_20 = 1 / 10 * math.sqrt(105) * (-sh_9_0 * x + sh_9_18 * z)
    if lmax == 10:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4, sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6, sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8, sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10, sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12, sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14, sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16, sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18, sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20], dim=-1)
    sh_11_0 = 1 / 22 * math.sqrt(506) * (sh_10_0 * z + sh_10_20 * x)
    sh_11_1 = 1 / 11 * math.sqrt(23) * sh_10_0 * y + 1 / 11 * math.sqrt(115) * sh_10_1 * z + 1 / 11 * math.sqrt(115) * sh_10_19 * x
    sh_11_2 = -1 / 462 * math.sqrt(966) * sh_10_0 * z + 2 / 231 * math.sqrt(4830) * sh_10_1 * y + 1 / 231 * math.sqrt(45885) * sh_10_18 * x + 1 / 231 * math.sqrt(45885) * sh_10_2 * z + 1 / 462 * math.sqrt(966) * sh_10_20 * x
    sh_11_3 = -1 / 154 * math.sqrt(322) * sh_10_1 * z + 1 / 154 * math.sqrt(18354) * sh_10_17 * x + 1 / 154 * math.sqrt(322) * sh_10_19 * x + 1 / 77 * math.sqrt(3059) * sh_10_2 * y + 1 / 154 * math.sqrt(18354) * sh_10_3 * z
    sh_11_4 = 1 / 154 * math.sqrt(16422) * sh_10_16 * x + 1 / 77 * math.sqrt(161) * sh_10_18 * x - 1 / 77 * math.sqrt(161) * sh_10_2 * z + 2 / 77 * math.sqrt(966) * sh_10_3 * y + 1 / 154 * math.sqrt(16422) * sh_10_4 * z
    sh_11_5 = 2 / 231 * math.sqrt(8211) * sh_10_15 * x + 1 / 231 * math.sqrt(2415) * sh_10_17 * x - 1 / 231 * math.sqrt(2415) * sh_10_3 * z + 1 / 231 * math.sqrt(41055) * sh_10_4 * y + 2 / 231 * math.sqrt(8211) * sh_10_5 * z
    sh_11_6 = 2 / 77 * math.sqrt(805) * sh_10_14 * x + 1 / 154 * math.sqrt(1610) * sh_10_16 * x - 1 / 154 * math.sqrt(1610) * sh_10_4 * z + 4 / 77 * math.sqrt(322) * sh_10_5 * y + 2 / 77 * math.sqrt(805) * sh_10_6 * z
    sh_11_7 = 1 / 22 * math.sqrt(230) * sh_10_13 * x + 1 / 22 * math.sqrt(46) * sh_10_15 * x - 1 / 22 * math.sqrt(46) * sh_10_5 * z + 1 / 11 * math.sqrt(115) * sh_10_6 * y + 1 / 22 * math.sqrt(230) * sh_10_7 * z
    sh_11_8 = 1 / 66 * math.sqrt(1794) * sh_10_12 * x + 1 / 33 * math.sqrt(138) * sh_10_14 * x - 1 / 33 * math.sqrt(138) * sh_10_6 * z + 4 / 33 * math.sqrt(69) * sh_10_7 * y + 1 / 66 * math.sqrt(1794) * sh_10_8 * z
    sh_11_9 = 1 / 77 * math.sqrt(2093) * sh_10_11 * x + 1 / 77 * math.sqrt(966) * sh_10_13 * x - 1 / 77 * math.sqrt(966) * sh_10_7 * z + 1 / 77 * math.sqrt(6279) * sh_10_8 * y + 1 / 77 * math.sqrt(2093) * sh_10_9 * z
    sh_11_10 = 1 / 77 * math.sqrt(3542) * sh_10_10 * x + 1 / 154 * math.sqrt(4830) * sh_10_12 * x - 1 / 154 * math.sqrt(4830) * sh_10_8 * z + 2 / 77 * math.sqrt(1610) * sh_10_9 * y
    sh_11_11 = 1 / 21 * math.sqrt(483) * sh_10_10 * y - 1 / 231 * math.sqrt(26565) * sh_10_11 * z - 1 / 231 * math.sqrt(26565) * sh_10_9 * x
    sh_11_12 = 1 / 77 * math.sqrt(3542) * sh_10_10 * z + 2 / 77 * math.sqrt(1610) * sh_10_11 * y - 1 / 154 * math.sqrt(4830) * sh_10_12 * z - 1 / 154 * math.sqrt(4830) * sh_10_8 * x
    sh_11_13 = 1 / 77 * math.sqrt(2093) * sh_10_11 * z + 1 / 77 * math.sqrt(6279) * sh_10_12 * y - 1 / 77 * math.sqrt(966) * sh_10_13 * z - 1 / 77 * math.sqrt(966) * sh_10_7 * x - 1 / 77 * math.sqrt(2093) * sh_10_9 * x
    sh_11_14 = 1 / 66 * math.sqrt(1794) * sh_10_12 * z + 4 / 33 * math.sqrt(69) * sh_10_13 * y - 1 / 33 * math.sqrt(138) * sh_10_14 * z - 1 / 33 * math.sqrt(138) * sh_10_6 * x - 1 / 66 * math.sqrt(1794) * sh_10_8 * x
    sh_11_15 = 1 / 22 * math.sqrt(230) * sh_10_13 * z + 1 / 11 * math.sqrt(115) * sh_10_14 * y - 1 / 22 * math.sqrt(46) * sh_10_15 * z - 1 / 22 * math.sqrt(46) * sh_10_5 * x - 1 / 22 * math.sqrt(230) * sh_10_7 * x
    sh_11_16 = 2 / 77 * math.sqrt(805) * sh_10_14 * z + 4 / 77 * math.sqrt(322) * sh_10_15 * y - 1 / 154 * math.sqrt(1610) * sh_10_16 * z - 1 / 154 * math.sqrt(1610) * sh_10_4 * x - 2 / 77 * math.sqrt(805) * sh_10_6 * x
    sh_11_17 = 2 / 231 * math.sqrt(8211) * sh_10_15 * z + 1 / 231 * math.sqrt(41055) * sh_10_16 * y - 1 / 231 * math.sqrt(2415) * sh_10_17 * z - 1 / 231 * math.sqrt(2415) * sh_10_3 * x - 2 / 231 * math.sqrt(8211) * sh_10_5 * x
    sh_11_18 = 1 / 154 * math.sqrt(16422) * sh_10_16 * z + 2 / 77 * math.sqrt(966) * sh_10_17 * y - 1 / 77 * math.sqrt(161) * sh_10_18 * z - 1 / 77 * math.sqrt(161) * sh_10_2 * x - 1 / 154 * math.sqrt(16422) * sh_10_4 * x
    sh_11_19 = -1 / 154 * math.sqrt(322) * sh_10_1 * x + 1 / 154 * math.sqrt(18354) * sh_10_17 * z + 1 / 77 * math.sqrt(3059) * sh_10_18 * y - 1 / 154 * math.sqrt(322) * sh_10_19 * z - 1 / 154 * math.sqrt(18354) * sh_10_3 * x
    sh_11_20 = -1 / 462 * math.sqrt(966) * sh_10_0 * x + 1 / 231 * math.sqrt(45885) * sh_10_18 * z + 2 / 231 * math.sqrt(4830) * sh_10_19 * y - 1 / 231 * math.sqrt(45885) * sh_10_2 * x - 1 / 462 * math.sqrt(966) * sh_10_20 * z
    sh_11_21 = -1 / 11 * math.sqrt(115) * sh_10_1 * x + 1 / 11 * math.sqrt(115) * sh_10_19 * z + 1 / 11 * math.sqrt(23) * sh_10_20 * y
    sh_11_22 = 1 / 22 * math.sqrt(506) * (-sh_10_0 * x + sh_10_20 * z)
    if lmax == 11:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4, sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6, sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8, sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10, sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12, sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14, sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16, sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18, sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20, sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22], dim=-1)
    sh_12_0 = 5 / 12 * math.sqrt(6) * (sh_11_0 * z + sh_11_22 * x)
    sh_12_1 = 5 / 12 * sh_11_0 * y + 5 / 24 * math.sqrt(22) * sh_11_1 * z + 5 / 24 * math.sqrt(22) * sh_11_21 * x
    sh_12_2 = -5 / 552 * math.sqrt(46) * sh_11_0 * z + 5 / 138 * math.sqrt(253) * sh_11_1 * y + 5 / 552 * math.sqrt(10626) * sh_11_2 * z + 5 / 552 * math.sqrt(10626) * sh_11_20 * x + 5 / 552 * math.sqrt(46) * sh_11_22 * x
    sh_12_3 = -5 / 552 * math.sqrt(138) * sh_11_1 * z + 5 / 276 * math.sqrt(2415) * sh_11_19 * x + 5 / 92 * math.sqrt(161) * sh_11_2 * y + 5 / 552 * math.sqrt(138) * sh_11_21 * x + 5 / 276 * math.sqrt(2415) * sh_11_3 * z
    sh_12_4 = 5 / 276 * math.sqrt(2185) * sh_11_18 * x - 5 / 276 * math.sqrt(69) * sh_11_2 * z + 5 / 276 * math.sqrt(69) * sh_11_20 * x + 5 / 69 * math.sqrt(115) * sh_11_3 * y + 5 / 276 * math.sqrt(2185) * sh_11_4 * z
    sh_12_5 = 5 / 184 * math.sqrt(874) * sh_11_17 * x + 5 / 276 * math.sqrt(115) * sh_11_19 * x - 5 / 276 * math.sqrt(115) * sh_11_3 * z + 5 / 276 * math.sqrt(2185) * sh_11_4 * y + 5 / 184 * math.sqrt(874) * sh_11_5 * z
    sh_12_6 = 5 / 552 * math.sqrt(3) * (math.sqrt(2346) * sh_11_16 * x + math.sqrt(230) * sh_11_18 * x - math.sqrt(230) * sh_11_4 * z + 12 * math.sqrt(23) * sh_11_5 * y + math.sqrt(2346) * sh_11_6 * z)
    sh_12_7 = 5 / 138 * math.sqrt(391) * sh_11_15 * x + 5 / 552 * math.sqrt(966) * sh_11_17 * x - 5 / 552 * math.sqrt(966) * sh_11_5 * z + 5 / 276 * math.sqrt(2737) * sh_11_6 * y + 5 / 138 * math.sqrt(391) * sh_11_7 * z
    sh_12_8 = 5 / 138 * math.sqrt(345) * sh_11_14 * x + 5 / 276 * math.sqrt(322) * sh_11_16 * x - 5 / 276 * math.sqrt(322) * sh_11_6 * z + 10 / 69 * math.sqrt(46) * sh_11_7 * y + 5 / 138 * math.sqrt(345) * sh_11_8 * z
    sh_12_9 = 5 / 552 * math.sqrt(4830) * sh_11_13 * x + 5 / 92 * math.sqrt(46) * sh_11_15 * x - 5 / 92 * math.sqrt(46) * sh_11_7 * z + 5 / 92 * math.sqrt(345) * sh_11_8 * y + 5 / 552 * math.sqrt(4830) * sh_11_9 * z
    sh_12_10 = 5 / 552 * math.sqrt(4186) * sh_11_10 * z + 5 / 552 * math.sqrt(4186) * sh_11_12 * x + 5 / 184 * math.sqrt(230) * sh_11_14 * x - 5 / 184 * math.sqrt(230) * sh_11_8 * z + 5 / 138 * math.sqrt(805) * sh_11_9 * y
    sh_12_11 = 5 / 276 * math.sqrt(3289) * sh_11_10 * y + 5 / 276 * math.sqrt(1794) * sh_11_11 * x + 5 / 552 * math.sqrt(2530) * sh_11_13 * x - 5 / 552 * math.sqrt(2530) * sh_11_9 * z
    sh_12_12 = -5 / 276 * math.sqrt(1518) * sh_11_10 * x + 5 / 23 * math.sqrt(23) * sh_11_11 * y - 5 / 276 * math.sqrt(1518) * sh_11_12 * z
    sh_12_13 = 5 / 276 * math.sqrt(1794) * sh_11_11 * z + 5 / 276 * math.sqrt(3289) * sh_11_12 * y - 5 / 552 * math.sqrt(2530) * sh_11_13 * z - 5 / 552 * math.sqrt(2530) * sh_11_9 * x
    sh_12_14 = -5 / 552 * math.sqrt(4186) * sh_11_10 * x + 5 / 552 * math.sqrt(4186) * sh_11_12 * z + 5 / 138 * math.sqrt(805) * sh_11_13 * y - 5 / 184 * math.sqrt(230) * sh_11_14 * z - 5 / 184 * math.sqrt(230) * sh_11_8 * x
    sh_12_15 = 5 / 552 * math.sqrt(4830) * sh_11_13 * z + 5 / 92 * math.sqrt(345) * sh_11_14 * y - 5 / 92 * math.sqrt(46) * sh_11_15 * z - 5 / 92 * math.sqrt(46) * sh_11_7 * x - 5 / 552 * math.sqrt(4830) * sh_11_9 * x
    sh_12_16 = 5 / 138 * math.sqrt(345) * sh_11_14 * z + 10 / 69 * math.sqrt(46) * sh_11_15 * y - 5 / 276 * math.sqrt(322) * sh_11_16 * z - 5 / 276 * math.sqrt(322) * sh_11_6 * x - 5 / 138 * math.sqrt(345) * sh_11_8 * x
    sh_12_17 = 5 / 138 * math.sqrt(391) * sh_11_15 * z + 5 / 276 * math.sqrt(2737) * sh_11_16 * y - 5 / 552 * math.sqrt(966) * sh_11_17 * z - 5 / 552 * math.sqrt(966) * sh_11_5 * x - 5 / 138 * math.sqrt(391) * sh_11_7 * x
    sh_12_18 = 5 / 552 * math.sqrt(3) * (math.sqrt(2346) * sh_11_16 * z + 12 * math.sqrt(23) * sh_11_17 * y - math.sqrt(230) * sh_11_18 * z - math.sqrt(230) * sh_11_4 * x - math.sqrt(2346) * sh_11_6 * x)
    sh_12_19 = 5 / 184 * math.sqrt(874) * sh_11_17 * z + 5 / 276 * math.sqrt(2185) * sh_11_18 * y - 5 / 276 * math.sqrt(115) * sh_11_19 * z - 5 / 276 * math.sqrt(115) * sh_11_3 * x - 5 / 184 * math.sqrt(874) * sh_11_5 * x
    sh_12_20 = 5 / 276 * math.sqrt(2185) * sh_11_18 * z + 5 / 69 * math.sqrt(115) * sh_11_19 * y - 5 / 276 * math.sqrt(69) * sh_11_2 * x - 5 / 276 * math.sqrt(69) * sh_11_20 * z - 5 / 276 * math.sqrt(2185) * sh_11_4 * x
    sh_12_21 = -5 / 552 * math.sqrt(138) * sh_11_1 * x + 5 / 276 * math.sqrt(2415) * sh_11_19 * z + 5 / 92 * math.sqrt(161) * sh_11_20 * y - 5 / 552 * math.sqrt(138) * sh_11_21 * z - 5 / 276 * math.sqrt(2415) * sh_11_3 * x
    sh_12_22 = -5 / 552 * math.sqrt(46) * sh_11_0 * x - 5 / 552 * math.sqrt(10626) * sh_11_2 * x + 5 / 552 * math.sqrt(10626) * sh_11_20 * z + 5 / 138 * math.sqrt(253) * sh_11_21 * y - 5 / 552 * math.sqrt(46) * sh_11_22 * z
    sh_12_23 = -5 / 24 * math.sqrt(22) * sh_11_1 * x + 5 / 24 * math.sqrt(22) * sh_11_21 * z + 5 / 12 * sh_11_22 * y
    sh_12_24 = 5 / 12 * math.sqrt(6) * (-sh_11_0 * x + sh_11_22 * z)
    return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4, sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6, sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8, sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10, sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12, sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14, sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16, sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18, sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20, sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22, sh_12_0, sh_12_1, sh_12_2, sh_12_3, sh_12_4, sh_12_5, sh_12_6, sh_12_7, sh_12_8, sh_12_9, sh_12_10, sh_12_11, sh_12_12, sh_12_13, sh_12_14, sh_12_15, sh_12_16, sh_12_17, sh_12_18, sh_12_19, sh_12_20, sh_12_21, sh_12_22, sh_12_23, sh_12_24], dim=-1)


def _self_test(full=False):
    """Run on a real GPU: reference AD, finite-difference AD, force-loss chain."""
    if triton is None or not torch.cuda.is_available():
        raise RuntimeError('--test requires a CUDA/ROCm GPU, torch, e3nn and triton')
    from e3nn.o3 import spherical_harmonics as e3nn_sh
    torch.manual_seed(20260909)
    device='cuda'
    cases=[(0,), (0,1,2,3), (3,1,3,0)]
    if full: cases += [(l,) for l in range(1,13)] + [tuple(range(13))]
    totals={'forward':0.,'backward':0.,'double_x':0.,'double_grad_out':0.}
    count=0
    for dtype in (torch.float64,torch.float32):
        # FP32 high-degree contractions may cancel near zero.
        atol,rtol = (2e-9,2e-8) if dtype==torch.float64 else (3e-3,3e-3)
        for ls in cases:
            for normalize in (False,True):
                for normalization in ('component','norm','integral'):
                    module=SphericalHarmonics(list(ls),normalize,normalization)
                    # Deliberately strided X, G and V; include multiple leading axes.
                    x=(torch.randn(2,3,6,device=device,dtype=dtype)*0.35)[...,::2].requires_grad_()
                    xr=x.detach().clone().requires_grad_()
                    y=module(x)
                    yr=module._reference_forward(xr)
                    ye=e3nn_sh(list(ls),xr,normalize,normalization)
                    torch.testing.assert_close(y,ye,atol=atol,rtol=rtol)
                    dim=y.shape[-1]
                    g=torch.randn(2,3,2*dim,device=device,dtype=dtype)[...,::2].requires_grad_()
                    gr=g.detach().clone().requires_grad_()
                    v=torch.randn(2,3,6,device=device,dtype=dtype)[...,::2]
                    gx=torch.autograd.grad(y,x,g,create_graph=True)[0]
                    # Add a zero path for the constant l=0 reference output.
                    yr=yr+xr.sum()*0.0
                    gxr=torch.autograd.grad(yr,xr,gr,create_graph=True)[0]
                    gxr=gxr+xr*0.0+gr.sum()*0.0
                    dx,dg=torch.autograd.grad(gx,(x,g),v)
                    dxr,dgr=torch.autograd.grad(gxr,(xr,gr),v)
                    for label,a,b in [('forward',y,yr),('backward',gx,gxr),
                                      ('double_x',dx,dxr),('double_grad_out',dg,dgr)]:
                        torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
                        totals[label]=max(totals[label],(a-b).abs().max().item())
                    count+=1
        print(f'{dtype}: passed reference comparisons',flush=True)
    # Finite-difference gradcheck verifies the actual custom Function chain.
    x=(torch.randn(2,3,device=device,dtype=torch.float64)+0.5).requires_grad_()
    for normalize in (False,True):
        fn=lambda x: spherical_harmonics([0,1,2,3],x,normalize,'component')
        assert torch.autograd.gradcheck(fn,(x,),eps=1e-6,atol=1e-5,rtol=1e-4)
        assert torch.autograd.gradgradcheck(fn,(x,),eps=1e-6,atol=2e-5,rtol=2e-4)
    # Test both pruned double-backward variants independently.
    ls=(0,1,2,3)
    for nx,ng in ((True,False),(False,True)):
        a=torch.randn(3,3,device=device,dtype=torch.float64,requires_grad=nx)
        g=torch.randn(3,16,device=device,dtype=torch.float64,requires_grad=ng)
        v=torch.randn_like(a)
        gx=_SHBackward.apply(a,g,ls,True,'component')
        target=a if nx else g
        actual=torch.autograd.grad(gx,target,v)[0]
        ar=a.detach().requires_grad_(); gr=g.detach().requires_grad_()
        yr=SphericalHarmonics(list(ls),True,'component')._reference_forward(ar)
        gxr=torch.autograd.grad(yr,ar,gr,create_graph=True)[0]
        expected=torch.autograd.grad(gxr,ar if nx else gr,v)[0]
        torch.testing.assert_close(actual,expected,atol=2e-9,rtol=2e-8)
    # A nonlinear energy head makes grad_out depend on both x and model weights.
    a=torch.randn(4,3,device=device,dtype=torch.float64,requires_grad=True)
    w=torch.randn(16,device=device,dtype=torch.float64,requires_grad=True)
    def force_loss_gradient(reference):
        mod=SphericalHarmonics(list(ls),True,'component')
        y=mod._reference_forward(a) if reference else mod(a)
        energy=(y*w).sum(-1).sin().sum()
        force=-torch.autograd.grad(energy,a,create_graph=True)[0]
        return torch.autograd.grad(force.square().sum(),(a,w))
    actual=force_loss_gradient(False); expected=force_loss_gradient(True)
    for p,q in zip(actual,expected):
        torch.testing.assert_close(p,q,atol=2e-8,rtol=2e-8)
    # Empty inputs and a single unbatched vector.
    for shape in ((0,3),(2,0,3),(3,)):
        a=torch.randn(shape,device=device,dtype=torch.float64,requires_grad=True)
        y=spherical_harmonics(list(ls),a,True,'norm')
        gx=torch.autograd.grad(y,a,torch.ones_like(y),create_graph=True)[0]
        dx=torch.autograd.grad(gx,a,torch.ones_like(gx))[0]
        assert dx.shape==a.shape
    # Clamp region including origin: test the mathematically finite GPU path
    # against polynomial(x/eps), avoiding PyTorch norm's origin Hessian.
    a=torch.tensor([[0.,0.,0.],[1e-14,-2e-14,1e-14]],device=device,
                   dtype=torch.float64,requires_grad=True)
    ar=a.detach().clone().requires_grad_()
    mod=SphericalHarmonics(list(ls),True,'component')
    y=mod(a)
    yr=SphericalHarmonics(list(ls),False,'component')._reference_forward(ar/1e-12)
    g=torch.randn_like(y,requires_grad=True); gr=g.detach().clone().requires_grad_()
    v=torch.randn_like(a)*1e-12
    gx=torch.autograd.grad(y,a,g,create_graph=True)[0]
    gxr=torch.autograd.grad(yr,ar,gr,create_graph=True)[0]
    dx,dg=torch.autograd.grad(gx,(a,g),v)
    dxr,dgr=torch.autograd.grad(gxr,(ar,gr),v)
    for p,q in ((y,yr),(gx,gxr),(dx,dxr),(dg,dgr)):
        assert torch.isfinite(p).all()
        torch.testing.assert_close(p,q,atol=1e-8,rtol=1e-9)
    # Parity, multiplicity and scripting compatibility.
    mod=SphericalHarmonics('2x1e + 1x0e',True,'integral',irreps_in='1e')
    assert mod._ls_list==[1,1,0]
    a=torch.randn(5,3,device=device,dtype=torch.float64)
    torch.testing.assert_close(mod(a),torch.jit.script(mod)(a))
    try:
        SphericalHarmonics('1e',True,irreps_in='1o')
    except ValueError:
        pass
    else:
        raise AssertionError('parity mismatch was accepted')
    print(f'PASS: {count} reference cases; gradcheck, gradgradcheck, pruning, '
          'force-loss gradients, shape/zero/parity/script cases')
    print('Maximum absolute errors by stage (combined FP32/FP64):',totals)


def _main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--dump-kernels',type=Path)
    parser.add_argument('--lmax',type=int,default=3)
    args=parser.parse_args()
    if args.test: _self_test(args.full)
    if args.dump_kernels:
        if not 0<=args.lmax<=12: parser.error('--lmax must be 0..12')
        args.dump_kernels.mkdir(parents=True,exist_ok=True)
        for mode in ('forward','backward','double_backward'):
            source=_kernel_source(tuple(range(args.lmax+1)),True,'component',mode)
            path=args.dump_kernels/f'sh_{mode}_l{args.lmax}.py'
            path.write_text('import triton\nimport triton.language as tl\n\n@triton.jit\n'+source)
            print(path)
    if not args.test and not args.dump_kernels: parser.print_help()


if __name__=='__main__':
    _main()
