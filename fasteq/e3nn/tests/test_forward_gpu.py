"""Per-function GPU parity against upstream e3nn.

Set E3NN_TRITON_PACKAGE=fasteq.e3nn.triton for the FastEq package layout.
"""

import importlib
import os
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

from e3nn import o3, math as emath
from e3nn.o3 import _rotation as reference_rotation
from e3nn.o3 import _s2grid as reference_s2
from e3nn.o3 import _so3grid as reference_so3
from e3nn.o3 import _angular_spherical_harmonics as reference_angular
from e3nn.math import _linalg as reference_linalg
from e3nn.math import perm as reference_perm

import fasteq.e3nn.triton.o3._rotation as tr
import fasteq.e3nn.triton.o3._s2grid as ts
import fasteq.e3nn.triton.o3._so3grid as tg
import fasteq.e3nn.triton.o3._angular_spherical_harmonics as ta
import fasteq.e3nn.triton.math._bessel as tm_bessel
import fasteq.e3nn.triton.math._linalg as tm_linalg
import fasteq.e3nn.triton.math._soft_one_hot_linspace as tm_one_hot
import fasteq.e3nn.triton.math._soft_unit_step as tm_step
import fasteq.e3nn.triton.math.perm as tm_perm


def assert_same(actual, expected, tol=3e-5):
    if isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_same(a, b, tol)
    else:
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol, equal_nan=True)


ROTATION_NAMES = (
    "identity_angles", "inverse_angles", "identity_quaternion", "compose_quaternion",
    "inverse_quaternion", "compose_axis_angle", "matrix_x", "matrix_y", "matrix_z",
    "angles_to_matrix", "angles_to_quaternion", "axis_angle_to_quaternion",
    "quaternion_to_axis_angle", "axis_angle_to_matrix", "quaternion_to_matrix",
    "angles_to_xyz", "xyz_to_angles",
)


@pytest.mark.parametrize("name", ROTATION_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotation(name, dtype):
    torch.manual_seed(0)
    d = dict(device="cuda", dtype=dtype)
    a, b, c = (torch.randn(13, **d) for _ in range(3))
    q1 = torch.randn(13, 4, **d)
    q2 = torch.randn(13, 4, **d)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)
    axis1 = torch.randn(13, 3, **d)
    axis2 = torch.randn(13, 3, **d)
    arguments = {
        "identity_angles": (13,), "inverse_angles": (a, b, c),
        "identity_quaternion": (13,), "compose_quaternion": (q1, q2),
        "inverse_quaternion": (q1,), "compose_axis_angle": (axis1, a, axis2, b),
        "matrix_x": (a,), "matrix_y": (a,), "matrix_z": (a,),
        "angles_to_matrix": (a, b, c), "angles_to_quaternion": (a, b, c),
        "axis_angle_to_quaternion": (axis1, a),
        "quaternion_to_axis_angle": (q1,), "axis_angle_to_matrix": (axis1, a),
        "quaternion_to_matrix": (q1,), "angles_to_xyz": (a, b),
        "xyz_to_angles": (axis1,),
    }
    kwargs = d if name.startswith("identity_") else {}
    args = arguments[name]
    assert_same(getattr(tr, name)(*args, **kwargs),
                getattr(reference_rotation, name)(*args, **kwargs), 1e-4)


@pytest.mark.parametrize("name", [
    "spherical_harmonics_alpha", "_mul_m_lm", "SphericalHarmonicsAlphaBeta.forward",
])
def test_angular(name):
    torch.manual_seed(1)
    a = torch.randn(17, device="cuda")
    b = torch.randn_like(a)
    if name == "spherical_harmonics_alpha":
        assert_same(ta.spherical_harmonics_alpha(3, a), reference_angular.spherical_harmonics_alpha(3, a))
    elif name == "_mul_m_lm":
        xm = torch.randn(17, 7, device="cuda")
        xlm = torch.randn(17, 12, device="cuda")
        mul_l = [(1, 1), (3, 1)]
        assert_same(ta._mul_m_lm(mul_l, xm, xlm), reference_angular._mul_m_lm(mul_l, xm, xlm))
    else:
        for normalization in ("integral", "component", "norm"):
            actual = ta.SphericalHarmonicsAlphaBeta([0, 1, 2], normalization).cuda()
            expected = reference_angular.SphericalHarmonicsAlphaBeta([0, 1, 2], normalization).cuda()
            assert_same(actual(a, b), expected(a, b), 1e-4)


@pytest.mark.parametrize("name", [
    "_quadrature_weights", "s2_grid", "_expand_matrix", "ToS2Grid.grid", "FromS2Grid.grid",
])
def test_s2(name):
    d = dict(device="cuda", dtype=torch.float32)
    if name == "_quadrature_weights":
        assert_same(ts._quadrature_weights(5, **d), reference_s2._quadrature_weights(5, **d))
    elif name == "s2_grid":
        assert_same(ts.s2_grid(10, 11, **d), reference_s2.s2_grid(10, 11, **d))
    elif name == "_expand_matrix":
        assert_same(ts._expand_matrix([0, 1, 2], **d), reference_s2._expand_matrix([0, 1, 2], **d))
    else:
        cls = name.split(".")[0]
        actual = getattr(ts, cls)(lmax=2, res=(8, 9), **d)
        expected = getattr(reference_s2, cls)(lmax=2, res=(8, 9), **d)
        assert_same(actual.grid, expected.grid)


@pytest.mark.parametrize("name", ["to_grid", "from_grid"])
def test_so3(name):
    torch.manual_seed(2)
    actual = tg.SO3Grid(1, 2).cuda()
    expected = reference_so3.SO3Grid(1, 2).cuda()
    x = torch.randn(3, actual.D.shape[-1], device="cuda")
    grid_x = torch.randn(3, actual.res_alpha, actual.res_beta, actual.res_gamma, device="cuda")
    arg = x if name == "to_grid" else grid_x
    assert_same(getattr(actual, name)(arg), getattr(expected, name)(arg), 1e-4)


def test_bessel():
    x = torch.tensor([0., 0.2, 0.8], device="cuda")
    assert_same(tm_bessel.bessel(x, 4), emath.bessel(x, 4))


@pytest.mark.parametrize("basis", ["gaussian", "cosine", "smooth_finite", "fourier", "bessel"])
@pytest.mark.parametrize("cutoff", [False, True])
def test_soft_one_hot_linspace(basis, cutoff):
    x = torch.tensor([0., 0.2, 0.8], device="cuda")
    assert_same(tm_one_hot.soft_one_hot_linspace(x, 0., 1., 4, basis, cutoff),
                emath.soft_one_hot_linspace(x, 0., 1., 4, basis, cutoff), 1e-4)


def test_direct_sum():
    torch.manual_seed(3)
    a = torch.randn(3, 2, 4, device="cuda")
    b = torch.randn(3, 5, 2, device="cuda")
    assert_same(tm_linalg.direct_sum(a, b), reference_linalg.direct_sum(a, b))


def test_direct_sum_distinct_blocks():
    left = torch.arange(8, device="cuda", dtype=torch.float32).reshape(2, 1, 4)
    right = torch.arange(20, device="cuda", dtype=torch.float32).reshape(2, 5, 2) + 100
    actual = tm_linalg.direct_sum(left, right)
    expected = reference_linalg.direct_sum(left, right)
    assert_same(actual, expected, 0)
    assert_same(actual[:, 1:, 4:], right, 0)


def test_soft_unit_step_forward():
    x = torch.tensor([-1., 0., 0.2, 0.8], device="cuda")
    assert_same(tm_step.soft_unit_step(x), emath.soft_unit_step(x))


def test_soft_unit_step_backward():
    x = torch.tensor([-1., 0., 0.2, 2.], device="cuda", requires_grad=True)
    y = tm_step.soft_unit_step(x).sum()
    y.backward()
    expected = x.detach().clone().requires_grad_(True)
    emath.soft_unit_step(expected).sum().backward()
    assert_same(x.grad, expected.grad)


def test_natural_representation():
    p = (2, 0, 1)
    assert_same(tm_perm.natural_representation(p, device="cuda"),
                reference_perm.natural_representation(p, device="cuda"))
