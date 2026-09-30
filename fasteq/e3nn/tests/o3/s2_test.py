"""GPU invariants adapted from tests/o3, excluding tensor-product tests."""

import importlib
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from e3nn import o3
from e3nn.o3 import _rotation as reference_rotation
from e3nn.o3 import _angular_spherical_harmonics as reference_angular
from e3nn.o3 import _s2grid as reference_s2
from e3nn.o3 import _so3grid as reference_so3

PREFIX = os.environ.get("E3NN_TRITON_PACKAGE", "fasteq.e3nn.triton")
rotation = importlib.import_module(PREFIX + ".o3._rotation")
angular = importlib.import_module(PREFIX + ".o3._angular_spherical_harmonics")
s2 = importlib.import_module(PREFIX + ".o3._s2grid")
so3 = importlib.import_module(PREFIX + ".o3._so3grid")


@pytest.fixture
def dtype(request):
    # e3nn's quadrature intermediates use the default dtype, even when the
    # result is explicitly requested as float64. Match upstream's dtype fixture.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("res", [(8, 9), (8, 10), (10, 13)])
@pytest.mark.parametrize("normalization", ["component", "norm"])
def test_s2_projection_roundtrip_matches_reference(res, normalization):
    # Reuses the odd/even alpha resolutions in tests/o3/s2_test.py.
    nb, na = res
    params = dict(device="cuda", dtype=torch.float32, normalization=normalization)
    to_grid = s2.ToS2Grid(lmax=2, res=res, **params)
    from_grid = s2.FromS2Grid(res=res, lmax=2, **params)
    ref_to = reference_s2.ToS2Grid(lmax=2, res=res, **params)
    ref_from = reference_s2.FromS2Grid(res=res, lmax=2, **params)
    coeff = torch.randn(2, 9, device="cuda")
    signal = torch.randn(2, nb, na, device="cuda")
    torch.testing.assert_close(from_grid(to_grid(coeff)), ref_from(ref_to(coeff)),
                               rtol=4e-4, atol=4e-4)
    torch.testing.assert_close(to_grid(from_grid(signal)), ref_to(ref_from(signal)),
                               rtol=4e-4, atol=4e-4)


@pytest.mark.parametrize("resolution,aspect_ratio", [(2, 1), (2, 2), (3, 2)])
def test_so3_grid_two_way_projection(resolution, aspect_ratio):
    actual = so3.SO3Grid(1, resolution, aspect_ratio=aspect_ratio).cuda()
    expected = reference_so3.SO3Grid(1, resolution, aspect_ratio=aspect_ratio).cuda()
    x = torch.randn(2, actual.D.shape[-1], device="cuda")
    y = torch.randn(2, actual.res_alpha, actual.res_beta, actual.res_gamma, device="cuda")
    torch.testing.assert_close(actual.from_grid(actual.to_grid(x)),
                               expected.from_grid(expected.to_grid(x)), rtol=4e-4, atol=4e-4)
    torch.testing.assert_close(actual.to_grid(actual.from_grid(y)),
                               expected.to_grid(expected.from_grid(y)), rtol=4e-4, atol=4e-4)


# Match the full parameter grids in upstream s2_test.py: 7 * 4 * 4 cases
# for each inverse direction, including inferred resolutions and lmax.
@pytest.mark.parametrize("res_a", [11, 12, 13, 14, 15, 16, None])
@pytest.mark.parametrize("res_b", [12, 14, 16, None])
@pytest.mark.parametrize("lmax", [0, 1, 5, None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], indirect=True)
def test_inverse1_parameter_grid(lmax, res_b, res_a, dtype):
    if lmax is None and res_b is None and res_a is None:
        return
    kwargs = dict(dtype=dtype, device="cuda")
    actual_from = s2.FromS2Grid((res_b, res_a), lmax, **kwargs)
    actual_to = s2.ToS2Grid(lmax, (res_b, res_a), **kwargs)
    expected_from = reference_s2.FromS2Grid((res_b, res_a), lmax, **kwargs)
    expected_to = reference_s2.ToS2Grid(lmax, (res_b, res_a), **kwargs)
    x = torch.randn(actual_from.res_beta, actual_from.res_alpha, device="cuda", dtype=dtype)
    projected = actual_to(actual_from(x))
    reference_projected = expected_to(expected_from(x))
    tol = 1e-3 if dtype == torch.float32 else 1e-8
    torch.testing.assert_close(projected, reference_projected, rtol=tol, atol=tol)
    torch.testing.assert_close(actual_to(actual_from(projected)), projected,
                               rtol=tol, atol=tol)


@pytest.mark.parametrize("res_a", [11, 12, 13, 14, 15, 16, None])
@pytest.mark.parametrize("res_b", [12, 14, 16, None])
@pytest.mark.parametrize("lmax", [0, 1, 5, None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], indirect=True)
def test_inverse2_parameter_grid(lmax, res_b, res_a, dtype):
    if lmax is None and res_b is None and res_a is None:
        return
    kwargs = dict(dtype=dtype, device="cuda")
    actual_from = s2.FromS2Grid((res_b, res_a), lmax, **kwargs)
    actual_to = s2.ToS2Grid(lmax, (res_b, res_a), **kwargs)
    expected_from = reference_s2.FromS2Grid((res_b, res_a), lmax, **kwargs)
    expected_to = reference_s2.ToS2Grid(lmax, (res_b, res_a), **kwargs)
    x = torch.randn((actual_from.lmax + 1) ** 2, device="cuda", dtype=dtype)
    got = actual_from(actual_to(x))
    want = expected_from(expected_to(x))
    tol = 1e-3 if dtype == torch.float32 else 1e-8
    torch.testing.assert_close(got, want, rtol=tol, atol=tol)
    torch.testing.assert_close(got, x, rtol=tol, atol=tol)


@pytest.mark.parametrize("res_a", [100, 101])
@pytest.mark.parametrize("res_b", [98, 100])
@pytest.mark.parametrize("lmax", [1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], indirect=True)
def test_s2_exp_projection_parameter_grid(lmax, res_b, res_a, dtype):
    kwargs = dict(dtype=dtype, device="cuda")
    actual_from = s2.FromS2Grid((res_b, res_a), lmax, **kwargs)
    actual_to = s2.ToS2Grid(lmax, (res_b, res_a), **kwargs)
    expected_from = reference_s2.FromS2Grid((res_b, res_a), lmax, **kwargs)
    expected_to = reference_s2.ToS2Grid(lmax, (res_b, res_a), **kwargs)
    x = torch.randn((lmax + 1) ** 2, device="cuda", dtype=dtype) * 0.2
    tol = 1e-3 if dtype == torch.float32 else 1e-8
    torch.testing.assert_close(actual_from(actual_to(x).exp()),
                               expected_from(expected_to(x).exp()),
                               rtol=tol, atol=tol)
