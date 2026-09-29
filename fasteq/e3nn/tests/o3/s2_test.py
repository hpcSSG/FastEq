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

