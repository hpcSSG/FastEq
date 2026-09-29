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


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
def test_angular_matches_e3nn_and_cartesian(l, normalization):
    # Source: angular_spherical_harmonics_test.py, including the zero vector's
    # avoided singularity by generating nonzero coordinates.
    xyz = torch.randn(11, 3, device="cuda") + torch.tensor([2., 0., 0.], device="cuda")
    alpha, beta = reference_rotation.xyz_to_angles(xyz)
    got = angular.spherical_harmonics_alpha_beta(l, alpha, beta, normalization=normalization)
    want = reference_angular.spherical_harmonics_alpha_beta(l, alpha, beta,
                                                             normalization=normalization)
    torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-4)
    if normalization == "integral":
        cartesian = o3.spherical_harmonics(l, xyz, True, normalization="integral")
        torch.testing.assert_close(got, cartesian, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_angular_equivariance(l):
    # Source: tests/o3/angular_spherical_harmonics_test.py.
    a = torch.tensor(0.45, device="cuda")
    b = torch.tensor(1.1, device="cuda")
    r = (torch.tensor(0.25, device="cuda"), torch.tensor(0.8, device="cuda"),
         torch.tensor(-0.35, device="cuda"))
    ra, rb, _ = rotation.compose_angles(*r, a, b, torch.tensor(0., device="cuda"))
    rotated = angular.spherical_harmonics_alpha_beta(l, ra, rb)
    original = angular.spherical_harmonics_alpha_beta(l, a, b)
    # e3nn's wigner_D currently constructs its generators on CPU.
    D = o3.wigner_D(l, *(angle.cpu() for angle in r)).to(original.device)
    torch.testing.assert_close(rotated, D @ original,
                               rtol=3e-4, atol=3e-4)
