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



@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_cartesian_spherical_harmonics_reference_and_gradient(l):
    module = importlib.import_module(PREFIX + ".o3._spherical_harmonics")
    assert module.spherical_harmonics is o3.spherical_harmonics
    xyz = torch.randn(4, 3, device="cuda", requires_grad=True)
    got = module.spherical_harmonics(l, xyz, True)
    want = o3.spherical_harmonics(l, xyz, True)
    torch.testing.assert_close(got, want)
    if l == 0:
        # Y_0 is constant and has no autograd connection to xyz.
        assert not got.requires_grad
        return
    dx, = torch.autograd.grad(got.sum(), xyz, retain_graph=True)
    dy, = torch.autograd.grad(want.sum(), xyz)
    torch.testing.assert_close(dx, dy)
