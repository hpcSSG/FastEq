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



@pytest.mark.parametrize("squared", [True, False])
@pytest.mark.parametrize("irreps", ["5x0e", "2x0e+4x1o+2x2e"])
def test_norm_forward_and_zero_gradient(squared, irreps):
    module = importlib.import_module(PREFIX + ".o3._norm")
    actual = module.Norm(irreps, squared=squared).cuda()
    expected = o3.Norm(irreps, squared=squared).cuda()
    x = torch.randn(3, actual.irreps_in.dim, device="cuda")
    torch.testing.assert_close(actual(x), expected(x), rtol=3e-5, atol=3e-5)
    zeros = torch.zeros_like(x, requires_grad=True)
    ga, = torch.autograd.grad(actual(zeros).sum(), zeros, retain_graph=True)
    gb, = torch.autograd.grad(expected(zeros).sum(), zeros)
    torch.testing.assert_close(ga, gb)
