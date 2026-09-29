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



@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("irreps", [("4x0e", "3x0e"), ("3x1o", "2x1o")])
def test_simple_linear_and_weight_views(dtype, irreps):
    module = importlib.import_module(PREFIX + ".o3._linear")
    actual = module.Linear(*irreps).to(device="cuda", dtype=dtype)
    expected = o3.Linear(*irreps).to(device="cuda", dtype=dtype)
    expected.load_state_dict(actual.state_dict())
    x = torch.randn(2, 3, actual.irreps_in.dim, device="cuda", dtype=dtype)
    with torch.no_grad():
        torch.testing.assert_close(actual(x), expected(x), rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(actual.weight_view_for_instruction(0),
                               expected.weight_view_for_instruction(0))

def test_bias_and_gradient_fallback():
    module = importlib.import_module(PREFIX + ".o3._linear")
    actual = module.Linear("2x0e+1x1o", "3x0e+1x1o", biases=True).cuda()
    expected = o3.Linear("2x0e+1x1o", "3x0e+1x1o", biases=True).cuda()
    expected.load_state_dict(actual.state_dict())
    x = torch.randn(4, actual.irreps_in.dim, device="cuda", requires_grad=True)
    out_a, out_b = actual(x), expected(x)
    torch.testing.assert_close(out_a, out_b, rtol=3e-5, atol=3e-5)
    ga, = torch.autograd.grad(out_a.sum(), x, retain_graph=True)
    gb, = torch.autograd.grad(out_b.sum(), x)
    torch.testing.assert_close(ga, gb, rtol=3e-5, atol=3e-5)
