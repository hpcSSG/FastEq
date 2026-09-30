"""GPU cases adapted from tests/nn, excluding models and tensor products."""

import importlib
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from e3nn import nn as reference

optimized = importlib.import_module(os.environ.get("E3NN_TRITON_PACKAGE", "fasteq.e3nn.triton") + ".nn")


def close(actual, expected, tol=3e-5):
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            close(a, b, tol)
    else:
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol, equal_nan=True)


def test_identity_and_fully_connected_are_retained():
    assert optimized.Identity is reference.Identity
    assert optimized.FullyConnectedNet is reference.FullyConnectedNet
    x = torch.randn(3, 4, device="cuda")
    assert optimized.Identity("4x0e", "4x0e").cuda()(x) is x


@pytest.mark.parametrize("act", [None, torch.tanh])
@pytest.mark.parametrize("var_in,var_out,out_act", [
    (1, 1, False), (1, 1, True), (0.1, 10.0, False), (0.1, 0.05, True),
])
def test_fully_connected_variance_parameter_grid(act, var_in, var_out, out_act):
    args = ((64, 48, 32, 4), act, var_in, var_out, out_act)
    actual = optimized.FullyConnectedNet(*args).cuda()
    expected = reference.FullyConnectedNet(*args).cuda()
    expected.load_state_dict(actual.state_dict())
    x = torch.randn(128, args[0][0], device="cuda") * var_in ** 0.5
    with torch.no_grad():
        close(actual(x), expected(x))
