"""GPU cases adapted from tests/nn, excluding models and tensor products."""

import importlib
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from e3nn import nn as reference, o3

optimized = importlib.import_module(os.environ.get("E3NN_TRITON_PACKAGE", "fasteq.e3nn.triton") + ".nn")


def close(actual, expected, tol=3e-5):
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            close(a, b, tol)
    else:
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol, equal_nan=True)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("nonlinearity", [torch.tanh, torch.sigmoid])
def test_norm_activation_zero_and_nonzero(bias, nonlinearity):
    irreps = o3.Irreps("2x1e+1x0e")
    actual = optimized.NormActivation(irreps, nonlinearity, bias=bias).cuda()
    expected = reference.NormActivation(irreps, nonlinearity, bias=bias).cuda()
    expected.load_state_dict(actual.state_dict())
    for x in (torch.zeros(2, 7, device="cuda"), torch.randn(2, 7, device="cuda")):
        with torch.no_grad():
            close(actual(x), expected(x), 1e-4)
