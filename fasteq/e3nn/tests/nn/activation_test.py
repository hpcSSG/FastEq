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


@pytest.mark.parametrize("irreps,acts", [
    ("6x0o", [torch.abs]),
    ("3x0e", [torch.tanh]),
    ("4x0e+3x0o", [torch.nn.functional.silu, torch.abs]),
    ("2x0e+1x1o+2x0o", [torch.sigmoid, None, torch.abs]),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_activation_parity_and_normalization(irreps, acts, dtype):
    # Cases and parity checks from tests/nn/activation_test.py.
    actual = optimized.Activation(irreps, acts)
    expected = reference.Activation(irreps, acts)
    assert actual.irreps_out == expected.irreps_out
    x = torch.randn(13, actual.irreps_in.dim, device="cuda", dtype=dtype)
    with torch.no_grad():
        close(actual(x), expected(x), 1e-4)


def test_activation_custom_callable_and_nonlast_dimension_fallback():
    fn = lambda x: x * x
    actual = optimized.Activation("2x0o", [fn])
    expected = reference.Activation("2x0o", [fn])
    x = torch.randn(3, 2, 4, device="cuda", requires_grad=True)
    y = actual(x, dim=1)
    z = expected(x, dim=1)
    close(y, z)
    gx, = torch.autograd.grad(y.sum(), x, retain_graph=True)
    gz, = torch.autograd.grad(z.sum(), x)
    close(gx, gz)
