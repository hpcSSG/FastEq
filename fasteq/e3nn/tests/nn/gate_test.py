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


def test_gate_multiple_irreps_and_autograd_fallback():
    # The irreps layout follows tests/nn/gate_test.py at smaller multiplicity.
    args = ("2x0o", [torch.tanh], "1x0o+1x0e", [torch.tanh, torch.sigmoid],
            "1x1e+1x1o")
    actual = optimized.Gate(*args).cuda()
    expected = reference.Gate(*args).cuda()
    x = torch.randn(4, actual.irreps_in.dim, device="cuda")
    with torch.no_grad():
        close(actual(x), expected(x), 1e-4)
    x.requires_grad_()
    gx, = torch.autograd.grad(actual(x).sum(), x, retain_graph=True)
    gy, = torch.autograd.grad(expected(x).sum(), x)
    close(gx, gy, 1e-4)

