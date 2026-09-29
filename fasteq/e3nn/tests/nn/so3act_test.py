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


@pytest.mark.parametrize("aspect_ratio", [1, 2])
def test_so3_activation_grid_roundtrip(aspect_ratio):
    args = (1, 1, torch.tanh, 3)
    actual = optimized.SO3Activation(*args, aspect_ratio=aspect_ratio).cuda()
    expected = reference.SO3Activation(*args, aspect_ratio=aspect_ratio).cuda()
    x = torch.randn(2, actual.grid_in.D.shape[-1], device="cuda")
    with torch.no_grad():
        close(actual(x), expected(x), 5e-4)

