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


@pytest.mark.parametrize("squeeze", [True, False])
def test_extract_squeeze_and_repeated_indices(squeeze):
    args = ("1x1e+1x0e+1x0e", ["1x0e+1x0e"], [(2, 1)])
    actual = optimized.Extract(*args, squeeze_out=squeeze)
    expected = reference.Extract(*args, squeeze_out=squeeze)
    x = torch.arange(5, dtype=torch.float64, device="cuda").repeat(2, 1)
    close(actual(x), expected(x), 0)
    ir_actual = optimized.ExtractIr(args[0], "0e")
    ir_expected = reference.ExtractIr(args[0], "0e")
    close(ir_actual(x), ir_expected(x), 0)

