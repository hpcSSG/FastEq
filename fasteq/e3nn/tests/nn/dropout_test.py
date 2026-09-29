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


@pytest.mark.parametrize("p", [0.0, 0.4, 0.75, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dropout_mask_reproducibility_and_eval_alias(p, dtype):
    # Source: tests/nn/dropout_test.py. One Bernoulli draw per irrep, reused
    # across spatial samples and every component of that irrep.
    actual = optimized.Dropout("2x1e+1x0e", p)
    expected = reference.Dropout("2x1e+1x0e", p)
    x = torch.randn(5, 2, 7, device="cuda", dtype=dtype)
    actual.eval()
    assert actual(x) is x
    expected.train()
    actual.train()
    torch.manual_seed(109)
    want = expected(x)
    next_want = torch.rand(8, device="cuda")
    torch.manual_seed(109)
    got = actual(x)
    next_got = torch.rand(8, device="cuda")
    close(got, want, 0)
    assert torch.equal(next_got, next_want)
    torch.testing.assert_close(got[:, 0] * x[:, 1], got[:, 1] * x[:, 0],
                               rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(got[..., 0] * x[..., 1], got[..., 1] * x[..., 0],
                               rtol=2e-5, atol=2e-5)

