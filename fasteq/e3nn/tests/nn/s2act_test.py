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


@pytest.mark.parametrize("random_rot", [False, True])
def test_s2_activation_parity_and_random_rotation(random_rot):
    args = ("1x0e+1x1o", torch.tanh, 8)
    # e3nn's wigner_D builds CPU generators, so its random rotation path
    # currently requires CPU angles. Keep the CUDA projection case separate.
    device = "cpu" if random_rot else "cuda"
    actual = optimized.S2Activation(*args, random_rot=random_rot).to(device)
    expected = reference.S2Activation(*args, random_rot=random_rot).to(device)
    assert actual.irreps_out == expected.irreps_out
    x = torch.randn(2, 4, device=device)
    torch.manual_seed(71)
    with torch.no_grad():
        want = expected(x)
    torch.manual_seed(71)
    with torch.no_grad():
        got = actual(x)
    close(got, want, 5e-4)
