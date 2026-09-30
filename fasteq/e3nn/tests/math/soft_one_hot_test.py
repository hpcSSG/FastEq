"""GPU cases adapted from tests/math, comparing FastEq with e3nn."""

import importlib
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from e3nn import math as reference
from e3nn.math import _linalg as reference_linalg
from e3nn.math import perm as reference_perm

PREFIX = os.environ.get("E3NN_TRITON_PACKAGE", "fasteq.e3nn.triton")
bessel = importlib.import_module(PREFIX + ".math._bessel").bessel
direct_sum = importlib.import_module(PREFIX + ".math._linalg").direct_sum
one_hot = importlib.import_module(PREFIX + ".math._soft_one_hot_linspace").soft_one_hot_linspace
soft_unit_step = importlib.import_module(PREFIX + ".math._soft_unit_step").soft_unit_step
natural_representation = importlib.import_module(PREFIX + ".math.perm").natural_representation


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("basis", ["gaussian", "cosine", "fourier", "bessel", "smooth_finite"])
@pytest.mark.parametrize("cutoff", [True, False])
def test_one_hot_outside_and_ends(dtype, basis, cutoff):
    # Mirrors the cutoff/normalization cases in tests/math/soft_one_hot_test.py.
    x = torch.tensor([[-2.0, -1.1, -1.0, -0.2], [0.5, 2.0, 2.1, 3.0]],
                     device="cuda", dtype=dtype)
    args = (-1.0, 2.0, 5, basis, cutoff)
    actual = one_hot(x, *args)
    expected = reference.soft_one_hot_linspace(x, *args)
    assert actual.shape == (2, 4, 5)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4, equal_nan=True)
    if cutoff and basis not in ("gaussian", "bessel"):
        assert torch.count_nonzero(actual.reshape(-1, 5)[[0, 1, 6, 7]]) == 0


@pytest.mark.parametrize("basis", ["gaussian", "cosine", "fourier", "bessel", "smooth_finite"])
def test_one_hot_zero_out_parameter_grid(basis):
    x = torch.cat((torch.linspace(-2., -1.1, 20, device="cuda"),
                   torch.linspace(2.1, 3., 20, device="cuda")))
    args = (-1., 2., 5, basis, True)
    torch.testing.assert_close(one_hot(x, *args), reference.soft_one_hot_linspace(x, *args),
                               rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("basis", ["gaussian", "cosine", "fourier", "smooth_finite"])
@pytest.mark.parametrize("cutoff", [True, False])
def test_one_hot_normalized_parameter_grid(basis, cutoff):
    x = torch.linspace(-14., 105., 50, device="cuda")
    args = (-20., 120., 12, basis, cutoff)
    got = one_hot(x, *args)
    want = reference.soft_one_hot_linspace(x, *args)
    torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-4)
    norm = got.square().sum(-1)
    assert norm.min() > 0.4
    assert norm.max() < 2.
