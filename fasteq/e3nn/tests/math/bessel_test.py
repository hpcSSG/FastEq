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
@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("x_max", [1.0, 2.5])
def test_bessel_shape_zero_limit_and_scale(dtype, n, x_max):
    x = torch.tensor([[0., 0.125], [x_max / 2, x_max]], device="cuda", dtype=dtype)
    actual = bessel(x, n, x_max=x_max)
    expected = reference.bessel(x, n, x_max=x_max)
    assert actual.shape == x.shape + (n,)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    assert torch.isfinite(actual).all()

