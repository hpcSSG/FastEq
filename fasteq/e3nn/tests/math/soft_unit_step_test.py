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
def test_soft_unit_step_forward_and_first_derivative(dtype):
    x = torch.tensor([-3., -1., 0., 0.125, 0.5, 2.],
                     device="cuda", dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual = soft_unit_step(x)
    expected = reference.soft_unit_step(y)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    grad = torch.arange(1, 7, device="cuda", dtype=dtype)
    gx, = torch.autograd.grad(actual, x, grad)
    gy, = torch.autograd.grad(expected, y, grad)
    torch.testing.assert_close(gx, gy, rtol=3e-5, atol=3e-5)
    assert torch.isfinite(gx).all()

