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


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 7, 15])
def test_natural_permutation_representation_and_composition(n):
    # Adapted from tests/math/perm_test.py; use a deterministic permutation.
    p = tuple(reversed(range(n)))
    q = tuple((i + 1) % n for i in range(n))
    actual = natural_representation(p, dtype=torch.float64, device="cuda")
    expected = reference_perm.natural_representation(p, dtype=torch.float64, device="cuda")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    eye = torch.eye(n, dtype=torch.float64, device="cuda")
    torch.testing.assert_close(actual @ actual.T, eye, rtol=0, atol=0)
    pq = reference_perm.compose(p, q)
    torch.testing.assert_close(
        natural_representation(p, device="cuda") @ natural_representation(q, device="cuda"),
        natural_representation(pq, device="cuda"), rtol=0, atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_direct_sum_three_rectangular_blocks(dtype):
    a = torch.randn(2, 3, 1, 4, device="cuda", dtype=dtype)
    b = torch.randn(2, 3, 2, 1, device="cuda", dtype=dtype)
    c = torch.randn(2, 3, 3, 2, device="cuda", dtype=dtype)
    actual = direct_sum(a, b, c)
    expected = reference_linalg.direct_sum(a, b, c)
    assert actual.shape == (2, 3, 6, 7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[..., 3:, 5:], c, rtol=0, atol=0)
