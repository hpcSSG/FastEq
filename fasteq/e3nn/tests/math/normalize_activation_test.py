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



from e3nn.math._normalize_activation import moment as reference_moment

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_moment_matches_e3nn(dtype):
    moment = importlib.import_module(PREFIX + ".math._normalize_activation").moment
    got = moment(torch.tanh, 2, dtype=dtype, device="cuda")
    want = reference_moment(torch.tanh, 2, dtype=dtype, device="cuda")
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

def test_normalize2mom_retains_original_class():
    module = importlib.import_module(PREFIX + ".math._normalize_activation")
    assert module.normalize2mom is reference.normalize2mom
