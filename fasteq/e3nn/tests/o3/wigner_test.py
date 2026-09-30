"""GPU invariants adapted from tests/o3, excluding tensor-product tests."""

import importlib
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from e3nn import o3
from e3nn.o3 import _rotation as reference_rotation
from e3nn.o3 import _angular_spherical_harmonics as reference_angular
from e3nn.o3 import _s2grid as reference_s2
from e3nn.o3 import _so3grid as reference_so3

PREFIX = os.environ.get("E3NN_TRITON_PACKAGE", "fasteq.e3nn.triton")
rotation = importlib.import_module(PREFIX + ".o3._rotation")
angular = importlib.import_module(PREFIX + ".o3._angular_spherical_harmonics")
s2 = importlib.import_module(PREFIX + ".o3._s2grid")
so3 = importlib.import_module(PREFIX + ".o3._so3grid")



def test_wigner_reference_symmetry_and_orthogonality():
    module = importlib.import_module(PREFIX + ".o3._wigner")
    assert module.wigner_3j is o3.wigner_3j
    assert module.wigner_D is o3.wigner_D
    coeff = module.wigner_3j(1, 1, 1, dtype=torch.float64, device="cuda")
    torch.testing.assert_close(coeff, o3.wigner_3j(1, 1, 1, dtype=torch.float64, device="cuda"))
    # wigner_D's generators follow the default CPU device in this e3nn version.
    a = torch.tensor(0.4)
    D = module.wigner_D(2, a, a * 2, -a)
    torch.testing.assert_close(D @ D.T, torch.eye(5), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("l1,l2,l3", [
    (1, 2, 3), (2, 3, 4), (3, 4, 5), (1, 1, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 2, 2),
])
def test_wigner_3j_parameter_grid(l1, l2, l3):
    module = importlib.import_module(PREFIX + ".o3._wigner")
    got = module.wigner_3j(l1, l2, l3, device="cuda")
    want = o3.wigner_3j(l1, l2, l3, device="cuda")
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("j", [0, 0.5, 1, 1.5, 2, 2.5])
def test_su2_generators_parameter_grid(j):
    module = importlib.import_module(PREFIX + ".o3._wigner")
    got = module.su2_generators(j)
    want = o3.su2_generators(j)
    torch.testing.assert_close(got, want)
