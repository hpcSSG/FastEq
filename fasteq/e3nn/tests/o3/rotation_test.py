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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotation_composition_and_inverse(dtype):
    # Adapted from tests/o3/rotation_test.py: compare rotation matrices,
    # since different valid Euler angles can represent the same rotation.
    a = torch.linspace(0.2, 1.2, 7, device="cuda", dtype=dtype)
    b = torch.linspace(0.3, 2.3, 7, device="cuda", dtype=dtype)
    c = torch.linspace(-1.1, 0.8, 7, device="cuda", dtype=dtype)
    abc = rotation.compose_angles(a, b, c, a * 0.4, b * 0.7, c * 0.3)
    expected = reference_rotation.angles_to_matrix(a, b, c) @ reference_rotation.angles_to_matrix(
        a * 0.4, b * 0.7, c * 0.3)
    torch.testing.assert_close(rotation.angles_to_matrix(*abc), expected, rtol=2e-4, atol=2e-4)
    inverse = rotation.inverse_angles(a, b, c)
    eye = torch.eye(3, device="cuda", dtype=dtype).expand(7, 3, 3)
    torch.testing.assert_close(rotation.angles_to_matrix(a, b, c)
                               @ rotation.angles_to_matrix(*inverse), eye,
                               rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotation_xyz_and_quaternion_matrix(dtype):
    torch.manual_seed(47)
    abc = reference_rotation.rand_angles(13, dtype=dtype, device="cuda")
    matrix = rotation.angles_to_matrix(*abc)
    quaternion = rotation.angles_to_quaternion(*abc)
    torch.testing.assert_close(rotation.quaternion_to_matrix(quaternion), matrix,
                               rtol=1e-4, atol=1e-4)
    pos = matrix @ torch.tensor([0., 1., 0.], device="cuda", dtype=dtype)
    torch.testing.assert_close(rotation.angles_to_xyz(*abc[:2]), pos,
                               rtol=1e-4, atol=1e-4)
    xyz_angles = rotation.xyz_to_angles(pos)
    torch.testing.assert_close(rotation.angles_to_xyz(*xyz_angles), pos,
                               rtol=1e-4, atol=1e-4)

