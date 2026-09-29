"""Static contract for the first Triton migration batch.

Run with plain Python when PyTorch and a GPU are unavailable.
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "e3nn" / "triton"
EXPECTED = {
    "o3/_angular_spherical_harmonics.py": {
        "SphericalHarmonicsAlphaBeta.forward", "spherical_harmonics_alpha", "_mul_m_lm"
    },
    "o3/_rotation.py": {
        "identity_angles", "inverse_angles", "identity_quaternion", "compose_quaternion",
        "inverse_quaternion", "compose_axis_angle", "matrix_x", "matrix_y", "matrix_z",
        "angles_to_matrix", "angles_to_quaternion", "axis_angle_to_quaternion",
        "quaternion_to_axis_angle", "axis_angle_to_matrix", "quaternion_to_matrix",
        "angles_to_xyz", "xyz_to_angles",
    },
    "o3/_s2grid.py": {"_quadrature_weights", "s2_grid", "_expand_matrix", "ToS2Grid.grid", "FromS2Grid.grid"},
    "o3/_so3grid.py": {"SO3Grid.to_grid", "SO3Grid.from_grid"},
    "math/_bessel.py": {"bessel"},
    "math/_linalg.py": {"direct_sum"},
    "math/_soft_one_hot_linspace.py": {"soft_one_hot_linspace"},
    "math/_soft_unit_step.py": {"_SoftUnitStep.forward", "_SoftUnitStep.backward"},
    "math/perm.py": {"natural_representation"},
}


def test_inventory():
    for filename, expected in EXPECTED.items():
        tree = ast.parse((ROOT / filename).read_text())
        actual = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                actual.add(node.name)
            elif isinstance(node, ast.ClassDef):
                actual.update(f"{node.name}.{child.name}" for child in node.body if isinstance(child, ast.FunctionDef))
        assert expected <= actual, (filename, sorted(expected - actual))


if __name__ == "__main__":
    test_inventory()
