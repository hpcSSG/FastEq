"""Catch Python math calls that Triton cannot lower from device code."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "e3nn" / "triton"


def test_no_python_sqrt_in_jit():
    bad = []
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list):
                continue
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name) and call.func.value.id == "math"
                        and call.func.attr == "sqrt"):
                    bad.append(f"{path.name}:{call.lineno}")
    assert not bad, bad


if __name__ == "__main__":
    test_no_python_sqrt_in_jit()
