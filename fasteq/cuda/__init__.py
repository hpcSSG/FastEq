import ctypes
import os, glob
import torch

_this_dir = os.path.dirname(__file__)
cands = glob.glob(os.path.join(_this_dir, "_cuda*.so"))
if not cands:
    raise ImportError(f"Cannot find _cuda*.so under: {_this_dir}")

# eqlinear is split into one DSO per precision so the frozen CUTLASS forks never share a symbol
# table.  Preload them before _cuda.so; its $ORIGIN RPATH would also resolve them.
_companions = (
    "libfasteq_eqlinear_tf32.so",
    "libfasteq_eqlinear_fp32.so",
    "libfasteq_eqlinear_fp64.so",
)
for _name in _companions:
    _path = os.path.join(_this_dir, _name)
    if not os.path.exists(_path):
        raise ImportError(
            f"Cannot find eqlinear companion {_name} under: {_this_dir}. "
            "Rebuild with: FASTEQ_BACKEND=cuda pip install -e . --no-build-isolation"
        )
    ctypes.CDLL(_path)

print(f"[Debug] torch.ops.load_library from{cands}")

so_path = max(cands, key=os.path.getmtime)
torch.ops.load_library(so_path)
print(f"[Debug] torch.ops.load_library from{so_path}")
