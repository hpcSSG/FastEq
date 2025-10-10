import torch
import os

from pathlib import Path
from threading import Lock

# ---- 全局状态控制 ----
_LOADED_LIBS = set()
_ENV_SET = False
_LOCK = Lock()

'''
# Set environment variables for CUDA
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
so_path = os.path.join(os.path.dirname(__file__), "_kernels/cuda/build/bin/", "libfasteq.so")
torch.ops.load_library(so_path)
'''

def load_fasteq_once():
    global _ENV_SET
    with _LOCK:
        if not _ENV_SET:
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
            _ENV_SET = True

        so_path = Path(__file__).parent / "_kernels" / "cuda" / "build" / "bin" / "libfasteq.so"
        so_path = so_path.resolve()

        if not so_path.exists():
            raise FileNotFoundError(f"[ERROR] Shared library not found: {so_path}")

        if str(so_path) not in _LOADED_LIBS:
            torch.ops.load_library(str(so_path))
            _LOADED_LIBS.add(str(so_path))
            print(f"[INFO] Loaded CUDA extension: {so_path}")
        else:
            pass

        return so_path

load_fasteq_once()
_loaded_kernels = {}
def load_kernel_from_lib(name: str):
    
    if name not in _loaded_kernels:
        print("Load CUDA kernel:", name)
        if name == "stc_fwd":
            _loaded_kernels[name] = torch.ops.stc_fwd
        elif name == "stc_bwd":
            _loaded_kernels[name] = torch.ops.stc_bwd
        else:
            raise ValueError(f"Unknown kernel name: {name}")
    return _loaded_kernels[name]
