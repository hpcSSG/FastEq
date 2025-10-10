import torch
import os

from pathlib import Path
from torch.utils.cpp_extension import load


# Set environment variables for CUDA
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
#so_path = os.path.join(os.path.dirname(__file__), "_kernels/cuda/build/bin/", "libfasteq.so")
#torch.ops.load_library(so_path)

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
