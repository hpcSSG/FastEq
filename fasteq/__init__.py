# fasteq/__init__.py
from __future__ import annotations

import os
import importlib
import torch


def _detect_backend() -> str:
    """
    Returns: "cuda" | "hip" | "cpu"
    Priority:
      1) FASTEQ_BACKEND env override: cuda/hip/cpu
      2) torch ROCm build + HIP available -> hip
      3) torch CUDA build + CUDA available -> cuda
      4) cpu
    """
    override = os.environ.get("FASTEQ_BACKEND", "").strip().lower()
    if override in {"cuda", "hip", "cpu"}:
        return override

    hip_ver = getattr(torch.version, "hip", None)
    cuda_ver = getattr(torch.version, "cuda", None)

    # ROCm build (torch.version.hip is not None / not empty)
    if hip_ver:
        # torch.cuda.is_available() is True on ROCm too (it means GPU available via HIP runtime)
        if torch.cuda.is_available():
            return "hip"
        return "cpu"

    # CUDA build
    if cuda_ver:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    return "cpu"


_BACKEND = _detect_backend()

# Import selected backend module
if _BACKEND == "cuda":
    _mod = importlib.import_module("fasteq.cuda")
elif _BACKEND == "hip":
    _mod = importlib.import_module("fasteq.hip")
else:
    _mod = None  # optional: provide a pure python fallback

# Re-export a unified API (pick the symbols you want)
# Example: assume both fasteq.cuda and fasteq.hip expose `ops` or same function names.
if _mod is not None:
    # If both backends define the same public names, you can do:
    # from .cuda import *  / from .hip import *  (not recommended)
    # Better: explicitly re-export what you need:
    ops = getattr(_mod, "ops", _mod)
    __all__ = ["ops", "_BACKEND"]
else:
    __all__ = ["_BACKEND"]
