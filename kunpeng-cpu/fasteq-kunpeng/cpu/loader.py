"""Compile the packaged C++ source once using PyTorch's extension cache."""
import os
import sys
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def load_extension():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The native CPU build currently targets Linux; validate on Kunpeng Linux.")
    from torch.utils.cpp_extension import load
    flags = ["-O3", "-fopenmp", "-march=native", "-funroll-loops"]
    if os.environ.get("FASTEQ_CPU_FAST_MATH", "1") == "1":
        flags += ["-ffast-math", "-fno-math-errno", "-fno-trapping-math"]
    includes, links = [], ["-fopenmp"]
    include = os.environ.get("NATIVE_CPU_BLAS_INCLUDE")
    lib = os.environ.get("NATIVE_CPU_BLAS_LIB")
    if bool(include) != bool(lib):
        raise ValueError("Set both NATIVE_CPU_BLAS_INCLUDE and NATIVE_CPU_BLAS_LIB, or neither.")
    if include:
        flags.append("-DNATIVE_CPU_USE_OPENBLAS")
        includes.append(include)
        links += [f"-L{lib}", f"-Wl,-rpath,{lib}", "-l" + os.environ.get("NATIVE_CPU_BLAS_NAME", "openblas")]
    return load(
        name="fasteq_native_equiv_cpu",
        sources=[str(Path(__file__).parent / "src" / "native_equiv_cpu.cpp")],
        extra_include_paths=includes, extra_cflags=flags, extra_ldflags=links,
        verbose=os.environ.get("FASTEQ_BUILD_VERBOSE", "0") == "1",
    )
