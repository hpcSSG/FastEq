#!/usr/bin/env bash
# Source after activating your Python environment; paths are site-specific.
export FASTEQ_BACKEND=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-2}"
# Set CXX to your C++ compiler if needed.
# Set BOTH NATIVE_CPU_BLAS_INCLUDE and NATIVE_CPU_BLAS_LIB for OpenBLAS.
# Optional: export TORCH_EXTENSIONS_DIR=/path/to/writable/cache
