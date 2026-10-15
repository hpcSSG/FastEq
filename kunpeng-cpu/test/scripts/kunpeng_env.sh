#!/usr/bin/env bash
# Source in an allocated compute-node job after activating the Python venv.
# This configures standalone FastEq CPU tests, not the full batchASE/EVD job.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-37}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/cpu_env.sh"

# Directory names are historical site configuration, not verified tool versions.
export KUNPENG_COMPILER_ROOT="${KUNPENG_COMPILER_ROOT:-/work_ssd/software/soft/tool/hpckit/2025.0.0/HPCKit/25.0.0/compiler/gcc}"
export KUNPENG_OPENBLAS_ROOT="${KUNPENG_OPENBLAS_ROOT:-/work_ssd/software/soft/lib/openblas/0.3.18-hpckit25.1.0.SPC001-gcc12.3.1}"
if [ -x "$KUNPENG_COMPILER_ROOT/bin/g++" ]; then
    export CXX="${CXX:-$KUNPENG_COMPILER_ROOT/bin/g++}"
fi
if [ -x "$KUNPENG_COMPILER_ROOT/bin/gcc" ]; then
    export CC="${CC:-$KUNPENG_COMPILER_ROOT/bin/gcc}"
fi
if [ -z "${NATIVE_CPU_BLAS_INCLUDE:-}" ] && [ -z "${NATIVE_CPU_BLAS_LIB:-}" ] && \
   [ -f "$KUNPENG_OPENBLAS_ROOT/include/cblas.h" ] && [ -d "$KUNPENG_OPENBLAS_ROOT/lib" ]; then
    export NATIVE_CPU_BLAS_INCLUDE="$KUNPENG_OPENBLAS_ROOT/include"
    export NATIVE_CPU_BLAS_LIB="$KUNPENG_OPENBLAS_ROOT/lib"
    export NATIVE_CPU_BLAS_NAME="${NATIVE_CPU_BLAS_NAME:-openblas}"
fi
if [ -n "${NATIVE_CPU_BLAS_LIB:-}" ]; then
    export LD_LIBRARY_PATH="$NATIVE_CPU_BLAS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ -d "$KUNPENG_COMPILER_ROOT/lib64" ]; then
    export LD_LIBRARY_PATH="$KUNPENG_COMPILER_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Isolate JIT caches by user, job and rank. Do not change CPU affinity here.
if [ -n "${CCS_JOB_ID:-}" ] && [ -z "${TORCH_EXTENSIONS_DIR:-}" ]; then
    export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp}/fasteq_${USER:-user}_${CCS_JOB_ID}_rank_${CCS_JOB_STEP_RANK:-0}"
fi
