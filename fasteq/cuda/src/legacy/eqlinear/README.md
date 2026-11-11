# Legacy eqlinear kernels

Pre-CUTLASS, hand-written CUDA/HIP implementation of the eqlinear forward and grad_x operators.
It is kept for reference and is **not built by the parent `fasteq/cuda/CMakeLists.txt`**.

- `eqlinear_legacy.cu`: old torch glue and `eqlinear_legacy` operator registration.
- `eqlinear_fwd_f32.cu`, `eqlinear_bwd_f32.cu`, `eqlinear_fwd_f64.cu`: CUDA kernels.
- `eqlinear_impl.h`, `eqlinear_impl_hip.h`: raw-pointer entry points.
- `device_*_helper.cuh`, `ptx_inst*.cuh`, `CMakeLists.txt`: helpers and an optional standalone
  build description; the target is not added to the package build.

The active production implementation is `fasteq/cuda/src/eqlinear/`. The legacy FP32 forward is
one-pass TF32 and its FP64 backward falls back to the transposed FP64 forward, so it is not a
drop-in replacement for the current `compute_precision` semantics.

To build it for an experiment, add this directory's `CMakeLists.txt` manually to a separate CMake
project; do not wire it into `fasteq/cuda/CMakeLists.txt` or the public torch namespace.
