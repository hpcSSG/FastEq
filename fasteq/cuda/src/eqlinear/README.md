# eqlinear source layout

```
fasteq/cuda/src/eqlinear/
  CMakeLists.txt                  single build file for the three backend DSOs
  common/                         shared, CUTLASS-free where possible
    eqlinear_common.h             Precision / Status / EqLinearProblemC / PlanInfo
    eqlinear_backend_api.h        C ABI vtable and export macro
    eqlinear_group_table.{h,cpp}  GroupTable, build_group_table, out_total_i
    packed_index_tables.{h,cpp}   shared A/B packed column maps + per-group spans
    eqlinear_prep_kernels.{h,cu}  templated permute/pack kernels (float instantiation)
    backend_adapter.h             traits-driven implementation of the C ABI vtable
    pointer_table.cuh             device pointer-table helpers
    device_lifetime.h             DeviceOwner / DeviceFence / DeviceBuffer
  backends/tf32|fp32|fp64/
    api.cpp                       precision traits + one exported C ABI entry
    dgrad.cu                      production B1 grad_x
    runner.cu                     forward orchestration and CUTLASS plan
    eqlinear.h / bwd.h            backend-internal interfaces
    row_plan.h                    backend-specific row layout (deliberately not shared)
    prep_kernels.h                alias header into common for tf32/fp32
    <CUTLASS kernels / collectives / fork-specific headers>
  dispatch/eqlinear_dispatch.cpp  torch-facing validation, handle registry, 12 ops
  experimental/                   archived unreachable implementations, never compiled
```

## Rules

- `common/` owns logic that is identical across precisions.  If a backend needs a different
  arithmetic order, alignment rule or row layout, keep it in that backend rather than adding
  branches to a shared abstraction.
- Each backend is a separate DSO (`libfasteq_eqlinear_<precision>.so`) with hidden visibility and
  one exported symbol.  The frozen CUTLASS forks must not share a symbol table.
- `add_subdirectory(src/eqlinear)` stays before `find_package(Torch)` in the parent CMake file so
  Torch cannot rewrite `sm_90a`.
- New files must be added to the single root `CMakeLists.txt`.  Nested `CMakeLists.txt` files would
  need a new exception to the repository's `*.txt` ignore rule.
- Archived code goes under `experimental/` with an entry in its README; it must not be referenced
  from `backends/` or `common/`.
