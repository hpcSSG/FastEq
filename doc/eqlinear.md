# eqlinear

CUTLASS-backed eqlinear forward and grad_x. The public Python entry point is unchanged:

```python
from fasteq.ops.so3_linear import fast_equi_linear

out = fast_equi_linear(descriptor, w, x)                 # FP32 -> tf32x3 by default
out = fast_equi_linear(descriptor, w, x, compute_precision="fp32")  # strict FP32
out = fast_equi_linear(descriptor, w, x, compute_precision="tf32")  # explicit tf32x3
```

## Precision routing

| x / w dtype | `compute_precision` | kernel |
|---|---|---|
| float64 / float64 | ignored | FP64 DMMA |
| float32 / float32 | `None` or `"tf32"` | TF32 tf32x3 |
| float32 / float32 | `"fp32"` | FP32 CUDA-core FFMA |
| mixed / fp16 / bf16 / other | any | `ValueError` |

FP32 inputs default to tf32x3; pass `compute_precision="fp32"` when strict FP32 numerics are
required. FP64 inputs always use the FP64 kernel.

## Frozen backends

- TF32: impl 5 (`sm90_3x_rs`) except `P == 4 and B <= 1024 and V <= 128`, which uses impl 0
  (CUTLASS 2.x SM80 tf32x3); backward uses dgrad B1.
- FP32: impl 3 (`kFused`); backward uses dgrad B1.
- FP64: impl 6 (`kAuto4`); backward uses dgrad B1.

The backends do not read runtime `EQLIN_*` or `FASTEQ_EQLINEAR_*` environment variables.
TF32 impl 0 reads the activation directly when each output row has one input path;
rows with multiple paths use the packed activation. Impl 5 retains its in-kernel split
activation view, and K-concatenated variants retain their packed path. These defaults
are compiled in. Tests exercise both the one-path view and the multi-path fallback
through their geometry; no runtime test switch is exposed.
Prepared plans are cached per weight tensor (identity + `_version`), shape and stream, so
`prepare_weight` is hoisted out of the steady-state call. `eqlinear_clear_plan_cache()` clears the
cache; `eqlinear_set_plan_cache(max_bytes)` sets its budget (0 disables caching).

Destroying a prepared plan explicitly waits for preceding work on its creation device
before freeing its pointer tables, prepared operands and workspaces. This protects
already queued forward and grad_x calls, including calls on non-default streams.
Destruction and cache eviction can therefore wait for other work on that device;
the same lifetime owner covers one-shot calls. Each entry selects the tensor/plan device
and restores the caller's current device. Inputs and outputs must belong to the plan device.
Independent streams use independent handles; concurrent unsynchronized runs sharing one
handle's mutable workspace are unsupported. Cleanup errors are reported without throwing
from destructors; if completion cannot be established, resources are retained until process exit.

Geometry caching accepts the canonical cuEquivariance descriptor, operand and Path classes
with immutable tuple segments and ordinary integer dimension sets. Every lookup retrieves
the current dimension dict and checks its identity and values. Dict replacement, structural
changes and unsupported types use full parsing. Coefficient-only changes can reuse validated
geometry when dimensions remain identical; coefficients are still read for each call.
If the canonical classes cannot be imported, geometry caching is disabled.

## Shape contract

- `x` is `[B, sum(i_dims)*U]` and is viewed as `[B, sum(i_dims), U]`.
- `w` contains `P*U*V` elements and is viewed as `[P, U, V]`.
- The flat no-grad fast path requires both `x` and `w` to be contiguous. Non-contiguous
  weights use the 3D `reshape().contiguous()` fallback, which prepares a contiguous copy.
  Plan keys use the original weight's actual shape/stride, identity and version.
- `out` is `[B, J*V]` with `J = out_total_i(i_dims)` (run-length grouping of equal i).
- TF32/FP32 require `B, V, I*U, P*U, J*V` to be multiples of 4; FP64 requires multiples of 2.
- FP32 `kFused` additionally requires `U == V`, `U % 16 == 0`, and `V in {96, 128, 224}`.
- The GPU must be Hopper (`sm_90a`).
- Unsupported shapes raise a precise exception; there is no Triton fallback.

## Build

CUTLASS is a submodule under `3rdparty/cutlass`, pinned to
`0b55a2f691d69981583568fd9eb69687b1f0de8a` (v4.8.0-3). Build with:

```bash
git submodule update --init 3rdparty/cutlass
FASTEQ_BACKEND=cuda pip install -e . --no-build-isolation
```

The build produces `_cuda.so` plus `libfasteq_eqlinear_{tf32,fp32,fp64}.so`; `fasteq.cuda`
preloads the companion libraries before loading `_cuda.so`.

## Source layout

`fasteq/cuda/src/eqlinear/` contains:

- `common/`: group table, A/B packed index tables, shared TF32/FP32 prep kernels, C ABI adapter
  and device lifetime helpers;
- `backends/{tf32,fp32,fp64}/`: per-precision C ABI entry, runner, dgrad, row plan and CUTLASS
  kernels;
- `dispatch/`: torch-facing validation, handle registry and the 12 operator definitions.

Each precision builds its own `libfasteq_eqlinear_<precision>.so` with hidden visibility; the
three CUTLASS forks never share a symbol table.

## Tests

```bash
python -m pytest -q \
  test/test_eqlinear.py test/test_eqlinear_real_descriptor.py \
  test/test_eqlinear_descriptor_mutation.py test/test_eqlinear_lifetime.py \
  test/test_eqlinear_weight_layout.py test/test_eqlinear_benchmark_io.py
```

Run on an idle Hopper (`sm_90a`) GPU.

## Numerics

FP64 inputs use the FP64 kernel. FP32 inputs use tf32x3 by default; pass
`compute_precision="fp32"` for strict FP32 CUDA-core arithmetic, or `"tf32"`/`"tf32x3"` to select
tf32x3 explicitly. The original CUDA f32 operator is one-pass TF32, so timings against it compare
different arithmetic work, not same-precision speedups.
