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
- FP64: impl 6 (`kAuto4`); backward uses dgrad B1. The FP64 source line's production entry impl 5
  (`kAuto3`) and the other auto variants produced wrong output on the 17-path
  `LARGE_P3` case in the vendored build; impl 6 is retained because it passes the supported
  forward and grad_x accuracy checks.

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
export TMPDIR=/app/work/fasteq-pub/tmp_dev
export CUDA_CACHE_PATH=/app/work/fasteq-pub/tmp_dev/cuda_cache
FASTEQ_BACKEND=cuda timeout --signal=INT --kill-after=10s 600s \
  uv pip install --python /app/work/fasteq_venv/bin/python \
  --no-build-isolation --no-deps --reinstall -e .
```

The build produces `_cuda.so` plus `libfasteq_eqlinear_{tf32,fp32,fp64}.so`; `fasteq.cuda`
preloads the companion libraries before loading `_cuda.so`.

## Source layout

`fasteq/cuda/src/eqlinear/` keeps the precision backends under `backends/{tf32,fp32,fp64}/` and the
cross-precision code under `common/`.  Shared pieces are the group table, the A/B packed index
tables, the float prep kernels (TF32+FP32) and the traits-driven C ABI adapter; each backend keeps
its own row plan, runner and CUTLASS kernels.  Unreachable wgrad/split-K implementations are
archived under `experimental/` and are never compiled.  See
[`fasteq/cuda/src/eqlinear/README.md`](../fasteq/cuda/src/eqlinear/README.md) for the file map and
the rules for adding shared code.

## Tests

```bash
/app/work/fasteq_venv/bin/python -m pytest -q \
  test/test_eqlinear.py test/test_eqlinear_real_descriptor.py \
  test/test_eqlinear_descriptor_mutation.py test/test_eqlinear_lifetime.py \
  test/test_eqlinear_weight_layout.py test/test_eqlinear_benchmark_io.py
```

Use the existing editable environment for later rebuilds; do not create a new environment
for each version. On a shared GPU machine, choose an idle device before tests and wrap
pytest, compare and smoke runs with appropriate timeouts.

## Performance interpretation

The retained implementation uses the P14 host fast paths with the P27 correctness fixes.
The merge accepts a three-percentage-point tolerance for improvement targets (or 3%
relative latency/overhead where applicable), as authorized by the user. P29's strict
spread and cross-round stability checks failed; acceptance does not mean the measurements
became stable. Code is unchanged and performance is inherited from P29, not a new scan.

The original CUDA f32 operator is one-pass TF32; this implementation uses tf32x3 by default
and a separate strict FP32 path. Original/public/prepared timings therefore describe
both API overhead and different arithmetic work, not same-precision speedups. Raw and
tolerance-adjusted results, the original CUDA gap, and queued baseline limitations are
recorded in [the merge report](eqlinear_merge_report.md).
