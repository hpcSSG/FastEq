# eqlinear

CUTLASS-backed eqlinear forward and grad_x. The public Python entry point is unchanged:

```python
from fasteq.ops.so3_linear import fast_equi_linear

out = fast_equi_linear(descriptor, w, x)                 # FP32 -> tf32x3 by default
out = fast_equi_linear(descriptor, w, x, compute_precision="fp32")  # exact FP32
out = fast_equi_linear(descriptor, w, x, compute_precision="tf32")  # explicit tf32x3
```

## Precision routing

| x / w dtype | `compute_precision` | kernel |
|---|---|---|
| float64 / float64 | ignored | FP64 DMMA |
| float32 / float32 | `None` or `"tf32"` | TF32 tf32x3 |
| float32 / float32 | `"fp32"` | FP32 CUDA-core FFMA |
| mixed / fp16 / bf16 / other | any | `ValueError` |

FP32 inputs default to tf32x3; pass `compute_precision="fp32"` when exact FP32 numerics are
required. FP64 inputs always use the FP64 kernel.

## Frozen backends

- TF32: impl 5 (`sm90_3x_rs`) except `P == 4 and B <= 1024 and V <= 128`, which uses impl 0
  (CUTLASS 2.x SM80 tf32x3); backward uses dgrad B1.
- FP32: impl 3 (`kFused`); backward uses dgrad B1.
- FP64: impl 6 (`kAuto4`); backward uses dgrad B1. The FP64 source line's production entry impl 5
  (`kAuto3`) and the other auto variants were measured to produce wrong output on the 17-path
  `LARGE_P3` case in the vendored build; impl 6 is bit-correct on all 25 measured points and is
  the fastest correct variant (25-point forward 3.95 ms in this integration).

The backends do not read `EQLIN_*` environment variables; the measured defaults are compiled in.
Prepared plans are cached per weight tensor (identity + `_version`), shape and stream, so
`prepare_weight` is hoisted out of the steady-state call. `eqlinear_clear_plan_cache()` clears the
cache; `eqlinear_set_plan_cache(max_bytes)` sets its budget (0 disables caching).

## Shape contract

- `x` is `[B, sum(i_dims)*U]` and is viewed as `[B, sum(i_dims), U]`.
- `w` contains `P*U*V` elements and is viewed as `[P, U, V]`.
- `out` is `[B, J*V]` with `J = out_total_i(i_dims)` (run-length grouping of equal i).
- TF32/FP32 require `B, V, I*U, P*U, J*V` to be multiples of 4; FP64 requires multiples of 2.
- FP32 `kFused` additionally requires `U == V`, `U % 16 == 0`, and `V in {96, 128, 224}`.
- The GPU must be Hopper (`sm_90a`).
- Unsupported shapes raise a precise exception; there is no Triton fallback.

## Build

CUTLASS is a submodule under `3rdparty/cutlass`, pinned to
`0b55a2f691d69981583568fd9eb69687b1f0de8a` (v4.8.0-3). Build with:

```bash
# uv-created venvs do not ship pip:
FASTEQ_BACKEND=cuda uv pip install --python /path/to/venv/bin/python -e . --no-build-isolation
# or, in a venv that has pip:
FASTEQ_BACKEND=cuda python -m pip install -e . --no-build-isolation
```

The build produces `_cuda.so` plus `libfasteq_eqlinear_{tf32,fp32,fp64}.so`; `fasteq.cuda`
preloads the companion libraries before loading `_cuda.so`.

## Tests

```bash
python -m pytest test/test_eqlinear.py -q
```
