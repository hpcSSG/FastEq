# eqlinear experimental archive

This directory keeps implementations that are no longer reachable from the public eqlinear entry
points. **Nothing here is listed in CMake and nothing here is compiled or shipped.**

Why not delete it: the TF32/FP32/FP64 source lines and their iteration reports are preserved by the
user's request. The full development history also lives in the local branch
`xjw/eqlinear-prep-opt` and in the per-iteration reports (`M*_REPORT.md`, `TF32_*`, `B2_REPORT.md`,
etc.) kept outside this package.

Contents:

| Path | Status | Why archived |
|---|---|---|
| `tf32/split_k_plan.{h,cu}`, `tf32/reduce_fwd.{h,cu}` | unreachable | only used by `if (char const * = nullptr)` split-K probes that were removed from `backends/tf32/runner.cu` |
| `tf32/bwd.cu` | unreachable | full TF32 backward (B1+B2+B3); production only exposes B1 `DgradRunner`, now in `backends/tf32/dgrad.cu` |
| `tf32/transpose_kt.{h,cu}`, `tf32/reduce_dw.{h,cu}` | unreachable | only used by the archived TF32 wgrad path |
| `fp64/bwd.cu` | unreachable | full FP64 backward (dgrad+wgrad); production only exposes B1, now in `backends/fp64/dgrad.cu` |

`TF32 split_pack.*` is **not** here: it is live and used by the production `kSm90_3xTF32` forward.

To restore an archived implementation for an A/B experiment, move the files back under
`backends/<precision>/`, add them to `fasteq/cuda/src/eqlinear/CMakeLists.txt`, and re-validate
accuracy and performance. Do not wire them into the public C ABI without a new acceptance run.
