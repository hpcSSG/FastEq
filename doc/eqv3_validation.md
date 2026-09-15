# EQv3 operator validation

Archived performance measurements: FastEq **`3bc9a82d40ef646c3ce75f6f517e3f618fb12754`** on an H100 80GB, FP32, PyTorch 2.11.0+cu128 and Triton 3.6.0.
Operator files were unchanged during those measurements. The current scope cleanup was validated separately; see [regression.json](eqv3_validation/2026-09-15/regression.json). Timings were not rerun. Native EQv3 source files match commit `a7300c58df683dc99cb48027d5bfd4c887486c48` byte for byte.

**Validation is incomplete at the implementation level:** AttentionAlpha has large-N parameter-gradient failures; GraphSoftmax intermittently fails a broadcast-rescale gradient check; the EQv3-specific `fused_gate_activation` public class has no `forward()` method. Successful configurations do not certify these failing paths.

## Representative measurements

N=4096. Ratios below are Torch time / FastEq time for the public operator call; values below 1 indicate a slowdown. `F+B` includes forward and all first-order input/parameter gradients, without an optimizer. Both timing modes include the actual Torch work inside hybrid backward implementations.

| Operator | Original Torch reference | Correctness coverage | Forward | F+B |
|---|---|---|---:|---:|
| GraphSoftmax | EQv3 GraphSoftmax | Failures at other configurations | 4.82x | 1.23x |
| AttentionAlpha | EQv3 attention block | Failures at other configurations | 3.43x | 0.75x |
| e3nn Gate | e3nn.nn.Gate | Tested configurations pass | 2.17x | 3.11x |
| LayerNorm | EQv3 EquivariantLayerNorm | Tested configurations pass | 2.92x | 1.85x |
| SeparableLayerNorm | EQv3 EquivariantSeparableLayerNorm | Tested configurations pass | 1.55x | 1.40x |
| EquivariantDropout | EQv3 EquivariantDropout | Tested configurations pass | 0.61x | 0.85x |

GraphSoftmax ratios above reuse a prepared CSR graph. Per-call CSR construction, including host synchronization, is measured separately through N=65536; consult the `with_csr_wall_speedup` column in the CSV before using the cached-graph ratio for an integration decision.
The e3nn Gate result does not validate EQv3 GateActivation, which is a different interface and currently fails to run.
![Measured speedup at N=4096](eqv3_validation/2026-09-15/speedup_4096.png)

## What is fused

| Operator | Actual execution |
|---|---|
| GraphSoftmax | Soft cap, neighborhood max, exponential, rescale/dropout, neighborhood sum and division. CSR preparation is separate; broadcast parameter gradients also use a Torch reduction. |
| AttentionAlpha | Forward LayerNorm, scalar activation, dropout and weighted channel reduction. Backward recomputes Torch expressions with autograd. |
| e3nn Gate | Scalar activation, gate activation, component mapping and multiplication; Triton first backward. |
| Two EQv3 LayerNorm variants | Shared scalar centering, grouped statistics, inverse standard deviation and affine output. Default backward combines native Torch statistic/parameter reductions with shared Triton dX/partial kernels. Groups are per degree or scalar/higher degrees. |
| EquivariantDropout | Shared random-mask generation and multiplication; backward reuses the same seeded mask. |

## Accuracy and scale

After removing the out-of-scope adapter, the LayerNorm suite passed **218 tests with zero failures or skips** against unmodified V2 and V3 references. This is a new CUDA correctness run; the separate scaling results retain their original measurement revision. Separate scaling checks compare every output and requested gradient element against the native Torch implementation; results and 32 numeric samples per tensor are retained.
The scaling/small-shape check records contain **93 passing cases and 2 failing cases**. Extra API/layout/dropout tests are listed separately.
For GraphSoftmax, a one-off rerun passed after the initial failure. Five further runs against the original EQv3 class contained one failing assertion out of 1720: E=73, N=17, H=3, cap=3, eps=1e-16, per-head rescale [1,3], dropout=0. The gradient tolerance ratio was 1.016865. The failing inputs and gradients are saved; the intermittent failure has not been fixed.
Tolerance is `abs(actual-reference) <= atol + rtol*abs(reference)`: 5e-5/5e-4 normally; GraphSoftmax uses 3e-6/3e-5 forward and 3e-5/3e-4 backward; dropout is exact with the same sampled mask. Tolerances were not relaxed.
N doubles from 256. Each backend/mode/shape runs in a fresh process, with 5 warmups and 20 timed repetitions. Reported GPU-event intervals include host submission gaps; synchronous wall-clock medians are also retained. This is a shared host without locked clocks, not an exclusive machine reservation.
LayerNorm/Dropout use [N,16,128]; Gate uses [N,2432] -> [N,2048]. Graph operators use E=32N and 8 heads; AttentionAlpha has 32 channels. Main dropout timing is active p=0.3 under training mode, including for no_grad forward. Attention dropout is zero in the main sweep.
The run continues until each path reaches actual OOM, its explicit indexing limit, an address-overflow guard in the harness, or a public Torch fallback. A successful execution beyond Torch's memory ceiling does not constitute a matched accuracy check. Peak memory is allocator-allocated memory including inputs/results/gradients/temporaries, not total board occupancy.
Index-limit, address-guard and fallback rows are harness preflight stops based on the verified source conditions; oversized inputs were not allocated at those rows. OOM rows contain the actual allocation failures.
![Scaling speedup](eqv3_validation/2026-09-15/speedup_scaling.png)

## CUDA / HIP scope

Performance measurements above are CUDA/H100 results. The archived BW gfx936 LayerNorm record matches the implementation, tests, references and artifacts at the measurement revision: **305 passed, zero skipped** on Torch 2.7.1, HIP 6.3.26045 and HCU Triton 3.1.0. Its N=4096 timings are retained separately. The scope cleanup was not rerun on HIP. This run did not repeat the HIP sweep or validate the other operator families on HIP.
These are standalone operator checks, not complete EQv3 inference/training measurements. The main sweep tests first-order gradients; LayerNorm still rejects double backward.

## Records

- [All paired timings, speedups and memory](eqv3_validation/2026-09-15/paired.csv)
- [Execution/OOM/index/fallback boundaries](eqv3_validation/2026-09-15/boundaries.csv)
- [Accuracy summary](eqv3_validation/2026-09-15/correctness_summary.json) and [scaling failures](eqv3_validation/2026-09-15/failures.csv)
- [GraphSoftmax native repeated checks](eqv3_validation/2026-09-15/softmax_native_checks.json), [one-off rerun](eqv3_validation/2026-09-15/softmax_diagnosis.json) and [extra checks](eqv3_validation/2026-09-15/extra_checks.json)
- [Reused HIP LayerNorm timings](eqv3_validation/2026-09-15/hip_layernorm_reused.csv)
- [Current LayerNorm regression](eqv3_validation/2026-09-15/regression.json)
- [Raw point/check records](eqv3_validation/2026-09-15/raw_results.json.gz), [source manifest](eqv3_validation/2026-09-15/manifest.json) and [reproduction](eqv3_validation/2026-09-15/REPRODUCE.md)

Full local artifact directory: `/public-data/zhouxibo/zxb/equiformer_v3_baseline/artifacts/eqv3_operators_20260915`.
