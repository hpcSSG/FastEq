# GraphSoftmax and AttentionAlpha diagnosis

This is a new diagnosis of the unchanged operators at FastEq commit
`979c3504b490402603f1bc9816273155046730b4`. It supplements the archived
2026-09-15 and 2026-09-16 sweeps; it does not replace their measurements.

Fresh work ran on **gxn70, physical H100 GPU 5**, Torch 2.11.0+cu128 and
Triton 3.6.0. The GPU was idle before and after the work, but the host was
shared and clocks were not locked. The Hygon job `839547` could not start
because of `AssocGrpGRES`; it was cancelled while pending after the user
confirmed there was no reusable allocation. No new HIP execution is claimed.
Actual runtime/reference hashes are in [manifest.json](manifest.json).

## AttentionAlpha: the failing parameter reduction is not the native contraction

The public operator runs one Triton forward kernel. Its custom backward then
calls `_reference` under `torch.enable_grad()` and runs `torch.autograd.grad`.
That reference uses a manually expanded LayerNorm and `(activation * w).sum(-1)`.
The original EQv3 block uses `torch.nn.LayerNorm` and
`torch.einsum('bik,ik->bi', activation, w)`. These expressions are mathematically
equivalent but have different FP32 computation/reduction paths.

A 2x2 ablation on identical inputs separates normalization from contraction.
The table reports maximum error/tolerance for `alpha_dot` gradients; values
above 1 fail the original `atol=5e-5, rtol=5e-4` criterion.

| LayerNorm expression | Contraction | N=32,768 | N=131,072 |
| --- | --- | ---: | ---: |
| Manual | Multiply then sum (current backward) | **2.626102** | **7.680816** |
| Manual | Native einsum | 0.172189 | 0.316744 |
| Native | Multiply then sum | **2.723365** | **7.751282** |
| Native | Native einsum | 0 | 0 |

The original H100 failures were reproduced at both sizes. Changing only the
contraction removes these failures; changing only LayerNorm does not. Thus the
dominant failing `alpha_dot` error in these cases comes from the contraction's
parameter-gradient accumulation, rather than the fused Triton forward.
Normalization differences still exist, but were smaller in this ablation.
This is a localized result, not proof that every shape/backend will pass.

A diagnostic monkey-patch used native LayerNorm plus einsum for backward
recomputation while retaining the original Triton forward. Complete output,
input and parameter-gradient comparisons passed at N=4,096 / 8,192 / 16,384 /
32,768 / 131,072. The largest error/tolerance across these candidate checks
was 0.063888. The production source was not edited. No HIP, active-dropout or
higher-order candidate acceptance is implied by this experiment.

The existing module self-test compares against its own `_reference`, so it
cannot by itself establish agreement with the native EQv3 contraction.
Original-source tests remain necessary.

## AttentionAlpha: backward discards the forward fusion benefit

GPU-event medians in ms, including the public call and first-order autograd:

| N | Native Torch F+B | Manual Torch F+B | Triton F | Current FastEq F+B | Native-recompute candidate F+B |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 4.5456 | 5.3568 | 0.6811 | 6.0383 | 5.2406 |
| 32,768 | 35.5512 | 40.0390 | 5.0998 | 45.2481 | 40.8205 |
| 131,072 | 142.2696 | 159.8791 | 20.2116 | 180.3166 | 162.6523 |

The current cost is closely explained by **Triton forward + manual Torch
forward/backward recomputation**. For example, 0.6811 + 5.3568 = 6.0379 ms,
versus 6.0383 ms measured. The candidate improves precision and part of the
backward cost, but still runs an extra forward and remains slower than native
Torch. It is not a training-speed fix.

The N=4,096 profiler records 22 `aten::mul` calls in FastEq F+B versus 9 in
native Torch. Current FastEq also expands normalization into multiple reductions
and elementwise operations. A real training optimization needs a backward that
reuses saved forward information and computes gradients without reconstructing
the entire Torch graph. The contraction/reduction acceptance must be retained
when that backward is designed.

## GraphSoftmax: fixed-input replay identifies near-zero cancellation

Both saved fixtures were replayed on H100 with exactly the stored `x`, `index`,
`rescale` and upstream gradient. Their hashes are recorded. Each implementation
was repeated 20 times. Both fixtures use E=73, N=17, H=3, rescale `[1,3]`,
soft cap 3, epsilon `1e-16`, and dropout 0.

| Fixture origin | Triton dr[0,2], all repeats | Torch FP32 range | FP32 tolerance failures / 20 | FP64 CPU closed form |
| --- | ---: | ---: | ---: | ---: |
| H100 | -7.6294e-6 | -4.9591e-5 to -1.1444e-5 | 3 / 20 | -2.759396e-12 |
| Hygon | -2.6703e-5 | -5.5313e-5 to -1.7166e-5 | 0 / 20 | -2.759395e-12 |

The Triton result was repeatable on H100; the native Torch reference varied
for the same input. The current native reference uses scatter reductions.
The stored Hygon input did not fail in these H100 repeats, so a fixture's
original failure is not automatically reproduced on another backend.

For a positive rescale shared by a head, the same factor appears in numerator
and denominator and almost cancels. For one node, let `z_i` be the stabilized
exponential, `g_i` the upstream gradient, and `D = r * sum(z_i) + eps`.
The exact derivative of `sum(g_i * r*z_i/D)` with respect to the shared `r` is:

`eps * sum(g_i*z_i) / D**2`.

The CPU FP64 closed form agrees with the FP64 autograd diagnostic at about
`-2.76e-12` for the failing component. FP32 computes this tiny residual through
subtraction of much larger terms. FastEq's compact `q*(g-sum(g*y))` and native
Torch's division-backward/scatter path round differently; native scatter
execution also introduces observed repeat-to-repeat variation.

The fixture deliberately sets rescale's first head to zero. Its corresponding
gradient is approximately `2.37e16`, while the failing third-head gradient is
near zero. A whole-vector relative L2 metric would hide this issue, so the
per-component absolute tolerance remains relevant.

FP64 here diagnoses cancellation. It does not replace Torch FP32 acceptance:
a more accurate result near zero can itself differ from some native FP32
repeats by more than `3e-5`. No blanket FP64 conversion or tolerance increase
was made. A fix must address the broadcast case, quantify the native-repeat
variation and keep the agreed acceptance criterion explicit.

## GraphSoftmax: graph preparation and submission overhead matter

Fresh H100 public F+B medians, ms. The standard configuration is E=32N, H=8,
per-edge rescale `[E,1]`, soft cap 3 and dropout 0.

| N | Native Torch | FastEq, cached CSR | FastEq, rebuild CSR |
| ---: | ---: | ---: | ---: |
| 4,096 | 0.6486 | 0.3312 | 0.7217 |
| 32,768 | 2.7802 | 0.3229 | 0.7183 |
| 262,144 | 21.2549 | 2.0943 | 3.4432 |

At N=4,096, the separately profiled fused forward kernel took 11.46 us,
the backward kernel 9.98 us, and the remaining Torch reduction 8.86 us.
These are profiler observations, not sums to substitute for benchmark medians.
They show why the public call's Python/autograd/launch submission gaps matter
at small N even when the GPU kernels themselves are short. CSR reconstruction
adds sorting, bookkeeping and host synchronization; it can erase the small-N
benefit. Cached CSR is valid only while the graph topology is unchanged.

At N=262,144, varying the launch geometry gave cached F+B 2.0272 ms with
8 heads/1 warp, versus 2.0894 ms with 8 heads/4 warps. Using 4 heads/1 warp
instead took 6.4774 ms. All measured geometry comparisons passed the standard
case's output and gradient tolerances. This H100 result does not establish an
optimal HCU geometry; changing warp count alone is not a demonstrated cure for
the archived Hygon slowdown.

Hygon-specific steady-state performance still needs an available DCU for
profiling. The source's indirect edge accesses, one-node-per-program layout,
launch geometry and separate rescale-gradient reduction are candidates to
measure, not confirmed causes of the Hygon large-N regression.

## The Hygon launch errors hit a documented configuration limit

The site's HCU Triton launcher calls `hipModuleLaunchKernel` with blockDim.x
=`warp_size * num_warps` (driver.py line 335). The archived configuration uses
64 lanes and 4 warps, or 256 threads per block. HIP documents that each
`gridDim * blockDim` dimension must remain below `2**32`.

| Archived error | gridDim.x | blockDim.x | Product |
| --- | ---: | ---: | ---: |
| AttentionAlpha N=65,536: E*H = N*32*8 | 16,777,216 | 256 | 2**32 |
| GraphSoftmax forward N=16,777,216 | 16,777,216 | 256 | 2**32 |
| EquivariantDropout N=2,097,152, 256-element tile | 16,777,216 | 256 | 2**32 |

These configurations violate the documented launch limit and match the exact
first `invalid argument` points. They are distinct from allocation OOM and
from floating-point accuracy. A shared multidimensional launch layout or
chunked invocation can address the limit without backend-specific mathematical
branches; the proposed change still needs HIP execution to verify its repair.

The read-only site launcher audit is saved in [hip_launch_audit.json](hip_launch_audit.json).

Source: [HIP launch API](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.1/reference/hip_runtime_api/modules/launch_api.html).

## Evidence and reproduction

- [Artifact checksums](checksums.sha256).
- [Manifest](manifest.json), [successful alpha worker statuses](alpha_status.json), and [initial pipeline statuses](pipeline_status.json).
- [AttentionAlpha N=32,768](h100/alpha_32768.json), [N=131,072](h100/alpha_131072.json), and the other three complete runs in `h100/`.
- [Fixed-input GraphSoftmax repeats](h100/graph_4096.json) and [CPU FP64 closed form](h100/graph_closed_form.json).
- [GraphSoftmax timings and geometry checks](h100/graph_perf_262144.json), [profile](h100/profile_4096.json), and the per-operator profile text files in `h100/`.
- [Diagnosis script](diagnose.py), [unchanged benchmark cases](cases.py), [closed-form check](graph_closed_form.py), and both saved fixtures in `fixtures/`.

The initial alpha harness incorrectly bound the case's input list and stopped
before comparisons. That binding was corrected, and all five alpha sizes were
rerun successfully; the earlier harness errors are not operator failures.

To reproduce, use the manifest's source/environment and set `PYTHONPATH` to
the checkout. Set `FASTEQ_BACKEND=cpu` to bypass unrelated package extensions,
`FASTEQ_EQUIFORMER_V3_SOURCE_DIR` to the original EQv3 source directory and
`FASTEQ_REFERENCE_NORM_DIR` to a reference directory (unused by these two
operators but required when importing `cases.py`). Select an available GPU.
From this artifact directory, write each repetition to a new output directory:

```bash
python diagnose.py alpha --n 32768 --out /path/to/new-results
python diagnose.py graph --fixtures fixtures --out /path/to/new-results
python diagnose.py graph_perf --n 262144 --out /path/to/new-results
python diagnose.py profile --n 4096 --out /path/to/new-results
```

No production operator file was modified, and no end-to-end model or force-loss
training was evaluated in this diagnosis.
