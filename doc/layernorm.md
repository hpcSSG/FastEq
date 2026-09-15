# Equivariant LayerNorm

FastEq fuses equivariant normalization for EquiformerV3 and EquiformerV2 layers,
with GPU FP32 forward execution and first-order gradients on CUDA and HIP.

## Supported layers

Inputs have shape `[N, (L + 1)**2, C]`, where `N` is the number of atoms, `L` is
the maximum degree, and `C` is the channel count. Statistics are computed
independently for each atom. Let `q_l` be the mean squared feature value across
the `2*l + 1` components and `C` channels of degree `l`, after scalar centering.

| Model | Native layer | Default normalization statistics |
| --- | --- | --- |
| EquiformerV3 | `EquivariantLayerNorm` | Each degree uses its own `q_l`. |
| EquiformerV3 | `EquivariantSeparableLayerNorm` | Scalars use `q_0`; higher degrees share `mean(q_1, ..., q_L)`. |
| EquiformerV2 | `EquivariantLayerNormArraySphericalHarmonics` | Scalars use `q_0`; higher degrees share `mean(q_1, ..., q_L)`. |

The table uses the native defaults, including equal weighting of degrees where
applicable. The adapter preserves the source layer's normalization options.
Only scalar features (`l = 0`) are centered across channels. Affine scales are learned per degree and channel and shared across
that degree's components. Bias applies only to scalars when the source has it.

## Fused operations

Forward combines scalar mean, centering, squaring, weighted group reduction,
reciprocal square root with epsilon, and affine transformation in one kernel.
Each group supplies its normalization scale to the degrees listed above.

For native source adapters, Torch computes the source-ordered statistic-gradient
reductions before a shared Triton program produces `dX` and per-atom affine
parameter contributions. Torch then sums contributions across atoms, and a
shared Triton kernel writes the degree-shared parameter gradients. The same
code and reduction APIs run on CUDA and HIP; no backend-specific module is
selected. This splits part of backward fusion to preserve native FP32 tolerance
without reimplementing each backend's reduction scheduler.

Group boundaries are device buffers, avoiding unsupported `constexpr` tuple
indexing. Hardware lane width is read from the active Triton target; a
32-channel tile is independent of warp/wavefront width. The implementation uses
ordinary Triton expressions without CUDA libdevice calls or inline assembly.
The supported presets are `per_degree` (the default) and `scalar_high`.

Only requested gradients are computed. Double backward and `create_graph=True`
are unsupported. Source adapters default to `parameter_reduction="native"`.
The optional `parameter_reduction="stable"` policy uses fixed 256-row Triton
reduction trees, with `accumulation="fp32"` or `"fp64"` controlling cross-atom
accumulators. These are explicit numerical policies shared by both backends.
FP64 is a diagnostic reference, not a replacement for Torch FP32 acceptance.

## Usage

```python
from fasteq.triton import from_reference

# source is the original FP32 GPU layer; x has shape [N, (L + 1)**2, C].
op = from_reference(source)
y = op(x.requires_grad_(True))
y.square().mean().backward()
```

The adapter reuses the source parameters, so gradients reach those parameters.
Apply it to the individual normalization layers you want to replace. Both
contiguous NKC storage and KNC storage viewed as `[N, K, C]` are supported,
where `K = (L + 1)**2`. CUDA and HIP FP32 are supported execution targets.

## Validation coverage

All three native layers in the table have been compared against their unmodified
PyTorch FP32 implementations on NVIDIA H100 and Hygon BW/gfx936. Comparisons cover the output,
input gradient, and all affine parameter gradients, including NKC and KNC
storage. See the [native-layer comparison tests](../test/test_triton_equivariant_layer_norm_precision.py)
and [backward tests](../test/test_triton_equivariant_layer_norm_backward.py).

This validation covers individual model layers. Full-model inference and
training with this shared implementation have not been validated. eSEN has
not been directly tested.

## Current scope validation

After reducing the adaptation scope, the remaining suite passed **218 tests,
0 failed, 0 skipped** on H100 with Torch 2.11.0 and Triton 3.6.0. Comparisons use
the unmodified V2/V3 Torch forward and autograd implementations, including large-N
parameter gradients, NKC/KNC layouts, and two SGD steps through supported layers.
See [regression.json](eqv3_validation/2026-09-15/regression.json) for the exact
implementation, test and reference hashes. This cleanup was retested on CUDA;
HIP and performance measurements were not rerun.

The [EQv3 report](eqv3_validation.md) summarizes existing performance and accuracy
records for the retained operator list, with their original measurement revision.

## Historical validation record: 2026-09-15

The following snapshot predates the scope cleanup. Its 305-test count describes
the original suite, not the current one. Raw records are preserved for audit;
see the [snapshot note](layernorm_validation/2026-09-15/README.md).

The shared implementation was developed from FastEq commit
`40ba40e72bee769d74a869bb4a4ba820ee1c55c0`. Both machines tested the identical
`fasteq/triton/fused_equivariant_layer_norm.py`, with SHA256:

```text
f2bc0ea975eac5bf5a16b3297415cc37d896fb8a41d5375e04e57d7ed1461520
```

The final logs and measurements below belong to this source hash. They exclude
earlier experimental implementations and the original CUDA baseline. The
baseline remains available in Git history; production does not select it as a
backend fallback.

### Machines and software

| Item | Hygon | NVIDIA |
| --- | --- | --- |
| Host | `a01r3n07` | `gxn70` |
| Device | Hygon BW, `gfx936:sramecc+:xnack-` | H100 80GB HBM3, `sm_90` |
| Reported device memory | 65,520 MiB | 81,089 MiB |
| Compute units / SMs | 80 CUs | 132 SMs |
| Wavefront / warp width | 64 | 32 |
| Allocation | One DCU, Slurm job `836140`, partition `hx1hdnormal01` | GPU 7 on a shared machine; no exclusive reservation or locked clocks |
| Torch | `2.7.1` | `2.11.0+cu128` |
| HIP | `6.3.26045` | N/A |
| Triton | HCU Triton `3.1.0` | `3.6.0` |
| Full LayerNorm suite | **305 passed, 0 failed, 0 skipped** | **305 passed, 0 failed, 0 skipped** |
| Pytest elapsed time | 227.87 s | 155.20 s |

The Hygon allocation was released after the final suite and performance steps
completed. Test durations describe these runs, not comparative hardware speed.
Validation used an operator-only loader importing the production module while
bypassing unrelated compiled package extensions. A clean package installation
was not part of this run.

### Reference sources and accuracy

EquiformerV2 uses the unmodified official
[`nets/equiformer_v2/layer_norm.py`](https://github.com/atomicarchitects/equiformer_v2/blob/d5ad4be729b56f74012ebb7f097f77c5b00a1004/nets/equiformer_v2/layer_norm.py)
at commit `d5ad4be729b56f74012ebb7f097f77c5b00a1004`, SHA256
`fdd7955f1fc3ef4e0a0ccbfb761128b5b269ad09f50423428f77aee12a53480c`.
All 24 previously skipped V2 source-comparison cases ran and passed on both
devices.

EquiformerV3 uses the unmodified original file
`experimental/models/equiformer_v3/layer_norm.py` from the Hygon source tree
labelled `equiformer_v3-a7300c58df683dc99cb48027d5bfd4c887486c48`, SHA256
`41e00b377cbacef4713d88697c06e419f3b8c208d9fcc55be78b53fd5a23c2c6`.
The directory label was recorded; its Git commit was not independently verified.
The file hash identifies the actual reference used.

The three complete test modules were
[`test_triton_equivariant_layer_norm.py`](../test/test_triton_equivariant_layer_norm.py),
[`test_triton_equivariant_layer_norm_backward.py`](../test/test_triton_equivariant_layer_norm_backward.py),
and [`test_triton_equivariant_layer_norm_precision.py`](../test/test_triton_equivariant_layer_norm_precision.py).
They cover forward output, `dX`, all affine parameter gradients, NKC/KNC layouts,
irregular channel widths, severe cancellation, small epsilon, and the precision
regressions at N=32,768 / 131,072 / 262,144 / 524,288. These large-N checks are
accuracy regressions, not a performance sweep to OOM.

Native Torch FP32 comparisons normally require elementwise:

```text
abs(actual - reference) <= 5e-5 + 5e-4 * abs(reference)
```

Existing stricter mathematical and backward regression tolerances are retained.
Mean, moment, and rstd alignment tests now use the same tolerance on both
backends; bitwise equality is not required. For native comparison cases that
record `max_tolerance_ratio`, define the ratio as the maximum of
`abs(actual-reference) / (5e-5 + 5e-4*abs(reference))`; a value at most 1 passes.
The maxima below summarize those recorded properties (335 per device), not the
separate tests with stricter tolerances:

| Largest recorded tolerance ratio | Hygon | H100 |
| --- | ---: | ---: |
| Forward output | 0.013968 | 0.013116 |
| Input gradient `dX` | 0.007151 | 0.007277 |
| Affine parameter gradients | 0.277466 | 0.631784 |

The largest parameter-gradient ratio occurred at V3 `EquivariantLayerNorm`,
N=4096, NKC on Hygon, and V2 `EquivariantLayerNormArraySphericalHarmonics`,
N=4096, NKC on H100. Exact values and test identifiers are recorded in
[`validation.json`](layernorm_validation/2026-09-15/validation.json).

### Representative performance

Measurements use **N=4096, Lmax=3, C=128**, shape `[4096, 16, 128]`, FP32 and
NKC layout, with seed `20260914`. Each case has 5 warmups and 20 measured
iterations; the tables show median GPU-event time in milliseconds. The
baseline is the original eager Torch layer. Forward runs under `no_grad`;
forward + backward computes `dX` and every affine parameter gradient, with no
optimizer step.

Within each device, both implementations use the same input, parameters, and
upstream gradient. All representative cases passed output and gradient checks
before their timings were recorded. The recorded harness measures Torch
followed by the unified implementation for each operator and mode. These are
representative measurements rather than randomized, repeated benchmark trials.
Software versions and GPU random streams differ, and H100 was shared, so the
tables support within-device implementation comparisons, not hardware rankings.

“Unified” below means the shared Triton implementation including its Torch
backward products and reductions. Speedup is Torch time divided by unified
time; these numbers include the cost of the partial backward fusion split.

#### Hygon BW: one DCU

| Native layer | Torch forward (ms) | Unified forward (ms) | Speedup | Torch forward + backward (ms) | Unified forward + backward (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 LayerNorm | 0.8011 | 0.3758 | 2.13× | 2.7340 | 1.8670 | 1.46× |
| V3 Separable | 0.7174 | 0.3911 | 1.83× | 2.1048 | 1.2849 | 1.64× |
| V2 SphericalHarmonics | 0.6305 | 0.2951 | 2.14× | 2.4856 | 1.8120 | 1.37× |

#### NVIDIA H100

| Native layer | Torch forward (ms) | Unified forward (ms) | Speedup | Torch forward + backward (ms) | Unified forward + backward (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 LayerNorm | 0.2429 | 0.1233 | 1.97× | 1.0427 | 0.6991 | 1.49× |
| V3 Separable | 0.1916 | 0.1291 | 1.48× | 0.7198 | 0.5266 | 1.37× |
| V2 SphericalHarmonics | 0.2033 | 0.1096 | 1.85× | 0.9617 | 0.6991 | 1.38× |

Raw timings, peak allocated memory, and the per-case correctness measurements
are in [`perf_hip.json`](layernorm_validation/2026-09-15/perf_hip.json) and
[`perf_cuda.json`](layernorm_validation/2026-09-15/perf_cuda.json). Final pytest
logs are [`unified_hip.log`](layernorm_validation/2026-09-15/unified_hip.log) and
[`unified_cuda.log`](layernorm_validation/2026-09-15/unified_cuda.log).

These historical representative measurements alone do not establish a maximum
atom count or OOM boundary. See the [EQv3 report](eqv3_validation.md) for the
separate scaling measurements and their source revision.
GPU FP32, supported layouts, tile-size limits, and 32-bit indexing restrictions
still apply; second-order gradients and full-model training remain unvalidated
or unsupported as described above.

## EQv3 operator comparison plots

H100, FP32; measurements use FastEq `3bc9a82d40ef646c3ce75f6f517e3f618fb12754`.
The charts cover the retained EQv3 operator list. Speedup is original Torch time
divided by FastEq time; forward + backward includes all first-order gradients.
These existing measurements were not rerun after the adaptation scope cleanup.

### N = 4096

| Operator | Forward speedup | Forward + backward speedup |
| --- | ---: | ---: |
| GraphSoftmax (cached CSR) | 4.82× | 1.23× |
| AttentionAlpha | 3.43× | 0.75× |
| e3nn Gate | 2.17× | 3.11× |
| LayerNorm | 2.92× | 1.85× |
| SeparableLayerNorm | 1.55× | 1.40× |
| EquivariantDropout | 0.61× | 0.85× |

GraphSoftmax and AttentionAlpha have accuracy failures in other configurations.
The e3nn Gate measurements do not validate EQv3 GateActivation. See the
[full report](eqv3_validation.md) for correctness coverage and timing details.

![Measured speedup at N=4096](eqv3_validation/2026-09-15/speedup_4096.png)

### Doubling the atom count

![Speedup as the atom count doubles](eqv3_validation/2026-09-15/speedup_scaling.png)

The curves end at each path's measured memory boundary or explicit execution
limit. Raw [paired timings](eqv3_validation/2026-09-15/paired.csv) and
[stopping boundaries](eqv3_validation/2026-09-15/boundaries.csv) distinguish OOM,
index limits and fallback; a stopping point is not always an OOM.
