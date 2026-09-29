# Equivariant LayerNorm

FastEq fuses equivariant normalization for EquiformerV3 and EquiformerV2 layers,
with GPU FP32 forward execution and first-order gradients on CUDA and HIP.

The current-source Hygon scaling run includes a Separable parameter-gradient
failure. The 218-test regression pass does not imply that every larger sampled
shape passes; the exact result is recorded below.

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

## Current-source accuracy

Both H100 `gxn70` GPU 5 and Hygon BW/gfx936 `a14r1n09` pass **218 regression
tests, zero failures/errors/skips**, using the current source SHA-256
`35d689624e48422801cfdb0b8461243f7a88ee012dc20122206433098a90da5d`.
Software versions are in the [machine table](eqv3_validation.md#machines-and-software).

The original unmodified V3 and V2 source layers are compared in FP32, including
output, `dX` and all affine parameters, NKC/KNC layouts, irregular channels,
large-N cancellation regressions and two SGD steps through supported layers.
Ordinary elementwise tolerance is `atol=5e-5, rtol=5e-4`; existing stricter
mathematical tests are retained. These are individual-layer tests, not
full-model inference, training, force-loss or higher-order-autograd validation.
eSEN has not been directly tested.

References:

- EQv3 `layer_norm.py`, SHA-256 `41e00b377cbacef4713d88697c06e419f3b8c208d9fcc55be78b53fd5a23c2c6`.
- Official EQv2 `layer_norm.py` at commit `d5ad4be729b56f74012ebb7f097f77c5b00a1004`, SHA-256 `fdd7955f1fc3ef4e0a0ccbfb761128b5b269ad09f50423428f77aee12a53480c`.

The separate current-source Hygon scaling/small-shape suite has **15/15 passes
for per-degree LayerNorm and 14/15 for Separable LayerNorm**. At N=262,144,
Lmax=3, C=128, Separable `affine_weight` has **two failing elements**, with a
maximum tolerance ratio of **2.786550**. Outputs and input gradients pass.
The N=524,288 sample passes, illustrating cancellation sensitivity rather
than a monotonic size threshold. The failure remains unresolved in this
source and is not replaced by an analytical-reference acceptance.

## Current-source performance

Hygon measurements use `[N,16,128]`, FP32, NKC, native parameter reduction,
five warmups and 20 synchronized GPU-event samples. Times include the Torch
reductions used in backward. All table points below pass their own comparisons;
they do not remove the separate Separable failure at N=262,144.

### EquivariantLayerNorm

| Device | N | Mode | Torch ms | FastEq ms | Speedup | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Hygon BW | 4,096 | Forward | 0.8077 | 0.3678 | 2.20x | PASS |
| Hygon BW | 524,288 | Forward | 65.7555 | 22.7414 | 2.89x | PASS |
| Hygon BW | 4,096 | Forward + backward | 2.6886 | 1.8550 | 1.45x | PASS |
| Hygon BW | 524,288 | Forward + backward | 239.9429 | 153.5689 | 1.56x | PASS |

### EquivariantSeparableLayerNorm

| Device | N | Mode | Torch ms | FastEq ms | Speedup | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Hygon BW | 4,096 | Forward | 0.7069 | 0.3929 | 1.80x | PASS |
| Hygon BW | 524,288 | Forward | 70.1496 | 25.0154 | 2.80x | PASS |
| Hygon BW | 4,096 | Forward + backward | 2.1165 | 1.2658 | 1.67x | PASS |
| Hygon BW | 524,288 | Forward + backward | 225.7688 | 86.4217 | 2.61x | PASS |

No H100 performance table is retained for this exact source: its available
performance snapshot used a different implementation hash. The current H100
218-test correctness report remains valid. The same rule excludes older V2
performance numbers; current V2 correctness is covered by the regression suite.

Both variants last run at N=524,288 in the FastEq sweep and stop before
N=1,048,576 at the explicit 32-bit element-index bound for this shape. This is
an implementation limit, not a measured FastEq OOM. Torch's actual allocation
stops and all intermediate timings are in the per-variant boundary/raw files.

## Records and reproduction

[H100 regression](validation/layernorm/h100/summary.json),
[H100 JUnit](validation/layernorm/h100/pytest.xml),
[Hygon regression](validation/layernorm/hygon/summary.json), and
[Hygon JUnit](validation/layernorm/hygon/pytest.xml) identify the implementation,
tests and reference hashes.
[LayerNorm timings](validation/norm/hygon/paired.csv),
[LayerNorm accuracy](validation/norm/hygon/correctness.json),
[Separable timings](validation/separable/hygon/paired.csv),
[Separable accuracy](validation/separable/hygon/correctness.json), and
[failed elements](validation/separable/hygon/failures.csv) preserve current-source
scaling evidence. Full samples, errors and boundaries are in the adjacent
raw-record files, indexed by [the manifest](validation/manifest.json).
Use the [reproduction commands](validation/REPRODUCE.md) for the three current
LayerNorm test modules and the scaling harness.
