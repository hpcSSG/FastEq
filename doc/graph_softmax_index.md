# GraphSoftmax without CSR

[`FusedGraphSoftmaxIndex`](../fasteq/triton/graph_softmax_index.py) implements
FP32 GraphSoftmax forward and Triton first-order backward directly on raw
edge destination indices. It builds no CSR, sorts no features and caches no
topology. The [CSR implementation](graph_softmax.md) remains available for
reusable graphs and workloads where CSR traversal is faster even after
construction. Selection is explicit; neither interface replaces the other.

## Usage and supported behavior

```python
from fasteq.triton.graph_softmax_index import FusedGraphSoftmaxIndex

softmax = FusedGraphSoftmaxIndex(eps=1e-16, softcap=3.0, exp_dropout=0.0)
y = softmax(logits, edge_dst, num_nodes=num_nodes, exp_rescale=rescale)
```

The fused path accepts FP32 `logits[E,H]` and one-dimensional int32/int64
`edge_dst[E]` on the same CUDA/HIP device. Rescale can broadcast to `[E,H]`.
Indices must be in `[0, num_nodes)`. Pass `num_nodes` explicitly to avoid a
host synchronization for size inference. Each call reads current indices,
including in-place topology updates. Noncontiguous logits/indices are copied
to contiguous storage; strided rescale is supported. Empty inputs and
unsupported shapes/dtypes/devices use the existing Torch reference expression.

The functional entry point is `fused_graph_softmax_index`. Both interfaces
support soft cap, exponential rescale and training dropout. An optional
one-element int64 device `seed` tensor reproduces the same original-edge
Triton dropout mask as the CSR route. Ordinary training generates a seed on
the device. The fused autograd path supports first-order gradients only.

## Implementation

Forward fuses the soft cap, node maximum, exponential, rescale, dropout, node
sum and normalization into scatter-based stages. Backward uses centered
arithmetic and FP64 atomic node sums to reduce cancellation, and preserves
the analytical epsilon term for rescale shared across edges. Broadcast
rescale gradients reuse the common fixed-tree Triton reduction. A per-node
integer pivot supports centering; it is not a neighbor list or CSR structure.
The mathematical kernels have no CUDA/HIP-specific branch.

Floating-point atomic sums do not have a fixed execution order. Precision is
validated against tolerances, with no bitwise determinism guarantee. The
scatter cost depends on edge distribution and contention, so avoiding CSR
preparation does not guarantee the fastest total call at every size.

## Accuracy and machines

The same source and test hashes were validated on both devices on 2026-09-18:

| Device | Host / GPU | Torch | Triton | Regression suite | Matched graph cases |
| --- | --- | --- | --- | --- | --- |
| NVIDIA H100 80GB HBM3, sm90 | gxn70, physical GPU 0 | 2.11.0+cu128 | 3.6.0 | 105 passed | 12 passed |
| Hygon BW DCU, gfx936 | f11r1n20, one DCU | 2.7.1 | HCU 3.1.0 | 105 passed | 12 passed |

Both regression suites have zero failures, errors or skips. Coverage includes
broadcast/zero rescale, requested gradient subsets, empty nodes/inputs,
int32/int64 indices, strided tensors, in-place topology updates, degree 8,193,
and seeded dropout output/gradient alignment with CSR. A regression forbids
sorting, bincount/cumsum, CSR construction and host scalar reads during a
fused call with explicit `num_nodes`.

The reference is original EQv3 `GraphSoftmax` in pinned `softmax.py`, SHA-256
`643467db97b6168f387c1d63bcc4af88af5ea5aed979325ce37ecd4c906b17b9`.
Output uses `atol=3e-6, rtol=3e-5`; logits/rescale gradients use
`atol=3e-5, rtol=3e-4`. Only the existing demonstrated shared-rescale
cancellation and single-neighbor/small-rescale boundary fixtures use the
[analytical/FP64 exception](graph_softmax.md#accuracy). Ordinary tests and all
24 benchmark cases compare every output, `dx` and `dr` element to native
Torch FP32. This exception does not widen the ordinary tolerances.

Maximum errors below aggregate the **12 benchmark cases per device**, for the
raw-index route. A tolerance ratio of at most 1 passes.

| Device | Component | Maximum absolute error | Maximum tolerance ratio | Failing elements |
| --- | --- | --- | --- | --- |
| h100 | output | 2.98023e-07 | 0.024441 | 0 |
| h100 | dx | 2.83122e-07 | 0.003891 | 0 |
| h100 | dr | 1.43051e-06 | 0.028613 | 0 |
| hygon | output | 3.27826e-07 | 0.027746 | 0 |
| hygon | dx | 2.38419e-07 | 0.002383 | 0 |
| hygon | dr | 1.66893e-06 | 0.017492 | 0 |

## Performance

FP32 inputs are `[E,8]`, `E=32N`, rescale `[E,1]`, soft cap 3, epsilon
`1e-16` and dropout 0. `random` draws uniform destination indices;
`interleaved` uses `arange(E) % N`; `contiguous` places 32 edges per node
consecutively. Inputs are generated before timing. Each case uses five
warmups and 20 samples with rotated provider order.

Tables report median **synchronized wall milliseconds for forward + backward**,
including logits/rescale gradients. `Rebuilt CSR` reconstructs CSR on every
call with features in original order. `Rebuilt + packed` reconstructs the
sorted layout, packs features and restores outputs/gradients on every call.
Neither comparison reuses CSR or a permutation, even though the input graph
is fixed within each timing case. These complete-call costs are distinct from
the reusable-topology GPU-event measurements on the [CSR page](graph_softmax.md).

### NVIDIA H100

| Edge order | N | Torch ms | Raw index ms | Rebuilt CSR ms | Rebuilt + packed ms | Torch / raw index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 256 | 0.6205 | 0.5248 | 0.7852 | 1.1092 | 1.18x |
| random | 4,096 | 0.5421 | 0.4163 | 0.6820 | 0.9400 | 1.30x |
| random | 32,768 | 2.8285 | 0.5945 | 0.9968 | 1.1636 | 4.76x |
| random | 262,144 | 21.4027 | 4.2089 | 5.3631 | 6.0272 | 5.09x |
| interleaved | 256 | 0.5659 | 0.4968 | 0.7060 | 1.0207 | 1.14x |
| interleaved | 4,096 | 0.5443 | 0.4597 | 0.7111 | 0.9934 | 1.18x |
| interleaved | 32,768 | 2.8263 | 0.5534 | 0.8580 | 1.0162 | 5.11x |
| interleaved | 262,144 | 21.3045 | 4.0339 | 3.6274 | 4.4819 | 5.28x |
| contiguous | 256 | 0.5516 | 0.4950 | 0.6975 | 1.0135 | 1.11x |
| contiguous | 4,096 | 0.5838 | 0.5106 | 0.7858 | 1.1135 | 1.14x |
| contiguous | 32,768 | 2.9873 | 0.7618 | 1.0007 | 1.3668 | 3.92x |
| contiguous | 262,144 | 22.1665 | 3.6921 | 3.2625 | 4.0193 | 6.00x |

### Hygon BW

| Edge order | N | Torch ms | Raw index ms | Rebuilt CSR ms | Rebuilt + packed ms | Torch / raw index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 256 | 0.8709 | 0.7392 | 0.9629 | 1.5798 | 1.18x |
| random | 4,096 | 0.8544 | 0.7294 | 1.0656 | 1.6647 | 1.17x |
| random | 32,768 | 2.3269 | 1.3886 | 4.0347 | 2.9214 | 1.68x |
| random | 262,144 | 18.6225 | 16.0895 | 24.0930 | 22.0862 | 1.16x |
| interleaved | 256 | 0.8482 | 0.7247 | 0.9538 | 1.5518 | 1.17x |
| interleaved | 4,096 | 0.8558 | 0.7288 | 1.0702 | 1.6827 | 1.17x |
| interleaved | 32,768 | 2.2909 | 1.4545 | 3.9460 | 2.9017 | 1.58x |
| interleaved | 262,144 | 17.4091 | 10.7081 | 31.3718 | 21.2409 | 1.63x |
| contiguous | 256 | 0.8434 | 0.7141 | 0.9444 | 1.5394 | 1.18x |
| contiguous | 4,096 | 0.8529 | 0.7221 | 1.0598 | 1.6604 | 1.18x |
| contiguous | 32,768 | 2.6725 | 1.4954 | 2.1584 | 2.5988 | 1.79x |
| contiguous | 262,144 | 19.5629 | 10.5850 | 13.2978 | 15.0358 | 1.85x |

Raw-index forward + backward is **1.11–6.00x Torch on H100** and
**1.16–1.85x on Hygon** in this matrix. Hygon raw-index fusion is faster than
both rebuilt-CSR alternatives in all 12 measured cases. On H100 at N=262,144,
rebuilt CSR is faster for interleaved edges (3.6274 vs 4.0339 ms) and contiguous
edges (3.2625 vs 3.6921 ms). Both implementations are therefore retained.

The [summary](validation/graph_softmax_index/summary.json) includes forward-only
wall medians as well. Each raw JSON retains forward and forward + backward,
all 20 wall/GPU-event samples, source hashes, and full-tensor accuracy metrics
for all three fused providers. H100 used a shared host without exclusive
reservation or locked clocks; compare implementations within each device.
These measurements do not establish a hardware throughput ranking.

## Records and scope

[Summary and hashes](validation/graph_softmax_index/summary.json),
[H100 tests](validation/graph_softmax_index/h100/pytest.xml) and
[Hygon tests](validation/graph_softmax_index/hygon/pytest.xml) describe the
current source. The summary links all 24 raw timing/accuracy JSON files.
The [benchmark](../test/benchmark_triton_graph_softmax_index.py),
[regression suite](../test/test_triton_graph_softmax_index.py) and
[reproduction commands](validation/REPRODUCE.md#graphsoftmax-without-topology-reuse)
are included. The CPU-only evidence audit verifies current hashes, counts,
accuracy checks, sample medians and summary values.

Validation is limited to standalone operators and these synthetic graphs;
full EQv3 training, double backward and scaling to OOM are not validated.
