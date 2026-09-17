# GraphSoftmax

[GraphSoftmax](../fasteq/triton/graph_softmax.py) implements FP32 forward and
Triton first-order backward on CUDA and HIP. The optional
[node-major layout helper](../fasteq/triton/graph_edge_layout.py) stores each
node's incoming edges contiguously and reuses the topology permutation.
Both devices pass 215 regression tests and four matched performance cases.

## Implementation and integration

Forward preserves the sequence soft cap, maximum subtraction, exponential,
rescale, dropout, sum and normalization. Backward saves unnormalized
exponentials with the realized dropout mask and uses centered FP64 arithmetic
to avoid subtracting nearly equal rounded values. Shared-edge rescale
gradients retain the epsilon term analytically. Cross-edge/head reductions
use fixed Triton trees with FP64 partial sums. CUDA and HIP use the same
mathematical kernels. Unsupported shapes/dtypes/degrees retain the adapter's
Torch fallback; these fallback calls are not counted as fused performance.

```python
from fasteq.triton.graph_edge_layout import prepare_graph_softmax_layout

layout = prepare_graph_softmax_layout(edge_dst, num_nodes=num_nodes)
x_sorted = layout.pack(logits)
r_sorted = layout.pack_rescale(exp_rescale)
y_sorted = layout.softmax(x_sorted, exp_rescale=r_sorted, softcap=3.0)
y = layout.unpack(y_sorted)  # Only when the consumer requires original order.
```

Keep **all** edge-aligned features, rescale weights and messages in the same
order. Reuse the mapping while topology stays fixed, but recompute feature
values each model step. A topology mutation invalidates the mapping. Fixed-seed
Triton dropout uses original edge IDs, preserving its realized mask across
the original and sorted routes. Packing and inverse gradient copies are
differentiable. The largest gain assumes neighboring operators also keep the
node order; packing and restoring on every call has a separate measured cost.

## Accuracy

Tests ran on H100 `gxn70` GPU 7 and Hygon BW/gfx936 `a14r1n06`; software is in
the [machine table](eqv3_validation.md#machines-and-software).
The reference is original EQv3 `GraphSoftmax` in pinned `softmax.py`, SHA-256
`643467db97b6168f387c1d63bcc4af88af5ea5aed979325ce37ecd4c906b17b9`.
Output uses `atol=3e-6, rtol=3e-5`; gradients use `atol=3e-5, rtol=3e-4`.

Each device passes **215 tests, zero failures/errors/skips**: 198 precision-route
cases and 17 layout/gradient/dropout/message-alignment cases. Coverage includes
broadcast and zero rescale, empty nodes, strided/expanded inputs, requested
gradient subsets, multi-stage reduction and fixed-seed dropout. A small
weighted-message chain checks that sorting preserves output and gradients.

Only saved shared-rescale cancellation fixtures and demonstrated
single-neighbor/small-rescale boundaries use independent analytical/FP64
derivatives. Native FP32 itself is unstable there, so these cases are not
reported as passing a direct FP32 gradient comparison. Zero-rescale components
retain FP32 checks, and additional assertions preserve small nonzero
analytical gradients. All ordinary cases keep their original tolerances.

Original-order, resident-order and pack/restore routes all pass native FP32
output, input-gradient and rescale-gradient checks at N=4,096 / 262,144 /
1,048,576 and on an additional random-index graph at N=4,096.
At N=1,048,576, the full pack/restore path has:

| Device | Component | Maximum absolute error | Maximum tolerance ratio | Failing elements |
| --- | --- | --- | --- | --- |
| h100 | output | 2.682e-07 | 0.02294 | 0 |
| h100 | dx | 2.980e-07 | 0.00312 | 0 |
| h100 | dr | 1.669e-06 | 0.02211 | 0 |
| hygon | output | 2.980e-07 | 0.02464 | 0 |
| hygon | dx | 2.086e-07 | 0.00339 | 0 |
| hygon | dr | 1.907e-06 | 0.02423 | 0 |

## Performance

Inputs are `[E,8]`, `E=32N`, FP32, rescale `[E,1]`, soft cap 3, epsilon
`1e-16`, dropout 0. Scaling graphs use interleaved destinations
`arange(E) % N`. Values are median GPU-event milliseconds, with five warmups
and 20 samples in rotated provider order.

Forward + backward, including both logits and rescale gradients:

| Device | N | Torch ms | Original edge order ms | Resident node order ms | Pack + restore ms |
| --- | --- | --- | --- | --- | --- |
| h100 | 4,096 | 0.696 | 0.433 | 0.431 | 0.765 |
| h100 | 262,144 | 21.292 | 2.367 | 1.844 | 2.872 |
| h100 | 1,048,576 | 85.832 | 9.055 | 6.116 | 11.760 |
| hygon | 4,096 | 0.838 | 0.529 | 0.501 | 1.062 |
| hygon | 262,144 | 17.377 | 24.943 | 5.984 | 14.096 |
| hygon | 1,048,576 | 69.725 | 115.642 | 23.412 | 56.478 |

Forward:

| Device | N | Torch ms | Original edge order ms | Resident node order ms | Pack + restore ms |
| --- | --- | --- | --- | --- | --- |
| h100 | 4,096 | 0.363 | 0.130 | 0.124 | 0.284 |
| h100 | 262,144 | 12.390 | 0.746 | 0.455 | 0.976 |
| h100 | 1,048,576 | 50.007 | 2.986 | 1.505 | 3.782 |
| hygon | 4,096 | 0.391 | 0.196 | 0.188 | 0.450 |
| hygon | 262,144 | 7.205 | 8.147 | 1.811 | 5.557 |
| hygon | 1,048,576 | 28.846 | 32.947 | 6.784 | 21.972 |

Original order reuses CSR. Resident node order excludes feature permutation.
Pack + restore includes feature permutations, original-order outputs and
inverse gradient permutations, while reusing the topology mapping. At a
million nodes, including **mapping construction as well** gives complete
pack/restore forward + backward wall time of 17.781 ms on H100 and 83.703 ms
on Hygon (three repetitions). Those wall times are distinct from the GPU-event
table.

At that size, resident order improves Hygon from 115.642 to 23.412 ms
(4.94x), and H100 from 9.055 to 6.116 ms (1.48x). The corresponding gains
against Torch are 2.98x and 14.03x. This comparison supports the contiguous-edge
integration strategy for these graphs, not a speedup guarantee for arbitrary
graphs or a complete model. Small calls can lose the benefit to packing costs.

## Records and scope

[Summary and source hashes](validation/graph_softmax/summary.json),
[H100 tests](validation/graph_softmax/h100/pytest.xml),
[Hygon tests](validation/graph_softmax/hygon/pytest.xml), and all eight raw
accuracy/timing JSON files linked from the summary describe this exact source.
The [benchmark](../test/benchmark_triton_graph_softmax.py) and
[reproduction commands](validation/REPRODUCE.md) are included.
These runs stop at the specified sizes, not at OOM. Double backward and
full EQv3 model training are not validated.
