# GraphSoftmax validation — 2026-09-17

This snapshot validates the precision-corrected GraphSoftmax and its reusable
node-major edge layout. Both devices passed **215 tests with zero failures,
errors or skips**, plus four matched accuracy/performance cases per device.
Source-file hashes and raw-result hashes are in [summary.json](summary.json).

## Machines and reference

| Device | Host | PyTorch | Triton | Regression result |
| --- | --- | --- | --- | --- |
| NVIDIA H100 80GB, CUDA sm90 | gxn70, GPU 7 | 2.11.0+cu128 | 3.6.0 | [215 passed](h100/pytest.xml) |
| Hygon BW, HIP gfx936 | a14r1n06, one DCU | 2.7.1 | 3.1.0 | [215 passed](hygon/pytest.xml) |

The reference is the original EquiformerV3 `GraphSoftmax` class from
[`softmax.py`](https://github.com/malixian/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3/softmax.py).
Its SHA-256 is
`643467db97b6168f387c1d63bcc4af88af5ea5aed979325ce37ecd4c906b17b9`.

Ordinary cases retain native Torch FP32 tolerances:

- Output: `atol=3e-6, rtol=3e-5`.
- Input and rescale gradients: `atol=3e-5, rtol=3e-4`.

Only the archived shared-rescale cancellation fixtures and proven
single-neighbor/small-rescale boundaries use independent analytical/FP64
derivatives. Native FP32 itself is unstable in those cases; they are **not**
claimed to pass a direct FP32 gradient comparison. Zero-rescale components
retain FP32 checks. Additional assertions check that small nonzero analytical
gradients are preserved instead of being replaced with zero.

The suite covers original and node-sorted routes, broadcast weights, zero
weights, empty nodes, dropout, noncontiguous and expanded inputs, gradient
subsets, multi-stage reductions, permutation gradients, stale topology,
fixed-seed mask preservation and weighted-message edge alignment.
The recorded tests consist of 198 precision-route cases and 17 layout cases.

## Matched forward-plus-backward timings

All values below are milliseconds. Inputs are FP32 with 32 edges per node,
8 heads, per-edge rescale `[E,1]`, softcap 3, epsilon `1e-16`, and no dropout.
Each case uses five warmups and the median of 20 synchronized GPU-event samples
with rotated provider order. Gradients are requested for logits and rescale.
All measured variants passed output and gradient checks against native FP32.

| Device | Nodes | Torch | Original edge order | Resident node order | Pack and restore each call |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100 | 4,096 | 0.696 | 0.433 | 0.431 | 0.765 |
| H100 | 262,144 | 21.292 | 2.367 | 1.844 | 2.872 |
| H100 | 1,048,576 | 85.832 | 9.055 | 6.116 | 11.760 |
| Hygon | 4,096 | 0.838 | 0.529 | 0.501 | 1.062 |
| Hygon | 262,144 | 17.377 | 24.943 | 5.984 | 14.096 |
| Hygon | 1,048,576 | 69.725 | 115.642 | 23.412 | 56.478 |

Original-order timings reuse CSR. Resident-order timings assume that upstream
features and gradients are already in node order. Pack/restore timings include
feature permutations, original-order output restoration and inverse gradient
permutations, while reusing the topology mapping. They do not include rebuilding
that mapping. With mapping construction included, million-node pack/restore
forward-plus-backward wall time is 17.781 ms on H100 and 83.703 ms on Hygon.

The three scaling cases use balanced interleaved destination indices
`arange(E) % N`. An additional random-index case at 4,096 nodes passed on each
device; its raw measurements are included below. These synthetic distributions
do not establish a speedup for every real graph. Small eager-call timings also
include dispatch gaps between GPU operations.

| Device | Interleaved 4,096 | Interleaved 262,144 | Interleaved 1,048,576 | Random 4,096 |
| --- | --- | --- | --- | --- |
| H100 | [JSON](h100/perf_4096.json) | [JSON](h100/perf_262144.json) | [JSON](h100/perf_1048576.json) | [JSON](h100/perf_random.json) |
| Hygon | [JSON](hygon/perf_4096.json) | [JSON](hygon/perf_262144.json) | [JSON](hygon/perf_1048576.json) | [JSON](hygon/perf_random.json) |

## Reproduction

Run from the FastEq repository root in an environment with the matching
PyTorch/Triton backend, `pytest`, `torch_geometric`, and the original EQv3
reference dependencies. Point the reference variable at the directory
containing the pinned `softmax.py` above. `FASTEQ_BACKEND=cpu` avoids loading
unrelated compiled FastEq extensions; these directly imported Triton tests
still execute on the selected GPU/DCU.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export FASTEQ_BACKEND=cpu
export FASTEQ_EQUIFORMER_V3_SOURCE_DIR=/path/to/equiformer_v3/experimental/models/equiformer_v3

python -m pytest -q test/test_triton_graph_softmax_precision.py \
  test/test_triton_graph_edge_layout.py --junitxml=graph_softmax_tests.xml

for n in 4096 262144 1048576; do
  python test/benchmark_triton_graph_softmax.py --n "$n" --out "graph_softmax_${n}.json"
done
python test/benchmark_triton_graph_softmax.py --n 4096 --topology random \
  --out graph_softmax_random.json
```

## Integration scope

`prepare_graph_softmax_layout(index, num_nodes)` returns a reusable mapping.
Use `pack` for per-edge features and `pack_rescale` for supported rescale shapes;
`layout.softmax` consumes and returns packed tensors. `unpack` restores original
edge order when a consumer needs it. Every edge-aligned tensor must use the same
ordering. Rebuild the mapping when topology changes, and never cache feature
values across model updates. Fixed-seed Triton dropout uses original edge IDs.

The intended integration keeps this ordering across neighboring operators.
Per-call packing is an explicit option, with its complete cost shown above.
This snapshot validates forward and first-order backward, including a small
weighted-message aggregation chain. It does not validate higher-order autograd
or full EquiformerV3 model training. The precision kernels are shared by CUDA
and HIP; no backend-specific mathematical branch was added.
