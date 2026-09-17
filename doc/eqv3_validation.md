# EQv3 operator validation

This is the consolidated record for the production sources in this checkout.
Every retained measurement is tied to the current operator's SHA-256 in
[the source manifest](validation/manifest.json). Older implementation results
and duplicate debug snapshots are omitted. Dates identify when a measurement
was made; an unchanged operator can retain an earlier measurement.

## Current status

| Operator | Correctness on H100 | Correctness on Hygon BW | Performance and remaining limits |
| --- | --- | --- | --- |
| [GraphSoftmax](graph_softmax.md) | 215 tests passed | 215 tests passed | Four matched graph cases per device pass. Reusable node order substantially improves the large-graph Hygon result. |
| [AttentionAlpha](attention_alpha.md) | 66 tests + 13 stress cases passed | 66 tests + 13 stress cases passed | Triton forward and first-order backward. H100 forward + backward is 5.43–6.25x Torch at measured sizes; Hygon is 0.51–0.52x. |
| [Equivariant LayerNorm](layernorm.md) | 218 tests passed | 218 tests passed; separate scaling suite has one failing Separable case | Hygon Separable affine-weight gradient at N=262,144 has two elements outside tolerance. Current-source H100 performance was not rerun. |
| [e3nn Equivariant Gate](equivariant_gate.md) | 15 shape comparisons passed | 15 shape comparisons passed | The separate EQv3 GateActivation class still lacks a callable forward. |
| [Equivariant Dropout](equivariant_dropout.md) | 17 shape comparisons passed | 16 shape comparisons passed | Exact output/gradient agreement with a shared mask. Hygon launch error at N=2,097,152. |

Test counts describe different suites and are not added into one acceptance
score. Full-model training and higher-order autograd are not validated by this
record. A passed regression suite does not erase a failure from a separate
larger-scale comparison.

## Machines and software

| Measurement | Host / device | Torch | Triton | Date |
| --- | --- | --- | --- | --- |
| GraphSoftmax | gxn70, physical GPU 7, NVIDIA H100 80GB HBM3, sm90 | 2.11.0+cu128 | 3.6.0 | 2026-09-17 |
| GraphSoftmax | a14r1n06, one Hygon BW DCU, gfx936, approximately 64 GiB | 2.7.1 | HCU 3.1.0 | 2026-09-17 |
| AttentionAlpha | gxn70, physical GPU 5, NVIDIA H100 80GB HBM3 | 2.11.0+cu128 | 3.6.0 | 2026-09-16 |
| AttentionAlpha | a14r1n06, one Hygon BW DCU, gfx936 | 2.7.1 | HCU 3.1.0 | 2026-09-16 |
| LayerNorm regression; Gate / Dropout | gxn70, physical GPU 5, NVIDIA H100 80GB HBM3 | 2.11.0+cu128 | 3.6.0 | 2026-09-15 |
| LayerNorm; Gate / Dropout | a14r1n09, one Hygon BW DCU, gfx936, 65,520 MiB | 2.7.1 / HIP 6.3.26045 | HCU 3.1.0 | 2026-09-16 |

Gate uses e3nn 0.4.4; graph references use torch-geometric 2.6.1. H100 runs used
a shared host without exclusive reservation or locked clocks. Compare FastEq
with Torch **within each device**; differing software and workloads do not
establish a hardware throughput ranking. PyTorch's `torch.cuda` API also
selects HIP devices in these scripts.

## Accuracy and references

Ordinary comparisons use the original eager Torch FP32 expression and check
every output and requested first-order input/parameter gradient element:
`abs(actual-reference) <= atol + rtol*abs(reference)`.

| Operator | Absolute tolerance | Relative tolerance | Reference |
| --- | --- | --- | --- |
| GraphSoftmax output | 3e-6 | 3e-5 | Original EQv3 GraphSoftmax |
| GraphSoftmax gradients | 3e-5 | 3e-4 | Torch FP32, with the narrowly documented cancellation-boundary exception |
| AttentionAlpha, LayerNorm, Gate | 5e-5 | 5e-4 | Original Torch expressions / native source layers / e3nn Gate |
| Dropout with the same mask | 0 | 0 | Original EQv3 EquivariantDropout |

The maximum tolerance ratio is the maximum error divided by the allowed
absolute-plus-relative error; a ratio at most 1 passes. Existing stricter
mathematical regression tests remain in force. Only GraphSoftmax's demonstrated
unstable cancellation boundaries use independent analytical/FP64 derivatives,
as detailed on its page. This exception does not apply to LayerNorm or
AttentionAlpha failures.

EQv3 references are pinned to
[`a7300c58`](https://github.com/malixian/equiformer_v3/tree/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3).
LayerNorm also tests official EQv2
[`d5ad4be7`](https://github.com/atomicarchitects/equiformer_v2/blob/d5ad4be729b56f74012ebb7f097f77c5b00a1004/nets/equiformer_v2/layer_norm.py).
Reference file hashes are retained beside the operator records.

## Measurement method

All reported timings are FP32 standalone operator calls. Forward uses
`no_grad`; forward + backward includes all requested input and parameter
gradients, without an optimizer. Five warmups precede 20 synchronized
GPU-event samples; tables report medians. Host dispatch gaps are included.
GraphSoftmax rotates provider order; AttentionAlpha measures Torch then
Triton. The Gate/Dropout/LayerNorm sweep uses a fresh process per
implementation, mode and shape. These differences are retained rather than
combining separate runs into a hardware comparison.

GraphSoftmax reports resident node order, per-call feature packing/restoration,
and topology preparation separately. LayerNorm timings include its Torch
backward reductions. Active Dropout generates masks normally during timing;
only correctness comparisons replay a common mask. Raw sweep records include
all timing samples and allocator-allocated peak memory, which is not total
device occupancy.

For retained scaling sweeps, N starts at 256 and doubles. `OOM` is an actual
allocation failure; `LIMIT` or `INDEX_GUARD` stops before allocating an
unsupported index range; `ERROR` is a runtime failure. A FastEq-only point
beyond Torch's memory ceiling has no matched accuracy result. The latest
GraphSoftmax and AttentionAlpha measurements are bounded representative runs,
so prior implementation OOM limits are not carried forward.

## Reproduction and records

[Reproduction instructions](validation/REPRODUCE.md) cover the current test and
benchmark entry points. Raw metrics, JUnit reports, filtered scaling records,
and their checksums are under [doc/validation](validation/manifest.json).
The operator pages contain the supported behavior, precision conclusions,
performance tables and limitations; the raw files provide audit evidence.
