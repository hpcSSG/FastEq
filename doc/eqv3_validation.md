# EQv3 operator validation index

Operator-specific results are recorded separately. LayerNorm also documents
the shared implementation, source adapters and fusion; the other pages record
the scope and results of performance validation.

| Operator | Documentation |
| --- | --- |
| Equivariant LayerNorm and supported variants | [LayerNorm](layernorm.md) |
| GraphSoftmax | [GraphSoftmax performance](graph_softmax.md) |
| AttentionAlpha | [AttentionAlpha performance](attention_alpha.md) |
| Equivariant Gate | [Equivariant Gate performance](equivariant_gate.md) |
| Equivariant Dropout | [Equivariant Dropout performance](equivariant_dropout.md) |

## Measurement method

This section describes the archived H100 run. Fresh Hygon coverage and its
software differences are recorded in the dated section below.

The archived sweep tested FastEq commit
`3bc9a82d40ef646c3ce75f6f517e3f618fb12754`. Operator files were unchanged during
measurement. This documentation split reuses those results. The later LayerNorm
scope-cleanup regression is recorded separately in
[regression.json](eqv3_validation/2026-09-15/regression.json).

| Item | Configuration |
| --- | --- |
| Hardware | One H100 80GB on gxn70, physical GPU 5 |
| Software | PyTorch 2.11.0+cu128, Triton 3.6.0; e3nn 0.4.4 for Gate |
| Dtype | FP32 |
| Native EQv3 source | Commit a7300c58df683dc99cb48027d5bfd4c887486c48; files verified byte for byte |
| Timing | 5 warmups, 20 synchronized samples; median GPU-event interval and wall-clock time |
| Forward | no_grad; active training dropout where specified on the operator page |
| Forward + backward | Forward and requested first-order input/parameter gradients, no optimizer |
| Scale | N starts at 256 and doubles; fresh process per backend, mode and shape |
| Host conditions | Shared machine, no exclusive reservation or locked clocks |

The baseline is the original eager Torch implementation; Gate uses
`e3nn.nn.Gate`, while AttentionAlpha executes the original EQv3 attention
expressions. Each operator page identifies its reference and input shapes.
GPU-event intervals include all work in the public call and host-submission
gaps; hybrid backward costs are included. They are not isolated kernel timings
or pure-backward estimates. Speedup is Torch time divided by FastEq time.

Peak memory is PyTorch allocator-allocated memory, including inputs, outputs,
gradients and temporaries; it is not total device occupancy. GraphSoftmax's
cached-CSR and per-call graph-preparation measurements are distinguished on
its page.

Matched accuracy checks compare every output and requested gradient element
against Torch with `abs(actual-reference) <= atol + rtol*abs(reference)`.
Default tolerances are `atol=5e-5, rtol=5e-4`; GraphSoftmax uses stricter
forward/gradient tolerances, and Dropout requires exact equality for a shared
mask. Tolerances were not relaxed. Raw records retain 32 numeric samples per
tensor in addition to the complete comparison metrics.

`OOM` records an actual allocation failure. `LIMIT`, `INDEX_GUARD` and
`FALLBACK` are harness stops based on verified indexing/fallback conditions;
the next shape is not allocated. Successful FastEq-only shapes beyond Torch's
memory ceiling have no matched Torch accuracy check.

The archived 2026-09-15 sweep measures standalone operators on CUDA, without
full-model timing. Its records contain no new HIP measurements. Earlier
CUDA/HIP LayerNorm results retain their own source hashes and are described in [LayerNorm](layernorm.md).

## Shared records

- [Paired timings, speedups and memory](eqv3_validation/2026-09-15/paired.csv)
- [Execution and stopping boundaries](eqv3_validation/2026-09-15/boundaries.csv)
- [Accuracy summary](eqv3_validation/2026-09-15/correctness_summary.json) and [failures](eqv3_validation/2026-09-15/failures.csv)
- [Raw points and comparisons](eqv3_validation/2026-09-15/raw_results.json.gz), [source manifest](eqv3_validation/2026-09-15/manifest.json) and [source audit](eqv3_validation/2026-09-15/reference_audit.json)
- [Reproduction instructions](eqv3_validation/2026-09-15/REPRODUCE.md)
- [Per-operator plotting script](eqv3_validation/2026-09-15/plot_by_operator.py)

The original combined report and plots remain in the dated artifact directory
as a historical snapshot. The five pages above are the operator documentation.

## Hygon rerun: 2026-09-16

The five operator pages now combine the archived H100 measurements above with
a fresh Hygon run. Device columns distinguish the timing and stopping tables;
the figures use separate device panels. Existing H100 raw data and failures
remain unchanged.

| Item | Fresh Hygon run | H100 comparison record |
| --- | --- | --- |
| Host | `a14r1n09` | `gxn70` |
| Device | One Hygon BW DCU, 65,520 MiB, wavefront 64 | One H100 80GB, physical GPU 5, warp 32 |
| Allocation | Slurm `838109`, `hx1hdnormal01`, 8 CPUs | Shared host; no exclusive GPU reservation or locked clocks |
| Torch | 2.7.1 | 2.11.0+cu128 |
| GPU runtime | HIP 6.3.26045 | CUDA 12.8 build |
| Triton | HCU Triton 3.1.0 | Triton 3.6.0 |
| Gate / graph dependencies | e3nn 0.4.4 / torch-geometric 2.6.1 | e3nn 0.4.4 / torch-geometric 2.6.1 |
| FastEq source | `d37eacaf21b5048159a1211794efbb58941485a9` | Archived timings: `3bc9a82d40ef646c3ce75f6f517e3f618fb12754` |
| LayerNorm suite after scope cleanup | 218 passed, 0 failed, 0 skipped | Separate existing regression: 218 passed, 0 failed, 0 skipped |

The four non-LayerNorm production operator files are byte-identical between
these runs. Hygon tests the current LayerNorm source after Merge removal,
SHA256 `35d689624e48422801cfdb0b8461243f7a88ee012dc20122206433098a90da5d`.
The H100 performance archive used the earlier source hash
`f2bc0ea975eac5bf5a16b3297415cc37d896fb8a41d5375e04e57d7ed1461520`; its
later 218-test correctness run is recorded separately. The old 305-test
CUDA/HIP record remains a dated historical result, not the current suite size.

All five original EQv3 reference files match the verified H100 source hashes.
The e3nn version is also matched. The harness retains the same shapes, native
Torch arithmetic, full-tensor comparisons, tolerances, seed, 5 warmups,
20 synchronized samples, fresh child processes and doubling-to-stop rules.
Path configuration and report aggregation were adapted to the Hygon machine.
Within each device, Torch and FastEq receive matched inputs; identical GPU
random streams across CUDA and HIP are not assumed.

The five families use six sweep keys because LayerNorm covers both the
per-degree and scalar/high-degree variants. The Hygon sweep includes first-order
input and parameter gradients, active p=0.3 Dropout, cached-CSR GraphSoftmax and
separate per-call CSR measurements. It does not add second-order or full-model
validation. FastEq's Torch products/reductions in LayerNorm backward and Torch
recomputation in AttentionAlpha backward are included in the measured calls.

Accuracy failures stay visible in the tables, raw records and red-cross plot
markers. `OOM` is an actual allocation failure; `LIMIT`, `INDEX_GUARD` and
`FALLBACK` stop before allocating a shape that violates the documented bound.
`ERROR` records a runtime failure, including HIP kernel-launch errors; it is
not an allocation OOM.
A successful FastEq-only point beyond Torch's memory ceiling has no matched
Torch correctness result. Speedups compare implementations on the same device;
the two software stacks and source revisions do not support a direct hardware
throughput ranking.

The run completed 328 execution points and 91 full-tensor comparisons
(**87 PASS, 4 FAIL**), producing 145 paired timing rows. The failures are
AttentionAlpha `alpha_dot` gradients at N=4,096 / 8,192 / 16,384 and
SeparableLayerNorm `affine_weight` at N=262,144. Separately, the repeated native
GraphSoftmax checks failed 2 of 1,720 assertions. The e3nn Gate comparisons
pass; the distinct EQv3 GateActivation API still lacks a callable forward.

Runtime launch errors stop Hygon FastEq AttentionAlpha at N=65,536, Dropout at
N=2,097,152, and GraphSoftmax forward at N=16,777,216. Both LayerNorm variants
and e3nn Gate reach the inherited indexing prechecks. All 24 backend/mode
sequences reached an explicit stopping condition. The single-DCU allocation
was released after collection; the measured Slurm steps completed normally.

Fresh records:

- [Manifest and source hashes](eqv3_validation/2026-09-16-hygon/manifest.json)
- [Paired timings and memory](eqv3_validation/2026-09-16-hygon/paired.csv), [boundaries](eqv3_validation/2026-09-16-hygon/boundaries.csv), and [summary](eqv3_validation/2026-09-16-hygon/summary.json)
- [Accuracy](eqv3_validation/2026-09-16-hygon/correctness_summary.json), [failed elements](eqv3_validation/2026-09-16-hygon/failures.csv), and [repeated native GraphSoftmax checks](eqv3_validation/2026-09-16-hygon/softmax_native_checks.json)
- [LayerNorm log](eqv3_validation/2026-09-16-hygon/layernorm.log) and [JUnit record](eqv3_validation/2026-09-16-hygon/layernorm.xml)
- [Raw measurement records](eqv3_validation/2026-09-16-hygon/raw_results.json.gz), [reproduction](eqv3_validation/2026-09-16-hygon/REPRODUCE.md), and [comparison plotting script](eqv3_validation/2026-09-16-hygon/plot_by_operator.py)

## GraphSoftmax and AttentionAlpha diagnosis

See the [2026-09-16 diagnosis](eqv3_validation/2026-09-16-diagnosis/REPORT.md) for
fixed-input replay, AttentionAlpha contraction ablations, H100 profiling,
and the documented HIP launch-dimension limit behind the archived error points.
This investigation preserves the original timing archives and production code.
