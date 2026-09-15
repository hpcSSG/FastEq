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

This sweep measures standalone operators on CUDA, without full-model timing
or new HIP validation. Earlier CUDA/HIP LayerNorm results retain their own
source hashes and are described in [LayerNorm](layernorm.md).

## Shared records

- [Paired timings, speedups and memory](eqv3_validation/2026-09-15/paired.csv)
- [Execution and stopping boundaries](eqv3_validation/2026-09-15/boundaries.csv)
- [Accuracy summary](eqv3_validation/2026-09-15/correctness_summary.json) and [failures](eqv3_validation/2026-09-15/failures.csv)
- [Raw points and comparisons](eqv3_validation/2026-09-15/raw_results.json.gz), [source manifest](eqv3_validation/2026-09-15/manifest.json) and [source audit](eqv3_validation/2026-09-15/reference_audit.json)
- [Reproduction instructions](eqv3_validation/2026-09-15/REPRODUCE.md)
- [Per-operator plotting script](eqv3_validation/2026-09-15/plot_by_operator.py)

The original combined report and plots remain in the dated artifact directory
as a historical snapshot. The five pages above are the operator documentation.
