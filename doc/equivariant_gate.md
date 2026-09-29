# Equivariant Gate

[FastEquivariantGate](../fasteq/triton/fused_equivariant_gate.py) fuses scalar
activation and broadcast gating of higher-order features. The validated
interface is **e3nn.nn.Gate** from e3nn 0.4.4. The separate EQv3
[`fused_gate_activation`](../fasteq/triton/fused_gate_act.py) class still has no
`forward()` method and fails to run on both devices; these Gate results do not
validate that interface or full EQv3 integration.

## Accuracy and configuration

H100 `gxn70` GPU 5 and Hygon BW/gfx936 `a14r1n09` each pass all **15
scaling/small-shape comparisons**, plus irregular/strided-input checks. The
current implementation is byte-identical to both measured sources, SHA-256
`3a9e8211dd2c16ed96f47cdcbfa30d860ef521cd23b63c69e432eacf46834858`.
See the [machine table](eqv3_validation.md#machines-and-software) for versions.

Lmax=3, C=128: input `[N,2432]`, output `[N,2048]`, scalar SiLU and gate sigmoid.
FP32 output and input gradients use `atol=5e-5, rtol=5e-4`. Maximum recorded
tolerance ratios (output / input gradient) are 0.000782 / 0.013518 on H100
and 0.000643 / 0.012412 on Hygon. These are first-order standalone checks.

## Performance

Five warmups, 20 synchronized GPU-event samples, median milliseconds.
The table includes N=4,096 and the largest matched size for each mode.

| Device | N | Mode | Torch ms | FastEq ms | Speedup | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| H100 | 4,096 | Forward | 0.2278 | 0.1048 | 2.17x | PASS |
| H100 | 524,288 | Forward | 16.5866 | 5.1104 | 3.25x | PASS |
| H100 | 4,096 | Forward + backward | 1.2874 | 0.4136 | 3.11x | PASS |
| H100 | 524,288 | Forward + backward | 102.0716 | 34.3886 | 2.97x | PASS |
| Hygon BW | 4,096 | Forward | 0.6994 | 0.4930 | 1.42x | PASS |
| Hygon BW | 524,288 | Forward | 53.6779 | 47.5115 | 1.13x | PASS |
| Hygon BW | 4,096 | Forward + backward | 3.0166 | 2.5921 | 1.16x | PASS |
| Hygon BW | 524,288 | Forward + backward | 303.3184 | 313.4137 | 0.97x | PASS |

Hygon's largest forward + backward point is slightly slower than Torch.
The same callable e3nn-compatible API is used throughout.

## Execution limits

| Device | Mode | Implementation | Last runnable N | Next N | Stop |
| --- | --- | --- | --- | --- | --- |
| h100 | Forward | torch | 1,048,576 | 2,097,152 | OOM |
| h100 | Forward | fused | 524,288 | 1,048,576 | INDEX_GUARD |
| h100 | Forward + backward | torch | 1,048,576 | 2,097,152 | OOM |
| h100 | Forward + backward | fused | 524,288 | 1,048,576 | INDEX_GUARD |
| hygon | Forward | torch | 1,048,576 | 2,097,152 | OOM |
| hygon | Forward | fused | 524,288 | 1,048,576 | INDEX_GUARD |
| hygon | Forward + backward | torch | 1,048,576 | 2,097,152 | OOM |
| hygon | Forward + backward | fused | 524,288 | 1,048,576 | INDEX_GUARD |

`INDEX_GUARD` is the harness's precheck for signed 32-bit row-offset overflow;
the next shape is not allocated, and the source lacks an equivalent runtime
guard. This is not a FastEq OOM. Full-model integration is not established.

## Records

[H100 accuracy](validation/gate/h100/correctness.json),
[Hygon accuracy](validation/gate/hygon/correctness.json),
[H100 timings](validation/gate/h100/paired.csv),
[Hygon timings](validation/gate/hygon/paired.csv), and adjacent raw records,
extra checks, source manifests and boundary files retain the latest evidence
for this unchanged source. See [reproduction](validation/REPRODUCE.md).
