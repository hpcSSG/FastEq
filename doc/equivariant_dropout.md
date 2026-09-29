# Equivariant Dropout

[EquivariantDropout](../fasteq/triton/fused_equivariant_dropout.py) applies masks
shared across the components of each representation and reuses the mask in
backward. Both devices pass every recorded output/input-gradient comparison
with the same realized mask. Hygon has a separate large-shape launch limit.

## Accuracy and configuration

The current source SHA-256 is
`48444ccc2b51bf6268642fc071d4e04caa4e235354a1896aa051091db86abb3a`,
identical to both measured versions. H100 `gxn70` GPU 5 passes **17 shape
comparisons**; Hygon BW/gfx936 `a14r1n09` passes **16**. Additional self-tests,
C=13 strided inputs and exactly-zero-input mask replay pass. Versions are in
the [machine table](eqv3_validation.md#machines-and-software).

The reference is original EQv3 `EquivariantDropout` in `drop.py`, SHA-256
`293b128aa1b407154aae2e1a8c77c16d6957161c1a65a4396d3d91f6f35a4b3f`.
Inputs are `[N,16,128]`, Lmax=3, FP32, dropout p=0.3 and `training=True`.
This training setting remains active during forward `no_grad` timing.
Correctness replays an identical mask into native Torch and requires exact
equality: recorded maximum absolute errors for output and `dX` are zero.
Timing lets each implementation generate its mask normally. Matching masks
for correctness does not imply matching Torch/Triton random streams.

## Performance

Five warmups, 20 synchronized GPU-event samples, median milliseconds.
N=4,096 and the largest matched size for each mode are shown. Small forward +
backward calls are slower than Torch; the larger matched cases show a benefit.

| Device | N | Mode | Torch ms | FastEq ms | Speedup | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| H100 | 4,096 | Forward | 0.1083 | 0.1767 | 0.61x | PASS |
| H100 | 2,097,152 | Forward | 39.0111 | 14.6830 | 2.66x | PASS |
| H100 | 4,096 | Forward + backward | 0.2162 | 0.2557 | 0.85x | PASS |
| H100 | 1,048,576 | Forward + backward | 27.8281 | 14.7071 | 1.89x | PASS |
| Hygon BW | 4,096 | Forward | 0.2967 | 0.2680 | 1.11x | PASS |
| Hygon BW | 1,048,576 | Forward | 96.4064 | 30.0622 | 3.21x | PASS |
| Hygon BW | 4,096 | Forward + backward | 0.3786 | 0.4650 | 0.81x | PASS |
| Hygon BW | 1,048,576 | Forward + backward | 116.6152 | 60.0472 | 1.94x | PASS |

## Execution limits

| Device | Mode | Implementation | Last runnable N | Next N | Stop |
| --- | --- | --- | --- | --- | --- |
| h100 | Forward | torch | 2,097,152 | 4,194,304 | OOM |
| h100 | Forward | fused | 4,194,304 | 8,388,608 | OOM |
| h100 | Forward + backward | torch | 1,048,576 | 2,097,152 | OOM |
| h100 | Forward + backward | fused | 2,097,152 | 4,194,304 | OOM |
| hygon | Forward | torch | 2,097,152 | 4,194,304 | OOM |
| hygon | Forward | fused | 1,048,576 | 2,097,152 | ERROR |
| hygon | Forward + backward | torch | 1,048,576 | 2,097,152 | OOM |
| hygon | Forward + backward | fused | 1,048,576 | 2,097,152 | ERROR |

Hygon's `ERROR` is `Triton Error [HIP]: Code: 1, invalid argument` at kernel
launch, not an allocation OOM. Successful FastEq-only sizes above Torch's
memory ceiling have no matched Torch correctness check. No complete-model
training or higher-order autograd result is established here.

## Records

[H100 accuracy](validation/dropout/h100/correctness.json),
[Hygon accuracy](validation/dropout/hygon/correctness.json),
[H100 timings](validation/dropout/h100/paired.csv),
[Hygon timings](validation/dropout/hygon/paired.csv), and the adjacent raw
samples, extra checks, source manifests and boundary files record only this
unchanged source. See [reproduction](validation/REPRODUCE.md).
