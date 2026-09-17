# AttentionAlpha

[AttentionAlpha](../fasteq/triton/fused_attention_alpha.py) fuses normalization,
activation, dropout and channel-weighted reduction. Forward and first-order
backward are implemented in Triton. Backward reuses saved forward values;
it does not rebuild the forward expression with Torch autograd.
The latest implementation passes the recorded FP32 tests on both devices.
H100 shows a substantial forward + backward speedup; Hygon backward remains
slower than native Torch.

## Implementation

The input is `[E,H,C]`; the result is `[E,H]`. The adapter supports optional
LayerNorm and affine parameters, SiLU or SmoothLeakyReLU, training dropout,
noncontiguous inputs and requested-gradient subsets. It saves only the
statistics/normalized values, activated features and dropout mask needed by
the requested gradients. Triton computes row gradients and affine partials,
then reduces the parameter gradients. The weight contraction uses a separate
Triton kernel; “forward + backward” includes that kernel and final reductions.

The mathematical kernels are shared by CUDA and HIP. The current source does
retain a **device-dependent accumulation schedule** for `alpha_dot` gradients:
`gfx936` uses one FP32 accumulator chain; other targets use the target lane
width. This is not a second mathematical implementation, but it means the
current operator is not completely free of device-specific scheduling.
The serialized chain preserves the reproduced Torch FP32 cancellation cases
and contributes to Hygon's high backward cost. No faster schedule is claimed
validated here.

## Accuracy and machines

H100 tests ran on `gxn70`, physical GPU 5, Torch 2.11.0+cu128 / Triton 3.6.0.
Hygon tests ran on `a14r1n06`, one BW/gfx936 DCU, Torch 2.7.1 / HCU Triton 3.1.0.
Both executed identical production source SHA-256
`e5d339e77faa9d36ac816bad12f6edb7639b63e8d9940487eb12186024ab8002`.

The reference uses native Torch FP32 LayerNorm, the original EQv3
SmoothLeakyReLU expression and the weighted `einsum` from the attention block.
Every output and requested `dX`, `dalpha_dot`, normalization weight and bias
gradient uses `atol=5e-5, rtol=5e-4`; no FP64-reference exception applies.
Each device passes **66 regression tests, zero failures/errors/skips, plus
13 stress cases**. Coverage includes irregular channels, strided inputs,
dropout with the same realized mask, gradient subsets and large edge counts.

The following maxima cover retained benchmark and stress JSON records
(they do not claim to summarize every pytest assertion). Absolute-error and
ratio maxima can come from different elements:

| Device | Component | Maximum absolute error | Maximum tolerance ratio | Failing elements |
| --- | --- | --- | --- | --- |
| h100 | output | 1.52588e-05 | 0.085633 | 0 |
| h100 | x | 1.33514e-05 | 0.046742 | 0 |
| h100 | alpha_dot | 0.00488281 | 0.612996 | 0 |
| h100 | norm_weight | 0.00477982 | 0.782690 | 0 |
| h100 | norm_bias | 0.00585938 | 0.231342 | 0 |
| hygon | output | 3.43323e-05 | 0.176641 | 0 |
| hygon | x | 1.52588e-05 | 0.028061 | 0 |
| hygon | alpha_dot | 0.00463867 | 0.840867 | 0 |
| hygon | norm_weight | 0.0090332 | 0.102361 | 0 |
| hygon | norm_bias | 0.00488281 | 0.160848 | 0 |

## Performance

Configuration: `E=32N`, H=8, C=32, LayerNorm + SmoothLeakyReLU, FP32, dropout 0.
Five warmups and 20 synchronized GPU-event samples; medians in milliseconds.
Forward + backward requests all input and parameter gradients. Torch is
measured before Triton; these are representative single-run comparisons.
Every retained timing point passes its complete FP32 comparison.

| Device | N | Torch forward | Triton forward | Torch forward + backward | Triton forward + backward | F+B speedup |
| --- | --- | --- | --- | --- | --- | --- |
| h100 | 4,096 | 2.340 | 0.101 | 4.539 | 0.836 | 5.43x |
| h100 | 32,768 | 18.431 | 0.547 | 35.645 | 5.997 | 5.94x |
| h100 | 131,072 | 73.969 | 1.753 | 142.282 | 22.771 | 6.25x |
| hygon | 4,096 | 7.916 | 0.361 | 19.087 | 36.839 | 0.52x |
| hygon | 32,768 | 63.000 | 2.398 | 150.762 | 296.735 | 0.51x |

The N=131,072 Hygon attempt exhausted device memory during the native benchmark;
no complete matched timing is retained for that size. This was not a clean
fresh-process doubling sweep and does not establish a general OOM limit.
The latest implementation has no validated full-scale performance curve beyond
the points above. Full-model training and second-order gradients are outside
this validation; the Triton backward explicitly rejects higher-order autograd.

## Records and reproduction

[Source/environment summary](validation/attention_alpha/summary.json),
[H100 pytest](validation/attention_alpha/h100/pytest.xml),
[Hygon pytest](validation/attention_alpha/hygon/pytest.xml),
[H100 stress](validation/attention_alpha/h100/stress.json), and
[Hygon stress](validation/attention_alpha/hygon/stress.json) contain the latest
results. Per-size JSON files are listed in the summary.
Use the [regression tests](../test/test_triton_attention_alpha.py),
[benchmark](../test/benchmark_triton_attention_alpha.py),
[stress script](../test/stress_triton_attention_alpha.py), and
[reproduction instructions](validation/REPRODUCE.md).
