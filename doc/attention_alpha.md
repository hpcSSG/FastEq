# AttentionAlpha

[AttentionAlpha](../fasteq/triton/fused_attention_alpha.py) fuses normalization,
activation, dropout and channel-weighted reduction. Forward and first-order
backward are implemented in Triton. Backward reuses saved forward values;
it does not rebuild the forward expression with Torch autograd.
The latest implementation passes the recorded FP32 tests on both devices.
At the measured sizes, complete forward + backward is 2.67–6.44x native Torch
on H100 and 1.45–1.54x native Torch on Hygon BW.

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
and limits edge-level parallelism on Hygon. Each weight-gradient program now
prefetches eight separate lane vectors and accumulates them in the original
FP32 FMA order. This avoids repeated shared-memory layout conversions from
extracting rows of a two-dimensional tile on HCU Triton 3.1. The launch uses
one target warp/wavefront on both CUDA and HIP, without extra feature packing
or floating-point atomic accumulation. Splitting the remaining long chain
into parallel reductions would require renewed cancellation-case validation.

## Accuracy and machines

Tests were measured on 2026-09-17. H100 tests ran on `gxn70`, physical GPU 7,
Torch 2.11.0+cu128 / Triton 3.6.0. Hygon tests ran on `a14r2n09`, one BW/gfx936
DCU, Torch 2.7.1 / HCU Triton 3.1.0.
Both executed identical production source SHA-256
`a1fb1f31b9d07166b71171b9fa52da8d290a93bbad1585830f27d8fd4817593a`.

The reference uses native Torch FP32 LayerNorm, the original EQv3
SmoothLeakyReLU expression and the weighted `einsum` from the attention block.
Every output and requested `dX`, `dalpha_dot`, normalization weight and bias
gradient uses `atol=5e-5, rtol=5e-4`; no FP64-reference exception applies.
Each device passes **67 regression tests, zero failures/errors/skips, plus
13 stress cases**. Coverage includes irregular channels, strided inputs,
dropout with the same realized mask, gradient subsets, large edge counts and
padded parameter-gradient grids. The 66 existing cases and the added grid
case were run separately; both JUnit reports are retained for each device.

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
| hygon | alpha_dot | 0.0141602 | 0.840867 | 0 |
| hygon | norm_weight | 0.0317383 | 0.102361 | 0 |
| hygon | norm_bias | 0.0302734 | 0.160848 | 0 |

## Performance

Configuration: `E=32N`, H=8, C=32, LayerNorm + SmoothLeakyReLU, FP32, dropout 0.
Five warmups and 20 synchronized GPU-event samples; medians in milliseconds.
Forward + backward requests all input and parameter gradients. Provider order
rotates each iteration. The measured run included an independent control;
only Torch and the current implementation's measurements are retained here.
These are representative single-run comparisons on shared hosts.
Every retained timing point passes its complete FP32 comparison.

| Device | N | Torch forward | Triton forward | Torch forward + backward | Triton forward + backward | F+B speedup |
| --- | --- | --- | --- | --- | --- | --- |
| h100 | 4,096 | 2.362 | 0.148 | 4.658 | 1.742 | 2.67x |
| h100 | 32,768 | 18.540 | 0.614 | 35.653 | 5.847 | 6.10x |
| h100 | 131,072 | 73.976 | 1.774 | 142.254 | 22.105 | 6.44x |
| hygon | 4,096 | 7.925 | 0.375 | 19.097 | 12.362 | 1.54x |
| hygon | 32,768 | 63.001 | 2.402 | 150.714 | 103.925 | 1.45x |
| hygon | 131,072 | 251.726 | 7.878 | 601.252 | 403.942 | 1.49x |

The reference outputs and gradients are staged on the CPU, and GPU comparison
temporaries are released before timing. All three sizes complete on both
devices. These bounded measurements do not establish a general OOM limit.
GPU events include host dispatch gaps, which particularly affect small calls
on the shared H100 host; compare against the same run's Torch measurements.
The latest implementation has no validated full-scale performance curve beyond
the points above. Full-model training and second-order gradients are outside
this validation; the Triton backward explicitly rejects higher-order autograd.

## Records and reproduction

[Source/environment summary](validation/attention_alpha/summary.json),
[H100 pytest](validation/attention_alpha/h100/pytest.xml),
[Hygon pytest](validation/attention_alpha/hygon/pytest.xml),
[H100 grid regression](validation/attention_alpha/h100/pytest_grid.xml),
[Hygon grid regression](validation/attention_alpha/hygon/pytest_grid.xml),
[H100 stress](validation/attention_alpha/h100/stress.json), and
[Hygon stress](validation/attention_alpha/hygon/stress.json) contain the latest
results. Per-size JSON files are listed in the summary.
Use the [regression tests](../test/test_triton_attention_alpha.py),
[matched benchmark](../test/benchmark_triton_attention_alpha_matched.py),
[stress script](../test/stress_triton_attention_alpha.py), and
[reproduction instructions](validation/REPRODUCE.md).
