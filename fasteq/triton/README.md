# Equivariant LayerNorm

`fused_equivariant_layer_norm.py` provides a shared CUDA FP32 forward implementation
for per-degree, merged, and scalar/high-degree equivariant normalization. It has
no dependency on Equiformer, fairchem, or e3nn. PyTorch and Triton are required;
the implementation was tested with PyTorch 2.8.0+cu128 and Triton 3.4.0 on H100.

## Two versions, one contract

| Version | Execution | Intermediate statistics |
| --- | --- | --- |
| `v0` | Scalar mean, group statistics, affine output in three kernels | Mean, second moment, reciprocal standard deviation |
| `v1` (default) | One kernel; each program loads one atom, computes all its groups, then writes the output | None in ordinary forward |

With centering disabled, v0 omits the mean kernel. Both versions preserve scalar-only
centering/bias and degree-wise affine weights. v1 reuses the loaded input and squared
values across groups; group boundaries are compile-time metadata. It uses no
model-specific shape branches or tuning tables. Both versions directly index affine
parameters and write the final output, avoiding expanded weights and per-degree `cat`.

For FP32, the main-tensor logical traffic is `12*N*K*C + 4*N*C` bytes for centered v0
and `8*N*K*C` for v1, where `K=(lmax+1)**2`. This is a 34.21% reduction at K=25.
These estimates exclude parameters and statistics and are not measured DRAM traffic.

## Usage

```python
import torch
from fasteq.triton import EquivariantNormSpec, TritonEquivariantNorm

spec = EquivariantNormSpec.from_preset(
    lmax=4,
    channels=128,
    grouping="per_degree",       # "all" or "scalar_high" also supported
    weighting="component",       # "degree_balanced" or "norm" also supported
    center_scalar=True,
    eps=1e-5,
)
op = TritonEquivariantNorm(
    spec, version="v1", reduction_order="components_first", device="cuda"
)
x = torch.randn(108, 25, 128, device="cuda")
gamma = torch.ones(5, 128, device="cuda")
beta = torch.zeros(128, device="cuda")
with torch.inference_mode():
    y = op(x, weight=gamma, bias=beta)
    debug = op.forward_with_stats(x, weight=gamma, bias=beta)
    # debug.output; debug.mean [N]; debug.moments / debug.rstd [N, G]
```

In a source checkout without built FastEq native extensions, set
`FASTEQ_BACKEND=cpu` **before importing FastEq**. This skips the package's native
extension loader; these Triton operators still execute on CUDA tensors. It is
not a native-build option for `setup.py` or `pip install`.

The logical input is `[N,(lmax+1)**2,C]`, with all m components present and equal C
across degrees. Inputs may have NKC or KNC storage; outputs are contiguous NKC.
`weight` accepts `[L+1,C]` or `(scalar[C], higher[L,C])`, including positive strides.
`weight=None` and `bias=None` independently disable affine scale and scalar bias.

For a group S, the second moment is
`mean_c(sum_{l in S,m} a_l * Z[n,l,m,c]**2)`, where only the scalar row of Z is
centered. The coefficients are `1/sum_{l in S}(2*l+1)` for component weighting,
`1/(len(S)*(2*l+1))` for degree balancing, and 1 for norm weighting.
Each output uses its group's `rsqrt(moment+eps)` and computes
`Z*(rstd*gamma) + scalar_bias`. Coefficients and gamma are shared over m within
each degree, preserving the equivariant structure.

## Existing source modules

```python
from fasteq.triton import from_reference

# source is the original FP32 module on the intended device.
op = from_reference(source, version="v1")
with torch.inference_mode():
    y = op(x)
```

| Source class | Grouping | Reduction order |
| --- | --- | --- |
| `EquivariantLayerNorm`, `EquivariantLayerNormArray` | Each degree separately | Components, then channels |
| `EquivariantMergeLayerNorm` | All degrees | Channels, then weighted components |
| `EquivariantLayerNormArraySphericalHarmonics` | Scalar / higher degrees | Weighted components, then channels |
| `EquivariantSeparableLayerNorm` | Scalar / higher degrees | Channels, then weighted components |

Adapters read the source configuration and exact cached FP32 balance coefficients
once. Parameters remain live references: parameter replacement and `.to(device)`
are supported. The adapter's state keys have a `source.` prefix;
`op.source.state_dict()` retains the source keys. Recreate the adapter if epsilon,
normalization, centering, or statistic weights change. New contiguous partitions
can use `EquivariantNormSpec` directly without a source-class adapter.

## Validation and timing

From the repository root, in an environment with PyTorch, Triton and pytest:

```bash
PYTHONPATH=. FASTEQ_BACKEND=cpu CUDA_VISIBLE_DEVICES=1 \
  python -m pytest -q test/test_triton_equivariant_layer_norm.py

PYTHONPATH=. FASTEQ_BACKEND=cpu CUDA_VISIBLE_DEVICES=1 \
  python test/benchmark_triton_equivariant_layer_norm.py \
  --output /tmp/equivariant_layer_norm_benchmark.json
```

Select a free GPU. `TRITON_CACHE_DIR` can redirect the compilation cache to a
writable directory. The portable benchmark checks correctness before measuring
v0/v1; it does not label the mathematical reference as original-model performance.

The default tests require no external model checkout. Set
`FASTEQ_EQUIFORMER_V3_LAYER_NORM` and `FASTEQ_EQUIFORMER_V2_LAYER_NORM` to the original
`layer_norm.py` file paths to enable six additional source-adapter comparisons.
Integration validation on H100 passed all 139 tests with those comparisons enabled;
with CUDA hidden, 13 CPU tests passed and 126 GPU tests skipped. The repository's
12-case v0/v1 benchmark also passed correctness and showed lower v1 GPU latency in
every case. Add `--eager` to record dispatch/allocation/synchronization wall times
separately from CUDA Graph timings.

The pre-integration source audit used EquiformerV3 commit
`a7300c58df683dc99cb48027d5bfd4c887486c48`. Its 648 ordinary source comparisons had
maximum absolute error `1.43e-6`. A separate 60-case H100 sweep covered both layouts,
N=108 through 27648 and L/C=2/32, 3/96, 4/128, 6/64. v1 was faster than v0 in all
those cases, with CUDA Graph v0/v1 ratios of 1.16–1.69 for per-degree normalization,
1.47–2.10 for merged normalization, and 1.31–1.98 for scalar/high normalization.
These are standalone forward results, not model end-to-end speedups.

## Scope and numerical limits

- FP32 CUDA inference only; no backward or implicit PyTorch fallback. Use
  `torch.no_grad()` or `torch.inference_mode()` when parameters require gradients.
- GPU groups must be ordered, contiguous, disjoint full-degree ranges whose
  statistic sources equal the output degrees using that statistic. Overlapping
  statistics, cross-group mappings, and noncontiguous groups are rejected even
  though the mathematical spec can express them.
- Execution tiles are limited to 65536 padded elements: v0 checks its largest
  group tile, v1 checks its whole-atom tile. Input element count and parameter
  relative offsets must fit signed 32-bit indexing.
- Different FP32 reduction trees are tolerance-equivalent, not bitwise equivalent
  to source implementations. SH's scalar `nn.LayerNorm` is particularly sensitive
  to large means with tiny variance. At mean 1024 and standard deviation about
  0.01, observed source differences were 0.00652/0.00149 for v0/v1; the original
  source differed from FP64 by about 0.00675. Those cases do not pass the ordinary
  `3e-5` tolerance and were recorded separately, not counted as ordinary accuracy
  passes. This implementation does not promise source-level agreement at that
  tolerance for arbitrarily ill-conditioned inputs.
