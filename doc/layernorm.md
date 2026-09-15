# Equivariant LayerNorm

FastEq fuses equivariant normalization for EquiformerV3 and EquiformerV2 layers,
with CUDA FP32 forward execution and first-order gradients.

## Supported layers

Inputs have shape `[N, (L + 1)**2, C]`, where `N` is the number of atoms, `L` is
the maximum degree, and `C` is the channel count. Statistics are computed
independently for each atom. Let `q_l` be the mean squared feature value across
the `2*l + 1` components and `C` channels of degree `l`, after scalar centering.

| Model | Native layer | Default normalization statistics |
| --- | --- | --- |
| EquiformerV3 | `EquivariantLayerNorm` | Each degree uses its own `q_l`. |
| EquiformerV3 | `EquivariantMergeLayerNorm` | All degrees share `mean(q_0, ..., q_L)`. |
| EquiformerV3 | `EquivariantSeparableLayerNorm` | Scalars use `q_0`; higher degrees share `mean(q_1, ..., q_L)`. |
| EquiformerV2 | `EquivariantLayerNormArraySphericalHarmonics` | Scalars use `q_0`; higher degrees share `mean(q_1, ..., q_L)`. |

The table uses the native defaults, including equal weighting of degrees where
applicable. The adapter preserves the source layer's normalization options.
Only scalar features (`l = 0`) are centered across channels; Merge can disable
centering. Affine scales are learned per degree and channel and shared across
that degree's components. Bias applies only to scalars when the source has it.

## Fused operations

Forward combines scalar mean, centering, squaring, weighted group reduction,
reciprocal square root with epsilon, and affine transformation in one kernel.
Each group supplies its normalization scale to the degrees listed above.

Backward combines the group reductions needed for the input gradient `dX`
with per-atom contributions to the affine parameter gradients. Subsequent
reductions across atoms produce `dgamma` and `dbeta`. Only requested gradients
are computed. This supports first-order differentiation; double backward and
`create_graph=True` are unsupported.

## Usage

```python
from fasteq.triton import from_reference

# source is the original FP32 CUDA layer; x has shape [N, (L + 1)**2, C].
op = from_reference(source)
y = op(x.requires_grad_(True))
y.square().mean().backward()
```

The adapter reuses the source parameters, so gradients reach those parameters.
Apply it to the individual normalization layers you want to replace. Both
contiguous NKC storage and KNC storage viewed as `[N, K, C]` are supported,
where `K = (L + 1)**2`. CUDA FP32 is the supported execution target.

## Validation coverage

All four native layers in the table have been compared against their unmodified
PyTorch FP32 implementations on NVIDIA H100. Comparisons cover the output,
input gradient, and all affine parameter gradients, including NKC and KNC
storage. See the [native-layer comparison tests](../test/test_triton_equivariant_layer_norm_precision.py)
and [backward tests](../test/test_triton_equivariant_layer_norm_backward.py).

This validation covers individual model layers. Full-model inference and
training with this shared implementation have not been validated. eSEN has
not been directly tested.
