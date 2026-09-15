# FastEq

**High-performance GPU operators for equivariant neural networks.**

FastEq accelerates the geometric and equivariant computations at the core of atomistic machine learning. It brings together optimized tensor products, spherical harmonics, rotations, linear maps, and fused neural network operations for energy prediction, force evaluation, and model training.

The project combines structure-aware kernel generation with Triton-based operator fusion. Its integration effort targets PyTorch workflows and interfaces used by e3nn and cuEquivariance, with cross-platform development.


## Highlights

- **Structure-aware tensor products.** Exploit Clebsch–Gordan selection rules, specialize nonzero computation paths, and reuse intermediate values across paths.
- **Fused GPU execution.** Combine compatible operations to reduce kernel launches and intermediate memory traffic, including activation and gating, graph softmax, and rotation-related data movement.
- **Second-order derivative operators for training.** Support forward, backward, and double-backward computation for selected operators, enabling model training with force-based losses.
- **Model integration.** Adapt core operators to the layouts and conventions used in equivariant message-passing networks and transformers.
- **Hardware-aware optimization.** Tune tiling, scheduling, accumulation, and register/shared-memory placement for each workload and backend.

## Operator Catalog

Model associations below follow the project operator inventory. Exact usage depends on the model variant and implementation. Operator names are descriptive catalog names and do not imply a stable Python import path.

| Category | Operator | Function | Relevant model families | Feature | 
| --- | --- | --- | --- | --- |
| Geometric encoding | `SphericalHarmonics` | Encode directions as spherical harmonic features. | Allegro, EquFlash, EquFlashV2, MACE, NequIP, SevenNet, TACE/TECE | Fused SphericalHarmonics |
| Rotations and representation transforms | `FusedSO3Rotation` | Apply SO(3) rotations using Wigner-D matrices, with fusion of compatible surrounding operations. | EquiformerV3, EquiformerV2 | Fused pattern: Gather + Merge + SO3_Rotate |
| Rotations and representation transforms | `FusedSO3Grid` | Transform between spherical harmonic coefficients and spherical grid signals. | EquiformerV3, EquiformerV2, eSEN | Fused pattern:  so3_grid.to_grid + activation + so3_grid.from_grid |
| Tensor products and higher-order coupling | `ChannelwiseTensorProduct` | Couple equivariant representations through channelwise Clebsch–Gordan tensor products. | EquFlash, MACE, NequIP, SevenNet, TACE/TECE | Register optimization and path scheduling by JIT |
| Tensor products and higher-order coupling | `FullyConnectedTensorProduct` | Mix channels across allowed irreducible-representation coupling paths. | MACE, NequIP, SevenNet | Fused Gather + BatchGEMM + CG sparse  |
| Tensor products and higher-order coupling | `SymmetricContraction` | Construct higher-order equivariant features through symmetric contractions. | MACE, TACE, TECE | Similar to ChannelwiseTensorProduct |
| Equivariant linear maps | `SO3Linear` | Apply channel mixing between matching irreducible representations. | EquFlash, MACE, NequIP, SevenNet, TACE/TECE, EquiformerV3, EquiformerV2, eSEN | Highly optimized fused TF32 GEMM and Triton FP32/FP64 fused GEMM | 
| Equivariant linear maps | `SO2Linear` | Apply SO(2)-equivariant channel mixing in local coordinate frames. | EquiformerV3, TACE/TECE | Triton FP32/FP64 fused GEMM  |
| Graph Softmax | [`FusedGraphSoftmax`](doc/graph_softmax.md) | Normalize attention logits over graph neighborhoods, with optional soft capping and exponential rescaling or dropout. | EquiformerV3; TACE/TECE attention integration targets | Fused GraphSoftmax |
| Graph attention | [`FusedAttenAlpha`](doc/attention_alpha.md) | Fuse normalization, activation, dropout, and weighted reduction in graph attention into a single kernel | EquiformerV3; TACE/TECE attention integration targets | Fused Norm + Act + Dropout + Reduce |
| Equivariant Gate| [`FusedEquivariantGate`](doc/equivariant_gate.md) | Fuse scalar activations with broadcast gating of higher-order features. | EquFlash, SevenNet, NequIP; EquiformerV3 gate variants | Fused e3nn.nn.Gate |
| Equivariant normalization | [`FusedEquivariantLayerNorm`](doc/layernorm.md) | Normalize features while preserving the required SO(3) representation structure. | EquiformerV3, EquiformerV2 | Fused statistics and affine transformation; first-order backward |
| Equivariant dropout | [`FusedEquivariantDropout`](doc/equivariant_dropout.md) | Apply dropout with masks shared across components as required to preserve equivariance. | EquiformerV3 | Fused Dropout |

Graph softmax operates on attention weights; it supports equivariant attention but does not itself perform a representation rotation or tensor-product coupling.

### Equivariant LayerNorm

CUDA FP32 normalization with first-order gradients for EquiformerV3 and
EquiformerV2 layers. See [Equivariant LayerNorm](doc/layernorm.md) for supported
layers, fused operations, usage, and validation coverage.

## How FastEq Optimizes Equivariant Computation

### Sparse path specialization

Clebsch–Gordan tensor products contain structured zeros imposed by selection rules. FastEq's tensor-product development builds on the FastTP approach: represent nonzero paths explicitly, specialize indices and coefficients at compilation time, and expose reuse across paths.

Scheduling balances reuse against register pressure and occupancy. Depending on the workload, implementations can use register storage, shared-memory staging, or separate kernels to manage live intermediate values.

### Operator fusion

Many equivariant layers combine small transformations with elementwise operations and irregular data movement. FastEq targets these sequences with fused implementations, including:

- Scalar activation and higher-order feature gating.
- Neighborhood softmax and compatible attention-weight transformations.
- Node-feature gathering, source/target merging, radial weighting, and rotation where layouts permit.
- Spherical-grid transforms and compatible nonlinear operations.

Fusion is selected by measurement: its benefit depends on tensor shape, memory layout, reuse, and the efficiency of the unfused baseline.

### Second-order derivative operators for training

FastEq supports the second-order derivatives required during training through specialized backward and double-backward implementations for selected operators. These implementations enable gradients to propagate through force predictions when optimizing energy-and-force losses. Gradient computation can be restricted to inputs that require it, avoiding unnecessary work and temporary storage.

For atomistic models, energy inference, force inference, and force training have different differentiation requirements. Verify the complete operator path before using an inference-oriented implementation for training. Force evaluation requires coordinate gradients even when model parameters are not being optimized.

## Backends

FastEq combines Triton-based kernels with generated tensor-product implementations. Backend development includes CUDA/HIP paths and adaptation toward domestic AI accelerators through FlagOS. Support is operator-specific; cross-platform execution and performance must be validated on the target hardware.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/malixian/FastEq.git
git clone -b fasteq https://github.com/malixian/cuEquivariance_torch.git
```


### 2. Create an isolated environment

Using Conda:

```bash
conda create -n fasteq python=3.10 -y
conda activate fasteq
```

Install the Python dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Build FastEq

For an NVIDIA H100 (`sm_90a`):

```bash
TORCH_CUDA_ARCH_LIST="9.0a" \
pip install -e . --no-build-isolation
```

For another NVIDIA GPU, replace `9.0a` with the compute capability of the
target device.

To perform a non-editable installation:

```bash
TORCH_CUDA_ARCH_LIST="9.0a" \
pip install . --no-build-isolation
```

Do not run both editable and non-editable installation commands in the same
environment unless intentionally reinstalling the package.

### 4. Install the patched cuEquivariance interface

Install the required cuEquivariance packages:

```bash
pip install \
  cuequivariance==0.8.0 \
  cuequivariance-torch==0.8.0 \
```

Install the patched Python interface:

```bash
cd 3rdparty/cuequivariance_torch/cuequivariance_torch
pip install hatchling twine editables
pip install -e . --no-build-isolation
cd ../../..
```
