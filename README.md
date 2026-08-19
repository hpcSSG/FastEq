# FastEq

FastEq is a high-performance library for accelerating sparse tensor-product
operators in SO(3)-equivariant neural networks. It provides optimized
implementations of the core operators used by modern equivariant models while
preserving compatibility with the `cuequivariance_torch` API.

FastEq currently supports:

- **ChannelWiseTensorProduct (CWTP)**
- **MessagePassingTensorProduct (MPTP)**, implemented as CWTP fused with
  `scatter_sum`
- **SymmetricContraction (STC)**
- **EquivariantLinear(equi-linear)** 

FastEq uses hardware-aware JIT compilation, static path specialization,
path scheduling, data placement, and candidate benchmarking to generate
efficient GPU implementations.

## Reproducibility

This repository contains the source code required to build FastEq, integrate it
with supported equivariant models, and reproduce the computational experiments
reported in the paper.

The experiments do not introduce a new dataset. They use existing model
packages, molecular structures, and benchmark configurations described in the
paper and experiment scripts.

### Reproducibility checklist coverage

The repository provides:

- source code for the proposed operators and optimization methods;
- comments and implementation structure corresponding to the paper design;
- hardware and software requirements;
- installation instructions;
- example operator usage;
- benchmark entry points and experiment configurations;
- deterministic benchmark settings where randomness is not involved;
- repeated timing measurements for latency evaluation.

Exact benchmark commands, model checkpoints, structures, run counts, and
reported configurations should be kept synchronized with the final paper and
the scripts under the benchmark directories.

## Requirements

### Hardware

The NVIDIA implementation requires a CUDA-capable GPU. The main NVIDIA
experiments in the paper use an NVIDIA H100 GPU.

The paper also reports results on a HYGON DCU BW1000 where supported by the
corresponding software stack.

| Device | FP32 peak | FP64 peak | Memory |
|---|---:|---:|---:|
| NVIDIA H100 | 66.9 TFLOPS | 33.5 TFLOPS | 80 GB HBM |
| HYGON DCU BW1000 | 60 TFLOPS | 30 TFLOPS | 64 GB HBM |

### Software

Minimum requirements:

- Python 3.10
- CUDA Toolkit 12 or later
- PyTorch 2.4.1 or later
- cuEquivariance 0.8.0

Baseline versions used in the paper:

| Software | Version |
|---|---|
| e3nn | 0.4.4 |
| cuEquivariance | 0.10.0 |
| OpenEquivariance | 0.5.4 |
| FlashTP | commit `0fbbbbae1061afc9285092a939d3f9abd851a758` |

Use the exact PyTorch and CUDA versions recorded by the experiment scripts or
environment file when reproducing the final paper results.

## Installation

### 1. Clone the repository

git clone -b v0.3.0 https://github.com/malixian/FastEq.git
git clone -b fasteq https://github.com/malixian/cuEquivariance_torch.git
git clone -b fasteq https://github.com/malixian/mace.git
git clone -b fasteq https://github.com/malixian/SevenNet.git


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

### 4. Install the patched MACE package

```bash
cd 3rdparty/mace
pip install -e . --no-build-isolation
cd ../..
```

The patched package integrates FastEq operators into the MACE evaluation
workflow.

### 5. Install the patched cuEquivariance interface

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

## Using FastEq

FastEq preserves the `cuequivariance_torch` interface. In supported operators,
enable the optimized implementation with:

```python
use_fasteq=True
```

The following examples assume that `torch`, `cuequivariance` as `cue`, and
`cuequivariance_torch` as `cuet` have been imported and that the irreducible
representations and layout have been defined by the model.

### Channel-wise tensor product

```python
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet

conv_tp = cuet.ChannelWiseTensorProduct(
    cue.Irreps(cueq_config.group, irreps_in1),
    cue.Irreps(cueq_config.group, irreps_in2),
    cue.Irreps(cueq_config.group, irreps_out),
    layout=cueq_config.layout,
    shared_weights=shared_weights,
    internal_weights=internal_weights,
    dtype=torch.get_default_dtype(),
    math_dtype=torch.get_default_dtype(),
    use_fasteq=True,
)

mji = conv_tp(
    node_feats[edge_index[0]],
    edge_attrs,
    tp_weights,
)

message = scatter_sum(
    src=mji,
    index=edge_index[1],
    dim=0,
    dim_size=node_feats.shape[0],
)
```

### Message-passing tensor product

The MPTP path fuses the tensor product with message aggregation. The following
helper adapts the patched cuEquivariance operator to the MACE calling
convention.

```python
import types
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet


def with_cueq_conv_fusion(conv_tp: torch.nn.Module) -> torch.nn.Module:
    """Adapt a cuEquivariance convolution tensor product for fused execution."""
    conv_tp.original_forward = conv_tp.forward

    num_segments = conv_tp.m.buffer_num_segments[0]
    num_operands = conv_tp.m.operand_extent
    conv_tp.weight_numel = num_segments * num_operands

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]

        return self.original_forward(
            [tp_weights, node_feats, edge_attrs],
            {1: sender},
            {0: node_feats},
            {0: receiver},
        )[0]

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp


mptp = cuet.ChannelWiseTensorProduct(
    cue.Irreps(cueq_config.group, irreps_in1),
    cue.Irreps(cueq_config.group, irreps_in2),
    cue.Irreps(cueq_config.group, irreps_out),
    layout=cueq_config.layout,
    shared_weights=shared_weights,
    internal_weights=internal_weights,
    dtype=torch.get_default_dtype(),
    math_dtype=torch.get_default_dtype(),
    use_fasteq=True,
)

mptp = with_cueq_conv_fusion(mptp.ff)
message = mptp(node_feats, edge_attrs, tp_weights, edge_index)
```


### Symmetric contraction

```python
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet

symmetric_contractions = cuet.SymmetricContraction(
    cue.Irreps(cueq_config.group, irreps_in),
    cue.Irreps(cueq_config.group, irreps_out),
    layout_in=cue.ir_mul,
    layout_out=cueq_config.layout,
    contraction_degree=correlation,
    num_elements=num_elements,
    original_mace=(not use_reduced_cg),
    dtype=torch.get_default_dtype(),
    math_dtype=torch.get_default_dtype(),
    use_fasteq=True,
)

node_feats = symmetric_contractions(
    node_feats.flatten(1),
    index_attrs,
)
```


## Experimental methodology

### Evaluation targets

The paper evaluates FastEq at two levels:

1. **Operator-level evaluation**
   - CWTP
   - MPTP
   - FCTP
   - STC
   - Equivariant Linear

2. **End-to-end evaluation**
   - MACE-OFF
   - SevenNet
   - NequIP
   - Allegro

The selected workloads cover tensor products with different path counts,
coupling structures, model sizes, and aggregation patterns.

### Benchmarks

The benchmark scripts are located in the `test` directory.
The scripts report end-to-end inference latency for FastTP and the supported baseline implementations.

```bash
cd test
```

Run the MACE-OFF benchmark:

```bash
python3 mace_bench_new.py
```

Run the SevenNet benchmark:

```bash
python3 sevennet_bench_v2.py
```

Run the Nequip and Allegro benchmark:

```bash
cd nequip_allegro_test && bash run.sh
```
