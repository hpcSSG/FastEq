# FastEq

# Overview
FastEq is a high-performance equivariant operator library that implements cuEquivariance’s core operators while fully supporting its API and interfaces.

The operators implemented in FastEq include:
- **ChannelWiseTensorProduct(cwtp)**
- **MessagePassingTensorProduct(mptp, cwtp+scatter_sum)**
- **FullyConnectedTensorProduct(fctp)**  
- **SymmetricContraction(stc)** 
- **EquivariantLinear(equi-linear)** 

# Prerequisites
- Python 3.10
- CUDA Toolkit 12 or higher  
- PyTorch 2.4.1 or higher  
- [cuEquivariance](https://github.com/NVIDIA/cuEquivariance)  

# Installation
1. Clone this repository:  
   ```bash
   https://github.com/malixian/FastEq.git
   cd FastEq
   ```
2. Ensure CUDA toolkit 12 or higher is installed on your system.
3. Install the Python dependencies and the package itself
   ```bash
   pip install -r requirements.txt
   pip install . --no-build-isolation 
   ```
   We recommend using Virtualenv or Conda to manage the dependencies.
4. Install patched cuEquivariance_torch
   ```bash
   cd 3rdpatry/cuequivariance_torch
   pip install . --no-build-isolation 
   ```
5. Install patched MACE
   ```bash
   cd 3rdpatry/mace
   pip install . --no-build-isolation 
   ```

   

# How to use FastEq
## cwtp
```python
import cuequivariance_torch as cuet

# Initalization
conv_tp = cuet.ChannelWiseTensorProduct(
    cue.Irreps(cueq_config.group, irreps_in1),
    cue.Irreps(cueq_config.group, irreps_in2),
    cue.Irreps(cueq_config.group, irreps_out),
    layout=cueq_config.layout,
    shared_weights=shared_weights,
    internal_weights=internal_weights,
    dtype=torch.get_default_dtype(),
    math_dtype=torch.get_default_dtype(),
    use_fasteq=True, # Set use_fasteq=True
)

# Execution
mji = conv_tp(
    node_feats[edge_index[0]], edge_attrs, tp_weights
)
message = scatter_sum(
    src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
)
```

## mptp
```python
import cuequivariance_torch as cuet

# use a helper to rewrite forward, referencing `with_cueq_conv_fusion` in MACE 

def with_cueq_conv_fusion(conv_tp: torch.nn.Module) -> torch.nn.Module:
    """Wraps a cuet.ConvTensorProduct to use conv fusion"""
    conv_tp.original_forward = conv_tp.forward
    num_segment = conv_tp.m.buffer_num_segments[0]
    num_operands = conv_tp.m.operand_extent
    conv_tp.weight_numel = num_segment * num_operands
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

# Initalization
mptp = cuet.ChannelWiseTensorProduct(
    cue.Irreps(cueq_config.group, irreps_in1),
    cue.Irreps(cueq_config.group, irreps_in2),
    cue.Irreps(cueq_config.group, irreps_out),
    layout=cueq_config.layout,
    shared_weights=shared_weights,
    internal_weights=internal_weights,
    dtype=torch.get_default_dtype(),
    math_dtype=torch.get_default_dtype(),
    use_fasteq=True, # Set use_fasteq=True
)
return with_cueq_conv_fusion(mptp.ff)

# Execution
message = mptp(node_feats, edge_attrs, tp_weights, edge_index)
```

## fctp
```python
import cuequivariance_torch as cuet

# Initalization
skip_tp = cuet.FullyConnectedTensorProduct(
    cue.Irreps(cueq_config.group, irreps_in1),
    cue.Irreps(cueq_config.group, irreps_in2),
    cue.Irreps(cueq_config.group, irreps_out),
    layout=cueq_config.layout,
    shared_weights=shared_weights,
    internal_weights=internal_weights,
    method="naive",
    use_fasteq=True, # Set use_fasteq=True
)

# Execution
message = skip_tp(message, node_attrs)
```
## stc
```python
import cuequivariance_torch as cuet

# Initalization
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
    use_fasteq=True, # Set use_fasteq=True
)

# Execution
node_feats = symmetric_contractions(
    node_feats.flatten(1),
    index_attrs,
)
```


## equi-linear
```python
import cuequivariance_torch as cuet

# Initalization
linear = cuet.Linear(
    cue.Irreps(cueq_config.group, irreps_in),
    cue.Irreps(cueq_config.group, irreps_out),
    layout=cueq_config.layout,
    shared_weights=shared_weights,
    method="naive",
    use_fasteq=True, # Set use_fasteq=True
)

# Execution
message = linear(message)
```