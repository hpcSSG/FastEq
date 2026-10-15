# FastEq — Kunpeng CPU

# Overview

FastEq Kunpeng CPU provides native CPU implementations of equivariant operators for Kunpeng processors. It uses C++, OpenMP, and optional OpenBLAS, with PyTorch interfaces for model inference and input-gradient computation.

The operators implemented in this CPU version include:

- **ChannelWiseTensorProduct(cwtp, e3nn `uvu` mode)**
- **MessagePassingTensorProduct(mptp, cwtp + scatter_sum)**
- **FullyConnectedTensorProduct(fctp)**
- **SymmetricContraction(stc)**
- **EquivariantLinear(equi-linear)**

Additional operators include Cartesian spherical harmonics (`l=0..3`), scatter sum, radial MLP, Bessel radial embedding, and edge geometry.

This version uses e3nn/MACE metadata and does not yet support cuEquivariance's `use_fasteq=True` interface. MPTP currently combines separate tensor-product and scatter kernels. Most operators provide input gradients for fixed-weight inference; full parameter training and higher-order gradients are not supported. See [CPU API](fasteq-kunpeng/docs/API.md) for supported configurations.

Repository layout:

```text
kunpeng-cpu/
  3rdparty/           # Optional MACE integration and startup hook
  fasteq-kunpeng/     # CPU operators, native source, and documentation
  test/              # Regression tests, benchmarks, results, and scripts
  README.md
  pyproject.toml
  setup.py
  requirements.txt
```

The source directory is named `fasteq-kunpeng`; the installed Python import remains `fasteq`. Install the package before running the examples or tests.

# Prerequisites

- Kylin Linux on Kunpeng 920 series (aarch64; CPU series and OS confirmed by the user)
- Python 3.9 or higher (historical server: CPython 3.9.25)
- PyTorch 2.2.2 or higher, with CPU support
- e3nn 0.4.4 or higher
- A C++ compiler with OpenMP support
- Ninja
- OpenBLAS development headers and libraries (optional)
- MACE (optional, for model integration)

The historical server used PyTorch 2.2.2, e3nn 0.4.4 and MACE 0.3.13. The minimum requirements now admit that environment; the reorganized package still needs target-server validation. See [server configuration](fasteq-kunpeng/docs/ENVIRONMENT.md) for the recorded versions, compiler paths and thread settings. Use a separate Virtualenv or Conda environment because this version shares the `fasteq` package name with the GPU version.

# Installation

On the measured node cn19793 (2026-10-15, job 1886300), the OS reports two 304-core sockets, one thread per core, and 32 NUMA nodes: 16 with CPUs and 16 memory-only nodes. The configured HPCKit compiler path reports GCC/G++ 12.3.1. The sampling process was restricted to 16 CPU IDs; verify affinity inside the actual worker before benchmarking. See [measured environment](fasteq-kunpeng/docs/ENVIRONMENT.md) for the raw report and scope.

1. From the FastEq repository root, enter the CPU subproject directory:

   ```bash
   cd kunpeng-cpu
   ```

   Run the installation and test commands below from this directory. The
   repository-root build configuration belongs to the upstream GPU package.

2. Ensure a CPU build of PyTorch and an OpenMP-capable C++ compiler are installed on your system.

3. Install the Python dependencies and the package:

   ```bash
   python -m pip install -r requirements.txt -c test/constraints-kunpeng.txt
   python -m pip install -e . --no-build-isolation -c test/constraints-kunpeng.txt
   ```

   The constraints reproduce known historical versions, not a complete environment lock. Preinstall setuptools>=68 and wheel for the non-isolated build. On the existing server, activate the venv first and check package import paths; the old environment also depended on a base module installation.

4. Configure the CPU environment:

   ```bash
   source test/scripts/cpu_env.sh
   export FASTEQ_BACKEND=cpu
   export OMP_NUM_THREADS=37  # Adjust to the cores allocated to each process.
   export OPENBLAS_NUM_THREADS=1
   ```

   To enable OpenBLAS, set both paths to your installation:

   ```bash
   export NATIVE_CPU_BLAS_INCLUDE=/actual/openblas/include
   export NATIVE_CPU_BLAS_LIB=/actual/openblas/lib
   export NATIVE_CPU_BLAS_NAME=openblas
   # Optional: export CXX=/actual/compiler/bin/g++
   # Optional: export TORCH_EXTENSIONS_DIR=/actual/writable/cache
   ```

   Without these paths, the kernels use their internal computation path.

   On the documented PAC cluster, `source test/scripts/kunpeng_env.sh` provides the recorded compiler/OpenBLAS paths when present and a job/rank-local JIT cache. It does not change CPU affinity. This standalone test profile defaults to OpenBLAS=1; the historical batchASE end-to-end profile used OpenBLAS=37 and is documented separately.

5. Compile and load the CPU extension:

   ```bash
   python -c 'from fasteq.cpu import load_extension; print(load_extension().__file__)'
   ```

   The extension is compiled on first use through PyTorch JIT and cached for subsequent calls. Prebuild it before starting multiple workers. Compilation uses `-march=native`; rebuild when changing CPU architecture. Set `FASTEQ_CPU_FAST_MATH=0` to disable the original fast-math options.

6. Install MACE if model integration is needed:

   ```bash
   python -m pip install -e '.[mace]' --no-build-isolation -c test/constraints-kunpeng.txt
   ```

   Use the MACE version matching your model. For an existing CPU `MACECalculator` instance:

   ```python
   from fasteq.integrations.mace_runtime import install_native_mace

   install_native_mace(calculator)
   ```

# How to use FastEq

The examples below use CPU tensors. Float32 and float64 are supported by the native kernels; input dtypes must match, and indices must be int64. Packed metadata stores a fixed-weight snapshot and must be rebuilt after changing weights, dtype, or model structure.

## cwtp

The CPU implementation supports e3nn tensor products with `uvu` connection mode.

```python
import torch
from e3nn import o3
from fasteq.ops import fast_cwtp, fast_scatter_sum
from fasteq.ops.metadata import make_tpu_metadata

# Initialization
conv_tp = o3.TensorProduct(
    "8x0e + 8x1o",
    "1x0e + 1x1o",
    "8x0e + 8x1o",
    instructions=[
        (0, 0, 0, "uvu", True),
        (1, 0, 1, "uvu", True),
        (0, 1, 1, "uvu", True),
    ],
    shared_weights=False,
    internal_weights=False,
).double()
metadata = make_tpu_metadata(conv_tp)

node_feats = torch.randn(4, conv_tp.irreps_in1.dim, dtype=torch.float64)
edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
edge_attrs = torch.randn(5, conv_tp.irreps_in2.dim, dtype=torch.float64)
tp_weights = torch.randn(5, conv_tp.weight_numel, dtype=torch.float64)

# Execution
mji = fast_cwtp(node_feats[edge_index[0]], edge_attrs, tp_weights, metadata)
message = fast_scatter_sum(mji, edge_index[1], node_feats.shape[0])
```

## mptp

Using the same initialization as the CWTP example, combine the tensor product and message aggregation through one Python interface. The CPU kernels are currently executed separately.

```python
from fasteq.ops import fast_mptp

# Execution
message = fast_mptp(
    node_features=node_feats,
    edge_attrs=edge_attrs,
    weight=tp_weights,
    sender=edge_index[0],
    receiver=edge_index[1],
    metadata=metadata,
)
```

## fctp

The packed path supports shared weights and one-hot scalar node attributes. It computes gradients with respect to the input features only.

```python
import torch
from e3nn import o3
from fasteq.ops import fast_fctp_packed
from fasteq.ops.metadata import make_packed_fctp_metadata

# Initialization
skip_tp = o3.FullyConnectedTensorProduct(
    "8x0e + 8x1o",
    "4x0e",
    "8x0e + 8x1o",
    shared_weights=True,
    internal_weights=True,
).double()
metadata = make_packed_fctp_metadata(skip_tp, skip_tp.weight.detach())

node_feats = torch.randn(16, skip_tp.irreps_in1.dim, dtype=torch.float64)
index_attrs = torch.randint(0, 4, (16,))  # Element indices for one-hot attributes.

# Execution
message = fast_fctp_packed(node_feats, index_attrs, metadata)
```

## stc

The fused path supports correlation=3 with scalar output. Start from an existing compatible MACE `Contraction` module and its input features:

```python
from fasteq.ops import fast_stc_scalar_fused
from fasteq.ops.metadata import make_symmetric_contraction_metadata

# Initialization
# contraction: a MACE Contraction module with correlation=3 and scalar output.
# node_feats: CPU tensor [num_nodes, channels, irrep_dim].
# index_attrs: CPU int64 tensor [num_nodes] containing element indices.
metadata = make_symmetric_contraction_metadata(contraction)

# Execution
message = fast_stc_scalar_fused(node_feats, index_attrs, metadata)
```

The lower-level `fast_stc(U, weight, x, element)` interface implements one contraction step. General MACE contractions still combine this step with the remaining operations in the model integration.

## equi-linear

The packed path supports shared weights without bias instructions and computes input gradients.

```python
import torch
from e3nn import o3
from fasteq.ops import fast_equi_linear_packed
from fasteq.ops.metadata import make_packed_linear_metadata

# Initialization
linear = o3.Linear(
    "8x0e + 8x1o",
    "8x0e + 8x1o",
    shared_weights=True,
    internal_weights=True,
).double()
metadata = make_packed_linear_metadata(linear)
node_feats = torch.randn(
    16, linear.irreps_in.dim, dtype=torch.float64, requires_grad=True
)

# Execution
message = fast_equi_linear_packed(node_feats, metadata)
message.square().sum().backward()
```

# Testing

```bash
python -m pip install -e '.[test]' --no-build-isolation -c test/constraints-kunpeng.txt
python test/scripts/collect_environment.py > environment.json
python -m pytest -q
python test/benchmark_native_vs_e3nn.py --threads 1 37 --batches 19 172 --fused-sc
```

The reorganized package has passed local syntax and packaging checks. Native compilation, numerical regressions, and performance measurements still need to be rerun on Kunpeng. Files in `test/results` are historical results from the original implementation.

See [validation status](fasteq-kunpeng/docs/VALIDATION.md), [known issues from review](fasteq-kunpeng/docs/REVIEW.md), [remaining work](fasteq-kunpeng/docs/MISSING_ITEMS.md), and [source and license notes](NOTICE.md) for details.
