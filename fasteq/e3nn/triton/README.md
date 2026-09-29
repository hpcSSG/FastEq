# e3nn math and o3 Triton counterparts

Source baseline: e3nn `main` at `2aa7f58440a06b15352a2cbce01fa4c26f824969`.

This optional package has a matching module for every `.py` file directly under
`e3nn/math` and `e3nn/o3`, excluding `o3/_irreps.py`, `o3/irrep/`,
`o3/_tensor_product/`, and `o3/experimental/`. Every top-level function and
class in the included files is accessible from its matching module. The
package root exports the same selected public names as e3nn's `math` and `o3`
packages. The upstream e3nn implementation remains unchanged.

Import into FastEq by placing the `e3nn/triton` directory under
`fasteq/e3nn/triton`:

```python
import fasteq.e3nn.triton.o3._rotation as tr
from fasteq.e3nn.triton.o3._s2grid import ToS2Grid, rfft
```

## Implementation choices

- Triton GPU paths cover the existing 33 selected entries, S2 Fourier
  transforms and grid forwards, rotation matrix conversions, `rand_angles`
  postprocessing, the power/mean part of `moment`, inference `Norm`, and
  single-instruction, no-bias, no-grad `Linear`. A class method can call
  several kernels where the operation has multiple computational stages.
- `rand_angles` draws samples with the original two `torch.rand` calls in the
  original order. Triton only transforms those samples. Derived random
  rotations reuse that function. Other host RNG operations retain e3nn.
- Python permutation/group operations, arbitrary activation functions,
  symbolic generators, orthonormalization, general Linear code generation,
  reduced tensor products, Wigner coefficient construction, and Cartesian
  spherical harmonics retain the corresponding e3nn functions or classes.
  General `Linear` configurations and differentiable `Norm` calls also use
  the original e3nn implementation.
- S2 and angular module initialization may use SymPy and PyTorch to prepare
  coefficients and buffers. S2 `rfft` and `irfft` use direct DFT reduction;
  their cost grows with both grid resolution and retained frequencies.

Triton paths accept CUDA `float32` or `float64` unless a function documents
otherwise. Triton functions without a custom `torch.autograd.Function` do not
provide input gradients. This package is not yet a complete training backend.
The retained e3nn paths preserve e3nn's original autograd behavior.
Triton `matrix_to_angles` and `matrix_to_axis_angle` expect valid rotation
matrices; unlike the reference wrappers, they do not perform a separate
global determinant assertion before the kernel launch.

## Verification

```bash
python tests/triton/test_math_o3_coverage_static.py
python tests/triton/test_inventory_static.py
E3NN_TRITON_PACKAGE=fasteq.e3nn.triton pytest tests/triton -q
```

The local development workspace has no PyTorch, Triton, or GPU. Static
coverage, syntax, and source checks can run there; CUDA numerical comparisons
must run in a GPU environment.
