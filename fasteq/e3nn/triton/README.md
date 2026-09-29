# e3nn Triton single-kernel batch

Baseline: e3nn `main` at `2aa7f58440a06b15352a2cbce01fa4c26f824969`.

This optional `e3nn.triton` package provides the 33 forward entry points selected
from `o3` and `math`. It does not patch the upstream `e3nn.o3` or `e3nn.math`
exports. Import the matching function or class explicitly, for example:

```python
from e3nn.triton.o3._rotation import angles_to_matrix
from e3nn.triton.math._bessel import bessel
```

The one-kernel classification counts the **main GPU calculation**. Python
allocation, tensor views and first-use coefficient preparation are outside this
count. Broadcasting through `.contiguous()` may launch a copy for expanded
inputs. `SO3Grid`, `ToS2Grid`, and `FromS2Grid` inherit their original setup
and replace only the listed methods or properties. The angular harmonics
module caches polynomial coefficients on first use for each device and dtype.

GPU tensors in `float32` or `float64` are required. CUDA is the initial target;
HIP compilation and numerical parity have not been checked. Forward functions using
bare Triton launches do **not** currently implement autograd; the explicit
`_SoftUnitStep.backward` is the exception. Do not use this package as a drop-in
training replacement until the corresponding backward and double backward
paths are implemented and verified.

Run `python tests/triton/test_inventory_static.py` for the 33-function
inventory, and `pytest tests/triton/test_forward_gpu.py -q` on a CUDA or HIP GPU
with PyTorch, Triton, e3nn dependencies and pytest installed. The latter
compares against the original e3nn functions; benchmark performance separately
after numerical parity passes.
