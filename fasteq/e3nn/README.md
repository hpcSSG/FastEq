# FastEq e3nn Triton GPU tests

`tests` contains only `math`, `o3`, and `nn`. Each test filename matches the
corresponding file under e3nn's `tests/math`, `tests/o3`, or `tests/nn`. The
tensor product test files, `reduce_tensor_test.py`, and `nn/models` are outside
this suite. There are no static tests.

The suite imports the reference implementation from `e3nn` and, by default,
the implementation under test from `fasteq.e3nn.triton`. It covers supported
CUDA forward paths, reference fallbacks, parity, random state, first
derivatives where implemented, and rotation/grid invariants. The upstream
repository has no separate `math/linalg_test.py` or `o3/so3grid_test.py`;
their relevant checks live in `perm_test.py` and `s2_test.py` respectively.

Run from the FastEq repository root with PyTorch, Triton, e3nn, pytest and a
CUDA GPU available:

```bash
pytest tests/math tests/o3 tests/nn -q
```

Set `E3NN_TRITON_PACKAGE` to use another package prefix, such as
`e3nn.triton` in the development checkout.
