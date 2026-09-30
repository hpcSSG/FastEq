"""Adapted from e3nn/tests/o3/reduce_tensor_test.py, with Triton reduction.

Default: CUDA construction, native TensorProduct forward, FP32 and FP64.
Package override: E3NN_TRITON_PACKAGE=fasteq.e3nn.triton
CPU smoke run: E3NN_REDUCE_TEST_DEVICE=cpu
Compile backend override: E3NN_REDUCE_COMPILE_BACKEND=eager (default inductor)
"""

import copy
import functools
import importlib
import os
import tempfile
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
import e3nn
from e3nn import o3 as reference_o3
from e3nn.util.test import assert_auto_jitable
from e3nn.util.test import assert_equivariant as reference_assert_equivariant

PREFIX = os.environ.get("E3NN_TRITON_PACKAGE", "fasteq.e3nn.triton")
DEVICE = os.environ.get("E3NN_REDUCE_TEST_DEVICE", "cuda")
BACKEND = os.environ.get("E3NN_REDUCE_COMPILE_BACKEND", "inductor")
reduce_module = importlib.import_module(PREFIX + ".o3._reduce")
RTP = reduce_module.ReducedTensorProducts
pytestmark = pytest.mark.skipif(
    torch.device(DEVICE).type == "cuda" and not torch.cuda.is_available(), reason="CUDA required"
)
o3 = SimpleNamespace(
    ReducedTensorProducts=functools.partial(RTP, reduction_device=DEVICE),
    Irreps=reference_o3.Irreps,
)


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    old_dtype = torch.get_default_dtype()
    old_device = torch.get_default_device()
    # CPU Wigner coefficients/rotations in reference utilities; GPU operations
    # below always specify their device explicitly.
    torch.set_default_device("cpu")
    torch.set_default_dtype(request.param)
    torch.manual_seed(0)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_default_device(old_device)


@pytest.fixture(autouse=True)
def use_dtype(dtype):
    pass


@pytest.fixture
def float_tolerance(dtype):
    return 1e-3 if dtype == torch.float32 else 1e-9


def _randn(*shape):
    return torch.randn(*shape, device=DEVICE)


def assert_equivariant(module, **kwargs):
    # Preserve the upstream randomized checks on a CPU copy. Native Wigner
    # utilities can otherwise mix CPU generators with CUDA angles.
    reference_assert_equivariant(copy.deepcopy(module).cpu(), **kwargs)
    if torch.device(DEVICE).type == "cuda":
        dtype = module.change_of_basis.dtype
        rotation = reference_o3.rand_matrix(dtype=dtype, device="cpu")
        xs = [_randn(2, ir.dim) for ir in module.irreps_in]
        for matrix in (rotation, -rotation):
            matrices = [ir.D_from_matrix(matrix).to(DEVICE) for ir in module.irreps_in]
            dout = module.irreps_out.D_from_matrix(matrix).to(DEVICE)
            got = module(*(x @ d.T for x, d in zip(xs, matrices)))
            want = module(*xs) @ dout.T
            tol = 1e-3 if dtype == torch.float32 else 1e-9
            torch.testing.assert_close(got, want, rtol=tol, atol=tol)


def assert_torch_compile(compile_mode, factory, *args):
    # Same construction/jit-mode handling as upstream, with an explicit backend
    # override for machines where inductor cannot be run.
    previous = e3nn.get_optimization_defaults()["jit_mode"]
    try:
        e3nn.set_optimization_defaults(jit_mode=compile_mode)
        module = factory()
        torch._dynamo.reset()
        compiled = torch.compile(module, backend=BACKEND, fullgraph=True)
        torch.testing.assert_close(compiled(*args), module(*args))
    finally:
        e3nn.set_optimization_defaults(jit_mode=previous)


def test_save_load() -> None:
    tp1 = o3.ReducedTensorProducts("ij=-ji", i="5x0e + 1e")
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(tp1.state_dict(), tmp.name)
        tp2 = o3.ReducedTensorProducts("ij=-ji", i="5x0e + 1e")
        tp2.load_state_dict(torch.load(tmp.name, map_location=DEVICE, weights_only=True))

    xs = (_randn(2, 5 + 3), _randn(2, 5 + 3))
    assert torch.allclose(tp1(*xs), tp2(*xs))

    assert torch.allclose(tp1.change_of_basis, tp2.change_of_basis)


def test_antisymmetric_matrix(float_tolerance) -> None:

    tp = o3.ReducedTensorProducts("ij=-ji", i="5x0e + 1e")

    Q = tp.change_of_basis
    x = _randn(2, 5 + 3)

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_torch_compile("inductor", functools.partial(o3.ReducedTensorProducts, "ij=-ji", i="5x0e + 1e"), *x)
    assert_auto_jitable(tp)

    assert (tp(*x) - torch.einsum("xij,i,j", Q, *x)).abs().max() < float_tolerance

    assert (Q + torch.einsum("xij->xji", Q)).abs().max() < float_tolerance


def test_reduce_tensor_Levi_Civita_symbol(float_tolerance) -> None:
    tp = o3.ReducedTensorProducts("ijk=-ikj=-jik", i="1e")
    assert tp.irreps_out == o3.Irreps("0e")

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = _randn(3, 3)
    assert (tp(*x) - torch.einsum("xijk,i,j,k", Q, *x)).abs().max() < float_tolerance

    assert (Q + torch.einsum("xijk->xikj", Q)).abs().max() < float_tolerance
    assert (Q + torch.einsum("xijk->xjik", Q)).abs().max() < float_tolerance


def test_reduce_tensor_antisymmetric_L2(float_tolerance) -> None:
    tp = o3.ReducedTensorProducts("ijk=-ikj=-jik", i="2e")

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = _randn(3, 5)
    assert (tp(*x) - torch.einsum("xijk,i,j,k", Q, *x)).abs().max() < float_tolerance

    assert (Q + torch.einsum("xijk->xikj", Q)).abs().max() < float_tolerance
    assert (Q + torch.einsum("xijk->xjik", Q)).abs().max() < float_tolerance


def test_reduce_tensor_elasticity_tensor(float_tolerance) -> None:
    tp = o3.ReducedTensorProducts("ijkl=jikl=klij", i="1e")
    assert tp.irreps_out.dim == 21

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = _randn(4, 3)
    assert (tp(*x) - torch.einsum("xijkl,i,j,k,l", Q, *x)).abs().max() < float_tolerance

    assert (Q - torch.einsum("xijkl->xjikl", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xijlk", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xklij", Q)).abs().max() < float_tolerance


def test_reduce_tensor_elasticity_tensor_parity(float_tolerance) -> None:
    tp = o3.ReducedTensorProducts("ijkl=jikl=klij", i="1o")
    assert tp.irreps_out.dim == 21
    assert all(ir.p == 1 for _, ir in tp.irreps_out)

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = _randn(4, 3)
    assert (tp(*x) - torch.einsum("xijkl,i,j,k,l", Q, *x)).abs().max() < float_tolerance

    assert (Q - torch.einsum("xijkl->xjikl", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xijlk", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xklij", Q)).abs().max() < float_tolerance


CASES = [
    ("ij=ji", "1o"),
    ("ij=-ji", "5x0e + 1e"),
    ("ijk=-ikj=-jik", "1e"),
    ("ijk=-ikj=-jik", "2e"),
    ("ijkl=jikl=klij", "1e"),
    ("ijkl=jikl=klij", "1o"),
]


@pytest.mark.parametrize("formula,irreps", CASES)
def test_native_compatibility_and_gradients(dtype, formula, irreps):
    actual = RTP(formula, i=irreps, reduction_device=DEVICE)
    native = reference_o3.ReducedTensorProducts(formula, i=irreps).to(DEVICE)
    assert actual.irreps_in == native.irreps_in
    assert actual.irreps_out == native.irreps_out
    assert set(actual.state_dict()) == set(native.state_dict())
    tol = 1e-3 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(actual.change_of_basis, native.change_of_basis, rtol=tol, atol=tol)
    # Compare each independently constructed basis before loading identical
    # states, then compare forward and autograd with exactly the same graph data.
    xs = [_randn(2, ir.dim).requires_grad_() for ir in actual.irreps_in]
    torch.testing.assert_close(actual(*xs), native(*xs), rtol=tol, atol=tol)
    native.load_state_dict(actual.state_dict())
    got, want = actual(*xs), native(*xs)
    torch.testing.assert_close(got, want, rtol=tol, atol=tol)
    ga = torch.autograd.grad(got.square().sum(), xs, create_graph=True)
    gb = torch.autograd.grad(want.square().sum(), xs, create_graph=True)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)
    gga = torch.autograd.grad(sum(g.square().sum() for g in ga), xs)
    ggb = torch.autograd.grad(sum(g.square().sum() for g in gb), xs)
    for a, b in zip(gga, ggb):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("formula,irreps", CASES)
def test_broadcast_and_basis_contraction(dtype, formula, irreps):
    module = RTP(formula, i=irreps, reduction_device=DEVICE)
    xs = [_randn(2, 1, ir.dim) if j == 0 else _randn(1, 3, ir.dim)
          for j, ir in enumerate(module.irreps_in)]
    indices = "ijkl"[:len(xs)]
    equation = "o" + indices + "," + ",".join("..." + i for i in indices) + "->...o"
    expected = torch.einsum(equation, module.change_of_basis, *xs)
    tol = 1e-3 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(module(*xs), expected, rtol=tol, atol=tol)
    empty = [x.expand(2, 3, x.shape[-1])[:0] for x in xs]
    assert module(*empty).shape == (0, 3, module.irreps_out.dim)


def test_local_math_bindings_and_constructor_calls(dtype, monkeypatch):
    ml = importlib.import_module(PREFIX + ".math._linalg")
    mr = importlib.import_module(PREFIX + ".math._reduce")
    assert reduce_module.orthonormalize is ml.orthonormalize
    assert reduce_module.reduce_permutation is mr.reduce_permutation
    calls = {"orthonormalize": 0, "reduce_permutation": 0}
    native_ortho, native_reduce = reduce_module.orthonormalize, reduce_module.reduce_permutation
    if torch.device(DEVICE).type == "cuda":
        def forbidden_fallback(*args, **kwargs):
            raise AssertionError("supported CUDA reduction must execute Triton math")

        monkeypatch.setattr(ml, "_reference_orthonormalize", forbidden_fallback)
        monkeypatch.setattr(mr, "_reference_reduce_permutation", forbidden_fallback)

    def ortho(*args, **kwargs):
        calls["orthonormalize"] += 1
        if torch.device(DEVICE).type == "cuda":
            assert args[0].is_cuda and args[0].dtype == torch.float64
        return native_ortho(*args, **kwargs)

    def reduce(*args, **kwargs):
        calls["reduce_permutation"] += 1
        assert torch.device(kwargs["device"]) == torch.device(DEVICE)
        return native_reduce(*args, **kwargs)

    monkeypatch.setattr(reduce_module, "orthonormalize", ortho)
    monkeypatch.setattr(reduce_module, "reduce_permutation", reduce)
    # Verify the native constructor before JIT converts its objects into
    # RecursiveScriptModule instances, which no longer satisfy isinstance(TP).
    from e3nn.o3._tensor_product._tensor_product import TensorProduct
    assert reduce_module.TensorProduct is TensorProduct
    constructed_tps = []

    def recorded_tensor_product(*args, **kwargs):
        tp = TensorProduct(*args, **kwargs)
        constructed_tps.append(tp)
        return tp

    monkeypatch.setattr(reduce_module, "TensorProduct", recorded_tensor_product)
    module = RTP("ij=-ji", i="1o", reduction_device=DEVICE)
    assert calls["reduce_permutation"] == 1 and calls["orthonormalize"] > 0
    assert module.change_of_basis.device.type == torch.device(DEVICE).type
    assert constructed_tps and all(isinstance(tp, TensorProduct) for tp in constructed_tps)
    # The reduced forward must delegate to main, retaining native TP autograd.
    xs = [_randn(2, ir.dim) for ir in module.irreps_in]
    original_main = module.main
    expected = original_main(*xs)
    main_calls = []

    class RecordedMain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wrapped = original_main

        def forward(self, *args):
            main_calls.append(1)
            return self.wrapped(*args)

    module.main = RecordedMain()
    torch.testing.assert_close(module(*xs), expected, rtol=0, atol=0)
    assert len(main_calls) == 1
