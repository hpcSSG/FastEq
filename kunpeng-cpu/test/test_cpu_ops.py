"""Numerical regressions against independent e3nn/PyTorch computations.

Run on Linux with torch/e3nn/pytest installed. Missing dependencies or native
build failures are errors, not silently skipped successful validation.
"""
import pytest
import torch
from e3nn import o3

from fasteq.ops import (
    fast_cwtp, fast_fctp, fast_fctp_packed, fast_equi_linear,
    fast_equi_linear_packed, fast_stc, fast_stc_scalar_fused,
    fast_mptp, fast_scatter_sum, fast_spherical_harmonics,
    fast_radial_mlp, fast_radial_embedding, fast_edge_geometry,
)
from fasteq.ops.metadata import (
    make_tpu_metadata, make_fctp_metadata, make_packed_fctp_metadata,
    make_linear_metadata, make_packed_linear_metadata,
)


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    torch.manual_seed(123)
    torch.set_num_threads(1)
    return request.param


def compare(reference, native, inputs):
    a = [x.detach().clone().requires_grad_(True) for x in inputs]
    b = [x.detach().clone().requires_grad_(True) for x in inputs]
    expected, actual = reference(*a), native(*b)
    expected = expected if isinstance(expected, tuple) else (expected,)
    actual = actual if isinstance(actual, tuple) else (actual,)
    tol = dict(atol=1e-4, rtol=1e-4) if a[0].dtype == torch.float32 else dict(atol=1e-9, rtol=1e-9)
    assert len(expected) == len(actual)
    gradients = [torch.randn_like(x) for x in expected]
    for x, y in zip(expected, actual):
        torch.testing.assert_close(x, y, **tol)
    da = torch.autograd.grad(expected, a, gradients)
    db = torch.autograd.grad(actual, b, gradients)
    for x, y in zip(da, db):
        torch.testing.assert_close(x, y, **tol)


@pytest.mark.parametrize('normalization', ['integral', 'component'])
def test_spherical_harmonics(dtype, normalization):
    ref = o3.SphericalHarmonics([0, 1, 2, 3], normalize=True, normalization=normalization)
    compare(ref, lambda x: fast_spherical_harmonics(x, normalization), [torch.randn(5, 3, dtype=dtype)])


@pytest.mark.parametrize('packed', [False, True])
def test_linear(dtype, packed):
    ref = o3.Linear('2x0e + 2x1o', '3x0e + 3x1o').to(dtype=dtype)
    if packed:
        meta = make_packed_linear_metadata(ref)
        native = lambda x: fast_equi_linear_packed(x, meta)
    else:
        meta = make_linear_metadata(ref)
        native = lambda x: fast_equi_linear(x, ref.weight.detach(), meta)
    compare(ref, native, [torch.randn(5, 8, dtype=dtype)])


@pytest.mark.parametrize('packed', [False, True])
def test_fctp(dtype, packed):
    ref = o3.FullyConnectedTensorProduct('2x0e + 2x1o', '3x0e', '2x0e + 2x1o').to(dtype=dtype)
    element = torch.tensor([0, 1, 2, 0, 1])
    attrs = torch.nn.functional.one_hot(element, 3).to(dtype)
    weight = ref.weight.detach()
    if packed:
        meta = make_packed_fctp_metadata(ref, weight)
        native = lambda x: fast_fctp_packed(x, element, meta)
    else:
        meta = make_fctp_metadata(ref, element)
        native = lambda x: fast_fctp(x, attrs, weight, meta)
    compare(lambda x: ref(x, attrs), native, [torch.randn(5, 8, dtype=dtype)])


def tpu(dtype):
    return o3.TensorProduct(
        '2x0e + 2x1o', '1x0e + 1x1o', '2x0e + 2x1o',
        [(0, 0, 0, 'uvu', True), (1, 0, 1, 'uvu', True), (0, 1, 1, 'uvu', True)],
        internal_weights=False, shared_weights=False,
    ).to(dtype=dtype)


def test_uvu_all_input_gradients(dtype):
    ref = tpu(dtype)
    meta = make_tpu_metadata(ref)
    compare(ref, lambda a, b, w: fast_cwtp(a, b, w, meta), [
        torch.randn(5, 8, dtype=dtype), torch.randn(5, 4, dtype=dtype),
        torch.randn(5, ref.weight_numel, dtype=dtype),
    ])


def test_composed_message_passing(dtype):
    ref = tpu(dtype)
    meta = make_tpu_metadata(ref)
    sender, receiver = torch.tensor([0, 0, 1, 2, 3]), torch.tensor([1, 1, 2, 3, 0])
    def reference(x, y, w):
        return x.new_zeros(4, 8).index_add(0, receiver, ref(x[sender], y, w))
    compare(reference, lambda x, y, w: fast_mptp(x, y, w, sender, receiver, meta), [
        torch.randn(4, 8, dtype=dtype), torch.randn(5, 4, dtype=dtype),
        torch.randn(5, ref.weight_numel, dtype=dtype),
    ])


def test_stc_step(dtype):
    U, weight = torch.randn(2, 2, 2, 3, 4, dtype=dtype), torch.randn(2, 4, 2, dtype=dtype)
    element = torch.tensor([0, 1, 0])
    compare(lambda x: torch.einsum('wxyik,bkc,bci->bcwxy', U, weight[element], x),
            lambda x: fast_stc(U, weight, x, element), [torch.randn(3, 2, 3, dtype=dtype)])


def test_fused_stc(dtype):
    meta = {'effective': torch.randn(2, 2, 3, 4, dtype=dtype),
            'C0': torch.randn(2, 2, 4, dtype=dtype), 'C1': torch.randn(2, 2, 2, dtype=dtype)}
    element = torch.tensor([0, 1, 0])
    def reference(x):
        ct = (torch.einsum('bcim,bci->bcm', meta['effective'][element], x) + meta['C0'][element]).reshape(3, 2, 2, 2)
        partial = torch.einsum('bcwx,bcw->bcx', ct, x[..., :2]) + meta['C1'][element]
        return (partial * x[..., :2]).sum(-1)
    compare(reference, lambda x: fast_stc_scalar_fused(x, element, meta), [torch.randn(3, 2, 3, dtype=dtype)])


@pytest.mark.parametrize('vector', [False, True])
def test_scatter(dtype, vector):
    index = torch.tensor([2, 0, 2, 1, 0])
    shape = (5,) if vector else (5, 3)
    compare(lambda x: x.new_zeros((4,) + x.shape[1:]).index_add(0, index, x),
            lambda x: fast_scatter_sum(x, index, 4), [torch.randn(shape, dtype=dtype)])


@pytest.mark.parametrize('normalize', [False, True])
def test_geometry(dtype, normalize):
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    def reference(p, shifts):
        vectors = p[edges[1]] - p[edges[0]] + shifts
        lengths = vectors.norm(dim=-1, keepdim=True)
        return (vectors / lengths.clamp_min(1e-9) if normalize else vectors), lengths
    compare(reference, lambda p, s: fast_edge_geometry(p, edges, s, normalize),
            [torch.randn(3, 3, dtype=dtype), torch.randn(3, 3, dtype=dtype)])


def test_radial_embedding(dtype):
    weights = torch.arange(1, 5, dtype=dtype) * torch.pi / 5
    def reference(r):
        t, p = r / 5, 5
        envelope = 1 - (p+1)*(p+2)/2*t**p + p*(p+2)*t**(p+1) - p*(p+1)/2*t**(p+2)
        return (2/5)**0.5 * torch.sin(r * weights) / r * envelope
    compare(reference, lambda r: fast_radial_embedding(r, weights, 5.0, 5),
            [torch.tensor([[0.4], [1.1], [3.5]], dtype=dtype)])


def test_radial_mlp(dtype):
    w1, w2 = torch.randn(3, 4, dtype=dtype), torch.randn(4, 2, dtype=dtype)
    meta = {'weights': (w1, w2), 'native_scales': torch.tensor([1.2, float('nan')], dtype=torch.float64),
            'native_active': torch.tensor([1, 0], dtype=torch.int64)}
    compare(lambda x: (torch.nn.functional.silu(x @ w1) * 1.2) @ w2,
            lambda x: fast_radial_mlp(x, meta), [torch.randn(5, 3, dtype=dtype)])
