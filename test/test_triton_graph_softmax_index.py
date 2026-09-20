"""Reuse the existing precision contract and check genuinely CSR-free calls."""
import pytest
import torch

pytest.importorskip('triton')
import test_triton_graph_softmax_precision as precision
from test_triton_graph_softmax_precision import native_class
from fasteq.triton.graph_softmax import reference_graph_softmax
from fasteq.triton.graph_softmax_index import fused_graph_softmax_index, FusedGraphSoftmaxIndex

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/HIP GPU required')


@pytest.fixture(autouse=True)
def index_route(monkeypatch):
    def adapted(src, index=None, ptr=None, num_nodes=None, **kwargs):
        # The common suite has pointer-route cases. Convert their test input
        # to raw indices here; the implementation itself never consumes CSR.
        if ptr is not None:
            num_nodes = ptr.numel() - 1
            index = torch.repeat_interleave(torch.arange(num_nodes, device=ptr.device),
                                            ptr[1:] - ptr[:-1])
        kwargs.pop('csr', None)
        kwargs.pop('block_heads', None)
        return fused_graph_softmax_index(src, index, num_nodes=num_nodes, **kwargs)
    monkeypatch.setattr(precision, 'fused_graph_softmax', adapted)


# Import the individual parametrized tests, not the CSR/layout autouse fixture.
# This preserves the native-FP32 tolerances and narrow analytic exceptions.
for _name in dir(precision):
    if _name.startswith('test_'):
        globals()[_name] = getattr(precision, _name)


def test_no_preprocessing_or_host_scalar_read(monkeypatch):
    from fasteq.triton import graph_softmax
    index = torch.arange(73, device='cuda') % 17
    x = torch.randn(73, 3, device='cuda', requires_grad=True)
    r = torch.rand(73, 1, device='cuda', requires_grad=True)
    upstream = torch.randn_like(x)
    # Warm up compilation before checking the per-call execution contract.
    torch.autograd.grad(fused_graph_softmax_index(x, index, 17, exp_rescale=r), (x, r), upstream)
    def forbidden(*args, **kwargs):
        raise AssertionError('raw-index execution must not preprocess CSR or read host scalars')
    with monkeypatch.context() as patch:
        for name in ('argsort', 'sort', 'bincount', 'cumsum'):
            patch.setattr(torch, name, forbidden)
        for name in ('item', 'cpu', '__int__', '__bool__'):
            patch.setattr(torch.Tensor, name, forbidden)
        patch.setattr(graph_softmax, 'prepare_graph_softmax', forbidden)
        y = fused_graph_softmax_index(x, index, 17, exp_rescale=r)
        grads = torch.autograd.grad(y, (x, r), upstream)
    assert len(grads) == 2


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_changed_topology_and_strided_tensors(dtype):
    torch.manual_seed(86)
    x = torch.randn(3, 97, device='cuda').t().requires_grad_()
    r = torch.rand(3, 97, device='cuda').t().requires_grad_()
    index = torch.randint(17, (194,), device='cuda', dtype=dtype)[::2]
    g = torch.randn_like(x)
    module = FusedGraphSoftmaxIndex(eps=.1, softcap=3.)
    for _ in range(3):
        index.copy_(torch.randint(17, (97,), device='cuda', dtype=dtype))
        y = module(x, index, num_nodes=19, exp_rescale=r)
        ref = reference_graph_softmax(x, index=index.long(), num_nodes=19,
                                      exp_rescale=r, eps=.1, softcap=3.)
        torch.testing.assert_close(y, ref, atol=3e-6, rtol=3e-5)
        for a, b in zip(torch.autograd.grad(y, (x, r), g),
                        torch.autograd.grad(ref, (x, r), g)):
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)


def test_high_degree_is_fused():
    x = torch.randn(8193, 3, device='cuda', requires_grad=True)
    index = torch.zeros(8193, device='cuda', dtype=torch.int64)
    y = fused_graph_softmax_index(x, index, num_nodes=3)
    assert type(y.grad_fn).__name__ == '_GraphSoftmaxIndexFnBackward'
    ref = reference_graph_softmax(x, index=index, num_nodes=3)
    torch.testing.assert_close(y, ref, atol=3e-6, rtol=3e-5)
    g = torch.randn_like(x)
    a, = torch.autograd.grad(y, x, g)
    b, = torch.autograd.grad(ref, x, g)
    torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)


def test_empty_input():
    x = torch.empty(0, 3, device='cuda', requires_grad=True)
    r = torch.ones(1, 3, device='cuda', requires_grad=True)
    y = fused_graph_softmax_index(x, torch.empty(0, device='cuda', dtype=torch.int64),
                                  num_nodes=17, exp_rescale=r)
    dx, dr = torch.autograd.grad(y.sum(), (x, r))
    assert dx.shape == x.shape
    torch.testing.assert_close(dr, torch.zeros_like(r))


def test_dropout_matches_csr_seed_and_gradients():
    from fasteq.triton.graph_softmax import fused_graph_softmax
    torch.manual_seed(181)
    x = torch.randn(97, 8, device='cuda', requires_grad=True)
    r = torch.rand(97, 1, device='cuda', requires_grad=True)
    index = torch.randint(17, (97,), device='cuda')
    seed = torch.tensor(581, device='cuda', dtype=torch.int64)
    kwargs = dict(num_nodes=19, exp_rescale=r, exp_dropout=.3, training=True,
                  softcap=3., seed=seed)
    y = fused_graph_softmax_index(x, index, **kwargs)
    other = fused_graph_softmax(x, index, **kwargs)
    assert torch.equal(y == 0, other == 0)
    torch.testing.assert_close(y, other, atol=3e-6, rtol=3e-5)
    g = torch.randn_like(x)
    for a, b in zip(torch.autograd.grad(y, (x, r), g),
                    torch.autograd.grad(other, (x, r), g)):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
