"""Permutation alignment, gradient routing and seeded dropout invariance."""
import pytest
import torch

pytest.importorskip('triton')
from fasteq.triton.graph_edge_layout import prepare_graph_softmax_layout
from fasteq.triton.graph_softmax import fused_graph_softmax

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/HIP GPU required')


@pytest.mark.parametrize('shape', [(0, 3), (17, 0), (17, 3), (17, 3, 5)])
def test_permutation_roundtrip_and_gradient(shape):
    index = torch.arange(shape[0], device='cuda') % 5
    layout = prepare_graph_softmax_layout(index, num_nodes=7)
    x = torch.randn(shape, device='cuda', requires_grad=True)
    if len(shape) == 3:
        x = x.transpose(1, 2).detach().requires_grad_()
    g = torch.randn_like(x)
    packed = layout.pack(x)
    torch.testing.assert_close(packed, x[layout.order], atol=0, rtol=0)
    y = layout.unpack(packed)
    torch.testing.assert_close(y, x, atol=0, rtol=0)
    gx, = torch.autograd.grad(y, x, g)
    torch.testing.assert_close(gx, g, atol=0, rtol=0)


def test_topology_change_rejected():
    index = torch.tensor([2, 0, 1, 0], device='cuda')
    layout = prepare_graph_softmax_layout(index, num_nodes=4)
    index[0] = 1
    with pytest.raises(ValueError, match='topology changed'):
        layout.pack(torch.randn(4, 3, device='cuda'))


def test_edge_metadata_alignment():
    index = torch.tensor([2, 0, 1, 0], device='cuda')
    layout = prepare_graph_softmax_layout(index, num_nodes=4)
    endpoints = torch.stack((torch.arange(4, device='cuda'), index), dim=1)
    packed = layout.pack(endpoints)
    torch.testing.assert_close(packed[:, 1], index.sort(stable=True).values)
    torch.testing.assert_close(layout.unpack(packed), endpoints)
    with pytest.raises(ValueError, match='dimension zero'):
        layout.pack(torch.randn(3, 4, device='cuda'))


@pytest.mark.parametrize('p', [0., .25, 1.])
@pytest.mark.parametrize('shape', ['head', 'edge', 'full'])
def test_same_seed_mask_output_and_gradients(p, shape):
    torch.manual_seed(908)
    e, n, h = 73, 17, 8
    index = torch.randint(n-2, (e,), device='cuda')
    x = torch.randn(e, h, device='cuda', requires_grad=True)
    sizes = {'head': (1, h), 'edge': (e, 1), 'full': (e, h)}
    r = (.1+torch.rand(sizes[shape], device='cuda')).requires_grad_()
    g = torch.randn_like(x)
    seed = torch.tensor(9981, device='cuda', dtype=torch.int64)
    layout = prepare_graph_softmax_layout(index, num_nodes=n)
    opts = dict(eps=.1, softcap=3., exp_dropout=p, training=True, seed=seed)
    before = fused_graph_softmax(x, index, num_nodes=n, exp_rescale=r, **opts)
    after = layout.unpack(layout.softmax(layout.pack(x),
        exp_rescale=layout.pack_rescale(r), **opts))
    torch.testing.assert_close(after == 0, before == 0)
    torch.testing.assert_close(after, before, atol=3e-6, rtol=3e-5)
    a = torch.autograd.grad(after, (x, r), g)
    b = torch.autograd.grad(before, (x, r), g)
    for aa, bb in zip(a, b):
        torch.testing.assert_close(aa, bb, atol=3e-5, rtol=3e-4)


def test_pack_gradient_uses_permutation_not_atomic_index_add(monkeypatch):
    index = torch.arange(73, device='cuda') % 17
    layout = prepare_graph_softmax_layout(index, num_nodes=17)
    x = torch.randn(73, 8, device='cuda', requires_grad=True)
    y = layout.unpack(layout.pack(x))
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU permutation backward must use Triton copies')
    monkeypatch.setattr(torch.Tensor, 'index_add_', forbidden)
    monkeypatch.setattr(torch.Tensor, 'scatter_add_', forbidden)
    y.backward(torch.ones_like(y))
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_attention_messages_keep_same_edge_alignment():
    torch.manual_seed(109)
    e, n, h, c = 137, 19, 3, 5
    index = torch.randint(n, (e,), device='cuda')
    layout = prepare_graph_softmax_layout(index, num_nodes=n)
    x = torch.randn(e, h, device='cuda', requires_grad=True)
    r = (.1+torch.rand(e, 1, device='cuda')).requires_grad_()
    values = torch.randn(e, h, c, device='cuda', requires_grad=True)
    node_grad = torch.randn(n, h, c, device='cuda')
    alpha = fused_graph_softmax(x, index, num_nodes=n, exp_rescale=r)
    before = torch.zeros(n, h, c, device='cuda').index_add(
        0, index, alpha.unsqueeze(-1)*values)
    alpha_sorted = layout.softmax(layout.pack(x), exp_rescale=layout.pack_rescale(r))
    after = torch.zeros(n, h, c, device='cuda').index_add(
        0, layout.pack(index), alpha_sorted.unsqueeze(-1)*layout.pack(values))
    torch.testing.assert_close(after, before, atol=3e-6, rtol=3e-5)
    a = torch.autograd.grad(after, (x, r, values), node_grad)
    b = torch.autograd.grad(before, (x, r, values), node_grad)
    for aa, bb in zip(a, b):
        torch.testing.assert_close(aa, bb, atol=3e-5, rtol=3e-4)
