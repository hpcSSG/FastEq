"""First-order attention gradients against native LayerNorm and einsum.

Set FASTEQ_ALPHA_LARGE_TESTS=1 to include the original large reduction failures.
"""
import ast
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip('triton')
from fasteq.triton.fused_attention_alpha import fused_attention_alpha, fused_atten_alpha

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/HIP GPU required')
ATOL, RTOL = 5e-5, 5e-4


def native(x, w, g, b, ln=True, activation='smooth_leaky_relu', slope=.2,
           eps=1e-5, p=0., mask=None):
    z = F.layer_norm(x, (x.shape[-1],), g, b, eps) if ln else x
    a = F.silu(z) if activation == 'silu' else (
        ((1+slope)/2)*z + ((1-slope)/2)*z*(2*torch.sigmoid(z)-1))
    if p:
        a = a * (mask.to(a.dtype)/(1-p) if p < 1 else 0.)
    return torch.einsum('bik,ik->bi', a, w)


def compare(e=17, h=3, c=32, ln=True, affine='both', activation='smooth_leaky_relu',
            p=0., training=True, noncontiguous=False, subset=None, eps=1e-5,
            slope=.2, constant=False, seed=20260914):
    torch.manual_seed(seed)
    # Preserve the original failure harness's RNG/input order.
    w = torch.randn(h, c, device='cuda', requires_grad=True)
    g = torch.randn(c, device='cuda', requires_grad=True) if ln and affine in ('both','weight') else None
    b = torch.randn(c, device='cuda', requires_grad=True) if ln and affine in ('both','bias') else None
    x = torch.randn(e, h, c, device='cuda')
    if constant:
        x.fill_(.75)
    if noncontiguous:
        x = x.transpose(0, 1).contiguous().transpose(0, 1)
        w = w.t().contiguous().t().detach().requires_grad_(True)
    x.requires_grad_(True)
    dy = torch.randn(e, h, device='cuda')
    if noncontiguous:
        dy = dy.T.contiguous().T
    named = [('x',x),('w',w)]+([('g',g)] if g is not None else [])+([('b',b)] if b is not None else [])
    for name, t in named:
        t.requires_grad_(subset is None or name in subset)
    inputs = [t for _, t in named if t.requires_grad]
    y = fused_attention_alpha(x, w, g, b, use_layer_norm=ln, eps=eps,
        activation=activation, negative_slope=slope, dropout_p=p, training=training,
        seed=torch.tensor(123, device=x.device, dtype=torch.int64))
    mask = y.grad_fn.saved_tensors[-1]
    ref = native(x,w,g,b,ln,activation,slope,eps,p if training else 0.,mask)
    expected = torch.autograd.grad(ref, inputs, dy)
    actual = torch.autograd.grad(y, inputs, dy)
    torch.testing.assert_close(y, ref, atol=ATOL, rtol=RTOL)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('c', [1,3,13,32,127,256,1025,8192])
@pytest.mark.parametrize('ln', [False,True])
@pytest.mark.parametrize('activation', ['silu','smooth_leaky_relu'])
def test_shapes(c, ln, activation):
    compare(c=c,ln=ln,affine='both' if ln else 'none',activation=activation)


@pytest.mark.parametrize('affine', ['none','weight','bias','both'])
@pytest.mark.parametrize('p', [0.,.25,1.])
def test_dropout_and_strides(affine,p):
    compare(e=19,h=5,c=33,affine=affine,p=p,noncontiguous=True)


@pytest.mark.parametrize('subset', [['x'],['w'],['g'],['b'],['w','b']])
def test_partial_gradients(subset):
    compare(e=11,h=2,c=13,subset=subset)


@pytest.mark.parametrize('e,h,c', [(0,3,32),(1,1,1),(37,7,64)])
def test_empty_and_partial_tiles(e,h,c):
    compare(e=e,h=h,c=c)


@pytest.mark.parametrize('eps,slope', [(1e-3,.01),(1e-6,.5)])
def test_nondefault_parameters(eps,slope):
    compare(c=13,eps=eps,slope=slope)


def test_constant_inputs():
    compare(constant=True)


def test_eval_disables_dropout():
    compare(p=.7,training=False)


def test_no_torch_recomputation(monkeypatch):
    x = torch.randn(19,3,32,device='cuda',requires_grad=True)
    w = torch.randn(3,32,device='cuda',requires_grad=True)
    y = fused_attention_alpha(x,w)
    def forbidden(*args,**kwargs):
        raise AssertionError('backward must not rebuild a Torch autograd graph')
    monkeypatch.setattr(torch.autograd,'grad',forbidden)
    monkeypatch.setattr(F,'layer_norm',forbidden)
    y.backward(torch.randn_like(y))
    assert x.grad is not None and w.grad is not None


def test_higher_order_is_explicitly_unsupported():
    x = torch.randn(2,3,13,device='cuda',requires_grad=True)
    w = torch.randn(3,13,device='cuda',requires_grad=True)
    y = fused_attention_alpha(x,w)
    with pytest.raises(NotImplementedError,match='first-order'):
        torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True)


def test_original_eqv3_adapter():
    root = os.environ.get('FASTEQ_EQUIFORMER_V3_SOURCE_DIR')
    if not root:
        pytest.skip('original EQv3 source directory is required')
    path = Path(root)/'activation.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SmoothLeakyReLU')
    scope = {'torch':torch,'__name__':'eqv3_original_activation'}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
    act = scope['SmoothLeakyReLU']().cuda()
    norm = torch.nn.LayerNorm(32).cuda()
    drop = torch.nn.Dropout(.2).cuda().eval()
    x = torch.randn(73,8,32,device='cuda',requires_grad=True)
    w = torch.randn(8,32,device='cuda',requires_grad=True)
    ref = torch.einsum('bik,ik->bi',drop(act(norm(x))),w)
    y = fused_atten_alpha(x,w,norm,act,drop)
    dy = torch.randn_like(y)
    inputs = (x,w,norm.weight,norm.bias)
    expected = torch.autograd.grad(ref,inputs,dy)
    actual = torch.autograd.grad(y,inputs,dy)
    torch.testing.assert_close(y,ref,atol=ATOL,rtol=RTOL)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=ATOL,rtol=RTOL)


@pytest.mark.skipif(os.environ.get('FASTEQ_ALPHA_LARGE_TESTS')!='1', reason='opt-in large GPU allocation')
@pytest.mark.parametrize('n', [4096,8192,16384,32768,131072])
def test_original_large_reduction_cases(n):
    compare(e=n*32,h=8,c=32)
    torch.cuda.empty_cache()


@pytest.mark.parametrize('seed,c,activation', [
    (123,32,'smooth_leaky_relu'),
    (7,64,'silu'),
])
def test_parameter_cancellation_cases(seed,c,activation):
    compare(e=131072,h=8,c=c,activation=activation,seed=seed)
    torch.cuda.empty_cache()


def test_padded_weight_gradient_grid():
    # H*C exceeds the one-dimensional grid bound, leaving padded programs.
    compare(e=3,h=257,c=257,ln=False,affine='none')
