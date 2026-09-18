"""Ordinary cases use native EQv3 FP32; proven cancellation cases use analytics.

The exceptional reference applies only to the archived shared-rescale fixtures
and single-neighbor/small-rescale cases. It does not relax the FP32 tolerances
or waive failures on ordinary inputs. These are first-order tests.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

pytest.importorskip('triton')
from fasteq.triton.graph_softmax import fused_graph_softmax, prepare_graph_softmax
from fasteq.triton.graph_edge_layout import prepare_graph_softmax_layout

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/HIP GPU required')
ATOL, RTOL = 3e-5, 3e-4


@pytest.fixture(autouse=True, params=['original_order', 'node_sorted'])
def edge_layout_route(request, monkeypatch):
    """Run the complete precision suite through both original and packed paths."""
    if request.param == 'original_order':
        return
    original = fused_graph_softmax

    def packed(src, index=None, ptr=None, num_nodes=None, **kwargs):
        if ptr is not None:
            return original(src, index=index, ptr=ptr, num_nodes=num_nodes, **kwargs)
        layout = prepare_graph_softmax_layout(index, num_nodes=num_nodes)
        kwargs.pop('csr', None)
        r = kwargs.pop('exp_rescale', None)
        return layout.unpack(layout.softmax(layout.pack(src),
            exp_rescale=layout.pack_rescale(r), **kwargs))

    monkeypatch.setattr(__import__(__name__, fromlist=['fused_graph_softmax']),
                        'fused_graph_softmax', packed)


@pytest.fixture(scope='module')
def native_class():
    root = os.environ.get('FASTEQ_EQUIFORMER_V3_SOURCE_DIR')
    if not root:
        pytest.skip('original EQv3 softmax.py is required')
    path = Path(root)/'softmax.py'
    spec = importlib.util.spec_from_file_location('original_eqv3_softmax_precision', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GraphSoftmax


def compare(native_class, h=8, shape='edge', cap=3., eps=.1, p=0., route='index',
            heads=8, subset='both', noncontiguous=False):
    torch.manual_seed(71)
    e, n = 73, 17
    index = torch.randint(n-2, (e,), device='cuda')
    if route == 'ptr':
        index = index.sort().values
    csr = prepare_graph_softmax(index=index, num_nodes=n)
    kwargs = {'index':index, 'num_nodes':n}
    if route == 'ptr':
        kwargs = {'ptr':csr.ptr}
        csr = prepare_graph_softmax(ptr=csr.ptr)
    x = torch.randn(e,h,device='cuda',requires_grad=subset!='r')
    shapes = {'scalar':(), 'head':(h,), 'row':(1,h), 'edge':(e,1), 'full':(e,h)}
    r = None
    if shape != 'none':
        r = .1 + torch.rand(shapes[shape],device='cuda')
        if noncontiguous and r.ndim == 2:
            r = r.t().contiguous().t()
        r.requires_grad_(subset!='x')
        with torch.no_grad():
            if r.numel():
                r.reshape(-1)[0] = 0.
    grad = torch.randn(e,h,device='cuda')
    if noncontiguous:
        grad = grad.t().contiguous().t()
    seed = torch.tensor(123,device='cuda',dtype=torch.int64)
    y = fused_graph_softmax(x,exp_rescale=r,csr=csr,softcap=cap,eps=eps,
                           exp_dropout=p,training=True,seed=seed,block_heads=heads,**kwargs)
    rr = r
    if p:
        with torch.no_grad():
            uniform = fused_graph_softmax(torch.zeros_like(x),csr=csr,
                exp_dropout=p,training=True,seed=seed,block_heads=heads,**kwargs)
            mask = (uniform != 0).float()/(1-p) if p<1 else torch.zeros_like(x)
        rr = mask if r is None else r*mask
    ref = native_class(eps=eps,softcap=cap,exp_dropout=0.)(x,exp_rescale=rr,**kwargs)
    inputs = [v for v in (x,r) if v is not None and v.requires_grad]
    expected = torch.autograd.grad(ref,inputs,grad)
    actual = torch.autograd.grad(y,inputs,grad)
    torch.testing.assert_close(y,ref,atol=3e-6,rtol=3e-5)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=ATOL,rtol=RTOL)


@pytest.mark.parametrize('shape',['none','scalar','head','row','edge','full'])
@pytest.mark.parametrize('h',[1,3,8,17])
@pytest.mark.parametrize('cap',[None,3.])
def test_native_fp32_shapes(native_class,shape,h,cap):
    # Non-negligible eps: shared-rescale gradients do not cancel to ~zero.
    compare(native_class,shape=shape,h=h,cap=cap,eps=.1)


@pytest.mark.parametrize('shape',['none','edge','full'])
@pytest.mark.parametrize('route',['index','ptr'])
@pytest.mark.parametrize('p',[0.,.25,1.])
def test_native_fp32_default_eps_dropout(native_class,shape,route,p):
    compare(native_class,shape=shape,route=route,p=p,eps=1e-16,noncontiguous=True)


@pytest.mark.parametrize('shape',['scalar','head','row'])
@pytest.mark.parametrize('p',[.25,1.])
def test_native_fp32_shared_dropout(native_class,shape,p):
    compare(native_class,shape=shape,p=p,eps=.1)


@pytest.mark.parametrize('subset',['x','r'])
@pytest.mark.parametrize('shape',['scalar','head','edge','full'])
def test_requested_gradient_subset(native_class,subset,shape):
    compare(native_class,subset=subset,shape=shape)


@pytest.mark.parametrize('h,heads',[(3,1),(3,4),(17,4),(33,8),(300,32)])
def test_head_tiles_and_reduction_tail(native_class,h,heads):
    compare(native_class,h=h,heads=heads,shape='edge')


def test_expanded_input_has_distinct_gradients(native_class):
    torch.manual_seed(91)
    e,n,h=73,17,8
    idx=torch.randint(n,(e,),device='cuda')
    x=torch.randn(e,h,device='cuda',requires_grad=True)
    r=torch.rand(1,h,device='cuda').expand(e,h).detach().requires_grad_(True)
    g=torch.randn_like(x)
    y=fused_graph_softmax(x,idx,num_nodes=n,exp_rescale=r,eps=.1)
    ref=native_class(eps=.1)(x,idx,num_nodes=n,exp_rescale=r)
    a=torch.autograd.grad(y,(x,r),g);b=torch.autograd.grad(ref,(x,r),g)
    for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,atol=ATOL,rtol=RTOL)
    assert a[1].shape == (e,h)


@pytest.mark.parametrize('fixture',['h100','hygon'])
def test_archived_shared_cancellation_analytic(native_class,fixture):
    raw=json.loads((Path(__file__).parent/'data'/'graph_softmax'/f'{fixture}.json').read_text())
    data={k:torch.tensor(v,dtype=torch.int64 if k=='index' else torch.float32)
          for k,v in raw.items()}
    x=data['x'].cuda().requires_grad_();r=data['r'].cuda().requires_grad_()
    idx=data['index'].cuda();g=data['g'].cuda();eps=1e-16
    y=fused_graph_softmax(x,idx,num_nodes=17,exp_rescale=r,softcap=3.,eps=eps)
    dx,dr=torch.autograd.grad(y,(x,r),g)
    ref=native_class(eps=eps,softcap=3.)(x,idx,num_nodes=17,exp_rescale=r)
    gx,gr_fp32=torch.autograd.grad(ref,(x,r),g)
    torch.testing.assert_close(y,ref,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(dx,gx,atol=ATOL,rtol=RTOL)
    # r=0 has a large, non-cancelling gradient; keep its native FP32 check.
    torch.testing.assert_close(dr[r==0],gr_fp32[r==0],atol=ATOL,rtol=RTOL)
    # Independent per-node closed form; no subtraction of cancelling gradients.
    xd=data['x'].double();rd=data['r'].double();gd=data['g'].double()
    expected=torch.zeros_like(rd)
    z=3*torch.tanh(xd/3)
    for node in range(17):
        select=data['index']==node
        if not select.any():continue
        v=z[select];u=torch.exp(v-v.max(0).values)
        den=rd*u.sum(0,keepdim=True)+eps
        expected+=eps*(gd[select]*u).sum(0,keepdim=True)/den.square()
    torch.testing.assert_close(dr.cpu().double(),expected,atol=ATOL,rtol=RTOL)
    # Preserve the eps-sized signal, rather than merely passing by returning 0.
    torch.testing.assert_close(dr[0,2].cpu().double(),expected[0,2],atol=1e-18,rtol=3e-4)


@pytest.mark.parametrize('scale',[1e-2,1e-4,1e-6])
@pytest.mark.parametrize('shape',['edge','full'])
def test_single_neighbor_small_rescale_analytic(native_class,scale,shape):
    torch.manual_seed(71)
    e,h=17,8;idx=torch.arange(e,device='cuda')
    x=torch.randn(e,h,device='cuda',requires_grad=True)
    r=((.1+.9*torch.rand(e,1 if shape=='edge' else h,device='cuda'))*scale).requires_grad_()
    g=torch.randn_like(x);eps=1e-16
    y=fused_graph_softmax(x,idx,num_nodes=e,exp_rescale=r,softcap=3.,eps=eps)
    dx,dr=torch.autograd.grad(y,(x,r),g)
    ref=native_class(eps=eps,softcap=3.)(x,idx,num_nodes=e,exp_rescale=r)
    gx,=torch.autograd.grad(ref,x,g)
    torch.testing.assert_close(y,ref,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(dx,gx,atol=ATOL,rtol=RTOL)
    rd=r.detach().double();expected=eps*g.double()/(rd+eps).square()
    if shape=='edge':expected=expected.sum(-1,keepdim=True)
    torch.testing.assert_close(dr.double(),expected,atol=ATOL,rtol=RTOL)
    torch.testing.assert_close(dr.double(),expected,atol=1e-15,rtol=3e-5)


@pytest.mark.parametrize('nodes',[1,17,257,65537])
def test_shared_zero_rescale_reduction_levels(native_class,nodes):
    torch.manual_seed(301);h=3
    idx=torch.arange(nodes,device='cuda');x=torch.randn(nodes,h,device='cuda',requires_grad=True)
    r=torch.zeros((),device='cuda',requires_grad=True);g=torch.randn_like(x)
    y=fused_graph_softmax(x,idx,num_nodes=nodes,exp_rescale=r,eps=.1)
    ref=native_class(eps=.1)(x,idx,num_nodes=nodes,exp_rescale=r)
    a=torch.autograd.grad(y,(x,r),g);b=torch.autograd.grad(ref,(x,r),g)
    for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,atol=ATOL,rtol=RTOL)


def test_backward_does_not_use_torch_reductions(monkeypatch):
    idx=torch.arange(73,device='cuda')%17
    x=torch.randn(73,3,device='cuda',requires_grad=True)
    r=torch.rand(1,3,device='cuda',requires_grad=True)
    y=fused_graph_softmax(x,idx,num_nodes=17,exp_rescale=r,eps=.1)
    def forbidden(*args,**kwargs):raise AssertionError('backward must use Triton reductions')
    monkeypatch.setattr(torch.Tensor,'sum_to_size',forbidden)
    monkeypatch.setattr(torch.autograd,'grad',forbidden)
    y.backward(torch.randn_like(y))
    assert x.grad is not None and r.grad is not None
