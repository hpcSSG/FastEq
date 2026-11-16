"""HIP indexed forward: ownership, initialized holes, tails and dispatch cache."""
import os
import pytest
import torch
from fasteq.ops import uniform1d_jit as u1d
from fasteq.ops.auto_schedule.uniform1d_forward import generate_code_uniform1d_fwd_with_scheduler


def test_policy_keeps_scatter_dense_and_cuda_on_existing_route(monkeypatch):
    monkeypatch.setattr(u1d, '_detect_gpu_backend', lambda: 'hip')
    monkeypatch.setenv('FASTEQ_HIP_CWTP_FWD', '1')
    assert u1d._hip_cwtp_fwd_block_size({1: object()}, {}, 128) == 128
    assert u1d._hip_cwtp_fwd_block_size({2: object()}, {}, 65) == 128
    assert u1d._hip_cwtp_fwd_block_size({1: object()}, {}, 33) == 64
    assert u1d._hip_cwtp_fwd_block_size({1: object()}, {0: object()}, 128) is None
    assert u1d._hip_cwtp_fwd_block_size({}, {}, 128) is None
    monkeypatch.setattr(u1d, '_detect_gpu_backend', lambda: 'cuda')
    assert u1d._hip_cwtp_fwd_block_size({1: object()}, {}, 128) is None
    monkeypatch.setattr(u1d, '_detect_gpu_backend', lambda: 'hip')
    monkeypatch.setenv('FASTEQ_HIP_CWTP_FWD', '0')
    assert u1d._hip_cwtp_fwd_block_size({1: object()}, {}, 128) is None


@pytest.mark.parametrize('scatter,holes', [(False,False),(False,True),(True,False)])
def test_scheduler_proves_output_ownership_before_uninitialized_allocation(scatter,holes):
    v=[0,0,2] if holes else [0,0,1]
    cols=[torch.tensor(x,dtype=torch.int32) for x in [[0,1,0],[0,1,0],[0,0,1],v]]
    result=generate_code_uniform1d_fwd_with_scheduler(*cols,torch.tensor([.5,-.25,1.]),
        input_indices={1:object()},output_indices={0:object()} if scatter else {},
        u_dim=128,optimize_indexed=True,block_size=128,return_schedule=True)
    summary=result['schedule'].placement_summary;code=result['code']
    if scatter:
        assert not summary.get('exclusive_outputs',False)
        assert 'atomicAdd(&out[' in code and 'torch::empty({B, V, U}' not in code
    else:
        assert summary['exclusive_outputs']
        stores=[i for i in result['schedule'].instructions if i.op=='store_acc_placed']
        assert sorted(str(i.args[0]) for i in stores)==['out[0]','out[2]' if holes else 'out[1]']
        assert ('torch::empty({B, V, U}' in code) == (not holes)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.hip,reason='HIP GPU required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('kind',['x_tail_holes','xy_multi_path','y_tail','scatter_fallback'])
def test_indexed_forward_values_tail_shared_weights_strides(monkeypatch,dtype,kind):
    monkeypatch.setattr(u1d,'ENABLE_JIT_CANDIDATES',False)
    monkeypatch.setenv('FASTEQ_CODEGEN_BACKEND','native');monkeypatch.setenv('FASTEQ_HIP_CWTP_FWD','1')
    torch.manual_seed(1701);N,E=5,11;U=65 if kind in ['x_tail_holes','scatter_fallback'] else (128 if kind=='xy_multi_path' else 33)
    vector=kind=='xy_multi_path';shared=kind in ['x_tail_holes','xy_multi_path'];sx=kind!='y_tail';sy=kind in ['xy_multi_path','y_tail'];scatter=kind=='scatter_fallback'
    I,J,K=2,2,2;V=2 if vector else 3
    paths=[(0,0,0,0),(1,1,1,0),(0,1,0,1 if vector else 2)]
    src=(torch.arange(E,device='cuda')*3%N).int();dst=(torch.arange(E,device='cuda')*2%N).int()
    def value(rows,cols):
        # Exercise the native contiguity precondition before caller-side packing.
        return (torch.randn(rows,2*cols+3,device='cuda',dtype=dtype)*.2)[:,1:1+2*cols:2]
    w=value(1 if shared else E,I*U);x=value(N if sx else E,J*U);y=value(N if sy else E,K*(U if vector else 1))
    meta={name:torch.tensor(c,device='cuda',dtype=torch.int32) for name,c in zip(['i_list','j_list','k_list','v_list'],zip(*paths))}
    meta.update(coeff_list=torch.tensor([.5,-.25,.75],device='cuda',dtype=dtype),w_seg_num=I,x_seg_num=J,y_seg_num=K,out_seg_num=V,u_dim=U,size_list=[I*U,J*U,K*(U if vector else 1),V*U])
    inp={}
    if sx:inp[1]=src
    if sy:inp[2]=src
    ww=w.reshape(-1,I,U).expand(E,-1,-1);xx=x.reshape(-1,J,U);yy=y.reshape(-1,K,U if vector else 1)
    if sx:xx=xx[src.long()]
    if sy:yy=yy[src.long()]
    ref=torch.zeros(E,V,U,device='cuda',dtype=dtype)
    for (i,j,k,v),c in zip(paths,meta['coeff_list']):ref[:,v]+=c*ww[:,i]*xx[:,j]*yy[:,k]
    if scatter:ref=ref.new_zeros(N,V,U).index_add(0,dst.long(),ref)
    expected=ref.flatten(1)
    # The existing native public entry requires contiguous inputs. Keep that
    # contract, and test caller packing plus a nonzero contiguous storage offset.
    with pytest.raises(RuntimeError,match='contiguous'):
        u1d.fast_uniform1d_jit(w,x,y,inp,{0:dst} if scatter else {},meta,False)
    packed=[torch.cat((t.new_zeros(1),t.reshape(-1)))[1:].view(t.shape) for t in (w,x,y)]
    assert all(t.is_contiguous() and t.storage_offset()==1 for t in packed)
    actual=u1d.fast_uniform1d_jit(*packed,inp,{0:dst} if scatter else {},meta,False)
    torch.testing.assert_close(actual,expected,atol=2e-8 if dtype==torch.float64 else 2e-5,rtol=2e-7 if dtype==torch.float64 else 2e-4)
    if not vector:assert torch.count_nonzero(actual.reshape(-1,V,U)[:,1])==0


def test_disabled_placement_keeps_initialized_output():
    t=torch.tensor([0],dtype=torch.int32)
    code=generate_code_uniform1d_fwd_with_scheduler(t,t,t,t,torch.tensor([1.]),
        input_indices={1:object()},u_dim=128,optimize_indexed=True,block_size=128,
        placement_config={'enabled':False})
    assert 'torch::empty({B, V, U}' not in code
    assert 'torch::zeros({B, V, U}' in code
