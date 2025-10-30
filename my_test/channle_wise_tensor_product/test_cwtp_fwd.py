import torch, functools
torch.load = functools.partial(torch.load, weights_only=False)
import numpy as np
import os
import math
import time
import cuequivariance as cue
import cuequivariance_torch as cueq
import e3nn.o3 as o3
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.cpp_extension import load
# import cProfile, pstats
# import pynvml

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

so_path = "/home/malixian/repos/cuequivariance_torch/primitives/_kernels/cuda/build/bin/libfasteq.so"
torch.ops.load_library(str(so_path))
test_cwtp_fwd = torch.ops.cwtp_fwd

torch.manual_seed(42)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# torch.cuda.set_device(device) 
dtype = torch.float64

# cwtp_forward = load(
#     name="1",
#     sources=["cwtp_forward.cu"],
#     extra_cflags=["-O3"],
#     extra_cuda_cflags=["-O3", '-gencode=arch=compute_90,code=sm_90'],
#     verbose=True
# )

cwtp_forward_cuda = load(
    name="cwtp_fwd",
    sources=["cwtp_fwd_opt1.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", '-gencode=arch=compute_90,code=sm_90'],
    verbose=True
)


B, U, V = 26840, 96, 1
dim_list = [1, 3, 5, 7]
dim_sum = sum(dim_list)  # 16
path_num = len(dim_list) # 4

x_all = torch.randn(B, U, V, dtype=dtype, device=device, requires_grad=True)  # 96×1=96
y_all = torch.randn(B, dim_sum, dtype=dtype, device=device, requires_grad=True)    # 1×16=16
w_all = torch.randn(B, path_num * U * V, dtype=dtype, device=device, requires_grad=True)


def einsum_test(x, y, w):
    einsum_out = None
    w = w.reshape(B, path_num, U * V)
    offset = 0
    for did in range(0, len(dim_list)):
        w_seg = w[:, did, :].reshape(B, U, V)
        dim = dim_list[did]
        x_seg = x.unsqueeze(1)
        y_seg = y[:, offset:offset+dim].unsqueeze(2)
        cg_tensor = torch.eye(dim, dtype=torch.float64).unsqueeze(0).to("cuda")
        out = torch.einsum("ijk,Zuv,Ziu,Zjv->Zkuv", cg_tensor, w_seg, x_seg, y_seg)
        offset += dim
        if einsum_out == None:
            einsum_out = out
        else:
            einsum_out = torch.cat([einsum_out, out], dim=1)
    einsum_out = einsum_out.reshape(B, dim_sum * U * V)
    return einsum_out

def einsum_test_simplified(w_flat, x, y):
    """
    x: [B, U, V]
    y: [B, dim_sum]
    w_flat: [B, path_num * U * V]
    """
    path_num = len(dim_list)
    dim_sum = sum(dim_list)

    # 还原 w 形状：[B, path_num, U, V]
    w = w_flat.view(B, path_num, U, V)

    # 按 dim_list 对 y 分段
    y_chunks = y.split(dim_list, dim=1)

    outs = []
    for p, y_p in enumerate(y_chunks):
        base = x * w[:, p] 
        base = base.unsqueeze(1)
        y_p = y_p.unsqueeze(-1).unsqueeze(-1)
        out_p = base * y_p
        outs.append(out_p)

    out = torch.cat(outs, dim=1).reshape(B, dim_sum * U * V)
    return out

def test_multi_path_tp():
    irreps_in1 = cue.Irreps("O3", "96x0e")
    irreps_in2 = cue.Irreps("O3", "1x0e+1x1o+1x2e+1x3o")
    irreps_out = cue.Irreps("O3", "96x0e+96x1o+96x2e+96x3o")

    tp = cueq.ChannelWiseTensorProduct(
        irreps_in1, irreps_in2, 
        filter_irreps_out = irreps_out, 
        # instructions = instructions,
        layout=cue.mul_ir,
        shared_weights=False,     
        internal_weights=False,
        device = device    
    )

    retry = 20
    
    # ###### cuda(4 paths) ######
    for retry_id in range(0, retry):
        output = torch.zeros(B, U * dim_sum, dtype=dtype, device=device)
        torch.cuda.synchronize()
        start_total1 = time.perf_counter() * 1000
        #output = cwtp_forward_total.forward(x_all.contiguous(), y_all.contiguous(), w_all.contiguous())
        #output = test_cwtp_fwd.forward(x_all.contiguous(), y_all.contiguous(), w_all.contiguous()) 
        output = einsum_test_simplified(w_all.contiguous(), x_all.contiguous(), y_all.contiguous()).reshape(B, dim_sum * U * V)
        torch.cuda.synchronize()
        end_total1 = time.perf_counter() * 1000
        t = (end_total1 - start_total1)
        if retry_id == retry - 1:
            print(f"cuda cost: {t:.4f} ms")
            print(f"ref output.shape={output.shape}, output.stride={output.stride()}")
    
    x_ref = x_all.clone().detach().requires_grad_(True)
    y_ref = y_all.clone().detach().requires_grad_(True)
    w_ref = w_all.clone().detach().requires_grad_(True)
    x_ref = x_ref.reshape(B, U*V)

    
    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start_total = time.perf_counter() *1000
        #out_ref = tp(x_ref, y_ref, w_ref)
        #out_ref = torch.zeros(B, dim_sum, U, dtype=dtype, device=device)
        out_ref = cwtp_forward_cuda.forward(x_all.contiguous(), y_all.contiguous(), w_all.contiguous())
        out_ref = out_ref.reshape(B, -1)
        torch.cuda.synchronize()
        end_total = time.perf_counter() *1000
        t = (end_total - start_total)
        if retry_id == retry - 1:
            print(f"my cuda cost: {t:.4f} ms")

    print(f"out_ref.shape={out_ref.shape}, output.shape={output.shape}")
        
    print(f"compare out: {torch.allclose(out_ref, output, rtol=1e-05, atol=1e-08, equal_nan=False)}")
    print(f'差值最大值:', (output - out_ref).abs().max().item())
    #print(f'output[0:]={output[0:]}')
    #print(f'out[0:]={out[0:]}')
    """ for i in range(96*2):
        print(f'直接调用{i}列差值最大值:', output[0, i].item()) """


if __name__ == "__main__":
    test_multi_path_tp()
    #einsum_test_simplified(w_all, x_all, y_all)
