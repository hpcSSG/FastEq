import torch
import os, time
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.manual_seed(42)

batch_size = 32
batch_one = 184
BATCH = batch_one * batch_size
U, V, W = 96, 10, 96
UV = U * V

a = torch.randn(BATCH, U, dtype=torch.float64, device="cuda")
b = torch.randn(BATCH, V, dtype=torch.float64, device="cuda")
w = torch.randn(U, V, W, dtype=torch.float64, device="cuda")
tp_out = torch.zeros(BATCH, U, V, dtype=torch.float64, device="cuda")

def test_full_path_tp():
    i_dim = 16
    j_dim = 1
    k_dim = 16
    
    cg_list = []
    path_num = 4
    for pid in range(0, path_num):
        dim = pid * 2 + 1
        cg_tensor = torch.randn(dim, 1, dim, dtype=torch.float64, device="cuda")
        cg_list.append(cg_tensor)

    a_full = torch.randn(BATCH, i_dim, U, dtype=torch.float64, device="cuda")
    b_full = torch.randn(BATCH, j_dim, V, dtype=torch.float64, device="cuda")
    w_full = torch.randn(path_num, U, V, W, dtype=torch.float64, device="cuda")
    out_full = torch.zeros(BATCH, k_dim, W, dtype=torch.float64, device="cuda")

    i_offset = 0
    k_offset = 0
    pid = 0

    torch.cuda.synchronize()
    start_total = time.perf_counter() *1000

    for cg in cg_list:
        torch.cuda.synchronize()
        start = time.perf_counter() *1000

        dim = cg.shape[0]
        a = a_full[:, i_offset:i_offset+dim, :]
        b = b_full
        w = w_full[pid, :, :, :] 
        
        out_full[:, k_offset:k_offset+dim,:] = torch.einsum("ijk,biu,bjv,uvw->bkw", cg, a, b, w)

        i_offset += dim
        k_offset += dim
        pid += 1

        torch.cuda.synchronize()
        end = time.perf_counter() *1000
        t = (end - start)
        print(f"path:{pid} einsum cost: {t:.4f} ms")

    torch.cuda.synchronize()
    end_total = time.perf_counter() *1000
    t = (end_total - start_total)
    print(f"full path tensor product einsum cost: {t:.4f} ms")


sparse_einsum = load(
    name="sparse_einsum",
    sources=["sparse_einsum/baseline.cu"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'],
)

sparse_einsum_bwd = load(
    name="sparse_einsum_bwd",
    sources=["sparse_einsum/baseline_bwd.cu"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'],
)

def test_one_path_tp_bwd():
    dim = 7
    I, K = dim, dim
    J = 1

    B = BATCH
    
    cg_val = 0.03227486121839514
    #cg_tensor = torch.randn(I, J, K, dtype=torch.float64, device="cuda")
    cg_tensor = cg_val * torch.eye(I, K, dtype=torch.float64, device='cuda').unsqueeze(1)

    grad_out = torch.randn(B, W, K, dtype=torch.float64, device="cuda")

    cg_indices = [
                   [0, 0, 0],
                   [1, 0, 1],
                   [2, 0, 2],
                   [3, 0, 3],
                   [4, 0, 4],
                   [5, 0, 5],
                   [6, 0, 6]
                 ]

    cg_values = [cg_val] * dim
    
    cg_indices_tensor = torch.tensor(cg_indices, dtype=torch.int, device='cuda').reshape(dim, 3)
    cg_values_tensor = torch.tensor(cg_values, dtype=torch.float64, device='cuda')
     
    for i in range(0, 5):
        my_out = torch.zeros(BATCH, I, J, W, dtype=torch.float64, device="cuda")
        cuda_out = torch.zeros(BATCH, I, J, W, dtype=torch.float64, device="cuda")

        torch.cuda.synchronize()
        start_total = time.perf_counter() *1000 
        ref = torch.einsum("bwk,ijk->bijw", grad_out, cg_tensor)
        torch.cuda.synchronize()
        end_total = time.perf_counter() *1000
        t = (end_total - start_total)
        print(f"einsum one path tensor product cost: {t:.4f} ms")
       
        start_total = time.perf_counter() *1000
        sparse_einsum_bwd.backward(grad_out.contiguous(), cg_indices_tensor.contiguous(), cg_values_tensor.contiguous(), cuda_out.contiguous())

        torch.cuda.synchronize()
        end_total = time.perf_counter() *1000
        t = (end_total - start_total)
        print(f"my impl one path tensor product cost: {t:.4f} ms")

        error = (cuda_out - ref).abs().max().item()
        print("cuda out Max error:", error)



if __name__ == "__main__":
    test_one_path_tp_bwd()
