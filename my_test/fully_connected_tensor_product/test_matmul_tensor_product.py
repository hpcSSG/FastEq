import torch
import os, time
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.manual_seed(42)

BATCH, U, V, W = 5888, 96, 10, 96
UV = U * V

a = torch.randn(BATCH, U, dtype=torch.float64, device="cuda")
b = torch.randn(BATCH, V, dtype=torch.float64, device="cuda")
w = torch.randn(U, V, W, dtype=torch.float64, device="cuda")
tp_out = torch.zeros(BATCH, U, V, dtype=torch.float64, device="cuda")


sparse_einsum = torch.utils.cpp_extension.load(
    name="spmm",
    sources=["/workspace/FastcuEq/cuequivariance_ops_torch/_ext/cuda/sparse_einsum.cu"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'],
)

sparse_einsum_bwd = torch.utils.cpp_extension.load(
    name="spmm_bwd",
    sources=["/workspace/FastcuEq/cuequivariance_ops_torch/_ext/cuda/sparse_einsum_bwd.cu"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'],
)

def test_one_path_tp():
    dim = 1
    I, K = dim, dim
    J = 1

    B = BATCH
    
    cg_val = 0.03227486121839514
    #cg_tensor = torch.randn(I, J, K, dtype=torch.float64, device="cuda")
    cg_tensor = cg_val * torch.eye(I, K, dtype=torch.float64, device='cuda').unsqueeze(1)

    a = torch.randn(BATCH, I, U, dtype=torch.float64, device="cuda")
    #b = torch.randn(BATCH, J, V, dtype=torch.float64, device="cuda")
    
    indices = torch.randint(low=0, high=V, size=(B, J), device="cuda")
    # 转为 one-hot，得到 [B, J, V]
    embedding = torch.nn.functional.one_hot(indices, num_classes=V).to(torch.float64)
    b = embedding.reshape(B, J, V)

    w = torch.randn(U, V, W, dtype=torch.float64, device="cuda")

    cg_indices = [ [i, 0, i] for i in range(dim)]

    cg_values = [cg_val] * dim
    
    cg_indices_tensor = torch.tensor(cg_indices, dtype=torch.int, device='cuda').reshape(dim, 3)
    cg_values_tensor = torch.tensor(cg_values, dtype=torch.float64, device='cuda')
     
    for i in range(0, 5):
        my_out = torch.zeros(BATCH, K, W, dtype=torch.float64, device="cuda")

        torch.cuda.synchronize()
        start_total = time.perf_counter() *1000 
        ref = torch.einsum("ijk,biu,bjv,uvw->bkw", cg_tensor, a, b, w)
        torch.cuda.synchronize()
        end_total = time.perf_counter() *1000
        t = (end_total - start_total)
        print(f"einsum one path tensor product cost: {t:.4f} ms")
        
        # Schedule 3: "biu,bjv,uvw->bijw"
        start_total = time.perf_counter() *1000
        #bjuw = torch.einsum("bjv,uvw->bjuw", b, w)
        
        nnz_idx = b.argmax(dim=-1)              # [B, J]
        # 对 W 的第 2 维（V 维）做“批量索引”，得到 [U, B, J, W]，再换轴
        bjuw = w[:, nnz_idx, :].permute(1, 2, 0, 3).contiguous()   # [B, J, U, W]

        bijw = torch.einsum("biu,bjuw->bijw", a, bjuw)
        
        #cuda_out = sparse_einsum.forward(bijw.contiguous(), cg_indices_tensor.contiguous(), cg_values_tensor.contiguous())
        cuda_out = sparse_einsum.forward(bijw, cg_indices_tensor, cg_values_tensor)

        torch.cuda.synchronize()
        end_total = time.perf_counter() *1000
        t = (end_total - start_total)
        print(f"my impl one path tensor product cost: {t:.4f} ms")

        error = (cuda_out - ref).abs().max().item()
        print("cuda out Max error:", error)




def test_multi_path_tp():
    dim_list = [1, 3, 5, 7]
    dim_sum = sum(dim_list)
    path_num = len(dim_list)
    
    B = BATCH
    cg_val = 0.03227486121839514

    I_list = []
    J_list = []
    K_list = []

    a_all = torch.randn(BATCH, dim_sum, U, dtype=torch.float64, device="cuda")
    b_all = torch.randn(BATCH, 1, V, dtype=torch.float64, device="cuda")
    w_all = torch.randn(path_num, U, V, W, dtype=torch.float64, device="cuda")
    cuda_out_all = torch.zeros(BATCH, dim_sum, W, dtype=torch.float64, device="cuda")

    cg_indices_list = []
    cg_values_list = []

    cg_eyes_list = []

    for dim in dim_list:
      I_list.append(dim)
      K_list.append(dim)
      J_list.append(1)
      
      cg_indices = [[i, 0, i] for i in range(dim)]
      cg_indices_tensor = torch.tensor(cg_indices, dtype=torch.int, device='cuda').reshape(dim, 3)
      cg_indices_list.append(cg_indices_tensor)

      cg_values = [cg_val] * dim
      cg_values_tensor = torch.tensor(cg_values, dtype=torch.float64, device='cuda')
      cg_values_list.append(cg_values_tensor)

      cg_eye = cg_val * torch.eye(dim, dim, dtype=torch.float64, device='cuda').unsqueeze(1)
      cg_eyes_list.append(cg_eye)

    retry = 5
    for retry_id in range(0, retry):
        dim_offset = 0
        total_einsum_time = 0
        total_cuda_time = 0
        
        for dim_idx in range(0, len(dim_list)):
            dim = dim_list[dim_idx]
            cg_eye = cg_eyes_list[dim_idx]
            a = a_all[:, dim_offset:dim_offset+dim, :]
            b = b_all
            w = w_all[dim_idx, :, :, :].reshape(U, V, W)

            I = I_list[dim_idx]
            J = J_list[dim_idx]
            K = K_list[dim_idx]

            torch.cuda.synchronize()
            start_total = time.perf_counter() *1000

            ref = torch.einsum("ijk,biu,bjv,uvw->bkw", cg_eye, a, b, w)
        
            torch.cuda.synchronize()
            end_total = time.perf_counter() *1000
            t = (end_total - start_total)
            print(f"einsum path {dim_idx} tensor product cost: {t:.4f} ms")
            if retry_id == retry - 1:
                total_einsum_time += t

            
            torch.cuda.synchronize()
            start_total = time.perf_counter() *1000

            bjuw = torch.einsum("bjv,uvw->bjuw", b, w).reshape(B, U, J * W)

            a_reshaped = a.reshape(B, I, U)
            bijw = torch.matmul(a_reshaped, bjuw)
            bijw_reshaped = bijw.reshape(B, I, J, W)
            cuda_out = cuda_out_all[:, dim_offset:dim_offset+dim, :]

            sparse_einsum.forward(bijw_reshaped.contiguous(), cg_indices_tensor.contiguous(), cg_values_tensor.contiguous(), cuda_out.contiguous())

            torch.cuda.synchronize()

            error = (cuda_out - ref).abs().max().item()
            print("cuda out Max error:", error)

            end_total = time.perf_counter() *1000
            t = (end_total - start_total)
            print(f"my impl one path tensor product cost: {t:.4f} ms")
            if retry_id == retry - 1:
                total_cuda_time += t
            
            dim_offset += dim
        print(f"einsum all paths tensor product cost: {total_einsum_time:.4f} ms")
        print(f"cuda all paths tensor product cost: {total_cuda_time:.4f} ms")



if __name__ == "__main__":
    #test_multi_path_tp()
    test_one_path_tp()
