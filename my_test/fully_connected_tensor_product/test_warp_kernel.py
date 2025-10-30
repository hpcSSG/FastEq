import torch
import os, time
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.manual_seed(42)

BATCH, U, V, W = 5152, 96, 10, 96
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


tp_opt_kernel = load(
    name="tensor_product_opt",
    sources=["tp_opt.cu"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'],
)

def test_tensor_product_opt():
    tp_opt_out = torch.zeros(BATCH, U, V, dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter() *1000
    tp_opt_kernel.launch_tp_kernel(a, b, tp_opt_out)
    torch.cuda.synchronize()
    end = time.perf_counter() *1000
    t = (end - start)
    print(f"cuda tensor product optimized: {t:.4f} ms")


def check_result_wo_cg():
    tp_out = tp_out.contiguous()
    tp_opt_kernel.launch_tp_kernel(a, b, tp_out)
    tp_out = tp_out.reshape(BATCH, UV)
    wr = w.reshape(UV, W)
    out = tp_out @ wr

    ref = torch.einsum("bu,bv,uvw->bw", a, b, w)
    error = (out - ref).abs().max().item()
    print("Max error:", error)


def test_einsum_wo_cg():
    torch.cuda.synchronize()
    start = time.perf_counter() * 1000
    outer = torch.einsum("bu, bv, uvw -> bw", a, b, w)
    torch.cuda.synchronize()
    end = time.perf_counter() *1000
    t = (end - start)
    print(f"einsum: {t:.4f} ms")


def test_outer_gemm():
    start = time.perf_counter() * 1000
    outer = torch.zeros(BATCH, U, V, dtype=torch.float64, device="cuda")
    tp_opt_kernel.launch_tp_kernel(a, b, outer)
    #outer = torch.einsum("bu,bv->buv", a, b).reshape(BATCH, UV)
    torch.cuda.synchronize()
    end = time.perf_counter() *1000
    t = (end - start)
    print(f"einsum outer: {t:.4f} ms")
    wr = w.reshape(UV, W)
    ref = outer @ wr
    torch.cuda.synchronize()
    end = time.perf_counter() *1000
    t = (end - start)
    print(f"gemm: {t:.4f} ms")


def test_one_path_tp():
    dim = 7
    I, K = dim, dim
    J = 1

    B = BATCH
    
    cg_val = 0.03227486121839514
    #cg_tensor = torch.randn(I, J, K, dtype=torch.float64, device="cuda")
    cg_tensor = cg_val * torch.eye(I, K, dtype=torch.float64, device='cuda').unsqueeze(1)

    a = torch.randn(BATCH, I, U, dtype=torch.float64, device="cuda")
    b = torch.randn(BATCH, J, V, dtype=torch.float64, device="cuda")
    w = torch.randn(U, V, W, dtype=torch.float64, device="cuda")
    
    for i in range(0, 5):
        my_out = torch.zeros(BATCH, K, W, dtype=torch.float64, device="cuda")

        torch.cuda.synchronize()
        start_total = time.perf_counter() *1000 
        ref = torch.einsum("ijk,biu,bjv,uvw->bkw", cg_tensor, a, b, w)
        torch.cuda.synchronize()
        end_total = time.perf_counter() *1000
        t = (end_total - start_total)
        print(f"einsum one path tensor product cost: {t:.4f} ms")
        
        # Schedule 1: "ijk,biu,bjv,uvw->bkw"
        #bkuv = torch.einsum("ijk,biu,bjv->bkuv", cg_tensor, a, b)
        #out = torch.einsum("bkuv,uvw->bkw", bkuv, w)

        # Schedule 2: "ijk,biu,bjv,uvw->bkw"
        #bijw = torch.einsum("biu,bjv,uvw->bijw", a, b, w)
        #out = torch.einsum("bijw, ijk", bijw, cg_tensor)
        
        # ==================== Subschedule =============

        # Schedule 1: "biu,bjv,uvw->bijw"
        #bivw = torch.einsum("biu,uvw->bivw", a, w)
        #bijw = torch.einsum("bjv,bivw->bijw", b, bivw)
        
        # Schedule 2: "biu,bjv,uvw->bijw"
        #bjuw_ref = torch.einsum("bjv,uvw->bjuw", b, w)
        #bijw_ref = torch.einsum("biu,bjuw->bijw", a, bjuw_ref)

        # Schedule 3: "biu,bjv,uvw->bijw"

        start_total = time.perf_counter() *1000

        # Replace einsum("bjv,vuw->bjuw", b, w)
        # 使用 matmul有精度问题
        #b_reshaped = b.reshape(B * J, V)
        #w_reshaped = w.reshape(V, U * W)
        #bjuw = torch.matmul(b_reshaped, w_reshaped).reshape(B, J, U, W)
        bjuw = torch.einsum("bjv,uvw->bjuw", b, w).reshape(B, U, J * W)

        # Replace einsum("biu,bjuw->bijw", a, bjuw)
        a_reshaped = a.reshape(B, I, U)
        bijw = torch.matmul(a_reshaped, bjuw)
        #bijw = torch.einsum("biu,bjuw->bijw", a_reshaped, bjuw)
        
        
        # Replace einsum("bijw,ijk -> bwk", bijw, cg_tensor)
        bijw_reshaped = bijw.reshape(B, I, J, W)
        # sparse gemm 
        for idx in range(dim):
            i, j, k = idx, 0, idx
            my_out[:, k, :] = cg_tensor[i, j, k] * bijw_reshaped[:, i, j, :]

        #out = torch.einsum("bijw,ijk -> bwk", bijw, cg_tensor)
        
        torch.cuda.synchronize()
        end_total = time.perf_counter() *1000
        t = (end_total - start_total)
        print(f"my impl one path tensor product cost: {t:.4f} ms")

        error = (my_out - ref).abs().max().item()
        print("out Max error:", error)



if __name__ == "__main__":
    #test_one_path_tp()
    test_multi_path_tp()
