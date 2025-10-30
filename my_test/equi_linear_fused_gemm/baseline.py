import torch
import time
import os
from torch.utils.cpp_extension import load

# 加载编译好的共享库
#torch.ops.load_library("build/bin/libfused_gmm.so")
torch.ops.load_library("/home/malixian/repos/cuequivariance_torch/primitives/_kernels/cuda/build/bin/libfasteq.so")
my_fused_gmm = torch.ops.equi_linear.fused_gemm

# -------- 基本参数 ----------
B = 2300
u = v = 96
i_list = [1, 3, 5, 7]               # 4 条路径的 i 维
num_paths = len(i_list)
I_total = sum(i_list)               # 1+3+5+7 = 16

# -------- 构造输入 ----------
# x: [B, sum(i)*u] = [736, 1536]
torch.manual_seed(0)
x = torch.randn(B, I_total * u, dtype=torch.float64, device="cuda")
# x = torch.arange(
#     start=0.000,
#     end=0.001 * B * I_total * u,
#     step=0.001,
#     dtype=torch.float64,
#     device="cuda"
# ).reshape(B, I_total * u)

# weight: [1, 36864] = [1, num_paths * u * v] -> [num_paths, u, v]
weight_flat = torch.randn(1, num_paths * u * v, dtype=torch.float64, device="cuda")
# weight_flat = torch.arange(
#     start=0.000,
#     end=0.001 * num_paths * u * v,
#     step=0.001,
#     dtype=torch.float64,
#     device="cuda"
# )
W = weight_flat.view(num_paths, u, v).contiguous()   # W[p] ∈ [u, v]

warmup = 50

cg_val = 0.10206207261596577
cg_tensor = torch.tensor(cg_val, dtype=torch.float64, device=torch.device("cuda"))

for i in range(warmup):
    offset = 0
    ys_e = []

    torch.cuda.synchronize()
    start_time = time.perf_counter() * 1000

    for p, i in enumerate(i_list):

        Xi = x[:, offset:offset + i * u].view(B, i, u).contiguous()
        offset += i * u
        Wp = W[p].view(u, v)
        Yi_e = Xi.view(B * i, u) @ Wp
        Yi_e = cg_tensor * Yi_e
        Yi_e = Yi_e.view(B, i, v)
        ys_e.append(Yi_e)

    torch.cuda.synchronize()
    end_time = time.perf_counter() * 1000
    execution_time_ms = (end_time - start_time)
    print(f"========= tp serial einsum cost: {execution_time_ms:.3f} ms ========")
    
x = x.view(B, I_total, u).contiguous()
W = weight_flat.view(num_paths, u, v).contiguous()
    
for i in range(warmup):
    torch.cuda.synchronize()
    start_time = time.perf_counter() * 1000

    my_out = my_fused_gmm(x, W, i_list, cg_val)

    torch.cuda.synchronize()
    end_time = time.perf_counter() * 1000
    execution_time_ms = (end_time - start_time)
    print(f"========= my einsum cost: {execution_time_ms:.3f} ms ========")

assert my_out.shape[1] == I_total and my_out.shape[2] == v, "my_out 的shape不正确"

# ref
offset = 0
ref_out = None
for p, i in enumerate(i_list):
    Xi = x[:, offset:offset + i, :].view(B, i, u).contiguous()
    offset += i
    Wp = W[p].view(u, v)
    Yi_e = Xi.view(B * i, u) @ Wp
    Yi_e = cg_tensor * Yi_e
    Yi_e = Yi_e.view(B, -1)
    if p == 0:
        ref_out = Yi_e
    else:
        ref_out = torch.cat([ref_out, Yi_e], dim=1)

assert ref_out.shape[1] == I_total * v, "ref_out 的shape不正确"
ref_out = ref_out.view(B, I_total, -1)

torch.testing.assert_close(my_out, ref_out, msg=f"GEMM 与 einsum 结果不一致！\nmy: {my_out[0]}\nref:{ref_out[0]}")
print("结果正确！")
