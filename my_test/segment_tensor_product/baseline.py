import torch
import time

# -------- 基本参数 ----------
B = 1472
u = v = 96
i_list = [1, 3, 5, 7]               # 4 条路径的 i 维
num_paths = len(i_list)
I_total = sum(i_list)               # 1+3+5+7 = 16

# -------- 构造输入 ----------
# x: [B, sum(i)*u] = [736, 1536]
torch.manual_seed(0)
x = torch.randn(B, I_total * u, dtype=torch.float64, device="cuda")

# weight: [1, 36864] = [1, num_paths * u * v] -> [num_paths, u, v]
weight_flat = torch.randn(1, num_paths * u * v, dtype=torch.float64, device="cuda")
W = weight_flat.view(num_paths, u, v).contiguous()   # W[p] ∈ [u, v]

warmup = 5

cg_val = 0.10206207261596577
cg_tensor = torch.tensor(cg_val, dtype=torch.float64, device=torch.device("cuda"))

for warm_id in range(warmup):
    offset = 0
    ys_e = []

    print(f"======= warmup:{warm_id} =========")

    for p, i in enumerate(i_list):

        Xi = x[:, offset:offset + i * u].view(B, i, u).contiguous()
        offset += i * u
        
        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        Wp = W[p].reshape(u, v)
        Yi_e = cg_tensor * Xi.reshape(B * i, u) @ Wp
        Yi_e = Yi_e.reshape(B, i, v)
        ys_e.append(Yi_e)
        
        print(f"Xi.shape:{Xi.shape}, W[p].shape:{W[p].shape}")
        #Yi_e_ref = torch.einsum(",biu,uv->biv", cg_tensor, Xi, W[p].view(u, v))
        #assert torch.allclose(Yi_e_ref, Yi_e, atol=1e-12), "GEMM 与 einsum 结果不一致！"

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = (end_time - start_time)
        print(f"========= tp serial einsum cost: {execution_time_ms:.3f} ms ========")
