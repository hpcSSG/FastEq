import torch
import os, time
import ast
from torch.utils.cpp_extension import load

# 设置 CUDA 调试同步
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda")

# 参数设置
B, u = 5152, 96
num_a = num_b = num_c = 16
num_i = 13
num_paths = 94

stp_kernel = load(
    name="segmented_tensor_product_kernel",
    sources=["baseline_cueq_stp.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90"
    ]
)

stp_shm_kernel = load(
    name="segmented_tensor_product_shm_kernel",
    sources=["shm_cueq_stp.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90"
    ]
)

device = 'cuda'
x1 = torch.randn(B, num_a, u, dtype=torch.float64, device=device)
x0_g = torch.randn(B, num_i, u, dtype=torch.float64, device=device)

weight = torch.randn(u, u, dtype=torch.float64, device=device)


with open("cg_coeff_paths.txt", "r") as f:
    data = f.read()
host_paths = ast.literal_eval(data)
path_lens_tensor = torch.tensor([len(p) for p in host_paths], dtype=torch.int, device=device)
max_len = max(len(p) for p in host_paths)
padded_paths = [p + [0] * (max_len - len(p)) for p in host_paths]
paths_tensor = torch.tensor(padded_paths, dtype=torch.int, device=device)
coeffs = torch.randn(len(path_lens_tensor), dtype=torch.float64, device=device)

print(f"paths_tensor shape: {paths_tensor.shape}, coeffs shape: {coeffs.shape}")


# baseline
def baseline():
    baseline_out = torch.zeros(B, u, dtype=torch.float64, device=device)
    for path_idx in range(0, len(paths_tensor)):
        path = paths_tensor[path_idx]
        real_path_len = path_lens_tensor[path_idx]
        if real_path_len == 3:
            a = path[0]
            i = path[1]
            baseline_out += x1[:, a] * x0_g[:, i] * coeffs[path_idx]
        if real_path_len == 4:
            a = path[0]
            b = path[1]
            i = path[2]
            baseline_out += x1[:, a] * x1[:, b] * x0_g[:, i] * coeffs[path_idx]
        if real_path_len == 5:
            a = path[0]
            b = path[1]
            c = path[2]
            i = path[3]
            baseline_out += x1[:, a] * x1[:, b] * x1[:, c] * x0_g[:, i] * coeffs[path_idx]
    out = torch.matmul(baseline_out, weight)
    return baseline_out

def stp_cuda():
    output = torch.zeros(B, u, dtype=torch.float64, device=device) 
    stp_kernel.segmented_tensor_product(
        x1, x0_g, coeffs, paths_tensor, path_lens_tensor, output
    )
    return output

def stp_shm_cuda():
    output = torch.zeros(B, u, dtype=torch.float64, device=device)
    stp_shm_kernel.segmented_tensor_product(
        x1, x0_g, coeffs, paths_tensor, path_lens_tensor, output
    )
    return output

# 计时器
def benchmark(fn, label, retry=1):
    torch.cuda.synchronize()
    start = time.perf_counter() * 1000
    for _ in range(0, retry):
        out = fn()
    torch.cuda.synchronize()
    end = time.perf_counter() *1000
    return label, out, (end - start)/retry

# 执行测试
res = []
for fn, name in [(baseline, "baseline"), (stp_cuda, "stp_cuda"), (stp_shm_cuda, "shm_stp_cuda")]:
    label, out, t = benchmark(fn, name)
    res.append((label, out, t))

# 打印结果
print("\n=== 性能对比 ===")
for label, out, t in res:
    print(f"{label:<15}: {t:.4f} ms")

ref = res[0][1]
for label, out, _ in res[1:]:
    err = (out - ref).abs().max().item()
    print(f"{label:<15}: 最大误差 = {err:.2e}")
