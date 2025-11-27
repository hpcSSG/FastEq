import torch
import os, time
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

fused_fctp = torch.utils.cpp_extension.load(
    name="fused_fctp",
    sources=["./fused_fctp.cu"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'],
)

import torch
import time


def build_local_cg_lists(dim_list, device):
    """
    构造与你之前例子等价的 local cg_indices / cg_values：
    每个 path:
      i_local = [0..K_p-1], j_local = 0, k_local = [0..K_p-1]
      val = 0.0323
    """
    cg_indices = []
    cg_values = []
    for K_p in dim_list:
        i = torch.arange(K_p, device=device, dtype=torch.int32)   # local i: 0..K_p-1
        j = torch.zeros_like(i)                                   # j_local=0
        k = torch.arange(K_p, device=device, dtype=torch.int32)   # local k: 0..K_p-1
        idx = torch.stack([i, j, k], dim=1)                       # [K_p, 3]
        val = torch.full((K_p,), 0.0323, device=device, dtype=torch.float32)
        cg_indices.append(idx)
        cg_values.append(val)
    print(f"cg_indices:{cg_indices}, cg_values:{cg_values}")
    return cg_indices, cg_values


def pack_cg_for_kernel_global(cg_indices_local, cg_values, dim_list, device):
    """
    把 local (i_local, j_local, k_local) + dim_list -> 统一打包为 global (i,j,k)：
      i_global = i_local + path_offset[p]
      j_global = j_local + 0           （保留接口，将来可扩展）
      k_global = k_local + path_offset[p]
    输出：
      cg_i_all, cg_j_all, cg_k_all, cg_val_all: [P, nnz_max]
      nnz_per_path, K_per_path, path_offset:   [P]
      K_total, nnz_max: 标量
    """
    P = len(cg_indices_local)
    assert P == len(dim_list)

    # K_per_path & path_offset & K_total
    K_per_path = torch.tensor(dim_list, device=device, dtype=torch.int32)
    path_offset = torch.empty(P, device=device, dtype=torch.int32)
    path_offset[0] = 0
    if P > 1:
        path_offset[1:] = torch.cumsum(K_per_path[:-1], dim=0)
    K_total = int(K_per_path.sum().item())

    # nnz 信息
    nnz_list = [ci.shape[0] for ci in cg_indices_local]
    nnz_max = max(nnz_list)
    nnz_per_path = torch.tensor(nnz_list, device=device, dtype=torch.int32)

    dtype_val = cg_values[0].dtype

    cg_i_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_j_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_k_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_val_all = torch.zeros((P, nnz_max), device=device, dtype=dtype_val)

    for p in range(P):
        ci_local = cg_indices_local[p]  # [nnz_p, 3], (i_local, j_local, k_local)
        cv       = cg_values[p]         # [nnz_p]
        nnz_p    = nnz_list[p]
        offset_p = path_offset[p].item()

        i_local = ci_local[:, 0]
        j_local = ci_local[:, 1]
        k_local = ci_local[:, 2]

        # --- local -> global ---
        i_global = i_local + offset_p
        j_global = j_local             # 这里先保持 0，将来可换成真正的 j_global
        k_global = k_local + offset_p

        cg_i_all[p, :nnz_p]   = i_global
        cg_j_all[p, :nnz_p]   = j_global
        cg_k_all[p, :nnz_p]   = k_global
        cg_val_all[p, :nnz_p] = cv

    return (
        cg_i_all,
        cg_j_all,
        cg_k_all,
        cg_val_all,
        nnz_per_path,
        K_per_path,
        path_offset,
        K_total,
        nnz_max,
    )


def baseline_fctp_multipath_global(
    a_seg, b_seg, w_all,
    cg_indices_local, cg_values,
    dim_list, path_offset
):
    """
    纯 PyTorch baseline，使用 global i/k 逻辑：
      - a_seg: [B, I_total, U]，I_total = sum(dim_list)
      - b_seg: [B, 1, V]
      - w_all: [P, U, V, W]
      - cg_indices_local: list of [nnz_p, 3] (i_local, j_local, k_local)
      - path_offset: [P]，用来把 local i/k 映射为 global i/k
    最终输出: [B, K_total, W]
    """
    device = a_seg.device
    B, I_total, U = a_seg.shape
    P, U2, V, W = w_all.shape
    assert U2 == U
    dim_sum = sum(dim_list)
    assert I_total == dim_sum, "I_total 必须等于 sum(dim_list)"

    # v* = argmax_v b[b,0,v]
    b_logits = b_seg[:, 0, :]  # [B,V]
    vstar = torch.argmax(b_logits, dim=-1)  # [B]

    K_total = sum(dim_list)
    out = torch.zeros((B, K_total, W), device=device, dtype=a_seg.dtype)

    for p in range(P):
        K_p = dim_list[p]
        nnz_p = cg_indices_local[p].shape[0]
        ci_local = cg_indices_local[p]  # [nnz_p,3]
        cv = cg_values[p].to(a_seg.dtype)  # [nnz_p]
        offset_p = path_offset[p].item()

        # local i/k
        i_local = ci_local[:, 0].long()  # [nnz_p]
        k_local = ci_local[:, 2].long()  # [nnz_p]

        # global i/k
        i_global = i_local + offset_p    # [nnz_p]
        k_global = k_local + offset_p    # [nnz_p]

        # W_p[:, v*, :] -> W_sel[b,u,w]
        W_p = w_all[p]  # [U,V,W]
        W_sel = W_p[:, vstar, :].permute(1, 0, 2).contiguous()  # [B,U,W]

        # 遍历 nnz_p（<=7）
        for e in range(nnz_p):
            ig = i_global[e].item()
            kg = k_global[e].item()
            val_e = cv[e]

            a_row = a_seg[:, ig, :]              # [B,U]
            contrib = torch.einsum("bu,buw->bw", a_row, W_sel)  # [B,W]
            out[:, kg, :] += val_e * contrib

    return out


def benchmark(fn, warmup=5, iters=20, desc="fn"):
    """
    简单的 CUDA 性能测试工具：
      - warmup 次预热
      - iters 次正式计时
    返回平均耗时（毫秒）
    """
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / iters
    print(f"{desc}: avg {avg_ms:.3f} ms over {iters} iters (warmup={warmup})")
    return avg_ms


def main():
    torch.manual_seed(0)

    assert torch.cuda.is_available(), "需要一块 CUDA GPU 来跑这个测试"

    device = torch.device("cuda")

    '''
    # 显式加载 .so（如果你没在别处 import 过）
    # 把下面路径改成你实际编译出的 .so 路径
    try:
        _ = torch.ops.fctp_fused3_fwd.forward_multipath_concat
    except (RuntimeError, AttributeError):
        torch.ops.load_library("path/to/your/libfctp_fused3_fwd.so")
    '''

    # --------- 配置参数 ---------
    BATCH, U, V, W = 5888, 96, 10, 96
    dim_list = [1, 3, 5, 7]       # 每条 path 的 K_p
    path_num = len(dim_list)
    dim_sum = sum(dim_list)       # I_total = 16

    I_total = dim_sum

    print(f"BATCH={BATCH}, I_total={I_total}, U={U}, V={V}, W={W}, paths={path_num}")

    # --------- 构造输入张量 ---------
    a_seg = torch.randn(BATCH, I_total, U, device=device, dtype=torch.float32)
    b_seg = torch.randn(BATCH, 1, V, device=device, dtype=torch.float32)
    w_all = torch.randn(path_num, U, V, W, device=device, dtype=torch.float32)

    # --------- 构造 local cg_indices / cg_values ---------
    cg_indices_local, cg_values = build_local_cg_lists(dim_list, device=device)

    # --------- 打包为 kernel 需要的 global 形式 ---------
    (
        cg_i_all,
        cg_j_all,
        cg_k_all,
        cg_val_all,
        nnz_per_path,
        K_per_path,
        path_offset,
        K_total,
        nnz_max,
    ) = pack_cg_for_kernel_global(cg_indices_local, cg_values, dim_list, device=device)

    print(f"nnz_max={nnz_max}, K_total={K_total}, path_offset={path_offset.tolist()}")

    # --------- 调用 CUDA kernel（一次）做正确性检查 ---------
    print(f"cg_i_all:{cg_i_all}, cg_j_all:{cg_j_all}, cg_k_all:{cg_k_all}")
    out_kernel = fused_fctp.fwd_opt(
        a_seg, b_seg, w_all,
        cg_i_all, cg_j_all, cg_k_all, cg_val_all,
        nnz_per_path, K_per_path, path_offset
    )  # [BATCH, K_total, W]

    print("Kernel output shape:", out_kernel.shape)

    # --------- baseline（global index 版本） ---------
    out_ref = baseline_fctp_multipath_global(
        a_seg, b_seg, w_all,
        cg_indices_local, cg_values,
        dim_list, path_offset
    )

    print("Baseline output shape:", out_ref.shape)

    # --------- 对比误差 ---------
    diff = (out_kernel - out_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max abs diff = {max_diff:.6e}")
    print(f"Mean abs diff = {mean_diff:.6e}")

    tol = 1e-4  # float32 阈值
    if max_diff < tol:
        print(f"[OK] Kernel matches baseline within tolerance {tol}")
    else:
        print(f"[WARN] Kernel diff ({max_diff}) exceeds tolerance {tol}")

    # --------- 性能测试 ---------
    print("\n==== Performance Benchmark ====")

    a_seg_det = a_seg.detach()
    b_seg_det = b_seg.detach()
    w_all_det = w_all.detach()

    def run_kernel():
        fused_fctp.fwd_opt(
            a_seg_det, b_seg_det, w_all_det,
            cg_i_all, cg_j_all, cg_k_all, cg_val_all,
            nnz_per_path, K_per_path, path_offset
        )

    def run_baseline():
        baseline_fctp_multipath_global(
            a_seg_det, b_seg_det, w_all_det,
            cg_indices_local, cg_values,
            dim_list, path_offset
        )

    warmup = 3
    iters_kernel = 20
    iters_baseline = 10

    with torch.no_grad():
        t_kernel = benchmark(run_kernel, warmup=warmup, iters=iters_kernel,
                             desc="CUDA multipath kernel")
        t_baseline = benchmark(run_baseline, warmup=warmup, iters=iters_baseline,
                               desc="PyTorch baseline")

    if t_kernel > 0:
        speedup = t_baseline / t_kernel
        print(f"Speedup (baseline / kernel) = {speedup:.2f}x")
    else:
        print("Kernel time is 0 ms? Something is wrong.")


if __name__ == "__main__":
    main()
