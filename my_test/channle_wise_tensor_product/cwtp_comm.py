
'''
Example
================= MACE-OFF medium ==============
op_name:tp_channel_wise, descriptor:uv,iu,jv,kuv+ijk sizes=1280,512,16,5120 num_segments=10,2,4,10 num_paths=10 i={1, 3} j={1, 3, 5, 7} k={1, 3, 5, 7} u=128 v=1

input shape:torch.Size([53900, 1280])
input shape:torch.Size([53900, 512])
input shape:torch.Size([53900, 16])


operand: Operand(ndim=2 num_segments=10 dims=0=128 1=1), size: 1280, segments:((128, 1), (128, 1), (128, 1), (128, 1), (128, 1), (128, 1), (128, 1), (128, 1), (128, 1), (128, 1))
operand: Operand(ndim=2 num_segments=2 dims=0={1, 3} 1=128), size: 512, segments:((1, 128), (3, 128))
operand: Operand(ndim=2 num_segments=4 dims=0={1, 3, 5, 7} 1=1), size: 16, segments:((1, 1), (3, 1), (5, 1), (7, 1))
operand: Operand(ndim=3 num_segments=10 dims=0={1, 3, 5, 7} 1=128 2=1), size: 5120, segments:((1, 128, 1), (1, 128, 1), (3, 128, 1), (3, 128, 1), (3, 128, 1), (5, 128, 1), (5, 128, 1), (5, 128, 1), (7, 128, 1), (7, 128, 1))
 tp_weights.shape:torch.Size([53900, 1280]), node_feats.shape:torch.Size([53900, 512]), edge_attrs.shape:torch.Size([53900, 16]), output.shape:torch.Size([53900, 5120])
op_name:tp_channel_wise, descriptor:uv,iu,jv,kuv+ijk sizes=1280,512,16,5120 num_segments=10,2,4,10 num_paths=10 i={1, 3} j={1, 3, 5, 7} k={1, 3, 5, 7} u=128 v=1


slice lens:10, slice:[slice(0, 128, None), slice(128, 256, None), slice(256, 384, None), slice(384, 512, None), slice(512, 640, None), slice(640, 768, None), slice(768, 896, None), slice(896, 1024, None), slice(1024, 1152, None), slice(1152, 1280, None)]
slice lens:2, slice:[slice(0, 128, None), slice(128, 512, None)]
slice lens:4, slice:[slice(0, 1, None), slice(1, 4, None), slice(4, 9, None), slice(9, 16, None)]
slice lens:10, slice:[slice(0, 128, None), slice(128, 256, None), slice(256, 640, None), slice(640, 1024, None), slice(1024, 1408, None), slice(1408, 2048, None), slice(2048, 2688, None), slice(2688, 3328, None), slice(3328, 4224, None), slice(4224, 5120, None)]
path 0 indices: (0, 0, 0, 0)
formula:ijk,Zuv,Ziu,Zjv->Zkuv, c_tensor shape:torch.Size([1, 1, 1]), segments[0] shape:torch.Size([53900, 128, 1]), segments[1] shape:torch.Size([53900, 1, 128]), segments[2] shape:torch.Size([53900, 1, 1]), out shape:torch.Size([53900, 1, 128, 1])
path 1 indices: (2, 0, 1, 2)
formula:ijk,Zuv,Ziu,Zjv->Zkuv, c_tensor shape:torch.Size([1, 3, 3]), segments[0] shape:torch.Size([53900, 128, 1]), segments[1] shape:torch.Size([53900, 1, 128]), segments[2] shape:torch.Size([53900, 3, 1]), out shape:torch.Size([53900, 3, 128, 1])
path 2 indices: (5, 0, 2, 5)
formula:ijk,Zuv,Ziu,Zjv->Zkuv, c_tensor shape:torch.Size([1, 5, 5]), segments[0] shape:torch.Size([53900, 128, 1]), segments[1] shape:torch.Size([53900, 1, 128]), segments[2] shape:torch.Size([53900, 5, 1]), out shape:torch.Size([53900, 5, 128, 1])
path 3 indices: (8, 0, 3, 8)
formula:ijk,Zuv,Ziu,Zjv->Zkuv, c_tensor shape:torch.Size([1, 7, 7]), segments[0] shape:torch.Size([53900, 128, 1]), segments[1] shape:torch.Size([53900, 1, 128]), segments[2] shape:torch.Size([53900, 7, 1]), out shape:torch.Size([53900, 7, 128, 1])
path 4 indices: (3, 1, 0, 3)
formula:ijk,Zuv,Ziu,Zjv->Zkuv, c_tensor shape:torch.Size([3, 1, 3]), segments[0] shape:torch.Size([53900, 128, 1]), segments[1] shape:torch.Size([53900, 3, 128]), segments[2] shape:torch.Size([53900, 1, 1]), out shape:torch.Size([53900, 3, 128, 1])
path 5 indices: (1, 1, 1, 1)
formula:ijk,Zuv,Ziu,Zjv->Zkuv, c_tensor shape:torch.Size([3, 3, 1]), segments[0] shape:torch.Size([53900, 128, 1]), segments[1] shape:torch.Size([53900, 3, 128]), segments[2] shape:torch.Size([53900, 3, 1]), out shape:torch.Size([53900, 1, 128, 1])
path 6 indices: (6, 1, 1, 6)
formula:ijk,Zuv,Ziu,Zjv->Zkuv, c_tensor shape:torch.Size([3, 3, 5]), segments[0] shape:torch.Size([53900, 128, 1]), segments[1] shape:torch.Size([53900, 3, 128]), segments[2] shape:torch.Size([53900, 3, 1]), out shape:torch.Size([53900, 5, 128, 1])
path 7 indices: (4, 1, 2, 4)
formula:ijk,Zuv,Ziu,Zjv->Zkuv, c_tensor shape:torch.Size([3, 5, 3]), segments[0] shape:torch.Size([53900, 128, 1]), segments[1] shape:torch.Size([53900, 3, 128]), segments[2] shape:torch.Size([53900, 5, 1]), out shape:torch.Size([53900, 3, 128, 1])
path 8 indices: (9, 1, 2, 9)
formula:ijk,Zuv,Ziu,Zjv->Zkuv, c_tensor shape:torch.Size([3, 5, 7]), segments[0] shape:torch.Size([53900, 128, 1]), segments[1] shape:torch.Size([53900, 3, 128]), segments[2] shape:torch.Size([53900, 5, 1]), out shape:torch.Size([53900, 7, 128, 1])
path 9 indices: (7, 1, 3, 7)
formula:ijk,Zuv,Ziu,Zjv->Zkuv, c_tensor shape:torch.Size([3, 7, 5]), segments[0] shape:torch.Size([53900, 128, 1]), segments[1] shape:torch.Size([53900, 3, 128]), segments[2] shape:torch.Size([53900, 7, 1]), out shape:torch.Size([53900, 5, 128, 1])

'''

import torch
import os, time




def infer_slices_and_meta(path_indices, c_tensors, u, v,
                          UV_TOTAL, IU_TOTAL, JV_TOTAL):
    """
    根据 path_indices + c_tensors + 输入 shape 推导：
      - uv / iu / jv 的切片
      - iu_seg_offsets / jv_seg_offsets
      - kv_k_offsets (在 K 维上的起始下标)
      - K_TOTAL
      - i_dims / j_dims / k_dims / c_offsets（用于 c_all）
    输出一个 meta dict，供 Python 和 CUDA 共用。
    """
    # path_indices: list[(uv_idx, iu_idx, jv_idx, kv_idx)]
    uv_seg_count = max(p[0] for p in path_indices) + 1
    iu_seg_count = max(p[1] for p in path_indices) + 1
    jv_seg_count = max(p[2] for p in path_indices) + 1
    kv_seg_count = max(p[3] for p in path_indices) + 1

    # --- i_dim/j_dim/k_dim/c_offset ---
    i_dims, j_dims, k_dims = [], [], []
    c_offsets = []
    offset = 0
    for c in c_tensors:
        i, j, k = c.shape
        i_dims.append(i)
        j_dims.append(j)
        k_dims.append(k)
        c_offsets.append(offset)
        offset += i * j * k

    # --- uv_slices：每个 uv segment 固定长度 u * v ---
    uv_slices = []
    start = 0
    uv_stride = u * v
    for _ in range(uv_seg_count):
        end = start + uv_stride
        uv_slices.append(slice(start, end))
        start = end
    assert start == UV_TOTAL, f"UV total {start} != {UV_TOTAL}"

    # --- iu_slices + offsets (按照 i_dim * u 拼) ---
    iu_slices = []
    iu_seg_offsets = []
    start = 0
    for s in range(iu_seg_count):
        iu_seg_offsets.append(start)
        # 找任意一个 path 的 iu_idx == s 来确定 i_dim
        idx = next(idx for idx, p in enumerate(path_indices) if p[1] == s)
        i_dim = i_dims[idx]
        length = i_dim * u
        end = start + length
        iu_slices.append(slice(start, end))
        start = end
    assert start == IU_TOTAL, f"IU total {start} != {IU_TOTAL}"

    # --- jv_slices + offsets (按照 j_dim * v 拼) ---
    jv_slices = []
    jv_seg_offsets = []
    start = 0
    for s in range(jv_seg_count):
        jv_seg_offsets.append(start)
        idx = next(idx for idx, p in enumerate(path_indices) if p[2] == s)
        j_dim = j_dims[idx]
        length = j_dim * v
        end = start + length
        jv_slices.append(slice(start, end))
        start = end
    assert start == JV_TOTAL, f"JV total {start} != {JV_TOTAL}"

    # --- K 维上的 offsets：kv_k_offsets ---
    kv_k_offsets = []
    start = 0
    for s in range(kv_seg_count):
        kv_k_offsets.append(start)
        # 一个 kv_seg 对应一个 k_dim（可能多个 path 用同一个 seg，但 k_dim 相同）
        idx = next(idx for idx, p in enumerate(path_indices) if p[3] == s)
        k_dim = k_dims[idx]
        start += k_dim
    K_TOTAL = start

    meta = {
        "uv_slices": uv_slices,
        "iu_slices": iu_slices,
        "jv_slices": jv_slices,
        "iu_seg_offsets": torch.tensor(iu_seg_offsets, dtype=torch.int32),
        "jv_seg_offsets": torch.tensor(jv_seg_offsets, dtype=torch.int32),
        "kv_k_offsets": torch.tensor(kv_k_offsets, dtype=torch.int32),
        "i_dims": torch.tensor(i_dims, dtype=torch.int32),
        "j_dims": torch.tensor(j_dims, dtype=torch.int32),
        "k_dims": torch.tensor(k_dims, dtype=torch.int32),
        "c_offsets": torch.tensor(c_offsets, dtype=torch.int32),
        "K_TOTAL": K_TOTAL,
    }
    return meta


def tp_channel_wise_einsum(x_uv, x_iu, x_jv,
                           c_tensors, path_indices, meta,
                           u, v):
    Z = x_uv.shape[0]
    K_TOTAL = meta["K_TOTAL"]
    out = x_uv.new_zeros(Z, K_TOTAL, u, v)

    uv_slices = meta["uv_slices"]
    iu_slices = meta["iu_slices"]
    jv_slices = meta["jv_slices"]
    kv_k_offsets = meta["kv_k_offsets"].tolist()

    for p, (uv_idx, iu_idx, jv_idx, kv_idx) in enumerate(path_indices):
        c = c_tensors[p]  # [i,j,k]
        i_dim, j_dim, k_dim = c.shape

        seg_uv = x_uv[:, uv_slices[uv_idx]].reshape(Z, u, v)       # [Z,u,v]
        seg_iu = x_iu[:, iu_slices[iu_idx]].reshape(Z, i_dim, u)   # [Z,i,u]
        seg_jv = x_jv[:, jv_slices[jv_idx]].reshape(Z, j_dim, v)   # [Z,j,v]

        tmp = torch.einsum("ijk,zuv,ziu,zjv->zkuv",
                           c, seg_uv, seg_iu, seg_jv)              # [Z,k,u,v]

        k_base = kv_k_offsets[kv_idx]
        out[:, k_base:k_base + k_dim, :, :] = tmp

    return out


def tp_channel_wise_noeinsum(x_uv, x_iu, x_jv,
                             c_tensors, path_indices, meta,
                             u, v):
    Z = x_uv.shape[0]
    K_TOTAL = meta["K_TOTAL"]
    out = x_uv.new_zeros(Z, K_TOTAL, u, v)

    uv_slices = meta["uv_slices"]
    iu_slices = meta["iu_slices"]
    jv_slices = meta["jv_slices"]
    kv_k_offsets = meta["kv_k_offsets"].tolist()

    for p, (uv_idx, iu_idx, jv_idx, kv_idx) in enumerate(path_indices):
        c = c_tensors[p]                   # [i,j,k]
        i_dim, j_dim, k_dim = c.shape

        seg_uv = x_uv[:, uv_slices[uv_idx]].reshape(Z, u, v)       # [Z,u,v]
        seg_iu = x_iu[:, iu_slices[iu_idx]].reshape(Z, i_dim, u)   # [Z,i,u]
        seg_jv = x_jv[:, jv_slices[jv_idx]].reshape(Z, j_dim, v)   # [Z,j,v]

        # broadcast 到 [Z,i,j,u,v,k] 再 sum
        seg_uv_b = seg_uv[:, None, None, :, :, None]   # [Z,1,1,u,v,1]
        seg_iu_b = seg_iu[:, :, None, :, None, None]   # [Z,i,1,u,1,1]
        seg_jv_b = seg_jv[:, None, :, None, :, None]   # [Z,1,j,1,v,1]
        c_b      = c[None, :, :, None, None, :]        # [1,i,j,1,1,k]

        prod = seg_uv_b * seg_iu_b * seg_jv_b          # [Z,i,j,u,v,1]
        weighted = prod * c_b                          # [Z,i,j,u,v,k]

        res = weighted.sum(dim=2).sum(dim=1)           # [Z,u,v,k]
        res = res.permute(0, 3, 1, 2).contiguous()     # [Z,k,u,v]

        k_base = kv_k_offsets[kv_idx]
        out[:, k_base:k_base + k_dim, :, :] = res

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



if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float64

    Z = 53900 # Batch, the number of atoms
    u = 128
    v = 1

    UV_TOTAL, IU_TOTAL, JV_TOTAL = 1280, 512, 16

    path_indices = [
        (0, 0, 0, 0),
        (2, 0, 1, 2),
        (5, 0, 2, 5),
        (8, 0, 3, 8),
        (3, 1, 0, 3),
        (1, 1, 1, 1),
        (6, 1, 1, 6),
        (4, 1, 2, 4),
        (9, 1, 2, 9),
        (7, 1, 3, 7),
    ]

    c_shapes = [
        (1, 1, 1),
        (1, 3, 3),
        (1, 5, 5),
        (1, 7, 7),
        (3, 1, 3),
        (3, 3, 1),
        (3, 3, 5),
        (3, 5, 3),
        (3, 5, 7),
        (3, 7, 5),
    ]

    x_uv = torch.randn(Z, 1280, device=device, dtype=dtype)
    x_iu = torch.randn(Z,  512, device=device, dtype=dtype)
    x_jv = torch.randn(Z,   16, device=device, dtype=dtype)

    c_tensors = [torch.randn(*s, device=device, dtype=dtype) for s in c_shapes]

    meta = infer_slices_and_meta(path_indices, c_tensors, u, v, UV_TOTAL, IU_TOTAL, JV_TOTAL)
    K_TOTAL = meta["K_TOTAL"]

    c_all = torch.cat([c.reshape(-1) for c in c_tensors], dim=0).contiguous()

    out_ein = tp_channel_wise_einsum(x_uv, x_iu, x_jv, c_tensors, path_indices, meta, u, v)
    out_noe = tp_channel_wise_noeinsum(x_uv, x_iu, x_jv, c_tensors, path_indices, meta, u, v)

    i_dims = meta["i_dims"].to(device)
    j_dims = meta["j_dims"].to(device)
    k_dims = meta["k_dims"].to(device)
    c_offsets = meta["c_offsets"].to(device)
    iu_seg_offsets = meta["iu_seg_offsets"].to(device)
    jv_seg_offsets = meta["jv_seg_offsets"].to(device)
    kv_k_offsets = meta["kv_k_offsets"].to(device)

    path_indices_tensor = torch.tensor(path_indices, dtype=torch.int32, device=device)

    so_path = os.path.join("../../primitives/_kernels/cuda/build/bin/", "libfasteq.so")
    torch.ops.load_library(so_path)

    out_cuda = torch.ops.cwtp_fwd.comm(
        x_uv, x_iu, x_jv,
        c_all,
        path_indices_tensor,
        i_dims, j_dims, k_dims,
        c_offsets,
        iu_seg_offsets,
        jv_seg_offsets,
        kv_k_offsets,
        u, v, K_TOTAL
    )

    print("einsum vs noeinsum allclose:",
        torch.allclose(out_ein, out_noe, atol=1e-9, rtol=1e-7),
        "max diff", (out_ein - out_noe).abs().max().item())


    print("einsum vs cuda allclose:",
        torch.allclose(out_ein, out_cuda, atol=1e-9, rtol=1e-7),
        "max diff", (out_ein - out_cuda).abs().max().item())

    def run_cuda():
        torch.ops.cwtp_fwd.comm(
            x_uv, x_iu, x_jv,
            c_all,
            path_indices_tensor,
            i_dims, j_dims, k_dims,
            c_offsets,
            iu_seg_offsets,
            jv_seg_offsets,
            kv_k_offsets,
            u, v, K_TOTAL
        )

    def run_einsum():
        tp_channel_wise_einsum(x_uv, x_iu, x_jv, c_tensors, path_indices, meta, u, v)

    benchmark(run_cuda)

    benchmark(run_einsum)



