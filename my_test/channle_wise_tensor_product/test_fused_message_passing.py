import os, time
import torch
torch.set_default_dtype(torch.float64)


from torch.utils.cpp_extension import load
torch.set_printoptions(precision=4, sci_mode=False)
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

fused_mp = load(
    name="fused_mp",
    sources=["fused_message_passing_opt.cu"],  # 路径按你的实际放置
    extra_cuda_cflags=["-O3", "--use_fast_math", '-gencode=arch=compute_90,code=sm_90', '--ptxas-options=-v', "-Xptxas --maxrregcount=128"],
    extra_cflags=["-O3"],
    verbose=False,
)

# --------------------------
# 假数据（用你的真实张量替换即可）
# --------------------------
device   = 'cuda'
nnodes   = 2208
E        = 40760
U        = 96
dim_list = [1, 3, 5, 7]
offs     = [0, 1, 4, 9]
DIM_SUM  = sum(dim_list)
paths    = len(dim_list)

node_feats = torch.randn(nnodes, U, device=device)
edge_attrs = torch.randn(E, DIM_SUM, device=device)
tp_weights = torch.randn(E, paths, U, device=device)

# 你给的 sender 是单调递增；这里示例也生成单调递增
sender = torch.arange(E, device=device) * nnodes // E
sender = torch.clamp(sender, max=nnodes-1)

# receiver 生成
receiver = torch.randint(160, 2064 + 1, (E,), device='cuda')
receiver, _ = torch.sort(receiver)

# --------------------------
# baseline：conv_tp + scatter_sum
# --------------------------
def forward_ref(x, y, w, dim_list, offs):
    # x: [E,U], y: [E,DIM_SUM], w: [E,P,U]
    B, U = x.shape
    out = x.new_zeros((B, DIM_SUM, U))
    for p, d in enumerate(dim_list):
        o = offs[p]
        base = x[:, None, :] * w[:, p:p+1, :]          # [E,1,U]
        out[:, o:o+d, :] = base * y[:, o:o+d, None]    # [E,d,U]
    return out

def baseline_conv_scatter(node_feats, edge_attrs, tp_weights, sender, receiver):
    x = node_feats.index_select(0, sender)                       # [E,U]
    out_edges = forward_ref(x, edge_attrs, tp_weights, dim_list, offs)  # [E,DIM_SUM,U]
    out_nodes = node_feats.new_zeros((nnodes, DIM_SUM, U))
    out_nodes.index_add_(0, receiver, out_edges)
    return out_nodes

# --------------------------
# fused_basic：仍然一次性构造 x，但避免落地 out_edges
# --------------------------
def fused_conv_scatter(node_feats, edge_attrs, tp_weights, sender, receiver, tile_u=32):
    E, P, U = tp_weights.shape
    out_nodes = node_feats.new_zeros((nnodes, DIM_SUM, U))
    x = node_feats.index_select(0, sender)  # [E,U]

    for p, d in enumerate(dim_list):
        o = offs[p]
        base = x * tp_weights[:, p, :]                   # [E,U]
        for u0 in range(0, U, tile_u):
            u1 = min(u0 + tile_u, U)
            btu = base[:, u0:u1]                         # [E,t]
            msg = edge_attrs[:, o:o+d].unsqueeze(-1) * btu.unsqueeze(1)  # [E,d,t]
            out_nodes[:, o:o+d, u0:u1].index_add_(0, receiver, msg)
    return out_nodes

# --------------------------
# fused_streaming_no_x：不构造 x，手动 gather（按 sender 段 + U 分片）
# 需要 sender 单调递增
# --------------------------
def compute_sender_runs(sender: torch.Tensor, nnodes: int):
    # 返回每个节点 s 的起止边界 [start,end)
    # sender 单调递增时，可 O(E) 求段界
    # start_idx[s], end_idx[s]
    E = sender.numel()
    # 找变化点
    diff = torch.ones(E, device=sender.device, dtype=torch.bool)
    diff[1:] = sender[1:] != sender[:-1]
    starts = torch.nonzero(diff, as_tuple=False).flatten()       # 段起点列表
    # 对应的发送者 id
    senders_unique = sender[starts]
    # 段终点（起点右移一位 + 末尾 E）
    ends = torch.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1]  = E

    # 组装成长度 nnodes 的 start/end，没出边的节点填相同值
    start_idx = torch.empty(nnodes, device=sender.device, dtype=torch.int32)
    end_idx   = torch.empty(nnodes, device=sender.device, dtype=torch.int32)
    start_idx.fill_(0); end_idx.fill_(0)

    start_idx[senders_unique] = starts.to(torch.int32)
    end_idx[senders_unique]   = ends.to(torch.int32)

    return start_idx, end_idx  # [nnodes], [nnodes]



def fused_streaming_no_x(node_feats, edge_attrs, tp_weights, sender, receiver, tile_u=32):
    # 先预取 sender 段界（一次性，后续多层可复用）
    out_nodes = node_feats.new_zeros((nnodes, DIM_SUM, U))
    start_idx, end_idx = compute_sender_runs(sender, nnodes)
    # 双层流式：先按 path（小 d），再按 U 分片；内部遍历每个 sender 的 run
    for p, d in enumerate(dim_list):
        o = offs[p]
        for u0 in range(0, U, tile_u):
            u1 = min(u0 + tile_u, U)
            # 遍历每个 sender 的连续段
            # 注：若 nnodes 很大且多数 sender 没出边，可先筛选非空 sender 列表以减少 for 次数
            non_empty = torch.nonzero((end_idx - start_idx) > 0, as_tuple=False).flatten()
            for s in non_empty.tolist():
                st = int(start_idx[s].item())
                ed = int(end_idx[s].item())
                # 该 sender 的所有出边范围 [st, ed)
                # 仅取 U tile，保持小临时张量
                x_s = node_feats[s, u0:u1].unsqueeze(0)                        # [1,t]
                # base for this run
                btu = x_s * tp_weights[st:ed, p, u0:u1]                        # [len,t]
                # 消息 [len,d,t]
                msg = edge_attrs[st:ed, o:o+d].unsqueeze(-1) * btu.unsqueeze(1)
                # 聚合到接收节点
                out_nodes[:, o:o+d, u0:u1].index_add_(0, receiver[st:ed], msg)
    return out_nodes


start_idx, end_idx = fused_mp.compute_sender_runs_sorted(sender, nnodes)

dim_list_tensor = torch.tensor([1,3,5,7], dtype=torch.int32, device=device)
offs_tensor     = torch.tensor([0,1,4,9], dtype=torch.int32, device=device)


# --------------------------
# 正确性 & 基准
# --------------------------
@torch.inference_mode()
def check_and_bench():
    def timeit(fn, warmup=3, iters=10):
        for _ in range(warmup):
            torch.cuda.synchronize(); _ = fn(); torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0.record()
            _ = fn()
            t1.record(); torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
        return sum(times)/len(times)

    ref    = baseline_conv_scatter(node_feats, edge_attrs, tp_weights, sender, receiver)
    fused1 = fused_conv_scatter(node_feats, edge_attrs, tp_weights, sender, receiver, tile_u=32)
    fused2 = fused_streaming_no_x(node_feats, edge_attrs, tp_weights, sender, receiver, tile_u=32)

    cuda_fused = fused_mp.forward(node_feats, edge_attrs, tp_weights, receiver.int(), start_idx, end_idx, dim_list_tensor, offs_tensor, 32, 8)

    print("max_abs_err (baseline vs fused_basic)   :", (ref - fused1).abs().max().item())
    print("max_abs_err (baseline vs streaming_no_x):", (ref - fused2).abs().max().item())
    print("max_abs_err (baseline vs cuda fused):", (ref - cuda_fused).abs().max().item())

    t_base = timeit(lambda: baseline_conv_scatter(node_feats, edge_attrs, tp_weights, sender, receiver))
    t_fb   = timeit(lambda: fused_conv_scatter(node_feats, edge_attrs, tp_weights, sender, receiver, tile_u=U))
    t_fs   = timeit(lambda: fused_streaming_no_x(node_feats, edge_attrs, tp_weights, sender, receiver, tile_u=U))
    t_cuda = timeit(lambda: fused_mp.forward(node_feats, edge_attrs, tp_weights, receiver.int(), start_idx, end_idx, dim_list_tensor, offs_tensor, 32, 8))
    t_cuda_sender_ready = timeit(lambda: fused_mp.compute_sender_runs_sorted(sender, nnodes))


    print(f"baseline:             {t_base:.3f} ms")
    print(f"fused_basic (with x): {t_fb:.3f} ms")
    print(f"fused_streaming_no_x: {t_fs:.3f} ms")
    print(f"fused_cuda: {t_cuda:.3f} ms")
    print(f"speedup fused_basic:  {t_base/t_fb:.2f}×")
    print(f"speedup streaming:    {t_base/t_fs:.2f}×")
    print(f"speedup cuda:    {t_base/t_cuda:.2f}×")
    print(f"t_cuda_sender_ready :    {t_cuda_sender_ready:.3f}ms")


# 直接调用检查/性能对比（如不需要可注释）
check_and_bench()
