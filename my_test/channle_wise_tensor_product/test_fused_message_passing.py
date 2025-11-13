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

fused_mp_bwd = load(
    name="fused_mp_bwd",
    sources=["fused_message_passing_bwd.cu"],  # 路径按你的实际放置
    extra_cuda_cflags=["-O3", "--use_fast_math", '-gencode=arch=compute_90,code=sm_90', '--ptxas-options=-v', "-Xptxas --maxrregcount=128"],
    extra_cflags=["-O3"],
    verbose=False,
)


class FusedMPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, node_feats, edge_attrs, tp_weights,
                receiver, start_idx, end_idx, dim_list, offs):
        # 保存反向所需变量
        ctx.save_for_backward(node_feats, edge_attrs, tp_weights,
                              receiver, start_idx, end_idx,
                              dim_list, offs)
        out = fused_mp.forward(node_feats, edge_attrs, tp_weights,
                               receiver, start_idx, end_idx,
                               dim_list, offs, 32, 8)
        return out

    @staticmethod
    def backward(ctx, grad_out_nodes):
        node_feats, edge_attrs, tp_weights, \
        receiver, start_idx, end_idx, \
        dim_list, offs = ctx.saved_tensors

        grad_node_feats, grad_edge_attrs, grad_tp_weights = fused_mp_bwd.backward(
            grad_out_nodes.contiguous(),
            node_feats, edge_attrs, tp_weights,
            receiver, start_idx, end_idx,
            dim_list, offs,
        )

        # 对应 forward 的后面几个输入没有梯度的返回 None
        return (grad_node_feats,
                grad_edge_attrs,
                grad_tp_weights,
                None,  # receiver
                None,  # start_idx
                None,  # end_idx
                None,  # dim_list
                None)  # offs


def fused_mp_cuda(node_feats, edge_attrs, tp_weights,
                  receiver, start_idx, end_idx,
                  dim_list, offs):
    return FusedMPFunction.apply(
        node_feats, edge_attrs, tp_weights,
        receiver, start_idx, end_idx,
        dim_list, offs
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

node_feats = torch.randn(nnodes, U, device=device, requires_grad=True)
edge_attrs = torch.randn(E, DIM_SUM, device=device, requires_grad=True)
tp_weights = torch.randn(E, paths, U, device=device, requires_grad=True)

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


start_idx, end_idx = fused_mp.compute_sender_runs_sorted(sender, nnodes)

dim_list_tensor = torch.tensor([1,3,5,7], dtype=torch.int32, device=device)
offs_tensor     = torch.tensor([0,1,4,9], dtype=torch.int32, device=device)


# --------------------------
# 正确性 & 基准
# --------------------------
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


    node_feats_ref = node_feats.clone().detach().requires_grad_(True)
    edge_attrs_ref = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_ref = tp_weights.clone().detach().requires_grad_(True)
    ref = baseline_conv_scatter(node_feats_ref, edge_attrs_ref, tp_weights_ref, sender, receiver)
    loss_ref = ref.sum()
    loss_ref.backward()
    grad_node_feats_ref = node_feats_ref.grad
    grad_edge_attrs_ref = edge_attrs_ref.grad
    grad_tp_weights_ref = tp_weights_ref.grad

    # fused1 = fused_conv_scatter(node_feats, edge_attrs, tp_weights, sender, receiver, tile_u=32)
    # print("Forward max_abs_err (baseline vs fused_basic)   :", (ref - fused1).abs().max().item())

    node_feats_cu = node_feats.clone().detach().requires_grad_(True)
    edge_attrs_cu = edge_attrs.clone().detach().requires_grad_(True)
    tp_weights_cu = tp_weights.clone().detach().requires_grad_(True)
    
    # cuda_fused = fused_mp_cuda(node_feats, edge_attrs, tp_weights, receiver.int(), start_idx, end_idx, dim_list_tensor, offs_tensor, 32, 8)
    
    out_cu = fused_mp_cuda(node_feats_cu, edge_attrs_cu, tp_weights_cu,
                           receiver.int(), start_idx, end_idx,
                           dim_list_tensor, offs_tensor)
    loss_cu = out_cu.sum()
    loss_cu.backward()
    grad_node_feats_cu = node_feats_cu.grad
    grad_edge_attrs_cu = edge_attrs_cu.grad
    grad_tp_weights_cu = tp_weights_cu.grad

    print("Forward max_abs_err (baseline vs cuda fused):", (ref - out_cu).abs().max().item())
    print("Check grad node_feats:")
    print("  max abs diff:", (grad_node_feats_ref - grad_node_feats_cu).abs().max().item())
    print("Check grad edge_attrs:")
    print("  max abs diff:", (grad_edge_attrs_ref - grad_edge_attrs_cu).abs().max().item())
    print("Check grad tp_weights:")
    print("  max abs diff:", (grad_tp_weights_ref - grad_tp_weights_cu).abs().max().item())


    t_base = timeit(lambda: baseline_conv_scatter(node_feats, edge_attrs, tp_weights, sender, receiver))
    t_fb   = timeit(lambda: fused_conv_scatter(node_feats, edge_attrs, tp_weights, sender, receiver, tile_u=U))
    t_cuda = timeit(lambda: fused_mp.forward(node_feats, edge_attrs, tp_weights, receiver.int(), start_idx, end_idx, dim_list_tensor, offs_tensor, 32, 8))
    t_cuda_bwd = timeit(lambda: fused_mp_bwd.backward(out_cu, node_feats, edge_attrs, tp_weights, receiver.int(), start_idx, end_idx, dim_list_tensor, offs_tensor))
    t_cuda_sender_ready = timeit(lambda: fused_mp.compute_sender_runs_sorted(sender, nnodes))


    print(f"baseline:             {t_base:.3f} ms")
    print(f"fused_basic (with x): {t_fb:.3f} ms")
    print(f"fused_cuda: {t_cuda:.3f} ms")
    print(f"fused_cuda_backward: {t_cuda_bwd:.3f} ms")
    print(f"speedup fused_basic:  {t_base/t_fb:.2f}×")
    print(f"speedup cuda:    {t_base/t_cuda:.2f}×")
    print(f"t_cuda_sender_ready :    {t_cuda_sender_ready:.3f}ms")


# 直接调用检查/性能对比（如不需要可注释）
check_and_bench()
