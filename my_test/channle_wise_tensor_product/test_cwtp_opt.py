import os, time
import torch
from torch.utils.cpp_extension import load
torch.set_printoptions(precision=4, sci_mode=False)


os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

mod = load(
    name="cwtp_opt",
    sources=["cwtp_opt.cu"],  # 路径按你的实际放置
    extra_cuda_cflags=["-O3", "--use_fast_math", '-gencode=arch=compute_90,code=sm_90', '--ptxas-options=-v', "-Xptxas --maxrregcount=128"],
    extra_cflags=["-O3"],
    verbose=False,
)

device = "cuda"
dtype  = torch.float64

B, U = 40760, 96           # 你的大规模用例
dim_list = [1, 3, 5, 7]
DIM_SUM = sum(dim_list)    # 16
P = len(dim_list)
offs = [0, 1, 4, 9]        # 前缀

# -------------------------------
# 参考实现（与CUDA前向数学等价）
# out_ref: [B,16,U]
# -------------------------------
def forward_ref(x, y, w):  # x:[B,U], y:[B,16], w:[B,4,U]
    B, U = x.shape
    out = x.new_zeros((B, DIM_SUM, U))
    for p, d in enumerate(dim_list):
        o = offs[p]
        # out[:, o:o+d, :] = (x[:, None, :] * w[:, p:p+1, :]) * y[:, o:o+d, None]
        base = x[:, None, :] * w[:, p:p+1, :]          # [B,1,U]
        out[:, o:o+d, :] = base * y[:, o:o+d, None]    # 广播到 [B,d,U]
    return out

# -------------------------------
# 单次大规模前向正确性检查
# -------------------------------
def test_forward_big():
    torch.manual_seed(0)
    x = torch.randn(B, U, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(B, DIM_SUM, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(B, P, U, device=device, dtype=dtype, requires_grad=True)

    with torch.no_grad():
        out_ref = forward_ref(x, y, w)                 # [B,16,U]
        out_cuda, b_buf = mod.fwd(x.contiguous(), y.contiguous(), w.contiguous())
        # 逐元素比较
        diff = (out_cuda - out_ref).abs()
        max_abs = diff.max().item()
        rel = diff / (out_ref.abs() + 1e-16)
        max_rel = rel.max().item()
        print(f"[Forward-big] max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
        assert max_abs < 1e-12 or max_rel < 1e-12, "Forward mismatch!"

# -------------------------------
# 后向正确性检查（用ref作autograd真值）
# -------------------------------
def test_backward_big():
    torch.manual_seed(1)
    x = torch.randn(B, U, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(B, DIM_SUM, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(B, P, U, device=device, dtype=dtype, requires_grad=True)

    # forward
    out_cuda, out_flat, b_buf = mod.fwd(x.contiguous(), y.contiguous(), w.contiguous())
    out_ref = forward_ref(x, y, w) 
    g = torch.randn_like(out_ref)                       # 上游梯度

    # autograd 真值
    loss = (out_ref * g).sum()
    loss.backward()
    gx_ref, gy_ref, gw_ref = x.grad.detach(), y.grad.detach(), w.grad.detach()

    # 我们的CUDA后向
    gx, gy, gw = mod.bwd(g.contiguous(), x.detach(), y.detach(), w.detach(), b_buf.detach())

    # 三路比较
    def check(name, a, b):
        diff = (a - b).abs()
        max_abs = diff.max().item()
        rel = diff / (b.abs() + 1e-16)
        max_rel = rel.max().item()
        print(f"[Backward-big][{name}], max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
        assert max_abs < 1e-12 or max_rel < 1e-12, f"{name} mismatch!"

    check("gx", gx, gx_ref)
    check("gy", gy, gy_ref)
    check("gw", gw, gw_ref)

def bench():
    torch.manual_seed(1)
    x = torch.randn(B, U, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(B, DIM_SUM, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(B, P, U, device=device, dtype=dtype, requires_grad=True)
    out_cuda, b_buf = mod.fwd(x.contiguous(), y.contiguous(), w.contiguous())
    g = torch.randn_like(out_cuda)                       # 上游梯度
    #grad_out_T = g.permute(0, 2, 1).contiguous()

    retry = 10

    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start_total = time.perf_counter() *1000
        
        #gx, gy, gw = mod.bwd(g, x.detach(), y.detach(), w.detach(), b_buf.detach())
        out_cuda, b_buf = mod.fwd(x.contiguous(), y.contiguous(), w.contiguous())

        torch.cuda.synchronize()
        end_total = time.perf_counter() *1000
        t = (end_total - start_total)
        if retry_id == retry - 1:
            print(f"cuda backward cost: {t:.4f} ms")
# -------------------------------
# 运行
# -------------------------------
if __name__ == "__main__":
    '''
    torch.cuda.synchronize()
    test_forward_big()
    torch.cuda.synchronize()
    test_backward_big()
    torch.cuda.synchronize()
    print("✅ All correctness tests passed.")
    '''
    test_forward_big()
    #test_backward_big()
    bench()
