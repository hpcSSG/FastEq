import torch
from torch.autograd import Function


torch.ops.load_library("build/bin/libfused_gmm.so")
my_fused_gmm = torch.ops.fused_gmm


class MultiPathMatMul(Function):
    @staticmethod
    def forward(ctx, x, W, i_list, cg):
        """
        x: (B, I_total, u)
        W: (num_paths, u, v)
        i_list: python list of ints, lengths per path
        cg: scalar tensor or float (broadcastable)
        returns: out (B, I_total * v) where each path produces (B, i*v) concatenated along dim=1
        """
        device = x.device
        dtype = x.dtype

        num_paths = len(i_list)
        u = W.size(1)
        v = W.size(2)
        B = x.size(0)

        # compute forward as in your loop
        out_chunks = []
        offset = 0
        for p, i in enumerate(i_list):
            Xi = x[:, offset:offset + i, :].contiguous()   # (B, i, u)
            # reshape to (B*i, u) and multiply by W[p] (u,v) -> (B*i, v)
            Yi = Xi.view(B * i, u) @ W[p].view(u, v)       # (B*i, v)
            Yi = (cg * Yi)                                 # scale
            Yi = Yi.view(B, i * v)                         # (B, i*v)
            out_chunks.append(Yi)
            offset += i

        out = torch.cat(out_chunks, dim=1)  # (B, I_total * v)

        # save for backward
        # we need x and W and i_list and cg and shapes to compute grads
        ctx.save_for_backward(x, W, torch.tensor(i_list, dtype=torch.int64, device=device), cg)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (B, I_total * v)
        returns: gradients for (x, W, None(for i_list), None(for cg))
        """
        x, W, i_list_tensor, cg = ctx.saved_tensors
        i_list = i_list_tensor.cpu().tolist()
        device = x.device
        dtype = x.dtype

        B = x.size(0)
        num_paths = len(i_list)
        u = W.size(1)
        v = W.size(2)
        I_total = sum(i_list)

        # prepare gradients
        dX = torch.zeros_like(x)

        offset_out = 0
        out_chunks = []

        grad_out = grad_output.reshape(B, I_total, v)
        for p, i in enumerate(i_list):
            # slice grad_output for this path
            dY = grad_out[:, offset_out:offset_out + i, :].contiguous()

            # dXi = cg * dY @ W[p].T  -> (B, i, u)
            Wp = W[p].view(u, v)                                     # (u, v)
            dXi = torch.matmul(dY, Wp.t()) * cg                      # (B, i, u)
            out_chunks.append(dXi)

            offset_out += i

        dX = torch.cat(out_chunks, dim=1)
        # return gradients in same order as forward args: (x, W, i_list, cg)
        # i_list and cg are not tensors requiring grad (or we don't compute their grads) -> None
        return dX, None, None, None 


class FusedMultiPathMatMul(Function):
    @staticmethod
    def forward(ctx, x, W, i_list, cg):
        """
        x: (B, I_total, u)
        W: (num_paths, u, v)
        i_list: python list of ints, lengths per path
        cg: scalar tensor or float (broadcastable)
        returns: out (B, I_total * v) where each path produces (B, i*v) concatenated along dim=1
        """
        device = x.device
        dtype = x.dtype

        num_paths = len(i_list)
        u = W.size(1)
        v = W.size(2)
        B = x.size(0)

        out = my_fused_gmm.forward(x, W, i_list, cg).view(B, -1)

        # save for backward
        # we need x and W and i_list and cg and shapes to compute grads
        ctx.save_for_backward(x, W, torch.tensor(i_list, dtype=torch.int64, device=device), cg)
        ctx.i_list = i_list
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (B, I_total * v)
        returns: gradients for (x, W, None(for i_list), None(for cg))
        """
        x, W, i_list_tensor, cg = ctx.saved_tensors
        i_list = ctx.i_list

        B = x.size(0)
        num_paths = len(i_list)
        u = W.size(1)
        v = W.size(2)
        I_total = sum(i_list)

        dX = my_fused_gmm.forward(grad_output.reshape(B, I_total, v), W.transpose(1,2).contiguous(), i_list, cg)
        return dX, None, None, None


def multipath_matmul(x, W, i_list, cg):
    return MultiPathMatMul.apply(x, W, i_list, cg)

def fused_multipath_matmul(x, W, i_list, cg):
    return FusedMultiPathMatMul.apply(x, W, i_list, cg)

# --- correctness test ---
def test_correctness():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float64

    B = 736
    u = v = 96
    #i_list = [1, 3, 5, 7]               # 4 条路径的 i 维
    i_list = [1]
    num_paths = len(i_list)
    I_total = sum(i_list)               # 16

    # x: (B, I_total, u)
    x = torch.randn(B, I_total, u, dtype=dtype, device=device, requires_grad=True)

    # W: (num_paths, u, v)
    weight_flat = torch.randn(1, num_paths * u * v, dtype=dtype, device=device)
    W = weight_flat.view(num_paths, u, v).contiguous()
    W.requires_grad_(False)

    cg_val = 0.10206207261596577
    cg_tensor = torch.tensor(cg_val, dtype=dtype, device=device)

    # reference forward using naive loop (same as your original code)
    def ref_forward(x, W):
        offset = 0
        out = None
        for p, i in enumerate(i_list):
            Xi = x[:, offset:offset + i, :].contiguous()   # (B, i, u)
            offset += i
            Wp = W[p].view(u, v)
            Yi = (cg_tensor * (Xi.view(B * i, u) @ Wp)).view(B, -1)  # (B, i*v)
            out = Yi if out is None else torch.cat([out, Yi], dim=1)
        return out  # (B, I_total * v)
    

    # custom function
    x_ref = x.clone().detach().requires_grad_(True)
    W_ref = W.clone().detach().requires_grad_(False)
    Y_ref = ref_forward(x_ref, W_ref)
    loss_ref = (Y_ref* torch.arange(Y_ref.numel(), dtype=dtype, device=device).reshape(Y_ref.shape)).sum()
    loss_ref.backward()
    dX_ref = x_ref.grad 


    x_test = x.clone().detach().requires_grad_(True)
    W_test = W.clone().detach().requires_grad_(False)
    Y_test = multipath_matmul(x_test, W_test, i_list, cg_tensor)
    loss_test = (Y_test * torch.arange(Y_test.numel(), dtype=dtype, device=device).reshape(Y_test.shape)).sum()
    loss_test.backward()
    dX_test = x_test.grad

    x_fused = x.clone().detach().requires_grad_(True)
    W_fused = W.clone().detach().requires_grad_(False)
    Y_fused = fused_multipath_matmul(x_fused, W_fused, i_list, cg_tensor)
    loss_test = (Y_fused * torch.arange(Y_fused.numel(), dtype=dtype, device=device).reshape(Y_fused.shape)).sum()
    loss_test.backward()
    dX_fused = x_fused.grad

    print("max abs diff Y of ref and unfused:", (Y_ref - Y_test).abs().max().item())
    print("max abs diff dX of ref and unfused:", (dX_ref - dX_test).abs().max().item())
    assert torch.allclose(dX_ref, dX_test, atol=1e-10, rtol=1e-8)
    print("unfused PASSED: gradients match PyTorch autograd reference.")

    print("max abs diff Y of ref and fused:", (Y_ref - Y_fused).abs().max().item())
    print("max abs diff dX of ref and fused:", (dX_ref - dX_fused).abs().max().item())
    assert torch.allclose(dX_ref, dX_fused, atol=1e-10, rtol=1e-8)
    print("fused PASSED: gradients match PyTorch autograd reference.")

if __name__ == "__main__":
    test_correctness()

