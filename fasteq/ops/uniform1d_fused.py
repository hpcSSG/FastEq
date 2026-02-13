import torch
import os, math, time
from typing import List

    
class FastUniform1dFusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, y, src_idx, b_list, cls_offsets, meta):

        i_list = meta["i_list"].to(torch.int32)
        j_list = meta["j_list"].to(torch.int32)
        k_list = meta["k_list"].to(torch.int32)
        coeff_list = meta["coeff_list"]
        v_offsets = meta["v_offsets"].to(torch.int32)
        out_seg_num = meta["out_seg_num"]
        w_seg_num = meta["w_seg_num"]
        x_seg_num = meta["x_seg_num"]
        y_seg_num = meta["y_seg_num"]
        u_dim = meta["u_dim"]

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        w = w.view(-1, w_seg_num, u_dim)
        x = x.view(-1, x_seg_num, u_dim)
        y = y.view(-1, y_seg_num, 1)

        src_idx = src_idx.to(torch.int32)
        b_list = b_list.to(torch.int32)
        cls_offsets = cls_offsets.to(torch.int32)

        out = torch.ops.u1d_fused_fwd.forward(
            w, x, y, 
            src_idx, b_list, cls_offsets, 
            i_list, j_list, k_list, 
            coeff_list, v_offsets, out_seg_num
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq uniform1d fused forward cost: {execution_time_ms:.3f} ms >>")

        ctx.save_for_backward(w, x, y)
        ctx.src_idx = src_idx
        ctx.b_list = b_list
        ctx.cls_offsets = cls_offsets
        ctx.i_list = i_list
        ctx.j_list = j_list
        ctx.k_list = k_list
        ctx.coeff_list = coeff_list
        ctx.v_offsets = v_offsets
        ctx.out_seg_num = out_seg_num
        ctx.w_seg_num = w_seg_num
        ctx.x_seg_num = x_seg_num
        ctx.y_seg_num = y_seg_num
        ctx.u_dim = u_dim

        return out

    @staticmethod
    def backward(ctx, grad_out):

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        w, x, y = ctx.saved_tensors

        grad_out = grad_out.view(-1, ctx.out_seg_num, ctx.u_dim)
        w = w.view(-1, ctx.w_seg_num, ctx.u_dim)
        x = x.view(-1, ctx.x_seg_num, ctx.u_dim)
        y = y.view(-1, ctx.y_seg_num, 1)

        grad_w, grad_x, grad_y = torch.ops.u1d_fused_fwd.forward(
            grad_out, w, x, y, 
            ctx.src_idx, ctx.b_list, ctx.cls_offsets, 
            ctx.i_list, ctx.j_list, ctx.k_list, 
            ctx.coeff_list, ctx.v_offsets,
            ctx.w_seg_num, ctx.x_seg_num, ctx.y_seg_num, ctx.out_seg_num, ctx.u_dim
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq uniform1d backward cost: {execution_time_ms:.3f} ms >>")

        return grad_w, grad_x, grad_y, None

def fast_uniform1d_fused(w, x, y,  src_idx, b_list, cls_offsets, meta):
    return FastUniform1dFusedFunction.apply(
        w, x, y,  src_idx, b_list, cls_offsets, meta
    )
