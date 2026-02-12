import torch
import os, math, time
from typing import List

    
class FastUniform1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, y, meta):

        i_list = meta["i_list"]
        j_list = meta["j_list"]
        k_list = meta["k_list"]
        coeff_list = meta["coeff_list"]
        v_offsets = meta["v_offsets"]
        out_seg_num = meta["out_seg_num"]
        w_seg_num = meta["w_seg_num"]
        x_seg_num = meta["x_seg_num"]
        y_seg_num = meta["y_seg_num"]

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        w = w.view(-1, w_seg_num, self.u_dim)
        x = x.view(-1, x_seg_num, self.u_dim)
        y = y.view(-1, y_seg_num, 1)

        scatter_sum_dim = x.shape[0]

        out = torch.ops.u1d_fwd.forward(w, x, y, i_list, j_list, k_list, coeff_list, v_offsets, out_seg_num)

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq uniform1d forward cost: {execution_time_ms:.3f} ms >>")
        
        ctx.save_for_backward(w, x, y)
        ctx.i_list = i_list
        ctx.j_list = j_list
        ctx.k_list = k_list
        ctx.coeff_list = coeff_list
        ctx.v_offsets = v_offsets
        ctx.out_seg_num = out_seg_num

        return out

    @staticmethod
    def backward(ctx, grad_out):

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        w, x, y = ctx.saved_tensors
        grad_w, grad_x, grad_y = torch.ops.u1d_bwd.backward(
            w, x, y, grad_out, 
            ctx.i_list, ctx.j_list, ctx.k_list, cctx.oeff_list,
            ctx.v_offsets, ctx.out_seg_num
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq uniform1d backward cost: {execution_time_ms:.3f} ms >>")

        return grad_w, grad_x, grad_y, None

def fast_uniform1d(w, x, y, meta):
    return FastUniform1dFunction.apply(
        w, x, y, meta
    )
