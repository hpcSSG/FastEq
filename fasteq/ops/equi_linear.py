import torch
import os, math, time
from typing import List
import fasteq.cuda 

class _FastEquiLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, descriptor, math_dtype=torch.float64):
        # y = torch.ops.equi_linear.forward(...)
        num_paths = len(descriptor.paths)
        # descriptor.operands[1] coresponse tensor x ((1, 96), (3, 96), (5, 96), (7, 96))
        I_list = [segment[0] for segment in descriptor.operands[1]]
        I_total = sum(I_list)
        cg_val = descriptor.paths[0].coefficients
        all_equal = True
        for _, path in enumerate(descriptor.paths):
            if cg_val != path.coefficients:
                all_equal = False
        if not all_equal:
            raise ValueError(f"coefficients value is different, causes accuracy problems")
            
            B, iu = x.shape
            _, puv = w.shape
            u = int(iu / I_total)
            v = int(puv / num_paths / u)
            x = x.view(B, I_total, u).contiguous()
            w = w.view(num_paths, u, v).contiguous()

            out = torch.ops.equi_linear.fused_gemm(x, w, I_list, cg_val).view(B, iu)

            ctx.save_for_backward(w, x, out)
            ctx.I_list = I_list
            ctx.cg_val = cg_val
        return out

    @staticmethod
    def backward(ctx, grad_out):
        w, x, output = ctx.saved_tensors
        wt = w.transpose(1, 2).contiguous() 
        grad_out = grad_out.view(ctx.B, ctx.I_total, ctx.u).contiguous()
        grad_x = torch.ops.equi_linear.fused_gemm(grad_out, wt, ctx.I_list, ctx.cg_val).view(ctx.B, -1)
        return grad_w, grad_x, None, None

def fast_equi_linear(w, x, descriptor, math_dtype=torch.float64):
    return _FastEquiLinearFn.apply(w, x, descriptor, math_dtype)