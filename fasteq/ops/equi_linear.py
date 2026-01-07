import torch
import os, math, time
from typing import List
import fasteq.cuda 

class _FastEquiLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, descriptor, math_dtype=torch.float64):
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
        
        dtype = w.dtype
        B, _ = x.shape
        u = list(descriptor.get_dims('u'))[0]
        v = list(descriptor.get_dims('v'))[0]
        x = x.view(B, -1, u).contiguous()
        w = w.view(num_paths, u, v).contiguous()
        
        w = w.to(torch.float64)
        x = x.to(torch.float64)

        #torch.cuda.synchronize()
        #start_time = time.perf_counter() * 1000

        out = torch.ops.equi_linear.fused_gemm(x, w, I_list, cg_val).view(B, -1)

        #torch.cuda.synchronize()
        #end_time = time.perf_counter() * 1000
        #execution_time_ms = end_time - start_time
        #print(f"<< fasteq equi-linear forward cost: {execution_time_ms:.3f} ms >>")

        out = out.to(dtype)

        ctx.save_for_backward(w, x, out)
        ctx.B = B
        ctx.I_list = I_list
        ctx.I_total = I_total
        ctx.cg_val = cg_val
        ctx.u = u
        return out

    @staticmethod
    def backward(ctx, grad_out):

        #torch.cuda.synchronize()
        #start_time = time.perf_counter() * 1000

        w, x, output = ctx.saved_tensors
        wt = w.transpose(1, 2).contiguous()
        math_dtype = grad_out.dtype
        grad_out = grad_out.view(ctx.B, ctx.I_total, ctx.u).contiguous()

        grad_out = grad_out.to(torch.float64)
        wt = wt.to(torch.float64)
        
        grad_x = torch.ops.equi_linear.fused_gemm(grad_out, wt, ctx.I_list, ctx.cg_val).view(ctx.B, -1)

        #torch.cuda.synchronize()
        #end_time = time.perf_counter() * 1000
        #execution_time_ms = end_time - start_time
        #print(f"<< fasteq equi-linear backward cost: {execution_time_ms:.3f} ms >>")

        grad_x = grad_x.to(math_dtype)

        return None, grad_x, None, None

def fast_equi_linear(descriptor, w, x, math_dtype=torch.float64):
    return _FastEquiLinearFn.apply(w, x, descriptor, math_dtype)