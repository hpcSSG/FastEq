import torch
import os, math, time
from typing import List

class _FastEquiLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, descriptor):
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
        
        B, _ = x.shape
        u = list(descriptor.get_dims('u'))[0]
        v = list(descriptor.get_dims('v'))[0]
        x = x.view(B, -1, u).contiguous()
        w = w.view(num_paths, u, v).contiguous()

        #torch.cuda.synchronize()
        #start_time = time.perf_counter() * 1000

        out = torch.ops.equi_linear.forward(x, w, I_list, cg_val).view(B, -1)

        #torch.cuda.synchronize()
        #end_time = time.perf_counter() * 1000
        #execution_time_ms = end_time - start_time
        #print(f"<< fasteq equi-linear forward cost: {execution_time_ms:.3f} ms >>")

        ctx.save_for_backward(w)
        ctx.B = B
        ctx.I_list = I_list
        ctx.I_total = I_total
        ctx.cg_val = cg_val
        ctx.u = u
        return out

    @staticmethod
    def backward(ctx, grad_out):

        # torch.cuda.synchronize()
        # start_time = time.perf_counter() * 1000

        w = ctx.saved_tensors
        wt = w.transpose(1, 2).contiguous()
        grad_out = grad_out.view(ctx.B, ctx.I_total, ctx.u).contiguous()
        grad_x = torch.ops.equi_linear.backward(grad_out, wt, ctx.I_list, ctx.cg_val).view(ctx.B, -1)

        # torch.cuda.synchronize()
        # end_time = time.perf_counter() * 1000
        # execution_time_ms = end_time - start_time
        # print(f"<< fasteq equi-linear backward cost: {execution_time_ms:.3f} ms >>")

        return None, grad_x, None, None

def fast_equi_linear(descriptor, w, x):
    return _FastEquiLinearFn.apply(w, x, descriptor)
