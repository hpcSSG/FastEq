import torch
import os, math, time
from typing import List

class _FastEquiLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, descriptor):
        num_paths = len(descriptor.paths)
        if num_paths <=4:
            I_list = list(descriptor.get_dimensions_dict()["i"])
        else:
            I_list = [segment[0] for segment in descriptor.operands[1]]
        I_total = sum(I_list)
        print(f"desc:{descriptor}")
        #torch.cuda.synchronize()
        #start_time = time.perf_counter() * 1000
        cg_list = []
        cg_val = descriptor.paths[0].coefficients
        for pid, path in enumerate(descriptor.paths):
            cg_list.append(float(path.coefficients))
        
        B, _ = x.shape
        u = list(descriptor.get_dims('u'))[0]
        v = list(descriptor.get_dims('v'))[0]
        x = x.view(B, -1, u).contiguous()
        w = w.view(num_paths, u, v).contiguous()

        # MACE small
        #I_list = [1,3,5,7]
        #cg_vals = [0.10206207261596577, 0.10206207261596577, 0.10206207261596577, 0.10206207261596577]
        
        # MACE medium
        #I_list = [1, 1, 3, 3, 3, 5, 5, 5, 7, 7] 
        #cg_vals = [0.0625,0.0625,0.051031036307982884,0.051031036307982884,0.051031036307982884,0.051031036307982884,0.051031036307982884,0.051031036307982884,0.0625,0.0625]

        # MACE large
        # I_list = [1, 1, 1, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 7]
        # 1:0.03857583749052298; 3,5=0.02988071523335984; 7=0.03340765523905305
        #cg_vals = [0.03857583749052298, 0.02988071523335984, 0.03340765523905305]
        #I_total = sum(I_list)

        x = x.float()
        w = w.float()

        out = torch.ops.equi_linear.forward(x, w, I_list, cg_val)
        out =  out.view(B, -1)

        out = out.double()

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
        ctx.cg_list = cg_list
        return out

    @staticmethod
    def backward(ctx, grad_out):

        #print(f"eq-linear bwd grad_out:{grad_out.shape}")

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000
        w, = ctx.saved_tensors

        print(f"grad_out shape:{grad_out.shape}, w shape:{w.shape}")

        grad_out = grad_out.float()
        w = w.float()

        grad_out = grad_out.view(ctx.B, 16, ctx.u).contiguous() # 只支持out固定为16
        grad_x = torch.ops.equi_linear.backward(grad_out, w, ctx.I_list, ctx.cg_val).view(ctx.B, -1)

        print(f"grad_x shape after bwd:{grad_x.shape}")

        grad_x = grad_x.double()

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        #print(f"<< fasteq equi-linear backward cost: {execution_time_ms:.3f} ms >>")
        #print(f"<< gradx shape:{grad_x.shape}")
        return None, grad_x, None, None

def fast_equi_linear(descriptor, w, x):
    return _FastEquiLinearFn.apply(w, x, descriptor)
