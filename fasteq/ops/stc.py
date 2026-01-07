import torch
import os, math, time
from typing import List
import fasteq.cuda 

    
class FastSymmetricTensorContractionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x0, i0, coeffs_tensor, paths_tensor, path_lens_tensor, num_out_segments):

        #torch.cuda.synchronize()
        #start_time = time.perf_counter() * 1000

        x0_g = x0[i0]
        out = torch.ops.stc_fwd.forward(
            x1.contiguous(),
            x0_g.contiguous(),
            coeffs_tensor.contiguous(),
            paths_tensor.contiguous(),
            path_lens_tensor.contiguous(),
            num_out_segments
        )

        #torch.cuda.synchronize()
        #end_time = time.perf_counter() * 1000
        #execution_time_ms = end_time - start_time
        #print(f"<< fasteq stc forward cost: {execution_time_ms:.3f} ms >>")
        
        ctx.save_for_backward(x1, x0_g, coeffs_tensor, paths_tensor, path_lens_tensor)
        ctx.num_out_segments = num_out_segments

        return out

    @staticmethod
    def backward(ctx, grad_out):

        #torch.cuda.synchronize()
        #start_time = time.perf_counter() * 1000

        x1, x0_g, coeffs_tensor, paths_tensor, path_lens_tensor = ctx.saved_tensors
        grad_x1 = torch.ops.stc_bwd.backward(
            grad_out.contiguous(),
            x1,
            x0_g,
            coeffs_tensor,
            paths_tensor,
            path_lens_tensor,
            ctx.num_out_segments,
        )

        #torch.cuda.synchronize()
        #end_time = time.perf_counter() * 1000
        #execution_time_ms = end_time - start_time
        #print(f"<< fasteq stc backward cost: {execution_time_ms:.3f} ms >>")

        return grad_x1, None, None, None, None, None, None

def fast_stc(x1, x0, i0, coeffs_tensor, paths_tensor, path_lens_tensor, num_out_segments):
    return FastSymmetricTensorContractionFunction.apply(
        x1, x0, i0, coeffs_tensor, paths_tensor, path_lens_tensor, num_out_segments
    )