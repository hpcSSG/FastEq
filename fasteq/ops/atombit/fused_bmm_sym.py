import torch
import os, math, time
from typing import List

    
class FusedBmmSymFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, geom):

        out = torch.ops.fused_bmm_sym.forward(
            h.contiguous(),
            geom.contiguous(),
        )
        
        ctx.save_for_backward(h, geom)

        return out

    @staticmethod
    def backward(ctx, grad_out):

        h, geom = ctx.saved_tensors
        grad_h, grad_geom = torch.ops.fused_bmm_sym.backward(
            h,
            geom,
            grad_out.contiguous(),
        )  
        
        return grad_h, grad_geom, None

def fused_bmm_sym(h, geom):
    return FusedBmmSymFunction.apply(
        h, geom
    )
