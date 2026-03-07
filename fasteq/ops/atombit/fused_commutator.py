import torch
import os, math, time
from typing import List

    
class FusedCommutatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, Fdim):

        out = torch.ops.fused_commutator.forward(
            A.contiguous(),
            B.contiguous(),
            Fdim
        )
        
        ctx.save_for_backward(A, B)
        ctx.Fdim = Fdim

        return out

    @staticmethod
    def backward(ctx, grad_out):

        A, B = ctx.saved_tensors
        grad_A, grad_B = torch.ops.fused_commutator.backward(
            A,
            B,
            grad_out.contiguous(),
            ctx.Fdim
        )  
        
        return grad_A, grad_B, None

def fused_commutator(A, B, Fdim):
    return FusedCommutatorFunction.apply(
        A, B, Fdim
    )
