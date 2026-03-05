import torch
import os, math, time
from typing import List

    
class FusedCommutatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, Fdim):

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        out = torch.ops.fused_commutator.forward(
            A.contiguous(),
            B.contiguous(),
            Fdim
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq fused_commutator forward cost: {execution_time_ms:.3f} ms >>")
        
        ctx.save_for_backward(A, B)
        ctx.Fdim = Fdim

        return out

    @staticmethod
    def backward(ctx, grad_out):

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        A, B = ctx.saved_tensors
        grad_A, grad_B = torch.ops.fused_commutator.backward(
            grad_out.contiguous(),
            A,
            B,
            ctx.Fdim
        )  
        
        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq fused_commutator backward cost: {execution_time_ms:.3f} ms >>")

        return grad_A, grad_B, None

def fast_stc(A, B, Fdim):
    return FusedCommutatorFunction.apply(
        A, B, Fdim
    )
