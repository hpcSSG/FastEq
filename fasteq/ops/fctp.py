import torch
import os, math, time
from typing import List
import fasteq.cuda 


class FastFullyConnectedTensorProductPathFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, y, meta):

        cg_i_all = meta["cg_i_all"]
        cg_j_all = meta["cg_j_all"]
        cg_k_all = meta["cg_k_all"]
        cg_val_all = meta["cg_val_all"]
        nnz_per_path = meta["nnz_per_path"]
        K_per_path = meta["K_per_path"]
        path_offset = meta["path_offset"]
        U, V, W, K_total = meta["U"], meta["V"], meta["W"], meta["K_total"]
        
        #torch.cuda.synchronize()
        #start_time = time.perf_counter() * 1000
        
        output = torch.ops.fctp_fused_multipath_fwd.forward(w, x, y, 
                cg_i_all, cg_j_all, cg_k_all, cg_val_all,
                nnz_per_path, K_per_path, path_offset, U, V, W, K_total)
        
        ctx.save_for_backward(w, x, y)
        ctx.meta = meta


        #torch.cuda.synchronize()
        #end_time = time.perf_counter() * 1000
        #execution_time_ms = end_time - start_time
        #print(f"<< fasteq fctp forward cost: {execution_time_ms:.3f} ms >>")

        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        w, x, y = ctx.saved_tensors

        meta = ctx.meta
        cg_i_all = meta["cg_i_all"]
        cg_j_all = meta["cg_j_all"]
        cg_k_all = meta["cg_k_all"]
        cg_val_all = meta["cg_val_all"]
        nnz_per_path = meta["nnz_per_path"]
        K_per_path = meta["K_per_path"]
        path_offset = meta["path_offset"]
        U, V, W, K_total = meta["U"], meta["V"], meta["W"], meta["K_total"]
        
        #torch.cuda.synchronize()
        #start_time = time.perf_counter() * 1000
        
        grad_x = torch.ops.fctp_fused_multipath_bwd.backward(grad_out, w, x, y, 
                cg_i_all, cg_j_all, cg_k_all, cg_val_all,
                nnz_per_path, K_per_path, path_offset, U, V, W, K_total)
        
        #torch.cuda.synchronize()
        #end_time = time.perf_counter() * 1000
        #execution_time_ms = end_time - start_time
        #print(f"<< fasteq fctp backward cost: {execution_time_ms:.3f} ms >>")


        return None, grad_x, None, None  # None for w, y, meta gradients

def fast_fctp(w, x, y, meta):
    return FastFullyConnectedTensorProductPathFused.apply(w, x, y, meta)