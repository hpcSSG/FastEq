import torch
import os, math, time
from typing import List
import fasteq.cuda 

class FastChannelWiseTensorProductFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, y, meta, ell_meta):

        cg_i_groupk = meta["cg_i_all"]
        cg_j_groupk  = meta["cg_j_all"]
        cg_k_groupk  = meta["cg_k_all"]
        cg_val_groupk  = meta["cg_val_all"]

        nnz_per_path = meta["nnz_per_path"]
        nnz_offsets_groupk = meta["nnz_offsets"]
        nnz_k_offsets_groupk = meta["nnz_k_offsets"]
        nnz_k_counts_groupk = meta["nnz_k_counts"]

        c_all = meta["c_all"]
        i_dims = meta["i_dims"]
        j_dims = meta["j_dims"]
        k_dims = meta["k_dims"]
        c_offsets = meta["c_offsets"]
        uv_seg_offsets = meta["uv_seg_offsets"]
        iu_seg_offsets = meta["iu_seg_offsets"]
        jv_seg_offsets = meta["jv_seg_offsets"]
        kv_k_offsets = meta["kv_k_offsets"]
        path_indices = meta["path_indices_tensor"]
        U = meta["U"]
        V = meta["V"]
        K_TOTAL = meta["K_TOTAL"]

        num_paths = nnz_per_path.shape[0]

        '''
        print(f"path_indices.shape:{path_indices.shape}, k_dims.shape:{k_dims.shape}, nnz_per_path.shape:{nnz_per_path.shape} \
            nnz_offsets.shape:{nnz_offsets_groupk.shape}, iu_seg_offsets:{iu_seg_offsets.shape}, jv_seg_offsets:{jv_seg_offsets.shape} \
            kv_k_offsets.shape:{kv_k_offsets.shape}")
        
        print(f"nnz_k_offsets.shape:{nnz_k_offsets_groupk.shape}, nnz_k_counts.shape:{nnz_k_counts_groupk.shape}, \
            cg_i_groupk.shape:{cg_i_groupk.shape}, cg_j_groupk.shape:{cg_j_groupk.shape},cg_val_groupk.shape:{cg_val_groupk.shape}")
        '''
        
        ell_E, ell_base, ell_ij, ell_val = ell_meta["ell_E"], ell_meta["ell_base"], ell_meta["ell_ij"], ell_meta["ell_val"]
        meta1, meta2 = ell_meta["meta1"], ell_meta["meta2"]

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000
        if num_paths == 19:
            output = torch.ops.cwtp_fwd.forward_pp(
                w, x, y,
                c_all,
                path_indices,
                i_dims, j_dims, k_dims,
                c_offsets,
                iu_seg_offsets,
                jv_seg_offsets,
                kv_k_offsets,
                ell_E, ell_base, ell_ij, ell_val,
                U, V, K_TOTAL
            )
            
        else:
            output = torch.ops.cwtp_fwd.forward(
                w, x, y,
                c_all,
                path_indices,
                i_dims, j_dims, k_dims,
                c_offsets,
                iu_seg_offsets,
                jv_seg_offsets,
                kv_k_offsets,
                nnz_per_path,
                nnz_offsets_groupk,
                nnz_k_offsets_groupk,
                nnz_k_counts_groupk,
                cg_i_groupk,
                cg_j_groupk,
                cg_k_groupk,
                cg_val_groupk,
                #ell_E, ell_base, 
                ell_ij, ell_val,
                meta1, meta2,
                U, V, K_TOTAL
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq cwtp forward cost: {execution_time_ms:.3f} ms >>")

        ctx.save_for_backward(w, x, y)
        ctx.meta = meta
        ctx.ell_meta = ell_meta
        return output

    @staticmethod
    def backward(ctx, grad_output):

        w, x, y = ctx.saved_tensors
        meta = ctx.meta
        meta1 = ctx.ell_meta["meta1"]
        meta2 = ctx.ell_meta["meta2"]

        cg_i_groupk = meta["cg_i_all"]
        cg_j_groupk  = meta["cg_j_all"]
        cg_k_groupk  = meta["cg_k_all"]
        cg_val_groupk  = meta["cg_val_all"]
        nnz_per_path = meta["nnz_per_path"]
        nnz_offsets = meta["nnz_offsets"]
        nnz_k_offsets_groupk = meta["nnz_k_offsets"]
        nnz_k_counts_groupk = meta["nnz_k_counts"]

        c_all = meta["c_all"]
        i_dims = meta["i_dims"]
        j_dims = meta["j_dims"]
        k_dims = meta["k_dims"]
        c_offsets = meta["c_offsets"]
        uv_seg_offsets = meta["uv_seg_offsets"]
        iu_seg_offsets = meta["iu_seg_offsets"]
        jv_seg_offsets = meta["jv_seg_offsets"]
        kv_k_offsets = meta["kv_k_offsets"]
        path_indices = meta["path_indices_tensor"]

        U = meta["U"]
        V = meta["V"]
        K_TOTAL = meta["K_TOTAL"]

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        grad_w, grad_x, grad_y = torch.ops.cwtp_bwd.backward(
            grad_output, w, x, y, 
            c_all,
            path_indices,
            uv_seg_offsets,
            iu_seg_offsets,
            jv_seg_offsets,
            kv_k_offsets,
            c_offsets,
            i_dims, j_dims, k_dims, 
            K_TOTAL, U, V, 
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq cwtp backward cost: {execution_time_ms:.3f} ms >>")
        
        '''
        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        grad_w, grad_x, grad_y = torch.ops.cwtp_fwd.backward_opt(
            grad_output, w, x, y, 
            c_all,
            path_indices,
            uv_seg_offsets,
            iu_seg_offsets,
            jv_seg_offsets,
            kv_k_offsets,
            meta1, meta2,
            K_TOTAL, U, V, 
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq cwtp opt backward cost: {execution_time_ms:.3f} ms >>")
        '''

        #print(f"grad_w:{grad_w}, ref_grad_w:{ref_grad_w}")
        #print(f"grad_x:{grad_x}, ref_grad_x:{ref_grad_x}")
        #print(f"grad_y:{grad_y}, ref_grad_y:{ref_grad_y}")

        return (
            grad_w,   # w
            grad_x,   # x
            grad_y,   # y
            None,     # meta
            None,
        )

def fast_cwtp(w, x, y, meta, ell_meta):
    return FastChannelWiseTensorProductFunction.apply(w, x, y, meta, ell_meta)
