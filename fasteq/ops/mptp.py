import torch
import os, math, time
from typing import List
import fasteq.cuda 

class FusedMPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tp_weights, node_feats, edge_attrs, sender,
                receiver, meta):
        
        #torch.cuda.synchronize()
        #start_time = time.perf_counter() * 1000
        
        '''
        out, start_idx, end_idx = torch.ops.fused_mp_fwd.forward(node_feats, edge_attrs, tp_weights,
                                sender, receiver, dim_list, offs, False)
        ctx.save_for_backward(node_feats, edge_attrs, tp_weights, sender,
                                receiver, start_idx, end_idx, dim_list, offs)
        '''

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

        U, V, K_TOTAL = meta["U"], meta["V"], meta["K_TOTAL"]
        
        output, row_ptr_s = torch.ops.mptp_fwd.forward(
            tp_weights, node_feats, edge_attrs, 
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
            sender.to(torch.int32),
            receiver.to(torch.int32),
            U, V, K_TOTAL
        )
        
        ctx.save_for_backward(node_feats, edge_attrs, tp_weights, sender, receiver, row_ptr_s)
        ctx.meta = meta

        #torch.cuda.synchronize()
        #end_time = time.perf_counter() * 1000
        #execution_time_ms = end_time - start_time
        #print(f"<< fasteq mptp forward cost: {execution_time_ms:.3f} ms >>")
        return output

    @staticmethod
    def backward(ctx, grad_out_nodes):
        node_feats, edge_attrs, tp_weights, \
        sender, receiver, row_ptr_s = ctx.saved_tensors

        meta = ctx.meta

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
        num_paths = path_indices.shape[0]

        U, V, K_TOTAL = meta["U"], meta["V"], meta["K_TOTAL"]

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000
        
        # To be implemented: backward logic for fused message passing
        grad_tp_weights, grad_node_feats, grad_edge_attrs  = torch.ops.mptp_bwd.backward(
            grad_out_nodes, tp_weights, node_feats, edge_attrs, row_ptr_s, receiver.to(torch.int32),
            c_all,
            #path_indices,
            #i_dims, j_dims, 
            #k_dims,
            #c_offsets,
            uv_seg_offsets,
            iu_seg_offsets,
            jv_seg_offsets,
            kv_k_offsets,
            c_offsets,
            U, V, K_TOTAL,
            num_paths,
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq mptp backward cost: {execution_time_ms:.3f} ms >>")
        return (grad_tp_weights,
                grad_node_feats,
                grad_edge_attrs,
                None,  # sender
                None,  # receiver
                None,) # meta 

def fast_mptp(tp_weights, node_feats, edge_attrs,  sender,
                    receiver, meta):
    return FusedMPFunction.apply(tp_weights, node_feats, edge_attrs, sender,
                                 receiver, meta)