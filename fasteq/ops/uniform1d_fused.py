import torch
import os, math, time
from typing import List

from collections import defaultdict, OrderedDict
import numpy as np
from .code_gen import generate_code_uniform1d_fwd, generate_code_uniform1d_bwd
from .uniform1d_bwd_schedule import build_backward_schedule_from_lists, emit_backward_cuda_from_schedule, generate_full_uniform1d_bwd_split_cuda
from ..tilelang.uniform1d import stp_edge_parallel_kernel_tl

def pack_paths32(i_list: torch.Tensor,
                 j_list: torch.Tensor,
                 k_list: torch.Tensor,
                 coeff_list: torch.Tensor) -> torch.Tensor:
    assert i_list.dtype == torch.int32 and j_list.dtype == torch.int32 and k_list.dtype == torch.int32
    assert coeff_list.dtype in (torch.float32, torch.float64)

    P = i_list.numel()
    device = coeff_list.device

    i_list = i_list.contiguous()
    j_list = j_list.contiguous()
    k_list = k_list.contiguous()
    coeff_list = coeff_list.contiguous()

    packed = torch.zeros((P, 32), dtype=torch.uint8, device=device)

    packed[:, 0:4]  = i_list.view(torch.uint8).reshape(P, 4)
    packed[:, 4:8]  = j_list.view(torch.uint8).reshape(P, 4)
    packed[:, 8:12] = k_list.view(torch.uint8).reshape(P, 4)
    # 12:16 pad0 = 0

    if coeff_list.dtype == torch.float32:
        packed[:, 16:20] = coeff_list.view(torch.uint8).reshape(P, 4)
        # 20:32 padding = 0
    else:
        packed[:, 16:24] = coeff_list.view(torch.uint8).reshape(P, 8)
        # 24:32 padding = 0

    return packed



class FastUniform1dFusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, y, src_idx, dst_idx, b_list, cls_offsets, meta):

        i_list = meta["i_list"].to(torch.int32)
        j_list = meta["j_list"].to(torch.int32)
        k_list = meta["k_list"].to(torch.int32)
        v_list = meta["v_list"].to(torch.int32)
        coeff_list = meta["coeff_list"]
        v_offsets = meta["v_offsets"].to(torch.int32)
        out_seg_num = meta["out_seg_num"]
        w_seg_num = meta["w_seg_num"]
        x_seg_num = meta["x_seg_num"]
        y_seg_num = meta["y_seg_num"]
        u_dim = meta["u_dim"]

        w = w.view(-1, w_seg_num, u_dim)
        x = x.view(-1, x_seg_num, u_dim)
        y = y.view(-1, y_seg_num, 1)


        src_idx = src_idx.to(torch.int32)
        dst_idx = dst_idx.to(torch.int32)
        b_list = b_list.to(torch.int32)
        cls_offsets = cls_offsets.to(torch.int32)

        P = i_list.numel()
        #print(f"Uniform1d Path num P={P}")
        
        

        packed = pack_paths32(i_list, j_list, k_list, coeff_list)


        """ # 编译 kernel
        tl_kernel = stp_edge_parallel_kernel_tl(
            B=w.shape[0],
            S=x.shape[0],
            Iw=w_seg_num,
            Ix=x_seg_num,
            Ky=y_seg_num,
            V=out_seg_num,
            U=u_dim,
            P=P,
            WARPS_PER_BLOCK=1,
            dtype="float32",
        )

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        # tilelang调用
        tl_ref = tl_kernel(
            w,
            x,
            y.view(-1, y_seg_num),
            src_idx,
            dst_idx,
            b_list,
            i_list,
            j_list,
            k_list,
            v_list,
            coeff_list,
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"[[ tilelang uniform1d fused path={P} forward cost: {execution_time_ms:.3f} ms ]]") """
        
        generate_code_uniform1d_fwd(i_list, j_list, k_list, v_list, coeff_list, u_dim)
        print(f"generate_code_uniform1d_fwd called, P={P}, u_dim={u_dim}")


        '''
        out = torch.ops.u1d_fused_fwd.forward_np(
            w, x, y, 
            src_idx, b_list, cls_offsets,
            i_list, j_list, k_list, coeff_list,
            packed, v_offsets, out_seg_num
        )
        '''

        '''
        out = torch.ops.u1d_fused_fwd.forward(
            w, x, y, 
            src_idx, b_list, cls_offsets, 
            i_list, j_list, k_list, 
            coeff_list, v_offsets, out_seg_num
        )
        '''

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        if P == 22 and u_dim == 32:
            out = torch.ops.uniform1d_codegen_path22_u32_fwd_codegen.run(
                w, x, y, 
                src_idx, dst_idx, b_list, out_seg_num
            )
        elif P == 777 and u_dim == 32:
            out = torch.ops.uniform1d_codegen_path777_u32_fwd_codegen.run(
                w, x, y, 
                src_idx, dst_idx, b_list, out_seg_num
            )
        elif P == 1490 and u_dim == 32:
            out = torch.ops.uniform1d_codegen_path1490_u32_fwd_codegen.run(
                w, x, y, 
                src_idx, dst_idx, b_list, out_seg_num
            )
        elif P == 1554 and u_dim == 32:
            out = torch.ops.uniform1d_codegen_path1554_u32_fwd_codegen.run(
                w, x, y, 
                src_idx, dst_idx, b_list, out_seg_num
            )
        
        elif P == 16 and u_dim == 128:
            out = torch.ops.uniform1d_codegen_path16_u128_fwd_codegen.run(
                w, x, y,
                src_idx, dst_idx, b_list, out_seg_num
            )

        else:
            out = torch.ops.u1d_fused_fwd.forward(
                w, x, y, 
                src_idx, b_list, cls_offsets, 
                i_list, j_list, k_list, 
                coeff_list, v_offsets, out_seg_num
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq uniform1d fused forward cost: {execution_time_ms:.3f} ms >>")

        ctx.save_for_backward(w, x, y)
        ctx.src_idx = src_idx
        ctx.dst_idx = dst_idx
        ctx.b_list = b_list
        ctx.cls_offsets = cls_offsets
        ctx.i_list = i_list
        ctx.j_list = j_list
        ctx.k_list = k_list
        ctx.v_list = v_list
        ctx.path_packed = packed
        ctx.coeff_list = coeff_list
        ctx.v_offsets = v_offsets
        ctx.out_seg_num = out_seg_num
        ctx.w_seg_num = w_seg_num
        ctx.x_seg_num = x_seg_num
        ctx.y_seg_num = y_seg_num
        ctx.u_dim = u_dim
        ctx.P = P

        return out

    @staticmethod
    def backward(ctx, grad_out):

        w, x, y = ctx.saved_tensors

        grad_out = grad_out.view(-1, ctx.out_seg_num, ctx.u_dim)
        w = w.view(-1, ctx.w_seg_num, ctx.u_dim)
        x = x.view(-1, ctx.x_seg_num, ctx.u_dim)
        y = y.view(-1, ctx.y_seg_num, 1)

        

        """ 
        i_list_cpu = ctx.i_list.detach().cpu().tolist()
        j_list_cpu = ctx.j_list.detach().cpu().tolist()
        k_list_cpu = ctx.k_list.detach().cpu().tolist()
        v_list_cpu = ctx.v_list.detach().cpu().tolist()
        coeff_list_cpu = ctx.coeff_list.detach().cpu().tolist()
        
        if ctx.P > 512:
            generate_full_uniform1d_bwd_split_cuda(
                i_list=i_list_cpu, j_list=j_list_cpu, k_list=k_list_cpu, v_list=v_list_cpu, coeff_list=coeff_list_cpu,
                bundle_name=f"uniform1d_split_regalloc_bwd_u{ctx.u_dim}_p{ctx.P}"
            )
        else:
            sched = build_backward_schedule_from_lists(
                i_list=i_list_cpu, j_list=j_list_cpu, k_list=k_list_cpu, v_list=v_list_cpu, coeff_list=coeff_list_cpu,
                U_dim=ctx.u_dim
            )

            code = emit_backward_cuda_from_schedule(
                sched,
                kernel_name=f"uniform1d_combine_u{ctx.u_dim}_p{ctx.P}_bwd",
                scalar_t="double",
            ) 
        """
       
        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        """ grad_w, grad_x, grad_y = torch.ops.u1d_fused_bwd.backward_ep(
            grad_out, w, x, y, 
            ctx.src_idx, ctx.dst_idx, ctx.b_list,
            ctx.i_list, ctx.j_list, ctx.k_list, ctx.v_list, ctx.coeff_list,
        ) """

        """ grad_w, grad_x, grad_y = torch.ops.uniform1d_codegen.backward(
            grad_out, w, x, y, 
            ctx.src_idx, ctx.dst_idx, ctx.b_list, ctx.cls_offsets, 
            ctx.i_list, ctx.j_list, ctx.k_list, ctx.v_list, ctx.coeff_list,
            ctx.path_packed, ctx.v_offsets, ctx.out_seg_num
        ) """

        if ctx.P == 22 and ctx.u_dim == 32:
            grad_w, grad_x, grad_y = torch.ops.uniform1d_combine_u32_p22_bwd_codegen.run(
                grad_out, w, x, y, 
                ctx.src_idx, ctx.dst_idx, ctx.b_list, ctx.out_seg_num
            )
        elif ctx.P == 777 and ctx.u_dim == 32:
            grad_w, grad_x, grad_y = torch.ops.uniform1d_split_regalloc_bwd_u32_p777_codegen.run(
                grad_out, w, x, y, ctx.src_idx, ctx.dst_idx, ctx.b_list,
                ctx.w_seg_num, ctx.x_seg_num, ctx.y_seg_num, ctx.out_seg_num
            )
        elif ctx.P == 1490 and ctx.u_dim == 32:
            grad_w, grad_x, grad_y = torch.ops.uniform1d_split_regalloc_bwd_u32_p1490_codegen.run(
                grad_out, w, x, y, ctx.src_idx, ctx.dst_idx, ctx.b_list,
                ctx.w_seg_num, ctx.x_seg_num, ctx.y_seg_num, ctx.out_seg_num
            )
        elif ctx.P == 1554 and ctx.u_dim == 32:
            grad_w, grad_x, grad_y = torch.ops.uniform1d_split_regalloc_bwd_u32_p1554_codegen.run(
                grad_out, w, x, y, ctx.src_idx, ctx.dst_idx, ctx.b_list,
                ctx.w_seg_num, ctx.x_seg_num, ctx.y_seg_num, ctx.out_seg_num
            ) 
        
        elif ctx.P == 16 and ctx.u_dim == 128:
            grad_w, grad_x, grad_y = torch.ops.uniform1d_combine_u128_p16_bwd_codegen.run(
                grad_out, w, x, y, 
                ctx.src_idx, ctx.dst_idx, ctx.b_list, ctx.out_seg_num
            )
                
            """ grad_w = grad_w.view(-1, ctx.w_seg_num * ctx.u_dim)
            grad_x = grad_x.view(-1, ctx.x_seg_num * ctx.u_dim)
            grad_y = grad_y.view(-1, ctx.y_seg_num)

            ref_w = ref_w.view(-1, ctx.w_seg_num * ctx.u_dim)
            ref_x = ref_x.view(-1, ctx.x_seg_num * ctx.u_dim)
            ref_y = ref_y.view(-1, ctx.y_seg_num)

            print(f"grad_w.shape:{grad_w.shape}, ref_w.shape:{ref_w.shape}")
            print(f"grad_x.shape:{grad_x.shape}, ref_x.shape:{ref_x.shape}")
            print(f"grad_y.shape:{grad_y.shape}, ref_y.shape:{ref_y.shape}")

            print(f"grad_w:{grad_w}, ref_w:{ref_w}")
            print(f"grad_x:{grad_x}, ref_x:{ref_x}")
            print(f"grad_y:{grad_y}, ref_y:{ref_y}") """
            

        else:
            grad_w, grad_x, grad_y = torch.ops.u1d_fused_bwd.backward(
                grad_out, w, x, y, 
                ctx.src_idx, ctx.b_list, ctx.cls_offsets, 
                ctx.i_list, ctx.j_list, ctx.k_list, 
                ctx.coeff_list, ctx.v_offsets,
                ctx.w_seg_num, ctx.x_seg_num, ctx.y_seg_num, ctx.out_seg_num, ctx.u_dim
            )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        execution_time_ms = end_time - start_time
        print(f"<< fasteq uniform1d path:{ctx.P} backward cost: {execution_time_ms:.3f} ms >>")

        grad_w = grad_w.view(-1, ctx.w_seg_num * ctx.u_dim)
        grad_x = grad_x.view(-1, ctx.x_seg_num * ctx.u_dim)
        grad_y = grad_y.view(-1, ctx.y_seg_num)

        

        return grad_w, grad_x, grad_y, None, None, None, None, None

def fast_uniform1d_fused(w, x, y,  src_idx, dst_idx, b_list, cls_offsets, meta):
    return FastUniform1dFusedFunction.apply(
        w, x, y, src_idx, dst_idx, b_list, cls_offsets, meta
    )