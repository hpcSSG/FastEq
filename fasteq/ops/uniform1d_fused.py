import torch
import os, math, time
from typing import List

from collections import defaultdict, OrderedDict
import numpy as np
from .code_gen import generate_code_uniform1d_fwd, generate_code_uniform1d_bwd
from .uniform1d_bwd_schedule import build_backward_schedule_from_lists, emit_backward_cuda_from_schedule

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
        
        """ generate_code_uniform1d_fwd(i_list, j_list, k_list, v_list, coeff_list, u_dim)
        print(f"generate_code_uniform1d_fwd called, P={P}, u_dim={u_dim}") """

        """ generate_code_uniform1d_bwd(i_list, j_list, k_list, v_list, coeff_list, u_dim)
        print(f"generate_code_uniform1d_bwd called, P={P}, u_dim={u_dim}") """

        #print(f"b_list 10:{b_list[:10]}")
        #print(f"cls_offsets 10:{cls_offsets[:10]}")


        #print(f"max i:{torch.max(i_list)}, max j:{torch.max(j_list)}, max k:{torch.max(k_list)}, max v:{torch.max(v_list)}")
        #print(f"w shape:{w.shape}, x shape:{x.shape}, y shape:{y.shape}")

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

        if u_dim == 32 or u_dim == 224 or u_dim == 128:
            out = torch.ops.uniform1d_codegen.forward(
                w, x, y, 
                src_idx, dst_idx, b_list, cls_offsets,
                i_list, j_list, k_list, v_list, coeff_list, packed,
                v_offsets, out_seg_num
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

        i_list_cpu = ctx.i_list.detach().cpu().tolist()
        j_list_cpu = ctx.j_list.detach().cpu().tolist()
        k_list_cpu = ctx.k_list.detach().cpu().tolist()
        v_list_cpu = ctx.v_list.detach().cpu().tolist()
        coeff_list_cpu = ctx.coeff_list.detach().cpu().tolist()

        """ sched = build_backward_schedule_from_lists(
            i_list=i_list_cpu, j_list=j_list_cpu, k_list=k_list_cpu, v_list=v_list_cpu, coeff_list=coeff_list_cpu,
            U_dim=ctx.u_dim
        )

        code = emit_backward_cuda_from_schedule(
            sched,
            kernel_name=f"generated_uniform1d_u{ctx.u_dim}_P{ctx.P}_backward_kernel",
            scalar_t="double",
        )
        
        print(sched["strategy"])
        print(sched["launch_style"]) """
       
        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        if ctx.u_dim == 32 or ctx.u_dim == 224 or ctx.u_dim == 128:
            '''
            ref_w, ref_x, ref_y = torch.ops.u1d_fused_bwd.backward(
                grad_out, w, x, y, 
                ctx.src_idx, ctx.b_list, ctx.cls_offsets, 
                ctx.i_list, ctx.j_list, ctx.k_list, 
                ctx.coeff_list, ctx.v_offsets,
                ctx.w_seg_num, ctx.x_seg_num, ctx.y_seg_num, ctx.out_seg_num, ctx.u_dim
            )
            '''

            grad_w, grad_x, grad_y = torch.ops.uniform1d_codegen.backward(
                grad_out, w, x, y, 
                ctx.src_idx, ctx.dst_idx, ctx.b_list, ctx.cls_offsets, 
                ctx.i_list, ctx.j_list, ctx.k_list, ctx.v_list, ctx.coeff_list,
                ctx.path_packed, ctx.v_offsets, ctx.out_seg_num
            )

            '''
            grad_w = grad_w.view(-1, ctx.w_seg_num * ctx.u_dim)
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
            print(f"grad_y:{grad_y}, ref_y:{ref_y}")
            '''

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
        print(f"<< fasteq uniform1d backward cost: {execution_time_ms:.3f} ms >>")

        grad_w = grad_w.view(-1, ctx.w_seg_num * ctx.u_dim)
        grad_x = grad_x.view(-1, ctx.x_seg_num * ctx.u_dim)
        grad_y = grad_y.view(-1, ctx.y_seg_num)

        

        return grad_w, grad_x, grad_y, None, None, None, None, None

def fast_uniform1d_fused(w, x, y,  src_idx, dst_idx, b_list, cls_offsets, meta):
    return FastUniform1dFusedFunction.apply(
        w, x, y, src_idx, dst_idx, b_list, cls_offsets, meta
    )
