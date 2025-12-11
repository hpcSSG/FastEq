import torch
import os, math, time
from typing import List


def prod(numbers: List[int]):
    """
    This method is a workaround for script() not recognizing math.prod()
    """
    if torch.jit.is_scripting():
        product = 1
        for num in numbers:
            product *= num
        return product
    else:
        return math.prod(numbers)

# Set environment variables for CUDA
so_path = os.path.join(os.path.dirname(__file__), "_kernels/cuda/build/bin/", "libfasteq.so")
torch.ops.load_library(so_path)


def load_kernel(name: str):
    if name not in _loaded_kernels:
        raise ValueError(f"Unknown kernel name: {name}")
    return _loaded_kernels[name]

def make_FastFullyConnectedTensorProductFunction():
    class FastFullyConnectedTensorProductFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, a, b, descriptor, cg_indices, cg_values, math_dtype):
            
            #print(f"w.shape:{w.shape}, message.shape:{a.shape}, node_attrs.shape:{b.shape}")
            #print(f"cg_indices:{cg_indices}, cg_values:{cg_values}")

            inputs = [w, a, b]
            num_inputs = len(inputs)
            slices = [ope.segment_slices() for ope in descriptor.operands]
             
            outputs = []
            #bjuw_list = []
            if num_inputs > 0 and descriptor.num_paths > 0:

                slices = [ope.segment_slices() for ope in descriptor.operands]

                for path_idx, path in enumerate(descriptor.paths):
                    segments = []
                    for oid in range(num_inputs):
                        seg_shape = descriptor.get_segment_shape(oid, path)
                        inp = inputs[oid][..., slices[oid][path.indices[oid]]]
                        if len(seg_shape) > 0:
                            inp = inp.reshape(inputs[oid].shape[:-1] + seg_shape)
                        else:
                            inp = inp.reshape(inputs[oid].shape[:-1])
                        segments.append(inp.to(dtype=math_dtype))
                    
                    '''
                    _, U, V, W = segments[0].shape
                    B, I, U = segments[1].shape
                    B, J, V = segments[2].shape
                    w_seg = segments[0].reshape(U, V, W)
                    a_seg = segments[1]
                    b_seg = segments[2] 
                    '''

                    w_seg = segments[0]
                    a_seg = segments[1]
                    b_seg = segments[2]              

                    '''
                    # replace out = torch.einsum(formula, c_tensor, *segments)
                    # for fctp, einsum formula=ijk,Zuvw,Ziu,Zjv->Zkw, segments[0].shape=torch.Size([1, 96, 10, 96]), segments[1].shape=torch.Size([736, 7, 96]), segments[2].shape=torch.Size([736, 1, 10])

                    torch.cuda.synchronize()
                    start_time = time.perf_counter() * 1000

                    nnz_idx = b_seg.argmax(dim=-1).to(torch.int32)              # [B, J]                 
                    wT = w_seg.permute(1, 0, 2)
                    bjuw = wT[nnz_idx]

                    bjuw_list.append(bjuw)
                    bijw = torch.einsum("biu,bjuw->bijw", a_seg, bjuw)
                    # out = torch.einsum("bijw,ijk->bkw", bijw, c_tensor)
                    out = torch.ops.fctp_spmm_fwd.forward(bijw.contiguous(), cg_indices[path_idx].contiguous(), cg_values[path_idx].contiguous())
                    
                    torch.cuda.synchronize()
                    end_time = time.perf_counter() * 1000
                    execution_time_ms = end_time - start_time
                    print(f"einsum1 + einsum2 + einsum3 baseline cost: {execution_time_ms}")


                    torch.cuda.synchronize()
                    start_time = time.perf_counter() * 1000
                    K = I
                    bijw = torch.ops.fctp_fused2.forward(a_seg.contiguous(), b_seg.contiguous(), w_seg.contiguous())
                    out = torch.ops.fctp_spmm_fwd.forward(bijw.contiguous(), cg_indices[path_idx].contiguous(), cg_values[path_idx].contiguous())
                    torch.cuda.synchronize()
                    end_time = time.perf_counter() * 1000
                    execution_time_ms = end_time - start_time
                    print(f"einsum1 + einsum2 + einsum3 cuda cost: {execution_time_ms}")
                    '''
                    
                    out = torch.ops.fctp_fused3_fwd.forward(a_seg.contiguous(), b_seg.contiguous(), w_seg.contiguous(), cg_indices[path_idx].contiguous(), cg_values[path_idx].contiguous())
                    #print("max abs diff:", (out_ref - out).abs().max().item())
                    
                    seg_shape = descriptor.get_segment_shape(-1, path)
                    outputs += [
                        out.reshape(out.shape[: out.ndim - len(seg_shape)] + (prod(seg_shape),))
                    ]
                
                    if len(outputs) == 0:
                        raise NotImplementedError("No FX implementation for empty paths")

                    def _sum(tensors, *, shape=None, like=None):
                        if len(tensors) == 0:
                            return like.new_zeros(shape)
                        out = tensors[0]
                        for t in tensors[1:]:
                            out = torch.add(out, t)
                        return out

                    batch_shape = outputs[0].shape[:-1]

                    segment_lengths = [
                        prod(descriptor.operands[-1][i]) for i in range(descriptor.operands[-1].num_segments)
                    ]

                    final_output = torch.cat(
                        [
                            _sum(
                                [
                                    out
                                    for out, path in zip(outputs, descriptor.paths)
                                    if path.indices[-1] == i
                                ],
                                shape=batch_shape + (prod(descriptor.operands[-1][i]),),
                                like=outputs[0],
                            )
                            for i in range(descriptor.operands[-1].num_segments)
                        ],
                        dim=-1,
                    )
            else:
                raise NotImplementedError(
                    "No FX implementation for empty paths and non-empty inputs"
                )

            # 保存中间量，用于 backward
            ctx.save_for_backward(w, a, b, *outputs)
            ctx.descriptor = descriptor
            ctx.cg_indices = cg_indices
            ctx.cg_values = cg_values
            #ctx.c_tensors = c_tensors
            #ctx.bjuw_list = bjuw_list
            ctx.segment_lengths = segment_lengths
            ctx.math_dtype = math_dtype

            return final_output

        @staticmethod
        def backward(ctx, grad_out):
            """
            Backward: 将 grad_out 拆分到 segment，再回传到每条路径的中间张量
            """

            w, a, b, *outputs = ctx.saved_tensors
            descriptor = ctx.descriptor
            #c_tensor_list = ctx.c_tensors
            segment_lengths = ctx.segment_lengths
            math_dtype = ctx.math_dtype
            #bjuw_list = ctx.bjuw_list

            grad_a = torch.zeros_like(a)
            # 1. 拆分 grad_out 按 segment_lengths
            grad_segments = torch.split(grad_out, segment_lengths, dim=-1)

            # 2. 遍历路径，将对应 segment 的梯度反向回传
            for path_idx, path in enumerate(descriptor.paths):
                seg_idx = path.indices[-1]  # 对应 segment id
                grad_out_seg = grad_segments[seg_idx]  # dL/d(segment_out)

                # 取 forward 中 bijw、bjuw、w_seg 等中间量
                out = outputs[path_idx]
                slices = [ope.segment_slices() for ope in descriptor.operands]
                segments = []
                for oid, inp in enumerate([w, a, b]):
                    seg_shape = descriptor.get_segment_shape(oid, path)
                    seg_inp = inp[..., slices[oid][path.indices[oid]]]
                    seg_inp = seg_inp.reshape(inp.shape[:-1] + seg_shape).to(dtype=math_dtype)
                    segments.append(seg_inp)

                w_seg = segments[0].reshape(segments[0].shape[1:])
                a_seg = segments[1]
                b_seg = segments[2]
                K = len(ctx.cg_indices[path_idx])
                grad_out_seg = grad_out_seg.reshape(grad_out_seg.shape[0], K, -1)

                '''

                #print(f"b.shape={b_seg.shape}, w.shape={w_seg.shape}, grad_out.shape={grad_out_seg.shape}")
                # ======== 逐步 einsum 的 backward ========
                # grad_a_seg = torch.einsum("ijk,bjv,uvw,bkw->biu", c_tensor_list[path_idx], b_seg, w_seg, grad_out_seg)

                #grad_bijw = torch.einsum("bkw,ijk->bijw", grad_out_seg, c_tensor_list[path_idx])
                grad_bijw = torch.ops.fctp_spmm_bwd.backward(grad_out_seg.contiguous(),
                                                    ctx.cg_indices[path_idx].contiguous(), 
                                                    ctx.cg_values[path_idx].contiguous()
                                                    )
                
                grad_a_seg = torch.einsum("bijw,bjuw->biu", grad_bijw, bjuw_list[path_idx])
                '''

                grad_a_seg = torch.ops.fctp_fused3_bwd.backward(b_seg.contiguous(), w_seg.contiguous(), grad_out_seg.contiguous(), ctx.cg_indices[path_idx].contiguous(), ctx.cg_values[path_idx].contiguous())

                # 累加到总梯度
                grad_a[..., slices[1][path.indices[1]]] += grad_a_seg.reshape(a[..., slices[1][path.indices[1]]].shape)

            return None, grad_a, None, None, None, None, None  # grad_w, grad_b, descriptor, c_tensor_list, math_dtype 不需要梯度
    return FastFullyConnectedTensorProductFunction


def make_FastFullyConnectedTensorProductPathFused(
        cg_i_all, cg_j_all, cg_k_all, cg_val_all, 
        nnz_per_path, K_per_path, path_offset, U, V, W, K_total
    ):
    class FastFullyConnectedTensorProductPathFused(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, x, y):
            
            torch.cuda.synchronize()
            start_time = time.perf_counter() * 1000
            
            output = torch.ops.fctp_fused_multipath_fwd.forward(w, x, y, 
                    cg_i_all, cg_j_all, cg_k_all, cg_val_all,
                    nnz_per_path, K_per_path, path_offset, U, V, W, K_total)
            
            ctx.save_for_backward(w, x, y)


            torch.cuda.synchronize()
            end_time = time.perf_counter() * 1000
            execution_time_ms = end_time - start_time
            print(f"<< fasteq fctp forward cost: {execution_time_ms:.3f} ms >>")

            return output
        
        @staticmethod
        def backward(ctx, grad_out):
            w, x, y = ctx.saved_tensors
            
            torch.cuda.synchronize()
            start_time = time.perf_counter() * 1000
            
            grad_x = torch.ops.fctp_fused_multipath_bwd.backward(grad_out, w, x, y, 
                    cg_i_all, cg_j_all, cg_k_all, cg_val_all,
                    nnz_per_path, K_per_path, path_offset, U, V, W, K_total)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter() * 1000
            execution_time_ms = end_time - start_time
            print(f"<< fasteq fctp backward cost: {execution_time_ms:.3f} ms >>")


            return None, grad_x, None  #  descriptor 不需要梯度
    return FastFullyConnectedTensorProductPathFused


def make_FastEquiLinearFunction():
    class FastEquiLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, x, descriptor, math_dtype=torch.float64):
            
            torch.cuda.synchronize()
            start_time = time.perf_counter() * 1000
            
            num_paths = len(descriptor.paths)
            # descriptor.operands[1] coresponse tensor x ((1, 96), (3, 96), (5, 96), (7, 96))
            I_list = [segment[0] for segment in descriptor.operands[1]]
            I_total = sum(I_list)
            cg_val = descriptor.paths[0].coefficients
            all_equal = True
            for _, path in enumerate(descriptor.paths):
                if cg_val != path.coefficients:
                    all_equal = False
            if not all_equal:
                raise ValueError(f"coefficients value is different, causes accuracy problems")
            
            B, iu = x.shape
            _, puv = w.shape
            u = int(iu / I_total)
            v = int(puv / num_paths / u)
            x = x.view(B, I_total, u).contiguous()
            w = w.view(num_paths, u, v).contiguous()

            my_out = torch.ops.equi_linear.fused_gemm(x, w, I_list, cg_val).view(B, iu)

            ctx.save_for_backward(w, x, my_out)
            ctx.descriptor = descriptor
            ctx.B = B
            ctx.u = u
            ctx.v = v 
            ctx.I_total = I_total
            ctx.num_paths = num_paths
            ctx.I_list = I_list
            ctx.cg_val = cg_val

            torch.cuda.synchronize()
            end_time = time.perf_counter() * 1000
            execution_time_ms = end_time - start_time
            print(f"<< fasteq equi-linear forward cost: {execution_time_ms:.3f} ms >>")


            return my_out
        
        @staticmethod
        def backward(ctx, grad_out):

            torch.cuda.synchronize()
            start_time = time.perf_counter() * 1000

            w, x, output = ctx.saved_tensors
            wt = w.transpose(1, 2).contiguous() 
            grad_out = grad_out.view(ctx.B, ctx.I_total, ctx.u).contiguous()
            grad_x = torch.ops.equi_linear.fused_gemm(grad_out, wt, ctx.I_list, ctx.cg_val).view(ctx.B, -1)

            torch.cuda.synchronize()
            end_time = time.perf_counter() * 1000
            execution_time_ms = end_time - start_time
            print(f"<< fasteq equi-linear backward cost: {execution_time_ms:.3f} ms >>")


            return None, grad_x, None, None
    
    return FastEquiLinearFunction


def make_FastChannelWiseTensorProductFunction():
    class FastChannelWiseTensorProductFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, x, y):
            """
            cg系数矩阵在 channel_wise 这里为单位矩阵，理论上可省略
            """
            # w:[B, 4 * U], x:[B, U], y:[B, dim_sum=16], outputs:[B, U*dim_sum=96*16]
            output = torch.ops.cwtp_fwd.comm(x.contiguous(), y.contiguous(), w.contiguous())
            ctx.save_for_backward(w, x, y)
            #ctx.b_buf = b_buf
            return output
        
        @staticmethod
        def backward(ctx, grad_out):
            w, x, y = ctx.saved_tensors
            grad_x, grad_y, grad_w = torch.ops.cwtp_bwd.backward(grad_out.contiguous(), x.contiguous(), y.contiguous(), w.contiguous(), ctx.b_buf.detach())
            return grad_w, grad_x, grad_y  #  descriptor 不需要梯度
    return FastChannelWiseTensorProductFunction

def make_FastSymmetricTensorContractionFunction():
    
    class FastSymmetricTensorContractionFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x1, x0, i0, coeffs_tensor, paths_tensor, path_lens_tensor, num_out_segments):

            torch.cuda.synchronize()
            start_time = time.perf_counter() * 1000

            x0_g = x0[i0]
            out = torch.ops.stc_fwd.forward(
                x1.contiguous(),
                x0_g.contiguous(),
                coeffs_tensor.contiguous(),
                paths_tensor.contiguous(),
                path_lens_tensor.contiguous(),
                num_out_segments
            )

            torch.cuda.synchronize()
            end_time = time.perf_counter() * 1000
            execution_time_ms = end_time - start_time
            print(f"<< fasteq stc forward cost: {execution_time_ms:.3f} ms >>")
            
            ctx.save_for_backward(x1, x0_g, coeffs_tensor, paths_tensor, path_lens_tensor)
            ctx.num_out_segments = num_out_segments

            return out

        @staticmethod
        def backward(ctx, grad_out):

            torch.cuda.synchronize()
            start_time = time.perf_counter() * 1000

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

            torch.cuda.synchronize()
            end_time = time.perf_counter() * 1000
            execution_time_ms = end_time - start_time
            print(f"<< fasteq stc backward cost: {execution_time_ms:.3f} ms >>")

            return grad_x1, None, None, None, None, None, None
    
    return FastSymmetricTensorContractionFunction

def make_FastFusedMessagePassing():
    class FusedMPFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, node_feats, edge_attrs, tp_weights, sender,
                    receiver, dim_list, offs):
            
            torch.cuda.synchronize()
            start_time = time.perf_counter() * 1000
            
            out, start_idx, end_idx = torch.ops.fused_mp_fwd.forward(node_feats, edge_attrs, tp_weights,
                                    sender, receiver, dim_list, offs, False)
            ctx.save_for_backward(node_feats, edge_attrs, tp_weights, sender,
                                    receiver, start_idx, end_idx, dim_list, offs)

            torch.cuda.synchronize()
            end_time = time.perf_counter() * 1000
            execution_time_ms = end_time - start_time
            print(f"<< fasteq mptp forward cost: {execution_time_ms:.3f} ms >>")
            return out

        @staticmethod
        def backward(ctx, grad_out_nodes):
            node_feats, edge_attrs, tp_weights, \
            sender, receiver, start_idx, end_idx, \
            dim_list, offs = ctx.saved_tensors

            torch.cuda.synchronize()
            start_time = time.perf_counter() * 1000

            grad_node_feats, grad_edge_attrs, grad_tp_weights = torch.ops.fused_mp_bwd.backward(
                grad_out_nodes.contiguous(),
                node_feats, edge_attrs, tp_weights,
                sender, receiver, start_idx, end_idx,
                dim_list, offs,
            )


            torch.cuda.synchronize()
            end_time = time.perf_counter() * 1000
            execution_time_ms = end_time - start_time
            print(f"<< fasteq mptp backward cost: {execution_time_ms:.3f} ms >>")

            # 对应 forward 的后面几个输入没有梯度的返回 None
            return (grad_node_feats,
                    grad_edge_attrs,
                    grad_tp_weights,
                    None,  # sender
                    None,  # receiver
                    None,  # start_idx
                    None,  # end_idx
                    None,  # dim_list
                    None)  # offs
        
    return FusedMPFunction

