#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>


#include "./uniform1d_codegen_path777_u32_fwd.cu"
#include "./uniform1d_codegen_path777_u32_bwd.cu"
#include "./uniform1d_codegen_path1490_u32_fwd.cu"
#include "./uniform1d_codegen_path1490_u32_bwd.cu"
#include "./uniform1d_codegen_path1554_u32_fwd.cu"
#include "./uniform1d_codegen_path1554_u32_bwd.cu"
#include "./uniform1d_codegen_path215_u224_fwd.cu"
#include "./uniform1d_codegen_path215_u224_bwd.cu"


torch::Tensor uniform1d_codegen_fwd_launch(
    torch::Tensor w,          // [B,Iw,U]
    torch::Tensor x_all,      // [S,Ix,U]
    torch::Tensor y,          // [B,Ky,1]
    torch::Tensor src_idx,    // [B] int32
    torch::Tensor dst_idx,    // [B] int32
    torch::Tensor b_list,     // [B] int32
    torch::Tensor cls_offsets,// [S+1] int32
    torch::Tensor i_list,     // [P] int32
    torch::Tensor j_list,     // [P] int32
    torch::Tensor k_list,     // [P] int32
    torch::Tensor v_list,     // [P] int32
    torch::Tensor coeff_list, // [P] same dtype as w
    torch::Tensor packed_paths,
    torch::Tensor v_offsets,  // [V+1] int32
    int64_t V64)
{
    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(), "w/x_all/y must be CUDA");
    TORCH_CHECK(src_idx.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(dst_idx.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(), "w/x_all/y must be contiguous");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
    TORCH_CHECK(v_offsets.scalar_type() == torch::kInt32, "v_offsets must be int32");
    TORCH_CHECK(coeff_list.scalar_type() == w.scalar_type(), "coeff_list dtype must match w");

    int B  = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x_all.size(0);
    int Ix = (int)x_all.size(1);

    int Ky = (int)y.size(1);
    int V  = (int)V64;

    const int P = (int)i_list.size(0);

    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");
    TORCH_CHECK((int)y.size(0) == B && (int)y.size(2) == 1, "y must be [B,Ky,1]");
    TORCH_CHECK((int)src_idx.numel() == B, "src_idx must be [B]");
    TORCH_CHECK((int)dst_idx.numel() == B, "src_idx must be [B]");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");

    auto out = torch::zeros({S, V, U}, w.options());

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "u1d_fused_fwd_atomic", [&]{
        
        dim3 block(64);  // 2 warps
        dim3 grid(B, (U + 31) / 32);

        if (P == 777 && U == 32)
            uniform1d_codegen_two_warp_vgroup_path777_u32_fwd<scalar_t><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V);

        else if (P == 1490 && U == 32)
            uniform1d_codegen_two_warp_vgroup_path1490_u32_fwd<scalar_t><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V);

        else if (P == 1554 && U == 32)
             uniform1d_codegen_two_warp_vgroup_path1554_u32_fwd<scalar_t><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V);
        else if (P == 215 && U == 224)
            uniform1d_codegen_two_warp_vgroup_path215_u224_fwd<scalar_t><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V, U);

        else
            TORCH_CHECK(false, "Uniform1d Unsupported P: ", P);
        
    });
    
    return out; // [S, V, U]
}

std::vector<torch::Tensor> uniform1d_codegen_bwd_launch(
    torch::Tensor grad_out,    // [S,V,U]
    torch::Tensor w,           // [B,Iw,U]
    torch::Tensor x_all,       // [S,Ix,U]
    torch::Tensor y,           // [B,Ky,1]
    torch::Tensor src_idx,     // [B] int32
    torch::Tensor dst_idx,     // [B] int32
    torch::Tensor b_list,      // [B] int32
    torch::Tensor cls_offsets, // [S+1] int32
    torch::Tensor i_list,      // [P] int32
    torch::Tensor j_list,      // [P] int32
    torch::Tensor k_list,      // [P] int32
    torch::Tensor v_list,      // [P] int32
    torch::Tensor coeff_list,  // [P] same dtype as w
    torch::Tensor packed_paths,
    torch::Tensor v_offsets,   // [V+1] int32
    int64_t V64)
{
    TORCH_CHECK(grad_out.is_cuda() && w.is_cuda() && x_all.is_cuda() && y.is_cuda(),
                "grad_out/w/x_all/y must be CUDA");
    TORCH_CHECK(src_idx.is_cuda() && dst_idx.is_cuda(), "src_idx/dst_idx must be CUDA");
    TORCH_CHECK(b_list.is_cuda(), "b_list must be CUDA");
    TORCH_CHECK(grad_out.is_contiguous() && w.is_contiguous() &&
                x_all.is_contiguous() && y.is_contiguous(),
                "grad_out/w/x_all/y must be contiguous");

    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
    TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list must be int32");
    TORCH_CHECK(v_offsets.scalar_type() == torch::kInt32, "v_offsets must be int32");
    TORCH_CHECK(coeff_list.scalar_type() == w.scalar_type(),
                "coeff_list dtype must match w");
    TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(),
                "grad_out dtype must match w");

    const int B  = (int)w.size(0);
    const int Iw = (int)w.size(1);
    const int U  = (int)w.size(2);

    const int S  = (int)x_all.size(0);
    const int Ix = (int)x_all.size(1);

    const int Ky = (int)y.size(1);
    const int V  = (int)V64;

    const int P = (int)i_list.size(0);

    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");
    TORCH_CHECK((int)y.size(0) == B && (int)y.size(2) == 1, "y must be [B,Ky,1]");
    TORCH_CHECK((int)src_idx.numel() == B, "src_idx must be [B]");
    TORCH_CHECK((int)dst_idx.numel() == B, "dst_idx must be [B]");
    TORCH_CHECK((int)b_list.numel() == B, "b_list must be [B]");
    TORCH_CHECK((int)grad_out.size(0) == S &&
                (int)grad_out.size(1) == V &&
                (int)grad_out.size(2) == U,
                "grad_out must be [S,V,U]");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");

    auto grad_w = torch::zeros_like(w);      // [B,Iw,U]
    auto grad_x = torch::zeros_like(x_all);  // [S,Ix,U]
    auto grad_y = torch::zeros_like(y);      // [B,Ky,1]

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "uniform1d_codegen_bwd_launch", [&] {
        dim3 block(64);  // 2 warps
        dim3 grid(B, (U + 31) / 32);
        
        if (P == 777 && U == 32) {
            uniform1d_codegen_two_warp_vgroup_path777_u32_bwd<scalar_t><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V);
        }
        else if (P == 1490 && U == 32) {
            uniform1d_codegen_two_warp_vgroup_path1490_u32_bwd<scalar_t><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V);
        }
        else if (P == 1554 && U == 32) {
            uniform1d_codegen_two_warp_vgroup_path1554_u32_bwd<scalar_t><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V);
        }
        else if (P == 215 && U == 224) {
            uniform1d_codegen_two_warp_vgroup_path215_u224_bwd<scalar_t><<<grid, block, 0, stream>>>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                (const int32_t*)src_idx.data_ptr<int32_t>(),
                (const int32_t*)dst_idx.data_ptr<int32_t>(),
                (const int32_t*)b_list.data_ptr<int32_t>(),
                B, Iw, Ix, Ky, V, U);
        }
        else {
            TORCH_CHECK(false, "Uniform1d backward unsupported P: ", P, ", U: ", U);
        }
        

    });

    return {grad_w, grad_x, grad_y};
}

TORCH_LIBRARY(uniform1d_codegen, m)
{
    m.def("forward", &uniform1d_codegen_fwd_launch);
    m.def("backward", &uniform1d_codegen_bwd_launch);
}