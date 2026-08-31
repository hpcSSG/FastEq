#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#include "equi_linear_kernels/impl.h"

bool is_aligned_128_bitwise(const void *ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) & 127) == 0; // 128-1=127
}

torch::Tensor mutipath_equi_linear_fwd(const torch::Tensor &x,                 // [B, total_i, U]
                                       const torch::Tensor &w,                 // [num_paths, U, V]
                                       const std::vector<int64_t> &i_dims_vec, // i dims
                                       const std::vector<double> &cg_val_vec)
{
    TORCH_CHECK((x.dtype() == torch::kFloat32 || x.dtype() == torch::kFloat64), "X must be float32 or float64");
    TORCH_CHECK(x.dim() == 3, "X must be of 3 dimention");
    TORCH_CHECK(w.dtype() == x.dtype(), "W must be the same type of x");
    TORCH_CHECK(w.dim() == 3, "W must be of 3 dimention");
    TORCH_CHECK(cg_val_vec.size() == i_dims_vec.size(), "cg_val_vec dims must match input path nums");

    uint32_t B = x.size(0);
    uint32_t total_i = x.size(1);
    uint32_t U = x.size(2);
    uint32_t V = w.size(2);
    uint32_t num_paths = w.size(0);

    torch::Tensor out = torch::empty({B, 16, V}, x.options());

    void *out_ptr = out.data_ptr();
    void *x_ptr = x.data_ptr();
    void *w_ptr = w.data_ptr();
    TORCH_CHECK(is_aligned_128_bitwise(out_ptr), "Tensor out misaligned by 128 Byte");
    TORCH_CHECK(is_aligned_128_bitwise(x_ptr), "Tensor x misaligned by 128 Byte");
    TORCH_CHECK(is_aligned_128_bitwise(w_ptr), "Tensor W misaligned by 128 Byte");
    cudaStream_t cur_stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();

    if (x.dtype() == torch::kFloat32)
    {
        mutipath_equi_linear_fwd_f32_impl(num_paths, static_cast<float *>(out_ptr), static_cast<float *>(x_ptr),
                                          static_cast<float *>(w_ptr), B, total_i, i_dims_vec, U, V, cg_val_vec,
                                          cur_stream);
    }
    else
    {
        mutipath_equi_linear_fwd_f64_impl(num_paths, static_cast<double *>(out_ptr), static_cast<double *>(x_ptr),
                                          static_cast<double *>(w_ptr), B, total_i, i_dims_vec, U, V, cg_val_vec,
                                          cur_stream);
    }

    return out;
}

torch::Tensor mutipath_equi_linear_bwd(const torch::Tensor &grad,                  // [B, total_i, V]
                                       const torch::Tensor &w,                     // [num_paths, U, V]
                                       const std::vector<int64_t> &out_i_dims_vec, // i dims
                                       const std::vector<double> &cg_val_vec)
{
    TORCH_CHECK(grad.dtype() == torch::kFloat32 || grad.dtype() == torch::kFloat64, "grad must be float32 or float64");
    TORCH_CHECK(grad.dim() == 3, "grad must be of 3 dimention");
    TORCH_CHECK(w.dtype() == grad.dtype(), "W must be the same type of grad");
    TORCH_CHECK(w.dim() == 3, "W must be of 3 dimention");
    TORCH_CHECK(cg_val_vec.size() == out_i_dims_vec.size(), "cg_val_vec dims must match out path nums");

    uint32_t B = grad.size(0);
    uint32_t grad_total_i = grad.size(1);
    uint32_t out_total_i = std::accumulate(out_i_dims_vec.begin(), out_i_dims_vec.end(), 0);
    uint32_t U = w.size(1);
    uint32_t V = w.size(2);
    uint32_t out_num_paths = w.size(0);

    torch::Tensor out = torch::empty({B, out_total_i, U}, grad.options());

    void *out_ptr = out.data_ptr();
    void *grad_ptr = grad.data_ptr();
    void *w_ptr = w.data_ptr();
    TORCH_CHECK(is_aligned_128_bitwise(out_ptr), "Tensor out misaligned by 128 Byte");
    TORCH_CHECK(is_aligned_128_bitwise(grad_ptr), "Tensor grad misaligned by 128 Byte");
    TORCH_CHECK(is_aligned_128_bitwise(w_ptr), "Tensor W misaligned by 128 Byte");
    cudaStream_t cur_stream = c10::cuda::getCurrentCUDAStream(grad.device().index()).stream();

    if (grad.dtype() == torch::kFloat32)
    {
        mutipath_equi_linear_bwd_f32_impl(out_num_paths, static_cast<float *>(out_ptr), static_cast<float *>(grad_ptr),
                                          static_cast<float *>(w_ptr), B, out_total_i, out_i_dims_vec, U, V, cg_val_vec,
                                          cur_stream);
    }
    else
    {
        // Not implimented, fall back to fwd
        torch::Tensor wt = w.transpose(1, 2).contiguous();
        void *wt_ptr = wt.data_ptr();
        mutipath_equi_linear_fwd_f64_impl(out_num_paths, static_cast<double *>(out_ptr),
                                          static_cast<double *>(grad_ptr), static_cast<double *>(wt_ptr), B,
                                          out_total_i, out_i_dims_vec, V, U, cg_val_vec, cur_stream);
    }

    return out;
}

TORCH_LIBRARY(equi_linear, m)
{
    m.def("forward", &mutipath_equi_linear_fwd);
    m.def("backward", &mutipath_equi_linear_bwd);
}
