#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

__global__ void sparse_einsum_bwd_kernel(
    const double* __restrict__ grad_out, // [B, K, W]
    const int* __restrict__ cg_indices, // [N_paths, 3]
    const double* __restrict__ cg_values,   // [N_paths]
    double* __restrict__ grad_bijw,           // [B, I, J, W]
    int B, int I, int J, int W, int K, int N_paths
) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || w >= W) return;

    #pragma unroll
    for (int idx = 0; idx < N_paths; ++idx) {
        int i = cg_indices[idx * 3 + 0];
        int j = cg_indices[idx * 3 + 1];
        int k = cg_indices[idx * 3 + 2];
        double v = cg_values[idx];

        double x = grad_out[(b * K + k) * W + w];
        atomicAdd(&grad_bijw[((b * I + i) * J + j) * W + w], v * x);
    }
}

at::Tensor launch_sparse_einsum_bwd_kernel(
    at::Tensor grad_out,         // [B, W, K]
    at::Tensor cg_indices,   // [N_paths, 3] int64
    at::Tensor cg_values    // [N_paths] float64
    //at::Tensor grad_bijw           // [B, I, J, W]
) {
    const int B = grad_out.size(0);
    const int K = grad_out.size(1);
    const int W = grad_out.size(2);
    const int I = K;
    const int J = 1;
    const int N_paths = K;

    at::Tensor grad_bijw = at::zeros({B, I, J, W}, cg_values.options());
    grad_bijw = grad_bijw.contiguous();

    //dim3 threads(32, 4);
    //dim3 blocks((W + 31) / 32, (B + 3) / 4);
    dim3 threads(8, 32); // 可以调试
    dim3 blocks((W + threads.x - 1)/threads.x, (B + threads.y - 1)/threads.y);
    cudaStream_t cur_stream = c10::cuda::getCurrentCUDAStream(grad_out.device().index()).stream();
    sparse_einsum_bwd_kernel<<<blocks, threads, 0, cur_stream>>>(
        grad_out.data_ptr<double>(),
        cg_indices.data_ptr<int>(),
        cg_values.data_ptr<double>(),
        grad_bijw.data_ptr<double>(),
        B, I, J, W, K, N_paths
    );

    return grad_bijw;

}

TORCH_LIBRARY(fctp_spmm_bwd, m)
{
    m.def("backward", &launch_sparse_einsum_bwd_kernel);
}