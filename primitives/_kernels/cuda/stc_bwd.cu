#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define CUDA_CHECK(expr) do { \
  cudaError_t _err = (expr);  \
  if (_err != cudaSuccess)    \
    AT_ERROR("CUDA error: ", cudaGetErrorString(_err), " at ", __FILE__, ":", __LINE__); \
} while(0)


__global__ void stc_bwd(
    const double* __restrict__ grad_out, // [B, u] -> flattened
    const double* __restrict__ x1,       // [B, num_a, u] -> flattened
    const double* __restrict__ x0_g,     // [B, num_i, u] -> flattened
    const double* __restrict__ coeffs,   // [num_paths]
    const int* __restrict__ paths,       // [num_paths, 5] -> flattened
    const int* __restrict__ path_lens,   // [num_paths]
    double* __restrict__ grad_x1,        // [B, num_a, u] -> flattened (output)
    int B, int num_paths, int u, int num_a, int num_i)
{
    extern __shared__ double shared_mem[];   // dynamic shared memory
    double* x1_shared  = shared_mem;                 // size: num_a * u
    double* x0g_shared = x1_shared + num_a * u;      // size: num_i * u

    const int b = blockIdx.x;
    const int j = threadIdx.x;

    if (b >= B || j >= u) return;

    // 预加载 x1[b, :, :] 与 x0_g[b, :, :]（与前向一致）
    for (int a = threadIdx.x; a < num_a * u; a += blockDim.x) {
        int a_idx = a / u;
        int u_idx = a % u;
        x1_shared[a] = x1[b * num_a * u + a_idx * u + u_idx];
    }
    for (int i = threadIdx.x; i < num_i * u; i += blockDim.x) {
        int i_idx = i / u;
        int u_idx = i % u;
        x0g_shared[i] = x0_g[b * num_i * u + i_idx * u + u_idx];
    }

    __syncthreads();

    const double go = grad_out[b * u + j];  // dL/d out[b,j]

    // 遍历所有 path，累加对参与的 x1 索引的梯度
    for (int p = 0; p < num_paths; ++p) {
        const double coeff = coeffs[p];
        const int len = path_lens[p];
        const int* path = paths + p * 5;

        // 基本索引
        const int a_idx = path[0];
        const int i_idx = (len == 3) ? path[1] : path[len - 2];

        // 参与乘积的值
        const double x1_a = x1_shared[a_idx * u + j];
        const double x0g  = x0g_shared[i_idx * u + j];

        if (len == 3) {
            // path: [a, i, *]
            const double contrib_a = go * coeff * x0g;
            atomicAdd(&grad_x1[b * num_a * u + a_idx * u + j], contrib_a);
        } else if (len == 4) {
            // path: [a, b, i, *]
            const int b_idx = path[1];
            const double x1_b = x1_shared[b_idx * u + j];

            const double base = go * coeff * x0g;
            const double contrib_a = base * x1_b;
            const double contrib_b = base * x1_a;

            atomicAdd(&grad_x1[b * num_a * u + a_idx * u + j], contrib_a);
            atomicAdd(&grad_x1[b * num_a * u + b_idx * u + j], contrib_b);
        } else if (len == 5) {
            // path: [a, b, c, i, *]
            const int b_idx = path[1];
            const int c_idx = path[2];
            const double x1_b = x1_shared[b_idx * u + j];
            const double x1_c = x1_shared[c_idx * u + j];

            const double base = go * coeff * x0g;

            const double contrib_a = base * (x1_b * x1_c);
            const double contrib_b = base * (x1_a * x1_c);
            const double contrib_c = base * (x1_a * x1_b);

            atomicAdd(&grad_x1[b * num_a * u + a_idx * u + j], contrib_a);
            atomicAdd(&grad_x1[b * num_a * u + b_idx * u + j], contrib_b);
            atomicAdd(&grad_x1[b * num_a * u + c_idx * u + j], contrib_c);
        }
        // 其他 len 情形不在当前定义范围内，忽略
    }
}

at::Tensor stc_bwd_launcher(
    at::Tensor grad_out,     // [B, u], double, contiguous
    at::Tensor x1,           // [B, num_a, u], double
    at::Tensor x0_g,         // [B, num_i, u], double
    at::Tensor coeffs,       // [num_paths],  double
    at::Tensor paths_tensor, // [num_paths, 5], int
    at::Tensor path_lens     // [num_paths],  int
) {

    const int B         = x1.size(0);
    const int num_a     = x1.size(1);
    const int u         = x1.size(2);
    const int num_i     = x0_g.size(1);
    const int num_paths = coeffs.size(0);

    auto grad_x1 = at::zeros_like(x1, x1.options()); // [B, num_a, u]

    dim3 blockDim(u);
    dim3 gridDim(B);
    size_t shared_mem_bytes = sizeof(double) * (num_a * u + num_i * u);
    cudaStream_t cur_stream = c10::cuda::getCurrentCUDAStream(grad_out.device().index()).stream();

    stc_bwd<<<gridDim, blockDim, shared_mem_bytes, cur_stream>>>(
        grad_out.data_ptr<double>(),
        x1.data_ptr<double>(),
        x0_g.data_ptr<double>(),
        coeffs.data_ptr<double>(),
        paths_tensor.data_ptr<int>(),
        path_lens.data_ptr<int>(),
        grad_x1.data_ptr<double>(),
        B, num_paths, u, num_a, num_i
    );
    CUDA_CHECK(cudaGetLastError());

    return grad_x1;
}

TORCH_LIBRARY(stc_bwd, m)
{
    m.def("backward", &stc_bwd_launcher);
}