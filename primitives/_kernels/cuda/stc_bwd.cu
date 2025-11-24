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


template<typename scalar_t>
__global__ void stc_bwd_kernel(
    const scalar_t* __restrict__ grad_out, // [B, u] -> flattened
    const scalar_t* __restrict__ x1,       // [B, num_a, u] -> flattened
    const scalar_t* __restrict__ x0_g,     // [B, num_i, u] -> flattened
    const scalar_t* __restrict__ coeffs,   // [num_paths]
    const int* __restrict__ paths,         // [num_paths, 5] -> flattened
    const int* __restrict__ path_lens,     // [num_paths]
    scalar_t* __restrict__ grad_x1,        // [B, num_a, u] -> flattened (output)
    int B, int num_paths, int u, int num_a, int num_i)
{
    // 动态 shared memory，按标量类型对齐
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* x1_shared  = reinterpret_cast<scalar_t*>(smem);           // size: num_a * u
    scalar_t* x0g_shared = x1_shared + num_a * u;                       // size: num_i * u

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

    const scalar_t go = grad_out[b * u + j];  // dL/d out[b,j]

    // 遍历所有 path，累加对参与的 x1 索引的梯度
    for (int p = 0; p < num_paths; ++p) {
        const scalar_t coeff = coeffs[p];
        const int len = path_lens[p];
        const int* path = paths + p * 5;

        // 基本索引
        const int a_idx = path[0];
        const int i_idx = (len == 3) ? path[1] : path[len - 2];

        // 参与乘积的值
        const scalar_t x1_a = x1_shared[a_idx * u + j];
        const scalar_t x0g  = x0g_shared[i_idx * u + j];

        if (len == 3) {
            // path: [a, i, *]
            const scalar_t contrib_a = go * coeff * x0g;
            atomicAdd(&grad_x1[b * num_a * u + a_idx * u + j], contrib_a);

        } else if (len == 4) {
            // path: [a, b, i, *]
            const int b_idx = path[1];
            const scalar_t x1_b = x1_shared[b_idx * u + j];

            const scalar_t base = go * coeff * x0g;
            const scalar_t contrib_a = base * x1_b;
            const scalar_t contrib_b = base * x1_a;

            atomicAdd(&grad_x1[b * num_a * u + a_idx * u + j], contrib_a);
            atomicAdd(&grad_x1[b * num_a * u + b_idx * u + j], contrib_b);

        } else if (len == 5) {
            // path: [a, b, c, i, *]
            const int b_idx = path[1];
            const int c_idx = path[2];
            const scalar_t x1_b = x1_shared[b_idx * u + j];
            const scalar_t x1_c = x1_shared[c_idx * u + j];

            const scalar_t base = go * coeff * x0g;

            const scalar_t contrib_a = base * (x1_b * x1_c);
            const scalar_t contrib_b = base * (x1_a * x1_c);
            const scalar_t contrib_c = base * (x1_a * x1_b);

            atomicAdd(&grad_x1[b * num_a * u + a_idx * u + j], contrib_a);
            atomicAdd(&grad_x1[b * num_a * u + b_idx * u + j], contrib_b);
            atomicAdd(&grad_x1[b * num_a * u + c_idx * u + j], contrib_c);
        }
        // 其他 len 情形不在当前定义范围内，忽略
    }
}


// -----------------------------
// Launcher：支持 float32 / float64
// -----------------------------
at::Tensor stc_bwd_launcher(
    at::Tensor grad_out,     // [B, u], float32/float64, contiguous
    at::Tensor x1,           // [B, num_a, u], same dtype
    at::Tensor x0_g,         // [B, num_i, u], same dtype
    at::Tensor coeffs,       // [num_paths],  same dtype
    at::Tensor paths_tensor, // [num_paths, 5], int
    at::Tensor path_lens     // [num_paths],  int
) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(x1.is_cuda(),       "x1 must be CUDA");
    TORCH_CHECK(x0_g.is_cuda(),     "x0_g must be CUDA");
    TORCH_CHECK(coeffs.is_cuda(),   "coeffs must be CUDA");
    TORCH_CHECK(paths_tensor.is_cuda(), "paths_tensor must be CUDA");
    TORCH_CHECK(path_lens.is_cuda(),    "path_lens must be CUDA");

    auto dtype = x1.scalar_type();
    TORCH_CHECK(
        dtype == at::kFloat || dtype == at::kDouble,
        "x1 must be float32 or float64");
    TORCH_CHECK(x0_g.scalar_type() == dtype,
                "x0_g must have same dtype as x1");
    TORCH_CHECK(coeffs.scalar_type() == dtype,
                "coeffs must have same dtype as x1");
    TORCH_CHECK(grad_out.scalar_type() == dtype,
                "grad_out must have same dtype as x1");

    TORCH_CHECK(paths_tensor.scalar_type() == at::kInt,
                "paths_tensor must be int32");
    TORCH_CHECK(path_lens.scalar_type() == at::kInt,
                "path_lens must be int32");

    grad_out     = grad_out.contiguous();
    x1           = x1.contiguous();
    x0_g         = x0_g.contiguous();
    coeffs       = coeffs.contiguous();
    paths_tensor = paths_tensor.contiguous();
    path_lens    = path_lens.contiguous();

    const int B         = x1.size(0);
    const int num_a     = x1.size(1);
    const int u         = x1.size(2);
    const int num_i     = x0_g.size(1);
    const int num_paths = coeffs.size(0);

    auto grad_x1 = at::zeros_like(x1); // [B, num_a, u]

    dim3 blockDim(u);
    dim3 gridDim(B);

    // shared memory: (num_a*u + num_i*u) * sizeof(scalar_t)
    const size_t shared_elems = static_cast<size_t>(num_a) * u
                              + static_cast<size_t>(num_i) * u;
    const size_t shared_mem_bytes = shared_elems * x1.element_size();

    cudaStream_t cur_stream =
        c10::cuda::getCurrentCUDAStream(grad_out.device().index()).stream();

    AT_DISPATCH_FLOATING_TYPES(dtype, "stc_bwd_kernel", [&] {
        using scalar_t = scalar_t;
        stc_bwd_kernel<scalar_t><<<gridDim, blockDim, shared_mem_bytes, cur_stream>>>(
            grad_out.data_ptr<scalar_t>(),
            x1.data_ptr<scalar_t>(),
            x0_g.data_ptr<scalar_t>(),
            coeffs.data_ptr<scalar_t>(),
            paths_tensor.data_ptr<int>(),
            path_lens.data_ptr<int>(),
            grad_x1.data_ptr<scalar_t>(),
            B, num_paths, u, num_a, num_i
        );
    });

    CUDA_CHECK(cudaGetLastError());
    return grad_x1;
}

TORCH_LIBRARY(stc_bwd, m)
{
    m.def("backward", &stc_bwd_launcher);
}
