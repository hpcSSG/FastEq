#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void stc_fwd_kernel(
    const scalar_t* __restrict__ x1,        // [B, num_a, u] -> flattened
    const scalar_t* __restrict__ x0_g,      // [B, num_i, u] -> flattened
    const scalar_t* __restrict__ coeffs,    // [num_paths]
    const int* __restrict__ paths,          // [num_paths, 5] -> flattened
    const int* __restrict__ path_lens,      // [num_paths]
    scalar_t* __restrict__ out,             // [B, u] -> flattened
    int B, int num_paths, int u, int num_a, int num_i)
{
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];
    scalar_t* x1_shared  = reinterpret_cast<scalar_t*>(smem);                 // size: num_a * u
    scalar_t* x0g_shared = x1_shared + num_a * u;                             // size: num_i * u

    int b = blockIdx.x;
    int j = threadIdx.x;

    if (b >= B || j >= u) return;

    // 每个 block 对应一个样本 b，每个线程处理一个 j
    // 预加载 x1[b, :, :] 到 shared memory
    for (int a = threadIdx.x; a < num_a * u; a += blockDim.x) {
        int a_idx = a / u;
        int u_idx = a % u;
        x1_shared[a] = x1[b * num_a * u + a_idx * u + u_idx];
    }

    // 预加载 x0_g[b, :, :] 到 shared memory
    for (int i = threadIdx.x; i < num_i * u; i += blockDim.x) {
        int i_idx = i / u;
        int u_idx = i % u;
        x0g_shared[i] = x0_g[b * num_i * u + i_idx * u + u_idx];
    }

    __syncthreads();

    scalar_t acc = scalar_t(0);

    for (int p = 0; p < num_paths; ++p) {
        const scalar_t coeff = coeffs[p];
        const int len = path_lens[p];
        const int* path = paths + p * 5;

        const int a = path[0];
        const int i = (len == 3) ? path[1] : path[len - 2];

        scalar_t val = x1_shared[a * u + j];

        if (len >= 4) {
            int b_idx = path[1];
            val *= x1_shared[b_idx * u + j];
        }
        if (len == 5) {
            int c_idx = path[2];
            val *= x1_shared[c_idx * u + j];
        }

        val *= x0g_shared[i * u + j];
        val *= coeff;

        acc += val;
    }

    // accumulate 到 global memory
    atomicAdd(&out[b * u + j], acc);
}

// --------------------------------------
// Launcher：支持 float32 / float64
// --------------------------------------
at::Tensor stc_fwd_launcher(
    at::Tensor x1,            // [B, num_a, u]
    at::Tensor x0_g,          // [B, num_i, u]
    at::Tensor coeffs,        // [num_paths]
    at::Tensor paths_tensor,  // [num_paths, 5]
    at::Tensor path_lens)     // [num_paths]
{
    TORCH_CHECK(x1.is_cuda(), "x1 must be CUDA");
    TORCH_CHECK(x0_g.is_cuda(), "x0_g must be CUDA");
    TORCH_CHECK(coeffs.is_cuda(), "coeffs must be CUDA");
    TORCH_CHECK(paths_tensor.is_cuda(), "paths_tensor must be CUDA");
    TORCH_CHECK(path_lens.is_cuda(), "path_lens must be CUDA");

    auto dtype = x1.scalar_type();
    TORCH_CHECK(
        dtype == at::kFloat || dtype == at::kDouble,
        "x1 must be float32 or float64");
    TORCH_CHECK(x0_g.scalar_type() == dtype,
                "x0_g must have same dtype as x1");
    TORCH_CHECK(coeffs.scalar_type() == dtype,
                "coeffs must have same dtype as x1");

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

    at::Tensor out = at::zeros({B, u}, x1.options()).contiguous();

    dim3 blockDim(u);   // 每个线程处理一个 j（假设 u <= 最大线程数 1024）
    dim3 gridDim(B);    // 每个 block 处理一个样本 b

    // shared memory: (num_a * u + num_i * u) * sizeof(scalar_t)
    const size_t shared_elems = static_cast<size_t>(num_a) * u
                              + static_cast<size_t>(num_i) * u;
    const size_t shared_mem_bytes = shared_elems * x1.element_size();

    cudaStream_t cur_stream =
        c10::cuda::getCurrentCUDAStream(x1.device().index()).stream();

    AT_DISPATCH_FLOATING_TYPES(dtype, "stc_fwd_kernel", [&] {
        using scalar_t = scalar_t;
        stc_fwd_kernel<scalar_t><<<gridDim, blockDim, shared_mem_bytes, cur_stream>>>(
            x1.data_ptr<scalar_t>(),
            x0_g.data_ptr<scalar_t>(),
            coeffs.data_ptr<scalar_t>(),
            paths_tensor.data_ptr<int>(),
            path_lens.data_ptr<int>(),
            out.data_ptr<scalar_t>(),
            B, num_paths, u, num_a, num_i);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// --------------------------------------
// Torch 注册
// --------------------------------------
TORCH_LIBRARY(stc_fwd, m)
{
    m.def("forward", &stc_fwd_launcher);
}
