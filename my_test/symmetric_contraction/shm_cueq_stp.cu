#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void stp_shared_kernel(
    const double* __restrict__ x1,        // [B, num_a, u] -> flattened
    const double* __restrict__ x0_g,      // [B, num_i, u] -> flattened
    const double* __restrict__ coeffs,    // [num_paths]
    const int* __restrict__ paths,    // [num_paths, 5] -> flattened
    const int* __restrict__ path_lens,// [num_paths]
    double* __restrict__ out,             // [B, u] -> flattened
    int B, int num_paths, int u, int num_a, int num_i)
{
    extern __shared__ double shared_mem[];  // dynamic shared memory
    double* x1_shared = shared_mem;                // size: num_a * u
    double* x0g_shared = x1_shared + num_a * u;    // size: num_i * u

    int b = blockIdx.x;
    int j = threadIdx.x;

    if (b >= B || j >= u) return;

    // 每个 block 对应一个样本 b，每个线程处理一个 j
    // 预加载 x1[b, :, j] 到 shared memory
    for (int a = threadIdx.x; a < num_a * u; a += blockDim.x) {
        int a_idx = a / u;
        int u_idx = a % u;
        x1_shared[a] = x1[b * num_a * u + a_idx * u + u_idx];
    }

    // 预加载 x0_g[b, :, j] 到 shared memory
    for (int i = threadIdx.x; i < num_i * u; i += blockDim.x) {
        int i_idx = i / u;
        int u_idx = i % u;
        x0g_shared[i] = x0_g[b * num_i * u + i_idx * u + u_idx];
    }

    __syncthreads();

    double acc = 0.0;

    for (int p = 0; p < num_paths; ++p) {
        const double coeff = coeffs[p];
        const int len = path_lens[p];
        const int* path = paths + p * 5;

        const int a = path[0];
        const int i = (len == 3) ? path[1] : path[len - 2];

        double val = x1_shared[a * u + j];

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



void stp_shm_launcher(
    at::Tensor x1,
    at::Tensor x0_g,
    at::Tensor coeffs,
    at::Tensor paths_tensor,
    at::Tensor path_lens,
    at::Tensor out) {

    const int B = x1.size(0);
    const int num_paths = coeffs.size(0);
    const int u = x1.size(2);
    const int num_a = x1.size(1);
    const int num_i = x0_g.size(1);

    dim3 blockDim(u);        // 每个线程处理一个 j
    dim3 gridDim(B);         // 每个 block 处理一个样本 b
    size_t shared_mem_bytes = sizeof(double) * (num_a * u + num_i * u);

    stp_shared_kernel<<<gridDim, blockDim, shared_mem_bytes>>>(
        x1.data_ptr<double>(),
        x0_g.data_ptr<double>(),
        coeffs.data_ptr<double>(),
        paths_tensor.data_ptr<int>(),
        path_lens.data_ptr<int>(),
        out.data_ptr<double>(),
        B, num_paths, u, num_a, num_i);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("segmented_tensor_product", &stp_shm_launcher, "Segment Tensor Product Baseline");
}