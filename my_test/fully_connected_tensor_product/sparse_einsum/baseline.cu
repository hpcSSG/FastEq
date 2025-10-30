#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void sparse_einsum_kernel(
    const double* __restrict__ bijw,    // [B, I, J, W]
    const int* __restrict__ cg_indices, // [N_paths, 3]
    const double* __restrict__ cg_values,   // [N_paths]
    double* __restrict__ out,           // [B, K, W]
    int B, int I, int J, int K, int W, int N_paths
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

        double x = bijw[((b * I + i) * J + j) * W + w];
        atomicAdd(&out[(b * K + k) * W + w], v * x);
    }
}

void launch_sparse_einsum_kernel(
    at::Tensor bijw,         // [B, I, J, W]
    at::Tensor cg_indices,   // [N_paths, 3] int64
    at::Tensor cg_values,    // [N_paths] float64
    at::Tensor out           // [B, K, W]
) {
    const int B = bijw.size(0);
    const int I = bijw.size(1);
    const int J = bijw.size(2);
    const int W = bijw.size(3);
    const int K = out.size(1);
    const int N_paths = cg_values.size(0);

    dim3 threads(32, 4);
    dim3 blocks((W + 31) / 32, (B + 3) / 4);

    sparse_einsum_kernel<<<blocks, threads>>>(
        bijw.data_ptr<double>(),
        cg_indices.data_ptr<int>(),
        cg_values.data_ptr<double>(),
        out.data_ptr<double>(),
        B, I, J, K, W, N_paths
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_sparse_einsum_kernel, "Sparse Einsum Kernel (CUDA)");
}
