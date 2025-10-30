#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void sparse_einsum_bwd_kernel_4d(
    const double* __restrict__ grad_out,   // [B, W, K]
    const int* __restrict__ cg_indices,    // [N_paths, 3]
    const double* __restrict__ cg_values,  // [N_paths]
    double* __restrict__ grad_bijw,        // [B, I, J, W]
    int B, int I, int J, int W, int K, int N_paths
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int ij = blockIdx.z * blockDim.z + threadIdx.z;

    if (w >= W || b >= B || ij >= I * J) return;

    int i = ij / J;
    int j = ij % J;

    double sum = 0.0;

    // 遍历稀疏路径
    for (int idx = 0; idx < N_paths; ++idx) {
        int ii = cg_indices[idx * 3 + 0];
        int jj = cg_indices[idx * 3 + 1];
        int kk = cg_indices[idx * 3 + 2];
        double v = cg_values[idx];

        if (ii == i && jj == j) {
            double x = grad_out[(b * W + w) * K + kk];  // [B,W,K]
            sum += v * x;
        }
    }

    // 唯一写入位置，无需 atomicAdd
    grad_bijw[((b * I + i) * J + j) * W + w] = sum;
}



void launch_sparse_einsum_bwd_kernel(
    at::Tensor grad_out,   // [B, W, K]
    at::Tensor cg_indices, // [N_paths, 3]
    at::Tensor cg_values,  // [N_paths]
    at::Tensor grad_bijw   // [B, I, J, W]
) {
    const int B = grad_out.size(0);
    const int W = grad_out.size(1);
    const int K = grad_out.size(2);
    const int I = grad_bijw.size(1);
    const int J = grad_bijw.size(2);
    const int N_paths = cg_values.size(0);

    dim3 threads(16, 16, I);  // (x,y,z) 可调
    dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (B + threads.y - 1) / threads.y,
        ((I*J) + threads.z - 1) / threads.z
    );

    sparse_einsum_bwd_kernel_4d<<<blocks, threads>>>(
        grad_out.data_ptr<double>(),
        cg_indices.data_ptr<int>(),
        cg_values.data_ptr<double>(),
        grad_bijw.data_ptr<double>(),
        B, I, J, W, K, N_paths
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &launch_sparse_einsum_bwd_kernel, "Sparse Einsum Kernel (CUDA)");
}
