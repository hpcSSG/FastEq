#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void stp_baseline_kernel(
    const double* __restrict__ x1,
    const double* __restrict__ x0_g,
    const double* __restrict__ coeffs,
    const int* __restrict__ paths,
    const int* __restrict__ path_lens,
    double* __restrict__ out,
    int B, int num_paths, int u, int num_a, int num_i) {

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    for (int p = 0; p < num_paths; ++p) {
        const double coeff = coeffs[p];
        const int len = path_lens[p];
        const int* path = paths + p * 5;

        const int a = path[0];
        const int i = (len == 3) ? path[1] : path[len - 2];

        for (int j = 0; j < u; ++j) {
            double val = x1[b * num_a * u + a * u + j];

            if (len >= 4) {
                int b_idx = path[1];
                val *= x1[b * num_a * u + b_idx * u + j];
            }
            if (len == 5) {
                int c_idx = path[2];
                val *= x1[b * num_a * u + c_idx * u + j];
            }

            val *= x0_g[b * num_i * u + i * u + j];
            val *= coeff;

            atomicAdd(&out[b * u + j], val);
        }
    }
}

void stp_baseline_launcher(
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

    const int threads = 128;
    const int blocks = (B + threads - 1) / threads;

    stp_baseline_kernel<<<blocks, threads>>>(
        x1.data_ptr<double>(),
        x0_g.data_ptr<double>(),
        coeffs.data_ptr<double>(),
        paths_tensor.data_ptr<int>(),
        path_lens.data_ptr<int>(),
        out.data_ptr<double>(),
        B, num_paths, u, num_a, num_i);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("segmented_tensor_product", &stp_baseline_launcher, "Segment Tensor Product Baseline");
}