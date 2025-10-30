#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

__constant__ int kDims[4] = {1, 3, 5, 7};
__constant__ int kOff[4]  = {0, 1, 4, 9};

__global__ void channel_wise_tensor_product_fwd(
    const double* __restrict__ x,   // [B, U]
    const double* __restrict__ y,   // [B, dim_sum]
    const double* __restrict__ w,   // [B, num_paths*U]  (按 path 访问)
    double* __restrict__ out,       // [B, dim_sum, U]
    int B, int dim, int U, int num_paths, int dim_sum)
{
    // 这里 dim == dim_sum
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim_sum * U;               // 每个 batch 的元素数
    if (tid >= total) return;

    int b = blockIdx.y;                    // 当前 batch

    // 先在“展平”的 [dim_sum, U] 空间里确定 path 与局部坐标
    int path = 0, tmp = tid;
    while (path < num_paths && tmp >= U * kDims[path]) {
        tmp -= U * kDims[path];
        ++path;
    }

    int dp = kDims[path];                  // 当前 path 的段长
    int dim_offset = kOff[path];           // 在 y（dim_sum 维）的偏移
    int u  = tmp / dp;                     // 当前 path 内的通道 u
    int i  = tmp % dp;                     // 当前 path 内的局部 dim 索引

    double xv = x[b * U + u];
    double yv = y[b * dim_sum + (dim_offset + i)];
    double wv = w[b * (num_paths * U) + path * U + u];

    // 写回到 [B, dim_sum, U] 的线性地址
    // out_idx = ((b * dim_sum) + k_global) * U + u
    int k_global = dim_offset + i;
    int out_idx = ((b * dim_sum) + k_global) * U + u;

    out[out_idx] = xv * yv * wv;
}

at::Tensor cwtp_forward(
    at::Tensor x,   // [B, U], double, CUDA, contiguous
    at::Tensor y,   // [B, dim_sum], double, CUDA, contiguous
    at::Tensor w)   // [B, num_paths*U], double, CUDA, contiguous
{
    const int B         = x.size(0);
    const int U         = x.size(1);
    const int dim_sum   = y.size(1);        // 1+3+5+7 = 16
    const int num_paths = 4;
    const int dim       = dim_sum;          // 沿用原参数命名

    // 分配输出为 [B, dim_sum, U]
    at::Tensor out = at::empty({B, dim_sum, U}, x.options()).contiguous();

    // 每个 batch 的总元素：dim_sum * U
    const int total = dim_sum * U;
    const int threads = 512;
    dim3 blockDim(threads);
    dim3 gridDim((total + threads - 1) / threads, B);

    channel_wise_tensor_product_fwd<<<gridDim, blockDim>>>(
        x.data_ptr<double>(),
        y.data_ptr<double>(),
        w.data_ptr<double>(),
        out.data_ptr<double>(),
        B, dim, U, num_paths, dim_sum);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;  // [B, dim_sum, U]
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cwtp_forward, "Segment Tensor Product Baseline");
}