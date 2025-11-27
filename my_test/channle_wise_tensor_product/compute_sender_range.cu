// compute_sender_runs_sorted.cu（可合并进现有 .cu）
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void k_fill_zero_i32(int32_t* arr, int32_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) arr[i] = 0;
}

// sender 必须非降序（已按 sender 排序）
// 对每条边 i 并行：检测 run 起点/终点，无需原子
__global__ void k_runs_from_sorted_sender(
    const int32_t* __restrict__ sender, // [E], sorted
    int32_t E,
    int32_t* __restrict__ start_idx,    // [N]
    int32_t* __restrict__ end_idx       // [N]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;

    int32_t s = sender[i];

    // run start
    if (i == 0 || sender[i - 1] != s) {
        start_idx[s] = i;
    }
    // run end  (end is exclusive: i+1)
    if (i == E - 1 || sender[i + 1] != s) {
        end_idx[s] = i + 1;
    }
}

extern "C" void compute_sender_runs_sorted_cuda(
    const int32_t* sender,  // [E], int32, sorted, cuda
    int32_t E,
    int32_t N,
    int32_t* start_idx,     // [N], int32, cuda (output)
    int32_t* end_idx,       // [N], int32, cuda (output)
    cudaStream_t stream
){
    const int threads = 256;
    const int blocksN = (N + threads - 1) / threads;
    const int blocksE = (E + threads - 1) / threads;

    // 初始化为 0，表示无出边的节点保持空段 [0,0)
    k_fill_zero_i32<<<blocksN, threads, 0, stream>>>(start_idx, N);
    k_fill_zero_i32<<<blocksN, threads, 0, stream>>>(end_idx, N);

    // 单 pass 根据排序好的 sender 写入每个节点的起止索引
    k_runs_from_sorted_sender<<<blocksE, threads, 0, stream>>>(
        sender, E, start_idx, end_idx
    );
}


std::vector<torch::Tensor> compute_sender_runs_sorted_binding(
    torch::Tensor sender, // [E], int32 或 int64, CUDA, 且已排序
    int64_t nnodes
){
    TORCH_CHECK(sender.is_cuda(), "sender must be CUDA tensor");
    TORCH_CHECK(sender.dtype() == at::kInt || sender.dtype() == at::kLong,
                "sender must be int32 or int64");
    TORCH_CHECK(nnodes >= 0, "nnodes must be non-negative");

    auto dev = sender.device();
    int64_t E64 = sender.numel();
    int32_t N = static_cast<int32_t>(nnodes);
    int32_t E = static_cast<int32_t>(E64);
    TORCH_CHECK((int64_t)N == nnodes && (int64_t)E == E64, "size too large for int32");

    // 若是 int64 转 int32（假定 sender 值域 < 2^31）
    torch::Tensor sender_i32 = (sender.dtype() == at::kInt)
        ? sender.contiguous()
        : sender.to(at::kInt, /*non_blocking=*/true).contiguous();

    auto opts_i32 = torch::TensorOptions().dtype(at::kInt).device(dev);
    auto start_idx = torch::empty({nnodes}, opts_i32);
    auto end_idx   = torch::empty({nnodes}, opts_i32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    compute_sender_runs_sorted_cuda(
        sender_i32.data_ptr<int32_t>(),
        E, N,
        start_idx.data_ptr<int32_t>(),
        end_idx.data_ptr<int32_t>(),
        stream
    );

    return {start_idx, end_idx};
}