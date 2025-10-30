#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define U 96
#define V 10

#define WARPS_PER_BLOCK 3
#define WARP_SIZE 32
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

// Tile sizes 
#define TILE_U 32  // 每个 warp 处理多少 u 行
#define NUM_U_PER_THREAD (TILE_U / WARP_SIZE)  // 每线程处理多少个 u 行



__global__ void tp_opt_kernel(
    const double* __restrict__ A,   // [BATCH, U]
    const double* __restrict__ B,   // [BATCH, V]
    double* __restrict__ out,       // [BATCH, U, V]
    int BATCH
) {
    int b = blockIdx.x;  // 每个 block 处理一个 batch
    if (b >= BATCH) return;

    int thread_id = threadIdx.x;
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;

    int warp_u_base = warp_id * TILE_U;
    int thread_u0 = warp_u_base + lane_id * NUM_U_PER_THREAD;

    if (thread_u0 >= U) return;

    const double* a_row = A + b * U;
    const double* b_row = B + b * V;
    double* out_ptr = out + b * U * V;

    // Load B[b, :] into register (共享每线程)
    double b_val[V];
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        b_val[v] = b_row[v];
    }

    // 每线程处理 NUM_U_PER_THREAD 行 u
    #pragma unroll
    for (int i = 0; i < NUM_U_PER_THREAD; ++i) {
        int u = thread_u0 + i;
        if (u >= U) continue;

        double a_val = a_row[u];

        #pragma unroll
        for (int v = 0; v < V; ++v) {
            out_ptr[u * V + v] = a_val * b_val[v];
        }
    }
} 


void launch_tp_kernel(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    int Batch = A.size(0);

    dim3 grid(Batch);  // 每 block 处理一个 batch
    dim3 block(THREADS_PER_BLOCK);  // 每 block 有多个 warp 并行处理 U
    
    tp_opt_kernel<<<grid, block>>>(
        A.data_ptr<double>(),
        B.data_ptr<double>(),
        out.data_ptr<double>(),
	Batch
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_tp_kernel", &launch_tp_kernel, "Warp tile tensor product kernel (CUDA)");
}
