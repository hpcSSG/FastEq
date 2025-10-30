#include <cuda.h>
#include <cuda_runtime.h> 
#include <torch/extension.h>

#define U 96
#define V 10
#define UV (U * V)
#define W 96


// Tile sizes 
#define TILE_U 32
#define TILE_V 10 
#define TILE_UV (TILE_U * TILE_V)  

// Kernel: warp-based tensor product
__global__ void warp_tile_tp_kernel(
    const double* __restrict__ A,   // [BATCH, U]
    const double* __restrict__ B,   // [BATCH, V]
    double* __restrict__ out,       // [BATCH, U, V]
    int BATCH
) {
    int b = threadIdx.x % 32;
    if (b >= BATCH) return;

    const double* a_row = A + b * U;
    const double* b_row = B + b * V;
    double* out_tensor = out + b * U * V;

    // iterate over U dimension by TILE_U
    for (int u0 = 0; u0 < U; u0 += TILE_U) {
        double a_tile[TILE_U];
        #pragma unroll
        for (int i = 0; i < TILE_U; ++i) {
            a_tile[i] = a_row[u0 + i];
        }

        // outer product: A_tile[u] * B[v] → O_tile[u * V + v]
        #pragma unroll
        for (int u = 0; u < TILE_U; ++u) {
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                out_tensor[(u0 + u) * V + v] = a_tile[u] * b_row[v];
            }
        }
    }
}


void launch_tp_kernel(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    const int threads = 32;
    const int blocks = 1;
    int Batch = A.size(0);
    warp_tile_tp_kernel<<<blocks, threads>>>(
        A.data_ptr<double>(),
        B.data_ptr<double>(),
        out.data_ptr<double>(),
	Batch
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_tp_kernel", &launch_tp_kernel, "Warp tile tensor product kernel (CUDA)");
}
