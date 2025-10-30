#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

#define U 96
#define V 10
#define UV (U * V)
#define W 96


// Tile sizes 
#define TILE_U 32
#define TILE_V 10 
#define TILE_UV (TILE_U * TILE_V)  

// Kernel: warp-based fused outer product and GEMM
__global__ void fused_outer_gemm_tile_kernel(
    const double* __restrict__ A,        // [BATCH, U]
    const double* __restrict__ B,        // [BATCH, V]
    const double* __restrict__ Weight,   // [UV, W]
    double* __restrict__ out,            // [BATCH, W]
    int BATCH
) {
    int lane_id = threadIdx.x % 32;
    if (lane_id >= BATCH) return;

    const double* a_row = A + lane_id * U;
    const double* b_row = B + lane_id * V;
    double* out_row = out + lane_id * W;

    // init output
    double acc[W] = {0.0};

    // iterate over u_tile
    for (int u0 = 0; u0 < U; u0 += TILE_U) {
        // A tile: a_tile[TILE_U]
        double a_tile[TILE_U];
        //#pragma unroll
        for (int i = 0; i < TILE_U; ++i) {
            a_tile[i] = a_row[u0 + i];
        }

        // Compute outer product [TILE_U x TILE_V]
        double o_tile[TILE_UV];
        //#pragma unroll
        for (int u = 0; u < TILE_U; ++u) {
            //#pragma unroll
            for (int v = 0; v < TILE_V; ++v) {
                o_tile[u * TILE_V + v] = a_tile[u] * b_row[v];
            }
        }

	// GEMM: o_tile[TILE_UV] @ W_part[TILE_UV x W]
        //#pragma unroll
        for (int k = 0; k < TILE_UV; ++k) {
            const double* w_row = Weight + (u0 * V + k) * W;
            double o_val = o_tile[k];
            //#pragma unroll
            for (int w = 0; w < W; ++w) {
                acc[w] += o_val * w_row[w];
            }
        }
    }

    //#pragma unroll
    for (int w = 0; w < W; ++w) {
        out_row[w] = acc[w];
    }
}



void launch_fused_kernel(torch::Tensor A, torch::Tensor B, torch::Tensor Weight, torch::Tensor out) {
    const int threads = 32;
    const int blocks = 1;
    int Batch = A.size(0);
    fused_outer_gemm_tile_kernel<<<blocks, threads>>>(
        A.data_ptr<double>(),
        B.data_ptr<double>(),
        Weight.data_ptr<double>(),
        out.data_ptr<double>(),
	Batch
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_kernel", &launch_fused_kernel, "Fused outer + GEMM kernel (CUDA)");
}
