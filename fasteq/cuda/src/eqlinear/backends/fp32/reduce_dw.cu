/***************************************************************************************************
 * fasteq eqlinear backends/fp32/reduce_dw.cu
 **************************************************************************************************/
#include "reduce_dw.h"

namespace eqlin_fp32 {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 8192;

__global__ void reduce_dw_kernel(float const *__restrict__ partials,
                                 int const *__restrict__ split_off,
                                 int const *__restrict__ splits, CgVals cg,
                                 float *__restrict__ dw, long long uv, long long total) {
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    long long const p = idx / uv;
    int const e = static_cast<int>(idx - p * uv);
    int const s0 = split_off[p];
    int const n = splits[p];
    double acc = 0.0;
    for (int s = 0; s < n; ++s) {
      acc += static_cast<double>(partials[(long long)(s0 + s) * uv + e]);
    }
    dw[idx] = static_cast<float>(cg.v[p] * acc);
  }
}

}  // namespace

void launch_reduce_dw(float const *partials, int const *split_off, int const *splits, CgVals cg,
                      float *dw, int P, int U, int V, cudaStream_t stream) {
  long long const uv = (long long)U * V;
  long long const total = (long long)P * uv;
  if (total == 0) return;
  long long blocks = (total + kThreads - 1) / kThreads;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  if (blocks < 1) blocks = 1;
  reduce_dw_kernel<<<static_cast<int>(blocks), kThreads, 0, stream>>>(
      partials, split_off, splits, cg, dw, uv, total);
}

}  // namespace eqlin_fp32
