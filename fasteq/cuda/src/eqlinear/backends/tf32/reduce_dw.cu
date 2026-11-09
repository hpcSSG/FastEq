/***************************************************************************************************
 * fasteq eqlinear backends/tf32/reduce_dw.cu
 **************************************************************************************************/
#include "reduce_dw.h"

namespace eqlin {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocksX = 64;

/// One block per path; threads walk the [U,V] tile, each summing its own element across splits.
/// The reads for consecutive threads are consecutive, so every split is a fully coalesced pass.
__global__ void reduce_dw_kernel(float const *__restrict__ partials, CgVals cg,
                                 float *__restrict__ dw, int uv, int const *sp,
                                 long long const *off) {
  int const p = blockIdx.y;
  float const scale = cg.v[p];
  long long const base = off[p];
  int const splits = sp[p];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < uv; i += gridDim.x * blockDim.x) {
    double acc = 0.0;
    for (int s = 0; s < splits; ++s) acc += static_cast<double>(partials[base + (long long)s * uv + i]);
    dw[static_cast<long long>(p) * uv + i] = static_cast<float>(acc * scale);
  }
}

}  // namespace

void launch_reduce_dw(float const *partials, CgVals cg, float *dw, int uv, int P, int const *sp,
                      long long const *off, cudaStream_t stream) {
  if (P <= 0 || uv <= 0) return;
  int bx = (uv + kThreads - 1) / kThreads;
  if (bx > kMaxBlocksX) bx = kMaxBlocksX;
  reduce_dw_kernel<<<dim3(bx, P), kThreads, 0, stream>>>(partials, cg, dw, uv, sp, off);
}

}  // namespace eqlin
