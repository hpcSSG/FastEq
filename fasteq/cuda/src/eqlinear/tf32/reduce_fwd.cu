/***************************************************************************************************
 * fasteq eqlinear tf32/csrc/reduce_fwd.cu
 **************************************************************************************************/
#include "reduce_fwd.h"

namespace eqlin {
namespace {

constexpr int kThreads = 128;

/// One block per (b, j); threads walk v.  `double` for the same reason `reduce_dw` uses it: summing
/// S fp32 partials of a chain that has already lost its bits in fp32 would add a second rounding
/// error of the same size as the one we are trying to remove.
__global__ void reduce_fwd_kernel(float const *__restrict__ partials, float *__restrict__ out,
                                  int J, int V, int S) {
  int const j = blockIdx.y;
  int const b = blockIdx.z;
  long long const pbase = static_cast<long long>(j) * S * V + static_cast<long long>(b) * J * S * V;
  long long const obase = (static_cast<long long>(b) * J + j) * V;
  long long const pstride = static_cast<long long>(V);
  for (int v = blockIdx.x * kThreads + threadIdx.x; v < V; v += gridDim.x * kThreads) {
    double acc = 0.0;
    for (int s = 0; s < S; ++s) acc += static_cast<double>(partials[pbase + s * pstride + v]);
    out[obase + v] = static_cast<float>(acc);
  }
}

}  // namespace

void launch_reduce_fwd(float const *partials, float *out, int J, int B, int V, int S,
                       cudaStream_t stream) {
  if (J <= 0 || B <= 0 || V <= 0 || S < 2) return;
  int bx = (V + kThreads - 1) / kThreads;
  if (bx > 8) bx = 8;
  reduce_fwd_kernel<<<dim3(bx, J, B), kThreads, 0, stream>>>(partials, out, J, V, S);
}

}  // namespace eqlin
