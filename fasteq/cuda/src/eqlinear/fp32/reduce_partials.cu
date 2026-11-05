/***************************************************************************************************
 * fasteq eqlinear fp32/csrc/reduce_partials.cu
 **************************************************************************************************/
#include "reduce_partials.h"

namespace eqlin_fp32 {
namespace {

constexpr int kThreads = 256;
/// x-extent cap: a few long-lived blocks beat one block per element (same reasoning as the prep
/// kernels: retiring after a single iteration drops achieved occupancy).
constexpr int kMaxSlices = 64;

/// blockIdx.x = output row j, blockIdx.y = a slice of the (B*V/4) float4 elements.  The reads for
/// consecutive threads are consecutive, so each of the pg slabs is a coalesced pass.
__global__ void reduce_partials_kernel(float const *__restrict__ partials,
                                       int const *__restrict__ rbase, int const *__restrict__ pg,
                                       float *__restrict__ out, int B, int J, int V, int R) {
  int const j = blockIdx.x;
  int const g = pg[j];
  int const v4 = V / 4;
  long long const n4 = static_cast<long long>(B) * v4;
  float const *base = partials + static_cast<long long>(rbase[j]) * V;

  for (long long t = blockIdx.y * kThreads + threadIdx.x; t < n4;
       t += static_cast<long long>(gridDim.y) * kThreads) {
    long long const b = t / v4;
    int const c = static_cast<int>(t - b * v4);
    float4 const *src =
        reinterpret_cast<float4 const *>(base + b * static_cast<long long>(R) * V) + c;
    float4 acc = *src;
    for (int m = 1; m < g; ++m) {
      float4 const x = *(src + static_cast<long long>(m) * v4);
      acc.x += x.x;
      acc.y += x.y;
      acc.z += x.z;
      acc.w += x.w;
    }
    *reinterpret_cast<float4 *>(out + (static_cast<long long>(b) * J + j) * V + 4 * c) = acc;
  }
}

}  // namespace

void launch_reduce_partials(float const *partials, int const *rbase, int const *pg, float *out,
                            int B, int J, int V, int R, cudaStream_t stream) {
  if (B <= 0 || J <= 0 || V <= 0 || R <= 0 || (V % 4) != 0) return;
  long long const n4 = static_cast<long long>(B) * (V / 4);
  int slices = static_cast<int>((n4 + kThreads - 1) / kThreads);
  if (slices > kMaxSlices) slices = kMaxSlices;
  if (slices < 1) slices = 1;
  dim3 const grid(J, slices);
  reduce_partials_kernel<<<grid, kThreads, 0, stream>>>(partials, rbase, pg, out, B, J, V, R);
}

}  // namespace eqlin_fp32
