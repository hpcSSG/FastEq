/***************************************************************************************************
 * fasteq eqlinear tf32/csrc/prep_kernels.cu
 **************************************************************************************************/
#include "prep_kernels.h"

namespace eqlin {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 8192;

__global__ void permute_a_kernel(float const *__restrict__ x,
                                 float *__restrict__ a_prep,
                                 int const *__restrict__ src_index,
                                 int I_U,
                                 long long total) {
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    long long const b = idx / I_U;
    int const col = static_cast<int>(idx - b * I_U);
    a_prep[idx] = x[b * I_U + src_index[col]];
  }
}

__global__ void pack_b_kernel(float const *__restrict__ w,
                              float *__restrict__ b_prep,
                              int const *__restrict__ src_base,
                              int const *__restrict__ p_of_col,
                              CgVals cg,
                              int PU,
                              long long total) {
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    long long const n = idx / PU;
    int const col = static_cast<int>(idx - n * PU);
    b_prep[idx] = cg.v[p_of_col[col]] * w[src_base[col] + n];
  }
}

inline int grid_for(long long total) {
  long long blocks = (total + kThreads - 1) / kThreads;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  if (blocks < 1) blocks = 1;
  return static_cast<int>(blocks);
}

}  // namespace

void launch_permute_a(float const *x, float *a_prep, int const *src_index,
                      int I, int U, int B, cudaStream_t stream) {
  long long const total = (long long)B * I * U;
  if (total == 0) return;
  permute_a_kernel<<<grid_for(total), kThreads, 0, stream>>>(x, a_prep, src_index, I * U, total);
}

void launch_pack_b(float const *w, float *b_prep, int const *src_base, int const *p_of_col,
                   CgVals cg, int P, int U, int V, cudaStream_t stream) {
  long long const total = (long long)V * P * U;
  if (total == 0) return;
  pack_b_kernel<<<grid_for(total), kThreads, 0, stream>>>(w, b_prep, src_base, p_of_col, cg,
                                                          P * U, total);
}

}  // namespace eqlin
