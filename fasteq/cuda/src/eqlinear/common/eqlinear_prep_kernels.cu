#include "eqlinear_prep_kernels.h"

namespace fasteq::eqlinear {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 8192;

template <typename T>
__global__ void permute_a_kernel(T const *__restrict__ x, T *__restrict__ a_prep,
                                 int const *__restrict__ src_index, int I_U, long long total) {
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    long long const b = idx / I_U;
    int const col = static_cast<int>(idx - b * I_U);
    a_prep[idx] = x[b * I_U + src_index[col]];
  }
}

template <typename T>
__global__ void pack_b_kernel(T const *__restrict__ w, T *__restrict__ b_prep,
                              int const *__restrict__ src_base,
                              int const *__restrict__ p_of_col, CgValues<T> cg, int PU,
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

template <typename T>
void launch_permute_a(T const *x, T *a_prep, int const *src_index, int I, int U, int B,
                      cudaStream_t stream) {
  long long const total = (long long)B * I * U;
  if (total == 0) return;
  permute_a_kernel<T><<<grid_for(total), kThreads, 0, stream>>>(x, a_prep, src_index, I * U,
                                                               total);
}

template <typename T>
void launch_pack_b(T const *w, T *b_prep, int const *src_base, int const *p_of_col,
                   CgValues<T> cg, int P, int U, int V, cudaStream_t stream) {
  long long const total = (long long)V * P * U;
  if (total == 0) return;
  pack_b_kernel<T><<<grid_for(total), kThreads, 0, stream>>>(w, b_prep, src_base, p_of_col, cg,
                                                             P * U, total);
}

template void launch_permute_a<float>(float const *, float *, int const *, int, int, int,
                                      cudaStream_t);
template void launch_pack_b<float>(float const *, float *, int const *, int const *, CgValues<float>,
                                   int, int, int, cudaStream_t);

}  // namespace fasteq::eqlinear
