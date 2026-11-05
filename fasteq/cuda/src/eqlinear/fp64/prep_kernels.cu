/***************************************************************************************************
 * fasteq eqlinear fp64/csrc/prep_kernels.cu
 **************************************************************************************************/
#include "prep_kernels.h"

#include <cstdlib>

namespace eqlin {
namespace {

constexpr int kThreads = 256;

/// x-extent cap for the launch grid.  gridDim.y carries the row index (A: b, B: n) so the kernel
/// needs no 64-bit division per element; gridDim.x is capped so blocks stay long-lived and
/// grid-stride over the columns instead of each retiring after one iteration (the same geometry
/// the tf32 line measured for its pack kernels).
constexpr int kPrepBlocksX = 16;

__global__ void permute_a_kernel(double const *__restrict__ x,
                                 double *__restrict__ a_prep,
                                 int const *__restrict__ src_index,
                                 int I_U,
                                 int cols) {
  double const *__restrict__ xr = x + static_cast<long long>(blockIdx.y) * I_U;
  double *__restrict__ ar = a_prep + static_cast<long long>(blockIdx.y) * cols;
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x) {
    ar[col] = xr[src_index[col]];
  }
}

__global__ void pack_b_kernel(double const *__restrict__ w,
                              double *__restrict__ b_prep,
                              int const *__restrict__ src_base,
                              int const *__restrict__ p_of_col,
                              CgVals cg,
                              int cols) {
  int const n = blockIdx.y;
  double *__restrict__ br = b_prep + static_cast<long long>(n) * cols;
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x) {
    br[col] = cg.v[p_of_col[col]] * w[src_base[col] + n];
  }
}

inline dim3 grid_for(int cols, int rows) {
  int cap = kPrepBlocksX;
  if (char const *e = nullptr) cap = atoi(e);
  int blocks = (cols + kThreads - 1) / kThreads;
  if (blocks > cap) blocks = cap;
  if (blocks < 1) blocks = 1;
  return dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(rows));
}

}  // namespace

void launch_permute_a(double const *x, double *a_prep, int const *src_index,
                      int I, int U, int B, cudaStream_t stream) {
  int const cols = I * U;
  if (B == 0 || cols == 0) return;
  permute_a_kernel<<<grid_for(cols, B), kThreads, 0, stream>>>(x, a_prep, src_index, cols, cols);
}

void launch_pack_b(double const *w, double *b_prep, int const *src_base, int const *p_of_col,
                   CgVals cg, int P, int U, int V, cudaStream_t stream) {
  int const cols = P * U;
  if (V == 0 || cols == 0) return;
  pack_b_kernel<<<grid_for(cols, V), kThreads, 0, stream>>>(w, b_prep, src_base, p_of_col, cg,
                                                            cols);
}

}  // namespace eqlin
