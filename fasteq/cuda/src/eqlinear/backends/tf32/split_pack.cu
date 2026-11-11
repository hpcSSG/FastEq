/***************************************************************************************************
 * fasteq eqlinear backends/tf32/split_pack.cu
 **************************************************************************************************/
#include "split_pack.h"

#include <cstdio>
#include <cstdlib>

#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

namespace eqlin {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 8192;

/// x-extent cap for the launch grid; see grid_for() for why a small cap wins.
constexpr int kPrepBlocksX = 16;

/// fp32 -> (big, res) tf32 pair, exactly the decomposition the tf32x3 GEMMs assume.
struct SplitF32 {
  cutlass::NumericConverterFastF32<> conv;
  CUTLASS_DEVICE cutlass::Array<cutlass::tfloat32_t, 2> operator()(float v) const { return conv(v); }
};

/// Same pair, round-to-nearest instead of truncation.  `round_half_ulp_truncate` writes back
/// `bits + 0x1000` and the tensor core truncates its operand, so the value the MMA sees is RN(v);
/// the residual is measured against that same value, which keeps big + res == v to within the
/// dropped res*res term.  Both bounds halve, so that term quarters.
struct SplitF32Rn {
  cutlass::NumericConverterFastF32<cutlass::FloatRoundStyle::round_half_ulp_truncate,
                                   cutlass::FloatRoundStyle::round_half_ulp_truncate>
      conv;
  CUTLASS_DEVICE cutlass::Array<cutlass::tfloat32_t, 2> operator()(float v) const { return conv(v); }
};

/// One splitter per launch, chosen once on the host so the inner loop stays a straight-line call.
struct SplitAny {
  SplitF32 trunc;
  SplitF32Rn rn;
  bool use_rn;
  CUTLASS_DEVICE cutlass::Array<cutlass::tfloat32_t, 2> operator()(float v) const {
    return use_rn ? rn(v) : trunc(v);
  }
};

/// The row index (b for A, n for B) is carried by blockIdx.y rather than recovered from a flat
/// index.  A `idx / cols` on every element is a 64-bit integer division -- ~20 instructions of
/// issue per element -- which dominated this kernel (it ran at 50% of DRAM peak with 95%
/// occupancy, i.e. issue-bound, not bandwidth-bound).
__global__ void split_a_kernel(float const *__restrict__ x,
                               float *__restrict__ a3,
                               int const *__restrict__ src_index,
                               unsigned char const *__restrict__ is_res,
                               int I_U,
                               int cols) {
  SplitF32 split;
  float const *__restrict__ xr = x + static_cast<long long>(blockIdx.y) * I_U;
  float *__restrict__ ar = a3 + static_cast<long long>(blockIdx.y) * cols;
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x) {
    cutlass::Array<cutlass::tfloat32_t, 2> const r = split(xr[src_index[col]]);
    ar[col] = is_res[col] ? static_cast<float>(r[1]) : static_cast<float>(r[0]);
  }
}

__global__ void split_b_kernel(float const *__restrict__ w,
                               float *__restrict__ b3,
                               int const *__restrict__ src_base,
                               int const *__restrict__ p_of_col,
                               unsigned char const *__restrict__ is_res,
                               CgVals cg,
                               int cols) {
  SplitF32 split;
  int const n = blockIdx.y;
  float *__restrict__ br = b3 + static_cast<long long>(n) * cols;
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x) {
    float const v = cg.v[p_of_col[col]] * w[src_base[col] + n];
    cutlass::Array<cutlass::tfloat32_t, 2> const r = split(v);
    br[col] = is_res[col] ? static_cast<float>(r[1]) : static_cast<float>(r[0]);
  }
}

/// M3-2: both tf32 halves of B in a single pass over W.  The K-concatenated `split_b_kernel` reads
/// W (and rescales it) twice per element because its two "big" segments are duplicates; here the
/// two outputs come off one product.
__global__ void pack_b_split_kernel(float const *__restrict__ w,
                                    float *__restrict__ b_big,
                                    float *__restrict__ b_res,
                                    int const *__restrict__ src_base,
                                    int const *__restrict__ p_of_col,
                                    CgVals cg,
                                    int cols,
                                    SplitAny split) {
  int const n = blockIdx.y;
  float const *__restrict__ wn = w + n;
  float *__restrict__ big = b_big + static_cast<long long>(n) * cols;
  float *__restrict__ res = b_res + static_cast<long long>(n) * cols;
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x) {
    cutlass::Array<cutlass::tfloat32_t, 2> const r = split(cg.v[p_of_col[col]] * wn[src_base[col]]);
    big[col] = static_cast<float>(r[0]);
    res[col] = static_cast<float>(r[1]);
  }
}

/// (x-blocks, rows) launch geometry for the two kernels above.
///
/// gridDim.y carries the row index (picking it out of a flat index costs a 64-bit division per
/// element).  gridDim.x must not be so large that every block retires after one iteration --
/// measured on LARGE_P3/B5888, going from a handful of long-lived blocks to 187 one-iteration
/// blocks dropped achieved occupancy from 95% to 74% and cost 15% of the kernel's time.  So the
/// x-extent is capped and the kernels grid-stride over the columns.
inline dim3 grid_for(int cols, int rows) {
  int cap = kPrepBlocksX;
  int blocks = (cols + kThreads - 1) / kThreads;
  if (blocks > cap) blocks = cap;
  if (blocks < 1) blocks = 1;
  return dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(rows));
}

}  // namespace

void launch_split_a(float const *x, float *a3, int const *src_index,
                    unsigned char const *is_res, long long cols, int I, int U, int B,
                    cudaStream_t stream) {
  if (B == 0 || cols == 0) return;
  split_a_kernel<<<grid_for(static_cast<int>(cols), B), kThreads, 0, stream>>>(
      x, a3, src_index, is_res, I * U, static_cast<int>(cols));
}

void launch_split_b(float const *w, float *b3, int const *src_base, int const *p_of_col,
                    unsigned char const *is_res, CgVals cg, long long cols, int P, int U, int V,
                    cudaStream_t stream) {
  if (V == 0 || cols == 0) return;
  if (nullptr) {
    fprintf(stderr, "[split_b] V=%d cols=%lld (%lld bytes)\n", V, cols, (long long)V * cols * 4);
  }
  split_b_kernel<<<grid_for(static_cast<int>(cols), V), kThreads, 0, stream>>>(
      w, b3, src_base, p_of_col, is_res, cg, static_cast<int>(cols));
}

void launch_pack_b_split(float const *w, float *b_big, float *b_res, int const *src_base,
                         int const *p_of_col, CgVals cg, long long cols, int P, int U, int V,
                         bool rn, cudaStream_t stream) {
  (void)P;
  (void)U;
  if (V == 0 || cols == 0) return;
  SplitAny split{SplitF32{}, SplitF32Rn{}, rn};
  pack_b_split_kernel<<<grid_for(static_cast<int>(cols), V), kThreads, 0, stream>>>(
      w, b_big, b_res, src_base, p_of_col, cg, static_cast<int>(cols), split);
}

}  // namespace eqlin
