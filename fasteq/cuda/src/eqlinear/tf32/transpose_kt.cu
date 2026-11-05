/***************************************************************************************************
 * fasteq eqlinear tf32/csrc/transpose_kt.cu
 **************************************************************************************************/
#include "transpose_kt.h"

#include "cutlass/numeric_conversion.h"

namespace eqlin {
namespace {

constexpr int kTile = 32;        // both axes; K and dim are always multiples of it for our shapes
constexpr int kThreads = 256;
constexpr int kVec = 4;          // float4 on both the read (along dim) and the write (along k)

/// 32x32 tile transpose, float4 on both sides.
///
/// Read side: thread (j, i4) takes k = k0+j and four consecutive `dim` elements — coalesced along
/// `dim`, which is the source's fast axis.  Write side: thread (i, j4) takes u = u0+i and four
/// consecutive k — coalesced along k, the destination's fast axis.  The tile sits in shared memory
/// as [u][k] with a +1 pad, so neither side hits bank conflicts.
__global__ void transpose_kt_kernel(float const *__restrict__ src, float *__restrict__ dst_big,
                                    float *__restrict__ dst_res,
                                    KtPath const *__restrict__ paths, int B, int dim,
                                    long long row_stride) {
  KtPath const p = paths[blockIdx.z];
  long long const K = static_cast<long long>(B) * p.i_g;
  int const u0 = blockIdx.x * kTile;
  int const k0 = blockIdx.y * kTile;
  if (u0 >= dim || k0 >= K) return;

  __shared__ float tile[kTile][kTile + 1];

  int const tid = threadIdx.x;
  // 32 k-rows x 8 float4 groups = 256 units; each row's four elements come from one (b, t).
  for (int idx = tid; idx < kTile * (kTile / kVec); idx += kThreads) {
    int const j = idx / (kTile / kVec);
    int const i4 = idx % (kTile / kVec);
    long long const k = k0 + j;
    int const u = u0 + i4 * kVec;
    if (k < K && u < dim) {
      int const b = static_cast<int>(k / p.i_g);
      int const t = static_cast<int>(k - static_cast<long long>(b) * p.i_g);
      float4 const v = *reinterpret_cast<float4 const *>(
          src + static_cast<long long>(b) * row_stride +
          static_cast<long long>(p.run_off + t) * dim + u);
      tile[u - u0][j] = v.x;
      tile[u - u0 + 1][j] = v.y;
      tile[u - u0 + 2][j] = v.z;
      tile[u - u0 + 3][j] = v.w;
    }
  }
  __syncthreads();

  for (int idx = tid; idx < kTile * (kTile / kVec); idx += kThreads) {
    int const i = idx / (kTile / kVec);
    int const j4 = idx % (kTile / kVec);
    int const u = u0 + i;
    long long const k = k0 + j4 * kVec;
    if (u < dim && k + kVec <= K) {
      long long const o = p.dst_off + static_cast<long long>(u) * K + k;
      if (dst_res == nullptr) {
        *reinterpret_cast<float4 *>(dst_big + o) =
            float4{tile[i][j4 * kVec], tile[i][j4 * kVec + 1], tile[i][j4 * kVec + 2],
                   tile[i][j4 * kVec + 3]};
      } else {
        cutlass::NumericConverterFastF32<> conv;
        float out_b[kVec], out_r[kVec];
        for (int e = 0; e < kVec; ++e) {
          cutlass::Array<cutlass::tfloat32_t, 2> const r = conv(tile[i][j4 * kVec + e]);
          out_b[e] = static_cast<float>(r[0]);
          out_r[e] = static_cast<float>(r[1]);
        }
        *reinterpret_cast<float4 *>(dst_big + o) = float4{out_b[0], out_b[1], out_b[2], out_b[3]};
        *reinterpret_cast<float4 *>(dst_res + o) = float4{out_r[0], out_r[1], out_r[2], out_r[3]};
      }
    }
  }
}

}  // namespace

void launch_transpose_kt(float const *src, float *dst_big, float *dst_res, KtPath const *paths,
                         int num_paths, int B, int dim, long long src_row_stride, int k_tiles_max,
                         cudaStream_t stream) {
  if (num_paths <= 0 || k_tiles_max <= 0 || dim <= 0) return;
  dim3 const grid((dim + kTile - 1) / kTile, k_tiles_max, num_paths);
  transpose_kt_kernel<<<grid, kThreads, 0, stream>>>(src, dst_big, dst_res, paths, B, dim,
                                                     src_row_stride);
}

}  // namespace eqlin
