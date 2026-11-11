/***************************************************************************************************
 * fasteq eqlinear backends/fp32/prep_bwd.cu
 **************************************************************************************************/
#include "prep_bwd.h"

namespace eqlin_fp32 {
namespace {

constexpr int kTile = 32;     // both axes; dim and K are multiples of it for our shapes
constexpr int kVec = 4;       // float4 on the read (along dim) and the write (along k)
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 8192;

__global__ void pack_cgw_kernel(float const *__restrict__ w, float *__restrict__ cgw, CgVals cg,
                                long long uv, long long total) {
  for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
       idx += (long long)gridDim.x * blockDim.x) {
    long long const p = idx / uv;
    cgw[idx] = cg.v[p] * w[idx];
  }
}

/// 32x32 tile transpose of `[B, rows, cols]` into one `[cols, B*rows]` slab, float4 on both
/// sides.  Read side: thread (j, i4) takes k = k0+j and four consecutive `cols` elements --
/// coalesced along `cols`, the source's fast axis.  Write side: thread (i, j4) takes u = u0+i and
/// four consecutive k -- coalesced along k, the destination's fast axis.  The tile sits in shared
/// memory as [u][k] with a +1 pad, so neither side hits bank conflicts.  This is the design the
/// tf32 line measured at 2.5 TB/s (`../cutlass_impl/csrc/transpose_kt.cu`); the M7 version tiled
/// per (path, batch item) and only reached 230 GB/s because rows=7 left most threads idle.
__global__ void transpose_kt_kernel(float const *__restrict__ src, float *__restrict__ dst,
                                    TransposeDesc const *__restrict__ descs, int B,
                                    long long src_batch_stride) {
  TransposeDesc const d = descs[blockIdx.z];
  if (d.rows <= 0 || d.cols <= 0) return;
  long long const K = static_cast<long long>(B) * d.rows;
  int const u0 = blockIdx.x * kTile;
  int const k0 = blockIdx.y * kTile;
  if (u0 >= d.cols || k0 >= K) return;

  __shared__ float tile[kTile][kTile + 1];

  int const tid = threadIdx.x;
  // 32 k-rows x 8 float4 groups = 256 units; each row's four elements come from one (b, t).
  for (int idx = tid; idx < kTile * (kTile / kVec); idx += kThreads) {
    int const j = idx / (kTile / kVec);
    int const i4 = idx % (kTile / kVec);
    long long const k = k0 + j;
    int const u = u0 + i4 * kVec;
    if (k < K && u < d.cols) {
      int const b = static_cast<int>(k / d.rows);
      int const t = static_cast<int>(k - static_cast<long long>(b) * d.rows);
      float4 const v = *reinterpret_cast<float4 const *>(
          src + static_cast<long long>(b) * src_batch_stride + d.src_off +
          static_cast<long long>(t) * d.cols + u);
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
    if (u < d.cols && k + kVec <= K) {
      *reinterpret_cast<float4 *>(dst + d.dst_off + static_cast<long long>(u) * K + k) =
          float4{tile[i][j4 * kVec], tile[i][j4 * kVec + 1], tile[i][j4 * kVec + 2],
                 tile[i][j4 * kVec + 3]};
    }
  }
}

inline int blocks_for(long long total) {
  long long blocks = (total + kThreads - 1) / kThreads;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  if (blocks < 1) blocks = 1;
  return static_cast<int>(blocks);
}

inline int grid_dim(int n) { return (n + kTile - 1) / kTile; }

}  // namespace

void launch_pack_cgw(float const *w, float *cgw, CgVals cg, int P, int U, int V,
                     cudaStream_t stream) {
  long long const total = (long long)P * U * V;
  if (total == 0) return;
  pack_cgw_kernel<<<blocks_for(total), kThreads, 0, stream>>>(w, cgw, cg,
                                                             (long long)U * V, total);
}

void launch_transpose_x(float const *x, float *xt, TransposeDesc const *descs, int num_desc, int B,
                        int src_batch_stride, int max_rows, int U, cudaStream_t stream) {
  if (num_desc == 0 || B == 0 || max_rows == 0) return;
  long long const kmax = static_cast<long long>(B) * max_rows;
  dim3 const grid(grid_dim(U), static_cast<int>((kmax + kTile - 1) / kTile), num_desc);
  transpose_kt_kernel<<<grid, kThreads, 0, stream>>>(x, xt, descs, B, src_batch_stride);
}

void launch_transpose_dy(float const *dy, float *yt, TransposeDesc const *descs, int num_desc,
                         int B, int src_batch_stride, int max_rows, int V,
                         cudaStream_t stream) {
  if (num_desc == 0 || B == 0 || max_rows == 0) return;
  long long const kmax = static_cast<long long>(B) * max_rows;
  dim3 const grid(grid_dim(V), static_cast<int>((kmax + kTile - 1) / kTile), num_desc);
  transpose_kt_kernel<<<grid, kThreads, 0, stream>>>(dy, yt, descs, B, src_batch_stride);
}

}  // namespace eqlin_fp32
