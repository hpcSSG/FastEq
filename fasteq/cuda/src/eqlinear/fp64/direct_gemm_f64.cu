/***************************************************************************************************
 * fasteq eqlinear fp64/csrc/direct_gemm_f64.cu
 *
 * The M5-B kernel: a persistent grouped GEMM with 4-warp CTAs (3 per SM), the composite 32x32x16
 * FP64 DMMA atom, 8-byte cp.async staging in a fragment-major smem layout, and register->global
 * epilogue stores (no epilogue smem, which is what keeps 3 CTAs/SM possible).
 *
 * A is addressed through the composed K axis (path block `p = k/u`, inside-block `k%u`), so the
 * multi-path shapes read x in place exactly like the 3.x path, with no permute.
 **************************************************************************************************/
#include "direct_gemm_f64.cuh"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include <cuda_pipeline.h>

#include "collective/fp64_dmma_atom.hpp"
#include "device_buffer.h"

namespace eqlin {
namespace {

using eqlin::fp64::Op32x32x16;

constexpr int kTB_M = 64, kTB_N = 64, kBK = 16;
constexpr int kThreads = 128;
constexpr int kStages = 4;
constexpr int kCtas = 3;
// Upper-bound probe (results WRONG, timing only): skip the steady-state fill to measure what a free
// fill would be worth.  Tells us whether a TMA-based fill is worth building at all.
#ifndef F64D_NOFILL
#define F64D_NOFILL 0
#endif
// Probe 2: same 16 cp.async per thread (same MIO/issue cost, same smem writes) but all reading one
// L2-resident address, so the fill's DRAM traffic disappears.  Results are wrong.
#ifndef F64D_FAKEFILL
#define F64D_FAKEFILL 0
#endif
constexpr int kABytes = kTB_M * kBK * 8;                 // 8192
constexpr int kBBytes = kTB_N * kBK * 8;                 // 8192
constexpr int kStageBytes = kABytes + kBBytes;           // 16384
constexpr int kSmemBytes = kStages * kStageBytes;        // 65536

struct Problem {
  double const *a;
  double const *b;
  double *c;
  int M, N, K;
  int u;          // 0 = flat K (packed A); else the path-block length for the composed A
  int m_stride;   // element stride between path blocks (composed A only)
  int b_off;      // (a's column offset is folded into the pointer by bind_a, like the other backends)
  int lda, ldb, ldc;
};

struct Tile {
  int problem;
  int mt, nt;
};

// ---- smem element offsets (bytes) inside a warp block, see the header comment --------------------
__device__ __forceinline__ int a_off(int m, int k) {
  int const mw = m & 31;
  return 512 * ((k >> 2) + 4 * (mw >> 4)) + 16 * ((mw & 7) * 4 + (k & 3)) + 8 * ((mw & 15) >> 3);
}
__device__ __forceinline__ int b_off(int n, int k) {
  int const nw = n & 31;
  return 512 * (2 * (nw >> 3) + ((k & 15) >> 3)) + 16 * ((nw & 7) * 4 + (k & 3)) +
         8 * ((k & 7) >> 2);
}

/// The per-lane offsets reduce to a few scalars: the read offsets are affine in the chunk index and
/// the staging offsets are affine in the element index (i>>2, (i>>1)&1, i&1), so nothing has to be
/// kept in an array (which spilled 88 B of local memory in the first version).
struct LanePlan {
  int a_read;   // + 512*v1 + 2048*v2
  int b_read;   // + 512*chunk
  int a_fill0;  // + 4096*(i>>2) + 2048*((i>>1)&1) + 8*(i&1)
  int b_fill0;  // + 4096*(i>>2) + 1024*(i&3)
  int kk;       // staged element's k index (lane & 15)
};

__device__ LanePlan make_lane_plan(int tid) {
  LanePlan L{};
  int const lane = tid & 31;
  int const warp = tid >> 5;
  int const wm = warp >> 1, wn = warp & 1;
  int const m0 = lane >> 2, k0 = lane & 3;
  int const kk = lane & 15;
  int const base = 2 * warp + (lane >> 4);  // row (A) / column (B) of the lane's first element

  L.a_read = 16 * (m0 * 4 + k0) + wm * 4096;
  L.b_read = kABytes + 16 * ((lane >> 2 & 7) * 4 + k0) + wn * 4096;
  L.kk = kk;
  // a_fill(i) = block(i>>2)*4096 + 512*(kk>>2) + 2048*((i>>1)&1) + 16*((base&7)*4 + (kk&3)) + 8*(i&1)
  L.a_fill0 = 512 * (kk >> 2) + 16 * ((base & 7) * 4 + (kk & 3));
  // b_fill(i) = block(i>>2)*4096 + kABytes + 512*(2*(i&3) + ((kk&15)>>3))
  //             + 16*((base&7)*4 + (kk&3)) + 8*((kk&7)>>2)
  L.b_fill0 = kABytes + 512 * ((kk & 15) >> 3) + 16 * ((base & 7) * 4 + (kk & 3)) +
              8 * ((kk & 7) >> 2);
  return L;
}

__device__ __forceinline__ int a_fill_off(LanePlan const &L, int i) {
  return L.a_fill0 + 4096 * (i >> 2) + 2048 * ((i >> 1) & 1) + 8 * (i & 1);
}
__device__ __forceinline__ int b_fill_off(LanePlan const &L, int i) {
  return L.b_fill0 + 4096 * (i >> 2) + 1024 * (i & 3);
}

/// Fill one k-tile (1024 A + 1024 B elements over 128 threads) with 8-byte cp.async.
/// `a_row`/`b_row` are the per-lane row pointers; only the k offset changes with `kt`.
__device__ __forceinline__ void fill_stage(char *smem, int stage, Problem const &p,
                                           LanePlan const &L, int kt,
                                           double const *const (&a_row)[8],
                                           double const *const (&b_row)[8]) {
  char *base = smem + stage * kStageBytes;
  int const kg = kt * kBK;
  int const a_col = p.u ? ((kg / p.u) * p.m_stride + (kg % p.u)) : kg;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    double const *asrc = F64D_FAKEFILL ? (a_row[0] + L.kk) : (a_row[i] + a_col + L.kk);
    double const *bsrc = F64D_FAKEFILL ? (b_row[0] + L.kk) : (b_row[i] + p.b_off + kg + L.kk);
    __pipeline_memcpy_async(base + a_fill_off(L, i), asrc, 8);
    __pipeline_memcpy_async(base + b_fill_off(L, i), bsrc, 8);
  }
  __pipeline_commit();
}

/// One k8 half of a BK=16 stage: 4 LDS.128 for A and 4 for B.  The half is just the two k-blocks
/// (k/4 in {2h, 2h+1}) of the existing k16 smem layout, so no layout change is needed.
__device__ __forceinline__ void load_half(char const *smem, int stage, LanePlan const &L, int h,
                                          double (&a)[2][4], double (&b)[4][2]) {
  char const *base = smem + stage * kStageBytes;
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int v1 = 0; v1 < 2; ++v1) {
      double2 v = *reinterpret_cast<double2 const *>(base + L.a_read + 1024 * h + 512 * v1 +
                                                     2048 * mi);
      a[mi][2 * v1 + 0] = v.x;  // (m0 + 16*mi,     k0 + 4*v1 + 8*h)
      a[mi][2 * v1 + 1] = v.y;  // (m0 + 16*mi + 8, k0 + 4*v1 + 8*h)
    }
#pragma unroll
  for (int ni = 0; ni < 4; ++ni) {
    double2 v = *reinterpret_cast<double2 const *>(base + L.b_read + 512 * (2 * ni + h));
    b[ni][0] = v.x;  // (n0 + 8*ni, k0 + 8*h)
    b[ni][1] = v.y;  // (n0 + 8*ni, k0 + 4 + 8*h)
  }
}

/// 8 sub-atom k8 DMMAs on fragments already in registers.
__device__ __forceinline__ void mma_half(double const (&a)[2][4], double const (&b)[4][2],
                                         double (&c)[8][4]) {
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      int const ch = mi + 2 * ni;
      cute::SM90::MMA_16x8x8_F64F64F64F64_TN::fma(
          c[ch][0], c[ch][1], c[ch][2], c[ch][3], a[mi][0], a[mi][1], a[mi][2], a[mi][3],
          b[ni][0], b[ni][1], c[ch][0], c[ch][1], c[ch][2], c[ch][3]);
    }
}

/// One BK=16 stage as two k8 halves: the second half's 8 LDS are issued before the first half's
/// 8 DMMAs, so their latency hides under the MMAs instead of stalling on them (the k16 atom has no
/// such second chunk inside a stage).  Bitwise identical to the k16 atom -- checked in the CP0
/// spike: 0 differing accumulators over a 32x32x16 tile.
__device__ __forceinline__ void mma_stage(char const *smem, int stage, LanePlan const &L,
                                          double (&c)[8][4]) {
  double a0[2][4], b0[4][2], a1[2][4], b1[4][2];
  load_half(smem, stage, L, 0, a0, b0);
  load_half(smem, stage, L, 1, a1, b1);
  mma_half(a0, b0, c);
  mma_half(a1, b1, c);
}

/// BK=16 variant: load the original k16 fragments and issue one DMMA per sub-atom.
__device__ __forceinline__ void mma_stage_k16(char const *smem, int stage, LanePlan const &L,
                                              double (&c)[8][4]) {
  char const *base = smem + stage * kStageBytes;
  double a[2][8], b[4][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int v1 = 0; v1 < 4; ++v1) {
      double2 v = *reinterpret_cast<double2 const *>(base + L.a_read + 512 * v1 + 2048 * mi);
      a[mi][2 * v1] = v.x;
      a[mi][2 * v1 + 1] = v.y;
    }
#pragma unroll
  for (int ni = 0; ni < 4; ++ni)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      double2 v = *reinterpret_cast<double2 const *>(base + L.b_read + 512 * (2 * ni + h));
      b[ni][2 * h] = v.x;
      b[ni][2 * h + 1] = v.y;
    }
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      int const ch = mi + 2 * ni;
      cute::SM90::MMA_16x8x16_F64F64F64F64_TN::fma(
          c[ch][0], c[ch][1], c[ch][2], c[ch][3], a[mi][0], a[mi][1], a[mi][2], a[mi][3],
          a[mi][4], a[mi][5], a[mi][6], a[mi][7], b[ni][0], b[ni][1], b[ni][2], b[ni][3],
          c[ch][0], c[ch][1], c[ch][2], c[ch][3]);
    }
}

__device__ __forceinline__ void store_tile(Problem const &p, int m0, int n0, int warp,
                                           double const (&c)[8][4]) {
  int const lane = threadIdx.x & 31;
  int const wm = warp >> 1, wn = warp & 1;
  int const cm = m0 + 32 * wm + (lane >> 2);
  int const cn = n0 + 32 * wn + 2 * (lane & 3);
  for (int chunk = 0; chunk < 8; ++chunk) {
    int const mi = chunk & 1, ni = chunk >> 1;
    for (int j = 0; j < 4; j += 2) {
      int const m = cm + 8 * (j >> 1) + 16 * mi;
      int const n = cn + (j & 1) + 8 * ni;
      if (m < p.M && n + 1 < p.N) {
        double2 v{c[chunk][j], c[chunk][j + 1]};
        *reinterpret_cast<double2 *>(p.c + static_cast<long long>(m) * p.ldc + n) = v;
      } else if (m < p.M && n < p.N) {
        p.c[static_cast<long long>(m) * p.ldc + n] = c[chunk][j];
      }
    }
  }
}

template <bool UseK8>
__global__ void __launch_bounds__(kThreads, kCtas)
direct_kernel(Problem const *__restrict__ probs, int num_probs, Tile const *__restrict__ tiles,
              int num_tiles, unsigned *__restrict__ counter) {
  extern __shared__ char smem[];
  int const tid = threadIdx.x;
  int const warp = tid >> 5;
  LanePlan const L = make_lane_plan(tid);

  // The work counter must be advanced once per *block*: a per-thread atomicAdd would give every
  // lane a different tile and the __syncthreads() in the loop would deadlock.
  __shared__ int s_tile;
  for (;;) {
    if (tid == 0) s_tile = static_cast<int>(atomicAdd(counter, 1u));
    __syncthreads();
    int const t = s_tile;
    if (t >= num_tiles) break;
    Tile const tl = tiles[t];
    if (tl.problem >= num_probs) continue;
    Problem const p = probs[tl.problem];
    int const m0 = tl.mt * kTB_M, n0 = tl.nt * kTB_N;
    int const k_tiles = p.K / kBK;

    double c[8][4];
    for (int i = 0; i < 8; ++i)
      for (int q = 0; q < 4; ++q) c[i][q] = 0.0;

    // Per-tile row pointers (clamped: the padded rows' C is never stored, so their garbage cannot
    // leak into a stored element).
    // Staged element i of this lane sits on row base + 8*i (see make_lane_plan).
    int const base_row = 2 * warp + (tid >> 4 & 1);
    int const row0_a = min(m0 + base_row, p.M - 1), row0_b = min(n0 + base_row, p.N - 1);
    double const *a_row[8], *b_row[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      a_row[i] = p.a + static_cast<long long>(min(row0_a + 8 * i, p.M - 1)) * p.lda;
      b_row[i] = p.b + static_cast<long long>(min(row0_b + 8 * i, p.N - 1)) * p.ldb;
    }

    for (int s = 0; s < kStages - 1; ++s) {
      if (s < k_tiles) fill_stage(smem, s, p, L, s, a_row, b_row);
      else __pipeline_commit();
    }
    for (int kt = 0; kt < k_tiles; ++kt) {
      __pipeline_wait_prior(kStages - 2);
      __syncthreads();
      int const pf = kt + kStages - 1;
      if (F64D_NOFILL) {
        __pipeline_commit();
      } else if (pf < k_tiles) {
        // Launch the next global-to-shared transfer before the current tile's DMMAs so the
        // asynchronous copy can progress while the warp consumes this stage.
        fill_stage(smem, pf % kStages, p, L, pf, a_row, b_row);
      } else {
        __pipeline_commit();
      }
      if constexpr (UseK8)
        mma_stage(smem, kt % kStages, L, c);
      else
        mma_stage_k16(smem, kt % kStages, L, c);
    }
    __pipeline_wait_prior(0);
    __syncthreads();
    store_tile(p, m0, n0, warp, c);
    __syncthreads();  // before the next tile reuses the ring
  }
}

}  // namespace

struct DirectGemmPlan::ImplBase {
  virtual ~ImplBase() = default;
  virtual int launch(double *out, cudaStream_t stream) = 0;
  virtual void bind_a(double const *a_base, cudaStream_t stream) = 0;
};

namespace {

class PlanImpl : public DirectGemmPlan::ImplBase {
 public:
  PlanImpl(int B, int V, int num_rows, long long lda, long long ldb, long long ldc,
           std::vector<RowPlan> const &rows, double const *a_data, double const *b_data, int u,
           bool use_k8)
      : num_rows_(num_rows), V_(V), ldc_(ldc), rows_(rows), u_(u), use_k8_(use_k8) {
    if (num_rows <= 0) throw std::invalid_argument("DirectGemmPlan: no rows");
    if (b_data == nullptr) throw std::invalid_argument("DirectGemmPlan: missing B");

    host_problem_.resize(num_rows);
    host_c_.resize(num_rows);
    int num_tiles = 0;
    for (int j = 0; j < num_rows; ++j) {
      RowPlan const &r = rows[j];
      if (r.K % kBK != 0) throw std::invalid_argument("DirectGemmPlan: K must be a multiple of 16");
      Problem &p = host_problem_[j];
      p = Problem{};
      p.a = nullptr;  // filled by bind_a
      p.b = b_data + r.b_off;
      p.c = nullptr;  // filled per launch
      p.M = r.M ? r.M : B;
      p.N = r.N ? r.N : V;
      p.K = r.K;
      p.u = u_;
      p.m_stride = static_cast<int>(r.m_stride);
      p.b_off = 0;  // folded into p.b
      p.lda = static_cast<int>(r.lda ? r.lda : lda);
      p.ldb = static_cast<int>(r.ldb ? r.ldb : ldb);
      p.ldc = static_cast<int>(r.ldc ? r.ldc : ldc);
      int const mt = (p.M + kTB_M - 1) / kTB_M;
      int const nt = (p.N + kTB_N - 1) / kTB_N;
      num_tiles += mt * nt;
      tiles_per_row_.push_back(mt * nt);
    }

    host_tiles_.reserve(num_tiles);
    for (int j = 0; j < num_rows; ++j) {
      int const mt = (host_problem_[j].M + kTB_M - 1) / kTB_M;
      int const nt = (host_problem_[j].N + kTB_N - 1) / kTB_N;
      for (int a = 0; a < mt; ++a)
        for (int b = 0; b < nt; ++b) host_tiles_.push_back(Tile{j, a, b});
    }
    num_tiles_ = num_tiles;

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, dev);
    grid_ = std::min(num_tiles_, 3 * sm_count_);
    if (grid_ <= 0) throw std::invalid_argument("DirectGemmPlan: empty grid");
    // 64 KB of dynamic smem needs the opt-in; without it the launch fails with "invalid argument".
    auto kernel = use_k8_ ? direct_kernel<true> : direct_kernel<false>;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes) !=
        cudaSuccess) {
      throw std::runtime_error("DirectGemmPlan: cannot raise the dynamic smem limit");
    }

    if (d_problem_.reset(num_rows * sizeof(Problem)) ||
        d_tiles_.reset(num_tiles_ * sizeof(Tile)) || d_c_.reset(num_rows * sizeof(double *)) ||
        d_counter_.reset(sizeof(unsigned))) {
      throw std::runtime_error("DirectGemmPlan: device allocation failed");
    }
    cudaMemcpy(d_problem_.as<Problem>(), host_problem_.data(), num_rows * sizeof(Problem),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_tiles_.as<Tile>(), host_tiles_.data(), num_tiles_ * sizeof(Tile),
               cudaMemcpyHostToDevice);
    if (a_data) bind_a(a_data, nullptr);
    if (nullptr) {
      fprintf(stderr, "[direct-plan] rows=%d tiles=%d grid=%d u=%d\n", num_rows_, num_tiles_, grid_,
              u_);
    }
  }

  ~PlanImpl() override = default;

  void bind_a(double const *a_base, cudaStream_t stream) override {
    if (a_base == a_base_ && a_bound_) return;
    a_base_ = a_base;
    for (int j = 0; j < num_rows_; ++j) host_problem_[j].a = a_base + rows_[j].a_off;
    cudaMemcpyAsync(d_problem_.as<Problem>(), host_problem_.data(), num_rows_ * sizeof(Problem),
                    cudaMemcpyHostToDevice, stream);
    a_bound_ = true;
  }

  int launch(double *out, cudaStream_t stream) override {
    if (!a_bound_) return 1;
    for (int j = 0; j < num_rows_; ++j)
      host_problem_[j].c = out + (rows_[j].c_off >= 0 ? rows_[j].c_off
                                                       : static_cast<long long>(j) * V_);
    cudaMemcpyAsync(d_problem_.as<Problem>(), host_problem_.data(), num_rows_ * sizeof(Problem),
                    cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_counter_.as<unsigned>(), 0, sizeof(unsigned), stream);
    if (use_k8_)
      direct_kernel<true><<<grid_, kThreads, kSmemBytes, stream>>>(
          d_problem_.as<Problem>(), num_rows_, d_tiles_.as<Tile>(), num_tiles_,
          d_counter_.as<unsigned>());
    else
      direct_kernel<false><<<grid_, kThreads, kSmemBytes, stream>>>(
          d_problem_.as<Problem>(), num_rows_, d_tiles_.as<Tile>(), num_tiles_,
          d_counter_.as<unsigned>());
    return 0;
  }

 private:
  int num_rows_, V_, num_tiles_ = 0, grid_ = 0, sm_count_ = 0, u_ = 0;
  bool use_k8_;
  long long ldc_;
  std::vector<RowPlan> rows_;
  std::vector<Problem> host_problem_;
  std::vector<Tile> host_tiles_;
  std::vector<int> tiles_per_row_;
  std::vector<double *> host_c_;
  double const *a_base_ = nullptr;
  bool a_bound_ = false;
  DeviceBuffer d_problem_, d_tiles_, d_c_, d_counter_;
};

}  // namespace

DirectGemmPlan::DirectGemmPlan(int B, int V, int num_rows, long long lda, long long ldb,
                               long long ldc, std::vector<RowPlan> const &rows,
                               double const *a_data, double const *b_data, int u, bool use_k8) {
  p_ = std::make_unique<PlanImpl>(B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data, u,
                                  use_k8);
}

DirectGemmPlan::~DirectGemmPlan() = default;

int DirectGemmPlan::launch(double *out, cudaStream_t stream) { return p_->launch(out, stream); }

void DirectGemmPlan::bind_a(double const *a_base, cudaStream_t stream) {
  p_->bind_a(a_base, stream);
}

}  // namespace eqlin
