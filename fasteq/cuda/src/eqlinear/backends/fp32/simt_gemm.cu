/***************************************************************************************************
 * fasteq eqlinear backends/fp32/simt_gemm.cu
 *
 * M2: hand-written grouped SIMT GEMM (impl 1).
 *
 * Shape-selected FP32 FMA chains, with compensated merging on the accuracy path. The single-chain
 * specialization retains the original ascending-K arithmetic. The tile is chosen per shape (exact-fit N),
 * and the accumulator/operand register budget is the only thing the compiler has to allocate -- no
 * stock iterator or epilogue state -- so a 96-128 accumulator thread tile still fits a CTA with
 * 2-3 blocks per SM instead of the stock path's 255 registers.
 *
 * Shared-memory layout: the operands stay K-contiguous (that is what cp.async can copy) and each
 * row is padded from 16 to 20 floats. Eight interleaved M-lanes address distinct bank groups.
 **************************************************************************************************/
#include "simt_gemm.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "device_buffer.h"
#include "numeric_policy.h"
#include "tiny_linear.cuh"
#include "tma_desc.h"

namespace eqlin_fp32 {
namespace {

constexpr int kK = 16;       // K-tile
constexpr int kPad = 4;      // row padding, in floats: 20 floats = 80 B per row
constexpr int kLds = kK + kPad;

/// One shared-memory layout for the device kernel and both host plan builders.
template <class Cfg, int Stages, bool Fused>
struct SharedPlan {
  static constexpr int kElements = (Cfg::kWarpM / 8) * (Cfg::kWarpN / 4);
  static constexpr int kPayload = (kElements + 3) & ~3;
  static constexpr int kStride = kPayload + 4;
  static constexpr int kOperands = Stages * (Cfg::kTbM + Cfg::kTbN) * kLds;
  static constexpr int kSlab = Fused ? Cfg::kThreads * kStride : 0;
  static constexpr int kFloats = kOperands + kSlab;
  static constexpr int kBarrierOffset = (kFloats + 1) & ~1;
};

/// Bank layout.  A warp's 8 M-lanes own rows `lane_m + 8*r` (interleaved, not contiguous blocks)
/// and its 4 N-lanes own columns `lane_n + 4*j`, so within one load instruction the lanes differ
/// by exactly one row.  With a 20-float row stride that puts them on banks 20*lane mod 32 =
/// 0,20,8,28,16,4,24,12 -- all eight 4-bank groups, conflict-free.  (The earlier contiguous
/// mapping put the lanes 8 rows apart, where 8*20 = 160 == 0 mod 32, and every load replayed.)
__device__ __forceinline__ int smem_off(int row, int k) { return row * kLds + k; }

__device__ __forceinline__ void cp_async16(void *smem_ptr, void const *gmem_ptr) {
  unsigned const s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ unsigned smem_addr(void const *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

/// One 2-D TMA tile: `map` describes [rows x cols] fp32 with a fixed row stride; the tile is the
/// (box_rows x box_cols) box at origin (c0, r0).  `.tile` preserves the source layout, so a box of
/// kLds floats lands as kLds-float rows -- exactly the padded layout the LDS code above assumes
/// (verified bit-exact by the M6 probe; see csrc/tma_desc.h).
__device__ __forceinline__ void tma_2d(void *dst, CUtensorMap const *map, int c0, int r0,
                                      uint64_t *bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(smem_addr(dst)),
      "l"(map), "r"(c0), "r"(r0), "r"(smem_addr(bar)));
}

__device__ __forceinline__ void mbar_init(uint64_t *bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect(uint64_t *bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_addr(bar)),
               "r"(bytes));
}
/// Bounded spin: a barrier bug must report instead of wedging the card (M6 hit exactly one -- the
/// ring was not re-armed between the tiles a CTA walks in the grid-stride loop).
__device__ __forceinline__ void mbar_wait(uint64_t *bar, int phase) {
  int ok = 0;
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      ".reg .u32 N;\n"
      "mov.u32 N, 0;\n"
      "MBWAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n"
      "@P bra MBDONE;\n"
      "add.u32 N, N, 1;\n"
      "setp.lt.u32 P, N, 100000000;\n"
      "@P bra MBWAIT;\n"
      "bra MBOUT;\n"
      "MBDONE:\n"
      "mov.u32 %0, 1;\n"
      "MBOUT:\n"
      "}\n"
      : "+r"(ok)
      : "r"(smem_addr(bar)), "r"(phase));
  if (!ok && threadIdx.x == 0) printf("[eqlin] mbarrier timeout (phase %d)\n", phase);
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/// Strides are plan-wide constants, and output row j lives at `out + j*c_stride` (the grouped
/// convention), so all three are scalars.  They used to be per-row arrays filled through a pinned
/// host buffer on every launch -- which raced with the async copy whenever two launches followed
/// each other quickly, and silently cost the first launch its output pointer.
struct SimtParams {
  int num_tiles;
  int3 const *problems;     // (M, N, K) per problem
  float const *a_base;      ///< A tensor base (packed buffer, or the caller's x in view mode)
  float const *const *ptr_b;
  int const *a_col;         ///< per problem: K-column origin of A, and the TMA column coordinate
  int const *b_col;         ///< per problem: K-column origin inside the packed B tensor (TMA)
  float *out;               // output row j starts at out + j*c_stride
  int c_stride;
  int lda, ldb, ldc;
  int4 const *tiles;        // (row, m_tile, n_tile, unused)
  int const *rbase;         // FUSED only: first problem index of output row j
  int const *pg;            // FUSED only: how many per-path problems output row j has

  // BWD only: per-problem addressing.  The forward path leaves all of these null and keeps using
  // the plan-wide scalars above; `if constexpr (BWD)` in the kernel eliminates the loads.
  int const *kind = nullptr;        ///< 0 = dgrad (dy/dx), 1 = wgrad (Xt/partials)
  float const *a_base0 = nullptr;   ///< dgrad A base (caller's dy)
  float const *a_base1 = nullptr;   ///< wgrad A base (Xt)
  float *c_base0 = nullptr;         ///< dgrad C base (caller's dx)
  float *c_base1 = nullptr;         ///< wgrad C base (partials)
  long long const *a_off = nullptr; ///< per problem: element offset into the kind's A base
  long long const *c_off = nullptr; ///< per problem: element offset into the kind's C base
  int const *lda_arr = nullptr;
  int const *ldb_arr = nullptr;
  int const *ldc_arr = nullptr;
  float const *scales = nullptr;  // accuracy specialization: immutable per-problem CG
};

template <int CHAINS, int TM, int TN>
__device__ __forceinline__ void fold_chains(float (&acc)[CHAINS][TM][TN]) {
  if constexpr (CHAINS > 1) {
#pragma unroll
    for (int r = 0; r < TM; ++r) {
#pragma unroll
      for (int c = 0; c < TN; ++c) {
        float sum = acc[0][r][c];
        float correction = 0.f;
#pragma unroll
        for (int h = 1; h < CHAINS; ++h) {
          float const term = acc[h][r][c];
          float const next = __fadd_rn(sum, term);
          float const recovered = __fsub_rn(next, sum);
          // TwoSum retains the rounding error when independent FP32 chains are merged.
          float const error = __fadd_rn(__fsub_rn(sum, __fsub_rn(next, recovered)),
                                        __fsub_rn(term, recovered));
          correction = __fadd_rn(correction, error);
          sum = next;
        }
        acc[0][r][c] = __fadd_rn(sum, correction);
      }
    }
  }
}

/// \param FUSED  chain the `pg` per-path problems of one output row inside one CTA, summing them
///        in smem between paths instead of in a second kernel.  Each path keeps its own fp32
///        accumulator, so the accumulation order is torch's (and identical to the split table's).
template <class Cfg, int STAGES, bool FUSED, bool TMA, bool BWD = false>
__global__ void __launch_bounds__(Cfg::kThreads, Cfg::kMinBlocks)
simt_gemm_kernel(SimtParams p, __grid_constant__ const CUtensorMap map_a,
                 __grid_constant__ const CUtensorMap map_b) {
  constexpr int kStages = STAGES;
  constexpr int TM = Cfg::kWarpM / 8;  // 8 M-lanes per warp
  constexpr int TN = Cfg::kWarpN / 4;  // 4 N-lanes per warp
  constexpr int KC = Cfg::kChunk;      // k values consumed per smem pass
  constexpr int CHAINS = Cfg::kChains;
  using Shared = SharedPlan<Cfg, STAGES, FUSED>;
  constexpr int THREADS = Cfg::kThreads;
  constexpr int A_STAGE = Cfg::kTbM * kLds;
  constexpr int B_STAGE = Cfg::kTbN * kLds;
  constexpr int STAGE_FLOATS = A_STAGE + B_STAGE;
  /// Running sum of the paths already finished, one float per element of the CTA tile.  Only the
  /// fused kernel pays for it; the plain one keeps the same occupancy it always had.
  constexpr int RES_OFF = Shared::kOperands;
  /// Running-sum slab: thread `tid` owns RES_STRIDE contiguous floats, so its TM*TN accumulators
    /// are written as float4s. Round the payload before adding one float4 padding element.
  /// the 8 lanes of a 128-bit access phase on 8 distinct bank groups.
  constexpr int RES_ELEMS = Shared::kPayload;
  constexpr int RES_STRIDE = Shared::kStride;
  /// Per-stage mbarrier (8 B each) after the operand stages and the running-sum slab, at an even
  /// float offset so the barriers are 8 B aligned.  Only the TMA mainloop uses them.
  constexpr int RES_OFF_ALIGNED = Shared::kBarrierOffset;
  /// A TMA box covers the whole padded row (kK + kPad floats), so the per-stage byte count is a
  /// compile-time constant.
  constexpr int TMA_BYTES = (Cfg::kTbM + Cfg::kTbN) * kLds * static_cast<int>(sizeof(float));

  extern __shared__ float smem[];
  uint64_t *bars = reinterpret_cast<uint64_t *>(&smem[RES_OFF_ALIGNED]);

  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp = tid >> 5;
  int const wm = warp / Cfg::kWarpsN;
  int const wn = warp % Cfg::kWarpsN;
  int const lane_m = lane / 4;  // 0..7
  int const lane_n = lane % 4;  // 0..3
  int const m_base = wm * Cfg::kWarpM + lane_m;   // row of the thread's first accumulator
  int const n_base = wn * Cfg::kWarpN + lane_n;   // and its first accumulator column

  for (int t = blockIdx.x; t < p.num_tiles; t += gridDim.x) {
    int4 const tile = p.tiles[t];
    int const row = tile.x;                       // output row: indexes ptr_c / ldc / rbase / pg
    int const base = FUSED ? p.rbase[row] : row;  // first problem index
    int const np = FUSED ? p.pg[row] : 1;         // problems chained by this tile
    int const m0 = tile.y * Cfg::kTbM;
    int const n0 = tile.z * Cfg::kTbN;
    int const Mp = p.problems[base].x;
    int const Kp = p.problems[base].z;  // K per path; the split table gives every path the same U
    int const rows_a = Mp - m0 < Cfg::kTbM ? Mp - m0 : Cfg::kTbM;

    int lda = p.lda;
    int ldb = p.ldb;
    int ldc = p.ldc;
    float const *A;
    float const *B;
    float *C;
    if constexpr (BWD) {
      // Backward row table: every problem carries its own operands and strides.  The caller's
      // dy/dx are kernel arguments (kind selects them), so no device pointer array is rewritten
      // per launch.
      int const kd = p.kind[base];
      lda = p.lda_arr[base];
      ldb = p.ldb_arr[base];
      ldc = p.ldc_arr[base];
      float const *ab = kd == 0 ? p.a_base0 : p.a_base1;
      float *cb = kd == 0 ? p.c_base0 : p.c_base1;
      A = ab + p.a_off[base] + static_cast<long long>(m0) * lda;
      B = p.ptr_b[base] + static_cast<long long>(n0) * ldb;
      C = cb + p.c_off[base] + static_cast<long long>(m0) * ldc + n0;
    } else {
      // h_a[i] was exactly a_data + a_col[i], so a scalar base plus the column table is the same
      // address -- and it is the only form that also works when A is the caller's x.
      A = p.a_base + p.a_col[base] + static_cast<long long>(m0) * lda;
      B = p.ptr_b[base] + static_cast<long long>(n0) * ldb;
      C = p.out + static_cast<long long>(row) * p.c_stride +
          static_cast<long long>(m0) * ldc + n0;
    }

    int const nk = Kp / kK;   // K-tiles per path
    int const nkt = np * nk;  // K-tiles per tile

    int a_col0 = 0, b_col0 = 0;  // path 0's K-column origins, for the TMA coordinates
    if constexpr (!BWD) {
      a_col0 = p.a_col[base];
      b_col0 = p.b_col[base];
    }

    // Operand bases of chained path `m` (path 0's are already in A/B).  Only the refill needs
    // them, and only once per path.
    int curv_a_col = a_col0, curv_b_col = b_col0;
    auto path_bases = [&](int m, float const *&pa, float const *&pb) {
      if constexpr (FUSED) {
        if (m > 0 && m < np) {
          pa = p.a_base + p.a_col[base + m] + static_cast<long long>(m0) * lda;
          pb = p.ptr_b[base + m] + static_cast<long long>(n0) * ldb;
          curv_a_col = p.a_col[base + m];
          curv_b_col = p.b_col[base + m];
        }
      }
    };

    // One stage's worth of A and B.  Both operands are K-contiguous in global memory, so the copy
    // is a flat 16 B chunk loop; the destination row is padded by kPad floats.  The commit always
    // happens, so a skipped copy keeps the group counts (and every cp_wait) in step.
    auto load_stage = [&](int stage, float const *Ag, float const *Bg, int acol, int bcol, int k0,
                          bool valid) {
      if constexpr (TMA) {
        // One instruction per operand per K-tile, issued by a single lane: the whole (TbM x kLds)
        // tile replaces (TbM + TbN) x 4 cp.async requests.
        if (tid == 0) {
          mbar_arrive_expect(&bars[stage], valid ? TMA_BYTES : 0);
          if (valid) {
            float *dst = &smem[stage * STAGE_FLOATS];
            tma_2d(dst, &map_a, acol + k0, m0, &bars[stage]);
            tma_2d(dst + A_STAGE, &map_b, bcol + k0, n0, &bars[stage]);
          }
        }
        return;
      }
      if (valid) {
        float *dst = &smem[stage * STAGE_FLOATS];
        int const chunks = (Cfg::kTbM + Cfg::kTbN) * (kK / 4);
        for (int c = tid; c < chunks; c += THREADS) {
          if (c < Cfg::kTbM * (kK / 4)) {
            int const arow = c / (kK / 4);
            int const half = c % (kK / 4);
            // Partial M tile: rows past the problem's M are never consumed (their output rows are
            // not stored), so the copy is skipped and the smem rows stay undefined.
            if (arow < rows_a) {
              cp_async16(dst + arow * kLds + half * 4,
                         Ag + static_cast<long long>(arow) * lda + k0 + half * 4);
            }
          } else {
            int const cc = c - Cfg::kTbM * (kK / 4);
            int const brow = cc / (kK / 4);
            int const half = cc % (kK / 4);
            cp_async16(dst + A_STAGE + brow * kLds + half * 4,
                       Bg + static_cast<long long>(brow) * ldb + k0 + half * 4);
          }
        }
      }
      cp_commit();
    };

    float acc[CHAINS][TM][TN];
#pragma unroll
    for (int h = 0; h < CHAINS; ++h)
#pragma unroll
    for (int r = 0; r < TM; ++r)
#pragma unroll
      for (int c = 0; c < TN; ++c) acc[h][r][c] = 0.f;

    // Prologue: fill stages 0 .. kStages-2, all of them inside path 0 (the plan asserts
    // nk >= kStages, i.e. every path is longer than the pipeline).  Each iteration then refills the
    // stage released *last* one ((kt + kStages - 1) % kStages): every warp has finished reading it
    // before this iteration's barrier, so one barrier per K-tile is enough.  The refill is issued
    // *before* the compute of tile kt, so the copy overlaps this tile's FFMA; with kStages == 2
    // that overlap is the whole pipeline (issue-after-compute exposes a full load latency per tile).
    if constexpr (!TMA) {
      // A persistent CTA reuses these stages for its next output tile. The last K-step's
      // barrier precedes its reads, so it does not protect them from the next prologue.
      // The first tile has no preceding readers and needs no additional barrier.
      if (t != blockIdx.x) __syncthreads();
    }
    if constexpr (TMA) {
      // The ring is per tile: a CTA walks several tiles, and the phases the previous tile left
      // behind do not match this tile's first wait, so re-arm the barriers here.
      __syncthreads();
      if (tid < kStages) mbar_init(&bars[tid], 1);
      __syncthreads();
    }
    for (int s = 0; s < kStages - 1; ++s) {
      load_stage(s, A, B, a_col0, b_col0, s * kK, s < nk);
    }

    // Prefetch cursor: the ring buffer runs straight through the path boundary, so the refill has
    // to know which path the tile it copies belongs to.  Stepping the cursor by hand instead of
    // computing kt / nk keeps an integer division (and a dependent pointer load) out of the loop.
    int ref_path = 0, ref_k = kStages - 1;
    float const *refA = A, *refB = B;
    int ref_a_col = a_col0, ref_b_col = b_col0;
    if constexpr (FUSED) {
      if (ref_k >= nk) {
        ref_k -= nk;
        ++ref_path;
      }
      path_bases(ref_path, refA, refB);
      ref_a_col = curv_a_col;
      ref_b_col = curv_b_col;
    }

    // Compute cursor: which path the tile about to be computed belongs to, and its k index.
    int cur_path = 0, cur_k = 0;

    for (int kt = 0; kt < nkt; ++kt) {
      if constexpr (TMA) {
        mbar_wait(&bars[kt % kStages], (kt / kStages) & 1);
      } else {
        cp_wait<kStages - 2>();
      }
      __syncthreads();

      if constexpr (FUSED) {
        load_stage((kt + kStages - 1) % kStages, refA, refB, ref_a_col, ref_b_col, ref_k * kK,
                   ref_path < np);
      } else {
        int const rf = kt + kStages - 1;
        load_stage(rf % kStages, A, B, a_col0, b_col0, rf * kK, rf < nk);
      }

      int const stage = kt % kStages;
      float const *sA = &smem[stage * STAGE_FLOATS];
      float const *sB = sA + A_STAGE;

#pragma unroll
      for (int kc = 0; kc < kK; kc += KC) {
        float a[TM][KC];
#pragma unroll
        for (int r = 0; r < TM; ++r) {
          float const *src = sA + smem_off(m_base + r * 8, kc);
          if constexpr (KC == 4) {
            float4 const v = *reinterpret_cast<float4 const *>(src);
            a[r][0] = v.x; a[r][1] = v.y; a[r][2] = v.z; a[r][3] = v.w;
          } else if constexpr (KC == 2) {
            float2 const v = *reinterpret_cast<float2 const *>(src);
            a[r][0] = v.x; a[r][1] = v.y;
          } else {
            a[r][0] = src[0];
          }
        }
        constexpr int NG = (TN + 3) / 4;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
          float b[4][KC];
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            if (g * 4 + c < TN) {
              float const *src = sB + smem_off(n_base + (g * 4 + c) * 4, kc);
              if constexpr (KC == 4) {
                float4 const v = *reinterpret_cast<float4 const *>(src);
                b[c][0] = v.x; b[c][1] = v.y; b[c][2] = v.z; b[c][3] = v.w;
              } else if constexpr (KC == 2) {
                float2 const v = *reinterpret_cast<float2 const *>(src);
                b[c][0] = v.x; b[c][1] = v.y;
              } else {
                b[c][0] = src[0];
              }
            }
          }
#pragma unroll
          for (int r = 0; r < TM; ++r) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              if (g * 4 + c < TN) {
#pragma unroll
                for (int j = 0; j < KC; ++j)
                  acc[(kc + j) % CHAINS][r][g * 4 + c] += a[r][j] * b[c][j];
              }
            }
          }
        }
      }

      // Path boundary: park this path's accumulator in the smem running sum and start a fresh one,
      // so every path is a full fp32 chain of its own -- torch's accumulation order.  A thread
      // owns slab elements [tid*RES_STRIDE, +TM*TN), written as float4s; the stride is chosen so
      // the 8 lanes of a 128-bit phase land on distinct bank groups.
      if constexpr (FUSED) {
        if (++ref_k == nk) {
          ref_k = 0;
          ++ref_path;
          path_bases(ref_path, refA, refB);
          ref_a_col = curv_a_col;
          ref_b_col = curv_b_col;
        }
        if (++cur_k == nk) {
          cur_k = 0;
          ++cur_path;
          if (cur_path < np) {
            fold_chains(acc);
            if constexpr (CHAINS > 1) {
              float const scale = p.scales ? p.scales[base + cur_path - 1] : 1.f;
#pragma unroll
              for (int r = 0; r < TM; ++r)
#pragma unroll
                for (int c = 0; c < TN; ++c) acc[0][r][c] *= scale;
            }
            float4 *slot = reinterpret_cast<float4 *>(&smem[RES_OFF + tid * RES_STRIDE]);
            auto at = [&](int i) { return i < TM * TN ? acc[0][i / TN][i % TN] : 0.f; };
            if (cur_path == 1) {  // first boundary: store, later ones accumulate
#pragma unroll
              for (int k = 0; k < RES_ELEMS; k += 4) {
                slot[k / 4] = make_float4(at(k), at(k + 1), at(k + 2), at(k + 3));
              }
            } else {
#pragma unroll
              for (int k = 0; k < RES_ELEMS; k += 4) {
                float4 w = slot[k / 4];
                w.x += at(k);
                w.y += at(k + 1);
                w.z += at(k + 2);
                w.w += at(k + 3);
                slot[k / 4] = w;
              }
            }
#pragma unroll
            for (int h = 0; h < CHAINS; ++h)
#pragma unroll
            for (int r = 0; r < TM; ++r)
#pragma unroll
              for (int c = 0; c < TN; ++c) acc[h][r][c] = 0.f;
          }
        }
      }
    }

    // Epilogue: scalar stores; the stores are a small fraction of the runtime and TN is not always
    // a multiple of 4 (V=224 gives TN=14), so vectorising them is not worth the tail handling.
    // FUSED folds in the running sum here, so the output tile is written exactly once.
    fold_chains(acc);
    float const *slab = &smem[RES_OFF + tid * RES_STRIDE];
#pragma unroll
    for (int r = 0; r < TM; ++r) {
      if (m_base + r * 8 >= rows_a) continue;
      float *crow = C + static_cast<long long>(m_base + r * 8) * ldc;
#pragma unroll
      for (int c = 0; c < TN; ++c) {
        float v = acc[0][r][c];
        if constexpr (CHAINS > 1) v *= p.scales ? p.scales[base + np - 1] : 1.f;
        if constexpr (FUSED) {
          if (np > 1) v += slab[r * TN + c];
        }
        crow[n_base + c * 4] = v;
      }
    }
  }
}

/// Compile-time tile configuration.  Lane layout is always 8 M-lanes x 4 N-lanes.
struct Cfg128 {  // V <= 128, B > 1024 (only reached by impl 1; impl 2 sends V=128 to the stock path)
  static constexpr int kMinBlocks = 1;
  static constexpr int kTbM = 128, kTbN = 128, kWarpM = 64, kWarpN = 64;
  // 16x8 = 128 accumulators + 24 live operand floats do not fit with KC=2 without a 40 B spill;
  // scalar operand loads remove the spill and measure the same, so keep them.
  static constexpr int kWarpsM = 2, kWarpsN = 2, kChunk = 1;
  static constexpr int kThreads = kWarpsM * kWarpsN * 32;
};
struct Cfg96 {  // V <= 96, B > 1024: 8x12 thread tile (0.833 B/FMA)
  static constexpr int kMinBlocks = 1;
  static constexpr int kTbM = 128, kTbN = 96, kWarpM = 64, kWarpN = 48;
  static constexpr int kWarpsM = 2, kWarpsN = 2, kChunk = 2;
  static constexpr int kThreads = kWarpsM * kWarpsN * 32;
};
struct Cfg224 {  // V > 128 (224): 8x14 thread tile (0.786 B/FMA), TB_M=64 keeps the grid wide
  static constexpr int kMinBlocks = 1;
  static constexpr int kTbM = 64, kTbN = 224, kWarpM = 64, kWarpN = 56;
  static constexpr int kWarpsM = 1, kWarpsN = 4, kChunk = 2;
  static constexpr int kThreads = kWarpsM * kWarpsN * 32;
};
struct Cfg224W {  // V > 128: one 8-warp CTA spanning 128 rows (same 8x14 thread tile as Cfg224).
  // Registers cap it at 1 CTA/SM, which is the point: the same 8 warps in one block, twice the M
  // reuse from B, and enough shared memory left over for a fused per-path accumulator slab.
  static constexpr int kMinBlocks = 1;
  static constexpr int kTbM = 128, kTbN = 224, kWarpM = 64, kWarpN = 56;
  static constexpr int kWarpsM = 2, kWarpsN = 4, kChunk = 2;
  static constexpr int kThreads = kWarpsM * kWarpsN * 32;
};
struct Cfg224N {  // M11 experiment: 8x7 thread tile (56 accumulators), TbN=112 (two N tiles),
                  // 4 warps / 128 threads.  Lower register pressure is the point: the occupancy
                  // query decides whether it buys a third CTA (8 -> 12 warps/SM).  The cost is a
                  // second read of the A tile (the two N tiles of one problem are adjacent in the
                  // tile list, so L2 may absorb it) and a worse smem/FMA ratio.
  static constexpr int kMinBlocks = 1;
  static constexpr int kTbM = 64, kTbN = 112, kWarpM = 64, kWarpN = 28;
  static constexpr int kWarpsM = 1, kWarpsN = 4, kChunk = 2;
  static constexpr int kThreads = kWarpsM * kWarpsN * 32;  // 128
};
struct Cfg128W {  // V = 128, path-chained: 256 rows over 8 warps, 1 CTA/SM.
  static constexpr int kMinBlocks = 1;
  static constexpr int kTbM = 256, kTbN = 128, kWarpM = 64, kWarpN = 64;
  static constexpr int kWarpsM = 4, kWarpsN = 2, kChunk = 1;
  static constexpr int kThreads = kWarpsM * kWarpsN * 32;
};
struct CfgS128 {  // V <= 128, B <= 1024: half the M tile to double the grid
  static constexpr int kMinBlocks = 1;
  static constexpr int kTbM = 64, kTbN = 128, kWarpM = 64, kWarpN = 64;
  static constexpr int kWarpsM = 1, kWarpsN = 2, kChunk = 1;
  static constexpr int kThreads = kWarpsM * kWarpsN * 32;
};
struct CfgS96 {
  static constexpr int kMinBlocks = 1;
  static constexpr int kTbM = 64, kTbN = 96, kWarpM = 64, kWarpN = 48;
  static constexpr int kWarpsM = 1, kWarpsN = 2, kChunk = 2;
  static constexpr int kThreads = kWarpsM * kWarpsN * 32;
};

// Eight numerical chains need their own tile budget. Halving N halves the
// accumulator state without changing K-to-chain assignment or fold_chains.
// Four adjacent K values are loaded together to reduce LDS/MIO pressure.
struct Cfg128Short : NumericConfig<Cfg128, 8> {
  static constexpr int kTbN = 64, kWarpN = 32, kChunk = 4;
  static constexpr int kMinBlocks = 3;
};
struct Cfg224Short : NumericConfig<Cfg224N, 8> {
  static constexpr int kChunk = 4, kMinBlocks = 3;
};

/// Which kernel shape, and how deep its pipeline.  Read once per plan so both can be A/B'd in one
/// process; 2/3/4 are the only depths that are instantiated (a deeper pipeline is not always
/// better: a multi-path problem's K is only U, so the prologue fill is a larger share of it).
int stages_from_env(int def) {
  char const *env = nullptr;
  int const s = env != nullptr ? atoi(env) : 0;
  return (s == 2 || s == 3 || s == 4) ? s : def;
}
char const *cfg_from_env() {
  char const *env = nullptr;
  return env != nullptr ? env : "";
}

/// One plan for both kernels.  They differ only in what a tile means:
///
///   plain (FUSED=false) -- a tile is (problem, m, n); every problem writes its own C rows.  The
///                          split table hands it one problem per (output row, path) and a second
///                          kernel sums the per-path slabs (M4).
///   fused (FUSED=true)  -- a tile is (output row, m, n) and chains that row's `pg` per-path
///                          problems, so the output tile is written once and no slab exists (M5).
///                          `rbase`/`pg` (J entries each) describe the runs in `rows`.
template <class Cfg, int STAGES>
class SimtPlanImpl : public GroupedGemmPlan::ImplBase {
 public:
  SimtPlanImpl(int M, int N, int num_problems, int num_out_rows, long long lda, long long ldb,
               long long ldc, std::vector<RowPlan> const &rows, std::vector<int> const *rbase,
               std::vector<int> const *pg, float const *a_data, float const *b_data, bool a_view)
      : M_(M), N_(N), NP_(num_problems), J_(num_out_rows), fused_(rbase != nullptr) {
    a_base_ = a_data;
    a_bound_ = !a_view;  // packed mode is usable immediately; view mode waits for bind_a
    // Mainloop choice is a run-time knob so the two can be A/B'd in one process (M6).  The TMA path
    // also needs the driver's tensor-map encoder and a box that fits the tile; if either is missing
    // the plan silently keeps cp.async rather than failing.
    {
      // TMA is the default from M6 on (12-14% faster, bit-identical); the cp.async mainloop stays
      // as the A/B baseline and as the fallback when a descriptor cannot be encoded.
      // TMA is the default from M6 on (12-14% faster at 21 of the 25 points, bit-identical).
      // The four points where it does not pay are the small-TILE ones -- V <= 128 at B <= 512 --
      // where the per-tile ring reset and the 20-float (vs 16) box dominate; those stay on cp.async
      // by shape, the same way impl 2 dispatches (measured in M6_REPORT.md §4).
      char const *env = nullptr;
      bool const tma_ok_by_shape = (N_ > 128) || (M_ >= 1024);
      want_tma_ = (env == nullptr || strcmp(env, "cpasync") != 0) && tma_ok_by_shape;
    }
    if (N % Cfg::kTbN) {
      throw std::invalid_argument("SimtPlanImpl: N must divide the tile exactly");
    }
    if (static_cast<int>(rows.size()) != NP_) {
      throw std::invalid_argument("SimtPlanImpl: problem table size mismatch");
    }
    for (int i = 0; i < NP_; ++i) {
      if (rows[i].K % kK) throw std::invalid_argument("SimtPlanImpl: K must be a multiple of the K-tile");
      // The prologue fills the pipeline inside path 0, so a path must be at least that long.
      if (fused_ && rows[i].K / kK < STAGES) {
        throw std::invalid_argument("SimtPlanImpl: per-path K is shorter than the pipeline");
      }
    }
    if (fused_ && (static_cast<int>(rbase->size()) != J_ || static_cast<int>(pg->size()) != J_)) {
      throw std::invalid_argument("SimtPlanImpl: row table size mismatch");
    }
    if (!fused_ && NP_ != J_) {
      throw std::invalid_argument("SimtPlanImpl: plain plan needs one problem per output row");
    }

    std::vector<int3> h_problem(NP_);
    std::vector<float const *> h_b(NP_);
    std::vector<int> h_a_col(NP_), h_b_col(NP_);
    for (int i = 0; i < NP_; ++i) {
      h_problem[i] = make_int3(M_, N_, rows[i].K);
      h_b[i] = b_data + rows[i].b_off;
      h_a_col[i] = static_cast<int>(rows[i].a_off);
      h_b_col[i] = static_cast<int>(rows[i].b_off);
    }

    int const mt = (M_ + Cfg::kTbM - 1) / Cfg::kTbM;
    int const nt = N_ / Cfg::kTbN;
    num_tiles_ = mt * nt * J_;
    std::vector<int4> h_tiles(num_tiles_);
    int cursor = 0;
    for (int j = 0; j < J_; ++j)
      for (int m = 0; m < mt; ++m)
        for (int n = 0; n < nt; ++n) h_tiles[cursor++] = make_int4(j, m, n, 0);

    size_t const n = static_cast<size_t>(NP_);
    size_t const nj = static_cast<size_t>(J_);
    lda_ = static_cast<int>(lda);
    ldb_ = static_cast<int>(ldb);
    ldc_ = static_cast<int>(ldc);
    c_stride_ = N_;
    if (d_problem_.reset(n * sizeof(int3)) || d_b_.reset(n * sizeof(float const *)) ||
        d_a_col_.reset(n * sizeof(int)) ||
        d_b_col_.reset(n * sizeof(int)) ||
        d_tiles_.reset(static_cast<size_t>(num_tiles_) * sizeof(int4))) {
      throw std::runtime_error("SimtPlanImpl: device allocation failed");
    }
    if (fused_ &&
        (d_rbase_.reset(nj * sizeof(int)) || d_pg_.reset(nj * sizeof(int)))) {
      throw std::runtime_error("SimtPlanImpl: row table allocation failed");
    }

    cudaMemcpy(d_problem_.as<int3>(), h_problem.data(), n * sizeof(int3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_.as<float const *>(), h_b.data(), n * sizeof(float const *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_col_.as<int>(), h_a_col.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_col_.as<int>(), h_b_col.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tiles_.as<int4>(), h_tiles.data(), num_tiles_ * sizeof(int4),
               cudaMemcpyHostToDevice);
    if (fused_) {
      cudaMemcpy(d_rbase_.as<int>(), rbase->data(), nj * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_pg_.as<int>(), pg->data(), nj * sizeof(int), cudaMemcpyHostToDevice);
    }

    // TMA: one tile per operand per K-tile, box = (rows of the tile, kLds) with kLds = kK + kPad so
    // the destination keeps the padded layout.  Both descriptors are built from plan-owned buffers,
    // so they are encoded once, here.
    tma_ = false;
    if (want_tma_) {
      bool ok = encode_2d_f32(&map_a_, a_data, static_cast<uint64_t>(M_),
                              static_cast<uint64_t>(lda), static_cast<uint64_t>(lda) * sizeof(float),
                              Cfg::kTbM, kLds);
      ok = ok && encode_2d_f32(&map_b_, b_data, static_cast<uint64_t>(N_),
                               static_cast<uint64_t>(ldb),
                               static_cast<uint64_t>(ldb) * sizeof(float), Cfg::kTbN, kLds);
      tma_ = ok;
      if (!ok) fprintf(stderr, "[eqlin] TMA unavailable here; using cp.async\n");
    }

    // K-tile 16 pushes the per-CTA dynamic smem past 48 KB; the opt-in must happen before the
    // occupancy query, otherwise the launch fails.
    cudaFuncSetAttribute(kernel_ptr(), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes());

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_ptr(), Cfg::kThreads,
                                                  smem_bytes());
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    blocks_per_sm_ = blocks_per_sm;
    grid_ = std::min(num_tiles_, sm_count_ * blocks_per_sm);
    if (grid_ < 1) grid_ = 1;
  }

  ~SimtPlanImpl() override = default;
  bool scales_after_dot() const override { return Cfg::kChains > 1; }

  int set_scales(std::vector<double> const &scales, cudaStream_t) override {
    if constexpr (Cfg::kChains > 1) {
      if (static_cast<int>(scales.size()) != NP_ ||
          d_scales_.reset(scales.size() * sizeof(float)) != 0) return 1;
      std::vector<float> const rounded(scales.begin(), scales.end());
      // Complete the upload before pageable host storage is destroyed. The plan owns this
      // immutable table; no host staging memory or pointer table is rewritten per launch.
      if (cudaMemcpy(d_scales_.as<float>(), rounded.data(), rounded.size() * sizeof(float),
                     cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    }
    return 0;
  }

  /// Is this problem set wide enough to fill the machine?  A path-chained tile is 2.5-4.4x fewer
  /// tiles than the split table's, so at small B it cannot fill the SMs and the split (whose extra
  /// problems do) stays faster; the caller falls back in that case.  One full wave is the cut.
  bool fills_machine() const {
    return static_cast<long long>(num_tiles_) * 10 >=
           static_cast<long long>(sm_count_) * blocks_per_sm_ * 9;
  }

  /// Zero-copy A: swap the base and re-encode the TMA descriptor.  Pure host work -- the kernel
  /// takes the base as a launch argument, and the descriptor is snapshotted by the launch call, so
  /// nothing here needs stream ordering.  A stale descriptor is the silent failure mode (TMA
  /// zero-fills out of bounds), which is why the switch is exercised with two alternating x buffers
  /// in the tests.
  int bind_a(float const *x, cudaStream_t stream) override {
    if (x == a_base_ && a_bound_) return 0;
    if (tma_) {
      if (!encode_2d_f32(&map_a_, x, static_cast<uint64_t>(M_), static_cast<uint64_t>(lda_),
                         static_cast<uint64_t>(lda_) * sizeof(float), Cfg::kTbM, kLds)) {
        fprintf(stderr, "[eqlin] bind_a: TMA re-encode failed (x=%p); refusing to launch\n",
                static_cast<void const *>(x));
        return 1;
      }
    }
    a_base_ = x;
    a_bound_ = true;
    bind_stream_ = stream;
    return 0;
  }

  int launch(float *out, cudaStream_t stream) override {
    if (!a_bound_) {
      fprintf(stderr, "[eqlin] launch: A not bound -- call prepare_act(x) first\n");
      return 1;
    }
    if (bind_stream_ != nullptr && stream != bind_stream_) {
      fprintf(stderr, "[eqlin] launch: stream differs from the one A was bound on\n");
      return 1;
    }
    SimtParams p{num_tiles_,
                 d_problem_.as<int3>(),
                 a_base_,
                 d_b_.as<float const *>(),
                 d_a_col_.as<int>(),
                 d_b_col_.as<int>(),
                 out,
                 c_stride_,
                 lda_,
                 ldb_,
                 ldc_,
                 d_tiles_.as<int4>(),
                 fused_ ? d_rbase_.as<int>() : nullptr,
                 fused_ ? d_pg_.as<int>() : nullptr};
    p.scales = d_scales_.as<float>();
    if (fused_) {
      if (tma_) {
        simt_gemm_kernel<Cfg, STAGES, true, true><<<grid_, Cfg::kThreads, smem_bytes(), stream>>>(
            p, map_a_, map_b_);
      } else {
        simt_gemm_kernel<Cfg, STAGES, true, false><<<grid_, Cfg::kThreads, smem_bytes(), stream>>>(
            p, map_a_, map_b_);
      }
      return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }
    if (tma_) {
      simt_gemm_kernel<Cfg, STAGES, false, true><<<grid_, Cfg::kThreads, smem_bytes(), stream>>>(
          p, map_a_, map_b_);
    } else {
      simt_gemm_kernel<Cfg, STAGES, false, false><<<grid_, Cfg::kThreads, smem_bytes(), stream>>>(
          p, map_a_, map_b_);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
  }

 private:
  void const *kernel_ptr() const {
    if (fused_) {
      return tma_ ? reinterpret_cast<void const *>(simt_gemm_kernel<Cfg, STAGES, true, true>)
                  : reinterpret_cast<void const *>(simt_gemm_kernel<Cfg, STAGES, true, false>);
    }
    return tma_ ? reinterpret_cast<void const *>(simt_gemm_kernel<Cfg, STAGES, false, true>)
                : reinterpret_cast<void const *>(simt_gemm_kernel<Cfg, STAGES, false, false>);
  }
  int smem_bytes() const {
    int floats = SharedPlan<Cfg, STAGES, false>::kFloats;
    if (fused_) floats = SharedPlan<Cfg, STAGES, true>::kFloats;
    int const operands_end = floats * static_cast<int>(sizeof(float));
    if (!tma_) return operands_end;
    // The barriers sit at RES_OFF_ALIGNED in the kernel: 2-float aligned, 8 B each.
    int const bar_off = ((floats + 1) & ~1) * static_cast<int>(sizeof(float));
    return bar_off + STAGES * static_cast<int>(sizeof(uint64_t));
  }

  int M_, N_, NP_, J_;
  bool fused_ = false;
  int num_tiles_ = 0, grid_ = 0, sm_count_ = 0, blocks_per_sm_ = 1;
  int lda_ = 0, ldb_ = 0, ldc_ = 0, c_stride_ = 0;
  /// A base + its bound state.  View mode starts unbound: `bind_a` must run before any launch.
  float const *a_base_ = nullptr;
  bool a_bound_ = false;
  cudaStream_t bind_stream_ = nullptr;
  bool want_tma_ = false, tma_ = false;
  CUtensorMap map_a_{}, map_b_{};
  DeviceBuffer d_problem_, d_b_, d_a_col_, d_b_col_, d_tiles_, d_rbase_, d_pg_;
  DeviceBuffer d_scales_;
};

/// Backward plan: one tile per (problem, m, n).  Every problem carries its own A/C offsets and
/// strides, so dgrad (M=B) and wgrad (M=U, split-K) can share one launch.  The caller's dy/dx are
/// kernel arguments; the plan owns only the immutable row/tile tables.
template <class Cfg, int STAGES>
class SimtBwdPlanImpl : public BwdPlan {
 public:
  bool scales_after_dot() const override { return Cfg::kChains > 1; }
  SimtBwdPlanImpl(int N, std::vector<BwdProblem> const &problems) {
    int const np = static_cast<int>(problems.size());
    if (np <= 0) throw std::invalid_argument("SimtBwdPlanImpl: empty problem table");
    if (N <= 0 || N % Cfg::kTbN) {
      throw std::invalid_argument("SimtBwdPlanImpl: N must divide the tile exactly");
    }

    std::vector<int3> h_problem(np);
    std::vector<float const *> h_b(np);
    std::vector<int> h_kind(np), h_lda(np), h_ldb(np), h_ldc(np);
    std::vector<long long> h_a_off(np), h_c_off(np);
    std::vector<float> h_scales(np);
    std::vector<int4> h_tiles;
    for (int i = 0; i < np; ++i) {
      BwdProblem const &q = problems[i];
      if (q.N != N) throw std::invalid_argument("SimtBwdPlanImpl: mixed N in one launch");
      if (q.K % kK) throw std::invalid_argument("SimtBwdPlanImpl: K must be a multiple of 16");
      if (q.K / kK < STAGES) {
        throw std::invalid_argument("SimtBwdPlanImpl: K is shorter than the pipeline");
      }
      h_problem[i] = make_int3(q.M, q.N, q.K);
      h_b[i] = q.b;
      h_kind[i] = q.kind;
      h_lda[i] = q.lda;
      h_ldb[i] = q.ldb;
      h_ldc[i] = q.ldc;
      h_a_off[i] = q.a_off;
      h_c_off[i] = q.c_off;
      h_scales[i] = q.scale;
      int const mt = (q.M + Cfg::kTbM - 1) / Cfg::kTbM;
      int const nt = q.N / Cfg::kTbN;
      for (int m = 0; m < mt; ++m)
        for (int n = 0; n < nt; ++n) h_tiles.push_back(make_int4(i, m, n, 0));
    }
    num_tiles_ = static_cast<int>(h_tiles.size());

    size_t const n = static_cast<size_t>(np);
    size_t const nt = static_cast<size_t>(num_tiles_);
    if (d_problem_.reset(n * sizeof(int3)) || d_b_.reset(n * sizeof(float const *)) ||
        d_kind_.reset(n * sizeof(int)) || d_lda_.reset(n * sizeof(int)) ||
        d_ldb_.reset(n * sizeof(int)) || d_ldc_.reset(n * sizeof(int)) ||
        d_a_off_.reset(n * sizeof(long long)) || d_c_off_.reset(n * sizeof(long long)) ||
        d_tiles_.reset(nt * sizeof(int4))) {
      throw std::runtime_error("SimtBwdPlanImpl: device allocation failed");
    }
    cudaMemcpy(d_problem_.as<int3>(), h_problem.data(), n * sizeof(int3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_.as<float const *>(), h_b.data(), n * sizeof(float const *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_kind_.as<int>(), h_kind.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lda_.as<int>(), h_lda.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldb_.as<int>(), h_ldb.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldc_.as<int>(), h_ldc.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_off_.as<long long>(), h_a_off.data(), n * sizeof(long long),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_off_.as<long long>(), h_c_off.data(), n * sizeof(long long),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_tiles_.as<int4>(), h_tiles.data(), nt * sizeof(int4), cudaMemcpyHostToDevice);
    if constexpr (Cfg::kChains > 1) {
      if (d_scales_.reset(n * sizeof(float)) != 0)
        throw std::runtime_error("SimtBwdPlanImpl: scale allocation failed");
      if (cudaMemcpy(d_scales_.as<float>(), h_scales.data(), n * sizeof(float),
                     cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("SimtBwdPlanImpl: scale upload failed");
    }

    cudaFuncSetAttribute(kernel_ptr(), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes());
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_ptr(), Cfg::kThreads,
                                                  smem_bytes());
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    blocks_per_sm_ = blocks_per_sm;
    grid_ = std::min(num_tiles_, sm_count_ * blocks_per_sm_);
    if (grid_ < 1) grid_ = 1;
  }

  int launch(float const *dy, float *dx, float const *xt, float *partials,
             cudaStream_t stream) override {
    SimtParams p{};
    p.num_tiles = num_tiles_;
    p.problems = d_problem_.as<int3>();
    p.ptr_b = d_b_.as<float const *>();
    p.tiles = d_tiles_.as<int4>();
    p.kind = d_kind_.as<int>();
    p.a_base0 = dy;
    p.a_base1 = xt;
    p.c_base0 = dx;
    p.c_base1 = partials;
    p.a_off = d_a_off_.as<long long>();
    p.c_off = d_c_off_.as<long long>();
    p.lda_arr = d_lda_.as<int>();
    p.ldb_arr = d_ldb_.as<int>();
    p.ldc_arr = d_ldc_.as<int>();
    p.scales = d_scales_.as<float>();
    simt_gemm_kernel<Cfg, STAGES, false, false, true>
        <<<grid_, Cfg::kThreads, smem_bytes(), stream>>>(p, CUtensorMap{}, CUtensorMap{});
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
  }

 private:
  void const *kernel_ptr() const {
    return reinterpret_cast<void const *>(simt_gemm_kernel<Cfg, STAGES, false, false, true>);
  }
  int smem_bytes() const {
    return SharedPlan<Cfg, STAGES, false>::kFloats * static_cast<int>(sizeof(float));
  }

  int num_tiles_ = 0, grid_ = 0, sm_count_ = 0, blocks_per_sm_ = 1;
  DeviceBuffer d_problem_, d_b_, d_kind_, d_lda_, d_ldb_, d_ldc_, d_a_off_, d_c_off_, d_tiles_;
  DeviceBuffer d_scales_;
};

template <class Cfg, int STAGES>
std::unique_ptr<SimtPlanImpl<Cfg, STAGES>> build(int M, int N, int num_problems, int num_out_rows,
                                                 long long lda, long long ldb, long long ldc,
                                                 std::vector<RowPlan> const &rows,
                                                 std::vector<int> const *rbase,
                                                 std::vector<int> const *pg, float const *a,
                                                 float const *b, bool a_view) {
  return std::make_unique<SimtPlanImpl<Cfg, STAGES>>(M, N, num_problems, num_out_rows, lda, ldb,
                                                     ldc, rows, rbase, pg, a, b, a_view);
}

template <class Cfg>
std::unique_ptr<GroupedGemmPlan::ImplBase> make(int M, int N, int num_rows, long long lda,
                                                long long ldb, long long ldc,
                                                std::vector<RowPlan> const &rows, float const *a,
                                                float const *b, bool a_view) {
  switch (stages_from_env(4)) {
    case 2:
      return build<Cfg, 2>(M, N, num_rows, num_rows, lda, ldb, ldc, rows, nullptr, nullptr, a, b,
                           a_view);
    case 3:
      return build<Cfg, 3>(M, N, num_rows, num_rows, lda, ldb, ldc, rows, nullptr, nullptr, a, b,
                           a_view);
    default:
      return build<Cfg, 4>(M, N, num_rows, num_rows, lda, ldb, ldc, rows, nullptr, nullptr, a, b,
                           a_view);
  }
}

/// Path-chained plan, or nullptr when it would not pay (see `fills_machine`).
template <class Cfg, int STAGES>
std::unique_ptr<GroupedGemmPlan::ImplBase> make_fused_stages(
    int M, int N, int num_problems, int num_out_rows, long long lda, long long ldb, long long ldc,
    std::vector<RowPlan> const &rows, std::vector<int> const &rbase, std::vector<int> const &pg,
    float const *a, float const *b, bool a_view) {
  auto plan = build<Cfg, STAGES>(M, N, num_problems, num_out_rows, lda, ldb, ldc, rows, &rbase,
                                 &pg, a, b, a_view);
  if (!plan->fills_machine()) return nullptr;
  return plan;
}

template <class Cfg>
std::unique_ptr<GroupedGemmPlan::ImplBase> make_fused(int M, int N, int num_problems,
                                                      int num_out_rows, long long lda,
                                                      long long ldb, long long ldc,
                                                      std::vector<RowPlan> const &rows,
                                                      std::vector<int> const &rbase,
                                                      std::vector<int> const &pg, float const *a,
                                                      float const *b, int def_stages,
                                                      bool a_view) {
  switch (stages_from_env(def_stages)) {
    case 2:
      return make_fused_stages<Cfg, 2>(M, N, num_problems, num_out_rows, lda, ldb, ldc, rows, rbase,
                                       pg, a, b, a_view);
    case 3:
      return make_fused_stages<Cfg, 3>(M, N, num_problems, num_out_rows, lda, ldb, ldc, rows, rbase,
                                       pg, a, b, a_view);
    default:
      return make_fused_stages<Cfg, 4>(M, N, num_problems, num_out_rows, lda, ldb, ldc, rows, rbase,
                                       pg, a, b, a_view);
  }
}

template <class Cfg>
std::unique_ptr<BwdPlan> make_bwd_cfg(int N, std::vector<BwdProblem> const &problems,
                                      int def_stages) {
  switch (stages_from_env(def_stages)) {
    case 2:
      return std::make_unique<SimtBwdPlanImpl<Cfg, 2>>(N, problems);
    case 3:
      return std::make_unique<SimtBwdPlanImpl<Cfg, 3>>(N, problems);
    default:
      return std::make_unique<SimtBwdPlanImpl<Cfg, 4>>(N, problems);
  }
}

template <class Cfg>
std::unique_ptr<GroupedGemmPlan::ImplBase> make_numeric(
    int M, int N, int num_rows, long long lda, long long ldb, long long ldc,
    std::vector<RowPlan> const &rows, float const *a, float const *b, bool a_view) {
  // Independent 1+3 component problems; retain the established arithmetic and
  // use a smaller resource tile only on the profiled large-batch shape class.
  if (M >= 1024 && N == 128 && num_rows == 4)
    return make<Cfg128Short>(M, N, num_rows, lda, ldb, ldc, rows, a, b, a_view);
  if (NumericPolicy::short_chains(M, N, num_rows))
    return make<NumericConfig<Cfg, 8>>(M, N, num_rows, lda, ldb, ldc, rows, a, b, a_view);
  return make<NumericConfig<Cfg, 1>>(M, N, num_rows, lda, ldb, ldc, rows, a, b, a_view);
}

template <class Cfg>
std::unique_ptr<GroupedGemmPlan::ImplBase> make_numeric_fused(
    int M, int N, int num_problems, int num_out_rows, long long lda, long long ldb, long long ldc,
    std::vector<RowPlan> const &rows, std::vector<int> const &rbase, std::vector<int> const &pg,
    float const *a, float const *b, int stages, bool a_view) {
  if (NumericPolicy::tiny_batch(M)) return nullptr;  // Runner uses the Tiny forward plan here.
  if (NumericPolicy::short_chains(M, N, num_problems))
    return make_fused<NumericConfig<Cfg, 8>>(M, N, num_problems, num_out_rows, lda, ldb, ldc,
                                            rows, rbase, pg, a, b, stages, a_view);
  return make_fused<NumericConfig<Cfg, 1>>(M, N, num_problems, num_out_rows, lda, ldb, ldc,
                                          rows, rbase, pg, a, b, stages, a_view);
}

template <class Cfg>
std::unique_ptr<BwdPlan> make_numeric_bwd(int N, std::vector<BwdProblem> const &problems,
                                         int stages, int batch) {
  bool const dgrad_only = std::all_of(problems.begin(), problems.end(),
                                    [](BwdProblem const &p) { return p.kind == 0; });
  if (dgrad_only && batch >= 1024 && N == 128 && problems.size() == 4)
    return make_bwd_cfg<Cfg128Short>(N, problems, stages);
  if (dgrad_only && batch >= 1024 && N == 224 && problems.size() == 9)
    return make_bwd_cfg<Cfg224Short>(N, problems, stages);
  if (dgrad_only && NumericPolicy::short_chains(batch, N, static_cast<int>(problems.size()), true))
    return make_bwd_cfg<NumericConfig<Cfg, 8>>(N, problems, stages);
  return make_bwd_cfg<NumericConfig<Cfg, 1>>(N, problems, stages);
}

}  // namespace

std::unique_ptr<GroupedGemmPlan::ImplBase> make_simt_custom_plan(
    int M, int N, int num_rows, long long lda, long long ldb, long long ldc,
    std::vector<RowPlan> const &rows, float const *a_data, float const *b_data, bool a_view) {
  bool const small = (M <= 1024);
  if (N <= 96) {
    if (small) return make_numeric<CfgS96>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
    return make_numeric<Cfg96>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
  }
  if (N <= 128) {
    if (small) return make_numeric<CfgS128>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
    return make_numeric<Cfg128>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
  }
  // V > 128: one 8x14 shape for every B (TB_M=64 already doubles the grid at small B; splitting
  // N was measured 7% slower at B=512).  EQLIN_FP32_CFG=wide swaps in the 8-warp / 128-row variant.
  if (strcmp(cfg_from_env(), "wide") == 0) {
    return make_numeric<Cfg224W>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
  }
  return make_numeric<Cfg224>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
}

std::unique_ptr<GroupedGemmPlan::ImplBase> make_simt_fused_plan(
    int M, int N, int num_problems, int num_out_rows, long long lda, long long ldb, long long ldc,
    std::vector<RowPlan> const &rows, std::vector<int> const &rbase, std::vector<int> const &pg,
    float const *a_data, float const *b_data, bool a_view) {
  if (N < 128) return nullptr;  // N=96 uses the exact-fit split plan, never an N=128 tile.
  // Shape per V.  Both are 8x14 / 8x16 thread tiles, i.e. the same mainloop the plain kernel uses;
  // what differs is where the running-sum slab fits.  V=224: 128 rows over 8 warps leaves 25 KB of
  // headroom at 3 stages, 1 CTA/SM (registers are the limit either way).  V=128: the 128x128 tile
  // at 2 stages still fits 2 CTAs/SM, which is what keeps 8 warps at small B (M5 §2).
  bool const want_wide = strcmp(cfg_from_env(), "wide") == 0;
  bool const want_narrow = strcmp(cfg_from_env(), "narrow") == 0;
  if (N > 128) {
    if (want_narrow) {
      return make_numeric_fused<Cfg224>(M, N, num_problems, num_out_rows, lda, ldb, ldc, rows, rbase, pg,
                                a_data, b_data, 2, a_view);
    }
    return make_numeric_fused<Cfg224W>(M, N, num_problems, num_out_rows, lda, ldb, ldc, rows, rbase, pg,
                               a_data, b_data, 3, a_view);
  }
  if (N <= 128) {
    if (want_wide) {
      return make_numeric_fused<Cfg128W>(M, N, num_problems, num_out_rows, lda, ldb, ldc, rows, rbase, pg,
                                 a_data, b_data, 3, a_view);
    }
    return make_numeric_fused<Cfg128>(M, N, num_problems, num_out_rows, lda, ldb, ldc, rows, rbase, pg,
                              a_data, b_data, 2, a_view);
  }
  return nullptr;
}

std::unique_ptr<BwdPlan> make_simt_bwd_plan(int N, int B,
                                            std::vector<BwdProblem> const &problems,
                                            float const *weights) {
  if (NumericPolicy::tiny_batch(B) && std::all_of(problems.begin(), problems.end(),
                                              [](BwdProblem const &p) { return p.kind == 0; }))
    return make_tiny_dgrad_plan(B, N, problems, weights);
  bool const small = (B <= 1024);
  if (N <= 96) {
    return small ? make_numeric_bwd<CfgS96>(N, problems, 2, B)
                 : make_numeric_bwd<Cfg96>(N, problems, 3, B);
  }
  if (N <= 128) {
    return small ? make_numeric_bwd<CfgS128>(N, problems, 2, B)
                 : make_numeric_bwd<Cfg128>(N, problems, 3, B);
  }
  // Backward's default is the narrow 64-row tile: wgrad's M is only U (96/128/224), so the
  // 128-row tile wastes 12.5% of the MMA rows and leaves half the SM idle at small B.  The
  // measured A/B (LARGE_P1/P3, B=512/2048/5888) is 6.4% faster for the fused core.
  // EQLIN_FP32_CFG=wide restores the forward's 128-row shape; =halfn is the M11 56-accumulator
  // experiment (TbN=112, two N tiles).
  // M12: the per-point M10 data says the wide/4 win is confined to N>128 with B>=2048; keep the
  // narrow/3 shape everywhere else.  This is the shape rule the global wide/4 could not express.
  if (strcmp(cfg_from_env(), "hybrid") == 0) {
    if (B >= 2048) return make_numeric_bwd<Cfg224W>(N, problems, 4, B);
    return make_numeric_bwd<Cfg224>(N, problems, 3, B);
  }
  if (strcmp(cfg_from_env(), "halfn") == 0) {
    return make_numeric_bwd<Cfg224N>(N, problems, 3, B);
  }
  if (strcmp(cfg_from_env(), "wide") == 0) {
    return make_numeric_bwd<Cfg224W>(N, problems, 3, B);
  }
  return make_numeric_bwd<Cfg224>(N, problems, 3, B);
}

}  // namespace eqlin_fp32
