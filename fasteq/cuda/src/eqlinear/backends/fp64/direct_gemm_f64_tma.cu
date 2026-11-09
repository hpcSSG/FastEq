/***************************************************************************************************
 * fasteq eqlinear backends/fp64/direct_gemm_f64_tma.cu
 *
 * M6: the hand-written 12-warp kernel with a TMA fill.
 *
 * M5-C measured where the M5-B kernel's time goes at LARGE_P3/B5888 (FP64_M5C_REPORT.md §3.1):
 * 1.29 ms = 0.26 fill-issue + 0.23 fill-traffic + 0.80 compute.  TMA removes the issue cost; the
 * price is that the smem tile must be a canonical SW128 tile, so the fragment fetch becomes 32
 * eight-byte LDS instead of 16 LDS.128.  What the 12-warp/128-thread structure buys back, and what
 * no stock path can do at that width, is the 128x32 tile: V=224 becomes seven exact N tiles.
 *
 * Tile instantiations: 64x64 (V % 64 == 0) and 128x32 (exact-N for V=224 / V=96).  Both run
 * 128-thread CTAs, 3 per SM.  A always goes through the composed (M, U, m) view (no permute).
 **************************************************************************************************/
#include "direct_gemm_f64_tma.cuh"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "cute/tensor.hpp"

#include "cutlass/arch/barrier.h"

#include "collective/fp64_dmma_atom.hpp"
#include "device_buffer.h"
#include "prep_kernels.h"

using namespace cute;

namespace eqlin {
namespace {

using eqlin::fp64::Op32x32x16;

// ---- tile configuration -------------------------------------------------------------------------
// The smem layouts are built the way sm90_mma_fp64_array_tma_warpspecialized.hpp:131-141 does it
// (rank-3 pipelined tile, sliced per stage) -- that shape is what make_tma_copy expects.
template <int TM, int TN>
struct Cfg {
  static constexpr int tb_m = TM, tb_n = TN, bk = 16;
  static constexpr int threads = 128;
  static constexpr bool exact_n = (TN == 32);
  static constexpr int stages = exact_n ? 3 : 4;  // keep 3 CTAs/SM inside 227 KB
  static constexpr int a_bytes = tb_m * bk * 8;
  static constexpr int b_bytes = tb_n * bk * 8;
  static constexpr int stage_bytes = a_bytes + b_bytes;
  static constexpr int smem_bytes = stages * stage_bytes;
  using TileShape = Shape<Int<tb_m>, Int<tb_n>, Int<bk>>;
  using SmemLayoutA = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<double>{},
      make_shape(Int<tb_m>{}, Int<bk>{}, Int<stages>{}), Step<_1, _2, _3>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<double>{},
      make_shape(Int<tb_n>{}, Int<bk>{}, Int<stages>{}), Step<_1, _2, _3>{}));
};

struct Problem {
  double const *a;
  double const *b;
  double *c;
  int M, N, K;
  int K_total;   // I*U: the flat extent of the A descriptor
  int m_stride;
  int lda, ldb, ldc;
};

struct Tile {
  int problem;
  int mt, nt;
};

/// A is described as a *flat* (M, I*U, 1) tensor: the composed path blocks are reached by feeding
/// the TMA a k coordinate of `path*m_stride + u0`, which is the same address the nested (U, m)
/// descriptor would produce.  A nested gmem mode makes cute's cta_v_map non-constant (the box
/// mapping through a runtime-extent mode), which is exactly what broke the first attempt.
using FlatStrideA = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using FlatStrideB = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

template <int TM, int TN>
struct TmaTypes {
  using CfgT = Cfg<TM, TN>;
  using TMA_A = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(static_cast<double const *>(nullptr), repeat_like(FlatStrideA{}, int32_t(0)),
                  FlatStrideA{}),
      typename CfgT::SmemLayoutA{}(_, _, Int<0>{}), make_shape(Int<TM>{}, Int<16>{}), Int<1>{}));
  using TMA_B = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(static_cast<double const *>(nullptr), repeat_like(FlatStrideB{}, int32_t(0)),
                  FlatStrideB{}),
      typename CfgT::SmemLayoutB{}(_, _, Int<0>{}), make_shape(Int<TN>{}, Int<16>{}), Int<1>{}));
};

template <int TM, int TN>
struct Params {
  using TMA_A = typename TmaTypes<TM, TN>::TMA_A;
  using TMA_B = typename TmaTypes<TM, TN>::TMA_B;

  TMA_A tma_a[kMaxPaths];
  TMA_B tma_b[kMaxPaths];
  Problem probs[kMaxPaths];
  int num_probs;
  int u;
  int m_max;
};

/// Fragment fetch from the canonical SW128 tile.  The addressing is cute's own layout algebra --
/// the same layout the TMA descriptor was built from -- so the LDS offsets cannot drift from what
/// the hardware wrote.  32 eight-byte loads per k-tile (3.x's shape); the 16-element groups that
/// a 128 B swizzle permutes stay contiguous, so this is the cost the 12 warps have to absorb.
template <int TM, int TN>
__device__ __forceinline__ void fetch_stage(char *smem, int stage, int warp, int lane,
                                           double (&a)[2][8], double (&b)[4][4]) {
  using CfgT = Cfg<TM, TN>;
  Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<double *>(smem)),
                          typename CfgT::SmemLayoutA{});
  // B starts after ALL of A's stages: both layouts are stage-major, so one A stage of offset
  // would land inside A's second stage.
  Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<double *>(smem) +
                                        CfgT::stages * CfgT::a_bytes / 8),
                          typename CfgT::SmemLayoutB{});
  Tensor sAs = sA(_, _, stage);  // (TM, BK)
  Tensor sBs = sB(_, _, stage);  // (TN, BK)
  int const m0 = lane >> 2, k0 = lane & 3;
  int const wm = warp / (TN / 32), wn = warp % (TN / 32);
  int const ma = wm * 32 + m0, nb = wn * 32 + m0;
#pragma unroll
  for (int v1 = 0; v1 < 4; ++v1)
#pragma unroll
    for (int v0 = 0; v0 < 2; ++v0)
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
        a[mi][v0 + 2 * v1] = sAs(ma + 8 * v0 + 16 * mi, k0 + 4 * v1);
#pragma unroll
  for (int ni = 0; ni < 4; ++ni)
#pragma unroll
    for (int c = 0; c < 4; ++c) b[ni][c] = sBs(nb + 8 * ni, k0 + 4 * c);
}

__device__ __forceinline__ void mma_frags(double const (&a)[2][8], double const (&b)[4][4],
                                          double (&c)[8][4]) {
  Op32x32x16::fma(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], a[0], a[1], b[0], b[1], b[2],
                  b[3], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}

template <int TM, int TN>
__device__ __forceinline__ void store_tile(Problem const &p, int m0, int n0, int warp, int lane,
                                           double const (&c)[8][4]) {
  int const wm = warp / (TN / 32), wn = warp % (TN / 32);
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

/// TMA issue for one stage: one thread arms the transaction barrier and fires both copies.
template <int TM, int TN>
__device__ __forceinline__ void tma_fill(Params<TM, TN> const &params, int j, int mt, int nt, int kt,
                                         char *smem, int stage, uint64_t &bar) {
  using CfgT = Cfg<TM, TN>;
  Problem const &p = params.probs[j];
  cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&bar, CfgT::a_bytes +
                                                                          CfgT::b_bytes);

  Tensor mA = params.tma_a[j].get_tma_tensor(make_shape(p.M, p.K_total, 1));
  Tensor gA = local_tile(mA, make_tile(Int<TM>{}, Int<16>{}, Int<1>{}), make_coord(_, _, _));
  int const u_tiles = params.u / 16;
  // Flat k-tile index: the path block starts at (kt/u_tiles)*m_stride elements.
  int const k_tile_flat = (kt / u_tiles) * (p.m_stride / 16) + (kt % u_tiles);
  Tensor gA_tile = gA(_, _, Int<0>{}, mt, k_tile_flat, Int<0>{});  // (TM, BK)

  Tensor mB = params.tma_b[j].get_tma_tensor(make_shape(p.N, p.K, 1));
  Tensor gB = local_tile(mB, make_tile(Int<TN>{}, Int<16>{}, Int<1>{}), make_coord(_, _, _));
  Tensor gB_tile = gB(_, _, Int<0>{}, nt, kt, Int<0>{});  // (TN, BK)

  Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<double *>(smem)),
                          typename CfgT::SmemLayoutA{});  // (TM, BK, stages)
  // B starts after ALL of A's stages: both layouts are stage-major, so one A stage of offset
  // would land inside A's second stage.
  Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<double *>(smem) +
                                        CfgT::stages * CfgT::a_bytes / 8),
                          typename CfgT::SmemLayoutB{});
  Tensor tAgA = params.tma_a[j].get_slice(Int<0>{}).partition_S(gA_tile);
  Tensor tAsA = params.tma_a[j].get_slice(Int<0>{}).partition_D(sA(_, _, stage));
  Tensor tBgB = params.tma_b[j].get_slice(Int<0>{}).partition_S(gB_tile);
  Tensor tBsB = params.tma_b[j].get_slice(Int<0>{}).partition_D(sB(_, _, stage));
  copy(params.tma_a[j].with(bar), tAgA, tAsA);
  copy(params.tma_b[j].with(bar), tBgB, tBsB);
}

template <int TM, int TN>
__global__ void __launch_bounds__(Cfg<TM, TN>::threads, 3)
direct_tma_kernel(CUTE_GRID_CONSTANT Params<TM, TN> const params, Tile const *__restrict__ tiles,
                  int num_tiles, unsigned *__restrict__ counter) {
  using CfgT = Cfg<TM, TN>;
  // 128 B: the canonical SW128 tiles are described to TMA at that alignment.
  extern __shared__ __align__(128) char smem[];
  // Raw mbarriers: cute's TMA copy atom takes the mbarrier as a uint64_t& for .with().
  auto *bars = reinterpret_cast<uint64_t *>(smem + CfgT::smem_bytes);
  int const tid = threadIdx.x;
  int const warp = tid >> 5, lane = tid & 31;

  if (tid == 0) {
    for (int s = 0; s < CfgT::stages; ++s)
      cutlass::arch::ClusterTransactionBarrier::init(&bars[s], 1);
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  __shared__ int s_tile;
  int phase[CfgT::stages];
  for (int s = 0; s < CfgT::stages; ++s) phase[s] = 0;

  for (;;) {
    if (tid == 0) s_tile = static_cast<int>(atomicAdd(counter, 1u));
    __syncthreads();
    int const t = s_tile;
    if (t >= num_tiles) break;
    Tile const tl = tiles[t];
    Problem const p = params.probs[tl.problem];
    int const m0 = tl.mt * TM, n0 = tl.nt * TN;
    int const k_tiles = p.K / 16;

    double a[2][8], b[4][4], c[8][4];
    for (int i = 0; i < 8; ++i)
      for (int q = 0; q < 4; ++q) c[i][q] = 0.0;

    if (tid == 0) {
      for (int s = 0; s < CfgT::stages - 1 && s < k_tiles; ++s)
        tma_fill<TM, TN>(params, tl.problem, tl.mt, tl.nt, s, smem, s, bars[s]);
    }
    for (int kt = 0; kt < k_tiles; ++kt) {
      int const st = kt % CfgT::stages;
      cutlass::arch::ClusterTransactionBarrier::wait(&bars[st], phase[st]);
      phase[st] ^= 1;
      __syncthreads();
      fetch_stage<TM, TN>(smem, st, warp, lane, a, b);
      mma_frags(a, b, c);
      __syncthreads();  // everyone is done with this stage before it is refilled below
      int const pf = kt + CfgT::stages - 1;
      if (tid == 0 && pf < k_tiles)
        tma_fill<TM, TN>(params, tl.problem, tl.mt, tl.nt, pf, smem, pf % CfgT::stages,
                         bars[pf % CfgT::stages]);
    }
    __syncthreads();
    store_tile<TM, TN>(p, m0, n0, warp, lane, c);
    __syncthreads();
  }
}

template <int TM, int TN>
int smem_needed() {
  return Cfg<TM, TN>::smem_bytes + Cfg<TM, TN>::stages * 8;
}

}  // namespace

struct TmaGemmPlan::ImplBase {
  virtual ~ImplBase() = default;
  virtual int launch(double *out, cudaStream_t stream) = 0;
  virtual void bind_a(double const *a_base, cudaStream_t stream) = 0;
};

namespace {

template <int TM, int TN>
class PlanImplT : public TmaGemmPlan::ImplBase {
 public:
  using CfgT = Cfg<TM, TN>;
  using ParamsT = Params<TM, TN>;

  PlanImplT(int B, int V, int num_rows, long long lda, long long ldb, long long ldc,
            std::vector<RowPlan> const &rows, double const *a_data, double const *b_data, int u)
      : B_(B), V_(V), lda_(lda), ldb_(ldb), ldc_(ldc), u_(u) {
    if (num_rows <= 0 || num_rows > kMaxPaths) throw std::invalid_argument("TmaGemmPlan: bad rows");
    rows_ = rows;
    for (int j = 0; j < num_rows; ++j) m_max_ = std::max<int>(m_max_, rows[j].m);
    params_.num_probs = num_rows;
    params_.u = u;
    params_.m_max = m_max_;
    for (int j = 0; j < num_rows; ++j) {
      Problem &p = params_.probs[j];
      p = Problem{};
      p.a = a_data ? a_data + rows[j].a_off : nullptr;
      p.b = b_data + rows[j].b_off;
      p.c = nullptr;
      p.M = B;
      p.N = V;
      p.K = rows[j].K;
      p.K_total = static_cast<int>(lda);
      p.m_stride = static_cast<int>(rows[j].m_stride);
      p.lda = static_cast<int>(lda);
      p.ldb = static_cast<int>(ldb);
      p.ldc = static_cast<int>(ldc);
      num_tiles_ += ((B + TM - 1) / TM) * ((V + TN - 1) / TN);
    }
    if (a_data) build_tma(a_data);
    host_tiles_.reserve(num_tiles_);
    for (int j = 0; j < params_.num_probs; ++j) {
      int const mt = (B + TM - 1) / TM, nt = (V + TN - 1) / TN;
      for (int a = 0; a < mt; ++a)
        for (int b = 0; b < nt; ++b) host_tiles_.push_back(Tile{j, a, b});
    }
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, dev);
    grid_ = std::min(num_tiles_, 3 * sm_count_);
    if (grid_ <= 0) throw std::invalid_argument("TmaGemmPlan: empty grid");
    smem_bytes_ = smem_needed<TM, TN>();
    if (cudaFuncSetAttribute(direct_tma_kernel<TM, TN>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes_) != cudaSuccess)
      throw std::runtime_error("TmaGemmPlan: cannot raise the dynamic smem limit");
    if (d_tiles_.reset(num_tiles_ * sizeof(Tile)) || d_counter_.reset(sizeof(unsigned)))
      throw std::runtime_error("TmaGemmPlan: device allocation failed");
    cudaMemcpy(d_tiles_.as<Tile>(), host_tiles_.data(), num_tiles_ * sizeof(Tile),
               cudaMemcpyHostToDevice);
    bound_ = a_data != nullptr;
    if (nullptr)
      fprintf(stderr, "[tma-plan] %dx%d rows=%d tiles=%d grid=%d u=%d m_max=%d\n", TM, TN,
              params_.num_probs, num_tiles_, grid_, u_, m_max_);
  }

  void bind_a(double const *a_base, cudaStream_t) override {
    if (a_base == a_base_ && bound_) return;
    a_base_ = a_base;
    build_tma(a_base);
    bound_ = true;
  }

  int launch(double *out, cudaStream_t stream) override {
    if (!bound_) return 1;
    for (int j = 0; j < params_.num_probs; ++j)
      params_.probs[j].c = out + static_cast<long long>(j) * V_;
    cudaMemsetAsync(d_counter_.as<unsigned>(), 0, sizeof(unsigned), stream);
    direct_tma_kernel<TM, TN><<<grid_, CfgT::threads, smem_bytes_, stream>>>(
        params_, d_tiles_.as<Tile>(), num_tiles_, d_counter_.as<unsigned>());
    return 0;
  }

 private:
  void build_tma(double const *a_base) {
    for (int j = 0; j < params_.num_probs; ++j) {
      Problem &p = params_.probs[j];
      p.a = a_base + rows_[j].a_off;
      Tensor gA = make_tensor(p.a, make_layout(make_shape(p.M, p.K_total, 1),
                                               make_stride(int64_t(lda_), Int<1>{}, Int<0>{})));
      params_.tma_a[j] = make_tma_copy(SM90_TMA_LOAD{}, gA, typename CfgT::SmemLayoutA{}(_, _, Int<0>{}),
                                       make_shape(Int<TM>{}, Int<16>{}), Int<1>{});
      Tensor gB = make_tensor(p.b, make_layout(make_shape(p.N, p.K, 1),
                                               make_stride(int64_t(ldb_), Int<1>{}, Int<0>{})));
      params_.tma_b[j] = make_tma_copy(SM90_TMA_LOAD{}, gB, typename CfgT::SmemLayoutB{}(_, _, Int<0>{}),
                                       make_shape(Int<TN>{}, Int<16>{}), Int<1>{});
    }
  }

  int B_, V_, num_tiles_ = 0, grid_ = 0, sm_count_ = 0, u_ = 0, m_max_ = 1, smem_bytes_ = 0;
  long long lda_, ldb_, ldc_;
  std::vector<RowPlan> rows_;
  std::vector<Tile> host_tiles_;
  double const *a_base_ = nullptr;
  bool bound_ = false;
  ParamsT params_{};
  DeviceBuffer d_tiles_, d_counter_;
};

}  // namespace

TmaGemmPlan::TmaGemmPlan(int B, int V, int num_rows, long long lda, long long ldb, long long ldc,
                         std::vector<RowPlan> const &rows, double const *a_data,
                         double const *b_data, int u, bool exact_n) {
  if (exact_n) {
    p_ = std::make_unique<PlanImplT<128, 32>>(B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data,
                                              u);
  } else {
    p_ = std::make_unique<PlanImplT<64, 64>>(B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data,
                                             u);
  }
}

TmaGemmPlan::~TmaGemmPlan() = default;

int TmaGemmPlan::launch(double *out, cudaStream_t stream) { return p_->launch(out, stream); }

void TmaGemmPlan::bind_a(double const *a_base, cudaStream_t stream) {
  p_->bind_a(a_base, stream);
}

}  // namespace eqlin
