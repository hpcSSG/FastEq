/***************************************************************************************************
 * fasteq eqlinear backends/fp64/sm90_fp64_grouped_gemm.cu
 *
 * M2/M3 backend: the stock SM90 PtrArray cooperative array kernel with the custom FP64 DMMA TMA
 * mainloop, in two flavours:
 *
 *   flat (composed = false) -- M2: A is the packed A_prep (M, K) with a single K stride.
 *   composed (composed = true) -- M3-A: A is read straight from x as (M, U, m) with the path index
 *                               in the descriptor's batch mode and stride i_g*U.  The problem's
 *                               flat K = m*U is still what the tile scheduler iterates; the
 *                               mainloop decodes each k-tile into (m_path, u_tile).
 *
 * The two flavours only differ in the A stride type and the Params' (u, m_max), so they share
 * this host code via a template bool.  Outputs are bitwise identical for the same A bytes.
 **************************************************************************************************/
#include "sm90_fp64_grouped_gemm.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "cutlass/cutlass.h"
#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

#include "cute/tensor.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "collective/dispatch_policy_fp64.hpp"
#include "collective/fp64_dmma_atom.hpp"
#include "collective/sm90_mma_fp64_array_tma_warpspecialized.hpp"
#include "device_buffer.h"
#include "pointer_table.cuh"

using namespace cute;

namespace eqlin {
namespace {

#ifndef F64M2_TB_M
#define F64M2_TB_M 128
#endif
#ifndef F64M2_TB_N
#define F64M2_TB_N 64
#endif
#ifndef F64M2_TB_K
#define F64M2_TB_K 16
#endif
#ifndef F64M2_STAGES
#define F64M2_STAGES 4
#endif
// Shared-memory layout atom for A and B.  The composite DMMA atom reads its operands with 16-byte
// per-thread LDS, whose bank pattern does not match the WGMMA-oriented SW128 atom: M4 measured 50 %
// bank-conflict wavefronts on shared loads with SW128 (2.x: 0.5 %).  64 = SW64, 0 = no swizzle.
#ifndef F64M2_SMEM_ATOM
#define F64M2_SMEM_ATOM 128
#endif
// Epilogue smem stages (the C-buffer depth).  One epilogue tile per CTA, so 1 is the useful value;
// 2 reproduces the stock builder's choice.  Every extra stage costs TB_M*TB_N*8 bytes of smem.
#ifndef F64M2_EPI_STAGES
#define F64M2_EPI_STAGES 2
#endif
#ifndef F64M2_PINGPONG
#define F64M2_PINGPONG 0
#endif

using ElementA = double;
using ElementB = double;
using ElementC = double;
using ElementAcc = double;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
constexpr int kAlign = 2;

using TileShape = Shape<Int<F64M2_TB_M>, Int<F64M2_TB_N>, Int<F64M2_TB_K>>;
using ClusterShape = Shape<_1, _1, _1>;
using TiledMma = TiledMMA<
    MMA_Atom<fp64::Op32x32x16>,
    Layout<Shape<Int<F64M2_TB_M / 32>, Int<F64M2_TB_N / 32>, Int<F64M2_TB_K / 16>>>>;
static_assert(size(TiledMma{}) == (F64M2_PINGPONG ? 128 : 256),
              "MMA thread count must match the selected SM90 schedule");
static_assert(size<0>(TileShape{}) % 32 == 0 && size<1>(TileShape{}) % 32 == 0 &&
                  size<2>(TileShape{}) % 16 == 0,
              "TileShape must be divisible by the 32x32x16 composite atom");

using ScheduleTag = std::conditional_t<
    F64M2_PINGPONG != 0, cutlass::gemm::KernelPtrArrayFp64TmaWarpSpecializedPingpong,
    cutlass::gemm::KernelPtrArrayFp64TmaWarpSpecializedCooperative>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB*>;
#if F64M2_SMEM_ATOM == 128
using SmemLayoutAtomA = decltype(GMMA::Layout_K_SW128_Atom<double>{});
using SmemLayoutAtomB = decltype(GMMA::Layout_K_SW128_Atom<double>{});
#elif F64M2_SMEM_ATOM == 64
using SmemLayoutAtomA = decltype(GMMA::Layout_K_SW64_Atom<double>{});
using SmemLayoutAtomB = decltype(GMMA::Layout_K_SW64_Atom<double>{});
#else
using SmemLayoutAtomA = decltype(GMMA::Layout_K_INTER_Atom<double>{});
using SmemLayoutAtomB = decltype(GMMA::Layout_K_INTER_Atom<double>{});
#endif
using SmemCopyAtom = Copy_Atom<AutoVectorizingCopy, double>;

/// Flat A: (M, K, L=1) with a single K stride.
using FlatStrideA = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
/// Composed A: (M, U, m) with the path in the batch mode.
using ComposedStrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
template <bool Composed>
using StrideA_t = std::conditional_t<Composed, ComposedStrideA, FlatStrideA>;
template <bool Composed>
using StrideAPtr = StrideA_t<Composed>*;

template <bool Composed>
using MainloopT = cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm90ArrayTmaDmmaWarpSpecializedFp64<
        F64M2_STAGES, ClusterShape, ScheduleTag, Composed>,
    TileShape,
    ElementA, StrideAPtr<Composed>,
    ElementB, StrideB,
    TiledMma,
    SM90_TMA_LOAD, SmemLayoutAtomA, SmemCopyAtom, cute::identity,
    SM90_TMA_LOAD, SmemLayoutAtomB, SmemCopyAtom, cute::identity>;

/// Epilogue tile == the CTA tile.  `EpilogueTileAuto` picks (128,32) for the cooperative schedule,
/// and with 64-bit C the second subtile comes out as NaN: the builder hardcodes
/// `CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, half_t>` (sm90_builder.inl:319-330), whose thread/value
/// grouping is 16-bit-specific.  A single 128x64 tile never enters the subtile path.
///
/// The dispatch policy is built here instead of by `CollectiveBuilder`, because the stock policy
/// derives `StagesC = max(min(EpiTiles,4), StagesD+1)` (sm90_builder.inl:64-88) = 2 for a single
/// epilogue tile -- and a CTA stores exactly one epilogue tile, so the second smem stage buys
/// nothing while costing 64 KB.  That 64 KB is what pays for mainloop stages 5-6.
using EpiTile = Shape<Int<F64M2_TB_M>, Int<F64M2_TB_N>>;
using EpiDispatchPolicy = cutlass::epilogue::Sm90PtrArrayTmaWarpSpecialized<
    /*StagesC=*/F64M2_EPI_STAGES,
    /*StagesD=*/1,
    /*FragmentSize=*/decltype(cute::size(EpiTile{}))::value / (2 * 128),
    /*ReuseSmemC=*/true,
    /*DelayTmaStore=*/false,
    /*NumEpilogueWarpGroups=*/2>;

using Epilogue = typename cutlass::epilogue::collective::detail::Sm90TmaBuilderImpl<
    TileShape, EpiTile,
    ElementAcc, ElementAcc,
    ElementC, LayoutC *, kAlign,
    ElementC, LayoutC *, kAlign,
    cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAcc>,
    EpiDispatchPolicy>::CollectiveOp;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

template <bool Composed>
using GemmKernelT = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, MainloopT<Composed>, Epilogue, cutlass::gemm::GroupScheduler>;
template <bool Composed>
using GemmT = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelT<Composed>>;

using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;

}  // namespace

struct Sm90Fp64Plan::ImplBase {
  virtual ~ImplBase() = default;
  virtual int launch(double *out, cudaStream_t stream) = 0;
  virtual void bind_a(double const *a_base, cudaStream_t stream) = 0;
};

namespace {

template <bool Composed>
class PlanImpl : public Sm90Fp64Plan::ImplBase {
 public:
  using Gemm = GemmT<Composed>;
  using GemmKernel = GemmKernelT<Composed>;
  using ElementA_ = ElementA;
  using InternalStrideA = StrideA_t<Composed>;
  using InternalStrideB = typename GemmKernel::InternalStrideB;
  using InternalStrideC = typename GemmKernel::InternalStrideC;

  PlanImpl(int B, int V, int num_rows, long long lda, long long ldb, long long ldc,
           std::vector<RowPlan> const &rows, double const *a_data, double const *b_data,
           int u, int m_max)
      : num_rows_(num_rows), u_(u), m_max_(m_max), a_bound_(a_data != nullptr),
        pointers_(pointer_rows(rows, V)) {
    if (num_rows <= 0) throw std::invalid_argument("Sm90Fp64Plan: no rows");
    if (b_data == nullptr) throw std::invalid_argument("Sm90Fp64Plan: missing B");
    if constexpr (Composed) {
      if (u <= 0 || m_max <= 0) throw std::invalid_argument("Sm90Fp64Plan: bad u/m_max");
    }
    parse_scheduler_env();

    host_problem_.resize(num_rows);
    host_a_.resize(num_rows);
    host_b_.resize(num_rows);
    host_sa_.resize(num_rows);
    host_sb_.resize(num_rows);
    host_sc_.resize(num_rows);

    for (int j = 0; j < num_rows; ++j) {
      int const row_M = B;
      host_problem_[j] = UnderlyingProblemShape(row_M, V, rows[j].K);
      host_a_[j] = a_data ? static_cast<ElementA const *>(a_data + rows[j].a_off) : nullptr;
      host_b_[j] = static_cast<ElementB const *>(b_data + rows[j].b_off);
      if constexpr (Composed) {
        long long const ms = rows[j].m_stride != 0 ? rows[j].m_stride : lda;
        host_sa_[j] = make_stride(static_cast<int64_t>(lda), Int<1>{}, static_cast<int64_t>(ms));
      } else {
        long long const row_lda = rows[j].lda != 0 ? rows[j].lda : lda;
        long long const row_ldb = rows[j].ldb != 0 ? rows[j].ldb : ldb;
        long long const row_ldc = rows[j].ldc != 0 ? rows[j].ldc : ldc;
        host_sa_[j] = cutlass::make_cute_packed_stride(InternalStrideA{},
                                                       Shape<int, int, int>{row_M, (int)row_lda, 1});
        host_sb_[j] = cutlass::make_cute_packed_stride(InternalStrideB{},
                                                       Shape<int, int, int>{V, (int)row_ldb, 1});
        host_sc_[j] = cutlass::make_cute_packed_stride(InternalStrideC{},
                                                       Shape<int, int, int>{row_M, (int)row_ldc, 1});
      }
    }

    size_t const n = static_cast<size_t>(num_rows);
    if (d_problem_.reset(n * sizeof(UnderlyingProblemShape)) ||
        d_a_.reset(n * sizeof(ElementA const *)) ||
        d_b_.reset(n * sizeof(ElementB const *)) ||
        d_c_.reset(n * sizeof(ElementC *)) ||
        d_cc_.reset(n * sizeof(ElementC const *)) ||
        d_sa_.reset(n * sizeof(InternalStrideA)) ||
        d_sb_.reset(n * sizeof(InternalStrideB)) ||
        d_sc_.reset(n * sizeof(InternalStrideC)) ||
        d_sd_.reset(n * sizeof(InternalStrideC))) {
      throw std::runtime_error("Sm90Fp64Plan: device allocation failed");
    }

    cudaMemcpy(d_problem_.as<UnderlyingProblemShape>(), host_problem_.data(),
               n * sizeof(UnderlyingProblemShape), cudaMemcpyHostToDevice);
    if (a_data) {
      cudaMemcpy(d_a_.as<ElementA const *>(), host_a_.data(), n * sizeof(ElementA const *),
                 cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_b_.as<ElementB const *>(), host_b_.data(), n * sizeof(ElementB const *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sa_.as<InternalStrideA>(), host_sa_.data(), n * sizeof(InternalStrideA),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sb_.as<InternalStrideB>(), host_sb_.data(), n * sizeof(InternalStrideB),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc_.as<InternalStrideC>(), host_sc_.data(), n * sizeof(InternalStrideC),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sd_.as<InternalStrideC>(), host_sc_.data(), n * sizeof(InternalStrideC),
               cudaMemcpyHostToDevice);

    hw_info_.device_id = 0;
    cudaGetDevice(&hw_info_.device_id);
    hw_info_.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info_.device_id);

    op_ = std::make_unique<Gemm>();
    {
      typename Gemm::Arguments args = make_arguments(d_cc_.as<ElementC const *>(),
                                                     d_c_.as<ElementC *>(), u, m_max);
      cutlass::Status st = op_->can_implement(args);
      if (st != cutlass::Status::kSuccess) throw std::runtime_error("Sm90Fp64Plan: can_implement");
      size_t const ws = Gemm::get_workspace_size(args);
      if (nullptr) fprintf(stderr, "[fp64-sm90] ctor workspace=%zu\n", ws);
      if (ws && d_workspace_.reset(ws)) throw std::runtime_error("Sm90Fp64Plan: workspace");
      st = op_->initialize(args, d_workspace_.as<void>());
      if (st != cutlass::Status::kSuccess) throw std::runtime_error("Sm90Fp64Plan: initialize");
    }
  }

  ~PlanImpl() override = default;

  void bind_a(double const *a_base, cudaStream_t stream) override {
    auto const *pa = static_cast<ElementA const *>(a_base);
    if (pa == last_a_base_) return;
    if (pointers_.input(pa, d_a_.as<double *>(), stream) != cudaSuccess)
      throw std::runtime_error("Sm90Fp64Plan: input pointer update failed");
    last_a_base_ = pa;
    a_bound_ = true;
  }

  int launch(double *out, cudaStream_t stream) override {
    if (!a_bound_) return static_cast<int>(cutlass::Status::kErrorInternal);

    if (pointers_.output(out, d_c_.as<ElementC *>(), d_cc_.as<ElementC const *>(), stream)
        != cudaSuccess) return static_cast<int>(cutlass::Status::kErrorInternal);

    typename Gemm::Arguments args = make_arguments(d_cc_.as<ElementC const *>(),
                                                   d_c_.as<ElementC *>(), u_, m_max_);
    cutlass::Status st = op_->initialize(args, d_workspace_.as<void>(), stream);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st);
    return static_cast<int>(op_->run(stream));
  }

 private:
  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  /// Tile-order knobs.  The persistent group scheduler walks its *minor* axis fastest, and under
  /// AlongN the minor axis is M while under AlongM it is N.  Measured with an interleaved in-process
  /// A/B on the two multi-path configs (all five B): AlongN/1 is 3.7 % faster at LARGE_P3/B5888 and
  /// 1.8 % at MEDIUM_P2/B5888, neutral or slightly better on the other eight points, so it is the
  /// default; `EQLIN_SM90_RASTER`/`EQLIN_SM90_SWIZZLE` restore the stock AlongM/1 or sweep others.
  void parse_scheduler_env() {
    char const *r = nullptr;
    if (r != nullptr && *r != '\0') {
      if (!strcmp(r, "alongn")) {
        raster_ = RasterOrderOptions::AlongN;
      } else if (!strcmp(r, "alongm")) {
        raster_ = RasterOrderOptions::AlongM;
      } else {
        throw std::invalid_argument("Sm90Fp64Plan: EQLIN_SM90_RASTER must be alongm|alongn");
      }
    }
    char const *s = nullptr;
    if (s != nullptr && *s != '\0') {
      int const v = atoi(s);
      if (v != 1 && v != 2 && v != 4 && v != 8) {
        throw std::invalid_argument("Sm90Fp64Plan: EQLIN_SM90_SWIZZLE must be 1, 2, 4 or 8");
      }
      max_swizzle_ = v;
    }
    if (nullptr) {
      fprintf(stderr, "[sm90-plan] raster=%s swizzle=%d\n",
              raster_ == RasterOrderOptions::AlongN ? "alongn" : "alongm", max_swizzle_);
    }
  }

  typename Gemm::Arguments make_arguments(ElementC const **ptr_c, ElementC **ptr_d,
                                          int u, int m_max) {
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_rows_, d_problem_.as<UnderlyingProblemShape>(), host_problem_.data()},
        {d_a_.as<ElementA const *>(), d_sa_.as<InternalStrideA>(),
         d_b_.as<ElementB const *>(), d_sb_.as<InternalStrideB>(), u, m_max},
        {{ElementAcc(1), ElementAcc(0)}, ptr_c, d_sc_.as<InternalStrideC>(),
         ptr_d, d_sd_.as<InternalStrideC>()},
        hw_info_};
    args.scheduler.raster_order = raster_;
    args.scheduler.max_swizzle_size = max_swizzle_;
    return args;
  }

  int num_rows_ = 0;
  int u_ = 0;
  int m_max_ = 1;
  RasterOrderOptions raster_ = RasterOrderOptions::AlongN;  // measured (see parse_scheduler_env)
  int max_swizzle_ = 1;
  bool a_bound_ = false;
  ElementA const *last_a_base_ = nullptr;

  std::vector<UnderlyingProblemShape> host_problem_;
  std::vector<ElementA const *> host_a_;
  std::vector<ElementB const *> host_b_;
  std::vector<InternalStrideA> host_sa_;
  std::vector<InternalStrideB> host_sb_;
  std::vector<InternalStrideC> host_sc_;
  PointerTable<double> pointers_;

  DeviceBuffer d_problem_, d_a_, d_b_, d_c_, d_cc_, d_sa_, d_sb_, d_sc_, d_sd_, d_workspace_;
  std::unique_ptr<Gemm> op_;
  cutlass::KernelHardwareInfo hw_info_{};
};

}  // namespace

bool sm90_fp64_available() { return true; }

Sm90Fp64Plan::Sm90Fp64Plan(int B, int V, int num_rows, long long lda, long long ldb, long long ldc,
                           std::vector<RowPlan> const &rows, double const *a_data,
                           double const *b_data, bool composed, int u, int m_max) {
  if (composed) {
    p_ = std::make_unique<PlanImpl<true>>(B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data,
                                          u, m_max);
  } else {
    p_ = std::make_unique<PlanImpl<false>>(B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data,
                                           0, 1);
  }
}

Sm90Fp64Plan::~Sm90Fp64Plan() = default;

int Sm90Fp64Plan::launch(double *out, cudaStream_t stream) { return p_->launch(out, stream); }

void Sm90Fp64Plan::bind_a(double const *a_base, cudaStream_t stream) {
  p_->bind_a(a_base, stream);
}

}  // namespace eqlin

#else  // !CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED

namespace eqlin {

bool sm90_fp64_available() { return false; }

Sm90Fp64Plan::Sm90Fp64Plan(int, int, int, long long, long long, long long,
                           std::vector<RowPlan> const &, double const *, double const *, bool,
                           int, int) {
  throw std::runtime_error("SM90 FP64 collective not available in this build");
}
Sm90Fp64Plan::~Sm90Fp64Plan() = default;
int Sm90Fp64Plan::launch(double *, cudaStream_t) { return 1; }
void Sm90Fp64Plan::bind_a(double const *, cudaStream_t) {}

}  // namespace eqlin

#endif
