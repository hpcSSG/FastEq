/***************************************************************************************************
 * fasteq eqlinear tf32/csrc/sm90_grouped_gemm.cu
 *
 * Instantiation follows test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_group_gemm.cu
 * and examples/57_hopper_grouped_gemm in structure.
 *
 * Three operand modes share one launch path (see sm90_grouped_gemm.cuh for what they mean):
 *   kFp32                -- stock mainloop, float operands (1xTF32 probe)
 *   kTf32KConcat         -- stock mainloop, K-concatenated tf32x3 operands (M2)
 *   kFp32InKernelSplit   -- custom mainloop, raw fp32 A + pre-split tf32 B (M3-2)
 **************************************************************************************************/
#include "sm90_grouped_gemm.cuh"

#include <cstdio>
#include <type_traits>
#include <cstdlib>
#include <stdexcept>

#include "cutlass/cutlass.h"
// Needed before the feature-test below: the macro lives here, and cutlass.h alone does not
// necessarily pull it in ahead of the SM90-only headers.
#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "collective/collective_builder_tf32x3.hpp"
#include "collective/collective_builder_tf32x3_staged.hpp"
#include "collective/dispatch_policy_tf32x3.hpp"
#include "collective/tile_scheduler_streamk_group.hpp"
#include "device_buffer.h"

using namespace cute;

namespace eqlin {
namespace {

using LayoutA = cutlass::layout::RowMajor;     // (M, K) row-major = K-major
using LayoutB = cutlass::layout::ColumnMajor;  // (N, K) row-major = K-major (GMMA requires K-major B)
using LayoutC = cutlass::layout::RowMajor;     // (M, N) row-major
using ElementC = float;
using ElementAcc = float;

constexpr int kAlign = 4;  // 16 bytes / 4 bytes == 4 elements.  Required by the TMA path.

#ifndef EQLIN_TB_M
#define EQLIN_TB_M 128
#endif
#ifndef EQLIN_TB_N
#define EQLIN_TB_N 128
#endif
#ifndef EQLIN_TB_K
#define EQLIN_TB_K 32
#endif
#ifndef EQLIN_TB_K3
// K-tile for the in-kernel-split tf32x3 mainloop.  That mainloop's pipeline stage holds four tiles
// (A_big, A_res, B_big, B_res) instead of two, so at TB_K = 32 an N = 256 tile would leave room for
// a single stage -- below what `MainloopSm90ArrayTmaGmmaWarpSpecializedTf32x3` requires.  Halving
// the K-tile keeps three stages in the same smem budget.
#define EQLIN_TB_K3 16
#endif
#ifndef EQLIN_CLUSTER_N
#define EQLIN_CLUSTER_N 1        // cluster extent along N: >1 turns on A multicast
#endif
#ifndef EQLIN_CLUSTER_M
// Cluster extent along M when the `EQLIN_CLUSTER_M` environment variable is unset.  The extent
// itself is a run-time choice; this macro only fixes the default a build starts from, so a build
// configured with -DEQLIN_CLUSTER_M=2 still behaves as it did before the knob existed.
#define EQLIN_CLUSTER_M 1
#endif
#ifndef EQLIN_PINGPONG
#define EQLIN_PINGPONG 0         // 1 -> Pingpong schedule (256 threads) instead of Cooperative (384)
#endif
#ifndef EQLIN_STAGES
#define EQLIN_STAGES 0           // 0 -> StageCountAutoCarveout, else an explicit stage count
#endif

/// Cluster shape.  The N axis has nothing to multicast here -- with TB_N >= V every row has exactly
/// one N-tile, so an N-cluster only ever broadcasts into a padding tile.  The M axis is the one that
/// can pay: B (the weights) is the same tensor for every M-tile of a row and is ~80% of the bytes a
/// K-tile pulls from L2, so a (2,1,1) cluster asks for it once per pair of CTAs.  What multicast
/// cannot do is reduce the per-CTA smem fill -- each CTA still receives its own copy -- so it only
/// pays while the L2 read-request rate is what the kernel waits on.  `microbench/cluster_tma_broadcast`
/// (M6-C) measures where that condition holds, and M6-D confirms on this kernel that it does not
/// hold here: over all 25 shapes the paired kernel time moves a median 0.3% (M6_REPORT.md D.3/D.4),
/// so the extent stays a diagnostic knob and nothing selects it by shape.
///
/// The M extent is a *template* parameter rather than a macro because the two extents have to be
/// measurable inside one process: two `.so` builds cannot be loaded together (the op names would
/// collide) and a cross-process comparison carries this box's 2.5-5.6% per-point session noise,
/// which is larger than the effect being measured.
template <int CM>
using ClusterShapeT = Shape<Int<CM>, Int<EQLIN_CLUSTER_N>, _1>;

#if EQLIN_PINGPONG
using KernelScheduleTag   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using EpilogueScheduleTag = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
#else
using KernelScheduleTag   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueScheduleTag = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
#endif
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

/// M is always 128 (Cooperative requires it) and K is small so the pipeline stays deep; N is left
/// open because the right choice depends on V: V <= 128 wants a 128-wide tile, V = 224 wants 256
/// (one N-tile instead of two, halving the A traffic).  The plan picks at run time.
template <int TBN, int TBK = EQLIN_TB_K>
using TileShapeT = Shape<Int<EQLIN_TB_M>, Int<TBN>, Int<TBK>>;

// The epilogue does not depend on the operand element types, so it is instantiated once at
// namespace scope.  (Instantiating it inside the operand-parameterised template and then taking
// `sizeof(SharedStorage)` for StageCountAutoCarveout trips a front-end error in this toolchain.)
template <int TBN, int CM>
using CollectiveEpilogueT = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShapeT<TBN>, ClusterShapeT<CM>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutC *, kAlign,
    ElementC, LayoutC *, kAlign,
    EpilogueScheduleTag,
    cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAcc>
  >::CollectiveOp;

template <int TBN, int CM>
constexpr int epilogue_smem_bytes() {
  return static_cast<int>(sizeof(typename CollectiveEpilogueT<TBN, CM>::SharedStorage));
}

/// Stage count: either explicit or auto-derived from the smem budget minus the epilogue carveout.
template <int TBN, int CM>
using StageCountT = std::conditional_t<
    (EQLIN_STAGES > 0),
    cute::Int<EQLIN_STAGES>,
    cutlass::gemm::collective::StageCountAutoCarveout<epilogue_smem_bytes<TBN, CM>()>>;

/// Operand element types per mode.  kFp32InKernelSplit keeps A in fp32 (the residual lives in the
/// low mantissa bits, which a TMA-side tf32 convert would round away) and B in tf32 (already split).
template <Sm90Operands Mode> struct OperandTypes;
template <> struct OperandTypes<Sm90Operands::kFp32> {
  using A = float; using B = float;
};
template <> struct OperandTypes<Sm90Operands::kTf32KConcat> {
  using A = cutlass::tfloat32_t; using B = cutlass::tfloat32_t;
};
template <> struct OperandTypes<Sm90Operands::kFp32InKernelSplit> {
  using A = float; using B = cutlass::tfloat32_t;
};
template <> struct OperandTypes<Sm90Operands::kFp32RegSplit> {
  using A = float; using B = cutlass::tfloat32_t;
};
template <> struct OperandTypes<Sm90Operands::kFp32RegSplitRn> {
  using A = float; using B = cutlass::tfloat32_t;
};
template <> struct OperandTypes<Sm90Operands::kFp32RegSplitFold> {
  using A = float; using B = cutlass::tfloat32_t;
};
template <> struct OperandTypes<Sm90Operands::kFp32RegSplitRnFold> {
  using A = float; using B = cutlass::tfloat32_t;
};

/// The custom tf32x3 collective only exists in the Cooperative flavour (its MMA atom layout
/// assumes two M-warpgroups), so it ignores EQLIN_PINGPONG.
template <Sm90Operands Mode> struct SchedulesOf;
template <> struct SchedulesOf<Sm90Operands::kFp32> {
  using Mainloop = KernelScheduleTag; using Epilogue = EpilogueScheduleTag;
};
template <> struct SchedulesOf<Sm90Operands::kTf32KConcat> {
  using Mainloop = KernelScheduleTag; using Epilogue = EpilogueScheduleTag;
};
template <> struct SchedulesOf<Sm90Operands::kFp32InKernelSplit> {
  using Mainloop = cutlass::gemm::KernelPtrArrayTf32x3TmaWarpSpecializedCooperative;
  using Epilogue = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
};
template <> struct SchedulesOf<Sm90Operands::kFp32RegSplit> {
  using Mainloop = cutlass::gemm::KernelPtrArrayTf32x3RsTmaWarpSpecializedCooperative;
  using Epilogue = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
};
template <> struct SchedulesOf<Sm90Operands::kFp32RegSplitRn> {
  using Mainloop = cutlass::gemm::KernelPtrArrayTf32x3RsStageTmaWarpSpecializedCooperative<true>;
  using Epilogue = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
};
template <> struct SchedulesOf<Sm90Operands::kFp32RegSplitFold> {
  using Mainloop = cutlass::gemm::KernelPtrArrayTf32x3RsStageTmaWarpSpecializedCooperative<false, true>;
  using Epilogue = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
};
template <> struct SchedulesOf<Sm90Operands::kFp32RegSplitRnFold> {
  using Mainloop = cutlass::gemm::KernelPtrArrayTf32x3RsStageTmaWarpSpecializedCooperative<true, true>;
  using Epilogue = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
};

/// K-tile per mode.  The tf32x3 mainloops halve it: the SS one because a stage carries four tiles
/// instead of two, and the RS one because a register-staged A keeps the same K-tile so the two are
/// directly comparable (see M4_REPORT.md).
template <Sm90Operands Mode> constexpr int kTbK = EQLIN_TB_K;
template <> constexpr int kTbK<Sm90Operands::kFp32InKernelSplit> = EQLIN_TB_K3;
template <> constexpr int kTbK<Sm90Operands::kFp32RegSplit> = EQLIN_TB_K3;
template <> constexpr int kTbK<Sm90Operands::kFp32RegSplitRn> = EQLIN_TB_K3;
template <> constexpr int kTbK<Sm90Operands::kFp32RegSplitFold> = EQLIN_TB_K3;
template <> constexpr int kTbK<Sm90Operands::kFp32RegSplitRnFold> = EQLIN_TB_K3;

/// True when the mainloop takes a second, residual B operand alongside the A and B pointers.  Both
/// tf32x3 modes do; it is what distinguishes their `Arguments` aggregate from the stock shape.
template <Sm90Operands Mode> constexpr bool kHasBRes = false;
template <> constexpr bool kHasBRes<Sm90Operands::kFp32InKernelSplit> = true;
template <> constexpr bool kHasBRes<Sm90Operands::kFp32RegSplit> = true;
template <> constexpr bool kHasBRes<Sm90Operands::kFp32RegSplitRn> = true;
template <> constexpr bool kHasBRes<Sm90Operands::kFp32RegSplitFold> = true;
template <> constexpr bool kHasBRes<Sm90Operands::kFp32RegSplitRnFold> = true;

/// True when this mode's mainloop is the numeric fork, whose `Arguments` carries `fold_k`.  That is
/// a property of the *type*, not of whether grading is switched on: impl 6 is the fork with
/// `Fold = false` and still has the field.
template <Sm90Operands Mode> constexpr bool kHasFold = false;
template <> constexpr bool kHasFold<Sm90Operands::kFp32RegSplitRn> = true;
template <> constexpr bool kHasFold<Sm90Operands::kFp32RegSplitFold> = true;
template <> constexpr bool kHasFold<Sm90Operands::kFp32RegSplitRnFold> = true;

template <int TBN, int CM, Sm90Operands Mode>
using CollectiveMainloopT = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    typename OperandTypes<Mode>::A, LayoutA *, kAlign,
    typename OperandTypes<Mode>::B, LayoutB *, kAlign,
    ElementAcc,
    TileShapeT<TBN, kTbK<Mode>>, ClusterShapeT<CM>,
    StageCountT<TBN, CM>,
    typename SchedulesOf<Mode>::Mainloop
  >::CollectiveOp;

/// `Sched` is the tile-scheduler *tag* handed to the array kernel as `GemmUniversal`'s fourth
/// template argument.  It defaults to the stock `GroupScheduler`, so every instantiation that does
/// not name one keeps the pre-M7 scheduler and its generated code.
template <int TBN, int CM, Sm90Operands Mode, class Sched = cutlass::gemm::GroupScheduler>
using GemmT = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<ProblemShape,
                                         CollectiveMainloopT<TBN, CM, Mode>,
                                         CollectiveEpilogueT<TBN, CM>,
                                         Sched>>;

/// Whether this instantiation's scheduler is the M7 stream-K one.  Used to decide whether a plan
/// has a decomposition to compute and hand over -- nothing else branches on the scheduler.
template <class Sched>
constexpr bool kSchedulesStreamK = std::is_same_v<Sched, StreamKGroupScheduler>;

/// The two L2-locality knobs, read from the environment once per plan rather than once per launch.
/// They live here rather than in `make_arguments` because the swizzle is also a *precondition* of
/// the split (the tile space this scheduler counts and the base's swizzle-padded decode agree only
/// while the swizzle cannot pad), so the value the decomposition is computed with and the value the
/// kernel is launched with have to be the same one.
inline int swizzle_from_env() {
  char const *z = nullptr;
  return z == nullptr ? 1 : atoi(z);
}

/// P1: K-tiles per accumulation group.  Read once per plan construction, like the other knobs
/// below.  0 (the default) leaves the chain untouched, so a build that never sets it behaves
/// exactly as the ungated path.
inline int fold_k_from_env() {
  char const *f = nullptr;
  return f == nullptr ? 0 : atoi(f);
}

inline cutlass::gemm::kernel::detail::RasterOrderOptions raster_from_env() {
  char const *r = nullptr;
  return (r != nullptr && atoi(r) == 1)
             ? cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN
             : cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
}

}  // namespace

struct Sm90GroupedPlan::ImplBase {
  virtual ~ImplBase() = default;
  virtual int launch(float *out, cudaStream_t stream) = 0;
  /// Default: no activation view, so there is nothing to bind.
  virtual void bind_a(void const *a_base, cudaStream_t stream) {
    (void)a_base;
    (void)stream;
  }
};

bool sm90_available() { return true; }

namespace {

/// Implementation parameterised on the tile-N, the operand mode and (via the mode) the element
/// types and schedules.  The host-side metadata (per-row pointers, strides, problem shapes) and the
/// workspace handling are identical across modes; only the mainloop `Arguments` differ.
template <int TBN, int CM, Sm90Operands Mode, class Sched = cutlass::gemm::GroupScheduler>
class Sm90PlanImpl : public Sm90GroupedPlan::ImplBase {
 public:
  using Gemm = GemmT<TBN, CM, Mode, Sched>;
  using ElementA_ = typename OperandTypes<Mode>::A;
  using ElementB_ = typename OperandTypes<Mode>::B;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
  /// The kernel's own scheduler class -- the array kernel rewrites the tag through
  /// `TileSchedulerSelector`, so the class that carries `compute_decomposition` is only reachable
  /// through the kernel.
  using StreamKScheduler = typename Gemm::GemmKernel::TileScheduler;

  Sm90PlanImpl(int B, int V, int num_rows, long long lda, long long ldb, long long ldc,
               std::vector<RowPlan> const &rows, void const *a_data, void const *b_data,
               void const *b_res_data)
      : num_rows_(num_rows), ldc_(ldc), c_col_stride_(V), rows_(rows),
        a_bound_(a_data != nullptr) {
    if (num_rows <= 0) throw std::invalid_argument("Sm90GroupedPlan: no rows");
    if constexpr (kHasBRes<Mode>) {
      if (b_res_data == nullptr) throw std::invalid_argument("Sm90GroupedPlan: missing B residual");
    }

    host_problem_.resize(num_rows);
    host_a_.resize(num_rows);
    host_b_.resize(num_rows);
    host_b_res_.resize(num_rows);
    host_sa_.resize(num_rows);
    host_sb_.resize(num_rows);
    host_sc_.resize(num_rows);

    auto const *pa = static_cast<ElementA_ const *>(a_data);
    auto const *pb = static_cast<ElementB_ const *>(b_data);
    auto const *pb_res = static_cast<ElementB_ const *>(b_res_data);

    for (int j = 0; j < num_rows; ++j) {
      int const row_M = rows[j].M != 0 ? rows[j].M : B;
      host_problem_[j] = UnderlyingProblemShape(row_M, V, rows[j].K);
      // A null `a_data` means the caller has no packed A at all: the activation-view mode supplies
      // the base through `bind_a` before the first launch, and a *fused* backward plan marks the
      // rows whose base is not `a_data` with `a_base`.
      auto const *pa_row = rows[j].a_base ? static_cast<ElementA_ const *>(rows[j].a_base) : pa;
      host_a_[j] = pa_row ? pa_row + rows[j].a_off : nullptr;
      auto const *pb_row = rows[j].b_base ? static_cast<ElementB_ const *>(rows[j].b_base) : pb;
      host_b_[j] = pb_row ? pb_row + rows[j].b_off : nullptr;
      // Both B tensors are allocated with the same leading dimension and the same per-row column
      // offsets, so one stride array serves them.
      auto const *pb_res_row =
          rows[j].b_res_base ? static_cast<ElementB_ const *>(rows[j].b_res_base) : pb_res;
      host_b_res_[j] = pb_res_row ? pb_res_row + rows[j].b_off : nullptr;
      // `make_cute_packed_stride` with the *actual* leading dimension substituted for the
      // contracted mode: the operands are column-strips of larger packed tensors (A_prep has
      // lda columns while K = g*U), so the packed stride would be wrong.
      // Each group carries its own base pointer, so the L mode is never stepped.
      long long const row_lda = rows[j].lda != 0 ? rows[j].lda : lda;
      long long const row_ldb = rows[j].ldb != 0 ? rows[j].ldb : ldb;
      long long const row_ldc = rows[j].ldc != 0 ? rows[j].ldc : ldc;
      host_sa_[j] = cutlass::make_cute_packed_stride(StrideA{}, Shape<int, int, int>{row_M, (int)row_lda, 1});
      host_sb_[j] = cutlass::make_cute_packed_stride(StrideB{}, Shape<int, int, int>{V, (int)row_ldb, 1});
      host_sc_[j] = cutlass::make_cute_packed_stride(StrideC{}, Shape<int, int, int>{row_M, (int)row_ldc, 1});
    }

    size_t const n = static_cast<size_t>(num_rows);
    bool const need_b_res = (kHasBRes<Mode>);
    if (d_problem_.reset(n * sizeof(UnderlyingProblemShape)) ||
        d_a_.reset(n * sizeof(ElementA_ const *)) || d_b_.reset(n * sizeof(ElementB_ const *)) ||
        d_c_.reset(n * sizeof(ElementC *)) || d_cc_.reset(n * sizeof(ElementC const *)) ||
        d_sa_.reset(n * sizeof(StrideA)) ||
        d_sb_.reset(n * sizeof(StrideB)) || d_sc_.reset(n * sizeof(StrideC)) ||
        (need_b_res && d_b_res_.reset(n * sizeof(ElementB_ const *)))) {
      throw std::runtime_error("Sm90GroupedPlan: device allocation failed");
    }
    if (cudaMallocHost(&host_c_pinned_, n * sizeof(ElementC *)) != cudaSuccess ||
        cudaMallocHost(&host_a_pinned_, n * sizeof(ElementA_ const *)) != cudaSuccess) {
      throw std::runtime_error("Sm90GroupedPlan: pinned allocation failed");
    }

    cudaMemcpy(d_problem_.as<UnderlyingProblemShape>(), host_problem_.data(),
               n * sizeof(UnderlyingProblemShape), cudaMemcpyHostToDevice);
    if (a_bound_) {
      cudaMemcpy(d_a_.as<ElementA_ const *>(), host_a_.data(), n * sizeof(ElementA_ const *),
                 cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_b_.as<ElementB_ const *>(), host_b_.data(), n * sizeof(ElementB_ const *),
               cudaMemcpyHostToDevice);
    if (need_b_res) {
      cudaMemcpy(d_b_res_.as<ElementB_ const *>(), host_b_res_.data(),
                 n * sizeof(ElementB_ const *), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_sa_.as<StrideA>(), host_sa_.data(), n * sizeof(StrideA), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sb_.as<StrideB>(), host_sb_.data(), n * sizeof(StrideB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc_.as<StrideC>(), host_sc_.data(), n * sizeof(StrideC), cudaMemcpyHostToDevice);

    // Reuse the de-quantised shapes for the offset arrays: D shares C's strides here.
    if (d_sd_.reset(n * sizeof(StrideD))) throw std::runtime_error("Sm90GroupedPlan: alloc");
    cudaMemcpy(d_sd_.as<StrideD>(), host_sc_.data(), n * sizeof(StrideD), cudaMemcpyHostToDevice);

    hw_info_.device_id = 0;
    cudaGetDevice(&hw_info_.device_id);
    hw_info_.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info_.device_id);

    // The stream-K decomposition, computed here because this is the only place that holds both the
    // problem set and the SM count.  Two reasons it is not computed inside the scheduler:
    //   * the kernel calls the scheduler's workspace statics with a *default-constructed* problem
    //     shape (see `GroupStreamKScheduler::get_workspace_size`), so a decomposition recomputed
    //     there would size the reduction buffer for an empty problem set;
    //   * `initialize()` is re-run on every launch and the timing harness brackets the launch, so a
    //     16-group host walk per launch would land inside the measured region.
    // The scheduler keeps the arithmetic (`compute_decomposition`); this only decides when to ask.
    if constexpr (kSchedulesStreamK<Sched>) {
      decomp_ = StreamKScheduler::compute_decomposition(group_shapes(), hw_info_.sm_count,
                                                        swizzle_from_env(),
                                                        cutlass::gemm::kernel::detail::DecompositionMode::Heuristic);
    }

    // Hoist the kernel object, the workspace sizing and the one-time `initialize` (which encodes
    // the TMA descriptors) out of the hot path; per launch only `update` is needed.
    op_ = std::make_unique<Gemm>();
    {
      typename Gemm::Arguments args = make_arguments(d_c_.as<ElementC const *>(),
                                                     d_c_.as<ElementC *>());
      cutlass::Status st = op_->can_implement(args);
      if (st != cutlass::Status::kSuccess) throw std::runtime_error("Sm90GroupedPlan: can_implement");
      size_t const ws = Gemm::get_workspace_size(args);
      if (nullptr) fprintf(stderr, "[sm90] ctor workspace=%zu\n", ws);
      if (ws && d_workspace_.reset(ws)) throw std::runtime_error("Sm90GroupedPlan: workspace");
      st = op_->initialize(args, d_workspace_.as<void>());
      if (st != cutlass::Status::kSuccess) throw std::runtime_error("Sm90GroupedPlan: initialize");
    }
  }

  ~Sm90PlanImpl() override {
    if (host_c_pinned_) cudaFreeHost(host_c_pinned_);
    if (host_a_pinned_) cudaFreeHost(host_a_pinned_);
  }

  void bind_a(void const *a_base, cudaStream_t stream) override {
    auto const *pa = static_cast<ElementA_ const *>(a_base);
    if (pa == last_a_base_) return;  // steady state: same activation buffer as last call
    last_a_base_ = pa;
    for (int j = 0; j < num_rows_; ++j) {
      // Rows that carry their own `a_base` (the fused plan's wgrad rows) keep it; the rest follow
      // the base being bound (dgrad's dy, which changes every call).
      auto const *row_row = rows_[j].a_base ? static_cast<ElementA_ const *>(rows_[j].a_base) : pa;
      host_a_pinned_[j] = row_row + rows_[j].a_off;
    }
    cudaMemcpyAsync(d_a_.as<ElementA_ const *>(), host_a_pinned_,
                    num_rows_ * sizeof(ElementA_ const *), cudaMemcpyHostToDevice, stream);
    a_bound_ = true;
  }

  int launch(float *out, cudaStream_t stream) override {
    if (!a_bound_) return static_cast<int>(cutlass::Status::kErrorInternal);
    for (int j = 0; j < num_rows_; ++j) {
      // Row j's tile base: `j*ldc` for the forward (row j writes output row j), or the explicit
      // offset the backward plans carry (dgrad row j writes input row i(j)).
      // A row with an explicit `c_base` (the fused backward's rows, which land in different
      // allocations) ignores `out` *and* uses `c_off` as an absolute offset -- 0 is a real offset
      // there, not "use the default".  Rows without one keep the forward/dgrad behaviour.
      if (rows_[j].c_base) {
        host_c_pinned_[j] = static_cast<float *>(rows_[j].c_base) + rows_[j].c_off;
        continue;
      }
      host_c_pinned_[j] = out + (rows_[j].c_off >= 0
                                     ? rows_[j].c_off
                                     : static_cast<long long>(j) * c_col_stride_);
    }
    cudaMemcpyAsync(d_c_.as<ElementC *>(), host_c_pinned_, num_rows_ * sizeof(ElementC *),
                    cudaMemcpyHostToDevice, stream);
    // The epilogue takes C as `ElementC const**` and D as `ElementD**`; C and D alias the same
    // memory here, so the two device arrays hold the same pointer values.
    cudaMemcpyAsync(d_cc_.as<ElementC const *>(), host_c_pinned_, num_rows_ * sizeof(ElementC *),
                    cudaMemcpyHostToDevice, stream);

    typename Gemm::Arguments args = make_arguments(d_cc_.as<ElementC const *>(),
                                                   d_c_.as<ElementC *>());
    // NOTE: `update(args)` cannot be used here -- the grouped tile scheduler needs the workspace
    // and the adapter's 1-arg update() has no way to pass it (initialize() only wires the
    // workspace for GemmUniversalMode::kGemm).  So initialize() is re-run per launch; the Gemm
    // object itself stays hoisted, which avoids re-constructing it every call.
    cutlass::Status st = op_->initialize(args, d_workspace_.as<void>(), stream);
    if (nullptr) fprintf(stderr, "[sm90] initialize -> %s\n", cutlass::cutlassGetStatusString(st));
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st);
    st = op_->run(stream);
    if (nullptr) fprintf(stderr, "[sm90] run -> %s\n", cutlass::cutlassGetStatusString(st));
    return static_cast<int>(st);
  }

 private:
  using FusionArgs = decltype(std::declval<typename Gemm::Arguments>().epilogue.thread);

  /// The problem set in the form the kernel's `Arguments` carries it: `num_rows` problems, device
  /// pointers for the kernel and host pointers for anything the host has to reason about.  Built in
  /// one place because both the arguments the kernel runs on and the decomposition the scheduler is
  /// sized from derive from it.
  auto group_shapes() {
    return decltype(std::declval<typename Gemm::Arguments>().problem_shape){
        num_rows_, d_problem_.as<UnderlyingProblemShape>(), host_problem_.data()};
  }

  typename Gemm::Arguments make_arguments(ElementC const **ptr_c, ElementC **ptr_d) {
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        group_shapes(),
        mainloop_args(),
        {epilogue_args(), ptr_c, d_sc_.as<StrideC>(), ptr_d, d_sd_.as<StrideD>()},
        hw_info_};
    // L2-locality knobs (PersistentTileSchedulerSm90Group::Arguments).  Driven from the
    // environment so a raster/swizzle sweep needs no recompile.
    args.scheduler.raster_order = raster_from_env();
    args.scheduler.max_swizzle_size = swizzle_from_env();
    if constexpr (kSchedulesStreamK<Sched>) {
      args.scheduler.decomp = decomp_;
    }
    return args;
  }

  using MainloopArgs = decltype(std::declval<typename Gemm::Arguments>().mainloop);

  MainloopArgs mainloop_args() {
    if constexpr (kHasFold<Mode>) {
      return MainloopArgs{d_a_.as<ElementA_ const *>(), d_sa_.as<StrideA>(),
                          d_b_.as<ElementB_ const *>(), d_sb_.as<StrideB>(),
                          d_b_res_.as<ElementB_ const *>(), fold_k_from_env()};
    } else if constexpr (kHasBRes<Mode>) {
      return MainloopArgs{d_a_.as<ElementA_ const *>(), d_sa_.as<StrideA>(),
                          d_b_.as<ElementB_ const *>(), d_sb_.as<StrideB>(),
                          d_b_res_.as<ElementB_ const *>()};
    } else {
      return MainloopArgs{d_a_.as<ElementA_ const *>(), d_sa_.as<StrideA>(),
                          d_b_.as<ElementB_ const *>(), d_sb_.as<StrideB>()};
    }
  }

  FusionArgs epilogue_args() {
    FusionArgs a{};
    a.alpha = ElementAcc(1);
    a.beta = ElementAcc(0);
    return a;
  }

  int num_rows_ = 0;
  long long ldc_ = 0;
  long long c_col_stride_ = 0;
  /// Kept so `bind_a` can re-derive each row's pointer from a new base.  Cheap: one RowPlan per
  /// output row.
  std::vector<RowPlan> rows_;
  /// False until a base is supplied (activation-view mode); `launch` refuses to run before that.
  bool a_bound_ = false;
  /// Last base handed to `bind_a`, so a repeated call with the same activation buffer costs nothing.
  ElementA_ const *last_a_base_ = nullptr;

  /// Computed once, in the constructor, for the stream-K instantiation only; unused (and all-zero,
  /// i.e. the data-parallel decomposition) otherwise.
  eqlin::GroupStreamKDecomposition decomp_{};

  std::vector<UnderlyingProblemShape> host_problem_;
  std::vector<ElementA_ const *> host_a_;
  std::vector<ElementB_ const *> host_b_;
  std::vector<ElementB_ const *> host_b_res_;
  std::vector<StrideA> host_sa_;
  std::vector<StrideB> host_sb_;
  std::vector<StrideC> host_sc_;
  ElementC **host_c_pinned_ = nullptr;
  ElementA_ const **host_a_pinned_ = nullptr;

  DeviceBuffer d_problem_, d_a_, d_b_, d_b_res_, d_c_, d_cc_, d_sa_, d_sb_, d_sc_, d_sd_,
      d_workspace_;
  std::unique_ptr<Gemm> op_;
  cutlass::KernelHardwareInfo hw_info_{};
};

/// Fully specified instantiation: tile-N and cluster extent both pinned.
template <int TBN, int CM, Sm90Operands Mode, class Sched = cutlass::gemm::GroupScheduler>
std::unique_ptr<Sm90GroupedPlan::ImplBase> make_plan_impl(int B, int V, int num_rows, long long lda,
                                                          long long ldb, long long ldc,
                                                          std::vector<RowPlan> const &rows,
                                                          void const *a, void const *b,
                                                          void const *b_res) {
  return std::make_unique<Sm90PlanImpl<TBN, CM, Mode, Sched>>(B, V, num_rows, lda, ldb, ldc, rows,
                                                              a, b, b_res);
}

/// TB_N dispatch, for a fixed cluster extent.  Two policies:
///
///   0 -- the M3-era pair: V > 128 picks 256 to keep a single N-tile (two 128-tiles would read A
///        twice, which cost 23% on LARGE_P3), otherwise 128.
///   1 -- exact fit (M5-A): the smallest supported tile that still covers V, so the padding shrinks
///        from 12.5% (224 -> 256) and 33% (96 -> 128) to nothing.  The wasted fraction is pure MMA
///        work plus B smem/TMA traffic per K-tile, and nothing downstream changes because the tile
///        count stays at one N-tile per row either way.
///
/// `EQLIN_N_POLICY` selects between them at run time (0 = policy 0, anything else = policy 1),
/// which is what lets `tests/ab_pair.py` interleave the two inside one process -- two .so builds
/// cannot be loaded together (the op names would collide) and cross-process comparison carries this
/// box's 2.5-5.6% per-point session noise.  Note that `EQLIN_TB_N` is *not* a knob for this: it is
/// never expanded anywhere in the tree, and the M3 sweep that appeared to use it was really
/// varying the source.
///
/// Tile N must be a multiple of 8 (atom shape) and of 32 (the (8,32) SW128 epilogue smem atom), and
/// the K-tile must divide into it; 96 and 224 satisfy all of that.  224 additionally needs
/// CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED for the native 64x224x8 tf32 atom (see CMakeLists.txt).
template <int CM, Sm90Operands Mode, class Sched = cutlass::gemm::GroupScheduler>
std::unique_ptr<Sm90GroupedPlan::ImplBase> make_plan_tbn(int B, int V, int num_rows, long long lda,
                                                           long long ldb, long long ldc,
                                                           std::vector<RowPlan> const &rows,
                                                           void const *a, void const *b,
                                                           void const *b_res) {
  // The tf32x3 mainloops are the only ones the tile-N policy applies to; the M1/M2 baselines keep
  // the M3 pair so their numbers stay comparable with the earlier reports.
  constexpr bool kSplit = (Mode == Sm90Operands::kFp32InKernelSplit ||
                           Mode == Sm90Operands::kFp32RegSplit ||
                           Mode == Sm90Operands::kFp32RegSplitRn ||
                           Mode == Sm90Operands::kFp32RegSplitFold ||
                           Mode == Sm90Operands::kFp32RegSplitRnFold);
  char const *policy = nullptr;
  if constexpr (kSplit) {
    if (policy == nullptr || atoi(policy) != 0) {
      if (V <= 96) {
        return make_plan_impl<96, CM, Mode, Sched>(B, V, num_rows, lda, ldb, ldc, rows, a, b, b_res);
      }
      if (V <= 128) {
        return make_plan_impl<128, CM, Mode, Sched>(B, V, num_rows, lda, ldb, ldc, rows, a, b, b_res);
      }
      if (V <= 224) {
        return make_plan_impl<224, CM, Mode, Sched>(B, V, num_rows, lda, ldb, ldc, rows, a, b, b_res);
      }
      return make_plan_impl<256, CM, Mode, Sched>(B, V, num_rows, lda, ldb, ldc, rows, a, b, b_res);
    }
  }
  if (V > 128) {
    return make_plan_impl<256, CM, Mode, Sched>(B, V, num_rows, lda, ldb, ldc, rows, a, b, b_res);
  }
  return make_plan_impl<128, CM, Mode, Sched>(B, V, num_rows, lda, ldb, ldc, rows, a, b, b_res);
}

/// Cluster extent along M, selected at run time.  The alternative -- a `-DEQLIN_CLUSTER_M=2` build
/// -- cannot be paired against the extent-1 build inside one process, and the cross-process
/// alternative carries more session noise than the effect being measured (see ClusterShapeT).
///
/// Only the extents listed in the switch are instantiated; widening it is the one-line change a
/// throwaway build needs to profile an extent outside this set (M = 4 for the ncu comparison).
/// Note the extent is passed to the scheduler as-is: CUTLASS rounds the persistent grid up so
/// `grid.x % cluster.x == 0` always holds, and a grid that is not a multiple fails to launch
/// rather than being rounded up (M6-C, limit L3).
template <Sm90Operands Mode, class Sched>
std::unique_ptr<Sm90GroupedPlan::ImplBase> make_plan_cluster(int B, int V, int num_rows,
                                                             long long lda, long long ldb,
                                                             long long ldc,
                                                             std::vector<RowPlan> const &rows,
                                                             void const *a, void const *b,
                                                             void const *b_res) {
  char const *env = nullptr;
  int const cm = (env == nullptr) ? EQLIN_CLUSTER_M : atoi(env);
  switch (cm) {
    case 1:
      return make_plan_tbn<1, Mode, Sched>(B, V, num_rows, lda, ldb, ldc, rows, a, b, b_res);
    case 2:
      return make_plan_tbn<2, Mode, Sched>(B, V, num_rows, lda, ldb, ldc, rows, a, b, b_res);
    default:
      throw std::invalid_argument("Sm90GroupedPlan: EQLIN_CLUSTER_M must be 1 or 2");
  }
}

/// Whether this operand mode gets a stream-K instantiation at all.  Only the RS tf32x3 mainloop
/// (impl 5) does: the scheduler is a template argument, so every extra scheduler doubles the
/// `GemmUniversal` instantiations reachable for that mode, and the M1/M2 baselines must keep
/// bit-identical generated code so their 25-point numbers stay comparable with the earlier reports.
template <Sm90Operands Mode> constexpr bool kStreamKEligible = false;
template <> constexpr bool kStreamKEligible<Sm90Operands::kFp32RegSplit> = true;

/// Outermost dispatch: which tile scheduler.  `EQLIN_STREAMK` is read here (like `EQLIN_N_POLICY`
/// in `make_plan_tbn`) rather than being a build-time knob, because the two variants have to be
/// interleavable inside one process for `tests/ab_pair.py` to pair them at the box's ~2% in-process
/// resolution instead of the 6% a cross-process comparison carries.
///
/// Whether the stream-K scheduler actually splits anything is *its* decision, taken from the shape
/// by CUTLASS's own decomposition heuristic -- this knob only chooses whether that scheduler is
/// reachable, so there is no per-shape policy written here.
template <Sm90Operands Mode>
std::unique_ptr<Sm90GroupedPlan::ImplBase> make_plan(int B, int V, int num_rows, long long lda,
                                                      long long ldb, long long ldc,
                                                      std::vector<RowPlan> const &rows,
                                                      void const *a, void const *b,
                                                      void const *b_res) {
  if constexpr (kStreamKEligible<Mode>) {
    if (char const *sk = nullptr; sk != nullptr && atoi(sk) != 0) {
      return make_plan_cluster<Mode, StreamKGroupScheduler>(B, V, num_rows, lda, ldb, ldc, rows, a,
                                                            b, b_res);
    }
  }
  return make_plan_cluster<Mode, cutlass::gemm::GroupScheduler>(B, V, num_rows, lda, ldb, ldc, rows,
                                                                a, b, b_res);
}

}  // namespace

Sm90GroupedPlan::Sm90GroupedPlan(Sm90Operands operands, int B, int V, int num_rows,
                                 long long lda, long long ldb, long long ldc,
                                 std::vector<RowPlan> const &rows,
                                 void const *a_data, void const *b_data, void const *b_res_data) {
  switch (operands) {
    case Sm90Operands::kFp32:
      p_ = make_plan<Sm90Operands::kFp32>(B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data, nullptr);
      break;
    case Sm90Operands::kTf32KConcat:
      p_ = make_plan<Sm90Operands::kTf32KConcat>(B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data, nullptr);
      break;
    case Sm90Operands::kFp32InKernelSplit:
      p_ = make_plan<Sm90Operands::kFp32InKernelSplit>(B, V, num_rows, lda, ldb, ldc, rows, a_data,
                                                       b_data, b_res_data);
      break;
    case Sm90Operands::kFp32RegSplit:
      p_ = make_plan<Sm90Operands::kFp32RegSplit>(B, V, num_rows, lda, ldb, ldc, rows, a_data,
                                                  b_data, b_res_data);
      break;
    case Sm90Operands::kFp32RegSplitRn:
      p_ = make_plan<Sm90Operands::kFp32RegSplitRn>(B, V, num_rows, lda, ldb, ldc, rows, a_data,
                                                    b_data, b_res_data);
      break;
    case Sm90Operands::kFp32RegSplitFold:
      p_ = make_plan<Sm90Operands::kFp32RegSplitFold>(B, V, num_rows, lda, ldb, ldc, rows, a_data,
                                                      b_data, b_res_data);
      break;
    case Sm90Operands::kFp32RegSplitRnFold:
      p_ = make_plan<Sm90Operands::kFp32RegSplitRnFold>(B, V, num_rows, lda, ldb, ldc, rows, a_data,
                                                        b_data, b_res_data);
      break;
  }
}

Sm90GroupedPlan::~Sm90GroupedPlan() = default;

int Sm90GroupedPlan::launch(float *out, cudaStream_t stream) { return p_->launch(out, stream); }

void Sm90GroupedPlan::bind_a(void const *a_base, cudaStream_t stream) {
  p_->bind_a(a_base, stream);
}

}  // namespace eqlin

#else  // !CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED

namespace eqlin {

bool sm90_available() { return false; }

Sm90GroupedPlan::Sm90GroupedPlan(Sm90Operands, int, int, int, long long, long long, long long,
                                 std::vector<RowPlan> const &, void const *, void const *,
                                 void const *) {
  throw std::runtime_error("SM90 collective not available in this build");
}
Sm90GroupedPlan::~Sm90GroupedPlan() = default;
int Sm90GroupedPlan::launch(float *, cudaStream_t) { return 1; }
void Sm90GroupedPlan::bind_a(void const *, cudaStream_t) {}

}  // namespace eqlin

#endif
