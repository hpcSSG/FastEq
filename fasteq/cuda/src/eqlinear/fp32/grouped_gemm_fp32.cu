/***************************************************************************************************
 * fasteq eqlinear fp32/csrc/grouped_gemm_fp32.cu
 *
 * Instantiation follows test/unit/gemm/device/gemm_grouped_sm80.cu (the grouped SIMT reference in
 * the pinned CUTLASS v4.8.0 tree) and cutlass_impl/csrc/grouped_gemm.cu (the M1 host-side plumbing).
 *
 * --- Why these template arguments ---
 *
 * * `arch::OpClassSimt` + `arch::Sm80`: the SIMT `DefaultGemm` specialization is keyed on the Sm80
 *   arch tag.  `arch::Sm90` selects a different specialization that pins Stages to 2 and builds an
 *   `arch::Sm50` MmaCore, i.e. no cp.async.
 * * `Stages = 3` (>= 3): the SIMT `DefaultMma` specialization with a generic stage count selects
 *   `MmaMultistage` (cp.async).  Stages == 2 selects `MmaPipelined`, which static-asserts exactly 2.
 * * `GemmShape<1,1,1>` instruction shape: a SIMT thread computes with scalar FMA.
 * * Alignment 1 on both operands: SIMT thread maps are `kElementsPerAccess == 1`, and
 *   `PredicatedTileAccessIterator` static-asserts against a wider access type.
 * * `LinearCombination<float, 1, ...>`: the SIMT epilogue must operate on scalars (`kCount == 1`).
 * * K-tile is fixed at 8: the SIMT multistage core's smem-padding assert
 *   (`default_mma_core_sm80.h:2377`) rejects TB_K = 32/64/96, and TB_K = 16 fails at launch.
 *
 * --- Why a tile policy ---
 *
 * M1 measured the shared-memory data path at 86% while every other L1 sub-unit was under 22%: the
 * binding constraint is shared-memory bytes per FMA, which for a `tm x tn` thread tile is
 * `(tm+tn)/(tm*tn)`.  A bigger thread tile is therefore the only lever.  CUTLASS derives the thread
 * tile from the warp shape and its 4x8 / 8x4 lane split:
 *
 *   warp 32x64 -> 8x8  (64 acc, 0.250 elem/FMA)   -- M1
 *   warp 64x64 -> 16x8 (128 acc, 0.1875)          -- upstream-tested (simt_sgemm_nt_sm80.cu:235)
 *   warp 64x96 -> 16x12 (192 acc, 0.1458)         -- legal but likely to spill
 *
 * The policy is a *runtime* choice (env `EQLIN_FP32_TILE_POLICY`) so two shapes can be A/B'd in one
 * process at the box's ~2% resolution instead of a cross-process ~6%.
 **************************************************************************************************/
#include "grouped_gemm_fp32.cuh"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/layout/matrix.h"

#include "device_buffer.h"
#include "simt_gemm.cuh"

namespace eqlin_fp32 {
namespace {

using Element = float;
using LayoutA = cutlass::layout::RowMajor;     // (M, K) row-major = K-major
using LayoutB = cutlass::layout::ColumnMajor;  // (N, K) row-major = K-major
using LayoutC = cutlass::layout::RowMajor;     // (M, N) row-major
using ElementAcc = float;
using LongIndex = typename LayoutA::Stride::LongIndex;
using GemmCoord = cutlass::gemm::GemmCoord;

/// SIMT thread maps consume one element per access; a wider alignment does not compile.
constexpr int kAlign = 1;

/// Not a knob: see the K-tile note in the file header.
constexpr int kTbK = 8;
constexpr int kWarpK = 8;
constexpr int kStages = 3;

template <int TBM, int TBN, int WMK, int WNK>
struct SimtKernelT {
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<Element, 1, ElementAcc, ElementAcc>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Element, LayoutA, cutlass::ComplexTransform::kNone, kAlign,
      Element, LayoutB, cutlass::ComplexTransform::kNone, kAlign,
      Element, LayoutC, ElementAcc,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TBM, TBN, kTbK>,
      cutlass::gemm::GemmShape<WMK, WNK, kWarpK>,
      cutlass::gemm::GemmShape<1, 1, 1>,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      /*Stages=*/kStages,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
    >::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmGrouped<Kernel>;
};

}  // namespace

namespace {

/// The grouped GEMM's C pointer array moves with the caller's `out`, so it is rewritten every
/// launch.  It used to be staged through a pinned host buffer and copied asynchronously -- which
/// races with the next launch (the driver does not read the staging buffer at call time, so
/// back-to-back launches could both see the *later* pointer).  Writing it on the device from kernel
/// arguments removes the staging buffer and the race.
__global__ void fill_c_ptrs_kernel(float *out, long long col_stride, float **dst, int n) {
  int const j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) dst[j] = out + static_cast<long long>(j) * col_stride;
}

/// Zero-copy A: re-point every problem's A at `x + offs[j]`.  Written on the device for the same
/// reason as the C pointers -- a pinned host staging buffer races with the next launch (M5 §5.3).
__global__ void fill_a_ptrs_kernel(float const *base, int const *offs, float **dst, int n) {
  int const j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) dst[j] = const_cast<float *>(base) + offs[j];
}

/// Host-side metadata + the one-time kernel initialization.  Mirrors the tf32-M1 plan: the problem
/// table, pointer arrays and strides are built once; a launch only rewrites the C pointer array
/// (which moves with the caller's `out`) and re-runs `update()`.
template <class GemmT>
class PlanImpl : public GroupedGemmPlan::ImplBase {
 public:
  PlanImpl(int M, int N, int num_rows, long long lda, long long ldb, long long ldc,
           std::vector<RowPlan> const &rows, float const *a_data, float const *b_data, bool a_view)
      : num_rows_(num_rows), c_col_stride_(N), a_view_(a_view), a_bound_(!a_view) {
    if (num_rows <= 0) throw std::invalid_argument("GroupedGemmPlan: no rows");

    host_problem_.resize(num_rows);
    host_a_.resize(num_rows);
    host_b_.resize(num_rows);
    host_a_off_.resize(num_rows);
    host_lda_.assign(num_rows, static_cast<LongIndex>(lda));
    host_ldb_.assign(num_rows, static_cast<LongIndex>(ldb));
    host_ldc_.assign(num_rows, static_cast<LongIndex>(ldc));

    for (int j = 0; j < num_rows; ++j) {
      host_problem_[j] = GemmCoord(M, N, rows[j].K);
      host_a_[j] = const_cast<float *>(a_data + rows[j].a_off);
      host_a_off_[j] = static_cast<int>(rows[j].a_off);
      host_b_[j] = const_cast<float *>(b_data + rows[j].b_off);
    }

    size_t const n = static_cast<size_t>(num_rows);
    if (d_problem_.reset(n * sizeof(GemmCoord)) || d_a_.reset(n * sizeof(float *)) ||
        d_b_.reset(n * sizeof(float *)) || d_c_.reset(n * sizeof(float *)) ||
        d_a_off_.reset(n * sizeof(int)) || d_lda_.reset(n * sizeof(LongIndex)) ||
        d_ldb_.reset(n * sizeof(LongIndex)) ||
        d_ldc_.reset(n * sizeof(LongIndex))) {
      throw std::runtime_error("GroupedGemmPlan: device allocation failed");
    }

    cudaMemcpy(d_problem_.as<GemmCoord>(), host_problem_.data(), n * sizeof(GemmCoord),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_.as<float *>(), host_a_.data(), n * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_off_.as<int>(), host_a_off_.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_.as<float *>(), host_b_.data(), n * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lda_.as<LongIndex>(), host_lda_.data(), n * sizeof(LongIndex),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldb_.as<LongIndex>(), host_ldb_.data(), n * sizeof(LongIndex),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldc_.as<LongIndex>(), host_ldc_.data(), n * sizeof(LongIndex),
               cudaMemcpyHostToDevice);

    threadblock_count_ = GemmT::sufficient(host_problem_.data(), num_rows);

    workspace_bytes_ = GemmT::get_workspace_size(arguments_for(nullptr));
    if (workspace_bytes_ && d_workspace_.reset(workspace_bytes_)) {
      throw std::runtime_error("GroupedGemmPlan: workspace allocation failed");
    }

    // Build the kernel object and run the one-time initialization (which includes the
    // cudaFuncSetAttribute for the >48KB dynamic smem).  Per-launch work then only needs update().
    op_ = std::make_unique<GemmT>();
    {
      typename GemmT::Arguments args = arguments_for(nullptr);
      cutlass::Status st = op_->can_implement(args);
      if (st != cutlass::Status::kSuccess) throw std::runtime_error("GroupedGemmPlan: can_implement");
      st = op_->initialize(args, d_workspace_.as<void>());
      if (st != cutlass::Status::kSuccess) throw std::runtime_error("GroupedGemmPlan: initialize");
    }
  }

  PlanImpl(PlanImpl const &) = delete;
  PlanImpl &operator=(PlanImpl const &) = delete;

  ~PlanImpl() override = default;

  /// Zero-copy A: CUTLASS's grouped Arguments take a *device pointer array*, so unlike the SIMT plan
  /// this needs one tiny kernel per rebind -- stream-ordered, which is exactly what the pinned-host
  /// version was missing (M5 §5.3).
  int bind_a(float const *x, cudaStream_t stream) override {
    if (a_bound_ && x == a_base_) return 0;
    fill_a_ptrs_kernel<<<(num_rows_ + 127) / 128, 128, 0, stream>>>(
        x, d_a_off_.as<int>(), d_a_.as<float *>(), num_rows_);
    a_base_ = x;
    a_bound_ = true;
    bind_stream_ = stream;
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
  }

  int launch(float *out, cudaStream_t stream) override {
    if (!a_bound_) {
      fprintf(stderr, "[eqlin] launch: A not bound -- call prepare_act(x) first\n");
      return static_cast<int>(cutlass::Status::kErrorInternal);
    }
    if (bind_stream_ != nullptr && stream != bind_stream_) {
      fprintf(stderr, "[eqlin] launch: stream differs from the one A was bound on\n");
      return static_cast<int>(cutlass::Status::kErrorInternal);
    }
    // Row j of the output is the column-strip out[:, j, :]: it starts at out + j*V and its rows are
    // `ldc` apart.  (j*V != j*ldc unless J == 1 -- getting this wrong silently works for a single
    // output row and corrupts every multi-row case.)
    fill_c_ptrs_kernel<<<(num_rows_ + 127) / 128, 128, 0, stream>>>(
        out, c_col_stride_, d_c_.as<float *>(), num_rows_);

    typename GemmT::Arguments args = arguments_for(d_c_.as<float *>());
    cutlass::Status st = op_->update(args, d_workspace_.as<void>(), stream);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st);
    return static_cast<int>(op_->run(stream));
  }

 private:
  using Arguments = typename GemmT::Arguments;

  Arguments arguments_for(float **ptr_c) {
    float **c = ptr_c ? ptr_c : d_c_.as<float *>();
    return Arguments(d_problem_.as<GemmCoord>(), num_rows_, threadblock_count_,
                     {ElementAcc(1), ElementAcc(0)},
                     d_a_.as<float *>(), d_b_.as<float *>(), c, c,
                     d_lda_.as<LongIndex>(), d_ldb_.as<LongIndex>(),
                     d_ldc_.as<LongIndex>(), d_ldc_.as<LongIndex>(),
                     // host-side copy; the grouped Arguments ctor takes a non-const pointer even
                     // though it only reads it (GemmGrouped::sufficient / get_workspace_size).
                     const_cast<GemmCoord *>(host_problem_.data()));
  }

  int num_rows_ = 0;
  long long c_col_stride_ = 0;
  int threadblock_count_ = 0;
  size_t workspace_bytes_ = 0;

  std::vector<GemmCoord> host_problem_;
  std::vector<float *> host_a_, host_b_;
  std::vector<LongIndex> host_lda_, host_ldb_, host_ldc_;
  std::vector<int> host_a_off_;  ///< per problem: A's element offset inside the A tensor
  bool a_view_ = false;
  bool a_bound_ = true;
  float const *a_base_ = nullptr;
  cudaStream_t bind_stream_ = nullptr;
  DeviceBuffer d_a_off_;

  DeviceBuffer d_problem_, d_a_, d_b_, d_c_, d_lda_, d_ldb_, d_ldc_, d_workspace_;
  std::unique_ptr<GemmT> op_;
};

template <int TBM, int TBN, int WMK, int WNK>
std::unique_ptr<GroupedGemmPlan::ImplBase> make_cfg(int M, int N, int num_rows, long long lda,
                                                    long long ldb, long long ldc,
                                                    std::vector<RowPlan> const &rows,
                                                    float const *a_data, float const *b_data,
                                                    bool a_view) {
  using Gemm = typename SimtKernelT<TBM, TBN, WMK, WNK>::Gemm;
  return std::make_unique<PlanImpl<Gemm>>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data,
                                          a_view);
}

/// Tile policy.  The default is read once per plan from `EQLIN_FP32_TILE_POLICY`:
///
///   0  M1's baseline: 128x128x8 / warp 32x64 (thread tile 8x8)
///   1  16x8 thread tile: warp 64x64; V > 128 pads to TB_N = 256 (CUTLASS's warp N must be a
///      multiple of 32, and 224 has no exact-fit warp tile in the stock lane arrangement)
///   2  24x8 / 16x12 thread tiles: TB_M = 192 / warp 96x64 for V > 96, exact-fit TB_N = 96 with
///      warp 64x96 for V <= 96.  Both carry 192 accumulators -- watch ptxas for spills.
///   3  exact-fit N, low register pressure: TB_N = 96 / warp 32x96 (8x12) for V <= 96, else the
///      same shapes as policy 1
///
/// Every policy falls back to a 64x128 / warp 64x64 (16x8) tile when B <= 1024: at B=512 the
/// baseline grid is 64 CTAs on 132 SMs (0.24 wave), and doubling the M-tile count is worth much
/// more there than the per-tile efficiency of the large tiles.
std::unique_ptr<GroupedGemmPlan::ImplBase> make_impl(int policy, int M, int N, int num_rows,
                                                     long long lda, long long ldb, long long ldc,
                                                     std::vector<RowPlan> const &rows,
                                                     float const *a_data, float const *b_data,
                                                     bool a_view) {
  bool const small = (M <= 1024);
  switch (policy) {
    case 0:
      return make_cfg<128, 128, 32, 64>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
    case 1:
      if (small) return make_cfg<64, 128, 64, 64>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
      if (N <= 128) return make_cfg<128, 128, 64, 64>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
      return make_cfg<128, 256, 64, 64>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
    case 2:
      if (small) return make_cfg<64, 128, 64, 64>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
      if (N <= 96) return make_cfg<128, 96, 64, 96>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
      return make_cfg<192, 128, 96, 64>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
    case 3:
      if (small) return make_cfg<64, 128, 64, 64>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
      if (N <= 96) return make_cfg<128, 96, 32, 96>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
      if (N <= 128) return make_cfg<128, 128, 64, 64>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
      return make_cfg<128, 256, 64, 64>(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
    default:
      throw std::invalid_argument("GroupedGemmPlan: EQLIN_FP32_TILE_POLICY must be 0..3");
  }
}

int policy_from_env() {
  char const *p = nullptr;
  return p == nullptr ? 0 : atoi(p);
}

}  // namespace

char const *status_string(int code) {
  return cutlass::cutlassGetStatusString(static_cast<cutlass::Status>(code));
}

GroupedGemmPlan::GroupedGemmPlan(int M, int N, int num_rows, long long lda, long long ldb,
                                 long long ldc, std::vector<RowPlan> const &rows,
                                 float const *a_data, float const *b_data, int mode, bool a_view) {
  if (mode == 1 || (mode == 2 && (N <= 96 || N > 128))) {
    p_ = make_simt_custom_plan(M, N, num_rows, lda, ldb, ldc, rows, a_data, b_data, a_view);
  } else {
    // mode 2's stock half is policy 2 (24x8 / exact-fit N), not the environment's default.
    p_ = make_impl(mode == 2 ? 2 : policy_from_env(), M, N, num_rows, lda, ldb, ldc, rows, a_data,
                   b_data, a_view);
  }
}

GroupedGemmPlan::~GroupedGemmPlan() = default;

int GroupedGemmPlan::launch(float *out, cudaStream_t stream) { return p_->launch(out, stream); }

int GroupedGemmPlan::bind_a(float const *x, cudaStream_t stream) { return p_->bind_a(x, stream); }

}  // namespace eqlin_fp32
