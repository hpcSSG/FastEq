/***************************************************************************************************
 * fasteq eqlinear backends/fp64/grouped_gemm_f64.cu
 *
 * M1 backend: CUTLASS 2.x device::GemmGrouped with FP64 operands, FP64 accumulate and the Hopper
 * FP64 tensor-core instruction (mma.sync.m16n8k{4,8}.f64).
 *
 * Instantiation follows the tf32 line's M1 (grouped_gemm.cu) and CUTLASS's own SM90 FP64 unit test
 * test/unit/gemm/device/gemm_f64t_f64n_f64t_tensor_op_f64_sm90.cu.
 *
 * Two facts fixed by the compile spike (see FP64_M1_REPORT.md):
 *   * the grouped + arch::Sm90 + double instantiation compiles and runs on sm_90a;
 *   * kAlignment for A/B must be 1: the 2.x f64 predicated iterator with 2-element (16 B) access
 *     produces a non-constant cp.async size and does not compile.  FP64 still gets 8 B per thread
 *     per access from the 64-bit access-size path.
 * InstructionShape is 16x8x4 in the baseline; 16x8x8 (also a legal FP64 DMMA) is kept as an
 * alternative tile config.  16x8x16 needs a warp K of at least 32 and spills in the 2.x iterator.
 *
 * M4: the tile is a compile-time *parameter* (`Gemm2xConfig`), so several shapes live in one library
 * and the host picks one per problem shape.  `kBaseline` is expressed with the F64_* CMake knobs and
 * is the default, so impl 0 keeps the M1 behaviour byte for byte.
 **************************************************************************************************/
#include "grouped_gemm_f64.cuh"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "device_buffer.h"
#include "pointer_table.cuh"

#ifndef F64_TB_M
#define F64_TB_M 64
#endif
#ifndef F64_TB_N
#define F64_TB_N 64
#endif
#ifndef F64_TB_K
#define F64_TB_K 16
#endif
#ifndef F64_WARP_M
#define F64_WARP_M 32
#endif
#ifndef F64_WARP_N
#define F64_WARP_N 32
#endif
#ifndef F64_INS_K
#define F64_INS_K 4
#endif
#ifndef F64_STAGES
#define F64_STAGES 4
#endif
#ifndef F64_EPI_COUNT
#define F64_EPI_COUNT 2
#endif

namespace eqlin {
namespace {

using ElementA = double;
using ElementB = double;
using ElementC = double;
using ElementAcc = double;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // (K, N) column-major: B_prep rows are K-contiguous
using LayoutC = cutlass::layout::RowMajor;
using LongIndex = typename LayoutA::Stride::LongIndex;
using GemmCoord = cutlass::gemm::GemmCoord;

/// 8-byte elements: the 2.x f64 iterator cannot express a 16-byte (2-element) access (see header).
constexpr int kAlign = 1;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementC, F64_EPI_COUNT,
                                                                ElementAcc, ElementAcc>;

// ---------------------------------------------------------------------------------------------
// Tile configurations
//
// `kBaseline` is the M1 tile, expressed with the F64_* CMake knobs so the existing rebuild-based
// sweep interface still works.  The fixed entries exist for shape dispatch: 32-wide N strips are
// exact for V=96/224, and 128x32 at 3 stages still fits 3 CTAs/SM (12 warps), which is the occupancy
// ceiling for this register footprint (166 regs/thread).
// ---------------------------------------------------------------------------------------------
template <Gemm2xConfig C>
struct TileConfig;

template <Gemm2xConfig C>
struct GroupSchedule {
  static constexpr cutlass::gemm::kernel::GroupScheduleMode mode =
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly;
};

template <>
struct TileConfig<Gemm2xConfig::kBaseline> {
  static constexpr int tb_m = F64_TB_M, tb_n = F64_TB_N, tb_k = F64_TB_K;
  static constexpr int warp_m = F64_WARP_M, warp_n = F64_WARP_N;
  static constexpr int ins_k = F64_INS_K, stages = F64_STAGES;
};

template <>
struct TileConfig<Gemm2xConfig::kHostSchedule> : TileConfig<Gemm2xConfig::kBaseline> {};

template <>
struct GroupSchedule<Gemm2xConfig::kHostSchedule> {
  static constexpr cutlass::gemm::kernel::GroupScheduleMode mode =
      cutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute;
};

template <>
struct TileConfig<Gemm2xConfig::kM128N32S3> {
  static constexpr int tb_m = 128, tb_n = 32, tb_k = 16;
  static constexpr int warp_m = 32, warp_n = 32, ins_k = 4, stages = 3;
};

template <>
struct TileConfig<Gemm2xConfig::kM128N32S4> {
  static constexpr int tb_m = 128, tb_n = 32, tb_k = 16;
  static constexpr int warp_m = 32, warp_n = 32, ins_k = 4, stages = 4;
};

template <>
struct TileConfig<Gemm2xConfig::kM64N32S4> {
  static constexpr int tb_m = 64, tb_n = 32, tb_k = 16;
  static constexpr int warp_m = 32, warp_n = 32, ins_k = 4, stages = 4;
};

template <>
struct TileConfig<Gemm2xConfig::kM64N64S3> {
  static constexpr int tb_m = 64, tb_n = 64, tb_k = 16;
  static constexpr int warp_m = 32, warp_n = 32, ins_k = 4, stages = 3;
};

template <>
struct TileConfig<Gemm2xConfig::kIns8> {
  static constexpr int tb_m = 64, tb_n = 64, tb_k = 16;
  static constexpr int warp_m = 32, warp_n = 32, ins_k = 8, stages = 4;
};

template <>
struct TileConfig<Gemm2xConfig::kM32N128S3> {
  static constexpr int tb_m = 32, tb_n = 128, tb_k = 16;
  static constexpr int warp_m = 32, warp_n = 32, ins_k = 4, stages = 3;
};

template <>
struct TileConfig<Gemm2xConfig::kM32N128S4> {
  static constexpr int tb_m = 32, tb_n = 128, tb_k = 16;
  static constexpr int warp_m = 32, warp_n = 32, ins_k = 4, stages = 4;
};

template <>
struct TileConfig<Gemm2xConfig::kM32N32S3> {
  static constexpr int tb_m = 32, tb_n = 32, tb_k = 16;
  static constexpr int warp_m = 16, warp_n = 32, ins_k = 4, stages = 3;
};

template <>
struct TileConfig<Gemm2xConfig::kM128N128S3> {
  static constexpr int tb_m = 128, tb_n = 128, tb_k = 16;
  static constexpr int warp_m = 64, warp_n = 64, ins_k = 4, stages = 3;
};

template <>
struct TileConfig<Gemm2xConfig::kM128N64S3> {
  static constexpr int tb_m = 128, tb_n = 64, tb_k = 16;
  static constexpr int warp_m = 64, warp_n = 32, ins_k = 4, stages = 3;
};

template <>
struct TileConfig<Gemm2xConfig::kM64N128S3> {
  static constexpr int tb_m = 64, tb_n = 128, tb_k = 16;
  static constexpr int warp_m = 32, warp_n = 64, ins_k = 4, stages = 3;
};

template <>
struct TileConfig<Gemm2xConfig::kM32N64S3> {
  static constexpr int tb_m = 32, tb_n = 64, tb_k = 16;
  static constexpr int warp_m = 16, warp_n = 32, ins_k = 4, stages = 3;
};

template <>
struct TileConfig<Gemm2xConfig::kM128N64W32S3> {
  static constexpr int tb_m = 128, tb_n = 64, tb_k = 16;
  static constexpr int warp_m = 32, warp_n = 32, ins_k = 4, stages = 3;
};

/// The 2.x 64-bit warp iterators require warp M/N to be multiples of 16 and a 4-element K step
/// (mma_tensor_op_tile_iterator_sm80.h, the *_64bCrosswise atoms).
template <Gemm2xConfig C>
struct TileLegal {
  static constexpr bool ok = (TileConfig<C>::tb_m % TileConfig<C>::warp_m == 0) &&
                             (TileConfig<C>::tb_n % TileConfig<C>::warp_n == 0) &&
                             (TileConfig<C>::tb_k % TileConfig<C>::ins_k == 0) &&
                             (TileConfig<C>::warp_m % 16 == 0) &&
                             (TileConfig<C>::warp_n % 16 == 0) && (TileConfig<C>::tb_k % 4 == 0);
};

static_assert(TileLegal<Gemm2xConfig::kBaseline>::ok, "kBaseline tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM128N32S3>::ok, "kM128N32S3 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM128N32S4>::ok, "kM128N32S4 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM64N32S4>::ok, "kM64N32S4 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM64N64S3>::ok, "kM64N64S3 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kIns8>::ok, "kIns8 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM32N128S3>::ok, "kM32N128S3 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM32N128S4>::ok, "kM32N128S4 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM32N32S3>::ok, "kM32N32S3 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM128N128S3>::ok, "kM128N128S3 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM128N64S3>::ok, "kM128N64S3 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM64N128S3>::ok, "kM64N128S3 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM32N64S3>::ok, "kM32N64S3 tile is not legal");
static_assert(TileLegal<Gemm2xConfig::kM128N64W32S3>::ok, "kM128N64W32S3 tile is not legal");

template <Gemm2xConfig C>
using GroupedKernelT = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, kAlign,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, kAlign,
    ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<TileConfig<C>::tb_m, TileConfig<C>::tb_n, TileConfig<C>::tb_k>,
    cutlass::gemm::GemmShape<TileConfig<C>::warp_m, TileConfig<C>::warp_n, TileConfig<C>::tb_k>,
    cutlass::gemm::GemmShape<16, 8, TileConfig<C>::ins_k>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    /* Stages = */ TileConfig<C>::stages,
    GroupSchedule<C>::mode,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;

template <Gemm2xConfig C>
using GroupedGemmT = cutlass::gemm::device::GemmGrouped<GroupedKernelT<C>>;

}  // namespace

char const *gemm2x_config_name(Gemm2xConfig cfg) {
  switch (cfg) {
    case Gemm2xConfig::kBaseline: return "base";
    case Gemm2xConfig::kM128N32S3: return "m128n32s3";
    case Gemm2xConfig::kM128N32S4: return "m128n32s4";
    case Gemm2xConfig::kM64N32S4: return "m64n32s4";
    case Gemm2xConfig::kM64N64S3: return "m64n64s3";
    case Gemm2xConfig::kIns8: return "ins8";
    case Gemm2xConfig::kM32N128S3: return "m32n128s3";
    case Gemm2xConfig::kM32N128S4: return "m32n128s4";
    case Gemm2xConfig::kM32N32S3: return "m32n32s3";
    case Gemm2xConfig::kM128N128S3: return "m128n128s3";
    case Gemm2xConfig::kM128N64S3: return "m128n64s3";
    case Gemm2xConfig::kM64N128S3: return "m64n128s3";
    case Gemm2xConfig::kM32N64S3: return "m32n64s3";
    case Gemm2xConfig::kM128N64W32S3: return "m128n64w32s3";
    case Gemm2xConfig::kHostSchedule: return "host_schedule";
  }
  return "?";
}

struct GroupedGemmPlan::ImplBase {
  virtual ~ImplBase() = default;
  virtual int launch(double *out, cudaStream_t stream) = 0;
  virtual void bind_a(double const *a_base, cudaStream_t stream) = 0;
};

namespace {

/// One instantiation of the 2.x grouped kernel for a compile-time tile config.  The host-side
/// metadata (problem sizes, pointer/stride arrays, workspace) is identical for every config; only
/// the kernel type changes.
template <Gemm2xConfig C>
class PlanImplT : public GroupedGemmPlan::ImplBase {
 public:
  using Gemm = GroupedGemmT<C>;

  PlanImplT(int B, int V, int num_rows, long long lda, long long ldb, long long ldc,
            std::vector<RowPlan> const &rows, double const *a_data, double const *b_data)
      : num_rows_(num_rows), a_base_(a_data), a_bound_(a_data != nullptr),
        pointers_(pointer_rows(rows, V)) {
    if (num_rows <= 0) throw std::invalid_argument("GroupedGemmPlan: no rows");
    if (b_data == nullptr) throw std::invalid_argument("GroupedGemmPlan: missing B");

    host_problem_.resize(num_rows);
    host_a_.resize(num_rows);
    host_b_.resize(num_rows);
    host_lda_.resize(num_rows);
    host_ldb_.resize(num_rows);
    host_ldc_.resize(num_rows);

    for (int j = 0; j < num_rows; ++j) {
      host_problem_[j] = GemmCoord(rows[j].M ? rows[j].M : B,
                                   rows[j].N ? rows[j].N : V, rows[j].K);
      host_a_[j] = a_data ? const_cast<double *>(a_data + rows[j].a_off) : nullptr;
      host_b_[j] = const_cast<double *>(b_data + rows[j].b_off);
      host_lda_[j] = static_cast<LongIndex>(rows[j].lda ? rows[j].lda : lda);
      host_ldb_[j] = static_cast<LongIndex>(rows[j].ldb ? rows[j].ldb : ldb);
      host_ldc_[j] = static_cast<LongIndex>(rows[j].ldc ? rows[j].ldc : ldc);
    }

    size_t const n = static_cast<size_t>(num_rows);
    if (d_problem_.reset(n * sizeof(GemmCoord)) || d_a_.reset(n * sizeof(double *)) ||
        d_b_.reset(n * sizeof(double *)) || d_c_.reset(n * sizeof(double *)) ||
        d_lda_.reset(n * sizeof(LongIndex)) || d_ldb_.reset(n * sizeof(LongIndex)) ||
        d_ldc_.reset(n * sizeof(LongIndex))) {
      throw std::runtime_error("GroupedGemmPlan: device allocation failed");
    }

    cudaMemcpy(d_problem_.as<GemmCoord>(), host_problem_.data(), n * sizeof(GemmCoord),
               cudaMemcpyHostToDevice);
    if (a_data) {
      cudaMemcpy(d_a_.as<double *>(), host_a_.data(), n * sizeof(double *),
                 cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_b_.as<double *>(), host_b_.data(), n * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lda_.as<LongIndex>(), host_lda_.data(), n * sizeof(LongIndex),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldb_.as<LongIndex>(), host_ldb_.data(), n * sizeof(LongIndex),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldc_.as<LongIndex>(), host_ldc_.data(), n * sizeof(LongIndex),
               cudaMemcpyHostToDevice);

    threadblock_count_ = Gemm::sufficient(host_problem_.data(), num_rows);
    if (threadblock_count_ <= 0) {
      // `sufficient` uses the occupancy API with this instantiation's static shared storage; 0
      // means the tile does not fit in one SM's shared memory.  Fail at construction rather than
      // launching a zero-block grid.
      throw std::invalid_argument("GroupedGemmPlan: tile shape does not fit in shared memory");
    }

    workspace_bytes_ = Gemm::get_workspace_size(arguments_for(nullptr));
    if (workspace_bytes_ && d_workspace_.reset(workspace_bytes_)) {
      throw std::runtime_error("GroupedGemmPlan: workspace allocation failed");
    }

    op_ = std::make_unique<Gemm>();
    {
      typename Gemm::Arguments args = arguments_for(nullptr);
      cutlass::Status st = op_->can_implement(args);
      if (st != cutlass::Status::kSuccess) throw std::runtime_error("GroupedGemmPlan: can_implement");
      st = op_->initialize(args, d_workspace_.as<void>());
      if (st != cutlass::Status::kSuccess) throw std::runtime_error("GroupedGemmPlan: initialize");
    }
    if (nullptr) {
      fprintf(stderr, "[f64-plan] tile=%s rows=%d tb=%d ws=%zu a_bound=%d\n", gemm2x_config_name(C),
              num_rows_, threadblock_count_, workspace_bytes_, static_cast<int>(a_bound_));
    }
  }

  PlanImplT(PlanImplT const &) = delete;
  PlanImplT &operator=(PlanImplT const &) = delete;

  ~PlanImplT() override = default;

  void bind_a(double const *a_base, cudaStream_t stream) override {
    if (a_base == a_base_) return;  // steady state: the same activation buffer as last call
    if (pointers_.input(a_base, d_a_.as<double *>(), stream) != cudaSuccess)
      throw std::runtime_error("GroupedGemmPlan: input pointer update failed");
    a_base_ = a_base;
    a_bound_ = true;
  }

  int launch(double *out, cudaStream_t stream) override {
    if (!a_bound_) {
      if (nullptr) fprintf(stderr, "[f64-plan] launch before bind_a\n");
      return static_cast<int>(cutlass::Status::kErrorInternal);
    }

    if (pointers_.output(out, d_c_.as<double *>(), nullptr, stream) != cudaSuccess)
      return static_cast<int>(cutlass::Status::kErrorInternal);

    typename Gemm::Arguments args = arguments_for(d_c_.as<double *>());
    cutlass::Status st = op_->update(args, d_workspace_.as<void>(), stream);
    if (nullptr) {
      fprintf(stderr, "[f64-plan] update -> %s\n", cutlass::cutlassGetStatusString(st));
    }
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st);
    st = op_->run(stream);
    if (nullptr) {
      fprintf(stderr, "[f64-plan] run -> %s\n", cutlass::cutlassGetStatusString(st));
    }
    return static_cast<int>(st);
  }

 private:
  using Arguments = typename Gemm::Arguments;

  Arguments arguments_for(double **ptr_c) {
    double **c = ptr_c ? ptr_c : d_c_.as<double *>();
    return Arguments(d_problem_.as<GemmCoord>(), num_rows_, threadblock_count_,
                     {ElementAcc(1), ElementAcc(0)},
                     d_a_.as<double *>(), d_b_.as<double *>(), c, c,
                     d_lda_.as<LongIndex>(), d_ldb_.as<LongIndex>(),
                     d_ldc_.as<LongIndex>(), d_ldc_.as<LongIndex>(),
                     // host-side copy; the grouped Arguments ctor takes a non-const pointer even
                     // though it only reads it.
                     const_cast<GemmCoord *>(host_problem_.data()));
  }

  int num_rows_ = 0;
  int threadblock_count_ = 0;
  size_t workspace_bytes_ = 0;

  double const *a_base_ = nullptr;
  bool a_bound_ = false;

  std::vector<GemmCoord> host_problem_;
  std::vector<double *> host_a_, host_b_;
  PointerTable<double> pointers_;
  std::vector<LongIndex> host_lda_, host_ldb_, host_ldc_;

  DeviceBuffer d_problem_, d_a_, d_b_, d_c_;
  DeviceBuffer d_lda_, d_ldb_, d_ldc_, d_workspace_;
  std::unique_ptr<Gemm> op_;
};

}  // namespace

char const *status_string(int code) {
  return cutlass::cutlassGetStatusString(static_cast<cutlass::Status>(code));
}

GroupedGemmPlan::GroupedGemmPlan(int B, int V, int num_rows, long long lda, long long ldb,
                                 long long ldc, std::vector<RowPlan> const &rows,
                                 double const *a_data, double const *b_data, Gemm2xConfig cfg) {
  switch (cfg) {
    case Gemm2xConfig::kM128N32S3:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM128N32S3>>(B, V, num_rows, lda, ldb, ldc,
                                                                rows, a_data, b_data);
      break;
    case Gemm2xConfig::kM128N32S4:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM128N32S4>>(B, V, num_rows, lda, ldb, ldc,
                                                                rows, a_data, b_data);
      break;
    case Gemm2xConfig::kM64N32S4:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM64N32S4>>(B, V, num_rows, lda, ldb, ldc,
                                                               rows, a_data, b_data);
      break;
    case Gemm2xConfig::kM64N64S3:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM64N64S3>>(B, V, num_rows, lda, ldb, ldc,
                                                               rows, a_data, b_data);
      break;
    case Gemm2xConfig::kIns8:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kIns8>>(B, V, num_rows, lda, ldb, ldc, rows,
                                                           a_data, b_data);
      break;
    case Gemm2xConfig::kM32N128S3:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM32N128S3>>(B, V, num_rows, lda, ldb, ldc,
                                                                rows, a_data, b_data);
      break;
    case Gemm2xConfig::kM32N128S4:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM32N128S4>>(B, V, num_rows, lda, ldb, ldc,
                                                                rows, a_data, b_data);
      break;
    case Gemm2xConfig::kM32N32S3:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM32N32S3>>(B, V, num_rows, lda, ldb, ldc,
                                                               rows, a_data, b_data);
      break;
    case Gemm2xConfig::kM128N128S3:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM128N128S3>>(
          B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data);
      break;
    case Gemm2xConfig::kM128N64S3:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM128N64S3>>(
          B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data);
      break;
    case Gemm2xConfig::kM64N128S3:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM64N128S3>>(
          B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data);
      break;
    case Gemm2xConfig::kM32N64S3:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM32N64S3>>(
          B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data);
      break;
    case Gemm2xConfig::kM128N64W32S3:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kM128N64W32S3>>(
          B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data);
      break;
    case Gemm2xConfig::kHostSchedule:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kHostSchedule>>(
          B, V, num_rows, lda, ldb, ldc, rows, a_data, b_data);
      break;
    case Gemm2xConfig::kBaseline:
    default:
      p_ = std::make_unique<PlanImplT<Gemm2xConfig::kBaseline>>(B, V, num_rows, lda, ldb, ldc,
                                                               rows, a_data, b_data);
      break;
  }
}

GroupedGemmPlan::~GroupedGemmPlan() = default;

int GroupedGemmPlan::launch(double *out, cudaStream_t stream) { return p_->launch(out, stream); }

void GroupedGemmPlan::bind_a(double const *a_base, cudaStream_t stream) {
  p_->bind_a(a_base, stream);
}

}  // namespace eqlin
