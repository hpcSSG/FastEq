/***************************************************************************************************
 * fasteq eqlinear backends/tf32/grouped_gemm.cu
 **************************************************************************************************/
#include "grouped_gemm.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/layout/matrix.h"

#include "device_buffer.h"
#include "pointer_table.cuh"

namespace eqlin {
namespace {

using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // (K, N) column-major; ldb = row stride of B_prep
using LayoutC = cutlass::layout::RowMajor;
using LongIndex = typename LayoutA::Stride::LongIndex;
using GemmCoord = cutlass::gemm::GemmCoord;

/// 128-bit = 4 x fp32.  Required by every tf32x3 kernel.
constexpr int kAlign = 4;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementC, kAlign, ElementAcc, ElementAcc>;

template <typename MathOperator>
using GroupedKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, kAlign,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, kAlign,
    ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    /* Stages = */ 3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
    MathOperator>::GemmKernel;

template <typename MathOperator>
using GroupedGemmT = cutlass::gemm::device::GemmGrouped<GroupedKernel<MathOperator>>;

}  // namespace

struct GroupedGemmPlan::ImplBase {
  virtual ~ImplBase() = default;
  virtual int launch(float *out, cudaStream_t stream) = 0;
  virtual int bind_a(float const *a_base, cudaStream_t stream) = 0;
};

namespace {

/// Implementation parameterised on the grouped-GEMM instantiation.
template <typename GemmT>
class PlanImpl : public GroupedGemmPlan::ImplBase {
 public:
  PlanImpl(int B, int V, int num_rows, long long lda, long long ldb, long long ldc,
           std::vector<RowPlan> const &rows, float const *a_prep, float const *b_prep)
      : num_rows_(num_rows), a_bound_(a_prep != nullptr), pointers_(pointer_rows(rows, V)) {
    if (num_rows <= 0) throw std::invalid_argument("GroupedGemmPlan: no rows");

    host_problem_.resize(num_rows);
    host_a_.resize(num_rows);
    host_b_.resize(num_rows);
    host_lda_.resize(num_rows);
    host_ldb_.resize(num_rows);
    host_ldc_.resize(num_rows);

    for (int j = 0; j < num_rows; ++j) {
      host_problem_[j] = GemmCoord(B, V, rows[j].K);
      host_a_[j] = a_prep ? const_cast<float *>(a_prep + rows[j].a_off) : nullptr;
      host_b_[j] = const_cast<float *>(b_prep + rows[j].b_off);
      host_lda_[j] = static_cast<LongIndex>(rows[j].lda ? rows[j].lda : lda);
      host_ldb_[j] = static_cast<LongIndex>(ldb);
      host_ldc_[j] = static_cast<LongIndex>(ldc);
    }

    size_t const n = static_cast<size_t>(num_rows);
    if (d_problem_.reset(n * sizeof(GemmCoord)) || d_a_.reset(n * sizeof(float *)) ||
        d_b_.reset(n * sizeof(float *)) || d_c_.reset(n * sizeof(float *)) ||
        d_lda_.reset(n * sizeof(LongIndex)) || d_ldb_.reset(n * sizeof(LongIndex)) ||
        d_ldc_.reset(n * sizeof(LongIndex))) {
      throw std::runtime_error("GroupedGemmPlan: device allocation failed");
    }

    cudaMemcpy(d_problem_.as<GemmCoord>(), host_problem_.data(), n * sizeof(GemmCoord), cudaMemcpyHostToDevice);
    if (a_prep) cudaMemcpy(d_a_.as<float *>(), host_a_.data(), n * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_.as<float *>(), host_b_.data(), n * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lda_.as<LongIndex>(), host_lda_.data(), n * sizeof(LongIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldb_.as<LongIndex>(), host_ldb_.data(), n * sizeof(LongIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldc_.as<LongIndex>(), host_ldc_.data(), n * sizeof(LongIndex), cudaMemcpyHostToDevice);

    threadblock_count_ = GemmT::sufficient(host_problem_.data(), num_rows);

    workspace_bytes_ = GemmT::get_workspace_size(arguments_for(nullptr));
    if (workspace_bytes_ && d_workspace_.reset(workspace_bytes_)) {
      throw std::runtime_error("GroupedGemmPlan: workspace allocation failed");
    }

    // Build the kernel object and run the one-time initialization (which includes a
    // cudaFuncSetAttribute for the >48KB dynamic smem).  Per-launch work then only needs
    // update(), which is much cheaper than initialize().
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

  int bind_a(float const *a_base, cudaStream_t stream) override {
    if (a_base == last_a_base_ && a_bound_) return 0;
    if (pointers_.input(a_base, d_a_.as<float *>(), stream) != cudaSuccess)
      return static_cast<int>(cutlass::Status::kErrorInternal);
    last_a_base_ = a_base;
    a_bound_ = true;
    return 0;
  }

  int launch(float *out, cudaStream_t stream) override {
    if (!a_bound_) return static_cast<int>(cutlass::Status::kErrorInternal);
    if (pointers_.output(out, d_c_.as<float *>(), nullptr, stream) != cudaSuccess)
      return static_cast<int>(cutlass::Status::kErrorInternal);

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
  bool a_bound_ = false;
  float const *last_a_base_ = nullptr;
  int threadblock_count_ = 0;
  size_t workspace_bytes_ = 0;

  std::vector<GemmCoord> host_problem_;
  std::vector<float *> host_a_, host_b_;
  std::vector<LongIndex> host_lda_, host_ldb_, host_ldc_;

  PointerTable<float> pointers_;
  DeviceBuffer d_problem_, d_a_, d_b_, d_c_;
  DeviceBuffer d_lda_, d_ldb_, d_ldc_, d_workspace_;
  std::unique_ptr<GemmT> op_;
};

}  // namespace

char const *status_string(int code) {
  return cutlass::cutlassGetStatusString(static_cast<cutlass::Status>(code));
}

GroupedGemmPlan::GroupedGemmPlan(Impl impl, int B, int V, int num_rows,
                                 long long lda, long long ldb, long long ldc,
                                 std::vector<RowPlan> const &rows,
                                 float const *a_prep, float const *b_prep) {
  switch (impl) {
    case Impl::k3xTF32:
      p_ = std::make_unique<PlanImpl<GroupedGemmT<cutlass::arch::OpMultiplyAddFastF32>>>(
          B, V, num_rows, lda, ldb, ldc, rows, a_prep, b_prep);
      break;
    case Impl::k1xTF32:
      p_ = std::make_unique<PlanImpl<GroupedGemmT<cutlass::arch::OpMultiplyAdd>>>(
          B, V, num_rows, lda, ldb, ldc, rows, a_prep, b_prep);
      break;
    default:
      throw std::invalid_argument("GroupedGemmPlan: unknown Impl");
  }
}

GroupedGemmPlan::~GroupedGemmPlan() = default;

int GroupedGemmPlan::bind_a(float const *a_base, cudaStream_t stream) {
  return p_->bind_a(a_base, stream);
}

int GroupedGemmPlan::launch(float *out, cudaStream_t stream) {
  return p_->launch(out, stream);
}

}  // namespace eqlin
