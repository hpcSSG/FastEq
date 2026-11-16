#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "../../common/single_path.h"
#include "device_buffer.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace fasteq::eqlinear::tf32_stock {
namespace {

void stock_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("eqlinear tf32 stock: ") + operation + ": " +
                             cudaGetErrorString(status));
}

template <bool Dgrad>
__global__ void snapshot_weight(const float* input, float* output, int u, int v,
                                 uint64_t count) {
  const int k_extent = Dgrad ? v : u;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t n = i / k_extent;
    const int k = i % k_extent;
    output[i] = input[Dgrad ? i : uint64_t(k) * v + n];
  }
}

// Stock FastF32 performs the true three-term TensorCore dot. OnlyAlphaScaling
// suppresses destination reads; double compute preserves the post-dot CG rule.
template <int Alignment>
using StockGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
        float, Alignment, float, double, cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3, Alignment, Alignment, false, cutlass::arch::OpMultiplyAddFastF32>;

template <int Alignment>
class StockPlan final : public ShapePlan {
  using Gemm = StockGemm<Alignment>;
  using Kernel = typename Gemm::GemmKernel;
  using ScalarGemm = StockGemm<1>;
  using ScalarKernel = typename ScalarGemm::GemmKernel;
 public:
  StockPlan(const SinglePathGeometry& geometry, const void* weight, bool dgrad,
             cudaStream_t stream) {
    const auto& g = geometry.geometry;
    const auto& path = geometry.path;
    if (g.I != g.J || path.rows != g.I || path.input_row || path.output_row || !weight)
      throw std::invalid_argument("eqlinear tf32 stock: expected full contiguous single path");
    const int64_t m = int64_t(g.B) * g.I;
    const int k = dgrad ? g.V : g.U, n = dgrad ? g.U : g.V;
    if (m > std::numeric_limits<int>::max())
      throw std::invalid_argument("eqlinear tf32 stock: M exceeds GemmCoord");
    const uint64_t count = uint64_t(k) * n;
    if (count > std::numeric_limits<size_t>::max() / sizeof(float))
      throw std::invalid_argument("eqlinear tf32 stock: weight size overflow");
    if (weight_.reset(count * sizeof(float)))
      throw std::runtime_error("eqlinear tf32 stock: weight allocation failed");
    cutlass::gemm::GemmCoord problem(static_cast<int>(m), n, k);
    typename Gemm::ThreadblockSwizzle swizzle;
    const auto tiled = swizzle.get_tiled_shape(problem, {64, 64, 16}, 1);
    grid_ = swizzle.get_grid_shape(tiled);
    params_ = typename Kernel::Params(problem, tiled,
        {nullptr, cutlass::layout::RowMajor(k)},
        {weight_.template as<float>(), cutlass::layout::ColumnMajor(k)},
        {nullptr, cutlass::layout::RowMajor(n)},
        {nullptr, cutlass::layout::RowMajor(n)},
        typename Gemm::EpilogueOutputOp::Params(path.coefficient));
    if constexpr (Alignment > 1)
      scalar_params_ = typename ScalarKernel::Params(problem, tiled,
          {nullptr, cutlass::layout::RowMajor(k)},
          {weight_.template as<float>(), cutlass::layout::ColumnMajor(k)},
          {nullptr, cutlass::layout::RowMajor(n)},
          {nullptr, cutlass::layout::RowMajor(n)},
          typename ScalarGemm::EpilogueOutputOp::Params(path.coefficient));
    // This tile/stage budget needs no opt-in shared-memory attribute and has
    // no split-K workspace, pointer tables, initialize or reset on the run path.
    static_assert(sizeof(typename Kernel::SharedStorage) < 48 * 1024);
    static_assert(sizeof(typename ScalarKernel::SharedStorage) < 48 * 1024);
    const int blocks = std::min<uint64_t>((count + 255) / 256, 65535);
    if (dgrad)
      snapshot_weight<true><<<blocks, 256, 0, stream>>>(
          static_cast<const float*>(weight), weight_.template as<float>(), g.U, g.V, count);
    else
      snapshot_weight<false><<<blocks, 256, 0, stream>>>(
          static_cast<const float*>(weight), weight_.template as<float>(), g.U, g.V, count);
    stock_check(cudaGetLastError(), "snapshot weight");
  }

  int run(const void* input, void* output, cudaStream_t stream) override {
    if (!input || !output) return static_cast<int>(cudaErrorInvalidValue);
    if constexpr (Alignment > 1) {
      if ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) & 15) {
        // Reuse the same snapshot for arbitrary base alignment. Neither a
        // second weight pack nor activation padding is needed.
        auto params = scalar_params_;
        params.ref_A.reset(const_cast<float*>(static_cast<const float*>(input)));
        params.ref_D.reset(static_cast<float*>(output));
        cutlass::Kernel<ScalarKernel><<<grid_, ScalarKernel::kThreadCount,
            sizeof(typename ScalarKernel::SharedStorage), stream>>>(params);
        return static_cast<int>(cudaGetLastError());
      }
    }
    auto params = params_;
    params.ref_A.reset(const_cast<float*>(static_cast<const float*>(input)));
    params.ref_D.reset(static_cast<float*>(output));
    cutlass::Kernel<Kernel><<<grid_, Kernel::kThreadCount,
        sizeof(typename Kernel::SharedStorage), stream>>>(params);
    return static_cast<int>(cudaGetLastError());
  }

 private:
  DeviceBuffer weight_;
  typename Kernel::Params params_;
  typename ScalarKernel::Params scalar_params_;
  dim3 grid_;
};

}  // namespace

// Only measured wins over both the frozen native API and the manual route.
// Keep all other shapes on their existing plan until measured on the same lease.
inline bool selected(const SinglePathGeometry& g) {
  const auto& p = g.geometry;
  if (p.I != p.J || g.path.rows != p.I || g.path.input_row || g.path.output_row ||
      p.U != p.V)
    return false;
  const int64_t m = int64_t(p.B) * p.I;
  if (m != 108 && m != 396 && m != 1584 && m != 6312 && m != 25116) return false;
  return p.U == 96 || p.U == 128 || (p.U == 224 && m != 25116);
}

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                const double* cg, bool dgrad, cudaStream_t stream,
                                bool scalar_access = false) {
  SinglePathGeometry geometry(p, cg);
  if (!scalar_access && p.U % 4 == 0 && p.V % 4 == 0)
    return std::make_unique<StockPlan<4>>(geometry, w, dgrad, stream);
  return std::make_unique<StockPlan<1>>(geometry, w, dgrad, stream);
}

}  // namespace fasteq::eqlinear::tf32_stock
