#include "shape_plan.cuh"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/numeric_conversion.h"

#include "device_buffer.h"

namespace fasteq::eqlinear::tf32_shape {
namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxBlocks = 65535;

struct WeightSplit {
  cutlass::tfloat32_t big, small;
};

struct KernelGeometry {
  int batch, source_rows, target_rows, k, n, paths;
  uint64_t batch_tiles, column_tiles, tiles;
};

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("eqlinear tf32 shape: ") + operation + ": " +
                             cudaGetErrorString(status));
}

size_t elements(int a, int b, int c) {
  size_t count = static_cast<size_t>(a);
  const size_t limit = std::numeric_limits<size_t>::max() / sizeof(WeightSplit);
  for (int extent : {b, c}) {
    if (count > limit / static_cast<size_t>(extent))
      throw std::invalid_argument("eqlinear tf32 shape: operand size overflow");
    count *= static_cast<size_t>(extent);
  }
  return count;
}

// Static B[N,K] preparation only: transpose W for forward, keep W's order for
// dgrad. No activation-sized global workspace or padded logical operand.
template <bool Dgrad>
__global__ void split_weights(const float* __restrict__ w,
                              WeightSplit* __restrict__ packed, int u, int v,
                              uint64_t count) {
  const int k_extent = Dgrad ? v : u;
  const int n_extent = Dgrad ? u : v;
  cutlass::NumericConverterFastF32<> split;
  for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += uint64_t(gridDim.x) * blockDim.x) {
    const int k = index % k_extent;
    const uint64_t column = index / k_extent;
    const int n = column % n_extent;
    const uint64_t path = column / n_extent;
    const uint64_t source = Dgrad ? index : (path * u + k) * v + n;
    const auto pair = split(w[source]);
    packed[index] = WeightSplit{pair[0], pair[1]};
  }
}

// CuTe's TV layouts are the source of truth for the register fragments. A
// warp computes [16 batches, 8 channels]; masked scalars fill only registers.
using Mma = cute::MMA_Atom<cute::SM80_16x8x8_F32TF32TF32F32_TN>;

template <bool Dgrad>
__global__ void fused_shape(const float* __restrict__ input, float* __restrict__ output,
                            const WeightSplit* __restrict__ weights,
                            const ShapePath* __restrict__ paths, KernelGeometry g) {
  const int lane = threadIdx.x % 32;
  const uint64_t warp = uint64_t(blockIdx.x) * kWarps + threadIdx.x / 32;
  Mma mma;
  cutlass::NumericConverterFastF32<> split;
  for (uint64_t tile = warp; tile < g.tiles; tile += uint64_t(gridDim.x) * kWarps) {
    const int n0 = (tile % g.column_tiles) * 8;
    const uint64_t row_batch = tile / g.column_tiles;
    const int b0 = (row_batch % g.batch_tiles) * 16;
    const int row = row_batch / g.batch_tiles;
    auto total = cute::make_tensor<float>(cute::Shape<cute::_4>{});
    cute::clear(total);

    for (int q = 0; q < g.paths; ++q) {
      const ShapePath path = paths[q];
      const int destination = Dgrad ? path.input_row : path.output_row;
      if (row < destination || row - destination >= path.rows) continue;
      const int source_row = (Dgrad ? path.output_row : path.input_row) + row - destination;
      auto dot = cute::make_tensor<float>(cute::Shape<cute::_4>{});
      cute::clear(dot);
      // A whole warp follows each path/K loop, including K=1 readout dgrad.
      for (int64_t k0 = 0; k0 < g.k; k0 += 8) {
        auto a_big = cute::make_tensor<cutlass::tfloat32_t>(cute::Shape<cute::_4>{});
        auto a_small = cute::make_tensor_like(a_big);
        auto b_big = cute::make_tensor<cutlass::tfloat32_t>(cute::Shape<cute::_2>{});
        auto b_small = cute::make_tensor_like(b_big);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int coordinate = Mma::LayoutA_TV{}(lane, i);
          const int batch = b0 + coordinate % 16;
          const int64_t k = k0 + coordinate / 16;
          float value = 0.f;
          if (batch < g.batch && k < g.k)
            value = input[(uint64_t(batch) * g.source_rows + source_row) * g.k + k];
          const auto pair = split(value);
          a_big(i) = pair[0];
          a_small(i) = pair[1];
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int coordinate = Mma::LayoutB_TV{}(lane, i);
          const int n = n0 + coordinate % 8;
          const int64_t k = k0 + coordinate / 8;
          WeightSplit value{cutlass::tfloat32_t::bitcast(0), cutlass::tfloat32_t::bitcast(0)};
          if (n < g.n && k < g.k)
            value = weights[(uint64_t(path.weight_path) * g.n + n) * g.k + k];
          b_big(i) = value.big;
          b_small(i) = value.small;
        }
        // True TF32x3 TensorCore arithmetic: two cross terms, then the high
        // product. The small*small term is intentionally absent, as in legacy.
        mma.call(a_big, b_small, dot);
        mma.call(a_small, b_big, dot);
        mma.call(a_big, b_big, dot);
      }
      #pragma unroll
      for (int i = 0; i < 4; ++i)
        total(i) += static_cast<float>(static_cast<double>(dot(i)) * path.coefficient);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int coordinate = Mma::LayoutC_TV{}(lane, i);
      const int batch = b0 + coordinate % 16;
      const int n = n0 + coordinate / 16;
      if (batch < g.batch && n < g.n)
        output[(uint64_t(batch) * g.target_rows + row) * g.n + n] = total(i);
    }
  }
}

class Tf32ShapePlan final : public ShapePlan {
 public:
  Tf32ShapePlan(const EqLinearProblemC& p, const void* w, const double* cg,
                bool dgrad, cudaStream_t stream)
      : geometry_(p, cg), dgrad_(dgrad) {
    if (!w) throw std::invalid_argument("eqlinear tf32 shape: null weight");
    // Validate pointer arithmetic without touching input/output or the GPU.
    (void)elements(p.B, p.I, p.U);
    (void)elements(p.B, p.J, p.V);
    const size_t count = elements(p.P, p.U, p.V);
    kernel_ = KernelGeometry{p.B, dgrad ? p.J : p.I, dgrad ? p.I : p.J,
                              dgrad ? p.V : p.U, dgrad ? p.U : p.V, p.P,
                              (uint64_t(p.B) + 15) / 16,
                              (uint64_t(dgrad ? p.U : p.V) + 7) / 8, 0};
    kernel_.tiles = uint64_t(kernel_.target_rows) * kernel_.batch_tiles * kernel_.column_tiles;
    if (paths_.reset(geometry_.paths.size() * sizeof(ShapePath)) ||
        weights_.reset(count * sizeof(WeightSplit)))
      throw std::runtime_error("eqlinear tf32 shape: device allocation failed");
    check(cudaMemcpyAsync(paths_.as<ShapePath>(), geometry_.paths.data(),
                          geometry_.paths.size() * sizeof(ShapePath),
                          cudaMemcpyHostToDevice, stream), "copy paths");
    const int blocks = static_cast<int>(std::min<uint64_t>((count + kThreads - 1) / kThreads,
                                                          kMaxBlocks));
    if (dgrad)
      split_weights<true><<<blocks, kThreads, 0, stream>>>(
          static_cast<const float*>(w), weights_.as<WeightSplit>(), p.U, p.V, count);
    else
      split_weights<false><<<blocks, kThreads, 0, stream>>>(
          static_cast<const float*>(w), weights_.as<WeightSplit>(), p.U, p.V, count);
    check(cudaGetLastError(), "split weights launch");
  }

  int run(const void* input, void* output, cudaStream_t stream) override {
    if (!input || !output) return static_cast<int>(cudaErrorInvalidValue);
    const int blocks = static_cast<int>(std::min<uint64_t>(
        (kernel_.tiles + kWarps - 1) / kWarps, kMaxBlocks));
    if (dgrad_)
      fused_shape<true><<<blocks, kThreads, 0, stream>>>(
          static_cast<const float*>(input), static_cast<float*>(output),
          weights_.as<WeightSplit>(), paths_.as<ShapePath>(), kernel_);
    else
      fused_shape<false><<<blocks, kThreads, 0, stream>>>(
          static_cast<const float*>(input), static_cast<float*>(output),
          weights_.as<WeightSplit>(), paths_.as<ShapePath>(), kernel_);
    return static_cast<int>(cudaGetLastError());
  }

 private:
  ShapeGeometry geometry_;
  bool dgrad_;
  KernelGeometry kernel_{};
  DeviceBuffer paths_, weights_;
};

}  // namespace

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                  const double* cg, bool dgrad, cudaStream_t stream) {
  return std::make_unique<Tf32ShapePlan>(p, w, cg, dgrad, stream);
}

}  // namespace fasteq::eqlinear::tf32_shape
