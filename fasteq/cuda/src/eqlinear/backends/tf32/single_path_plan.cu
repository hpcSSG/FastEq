#include "single_path_plan.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/numeric_conversion.h"
#include "device_buffer.h"
#include "stock_gemm_plan.cuh"

namespace fasteq::eqlinear::tf32_single {
namespace {

constexpr int kThreads = 128;
constexpr int kMaxGrid = 65535;
using Tf32 = cutlass::tfloat32_t;
struct WeightSplit { uint32_t big, small; };

struct Parameters {
  uint64_t rows;
  int source_rows, target_rows, source_first, target_first, path_rows;
  int k, n, packed_k;
  double coefficient;
};

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("eqlinear tf32 single: ") + operation + ": " +
                             cudaGetErrorString(status));
}

size_t checked_bytes(uint64_t rows, uint64_t columns, size_t element_size) {
  const size_t limit = std::numeric_limits<size_t>::max() / element_size;
  if (rows > limit / columns)
    throw std::invalid_argument("eqlinear tf32 single: operand size overflow");
  return static_cast<size_t>(rows * columns) * element_size;
}

// Only the immutable weight is padded, transposed and split. CG is deliberately
// absent: its double multiplication belongs after the complete FP32 dot.
template <bool Dgrad>
__global__ void pack_weights(const float* __restrict__ w, WeightSplit* __restrict__ packed,
                             int u, int v, int packed_k, uint64_t count) {
  const int k_extent = Dgrad ? v : u;
  cutlass::NumericConverterFastF32<> split;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const int k = i % packed_k;
    const uint64_t n = i / packed_k;
    float value = 0.f;
    if (k < k_extent) value = w[Dgrad ? n * v + k : uint64_t(k) * v + n];
    const auto pair = split(value);
    packed[i] = {pair[0].raw(), pair[1].raw()};
  }
}

// K=1 still executes a legal TensorCore TF32x3 operation, using the minimum
// m16n8k4 atom. All other contractions use m16n8k8, including masked U=10.
template <bool ScalarK> struct AtomPolicy;
template <> struct AtomPolicy<false> {
  using Atom = cute::MMA_Atom<cute::SM80_16x8x8_F32TF32TF32F32_TN>;
  static constexpr int k = 8, a = 4, b = 2, block_k = 32;
};
template <> struct AtomPolicy<true> {
  using Atom = cute::MMA_Atom<cute::SM80_16x8x4_F32TF32TF32F32_TN>;
  static constexpr int k = 4, a = 2, b = 1, block_k = 4;
};

template <int StaticK, int BlockN, bool DenseRows>
__global__ __launch_bounds__(kThreads)
void direct_gemm(const float* __restrict__ input, float* __restrict__ output,
                 const WeightSplit* __restrict__ weights, Parameters g) {
  using Policy = AtomPolicy<StaticK == 1>;
  using Mma = typename Policy::Atom;
  using LayoutA = typename Mma::LayoutA_TV;
  using LayoutB = typename Mma::LayoutB_TV;
  using LayoutC = typename Mma::LayoutC_TV;
  constexpr int BK = Policy::block_k;
  constexpr int BN = BlockN;
  constexpr int BM = BN == 8 ? 64 : 32;
  constexpr int WarpN = BN == 8 ? 8 : BN / 2;
  constexpr int Repeats = WarpN / 8;
  // Four extra words break the repeated bank mapping of the MMA row loads.
  constexpr int Pitch = BK + 4;
  __shared__ uint32_t a_big[BM][Pitch], a_small[BM][Pitch];
  __shared__ uint32_t b_big[BN][Pitch], b_small[BN][Pitch];
  __shared__ float result[BM][BN + 1];

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int warp_m = (BN == 8 ? warp : warp / 2) * 16;
  const int warp_n = BN == 8 ? 0 : (warp % 2) * WarpN;
  const int n0 = blockIdx.x * BN;
  const int k_extent = StaticK ? StaticK : g.k;
  Mma mma;
  cutlass::NumericConverterFastF32<> split;

  for (uint64_t m0 = uint64_t(blockIdx.y) * BM; m0 < g.rows;
       m0 += uint64_t(gridDim.y) * BM) {
    auto dot = cute::make_tensor<float>(cute::Shape<cute::_4, cute::Int<Repeats>>{});
    cute::clear(dot);
    for (int64_t k0 = 0; k0 < k_extent; k0 += BK) {
      for (int i = threadIdx.x; i < BM * BK; i += kThreads) {
        const int m = i / BK, kk = i % BK;
        const uint64_t logical_m = m0 + m;
        const int row = DenseRows ? 0 : logical_m % g.target_rows;
        float value = 0.f;
        if (logical_m < g.rows && k0 + kk < k_extent &&
            (DenseRows || (row >= g.target_first && row - g.target_first < g.path_rows))) {
          const uint64_t source = DenseRows ? logical_m :
              (logical_m / g.target_rows) * g.source_rows + g.source_first + row - g.target_first;
          // Scalar masked loads accept raw row pitches such as U10's 40 B,
          // and arbitrary input base alignment. No global activation padding.
          value = input[source * k_extent + k0 + kk];
        }
        const auto pair = split(value);
        a_big[m][kk] = pair[0].raw();
        a_small[m][kk] = pair[1].raw();
      }
      for (int i = threadIdx.x; i < BN * BK; i += kThreads) {
        const int n = i / BK, kk = i % BK;
        WeightSplit pair{0, 0};
        if (n0 + n < g.n && k0 + kk < k_extent)
          pair = weights[uint64_t(n0 + n) * g.packed_k + k0 + kk];
        b_big[n][kk] = pair.big;
        b_small[n][kk] = pair.small;
      }
      __syncthreads();

      #pragma unroll
      for (int kk = 0; kk < BK; kk += Policy::k) {
        auto ah = cute::make_tensor<Tf32>(cute::Shape<cute::Int<Policy::a>>{});
        auto al = cute::make_tensor_like(ah);
        #pragma unroll
        for (int i = 0; i < Policy::a; ++i) {
          const int coordinate = LayoutA{}(lane, i);
          const int m = warp_m + coordinate % 16, k = kk + coordinate / 16;
          ah(i) = Tf32::bitcast(a_big[m][k]);
          al(i) = Tf32::bitcast(a_small[m][k]);
        }
        #pragma unroll
        for (int tile = 0; tile < Repeats; ++tile) {
          auto bh = cute::make_tensor<Tf32>(cute::Shape<cute::Int<Policy::b>>{});
          auto bl = cute::make_tensor_like(bh);
          #pragma unroll
          for (int i = 0; i < Policy::b; ++i) {
            const int coordinate = LayoutB{}(lane, i);
            const int n = warp_n + tile * 8 + coordinate % 8, k = kk + coordinate / 8;
            bh(i) = Tf32::bitcast(b_big[n][k]);
            bl(i) = Tf32::bitcast(b_small[n][k]);
          }
          auto accum = dot(cute::_, tile);
          // Keep the shape path's three-term arithmetic and rounding. Raw
          // residual bits must never pass through Tf32(float)'s constructor.
          mma.call(ah, bl, accum);
          mma.call(al, bh, accum);
          mma.call(ah, bh, accum);
        }
      }
      __syncthreads();  // Readers finish before the next A/B stage reuses storage.
    }

    #pragma unroll
    for (int tile = 0; tile < Repeats; ++tile)
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int coordinate = LayoutC{}(lane, i);
        const int m = warp_m + coordinate % 16;
        const int n = warp_n + tile * 8 + coordinate / 16;
        const int row = DenseRows ? 0 : (m0 + m) % g.target_rows;
        const bool active = DenseRows || (row >= g.target_first && row - g.target_first < g.path_rows);
        result[m][n] = active ? static_cast<float>(static_cast<double>(dot(i, tile)) *
                                                   g.coefficient) : 0.f;
      }
    __syncthreads();

    // Remap the MMA fragments to consecutive channels. Scalar K=1 dgrad
    // writes full float4 vectors for N=16/96/128/224, rather than strided lanes.
    if (g.n % 4 == 0 && (reinterpret_cast<uintptr_t>(output) & 15) == 0) {
      for (int i = threadIdx.x; i < BM * BN / 4; i += kThreads) {
        const int m = i / (BN / 4), n = (i % (BN / 4)) * 4;
        if (m0 + m < g.rows && n0 + n + 3 < g.n)
          *reinterpret_cast<float4*>(output + (m0 + m) * g.n + n0 + n) =
              make_float4(result[m][n], result[m][n + 1], result[m][n + 2], result[m][n + 3]);
      }
    } else {
      for (int i = threadIdx.x; i < BM * BN; i += kThreads) {
        const int m = i / BN, n = i % BN;
        if (m0 + m < g.rows && n0 + n < g.n)
          output[(m0 + m) * g.n + n0 + n] = result[m][n];
      }
    }
    __syncthreads();  // Also protects the epilogue in the capped-grid loop.
  }
}

using Launch = void (*)(const float*, float*, const WeightSplit*, Parameters, cudaStream_t);

// A whole warpgroup owns a small output tile. B's two immutable halves are
// staged together; A is split directly into the CuTe-defined RS registers.
// Drain each group before reusing any operand, preserving RS register and
// shared stage lifetimes explicitly.
template <int K, int BN>
__global__ __launch_bounds__(kThreads)
void dense_gmma(const float* __restrict__ input, float* __restrict__ output,
                const WeightSplit* __restrict__ weights, Parameters g) {
  using Op = std::conditional_t<BN == 32, cute::SM90_64x32x8_F32TF32TF32_RS_TN<>,
                                         cute::SM90_64x64x8_F32TF32TF32_RS_TN<>>;
  using Mma = cute::MMA_Atom<Op>;
  using LayoutA = typename Mma::LayoutA_TV;
  using LayoutC = typename Mma::LayoutC_TV;
  using LayoutB = decltype(cute::tile_to_shape(cute::GMMA::Layout_K_INTER_Atom<Tf32>{},
                                              cute::Shape<cute::Int<BN>, cute::_8>{}));
  __shared__ __align__(16) uint32_t b_big[BN * 8], b_small[BN * 8];
  __shared__ float result[64][BN + 1];
  auto sbh = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Tf32*>(b_big)), LayoutB{});
  auto sbl = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Tf32*>(b_small)), LayoutB{});
  auto bh = cute::make_tensor<uint64_t>(cute::Shape<cute::_1>{});
  auto bl = cute::make_tensor_like(bh);
  bh(0) = cute::GMMA::make_gmma_desc<cute::GMMA::Major::K>(sbh);
  bl(0) = cute::GMMA::make_gmma_desc<cute::GMMA::Major::K>(sbl);
  const int n0 = blockIdx.x * BN;
  Mma mma;
  cutlass::NumericConverterFastF32<> split;
  for (uint64_t m0 = uint64_t(blockIdx.y) * 64; m0 < g.rows;
       m0 += uint64_t(gridDim.y) * 64) {
    auto dot = cute::make_tensor<float>(cute::Shape<cute::Int<BN / 2>>{});
    cute::clear(dot);
    for (int k0 = 0; k0 < K; k0 += 8) {
      auto ah = cute::make_tensor<Tf32>(cute::Shape<cute::_4>{});
      auto al = cute::make_tensor_like(ah);
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int coordinate = LayoutA{}(threadIdx.x, i);
        const uint64_t m = m0 + coordinate % 64;
        const int k = k0 + coordinate / 64;
        const auto pair = split(m < g.rows ? input[m * K + k] : 0.f);
        ah(i) = pair[0];
        al(i) = pair[1];
      }
      #pragma unroll
      for (int i = threadIdx.x; i < BN * 8; i += kThreads) {
        const int n = i / 8, k = i % 8;
        WeightSplit pair{0, 0};
        if (n0 + n < g.n) pair = weights[uint64_t(n0 + n) * K + k0 + k];
        sbh(n, k) = Tf32::bitcast(pair.big);
        sbl(n, k) = Tf32::bitcast(pair.small);
      }
      cutlass::arch::fence_view_async_shared();
      __syncthreads();
      cute::warpgroup_fence_operand(dot);
      cute::warpgroup_fence_operand(ah);
      cute::warpgroup_fence_operand(al);
      cute::warpgroup_arrive();
      mma.call(ah, bl, dot);
      mma.call(al, bh, dot);
      mma.call(ah, bh, dot);
      cute::warpgroup_commit_batch();
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(dot);
      cute::warpgroup_fence_operand(ah);
      cute::warpgroup_fence_operand(al);
      __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < BN / 2; ++i) {
      const int coordinate = LayoutC{}(threadIdx.x, i);
      const int m = coordinate % 64, n = coordinate / 64;
      result[m][n] = static_cast<float>(static_cast<double>(dot(i)) * g.coefficient);
    }
    __syncthreads();
    if ((reinterpret_cast<uintptr_t>(output) & 15) == 0) {
      for (int i = threadIdx.x; i < 64 * BN / 4; i += kThreads) {
        const int m = i / (BN / 4), n = (i % (BN / 4)) * 4;
        if (m0 + m < g.rows && n0 + n + 3 < g.n)
          *reinterpret_cast<float4*>(output + (m0 + m) * g.n + n0 + n) =
              make_float4(result[m][n], result[m][n + 1], result[m][n + 2], result[m][n + 3]);
      }
    } else {
      for (int i = threadIdx.x; i < 64 * BN; i += kThreads) {
        const int m = i / BN, n = i % BN;
        if (m0 + m < g.rows && n0 + n < g.n)
          output[(m0 + m) * g.n + n0 + n] = result[m][n];
      }
    }
    __syncthreads();
  }
}

template <int K, int N>
void launch_gmma(const float* input, float* output, const WeightSplit* weights,
                  Parameters g, cudaStream_t stream) {
  const dim3 grid((g.n + N - 1) / N, std::min<uint64_t>((g.rows + 63) / 64, kMaxGrid));
  dense_gmma<K, N><<<grid, kThreads, 0, stream>>>(input, output, weights, g);
}

template <int K, int N, bool DenseRows>
void launch(const float* input, float* output, const WeightSplit* weights,
             Parameters g, cudaStream_t stream) {
  constexpr int M = N == 8 ? 64 : 32;
  const dim3 grid((uint64_t(g.n) + N - 1) / N,
                  std::min<uint64_t>((g.rows + M - 1) / M, kMaxGrid));
  direct_gemm<K, N, DenseRows><<<grid, kThreads, 0, stream>>>(input, output, weights, g);
}

template <int K>
Launch select_columns(int n, bool dense) {
  if (n <= 8) return dense ? &launch<K, 8, true> : &launch<K, 8, false>;
  if (n <= 16) return dense ? &launch<K, 16, true> : &launch<K, 16, false>;
  return dense ? &launch<K, 64, true> : &launch<K, 64, false>;
}

Launch select_kernel(int k, int n, bool dense, uint64_t rows) {
  if (dense && k == n) {
    // Conservative crossover points from standalone factory event timings.
    // Small M retains the shorter SM80 path; no public speedup is implied.
    if (k == 96 && rows >= 25116) return &launch_gmma<96, 32>;
    if (k == 128 && rows >= 25116) return &launch_gmma<128, 64>;
    if (k == 224 && rows >= 6312) return &launch_gmma<224, 64>;
  }
  switch (k) {
    case 1: return select_columns<1>(n, dense);
    case 10: return select_columns<10>(n, dense);
    case 16: return select_columns<16>(n, dense);
    case 96: return select_columns<96>(n, dense);
    case 128: return select_columns<128>(n, dense);
    case 224: return select_columns<224>(n, dense);
    default: return select_columns<0>(n, dense);
  }
}

class SinglePathPlan final : public ShapePlan {
 public:
  SinglePathPlan(const EqLinearProblemC& p, const void* w, const double* cg,
                 bool dgrad, cudaStream_t stream) : geometry_(p, cg) {
    if (!w) throw std::invalid_argument("eqlinear tf32 single: null weight");
    const auto& path = geometry_.path;
    const int k = dgrad ? p.V : p.U, n = dgrad ? p.U : p.V;
    const int alignment = k == 1 ? 4 : 32;
    const int64_t packed_k = ((int64_t(k) + alignment - 1) / alignment) * alignment;
    if (packed_k > std::numeric_limits<int>::max())
      throw std::invalid_argument("eqlinear tf32 single: packed K overflow");
    const int source_rows = dgrad ? p.J : p.I, target_rows = dgrad ? p.I : p.J;
    (void)checked_bytes(uint64_t(p.B) * source_rows, k, sizeof(float));
    (void)checked_bytes(uint64_t(p.B) * target_rows, n, sizeof(float));
    parameters_ = {uint64_t(p.B) * target_rows, source_rows, target_rows,
                   dgrad ? path.output_row : path.input_row,
                   dgrad ? path.input_row : path.output_row, path.rows,
                   k, n, static_cast<int>(packed_k), path.coefficient};
    const bool dense = source_rows == target_rows && path.rows == target_rows &&
                        path.input_row == 0 && path.output_row == 0;
    launch_ = select_kernel(k, n, dense, parameters_.rows);
    const size_t bytes = checked_bytes(n, packed_k, sizeof(WeightSplit));
    if (weights_.reset(bytes)) throw std::runtime_error("eqlinear tf32 single: weight allocation failed");
    const uint64_t count = uint64_t(n) * packed_k;
    const int blocks = std::min<uint64_t>((count + kThreads - 1) / kThreads, kMaxGrid);
    if (dgrad)
      pack_weights<true><<<blocks, kThreads, 0, stream>>>(
          static_cast<const float*>(w), weights_.as<WeightSplit>(), p.U, p.V, packed_k, count);
    else
      pack_weights<false><<<blocks, kThreads, 0, stream>>>(
          static_cast<const float*>(w), weights_.as<WeightSplit>(), p.U, p.V, packed_k, count);
    check(cudaGetLastError(), "pack weights launch");
  }

  int run(const void* input, void* output, cudaStream_t stream) override {
    if (!input || !output) return static_cast<int>(cudaErrorInvalidValue);
    launch_(static_cast<const float*>(input), static_cast<float*>(output),
            weights_.as<WeightSplit>(), parameters_, stream);
    return static_cast<int>(cudaGetLastError());
  }

 private:
  SinglePathGeometry geometry_;
  Parameters parameters_{};
  Launch launch_ = nullptr;
  DeviceBuffer weights_;
};

}  // namespace

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                const double* cg, bool dgrad, cudaStream_t stream) {
  const SinglePathGeometry geometry(p, cg);
  if (tf32_stock::selected(geometry))
    return tf32_stock::create(p, w, cg, dgrad, stream);
  return std::make_unique<SinglePathPlan>(p, w, cg, dgrad, stream);
}

}  // namespace fasteq::eqlinear::tf32_single
