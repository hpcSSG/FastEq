#include "single_path_plan.cuh"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "../../common/single_path.h"
#include "block_accumulator.cuh"
#include "device_buffer.h"
#include "shape_plan.cuh"
#include "single_path_math.cuh"

namespace fasteq::eqlinear::fp32_single {
namespace {

constexpr int kThreads = 128;

struct Parameters {
  const float* weights;
  int K, N;
  int64_t M, tiles;
  float2 coefficient;
};

struct BlockedTile {
  static constexpr int kM = 64, kN = 64, kK = 16, kBlockK = 8;
  static constexpr int kRows = 8, kColumns = 4;

  static bool accepts(const Parameters& p, bool dgrad) {
    if (p.K < 32 || p.N < kN || p.M < 1024) return false;
    // Small square contractions are sensitive to tile parallelism and public
    // autograd overhead; retain their established kernels below this work size.
    if (p.K == p.N && p.N <= 128)
      return !dgrad && p.M >= 4096;
    return true;
  }
};

// Static, unscaled weights in true [K,N] order: no K16/N32 padding and no
// activation packing. Dgrad transposes once during plan construction.
__global__ void pack_weight(const float* input, float* output, int V,
                            int N, int64_t count, bool dgrad) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t k = i / N, n = i % N;
    output[i] = dgrad ? input[n * V + k] : input[k * V + n];
  }
}

// One CTA owns one row and eight output columns. Lanes cooperate over K,
// keeping both hi and lo in every deterministic shuffle merge. For N=1 a
// single-warp CTA owns a complete dot. No global split-K/atomic/temp buffer.
template <int Columns, int Warps>
__global__ void __launch_bounds__(32 * Warps) cooperative_dot(
    Parameters p, const float* __restrict__ input, float* __restrict__ output) {
  constexpr int kLanes = 32 / Columns;
  constexpr int kWidth = Columns * Warps;
  const int lane = threadIdx.x & 31;
  const int part = lane / Columns;
  const int column = (threadIdx.x / 32) * Columns + lane % Columns;
  const int64_t nt = (int64_t(p.N) + kWidth - 1) / kWidth;
  for (int64_t tile = blockIdx.x; tile < p.tiles; tile += gridDim.x) {
    const int64_t row = tile / nt;
    const int64_t n = (tile % nt) * kWidth + column;
    DotAccumulator partial;
    if (n < p.N) {
      for (int64_t k = part; k < p.K; k += kLanes)
        partial.add_product(input[row * p.K + k], p.weights[k * p.N + n]);
    }
    FloatPair dot = partial.finish();
#pragma unroll
    for (int offset = 16; offset >= Columns; offset /= 2) {
      const FloatPair other{__shfl_down_sync(0xffffffffu, dot.hi, offset),
                            __shfl_down_sync(0xffffffffu, dot.lo, offset)};
      if (part < offset / Columns) dot.add(other);
    }
    if (part == 0 && n < p.N) output[row * p.N + n] = dot.scale(p.coefficient);
  }
}

// K=1 is an outer product, including the V1 dgrad case. Consecutive lanes
// write consecutive output channels. Only valid logical activations are read.
__global__ void __launch_bounds__(kThreads) outer_product(
    Parameters p, const float* __restrict__ input, float* __restrict__ output) {
  const int64_t count = p.M * p.N;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = i / p.N, n = i % p.N;
    output[i] = FloatPair::two_prod(input[row], p.weights[n]).scale(p.coefficient);
  }
}

// True short-K specialization: each warp owns a row and consecutive N lanes,
// with Rows/4 rows/warp. U10 executes exactly ten compensated products per output.
template <int FixedK, int Rows = 8>
__global__ void __launch_bounds__(kThreads, 4) short_k(
    Parameters p, const float* __restrict__ input, float* __restrict__ output) {
  constexpr int kM = Rows, kN = 32;
  __shared__ float a[kM][FixedK];
  __shared__ float w[FixedK][kN];
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int64_t nt = (int64_t(p.N) + kN - 1) / kN;
  for (int64_t tile = blockIdx.x; tile < p.tiles; tile += gridDim.x) {
    const int64_t m0 = (tile / nt) * kM, n0 = (tile % nt) * kN;
    for (int i = threadIdx.x; i < kM * FixedK; i += kThreads) {
      const int m = i / FixedK, k = i % FixedK;
      a[m][k] = m0 + m < p.M ? input[(m0 + m) * p.K + k] : 0.f;
    }
    for (int i = threadIdx.x; i < FixedK * kN; i += kThreads) {
      const int k = i / kN, n = i % kN;
      w[k][n] = n0 + n < p.N ? p.weights[int64_t(k) * p.N + n0 + n] : 0.f;
    }
    __syncthreads();
    DotAccumulator dot[Rows / 4]{};
#pragma unroll 1
    for (int k = 0; k < FixedK; ++k) {
      const float weight = w[k][lane];
#pragma unroll
      for (int m = 0; m < Rows / 4; ++m)
        dot[m].add_product(a[warp + 4 * m][k], weight);
    }
#pragma unroll
    for (int m = 0; m < Rows / 4; ++m) {
      const int64_t row = m0 + warp + 4 * m, n = n0 + lane;
      if (row < p.M && n < p.N) output[row * p.N + n] = dot[m].finish().scale(p.coefficient);
    }
    __syncthreads();
  }
}

// K10: five independent two-term FP32 FMA chains, followed by a TwoSum fold
// before applying both CG components. No per-product compensation is needed on
// this accuracy-gated shape class. This is not bitwise FloatPair arithmetic.
// A lane owns four adjacent N values; all public N=96/128/224 use aligned float4
// stores. Eight rows share the static weight tile, with no padded K arithmetic.
__global__ void __launch_bounds__(kThreads, 4) short_ten_chains(
    Parameters p, const float* __restrict__ input, float* __restrict__ output) {
  constexpr int kM = 8, kN = 128;
  __shared__ float a[kM][10];
  __shared__ __align__(16) float w[10][kN];
  const int n4 = (threadIdx.x & 31) * 4, warp = threadIdx.x / 32;
  const int64_t nt = (int64_t(p.N) + kN - 1) / kN;
  for (int64_t tile = blockIdx.x; tile < p.tiles; tile += gridDim.x) {
    const int64_t m0 = tile / nt * kM, n0 = tile % nt * kN;
    if (threadIdx.x < kM * 10) {
      const int m = threadIdx.x / 10, k = threadIdx.x % 10;
      a[m][k] = m0 + m < p.M ? input[(m0 + m) * 10 + k] : 0.f;
    }
    for (int i = threadIdx.x; i < 10 * kN / 4; i += kThreads) {
      const int k = i / (kN / 4), n = i % (kN / 4) * 4;
      *reinterpret_cast<float4*>(&w[k][n]) = n0 + n + 3 < p.N
          ? *reinterpret_cast<const float4*>(p.weights + int64_t(k) * p.N + n0 + n)
          : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();
    float partial[5][2][4];
#pragma unroll
    for (int h = 0; h < 5; ++h) {
      const float4 first = *reinterpret_cast<const float4*>(&w[2 * h][n4]);
      const float4 second = *reinterpret_cast<const float4*>(&w[2 * h + 1][n4]);
      const float w0[4]{first.x, first.y, first.z, first.w};
      const float w1[4]{second.x, second.y, second.z, second.w};
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const float a0 = a[warp + 4 * m][2 * h];
        const float a1 = a[warp + 4 * m][2 * h + 1];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          partial[h][m][c] = __fmaf_rn(a1, w1[c], __fmul_rn(a0, w0[c]));
        }
      }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      float result[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        float sum = partial[0][m][c], low = 0.f;
#pragma unroll
        for (int h = 1; h < 5; ++h) {
          const FloatPair added = FloatPair::two_sum(sum, partial[h][m][c]);
          low = __fadd_rn(low, added.lo);
          sum = added.hi;
        }
        result[c] = FloatPair::two_sum(sum, low).scale(p.coefficient);
      }
      const int64_t row = m0 + warp + 4 * m, col = n0 + n4;
      if (row < p.M && col + 3 < p.N) {
        float* destination = output + row * p.N + col;
        // A float4 assignment was scalarized into four strided stores in SASS.
        // Keep the 128-bit transaction explicit, but retain the public float*
        // contract for a caller-provided output view with a scalar offset.
        if ((reinterpret_cast<uintptr_t>(destination) & 15u) == 0) {
          asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                       :: "l"(destination), "f"(result[0]), "f"(result[1]),
                          "f"(result[2]), "f"(result[3]) : "memory");
        } else {
#pragma unroll
          for (int c = 0; c < 4; ++c) destination[c] = result[c];
        }
      }
    }
    __syncthreads();
  }
}

// A single logical SIMT GEMM, with four outputs/thread, no path/segment lookup,
// no separate total accumulator and a 3 KiB shared footprint. Looping K keeps
// the compensated instruction body small rather than expanding a K16 body.
__global__ void __launch_bounds__(kThreads, 4) tiled_dot(
    Parameters p, const float* __restrict__ input, float* __restrict__ output) {
  constexpr int kM = 16, kN = 32, kK = 16;
  __shared__ float a[kM][kK + 1];
  __shared__ float w[kK][kN];
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int64_t nt = (int64_t(p.N) + kN - 1) / kN;
  for (int64_t tile = blockIdx.x; tile < p.tiles; tile += gridDim.x) {
    const int64_t m0 = (tile / nt) * kM, n0 = (tile % nt) * kN;
    DotAccumulator dot[4]{};
    for (int64_t k0 = 0; k0 < p.K; k0 += kK) {
      for (int i = threadIdx.x; i < kM * kK; i += kThreads) {
        const int m = i / kK, k = i % kK;
        a[m][k] = m0 + m < p.M && k0 + k < p.K
            ? input[(m0 + m) * p.K + k0 + k] : 0.f;
      }
      for (int i = threadIdx.x; i < kK * kN; i += kThreads) {
        const int k = i / kN, n = i % kN;
        w[k][n] = k0 + k < p.K && n0 + n < p.N
            ? p.weights[(k0 + k) * p.N + n0 + n] : 0.f;
      }
      __syncthreads();
      const int valid_k = static_cast<int>(min(int64_t(kK), int64_t(p.K) - k0));
#pragma unroll 1
      for (int k = 0; k < valid_k; ++k) {
        const float weight = w[k][lane];
#pragma unroll
        for (int m = 0; m < 4; ++m)
          dot[m].add_product(a[warp + 4 * m][k], weight);
      }
      __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const int64_t row = m0 + warp + 4 * m, n = n0 + lane;
      if (row < p.M && n < p.N) output[row * p.N + n] = dot[m].finish().scale(p.coefficient);
    }
    __syncthreads();
  }
}

// Square P1 uses the existing accuracy policy's eight independent true-FP32
// FMA chains, with the same k%8 assignment. A 16x64 tile limits the chain state
// to 64 registers/thread, instead of replicating the large legacy thread tile.
// Folding retains a FloatPair through CG scaling rather than rounding early.
__global__ void __launch_bounds__(kThreads, 4) square_dot(
    Parameters p, const float* __restrict__ input, float* __restrict__ output) {
  constexpr int kM = 16, kN = 64, kK = 16;
  __shared__ float a[kM][kK + 1];
  __shared__ float w[kK][kN];
  const int lane_m = (threadIdx.x & 31) / 8;
  const int lane_n = threadIdx.x % 8;
  const int warp_m = (threadIdx.x / 32) / 2;
  const int warp_n = (threadIdx.x / 32) % 2;
  const int64_t nt = (int64_t(p.N) + kN - 1) / kN;
  for (int64_t tile = blockIdx.x; tile < p.tiles; tile += gridDim.x) {
    const int64_t m0 = (tile / nt) * kM, n0 = (tile % nt) * kN;
    float acc[8][2][4]{};
    for (int64_t k0 = 0; k0 < p.K; k0 += kK) {
      for (int i = threadIdx.x; i < kM * kK; i += kThreads) {
        const int m = i / kK, k = i % kK;
        a[m][k] = m0 + m < p.M && k0 + k < p.K
            ? input[(m0 + m) * p.K + k0 + k] : 0.f;
      }
      for (int i = threadIdx.x; i < kK * kN; i += kThreads) {
        const int k = i / kN, n = i % kN;
        w[k][n] = k0 + k < p.K && n0 + n < p.N
            ? p.weights[(k0 + k) * p.N + n0 + n] : 0.f;
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < kK; ++k) {
        float av[2], bv[4];
#pragma unroll
        for (int m = 0; m < 2; ++m) av[m] = a[warp_m * 8 + lane_m + 4 * m][k];
#pragma unroll
        for (int n = 0; n < 4; ++n) bv[n] = w[k][warp_n * 32 + lane_n + 8 * n];
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
          for (int n = 0; n < 4; ++n)
            acc[k % 8][m][n] = __fmaf_rn(av[m], bv[n], acc[k % 8][m][n]);
      }
      __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        FloatPair dot{acc[0][m][n], 0.f};
#pragma unroll
        for (int h = 1; h < 8; ++h) {
          const FloatPair sum = FloatPair::two_sum(dot.hi, acc[h][m][n]);
          dot.lo = __fadd_rn(dot.lo, sum.lo);
          dot.hi = sum.hi;
        }
        const int64_t row = m0 + warp_m * 8 + lane_m + 4 * m;
        const int64_t col = n0 + warp_n * 32 + lane_n + 8 * n;
        if (row < p.M && col < p.N)
          output[row * p.N + col] = FloatPair::two_sum(dot.hi, dot.lo).scale(p.coefficient);
      }
    __syncthreads();
  }
}

// Dense, sufficiently large single-path GEMMs: one 8x4 output accumulator per
// thread. Each K8 dot rounds before joining that accumulator; the final CG
// multiply retains both coefficient components. The original kernels remain
// available for short K, narrow N, small M and numerical/performance fallback.
__global__ void __launch_bounds__(kThreads, 2) blocked_dot(
    Parameters p, const float* __restrict__ input, float* __restrict__ output) {
  using Tile = BlockedTile;
  __shared__ float a[Tile::kM][Tile::kK + 1];
  __shared__ float w[Tile::kK][Tile::kN];
  const int lane_m = (threadIdx.x & 31) / 8;
  const int lane_n = threadIdx.x % 8;
  const int warp_m = (threadIdx.x / 32) / 2;
  const int warp_n = (threadIdx.x / 32) % 2;
  const int64_t nt = (int64_t(p.N) + Tile::kN - 1) / Tile::kN;
  for (int64_t tile = blockIdx.x; tile < p.tiles; tile += gridDim.x) {
    const int64_t m0 = (tile / nt) * Tile::kM, n0 = (tile % nt) * Tile::kN;
    float acc[Tile::kRows][Tile::kColumns]{};
    for (int64_t k0 = 0; k0 < p.K; k0 += Tile::kK) {
#pragma unroll
      for (int i = threadIdx.x; i < Tile::kM * Tile::kK; i += kThreads) {
        const int m = i / Tile::kK, k = i % Tile::kK;
        a[m][k] = m0 + m < p.M && k0 + k < p.K
            ? input[(m0 + m) * p.K + k0 + k] : 0.f;
      }
#pragma unroll
      for (int i = threadIdx.x; i < Tile::kK * Tile::kN; i += kThreads) {
        const int k = i / Tile::kN, n = i % Tile::kN;
        w[k][n] = k0 + k < p.K && n0 + n < p.N
            ? p.weights[(k0 + k) * p.N + n0 + n] : 0.f;
      }
      __syncthreads();
#pragma unroll
      for (int chunk = 0; chunk < Tile::kK; chunk += Tile::kBlockK) {
        float av[Tile::kRows][Tile::kBlockK];
        float bv[Tile::kColumns][Tile::kBlockK];
#pragma unroll
        for (int m = 0; m < Tile::kRows; ++m)
#pragma unroll
          for (int k = 0; k < Tile::kBlockK; ++k)
            av[m][k] = a[warp_m * (Tile::kM / 2) + lane_m + 4 * m][chunk + k];
#pragma unroll
        for (int n = 0; n < Tile::kColumns; ++n)
#pragma unroll
          for (int k = 0; k < Tile::kBlockK; ++k)
            bv[n][k] = w[chunk + k][warp_n * (Tile::kN / 2) + lane_n + 8 * n];
#pragma unroll
        for (int m = 0; m < Tile::kRows; ++m)
#pragma unroll
          for (int n = 0; n < Tile::kColumns; ++n)
            eqlin_fp32::BlockAccumulator::add<Tile::kBlockK>(acc[m][n], av[m], bv[n]);
      }
      __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < Tile::kRows; ++m)
#pragma unroll
      for (int n = 0; n < Tile::kColumns; ++n) {
        const int64_t row = m0 + warp_m * (Tile::kM / 2) + lane_m + 4 * m;
        const int64_t col = n0 + warp_n * (Tile::kN / 2) + lane_n + 8 * n;
        if (row < p.M && col < p.N)
          output[row * p.N + col] = FloatPair{acc[m][n], 0.f}.scale(p.coefficient);
      }
  }
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("eqlinear FP32 single: ") + operation + ": " +
                             cudaGetErrorString(status));
}

int64_t checked_product(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b))
    throw std::invalid_argument("eqlinear FP32 single: extent overflow");
  return a * b;
}

class Plan final : public ShapePlan {
 public:
  Plan(const SinglePathGeometry& single, const void* weight, bool dgrad, cudaStream_t stream) {
    if (!weight) throw std::invalid_argument("eqlinear FP32 single: null weight");
    const auto& g = single.geometry;
    p_.K = dgrad ? g.V : g.U;
    p_.N = dgrad ? g.U : g.V;
    p_.M = checked_product(g.B, single.path.rows);
    checked_product(checked_product(p_.M, p_.K), sizeof(float));
    const int64_t elements = checked_product(p_.M, p_.N);
    checked_product(elements, sizeof(float));
    const float hi = static_cast<float>(single.path.coefficient);
    p_.coefficient = {hi, static_cast<float>(single.path.coefficient - double(hi))};
    if (p_.K == 1) {
      p_.tiles = (elements + kThreads - 1) / kThreads;
      launch_ = &launch_outer;
    } else if (p_.N == 1) {
      p_.tiles = p_.M;
      launch_ = &launch_cooperative<1, 1>;
    } else if (p_.K > 16 && p_.M <= 512) {
      p_.tiles = checked_product(p_.M, (int64_t(p_.N) + 7) / 8);
      launch_ = &launch_cooperative<2, 4>;
    } else if (BlockedTile::accepts(p_, dgrad)) {
      p_.tiles = checked_product((p_.M + BlockedTile::kM - 1) / BlockedTile::kM,
                                 (int64_t(p_.N) + BlockedTile::kN - 1) / BlockedTile::kN);
      launch_ = &launch_blocked;
    } else if (p_.K == p_.N && (p_.K == 96 || p_.K == 128 || p_.K == 224)) {
      p_.tiles = checked_product((p_.M + 15) / 16, (int64_t(p_.N) + 63) / 64);
      launch_ = &launch_square;
    } else if (p_.K == 10) {
      if (p_.M >= 1024 && p_.N == 224) {
        p_.tiles = checked_product((p_.M + 7) / 8, (int64_t(p_.N) + 127) / 128);
        launch_ = &launch_ten_chains;
      } else if (p_.M >= 1024) {
        configure_tile(32);
        launch_ = &launch_short<10, 32>;
      } else {
        configure_tile(8);
        launch_ = &launch_short<10>;
      }
    } else if (p_.K == 16) {
      configure_tile(8);
      launch_ = &launch_short<16>;
    } else {
      configure_tile(16);
      launch_ = &launch_tiled;
    }
    const int64_t count = checked_product(p_.K, p_.N);
    const int64_t bytes = checked_product(count, sizeof(float));
    if (weights_.reset(static_cast<size_t>(bytes)))
      throw std::runtime_error("eqlinear FP32 single: allocation failed");
    p_.weights = weights_.as<float>();
    pack_weight<<<blocks((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const float*>(weight), weights_.as<float>(), g.V,
        p_.N, count, dgrad);
    check_cuda(cudaGetLastError(), "pack weight");
    check_cuda(cudaStreamSynchronize(stream), "prepare");
  }

  int run(const void* input, void* output, cudaStream_t stream) override {
    if (!input || !output) return static_cast<int>(cudaErrorInvalidValue);
    return launch_(p_, static_cast<const float*>(input), static_cast<float*>(output), stream);
  }

 private:
  using Launch = int (*)(Parameters, const float*, float*, cudaStream_t);
  static unsigned blocks(int64_t count) {
    return static_cast<unsigned>(std::min<int64_t>(count, 65535));
  }
  void configure_tile(int m) {
    p_.tiles = checked_product((p_.M + m - 1) / m, (int64_t(p_.N) + 31) / 32);
  }
  static int launch_outer(Parameters p, const float* x, float* y, cudaStream_t s) {
    outer_product<<<blocks(p.tiles), kThreads, 0, s>>>(p, x, y);
    return static_cast<int>(cudaGetLastError());
  }
  template <int C, int W>
  static int launch_cooperative(Parameters p, const float* x, float* y, cudaStream_t s) {
    cooperative_dot<C, W><<<blocks(p.tiles), 32 * W, 0, s>>>(p, x, y);
    return static_cast<int>(cudaGetLastError());
  }
  template <int K, int Rows = 8>
  static int launch_short(Parameters p, const float* x, float* y, cudaStream_t s) {
    short_k<K, Rows><<<blocks(p.tiles), kThreads, 0, s>>>(p, x, y);
    return static_cast<int>(cudaGetLastError());
  }
  static int launch_tiled(Parameters p, const float* x, float* y, cudaStream_t s) {
    tiled_dot<<<blocks(p.tiles), kThreads, 0, s>>>(p, x, y);
    return static_cast<int>(cudaGetLastError());
  }
  static int launch_square(Parameters p, const float* x, float* y, cudaStream_t s) {
    square_dot<<<blocks(p.tiles), kThreads, 0, s>>>(p, x, y);
    return static_cast<int>(cudaGetLastError());
  }
  static int launch_blocked(Parameters p, const float* x, float* y, cudaStream_t s) {
    blocked_dot<<<blocks(p.tiles), kThreads, 0, s>>>(p, x, y);
    return static_cast<int>(cudaGetLastError());
  }
  static int launch_ten_chains(Parameters p, const float* x, float* y, cudaStream_t s) {
    short_ten_chains<<<blocks(p.tiles), kThreads, 0, s>>>(p, x, y);
    return static_cast<int>(cudaGetLastError());
  }

  Parameters p_{};
  Launch launch_ = nullptr;
  DeviceBuffer weights_;
};

}  // namespace

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                const double* cg, bool dgrad, cudaStream_t stream) {
  if (!required(p, dgrad)) return fp32_shape::create(p, w, cg, dgrad, stream);
  const SinglePathGeometry single(p, cg);
  // Preserve full logical strides and zero-filled holes through the established
  // shape implementation until those sparse mappings have a specialized kernel.
  if (single.path.input_row != 0 || single.path.output_row != 0 ||
      single.path.rows != single.geometry.I || single.path.rows != single.geometry.J)
    return fp32_shape::create(p, w, cg, dgrad, stream);
  return std::make_unique<Plan>(single, w, dgrad, stream);
}

}  // namespace fasteq::eqlinear::fp32_single
