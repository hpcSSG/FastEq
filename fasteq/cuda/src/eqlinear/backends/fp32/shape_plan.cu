#include "shape_plan.cuh"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma.hpp>
#include <cute/tensor.hpp>

#include "device_buffer.h"

namespace fasteq::eqlinear::fp32_shape {
namespace {

// The same FP32 error transforms as tiny_linear.cu. Explicit rounding keeps
// product residuals and cancellation terms intact under multiply/add contraction.
struct FloatPair {
  float hi = 0.f, lo = 0.f;

  __device__ __forceinline__ static FloatPair two_sum(float a, float b) {
    const float hi = __fadd_rn(a, b);
    const float bv = __fsub_rn(hi, a);
    return {hi, __fadd_rn(__fsub_rn(a, __fsub_rn(hi, bv)), __fsub_rn(b, bv))};
  }

  __device__ __forceinline__ static FloatPair two_prod(float a, float b) {
    const float hi = __fmul_rn(a, b);
    float lo;
    cute::UniversalFMA<float>::fma(lo, a, b, -hi);
    return {hi, lo};
  }

  __device__ __forceinline__ void add(FloatPair b) {
    const FloatPair s = two_sum(hi, b.hi);
    const FloatPair t = two_sum(lo, b.lo);
    const FloatPair u = two_sum(s.lo, t.hi);
    const FloatPair v = two_sum(s.hi, u.hi);
    *this = two_sum(v.hi, __fadd_rn(v.lo, __fadd_rn(u.lo, t.lo)));
  }

  __device__ __forceinline__ FloatPair multiply(FloatPair b) const {
    FloatPair result = two_prod(hi, b.hi);
    result.add(two_prod(hi, b.lo));
    result.add(two_prod(lo, b.hi));
    result.add(two_prod(lo, b.lo));
    return result;
  }

  __device__ __forceinline__ float rounded() const { return __fadd_rn(hi, lo); }
};

struct Path {
  int source_row;
  int weight_path;
  float2 coefficient;
};

// A segment has the same contributing paths for every row. Endpoints, rather
// than an expanded row table, bound metadata by O(P^2), independent of I/J/B.
struct Segment {
  int row, rows, first, count;
  int64_t tile_end;
};

template <int M_, int N_, int ThreadsM_, int ThreadsN_>
struct Tile {
  static constexpr int M = M_, N = N_, K = 16;
  static constexpr int ThreadsM = ThreadsM_, ThreadsN = ThreadsN_;
  static constexpr int Threads = ThreadsM * ThreadsN;
  static constexpr int TM = M / ThreadsM, TN = N / ThreadsN;
  static_assert(Threads == 128 && M % ThreadsM == 0 && N % ThreadsN == 0);
};

using ScalarTile = Tile<128, 1, 128, 1>;
using ThinTile = Tile<64, 4, 32, 4>;
using NarrowTile = Tile<64, 16, 32, 4>;
using WideTile = Tile<32, 32, 16, 8>;
using OuterTile = Tile<16, 32, 4, 32>;

struct Parameters {
  const Path* paths;
  const Segment* segments;
  const float* weights;
  int segments_count, B, source_rows, target_rows, K, N;
  int64_t packed_k, packed_n, tiles;
};

__device__ __forceinline__ int find_segment(Parameters p, int64_t tile) {
  int lo = 0, hi = p.segments_count;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (tile >= p.segments[mid].tile_end) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// V=1 dgrad is a direct outer product, not a 16-wide reduction. Each warp
// broadcasts one logical activation per row and reuses a weight across four
// rows. Empty segments execute only the zero stores. No shared tile, padded K
// work, block barrier, or temporary output is needed; path order and the exact
// FloatPair product/CG transforms are the same as the general kernel.
__global__ void __launch_bounds__(OuterTile::Threads) outer_product_kernel(
    Parameters p, const float* __restrict__ input, float* __restrict__ output) {
  const int lane = threadIdx.x % OuterTile::ThreadsN;
  const int warp = threadIdx.x / OuterTile::ThreadsN;
  const int64_t n_tiles = (int64_t(p.N) + OuterTile::N - 1) / OuterTile::N;
  for (int64_t tile = blockIdx.x; tile < p.tiles; tile += gridDim.x) {
    const int s = find_segment(p, tile);
    const Segment segment = p.segments[s];
    const int64_t local = tile - (s ? p.segments[s - 1].tile_end : 0);
    const int64_t m0 = (local / n_tiles) * OuterTile::M;
    const int64_t n = (local % n_tiles) * OuterTile::N + lane;
    const int64_t logical_m = int64_t(p.B) * segment.rows;
    FloatPair total[OuterTile::TM]{};
    for (int q = 0; q < segment.count; ++q) {
      const Path path = p.paths[segment.first + q];
      const float weight = n < p.N
          ? p.weights[int64_t(path.weight_path) * p.packed_k * p.packed_n + n]
          : 0.f;
#pragma unroll
      for (int m = 0; m < OuterTile::TM; ++m) {
        const int64_t row = m0 + warp + m * OuterTile::ThreadsM;
        float activation = 0.f;
        if (lane == 0 && row < logical_m)
          activation = input[(row / segment.rows) * p.source_rows +
                              path.source_row + row % segment.rows];
        activation = __shfl_sync(0xffffffffu, activation, 0);
        const FloatPair product = FloatPair::two_prod(activation, weight);
        total[m].add(product.multiply({path.coefficient.x, path.coefficient.y}));
      }
    }
#pragma unroll
    for (int m = 0; m < OuterTile::TM; ++m) {
      const int64_t row = m0 + warp + m * OuterTile::ThreadsM;
      if (row < logical_m && n < p.N) {
        const int64_t offset = ((row / segment.rows) * p.target_rows + segment.row +
                                 row % segment.rows) * int64_t(p.N) + n;
        output[offset] = total[m].rounded();
      }
    }
  }
}

template <class T>
struct SharedTile {
  // Odd A row stride avoids bank aliases between the M lanes. B is N contiguous.
  float a[2][T::M * (T::K + 1)];
  float w[2][T::K * T::N];
};

template <class T>
__device__ __forceinline__ void load_stage(
    SharedTile<T>& shared, int stage, int64_t k0, int64_t m0, int64_t n0,
    int64_t logical_m, Segment segment, Path path, Parameters p, const float* input) {
  auto a = cute::make_tensor(cute::make_smem_ptr(shared.a[stage]),
      cute::make_layout(cute::make_shape(cute::Int<T::M>{}, cute::Int<T::K>{}),
                        cute::make_stride(cute::Int<T::K + 1>{}, cute::Int<1>{})));
  auto w = cute::make_tensor(cute::make_smem_ptr(shared.w[stage]),
      cute::make_layout(cute::make_shape(cute::Int<T::N>{}, cute::Int<T::K>{}),
                        cute::make_stride(cute::Int<1>{}, cute::Int<T::N>{})));
#pragma unroll
  for (int i = threadIdx.x; i < T::M * T::K; i += T::Threads) {
    const int m = i / T::K, k = i % T::K;
    const int64_t row = m0 + m;
    const bool valid = row < logical_m && k0 + k < p.K;
    // Select a valid address even on the zero-fill path; cp.async src-size=0
    // performs no global read, including when unused logical rows contain NaNs.
    const int64_t offset = valid
        ? ((row / segment.rows) * p.source_rows + path.source_row + row % segment.rows) *
              int64_t(p.K) + k0 + k
        : 0;
    cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(input[offset], a(m, k), valid);
  }
#pragma unroll
  for (int i = threadIdx.x; i < T::N * T::K; i += T::Threads) {
    const int k = i / T::N, n = i % T::N;
    const bool valid = k0 + k < p.K && n0 + n < p.N;
    const int64_t offset = valid
        ? (int64_t(path.weight_path) * p.packed_k + k0 + k) * p.packed_n + n0 + n
        : 0;
    cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<float>::copy(p.weights[offset], w(n, k), valid);
  }
}

template <class T>
__global__ void __launch_bounds__(T::Threads) fused_kernel(
    Parameters p, const float* __restrict__ input, float* __restrict__ output) {
  __shared__ SharedTile<T> shared;
  const int lane_m = threadIdx.x / T::ThreadsN;
  const int lane_n = threadIdx.x % T::ThreadsN;
  const int64_t n_tiles = (int64_t(p.N) + T::N - 1) / T::N;
  for (int64_t tile = blockIdx.x; tile < p.tiles; tile += gridDim.x) {
    const int lo = find_segment(p, tile);
    const Segment segment = p.segments[lo];
    const int64_t local = tile - (lo ? p.segments[lo - 1].tile_end : 0);
    const int64_t m0 = (local / n_tiles) * T::M;
    const int64_t n0 = (local % n_tiles) * T::N;
    const int64_t logical_m = int64_t(p.B) * segment.rows;
    FloatPair total[T::TM][T::TN]{};
    for (int q = 0; q < segment.count; ++q) {
      const Path path = p.paths[segment.first + q];
      FloatPair dot[T::TM][T::TN]{};
      load_stage<T>(shared, 0, 0, m0, n0, logical_m, segment, path, p, input);
      cute::cp_async_fence();
      for (int64_t k0 = 0; k0 < p.K; k0 += T::K) {
        const int stage = int(k0 / T::K) & 1;
        cute::cp_async_wait<0>();
        __syncthreads();
        if (k0 + T::K < p.K) {
          load_stage<T>(shared, stage ^ 1, k0 + T::K, m0, n0,
                        logical_m, segment, path, p, input);
          cute::cp_async_fence();
        }
        auto a = cute::make_tensor(cute::make_smem_ptr(shared.a[stage]),
            cute::make_layout(cute::make_shape(cute::Int<T::M>{}, cute::Int<T::K>{}),
                              cute::make_stride(cute::Int<T::K + 1>{}, cute::Int<1>{})));
        auto w = cute::make_tensor(cute::make_smem_ptr(shared.w[stage]),
            cute::make_layout(cute::make_shape(cute::Int<T::N>{}, cute::Int<T::K>{}),
                              cute::make_stride(cute::Int<1>{}, cute::Int<T::N>{})));
#pragma unroll
        for (int k = 0; k < T::K; ++k) {
          float av[T::TM], wv[T::TN];
#pragma unroll
          for (int m = 0; m < T::TM; ++m) av[m] = a(lane_m + m * T::ThreadsM, k);
#pragma unroll
          for (int n = 0; n < T::TN; ++n) wv[n] = w(lane_n + n * T::ThreadsN, k);
#pragma unroll
          for (int m = 0; m < T::TM; ++m)
#pragma unroll
            for (int n = 0; n < T::TN; ++n) dot[m][n].add(FloatPair::two_prod(av[m], wv[n]));
        }
        // All readers finish before the next prefetch reuses this stage.
        __syncthreads();
      }
#pragma unroll
      for (int m = 0; m < T::TM; ++m)
#pragma unroll
        for (int n = 0; n < T::TN; ++n)
          total[m][n].add(dot[m][n].multiply({path.coefficient.x, path.coefficient.y}));
    }
#pragma unroll
    for (int m = 0; m < T::TM; ++m) {
      const int64_t row = m0 + lane_m + m * T::ThreadsM;
#pragma unroll
      for (int n = 0; n < T::TN; ++n) {
        const int64_t col = n0 + lane_n + n * T::ThreadsN;
        if (row < logical_m && col < p.N) {
          const int64_t offset = ((row / segment.rows) * p.target_rows + segment.row +
                                   row % segment.rows) * int64_t(p.N) + col;
          // Empty segments arrive here with exactly zero and never load input.
          output[offset] = total[m][n].rounded();
        }
      }
    }
    __syncthreads();
  }
}

// Immutable, unscaled W in [P,K,N]. Dgrad reverses the true logical K/N; it
// never substitutes V for the dx channel count U. Only static W is padded.
__global__ void pack_weight(const float* input, float* output,
                            int U, int V, int64_t kp, int64_t np,
                            int64_t count, bool dgrad) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t n = i % np, k = (i / np) % kp, path = i / (np * kp);
    const int64_t u = dgrad ? n : k, v = dgrad ? k : n;
    output[i] = u < U && v < V ? input[(path * U + u) * V + v] : 0.f;
  }
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("eqlinear FP32 shape: ") + operation + ": " +
                             cudaGetErrorString(status));
}

int64_t checked_product(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b))
    throw std::invalid_argument("eqlinear FP32 shape: extent overflow");
  return a * b;
}

void validate_operand(int B, int rows, int channels) {
  checked_product(checked_product(checked_product(B, rows), channels), sizeof(float));
}

class Plan final : public ShapePlan {
 public:
  Plan(const EqLinearProblemC& problem, const void* weight, const double* cg,
       bool dgrad, cudaStream_t stream) {
    const ShapeGeometry geometry(problem, cg);
    if (!weight) throw std::invalid_argument("eqlinear FP32 shape: null weight");
    validate_operand(geometry.B, geometry.I, geometry.U);
    validate_operand(geometry.B, geometry.J, geometry.V);
    p_.B = geometry.B;
    p_.source_rows = dgrad ? geometry.J : geometry.I;
    p_.target_rows = dgrad ? geometry.I : geometry.J;
    p_.K = dgrad ? geometry.V : geometry.U;
    p_.N = dgrad ? geometry.U : geometry.V;
    p_.packed_k = (int64_t(p_.K) + 15) / 16 * 16;
    p_.packed_n = (int64_t(p_.N) + 31) / 32 * 32;
    const int64_t count = checked_product(checked_product(geometry.P, p_.packed_k), p_.packed_n);
    const int64_t bytes = checked_product(count, sizeof(float));

    std::vector<int> boundaries{0, p_.target_rows};
    for (const ShapePath& path : geometry.paths) {
      const int row = dgrad ? path.input_row : path.output_row;
      boundaries.push_back(row);
      boundaries.push_back(row + path.rows);
    }
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    for (size_t s = 1; s < boundaries.size(); ++s) {
      Segment segment{boundaries[s - 1], boundaries[s] - boundaries[s - 1],
                      static_cast<int>(paths_.size()), 0, 0};
      for (const ShapePath& path : geometry.paths) {
        const int target = dgrad ? path.input_row : path.output_row;
        if (segment.row < target || segment.row >= target + path.rows) continue;
        const int source = dgrad ? path.output_row : path.input_row;
        const float hi = static_cast<float>(path.coefficient);
        const float lo = static_cast<float>(path.coefficient - static_cast<double>(hi));
        paths_.push_back({source + segment.row - target, path.weight_path, {hi, lo}});
        ++segment.count;
      }
      segments_.push_back(segment);
    }
    if (dgrad && p_.K == 1) configure<OuterTile>(&launch_outer);
    else if (p_.N == 1) configure<ScalarTile>();
    else if (p_.N <= 4) configure<ThinTile>();
    else if (p_.N <= 16) configure<NarrowTile>();
    else configure<WideTile>();

    if (weights_.reset(static_cast<size_t>(bytes)) ||
        device_paths_.reset(paths_.size() * sizeof(Path)) ||
        device_segments_.reset(segments_.size() * sizeof(Segment)))
      throw std::runtime_error("eqlinear FP32 shape: allocation failed");
    p_.weights = weights_.as<float>();
    p_.paths = device_paths_.as<Path>();
    p_.segments = device_segments_.as<Segment>();
    check_cuda(cudaMemcpyAsync(device_paths_.as<Path>(), paths_.data(), paths_.size() * sizeof(Path),
                                cudaMemcpyHostToDevice, stream), "copy paths");
    check_cuda(cudaMemcpyAsync(device_segments_.as<Segment>(), segments_.data(),
                                segments_.size() * sizeof(Segment), cudaMemcpyHostToDevice, stream),
               "copy segments");
    const unsigned blocks = static_cast<unsigned>(std::min<int64_t>((count + 255) / 256, 65535));
    pack_weight<<<blocks, 256, 0, stream>>>(static_cast<const float*>(weight), weights_.as<float>(),
                                          geometry.U, geometry.V, p_.packed_k, p_.packed_n,
                                          count, dgrad);
    check_cuda(cudaGetLastError(), "pack weight");
    // Creation owns its copies and static pack completely on return. Later runs
    // may use another stream; the parent supplies the plan destruction fence.
    check_cuda(cudaStreamSynchronize(stream), "prepare");
  }

  int run(const void* input, void* output, cudaStream_t stream) override {
    if (!input || !output) return static_cast<int>(cudaErrorInvalidValue);
    return launch_(p_, static_cast<const float*>(input), static_cast<float*>(output), stream);
  }

 private:
  using Launch = int (*)(Parameters, const float*, float*, cudaStream_t);

  template <class T>
  static int launch(Parameters p, const float* input, float* output, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(p.tiles, 65535));
    fused_kernel<T><<<blocks, T::Threads, 0, stream>>>(p, input, output);
    return static_cast<int>(cudaGetLastError());
  }

  static int launch_outer(Parameters p, const float* input, float* output, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(p.tiles, 65535));
    outer_product_kernel<<<blocks, OuterTile::Threads, 0, stream>>>(p, input, output);
    return static_cast<int>(cudaGetLastError());
  }

  template <class T>
  void configure(Launch launch_function = &launch<T>) {
    const int64_t nt = (int64_t(p_.N) + T::N - 1) / T::N;
    for (Segment& segment : segments_) {
      const int64_t mt = (int64_t(p_.B) * segment.rows + T::M - 1) / T::M;
      const int64_t count = checked_product(mt, nt);
      if (p_.tiles > std::numeric_limits<int64_t>::max() - count)
        throw std::invalid_argument("eqlinear FP32 shape: tile count overflow");
      p_.tiles += count;
      segment.tile_end = p_.tiles;
    }
    p_.segments_count = static_cast<int>(segments_.size());
    launch_ = launch_function;
  }

  Parameters p_{};
  Launch launch_ = nullptr;
  std::vector<Path> paths_;
  std::vector<Segment> segments_;
  // Declared last so failed construction fences uploads before host metadata dies.
  DeviceBuffer weights_, device_paths_, device_segments_;
};

}  // namespace

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                const double* cg, bool dgrad, cudaStream_t stream) {
  return std::make_unique<Plan>(p, w, cg, dgrad, stream);
}

}  // namespace fasteq::eqlinear::fp32_shape
