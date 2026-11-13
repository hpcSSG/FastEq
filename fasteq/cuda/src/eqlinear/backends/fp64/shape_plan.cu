#include "shape_plan.cuh"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include <cute/arch/mma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/tensor.hpp>

#include "device_buffer.h"

namespace fasteq::eqlinear::fp64_shape {
namespace {

using detail::RowPath;
constexpr int kThreads = 128, kTileM = 16, kTileN = 32, kTileK = 16;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("eqlinear fp64 shape: ") + operation + ": " +
                             cudaGetErrorString(status));
}

int blocks_for(int64_t count, int elements_per_block) {
  return static_cast<int>(std::min<int64_t>(65535, (count - 1) / elements_per_block + 1));
}

size_t tensor_bytes(int a, int b, int c) {
  size_t count = static_cast<size_t>(a);
  for (int extent : {b, c}) {
    if (count > size_t(std::numeric_limits<int64_t>::max()) / sizeof(double) / extent)
      throw std::invalid_argument("eqlinear fp64 shape: operand size overflow");
    count *= extent;
  }
  return count * sizeof(double);
}

// Both plans own a private immutable weight snapshot. Forward copies [P,U,V];
// dgrad prepares [P,V,U], so all steady-state loads use the same K,N layout.
__global__ void transpose_weight(const double* __restrict__ w, double* __restrict__ packed,
                                 int U, int V, int64_t count) {
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += int64_t(gridDim.x) * blockDim.x) {
    int u = static_cast<int>(index % U);
    int v = static_cast<int>((index / U) % V);
    int64_t p = index / (int64_t(U) * V);
    packed[index] = w[(p * U + u) * V + v];
  }
}

// Four independent outputs per thread. This is real FP64 CUDA-core FMA;
// narrow-U/outer-product rooflines must use CUDA-core, not DMMA peak throughput.
struct FmaTile {
  __device__ static int row(int tid, int output) { return (tid + output * kThreads) / kTileN; }
  __device__ static int col(int tid, int output) { return (tid + output * kThreads) % kTileN; }

  template <class A, class W>
  __device__ static void multiply(const A& a, const W& w, double (&dot)[4]) {
    #pragma unroll
    for (int k = 0; k < kTileK; ++k) {
      #pragma unroll
      for (int o = 0; o < 4; ++o)
        cute::UniversalFMA<double>::fma(dot[o], a(row(threadIdx.x, o), k),
                                        w(k, col(threadIdx.x, o)), dot[o]);
    }
  }
};

// 2x2 warps, each owning two 8x8 tiles. The register mapping follows PTX's
// m8n8k4.f64 fragment layout; CuTe emits the actual FP64 DMMA instruction.
struct DmmaTile {
  __device__ static int row(int tid, int) { return (tid / 64) * 8 + (tid % 32) / 4; }
  __device__ static int col(int tid, int o) {
    return ((tid / 32) % 2) * 16 + (o / 2) * 8 + (tid % 4) * 2 + o % 2;
  }

  template <class A, class W>
  __device__ static void multiply(const A& a, const W& w, double (&dot)[4]) {
    int tid = threadIdx.x;
    #pragma unroll
    for (int k = 0; k < kTileK; k += 4) {
      double av = a(row(tid, 0), k + tid % 4);
      #pragma unroll
      for (int n = 0; n < 2; ++n) {
        double bv = w(k + tid % 4, ((tid / 32) % 2) * 16 + n * 8 + (tid % 32) / 4);
        cute::SM80_8x8x4_F64F64F64F64_TN::fma(
            dot[2 * n], dot[2 * n + 1], av, bv, dot[2 * n], dot[2 * n + 1]);
      }
    }
  }
};

struct Contract {
  int batch, source_rows, target_rows, reduction, columns;
  const int64_t* offsets;
  const RowPath* paths;
  const double* weight;
};

// A single CTA owns each (logical row, B tile, N tile), including unused rows.
// All path dots are reduced here: no atomics, activation scratch, or crop pass.
template <class Math>
__global__ void tiled_contract(Contract p, const double* __restrict__ input,
                                double* __restrict__ output, int64_t tiles) {
  __shared__ double a_storage[kTileM * kTileK];
  __shared__ double w_storage[kTileK * kTileN];
  auto a = cute::make_tensor(cute::make_smem_ptr(a_storage),
      cute::make_layout(cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileK>{}),
                        cute::make_stride(cute::Int<kTileK>{}, cute::Int<1>{})));
  auto w = cute::make_tensor(cute::make_smem_ptr(w_storage),
      cute::make_layout(cute::make_shape(cute::Int<kTileK>{}, cute::Int<kTileN>{}),
                        cute::make_stride(cute::Int<kTileN>{}, cute::Int<1>{})));
  int64_t nt = (int64_t(p.columns) + kTileN - 1) / kTileN;
  int64_t mt = (int64_t(p.batch) + kTileM - 1) / kTileM;
  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    int64_t n0 = tile % nt * kTileN;
    int64_t b0 = (tile / nt) % mt * kTileM;
    int row = static_cast<int>(tile / nt / mt);
    double result[4] = {};
    for (int64_t q = p.offsets[row]; q < p.offsets[row + 1]; ++q) {
      RowPath path = p.paths[q];
      double dot[4] = {};
      for (int64_t k0 = 0; k0 < p.reduction; k0 += kTileK) {
        for (int i = threadIdx.x; i < kTileM * kTileK; i += kThreads) {
          int m = i / kTileK, k = i % kTileK;
          // Predicate the raw logical load, then zero-fill only shared memory.
          a(m, k) = b0 + m < p.batch && k0 + k < p.reduction
              ? input[((b0 + m) * p.source_rows + path.source_row) * p.reduction + k0 + k]
              : 0.0;
        }
        for (int i = threadIdx.x; i < kTileK * kTileN; i += kThreads) {
          int k = i / kTileN, n = i % kTileN;
          w(k, n) = k0 + k < p.reduction && n0 + n < p.columns
              ? p.weight[(int64_t(path.weight_path) * p.reduction + k0 + k) * p.columns + n0 + n]
              : 0.0;
        }
        __syncthreads();
        Math::multiply(a, w, dot);
        __syncthreads();
      }
      #pragma unroll
      for (int o = 0; o < 4; ++o)
        cute::UniversalFMA<double>::fma(result[o], dot[o], path.coefficient, result[o]);
    }
    #pragma unroll
    for (int o = 0; o < 4; ++o) {
      int64_t b = b0 + Math::row(threadIdx.x, o), n = n0 + Math::col(threadIdx.x, o);
      if (b < p.batch && n < p.columns)
        output[(b * p.target_rows + row) * p.columns + n] = result[o];
    }
  }
}

// Scalar readouts coalesce the contraction across each warp rather than
// spending a 32-column GEMM tile on V=1. Each lane still uses FP64 CuTe FMA.
__global__ void scalar_readout(Contract p, const double* __restrict__ input,
                                double* __restrict__ output, int64_t rows) {
  int lane = threadIdx.x % 32;
  for (int64_t task = int64_t(blockIdx.x) * (kThreads / 32) + threadIdx.x / 32;
       task < rows; task += int64_t(gridDim.x) * (kThreads / 32)) {
    int row = static_cast<int>(task % p.target_rows);
    int64_t batch = task / p.target_rows;
    double result = 0;
    for (int64_t q = p.offsets[row]; q < p.offsets[row + 1]; ++q) {
      RowPath path = p.paths[q];
      double dot = 0;
      for (int64_t k = lane; k < p.reduction; k += 32)
        cute::UniversalFMA<double>::fma(dot,
            input[(batch * p.source_rows + path.source_row) * p.reduction + k],
            p.weight[int64_t(path.weight_path) * p.reduction + k], dot);
      #pragma unroll
      for (int delta = 16; delta; delta /= 2) dot += __shfl_down_sync(0xffffffff, dot, delta);
      cute::UniversalFMA<double>::fma(result, dot, path.coefficient, result);
    }
    if (!lane) output[task] = result;
  }
}

// V=1 dgrad is a fused outer product, coalesced in logical U. Avoid shared
// staging and a mostly empty K tile when the contraction contains one element.
__global__ void scalar_outer(Contract p, const double* __restrict__ input,
                              double* __restrict__ output, int64_t count) {
  for (int64_t index = int64_t(blockIdx.x) * kThreads + threadIdx.x; index < count;
       index += int64_t(gridDim.x) * kThreads) {
    int n = static_cast<int>(index % p.columns);
    int row = static_cast<int>((index / p.columns) % p.target_rows);
    int64_t batch = index / p.columns / p.target_rows;
    double result = 0;
    for (int64_t q = p.offsets[row]; q < p.offsets[row + 1]; ++q) {
      RowPath path = p.paths[q];
      double dot = input[batch * p.source_rows + path.source_row] *
                   p.weight[int64_t(path.weight_path) * p.columns + n];
      cute::UniversalFMA<double>::fma(result, dot, path.coefficient, result);
    }
    output[index] = result;
  }
}

class Plan final : public ShapePlan {
 public:
  Plan(const EqLinearProblemC& p, const void* weight, const double* cg, bool dgrad,
       cudaStream_t stream)
      : geometry_(p, cg), rows_(geometry_, dgrad) {
    if (!weight) throw std::invalid_argument("eqlinear fp64 shape: null weight");
    // Check all products before using signed 64-bit offsets in device code.
    tensor_bytes(p.B, p.I, p.U);
    tensor_bytes(p.B, p.J, p.V);
    size_t bytes = tensor_bytes(p.P, p.U, p.V);
    if (weight_.reset(bytes) ||
        offsets_.reset(rows_.offsets().size() * sizeof(int64_t)) ||
        paths_.reset(rows_.paths().size() * sizeof(RowPath)))
      throw std::runtime_error("eqlinear fp64 shape: allocation failed");
    check(cudaMemcpyAsync(offsets_.as<int64_t>(), rows_.offsets().data(),
                           rows_.offsets().size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream),
          "copy row offsets");
    check(cudaMemcpyAsync(paths_.as<RowPath>(), rows_.paths().data(),
                           rows_.paths().size() * sizeof(RowPath), cudaMemcpyHostToDevice, stream),
          "copy paths");
    if (dgrad) {
      int64_t count = static_cast<int64_t>(bytes / sizeof(double));
      transpose_weight<<<blocks_for(count, 256), 256, 0, stream>>>(
          static_cast<const double*>(weight), weight_.as<double>(), p.U, p.V, count);
      check(cudaGetLastError(), "prepare transposed weight");
    } else {
      check(cudaMemcpyAsync(weight_.as<double>(), weight, bytes, cudaMemcpyDeviceToDevice, stream),
            "copy weight");
    }
    contract_ = {p.B, dgrad ? p.J : p.I, dgrad ? p.I : p.J,
                 dgrad ? p.V : p.U, dgrad ? p.U : p.V,
                 offsets_.as<int64_t>(), paths_.as<RowPath>(), weight_.as<double>()};
  }

  int run(const void* input, void* output, cudaStream_t stream) override {
    if (!input || !output) return static_cast<int>(cudaErrorInvalidValue);
    auto x = static_cast<const double*>(input);
    auto out = static_cast<double*>(output);
    const auto& p = contract_;
    if (p.columns == 1) {
      int64_t count = int64_t(p.batch) * p.target_rows;
      scalar_readout<<<blocks_for(count, kThreads / 32), kThreads, 0, stream>>>(p, x, out, count);
    } else if (p.reduction == 1) {
      int64_t count = int64_t(p.batch) * p.target_rows * p.columns;
      scalar_outer<<<blocks_for(count, kThreads), kThreads, 0, stream>>>(p, x, out, count);
    } else {
      int64_t tiles = int64_t(p.target_rows) * ((int64_t(p.batch) + kTileM - 1) / kTileM) *
                      ((int64_t(p.columns) + kTileN - 1) / kTileN);
      int blocks = blocks_for(tiles, 1);
      if (std::min(geometry_.U, geometry_.V) <= kTileK)
        tiled_contract<FmaTile><<<blocks, kThreads, 0, stream>>>(p, x, out, tiles);
      else
        tiled_contract<DmmaTile><<<blocks, kThreads, 0, stream>>>(p, x, out, tiles);
    }
    return static_cast<int>(cudaGetLastError());
  }

 private:
  // Host upload sources live until the parent-owned lifetime fence completes.
  ShapeGeometry geometry_;
  detail::RowTable rows_;
  eqlin::DeviceBuffer weight_, offsets_, paths_;
  Contract contract_{};
};

}  // namespace

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                const double* cg, bool dgrad, cudaStream_t stream) {
  return std::make_unique<Plan>(p, w, cg, dgrad, stream);
}

}  // namespace fasteq::eqlinear::fp64_shape
