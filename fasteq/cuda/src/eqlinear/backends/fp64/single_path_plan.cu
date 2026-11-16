#include "single_path_plan.cuh"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#include <cute/arch/mma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>

#include "device_buffer.h"
#include "shape_plan.cuh"

namespace fasteq::eqlinear::fp64_single {
namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("eqlinear fp64 single: ") + operation + ": " +
                             cudaGetErrorString(status));
}

size_t tensor_bytes(int a, int b, int c) {
  size_t count = size_t(a);
  for (int extent : {b, c}) {
    if (count > size_t(std::numeric_limits<int64_t>::max()) / sizeof(double) / extent)
      throw std::invalid_argument("eqlinear fp64 single: operand size overflow");
    count *= extent;
  }
  return count * sizeof(double);
}

int blocks_for(int64_t count, int tile = 1) {
  return int(std::min<int64_t>(65535, (count - 1) / tile + 1));
}

// Transpose only at plan construction. Coefficients stay out of the snapshot:
// both engines apply the full double coefficient AFTER the FP64 dot product.
__global__ void transpose_weight(const double* input, double* output, int rows, int cols,
                                 int64_t count) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(gridDim.x) * blockDim.x)
    output[(i % cols) * rows + i / cols] = input[i];
}

struct Contract {
  int batch, target_rows, source_row, target_row, rows, reduction, columns;
  int64_t source_stride, destination_stride;
  double coefficient;
  const double* weight;
};

constexpr int kContiguousRowsPerWarp = 8;

// Complete I=J=1 contraction only. The two N tiles are explicit grid planes;
// all logical strides and N bounds are constants, with no row/path divmod.
template <int RowsPerWarp>
__global__ void contiguous_k10_n224(const double* __restrict__ input,
                                   const double* __restrict__ weight,
                                   double* __restrict__ output, int batch,
                                   double coefficient) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int64_t b = int64_t(blockIdx.x) * (4 * RowsPerWarp) + warp * RowsPerWarp;
  int n0 = int(blockIdx.y & 1) * 128 + lane * 2;
  double a[RowsPerWarp] = {}, dot[RowsPerWarp][2][2] = {};
  #pragma unroll
  for (int m = 0; m < RowsPerWarp; ++m)
    if (b + m < batch && lane < 10) a[m] = input[(b + m) * 10 + lane];
  #pragma unroll
  for (int k = 0; k < 10; ++k) {
    double x[RowsPerWarp];
    #pragma unroll
    for (int m = 0; m < RowsPerWarp; ++m)
      x[m] = __shfl_sync(0xffffffff, a[m], k);
    #pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      int n = n0 + pair * 64;
      double2 w = n < 224
          ? *reinterpret_cast<const double2*>(weight + k * 224 + n)
          : make_double2(0.0, 0.0);
      #pragma unroll
      for (int m = 0; m < RowsPerWarp; ++m) {
        cute::UniversalFMA<double>::fma(dot[m][pair][0], x[m], w.x, dot[m][pair][0]);
        cute::UniversalFMA<double>::fma(dot[m][pair][1], x[m], w.y, dot[m][pair][1]);
      }
    }
  }
  #pragma unroll
  for (int m = 0; m < RowsPerWarp; ++m) {
    #pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      int n = n0 + pair * 64;
      if (b + m < batch && n < 224)
        *reinterpret_cast<double2*>(output + (b + m) * 224 + n) =
            make_double2(dot[m][pair][0] * coefficient, dot[m][pair][1] * coefficient);
    }
  }
}

// Native m8n8k4 consumes four K-contiguous doubles per column. Zero padding is
// private to this construction-time snapshot; activation loads remain masked.
__global__ void prepare_short_weight(const double* input, double* output, int V,
                                      bool dgrad, int K, int padded_k, int64_t count) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(gridDim.x) * blockDim.x) {
    int k = int(i % padded_k);
    int64_t n = i / padded_k;
    output[i] = k < K ? input[dgrad ? n * V + k : int64_t(k) * V + n] : 0.0;
  }
}

// Each lane owns two adjacent columns in four batch rows. A 32-column tile
// uses two independent half-warps, reusing each weight across eight batch rows.
// Aligned pairs use one 128-bit memory operation instead of two strided scalars.
// Input elements are loaded once per lane group and broadcast from registers.
// No shared memory, CSR traversal, padded K work, or block barrier is required.
template <int K, int TileN = 64, bool Vectorized = false, int Pairs = 1>
__global__ void short_fma_contract(Contract p, const double* __restrict__ input,
                               double* __restrict__ output, int64_t tiles) {
  static_assert(TileN == 32 || TileN == 64);
  static_assert(Vectorized || (TileN == 64 && Pairs == 1));
  constexpr int kGroup = TileN / 2, kRowsPerWarp = 128 / kGroup;
  constexpr int kTileM = kRowsPerWarp * 4, kTileN = TileN * Pairs;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int64_t nt = (int64_t(p.columns) + kTileN - 1) / kTileN;
  int64_t mt = (int64_t(p.batch) + kTileM - 1) / kTileM;
  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    int64_t n0 = tile % nt * kTileN + (lane % kGroup) * 2;
    int64_t b = (tile / nt) % mt * kTileM + warp * kRowsPerWarp + (lane / kGroup) * 4;
    int row = int(tile / nt / mt);
    bool active = row >= p.target_row && int64_t(row) < int64_t(p.target_row) + p.rows;
    double a[4] = {}, dot[4][Pairs][2] = {};
    if (active) {
      #pragma unroll
      for (int m = 0; m < 4; ++m)
        if (b + m < p.batch && lane % kGroup < K)
          a[m] = input[(b + m) * p.source_stride +
                       (int64_t(p.source_row) + row - p.target_row) * K + lane % kGroup];
      #pragma unroll
      for (int k = 0; k < K; ++k) {
        if constexpr (Vectorized) {
          double x[4];
          #pragma unroll
          for (int m = 0; m < 4; ++m) x[m] = __shfl_sync(0xffffffff, a[m], k, kGroup);
          #pragma unroll
          for (int pair = 0; pair < Pairs; ++pair) {
            int64_t n = n0 + pair * TileN;
            double w0 = 0.0, w1 = 0.0;
            if (n + 1 < p.columns) {
              double2 weight = *reinterpret_cast<const double2*>(p.weight + int64_t(k) * p.columns + n);
              w0 = weight.x;
              w1 = weight.y;
            }
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
              cute::UniversalFMA<double>::fma(dot[m][pair][0], x[m], w0, dot[m][pair][0]);
              cute::UniversalFMA<double>::fma(dot[m][pair][1], x[m], w1, dot[m][pair][1]);
            }
          }
        } else {
          // Keep the original scalar instruction order for 8-byte outputs and
          // odd N. Hoisting all shuffles changed its register schedule adversely.
          double w0 = n0 < p.columns ? p.weight[int64_t(k) * p.columns + n0] : 0.0;
          double w1 = n0 + 1 < p.columns ? p.weight[int64_t(k) * p.columns + n0 + 1] : 0.0;
          #pragma unroll
          for (int m = 0; m < 4; ++m) {
            double x = __shfl_sync(0xffffffff, a[m], k);
            cute::UniversalFMA<double>::fma(dot[m][0][0], x, w0, dot[m][0][0]);
            cute::UniversalFMA<double>::fma(dot[m][0][1], x, w1, dot[m][0][1]);
          }
        }
      }
    }
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
      #pragma unroll
      for (int pair = 0; pair < Pairs; ++pair) {
        int64_t n = n0 + pair * TileN;
        if constexpr (Vectorized) {
          if (b + m < p.batch && n + 1 < p.columns) {
            double2 result = active ? make_double2(dot[m][pair][0] * p.coefficient,
                                                  dot[m][pair][1] * p.coefficient)
                                    : make_double2(0.0, 0.0);
            *reinterpret_cast<double2*>(output + (b + m) * p.destination_stride +
                                       int64_t(row) * p.columns + n) = result;
          }
        } else {
          #pragma unroll
          for (int c = 0; c < 2; ++c)
            if (b + m < p.batch && n + c < p.columns)
              output[(b + m) * p.destination_stride + int64_t(row) * p.columns + n + c] =
                  active ? dot[m][pair][c] * p.coefficient : 0.0;
        }
      }
    }
  }
}

// Four warps each own 8x32 outputs. The m8n8k4 register mapping loads one A
// double per lane and reuses it across four N atoms, with eight accumulators.
// This replaces K-times-per-output warp shuffles with true FP64 tensor math.
// No shared memory, CSR traversal or block barrier is required.
template <int K>
__global__ void short_dmma_contract(Contract p, const double* __restrict__ input,
                               double* __restrict__ output, int64_t tiles) {
  constexpr int kTileM = 32, kTileN = 32, kPaddedK = (K + 3) / 4 * 4;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int64_t nt = (int64_t(p.columns) + kTileN - 1) / kTileN;
  int64_t mt = (int64_t(p.batch) + kTileM - 1) / kTileM;
  for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    int64_t n0 = tile % nt * kTileN;
    int64_t b = (tile / nt) % mt * kTileM + warp * 8 + lane / 4;
    int row = int(tile / nt / mt);
    bool active = row >= p.target_row && int64_t(row) < int64_t(p.target_row) + p.rows;
    double dot[8] = {};
    if (active) {
      #pragma unroll
      for (int k0 = 0; k0 < K; k0 += 4) {
        int k = k0 + lane % 4;
        double a = b < p.batch && k < K
            ? input[b * p.source_stride +
                    (int64_t(p.source_row) + row - p.target_row) * K + k] : 0.0;
        #pragma unroll
        for (int atom = 0; atom < 4; ++atom) {
          int64_t n = n0 + atom * 8 + lane / 4;
          double w = n < p.columns ? p.weight[n * kPaddedK + k] : 0.0;
          cute::SM80_8x8x4_F64F64F64F64_TN::fma(
              dot[atom * 2], dot[atom * 2 + 1], a, w, dot[atom * 2], dot[atom * 2 + 1]);
        }
      }
    }
    #pragma unroll
    for (int atom = 0; atom < 4; ++atom) {
      #pragma unroll
      for (int c = 0; c < 2; ++c) {
        int64_t n = n0 + atom * 8 + lane % 4 * 2 + c;
        if (b < p.batch && n < p.columns)
          output[b * p.destination_stride + int64_t(row) * p.columns + n] =
              active ? dot[atom * 2 + c] * p.coefficient : 0.0;
      }
    }
  }
}

// K=1 is an outer product, with exact operand strides and fused zeroing of
// inactive destination rows. Never read the unused (possibly NaN) source rows.
__global__ void outer_contract(Contract p, const double* __restrict__ input,
                               double* __restrict__ output, int64_t count) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(gridDim.x) * blockDim.x) {
    int n = int(i % p.columns);
    int row = int(i / p.columns % p.target_rows);
    int64_t batch = i / p.destination_stride;
    double result = 0;
    if (row >= p.target_row && int64_t(row) < int64_t(p.target_row) + p.rows) {
      double dot = input[batch * p.source_stride + int64_t(p.source_row) + row - p.target_row] * p.weight[n];
      result = dot * p.coefficient;
    }
    output[i] = result;
  }
}

// Standard CUTLASS 64-bit shared iterators replace the bespoke TMA fetch_stage.
// A is row major; a construction-time transpose makes B column major. Scalar
// epilogue accesses also support pointers with only double (8-byte) alignment.
template <int N>
using DenseGemm = cutlass::gemm::device::Gemm<
    double, cutlass::layout::RowMajor, double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::RowMajor, double, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90, cutlass::gemm::GemmShape<64, N, 16>,
    cutlass::gemm::GemmShape<32, 16, 16>, cutlass::gemm::GemmShape<16, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<double, 1, double, double>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 1, 1>;

class DenseLaunch {
 public:
  virtual ~DenseLaunch() = default;
  virtual int run(const double* input, double* output, cudaStream_t stream) = 0;
};

template <int N>
class PreparedDense final : public DenseLaunch {
 public:
  explicit PreparedDense(const Contract& p)
      : arguments_({int(int64_t(p.batch) * p.rows), p.columns, p.reduction},
                   {nullptr, p.reduction}, {p.weight, p.reduction},
                   {nullptr, p.columns}, {nullptr, p.columns}, {p.coefficient, 0.0}) {
    static_assert(sizeof(typename DenseGemm<N>::GemmKernel::SharedStorage) < (48 << 10),
                  "dense run must not require per-launch CUDA attributes");
    if (gemm_.can_implement(arguments_) != cutlass::Status::kSuccess ||
        gemm_.initialize(arguments_) != cutlass::Status::kSuccess)
      throw std::runtime_error("eqlinear fp64 single: prepare dense GEMM failed");
  }

  int run(const double* input, double* output, cudaStream_t stream) override {
    // Protect mutable launch parameters if callers share a handle across host
    // threads. CUDA copies them at launch, before this lock is released.
    std::lock_guard<std::mutex> lock(mutex_);
    arguments_.ref_A.reset(input);
    arguments_.ref_C.reset(output);
    arguments_.ref_D.reset(output);
    auto status = gemm_.update(arguments_);
    if (status == cutlass::Status::kSuccess) status = gemm_.run(stream);
    return status == cutlass::Status::kSuccess ? int(cudaSuccess) : int(cudaErrorUnknown);
  }

 private:
  DenseGemm<N> gemm_;
  typename DenseGemm<N>::Arguments arguments_;
  std::mutex mutex_;
};

class Plan final : public ShapePlan {
 public:
  Plan(const EqLinearProblemC& p, const void* w, const double* cg, bool dgrad,
       bool dense, cudaStream_t stream)
      : geometry_(p, cg), dense_(dense),
        // Small grids favor the shorter DMMA instruction path. Larger batches
        // retain register FMA: standalone comparisons measure higher DMMA
        // latency as grids grow.
        masked_(!dense && (dgrad ? p.V : p.U) == 10 && p.B <= 512 &&
                (dgrad ? p.U : p.V) >= 32 && (dgrad ? p.U : p.V) <= 256) {
    if (!w) throw std::invalid_argument("eqlinear fp64 single: null weight");
    tensor_bytes(p.B, p.I, p.U);
    tensor_bytes(p.B, p.J, p.V);
    int k = dgrad ? p.V : p.U, n = dgrad ? p.U : p.V;
    int padded_k = masked_ ? (k + 3) / 4 * 4 : k;
    size_t bytes = tensor_bytes(1, n, padded_k);
    if (weight_.reset(bytes)) throw std::runtime_error("eqlinear fp64 single: allocation failed");
    const auto& path = geometry_.path;
    contract_ = {p.B, dgrad ? p.I : p.J, dgrad ? path.output_row : path.input_row,
                 dgrad ? path.input_row : path.output_row, path.rows,
                 dgrad ? p.V : p.U, dgrad ? p.U : p.V,
                 geometry_.source_stride(dgrad), geometry_.destination_stride(dgrad),
                 path.coefficient, weight_.as<double>()};
    contiguous_ = !dense_ && !masked_ && p.B >= 6312 && k == 10 && n == 224 && p.I == 1 && p.J == 1 &&
                  path.input_row == 0 && path.output_row == 0 && path.rows == 1;
    // Dense uses [N,K], masked DMMA padded [N,K], register FMA [K,N].
    if (masked_) {
      prepare_short_weight<<<blocks_for(int64_t(bytes / sizeof(double)), 256), 256, 0, stream>>>(
          static_cast<const double*>(w), weight_.as<double>(), p.V, dgrad, k, padded_k,
          int64_t(bytes / sizeof(double)));
      check(cudaGetLastError(), "prepare masked DMMA weight");
    } else if (dgrad != dense_) {
      transpose_weight<<<blocks_for(int64_t(bytes / sizeof(double)), 256), 256, 0, stream>>>(
          static_cast<const double*>(w), weight_.as<double>(), p.U, p.V,
          int64_t(bytes / sizeof(double)));
      check(cudaGetLastError(), "transpose weight");
    } else {
      check(cudaMemcpyAsync(weight_.as<double>(), w, bytes, cudaMemcpyDeviceToDevice, stream),
            "snapshot weight");
    }
    if (dense_) {
      if (n <= 16) dense_launch_ = std::make_unique<PreparedDense<16>>(contract_);
      else dense_launch_ = std::make_unique<PreparedDense<32>>(contract_);
    }
  }

  int run(const void* input, void* output, cudaStream_t stream) override {
    if (!input || !output) return int(cudaErrorInvalidValue);
    const double* x = static_cast<const double*>(input);
    double* y = static_cast<double*>(output);
    const auto& p = contract_;
    if (dense_) return dense_launch_->run(x, y, stream);
    if (contiguous_ && (reinterpret_cast<uintptr_t>(y) & 15) == 0) {
      constexpr int kTileM = 4 * kContiguousRowsPerWarp;
      dim3 grid(static_cast<unsigned>((int64_t(p.batch) + kTileM - 1) / kTileM), 2);
      contiguous_k10_n224<kContiguousRowsPerWarp><<<grid, 128, 0, stream>>>(
          x, p.weight, y, p.batch, p.coefficient);
      return int(cudaGetLastError());
    }
    if (p.reduction == 1) {
      int64_t count = int64_t(p.batch) * p.destination_stride;
      outer_contract<<<blocks_for(count, 256), 256, 0, stream>>>(p, x, y, count);
    } else if (masked_) {
      int64_t tiles = int64_t(p.target_rows) * ((int64_t(p.batch) + 31) / 32) *
                      ((int64_t(p.columns) + 31) / 32);
      short_dmma_contract<10><<<blocks_for(tiles), 128, 0, stream>>>(p, x, y, tiles);
    } else {
      // cudaMalloc aligns the private W snapshot. Even N also aligns every W
      // row and every logical destination row; input needs only double alignment.
      if (p.reduction == 10 && p.columns % 2 == 0 &&
          (reinterpret_cast<uintptr_t>(y) & 15) == 0) {
        if (p.columns >= 128) return run_short<10, 64, true, 2>(x, y, stream);
        if (p.columns % 64 == 32) return run_short<10, 32, true>(x, y, stream);
        return run_short<10, 64, true>(x, y, stream);
      }
      return p.reduction == 10 ? run_short<10>(x, y, stream) : run_short<16>(x, y, stream);
    }
    return int(cudaGetLastError());
  }

 private:
  template <int K, int N = 64, bool Vectorized = false, int Pairs = 1>
  int run_short(const double* x, double* y, cudaStream_t stream) {
    constexpr int kTileM = 1024 / N;
    const auto& p = contract_;
    constexpr int kTileN = N * Pairs;
    int64_t tiles = int64_t(p.target_rows) * ((int64_t(p.batch) + kTileM - 1) / kTileM) *
                    ((int64_t(p.columns) + kTileN - 1) / kTileN);
    short_fma_contract<K, N, Vectorized, Pairs><<<blocks_for(tiles), 128, 0, stream>>>(p, x, y, tiles);
    return int(cudaGetLastError());
  }

  SinglePathGeometry geometry_;
  bool dense_, masked_;
  bool contiguous_ = false;
  eqlin::DeviceBuffer weight_;
  Contract contract_{};
  std::unique_ptr<DenseLaunch> dense_launch_;
};

}  // namespace

std::unique_ptr<ShapePlan> create(const EqLinearProblemC& p, const void* w,
                                const double* cg, bool dgrad, cudaStream_t stream) {
  SinglePathGeometry geometry(p, cg);
  int k = dgrad ? p.V : p.U, n = dgrad ? p.U : p.V;
  bool short_k = k == 1 || k == 10 || k == 16;
  const auto& path = geometry.path;
  bool dense = !short_k && n > 1 && path.input_row == 0 && path.output_row == 0 &&
               path.rows == p.I && path.rows == p.J &&
               int64_t(p.B) * path.rows <= std::numeric_limits<int>::max();
  if (short_k || dense) return std::make_unique<Plan>(p, w, cg, dgrad, dense, stream);
  return fp64_shape::create(p, w, cg, dgrad, stream);
}

}  // namespace fasteq::eqlinear::fp64_single
