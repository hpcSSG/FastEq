#include "bwd.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "device_buffer.h"
#include "direct_gemm_f64.cuh"
#include "eqlinear_group_table.h"
#include "grouped_gemm_f64.cuh"
#include "row_plan.h"
#include "sm90_fp64_grouped_gemm.cuh"

namespace eqlin {
namespace {

__global__ void scale_weight(double const *__restrict__ w, double *__restrict__ packed,
                             double const *__restrict__ cg, int U, int V, int P) {
  long long const total = static_cast<long long>(P) * U * V;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    int const p = static_cast<int>(i / (static_cast<long long>(U) * V));
    packed[i] = w[i] * cg[p];
  }
}

__global__ void reduce_split_dw(double const *__restrict__ partial, double *__restrict__ dw,
                                long long elements, int splits) {
  long long const stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < elements; idx += stride) {
    double sum = 0.0;
    for (int s = 0; s < splits; ++s)
      sum += partial[static_cast<long long>(s) * elements + idx];
    dw[idx] = sum;
  }
}

__global__ void transpose_slice(double const *__restrict__ src, double *__restrict__ dst,
                                int base_row, int rows, int dim, int B, int width) {
  __shared__ double tile[32][33];
  int const u = blockIdx.x * 32 + threadIdx.x;
  int const k0 = blockIdx.y * 32;
  for (int j = 0; j < 4; ++j) {
    int const k = k0 + threadIdx.y + j * 8;
    if (u < dim && k < rows) {
      int const b = k / width, t = k - b * width;
      tile[threadIdx.y + j * 8][threadIdx.x] =
          src[(static_cast<long long>(b) * B + base_row + t) * dim + u];
    }
  }
  __syncthreads();
  int const out_k = k0 + threadIdx.x;
  for (int j = 0; j < 4; ++j) {
    int const out_u = blockIdx.x * 32 + threadIdx.y + j * 8;
    if (out_u < dim && out_k < rows)
      dst[static_cast<long long>(out_u) * rows + out_k] =
          tile[threadIdx.x][threadIdx.y + j * 8];
  }
}

__global__ void transpose_paths(double const *__restrict__ src, double *__restrict__ dst,
                                double const *__restrict__ scales, int base_row, int rows,
                                int dim, int B, int width) {
  __shared__ double tile[32][33];
  int const path = blockIdx.z;
  int const u = blockIdx.x * 32 + threadIdx.x;
  int const k0 = blockIdx.y * 32;
  int const path_row = base_row + path * width;
  double const scale = scales[path];
  double *path_dst = dst + static_cast<long long>(path) * dim * rows;
  for (int j = 0; j < 4; ++j) {
    int const k = k0 + threadIdx.y + j * 8;
    if (u < dim && k < rows) {
      int const b = k / width, t = k - b * width;
      tile[threadIdx.y + j * 8][threadIdx.x] =
          src[(static_cast<long long>(b) * B + path_row + t) * dim + u] * scale;
    }
  }
  __syncthreads();
  int const out_k = k0 + threadIdx.x;
  for (int j = 0; j < 4; ++j) {
    int const out_u = blockIdx.x * 32 + threadIdx.y + j * 8;
    if (out_u < dim && out_k < rows)
      path_dst[static_cast<long long>(out_u) * rows + out_k] =
          tile[threadIdx.x][threadIdx.y + j * 8];
  }
}

__global__ void transpose_group(double const *__restrict__ x, double *__restrict__ xt,
                               double const *__restrict__ scales, int x_base_row,
                               double const *__restrict__ dy, double *__restrict__ dyt,
                               int dy_base_row, int rows, int U, int V, int x_total_rows,
                               int dy_total_rows, int width, int paths) {
  __shared__ double tile[32][33];
  int const path = blockIdx.z;
  int const col = blockIdx.x * 32 + threadIdx.x;
  int const k0 = blockIdx.y * 32;
  int const path_row = x_base_row + path * width;
  double *path_xt = xt + static_cast<long long>(path) * U * rows;
  double const scale = scales[path];

  for (int j = 0; j < 4; ++j) {
    int const k = k0 + threadIdx.y + j * 8;
    if (col < U && k < rows) {
      int const b = k / width, t = k - b * width;
      tile[threadIdx.y + j * 8][threadIdx.x] =
          x[(static_cast<long long>(b) * x_total_rows + path_row + t) * U + col] * scale;
    }
  }
  __syncthreads();
  int const out_k = k0 + threadIdx.x;
  for (int j = 0; j < 4; ++j) {
    int const out_u = blockIdx.x * 32 + threadIdx.y + j * 8;
    if (out_u < U && out_k < rows)
      path_xt[static_cast<long long>(out_u) * rows + out_k] =
          tile[threadIdx.x][threadIdx.y + j * 8];
  }
  __syncthreads();

  if (path == 0) {
    for (int j = 0; j < 4; ++j) {
      int const k = k0 + threadIdx.y + j * 8;
      if (col < V && k < rows) {
        int const b = k / width, t = k - b * width;
        tile[threadIdx.y + j * 8][threadIdx.x] =
            dy[(static_cast<long long>(b) * dy_total_rows + dy_base_row + t) * V + col];
      }
    }
    __syncthreads();
    for (int j = 0; j < 4; ++j) {
      int const out_v = blockIdx.x * 32 + threadIdx.y + j * 8;
      if (out_v < V && out_k < rows)
        dyt[static_cast<long long>(out_v) * rows + out_k] =
            tile[threadIdx.x][threadIdx.y + j * 8];
    }
  }
}

struct TransposePathMeta {
  long long x_offset;
  long long y_offset;
  int input_row;
  int output_row;
  int width;
  int first_path;
};

struct TransposeTask {
  int path;
  int tile_x;
  int tile_y;
};

__device__ __forceinline__ void transpose_group_tile(
    double const *__restrict__ x, double *__restrict__ xt,
    double const *__restrict__ dy, double *__restrict__ dyt,
    TransposePathMeta const &path, double (&tile)[32][33], int B, int I, int U, int V, int J,
    int rows, int col0, int k0, double scale) {
  int const col = col0 + threadIdx.x;
  for (int j = 0; j < 4; ++j) {
    int const k = k0 + threadIdx.y + j * 8;
    if (col < U && k < rows) {
      int const b = k / path.width;
      int const t = k - b * path.width;
      tile[threadIdx.y + j * 8][threadIdx.x] =
          x[(static_cast<long long>(b) * I + path.input_row + t) * U + col] * scale;
    }
  }
  __syncthreads();
  int const out_k = k0 + threadIdx.x;
  for (int j = 0; j < 4; ++j) {
    int const out_u = col0 + threadIdx.y + j * 8;
    if (out_u < U && out_k < rows)
      xt[path.x_offset + static_cast<long long>(out_u) * rows + out_k] =
          tile[threadIdx.x][threadIdx.y + j * 8];
  }
  __syncthreads();

  if (path.first_path) {
    for (int j = 0; j < 4; ++j) {
      int const k = k0 + threadIdx.y + j * 8;
      if (col < V && k < rows) {
        int const b = k / path.width;
        int const t = k - b * path.width;
        tile[threadIdx.y + j * 8][threadIdx.x] =
            dy[(static_cast<long long>(b) * J + path.output_row + t) * V + col];
      }
    }
    __syncthreads();
    for (int j = 0; j < 4; ++j) {
      int const out_v = col0 + threadIdx.y + j * 8;
      if (out_v < V && out_k < rows)
        dyt[path.y_offset + static_cast<long long>(out_v) * rows + out_k] =
            tile[threadIdx.x][threadIdx.y + j * 8];
    }
  }
}

template <int XGroup, int KGroup>
__global__ void transpose_all_groups(double const *__restrict__ x, double *__restrict__ xt,
                                     double const *__restrict__ scales,
                                     double const *__restrict__ dy, double *__restrict__ dyt,
                                     TransposePathMeta const *__restrict__ paths,
                                     TransposeTask const *__restrict__ tasks, int task_count,
                                     int B, int I, int U, int V, int J) {
  __shared__ double tile[32][33];
  int const task_index = blockIdx.x;
  if (task_index >= task_count) return;
  TransposeTask const task = tasks[task_index];
  TransposePathMeta const path = paths[task.path];
  int const rows = B * path.width;
  double const scale = scales[task.path];

#pragma unroll
  for (int x_group_idx = 0; x_group_idx < XGroup; ++x_group_idx) {
    int const col0 = (task.tile_x + x_group_idx) * 32;
#pragma unroll
    for (int k_group_idx = 0; k_group_idx < KGroup; ++k_group_idx) {
      int const k0 = (task.tile_y + k_group_idx) * 32;
      if constexpr (KGroup == 1) {
        transpose_group_tile(x, xt, dy, dyt, path, tile, B, I, U, V, J, rows, col0, k0, scale);
      } else if (k0 < rows) {
        transpose_group_tile(x, xt, dy, dyt, path, tile, B, I, U, V, J, rows, col0, k0, scale);
      }
    }
  }
}

void launch_transpose(double const *src, double *dst, int base_row, int rows, int dim, int B,
                      int width, cudaStream_t stream) {
  dim3 const block(32, 8);
  dim3 const grid((dim + 31) / 32, (rows + 31) / 32);
  transpose_slice<<<grid, block, 0, stream>>>(src, dst, base_row, rows, dim, B, width);
}

void launch_transpose_paths(double const *src, double *dst, double const *scales, int base_row,
                            int rows, int dim, int B, int width, int paths,
                            cudaStream_t stream) {
  dim3 const block(32, 8);
  dim3 const grid((dim + 31) / 32, (rows + 31) / 32, paths);
  transpose_paths<<<grid, block, 0, stream>>>(src, dst, scales, base_row, rows, dim, B, width);
}

void launch_transpose_group(double const *x, double *xt, double const *scales, int x_base_row,
                            double const *dy, double *dyt, int dy_base_row, int rows, int U,
                            int V, int x_total_rows, int dy_total_rows, int width, int paths,
                            cudaStream_t stream) {
  dim3 const block(32, 8);
  dim3 const grid((std::max(U, V) + 31) / 32, (rows + 31) / 32, paths);
  transpose_group<<<grid, block, 0, stream>>>(x, xt, scales, x_base_row, dy, dyt, dy_base_row,
                                              rows, U, V, x_total_rows, dy_total_rows, width,
                                              paths);
}

template <int XGroup, int KGroup>
void launch_transpose_all_groups(double const *x, double *xt, double const *scales,
                                 double const *dy, double *dyt,
                                 TransposePathMeta const *path_meta,
                                 TransposeTask const *tasks, int task_count,
                                 int B, int I, int U, int V, int J, cudaStream_t stream) {
  transpose_all_groups<XGroup, KGroup><<<task_count, dim3(32, 8), 0, stream>>>(
      x, xt, scales, dy, dyt, path_meta, tasks, task_count, B, I, U, V, J);
}

Gemm2xConfig wgrad_tile_config(int tile) {
  switch (tile) {
    case 0: return Gemm2xConfig::kBaseline;
    case 1: return Gemm2xConfig::kM128N32S3;
    case 2: return Gemm2xConfig::kM128N32S4;
    case 3: return Gemm2xConfig::kM64N32S4;
    case 4: return Gemm2xConfig::kM64N64S3;
    case 6: return Gemm2xConfig::kM32N128S3;
    case 7: return Gemm2xConfig::kM32N128S4;
    case 8: return Gemm2xConfig::kM32N32S3;
    case 9: return Gemm2xConfig::kM128N128S3;
    case 10: return Gemm2xConfig::kM128N64S3;
    case 11: return Gemm2xConfig::kM64N128S3;
    case 12: return Gemm2xConfig::kM32N64S3;
    case 13: return Gemm2xConfig::kM128N64W32S3;
    case 14: return Gemm2xConfig::kHostSchedule;
    default: throw std::invalid_argument("WgradRunner: unsupported tile id");
  }
}

}  // namespace

struct DgradRunner::Impl {
  int B, I, U, V, J;
  std::vector<RowPlan> rows;
  DeviceBuffer d_cg, d_weight;
  std::unique_ptr<GroupedGemmPlan> plan;

  Impl(int b, int i, int u, int p, int v, std::vector<int> const &dims, double const *w,
       std::vector<double> const &cg, cudaStream_t stream)
      : B(b), I(i), U(u), V(v), J(build_group_table(dims).out_total_i) {
    GroupTable const table = build_group_table(dims);
    if (b <= 0 || u <= 0 || v <= 0 || p != table.num_paths() ||
        i != table.in_prefix.back() + table.i_dims.back() || w == nullptr ||
        (!cg.empty() && static_cast<int>(cg.size()) != p))
      throw std::invalid_argument("DgradRunner: invalid shape or inputs");
    if (d_cg.reset(static_cast<size_t>(p) * sizeof(double)) ||
        d_weight.reset(static_cast<size_t>(p) * u * v * sizeof(double)))
      throw std::runtime_error("DgradRunner: allocation failed");
    std::vector<double> scales(p, 1.0);
    if (!cg.empty()) scales = cg;
    cudaMemcpyAsync(d_cg.as<double>(), scales.data(), p * sizeof(double), cudaMemcpyHostToDevice,
                    stream);
    int const blocks = std::min(4096, (p * u * v + 255) / 256);
    scale_weight<<<blocks, 256, 0, stream>>>(w, d_weight.as<double>(), d_cg.as<double>(), u, v, p);

    rows.reserve(I);
    for (int g = 0; g < table.num_groups(); ++g) {
      int const ig = table.out_i_dims[g];
      for (int m = 0; m < table.path_count[g]; ++m) {
        int const path = table.path_istart[g] + m;
        for (int t = 0; t < ig; ++t) {
          int const out_row = table.out_prefix[g] + t;
          int const in_row = table.row_start[g] + m * ig + t;
          RowPlan r{};
          r.K = V;
          r.a_off = static_cast<long long>(out_row) * V;
          r.b_off = static_cast<long long>(path) * U * V;
          r.lda = static_cast<long long>(J) * V;
          r.ldb = V;
          r.ldc = static_cast<long long>(I) * U;
          r.c_off = static_cast<long long>(in_row) * U;
          rows.push_back(r);
        }
      }
    }
    plan = std::make_unique<GroupedGemmPlan>(B, U, I, static_cast<long long>(J) * V, V,
                                             static_cast<long long>(I) * U, rows, nullptr,
                                             d_weight.as<double>());
  }
};

DgradRunner::DgradRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims,
                         double const *w, std::vector<double> const &cg, cudaStream_t stream)
    : p_(std::make_unique<Impl>(B, I, U, P, V, i_dims, w, cg, stream)) {}
DgradRunner::~DgradRunner() = default;

int DgradRunner::run(double const *dy, double *dx, cudaStream_t stream) {
  p_->plan->bind_a(dy, stream);
  return p_->plan->launch(dx, stream);
}

struct WgradRunner::Impl {
  int B, I, U, P, V;
  int split_k;
  GroupTable table;
  std::vector<RowPlan> rows;
  std::vector<long long> x_offsets, y_offsets;
  DeviceBuffer d_xt, d_yt, d_cg, d_partial;
  DeviceBuffer d_transpose_paths, d_transpose_tasks;
  int transpose_task_count = 0;
  std::unique_ptr<GroupedGemmPlan> plan;
  std::unique_ptr<DirectGemmPlan> direct_plan;
  std::unique_ptr<Sm90Fp64Plan> sm90_plan;

  Impl(int b, int i, int u, int p, int v, std::vector<int> const &dims,
       std::vector<double> const &scales, int tile, int split, int kernel, int transpose_mode)
      : B(b), I(i), U(u), P(p), V(v), split_k(split), table(build_group_table(dims)) {
    if (b <= 0 || u <= 0 || v <= 0 || p != table.num_paths() ||
        i != table.in_prefix.back() + table.i_dims.back() ||
        (!scales.empty() && static_cast<int>(scales.size()) != p) || split <= 0 || split > b ||
        (kernel < 0 || kernel > 3) || (transpose_mode < 0 || transpose_mode > 7))
      throw std::invalid_argument("WgradRunner: invalid shape or inputs");
    int const selected_transpose_mode =
        transpose_mode == 3 ? ((P == 10 && U == 128 && V == 128) ? 2 : 0) : transpose_mode;
    rows.reserve(static_cast<size_t>(P) * split_k);
    x_offsets.resize(P);
    y_offsets.resize(table.num_groups());
    long long xo = 0, yo = 0;
    for (int g = 0; g < table.num_groups(); ++g) {
      int const ig = table.out_i_dims[g], K = B * ig;
      y_offsets[g] = yo;
      yo += static_cast<long long>(V) * K;
      int const chunk = (K + split_k - 1) / split_k;
      for (int m = 0; m < table.path_count[g]; ++m) {
        int const path = table.path_istart[g] + m;
        x_offsets[path] = xo;
        for (int s = 0; s < split_k; ++s) {
          int const k0 = s * chunk;
          int const ks = std::min(chunk, K - k0);
          RowPlan r{};
          r.K = ks;
          r.a_off = xo + k0;
          r.b_off = y_offsets[g] + k0;
          r.lda = K;
          r.ldb = K;
          r.ldc = V;
          r.c_off = (static_cast<long long>(s) * P + path) * U * V;
          rows.push_back(r);
        }
        xo += static_cast<long long>(U) * K;
      }
    }
    char const *row_order = nullptr;
    bool const sort_rows = row_order == nullptr
                               ? (P == 10 && U == 128 && V == 128)
                               : (std::strcmp(row_order, "descending_k") == 0 ||
                                  std::strcmp(row_order, "ascending_k") == 0);
    if (sort_rows) {
      bool const ascending = row_order != nullptr &&
                             std::strcmp(row_order, "ascending_k") == 0;
      std::stable_sort(rows.begin(), rows.end(),
                       [ascending](RowPlan const &lhs, RowPlan const &rhs) {
                         return ascending ? lhs.K < rhs.K : lhs.K > rhs.K;
                       });
    }
    if (d_xt.reset(static_cast<size_t>(B) * I * U * sizeof(double)) ||
        d_yt.reset(static_cast<size_t>(B) * table.out_total_i * V * sizeof(double)) ||
        d_cg.reset(static_cast<size_t>(P) * sizeof(double)) ||
        (split_k > 1 && d_partial.reset(static_cast<size_t>(split_k) * P * U * V * sizeof(double))))
      throw std::runtime_error("WgradRunner: allocation failed");
    if (selected_transpose_mode == 2 || selected_transpose_mode == 4 ||
        selected_transpose_mode == 5 || selected_transpose_mode == 6 ||
        selected_transpose_mode == 7) {
      std::vector<TransposePathMeta> host_paths(P);
      std::vector<TransposeTask> host_tasks;
      int const tile_x_count = (std::max(U, V) + 31) / 32;
      for (int g = 0; g < table.num_groups(); ++g) {
        int const width = table.out_i_dims[g];
        int const rows = B * width;
        int const tile_y_count = (rows + 31) / 32;
        int const x_group = selected_transpose_mode == 6 || selected_transpose_mode == 7 ? 2 : 1;
        int const k_group = selected_transpose_mode == 5 || selected_transpose_mode == 7 ? 2 : 1;
        for (int step = 0; step < table.path_count[g]; ++step) {
          int const m = selected_transpose_mode == 4 ? table.path_count[g] - 1 - step : step;
          int const path = table.path_istart[g] + m;
          host_paths[path] = {x_offsets[path], y_offsets[g], table.in_prefix[path],
                              table.out_prefix[g], width, m == 0};
          for (int tile_y = 0; tile_y < tile_y_count; tile_y += k_group)
            for (int tile_x = 0; tile_x < tile_x_count; tile_x += x_group)
              host_tasks.push_back({path, tile_x, tile_y});
        }
      }
      transpose_task_count = static_cast<int>(host_tasks.size());
      if (d_transpose_paths.reset(host_paths.size() * sizeof(TransposePathMeta)) ||
          d_transpose_tasks.reset(host_tasks.size() * sizeof(TransposeTask)))
        throw std::runtime_error("WgradRunner: transpose task allocation failed");
      cudaMemcpy(d_transpose_paths.as<TransposePathMeta>(), host_paths.data(),
                 host_paths.size() * sizeof(TransposePathMeta), cudaMemcpyHostToDevice);
      cudaMemcpy(d_transpose_tasks.as<TransposeTask>(), host_tasks.data(),
                 host_tasks.size() * sizeof(TransposeTask), cudaMemcpyHostToDevice);
    }
    std::vector<double> host_cg(P, 1.0);
    if (!scales.empty()) host_cg = scales;
    cudaMemcpy(d_cg.as<double>(), host_cg.data(), P * sizeof(double), cudaMemcpyHostToDevice);
    if (kernel == 0) {
      plan = std::make_unique<GroupedGemmPlan>(U, V, rows.size(), 1, 1, V, rows,
                                               d_xt.as<double>(), d_yt.as<double>(),
                                               wgrad_tile_config(tile));
    } else if (kernel == 3) {
      if (tile != 0) throw std::invalid_argument("WgradRunner: SM90 kernel requires tile=0");
      sm90_plan = std::make_unique<Sm90Fp64Plan>(U, V, rows.size(), 1, 1, V, rows,
                                                 d_xt.as<double>(), d_yt.as<double>());
    } else {
      if (tile != 0) throw std::invalid_argument("WgradRunner: direct kernel requires tile=0");
      direct_plan = std::make_unique<DirectGemmPlan>(U, V, rows.size(), 1, 1, V, rows,
                                                     d_xt.as<double>(), d_yt.as<double>(), 0,
                                                     kernel == 1);
    }
    kernel_ = kernel;
    transpose_mode_ = selected_transpose_mode;
  }

  int kernel_ = 0;
  int transpose_mode_ = 0;
};

WgradRunner::WgradRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims,
                         std::vector<double> const &cg, int tile, int split_k, int kernel,
                         int transpose_mode)
    : p_(std::make_unique<Impl>(B, I, U, P, V, i_dims, cg, tile, split_k, kernel,
                                transpose_mode)) {}
WgradRunner::~WgradRunner() = default;

int WgradRunner::run(double const *x, double const *dy, double *dw, cudaStream_t stream) {
  Impl &s = *p_;
  if (s.transpose_mode_ == 2 || s.transpose_mode_ == 4 || s.transpose_mode_ == 5 ||
      s.transpose_mode_ == 6 || s.transpose_mode_ == 7) {
    if (s.transpose_mode_ == 5)
      launch_transpose_all_groups<1, 2>(x, s.d_xt.as<double>(), s.d_cg.as<double>(), dy,
          s.d_yt.as<double>(), s.d_transpose_paths.as<TransposePathMeta>(),
          s.d_transpose_tasks.as<TransposeTask>(), s.transpose_task_count,
          s.B, s.I, s.U, s.V, s.table.out_total_i, stream);
    else if (s.transpose_mode_ == 6)
      launch_transpose_all_groups<2, 1>(x, s.d_xt.as<double>(), s.d_cg.as<double>(), dy,
          s.d_yt.as<double>(), s.d_transpose_paths.as<TransposePathMeta>(),
          s.d_transpose_tasks.as<TransposeTask>(), s.transpose_task_count,
          s.B, s.I, s.U, s.V, s.table.out_total_i, stream);
    else if (s.transpose_mode_ == 7)
      launch_transpose_all_groups<2, 2>(x, s.d_xt.as<double>(), s.d_cg.as<double>(), dy,
          s.d_yt.as<double>(), s.d_transpose_paths.as<TransposePathMeta>(),
          s.d_transpose_tasks.as<TransposeTask>(), s.transpose_task_count,
          s.B, s.I, s.U, s.V, s.table.out_total_i, stream);
    else
      launch_transpose_all_groups<1, 1>(x, s.d_xt.as<double>(), s.d_cg.as<double>(), dy,
          s.d_yt.as<double>(), s.d_transpose_paths.as<TransposePathMeta>(),
          s.d_transpose_tasks.as<TransposeTask>(), s.transpose_task_count,
          s.B, s.I, s.U, s.V, s.table.out_total_i, stream);
  } else {
    for (int g = 0; g < s.table.num_groups(); ++g) {
      int const ig = s.table.out_i_dims[g], K = s.B * ig;
      if (s.transpose_mode_ == 1) {
        launch_transpose_group(x, s.d_xt.as<double>() + s.x_offsets[s.table.path_istart[g]],
                               s.d_cg.as<double>() + s.table.path_istart[g],
                               s.table.row_start[g], dy, s.d_yt.as<double>() + s.y_offsets[g],
                               s.table.out_prefix[g], K, s.U, s.V, s.I,
                               s.table.out_total_i, ig,
                               s.table.path_count[g], stream);
      } else {
        launch_transpose_paths(x, s.d_xt.as<double>() + s.x_offsets[s.table.path_istart[g]],
                               s.d_cg.as<double>() + s.table.path_istart[g],
                               s.table.row_start[g], K, s.U, s.I, ig,
                               s.table.path_count[g], stream);
        launch_transpose(dy, s.d_yt.as<double>() + s.y_offsets[g], s.table.out_prefix[g],
                         K, s.V, s.table.out_total_i, ig, stream);
      }
    }
  }
  auto launch = [&](double *out) {
    if (s.kernel_ == 0) return s.plan->launch(out, stream);
    if (s.kernel_ == 3) return s.sm90_plan->launch(out, stream);
    return s.direct_plan->launch(out, stream);
  };
  if (s.split_k == 1) return launch(dw);
  int rc = launch(s.d_partial.as<double>());
  if (rc != 0) return rc;
  long long const elements = static_cast<long long>(s.P) * s.U * s.V;
  int const blocks = std::min(4096, static_cast<int>((elements + 255) / 256));
  reduce_split_dw<<<blocks, 256, 0, stream>>>(s.d_partial.as<double>(), dw, elements, s.split_k);
  return 0;
}

struct BackwardRunner::Impl {
  DgradRunner dgrad;
  WgradRunner wgrad;

  Impl(int B, int I, int U, int P, int V, std::vector<int> const &i_dims,
       double const *w, std::vector<double> const &cg, cudaStream_t stream,
       int wgrad_tile, int wgrad_split_k, int wgrad_kernel, int wgrad_transpose_mode)
      : dgrad(B, I, U, P, V, i_dims, w, cg, stream),
        wgrad(B, I, U, P, V, i_dims, cg, wgrad_tile, wgrad_split_k, wgrad_kernel,
              wgrad_transpose_mode) {}
};

BackwardRunner::BackwardRunner(int B, int I, int U, int P, int V,
                               std::vector<int> const &i_dims, double const *w,
                               std::vector<double> const &cg, cudaStream_t stream,
                               int wgrad_tile, int wgrad_split_k, int wgrad_kernel,
                               int wgrad_transpose_mode)
    : p_(std::make_unique<Impl>(B, I, U, P, V, i_dims, w, cg, stream, wgrad_tile,
                                wgrad_split_k, wgrad_kernel, wgrad_transpose_mode)) {}
BackwardRunner::~BackwardRunner() = default;

int BackwardRunner::run(double const *x, double const *dy, double *dx, double *dw,
                        cudaStream_t stream) {
  Impl &s = *p_;
  int const rc = s.dgrad.run(dy, dx, stream);
  return rc == 0 ? s.wgrad.run(x, dy, dw, stream) : rc;
}

}  // namespace eqlin
