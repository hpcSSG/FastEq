/***************************************************************************************************
 * fasteq eqlinear backends/fp64/dgrad.cu
 *
 * Production B1 dgrad (dx) only.
 **************************************************************************************************/
#include "bwd.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "device_buffer.h"
#include "eqlinear_group_table.h"
#include "grouped_gemm_f64.cuh"
#include "row_plan.h"

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

}  // namespace eqlin
