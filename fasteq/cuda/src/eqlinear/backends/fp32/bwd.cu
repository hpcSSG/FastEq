/***************************************************************************************************
 * fasteq eqlinear backends/fp32/bwd.cu
 *
 * Backward glue: group table -> problem table -> prep -> one fused grouped GEMM -> reduce.
 *
 * The two families of problems differ only in (M,N,K), the operands and where C lands, so kB3
 * concatenates them into one row table (wgrad's longer-K problems first, dgrad's shorter ones
 * last -- shortest tail).  kB1/kB2 build only one family and are the A/B baselines.
 **************************************************************************************************/
#include "bwd.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "device_buffer.h"
#include "eqlinear_group_table.h"
#include "prep_bwd.h"
#include "reduce_dw.h"
#include "simt_gemm.cuh"
#include "tiny_linear.cuh"

namespace eqlin_fp32 {
namespace {

/// Split-K chain length is the accuracy lever (B2_REPORT.md §2).  M8's ladder measured, per batch
/// size, the longest chain whose dw error stays inside the 1.05x torch gate; the worst ratio over
/// the ladder points is in parentheses:
///   B <= 512: 6 K-tiles (0.89)   B <= 2944: 12 (0.97)   B >= 5888: 18 (0.88)
/// Longer chains are faster (fewer partials, longer K per problem) but fail the gate below that.
int default_chain_tiles(int B) {
  if (B <= 512) return 6;
  if (B <= 2944) return 12;
  return 18;
}

/// The plan's STAGES must not exceed the shortest split.  Mirror the plan's default choice
/// (N > 128 or B > 1024 -> 3, else 2) and honour the same EQLIN_FP32_STAGES override.
int stages_env_or(int def) {
  char const *env = nullptr;
  int const s = env != nullptr ? atoi(env) : 0;
  return (s == 2 || s == 3 || s == 4) ? s : def;
}

int plan_stages_hint(int N, int B) {
  char const *cfg = nullptr;
  bool const hybrid = cfg != nullptr && strcmp(cfg, "hybrid") == 0;
  int const def = (hybrid && N > 128 && B >= 2048) ? 4
                                                   : ((N > 128 || B > 1024) ? 3 : 2);
  return stages_env_or(def);
}

/// Chain-length ladder knob for the accuracy/performance trade (M8): `EQLIN_BWD_CHAIN=<k-tiles>`.
/// It is clamped up to the pipeline depth so a split can never be shorter than the mainloop.
int chain_tiles_from_env(int def) {
  char const *env = nullptr;
  int const c = env != nullptr ? atoi(env) : 0;
  return c > 0 ? c : def;
}

void require_alloc(DeviceBuffer &b, size_t bytes, char const *what) {
  if (b.reset(bytes) != 0) throw std::runtime_error(std::string("backward: ") + what);
}

}  // namespace

struct BackwardRunner::Impl_ {
  int B, I, U, P, V, J;
  BwdImpl impl;
  GroupTable table;
  CgVals cg{};
  std::vector<double> coefficients;
  int max_rows = 0;
  int max_splits = 0;

  std::vector<BwdProblem> problems;
  std::vector<int> h_split_off, h_splits;
  std::vector<TransposeDesc> h_xdesc, h_ydesc;
  std::vector<long long> x_slab, y_slab;

  DeviceBuffer d_cgw, d_xt, d_yt, d_partials;
  DeviceBuffer d_xdesc, d_ydesc, d_split_off, d_splits;
  std::unique_ptr<BwdPlan> plan;

  Impl_(int B_, int I_, int U_, int P_, int V_, std::vector<int> const &i_dims, BwdImpl bimpl,
        float const *w, double const *cgd, cudaStream_t stream)
      : B(B_), I(I_), U(U_), P(P_), V(V_), impl(bimpl), table(build_group_table(i_dims)) {
    J = table.out_total_i;
    if (B <= 0 || I <= 0 || U <= 0 || V <= 0) throw std::invalid_argument("backward: bad B/I/U/V");
    if (P <= 0 || P > kMaxPaths) throw std::invalid_argument("backward: P out of range");
    if (table.num_paths() != P) throw std::invalid_argument("backward: P != len(i_dims)");
    if (I != table.in_prefix[P - 1] + table.i_dims[P - 1]) {
      throw std::invalid_argument("backward: I != sum(i_dims)");
    }
    if (impl != BwdImpl::kB1 && impl != BwdImpl::kB2 && impl != BwdImpl::kB3) {
      throw std::invalid_argument("backward: unknown BwdImpl");
    }
    if (U % 16 || V % 16) throw std::invalid_argument("backward: U and V must be multiples of 16");
    if (U != V) throw std::invalid_argument("backward: U != V is not supported yet");

    coefficients.resize(P);
    for (int p = 0; p < P; ++p) {
      coefficients[p] = cgd ? cgd[p] : 1.;
      cg.v[p] = static_cast<float>(coefficients[p]);
    }
    for (int p = 0; p < P; ++p) max_rows = std::max(max_rows, table.i_dims[p]);

    bool const has_dgrad = impl != BwdImpl::kB2;
    bool const has_wgrad = impl != BwdImpl::kB1;

    if (has_dgrad) {
      require_alloc(d_cgw, static_cast<size_t>(P) * U * V * sizeof(float), "cgW allocation");
    }
    if (has_wgrad) {
      build_wgrad(stream);
    }
    if (has_dgrad) {
      build_dgrad();
    }
    if (problems.empty()) throw std::runtime_error("backward: empty problem table");
    plan = make_simt_bwd_plan(V, B, problems, d_cgw.as<float>());
    if (has_dgrad) {
      if (plan->transposes_weight()) {
        if (launch_tiny_weight_transpose(w, d_cgw.as<float>(), P, U, V, stream) != 0)
          throw std::runtime_error("backward: tiny weight transpose failed");
        return;
      }
      CgVals weight_scales = cg;
      if (plan->scales_after_dot())
        for (int p = 0; p < P; ++p) weight_scales.v[p] = 1.f;
      launch_pack_cgw(w, d_cgw.as<float>(), weight_scales, P, U, V, stream);
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("backward: cgW pack launch failed");
    }
  }

  int group_of_path(int p) const {
    for (int g = 0; g < table.num_groups(); ++g) {
      int const s = table.path_istart[g];
      if (p >= s && p < s + table.path_count[g]) return g;
    }
    throw std::logic_error("backward: path not in any group");
  }

  void build_wgrad(cudaStream_t stream) {
    int const G = table.num_groups();
    h_xdesc.resize(P);
    h_ydesc.resize(G);
    x_slab.assign(P, 0);
    y_slab.assign(G, 0);

    long long xoff = 0;
    for (int p = 0; p < P; ++p) {
      int const i = table.i_dims[p];
      h_xdesc[p] = TransposeDesc{table.in_prefix[p] * U, i, U, xoff, B * i};
      x_slab[p] = xoff;
      xoff += static_cast<long long>(U) * B * i;
    }
    long long yoff = 0;
    for (int g = 0; g < G; ++g) {
      int const i = table.out_i_dims[g];
      h_ydesc[g] = TransposeDesc{table.out_prefix[g] * V, i, V, yoff, B * i};
      y_slab[g] = yoff;
      yoff += static_cast<long long>(V) * B * i;
    }

    require_alloc(d_xt, static_cast<size_t>(xoff) * sizeof(float), "Xt allocation");
    require_alloc(d_yt, static_cast<size_t>(yoff) * sizeof(float), "Yt allocation");
    require_alloc(d_xdesc, h_xdesc.size() * sizeof(TransposeDesc), "x descriptor allocation");
    require_alloc(d_ydesc, h_ydesc.size() * sizeof(TransposeDesc), "y descriptor allocation");
    cudaMemcpy(d_xdesc.as<TransposeDesc>(), h_xdesc.data(), h_xdesc.size() * sizeof(TransposeDesc),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ydesc.as<TransposeDesc>(), h_ydesc.data(), h_ydesc.size() * sizeof(TransposeDesc),
               cudaMemcpyHostToDevice);

    int const min_slice_tiles = plan_stages_hint(V, B);
    int const chain_tiles =
        std::max(chain_tiles_from_env(default_chain_tiles(B)), min_slice_tiles);
    h_split_off.assign(P, 0);
    h_splits.assign(P, 0);
    int prob = 0;
    for (int p = 0; p < P; ++p) {
      int const i = table.i_dims[p];
      int const K = B * i;
      if (K % 16) throw std::invalid_argument("backward: wgrad K must be a multiple of 16");
      int const nk_total = K / 16;
      int const want = (nk_total + chain_tiles - 1) / chain_tiles;
      int const cap = std::max(1, nk_total / min_slice_tiles);
      int const S = std::max(1, std::min(want, cap));
      h_split_off[p] = prob;
      h_splits[p] = S;
      max_splits = std::max(max_splits, S);

      int const g = group_of_path(p);
      int const base = nk_total / S;
      int const rem = nk_total % S;
      int k_tile = 0;
      for (int s = 0; s < S; ++s) {
        int const nk_s = base + (s < rem ? 1 : 0);
        BwdProblem q;
        q.kind = 1;
        q.M = U;
        q.N = V;
        q.K = nk_s * 16;
        q.b = d_yt.as<float>() + y_slab[g] + static_cast<long long>(k_tile) * 16;
        q.a_off = x_slab[p] + static_cast<long long>(k_tile) * 16;
        q.c_off = static_cast<long long>(prob) * U * V;
        q.lda = K;
        q.ldb = K;
        q.ldc = V;
        problems.push_back(q);
        ++prob;
        k_tile += nk_s;
      }
    }
    require_alloc(d_partials, static_cast<size_t>(prob) * U * V * sizeof(float),
                  "partial slab allocation");
    require_alloc(d_split_off, static_cast<size_t>(P) * sizeof(int), "split_off allocation");
    require_alloc(d_splits, static_cast<size_t>(P) * sizeof(int), "splits allocation");
    cudaMemcpy(d_split_off.as<int>(), h_split_off.data(), P * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_splits.as<int>(), h_splits.data(), P * sizeof(int), cudaMemcpyHostToDevice);
  }

  void build_dgrad() {
    long long const lda = static_cast<long long>(J) * V;
    long long const ldc = static_cast<long long>(I) * U;
    for (int g = 0; g < table.num_groups(); ++g) {
      int const i = table.out_i_dims[g];
      int const m_paths = table.path_count[g];
      int const r0 = table.row_start[g];
      int const p0 = table.path_istart[g];
      int const j0 = table.out_prefix[g];
      for (int t = 0; t < i; ++t) {
        for (int m = 0; m < m_paths; ++m) {
          int const p = p0 + m;
          BwdProblem q;
          q.kind = 0;
          q.M = B;
          q.N = U;
          q.K = V;
          q.b = d_cgw.as<float>() + static_cast<long long>(p) * U * V;
          q.a_off = static_cast<long long>(j0 + t) * V;
          q.c_off = static_cast<long long>(r0 + t + m * i) * U;
          q.lda = static_cast<int>(lda);
          q.ldb = V;
          q.ldc = static_cast<int>(ldc);
          q.scale = coefficients[p];
          q.weight_path = p;
          problems.push_back(q);
        }
      }
    }
  }

  int run(float const *x, float const *dy, float *dx, float *dw, cudaStream_t stream) {
    bool const has_dgrad = impl != BwdImpl::kB2;
    bool const has_wgrad = impl != BwdImpl::kB1;
    if (has_wgrad) {
      launch_transpose_x(x, d_xt.as<float>(), d_xdesc.as<TransposeDesc>(), P, B, I * U, max_rows, U,
                         stream);
      launch_transpose_dy(dy, d_yt.as<float>(), d_ydesc.as<TransposeDesc>(), table.num_groups(), B,
                          J * V, max_rows, V, stream);
      if (cudaGetLastError() != cudaSuccess) return 1;  // a failed prep must not reach the GEMM
    }
    char const *prep_only = nullptr;
    if (has_wgrad && prep_only != nullptr && atoi(prep_only) != 0) return 0;

    int const rc = plan->launch(has_dgrad ? dy : nullptr, has_dgrad ? dx : nullptr,
                                has_wgrad ? d_xt.as<float>() : nullptr,
                                has_wgrad ? d_partials.as<float>() : nullptr, stream);
    if (rc != 0) return rc;
    if (has_wgrad) {
      launch_reduce_dw(d_partials.as<float>(), d_split_off.as<int>(), d_splits.as<int>(), cg, dw,
                       P, U, V, stream);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
  }
};

BackwardRunner::BackwardRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims,
                               BwdImpl impl, float const *w, double const *cg, cudaStream_t stream)
    : p_(new Impl_(B, I, U, P, V, i_dims, impl, w, cg, stream)) {}

BackwardRunner::~BackwardRunner() = default;

int BackwardRunner::j_rows() const { return p_->J; }

int BackwardRunner::num_paths() const { return p_->P; }

int BackwardRunner::u_cols() const { return p_->U; }

int BackwardRunner::v_cols() const { return p_->V; }

int BackwardRunner::splits() const { return p_->max_splits; }

int BackwardRunner::run(float const *x, float const *dy, float *dx, float *dw,
                        cudaStream_t stream) {
  return p_->run(x, dy, dx, dw, stream);
}

}  // namespace eqlin_fp32
