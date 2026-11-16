/***************************************************************************************************
 * fasteq eqlinear backends/fp32/runner.cu
 *
 * Glue: group table -> host index tables -> device packing kernels -> grouped GEMM (+ reduction).
 *
 *   out[b, j, n] = sum_{p in P(j)} cg[p] * sum_k x[b, in_prefix[p] + t, k] * W[p, k, n]
 *
 * realised as a block-sparse grouped GEMM.  The packing is always the same:
 *
 *   A_prep[b, a_col + t*K0 + m*U + k] = x[b, row_start(g) + m*i_g + t, k]
 *   B_prep[n, b_col + m*U + k]       = cg[p0+m] * W[p0+m, k, n]
 *   lda = I*U, ldb = P*U
 *
 * --- Two problem tables (M4) ---
 *
 * Single-path groups (path_count == 1) keep one problem per output row with K = U; that is exactly
 * torch's own contraction length and the two agree to 1.00x.
 *
 * Multi-path groups used to be one problem per output row with K = path_count*U, which sums g*U
 * terms in a single fp32 accumulator.  Torch runs g GEMMs of K=U and adds the g results, so our
 * round-off grew like sqrt(g) (1.55x / 1.96x on the two multi-path configs).  With
 * EQLIN_PATH_SPLIT=1 each (output row, path) becomes its own K=U problem, writing a partial slab
 * `partials[B][R][V]` (R = I); `launch_reduce_partials` then sums the g partials in fp32 in path
 * order, reproducing torch's accumulation.
 *
 * The partial layout needs no kernel change: both plans write problem r's tile at `out + r*V` with
 * a plan-wide row stride, so passing `partials` as `out` and `ldc = R*V` produces [B][R][V].
 **************************************************************************************************/
#include "eqlinear.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "device_buffer.h"
#include "eqlinear_group_table.h"
#include "grouped_gemm_fp32.cuh"
#include "packed_index_tables.h"
#include "prep_kernels.h"
#include "reduce_partials.h"
#include "simt_gemm.cuh"
#include "tiny_linear.cuh"
#include "numeric_policy.h"
#include "row_plan.h"

namespace eqlin_fp32 {

using fasteq::eqlinear::PackedIndexTables;
using fasteq::eqlinear::SegmentSpec;

namespace {
constexpr int kAlign = 4;  // keeps every per-row base 16B-aligned; see the ctor comment
}  // namespace

struct Runner::Impl_ {
  int B, I, U, P, V, J;
  /// Number of GEMM problems: J, or I when every output row is split per path (M4).
  int num_rows = 0;
  bool split = false;
  /// M5: the split problems are chained inside one CTA instead of written to a partial slab.
  bool fused = false;
  Impl impl;
  GroupTable table;
  std::vector<RowPlan> rows;
  /// Zero-copy A (M6/P1): the same problems with `a_off` reinterpreted as x's element offset.
  /// Empty when the view does not apply (see setup_a_view), in which case `rows` is used and the
  /// activation side is packed into `d_a_prep` as before.
  std::vector<RowPlan> rows_view;
  bool a_view = false;
  long long lda = 0, ldb = 0, ldc = 0;

  std::vector<int> h_src_index;   // [I*U] A column -> element offset within one x batch
  std::vector<int> h_b_src_base;  // [P*U] B column -> base offset into w
  std::vector<int> h_b_p_of_col;  // [P*U] B column -> input path index
  std::vector<int> h_rbase;       // [J] first problem index of each output row (split mode)
  std::vector<int> h_pg;          // [J] number of paths feeding each output row (split mode)

  DeviceBuffer d_src_index, d_b_src_base, d_b_p_of_col;
  DeviceBuffer d_a_prep, d_b_prep;
  /// 16 B placeholder base handed to the plans when `a_view` is on.  The plans encode their TMA
  /// descriptor and size their shared memory at construction, when `x` is not known yet; this keeps
  /// that decision on the real (TMA) variant.  It stays allocated after the bind -- a dangling
  /// `d_a_` entry would be a far worse failure than 16 wasted bytes.
  DeviceBuffer d_a_stub;
  DeviceBuffer d_partials, d_rbase, d_pg;
  std::unique_ptr<GroupedGemmPlan> plan;
  /// Set when `fused`: the path-chained plan owns its own row table, and there is no `plan`.
  std::unique_ptr<GroupedGemmPlan::ImplBase> fused_plan;

  Impl_(int B_, int I_, int U_, int P_, int V_, std::vector<int> const &i_dims, Impl impl_)
      : B(B_), I(I_), U(U_), P(P_), V(V_), impl(impl_), table(build_group_table(i_dims)) {
    J = table.out_total_i;
    if (B <= 0 || U <= 0 || V <= 0) throw std::invalid_argument("Runner: bad B/U/V");
    if (P <= 0 || P > kMaxPaths) throw std::invalid_argument("Runner: P out of range");
    if (table.num_paths() != P) throw std::invalid_argument("Runner: P != len(i_dims)");
    if (I != table.in_prefix[P - 1] + table.i_dims[P - 1]) {
      throw std::invalid_argument("Runner: I != sum(i_dims)");
    }
    if (impl != Impl::kSimt && impl != Impl::kSimtCustom && impl != Impl::kAuto &&
        impl != Impl::kFused) {
      throw std::invalid_argument("Runner: unknown Impl");
    }

    // Split whenever any output row draws on more than one path.  The split's purpose is to match
    // torch's per-path accumulation; single-path rows already do.
    bool const multi_path = std::any_of(table.path_count.begin(), table.path_count.end(),
                                        [](int m) { return m > 1; });
    // ON by default: it is what makes the multi-path configs match torch's accuracy (M4).  The
    // environment switch exists only to A/B the cost (`EQLIN_PATH_SPLIT=0` restores the old
    // K-concatenated problem table).
    char const *env = nullptr;
    split = multi_path && !(env != nullptr && atoi(env) == 0);
    multi_path_ = (P > 1);
    char const *fenv = nullptr;
    bool const want_fused =
        impl == Impl::kFused && !(fenv != nullptr && atoi(fenv) == 0);
    if (split && (U % 16) != 0) {
      // The K-tile of both backends is 8/16; per-path problems have K = U.
      throw std::invalid_argument("Runner: path split needs U to be a multiple of 16");
    }
    num_rows = split ? I : J;

    lda = static_cast<long long>(I) * U;
    ldb = static_cast<long long>(P) * U;
    ldc = static_cast<long long>(num_rows) * V;
    // Not required by the SIMT mainloop (its operands are 4-byte scalar accesses), but kept so the
    // FP32 line accepts exactly the shapes the tf32 line accepts -- and so a later vectorized or
    // TMA-based mainloop can assume 16B-aligned row starts without changing the contract.
    if (lda % kAlign || ldb % kAlign || ldc % kAlign || V % kAlign) {
      throw std::invalid_argument("Runner: strides/V must be multiples of 4");
    }

    want_fused_ = want_fused;
    build_host_tables();
    setup_a_view();
    allocate();
  }

  void build_host_tables() {
    PackedIndexTables const packed(table, U, V, SegmentSpec{});
    h_src_index = packed.src_index();
    h_b_src_base = packed.b_src_base();
    h_b_p_of_col = packed.b_p_of_col();
    if (split) {
      h_rbase.assign(J, 0);
      h_pg.assign(J, 1);
    }

    rows.clear();
    int rrow = 0;
    for (auto const &block : packed.blocks()) {
      if (split) {
        // Rows of one output row share a contiguous run of m problem indices, in path order, so the
        // reduction can read [rbase, rbase+m).
        for (int t = 0; t < block.i; ++t) {
          int const j = block.j0 + t;
          h_rbase[j] = rrow;
          h_pg[j] = block.m;
          for (int mm = 0; mm < block.m; ++mm) {
            rows.push_back(RowPlan{U, block.a_base + static_cast<long long>(t) * block.K0 + mm * U,
                                   block.b_base + static_cast<long long>(mm) * U});
          }
          rrow += block.m;
        }
      } else {
        for (int t = 0; t < block.i; ++t) {
          rows.push_back(RowPlan{block.K0,
                                 block.a_base + static_cast<long long>(t) * block.K0,
                                 block.b_base});
        }
      }
    }

    if (packed.a_cols() != static_cast<long long>(I) * U ||
        packed.b_cols() != static_cast<long long>(P) * U) {
      throw std::runtime_error("Runner: packed column count mismatch");
    }
    if (static_cast<int>(rows.size()) != num_rows) {
      throw std::logic_error("Runner: row table size mismatch");
    }
    if (split && rrow != I) {
      // R == I is an identity: each path contributes i_g rows and the groups partition [0, I).
      throw std::logic_error("Runner: per-path row count != I");
    }
  }

  /// Decide whether the GEMM can read the caller's `x` in place.  The packing is a pure *column*
  /// permutation with the same row stride (`lda = I*U`), so the view is legal exactly when every
  /// problem's K is one contiguous, U-aligned run inside a single row of x.  That holds for every
  /// default configuration -- a split problem has K == U -- and fails only for the
  /// `EQLIN_PATH_SPLIT=0` K-concatenation table, which spans g rows and keeps packing.
  void setup_a_view() {
    char const *env = nullptr;
    a_view = !(env != nullptr && atoi(env) == 0);
    if (!a_view) return;
    rows_view.clear();
    rows_view.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
      RowPlan r = rows[i];
      if (r.K != U) {  // K spans more than one path's run
        a_view = false;
        break;
      }
      long long const base = r.a_off;
      int const src = h_src_index[static_cast<size_t>(base)];
      if (src % U != 0) {
        a_view = false;
        break;
      }
      // Assert the run is contiguous in x rather than assuming it: if the packing table ever
      // changes, this fails loudly instead of reading the wrong bytes.
      for (int k = 0; k < r.K; ++k) {
        if (h_src_index[static_cast<size_t>(base) + k] != src + k) {
          a_view = false;
          break;
        }
      }
      if (!a_view) break;
      r.a_off = src;
      rows_view.push_back(r);
    }
    if (!a_view) rows_view.clear();
  }

  bool want_fused_ = false;
  /// More than one input path feeds the operator; drives the strict-FP32 contraction policy. Note
  /// that descriptors can have P>1 while every output row still belongs to one path (`path_count`
  /// all 1); this flag follows the operator's path count, not the per-row grouping.
  bool multi_path_ = false;

  void allocate() {
    size_t const n_a = static_cast<size_t>(B) * I * U;
    size_t const n_b = static_cast<size_t>(V) * P * U;
    bool bad = d_b_src_base.reset(h_b_src_base.size() * sizeof(int)) ||
               d_b_p_of_col.reset(h_b_p_of_col.size() * sizeof(int)) ||
               d_b_prep.reset(n_b * sizeof(float));
    // The two A-side staging buffers exist only for the packed fallback; in view mode the only
    // A allocation is the 16 B placeholder.
    if (a_view) {
      bad = bad || d_a_stub.reset(16);
    } else {
      bad = bad || d_src_index.reset(h_src_index.size() * sizeof(int)) ||
            d_a_prep.reset(n_a * sizeof(float));
    }
    if (bad) throw std::runtime_error("Runner: device allocation failed");

    std::vector<RowPlan> const &plan_rows = a_view ? rows_view : rows;
    float const *plan_a = (a_view ? d_a_stub : d_a_prep).as<float>();

    // Fused (M5): chain the per-path problems in one CTA, so there is no slab and no reduction.
    // The plan declines when its coarser tile grid would not fill the machine; then the split
    // path below runs instead.
    if (want_fused_ && NumericPolicy::tiny_batch(B)) {
      fused_plan = make_tiny_forward_plan(B, U, V, J, lda, plan_rows, h_rbase, h_pg,
                                          plan_a, d_b_prep.as<float>());
      fused = true;
    } else if (want_fused_ && split) {
      fused_plan = make_simt_fused_plan(B, V, num_rows, J, lda, ldb,
                                        static_cast<long long>(J) * V, plan_rows, h_rbase, h_pg,
                                        plan_a, d_b_prep.as<float>(), a_view, multi_path_);
      fused = (fused_plan != nullptr);
    }
    if (split && !fused) {
      size_t const n_p = static_cast<size_t>(B) * num_rows * V;
      if (d_partials.reset(n_p * sizeof(float)) ||
          d_rbase.reset(static_cast<size_t>(J) * sizeof(int)) ||
          d_pg.reset(static_cast<size_t>(J) * sizeof(int))) {
        throw std::runtime_error("Runner: partial buffer allocation failed");
      }
      cudaMemcpy(d_rbase.as<int>(), h_rbase.data(), J * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_pg.as<int>(), h_pg.data(), J * sizeof(int), cudaMemcpyHostToDevice);
    }

    if (!a_view) {
      cudaMemcpy(d_src_index.as<int>(), h_src_index.data(), h_src_index.size() * sizeof(int),
                 cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_b_src_base.as<int>(), h_b_src_base.data(), h_b_src_base.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_p_of_col.as<int>(), h_b_p_of_col.data(), h_b_p_of_col.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    if (fused) return;  // the fused plan writes `out` directly; no grouped-GEMM plan is built
    int const plan_mode = impl == Impl::kFused ? static_cast<int>(Impl::kAuto)
                                               : static_cast<int>(impl);
    plan = std::make_unique<GroupedGemmPlan>(B, V, num_rows, lda, ldb, ldc, plan_rows, plan_a,
                                             d_b_prep.as<float>(), plan_mode, a_view, multi_path_);
  }

  /// Weight side: constant across calls, so a caller can hoist it.
  int prepare_weight(float const *w, double const *cg, cudaStream_t stream) {
    // cg travels as a kernel argument -- no host->device copy on the critical path.
    CgVals cg_vals{};
    for (int p = 0; p < P; ++p) cg_vals.v[p] = cg ? static_cast<float>(cg[p]) : 1.f;
    if (fused ? fused_plan->scales_after_dot() : plan->scales_after_dot()) {
      std::vector<double> scales(rows.size());
      for (size_t r = 0; r < rows.size(); ++r) {
        int const p = static_cast<int>(rows[r].b_off / U);
        scales[r] = cg ? cg[p] : 1.;
      }
      int const rc = fused ? fused_plan->set_scales(scales, stream)
                           : plan->set_scales(scales, stream);
      if (rc != 0) return rc;
      for (int p = 0; p < P; ++p) cg_vals.v[p] = 1.f;
    }
    if (fused ? fused_plan->original_weight_layout() : plan->original_weight_layout())
      return cudaMemcpyAsync(d_b_prep.as<float>(), w, static_cast<size_t>(P) * U * V * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream) == cudaSuccess ? 0 : 1;
    launch_pack_b(w, d_b_prep.as<float>(), d_b_src_base.as<int>(), d_b_p_of_col.as<int>(), cg_vals,
                  P, U, V, stream);
    return 0;
  }

  /// Activation side: bind x in place (view mode: pure host work) or pack it (fallback).
  int prepare_act(float const *x, cudaStream_t stream) {
    if (a_view) return bind_a(x, stream);
    launch_permute_a(x, d_a_prep.as<float>(), d_src_index.as<int>(), I, U, B, stream);
    return 0;
  }

  int bind_a(float const *x, cudaStream_t stream) {
    if (fused) return fused_plan->bind_a(x, stream);
    return plan->bind_a(x, stream);
  }

  int prepare(float const *x, float const *w, double const *cg, cudaStream_t stream) {
    int const rc = prepare_weight(w, cg, stream);
    if (rc != 0) return rc;
    return prepare_act(x, stream);
  }

  /// One dispatch for the per-call path (the sibling line's `run_act`): bind + launch, so the e2e
  /// column measures a single Python call rather than two.
  int run_act(float const *x, float *out, cudaStream_t stream) {
    int const rc = prepare_act(x, stream);
    if (rc != 0) return rc;
    return run(out, stream);
  }

  int run(float *out, cudaStream_t stream) {
    if (fused) return fused_plan->launch(out, stream);
    if (!split) return plan->launch(out, stream);
    int rc = plan->launch(d_partials.as<float>(), stream);
    if (rc != 0) return rc;
    launch_reduce_partials(d_partials.as<float>(), d_rbase.as<int>(), d_pg.as<int>(), out, B, J, V,
                           num_rows, stream);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
  }
};

Runner::Runner(int B, int I, int U, int P, int V, std::vector<int> i_dims, Impl impl)
    : p_(new Impl_(B, I, U, P, V, i_dims, impl)) {}

Runner::~Runner() = default;

int Runner::j_rows() const { return p_->J; }

int Runner::prepare(float const *x, float const *w, double const *cg, cudaStream_t stream) {
  return p_->prepare(x, w, cg, stream);
}

int Runner::prepare_weight(float const *w, double const *cg, cudaStream_t stream) {
  return p_->prepare_weight(w, cg, stream);
}

int Runner::prepare_act(float const *x, cudaStream_t stream) { return p_->prepare_act(x, stream); }

int Runner::run_act(float const *x, float *out, cudaStream_t stream) {
  return p_->run_act(x, out, stream);
}

int Runner::run(float *out, cudaStream_t stream) { return p_->run(out, stream); }

int Runner::operator()(float const *x, float const *w, double const *cg, float *out,
                       cudaStream_t stream) {
  int const rc = prepare(x, w, cg, stream);
  if (rc != 0) return rc;
  return run(out, stream);
}

float const *Runner::a_prep(long long *numel) const {
  // View mode has no packed activation buffer: say so with a null/zero pair rather than handing
  // back a pointer into freed or unrelated memory.
  if (p_->a_view) {
    *numel = 0;
    return nullptr;
  }
  *numel = static_cast<long long>(p_->B) * p_->I * p_->U;
  return p_->d_a_prep.as<float>();
}

float const *Runner::b_prep(long long *numel) const {
  *numel = static_cast<long long>(p_->V) * p_->P * p_->U;
  return p_->d_b_prep.as<float>();
}


}  // namespace eqlin_fp32
