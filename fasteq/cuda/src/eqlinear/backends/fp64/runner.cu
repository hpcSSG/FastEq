/***************************************************************************************************
 * fasteq eqlinear backends/fp64/runner.cu
 *
 * Glue: group table -> host index tables -> device packing kernels -> grouped GEMM.
 *
 *   out[b, j, n] = sum_{p in P(j)} cg[p] * sum_k x[b, prefix[p]+t, k] * W[p, k, n]
 *
 * realised as a block-sparse grouped GEMM with one group per output row j:
 *   A_prep[b, a_col(t) + m*U + k] = x[b, row_start(g) + m*i_g + t, k]
 *   B_prep[n, b_col(g) + m*U + k] = cg[p0+m] * W[p0+m, k, n]
 *   group j: M = B, N = V, K = g_op*U,  lda = I*U, ldb = P*U, ldc = J*V
 *
 * FP64 has no split and no extra passes: the packed operands are plain double.
 **************************************************************************************************/
#include "eqlinear.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "device_buffer.h"
#include "group_table.h"
#include "direct_gemm_f64.cuh"
#include "direct_gemm_f64_tma.cuh"
#include "grouped_gemm_f64.cuh"
#include "prep_kernels.h"
#include "row_plan.h"
#include "sm90_fp64_grouped_gemm.cuh"

namespace eqlin {

namespace {
/// 16-byte alignment in units of double.  The 2.x FP64 kernel does 8-byte accesses, but the
/// epilogue stores two doubles per access and every future vectorized/TMA path will want this.
constexpr int kAlign = 2;

/// `EQLIN_2X_TILE` vocabulary; keep in sync with gemm2x_config_name().
Gemm2xConfig gemm2x_from_name(char const *name) {
  for (Gemm2xConfig c : {Gemm2xConfig::kBaseline, Gemm2xConfig::kM128N32S3, Gemm2xConfig::kM128N32S4,
                         Gemm2xConfig::kM64N32S4, Gemm2xConfig::kM64N64S3, Gemm2xConfig::kIns8,
                         Gemm2xConfig::kM32N128S3, Gemm2xConfig::kM32N128S4}) {
    if (!strcmp(name, gemm2x_config_name(c))) return c;
  }
  throw std::invalid_argument("Runner: unknown EQLIN_2X_TILE");
}
}  // namespace

struct Runner::Impl_ {
  int B, I, U, P, V, J;
  F64Impl impl;
  /// A is bound straight to x instead of being permuted: possible when every output row draws on
  /// exactly one input path, because then the packed column strip is bit-identical to a slice of x
  /// (both `a_off` and `lda` come out the same).  `EQLIN_A_VIEW=0` forces the packed path back on.
  bool a_view = false;
  F64Impl effective;         // impl after kAuto resolution
  Gemm2xConfig cfg2x = Gemm2xConfig::kBaseline;  // 2.x tile, resolved by shape for kAuto2
  bool sm90_direct = false;  // impl 1: A read straight from x with the composed (m,u) descriptor
  bool sm90_packed = false;  // impl 1: packed A (the `EQLIN_A_PACK=1` parity/debug path)
  std::vector<RowPlan> rows_tma;  // per-row A offsets/paths for the composed descriptor
  int m_max = 1;
  GroupTable table;
  std::vector<RowPlan> rows;
  std::vector<RowPlan> rows_view;  // per-row A offsets for the activation-view mode
  long long lda = 0, ldb = 0, ldc = 0;

  std::vector<int> h_src_index;   // [I*U] A column -> element offset within one x batch
  std::vector<int> h_b_src_base;  // [P*U] B column -> base offset into w
  std::vector<int> h_b_p_of_col;  // [P*U] B column -> input path index

  DeviceBuffer d_src_index, d_b_src_base, d_b_p_of_col;
  DeviceBuffer d_a_prep, d_b_prep;
  std::unique_ptr<GroupedGemmPlan> plan;      // impl 0: 2.x grouped
  std::unique_ptr<Sm90Fp64Plan> sm90_plan;    // impl 1: 3.x TMA + DMMA mainloop
  std::unique_ptr<DirectGemmPlan> direct_plan;  // impl 4: hand-written 4-warp kernel
  std::unique_ptr<TmaGemmPlan> tma_plan;        // impl 6: hand-written kernel with a TMA fill
  bool tma_exact_n = false;                     // (128,32) vs (64,64) tile for impl 6

  Impl_(int B_, int I_, int U_, int P_, int V_, std::vector<int> const &i_dims, F64Impl impl_)
      : B(B_), I(I_), U(U_), P(P_), V(V_), impl(impl_), table(build_group_table(i_dims)) {
    J = table.out_total_i;
    if (B <= 0 || U <= 0 || V <= 0) throw std::invalid_argument("Runner: bad B/U/V");
    if (table.num_paths() != P) throw std::invalid_argument("Runner: P != len(i_dims)");
    if (P > kMaxPaths) throw std::invalid_argument("Runner: P exceeds kMaxPaths");
    if (I != table.in_prefix[P - 1] + table.i_dims[P - 1]) {
      throw std::invalid_argument("Runner: I != sum(i_dims)");
    }

    lda = static_cast<long long>(I) * U;
    ldb = static_cast<long long>(P) * U;
    ldc = static_cast<long long>(J) * V;
    if (lda % kAlign || ldb % kAlign || ldc % kAlign || V % kAlign) {
      throw std::invalid_argument(
          "Runner: FP64 needs 16-byte-aligned strides (I*U, P*U, J*V) and V");
    }

    build_host_tables();
    build_view_rows_if_eligible();
    // M3-A dispatch.  kAuto picks the backend by shape: multi-path groups read A straight from x
    // (the 3.x composed descriptor), single-path shapes stay on the faster 2.x kernel.
    effective = impl;
    bool const any_multi = std::any_of(table.path_count.begin(), table.path_count.end(),
                                       [](int m) { return m > 1; });
    // M5-C's rule, factored out: the 4-warp cp.async kernel wins on the narrow-N shapes.
    auto const auto3_choice = [&]() {
      return (V <= 128) ? F64Impl::kDirect
                        : (any_multi ? F64Impl::kSm90Dmma : F64Impl::k2xGrouped);
    };
    if (impl == F64Impl::kAuto4 || impl == F64Impl::kAuto5) {
      // M6: the TMA+canonical kernel's win is one-off (the fill issue) while its cost -- 32 eight-
      // byte LDS instead of 16 LDS.128 -- scales with K.  Measured across the presets, it wins at
      // every B once a row has <= 64 k-tiles (K <= 1024) and loses above that (LARGE_P3's 70).
      // kAuto5 uses that boundary; kAuto4 runs the kernel unconditionally for the A/B.
      bool const use_tma = impl == F64Impl::kAuto4 || max_row_k() <= 1024;
      // M6: the (128,32) exact-N tile is worth it exactly where V is not a multiple of 64 -- V=224
      // becomes seven exact N tiles instead of 8 (12.5 % of the DMMA work was padding), V=96 three
      // instead of four.  `EQLIN_TMA_N=32|64` forces the tile for the mechanism A/B.
      tma_exact_n = (V % 64 != 0);
      if (char const *env = nullptr; env != nullptr && *env != '\0')
        tma_exact_n = atoi(env) == 32;
      effective = use_tma ? F64Impl::kAuto4 : auto3_choice();
      if (effective == F64Impl::kAuto4 || effective == F64Impl::kDirect) build_rows_tma();
      if (effective == F64Impl::kSm90Dmma) sm90_direct = true;
    } else if (impl == F64Impl::kAuto || impl == F64Impl::kAuto2 || impl == F64Impl::kAuto3) {
      // M5-C: the hand-written 4-warp kernel wins where the per-CTA tile work is small, i.e. on the
      // narrow-N shapes (measured: V=96/128 beat both stock paths on 3 of 5 B values and on the
      // 5-point sums; V=224 loses because its 64-wide M tile pays more fill bytes per MAC than the
      // 128-wide 3.x tile).  Wider N keeps the M4 choice.
      bool const narrow_n = impl == F64Impl::kAuto3 && V <= 128;
      effective = narrow_n ? F64Impl::kDirect
                           : (any_multi ? F64Impl::kSm90Dmma : F64Impl::k2xGrouped);
    }
    cfg2x = select_gemm2x(impl == F64Impl::kAuto2);
    if (effective == F64Impl::kDirect) {
      // The hand-written kernel always reads A through the composed (U, m) K axis, so it needs the
      // rows_tma offsets for single-path shapes too.
      build_rows_tma();
    }
    if (effective == F64Impl::kSm90Dmma) {
      char const *pack = nullptr;
      sm90_packed = pack != nullptr && atoi(pack) != 0;
      sm90_direct = !sm90_packed;
      if (sm90_direct) build_rows_tma();
    }
    allocate();
  }

  /// 2.x tile selection.  `EQLIN_2X_TILE` forces a config, but only for the shape-dispatched entry
  /// (kAuto2), so impl 0/1/2 keep their frozen semantics and can serve as the A/B baseline.
  /// Without the override, kAuto2 deviates from the M1 baseline by mechanism: a 32-wide N strip is
  /// exact when V % 64 != 0 (V=224 spends 256/224 = 12.5 % of the DMMA work on padding, V=96 spends
  /// 128/96 = 33 %).  The interleaved A/B on both shapes and all five B values shows that saving
  /// reaches the clock (-9.6 % on SMALL_P1, -3.7 % on LARGE_P1 for the five-B sums) even though the
  /// narrower tile costs occupancy (196 registers -> 2 CTAs/SM instead of 166 -> 3).  V=128 is
  /// already exact and keeps the baseline tile, which is measurably better there.
  Gemm2xConfig select_gemm2x(bool shape_dispatch) {
    Gemm2xConfig cfg = Gemm2xConfig::kBaseline;
    if (shape_dispatch) {
      if (char const *env = nullptr; env != nullptr && *env != '\0') {
        cfg = gemm2x_from_name(env);
      } else if (V % 64 != 0 && V >= 32) {
        cfg = Gemm2xConfig::kM128N32S3;
      }
    }
    if (nullptr) {
      fprintf(stderr, "[runner] 2x tile=%s ai=%.1f\n", gemm2x_config_name(cfg),
              arithmetic_intensity());
    }
    return cfg;
  }

  /// Longest contraction among the output rows -- the M6 dispatch axis (see the kAuto5 rule).
  int max_row_k() const {
    int k = 0;
    for (auto const &r : rows) k = std::max(k, r.K);
    return k;
  }

  /// FLOP/byte of one forward call at the ideal traffic (the same formula as tests/cost_model.py).
  double arithmetic_intensity() const {
    double const flops = 2.0 * B * I * U * V;
    double const bytes = 8.0 * (static_cast<double>(B) * I * U + static_cast<double>(P) * U * V +
                                static_cast<double>(B) * J * V);
    return flops / bytes;
  }


  /// Per-output-row A offsets/paths for the composed (M, U, m) descriptor: row j=(g,t) starts at
  /// x[b, row_start[g]+t, :] and walks the group's paths with stride i_g*U.
  void build_rows_tma() {
    rows_tma.assign(rows.size(), RowPlan{});
    m_max = 1;
    for (int g = 0; g < table.num_groups(); ++g) {
      int const i = table.out_i_dims[g];
      int const m = table.path_count[g];
      int const r0 = table.row_start[g];
      m_max = std::max(m_max, m);
      for (int t = 0; t < i; ++t) {
        int const j = table.out_prefix[g] + t;
        RowPlan r = rows[j];
        r.a_off = static_cast<long long>(r0 + t) * U;
        r.m = m;
        r.m_stride = static_cast<long long>(i) * U;
        rows_tma[j] = r;
      }
    }
  }

  /// Packed A offsets/pitches for the activation-view mode, one entry per output row.
  void build_view_rows_if_eligible() {
    bool const one_path_per_row = std::all_of(table.path_count.begin(), table.path_count.end(),
                                              [](int m) { return m == 1; });
    char const *v = nullptr;
    a_view = one_path_per_row && !(v != nullptr && atoi(v) == 0);
    if (!a_view) return;

    rows_view.clear();
    rows_view.reserve(rows.size());
    for (int g = 0; g < table.num_groups(); ++g) {
      int const i = table.out_i_dims[g];
      for (int t = 0; t < i; ++t) {
        RowPlan r = rows[table.out_prefix[g] + t];
        // With one path per row the packed strip starts at U*(row_start+t) == U*j and the pitch is
        // I*U, both identical to x's.  Checked rather than assumed: if the packing ever changes,
        // this must fail loudly instead of reading the wrong bytes.
        if (r.a_off != static_cast<long long>(table.out_prefix[g] + t) * U) {
          throw std::logic_error("Runner: activation view assumes a_off == U*j on single-path rows");
        }
        rows_view.push_back(r);
      }
    }
  }

  void build_host_tables() {
    int const G = table.num_groups();

    h_src_index.assign(static_cast<size_t>(I) * U, 0);
    h_b_src_base.assign(static_cast<size_t>(P) * U, 0);
    h_b_p_of_col.assign(static_cast<size_t>(P) * U, 0);
    rows.clear();

    // ---- A columns + per-row plan ----
    long long a_col = 0;
    for (int g = 0; g < G; ++g) {
      int const i = table.out_i_dims[g];
      int const m = table.path_count[g];
      int const K = m * U;
      int const r0 = table.row_start[g];
      for (int t = 0; t < i; ++t) {
        for (int mm = 0; mm < m; ++mm) {
          for (int k = 0; k < U; ++k) {
            long long const col = a_col + static_cast<long long>(t) * K + mm * U + k;
            h_src_index[col] = (r0 + mm * i + t) * U + k;
          }
        }
        rows.push_back(RowPlan{K, a_col + static_cast<long long>(t) * K, 0});
      }
      a_col += static_cast<long long>(i) * K;
    }

    // ---- B columns (+ row -> group B block) ----
    long long b_col = 0;
    for (int g = 0; g < G; ++g) {
      int const m = table.path_count[g];
      int const p0 = table.path_istart[g];
      int const K0 = m * U;
      for (int mm = 0; mm < m; ++mm) {
        for (int k = 0; k < U; ++k) {
          long long const col = b_col + static_cast<long long>(mm) * U + k;
          h_b_src_base[col] = static_cast<long long>(p0 + mm) * U * V + static_cast<long long>(k) * V;
          h_b_p_of_col[col] = p0 + mm;
        }
      }
      for (int t = 0; t < table.out_i_dims[g]; ++t) {
        rows[table.out_prefix[g] + t].b_off = b_col;
      }
      b_col += K0;
    }
    if (a_col != static_cast<long long>(I) * U || b_col != static_cast<long long>(P) * U) {
      throw std::runtime_error("Runner: packed column count mismatch");
    }
  }

  void allocate() {
    size_t const n_a = static_cast<size_t>(B) * I * U;
    size_t const n_b = static_cast<size_t>(V) * P * U;
    // A_prep is needed by the 2.x packed path, and by the 3.x flat (EQLIN_A_PACK=1) parity path;
    // the 3.x composed path reads x in place.
    bool const need_a_prep = (effective == F64Impl::kDirect || effective == F64Impl::kAuto4)
                                 ? false
                                 : (effective == F64Impl::kSm90Dmma ? sm90_packed : !a_view);
    if (d_src_index.reset(h_src_index.size() * sizeof(int)) ||
        d_b_src_base.reset(h_b_src_base.size() * sizeof(int)) ||
        d_b_p_of_col.reset(h_b_p_of_col.size() * sizeof(int)) ||
        (need_a_prep && d_a_prep.reset(n_a * sizeof(double))) ||
        d_b_prep.reset(n_b * sizeof(double))) {
      throw std::runtime_error("Runner: device allocation failed");
    }

    cudaMemcpy(d_src_index.as<int>(), h_src_index.data(), h_src_index.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_src_base.as<int>(), h_b_src_base.data(), h_b_src_base.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_p_of_col.as<int>(), h_b_p_of_col.data(), h_b_p_of_col.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    if (effective == F64Impl::kSm90Dmma) {
      if (!sm90_fp64_available()) throw std::runtime_error("Runner: SM90 FP64 backend unavailable");
      if (sm90_direct) {
        sm90_plan = std::make_unique<Sm90Fp64Plan>(
            B, V, J, lda, ldb, ldc, rows_tma, nullptr, d_b_prep.as<double>(), /*composed=*/true,
            U, m_max);
      } else {
        sm90_plan = std::make_unique<Sm90Fp64Plan>(
            B, V, J, lda, ldb, ldc, rows, d_a_prep.as<double>(), d_b_prep.as<double>(),
            /*composed=*/false, 0, 1);
      }
    } else if (effective == F64Impl::kDirect) {
      direct_plan = std::make_unique<DirectGemmPlan>(B, V, J, lda, ldb, ldc, rows_tma, nullptr,
                                                     d_b_prep.as<double>(), U);
    } else if (effective == F64Impl::kAuto4) {
      tma_plan = std::make_unique<TmaGemmPlan>(B, V, J, lda, ldb, ldc, rows_tma, nullptr,
                                               d_b_prep.as<double>(), U, tma_exact_n);
    } else {
      plan = std::make_unique<GroupedGemmPlan>(
          B, V, J, lda, ldb, ldc, a_view ? rows_view : rows,
          a_view ? nullptr : d_a_prep.as<double>(), d_b_prep.as<double>(), cfg2x);
    }
  }

  int prepare_weight(double const *w, double const *cg, cudaStream_t stream) {
    // cg travels as a kernel argument -- no host->device copy on the critical path.
    CgVals cg_vals{};
    for (int p = 0; p < P; ++p) cg_vals.v[p] = cg ? cg[p] : 1.0;
    launch_pack_b(w, d_b_prep.as<double>(), d_b_src_base.as<int>(), d_b_p_of_col.as<int>(), cg_vals,
                  P, U, V, stream);
    return 0;
  }

  int prepare_act(double const *x, cudaStream_t stream) {
    if (effective == F64Impl::kDirect) {
      direct_plan->bind_a(x, stream);
      return 0;
    }
    if (effective == F64Impl::kAuto4) {
      tma_plan->bind_a(x, stream);
      return 0;
    }
    if (sm90_direct) {
      // Composed A: the descriptor reads x in place, single- and multi-path alike.
      sm90_plan->bind_a(x, stream);
      return 0;
    }
    if (effective == F64Impl::kSm90Dmma) {
      // EQLIN_A_PACK=1: the flat parity/debug path always packs A.
      launch_permute_a(x, d_a_prep.as<double>(), d_src_index.as<int>(), I, U, B, stream);
      return 0;
    }
    if (a_view) {
      // Nothing to pack: the GEMM reads x in place.  Only the per-row base pointers move.
      plan->bind_a(x, stream);
      return 0;
    }
    launch_permute_a(x, d_a_prep.as<double>(), d_src_index.as<int>(), I, U, B, stream);
    return 0;
  }
};

Runner::Runner(int B, int I, int U, int P, int V, std::vector<int> i_dims, F64Impl impl)
    : p_(new Impl_(B, I, U, P, V, i_dims, impl)) {}

Runner::~Runner() = default;

int Runner::j_rows() const { return p_->J; }

int Runner::prepare_weight(double const *w, double const *cg, cudaStream_t stream) {
  return p_->prepare_weight(w, cg, stream);
}

int Runner::prepare_act(double const *x, cudaStream_t stream) { return p_->prepare_act(x, stream); }

int Runner::prepare(double const *x, double const *w, double const *cg, cudaStream_t stream) {
  int const rc = p_->prepare_weight(w, cg, stream);
  if (rc != 0) return rc;
  return p_->prepare_act(x, stream);
}

int Runner::run(double *out, cudaStream_t stream) {
  if (p_->tma_plan) return p_->tma_plan->launch(out, stream);
  if (p_->direct_plan) return p_->direct_plan->launch(out, stream);
  if (p_->sm90_plan) return p_->sm90_plan->launch(out, stream);
  return p_->plan->launch(out, stream);
}

int Runner::operator()(double const *x, double const *w, double const *cg, double *out,
                       cudaStream_t stream) {
  int const rc = prepare(x, w, cg, stream);
  if (rc != 0) return rc;
  return run(out, stream);
}

double const *Runner::a_prep(long long *numel) const {
  if (!p_->d_a_prep) {
    *numel = 0;
    return nullptr;
  }
  *numel = static_cast<long long>(p_->B) * p_->I * p_->U;
  return p_->d_a_prep.as<double>();
}

double const *Runner::b_prep(long long *numel) const {
  *numel = static_cast<long long>(p_->V) * p_->P * p_->U;
  return p_->d_b_prep.as<double>();
}

bool Runner::a_view() const {
  return p_->a_view || p_->sm90_direct || p_->direct_plan || p_->tma_plan;
}

int out_total_i(std::vector<int> const &i_dims) { return build_group_table(i_dims).out_total_i; }

}  // namespace eqlin
