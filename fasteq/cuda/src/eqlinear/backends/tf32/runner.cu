/***************************************************************************************************
 * fasteq eqlinear backends/tf32/runner.cu
 *
 * Glue: group table -> host index tables -> device packing kernels -> grouped GEMM.
 *
 *   out[b, j, n] = sum_{p in P(j)} cg[p] * sum_k x[b, in_prefix[p] + t, k] * W[p, k, n]
 *
 * realised as a block-sparse grouped GEMM with one group per output row j:
 *   A_prep[b, a_col(op) + t*K + m*U + k] = x[b, row_start(op) + m*i_op + t, k]
 *   B_prep[n, b_col(op) + m*U + k]       = cg[p0+m] * W[p0+m, k, n]
 *   group j: M = B, N = V, K = g_op*U,  lda = I*U, ldb = P*U, ldc = J*V
 *
 * For the tf32x3 K-concatenation variants (Impl::kSm90_3xTF32) both operands are split into a
 * (big, res) tf32 pair and replicated over 3 K-segments -- see split_pack.h -- which triples the
 * packed column counts (K = 3*g_op*U) without changing the group structure.
 **************************************************************************************************/
#include "eqlinear.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef EQLIN_TB_K3
#define EQLIN_TB_K3 16
#endif

#include "device_buffer.h"
#include "eqlinear_group_table.h"
#include "grouped_gemm.cuh"
#include "packed_index_tables.h"
#include "prep_kernels.h"
#include "row_plan.h"
#include "sm90_grouped_gemm.cuh"
#include "split_pack.h"

namespace eqlin {

using fasteq::eqlinear::PackedIndexTables;
using fasteq::eqlinear::SegmentSpec;

namespace {
constexpr int kAlign = 4;  // 128-bit accesses; required by the tf32x3 kernels
}  // namespace

struct Runner::Impl_ {
  int B, I, U, P, V, J;
  Impl impl;
  bool kconcat = false;   // 3-way K-concatenated tf32x3 operands
  bool inkernel = false;  // tf32x3 with the A split done inside the mainloop
  /// A is bound straight to x instead of being permuted: possible when every output row draws on
  /// exactly one input path, because then the packed column strip is bit-identical to a slice of x
  /// (both `a_off` and `lda` come out the same).
  bool a_view = false;
  /// 0 = x is [B, I, U]; 1 = x is P-major [P][B, d_p, U] (one buffer).  Only affects the offsets and
  /// pitches of the A view; the production layout is [B, I, U].
  int a_layout = 0;
  int seg = 1;            // number of K segments per operand (1 or 3)
  GroupTable table;
  std::vector<RowPlan> rows;
  std::vector<RowPlan> rows_view;  // per-row A offsets/pitches for the activation-view mode
  long long lda = 0, ldb = 0, ldc = 0;
  long long a_cols = 0, b_cols = 0;  // packed column counts

  std::vector<int> h_src_index;           // [a_cols] A column -> element offset within one x batch
  std::vector<int> h_b_src_base;          // [b_cols] B column -> base offset into w
  std::vector<int> h_b_p_of_col;          // [b_cols] B column -> input path index
  std::vector<unsigned char> h_a_is_res;  // [a_cols] take the residual instead of the big part
  std::vector<unsigned char> h_b_is_res;  // [b_cols]

  DeviceBuffer d_src_index, d_b_src_base, d_b_p_of_col, d_a_is_res, d_b_is_res;
  DeviceBuffer d_a_prep, d_b_prep, d_b_prep_res;
  std::unique_ptr<GroupedGemmPlan> plan;     // M1: CUTLASS 2.x / SM80 mma.sync path
  std::unique_ptr<Sm90Launch> sm90_plan;     // M2: CUTLASS 3.x / SM90 WGMMA path (plain or split-K)

  Impl_(int B_, int I_, int U_, int P_, int V_, std::vector<int> const &i_dims, Impl impl_)
      : B(B_), I(I_), U(U_), P(P_), V(V_), impl(impl_), kconcat(is_kconcat(impl_)),
        inkernel(is_inkernel_tf32x3(impl_)), table(build_group_table(i_dims)) {
    seg = kconcat ? 3 : 1;
    J = table.out_total_i;
    if (B <= 0 || U <= 0 || V <= 0) throw std::invalid_argument("Runner: bad B/U/V");
    if (table.num_paths() != P) throw std::invalid_argument("Runner: P != len(i_dims)");
    if (I != table.in_prefix[P - 1] + table.i_dims[P - 1]) {
      throw std::invalid_argument("Runner: I != sum(i_dims)");
    }

    a_cols = static_cast<long long>(seg) * I * U;
    b_cols = static_cast<long long>(seg) * P * U;
    lda = a_cols;
    ldb = b_cols;
    ldc = static_cast<long long>(J) * V;
    if (lda % kAlign || ldb % kAlign || ldc % kAlign || V % kAlign || B % kAlign) {
      throw std::invalid_argument(
          "Runner: tf32x3 needs M=N=K=4k and 4-aligned strides (I*U, P*U, J*V)");
    }

    build_host_tables();
    // Activation-view mode.  With one input path per output row the packed A column strip of row j
    // is exactly x[b, j*U : j*U+U], so the permute would be an identity copy and the kernel can read
    // x in place.  Grouped FP32-A and in-kernel-split mainloops can both consume this view;
    // the K-concatenated variant needs a 3-segment A and keeps the packed path.
    bool const grouped_view = impl == Impl::k3xTF32;
    if (inkernel || grouped_view) {
      bool const one_path_per_row = std::all_of(table.path_count.begin(), table.path_count.end(),
                                                [](int m) { return m == 1; });
      a_view = one_path_per_row;
      a_layout = 0;
      if (a_view) build_view_rows();
    }
    allocate();
  }

  /// Per-row A offsets and pitches for the activation-view mode, one entry per output row in the
  /// same order as `rows`.
  void build_view_rows() {
    rows_view.clear();
    rows_view.reserve(rows.size());
    for (int g = 0; g < table.num_groups(); ++g) {
      int const i = table.out_i_dims[g];
      int const p0 = table.path_istart[g];  // the group's only path
      for (int t = 0; t < i; ++t) {
        RowPlan r = rows[table.out_prefix[g] + t];
        if (a_layout == 1) {
          // x is P-major: path p starts at in_prefix[p]*B*U, and this row is its component-t slab
          // with pitch d_p*U.
          r.a_off = static_cast<long long>(table.in_prefix[p0]) * B * U +
                    static_cast<long long>(t) * U;
          r.lda = static_cast<long long>(i) * U;
        } else if (r.a_off != static_cast<long long>(table.out_prefix[g] + t) * U) {
          // This equality is the whole reason the view is legal for [B,I,U]: with one path per row
          // the packed column strip starts at U*(row_start+t) == U*j and the pitch is I*U, both
          // identical to x's.  Checked rather than assumed -- if the packing ever changes, this
          // must fail loudly instead of reading the wrong bytes.
          throw std::logic_error("Runner: activation view assumes a_off == U*j on single-path rows");
        }
        rows_view.push_back(r);
      }
    }
  }

  void build_host_tables() {
    PackedIndexTables const packed(
        table, U, V, SegmentSpec{seg, kconcat, kconcat});
    h_src_index = packed.src_index();
    h_b_src_base = packed.b_src_base();
    h_b_p_of_col = packed.b_p_of_col();
    if (kconcat) {
      h_a_is_res = packed.a_is_res();
      h_b_is_res = packed.b_is_res();
    }

    rows.clear();
    rows.reserve(static_cast<size_t>(J));
    for (auto const &block : packed.blocks()) {
      for (int t = 0; t < block.i; ++t) {
        rows.push_back(RowPlan{block.K, block.a_base + static_cast<long long>(t) * block.K,
                               block.b_base});
      }
    }
    if (packed.a_cols() != a_cols || packed.b_cols() != b_cols) {
      throw std::runtime_error("Runner: packed column count mismatch");
    }
  }

  void allocate() {
    size_t const n_a = static_cast<size_t>(B) * a_cols;
    size_t const n_b = static_cast<size_t>(V) * b_cols;
    if (d_src_index.reset(h_src_index.size() * sizeof(int)) ||
        d_b_src_base.reset(h_b_src_base.size() * sizeof(int)) ||
        d_b_p_of_col.reset(h_b_p_of_col.size() * sizeof(int)) ||
        (!a_view && d_a_prep.reset(n_a * sizeof(float))) || d_b_prep.reset(n_b * sizeof(float)) ||
        (inkernel && d_b_prep_res.reset(n_b * sizeof(float)))) {
      throw std::runtime_error("Runner: device allocation failed");
    }
    if (kconcat &&
        (d_a_is_res.reset(h_a_is_res.size() * sizeof(unsigned char)) ||
         d_b_is_res.reset(h_b_is_res.size() * sizeof(unsigned char)))) {
      throw std::runtime_error("Runner: device allocation failed");
    }

    cudaMemcpy(d_src_index.as<int>(), h_src_index.data(), h_src_index.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_src_base.as<int>(), h_b_src_base.data(), h_b_src_base.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_p_of_col.as<int>(), h_b_p_of_col.data(), h_b_p_of_col.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    if (kconcat) {
      cudaMemcpy(d_a_is_res.as<unsigned char>(), h_a_is_res.data(), h_a_is_res.size(),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(d_b_is_res.as<unsigned char>(), h_b_is_res.data(), h_b_is_res.size(),
                 cudaMemcpyHostToDevice);
    }

    if (is_sm90(impl)) {
      // kSm90_1xTF32 shares M1's packed operands (float) so it is a drop-in probe of the Hopper
      // path; kSm90_3xTF32 feeds the split/K-concatenated tf32 buffers; the two in-kernel-split
      // variants read a plain fp32 permute of A plus the two pre-split halves of B and differ only
      // in which mainloop consumes them.
      Sm90Operands const operands =
          inkernel  ? (impl == Impl::kSm90_3xTF32RegSplitRnFold ? Sm90Operands::kFp32RegSplitRnFold
                     : impl == Impl::kSm90_3xTF32RegSplitFold   ? Sm90Operands::kFp32RegSplitFold
                     : impl == Impl::kSm90_3xTF32RegSplitRn     ? Sm90Operands::kFp32RegSplitRn
                     : impl == Impl::kSm90_3xTF32RegSplit   ? Sm90Operands::kFp32RegSplit
                                                            : Sm90Operands::kFp32InKernelSplit)
          : kconcat ? Sm90Operands::kTf32KConcat
                    : Sm90Operands::kFp32;
      std::unique_ptr<Sm90Launch> sm90 = std::make_unique<Sm90GroupedPlan>(
          operands, B, V, J, lda, ldb, ldc, a_view ? rows_view : rows,
          a_view ? nullptr : static_cast<void const *>(d_a_prep.as<float>()),
          static_cast<void const *>(d_b_prep.as<float>()),
          static_cast<void const *>(d_b_prep_res.as<float>()));
      sm90_plan = std::move(sm90);
    } else {
      plan = std::make_unique<GroupedGemmPlan>(impl, B, V, J, lda, ldb, ldc,
                                               a_view ? rows_view : rows,
                                               a_view ? nullptr : d_a_prep.as<float>(),
                                               d_b_prep.as<float>());
    }
  }

  int prepare_weight(float const *w, double const *cg, cudaStream_t stream) {
    // cg travels as a kernel argument -- no host->device copy on the critical path.
    CgVals cg_vals{};
    for (int p = 0; p < P; ++p) {
      cg_vals.v[p] = cg ? static_cast<float>(cg[p]) : 1.f;
    }

    if (kconcat) {
      launch_split_b(w, d_b_prep.as<float>(), d_b_src_base.as<int>(), d_b_p_of_col.as<int>(),
                     d_b_is_res.as<unsigned char>(), cg_vals, b_cols, P, U, V, stream);
    } else if (inkernel) {
      // B is pre-split here rather than in-kernel: it is a few MB and never changes, so the extra
      // smem tile would sit on the critical path for nothing.
      launch_pack_b_split(w, d_b_prep.as<float>(), d_b_prep_res.as<float>(),
                          d_b_src_base.as<int>(), d_b_p_of_col.as<int>(), cg_vals, b_cols, P, U, V,
                          /*rn=*/impl == Impl::kSm90_3xTF32RegSplitRn ||
                              impl == Impl::kSm90_3xTF32RegSplitRnFold,
                          stream);
    } else {
      launch_pack_b(w, d_b_prep.as<float>(), d_b_src_base.as<int>(), d_b_p_of_col.as<int>(),
                    cg_vals, P, U, V, stream);
    }
    return 0;
  }

  int prepare_act(float const *x, cudaStream_t stream) {
    if (kconcat) {
      launch_split_a(x, d_a_prep.as<float>(), d_src_index.as<int>(), d_a_is_res.as<unsigned char>(),
                     a_cols, I, U, B, stream);
      return 0;
    }
    if (a_view) {
      // Nothing to pack: the GEMM reads x in place.  Only the per-row base pointers move.
      if (sm90_plan) {
        sm90_plan->bind_a(x, stream);
        return 0;
      }
      return plan->bind_a(x, stream);
    }
    launch_permute_a(x, d_a_prep.as<float>(), d_src_index.as<int>(), I, U, B, stream);
    return 0;
  }

  int prepare(float const *x, float const *w, double const *cg, cudaStream_t stream) {
    int const rc = prepare_weight(w, cg, stream);
    if (rc != 0) return rc;
    return prepare_act(x, stream);
  }

  int run(float *out, cudaStream_t stream) {
    if (sm90_plan) return sm90_plan->launch(out, stream);
    return plan->launch(out, stream);
  }
};

Runner::Runner(int B, int I, int U, int P, int V, std::vector<int> i_dims, Impl impl)
    : p_(new Impl_(B, I, U, P, V, i_dims, impl)) {}

Runner::~Runner() = default;

int Runner::j_rows() const { return p_->J; }

bool Runner::a_view() const { return p_->a_view; }

int Runner::prepare_weight(float const *w, double const *cg, cudaStream_t stream) {
  return p_->prepare_weight(w, cg, stream);
}

int Runner::prepare_act(float const *x, cudaStream_t stream) { return p_->prepare_act(x, stream); }

int Runner::prepare(float const *x, float const *w, double const *cg, cudaStream_t stream) {
  return p_->prepare(x, w, cg, stream);
}

int Runner::run(float *out, cudaStream_t stream) { return p_->run(out, stream); }

int Runner::operator()(float const *x, float const *w, double const *cg, float *out,
                       cudaStream_t stream) {
  int const rc = prepare(x, w, cg, stream);
  if (rc != 0) return rc;
  return run(out, stream);
}

float const *Runner::a_prep(long long *numel) const {
  // In activation-view mode there is no packed A at all; report an empty buffer rather than a
  // dangling pointer.
  if (!p_->d_a_prep) {
    *numel = 0;
    return nullptr;
  }
  *numel = static_cast<long long>(p_->B) * p_->a_cols;
  return p_->d_a_prep.as<float>();
}

float const *Runner::b_prep(long long *numel) const {
  *numel = static_cast<long long>(p_->V) * p_->b_cols;
  return p_->d_b_prep.as<float>();
}


}  // namespace eqlin
