/***************************************************************************************************
 * fasteq eqlinear backends/tf32/bwd.cu
 **************************************************************************************************/
#include "bwd.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "cutlass/numeric_conversion.h"

#include "device_buffer.h"
#include "group_table.h"
#include "prep_kernels.h"  // CgVals
#include "reduce_dw.h"
#include "row_plan.h"
#include "sm90_grouped_gemm.cuh"
#include "transpose_kt.h"

namespace eqlin {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 32;
constexpr int kAlign = 4;  // 128-bit accesses, same constraint as the forward

/// dgrad's B operand is `B[n=u, k=v] = cg[p] * W[p,u,v]`, i.e. **W's own layout** -- the weight side
/// needs no transpose, only the cg scale and the tf32 (big, res) split.  One buffer per half.
__global__ void pack_w_dgrad_kernel(float const *__restrict__ w, float *__restrict__ big,
                                    float *__restrict__ res, CgVals cg, int UV) {
  cutlass::NumericConverterFastF32<> conv;
  int const p = blockIdx.y;
  float const scale = cg.v[p];
  float const *__restrict__ wp = w + static_cast<long long>(p) * UV;
  float *__restrict__ bp = big + static_cast<long long>(p) * UV;
  float *__restrict__ rp = res + static_cast<long long>(p) * UV;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < UV; i += gridDim.x * blockDim.x) {
    cutlass::Array<cutlass::tfloat32_t, 2> const r = conv(scale * wp[i]);
    bp[i] = static_cast<float>(r[0]);
    rp[i] = static_cast<float>(r[1]);
  }
}

void launch_pack_w_dgrad(float const *w, float *big, float *res, CgVals cg, int P, int UV,
                         cudaStream_t stream) {
  int bx = (UV + kThreads - 1) / kThreads;
  if (bx > kMaxBlocks) bx = kMaxBlocks;
  if (bx < 1) bx = 1;
  pack_w_dgrad_kernel<<<dim3(bx, P), kThreads, 0, stream>>>(w, big, res, cg, UV);
}

}  // namespace


namespace {

/// Per-path {source row, i_g, destination offset} for the two transposed wgrad operands, plus the
/// grid extents.  B2 hands these offsets to a plan-wide base; B3 puts them in each row's base.
struct WgradPaths {
  std::vector<KtPath> x;  // one per path
  std::vector<KtPath> y;  // one per group
  int k_tiles_x = 0;
  int k_tiles_y = 0;
};

WgradPaths make_wgrad_paths(GroupTable const &t, int B, int U, int V) {
  WgradPaths w;
  int const P = t.num_paths(), G = t.num_groups();
  w.x.resize(P);
  w.y.resize(G);
  long long off = 0;
  for (int g = 0; g < G; ++g) {
    int const i_g = t.out_i_dims[g];
    for (int m = 0; m < t.path_count[g]; ++m) {
      int const p = t.path_istart[g] + m;
      w.x[p] = KtPath{t.row_start[g] + m * i_g, i_g, off};
      off += static_cast<long long>(U) * B * i_g;
    }
  }
  long long off_y = 0;
  for (int g = 0; g < G; ++g) {
    int const i_g = t.out_i_dims[g];
    w.y[g] = KtPath{t.out_prefix[g], i_g, off_y};
    off_y += static_cast<long long>(V) * B * i_g;
  }
  for (int p = 0; p < P; ++p) w.k_tiles_x = std::max(w.k_tiles_x, (B * t.i_dims[p] + 31) / 32);
  for (int g = 0; g < G; ++g) w.k_tiles_y = std::max(w.k_tiles_y, (B * t.out_i_dims[g] + 31) / 32);
  return w;
}

/// Device SM count, queried once per runner.  0 means "unknown" -- the split rule is then skipped and
/// the accuracy-driven default stands.
int device_sm_count() {
  int dev = 0, sm = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0;
  if (cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return 0;
  return sm;
}

/// Partial buffer budget for the split rule: the fp64 reduction reads all of it back, so a shorter
/// chain (more splits) has to earn its keep against that traffic.
constexpr long long kPartialsBudgetBytes = 16LL << 20;

/// The per-split accumulator length for wgrad.
///
/// The accuracy rule sizes it to ~512 terms (the fp32 error is ~linear in the chain, ~7e-9/term).
/// That says nothing about the *machine*, and in the fused backward (B3) the makespan per wave is
/// set by the longest tile: dgrad's tiles are fixed at K = V, wgrad's are K_p/S_p.  On the small
/// points the accuracy chain leaves wgrad's tiles 2-4x longer than dgrad's and the whole grid under
/// one wave -- measured at 80 CTAs / 0.61 waves / 10% SM utilisation (B3_REPORT.md §7).
///
/// So shorten the chain as far as the tile count allows *without adding a wave* and while the
/// partials stay inside the budget. Shorter is strictly better for accuracy, and the rule never
/// lengthens the chain. Skipping it when a wave would be added is the whole point: a blanket split
/// floor was measured 1.53x *slower* on a 1.9-wave point (B2_REPORT.md §2).
int pick_chain(GroupTable const &t, int B, int U, int V, int sm_count, int base_chain) {
  int const P = t.num_paths();
  auto sum_splits = [&](int chain) {
    long long sum = 0;
    for (int p = 0; p < P; ++p) {
      long long const K = static_cast<long long>(B) * t.i_dims[p];
      long long const sp = (K + chain - 1) / chain;
      sum += std::min<long long>(std::max<long long>(sp, 1), std::max<long long>(1, K / 16));
    }
    return sum;
  };
  long long const dx_tiles =
      static_cast<long long>(t.in_prefix[P - 1] + t.i_dims[P - 1]) * ((B + 127) / 128);
  long long const mtiles = (U + 127) / 128;
  auto waves = [&](int chain) {
    long long const total = dx_tiles + mtiles * sum_splits(chain);
    return (total + sm_count - 1) / sm_count;
  };
  int const w0 = waves(base_chain);
  // Only the under-one-wave case.  Measured: relaxing this to "keep the wave count" also shortens
  // the chain on points that already have >=2 waves, where the extra per-CTA fill costs 2-3% more
  // than the shorter tiles save -- the aggregate across 25 points cancelled out to 1.00x
  // (B3_REPORT.md §7.3).  Under one wave there is no such trade: idling SMs are pure loss.
  if (w0 > 1) return base_chain;
  int best = base_chain;
  for (int chain : {384, 256, 192, 160, 128, 112, 96, 80, 64}) {
    if (waves(chain) != w0) break;  // both metrics are monotone in the chain, so stop at the first
    if (sum_splits(chain) * static_cast<long long>(U) * V * 4 > kPartialsBudgetBytes) break;
    best = chain;
  }
  return best;
}

/// The wgrad problem table: one problem per (path, split).
///
/// Order is group -> split -> path-within-group: a group's paths read the same Yt slab at the same k
/// offset, so keeping them adjacent holds that slice in L2 -- measured worth 45% of the GEMM's DRAM
/// reads (B2_REPORT.md §6).
///
/// The split count is per path and targets a ~512-term accumulator chain: the fp32 error is
/// ~linear in that chain (~7e-9 per term) and the fp64 reduce adds none across splits.  Within one
/// config K_p spans max(i_dims)/min(i_dims) (7x), so a single global S either over-splits the short
/// paths into 4-k-tile CTAs or under-splits the long ones.  No "fill the machine" floor: measured on
/// LARGE_P1, forcing S up for occupancy cost 1.53x on dw for accuracy we do not need.
struct WgradRows {
  std::vector<RowPlan> rows;
  std::vector<int> splits;          // per path
  std::vector<long long> part_off;  // per path, element offset into the partial buffer
  long long n_partials = 0;         // number of partial tiles (== number of problems)
  int max_splits = 1;
};

WgradRows make_wgrad_rows(GroupTable const &t, int B, int U, int V, int splits_override,
                          int sm_count, WgradPaths const &paths) {
  WgradRows w;
  int const P = t.num_paths();
  // Per-split chain: EQLIN_WGRAD_CHAIN forces one (the ladder experiment); otherwise the rule in
  // pick_chain() may shorten the accuracy default when the machine would otherwise be left idle.
  int chain = 512;
  if (char const *c = nullptr; c != nullptr && atoi(c) > 0) {
    chain = atoi(c);
  } else if (splits_override <= 0 && sm_count > 0) {
    chain = pick_chain(t, B, U, V, sm_count, chain);
  }
  w.splits.assign(P, 1);
  w.part_off.assign(P, 0);
  long long part_cursor = 0;
  for (int p = 0; p < P; ++p) {
    long long const K = static_cast<long long>(B) * t.i_dims[p];
    int sp = splits_override > 0 ? splits_override : static_cast<int>((K + chain - 1) / chain);
    sp = std::min<long long>(sp, std::max<long long>(1, K / 16));
    w.splits[p] = sp;
    w.part_off[p] = part_cursor * static_cast<long long>(U) * V;
    part_cursor += sp;
    w.max_splits = std::max(w.max_splits, sp);
  }
  w.n_partials = part_cursor;
  w.rows.reserve(static_cast<size_t>(part_cursor));
  for (int g = 0; g < t.num_groups(); ++g) {
    int const npaths = t.path_count[g];
    int const p0 = t.path_istart[g];
    int const S = w.splits[p0];
    long long const K = static_cast<long long>(B) * t.out_i_dims[g];
    long long const per = (K / S / 16) * 16;
    for (int s = 0; s < S; ++s) {
      long long const koff = static_cast<long long>(s) * per;
      long long const len = (s == S - 1) ? (K - koff) : per;
      for (int m = 0; m < npaths; ++m) {
        int const p = p0 + m;
        // The offsets are the ones the transpose writes to (cumulative slabs), not p*U*K_p.
        w.rows.push_back(RowPlan{static_cast<int>(len),
                                 paths.x[p].dst_off + koff,
                                 paths.y[g].dst_off + koff,
                                 K,   // lda = K_p
                                 K,   // ldb = K_g (= K_p)
                                 0,   // ldc: the plan's V
                                 w.part_off[p] + static_cast<long long>(s) * U * V,
                                 nullptr, nullptr, nullptr, nullptr,
                                 U});
      }
    }
  }
  return w;
}

/// dgrad: one problem per (group, path, component).  (g,t,m) -> i is a bijection, so each x row is
/// written by exactly one problem and the K axis stays V (no accumulation, no scatter).  Order is
/// the derivation's; component-major was measured neutral (1.01x, inside the round spread).
std::vector<RowPlan> make_dgrad_rows(GroupTable const &t, int B, int U, int V) {
  std::vector<RowPlan> rows;
  int const P = t.num_paths();
  rows.reserve(t.in_prefix[P - 1] + t.i_dims[P - 1]);
  for (int g = 0; g < t.num_groups(); ++g) {
    int const i = t.out_i_dims[g];
    int const npaths = t.path_count[g];
    int const r0 = t.row_start[g];
    int const o0 = t.out_prefix[g];
    int const p0 = t.path_istart[g];
    for (int m = 0; m < npaths; ++m)
      for (int tt = 0; tt < i; ++tt)
        rows.push_back(RowPlan{V,
                               static_cast<long long>(o0 + tt) * V,
                               static_cast<long long>(p0 + m) * U * V,
                               0, 0, 0,
                               static_cast<long long>(r0 + m * i + tt) * U,
                               nullptr, nullptr, nullptr, nullptr,
                               B});
  }
  return rows;
}


/// cg-scaled tf32 big/res split of the weight, laid out exactly like W ([P][U][V]) because dgrad's
/// B operand is `B[n=u, k=v] = cg_p * W[p,u,v]` -- W's own layout, no transpose.
void pack_dgrad_weight(float const *w, float *big, float *res, CgVals cg, int P, int U, int V,
                       cudaStream_t stream) {
  launch_pack_w_dgrad(w, big, res, cg, P, U * V, stream);
}

}  // namespace

struct DgradRunner::Impl_ {
  int B, I, U, P, V, J;
  GroupTable table;
  std::vector<RowPlan> rows;
  DeviceBuffer d_b_big, d_b_res;
  std::unique_ptr<Sm90Launch> plan;

  Impl_(int B_, int I_, int U_, int P_, int V_, std::vector<int> const &i_dims, BwdImpl bimpl,
        float const *w, double const *cg, cudaStream_t stream)
      : B(B_), I(I_), U(U_), P(P_), V(V_), table(build_group_table(i_dims)) {
    J = table.out_total_i;
    if (B <= 0 || U <= 0 || V <= 0) throw std::invalid_argument("DgradRunner: bad B/U/V");
    if (table.num_paths() != P) throw std::invalid_argument("DgradRunner: P != len(i_dims)");
    if (I != table.in_prefix[P - 1] + table.i_dims[P - 1]) {
      throw std::invalid_argument("DgradRunner: I != sum(i_dims)");
    }
    if (bimpl != BwdImpl::kB1) throw std::invalid_argument("DgradRunner: unknown BwdImpl");
    if (!sm90_available()) throw std::runtime_error("DgradRunner: SM90 collective unavailable");
    long long const lda = static_cast<long long>(J) * V;  // dy row pitch
    long long const ldc = static_cast<long long>(I) * U;  // dx row pitch
    if (lda % kAlign || ldc % kAlign || V % kAlign || B % kAlign) {
      throw std::invalid_argument("DgradRunner: needs 4-aligned dy/dx strides and V/B");
    }

    rows = make_dgrad_rows(table, B, U, V);
    if (static_cast<int>(rows.size()) != I) throw std::logic_error("DgradRunner: row count != I");

    int const uv = U * V;
    if (d_b_big.reset(static_cast<size_t>(P) * uv * sizeof(float)) ||
        d_b_res.reset(static_cast<size_t>(P) * uv * sizeof(float))) {
      throw std::runtime_error("DgradRunner: device allocation failed");
    }
    CgVals cg_vals{};
    for (int p = 0; p < P; ++p) cg_vals.v[p] = cg ? static_cast<float>(cg[p]) : 1.f;
    launch_pack_w_dgrad(w, d_b_big.as<float>(), d_b_res.as<float>(), cg_vals, P, uv, stream);

    // B1: tf32x3, A from registers.  The operand set and the numerics are the forward's impl 5, but
    // the backward is numbered on its own B-series (see BwdImpl) and must not follow the forward's.
    Sm90Operands const operands = Sm90Operands::kFp32RegSplit;
    // N = U here, not V: the contraction is over V.  A is bound per call (dy changes), so the plan
    // is constructed without one and `bind_a` supplies it.
    plan = std::make_unique<Sm90GroupedPlan>(operands, B, U, I, lda, V, ldc, rows, nullptr,
                                             static_cast<void const *>(d_b_big.as<float>()),
                                             static_cast<void const *>(d_b_res.as<float>()));
  }
};

DgradRunner::DgradRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims,
                         BwdImpl bimpl, float const *w, double const *cg, cudaStream_t stream)
    : p_(new Impl_(B, I, U, P, V, i_dims, bimpl, w, cg, stream)) {}

DgradRunner::~DgradRunner() = default;

int DgradRunner::run(float const *dy, float *dx, cudaStream_t stream) {
  p_->plan->bind_a(dy, stream);
  return p_->plan->launch(dx, stream);
}

// ---------------------------------------------------------------------------------------------
// B2: wgrad
// ---------------------------------------------------------------------------------------------

struct WgradRunner::Impl_ {
  int B, I, U, P, V, J, splits;
  GroupTable table;
  std::vector<RowPlan> rows;
  CgVals cg{};

  // Per-call preparation targets.  `Xt` is [P][U, K_p] concatenated, `Yt` is [G][V, K_g] in two
  // tf32 halves (the mainloop splits A itself but B must arrive pre-split).
  DeviceBuffer d_xt, d_yt_big, d_yt_res, d_partials, d_paths_x, d_paths_y;
  int k_tiles_x = 0, k_tiles_y = 0, g_count = 0;
  std::vector<int> h_splits;         // per path
  std::vector<long long> h_part_off; // per path, in units of U*V
  DeviceBuffer d_sp, d_off;
  std::unique_ptr<Sm90Launch> plan;

  Impl_(int B_, int I_, int U_, int P_, int V_, std::vector<int> const &i_dims, BwdImpl bimpl,
        int splits_, double const *cg_in, cudaStream_t stream)
      : B(B_), I(I_), U(U_), P(P_), V(V_), table(build_group_table(i_dims)) {
    J = table.out_total_i;
    if (B <= 0 || U <= 0 || V <= 0) throw std::invalid_argument("WgradRunner: bad B/U/V");
    if (table.num_paths() != P) throw std::invalid_argument("WgradRunner: P != len(i_dims)");
    if (I != table.in_prefix[P - 1] + table.i_dims[P - 1]) {
      throw std::invalid_argument("WgradRunner: I != sum(i_dims)");
    }
    if (bimpl != BwdImpl::kB2) throw std::invalid_argument("WgradRunner: unknown BwdImpl");
    if (!sm90_available()) throw std::runtime_error("WgradRunner: SM90 collective unavailable");
    for (int p = 0; p < P; ++p) cg.v[p] = cg_in ? static_cast<float>(cg_in[p]) : 1.f;

    WgradPaths const paths = make_wgrad_paths(table, B, U, V);
    WgradRows const wr = make_wgrad_rows(table, B, U, V, splits_, device_sm_count(), paths);
    rows = wr.rows;
    splits = wr.max_splits;
    h_splits = wr.splits;
    h_part_off = wr.part_off;
    g_count = static_cast<int>(paths.y.size());
    k_tiles_x = paths.k_tiles_x;
    k_tiles_y = paths.k_tiles_y;

    size_t const n_xt = static_cast<size_t>(B) * I * U;
    size_t const n_yt = static_cast<size_t>(B) * J * V;
    size_t const n_part = static_cast<size_t>(wr.n_partials) * U * V;
    if (d_xt.reset(n_xt * sizeof(float)) || d_yt_big.reset(n_yt * sizeof(float)) ||
        d_yt_res.reset(n_yt * sizeof(float)) || d_partials.reset(n_part * sizeof(float)) ||
        d_sp.reset(P * sizeof(int)) || d_off.reset(P * sizeof(long long)) ||
        d_paths_x.reset(paths.x.size() * sizeof(KtPath)) ||
        d_paths_y.reset(paths.y.size() * sizeof(KtPath))) {
      throw std::runtime_error("WgradRunner: device allocation failed");
    }
    cudaMemcpy(d_sp.as<int>(), h_splits.data(), P * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off.as<long long>(), h_part_off.data(), P * sizeof(long long),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_paths_x.as<KtPath>(), paths.x.data(), paths.x.size() * sizeof(KtPath),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_paths_y.as<KtPath>(), paths.y.data(), paths.y.size() * sizeof(KtPath),
               cudaMemcpyHostToDevice);

    // M = U, N = V; every row carries its own lda/ldb/c_off (K differs per path).
    plan = std::make_unique<Sm90GroupedPlan>(Sm90Operands::kFp32RegSplit, U, V,
                                             static_cast<int>(rows.size()), 0, 0, V, rows,
                                             static_cast<void const *>(d_xt.as<float>()),
                                             static_cast<void const *>(d_yt_big.as<float>()),
                                             static_cast<void const *>(d_yt_res.as<float>()));
  }

  int run(float const *x, float const *dy, float *dw, cudaStream_t stream) {
    launch_transpose_kt(x, d_xt.as<float>(), nullptr, d_paths_x.as<KtPath>(), P, B, U,
                        static_cast<long long>(I) * U, k_tiles_x, stream);
    launch_transpose_kt(dy, d_yt_big.as<float>(), d_yt_res.as<float>(), d_paths_y.as<KtPath>(),
                        g_count, B, V, static_cast<long long>(J) * V, k_tiles_y, stream);
    // Diagnostic only: stop after the prep so the harness can attribute the e2e cost.
    if (char const *po = nullptr; po != nullptr && atoi(po) != 0) return 0;
    int rc = plan->launch(d_partials.as<float>(), stream);
    if (rc != 0) return rc;
    launch_reduce_dw(d_partials.as<float>(), cg, dw, U * V, P, d_sp.as<int>(),
                    d_off.as<long long>(), stream);
    return 0;
  }
};

WgradRunner::WgradRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims,
                         BwdImpl bimpl, int splits, double const *cg, cudaStream_t stream)
    : p_(new Impl_(B, I, U, P, V, i_dims, bimpl, splits, cg, stream)) {}

WgradRunner::~WgradRunner() = default;

int WgradRunner::splits() const { return p_->splits; }

int WgradRunner::run(float const *x, float const *dy, float *dw, cudaStream_t stream) {
  return p_->run(x, dy, dw, stream);
}

// ---------------------------------------------------------------------------------------------
// B3: the fused backward
// ---------------------------------------------------------------------------------------------

struct BackwardRunner::Impl_ {
  int B, I, U, P, V, J, splits, g_count = 0;
  GroupTable table;
  CgVals cg{};
  WgradPaths paths;

  // dgrad's B (cg*W split) and wgrad's operands + partials.  Both live for the handle's lifetime.
  DeviceBuffer d_w_big, d_w_res;
  DeviceBuffer d_xt, d_yt_big, d_yt_res, d_partials, d_paths_x, d_paths_y, d_sp, d_off;
  int k_tiles_x = 0, k_tiles_y = 0;
  std::vector<int> h_splits;
  std::vector<long long> h_part_off;
  std::unique_ptr<Sm90Launch> plan;

  Impl_(int B_, int I_, int U_, int P_, int V_, std::vector<int> const &i_dims, BwdImpl bimpl,
        int splits_, float const *w, double const *cg_in, cudaStream_t stream)
      : B(B_), I(I_), U(U_), P(P_), V(V_), table(build_group_table(i_dims)) {
    J = table.out_total_i;
    if (B <= 0 || U <= 0 || V <= 0) throw std::invalid_argument("BackwardRunner: bad B/U/V");
    if (table.num_paths() != P) throw std::invalid_argument("BackwardRunner: P != len(i_dims)");
    if (I != table.in_prefix[P - 1] + table.i_dims[P - 1]) {
      throw std::invalid_argument("BackwardRunner: I != sum(i_dims)");
    }
    if (bimpl != BwdImpl::kB3) throw std::invalid_argument("BackwardRunner: unknown BwdImpl");
    if (U != V) throw std::invalid_argument("BackwardRunner: the fused plan needs U == V");
    if (!sm90_available()) throw std::runtime_error("BackwardRunner: SM90 collective unavailable");
    if (static_cast<long long>(J) * V % kAlign || static_cast<long long>(I) * U % kAlign ||
        V % kAlign || B % kAlign) {
      throw std::invalid_argument("BackwardRunner: needs 4-aligned dy/dx strides and V/B");
    }
    for (int p = 0; p < P; ++p) cg.v[p] = cg_in ? static_cast<float>(cg_in[p]) : 1.f;

    paths = make_wgrad_paths(table, B, U, V);
    WgradRows const wr = make_wgrad_rows(table, B, U, V, splits_, device_sm_count(), paths);
    splits = wr.max_splits;
    h_splits = wr.splits;
    h_part_off = wr.part_off;
    g_count = static_cast<int>(paths.y.size());
    k_tiles_x = paths.k_tiles_x;
    k_tiles_y = paths.k_tiles_y;

    size_t const n_xt = static_cast<size_t>(B) * I * U;
    size_t const n_yt = static_cast<size_t>(B) * J * V;
    size_t const n_part = static_cast<size_t>(wr.n_partials) * U * V;
    if (d_w_big.reset(static_cast<size_t>(P) * U * V * sizeof(float)) ||
        d_w_res.reset(static_cast<size_t>(P) * U * V * sizeof(float)) ||
        d_xt.reset(n_xt * sizeof(float)) || d_yt_big.reset(n_yt * sizeof(float)) ||
        d_yt_res.reset(n_yt * sizeof(float)) || d_partials.reset(n_part * sizeof(float)) ||
        d_sp.reset(P * sizeof(int)) || d_off.reset(P * sizeof(long long)) ||
        d_paths_x.reset(paths.x.size() * sizeof(KtPath)) ||
        d_paths_y.reset(paths.y.size() * sizeof(KtPath))) {
      throw std::runtime_error("BackwardRunner: device allocation failed");
    }
    cudaMemcpy(d_sp.as<int>(), h_splits.data(), P * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off.as<long long>(), h_part_off.data(), P * sizeof(long long),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_paths_x.as<KtPath>(), paths.x.data(), paths.x.size() * sizeof(KtPath),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_paths_y.as<KtPath>(), paths.y.data(), paths.y.size() * sizeof(KtPath),
               cudaMemcpyHostToDevice);
    pack_dgrad_weight(w, d_w_big.as<float>(), d_w_res.as<float>(), cg, P, U, V, stream);

    // One row table: wgrad's (path, split) problems first -- they are the longer tasks, so putting
    // them first is the shortest-tail order -- then dgrad's (group, path, component) ones.
    std::vector<RowPlan> rows = wr.rows;
    for (RowPlan &r : rows) {
      r.a_base = d_xt.as<float>();
      r.b_base = d_yt_big.as<float>();
      r.b_res_base = d_yt_res.as<float>();
      r.c_base = d_partials.as<float>();
    }
    std::vector<RowPlan> const dx_rows = make_dgrad_rows(table, B, U, V);
    for (RowPlan r : dx_rows) {
      // dgrad's A is the caller's dy: leave `a_base` null so `bind_a` supplies it every call.
      r.b_base = d_w_big.as<float>();
      r.b_res_base = d_w_res.as<float>();
      r.c_base = nullptr;  // the C tile lives in the caller's dx, passed to `launch`
      r.lda = static_cast<long long>(J) * V;  // the plan-wide lda/ldb are 0 here
      r.ldb = V;
      r.ldc = static_cast<long long>(I) * U;
      rows.push_back(r);
    }

    plan = std::make_unique<Sm90GroupedPlan>(Sm90Operands::kFp32RegSplit, U, V,
                                             static_cast<int>(rows.size()), 0, 0, V, rows,
                                             nullptr,
                                             static_cast<void const *>(d_w_big.as<float>()),
                                             static_cast<void const *>(d_w_res.as<float>()));
  }

  int run(float const *x, float const *dy, float *dx, float *dw, cudaStream_t stream) {
    launch_transpose_kt(x, d_xt.as<float>(), nullptr, d_paths_x.as<KtPath>(), P, B, U,
                        static_cast<long long>(I) * U, k_tiles_x, stream);
    launch_transpose_kt(dy, d_yt_big.as<float>(), d_yt_res.as<float>(), d_paths_y.as<KtPath>(),
                        g_count, B, V, static_cast<long long>(J) * V, k_tiles_y, stream);
    // Diagnostic only: stop after the prep so the harness can attribute its cost.
    if (char const *po = nullptr; po != nullptr && atoi(po) != 0) return 0;
    // dgrad's rows follow this base; wgrad's carry their own (`a_base`), so binding is harmless to
    // them.  `c_base` is null only on the dgrad rows, which then write into `dx`.
    plan->bind_a(dy, stream);
    int rc = plan->launch(dx, stream);
    if (rc != 0) return rc;
    launch_reduce_dw(d_partials.as<float>(), cg, dw, U * V, P, d_sp.as<int>(),
                     d_off.as<long long>(), stream);
    return 0;
  }
};

BackwardRunner::BackwardRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims,
                               BwdImpl bimpl, int splits, float const *w, double const *cg,
                               cudaStream_t stream)
    : p_(new Impl_(B, I, U, P, V, i_dims, bimpl, splits, w, cg, stream)) {}

BackwardRunner::~BackwardRunner() = default;

int BackwardRunner::splits() const { return p_->splits; }

int BackwardRunner::run(float const *x, float const *dy, float *dx, float *dw,
                        cudaStream_t stream) {
  return p_->run(x, dy, dx, dw, stream);
}

}  // namespace eqlin
