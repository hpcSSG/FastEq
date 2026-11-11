/***************************************************************************************************
 * fasteq eqlinear backends/tf32/dgrad.cu
 *
 * Production B1 dgrad (dx) only.
 **************************************************************************************************/
#include "bwd.h"

#include <stdexcept>
#include <vector>

#include "cutlass/numeric_conversion.h"

#include "device_buffer.h"
#include "eqlinear_group_table.h"
#include "prep_kernels.h"  // CgVals
#include "row_plan.h"
#include "sm90_grouped_gemm.cuh"

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

/// dgrad: one problem per (group, path, component).  (g,t,m) -> i is a bijection, so each x row is
/// written by exactly one problem and the K axis stays V (no accumulation, no scatter).
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

/// cg-scaled tf32 big/res split of the weight, laid out exactly like W ([P][U][V]).
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

    // B1: tf32x3, A from registers.  N = U here, not V: the contraction is over V.
    Sm90Operands const operands = Sm90Operands::kFp32RegSplit;
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

}  // namespace eqlin
