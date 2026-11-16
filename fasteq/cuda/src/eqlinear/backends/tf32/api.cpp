// C ABI adapter for the TF32 (tf32x3 tensor-core) eqlinear backend.
//
// Frozen dispatch rule (measured best from the TF32 line):
//   * four-path, small-batch, narrow-N shapes: impl 0 (CUTLASS 2.x SM80 tf32x3)
//   * everything else:                        impl 5 (SM90 RS tf32x3)
// Backward is always the dgrad-only B1 path.

#include <memory>
#include <vector>

#include "backend_adapter.h"
#include "bwd.h"
#include "eqlinear.h"
#include "eqlinear_backend_api.h"
#include "shape_plan.cuh"
#include "single_path_plan.cuh"

namespace {

using fasteq::eqlinear::EqLinearBackendApi;
using fasteq::eqlinear::EqLinearProblemC;

struct Tf32Traits {
  using Scalar = float;
  using ForwardRunner = eqlin::Runner;
  using DgradRunner = eqlin::DgradRunner;

  static constexpr int kPrecision = 0;
  static constexpr int kDgradImpl = static_cast<int>(eqlin::BwdImpl::kB1);
  static constexpr char const *kStatusMessages[5] = {
      "eqlinear tf32: ok", "eqlinear tf32: unsupported configuration",
      "eqlinear tf32: invalid argument", "eqlinear tf32: CUDA error",
      "eqlinear tf32: internal error"};

  static bool use_single_path_plan(EqLinearProblemC const& p, bool dgrad) {
    // Scalar forward readout keeps the established register-only x3 shape
    // kernel. The shared-memory P1 engine regressed this measured shape class.
    if (p.P == 1 && p.V == 1 && p.U >= 128 && !dgrad) return false;
    // Same-lease eager and graph measurements retain the original Hopper plan
    // for this full-contiguous shape: both experimental P1 engines were slower.
    if (p.P == 1 && !p.paths && p.B == 25116 && p.I == 1 && p.J == 1 &&
        p.U == 224 && p.V == 224)
      return false;
    return fasteq::eqlinear::tf32_single::required(p, dgrad);
  }
  static std::unique_ptr<fasteq::eqlinear::ShapePlan> make_single_path_plan(
      EqLinearProblemC const& p, void const* w, double const* cg, bool dgrad,
      cudaStream_t stream) {
    return fasteq::eqlinear::tf32_single::create(p, w, cg, dgrad, stream);
  }

  static bool use_shape_plan(EqLinearProblemC const &p) {
    return fasteq::eqlinear::tf32_shape::required(p);
  }

  static std::unique_ptr<fasteq::eqlinear::ShapePlan> make_shape_plan(
      EqLinearProblemC const &p, void const *w, double const *cg, bool dgrad,
      cudaStream_t stream) {
    return fasteq::eqlinear::tf32_shape::create(p, w, cg, dgrad, stream);
  }

  static eqlin::Impl select_impl(EqLinearProblemC const &p) {
    if (p.P == 4 && p.B <= 1024 && p.V <= 128) return eqlin::Impl::k3xTF32;
    return eqlin::Impl::kSm90_3xTF32RegSplit;
  }

  static std::unique_ptr<ForwardRunner> make_forward(EqLinearProblemC const &p,
                                                     std::vector<int> const &dims, void const *,
                                                     double const *) {
    return std::make_unique<ForwardRunner>(p.B, p.I, p.U, p.P, p.V, dims, select_impl(p));
  }

  static int forward_impl(EqLinearProblemC const &p) {
    return static_cast<int>(select_impl(p));
  }

  static bool a_view(ForwardRunner const &runner) { return runner.a_view(); }

  static int forward_run(ForwardRunner &runner, void const *x, void *out, cudaStream_t stream) {
    const int rc = runner.prepare_act(static_cast<Scalar const *>(x), stream);
    if (rc != 0) return rc;
    return runner.run(static_cast<Scalar *>(out), stream);
  }

  static std::unique_ptr<DgradRunner> make_dgrad(EqLinearProblemC const &p,
                                                 std::vector<int> const &dims, void const *w,
                                                 double const *cg, cudaStream_t stream) {
    return std::make_unique<DgradRunner>(p.B, p.I, p.U, p.P, p.V, dims, eqlin::BwdImpl::kB1,
                                         static_cast<Scalar const *>(w), cg, stream);
  }

  static int dgrad_run(DgradRunner &runner, void const *dy, void *dx, cudaStream_t stream) {
    return runner.run(static_cast<Scalar const *>(dy), static_cast<Scalar *>(dx), stream);
  }

  static char const *backend_status_string(int code) { return eqlin::status_string(code); }
};

}  // namespace

FASTEQ_EQLINEAR_EXPORT const EqLinearBackendApi *fasteq_eqlinear_tf32_api() {
  return &fasteq::eqlinear::BackendAdapter<Tf32Traits>::api();
}
