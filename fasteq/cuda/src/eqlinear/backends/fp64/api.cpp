// C ABI adapter for the FP64 (DMMA tensor-core) eqlinear backend.
//
// Frozen dispatch: forward impl 6 (kAuto4: TMA-fed hand-written kernel, exact-N tile where
// useful), backward dgrad-only B1.

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

struct Fp64Traits {
  using Scalar = double;
  using ForwardRunner = eqlin::Runner;
  using DgradRunner = eqlin::DgradRunner;

  static constexpr int kPrecision = 2;
  static constexpr int kDgradImpl = static_cast<int>(eqlin::BwdImpl::kB1);
  static constexpr char const *kStatusMessages[5] = {
      "eqlinear fp64: ok", "eqlinear fp64: unsupported configuration",
      "eqlinear fp64: invalid argument", "eqlinear fp64: CUDA error",
      "eqlinear fp64: internal error"};

  static bool use_single_path_plan(EqLinearProblemC const& p, bool dgrad) {
    return fasteq::eqlinear::fp64_single::required(p, dgrad);
  }
  static std::unique_ptr<fasteq::eqlinear::ShapePlan> make_single_path_plan(
      EqLinearProblemC const& p, void const* w, double const* cg, bool dgrad,
      cudaStream_t stream) {
    return fasteq::eqlinear::fp64_single::create(p, w, cg, dgrad, stream);
  }

  static bool use_shape_plan(EqLinearProblemC const &p) {
    return fasteq::eqlinear::fp64_shape::required(p);
  }

  static std::unique_ptr<fasteq::eqlinear::ShapePlan> make_shape_plan(
      EqLinearProblemC const &p, void const *w, double const *cg, bool dgrad,
      cudaStream_t stream) {
    return fasteq::eqlinear::fp64_shape::create(p, w, cg, dgrad, stream);
  }

  static std::unique_ptr<ForwardRunner> make_forward(EqLinearProblemC const &p,
                                                     std::vector<int> const &dims, void const *,
                                                     double const *) {
    return std::make_unique<ForwardRunner>(p.B, p.I, p.U, p.P, p.V, dims,
                                           eqlin::F64Impl::kAuto4);
  }

  static int forward_impl(EqLinearProblemC const &) {
    return static_cast<int>(eqlin::F64Impl::kAuto4);
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
    std::vector<double> cg_vec;
    if (cg != nullptr) cg_vec.assign(cg, cg + p.P);
    return std::make_unique<DgradRunner>(p.B, p.I, p.U, p.P, p.V, dims,
                                         static_cast<Scalar const *>(w), cg_vec, stream);
  }

  static int dgrad_run(DgradRunner &runner, void const *dy, void *dx, cudaStream_t stream) {
    return runner.run(static_cast<Scalar const *>(dy), static_cast<Scalar *>(dx), stream);
  }

  static char const *backend_status_string(int code) { return eqlin::status_string(code); }
};

}  // namespace

FASTEQ_EQLINEAR_EXPORT const EqLinearBackendApi *fasteq_eqlinear_fp64_api() {
  return &fasteq::eqlinear::BackendAdapter<Fp64Traits>::api();
}
