// C ABI adapter for the FP32 (CUDA-core FFMA) eqlinear backend.
//
// Frozen dispatch: forward impl 3 (kFused, TMA mainloop, zero-copy A), backward B1 (dgrad only).

#include <memory>
#include <vector>

#include "backend_adapter.h"
#include "bwd.h"
#include "eqlinear.h"
#include "eqlinear_backend_api.h"

namespace {

using fasteq::eqlinear::EqLinearBackendApi;
using fasteq::eqlinear::EqLinearProblemC;

struct Fp32Traits {
  using Scalar = float;
  using ForwardRunner = eqlin_fp32::Runner;
  using DgradRunner = eqlin_fp32::BackwardRunner;

  static constexpr int kPrecision = 1;
  static constexpr int kDgradImpl = static_cast<int>(eqlin_fp32::BwdImpl::kB1);
  static constexpr char const *kStatusMessages[5] = {
      "eqlinear fp32: ok", "eqlinear fp32: unsupported configuration",
      "eqlinear fp32: invalid argument", "eqlinear fp32: CUDA error",
      "eqlinear fp32: internal error"};

  static std::unique_ptr<ForwardRunner> make_forward(EqLinearProblemC const &p,
                                                     std::vector<int> const &dims, void const *,
                                                     double const *) {
    return std::make_unique<ForwardRunner>(p.B, p.I, p.U, p.P, p.V, dims,
                                           eqlin_fp32::Impl::kFused);
  }

  static int forward_impl(EqLinearProblemC const &) {
    return static_cast<int>(eqlin_fp32::Impl::kFused);
  }

  static bool a_view(ForwardRunner const &) { return false; }

  static int forward_run(ForwardRunner &runner, void const *x, void *out, cudaStream_t stream) {
    return runner.run_act(static_cast<Scalar const *>(x), static_cast<Scalar *>(out), stream);
  }

  static std::unique_ptr<DgradRunner> make_dgrad(EqLinearProblemC const &p,
                                                 std::vector<int> const &dims, void const *w,
                                                 double const *cg, cudaStream_t stream) {
    return std::make_unique<DgradRunner>(p.B, p.I, p.U, p.P, p.V, dims,
                                         eqlin_fp32::BwdImpl::kB1,
                                         static_cast<Scalar const *>(w), cg, stream);
  }

  static int dgrad_run(DgradRunner &runner, void const *dy, void *dx, cudaStream_t stream) {
    return runner.run(nullptr, static_cast<Scalar const *>(dy), static_cast<Scalar *>(dx), nullptr,
                      stream);
  }

  static char const *backend_status_string(int code) { return eqlin_fp32::status_string(code); }
};

}  // namespace

FASTEQ_EQLINEAR_EXPORT const EqLinearBackendApi *fasteq_eqlinear_fp32_api() {
  return &fasteq::eqlinear::BackendAdapter<Fp32Traits>::api();
}
