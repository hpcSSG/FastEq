// C ABI adapter for the TF32 (tf32x3 tensor-core) eqlinear backend.
//
// The frozen dispatch rule is the measured best from the TF32 line:
//   * four-path, small-batch, narrow-N shapes: impl 0 (CUTLASS 2.x SM80 tf32x3)
//   * everything else:                        impl 5 (SM90 RS tf32x3)
// Backward is always the dgrad-only B1 path.

#include <exception>
#include <vector>

#include "bwd.h"
#include "eqlinear.h"
#include "eqlinear_backend_api.h"
#include "device_lifetime.h"

namespace {

using fasteq::eqlinear::EqLinearBackendApi;
using fasteq::eqlinear::EqLinearPlanInfo;
using fasteq::eqlinear::EqLinearProblemC;
using fasteq::eqlinear::Status;

using ForwardPlan = fasteq::eqlinear::DeviceOwner<eqlin::Runner>;

using DgradPlan = fasteq::eqlinear::DeviceOwner<eqlin::DgradRunner>;

eqlin::Impl select_forward_impl(const EqLinearProblemC& p) {
  if (p.P == 4 && p.B <= 1024 && p.V <= 128) {
    return eqlin::Impl::k3xTF32;
  }
  return eqlin::Impl::kSm90_3xTF32RegSplit;
}

int forward_create(const EqLinearProblemC* problem, const void* w, const double* cg,
                   cudaStream_t stream, void** plan, EqLinearPlanInfo* info) {
  if (problem == nullptr || w == nullptr || plan == nullptr || info == nullptr) {
    return static_cast<int>(Status::kInvalidArgument);
  }
  *plan = nullptr;
  try {
    std::vector<int> i_dims(problem->i_dims, problem->i_dims + problem->P);
    const auto impl = select_forward_impl(*problem);
    auto owner = std::make_unique<ForwardPlan>();
    owner->runner = std::make_unique<eqlin::Runner>(problem->B, problem->I, problem->U, problem->P, problem->V,
                                    i_dims, impl);
    auto* runner = owner->runner.get();
    const int rc = runner->prepare_weight(static_cast<const float*>(w), cg, stream);
    if (rc != 0) {
      return rc;
    }
    info->precision = 0;
    info->forward_impl = static_cast<int>(impl);
    info->dgrad_impl = static_cast<int>(eqlin::BwdImpl::kB1);
    info->a_view = runner->a_view() ? 1 : 0;
    info->j_rows = runner->j_rows();
    *plan = owner.release();
    return 0;
  } catch (const std::exception&) {
    return static_cast<int>(Status::kInvalidArgument);
  }
}

int forward_run(void* plan, const void* x, void* out, cudaStream_t stream) {
  if (plan == nullptr || x == nullptr || out == nullptr) {
    return static_cast<int>(Status::kInvalidArgument);
  }
  auto* p = static_cast<ForwardPlan*>(plan);
  const int prep = p->runner->prepare_act(static_cast<const float*>(x), stream);
  if (prep != 0) {
    return prep;
  }
  return p->runner->run(static_cast<float*>(out), stream);
}

void forward_destroy(void* plan) {
  delete static_cast<ForwardPlan*>(plan);
}

int dgrad_create(const EqLinearProblemC* problem, const void* w, const double* cg,
                 cudaStream_t stream, void** plan, EqLinearPlanInfo* info) {
  if (problem == nullptr || w == nullptr || plan == nullptr || info == nullptr) {
    return static_cast<int>(Status::kInvalidArgument);
  }
  *plan = nullptr;
  try {
    std::vector<int> i_dims(problem->i_dims, problem->i_dims + problem->P);
    auto owner = std::make_unique<DgradPlan>();
    owner->runner = std::make_unique<eqlin::DgradRunner>(
        problem->B, problem->I, problem->U, problem->P, problem->V, i_dims,
        eqlin::BwdImpl::kB1, static_cast<const float*>(w), cg, stream);
    info->precision = 0;
    info->forward_impl = -1;
    info->dgrad_impl = static_cast<int>(eqlin::BwdImpl::kB1);
    info->a_view = 0;
    info->j_rows = eqlin::out_total_i(i_dims);
    *plan = owner.release();
    return 0;
  } catch (const std::exception&) {
    return static_cast<int>(Status::kInvalidArgument);
  }
}

int dgrad_run(void* plan, const void* dy, void* dx, cudaStream_t stream) {
  if (plan == nullptr || dy == nullptr || dx == nullptr) {
    return static_cast<int>(Status::kInvalidArgument);
  }
  auto* p = static_cast<DgradPlan*>(plan);
  return p->runner->run(static_cast<const float*>(dy), static_cast<float*>(dx), stream);
}

void dgrad_destroy(void* plan) {
  delete static_cast<DgradPlan*>(plan);
}

const char* status_string(int code) {
  switch (code) {
    case static_cast<int>(Status::kOk):
      return "eqlinear tf32: ok";
    case static_cast<int>(Status::kUnsupported):
      return "eqlinear tf32: unsupported configuration";
    case static_cast<int>(Status::kInvalidArgument):
      return "eqlinear tf32: invalid argument";
    case static_cast<int>(Status::kCudaError):
      return "eqlinear tf32: CUDA error";
    case static_cast<int>(Status::kInternalError):
      return "eqlinear tf32: internal error";
    default:
      return eqlin::status_string(code);
  }
}

const EqLinearBackendApi kApi = {
    fasteq::eqlinear::kEqLinearAbiVersion,
    0,
    &forward_create,
    &forward_run,
    &forward_destroy,
    &dgrad_create,
    &dgrad_run,
    &dgrad_destroy,
    &status_string,
};

}  // namespace

FASTEQ_EQLINEAR_EXPORT const EqLinearBackendApi* fasteq_eqlinear_tf32_api() { return &kApi; }
