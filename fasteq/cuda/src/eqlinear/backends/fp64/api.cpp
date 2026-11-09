// C ABI adapter for the FP64 (DMMA tensor-core) eqlinear backend.
//
// Frozen dispatch: forward impl 5 (kAuto3: V<=128 -> direct, else multi-path -> SM90 DMMA,
// else 2.x grouped), backward dgrad-only B1.

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

int forward_create(const EqLinearProblemC* problem, const void* w, const double* cg,
                   cudaStream_t stream, void** plan, EqLinearPlanInfo* info) {
  if (problem == nullptr || w == nullptr || plan == nullptr || info == nullptr) {
    return static_cast<int>(Status::kInvalidArgument);
  }
  *plan = nullptr;
  try {
    std::vector<int> i_dims(problem->i_dims, problem->i_dims + problem->P);
    auto owner = std::make_unique<ForwardPlan>();
    owner->runner = std::make_unique<eqlin::Runner>(problem->B, problem->I, problem->U, problem->P, problem->V,
                                    i_dims, eqlin::F64Impl::kAuto4);
    auto* runner = owner->runner.get();
    const int rc = runner->prepare_weight(static_cast<const double*>(w), cg, stream);
    if (rc != 0) {
      return rc;
    }
    info->precision = 2;
    info->forward_impl = static_cast<int>(eqlin::F64Impl::kAuto4);
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
  const int prep = p->runner->prepare_act(static_cast<const double*>(x), stream);
  if (prep != 0) {
    return prep;
  }
  return p->runner->run(static_cast<double*>(out), stream);
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
    std::vector<double> cg_vec(cg, cg + problem->P);
    auto owner = std::make_unique<DgradPlan>();
    owner->runner = std::make_unique<eqlin::DgradRunner>(
        problem->B, problem->I, problem->U, problem->P, problem->V, i_dims,
        static_cast<const double*>(w), cg_vec, stream);
    info->precision = 2;
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
  return p->runner->run(static_cast<const double*>(dy), static_cast<double*>(dx), stream);
}

void dgrad_destroy(void* plan) {
  delete static_cast<DgradPlan*>(plan);
}

const char* status_string(int code) {
  switch (code) {
    case static_cast<int>(Status::kOk):
      return "eqlinear fp64: ok";
    case static_cast<int>(Status::kUnsupported):
      return "eqlinear fp64: unsupported configuration";
    case static_cast<int>(Status::kInvalidArgument):
      return "eqlinear fp64: invalid argument";
    case static_cast<int>(Status::kCudaError):
      return "eqlinear fp64: CUDA error";
    case static_cast<int>(Status::kInternalError):
      return "eqlinear fp64: internal error";
    default:
      return eqlin::status_string(code);
  }
}

const EqLinearBackendApi kApi = {
    fasteq::eqlinear::kEqLinearAbiVersion,
    2,
    &forward_create,
    &forward_run,
    &forward_destroy,
    &dgrad_create,
    &dgrad_run,
    &dgrad_destroy,
    &status_string,
};

}  // namespace

FASTEQ_EQLINEAR_EXPORT const EqLinearBackendApi* fasteq_eqlinear_fp64_api() { return &kApi; }
