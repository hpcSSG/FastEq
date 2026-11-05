// C ABI adapter for the FP32 (CUDA-core FFMA) eqlinear backend.
//
// Frozen dispatch: forward impl 3 (kFused, TMA mainloop, zero-copy A), backward B1 (dgrad only).

#include <exception>
#include <vector>

#include "bwd.h"
#include "eqlinear.h"
#include "eqlinear_backend_api.h"

namespace {

using fasteq::eqlinear::EqLinearBackendApi;
using fasteq::eqlinear::EqLinearPlanInfo;
using fasteq::eqlinear::EqLinearProblemC;
using fasteq::eqlinear::Status;

struct ForwardPlan {
  eqlin_fp32::Runner* runner = nullptr;
};

struct DgradPlan {
  eqlin_fp32::BackwardRunner* runner = nullptr;
};

int forward_create(const EqLinearProblemC* problem, const void* w, const double* cg,
                   cudaStream_t stream, void** plan, EqLinearPlanInfo* info) {
  if (problem == nullptr || w == nullptr || plan == nullptr || info == nullptr) {
    return static_cast<int>(Status::kInvalidArgument);
  }
  try {
    std::vector<int> i_dims(problem->i_dims, problem->i_dims + problem->P);
    auto runner = new eqlin_fp32::Runner(problem->B, problem->I, problem->U, problem->P,
                                         problem->V, i_dims, eqlin_fp32::Impl::kFused);
    const int rc = runner->prepare_weight(static_cast<const float*>(w), cg, stream);
    if (rc != 0) {
      delete runner;
      return rc;
    }
    *plan = new ForwardPlan{runner};
    info->precision = 1;
    info->forward_impl = static_cast<int>(eqlin_fp32::Impl::kFused);
    info->dgrad_impl = static_cast<int>(eqlin_fp32::BwdImpl::kB1);
    info->a_view = 0;
    info->j_rows = runner->j_rows();
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
  return p->runner->run_act(static_cast<const float*>(x), static_cast<float*>(out), stream);
}

void forward_destroy(void* plan) {
  auto* p = static_cast<ForwardPlan*>(plan);
  if (p != nullptr) {
    delete p->runner;
    delete p;
  }
}

int dgrad_create(const EqLinearProblemC* problem, const void* w, const double* cg,
                 cudaStream_t stream, void** plan, EqLinearPlanInfo* info) {
  if (problem == nullptr || w == nullptr || plan == nullptr || info == nullptr) {
    return static_cast<int>(Status::kInvalidArgument);
  }
  try {
    std::vector<int> i_dims(problem->i_dims, problem->i_dims + problem->P);
    auto runner = new eqlin_fp32::BackwardRunner(
        problem->B, problem->I, problem->U, problem->P, problem->V, i_dims,
        eqlin_fp32::BwdImpl::kB1, static_cast<const float*>(w), cg, stream);
    *plan = new DgradPlan{runner};
    info->precision = 1;
    info->forward_impl = -1;
    info->dgrad_impl = static_cast<int>(eqlin_fp32::BwdImpl::kB1);
    info->a_view = 0;
    info->j_rows = runner->j_rows();
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
  return p->runner->run(nullptr, static_cast<const float*>(dy), static_cast<float*>(dx), nullptr,
                        stream);
}

void dgrad_destroy(void* plan) {
  auto* p = static_cast<DgradPlan*>(plan);
  if (p != nullptr) {
    delete p->runner;
    delete p;
  }
}

const char* status_string(int code) {
  switch (code) {
    case static_cast<int>(Status::kOk):
      return "eqlinear fp32: ok";
    case static_cast<int>(Status::kUnsupported):
      return "eqlinear fp32: unsupported configuration";
    case static_cast<int>(Status::kInvalidArgument):
      return "eqlinear fp32: invalid argument";
    case static_cast<int>(Status::kCudaError):
      return "eqlinear fp32: CUDA error";
    case static_cast<int>(Status::kInternalError):
      return "eqlinear fp32: internal error";
    default:
      return eqlin_fp32::status_string(code);
  }
}

const EqLinearBackendApi kApi = {
    fasteq::eqlinear::kEqLinearAbiVersion,
    1,
    &forward_create,
    &forward_run,
    &forward_destroy,
    &dgrad_create,
    &dgrad_run,
    &dgrad_destroy,
    &status_string,
};

}  // namespace

FASTEQ_EQLINEAR_EXPORT const EqLinearBackendApi* fasteq_eqlinear_fp32_api() { return &kApi; }
