#pragma once

#include <exception>
#include <memory>
#include <vector>

#include "device_lifetime.h"
#include "eqlinear_backend_api.h"
#include "eqlinear_common.h"
#include "eqlinear_group_table.h"

namespace fasteq::eqlinear {

/// Shared implementation of the C ABI vtable.  Traits supply only what genuinely differs between
/// the precision backends: scalar type, runner types, implementation selection, run glue and
/// status labels.  Static (policy) dispatch keeps each backend's concrete CUDA types intact.
template <typename Traits>
struct BackendAdapter {
  using Scalar = typename Traits::Scalar;
  using ForwardRunner = typename Traits::ForwardRunner;
  using DgradRunner = typename Traits::DgradRunner;
  using ForwardPlan = DeviceOwner<ForwardRunner>;
  using DgradPlan = DeviceOwner<DgradRunner>;

  static int forward_create(const EqLinearProblemC *problem, const void *w, const double *cg,
                            cudaStream_t stream, void **plan, EqLinearPlanInfo *info) {
    if (problem == nullptr || w == nullptr || plan == nullptr || info == nullptr) {
      return static_cast<int>(Status::kInvalidArgument);
    }
    *plan = nullptr;
    try {
      std::vector<int> i_dims(problem->i_dims, problem->i_dims + problem->P);
      auto owner = std::make_unique<ForwardPlan>();
      owner->runner = Traits::make_forward(*problem, i_dims, w, cg);
      auto *runner = owner->runner.get();
      const int rc = runner->prepare_weight(static_cast<Scalar const *>(w), cg, stream);
      if (rc != 0) return rc;
      info->precision = Traits::kPrecision;
      info->forward_impl = Traits::forward_impl(*problem);
      info->dgrad_impl = Traits::kDgradImpl;
      info->a_view = Traits::a_view(*runner) ? 1 : 0;
      info->j_rows = out_total_i(i_dims);
      *plan = owner.release();
      return 0;
    } catch (std::exception const &) {
      return static_cast<int>(Status::kInvalidArgument);
    }
  }

  static int forward_run(void *plan, const void *x, void *out, cudaStream_t stream) {
    if (plan == nullptr || x == nullptr || out == nullptr) {
      return static_cast<int>(Status::kInvalidArgument);
    }
    auto *p = static_cast<ForwardPlan *>(plan);
    return Traits::forward_run(*p->runner, x, out, stream);
  }

  static void forward_destroy(void *plan) { delete static_cast<ForwardPlan *>(plan); }

  static int dgrad_create(const EqLinearProblemC *problem, const void *w, const double *cg,
                          cudaStream_t stream, void **plan, EqLinearPlanInfo *info) {
    if (problem == nullptr || w == nullptr || plan == nullptr || info == nullptr) {
      return static_cast<int>(Status::kInvalidArgument);
    }
    *plan = nullptr;
    try {
      std::vector<int> i_dims(problem->i_dims, problem->i_dims + problem->P);
      auto owner = std::make_unique<DgradPlan>();
      owner->runner = Traits::make_dgrad(*problem, i_dims, w, cg, stream);
      info->precision = Traits::kPrecision;
      info->forward_impl = -1;
      info->dgrad_impl = Traits::kDgradImpl;
      info->a_view = 0;
      info->j_rows = out_total_i(i_dims);
      *plan = owner.release();
      return 0;
    } catch (std::exception const &) {
      return static_cast<int>(Status::kInvalidArgument);
    }
  }

  static int dgrad_run(void *plan, const void *dy, void *dx, cudaStream_t stream) {
    if (plan == nullptr || dy == nullptr || dx == nullptr) {
      return static_cast<int>(Status::kInvalidArgument);
    }
    auto *p = static_cast<DgradPlan *>(plan);
    return Traits::dgrad_run(*p->runner, dy, dx, stream);
  }

  static void dgrad_destroy(void *plan) { delete static_cast<DgradPlan *>(plan); }

  static const char *status_string(int code) {
    if (code >= static_cast<int>(Status::kOk) && code <= static_cast<int>(Status::kInternalError)) {
      return Traits::kStatusMessages[code];
    }
    return Traits::backend_status_string(code);
  }

  static EqLinearBackendApi const &api() {
    static const EqLinearBackendApi value = {
        kEqLinearAbiVersion, Traits::kPrecision, &forward_create, &forward_run,
        &forward_destroy,    &dgrad_create,           &dgrad_run,     &dgrad_destroy,
        &status_string};
    return value;
  }
};

}  // namespace fasteq::eqlinear
