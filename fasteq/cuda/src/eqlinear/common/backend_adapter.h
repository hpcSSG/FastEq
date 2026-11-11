#pragma once

#include <exception>
#include <memory>
#include <vector>

#include "device_lifetime.h"
#include "eqlinear_backend_api.h"
#include "eqlinear_common.h"
#include "eqlinear_group_table.h"
#include "shape_plan.h"

namespace fasteq::eqlinear {

/// Shared implementation of the C ABI vtable.  Traits supply only what genuinely differs between
/// the precision backends: scalar type, runner types, implementation selection, run glue and
/// status labels.  Static (policy) dispatch keeps each backend's concrete CUDA types intact.
template <typename Traits>
struct BackendAdapter {
  using Scalar = typename Traits::Scalar;
  using ForwardRunner = typename Traits::ForwardRunner;
  using DgradRunner = typename Traits::DgradRunner;
  template <typename Runner>
  struct PlanState {
    std::unique_ptr<Runner> legacy;
    std::unique_ptr<ShapePlan> shape;
  };
  using ForwardPlan = DeviceOwner<PlanState<ForwardRunner>>;
  using DgradPlan = DeviceOwner<PlanState<DgradRunner>>;

  static int forward_create(const EqLinearProblemC *problem, const void *w, const double *cg,
                            cudaStream_t stream, void **plan, EqLinearPlanInfo *info) {
    if (problem == nullptr || w == nullptr || plan == nullptr || info == nullptr) {
      return static_cast<int>(Status::kInvalidArgument);
    }
    *plan = nullptr;
    try {
      std::vector<int> i_dims(problem->i_dims, problem->i_dims + problem->P);
      auto owner = std::make_unique<ForwardPlan>();
      owner->runner = std::make_unique<PlanState<ForwardRunner>>();
      auto* state = owner->runner.get();
      if (Traits::use_shape_plan(*problem)) {
        state->shape = Traits::make_shape_plan(*problem, w, cg, false, stream);
      } else {
        state->legacy = Traits::make_forward(*problem, i_dims, w, cg);
        const int rc = state->legacy->prepare_weight(static_cast<Scalar const*>(w), cg, stream);
        if (rc != 0) return rc;
      }
      info->precision = Traits::kPrecision;
      info->forward_impl = state->shape ? state->shape->impl() : Traits::forward_impl(*problem);
      info->dgrad_impl = Traits::kDgradImpl;
      info->a_view = state->shape ? 1 : (Traits::a_view(*state->legacy) ? 1 : 0);
      info->j_rows = problem->J;
      *plan = owner.release();
      return 0;
    } catch (UnsupportedConfiguration const &) {
      return static_cast<int>(Status::kUnsupported);
    } catch (std::exception const &) {
      return static_cast<int>(Status::kInvalidArgument);
    }
  }

  static int forward_run(void *plan, const void *x, void *out, cudaStream_t stream) {
    if (plan == nullptr || x == nullptr || out == nullptr) {
      return static_cast<int>(Status::kInvalidArgument);
    }
    auto *p = static_cast<ForwardPlan *>(plan);
    auto& state = *p->runner;
    return state.shape ? state.shape->run(x, out, stream)
                       : Traits::forward_run(*state.legacy, x, out, stream);
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
      owner->runner = std::make_unique<PlanState<DgradRunner>>();
      auto* state = owner->runner.get();
      if (Traits::use_shape_plan(*problem))
        state->shape = Traits::make_shape_plan(*problem, w, cg, true, stream);
      else
        state->legacy = Traits::make_dgrad(*problem, i_dims, w, cg, stream);
      info->precision = Traits::kPrecision;
      info->forward_impl = -1;
      info->dgrad_impl = state->shape ? state->shape->impl() : Traits::kDgradImpl;
      info->a_view = 0;
      info->j_rows = problem->J;
      *plan = owner.release();
      return 0;
    } catch (UnsupportedConfiguration const &) {
      return static_cast<int>(Status::kUnsupported);
    } catch (std::exception const &) {
      return static_cast<int>(Status::kInvalidArgument);
    }
  }

  static int dgrad_run(void *plan, const void *dy, void *dx, cudaStream_t stream) {
    if (plan == nullptr || dy == nullptr || dx == nullptr) {
      return static_cast<int>(Status::kInvalidArgument);
    }
    auto *p = static_cast<DgradPlan *>(plan);
    auto& state = *p->runner;
    return state.shape ? state.shape->run(dy, dx, stream)
                       : Traits::dgrad_run(*state.legacy, dy, dx, stream);
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
