#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "eqlinear_common.h"

namespace fasteq::eqlinear {

/// C ABI implemented by each per-precision backend DSO.  Only this struct's function pointers
/// cross the DSO boundary; every C++/CUTLASS symbol stays hidden inside the backend.
struct EqLinearBackendApi {
  uint32_t abi_version;
  int32_t precision;

  int (*forward_create)(const EqLinearProblemC* problem, const void* w, const double* cg,
                        cudaStream_t stream, void** plan, EqLinearPlanInfo* info);
  int (*forward_run)(void* plan, const void* x, void* out, cudaStream_t stream);
  void (*forward_destroy)(void* plan);

  int (*dgrad_create)(const EqLinearProblemC* problem, const void* w, const double* cg,
                      cudaStream_t stream, void** plan, EqLinearPlanInfo* info);
  int (*dgrad_run)(void* plan, const void* dy, void* dx, cudaStream_t stream);
  void (*dgrad_destroy)(void* plan);

  const char* (*status_string)(int code);
};

}  // namespace fasteq::eqlinear

#define FASTEQ_EQLINEAR_EXPORT extern "C" __attribute__((visibility("default")))
