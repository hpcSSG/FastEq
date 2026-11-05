#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace fasteq::eqlinear {

/// Public precision selector.  The Python layer owns the string -> enum mapping.
enum class Precision : int32_t {
  kTf32 = 0,
  kFp32 = 1,
  kFp64 = 2,
};

enum class Status : int32_t {
  kOk = 0,
  kUnsupported = 1,
  kInvalidArgument = 2,
  kCudaError = 3,
  kInternalError = 4,
};

/// Geometry passed from the dispatcher to a backend.  `i_dims` is borrowed for the duration of
/// the create call; the backend must copy it if it outlives the call.
struct EqLinearProblemC {
  int32_t B = 0;
  int32_t I = 0;
  int32_t U = 0;
  int32_t P = 0;
  int32_t V = 0;
  int32_t J = 0;
  const int32_t* i_dims = nullptr;
};

/// Frozen dispatch decisions reported back to Python for tests and diagnostics.
struct EqLinearPlanInfo {
  int32_t precision = 0;
  int32_t forward_impl = -1;
  int32_t dgrad_impl = -1;
  int32_t a_view = 0;
  int32_t j_rows = 0;
};

inline constexpr uint32_t kEqLinearAbiVersion = 1;

}  // namespace fasteq::eqlinear
