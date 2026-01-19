#pragma once
#include <cstdint>

// Make sure these are consistent across all TUs
#ifndef MAX_PATHS_CONST
#define MAX_PATHS_CONST 17
#endif

#ifndef ELL_MAX_CONST
#define ELL_MAX_CONST 2048
#endif

// If CWTP_DEFINE_CONSTANTS is defined in exactly ONE .cu before including this header,
// the constants are defined there; otherwise they are declared as extern.
#ifdef CWTP_DEFINE_CONSTANTS
  #define CWTP_EXTERN
#else
  #define CWTP_EXTERN extern
#endif

CWTP_EXTERN __device__ __constant__ int32_t  c_ell_E       [MAX_PATHS_CONST];
CWTP_EXTERN __device__ __constant__ int32_t  c_ell_base    [MAX_PATHS_CONST];
CWTP_EXTERN __device__ __constant__ uint16_t c_ell_ij      [ELL_MAX_CONST];
CWTP_EXTERN __device__ __constant__ float    c_ell_val_f32 [ELL_MAX_CONST];
CWTP_EXTERN __device__ __constant__ double   c_ell_val_f64 [ELL_MAX_CONST];

#undef CWTP_EXTERN

template <typename T>
__device__ __forceinline__ const T* ell_val_const_ptr();

template <>
__device__ __forceinline__ const float* ell_val_const_ptr<float>() {
  return c_ell_val_f32;
}
template <>
__device__ __forceinline__ const double* ell_val_const_ptr<double>() {
  return c_ell_val_f64;
}
