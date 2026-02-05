#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

void mutipath_equi_linear_f32_impl(const uint32_t &num_paths,              // path nums
                                   float *out,                             // [B, total_i, V]
                                   float *x,                               // [B, total_i, U]
                                   float *w,                               // [num_paths, U, V]
                                   const uint32_t &B,                      // batch
                                   const uint32_t &total_i,                // total_i
                                   const std::vector<int64_t> &i_dims_vec, // i dims
                                   const uint32_t &U,                      // U
                                   const uint32_t &V,                      // V
                                   const double &val,                      // cg_val
                                   const cudaStream_t &cur_stream          // current stream
);

void mutipath_equi_linear_f64_impl(const uint32_t &num_paths,              // path nums
                                   double *out,                            // [B, total_i, V]
                                   double *x,                              // [B, total_i, U]
                                   double *w,                              // [num_paths, U, V]
                                   const uint32_t &B,                      // batch
                                   const uint32_t &total_i,                // total_i
                                   const std::vector<int64_t> &i_dims_vec, // i dims
                                   const uint32_t &U,                      // U
                                   const uint32_t &V,                      // V
                                   const double &val,                      // cg_val
                                   const cudaStream_t &cur_stream          // current stream
);
