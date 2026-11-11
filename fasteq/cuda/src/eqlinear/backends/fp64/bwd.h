#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "eqlinear.h"

namespace eqlin {

enum class BwdImpl : int { kB1 = 1, kB2 = 2 };

class DgradRunner {
 public:
  DgradRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims,
              double const *w, std::vector<double> const &cg, cudaStream_t stream);
  ~DgradRunner();
  DgradRunner(DgradRunner const &) = delete;
  DgradRunner &operator=(DgradRunner const &) = delete;
  int run(double const *dy, double *dx, cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

class WgradRunner {
 public:
  WgradRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims,
              std::vector<double> const &cg, int tile = 0, int split_k = 1,
              int kernel = 0, int transpose_mode = 3);
  ~WgradRunner();
  WgradRunner(WgradRunner const &) = delete;
  WgradRunner &operator=(WgradRunner const &) = delete;
  int run(double const *x, double const *dy, double *dw, cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

class BackwardRunner {
 public:
  BackwardRunner(int B, int I, int U, int P, int V, std::vector<int> const &i_dims,
                 double const *w, std::vector<double> const &cg, cudaStream_t stream,
                 int wgrad_tile = 0, int wgrad_split_k = 1, int wgrad_kernel = 0,
                 int wgrad_transpose_mode = 3);
  ~BackwardRunner();
  BackwardRunner(BackwardRunner const &) = delete;
  BackwardRunner &operator=(BackwardRunner const &) = delete;
  int run(double const *x, double const *dy, double *dx, double *dw, cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace eqlin
