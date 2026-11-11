#pragma once

#include <memory>
#include <vector>
#include "eqlinear_common.h"
#include "eqlinear_group_table.h"

namespace fasteq::eqlinear {

// Same-precision fused plans for shapes that cannot use the frozen fast paths.
class ShapePlan {
 public:
  virtual ~ShapePlan() = default;
  virtual int run(const void* input, void* output, cudaStream_t stream) = 0;
  virtual int impl() const { return 100; }
};

struct ShapePath {
  int input_row, output_row, weight_path, rows;
  double coefficient;
};

// Host ownership only. Device plans copy these immutable descriptors once.
struct ShapeGeometry {
  int B, I, U, P, V, J;
  std::vector<ShapePath> paths;

  ShapeGeometry(const EqLinearProblemC& p, const double* cg)
      : B(p.B), I(p.I), U(p.U), P(p.P), V(p.V), J(p.J) {
    if (B <= 0 || I <= 0 || U <= 0 || P <= 0 || P > 32 || V <= 0 || J <= 0 ||
        !p.i_dims || !cg)
      throw std::invalid_argument("eqlinear: invalid shape geometry");
    int64_t input = 0, output = 0;
    int previous = -1;
    for (int q = 0; q < P; ++q) {
      EqLinearPathC path;
      if (p.paths) {
        path = p.paths[q];
      } else {
        if (q && p.i_dims[q] != previous) output += previous;
        if (input > I || output > J)
          throw std::invalid_argument("eqlinear: dense path prefix outside operand bounds");
        path = {static_cast<int32_t>(input), static_cast<int32_t>(output), q, p.i_dims[q]};
        input += p.i_dims[q];
        previous = p.i_dims[q];
      }
      if (path.rows <= 0 || path.rows != p.i_dims[q] || path.input_row < 0 ||
          path.output_row < 0 || path.weight_path < 0 || path.weight_path >= P ||
          int64_t(path.input_row) + path.rows > I ||
          int64_t(path.output_row) + path.rows > J)
        throw std::invalid_argument("eqlinear: path outside operand bounds");
      paths.push_back({path.input_row, path.output_row, path.weight_path, path.rows, cg[q]});
    }
  }
};

}  // namespace fasteq::eqlinear
