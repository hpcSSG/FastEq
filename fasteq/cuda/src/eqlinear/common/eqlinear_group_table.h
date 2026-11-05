#pragma once

#include <vector>

namespace fasteq::eqlinear {

/// J: sum of the distinct run-length-grouped i values.  This is the output-row count used by all
/// three backends and by the torch dispatcher.
int out_total_i(const std::vector<int>& i_dims);

}  // namespace fasteq::eqlinear
