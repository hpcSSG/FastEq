#pragma once

#include <cuda_runtime.h>

namespace fasteq::eqlinear {

inline constexpr int kMaxPaths = 32;

template <typename T>
struct CgValues {
  T v[kMaxPaths];
};

/// a_prep[b, col] = x[b, src_index[col]] for col in [0, I*U).
template <typename T>
void launch_permute_a(T const *x, T *a_prep, int const *src_index, int I, int U, int B,
                      cudaStream_t stream);

/// b_prep[n, col] = cg.v[p_of_col[col]] * w[src_base[col] + n]; tables have P*U entries.
template <typename T>
void launch_pack_b(T const *w, T *b_prep, int const *src_base, int const *p_of_col,
                   CgValues<T> cg, int P, int U, int V, cudaStream_t stream);

}  // namespace fasteq::eqlinear
