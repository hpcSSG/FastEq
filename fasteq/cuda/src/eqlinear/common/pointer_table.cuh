#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "device_lifetime.h"

namespace eqlin {

template <typename T>
struct PointerRow {
  const T *a_base;
  T *c_base;
  long long a_offset;
  long long c_offset;
};

// Reusing pinned host memory lets the next call overwrite pointer values before an earlier
// asynchronous copy consumes them. Generate the pointer arrays in stream order instead.
template <typename T>
__global__ void update_output_table(T *out, const PointerRow<T> *rows, T **c,
                                    T const **cc, int count) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < count) {
    T *ptr = (rows[j].c_base ? rows[j].c_base : out) + rows[j].c_offset;
    c[j] = ptr;
    if (cc) cc[j] = ptr;
  }
}

template <typename T>
__global__ void update_input_table(T const *in, const PointerRow<T> *rows, T **a, int count) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < count) a[j] = const_cast<T *>((rows[j].a_base ? rows[j].a_base : in) + rows[j].a_offset);
}

template <typename T>
class PointerTable {
 public:
  explicit PointerTable(std::vector<PointerRow<T>> const &rows) : count_(rows.size()) {
    if (device_.reset(rows.size() * sizeof(PointerRow<T>)) != 0)
      throw std::runtime_error("pointer table allocation failed");
    if (cudaMemcpy(device_.as<PointerRow<T>>(), rows.data(), rows.size() * sizeof(PointerRow<T>),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("pointer table initialization failed");
    }
  }
  ~PointerTable() = default;
  PointerTable(PointerTable const &) = delete;
  PointerTable &operator=(PointerTable const &) = delete;

  cudaError_t output(T *out, T **c, T const **cc, cudaStream_t stream) const {
    update_output_table<<<(count_ + 127) / 128, 128, 0, stream>>>(out, device_.as<PointerRow<T>>(), c, cc, count_);
    return cudaGetLastError();
  }
  cudaError_t input(T const *in, T **a, cudaStream_t stream) const {
    update_input_table<<<(count_ + 127) / 128, 128, 0, stream>>>(in, device_.as<PointerRow<T>>(), a, count_);
    return cudaGetLastError();
  }

 private:
  int count_;
  fasteq::eqlinear::DeviceBuffer device_;
};

}  // namespace eqlin
