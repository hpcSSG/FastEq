/***************************************************************************************************
 * fasteq eqlinear fp32/csrc/device_buffer.h
 **************************************************************************************************/
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace eqlin_fp32 {

/// Minimal RAII wrapper around cudaMalloc.  Returns nonzero from reset() on failure so callers can
/// turn allocation problems into a status code instead of an exception.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) { reset(bytes); }
  ~DeviceBuffer() { reset(0); }

  DeviceBuffer(DeviceBuffer const &) = delete;
  DeviceBuffer &operator=(DeviceBuffer const &) = delete;

  int reset(size_t bytes) {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    if (bytes) {
      if (cudaMalloc(&ptr_, bytes) != cudaSuccess) {
        ptr_ = nullptr;
        return 1;
      }
    }
    return 0;
  }

  template <typename T>
  T *as() const {
    return static_cast<T *>(ptr_);
  }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  void *ptr_ = nullptr;
};

}  // namespace eqlin_fp32
