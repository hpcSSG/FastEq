#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>

namespace fasteq::eqlinear {

inline bool lifetime_check(cudaError_t status, const char* operation) noexcept {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "eqlinear: %s failed during resource cleanup: %s\n", operation,
               cudaGetErrorString(status));
  return false;
}

// The fence stays on the allocation device until all protected resources are destroyed.
// cudaFree's documented possible implicit synchronization is not a completion guarantee.
class DeviceFence {
 public:
  explicit DeviceFence(int device) noexcept {
    if (!lifetime_check(cudaGetDevice(&previous_), "cudaGetDevice")) return;
    if (previous_ != device) {
      if (!lifetime_check(cudaSetDevice(device), "cudaSetDevice")) return;
      switched_ = true;
    }
    ready_ = lifetime_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }
  ~DeviceFence() noexcept {
    if (switched_) lifetime_check(cudaSetDevice(previous_), "restore CUDA device");
  }
  DeviceFence(const DeviceFence&) = delete;
  DeviceFence& operator=(const DeviceFence&) = delete;
  bool ready() const noexcept { return ready_; }

 private:
  int previous_ = -1;
  bool switched_ = false;
  bool ready_ = false;
};

// Normal plan destruction has one device fence and no per-run bookkeeping. Holding the
// whole runner until completion also protects its tables, workspaces and prepared operands.
template <typename Runner>
class DeviceOwner {
 public:
  DeviceOwner() {
    if (cudaGetDevice(&device_) != cudaSuccess)
      throw std::runtime_error("eqlinear: cannot identify plan device");
  }
  ~DeviceOwner() noexcept {
    if (!runner) return;
    DeviceFence fence(device_);
    if (fence.ready()) {
      runner.reset();
    } else {
      // A failed fence cannot establish a safe reclamation point. Keep GPU storage alive
      // until process exit rather than freeing a runner that may still be in use.
      (void)runner.release();
      std::fprintf(stderr, "eqlinear: retaining runner after failed destruction fence\n");
    }
  }
  DeviceOwner(const DeviceOwner&) = delete;
  DeviceOwner& operator=(const DeviceOwner&) = delete;

  std::unique_ptr<Runner> runner;

 private:
  int device_ = -1;
};

// Successful plans are fenced by DeviceOwner. A runner constructor can instead throw
// before DeviceOwner receives it: each completed allocation then fences before its own
// release during unwinding, including failures in subsequent member constructors.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) { reset(bytes); }
  ~DeviceBuffer() noexcept { reset(0); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  int reset(size_t bytes) noexcept {
    if (ptr_) {
      if (std::uncaught_exceptions() > 0) {
        DeviceFence fence(device_);
        if (fence.ready()) release();
        else {
          ptr_ = nullptr;
          std::fprintf(stderr, "eqlinear: retaining allocation after failed unwind fence\n");
          return 1;
        }
      } else {
        release();
      }
    }
    if (bytes && (cudaGetDevice(&device_) != cudaSuccess ||
                  cudaMalloc(&ptr_, bytes) != cudaSuccess)) {
      ptr_ = nullptr;
      return 1;
    }
    return 0;
  }

  template <typename T>
  T* as() const { return static_cast<T*>(ptr_); }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  void release() noexcept {
    lifetime_check(cudaFree(ptr_), "cudaFree");
    ptr_ = nullptr;
  }
  void* ptr_ = nullptr;
  int device_ = -1;
};

}  // namespace fasteq::eqlinear
