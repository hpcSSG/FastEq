#ifndef HIP_UTILS_HPP
#define HIP_UTILS_HPP

#include <hip/hip_runtime.h>

#ifdef __HIP_DEVICE_COMPILE__
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 64  // AMD GPUs use 64-thread wavefronts
#endif

#define HIP_CHECK(ans) do { \
    hipError_t err = (ans); \
    if (err != hipSuccess) { \
        printf("HIP Error: %s (%d) at %s:%d\n", hipGetErrorString(err), (int)err, __FILE__, __LINE__); \
    } \
} while(0)

template <typename T>
__device__ __forceinline__ T ld_g(const T* p) {
    // HIP does not have __ldg, direct access is used
    return *p;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v, unsigned mask) {
    // AMD uses 64-wide wavefronts, use __shfl_down for reduction
    for (int offset = 32; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset);
    }
    return v;
}

template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
    for (int d = 32; d > 0; d >>= 1) {
        v += __shfl_down(v, d);
    }
    return v;
}

__host__ __device__ inline int find_integer_divisor(int x, int bdim) {
    return (x + bdim - 1) / bdim;
}

template <class T>
__host__ __device__ inline T *shared_array(unsigned int n_elements, void *&ptr,
                              unsigned int *space) noexcept {
    const unsigned long long inptr = reinterpret_cast<unsigned long long>(ptr);
    const unsigned long long end = inptr + n_elements * sizeof(T);
    if (space)
        *space += static_cast<unsigned int>(end - inptr);
    ptr = reinterpret_cast<void *>(end);
    return reinterpret_cast<T *>(inptr);
}

#endif // HIP_UTILS_HPP