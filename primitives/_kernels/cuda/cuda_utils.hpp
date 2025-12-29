#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#ifdef __CUDA_ARCH__
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define CUDA_CHECK(ans) do { \
  cudaError_t err = (ans); \
  if (err != cudaSuccess) { \
    printf("CUDA Error: %s (%d) at %s:%d\n", cudaGetErrorString(err), (int)err, __FILE__, __LINE__); \
  } \
} while(0)

DEVICE inline int find_integer_divisor(int x, int bdim) {
  return (x + bdim - 1) / bdim;
}

template <class T>
DEVICE inline T *shared_array(unsigned int n_elements, void *&ptr,
                              unsigned int *space) noexcept {
  const unsigned long long inptr = reinterpret_cast<unsigned long long>(ptr);
  const unsigned long long end = inptr + n_elements * sizeof(T);
  if (space)
    *space += static_cast<unsigned int>(end - inptr);
  ptr = reinterpret_cast<void *>(end);
  return reinterpret_cast<T *>(inptr);
}

/*
// forward declare multiple types...
template float *shared_array<float>(unsigned int n_elements, void *&ptr,
                                    unsigned int *space) noexcept;
template double *shared_array<double>(unsigned int n_elements, void *&ptr,
                                      unsigned int *space) noexcept;
template int *shared_array<int>(unsigned int n_elements, void *&ptr,
                                unsigned int *space) noexcept;
template short *shared_array<short>(unsigned int n_elements, void *&ptr,
                                    unsigned int *space) noexcept; */

#endif // CUDA_UTILS_CUH