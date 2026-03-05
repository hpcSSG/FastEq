#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <typename T>
__device__ __forceinline__ T fmaT(T a, T b, T c) {
#if __CUDA_ARCH__ >= 530
  return fma(a, b, c);
#else
  return a * b + c;
#endif
}

// A,B: [N,3,3] contiguous (row-major inside the 3x3)
// out: [E,3,Fdim] contiguous with strides: out[(e*3 + c)*Fdim + f]
template <typename T>
__global__ void commutator3x3_extract_kernel(
    const T* __restrict__ A,  // [N,9]
    const T* __restrict__ B,  // [N,9]
    T* __restrict__ out,      // [E*3*Fdim]
    int64_t N,
    int64_t Fdim)
{
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  // map idx -> (e,f)
  int64_t f = idx % Fdim;
  int64_t e = idx / Fdim;

  const T* a = A + idx * 9;
  const T* b = B + idx * 9;

  // a[r,c] = a[r*3 + c]
  // b[r,c] = b[r*3 + c]

  // rx = (AB - BA)[2,1] = sum_k a[2,k]*b[k,1] - sum_k b[2,k]*a[k,1]
  T ab21 = fmaT(a[6], b[1], fmaT(a[7], b[4], a[8] * b[7]));  // a20*b01 + a21*b11 + a22*b21? careful: b[k,1] => b[0,1]=b[1], b[1,1]=b[4], b[2,1]=b[7]
  T ba21 = fmaT(b[6], a[1], fmaT(b[7], a[4], b[8] * a[7]));  // b20*a01 + b21*a11 + b22*a21
  T rx = ab21 - ba21;

  // ry = (AB - BA)[0,2] = sum_k a[0,k]*b[k,2] - sum_k b[0,k]*a[k,2]
  T ab02 = fmaT(a[0], b[2], fmaT(a[1], b[5], a[2] * b[8]));  // a00*b02 + a01*b12 + a02*b22
  T ba02 = fmaT(b[0], a[2], fmaT(b[1], a[5], b[2] * a[8]));  // b00*a02 + b01*a12 + b02*a22
  T ry = ab02 - ba02;

  // rz = (AB - BA)[1,0] = sum_k a[1,k]*b[k,0] - sum_k b[1,k]*a[k,0]
  T ab10 = fmaT(a[3], b[0], fmaT(a[4], b[3], a[5] * b[6]));  // a10*b00 + a11*b10 + a12*b20
  T ba10 = fmaT(b[3], a[0], fmaT(b[4], a[3], b[5] * a[6]));  // b10*a00 + b11*a10 + b12*a20
  T rz = ab10 - ba10;

  // write to out[e, :, f]  => contiguous [E,3,Fdim]
  // out_index = (e*3 + c)*Fdim + f
  int64_t base = (e * 3) * Fdim + f;
  out[base + 0 * Fdim] = rx;
  out[base + 1 * Fdim] = ry;
  out[base + 2 * Fdim] = rz;
}

template <typename T>
void launch_commutator3x3_extract(
    const T* A, const T* B, T* out,
    int64_t N, int64_t Fdim,
    cudaStream_t stream)
{
  constexpr int threads = 256;
  int64_t blocks = (N + threads - 1) / threads;
  commutator3x3_extract_kernel<T><<< (uint32_t)blocks, threads, 0, stream >>>(
      A, B, out, N, Fdim);
}

// include the kernel file or paste kernel code here
// (Assume you pasted the kernel+launcher above in this file)

void commutator3x3_extract_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor out, int64_t Fdim) {
  auto N = A.size(0);
  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "commutator3x3_extract_cuda", [&] {
    launch_commutator3x3_extract<scalar_t>(
      (const scalar_t*)A.data_ptr<scalar_t>(),
      (const scalar_t*)B.data_ptr<scalar_t>(),
      (scalar_t*)out.data_ptr<scalar_t>(),
      N, Fdim, stream.stream());
  });
}

torch::Tensor commutator3x3_extract(torch::Tensor A, torch::Tensor B, int64_t Fdim) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A/B dtype must match");
  TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "A/B must be [N,3,3]");
  TORCH_CHECK(A.size(1) == 3 && A.size(2) == 3, "A must be [N,3,3]");
  TORCH_CHECK(B.size(1) == 3 && B.size(2) == 3, "B must be [N,3,3]");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A/B must be contiguous [N,3,3]");

  auto N = A.size(0);
  TORCH_CHECK(Fdim > 0 && (N % Fdim) == 0, "Require N % Fdim == 0, got N=", N, " Fdim=", Fdim);
  auto E = N / Fdim;

  auto out = torch::empty({E, 3, Fdim}, A.options());
  commutator3x3_extract_cuda(A, B, out, Fdim);
  return out;
}

TORCH_LIBRARY(fused_commutator, m)
{
    m.def("forward", &commutator3x3_extract);
}