#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

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

template <typename T>
__global__ void commutator3x3_extract_backward_kernel(
    const T* __restrict__ A,         // [N,9]
    const T* __restrict__ B,         // [N,9]
    const T* __restrict__ grad_out,  // [E,3,Fdim]
    T* __restrict__ grad_A,          // [N,9]
    T* __restrict__ grad_B,          // [N,9]
    int64_t N,
    int64_t Fdim)
{
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  // idx -> (e, f)
  int64_t f = idx % Fdim;
  int64_t e = idx / Fdim;

  const T* a = A + idx * 9;
  const T* b = B + idx * 9;

  T* gA = grad_A + idx * 9;
  T* gB = grad_B + idx * 9;

  // grad_out[e, :, f]
  int64_t base = (e * 3) * Fdim + f;
  T gx = grad_out[base + 0 * Fdim];  // dL/drx
  T gy = grad_out[base + 1 * Fdim];  // dL/dry
  T gz = grad_out[base + 2 * Fdim];  // dL/drz

  // load A
  T a0 = a[0], a1 = a[1], a2 = a[2];
  T a3 = a[3], a4 = a[4], a5 = a[5];
  T a6 = a[6], a7 = a[7], a8 = a[8];

  // load B
  T b0 = b[0], b1 = b[1], b2 = b[2];
  T b3 = b[3], b4 = b[4], b5 = b[5];
  T b6 = b[6], b7 = b[7], b8 = b[8];

  // grad A
  gA[0] = fmaT(gy, b2, -gz * b3);
  gA[1] = fmaT(gy, b5, -gx * b6);
  gA[2] = gy * (b8 - b0);

  gA[3] = gz * (b0 - b4);
  gA[4] = fmaT(gz, b3, -gx * b7);
  gA[5] = fmaT(gz, b6, -gy * b1);

  gA[6] = fmaT(gx, b1, -gz * b5);
  gA[7] = gx * (b4 - b8);
  gA[8] = fmaT(gx, b7, -gy * b2);

  // grad B
  gB[0] = fmaT(gz, a3, -gy * a2);
  gB[1] = fmaT(gx, a6, -gy * a5);
  gB[2] = gy * (a0 - a8);

  gB[3] = gz * (a4 - a0);
  gB[4] = fmaT(gx, a7, -gz * a3);
  gB[5] = fmaT(gy, a1, -gz * a6);

  gB[6] = fmaT(gz, a5, -gx * a1);
  gB[7] = gx * (a8 - a4);
  gB[8] = fmaT(gy, a2, -gx * a7);
}

template <typename T>
void launch_commutator3x3_extract_backward(
    const T* A,
    const T* B,
    const T* grad_out,
    T* grad_A,
    T* grad_B,
    int64_t N,
    int64_t Fdim,
    cudaStream_t stream)
{
  constexpr int threads = 256;
  int64_t blocks = (N + threads - 1) / threads;
  commutator3x3_extract_backward_kernel<T>
      <<< (uint32_t)blocks, threads, 0, stream >>>(
          A, B, grad_out, grad_A, grad_B, N, Fdim);
}

void commutator3x3_extract_backward_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor grad_out,
    torch::Tensor grad_A,
    torch::Tensor grad_B,
    int64_t Fdim)
{
  auto N = A.size(0);
  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "commutator3x3_extract_backward_cuda", [&] {
    launch_commutator3x3_extract_backward<scalar_t>(
      (const scalar_t*)A.data_ptr<scalar_t>(),
      (const scalar_t*)B.data_ptr<scalar_t>(),
      (const scalar_t*)grad_out.data_ptr<scalar_t>(),
      (scalar_t*)grad_A.data_ptr<scalar_t>(),
      (scalar_t*)grad_B.data_ptr<scalar_t>(),
      N, Fdim, stream.stream());
  });
}

std::vector<torch::Tensor> commutator3x3_extract_backward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor grad_out,
    int64_t Fdim)
{
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && grad_out.is_cuda(),
              "A/B/grad_out must be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == B.scalar_type(),
              "A/B dtype must match");
  TORCH_CHECK(A.scalar_type() == grad_out.scalar_type(),
              "grad_out dtype must match A/B");
  TORCH_CHECK(A.dim() == 3 && B.dim() == 3,
              "A/B must be [N,3,3]");
  TORCH_CHECK(A.size(1) == 3 && A.size(2) == 3,
              "A must be [N,3,3]");
  TORCH_CHECK(B.size(1) == 3 && B.size(2) == 3,
              "B must be [N,3,3]");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
              "A/B must be contiguous [N,3,3]");

  auto N = A.size(0);
  TORCH_CHECK(Fdim > 0 && (N % Fdim) == 0,
              "Require N % Fdim == 0, got N=", N, " Fdim=", Fdim);

  auto E = N / Fdim;
  TORCH_CHECK(grad_out.dim() == 3, "grad_out must be [E,3,Fdim]");
  TORCH_CHECK(grad_out.size(0) == E &&
              grad_out.size(1) == 3 &&
              grad_out.size(2) == Fdim,
              "grad_out must have shape [E,3,Fdim]");
  TORCH_CHECK(grad_out.is_contiguous(),
              "grad_out must be contiguous [E,3,Fdim]");

  auto grad_A = torch::empty_like(A);
  auto grad_B = torch::empty_like(B);

  commutator3x3_extract_backward_cuda(A, B, grad_out, grad_A, grad_B, Fdim);
  return {grad_A, grad_B};
}

TORCH_LIBRARY(fused_commutator, m)
{
    m.def("forward", &commutator3x3_extract);
    m.def("backward", &commutator3x3_extract_backward);
}