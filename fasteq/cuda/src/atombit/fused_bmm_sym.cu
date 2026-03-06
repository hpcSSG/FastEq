#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

static __device__ __forceinline__ int idx4(int e, int a, int b, int f, int F) {
    return (((e * 3 + a) * 3 + b) * F + f);
}

// 以 float4 方式访问：把 f 维按 4 打包
static __device__ __forceinline__ int idx4_f4(int e, int a, int b, int f4, int F4) {
    // F4 = F/4
    return (((e * 3 + a) * 3 + b) * F4 + f4);
}

// 逐分量 FMA: acc += x * y
static __device__ __forceinline__ float4 fma4(float4 acc, float4 x, float4 y) {
    acc.x = fmaf(x.x, y.x, acc.x);
    acc.y = fmaf(x.y, y.y, acc.y);
    acc.z = fmaf(x.z, y.z, acc.z);
    acc.w = fmaf(x.w, y.w, acc.w);
    return acc;
}

static __device__ __forceinline__ float4 add4(float4 a, float4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    return a;
}
static __device__ __forceinline__ float4 sub4(float4 a, float4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
    return a;
}
static __device__ __forceinline__ float4 mul4s(float4 a, float s) {
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
    return a;
}

static __device__ __forceinline__ float4 make_zero4() {
    return float4{0.f, 0.f, 0.f, 0.f};
}

__global__ void fused_raw_sym_traceless_f4(
    const float* __restrict__ h,  // [E,3,3,F]
    const float* __restrict__ g,  // [E,3,3,F]
    float* __restrict__ out,      // [E,3,3,F]
    int E, int F)
{
    // 要求 F % 4 == 0
    int F4 = F >> 2;

    int e  = (int)blockIdx.x;
    int f4 = (int)blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= E || f4 >= F4) return;

    // 按 float4 读取
    const float4* __restrict__ h4 = reinterpret_cast<const float4*>(h);
    const float4* __restrict__ g4 = reinterpret_cast<const float4*>(g);
    float4* __restrict__ o4       = reinterpret_cast<float4*>(out);

    // raw_ij
    float4 r00 = {0,0,0,0}, r01 = {0,0,0,0}, r02 = {0,0,0,0};
    float4 r10 = {0,0,0,0}, r11 = {0,0,0,0}, r12 = {0,0,0,0};
    float4 r20 = {0,0,0,0}, r21 = {0,0,0,0}, r22 = {0,0,0,0};

    // k = 0
    {
        float4 h00 = h4[idx4_f4(e,0,0,f4,F4)], h10 = h4[idx4_f4(e,1,0,f4,F4)], h20 = h4[idx4_f4(e,2,0,f4,F4)];
        float4 g00 = g4[idx4_f4(e,0,0,f4,F4)], g01 = g4[idx4_f4(e,0,1,f4,F4)], g02 = g4[idx4_f4(e,0,2,f4,F4)];
        r00 = fma4(r00, h00, g00); r01 = fma4(r01, h00, g01); r02 = fma4(r02, h00, g02);
        r10 = fma4(r10, h10, g00); r11 = fma4(r11, h10, g01); r12 = fma4(r12, h10, g02);
        r20 = fma4(r20, h20, g00); r21 = fma4(r21, h20, g01); r22 = fma4(r22, h20, g02);
    }
    // k = 1
    {
        float4 h01 = h4[idx4_f4(e,0,1,f4,F4)], h11 = h4[idx4_f4(e,1,1,f4,F4)], h21 = h4[idx4_f4(e,2,1,f4,F4)];
        float4 g10 = g4[idx4_f4(e,1,0,f4,F4)], g11 = g4[idx4_f4(e,1,1,f4,F4)], g12 = g4[idx4_f4(e,1,2,f4,F4)];
        r00 = fma4(r00, h01, g10); r01 = fma4(r01, h01, g11); r02 = fma4(r02, h01, g12);
        r10 = fma4(r10, h11, g10); r11 = fma4(r11, h11, g11); r12 = fma4(r12, h11, g12);
        r20 = fma4(r20, h21, g10); r21 = fma4(r21, h21, g11); r22 = fma4(r22, h21, g12);
    }
    // k = 2
    {
        float4 h02 = h4[idx4_f4(e,0,2,f4,F4)], h12 = h4[idx4_f4(e,1,2,f4,F4)], h22 = h4[idx4_f4(e,2,2,f4,F4)];
        float4 g20 = g4[idx4_f4(e,2,0,f4,F4)], g21 = g4[idx4_f4(e,2,1,f4,F4)], g22 = g4[idx4_f4(e,2,2,f4,F4)];
        r00 = fma4(r00, h02, g20); r01 = fma4(r01, h02, g21); r02 = fma4(r02, h02, g22);
        r10 = fma4(r10, h12, g20); r11 = fma4(r11, h12, g21); r12 = fma4(r12, h12, g22);
        r20 = fma4(r20, h22, g20); r21 = fma4(r21, h22, g21); r22 = fma4(r22, h22, g22);
    }

    // sym = 0.5*(raw + raw^T)
    float4 s00 = r00;
    float4 s11 = r11;
    float4 s22 = r22;

    float4 s01 = mul4s(add4(r01, r10), 0.5f);
    float4 s02 = mul4s(add4(r02, r20), 0.5f);
    float4 s12 = mul4s(add4(r12, r21), 0.5f);

    // trace = s00 + s11 + s22
    float4 tr = add4(add4(s00, s11), s22);
    float4 t3 = mul4s(tr, (1.0f/3.0f));

    // traceless: 对角线减去 tr/3
    s00 = sub4(s00, t3);
    s11 = sub4(s11, t3);
    s22 = sub4(s22, t3);

    // 写回 out
    o4[idx4_f4(e,0,0,f4,F4)] = s00;
    o4[idx4_f4(e,0,1,f4,F4)] = s01;
    o4[idx4_f4(e,0,2,f4,F4)] = s02;

    o4[idx4_f4(e,1,0,f4,F4)] = s01;
    o4[idx4_f4(e,1,1,f4,F4)] = s11;
    o4[idx4_f4(e,1,2,f4,F4)] = s12;

    o4[idx4_f4(e,2,0,f4,F4)] = s02;
    o4[idx4_f4(e,2,1,f4,F4)] = s12;
    o4[idx4_f4(e,2,2,f4,F4)] = s22;
}

torch::Tensor fused_traceless_forward_cuda(torch::Tensor h, torch::Tensor geom) {
    TORCH_CHECK(h.is_cuda(), "h must be CUDA");
    TORCH_CHECK(geom.is_cuda(), "geom must be CUDA");
    TORCH_CHECK(h.dtype() == torch::kFloat32, "only float32 supported in this kernel");
    TORCH_CHECK(geom.dtype() == torch::kFloat32, "only float32 supported in this kernel");
    TORCH_CHECK(h.is_contiguous(), "h must be contiguous (call contiguous())");
    TORCH_CHECK(geom.is_contiguous(), "geom must be contiguous (call contiguous())");
    TORCH_CHECK(h.dim() == 4 && geom.dim() == 4, "expect 4D tensors [E,3,3,F]");
    TORCH_CHECK(h.size(1) == 3 && h.size(2) == 3, "h shape must be [E,3,3,F]");
    TORCH_CHECK(geom.size(1) == 3 && geom.size(2) == 3, "geom shape must be [E,3,3,F]");
    TORCH_CHECK(h.sizes() == geom.sizes(), "h and geom must have same shape");

    int64_t E = h.size(0);
    int64_t F = h.size(3);
    TORCH_CHECK(F % 4 == 0, "F must be divisible by 4 for float4 path");

    auto out = torch::empty_like(h);

    // 对齐检查
    const void* hp = h.data_ptr();
    const void* gp = geom.data_ptr();
    void* op = out.data_ptr();
    TORCH_CHECK(((uintptr_t)hp % 16) == 0 && ((uintptr_t)gp % 16) == 0 && ((uintptr_t)op % 16) == 0,
                "h/geom/out pointers must be 16-byte aligned for float4 path");

    int F4 = (int)(F / 4);
    dim3 block(32);
    dim3 grid((unsigned)E, 1);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    fused_raw_sym_traceless_f4<<<grid, block, 0, stream>>>(
        (const float*)h.data_ptr<float>(),
        (const float*)geom.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        (int)E, (int)F);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

__global__ void fused_raw_sym_traceless_bwd_f4(
    const float* __restrict__ h,         // [E,3,3,F]
    const float* __restrict__ g,         // [E,3,3,F]
    const float* __restrict__ grad_out,  // [E,3,3,F]
    float* __restrict__ grad_h,          // [E,3,3,F]
    float* __restrict__ grad_g,          // [E,3,3,F]
    int E, int F)
{
    // 要求 F % 4 == 0
    int F4 = F >> 2;

    int e  = (int)blockIdx.x;
    int f4 = (int)blockIdx.y * blockDim.x + threadIdx.x;
    if (e >= E || f4 >= F4) return;

    const float4* __restrict__ h4  = reinterpret_cast<const float4*>(h);
    const float4* __restrict__ g4  = reinterpret_cast<const float4*>(g);
    const float4* __restrict__ go4 = reinterpret_cast<const float4*>(grad_out);

    float4* __restrict__ gh4 = reinterpret_cast<float4*>(grad_h);
    float4* __restrict__ gg4 = reinterpret_cast<float4*>(grad_g);

    // -----------------------------
    // 1) 读取上游梯度 dO
    // -----------------------------
    float4 go00 = go4[idx4_f4(e,0,0,f4,F4)];
    float4 go01 = go4[idx4_f4(e,0,1,f4,F4)];
    float4 go02 = go4[idx4_f4(e,0,2,f4,F4)];

    float4 go10 = go4[idx4_f4(e,1,0,f4,F4)];
    float4 go11 = go4[idx4_f4(e,1,1,f4,F4)];
    float4 go12 = go4[idx4_f4(e,1,2,f4,F4)];

    float4 go20 = go4[idx4_f4(e,2,0,f4,F4)];
    float4 go21 = go4[idx4_f4(e,2,1,f4,F4)];
    float4 go22 = go4[idx4_f4(e,2,2,f4,F4)];

    // -----------------------------
    // 2) dR = sym(dO) - tr(dO)/3 * I
    // -----------------------------
    float4 tr = add4(add4(go00, go11), go22);
    float4 t3 = mul4s(tr, 1.0f / 3.0f);

    float4 dr00 = sub4(go00, t3);
    float4 dr11 = sub4(go11, t3);
    float4 dr22 = sub4(go22, t3);

    float4 dr01 = mul4s(add4(go01, go10), 0.5f);
    float4 dr02 = mul4s(add4(go02, go20), 0.5f);
    float4 dr12 = mul4s(add4(go12, go21), 0.5f);

    // 对称
    float4 dr10 = dr01;
    float4 dr20 = dr02;
    float4 dr21 = dr12;

    // -----------------------------
    // 3) 读取前向输入 H, G
    // -----------------------------
    float4 h00 = h4[idx4_f4(e,0,0,f4,F4)];
    float4 h01 = h4[idx4_f4(e,0,1,f4,F4)];
    float4 h02 = h4[idx4_f4(e,0,2,f4,F4)];

    float4 h10 = h4[idx4_f4(e,1,0,f4,F4)];
    float4 h11 = h4[idx4_f4(e,1,1,f4,F4)];
    float4 h12 = h4[idx4_f4(e,1,2,f4,F4)];

    float4 h20 = h4[idx4_f4(e,2,0,f4,F4)];
    float4 h21 = h4[idx4_f4(e,2,1,f4,F4)];
    float4 h22 = h4[idx4_f4(e,2,2,f4,F4)];

    float4 g00 = g4[idx4_f4(e,0,0,f4,F4)];
    float4 g01 = g4[idx4_f4(e,0,1,f4,F4)];
    float4 g02 = g4[idx4_f4(e,0,2,f4,F4)];

    float4 g10 = g4[idx4_f4(e,1,0,f4,F4)];
    float4 g11 = g4[idx4_f4(e,1,1,f4,F4)];
    float4 g12 = g4[idx4_f4(e,1,2,f4,F4)];

    float4 g20 = g4[idx4_f4(e,2,0,f4,F4)];
    float4 g21 = g4[idx4_f4(e,2,1,f4,F4)];
    float4 g22 = g4[idx4_f4(e,2,2,f4,F4)];

    // -----------------------------
    // 4) grad_h = dR @ G^T
    //    gh_ik = sum_j dR_ij * G_kj
    // -----------------------------
    float4 gh00 = make_zero4(), gh01 = make_zero4(), gh02 = make_zero4();
    float4 gh10 = make_zero4(), gh11 = make_zero4(), gh12 = make_zero4();
    float4 gh20 = make_zero4(), gh21 = make_zero4(), gh22 = make_zero4();

    // k = 0: 使用 G row 0 -> (g00, g01, g02)
    gh00 = fma4(gh00, dr00, g00);
    gh00 = fma4(gh00, dr01, g01);
    gh00 = fma4(gh00, dr02, g02);

    gh10 = fma4(gh10, dr10, g00);
    gh10 = fma4(gh10, dr11, g01);
    gh10 = fma4(gh10, dr12, g02);

    gh20 = fma4(gh20, dr20, g00);
    gh20 = fma4(gh20, dr21, g01);
    gh20 = fma4(gh20, dr22, g02);

    // k = 1: 使用 G row 1 -> (g10, g11, g12)
    gh01 = fma4(gh01, dr00, g10);
    gh01 = fma4(gh01, dr01, g11);
    gh01 = fma4(gh01, dr02, g12);

    gh11 = fma4(gh11, dr10, g10);
    gh11 = fma4(gh11, dr11, g11);
    gh11 = fma4(gh11, dr12, g12);

    gh21 = fma4(gh21, dr20, g10);
    gh21 = fma4(gh21, dr21, g11);
    gh21 = fma4(gh21, dr22, g12);

    // k = 2: 使用 G row 2 -> (g20, g21, g22)
    gh02 = fma4(gh02, dr00, g20);
    gh02 = fma4(gh02, dr01, g21);
    gh02 = fma4(gh02, dr02, g22);

    gh12 = fma4(gh12, dr10, g20);
    gh12 = fma4(gh12, dr11, g21);
    gh12 = fma4(gh12, dr12, g22);

    gh22 = fma4(gh22, dr20, g20);
    gh22 = fma4(gh22, dr21, g21);
    gh22 = fma4(gh22, dr22, g22);

    // -----------------------------
    // 5) grad_g = H^T @ dR
    //    gg_kj = sum_i H_ik * dR_ij
    // -----------------------------
    float4 gg00 = make_zero4(), gg01 = make_zero4(), gg02 = make_zero4();
    float4 gg10 = make_zero4(), gg11 = make_zero4(), gg12 = make_zero4();
    float4 gg20 = make_zero4(), gg21 = make_zero4(), gg22 = make_zero4();

    // k = 0: 使用 H col 0 -> (h00, h10, h20)
    gg00 = fma4(gg00, h00, dr00);
    gg00 = fma4(gg00, h10, dr10);
    gg00 = fma4(gg00, h20, dr20);

    gg01 = fma4(gg01, h00, dr01);
    gg01 = fma4(gg01, h10, dr11);
    gg01 = fma4(gg01, h20, dr21);

    gg02 = fma4(gg02, h00, dr02);
    gg02 = fma4(gg02, h10, dr12);
    gg02 = fma4(gg02, h20, dr22);

    // k = 1: 使用 H col 1 -> (h01, h11, h21)
    gg10 = fma4(gg10, h01, dr00);
    gg10 = fma4(gg10, h11, dr10);
    gg10 = fma4(gg10, h21, dr20);

    gg11 = fma4(gg11, h01, dr01);
    gg11 = fma4(gg11, h11, dr11);
    gg11 = fma4(gg11, h21, dr21);

    gg12 = fma4(gg12, h01, dr02);
    gg12 = fma4(gg12, h11, dr12);
    gg12 = fma4(gg12, h21, dr22);

    // k = 2: 使用 H col 2 -> (h02, h12, h22)
    gg20 = fma4(gg20, h02, dr00);
    gg20 = fma4(gg20, h12, dr10);
    gg20 = fma4(gg20, h22, dr20);

    gg21 = fma4(gg21, h02, dr01);
    gg21 = fma4(gg21, h12, dr11);
    gg21 = fma4(gg21, h22, dr21);

    gg22 = fma4(gg22, h02, dr02);
    gg22 = fma4(gg22, h12, dr12);
    gg22 = fma4(gg22, h22, dr22);

    // -----------------------------
    // 6) 写回
    // -----------------------------
    gh4[idx4_f4(e,0,0,f4,F4)] = gh00;
    gh4[idx4_f4(e,0,1,f4,F4)] = gh01;
    gh4[idx4_f4(e,0,2,f4,F4)] = gh02;

    gh4[idx4_f4(e,1,0,f4,F4)] = gh10;
    gh4[idx4_f4(e,1,1,f4,F4)] = gh11;
    gh4[idx4_f4(e,1,2,f4,F4)] = gh12;

    gh4[idx4_f4(e,2,0,f4,F4)] = gh20;
    gh4[idx4_f4(e,2,1,f4,F4)] = gh21;
    gh4[idx4_f4(e,2,2,f4,F4)] = gh22;

    gg4[idx4_f4(e,0,0,f4,F4)] = gg00;
    gg4[idx4_f4(e,0,1,f4,F4)] = gg01;
    gg4[idx4_f4(e,0,2,f4,F4)] = gg02;

    gg4[idx4_f4(e,1,0,f4,F4)] = gg10;
    gg4[idx4_f4(e,1,1,f4,F4)] = gg11;
    gg4[idx4_f4(e,1,2,f4,F4)] = gg12;

    gg4[idx4_f4(e,2,0,f4,F4)] = gg20;
    gg4[idx4_f4(e,2,1,f4,F4)] = gg21;
    gg4[idx4_f4(e,2,2,f4,F4)] = gg22;
}

std::vector<torch::Tensor> fused_traceless_backward_cuda(
    torch::Tensor h,
    torch::Tensor geom,
    torch::Tensor grad_out)
{
    TORCH_CHECK(h.is_cuda(), "h must be CUDA");
    TORCH_CHECK(geom.is_cuda(), "geom must be CUDA");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");

    TORCH_CHECK(h.dtype() == torch::kFloat32, "only float32 supported in this kernel");
    TORCH_CHECK(geom.dtype() == torch::kFloat32, "only float32 supported in this kernel");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32, "only float32 supported in this kernel");

    TORCH_CHECK(h.is_contiguous(), "h must be contiguous (call contiguous())");
    TORCH_CHECK(geom.is_contiguous(), "geom must be contiguous (call contiguous())");
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous (call contiguous())");

    TORCH_CHECK(h.dim() == 4 && geom.dim() == 4 && grad_out.dim() == 4,
                "expect 4D tensors [E,3,3,F]");

    TORCH_CHECK(h.size(1) == 3 && h.size(2) == 3, "h shape must be [E,3,3,F]");
    TORCH_CHECK(geom.size(1) == 3 && geom.size(2) == 3, "geom shape must be [E,3,3,F]");
    TORCH_CHECK(grad_out.size(1) == 3 && grad_out.size(2) == 3, "grad_out shape must be [E,3,3,F]");

    TORCH_CHECK(h.sizes() == geom.sizes(), "h and geom must have same shape");
    TORCH_CHECK(h.sizes() == grad_out.sizes(), "grad_out must have same shape as h/geom");

    int64_t E = h.size(0);
    int64_t F = h.size(3);
    TORCH_CHECK(F % 4 == 0, "F must be divisible by 4 for float4 path");

    auto grad_h    = torch::empty_like(h);
    auto grad_geom = torch::empty_like(geom);

    // 16-byte 对齐检查
    const void* hp  = h.data_ptr();
    const void* gp  = geom.data_ptr();
    const void* gop = grad_out.data_ptr();
    void* ghp       = grad_h.data_ptr();
    void* ggp       = grad_geom.data_ptr();

    TORCH_CHECK(((uintptr_t)hp  % 16) == 0 &&
                ((uintptr_t)gp  % 16) == 0 &&
                ((uintptr_t)gop % 16) == 0 &&
                ((uintptr_t)ghp % 16) == 0 &&
                ((uintptr_t)ggp % 16) == 0,
                "h/geom/grad_out/grad_h/grad_geom pointers must be 16-byte aligned for float4 path");

    int F4 = (int)(F / 4);
    dim3 block(32);
    dim3 grid((unsigned)E, (unsigned)((F4 + block.x - 1) / block.x));

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    fused_raw_sym_traceless_bwd_f4<<<grid, block, 0, stream>>>(
        (const float*)h.data_ptr<float>(),
        (const float*)geom.data_ptr<float>(),
        (const float*)grad_out.data_ptr<float>(),
        (float*)grad_h.data_ptr<float>(),
        (float*)grad_geom.data_ptr<float>(),
        (int)E, (int)F
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_h, grad_geom};
}


TORCH_LIBRARY(fused_bmm_sym, m)
{
    m.def("forward", &fused_traceless_forward_cuda);
    m.def("backward", &fused_traceless_backward_cuda);
}