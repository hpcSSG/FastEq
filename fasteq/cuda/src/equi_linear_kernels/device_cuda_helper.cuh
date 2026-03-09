#pragma once
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#include "./ptx_inst.cuh"

template <uint32_t _N> struct idim_T
{
    uint32_t _i[_N];
};

/* Device template function helper */
// dump single reg
template <typename T>
__device__ void dump_warp_reg(T reg,                     // reg
                              uint32_t wg_id,            // cur warpgroup id
                              uint32_t in_wg_tid,        // cur in warpgroup tid
                              uint32_t dump_wg_id,       // to dump wg_id
                              uint32_t dump_warp_id = 0, // to dump in-warpgroup-warp id [0:4)
                              const char reg_name[] = "" // reg_name
)
{
    uint32_t t_start = dump_warp_id * WARP_SIZE;
    if (wg_id == dump_wg_id)
    {
        if (in_wg_tid == t_start)
            printf("[Kernel Debug]: %s Data:\n", reg_name);
#pragma unroll 1
        for (uint32_t _t = t_start; _t < t_start + WARP_SIZE; ++_t)
        {
            if (in_wg_tid == _t)
            {
                printf("[%u]%.3f, ", in_wg_tid, float(reg));
                // printf("%.3f", in_wg_tid, float(reg));
                if (_t % 4 == 3)
                {
                    printf("\n");
                }
            }
            __syncwarp(0xffffffff);
        }
    }
}

// dump reg[start_id:end_id]
template <typename T, uint32_t NUM>
__device__ void dump_warp_reg(T reg[NUM],                // regs
                              uint32_t wg_id,            // cur warpgroup id
                              uint32_t in_wg_tid,        // cur in warpgroup tid
                              uint32_t dump_wg_id,       // to dump wg_id
                              uint32_t start_id = 0,     // regs start id
                              uint32_t end_id = NUM,     // regs end id
                              uint32_t dump_warp_id = 0, // to dump in-warpgroup-warp id [0:4)
                              const char reg_name[] = "" // reg_name
)
{
    end_id = (end_id > NUM) ? NUM : end_id;
    uint32_t t_start = dump_warp_id * WARP_SIZE;
    if (wg_id == dump_wg_id)
    {
        if (in_wg_tid == t_start)
            printf("[Kernel Debug]: %s Data:\n", reg_name);
#pragma unroll 1
        for (uint32_t _t = t_start; _t < t_start + WARP_SIZE; ++_t)
        {
            if (in_wg_tid == _t)
            {
                // printf("[%u]", in_wg_tid);
#pragma unroll 1
                for (uint32_t _n = start_id; _n < end_id; ++_n)
                {
                    // printf("%.3f, ", float(reg[_n]));
                    printf("%.3f,", float(reg[_n]));
                }
                if (_t % 4 == 3)
                {
                    printf("\n");
                }
            }
            __syncwarp(0xffffffff);
        }
    }
}

template <typename T> __device__ __forceinline__ T warp_reduce_sum(T val)
{
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T> __device__ __forceinline__ T warp_reduce_avg(T val)
{
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return static_cast<T>(val / WARP_SIZE);
}

template <typename T> __device__ __forceinline__ void check_aligned_128_bitwise(const T *ptr, const char name[])
{
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        if ((reinterpret_cast<uintptr_t>(ptr) & 127) != 0)
        {
            printf("[Debug]: block(%u, %u, %u) %s align 128 check failed!\n", (uint32_t)blockIdx.x,
                   (uint32_t)blockIdx.y, (uint32_t)blockIdx.z, name);
        }
    }
}

template <typename T> __device__ __forceinline__ void check_aligned_1024_bitwise(const T *ptr, const char name[])
{
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        if ((reinterpret_cast<uintptr_t>(ptr) & 1023) != 0)
        {
            printf("[Debug]: block(%u, %u, %u) %s align 1024 check failed!\n", (uint32_t)blockIdx.x,
                   (uint32_t)blockIdx.y, (uint32_t)blockIdx.z, name);
        }
    }
}

template <uint32_t N> __device__ __forceinline__ void swap_Nregs_f32(float a[N], float b[N])
{
    // 位运算交换，不占用额外寄存器，精度无损
#pragma unroll
    for (uint32_t _n = 0; _n < N; ++_n)
    {
        a[_n] = __uint_as_float(__float_as_uint(a[_n]) ^ __float_as_uint(b[_n]));
        b[_n] = __uint_as_float(__float_as_uint(a[_n]) ^ __float_as_uint(b[_n]));
        a[_n] = __uint_as_float(__float_as_uint(a[_n]) ^ __float_as_uint(b[_n]));
    }
}

template <typename T>
__device__ __forceinline__ void cp_async_bulk_tensor_3d_global_to_shared_multicast(T *smem_ptr, const void *tensor_map,
                                                                                   uint32_t crd0, uint32_t crd1,
                                                                                   uint32_t crd2, uint64_t *mbar_ptr,
                                                                                   uint16_t mcast_mask)
{
    // 需要将 generic 指针转换为 shared memory 的 32 位地址格式供 PTX 使用
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(static_cast<void *>(smem_ptr)));
    uint32_t mbar_int = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                 " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                 :
                 : "r"(smem_int), "l"(tensor_map), "r"(crd0), "r"(crd1), "r"(crd2), "r"(mbar_int), "h"(mcast_mask)
                 : "memory");
}

template <typename T>
__device__ __forceinline__ void cp_async_bulk_tensor_3d_global_to_shared(T *smem_ptr, const void *tensor_map,
                                                                         uint32_t crd0, uint32_t crd1, uint32_t crd2,
                                                                         uint64_t *mbar_ptr)
{
    // 需要将 generic 指针转换为 shared memory 的 32 位地址格式供 PTX 使用
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(static_cast<void *>(smem_ptr)));
    uint32_t mbar_int = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4}], [%5];"
                 :
                 : "r"(smem_int), "l"(tensor_map), "r"(crd0), "r"(crd1), "r"(crd2), "r"(mbar_int)
                 : "memory");
}

template <typename T>
__device__ __forceinline__ void cta_mbarrier_arrive_and_wait(T *barrier_ptr, cuda::ptx::scope_cluster_t)
{
    static_assert(std::is_same<T, uint64_t>::value, "Barrier type must be uint64_t");
    uint64_t state = cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cluster,
                                                cuda::ptx::space_shared, barrier_ptr);
    asm_mbarrier_wait(barrier_ptr, state);
}

template <typename T>
__device__ __forceinline__ void cta_mbarrier_arrive_and_wait(T *barrier_ptr, cuda::ptx::scope_cta_t)
{
    static_assert(std::is_same<T, uint64_t>::value, "Barrier type must be uint64_t");
    uint64_t state = cuda::ptx::mbarrier_arrive(barrier_ptr);
    asm_mbarrier_wait(barrier_ptr, state);
}

/* Host function helper for TMA (Tensor Memory Access) */

// Get pointer to cuTensorMapEncodeTiled function
inline PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled()
{
    static void *cuTensorMapEncodeTiled_ptr = nullptr;
    if (cuTensorMapEncodeTiled_ptr == nullptr)
    {
        cudaDriverEntryPointQueryResult driver_status;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,
                                         cudaEnableDefault, &driver_status);
        assert(driver_status == cudaDriverEntryPointSuccess);
    }
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

// Creates a tensor map to describe a 3D array of size B x M x N
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-multi-dimensional-arrays
inline void init_3d_tensormap(
    CUtensorMap *tensor_map_ptr,                                                      // Host empty tensor map
    void *tensor_ptr,                                                                 // global addr
    const uint32_t &B,                                                                // B
    const uint32_t &M,                                                                // M
    const uint32_t &N,                                                                // N
    const uint32_t &stride_B,                                                         // stride_B (elems)
    const uint32_t &stride_M,                                                         // stride_M (elems)
    const uint32_t &box_B,                                                            // B of shared memory buffer
    const uint32_t &box_M,                                                            // M of shared memory buffer
    const uint32_t &box_N,                                                            // N of shared memory buffer
    CUtensorMapSwizzle swizzle_mode = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE, // swizzle_mode
    CUtensorMapL2promotion l2_promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE, // l2 promotion type
    CUtensorMapFloatOOBfill oob_fill =
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE // out-of-bounds fill type
)
{
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 3;
    uint64_t size[rank] = {static_cast<uint64_t>(N), static_cast<uint64_t>(M), static_cast<uint64_t>(B)};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {static_cast<uint64_t>(stride_M) * sizeof(float),
                                 static_cast<uint64_t>(stride_B) * sizeof(float)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {box_N, box_M, box_B};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    static const uint32_t elem_stride[3] = {1, 1, 1};
    // Create the tensor descriptor.
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUresult res =
        cuTensorMapEncodeTiled(tensor_map_ptr, // CUtensorMap *tensorMap,
                               CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                               rank,        // cuuint32_t tensorRank,
                               tensor_ptr,  // void *globalAddress,
                               size,        // const cuuint64_t *globalDim,
                               stride,      // const cuuint64_t *globalStrides,
                               box_size,    // const cuuint32_t *boxDim,
                               elem_stride, // const cuuint32_t *elementStrides,
                               // Interleave patterns can be used to accelerate loading of values that
                               // are less than 4 bytes long.
                               CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                               // Swizzling can be used to avoid shared memory bank conflicts.
                               swizzle_mode,
                               // L2 Promotion can be used to widen the effect of a cache-policy to a wider
                               // set of L2 cache lines.
                               l2_promotion,
                               // Any element that is outside of bounds will be set to zero by the TMA transfer.
                               oob_fill);
}

// creates a tensor map to describe a two-dimensional row-major array of size B x M x N
//      https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-multi-dimensional-arrays
inline void init_2d_tensormap(
    CUtensorMap *tensor_map_ptr,                                                      // Host empty tensor map
    void *tensor_ptr,                                                                 // global addr
    const uint32_t &M,                                                                // M
    const uint32_t &N,                                                                // N
    const uint32_t &stride_M,                                                         // stride_M (elems)
    const uint32_t &box_M,                                                            // M of shared memory buffer
    const uint32_t &box_N,                                                            // N of shared memory buffer
    CUtensorMapSwizzle swizzle_mode = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE, // swizzle_mode
    CUtensorMapL2promotion l2_promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE, // l2 promotion type
    CUtensorMapFloatOOBfill oob_fill =
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE // out-of-bounds fill type
)
{
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {static_cast<uint64_t>(N), static_cast<uint64_t>(M)};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {static_cast<uint64_t>(stride_M) * sizeof(float)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {box_N, box_M};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    static const uint32_t elem_stride[2] = {1, 1};
    // Create the tensor descriptor.
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUresult res =
        cuTensorMapEncodeTiled(tensor_map_ptr, // CUtensorMap *tensorMap,
                               CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                               rank,        // cuuint32_t tensorRank,
                               tensor_ptr,  // void *globalAddress,
                               size,        // const cuuint64_t *globalDim,
                               stride,      // const cuuint64_t *globalStrides,
                               box_size,    // const cuuint32_t *boxDim,
                               elem_stride, // const cuuint32_t *elementStrides,
                               // Interleave patterns can be used to accelerate loading of values that
                               // are less than 4 bytes long.
                               CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                               // Swizzling can be used to avoid shared memory bank conflicts.
                               swizzle_mode,
                               // L2 Promotion can be used to widen the effect of a cache-policy to a wider
                               // set of L2 cache lines.
                               l2_promotion,
                               // Any element that is outside of bounds will be set to zero by the TMA transfer.
                               oob_fill);
}

template <uint32_t N> void cal_prefex_sum(idim_T<N> &dst, const std::vector<int64_t> &src)
{
    dst._i[0] = 0;
#pragma unroll
    for (uint32_t i = 1; i < N; ++i)
    {
        dst._i[i] = dst._i[i - 1] + (uint32_t)src[i - 1];
    }
}
