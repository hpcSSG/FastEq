#pragma once
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#ifdef __FCTP_DEBUG__
#include <cutlass/util/debug.h>
#include <cutlass/util/device_dump.h>
#endif

#include "./ptx_inst.cuh"

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