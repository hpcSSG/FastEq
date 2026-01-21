#pragma once
#ifndef EVAL_PID_GENERATED_CUH
#define EVAL_PID_GENERATED_CUH

#include <stdint.h>
#include <cuda_runtime.h>


// ---- generated eval_pid_<pid> ----
template <typename T>
__device__ __forceinline__ T eval_pid_0(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_1(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)256;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_2(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)512;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_3(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)768;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_4(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)1024;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_5(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)1;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_6(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)2;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_7(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)257;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)514;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_8(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)512;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1024;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)257;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)2;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_9(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)256;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)513;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)770;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_10(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)769;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)514;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1026;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_11(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)512;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)257;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)2;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)4;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_12(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)1;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)258;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)515;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_13(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)514;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)259;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)516;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_14(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)1024;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1536;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)769;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1281;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)514;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)259;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)4;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[6];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)516;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[7];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_15(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)256;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)513;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)770;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1027;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1284;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[6];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[7];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_16(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)512;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)257;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1026;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)771;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1283;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1028;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[6];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1540;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[7];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_17(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)1280;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_18(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)1536;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_19(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)512;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)2;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_20(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)256;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_21(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)513;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)258;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_22(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)514;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_23(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)1024;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)2;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_24(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)768;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)258;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_25(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)256;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1025;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)770;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_26(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1026;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_27(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)1024;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1536;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)257;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)2;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)514;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_28(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)768;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1280;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)513;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)258;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_29(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)512;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)769;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1026;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_30(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)256;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1025;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)770;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1282;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_31(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)512;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1281;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1026;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1538;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_32(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)3;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_33(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)4;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_34(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)512;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)4;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_35(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)256;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)513;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)3;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_36(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)1;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)515;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)260;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_37(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)516;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_38(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)257;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)514;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)771;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1028;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_39(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)512;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)769;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)2;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)259;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_40(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)768;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)513;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1025;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)258;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)3;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)260;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_41(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)257;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)514;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)771;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1028;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_42(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)256;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)770;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)515;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1027;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)772;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_43(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)257;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1026;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)771;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)516;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_44(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)1024;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1281;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)2;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)259;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)516;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[6];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[7];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_45(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)768;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1025;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1537;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)3;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)515;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[6];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[7];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_46(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)1;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)513;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1027;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1539;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)772;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[6];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[7];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

template <typename T>
__device__ __forceinline__ T eval_pid_47(
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    T acc = (T)0;
    {
        constexpr uint16_t ij = (uint16_t)512;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[0];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)257;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[1];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1538;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[2];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1283;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[3];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)1028;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[4];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[5];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[6];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    {
        constexpr uint16_t ij = (uint16_t)0;
        constexpr int i = (int)(ij & 0xFF);
        constexpr int j = (int)(ij >> 8);
        T c   = v_row[7];
        T xiu = s_iu[iu_base + i * U + u];
        T xjv = s_jv[jv_base + j];
        acc = fma(c, xiu * xjv, acc);
    }
    return acc;
}

// ---- generated eval_pid_dispatch ----
template <typename T>
__device__ __forceinline__ T eval_pid_dispatch(
    int pid,
    const T* __restrict__ v_row,
    const T* __restrict__ s_iu, int iu_base, int U, int u,
    const T* __restrict__ s_jv, int jv_base
) {
    switch (pid) {
      case 0: return eval_pid_0<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 1: return eval_pid_1<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 2: return eval_pid_2<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 3: return eval_pid_3<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 4: return eval_pid_4<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 5: return eval_pid_5<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 6: return eval_pid_6<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 7: return eval_pid_7<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 8: return eval_pid_8<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 9: return eval_pid_9<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 10: return eval_pid_10<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 11: return eval_pid_11<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 12: return eval_pid_12<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 13: return eval_pid_13<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 14: return eval_pid_14<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 15: return eval_pid_15<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 16: return eval_pid_16<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 17: return eval_pid_17<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 18: return eval_pid_18<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 19: return eval_pid_19<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 20: return eval_pid_20<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 21: return eval_pid_21<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 22: return eval_pid_22<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 23: return eval_pid_23<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 24: return eval_pid_24<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 25: return eval_pid_25<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 26: return eval_pid_26<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 27: return eval_pid_27<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 28: return eval_pid_28<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 29: return eval_pid_29<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 30: return eval_pid_30<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 31: return eval_pid_31<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 32: return eval_pid_32<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 33: return eval_pid_33<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 34: return eval_pid_34<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 35: return eval_pid_35<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 36: return eval_pid_36<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 37: return eval_pid_37<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 38: return eval_pid_38<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 39: return eval_pid_39<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 40: return eval_pid_40<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 41: return eval_pid_41<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 42: return eval_pid_42<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 43: return eval_pid_43<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 44: return eval_pid_44<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 45: return eval_pid_45<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 46: return eval_pid_46<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      case 47: return eval_pid_47<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);
      default: return (T)0;
    }
}


#endif // EVAL_PID_GENERATED_CUH
