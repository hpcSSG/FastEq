#include "ptx_inst.cuh"
#include <c10/cuda/CUDAStream.h>
#include <cudaTypedefs.h>
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

#ifdef __FCTP_DEBUG__
#include <cutlass/util/debug.h>
#include <cutlass/util/device_dump.h>
#endif

struct idim_T
{
    uint32_t _i[4];
};

template <uint32_t WARP_PER_BLOCK>
__device__ __forceinline__ void WarpATileG2SDirect16BAsync(double *smem_ptr0,        // 共享内存基地址
                                                           const double *A_ptr0,     // 矩阵A基地址
                                                           const uint32_t &t_m0,     // thread level的矩阵A的M方向偏移
                                                           const uint32_t &t_ak0,    // thread level的矩阵A的K方向偏移
                                                           const uint32_t &loop_k,   // k循环
                                                           const uint32_t &m_stride, // M方向步长
                                                           const uint32_t &m_count,  // M方向计数
                                                           const uint32_t &V,        // V
                                                           const uint32_t &tid,      // thread id
                                                           const uint32_t &wid       // warp id
)
{
    // Global offset
    // [B, V] -> [M, K]
    uint32_t t_batch = t_m0 + m_count * m_stride;
    uint32_t t_v = t_ak0 + loop_k;
    uint64_t a_offset = t_batch * V + t_v;
    const double *A_ptr = A_ptr0 + a_offset;

    // SMEM offset
    // 无需swizzle，直接存
    constexpr uint32_t DOUBLE_NUMS_PER_SMEM_LINE = 16; // smem每行可存16个double
    constexpr uint32_t WARP_TAKE_SMEM_LINES = 4;       // 每个warp一次load的数据要占4行SMEM
    uint32_t smem_x = (tid & 0b111) << 1;              // (tid % 8) * 2
    uint32_t smem_y = m_count * WARP_PER_BLOCK * WARP_TAKE_SMEM_LINES + wid * WARP_TAKE_SMEM_LINES + (tid >> 3);
    uint32_t smem_offset = smem_y * DOUBLE_NUMS_PER_SMEM_LINE + smem_x;
    uint32_t real_smem_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<void *>(smem_ptr0 + smem_offset)));

    // fetch req
    asm_cp_async_ca(real_smem_ptr, A_ptr, 16);
}

template <typename elem_T,          // 元素类型
          uint32_t SMEM_ELEM_COUNT, // 元素数量
          uint32_t WARP_PER_BLOCK   // block内warp数量
          >
__device__ __forceinline__ void SetSmemZeroSync(elem_T *smem_ptr0,   // 共享内存基地址
                                                const uint32_t &tid, // thread id
                                                const uint32_t &wid  // warp id
)
{
    constexpr uint32_t ELEMS_PER_16B = 16 / sizeof(elem_T);
    constexpr uint32_t LOOP_COUNT = SMEM_ELEM_COUNT / (WARP_PER_BLOCK * WARP_SIZE * ELEMS_PER_16B);
    constexpr uint32_t SMEM_LINES_PER_LOOP = 4 * WARP_PER_BLOCK;
    constexpr uint32_t ELEMS_PER_SMEM_LINE = 128 / sizeof(elem_T);

    uint32_t smem_x = (tid % 8) * ELEMS_PER_16B;
    uint32_t smem_y0 = (tid / 8) + wid * 4;
    float zero[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll
    for (uint32_t _l = 0; _l < LOOP_COUNT; ++_l)
    {
        uint32_t smem_y = smem_y0 + _l * SMEM_LINES_PER_LOOP;
        uint32_t smem_offset = smem_y * ELEMS_PER_SMEM_LINE + smem_x;
        FETCH_16B(smem_ptr0[smem_offset]) = FETCH_16B(zero[0]);
    }
}

template <uint32_t WARP_PER_BLOCK>
__device__ __forceinline__ void WarpATileG2SReverse8BAsync(double *smem_ptr0,        // 共享内存基地址
                                                           const double *A_ptr0,     // 矩阵A基地址
                                                           const uint32_t &t_m0,     // thread level的矩阵A的M方向偏移
                                                           const uint32_t &t_ak0,    // thread level的矩阵A的K方向偏移
                                                           const uint32_t &loop_k,   // k循环
                                                           const uint32_t &m_stride, // M方向步长
                                                           const uint32_t &m_count,  // M方向计数
                                                           const uint32_t &V,        // V
                                                           const uint32_t &tid,      // thread id
                                                           const uint32_t &wid       // warp id
)
{
    // Global offset
    // [B, V] -> [M, K]
    uint32_t t_batch = t_m0 + m_count * m_stride;
    uint32_t t_v = t_ak0 + loop_k;
    uint64_t a_offset = t_batch * V + t_v;
    const double *A_ptr = A_ptr0 + a_offset;

    // SMEM offset
    // 每行存8个连续thread，每两个连续thread一组，每组之间隔2个double，奇数行smem_x反向
    constexpr uint32_t DOUBLE_NUMS_PER_SMEM_LINE = 16; // smem每行可存16个double
    constexpr uint32_t WARP_TAKE_SMEM_LINES = 4;       // 每个warp一次load的数据要占4行SMEM
    uint32_t smem_y = m_count * WARP_PER_BLOCK * WARP_TAKE_SMEM_LINES + wid * WARP_TAKE_SMEM_LINES + (tid >> 3);
    uint32_t smem_x0 = ((tid & 0b110) << 1) + (tid & 0b1); // ((tid % 8) / 2) * 4 + (tid % 2)
    uint32_t smem_x = ((smem_y & 0b1) * 0b1111) ^ smem_x0; // 奇数行smem_x反向
    uint32_t smem_offset = smem_y * DOUBLE_NUMS_PER_SMEM_LINE + smem_x;
    uint32_t real_smem_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<void *>(smem_ptr0 + smem_offset)));

    // fetch req
    asm_cp_async_ca(real_smem_ptr, A_ptr, 8);
}

template <uint32_t WARP_PER_BLOCK>
__device__ __forceinline__ void WarpBTileG2SSwizzleAsync(double *smem_ptr0,        // 共享内存基地址
                                                         const double *B_ptr0,     // 矩阵B基地址
                                                         const uint32_t &t_n0,     // thread level的矩阵B的 N 方向偏移
                                                         const uint32_t &t_bk0,    // thread level的矩阵B的 K 方向偏移
                                                         const uint32_t &loop_k,   // k循环
                                                         const uint32_t &k_stride, // K方向步长
                                                         const uint32_t &k_count,  // K方向计数
                                                         const uint32_t &path_id,  // 当前的 path id
                                                         const uint32_t &U,        // U
                                                         const uint32_t &V,        // V
                                                         const uint32_t &W,        // W
                                                         const uint32_t &tid,      // thread id
                                                         const uint32_t &wid       // warp id
)
{
    // Global offset
    // [path, U, V, W], path->path, K -> V, N -> U*W
    uint32_t t_v = loop_k + k_count * k_stride + t_bk0;
    uint32_t t_u = t_n0 / W;
    uint32_t t_w = t_n0 % W;
    uint64_t b_offset = path_id * U * V * W + t_u * V * W + t_v * W + t_w;
    const double *B_ptr = B_ptr0 + b_offset;

    // SMEM offset
    // 每次load 64*4的一个tile，16*4为一组（包含两个连续的N8K4的MMA tile）
    // 每个16x4的组存在连续的4行，4行内每4个double为一组做swizzle
    constexpr uint32_t THREAD_LD_DOUBLES = 2;          // 每个thread每次load 16B，占4个bank
    constexpr uint32_t DOUBLE_NUMS_PER_SMEM_LINE = 16; // smem每行可存16个double
    constexpr uint32_t N8K4X2_TAKE_SMEM_LINES = 4;     // 每两个矩阵要占4行SMEM
    uint32_t group_id = tid >> 3;                      // (tid / 8)
    uint32_t in_group_x = tid & 0b111;                 // tid % 8
    uint32_t in_group_y = wid;
    uint32_t swizzle_group_x = in_group_x >> 1; // in_group_x / 2
    uint32_t smem_swizzle_y0 = in_group_y;
    uint32_t smem_swizzle_x0 = ((smem_swizzle_y0 ^ swizzle_group_x) << 1) + (in_group_x & 0b1);
    uint32_t smem_y =
        smem_swizzle_y0 + group_id * N8K4X2_TAKE_SMEM_LINES + k_count * (WARP_SIZE / 8) * N8K4X2_TAKE_SMEM_LINES;
    uint32_t smem_x = smem_swizzle_x0 * THREAD_LD_DOUBLES;
    uint32_t smem_offset = smem_y * DOUBLE_NUMS_PER_SMEM_LINE + smem_x;
    uint32_t real_smem_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<void *>(smem_ptr0 + smem_offset)));

    // fetch req
    asm_cp_async_ca(real_smem_ptr, B_ptr, 16);
}

// 加载一个A矩阵的8*4分块（Subtile）
template <uint32_t TOTAL_M_COUNT>
__device__ __forceinline__ void WarpASubtileM8K4S2RDirectSync(double &a_reg,           // dst
                                                              const double *smem_ptr0, // 共享内存基地址
                                                              const uint32_t &m_count, // m方向计数
                                                              const uint32_t &tid,     // thread id
                                                              const uint32_t &warp_mid // warp_m id
)
{
    constexpr uint32_t DOUBLE_NUMS_PER_SMEM_LINE = 16; // smem每行可存16个double
    constexpr uint32_t SMEM_LINES_PER_MCOUNT = 2;
    uint32_t smem_x = tid & 0b1111; // tid % 16
    uint32_t smem_y = (tid >> 4) + m_count * SMEM_LINES_PER_MCOUNT + warp_mid * TOTAL_M_COUNT * SMEM_LINES_PER_MCOUNT;
    uint32_t smem_offset = smem_y * DOUBLE_NUMS_PER_SMEM_LINE + smem_x;
    a_reg = smem_ptr0[smem_offset];
}

// 加载一个A矩阵的8*4分块（Subtile）
// 奇数行反向
template <uint32_t TOTAL_M_COUNT>
__device__ __forceinline__ void WarpASubtileM8K4S2RReverseSync(double &a_reg,           // dst
                                                               const double *smem_ptr0, // 共享内存基地址
                                                               const uint32_t &m_count, // m方向计数
                                                               const uint32_t &tid,     // thread id
                                                               const uint32_t &warp_mid // warp_m id
)
{
    constexpr uint32_t DOUBLE_NUMS_PER_SMEM_LINE = 16; // smem每行可存16个double
    constexpr uint32_t SMEM_LINES_PER_MCOUNT = 2;
    uint32_t smem_y = (tid >> 4) + m_count * SMEM_LINES_PER_MCOUNT + warp_mid * TOTAL_M_COUNT * SMEM_LINES_PER_MCOUNT;
    uint32_t smem_x0 = tid & 0b1111;                       // tid % 16
    uint32_t smem_x = ((smem_y & 0b1) * 0b1111) ^ smem_x0; // 奇数行smem_x反向
    uint32_t smem_offset = smem_y * DOUBLE_NUMS_PER_SMEM_LINE + smem_x;
    a_reg = smem_ptr0[smem_offset];
}

// 加载一个B矩阵的8*4分块（Subtile）
template <uint32_t TOTAL_N_COUNT>
__device__ __forceinline__ void WarpBSubtileN8K4S2RSwizzleSync(double &b_reg,           // dst
                                                               const double *smem_ptr0, // 共享内存基地址
                                                               const uint32_t &n_count, // n方向计数
                                                               const uint32_t &tid,     // thread id
                                                               const uint32_t &warp_nid // warp_n id
)
{
    constexpr uint32_t DOUBLE_NUMS_PER_SMEM_LINE = 16; // smem每行可存16个double
    constexpr uint32_t SWIZZLE_GROUP_DOUBLE_NUM = 4;   // 每4个double为一组
    constexpr uint32_t N8K4X2_TAKE_SMEM_LINES = 4;
    uint32_t smem_swizzle_y0 = tid & 0x3; // tid % 4
    uint32_t smem_unswizzle_group_id = (warp_nid * TOTAL_N_COUNT + n_count) >> 1;

    uint32_t smem_unswizzle_x = (tid >> 2) + ((n_count & 0x1) << 3); // (tid / 4) + (n_count % 2) * 8
    uint32_t smem_unswizzle_swgroup_id = smem_unswizzle_x >> 2;      // smem_unswizzle_x / 4
    uint32_t smem_unswizzle_in_swgroup_id = smem_unswizzle_x & 0x3;  // smem_unswizzle_x % 4
    uint32_t smem_swizzle_x0 =
        (smem_unswizzle_swgroup_id ^ smem_swizzle_y0) * SWIZZLE_GROUP_DOUBLE_NUM + smem_unswizzle_in_swgroup_id;

    uint32_t smem_y = smem_swizzle_y0 + smem_unswizzle_group_id * N8K4X2_TAKE_SMEM_LINES;
    uint32_t smem_x = smem_swizzle_x0;
    uint32_t smem_offset = smem_y * DOUBLE_NUMS_PER_SMEM_LINE + smem_x;

    b_reg = smem_ptr0[smem_offset];
}

// 加载一个M8N8的矩阵块到 SMEM_M * SMEM_N 的 Row Major SMEM中
// 每次加载其中的8行，(tid / 4) % 2 == 0的加载16个BANK，剩下的加载另外16个BANK
// 对于当前warp的TILE，N方向一共有TOTAL_N_COUNT个N8 TILE
// 对于当前block的TILE，N方向一共有TOTAL_N_COUNT * N_WARPS 个N8 TILE
template <uint32_t TOTAL_N_COUNT, uint32_t N_WARPS>
__device__ __forceinline__ void WarpTileM8N8R2SSwizzle2Sync(double *smem_ptr0,       // SMEM基地址(M warps需手动偏移)
                                                            const float4 &reg_16B,   // src
                                                            const uint32_t &m_count, // m方向计数
                                                            const uint32_t &n_count, // n方向计数
                                                            const uint32_t &tid,     // thread id
                                                            const uint32_t &warp_nid // warp nid
)
{
    constexpr uint32_t DOUBLE_NUMS_PER_SMEM_LINE = 16; // smem每行可存16个double
    constexpr uint32_t SMEM_N = N_WARPS * TOTAL_N_COUNT * 8;
    constexpr uint32_t SMEM_LINES_PER_NLINE = SMEM_N / DOUBLE_NUMS_PER_SMEM_LINE; // BANK视图下，SMEM_N相当于几行SMEM
    constexpr uint32_t SMEM_LINES_PER_MCOUNT =
        (64 * TOTAL_N_COUNT * N_WARPS) / DOUBLE_NUMS_PER_SMEM_LINE; // BANK试图下，每个m_count相对于几行SMEM
    uint32_t swizzle_id = ((n_count % 2) + ((tid / 4) % 2)) % 2;
    uint32_t smem_nline_id = (n_count / 2) + (warp_nid * (TOTAL_N_COUNT / 2));
    uint32_t smem_x = (tid % 4) * 2 + swizzle_id * 8;
    uint32_t smem_y = smem_nline_id + (tid / 4) * SMEM_LINES_PER_NLINE + m_count * SMEM_LINES_PER_MCOUNT;
    uint32_t smem_offset = smem_y * DOUBLE_NUMS_PER_SMEM_LINE + smem_x;

    FETCH_16B(smem_ptr0[smem_offset]) = reg_16B;
}

// 加载整个 ST_Buffer的M64N64的tensor tile到GMEM, N主序
__device__ __forceinline__ void WarpTileM64N64S2GTmaBulkTensorAsync(
    const CUtensorMap *out_map_ptr, // 输出矩阵的tensor map
    double *smem_ptr0,              // 共享内存基地址
    const uint32_t &b_mtile_id,     // block m tile
    const uint32_t &b_ntile_id,     // block n tile
    const uint32_t &b_path_id,      // block path
    const uint32_t &B               // batch size
)
{
    uint32_t warptile_coord_n = b_ntile_id * 64;
    uint32_t warptile_coord_m = (b_path_id * (B / 64) + b_mtile_id) * 64;
    uint64_t tmap_ptr = reinterpret_cast<uint64_t>(out_map_ptr);
    uint32_t real_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<void *>(smem_ptr0)));
    asm_cp_async_bulk_tensor_2d_global_shared(tmap_ptr, real_smem_ptr, warptile_coord_m, warptile_coord_n);
}

__device__ __forceinline__ void SwapDouble4(double a[4], double b[4])
{
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
    {
        double tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}

template <uint32_t DIM_0, uint32_t DIM_1, uint32_t DIM_2>
__device__ __forceinline__ void ZerosAccuReg(double accu[DIM_0][DIM_1][DIM_2])
{
#pragma unroll
    for (uint32_t _m = 0; _m < DIM_0; ++_m)
    {
#pragma unroll
        for (uint32_t _n = 0; _n < DIM_1; ++_n)
        {
#pragma unroll
            for (uint32_t _l = 0; _l < DIM_2; ++_l)
            {
                accu[_m][_n][_l] = 0;
            }
        }
    }
}

template <uint32_t LOAD_TAG_IDX> //
__device__ __forceinline__ void UpdateTags(uint16_t tags[2])
{
    constexpr uint32_t USE_TAG_IDX = 1 - LOAD_TAG_IDX;
    tags[USE_TAG_IDX] = tags[LOAD_TAG_IDX];
    tags[LOAD_TAG_IDX] = 1 - tags[LOAD_TAG_IDX];
}

/* Fused GMM:
 * 1. Tiling: M64N64K4, Warp=4
 * 2. Shared memory swizzle
 * 3. Multi-dim Tensor TMA async store (整个block一起存)
 * 4. K方向 2 stages pipline
 */
template <uint32_t M_WARPS = 2,                                                  // m方向warp数量
          uint32_t N_WARPS = 2,                                                  // n方向warp数量
          uint32_t W_TILE_M = 32,                                                // 每个warp的tile m大小
          uint32_t W_TILE_N = 32,                                                // 每个warp的tile n大小
          uint32_t TILE_K = 4>                                                   // tile k大小
__global__ void fused_gmm_kernel_v4(const __grid_constant__ CUtensorMap out_map, // 输出tensor map
                                    const double *__restrict__ y,                // [B, V]
                                    const double *__restrict__ w,                // [path, U, V, W]
                                    uint32_t num_paths,                          // path数量
                                    uint32_t B,                                  // batch size
                                    uint32_t U,                                  // U
                                    uint32_t V,                                  // V
                                    uint32_t W                                   // W
)
{
    // basic info
    uint32_t tid = threadIdx.x;
    uint32_t wid = threadIdx.y;
    uint32_t w_mid = wid / N_WARPS;
    uint32_t w_nid = wid % N_WARPS;
    uint32_t b_ntile_id = blockIdx.x;
    uint32_t b_mtile_id = blockIdx.y;
    uint32_t b_path_id = blockIdx.z;

    constexpr uint32_t TILE_M = M_WARPS * W_TILE_M;
    constexpr uint32_t TILE_N = N_WARPS * W_TILE_N;
    constexpr uint32_t WARP_PER_BLOCK = M_WARPS * N_WARPS;
    // 越界Block
    if (b_mtile_id * TILE_M >= B && b_ntile_id * TILE_N >= U * W)
        return;

    // 计算分块内各个warp的参数
    uint32_t b_m0 = b_mtile_id * TILE_M;
    uint32_t b_n0 = b_ntile_id * TILE_N;
    // 对于M和N矩阵，一个warp以最大带宽加载数据一次能加载几行(ROWS)、每行需要加载几次(PER_ROW)
    // M矩阵：K方向主序；N矩阵：N方向主序
    constexpr uint32_t W_LD16B_M_ROWS = CEIL_DIV(WARP_SIZE * 2, TILE_K); // 16
    constexpr uint32_t W_LD16B_N_ROWS = CEIL_DIV(WARP_SIZE * 2, TILE_N); // 1
    // MN矩阵需要的总加载次数
    constexpr uint32_t W_LD16B_M_COUNTS = CEIL_DIV(TILE_M * TILE_K, WARP_PER_BLOCK * WARP_SIZE * 2); // 1
    constexpr uint32_t W_LD16B_N_COUNTS = CEIL_DIV(TILE_N * TILE_K, WARP_PER_BLOCK * WARP_SIZE * 2); // 1
    // MN矩阵加载每行需要几个thread
    constexpr uint32_t THREADS_PER_M_ROW_LD = TILE_K / 2; // 2
    constexpr uint32_t THREADS_PER_N_ROW_LD = TILE_N / 2; // 32

    // thread m/n/k
    uint32_t t_ld_m0 = b_m0 + wid * W_LD16B_M_ROWS + (tid / THREADS_PER_M_ROW_LD);
    uint32_t t_ld_ak0 = (tid % THREADS_PER_M_ROW_LD) * 2;
    uint32_t t_ld_n0 = b_n0 + (tid % THREADS_PER_N_ROW_LD) * 2;
    uint32_t t_ld_bk0 = wid * W_LD16B_N_ROWS + (tid / THREADS_PER_N_ROW_LD);

    // SMEM
    extern __shared__ double smem[];
    constexpr uint32_t stages = 2; // 流水级
    constexpr uint32_t A_SMEM_SIZE_PER_STAGE = TILE_M * TILE_K;
    constexpr uint32_t B_SMEM_SIZE_PER_STAGE = TILE_N * TILE_K;
    constexpr uint32_t A_SMEM_SIZE = A_SMEM_SIZE_PER_STAGE * stages;
    constexpr uint32_t B_SMEM_SIZE = B_SMEM_SIZE_PER_STAGE * stages;
    constexpr uint32_t WARP_ST_BUFFER_SIZE = W_TILE_M * W_TILE_N;
    double *A_smem = smem;
    double *B_smem = A_smem + A_SMEM_SIZE;
    double *ST_smem = B_smem + B_SMEM_SIZE + w_mid * WARP_ST_BUFFER_SIZE * N_WARPS;

    // MMA指令参数
    constexpr uint32_t MMA_M = 16;
    constexpr uint32_t MMA_N = 8;
    constexpr uint32_t MMA_K = 4;

    // buffer tag
    constexpr uint32_t LOAD_TAG_IDX = 0;
    constexpr uint32_t USE_TAG_IDX = 1 - LOAD_TAG_IDX;
    uint16_t tags[2] = {0, 1};

    // preload
    uint32_t loop_k = 0;
    {
        // G2S
#pragma unroll
        for (uint32_t m_count = 0; m_count < W_LD16B_M_COUNTS; ++m_count)
        {
            WarpATileG2SDirect16BAsync<WARP_PER_BLOCK>((A_smem + tags[LOAD_TAG_IDX] * A_SMEM_SIZE_PER_STAGE), y,
                                                       t_ld_m0, t_ld_ak0, loop_k, (W_LD16B_M_ROWS * WARP_PER_BLOCK),
                                                       m_count, V, tid, wid);
        }
#pragma unroll
        for (uint32_t k_count = 0; k_count < W_LD16B_N_COUNTS; ++k_count)
        {
            WarpBTileG2SSwizzleAsync<WARP_PER_BLOCK>((B_smem + tags[LOAD_TAG_IDX] * B_SMEM_SIZE_PER_STAGE), w, t_ld_n0,
                                                     t_ld_bk0, loop_k, (W_LD16B_N_ROWS * WARP_PER_BLOCK), k_count,
                                                     b_path_id, U, V, W, tid, wid);
        }
        asm_cp_async_commit_group();
    }

    // Accumulator定义并置为0
    double accu[W_TILE_M / MMA_M][W_TILE_N / MMA_N][4];
    ZerosAccuReg<(W_TILE_M / MMA_M), (W_TILE_N / MMA_N), 4>(accu);

    // 等待预加载完成，更新tag
    asm_cp_async_waitgroup(0);
    UpdateTags<LOAD_TAG_IDX>(tags);
    __syncthreads();

    // 主循环
    for (loop_k = TILE_K; loop_k < (V - TILE_K); loop_k += TILE_K)
    {
        // G2S
#pragma unroll
        for (uint32_t m_count = 0; m_count < W_LD16B_M_COUNTS; ++m_count)
        {
            WarpATileG2SDirect16BAsync<WARP_PER_BLOCK>((A_smem + tags[LOAD_TAG_IDX] * A_SMEM_SIZE_PER_STAGE), y,
                                                       t_ld_m0, t_ld_ak0, loop_k, (W_LD16B_M_ROWS * WARP_PER_BLOCK),
                                                       m_count, V, tid, wid);
        }
#pragma unroll
        for (uint32_t k_count = 0; k_count < W_LD16B_N_COUNTS; ++k_count)
        {
            WarpBTileG2SSwizzleAsync<WARP_PER_BLOCK>((B_smem + tags[LOAD_TAG_IDX] * B_SMEM_SIZE_PER_STAGE), w, t_ld_n0,
                                                     t_ld_bk0, loop_k, (W_LD16B_N_ROWS * WARP_PER_BLOCK), k_count,
                                                     b_path_id, U, V, W, tid, wid);
        }
        asm_cp_async_commit_group();

        // S2R (m16n8k4)
        double B_reg[W_TILE_N / MMA_N];
        // 加载整个B矩阵Tile到寄存器中
#pragma unroll
        for (uint32_t _n = 0; _n < (W_TILE_N / MMA_N); ++_n)
        {
            WarpBSubtileN8K4S2RSwizzleSync<W_TILE_N / 8>(
                B_reg[_n], (B_smem + tags[USE_TAG_IDX] * B_SMEM_SIZE_PER_STAGE), _n, tid, w_nid);
        }

        // 加载A的同时做MMA
        for (uint32_t _m = 0; _m < (W_TILE_M / MMA_M); ++_m)
        {
            // 用M8K4 load一个M16K4要load 2次
            double A_reg[2];
            WarpASubtileM8K4S2RDirectSync<W_TILE_M / 8>(A_reg[0], (A_smem + tags[USE_TAG_IDX] * A_SMEM_SIZE_PER_STAGE),
                                                        (2 * _m), tid, w_mid);
            WarpASubtileM8K4S2RDirectSync<W_TILE_M / 8>(A_reg[1], (A_smem + tags[USE_TAG_IDX] * A_SMEM_SIZE_PER_STAGE),
                                                        (2 * _m + 1), tid, w_mid);

            // MMA
#pragma unroll
            for (uint32_t _n = 0; _n < (W_TILE_N / MMA_N); ++_n)
            {
                asm_mma_m16n8k4_f64_f64_f64_f64(accu[_m][_n][0], accu[_m][_n][1], accu[_m][_n][2], accu[_m][_n][3], // D
                                                A_reg[0], A_reg[1],                                                 // A
                                                B_reg[_n],                                                          // B
                                                accu[_m][_n][0], accu[_m][_n][1], accu[_m][_n][2], accu[_m][_n][3]  // C
                );
            }
        }

        asm_cp_async_waitgroup(0);
        UpdateTags<LOAD_TAG_IDX>(tags);
        __syncthreads();
    }

    // last loop (K方向会溢出2行/列)
    {
        uint32_t t_ld_m1 = b_m0 + wid * W_LD16B_M_ROWS + (tid / 2);
        uint32_t t_ld_ak1 = tid % 2;
        // G2S
#pragma unroll
        for (uint32_t m_count = 0; m_count < W_LD16B_M_COUNTS; ++m_count)
        {

            WarpATileG2SReverse8BAsync<WARP_PER_BLOCK>((A_smem + tags[LOAD_TAG_IDX] * A_SMEM_SIZE_PER_STAGE), y,
                                                       t_ld_m1, t_ld_ak1, loop_k, (W_LD16B_M_ROWS * WARP_PER_BLOCK),
                                                       m_count, V, tid, wid);
        }
        if (wid < 2) // N矩阵只需要load两行-> warp 0 && warp 1
        {
#pragma unroll
            for (uint32_t k_count = 0; k_count < W_LD16B_N_COUNTS; ++k_count)
            {
                WarpBTileG2SSwizzleAsync<WARP_PER_BLOCK>((B_smem + tags[LOAD_TAG_IDX] * B_SMEM_SIZE_PER_STAGE), w,
                                                         t_ld_n0, t_ld_bk0, loop_k, (W_LD16B_N_ROWS * WARP_PER_BLOCK),
                                                         k_count, b_path_id, U, V, W, tid, wid);
            }
        }
        asm_cp_async_commit_group();

        // S2R (m16n8k4)
        double B_reg[W_TILE_N / MMA_N];
        // 加载整个B矩阵Tile到寄存器中
#pragma unroll
        for (uint32_t _n = 0; _n < (W_TILE_N / MMA_N); ++_n)
        {
            WarpBSubtileN8K4S2RSwizzleSync<W_TILE_N / 8>(
                B_reg[_n], (B_smem + tags[USE_TAG_IDX] * B_SMEM_SIZE_PER_STAGE), _n, tid, w_nid);
        }

        // 加载A的同时做MMA
        for (uint32_t _m = 0; _m < (W_TILE_M / MMA_M); ++_m)
        {
            // 用M8K4 load一个M16K4要load 2次
            double A_reg[2];
            WarpASubtileM8K4S2RDirectSync<W_TILE_M / 8>(A_reg[0], (A_smem + tags[USE_TAG_IDX] * A_SMEM_SIZE_PER_STAGE),
                                                        (2 * _m), tid, w_mid);
            WarpASubtileM8K4S2RDirectSync<W_TILE_M / 8>(A_reg[1], (A_smem + tags[USE_TAG_IDX] * A_SMEM_SIZE_PER_STAGE),
                                                        (2 * _m + 1), tid, w_mid);

            // MMA
#pragma unroll
            for (uint32_t _n = 0; _n < (W_TILE_N / MMA_N); ++_n)
            {
                asm_mma_m16n8k4_f64_f64_f64_f64(accu[_m][_n][0], accu[_m][_n][1], accu[_m][_n][2], accu[_m][_n][3], // D
                                                A_reg[0], A_reg[1],                                                 // A
                                                B_reg[_n],                                                          // B
                                                accu[_m][_n][0], accu[_m][_n][1], accu[_m][_n][2], accu[_m][_n][3]  // C
                );
            }
        }

        asm_cp_async_waitgroup(0);
        UpdateTags<LOAD_TAG_IDX>(tags);
        __syncthreads();
    }

    // 尾声处理
    {
        // S2R (m16n8k4)
        double B_reg[W_TILE_N / MMA_N] = {0};
        // 加载整个B矩阵Tile到寄存器中
        if (tid % 4 < 2) // 最后两行为0不需要加载
        {
#pragma unroll
            for (uint32_t _n = 0; _n < (W_TILE_N / MMA_N); ++_n)
            {
                WarpBSubtileN8K4S2RSwizzleSync<W_TILE_N / 8>(
                    B_reg[_n], (B_smem + tags[USE_TAG_IDX] * B_SMEM_SIZE_PER_STAGE), _n, tid, w_nid);
            }
        }

        // 加载A的同时做MMA
        for (uint32_t _m = 0; _m < (W_TILE_M / MMA_M); ++_m)
        {
            // 用M8K4 load一个M16K4要load 2次
            double A_reg[2] = {0};
            if (tid % 4 < 2) // 最后两列为0不需要加载
            {
                WarpASubtileM8K4S2RReverseSync<W_TILE_M / 8>(
                    A_reg[0], (A_smem + tags[USE_TAG_IDX] * A_SMEM_SIZE_PER_STAGE), (2 * _m), tid, w_mid);
                WarpASubtileM8K4S2RReverseSync<W_TILE_M / 8>(
                    A_reg[1], (A_smem + tags[USE_TAG_IDX] * A_SMEM_SIZE_PER_STAGE), (2 * _m + 1), tid, w_mid);
            }
            // MMA
#pragma unroll
            for (uint32_t _n = 0; _n < (W_TILE_N / MMA_N); ++_n)
            {
                asm_mma_m16n8k4_f64_f64_f64_f64(                                        // 1
                    accu[_m][_n][0], accu[_m][_n][1], accu[_m][_n][2], accu[_m][_n][3], // D
                    A_reg[0], A_reg[1],                                                 // A
                    B_reg[_n],                                                          // B
                    accu[_m][_n][0], accu[_m][_n][1], accu[_m][_n][2], accu[_m][_n][3]  // C
                );
            }
        }
        /* NO SYNC */
    }

    /* 一个warp负责一个32 x 32的TILE，由4 x 4个M8N8的MMA-tile组成
     * 需要将这32 x 32个TILE的数据按照 Row major连续存储到该 warp 的 SMEM buffer中
     * 为了避免bank conflict，需要将(tid/4)%2 == 1的thread，n方向的accu两两互换
     */
    if ((tid >> 2) & 0x1 == 1) // (tid / 4) % 2 == 1
    {
#pragma unroll
        for (uint32_t _m = 0; _m < (W_TILE_M / MMA_M); ++_m)
        {
#pragma unroll
            for (uint32_t _n = 0; _n < (W_TILE_N / MMA_N); _n += 2)
            {
                SwapDouble4(accu[_m][_n], accu[_m][_n + 1]);
            }
        }
    }

    // ST accu to SMEM
#pragma unroll
    for (uint32_t _m = 0; _m < (W_TILE_M / 8); ++_m)
    {
#pragma unroll
        for (uint32_t _n = 0; _n < (W_TILE_N / 8); ++_n)
        {
            // 计算accu坐标
            // 每组accu四个double，前两个是m=0~7的，后两个是m=8~15的
            uint32_t accu_l0 = (_m & 0x1) << 1; // (out_m % 2) * 2
            uint32_t accu_m = _m >> 1;          // out_m / 2
            uint32_t accu_n = _n;
            // store to SMEM
            WarpTileM8N8R2SSwizzle2Sync<(W_TILE_N / 8), N_WARPS>(ST_smem, FETCH_16B(accu[accu_m][accu_n][accu_l0]), _m,
                                                                 _n, tid, w_nid);
        }
    }

    /* st.shared和cp.async.bulk.tensor.shared.global之间无法确保顺序一致性，
     * 需要显式地增加同步原语，同时确保同一 BLOCK 的所有thread全部写完smem
     *      https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar
     *      https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#ordering-memory-operations
     */
    asm_fence_proxy_async_shared_cta();
    __syncthreads();

    // ST to Global
    if (tid == 0 && wid == 0)
    {
        WarpTileM64N64S2GTmaBulkTensorAsync(&out_map, ST_smem, b_mtile_id, b_ntile_id, b_path_id, B);
        asm_cp_async_bulk_commit_group();
        asm_cp_async_bulk_waitgroup(0);
    }
}

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled()
{
    // Get pointer to cuTensorMapEncodeTiled
    cudaDriverEntryPointQueryResult driver_status;
    void *cuTensorMapEncodeTiled_ptr = nullptr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault,
                                     &driver_status);
    assert(driver_status == cudaDriverEntryPointSuccess);

    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

// reates a tensor map to describe a two-dimensional row-major array of size M x N
//      https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-multi-dimensional-arrays
void init_2d_tensormap_rmajor(CUtensorMap *tensor_map_ptr, // Host empty tensor map
                              void *tensor_ptr,            // global addr
                              const uint32_t &M,           // M
                              const uint32_t &N,           // N
                              const uint32_t &box_M,       // M of shared memory buffer
                              const uint32_t &box_N        // N of shared memory buffer
)
{
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {static_cast<uint64_t>(N), static_cast<uint64_t>(M)};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {static_cast<uint64_t>(N) * sizeof(double)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {box_N, box_M};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride[rank] = {1, 1};
    // Create the tensor descriptor.
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUresult res =
        cuTensorMapEncodeTiled(tensor_map_ptr, // CUtensorMap *tensorMap,
                               CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
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
                               CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                               // L2 Promotion can be used to widen the effect of a cache-policy to a wider
                               // set of L2 cache lines.
                               CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                               // Any element that is outside of bounds will be set to zero by the TMA transfer.
                               CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

torch::Tensor fctp_multi_path_mm(const torch::Tensor &y, // [B, V]
                                 const torch::Tensor &w  // [num_paths, U, V, W]
)
{
    TORCH_CHECK(y.dtype() == torch::kFloat64, "X must be float64");
    TORCH_CHECK(y.dim() == 2, "X must be of 2 dimention");
    TORCH_CHECK(w.dtype() == torch::kFloat64, "Y must be float64");
    TORCH_CHECK(w.dim() == 4, "Y must be of 4 dimention");
    TORCH_CHECK(w.size(2) == y.size(1), "the V dimention of y and w must be equal!");

    uint32_t B = y.size(0);
    uint32_t num_paths = w.size(0);
    uint32_t U = w.size(1);
    uint32_t V = w.size(2);
    uint32_t W = w.size(3);

    torch::Tensor out = torch::empty({num_paths, B, U, W}, y.options());

    constexpr uint32_t M_WARPS = 2;
    constexpr uint32_t N_WARPS = 2;
    constexpr uint32_t W_TILE_M = 32;
    constexpr uint32_t W_TILE_N = 32;
    constexpr uint32_t TILE_K = 4;
    constexpr uint32_t TILE_M = M_WARPS * W_TILE_M;
    constexpr uint32_t TILE_N = N_WARPS * W_TILE_N;
    constexpr uint32_t WARP_PER_BLOCK = M_WARPS * N_WARPS;

    cudaStream_t cur_stream = c10::cuda::getCurrentCUDAStream(y.device().index()).stream();
    TORCH_CHECK((U * W) % TILE_N == 0, "(U * W) must be divisible by %u", TILE_N);
    TORCH_CHECK(B % TILE_M == 0, "Dim B must be divisible by %u", TILE_M);
    TORCH_CHECK(V % TILE_K == 2, "Dim V mod %u must be 2", TILE_K);

    CUtensorMap out_map{};
    init_2d_tensormap_rmajor(&out_map, out.data_ptr<double>(), num_paths * B, U * W, TILE_M, TILE_N);

    dim3 grid(CEIL_DIV(U * W, TILE_N), CEIL_DIV(B, TILE_M), num_paths);
    dim3 block(WARP_SIZE, WARP_PER_BLOCK);
    constexpr uint32_t SMEM_SIZE = ((TILE_M * TILE_K + TILE_N * TILE_K) * 2 // 2 stages load buffer
                                    + (TILE_M * TILE_N))                    // store buffer
                                   * sizeof(double);

    auto cuda_func = fused_gmm_kernel_v4<M_WARPS, N_WARPS, W_TILE_M, W_TILE_N, TILE_K>;
    cudaFuncSetAttribute(cuda_func, cudaFuncAttributeMaxDynamicSharedMemorySize, 196 * 1024); // 196 KB

    cuda_func<<<grid, block, SMEM_SIZE, cur_stream>>>(out_map, y.data_ptr<double>(), w.data_ptr<double>(), num_paths, B,
                                                      U, V, W);
    return out;
}

TORCH_LIBRARY(fctp_multi_path_mm, m)
{
    m.def("forward", &fctp_multi_path_mm);
}
