#include "ptx_inst.cuh"
#include <c10/cuda/CUDAStream.h>
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
__device__ __forceinline__ void WarpATileG2SSwizzleAsync(
    double *smem_ptr0,            // 共享内存基地址
    const double *A_ptr0,         // 矩阵A基地址
    const uint32_t &t_m0,         // thread level的矩阵A的M方向偏移
    const uint32_t &t_ak0,        // thread level的矩阵A的K方向偏移
    const uint32_t &loop_k,       // k循环
    const uint32_t &m_stride,     // M方向步长
    const uint32_t &m_count,      // M方向计数
    const uint32_t &prefix_i_sum, // 当前path之前的所有path的i总数之和
    const uint32_t &path_size,    // 当前的path的大小
    const uint32_t &total_i,      // I的总数
    const uint32_t &B,            // B
    const uint32_t &U,            // U
    const uint32_t &tid,          // thread id
    const uint32_t &wid           // warp id
)
{
    // Global offset
    // [B * total_i, U], 已知 m, path -> [Batch, i]
    uint32_t t_m = t_m0 + m_count * m_stride;
    uint32_t t_batch = t_m / path_size;
    uint32_t t_i = (t_m % path_size) + prefix_i_sum;
    uint32_t t_u = t_ak0 + loop_k;
    uint64_t a_offset = t_batch * total_i * U + t_i * U + t_u;
    const double *A_ptr = A_ptr0 + a_offset;

    // SMEM offset
    // 每两个thread合为一组，占8Bank，4组之间swizzle
    constexpr uint32_t THREAD_LD_DOUBLES = 2;          // 每个thread每次load 16B，占4个bank
    constexpr uint32_t DOUBLE_NUMS_PER_SMEM_LINE = 16; // smem每行可存16个double
    constexpr uint32_t WARP_TAKE_SMEM_LINES = 4;       // 每个warp一次load的数据要占4行SMEM
    uint32_t group_id = (tid >> 1) & 0x3;              // (tid / 2) % 4
    uint32_t in_group_id = tid & 0x1;                  // tid % 2
    uint32_t smem_swizzle_y0 = tid >> 3;               // tid / 8
    uint32_t smem_swizzle_x0 = ((smem_swizzle_y0 ^ group_id) << 1) + in_group_id;
    uint32_t smem_y = smem_swizzle_y0 + wid * WARP_TAKE_SMEM_LINES + m_count * WARP_PER_BLOCK * WARP_TAKE_SMEM_LINES;
    uint32_t smem_x = smem_swizzle_x0 * THREAD_LD_DOUBLES;
    uint32_t smem_offset = smem_y * DOUBLE_NUMS_PER_SMEM_LINE + smem_x;
    uint32_t real_smem_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<void *>(smem_ptr0 + smem_offset)));

    // fetch req
    if (t_batch < B)
    { // Batch 维度没越界
        asm_cp_async_ca_l2_prefetch_128B(real_smem_ptr, A_ptr, 16);
    }
}

template <uint32_t WARP_PER_BLOCK>
__device__ __forceinline__ void WarpBTileG2SSwizzleAsync(double *smem_ptr0,        // 共享内存基地址
                                                         const double *B_ptr0,     // 矩阵B基地址
                                                         const uint32_t &t_n0,     // thread level的矩阵B的N方向偏移
                                                         const uint32_t &t_bk0,    // thread level的矩阵B的K方向偏移
                                                         const uint32_t &loop_k,   // k循环
                                                         const uint32_t &k_stride, // K方向步长
                                                         const uint32_t &k_count,  // K方向计数
                                                         const uint32_t &path_id,  // 当前的 path id
                                                         const uint32_t &U,        // U
                                                         const uint32_t &V,        // V
                                                         const uint32_t &tid,      // thread id
                                                         const uint32_t &wid       // warp id
)
{
    // Global offset
    // [path, U, V], k -> U, N -> V
    uint32_t t_u = loop_k + k_count * k_stride + t_bk0;
    uint32_t t_v = t_n0;
    uint64_t b_offset = path_id * U * V + t_u * V + t_v;
    const double *B_ptr = B_ptr0 + b_offset;

    // SMEM offset
    // 每两个thread合为一组，占8Bank，4组之间swizzle
    // 每个8x4的mma tile存在连续的两行，沿着N方向放矩阵
    // 每两个矩阵一起swizzle后存放
    constexpr uint32_t THREAD_LD_DOUBLES = 2;          // 每个thread每次load 16B，占4个bank
    constexpr uint32_t DOUBLE_NUMS_PER_SMEM_LINE = 16; // smem每行可存16个double
    constexpr uint32_t N8K4X2_TAKE_SMEM_LINES = 4;     // 每两个矩阵要占4行SMEM
    uint32_t group_id = (tid >> 1) & 0x3;              // (tid / 2) % 4
    uint32_t in_group_id = tid & 0x1;                  // tid % 2
    uint32_t warp_group_id = wid >> 1;                 // wid / 2
    uint32_t in_warp_group_id = wid & 0x1;             // wid % 2
    uint32_t smem_swizzle_y0 = in_warp_group_id * 2 + (tid >> 4);
    uint32_t smem_swizzle_x0 = ((smem_swizzle_y0 ^ group_id) << 1) + in_group_id;
    uint32_t smem_y = smem_swizzle_y0 + ((tid / 8) % 2) * N8K4X2_TAKE_SMEM_LINES +
                      warp_group_id * 2 * N8K4X2_TAKE_SMEM_LINES + k_count * WARP_PER_BLOCK * N8K4X2_TAKE_SMEM_LINES;
    uint32_t smem_x = smem_swizzle_x0 * THREAD_LD_DOUBLES;
    uint32_t smem_offset = smem_y * DOUBLE_NUMS_PER_SMEM_LINE + smem_x;
    uint32_t real_smem_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<void *>(smem_ptr0 + smem_offset)));

    // fetch req
    asm_cp_async_ca_l2_prefetch_256B(real_smem_ptr, B_ptr, 16);
}

// 加载一个A矩阵的8*4分块（Subtile）
template <uint32_t TOTAL_M_COUNT>
__device__ __forceinline__ void WarpASubtileM8K4S2RSwizzleSync(double &a_reg,           // dst
                                                               const double *smem_ptr0, // 共享内存基地址
                                                               const uint32_t &m_count, // m方向计数
                                                               const uint32_t &k_count, // k方向计数
                                                               const uint32_t &tid,     // thread id
                                                               const uint32_t &warp_mid // warp_m id
)
{
    constexpr uint32_t DOUBLE_NUMS_PER_SMEM_LINE = 16; // smem每行可存16个double
    constexpr uint32_t SWIZZLE_GROUP_DOUBLE_NUM = 4;   // 每4个double为一组
    uint32_t smem_swizzle_y0 = tid >> 2;               // tid / 4
    uint32_t smem_swizzle_x0 = (tid & 0x3) + ((k_count ^ (smem_swizzle_y0 & 0x3)) << 2);
    uint32_t smem_y = warp_mid * TOTAL_M_COUNT * 8 + m_count * 8 + smem_swizzle_y0;
    uint32_t smem_x = smem_swizzle_x0;
    uint32_t smem_offset = smem_y * DOUBLE_NUMS_PER_SMEM_LINE + smem_x;
    a_reg = smem_ptr0[smem_offset];
}

// 加载一个B矩阵的8*4分块（Subtile）
template <uint32_t TOTAL_N_COUNT, uint32_t N_WARPS>
__device__ __forceinline__ void WarpBSubtileN8K4S2RSwizzleSync(double &b_reg,           // dst
                                                               const double *smem_ptr0, // 共享内存基地址
                                                               const uint32_t &n_count, // n方向计数
                                                               const uint32_t &k_count, // k方向计数
                                                               const uint32_t &tid,     // thread id
                                                               const uint32_t &warp_nid // warp_n id
)
{
    constexpr uint32_t DOUBLE_NUMS_PER_SMEM_LINE = 16; // smem每行可存16个double
    constexpr uint32_t SWIZZLE_GROUP_DOUBLE_NUM = 4;   // 每4个double为一组
    constexpr uint32_t BLOCK_TILE_TOTAL_NCOUNTS = TOTAL_N_COUNT * N_WARPS;
    uint32_t block_tile_ncount_id = warp_nid * TOTAL_N_COUNT + n_count;
    uint32_t same_line_ncount_id = block_tile_ncount_id & 0x1; // count % 2
    uint32_t cross_line_ncount_id = block_tile_ncount_id >> 1; // count / 2
    uint32_t smem_unswizzle_x = same_line_ncount_id * 8 + (tid >> 2);
    uint32_t swizzle_group_id = smem_unswizzle_x >> 2;
    uint32_t in_swizzle_group_id = smem_unswizzle_x & 0x3;

    uint32_t smem_swizzle_y0 = tid & 0x3; // tid % 4
    uint32_t smem_swizzle_x0 = (swizzle_group_id ^ smem_swizzle_y0) * SWIZZLE_GROUP_DOUBLE_NUM + in_swizzle_group_id;
    uint32_t smem_y = k_count * (BLOCK_TILE_TOTAL_NCOUNTS / 2) * 4 + cross_line_ncount_id * 4 + smem_swizzle_y0;
    uint32_t smem_x = smem_swizzle_x0;
    uint32_t smem_offset = smem_y * DOUBLE_NUMS_PER_SMEM_LINE + smem_x;
    b_reg = smem_ptr0[smem_offset];
}

/* Fused GMM V1:
 * 1. Tiling: M32N32K16, Warp=4
 * 2. Shared memory swizzle
 * 3. 2 stage pipline (8 KB/block/stage, expecting Occupancy = 8 block/SM, need 128KB SMEM)
 *      https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#unified-shared-memory-l1-texture-cache
 * 4. better blocking strategy
 */
template <uint32_t M_WARPS = 2,                                   // m方向warp数量
          uint32_t N_WARPS = 2,                                   // n方向warp数量
          uint32_t W_TILE_M = 16,                                 // 每个warp的tile m大小
          uint32_t W_TILE_N = 16,                                 // 每个warp的tile n大小
          uint32_t TILE_K = 16>                                   // tile k大小
__global__ void fused_gmm_kernel_v2(const double *__restrict__ x, // [B, total_i, U]
                                    const double *__restrict__ w, // [num_paths, U, V]
                                    double *__restrict__ out,     // [B, total_i, V]
                                    idim_T i_dims,                // [num_paths]
                                    idim_T prefex_i_sum,          // [num_paths]
                                    uint32_t num_paths,           // path数量
                                    uint32_t total_i,             // i的总数
                                    uint32_t B,                   // batch size
                                    uint32_t U,                   // U
                                    uint32_t V,                   // V
                                    double cg_val                 // val
)
{
    constexpr uint32_t TILE_M = M_WARPS * W_TILE_M;
    constexpr uint32_t TILE_N = N_WARPS * W_TILE_N;
    constexpr uint32_t WARP_PER_BLOCK = M_WARPS * N_WARPS;

    // basic info
    uint32_t tid = threadIdx.x;
    uint32_t wid = threadIdx.y;
    uint32_t w_mid = wid / N_WARPS;
    uint32_t w_nid = wid % N_WARPS;
    uint32_t b_ntile_id = blockIdx.x;
    uint32_t b_bi_id = blockIdx.y;

    uint32_t block_nums_per_bi = CEIL_DIV(B, TILE_M);
    uint32_t b_i_id = b_bi_id / block_nums_per_bi;
    uint32_t b_path_id = 3;
#pragma unroll
    for (uint32_t _p = 0; _p < 3; ++_p)
    {
        b_path_id = (b_i_id >= prefex_i_sum._i[_p] && b_i_id < prefex_i_sum._i[_p + 1]) ? _p : b_path_id;
    }
    uint32_t b_path_size = i_dims._i[b_path_id];
    uint32_t b_prefix_i_sum = prefex_i_sum._i[b_path_id];
    uint32_t b_mtile_id = (b_i_id - b_prefix_i_sum) * block_nums_per_bi + (b_bi_id % block_nums_per_bi);

    // 越界Block
    if (b_mtile_id * TILE_M >= b_path_size * B)
    {
        return;
    }
    // 计算分块内各个warp的参数
    uint32_t b_m0 = b_mtile_id * TILE_M;
    uint32_t b_n0 = b_ntile_id * TILE_N;
    // 对于M和N矩阵，一个warp以最大带宽加载数据一次能加载几行(ROWS)、每行需要加载几次(PER_ROW)
    // M矩阵：K方向主序；N矩阵：N方向主序
    constexpr uint32_t W_LD16B_M_ROWS = CEIL_DIV(WARP_SIZE * 2, TILE_K); // 4
    // constexpr uint32_t W_LD16B_M_PER_ROW = CEIL_DIV(TILE_K, WARP_SIZE * 2); // 1
    constexpr uint32_t W_LD16B_N_ROWS = CEIL_DIV(WARP_SIZE * 2, TILE_N); // 2
    // constexpr uint32_t W_LD16B_N_PER_ROW = CEIL_DIV(TILE_N, WARP_SIZE * 2); // 1
    // MN矩阵需要的总加载次数
    constexpr uint32_t W_LD16B_M_COUNTS = CEIL_DIV(TILE_M * TILE_K, WARP_PER_BLOCK * WARP_SIZE * 2); // 2
    constexpr uint32_t W_LD16B_N_COUNTS = CEIL_DIV(TILE_N * TILE_K, WARP_PER_BLOCK * WARP_SIZE * 2); // 2
    // MN矩阵加载每行需要几个thread
    constexpr uint32_t THREADS_PER_M_ROW_LD = TILE_K / 2; // 8
    constexpr uint32_t THREADS_PER_N_ROW_LD = TILE_N / 2; // 16

    // thread m/n/k
    uint32_t t_ld_m0 = b_m0 + wid * W_LD16B_M_ROWS + (tid / THREADS_PER_M_ROW_LD);
    uint32_t t_ld_ak0 = (tid % THREADS_PER_M_ROW_LD) * 2;
    uint32_t t_ld_n0 = b_n0 + (tid % THREADS_PER_N_ROW_LD) * 2;
    uint32_t t_ld_bk0 = wid * W_LD16B_N_ROWS + (tid / THREADS_PER_N_ROW_LD);

    // SMEM
    extern __shared__ double smem[];
    constexpr uint32_t stage = 2; // 流水级
    constexpr uint32_t A_SMEM_SIZE_PER_STAGE = TILE_M * TILE_K;
    constexpr uint32_t B_SMEM_SIZE_PER_STAGE = TILE_N * TILE_K;
    constexpr uint32_t A_SMEM_SIZE = A_SMEM_SIZE_PER_STAGE * stage;
    constexpr uint32_t B_SMEM_SIZE = B_SMEM_SIZE_PER_STAGE * stage;
    double *A_smem = smem;
    double *B_smem = A_smem + A_SMEM_SIZE;

    // stage计数
    uint32_t ld_tag = 0;
    uint32_t use_tag = 1 - ld_tag;

    // 预取 SMEM(ld)
    {
        uint32_t loop_k = 0;
        // G2S
#pragma unroll
        for (uint32_t m_count = 0; m_count < W_LD16B_M_COUNTS; ++m_count)
        {
            WarpATileG2SSwizzleAsync<WARP_PER_BLOCK>((A_smem + ld_tag * A_SMEM_SIZE_PER_STAGE), x, t_ld_m0, t_ld_ak0,
                                                     loop_k, (W_LD16B_M_ROWS * WARP_PER_BLOCK), m_count, b_prefix_i_sum,
                                                     b_path_size, total_i, B, U, tid, wid);
        }
#pragma unroll
        for (uint32_t k_count = 0; k_count < W_LD16B_N_COUNTS; ++k_count)
        {
            WarpBTileG2SSwizzleAsync<WARP_PER_BLOCK>((B_smem + ld_tag * B_SMEM_SIZE_PER_STAGE), w, t_ld_n0, t_ld_bk0,
                                                     loop_k, (W_LD16B_N_ROWS * WARP_PER_BLOCK), k_count, b_path_id, U,
                                                     V, tid, wid);
        }
        asm_cp_async_commit_group();
    }

    // MMA指令参数
    constexpr uint32_t MMA_M = 16;
    constexpr uint32_t MMA_N = 8;
    constexpr uint32_t MMA_K = 4;

    // Accumulator定义并置为0
    double accu[W_TILE_M / MMA_M][W_TILE_N / MMA_N][4];
#pragma unroll
    for (uint32_t _m = 0; _m < W_TILE_M / MMA_M; ++_m)
    {
#pragma unroll
        for (uint32_t _n = 0; _n < W_TILE_N / MMA_N; ++_n)
        {
#pragma unroll
            for (uint32_t _l = 0; _l < 4; ++_l)
            {
                accu[_m][_n][_l] = 0.0;
            }
        }
    }

    // 更新标志位，等待预取的数据结束读取
    use_tag = ld_tag;
    ld_tag = 1 - ld_tag;
    asm_cp_async_waitgroup(0);
    __syncthreads();

    // 主循环
    for (uint32_t loop_k = TILE_K; loop_k < U; loop_k += TILE_K)
    {
        // G2S（本轮）
#pragma unroll
        for (uint32_t m_count = 0; m_count < W_LD16B_M_COUNTS; ++m_count)
        {
            WarpATileG2SSwizzleAsync<WARP_PER_BLOCK>((A_smem + ld_tag * A_SMEM_SIZE_PER_STAGE), x, t_ld_m0, t_ld_ak0,
                                                     loop_k, (W_LD16B_M_ROWS * WARP_PER_BLOCK), m_count, b_prefix_i_sum,
                                                     b_path_size, total_i, B, U, tid, wid);
        }
#pragma unroll
        for (uint32_t k_count = 0; k_count < W_LD16B_N_COUNTS; ++k_count)
        {
            WarpBTileG2SSwizzleAsync<WARP_PER_BLOCK>((B_smem + ld_tag * B_SMEM_SIZE_PER_STAGE), w, t_ld_n0, t_ld_bk0,
                                                     loop_k, (W_LD16B_N_ROWS * WARP_PER_BLOCK), k_count, b_path_id, U,
                                                     V, tid, wid);
        }
        asm_cp_async_commit_group();

        // 计算上一轮load的数据
        // S2R (m16n8k4)
        double B_reg[W_TILE_N / MMA_N][TILE_K / MMA_K];
        // 加载整个B矩阵Tile到寄存器中
#pragma unroll
        for (uint32_t _n = 0; _n < (W_TILE_N / MMA_N); ++_n)
        {
#pragma unroll
            for (uint32_t _k = 0; _k < (TILE_K / MMA_K); ++_k)
            {
                WarpBSubtileN8K4S2RSwizzleSync<W_TILE_N / 8, N_WARPS>(
                    B_reg[_n][_k], (B_smem + use_tag * B_SMEM_SIZE_PER_STAGE), _n, _k, tid, w_nid);
            }
        }
        // 加载A的同时做MMA
        for (uint32_t _m = 0; _m < (W_TILE_M / MMA_M); ++_m)
        {
            for (uint32_t _k = 0; _k < (TILE_K / MMA_K); ++_k)
            {
                // 用M8K4 load一个M16K4要load 2次
                double A_reg[2];
                WarpASubtileM8K4S2RSwizzleSync<W_TILE_M / 8>(A_reg[0], (A_smem + use_tag * A_SMEM_SIZE_PER_STAGE),
                                                             (2 * _m), _k, tid, w_mid);
                WarpASubtileM8K4S2RSwizzleSync<W_TILE_M / 8>(A_reg[1], (A_smem + use_tag * A_SMEM_SIZE_PER_STAGE),
                                                             (2 * _m + 1), _k, tid, w_mid);
                // Do MMA
#pragma unroll
                for (uint32_t _n = 0; _n < (W_TILE_N / MMA_N); ++_n)
                {
                    asm_mma_m16n8k4_f64_f64_f64_f64(
                        accu[_m][_n][0], accu[_m][_n][1], accu[_m][_n][2], accu[_m][_n][3], // D
                        A_reg[0], A_reg[1],                                                 // A
                        B_reg[_n][_k],                                                      // B
                        accu[_m][_n][0], accu[_m][_n][1], accu[_m][_n][2], accu[_m][_n][3]  // C
                    );
                }
            }
        }

        // 更新标志位，等待load返回
        use_tag = ld_tag;
        ld_tag = 1 - ld_tag;
        asm_cp_async_waitgroup(0);
        __syncthreads();
    }

    // 尾声处理
    {
        // S2R (m16n8k4)
        double B_reg[W_TILE_N / MMA_N][TILE_K / MMA_K];
        // 加载整个B矩阵Tile到寄存器中
#pragma unroll
        for (uint32_t _n = 0; _n < (W_TILE_N / MMA_N); ++_n)
        {
#pragma unroll
            for (uint32_t _k = 0; _k < (TILE_K / MMA_K); ++_k)
            {
                WarpBSubtileN8K4S2RSwizzleSync<W_TILE_N / 8, N_WARPS>(
                    B_reg[_n][_k], (B_smem + use_tag * B_SMEM_SIZE_PER_STAGE), _n, _k, tid, w_nid);
            }
        }
        // 加载A的同时做MMA
        for (uint32_t _m = 0; _m < (W_TILE_M / MMA_M); ++_m)
        {
            for (uint32_t _k = 0; _k < (TILE_K / MMA_K); ++_k)
            {
                // 用M8K4 load一个M16K4要load 2次
                double A_reg[2];
                WarpASubtileM8K4S2RSwizzleSync<W_TILE_M / 8>(A_reg[0], (A_smem + use_tag * A_SMEM_SIZE_PER_STAGE),
                                                             (2 * _m), _k, tid, w_mid);
                WarpASubtileM8K4S2RSwizzleSync<W_TILE_M / 8>(A_reg[1], (A_smem + use_tag * A_SMEM_SIZE_PER_STAGE),
                                                             (2 * _m + 1), _k, tid, w_mid);
                // Do MMA
#pragma unroll
                for (uint32_t _n = 0; _n < (W_TILE_N / MMA_N); ++_n)
                {
                    asm_mma_m16n8k4_f64_f64_f64_f64(
                        accu[_m][_n][0], accu[_m][_n][1], accu[_m][_n][2], accu[_m][_n][3], // D
                        A_reg[0], A_reg[1],                                                 // A
                        B_reg[_n][_k],                                                      // B
                        accu[_m][_n][0], accu[_m][_n][1], accu[_m][_n][2], accu[_m][_n][3]  // C
                    );
                }
            }
        }
    }

    // ST to Global
    for (uint32_t out_m0 = 0; out_m0 < (W_TILE_M / 8); ++out_m0)
    {
        for (uint32_t out_n0 = 0; out_n0 < (W_TILE_N / 8); ++out_n0)
        {
            // 计算accu坐标
            // 每组accu四个double，前两个是m=0~7的，后两个是m=8~15的
            uint32_t accu_l0 = (out_m0 & 0x1) << 1; // (out_m % 2) * 2
            uint32_t accu_m = out_m0 >> 1;          // out_m / 2
            uint32_t accu_n = out_n0;

            // 计算 global offset
            uint32_t out_m = b_m0 + w_mid * W_TILE_M + out_m0 * 8 + (tid >> 2);
            uint32_t out_n = b_n0 + w_nid * W_TILE_N + out_n0 * 8 + ((tid & 0x3) << 1);
            // 已知 path, m, n -> [B, i, V]
            uint32_t out_batch = out_m / b_path_size;
            uint32_t out_i = b_prefix_i_sum + out_m % b_path_size;
            uint32_t out_v = out_n;
            uint32_t out_offset = out_batch * total_i * V + out_i * V + out_v;
            // 乘 val
            accu[accu_m][accu_n][accu_l0] *= cg_val;
            accu[accu_m][accu_n][accu_l0 + 1] *= cg_val;
            // store re (sync)
            if (out_batch < B)
            {
                FETCH_16B(out[out_offset]) = FETCH_16B(accu[accu_m][accu_n][accu_l0]);
            }
        }
    }
}

torch::Tensor fused_gmm(const torch::Tensor &x,                 // [B, total_i, U]
                        const torch::Tensor &w,                 // [num_paths, U, V]
                        const std::vector<int64_t> &i_dims_vec, // i dims
                        double val)
{
    TORCH_CHECK(x.dtype() == torch::kFloat64, "X must be float64");
    TORCH_CHECK(x.dim() == 3, "X must be of 3 dimention");
    TORCH_CHECK(w.dtype() == torch::kFloat64, "Y must be float64");
    TORCH_CHECK(w.dim() == 3, "Y must be of 3 dimention");

    uint32_t B = x.size(0);
    uint32_t total_i = x.size(1);
    uint32_t U = x.size(2);
    uint32_t V = w.size(2);

    uint32_t num_paths = i_dims_vec.size();
    TORCH_CHECK(num_paths == 4, "num_paths must be 4");

    torch::Tensor out = torch::empty({B, total_i, V}, x.options());

    idim_T i_dims = {
        {(uint32_t)i_dims_vec[0], (uint32_t)i_dims_vec[1], (uint32_t)i_dims_vec[2], (uint32_t)i_dims_vec[3]}};
    idim_T prefix_i_sum = {{0, 0, 0, 0}};
#pragma unroll
    for (uint32_t i = 1; i < 4; ++i)
    {
        prefix_i_sum._i[i] = prefix_i_sum._i[i - 1] + i_dims_vec[i - 1];
    }

    constexpr uint32_t M_WARPS = 2;
    constexpr uint32_t N_WARPS = 2;
    constexpr uint32_t W_TILE_M = 16;
    constexpr uint32_t W_TILE_N = 16;
    constexpr uint32_t TILE_K = 16;
    constexpr uint32_t TILE_M = M_WARPS * W_TILE_M;
    constexpr uint32_t TILE_N = N_WARPS * W_TILE_N;
    constexpr uint32_t WARP_PER_BLOCK = M_WARPS * N_WARPS;

    TORCH_CHECK(V % TILE_N == 0, "Dim V must be divisible by %u", TILE_N);
    // TORCH_CHECK(B % TILE_M == 0, "Dim B must be divisible by %u", TILE_M);
    dim3 grid(CEIL_DIV(V, TILE_N), total_i * CEIL_DIV(B, TILE_M));
    dim3 block(WARP_SIZE, WARP_PER_BLOCK);

    constexpr uint32_t SMEM_SIZE = (TILE_M * TILE_K + TILE_N * TILE_K) * 2 * 8; // 2 stage, 8 B/Double
    cudaStream_t cur_stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();

    auto cuda_kernel = fused_gmm_kernel_v2<M_WARPS, N_WARPS, W_TILE_M, W_TILE_N, TILE_K>;
    cudaFuncSetAttribute(cuda_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 57); // 128 KB （128/228）

    cuda_kernel<<<grid, block, SMEM_SIZE, cur_stream>>>(x.data_ptr<double>(), w.data_ptr<double>(),
                                                        out.data_ptr<double>(), i_dims, prefix_i_sum, num_paths,
                                                        total_i, B, U, V, val);

    return out;
}

TORCH_LIBRARY(equi_linear, m)
{
    m.def("fused_gemm", &fused_gmm);
}
