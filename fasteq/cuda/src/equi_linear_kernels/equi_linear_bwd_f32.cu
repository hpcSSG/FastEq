#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "./device_cuda_helper.cuh"
#include "./impl.h"
#include "./ptx_inst.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

/* ===================== Device Helper Functions ===================== */
__device__ __forceinline__ void warp_ldmatirx_8x32f32_sw128B_sync(
    float d_reg[16],         // out
    float *smem_ptr,         // smem_ptr
    const uint32_t &cnt0,    // out base offset (0 or 1)
    const uint32_t &smem_y0, // smem_y base offset (in view of '32bit_T smem[y][32]')
    const uint32_t &warp_tid // warpgroup tid
)
{
#pragma unroll
    for (uint32_t cnt = 0; cnt < 2; ++cnt)
    {
        uint32_t sw_y = warp_tid % 8;
        uint32_t unsw_x_16B = (warp_tid / 8) + cnt * 4;
        uint32_t sw_x_16B = (sw_y ^ unsw_x_16B) % 8;
        uint32_t smem_offset = (sw_y + smem_y0) * 32 + sw_x_16B * 4;
        uint32_t real_smem_ptr =
            static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<void *>(smem_ptr + smem_offset)));
        asm_ldmatrix_x4(d_reg[cnt * 8 + cnt0 + 0], // r0
                        d_reg[cnt * 8 + cnt0 + 2], // r1
                        d_reg[cnt * 8 + cnt0 + 4], // r2
                        d_reg[cnt * 8 + cnt0 + 6], // r3
                        real_smem_ptr              // smem ptr
        );
    }
}

__device__ __forceinline__ void warpgroup_ldmatirx_64x32f32_sw128B_sync(
    float d_reg[16],          // out
    float *smem_ptr,          // smem_ptr
    const uint32_t &smem_y0,  // smem_y base offset (in view of '32bit_T smem[y][32]')
    const uint32_t &in_wg_tid // warpgroup tid
)
{
    uint32_t warp_tid = in_wg_tid % WARP_SIZE;
    uint32_t in_wg_wid = in_wg_tid / WARP_SIZE;
    uint32_t warp_smem_y0 = in_wg_wid * 16 + smem_y0;
#pragma unroll
    for (uint32_t cnt = 0; cnt < 2; ++cnt)
    {
        uint32_t warp_smem_y = warp_smem_y0 + 8 * cnt;
        warp_ldmatirx_8x32f32_sw128B_sync(d_reg, smem_ptr, cnt, warp_smem_y, warp_tid);
    }
}

__device__ __forceinline__ uint64_t make_smem_desc(float *ptr)
{
    // k-major
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    // Swizzled layouts: not used, assumed to be 1.
    desc |= (1llu) << 16;
    // The offset from the first 8 rows to the next 8 rows.
    desc |= matrix_descriptor_encode(1024) << 32;
    desc |= (1llu) << 62; // swizzle 128B
    return desc;
}

template <uint32_t TILE_N>
__device__ void wg_reg_swap_swizzle4_m64n32f32(float accu[TILE_N / 8][4], // accu
                                               const uint32_t &in_wg_tid  // in_wg_tid
)
{
    static_assert(TILE_N % 32 == 0);
    /* TILE_N must be divisable by 32 !!
     * input matrix m64nN shape: (same as wgmma.m64nNk8 defines)
     * output matrix m64nN - tilem4n32 shape:(8float, swizzle4)
     *    | T0~3{d0, d1}     |  T0~3{d4, d5}     |  T0~3{d8, d9}   | T0~3{d12, d13}
     *    | T4~7{d4, d5}     |  T4~7{d0, d1}     |  T4~7{d12, d13} | T4~7{d8, d9}
     *    | T8~11{d8, d9}    |  T8~11{d12, d13}  |  T8~11{d0, d1}  | T8~11{d4, d5}
     *    | T12~15{d12, d13} |  T12~15{d8, d9}   |  T12~15{d4, d5} | T12~15{d0, d1}
     * No bank-conflict when threads in a warp store float2 of same tags(e.g.{d0, d1}),
     * and the Matrix in SMEM matrix CU_TENSOR_MAP_SWIZZLE_NONE.
     */
    if (in_wg_tid % 8 >= 4)
    {
#pragma unroll
        for (uint32_t _cnt = 0; _cnt < (TILE_N / 32); ++_cnt)
        {
            swap_Nregs_f32<4>(accu[_cnt * 4 + 0], accu[_cnt * 4 + 1]);
            swap_Nregs_f32<4>(accu[_cnt * 4 + 2], accu[_cnt * 4 + 3]);
        }
    }
    if (in_wg_tid % 16 >= 8)
    {
#pragma unroll
        for (uint32_t _cnt = 0; _cnt < (TILE_N / 32); ++_cnt)
        {
            swap_Nregs_f32<4>(accu[_cnt * 4 + 0], accu[_cnt * 4 + 2]);
            swap_Nregs_f32<4>(accu[_cnt * 4 + 1], accu[_cnt * 4 + 3]);
        }
    }
}

template <uint32_t TILE_N>
__device__ void wg_R2S_swizzle4_m64n32f32(float *out_smem0,          // out smem base ptr
                                          float accu[TILE_N / 8][4], // accu
                                          const uint32_t &in_wg_tid  // in_wg_tid
)
{
    static_assert(TILE_N % 32 == 0);
    constexpr uint32_t SMEM_Y_STRIDE = TILE_N / 32; // N方向每32个数一行

    uint32_t smem_y0 = ((in_wg_tid / 32) * 16) + ((in_wg_tid % 32) / 4);
    uint32_t smem_in_swg_x0 = (in_wg_tid % 4) * 2;

    // We make swizzle4 within swg(swizzle group)
    uint32_t smem_unswg_x0 = (in_wg_tid / 4) % 4;
#pragma unroll
    for (uint32_t _y = 0; _y < SMEM_Y_STRIDE; ++_y)
    {
        uint32_t smem_y1 = smem_y0 + _y;
#pragma unroll
        for (uint32_t _cnt = 0; _cnt < 4; ++_cnt)
        {
            uint32_t smem_swg_x0 = _cnt ^ smem_unswg_x0; // (0~3)
            uint32_t smem_x = smem_swg_x0 * 8 + smem_in_swg_x0;
#pragma unroll
            for (uint32_t _idx = 0; _idx < 2; ++_idx) // accu[0,1] | accu[2, 3]
            {
                uint32_t smem_y = smem_y1 + _idx * 8;
                uint32_t smem_offset = smem_y * 32 + smem_x;
                FETCH_8B(out_smem0[smem_offset]) = FETCH_8B(accu[_y * 4 + _cnt][_idx * 2]);
            }
        }
    }
}

/* ========================================================== */
/* =================== P&C Step Functions =================== */

template <uint32_t TILE_K, uint32_t A_SMEM_SIZE_PER_STAGE, uint32_t B_SMEM_SIZE_PER_STAGE, uint32_t TOTAL_STAGES>
__device__ void mainloop_producer(barrier bars_ready[],          // buffer finish consuming barrier
                                  barrier bars_filled[],         // buffer finish filling barrier
                                  float *A_smem,                 //
                                  float *B_smem,                 //
                                  const CUtensorMap *grad_map,   //
                                  const CUtensorMap *w_map,      //
                                  const uint32_t &b_batch_id0,   //
                                  const uint32_t &b_grad_i_id,   //
                                  const uint32_t &b_out_path_id, //
                                  const uint32_t &b_u_id0,       //
                                  const uint32_t &V,             //
                                  const uint32_t &in_wg_tid      //
)
{
    // only producer wg tid0 execute
    if (in_wg_tid == 0)
    {
        uint32_t cur_buffer_id = 0;
        uint32_t coord_grad_i = b_grad_i_id;
        uint32_t coord_batch = b_batch_id0;
        uint32_t coord_out_path = b_out_path_id;
        uint32_t coord_u = b_u_id0;
        // loop
        uint32_t k_count;
        for (k_count = 0; k_count < (CEIL_DIV(V, TILE_K)); ++k_count)
        {
            // wait buffer finish consuming
            bars_ready[cur_buffer_id].arrive_and_wait();
            // load grad tile
            constexpr size_t TX_BYTES_PER_STAGE = (A_SMEM_SIZE_PER_STAGE + B_SMEM_SIZE_PER_STAGE) * sizeof(float);
            uint32_t coord_v = k_count * TILE_K;
            cde::cp_async_bulk_tensor_3d_global_to_shared(
                (A_smem + cur_buffer_id * A_SMEM_SIZE_PER_STAGE), // smem_ptr base
                grad_map,                                         // tensormap
                coord_v, coord_grad_i, coord_batch,               // coords
                bars_filled[cur_buffer_id]                        // barrier
            );
            cde::cp_async_bulk_tensor_3d_global_to_shared(
                (B_smem + cur_buffer_id * B_SMEM_SIZE_PER_STAGE), // smem_ptr base
                w_map,                                            // tensormap
                coord_v, coord_u, coord_out_path,                 // coords
                bars_filled[cur_buffer_id]                        // barrier
            );
            cuda::device::barrier_arrive_tx(bars_filled[cur_buffer_id], 1, TX_BYTES_PER_STAGE);
            // uint32_t next_coord_m0 = (k_count + 1) * TILE_K;
            // asm_cp_async_bulk_prefetch_tensor_3d_l2(grad_map, next_coord_m0, coord_m1, coord_m2);

            // switch to next buffer
            cur_buffer_id = (cur_buffer_id + 1) % TOTAL_STAGES;
        }
        // 循环结束时，bars_ready由于缺少producer wg的arrive，而没有被重置；
        // 需要等待所有buffer的tag闲置之后才能进行下一步操作。
#pragma unroll
        for (uint32_t i = 0; i < TOTAL_STAGES; ++i)
        {
            bars_ready[(cur_buffer_id + i) % TOTAL_STAGES].arrive_and_wait();
        }
    }
}

template <uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t A_SMEM_SIZE_PER_STAGE,
          uint32_t B_SMEM_SIZE_PER_STAGE, uint32_t PRODUCER_WG_NUM, uint32_t CONSUMER_WG_NUM, uint32_t TOTAL_STAGES>
__device__ void mainloop_consumer(barrier bars_ready[],  // buffer finish consuming barrier
                                  barrier bars_filled[], // buffer finish filling barrier
                                  float t_accu[(TILE_M / CONSUMER_WG_NUM) / 64][(TILE_N / 8)][4], //
                                  float *A_smem,                                                  //
                                  float *B_smem,                                                  //
                                  const uint32_t &V,                                              //
                                  const uint32_t &consumer_wg_id,                                 //
                                  const uint32_t &in_wg_tid                                       //
)
{
    // define wgmma(wgmma.mma_async.m64nNk8)
    constexpr uint32_t WGMMA_M = 64;
    constexpr uint32_t WGMMA_N = TILE_N;
    constexpr uint32_t WGMMA_K = 8;

    uint32_t cur_buffer_id = 0;
    // init (All buffers are empty)
#pragma unroll
    for (uint32_t i = 0; i < TOTAL_STAGES; ++i)
    {
        bars_ready[i].arrive(); // buffer_0 is ready for initial fill
    }
    // loop
    for (uint32_t k_count = 0; k_count < CEIL_DIV(V, TILE_K); ++k_count)
    {
        // Wait for the data to have arrived.
        bars_filled[cur_buffer_id].arrive_and_wait();

        // load A data from SMEM
        float a_reg[(TILE_M / CONSUMER_WG_NUM) / 64][16];
#pragma unroll
        for (uint32_t _m = 0; _m < ((TILE_M / CONSUMER_WG_NUM) / 64); ++_m)
        {
            warpgroup_ldmatirx_64x32f32_sw128B_sync(
                a_reg[_m],                                                      // out
                (A_smem + cur_buffer_id * A_SMEM_SIZE_PER_STAGE),               // smem_ptr
                (_m + consumer_wg_id * ((TILE_M / CONSUMER_WG_NUM) / 64)) * 64, // smem_y base offset
                in_wg_tid                                                       // warpgroup tid
            );
            // #pragma unroll
            //             for (uint32_t _l = 0; _l < 16; ++_l)
            //             {
            //                 uint32_t tmp;
            //                 asm_cvt_tf32_f32(tmp, a_reg[_m][_l]);
            //                 a_reg[_m][_l] = __uint_as_float(tmp);
            //             }
        }

        // do mma
        asm_warpgroup_arrive();
#pragma unroll
        for (uint32_t _k = 0; _k < (TILE_K / WGMMA_K); ++_k)
        {
            uint64_t b_desc = make_smem_desc(&B_smem[cur_buffer_id * B_SMEM_SIZE_PER_STAGE + WGMMA_K * _k]);
#pragma unroll
            for (uint32_t _m = 0; _m < ((TILE_M / CONSUMER_WG_NUM) / WGMMA_M); ++_m)
            {
                asm_wgmma_m64n32k8_tf32<1, 1, 1>(t_accu[_m], a_reg[_m][4 * _k], a_reg[_m][4 * _k + 1],
                                                 a_reg[_m][4 * _k + 2], a_reg[_m][4 * _k + 3], b_desc);
            }
        }
        asm_warpgroup_commit_batch();
        asm_warpgroup_wait(0);

        // tell producer that consuming finished
        bars_ready[cur_buffer_id].arrive();

        // switch to next buffer
        cur_buffer_id = (cur_buffer_id + 1) % TOTAL_STAGES;
    }
}

template <uint32_t TILE_M, uint32_t TILE_N, uint32_t CONSUMER_WG_NUM>
__device__ void store_re_async_consumer(
    float *O_smem,                                                  // output smem buffer base addr
    float t_accu[(TILE_M / CONSUMER_WG_NUM) / 64][(TILE_N / 8)][4], // accumulator regs
    const CUtensorMap *out_map,                                     // out tensor map
    const uint32_t &b_batch_id0,                                    // b_batch_id0
    const uint32_t &b_out_i_id,                                     // b_out_i_id
    const uint32_t &b_u_id0,                                        // b_u_id0
    const float &cg_val,                                            // cg_val
    const uint32_t &consumer_wg_id,                                 // consumer warpgroup id
    const uint32_t &in_wg_tid                                       // in warpgroup tid
)
{
    constexpr uint32_t SMEM_SIZE_PER_M_TILE = 64 * TILE_N;
    constexpr uint32_t OUT_SMEM_SIZE_PER_CONS_WG = (TILE_M / CONSUMER_WG_NUM) * TILE_N;
    // Swizzle regs within each thread to avoid SMEM bank-conflict.
    // Target layout (Layout from reg to SMEM): swizzle 4 (each element is 32B);
    // Target SMEM layout (Layout from SMEM to GLOBAL): row major, swizzle NONE;
    for (uint32_t _m = 0; _m < (TILE_M / CONSUMER_WG_NUM) / 64; ++_m)
    {
        // Attach cg_val effect
#pragma unroll
        for (uint32_t _n = 0; _n < (TILE_N / 8); ++_n)
        {
#pragma unroll
            for (uint32_t _cnt = 0; _cnt < 4; ++_cnt)
            {
                t_accu[_m][_n][_cnt] *= cg_val;
            }
        }
        wg_reg_swap_swizzle4_m64n32f32<TILE_N>(t_accu[_m], in_wg_tid);
        wg_R2S_swizzle4_m64n32f32<TILE_N>(
            (O_smem + consumer_wg_id * OUT_SMEM_SIZE_PER_CONS_WG + _m * SMEM_SIZE_PER_M_TILE), //
            t_accu[_m],                                                                        //
            in_wg_tid                                                                          //
        );
    }

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    asm_warpgroup_arrive();

    // Send TMA store request
    if (in_wg_tid == 0)
    {
        uint32_t coord_n0 = b_u_id0;                                                   // U
        uint32_t coord_n1 = b_out_i_id;                                                // out_total_i
        uint32_t coord_n2 = b_batch_id0 + consumer_wg_id * (TILE_M / CONSUMER_WG_NUM); // B
        // We transfer a (TILE_M x TIME_N) tile each time
        cde::cp_async_bulk_tensor_3d_shared_to_global(
            out_map,                                              // tensor map
            coord_n0, coord_n1, coord_n2,                         // coord
            (O_smem + consumer_wg_id * OUT_SMEM_SIZE_PER_CONS_WG) // smem start ptr
        );
        // Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cde::cp_async_bulk_commit_group();
    }
}

template <uint32_t GRAD_NUM_PATHS, uint32_t OUT_NUM_PATHS>
__device__ uint32_t calculate_out_i(const idim_T<GRAD_NUM_PATHS> &grad_prefex_i_sum, // [GRAD_NUM_PATHS]
                                    const idim_T<OUT_NUM_PATHS> &out_prefex_i_sum,   // [OUT_NUM_PATHS]
                                    const uint32_t &b_grad_i_id,                     // b_grad_i_id
                                    const uint32_t &b_out_path_id                    // b_out_path_id
)
{
    // get path info
    uint32_t cur_prefex_i_sum = 0;
#pragma unroll
    for (uint32_t n = 0; n < OUT_NUM_PATHS; ++n)
    {
        cur_prefex_i_sum = b_grad_i_id >= grad_prefex_i_sum._i[n] ? grad_prefex_i_sum._i[n] : cur_prefex_i_sum;
    }
    uint32_t offen = b_grad_i_id - cur_prefex_i_sum;
    return out_prefex_i_sum._i[b_out_path_id] + offen;
}

template <uint32_t GRAD_NUM_PATHS>
__device__ uint32_t calculate_grad_path(const idim_T<GRAD_NUM_PATHS> &grad_prefex_i_sum, // [GRAD_NUM_PATHS]
                                        const uint32_t &b_grad_i_id                      // b_grad_i_id
)
{
    uint32_t gp = 0;
#pragma unroll
    for (uint32_t n = 0; n < GRAD_NUM_PATHS; ++n)
    {
        gp = b_grad_i_id >= grad_prefex_i_sum._i[n] ? n : gp;
    }
    return gp;
}

/* ============================================================= */
/* =================== Main Kernel Functions =================== */

template <uint32_t GRAD_NUM_PATHS,      // in path数量
          uint32_t OUT_NUM_PATHS,       // out path数量
          uint32_t TILE_M,              // tile m大小
          uint32_t TILE_N,              // tile n大小
          uint32_t TILE_K = 32,         // tile k大小
          uint32_t PRODUCER_WG_NUM = 1, // producer warpgroup数量
          uint32_t CONSUMER_WG_NUM = 2  // consumer warpgroup数量
          >
__global__ void mutipath_equi_linear_bwd_f32_tf32_kernel(
    const __grid_constant__ CUtensorMap grad_map,  // grad tensor maps
    const __grid_constant__ CUtensorMap w_map,     // w tensor maps
    const __grid_constant__ CUtensorMap out_map,   // out tensor maps
    cg_T<OUT_NUM_PATHS> cg_vals,                   // cg_vals
    idim_T<GRAD_NUM_PATHS> grad_prefex_i_sum,      // [GRAD_NUM_PATHS]
    idim_T<OUT_NUM_PATHS> out_prefex_i_sum,        // [OUT_NUM_PATHS]
    idim_T<GRAD_NUM_PATHS> gradpath_repeated_nums, // [GRAD_NUM_PATHS]
    uint32_t B,                                    // batch size
    uint32_t U,                                    // U
    uint32_t V,                                    // V
    uint32_t grad_total_i,                         // input i的总数
    uint32_t out_total_i                           // output i的总数
)
{
    /* tensor shape:
     * grad: [B, grad_total_i, V]
     * w: [OUT_NUM_PATHS, U, V]
     * out: [B, out_total_i, U]
     * dim3 grid(max_reduce_path_nums * CEIL_DIV(U, TILE_N), grad_total_i * CEIL_DIV(B, TILE_M));
     */
    // basic info
    constexpr uint32_t WARPGROUP_NUM = PRODUCER_WG_NUM + CONSUMER_WG_NUM;
    constexpr uint32_t PRODUCER_WG_ID = 0;
    uint32_t in_wg_tid = threadIdx.x;
    uint32_t wg_id = threadIdx.y;
    uint32_t b_ntile_id = blockIdx.x;
    uint32_t b_mtile_id = blockIdx.y;

    // get path info
    uint32_t b_batch_id0 = (b_mtile_id % (CEIL_DIV(B, TILE_M))) * TILE_M;
    uint32_t b_grad_i_id = b_mtile_id / (CEIL_DIV(B, TILE_M));
    uint32_t b_u_id0 = (b_ntile_id % (CEIL_DIV(U, TILE_N))) * TILE_N;

    uint32_t b_grad_path_id = calculate_grad_path<GRAD_NUM_PATHS>(grad_prefex_i_sum, b_grad_i_id);
    uint32_t b_out_path_offset = b_ntile_id / (CEIL_DIV(U, TILE_N));
    uint32_t b_out_path_id = b_out_path_offset; // here we only got an index of outpath id
    for (uint32_t p = 0; p < b_grad_path_id; ++p)
    {
        b_out_path_id += gradpath_repeated_nums._i[p];
    }

    // check whether current (b_grad_i_id, b_out_path_id) pair is valid
    if (b_out_path_offset >= gradpath_repeated_nums._i[b_grad_path_id])
        return;

    // define SMEM-buffer
    extern __shared__ __align__(1024) float smem[];
    constexpr uint32_t TOTAL_STAGES = 2; // 流水级
    static_assert((TILE_M * TILE_N) % CONSUMER_WG_NUM == 0);
    constexpr uint32_t OUT_SMEM_SIZE_PER_CONS_WG = (TILE_M * TILE_N) / CONSUMER_WG_NUM;
    constexpr uint32_t A_SMEM_SIZE_PER_STAGE = TILE_M * TILE_K;
    constexpr uint32_t B_SMEM_SIZE_PER_STAGE = TILE_N * TILE_K;
    // constexpr uint32_t OUT_SMEM_SIZE = OUT_SMEM_SIZE_PER_CONS_WG * CONSUMER_WG_NUM;
    constexpr uint32_t A_SMEM_SIZE = A_SMEM_SIZE_PER_STAGE * TOTAL_STAGES;
    constexpr uint32_t B_SMEM_SIZE = B_SMEM_SIZE_PER_STAGE * TOTAL_STAGES;
    float *A_smem = smem; // must align 1024
    float *B_smem = A_smem + A_SMEM_SIZE;
    float *O_smem = B_smem + B_SMEM_SIZE;

    // define & init barrier
    // for buffer id=s:
    //      bars[0~OUTER_BARRIER_NUMS]: 保留备用
    //      bars[OUTER_BARRIER_NUMS + s]: buffer 's' ready to be filled barrier; (bars_ready)
    //      bars[OUTER_BARRIER_NUMS + TOTAL_STAGES + s]: buffer 's' are filled-in barrier; (bars_filled)
    constexpr uint32_t OUTER_BARRIER_NUMS = 0;
    constexpr uint32_t TOTAL_BARRIER_NUMS = OUTER_BARRIER_NUMS + TOTAL_STAGES * 2;
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bars[TOTAL_BARRIER_NUMS];
    static_assert(TOTAL_BARRIER_NUMS <= 128); // 不同thread并行初始化barriers
    if (in_wg_tid < TOTAL_BARRIER_NUMS && wg_id == 0)
    {
        // Initialize barrier.
        init(&bars[in_wg_tid], (CONSUMER_WG_NUM * 128 + 1));
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
    }
    barrier *bars_ready = bars + OUTER_BARRIER_NUMS;
    barrier *bars_filled = bars + OUTER_BARRIER_NUMS + TOTAL_STAGES;
    // Syncthreads so initialized barrier is visible to all threads.
    __syncthreads();

    if (wg_id == PRODUCER_WG_ID)
    {
        mainloop_producer<TILE_K, A_SMEM_SIZE_PER_STAGE, B_SMEM_SIZE_PER_STAGE, TOTAL_STAGES>(
            bars_ready,    // buffer finish consuming barrier
            bars_filled,   // buffer finish filling barrier
            A_smem,        //
            B_smem,        //
            &grad_map,     //
            &w_map,        //
            b_batch_id0,   //
            b_grad_i_id,   //
            b_out_path_id, //
            b_u_id0,       //
            V,             //
            in_wg_tid      //
        );

        /* STEP3: Store accumulator */
        // Producer do nothing
    }
    else
    {
        uint32_t consumer_wg_id = wg_id - PRODUCER_WG_NUM;
        // define accumulators
        float t_accu[(TILE_M / CONSUMER_WG_NUM) / 64][(TILE_N / 8)][4] = {0.0f};

        cde::fence_proxy_async_shared_cta();
        mainloop_consumer<TILE_M, TILE_N, TILE_K, A_SMEM_SIZE_PER_STAGE, B_SMEM_SIZE_PER_STAGE, PRODUCER_WG_NUM,
                          CONSUMER_WG_NUM, TOTAL_STAGES>(bars_ready,     // buffer finish consuming barrier
                                                         bars_filled,    // buffer finish filling barrier
                                                         t_accu,         //
                                                         A_smem,         //
                                                         B_smem,         //
                                                         V,              //
                                                         consumer_wg_id, //
                                                         in_wg_tid       //
        );

        /* STEP3: Store accumulator */
        // calculate b_out_i_id
        uint32_t b_out_i_id = calculate_out_i<GRAD_NUM_PATHS, OUT_NUM_PATHS>(grad_prefex_i_sum, out_prefex_i_sum,
                                                                             b_grad_i_id, b_out_path_id);
        float cur_cg_val = cg_vals._v[b_out_path_id];
        store_re_async_consumer<TILE_M, TILE_N, CONSUMER_WG_NUM>(O_smem,         // output smem buffer base addr
                                                                 t_accu,         // accumulator regs
                                                                 &out_map,       // out tensor map
                                                                 b_batch_id0,    // b_batch_id0
                                                                 b_out_i_id,     // b_out_i_id
                                                                 b_u_id0,        // b_u_id0
                                                                 cur_cg_val,     // cg_val
                                                                 consumer_wg_id, // consumer warpgroup id
                                                                 in_wg_tid       // in warpgroup tid
        );
        // Wait for accumulator finishing storing.
        if (in_wg_tid == 0)
        {
            cde::cp_async_bulk_wait_group_read<0>();
        }
    }
}

template <uint32_t OUT_NUM_PATHS, uint32_t GRAD_NUM_PATHS = 4>
void mutipath_equi_linear_bwd_f32(float *out,                                  // [B, out_total_i, U]
                                  float *grad,                                 // [B, grad_total_i, V]
                                  float *w,                                    // [OUT_NUM_PATHS, U, V]
                                  const std::vector<int64_t> &grad_i_dims_vec, // in i dims
                                  const std::vector<int64_t> &out_i_dims_vec,  // out i dims
                                  const uint32_t &B,                           // batch
                                  const uint32_t &grad_total_i,                // grad_total_i
                                  const uint32_t &out_total_i,                 // out_total_i
                                  const uint32_t &U,                           // U
                                  const uint32_t &V,                           // V
                                  const std::vector<double> &cg_val_vec,       // cg_vals
                                  const cudaStream_t &cur_stream               // current stream
)
{
    constexpr uint32_t PRODUCER_WG_NUM = 1; // producer warpgroup数量
    constexpr uint32_t CONSUMER_WG_NUM = 2; // consumer warpgroup数量
    constexpr uint32_t WARPGROUP_NUM = PRODUCER_WG_NUM + CONSUMER_WG_NUM;
    constexpr uint32_t TILE_K = 32;
    constexpr uint32_t TILE_M = 128;
    constexpr uint32_t TILE_N = 32;

    static_assert(TILE_K == 32);
    assert(U % TILE_N == 0);

    idim_T<GRAD_NUM_PATHS> grad_prefex_i_sum;
    cal_prefex_sum<GRAD_NUM_PATHS>(grad_prefex_i_sum, grad_i_dims_vec);

    idim_T<OUT_NUM_PATHS> out_prefex_i_sum;
    cal_prefex_sum<OUT_NUM_PATHS>(out_prefex_i_sum, out_i_dims_vec);

    idim_T<GRAD_NUM_PATHS> gradpath_repeated_nums;
    uint32_t max_reduce_path_nums = 0;
    {
        int64_t _cur_grad_path = 0;
        uint32_t _last_same_size_path_nums = 0;
        for (uint32_t _n = 0; _n < OUT_NUM_PATHS; ++_n)
        {
            if (_n != 0 && out_i_dims_vec[_n] != out_i_dims_vec[_n - 1])
            {
                max_reduce_path_nums = std::max(_last_same_size_path_nums, max_reduce_path_nums);
                gradpath_repeated_nums._i[_cur_grad_path] = _last_same_size_path_nums;
                _last_same_size_path_nums = 0;
                ++_cur_grad_path;
            }
            ++_last_same_size_path_nums;
        }
        gradpath_repeated_nums._i[_cur_grad_path] = _last_same_size_path_nums;
    }

    cg_T<OUT_NUM_PATHS> cg_vals(cg_val_vec);

    CUtensorMap grad_map{};
    CUtensorMap w_map{};
    CUtensorMap out_map{};
    init_3d_tensormap(&grad_map,                                     // tensor_map_ptr
                      static_cast<void *>(grad),                     // tensor_ptr
                      B,                                             // B
                      grad_total_i,                                  // M
                      V,                                             // N
                      grad_total_i * V,                              // stride B
                      V,                                             // stride M
                      TILE_M,                                        // box B
                      1,                                             // box M
                      TILE_K,                                        // box N
                      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B // swizzle 128B
    );
    init_3d_tensormap(&w_map,                                        // tensor_map_ptr
                      static_cast<void *>(w),                        // tensor_ptr
                      OUT_NUM_PATHS,                                 // B
                      U,                                             // M
                      V,                                             // N
                      U * V,                                         // stride B
                      V,                                             // stride M
                      1,                                             // box B
                      TILE_N,                                        // box M
                      TILE_K,                                        // box N
                      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B // swizzle 128B
    );
    init_3d_tensormap(&out_map,                   // tensor_map_ptr
                      static_cast<void *>(out),   // tensor_ptr
                      B,                          // B
                      out_total_i,                // M
                      U,                          // N
                      out_total_i * U,            // stride B
                      U,                          // stride M
                      (TILE_M / CONSUMER_WG_NUM), // box B
                      1,                          // box M
                      TILE_N                      // box N
    );
    dim3 grid(max_reduce_path_nums * CEIL_DIV(U, TILE_N), grad_total_i * CEIL_DIV(B, TILE_M));
    dim3 block(128, WARPGROUP_NUM);

    constexpr uint32_t STAGES = 2;
    size_t SMEM_SIZE = sizeof(float) * ((TILE_M * TILE_K + TILE_N * TILE_K) * STAGES // A_smem & B_smem
                                        + (TILE_M * TILE_N));                        // Out_smem

    auto cuda_kernel = mutipath_equi_linear_bwd_f32_tf32_kernel<GRAD_NUM_PATHS,  // in path数量
                                                                OUT_NUM_PATHS,   // out path数量
                                                                TILE_M,          // tile m大小
                                                                TILE_N,          // tile n大小
                                                                TILE_K,          // tile k大小
                                                                PRODUCER_WG_NUM, // producer warpgroup数量
                                                                CONSUMER_WG_NUM  // consumer warpgroup数量
                                                                >;
    cudaFuncSetAttribute(cuda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);
    cuda_kernel<<<grid, block, SMEM_SIZE, cur_stream>>>(grad_map,               // grad tensor maps
                                                        w_map,                  // w tensor maps
                                                        out_map,                // out tensor maps
                                                        cg_vals,                // cg vals
                                                        grad_prefex_i_sum,      // [GRAD_NUM_PATHS]
                                                        out_prefex_i_sum,       // [OUT_NUM_PATHS]
                                                        gradpath_repeated_nums, // [GRAD_NUM_PATHS]
                                                        B,                      // batch size
                                                        U,                      // U
                                                        V,                      // V
                                                        grad_total_i,           // input i的总数
                                                        out_total_i             // output i的总数
    );
}

// wrapper
void mutipath_equi_linear_bwd_f32_impl(const uint32_t &out_num_paths,          // path nums
                                       float *out,                             // [B, out_total_i, U]
                                       float *grad,                            // [B, grad_total_i, V]
                                       float *w,                               // [OUT_NUM_PATHS, U, V]
                                       const uint32_t &B,                      // batch
                                       const uint32_t &out_total_i,            // out_total_i
                                       const std::vector<int64_t> &out_i_dims, // out i dims
                                       const uint32_t &U,                      // U
                                       const uint32_t &V,                      // V
                                       const std::vector<double> &cg_val_vec,  // cg_vals
                                       const cudaStream_t &cur_stream          // current stream
)
{
    const std::vector<int64_t> grad_i_dims = {1, 3, 5, 7};
    const uint32_t grad_total_i = 16;
    auto call_impl = [&](auto &&...forwarded_args) {
        switch (out_num_paths)
        {
        case 4: // small
            mutipath_equi_linear_bwd_f32<4>(std::forward<decltype(forwarded_args)>(forwarded_args)...);
            break;
        case 10: // medium
            mutipath_equi_linear_bwd_f32<10>(std::forward<decltype(forwarded_args)>(forwarded_args)...);
            break;
        case 17: // large
            mutipath_equi_linear_bwd_f32<17>(std::forward<decltype(forwarded_args)>(forwarded_args)...);
            break;
        default:
            throw std::invalid_argument("Unsupported number of paths: " + std::to_string(out_num_paths) +
                                        ". Supported values are 4, 10, and 17.");
        }
    };

    // 调用lambda，完美转发参数
    call_impl(out, grad, w, grad_i_dims, out_i_dims, B, grad_total_i, out_total_i, U, V, cg_val_vec, cur_stream);
}