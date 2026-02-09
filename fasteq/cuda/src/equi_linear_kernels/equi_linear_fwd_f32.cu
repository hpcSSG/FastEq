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

template <uint32_t _N> struct idim_T
{
    uint32_t _i[_N];
};

struct param_maps_T
{
    CUtensorMap _x_map;
    CUtensorMap _w_map;
};

/* ===================== Device Helper Functions ===================== */

__device__ __forceinline__ void wg_trans_16x32f32tf32_sw128B_sync(
    float *out_smem_ptr,            // out buffer ptr to store output 32*32 data
    float *tmp_smem_ptr,            // tmp buffer ptr containning input 16x32 data
    const uint32_t &base_sw_x0_tag, // 0 or 1 (indicate swizzle 16B_x starts from 0 or 4)
    const uint32_t &in_wg_tid       // in warpgroup tid
)
{
    // sw128B means the transposed matrix leading dimention must be 32(floats).
    // so each 16x32 input matrix will be transposed to a 32x16 matirx,
    // and then make 128B swizzle with a neighbor 32x16 matirx (i.e. 32x32 matirx sw128B)
    uint32_t ld_col = in_wg_tid % 32;
    uint32_t ld_row0 = (in_wg_tid / 32) * 4;
    float tmp_reg[4];
#pragma unroll
    for (uint32_t cnt = 0; cnt < 4; ++cnt)
    {
        uint32_t ld_row = ld_row0 + cnt;
        float fp32_data = tmp_smem_ptr[ld_row * 32 + ld_col];
        uint32_t tf32_data;
        asm_cvt_tf32_f32(tf32_data, fp32_data);
        tmp_reg[cnt] = __uint_as_float(tf32_data);
    }

    // store to out smem sync
    uint32_t st_sw_y = ld_col;
    uint32_t st_unsw_16B_x = base_sw_x0_tag * 4 + (in_wg_tid / 32); // 0~7
    uint32_t st_sw_x = ((st_sw_y % 8) ^ st_unsw_16B_x) * 4;
    FETCH_16B(out_smem_ptr[st_sw_y * 32 + st_sw_x]) = FETCH_16B(tmp_reg[0]);
}

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

template <uint32_t TILE_N, uint32_t TILE_K, uint32_t TOTAL_STAGES>
__device__ void preloop_producer(barrier bars_ready[],         // buffer finish consuming barrier
                                 barrier bars_filled[],        // buffer finish filling barrier
                                 float *tmp_smem,              //
                                 const CUtensorMap *w_map,     //
                                 const uint32_t &b_ntile_id,   //
                                 const uint32_t &b_in_path_id, //
                                 const uint32_t &U,            //
                                 const uint32_t &in_wg_tid     //
)
{
    constexpr uint32_t PRELOOP_BUFFER_SIZE_PER_STAGE = TILE_N * TILE_K;

    uint32_t cur_buffer_id = 0;
    uint32_t coord_n0 = b_ntile_id * TILE_N;
    uint32_t coord_n2 = b_in_path_id * 1;

    // prefetch loop 0 data to L2
    if (in_wg_tid == 0)
    {
        asm_cp_async_bulk_prefetch_tensor_3d_l2(w_map, coord_n0, 0, coord_n2);
    }
    // loop
    uint32_t k_count;
    for (k_count = 0; k_count < (CEIL_DIV(U, TILE_K) - 1); ++k_count)
    {
        // wait buffer finish consuming
        bars_ready[cur_buffer_id].arrive_and_wait();
        // load w tile and prefech next tile
        if (in_wg_tid == 0)
        {
            uint32_t coord_n1 = k_count * TILE_K;
            uint32_t next_coord_n1 = (k_count + 1) * TILE_K;
            cde::cp_async_bulk_tensor_3d_global_to_shared(
                (tmp_smem + cur_buffer_id * PRELOOP_BUFFER_SIZE_PER_STAGE), // smem_ptr base
                w_map,                                                      // tensormap
                coord_n0, coord_n1, coord_n2,                               // coords
                bars_filled[cur_buffer_id]                                  // barrier
            );
            asm_cp_async_bulk_prefetch_tensor_3d_l2(w_map, coord_n0, next_coord_n1, coord_n2);
        }
        bars_filled[cur_buffer_id].arrive();
        // switch to next buffer
        cur_buffer_id = (cur_buffer_id + 1) % TOTAL_STAGES;
    }
    // loop epilogue (no prefetch)
    {
        // wait buffer finish consuming
        bars_ready[cur_buffer_id].arrive_and_wait();
        // load w tile and prefech next tile
        if (in_wg_tid == 0)
        {
            uint32_t coord_n1 = k_count * TILE_K;
            cde::cp_async_bulk_tensor_3d_global_to_shared(
                (tmp_smem + cur_buffer_id * PRELOOP_BUFFER_SIZE_PER_STAGE), // smem_ptr base
                w_map,                                                      // tensormap
                coord_n0, coord_n1, coord_n2,                               // coords
                bars_filled[cur_buffer_id]                                  // barrier
            );
        }
        bars_filled[cur_buffer_id].arrive();
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

template <uint32_t TILE_N, uint32_t TILE_K, uint32_t B_SMEM_SIZE_PER_TILE, uint32_t CONSUMER_WG_NUM,
          uint32_t TOTAL_STAGES>
__device__ void preloop_consumer(barrier bars_ready[],           // buffer finish consuming barrier
                                 barrier bars_filled[],          // buffer finish filling barrier
                                 float *B_smem,                  //
                                 float *tmp_smem,                //
                                 const uint32_t &U,              //
                                 const uint32_t &consumer_wg_id, //
                                 const uint32_t &in_wg_tid       //
)
{
    static_assert(TILE_N == 32 && TILE_K == 32);
    constexpr uint32_t PRELOOP_BUFFER_SIZE_PER_STAGE = TILE_N * TILE_K;
    // init (All buffers are empty)
#pragma unroll
    for (uint32_t i = 0; i < TOTAL_STAGES; ++i)
    {
        bars_ready[i].arrive(); // buffer_0 is ready for initial fill
    }

    uint32_t cur_buffer_id = 0;
    // loop
    for (uint32_t k_count = 0; k_count < CEIL_DIV(U, TILE_K); ++k_count)
    {
        barrier::arrival_token token;
        if (in_wg_tid == 0)
        { // 每个consumer WG的t0更新transaction count
            token = cuda::device::barrier_arrive_tx(bars_filled[cur_buffer_id], 1,
                                                    (B_SMEM_SIZE_PER_TILE * sizeof(float)) / CONSUMER_WG_NUM);
        }
        else
        {
            token = bars_filled[cur_buffer_id].arrive();
        }
        // Wait for the data to have arrived.
        bars_filled[cur_buffer_id].wait(std::move(token));
        // consume (TILE_N = 32)
        uint32_t base_sw_x0_tag = consumer_wg_id;
        wg_trans_16x32f32tf32_sw128B_sync(
            (B_smem + k_count * B_SMEM_SIZE_PER_TILE), // out buffer ptr to store output 32*32 data
            (tmp_smem + cur_buffer_id * PRELOOP_BUFFER_SIZE_PER_STAGE +
             base_sw_x0_tag * 512), // tmp buffer ptr containning input 16x32 data
            base_sw_x0_tag,         // 0 or 1
            in_wg_tid               // in warpgroup tid
        );
        // tell producer that consuming finished
        bars_ready[cur_buffer_id].arrive();
        // switch to next buffer
        cur_buffer_id = (cur_buffer_id + 1) % TOTAL_STAGES;
    }
}

template <uint32_t TILE_M, uint32_t TILE_K, uint32_t A_SMEM_SIZE_PER_STAGE, uint32_t TOTAL_STAGES>
__device__ void mainloop_producer(barrier bars_ready[],       // buffer finish consuming barrier
                                  barrier bars_filled[],      // buffer finish filling barrier
                                  float *A_smem,              //
                                  const CUtensorMap *x_map,   //
                                  const uint32_t &b_in_i_id,  //
                                  const uint32_t &b_mtile_id, //
                                  const uint32_t &B,          //
                                  const uint32_t &in_total_i, //
                                  const uint32_t &U,          //
                                  const uint32_t &in_wg_tid   //
)
{
    uint32_t cur_buffer_id = 0;
    uint32_t coord_n1 = b_in_i_id * 1;
    uint32_t coord_n2 = b_mtile_id * TILE_M;
    // prefetch
    asm_cp_async_bulk_prefetch_tensor_3d_l2(x_map, 0, coord_n1, coord_n2);
    // loop
    uint32_t k_count;
    for (k_count = 0; k_count < (CEIL_DIV(U, TILE_K) - 1); ++k_count)
    {
        // wait buffer finish consuming
        bars_ready[cur_buffer_id].arrive_and_wait();
        // load x tile
        if (in_wg_tid == 0)
        {
            uint32_t coord_n0 = k_count * TILE_K;
            uint32_t next_coord_n0 = (k_count + 1) * TILE_K;
            cde::cp_async_bulk_tensor_3d_global_to_shared(
                (A_smem + cur_buffer_id * A_SMEM_SIZE_PER_STAGE), // smem_ptr base
                x_map,                                            // tensormap
                coord_n0, coord_n1, coord_n2,                     // coords
                bars_filled[cur_buffer_id]                        // barrier
            );
            asm_cp_async_bulk_prefetch_tensor_3d_l2(x_map, next_coord_n0, coord_n1, coord_n2);
        }
        bars_filled[cur_buffer_id].arrive();
        // switch to next buffer
        cur_buffer_id = (cur_buffer_id + 1) % TOTAL_STAGES;
    }
    // loop epilogue (no prefetch)
    {
        // wait buffer finish consuming
        bars_ready[cur_buffer_id].arrive_and_wait();
        // load x tile
        if (in_wg_tid == 0)
        {
            uint32_t coord_n0 = k_count * TILE_K;
            cde::cp_async_bulk_tensor_3d_global_to_shared(
                (A_smem + cur_buffer_id * A_SMEM_SIZE_PER_STAGE), // smem_ptr base
                x_map,                                            // tensormap
                coord_n0, coord_n1, coord_n2,                     // coords
                bars_filled[cur_buffer_id]                        // barrier
            );
        }
        bars_filled[cur_buffer_id].arrive();
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

template <uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t A_SMEM_SIZE_PER_STAGE,
          uint32_t B_SMEM_SIZE_PER_TILE, uint32_t PRODUCER_WG_NUM, uint32_t CONSUMER_WG_NUM, uint32_t TOTAL_STAGES>
__device__ void mainloop_consumer(barrier bars_ready[],  // buffer finish consuming barrier
                                  barrier bars_filled[], // buffer finish filling barrier
                                  float t_accu[(TILE_M / CONSUMER_WG_NUM) / 64][(TILE_N / 8)][4], //
                                  float *A_smem,                                                  //
                                  float *B_smem,                                                  //
                                  const uint32_t &U,                                              //
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
    for (uint32_t k_count = 0; k_count < CEIL_DIV(U, TILE_K); ++k_count)
    {
        barrier::arrival_token token;
        if (in_wg_tid == 0)
        { // 每个consumer WG的t0更新transaction count
            token = cuda::device::barrier_arrive_tx(bars_filled[cur_buffer_id], 1,
                                                    (A_SMEM_SIZE_PER_STAGE * sizeof(float)) / CONSUMER_WG_NUM);
        }
        else
        {
            token = bars_filled[cur_buffer_id].arrive();
        }
        // Wait for the data to have arrived.
        bars_filled[cur_buffer_id].wait(std::move(token));

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
#pragma unroll
            for (uint32_t _l = 0; _l < 16; ++_l)
            {
                uint32_t tmp;
                asm_cvt_tf32_f32(tmp, a_reg[_m][_l]);
                a_reg[_m][_l] = __uint_as_float(tmp);
            }
        }

        // do mma
        asm_warpgroup_arrive();
#pragma unroll
        for (uint32_t _k = 0; _k < (TILE_K / WGMMA_K); ++_k)
        {
#pragma unroll
            for (uint32_t _m = 0; _m < ((TILE_M / CONSUMER_WG_NUM) / WGMMA_M); ++_m)
            {
                uint64_t b_desc = make_smem_desc(&B_smem[k_count * B_SMEM_SIZE_PER_TILE + WGMMA_K * _k]);
                asm_wgmma_m64n32k8_tf32<1, 1, 1>(t_accu[_m], a_reg[_m][4 * _k], a_reg[_m][4 * _k + 1],
                                                 a_reg[_m][4 * _k + 2], a_reg[_m][4 * _k + 3], b_desc);
            }
            asm_warpgroup_commit_batch();
            asm_warpgroup_wait(0);
        }

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
    const uint32_t &b_mtile_id,                                     // b_mtile_id
    const uint32_t &b_ntile_id,                                     // b_ntile_id
    const uint32_t &b_out_i_id,                                     // b_out_i_id
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

    // Send TMA store request
    if (in_wg_tid == 0)
    {
        uint32_t coord_n0 = b_ntile_id * TILE_N;                                               // V
        uint32_t coord_n1 = b_out_i_id;                                                        // in_total_i
        uint32_t coord_n2 = b_mtile_id * TILE_M + consumer_wg_id * (TILE_M / CONSUMER_WG_NUM); // B
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

/* ============================================================= */
/* =================== Main Kernel Functions =================== */

template <uint32_t IN_NUM_PATHS,        // in path数量
          uint32_t OUT_NUM_PATHS,       // out path数量
          uint32_t TILE_M,              // tile m大小
          uint32_t TILE_N,              // tile n大小
          uint32_t TILE_K = 32,         // tile k大小
          uint32_t PRODUCER_WG_NUM = 1, // producer warpgroup数量
          uint32_t CONSUMER_WG_NUM = 2  // consumer warpgroup数量
          >
__global__ void mutipath_equi_linear_f32_tf32_kernel(const __grid_constant__ CUtensorMap x_map,   // x tensor maps
                                                     const __grid_constant__ CUtensorMap w_map,   // w tensor maps
                                                     const __grid_constant__ CUtensorMap out_map, // out tensor maps
                                                     idim_T<IN_NUM_PATHS> in_prefex_i_sum,        // [IN_NUM_PATHS]
                                                     idim_T<OUT_NUM_PATHS> out_prefex_i_sum,      // [OUT_NUM_PATHS]
                                                     idim_T<OUT_NUM_PATHS> out_path_count,        // [OUT_NUM_PATHS]
                                                     idim_T<OUT_NUM_PATHS> out_path_in_istart,    // [OUT_NUM_PATHS]
                                                     uint32_t B,                                  // batch size
                                                     uint32_t U,                                  // U
                                                     uint32_t V,                                  // V
                                                     uint32_t in_total_i,                         // input i的总数
                                                     uint32_t out_total_i,                        // output i的总数
                                                     float cg_val                                 // val
)
{
    /* tensor shape:
     * x: [B, in_total_i, U]
     * w: [IN_NUM_PATHS, U, V]
     * out: [B, out_total_i, V]
     */
    // basic info
    constexpr uint32_t WARPGROUP_NUM = PRODUCER_WG_NUM + CONSUMER_WG_NUM;
    constexpr uint32_t PRODUCER_WG_ID = 0;
    uint32_t in_wg_tid = threadIdx.x;
    uint32_t wg_id = threadIdx.y;
    uint32_t b_ntile_id = blockIdx.x;
    uint32_t b_mtile_id = blockIdx.y;
    uint32_t b_out_i_id = blockIdx.z;

    // get path info
    uint32_t b_out_path_id = 0;
#pragma unroll
    for (uint32_t n = 0; n < OUT_NUM_PATHS; ++n)
    {
        b_out_path_id = b_out_i_id >= out_prefex_i_sum._i[n] ? n : b_out_path_id;
    }
    uint32_t b_out_reduce_loop_count = out_path_count._i[b_out_path_id];

    // define SMEM-buffer
    extern __shared__ __align__(1024) float smem[];
    constexpr uint32_t TOTAL_STAGES = 2; // 流水级
    static_assert((TILE_M * TILE_N) % CONSUMER_WG_NUM == 0);
    constexpr uint32_t OUT_SMEM_SIZE_PER_CONS_WG = (TILE_M * TILE_N) / CONSUMER_WG_NUM;
    constexpr uint32_t A_SMEM_SIZE_PER_STAGE = TILE_M * TILE_K;
    constexpr uint32_t B_SMEM_SIZE_PER_TILE = TILE_N * TILE_K;
    constexpr uint32_t OUT_SMEM_SIZE = OUT_SMEM_SIZE_PER_CONS_WG * CONSUMER_WG_NUM;
    constexpr uint32_t A_SMEM_SIZE = A_SMEM_SIZE_PER_STAGE * TOTAL_STAGES;
    float *A_smem = smem; // must align 1024
    float *O_smem = A_smem + A_SMEM_SIZE;
    float *B_smem = O_smem + OUT_SMEM_SIZE;

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
        // Initialize barrier. All `blockDim.x` threads in block participate.
        init(&bars[in_wg_tid], WARPGROUP_NUM * 128);
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
    }
    barrier *bars_ready = bars + OUTER_BARRIER_NUMS;
    barrier *bars_filled = bars + OUTER_BARRIER_NUMS + TOTAL_STAGES;
    // Syncthreads so initialized barrier is visible to all threads.
    __syncthreads();

    if (wg_id == PRODUCER_WG_ID)
    {
        // Get b_in_i_id, b_in_path_id
        uint32_t b_in_i_id0 = out_path_in_istart._i[b_out_path_id] + (b_out_i_id - out_prefex_i_sum._i[b_out_path_id]);
        uint32_t b_in_i_stride = (b_out_path_id < OUT_NUM_PATHS - 1)
                                     ? (out_prefex_i_sum._i[b_out_path_id + 1] - out_prefex_i_sum._i[b_out_path_id])
                                     : (out_total_i - out_prefex_i_sum._i[b_out_path_id]);
        uint32_t b_in_path_id0 = 0;
        for (uint32_t p = 0; p < b_out_path_id; ++p)
        {
            b_in_path_id0 += out_path_count._i[p];
        }

        // Out reduce loop
        for (uint32_t cnt = 0; cnt < b_out_reduce_loop_count; ++cnt)
        {
            uint32_t b_in_i_id = b_in_i_id0 + b_in_i_stride * cnt;
            uint32_t b_in_path_id = b_in_path_id0 + cnt;

            /* STEP1 (prepare loop): Fetch W tile and transpose (prepare loop)*/
            // We use O_smem as a tmp buffer (buffer size: TILE_M * TILE_N)
            preloop_producer<TILE_N, TILE_K, TOTAL_STAGES>(bars_ready,   //
                                                           bars_filled,  //
                                                           O_smem,       //
                                                           &w_map,       //
                                                           b_ntile_id,   //
                                                           b_in_path_id, //
                                                           U,            //
                                                           in_wg_tid     //
            );

            /* STEP2: Fetch X tile and do wgmma (main loop) */
            mainloop_producer<TILE_M, TILE_K, A_SMEM_SIZE_PER_STAGE, TOTAL_STAGES>(bars_ready,  //
                                                                                   bars_filled, //
                                                                                   A_smem,      //
                                                                                   &x_map,      //
                                                                                   b_in_i_id,   //
                                                                                   b_mtile_id,  //
                                                                                   B,           //
                                                                                   in_total_i,  //
                                                                                   U,           //
                                                                                   in_wg_tid    //
            );
        }

        /* STEP3: Store accumulator */
        // Producer do nothing
    }
    else
    {
        uint32_t consumer_wg_id = wg_id - PRODUCER_WG_NUM;
        // define accumulators
        float t_accu[(TILE_M / CONSUMER_WG_NUM) / 64][(TILE_N / 8)][4] = {0.0f};

        // Out reduce loop
        for (uint32_t cnt = 0; cnt < b_out_reduce_loop_count; ++cnt)
        {
            /* STEP1 (prepare loop): Fetch W tile and transpose (prepare loop)*/
            // We use O_smem as a tmp buffer (buffer size: TILE_M * TILE_N)
            preloop_consumer<TILE_N, TILE_K, B_SMEM_SIZE_PER_TILE, CONSUMER_WG_NUM, TOTAL_STAGES>(bars_ready,     //
                                                                                                  bars_filled,    //
                                                                                                  B_smem,         //
                                                                                                  O_smem,         //
                                                                                                  U,              //
                                                                                                  consumer_wg_id, //
                                                                                                  in_wg_tid       //
            );

            cde::fence_proxy_async_shared_cta();
            /* STEP2: Fetch X tile and do wgmma (main loop) */
            mainloop_consumer<TILE_M, TILE_N, TILE_K, A_SMEM_SIZE_PER_STAGE, B_SMEM_SIZE_PER_TILE, PRODUCER_WG_NUM,
                              CONSUMER_WG_NUM, TOTAL_STAGES>(bars_ready,     //
                                                             bars_filled,    //
                                                             t_accu,         //
                                                             A_smem,         //
                                                             B_smem,         //
                                                             U,              //
                                                             consumer_wg_id, //
                                                             in_wg_tid       //
            );
        }

        /* STEP3: Store accumulator */
        store_re_async_consumer<TILE_M, TILE_N, CONSUMER_WG_NUM>(O_smem,         // output smem buffer base addr
                                                                 t_accu,         // accumulator regs
                                                                 &out_map,       // out tensor map
                                                                 b_mtile_id,     // b_mtile_id
                                                                 b_ntile_id,     // b_ntile_id
                                                                 b_out_i_id,     // b_out_i_id
                                                                 cg_val,         // cg_val
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

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled()
{
    // Get pointer to cuTensorMapEncodeTiled
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

// reates a tensor map to describe a two-dimensional row-major array of size B x M x N
//      https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-multi-dimensional-arrays
void init_3d_tensormap(
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

template <uint32_t N> void cal_prefex_sum(idim_T<N> &dst, const std::vector<int64_t> &src)
{
    dst._i[0] = 0;
#pragma unroll
    for (uint32_t i = 1; i < N; ++i)
    {
        dst._i[i] = dst._i[i - 1] + (uint32_t)src[i - 1];
    }
}

template <uint32_t IN_NUM_PATHS, uint32_t OUT_NUM_PATHS = 4>
void mutipath_equi_linear_kernel_impl(float *out,                                 // [B, out_total_i, V]
                                      float *x,                                   // [B, in_total_i, U]
                                      float *w,                                   // [IN_NUM_PATHS, U, V]
                                      const std::vector<int64_t> &in_i_dims_vec,  // in i dims
                                      const std::vector<int64_t> &out_i_dims_vec, // out i dims
                                      const uint32_t &B,                          // batch
                                      const uint32_t &in_total_i,                 // in_total_i
                                      const uint32_t &out_total_i,                // out_total_i
                                      const uint32_t &U,                          // U
                                      const uint32_t &V,                          // V
                                      const double &val,                          // cg_val
                                      const cudaStream_t &cur_stream              // current stream
)
{
    constexpr uint32_t PRODUCER_WG_NUM = 1; // producer warpgroup数量
    constexpr uint32_t CONSUMER_WG_NUM = 2; // consumer warpgroup数量
    constexpr uint32_t WARPGROUP_NUM = PRODUCER_WG_NUM + CONSUMER_WG_NUM;
    constexpr uint32_t TILE_K = 32;
    constexpr uint32_t TILE_M = 128;
    constexpr uint32_t TILE_N = 32;

    static_assert(TILE_K == 32);
    assert(V % TILE_N == 0);

    idim_T<IN_NUM_PATHS> in_prefex_i_sum;
    cal_prefex_sum<IN_NUM_PATHS>(in_prefex_i_sum, in_i_dims_vec);

    idim_T<OUT_NUM_PATHS> out_prefex_i_sum;
    cal_prefex_sum<OUT_NUM_PATHS>(out_prefex_i_sum, out_i_dims_vec);

    idim_T<OUT_NUM_PATHS> out_path_count;
    uint32_t _cur_cnt = 0;
    uint32_t _cur_op = 0;
    uint32_t _last_i_dim = in_i_dims_vec[0];
    for (uint32_t p = 0; p < IN_NUM_PATHS; ++p)
    {
        if (in_i_dims_vec[p] == _last_i_dim)
        {
            ++_cur_cnt;
        }
        else
        {
            out_path_count._i[_cur_op] = _cur_cnt;
            _cur_cnt = 1;
            ++_cur_op;
            _last_i_dim = in_i_dims_vec[p];
        }
    }
    out_path_count._i[_cur_op] = _cur_cnt;

    idim_T<OUT_NUM_PATHS> out_path_in_istart;
    uint32_t _cur_i_path = 0;
    for (uint32_t op = 0; op < OUT_NUM_PATHS; ++op)
    {
        out_path_in_istart._i[op] = in_prefex_i_sum._i[_cur_i_path];
        _cur_i_path += out_path_count._i[op];
    }

    CUtensorMap x_map{};
    CUtensorMap w_map{};
    CUtensorMap out_map{};
    init_3d_tensormap(&x_map,                                        // tensor_map_ptr
                      static_cast<void *>(x),                        // tensor_ptr
                      B,                                             // B
                      in_total_i,                                    // M
                      U,                                             // N
                      in_total_i * U,                                // stride B
                      U,                                             // stride M
                      TILE_M,                                        // box B
                      1,                                             // box M
                      TILE_K,                                        // box N
                      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B // swizzle 128B
    );
    init_3d_tensormap(&w_map,                 // tensor_map_ptr
                      static_cast<void *>(w), // tensor_ptr
                      IN_NUM_PATHS,           // B
                      U,                      // M
                      V,                      // N
                      U * V,                  // stride B
                      V,                      // stride M
                      1,                      // box B
                      TILE_K,                 // box M
                      TILE_N                  // box N
    );
    init_3d_tensormap(&out_map,                   // tensor_map_ptr
                      static_cast<void *>(out),   // tensor_ptr
                      B,                          // B
                      out_total_i,                // M
                      V,                          // N
                      out_total_i * V,            // stride B
                      V,                          // stride M
                      (TILE_M / CONSUMER_WG_NUM), // box B
                      1,                          // box M
                      TILE_N                      // box N
    );
    dim3 grid(CEIL_DIV(V, TILE_N), CEIL_DIV(B, TILE_M), out_total_i);
    dim3 block(128, WARPGROUP_NUM);

    constexpr uint32_t STAGES = 2;
    size_t SMEM_SIZE = sizeof(float) * ((TILE_M * TILE_K) * STAGES // A_smem
                                        + (TILE_N * V)             // B_smem
                                        + (TILE_M * TILE_N)        // Out_smem
                                       );

    auto cuda_kernel = mutipath_equi_linear_f32_tf32_kernel<IN_NUM_PATHS, OUT_NUM_PATHS, TILE_M, TILE_N, TILE_K,
                                                            PRODUCER_WG_NUM, CONSUMER_WG_NUM>;
    cudaFuncSetAttribute(cuda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);
    cuda_kernel<<<grid, block, SMEM_SIZE, cur_stream>>>(x_map, w_map, out_map, in_prefex_i_sum, out_prefex_i_sum,
                                                        out_path_count, out_path_in_istart, B, U, V, in_total_i,
                                                        out_total_i, val);
}

// wrapper
void mutipath_equi_linear_f32_impl(const uint32_t &IN_NUM_PATHS,           // path nums
                                   float *out,                             // [B, out_total_i, V]
                                   float *x,                               // [B, in_total_i, U]
                                   float *w,                               // [IN_NUM_PATHS, U, V]
                                   const uint32_t &B,                      // batch
                                   const uint32_t &total_i,                // in_total_i
                                   const std::vector<int64_t> &i_dims_vec, // i dims
                                   const uint32_t &U,                      // U
                                   const uint32_t &V,                      // V
                                   const double &val,                      // cg_val
                                   const cudaStream_t &cur_stream          // current stream
)
{
    const std::vector<int64_t> out_i_dims_vec = {1, 3, 5, 7};
    const uint32_t out_total_i = 16;
    auto call_impl = [&](auto &&...forwarded_args) {
        switch (IN_NUM_PATHS)
        {
        case 4: // small
            mutipath_equi_linear_kernel_impl<4>(std::forward<decltype(forwarded_args)>(forwarded_args)...);
            break;
        case 10: // medium
            mutipath_equi_linear_kernel_impl<10>(std::forward<decltype(forwarded_args)>(forwarded_args)...);
            break;
        case 17: // large
            mutipath_equi_linear_kernel_impl<17>(std::forward<decltype(forwarded_args)>(forwarded_args)...);
            break;
        default:
            throw std::invalid_argument("Unsupported number of paths: " + std::to_string(IN_NUM_PATHS) +
                                        ". Supported values are 4, 10, and 17.");
        }
    };
    // 调用lambda，完美转发参数
    call_impl(out, x, w, i_dims_vec, out_i_dims_vec, B, total_i, out_total_i, U, V, val, cur_stream);
}
