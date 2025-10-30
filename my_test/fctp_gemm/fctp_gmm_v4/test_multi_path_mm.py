import torch
import os, time

torch.manual_seed(42)

BATCH, U, V, W = 5888, 96, 10, 96
UV = U * V

torch.ops.load_library("build/bin/libmulti_path_mm.so")
fctp_multi_path_mm = torch.ops.fctp_multi_path_mm


def test_multi_path_tp():
    dim_list = [1, 3, 5, 7]
    path_num = len(dim_list)

    B = BATCH

    b_all = torch.randn(BATCH, 1, V, dtype=torch.float64, device="cuda")
    # b_all = torch.arange(
    #             start=0.000,
    #             end=0.001 * B * V,
    #             step=0.001,
    #             dtype=torch.float64,
    #             device="cuda"
    #         ).reshape(BATCH, 1, V)
    
    w_all = torch.randn(path_num, U, V, W, dtype=torch.float64, device="cuda")
    # w_all = torch.arange(
    #             start=0.000,
    #             end=0.001 * path_num * U * V * W,
    #             step=0.001,
    #             dtype=torch.float64,
    #             device="cuda"
    #         ).reshape(path_num, U, V, W)

    retry = 50
    # test ref
    for retry_id in range(0, retry):
        total_einsum_time = 0
        ref_pbuw = None
        # ref
        for dim_idx in range(0, path_num):
            b = b_all
            w = w_all[dim_idx, :, :, :].reshape(U, V, W)

            torch.cuda.synchronize()
            start_total = time.perf_counter() *1000

            tmp_bjuw = torch.einsum("bjv,uvw->bjuw", b, w)
            
            torch.cuda.synchronize()
            end_total = time.perf_counter() *1000
            t = (end_total - start_total)
            # print(f"--- einsum path {dim_idx} tensor product cost: {t:.4f} ms")
            total_einsum_time += t
            if dim_idx == 0:
                ref_pbuw = tmp_bjuw.view(-1)
            else:
                ref_pbuw = torch.cat((ref_pbuw, tmp_bjuw.view(-1)), dim=0)

        print(f"ref einsum time: {total_einsum_time:.4f} ms")
        
    ref_pbuw = ref_pbuw.view(path_num, BATCH, U, -1)
    assert ref_pbuw.shape[3] == W, "ref_pbuw的最后一个维度大小不为W"
    
    # test my
    b = b_all.view(BATCH, -1)
    w = w_all
    for retry_id in range(0, retry):
        # my
        torch.cuda.synchronize()
        start_total = time.perf_counter() *1000

        my_pbuw = fctp_multi_path_mm.forward(b, w)
        
        torch.cuda.synchronize()
        end_total = time.perf_counter() *1000
        total_einsum_time = (end_total - start_total)
        print(f"my einsum time: {total_einsum_time:.4f} ms")
        
    # compare re
    # torch.set_printoptions(threshold=float('inf'), linewidth=999999, precision=3, sci_mode=False)
    # for _p in range(2):
    #     for _b in range(BATCH):
    #         if not torch.allclose(my_pbuw[_p][_b], ref_pbuw[_p][_b]):
                
    #             for _u in range(U):
    #                 if not torch.allclose(my_pbuw[_p][_b][_u], ref_pbuw[_p][_b][_u]):
    #                     print(f"p={_p} b={_b} u={_u} 结果不一致！")
    #                     print(f"my: {my_pbuw[_p][_b][_u]}")
    #                     print(F"ref: {ref_pbuw[_p][_b][_u]}")
    torch.testing.assert_close(my_pbuw, ref_pbuw, msg=f"My 与 Ref 结果不一致！\nmy: {my_pbuw[0]}\nref:{ref_pbuw[0]}")
    print("结果正确！")


if __name__ == "__main__":
    test_multi_path_tp()
    