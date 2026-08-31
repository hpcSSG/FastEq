import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[3])
def stp_edge_parallel_kernel_tl(
    B: int,
    S: int,
    Iw: int,
    Ix: int,
    Ky: int,
    V: int,
    U: int,
    P: int,
    WARPS_PER_BLOCK: int = 4,
    dtype: str = "float32",
):
    threads = WARPS_PER_BLOCK * 32

    @T.prim_func
    def main(
        w: T.Tensor((B, Iw, U), dtype),          # 0
        x_all: T.Tensor((S, Ix, U), dtype),      # 1
        y: T.Tensor((B, Ky), dtype),             # 2
        out: T.Tensor((S, V, U), dtype),         # 3 <- return value
        src_idx: T.Tensor((B,), "int32"),        # 4
        dst_idx: T.Tensor((B,), "int32"),        # 5
        b_list: T.Tensor((B,), "int32"),         # 6
        i_list: T.Tensor((P,), "int32"),         # 8
        j_list: T.Tensor((P,), "int32"),         # 9
        k_list: T.Tensor((P,), "int32"),         # 10
        v_list: T.Tensor((P,), "int32"),         # 11
        coeff_list: T.Tensor((P,), dtype),       # 12
    ):
        with T.Kernel(T.ceildiv(B, WARPS_PER_BLOCK), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            lane = tid % 32
            warp = tid // 32
            warp_global = bx * WARPS_PER_BLOCK + warp

            if warp_global < B:
                b = b_list[warp_global]
                src = src_idx[b]
                dst = dst_idx[b]

                if lane < U:
                    for t in T.serial(P):
                        i = i_list[t]
                        j = j_list[t]
                        k = k_list[t]
                        v = v_list[t]
                        c = coeff_list[t]

                        wval = w[b, i, lane]
                        xval = x_all[src, j, lane]
                        yval = y[b, k]
                        acc = c * wval * xval * yval

                        T.atomic_add(out[dst, v, lane], acc)

    return main