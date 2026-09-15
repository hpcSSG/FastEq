"""Resident-tile weighted dot matching current Torch CUDA cuBLAS gemvx.

Empirically verified on Torch 2.8/H100 for Merge einsum('ai,nic->nac')
with N=17/108/3456/262144/524288 and M=4/9/15/16/25/36.
This reproduces the observed two-contiguous-chunk FMA accumulation order;
it is not a universal assertion about arbitrary cuBLAS versions/shapes.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def source_weighted_sum(V, W, START: tl.constexpr, M: tl.constexpr,
                        SINGLETON: tl.constexpr=False):
    """V[BK,BC], W[BK], both FP32; reduce selected rows to [BC].

    START:M selects only source rows. Gathers do not arithmetic-mask excluded
    rows, so NaN/Inf outside this group never enters either accumulator.
    Input data remains resident; no global load or scratch is introduced.
    """
    BC: tl.constexpr = V.shape[1]
    if SINGLETON:
        value=tl.full((BC,),0.,tl.float32)
        for j in tl.static_range(M):
            selected=tl.reshape(tl.gather(V,tl.full((1,BC),START+j,tl.int32),0),(BC,))
            weight=tl.sum(tl.gather(W,tl.full((1,),START+j,tl.int32),0),0)
            value=libdevice.add_rn(value,libdevice.mul_rn(selected,weight))
        return value
    HALF: tl.constexpr = triton.cdiv(M, 2)
    a = tl.full((BC,), 0., tl.float32)
    b = tl.full((BC,), 0., tl.float32)
    for j in tl.static_range(HALF):
        ai = tl.full((1, BC), START + j, tl.int32)
        av = tl.reshape(tl.gather(V, ai, axis=0), (BC,))
        aw = tl.sum(tl.gather(W, tl.full((1,), START + j, tl.int32), axis=0), 0)
        a = tl.fma(av, aw, a)
        if j + HALF < M:
            bi = tl.full((1, BC), START + HALF + j, tl.int32)
            bv = tl.reshape(tl.gather(V, bi, axis=0), (BC,))
            bw = tl.sum(tl.gather(W, tl.full((1,), START + HALF + j, tl.int32), axis=0), 0)
            b = tl.fma(bv, bw, b)
    return a + b
