"""Resident-tile FP32 mean/rstd helpers matching PyTorch 2.8 CUDA.

MeanOps uses the same four streams/trees as Sum, then project(sum)=sum*factor.
Its host factor is float(num_outputs)/float(numel), not a padded-tile divisor.
Caller must keep enable_fp_fusion=False to retain separate source operations.
"""
import struct
import triton
import triton.language as tl
from ._torch_norm_order import m_config, configured_m_sum
from ._torch_norm_weighted import source_weighted_sum


def f32(value):
    return struct.unpack('f',struct.pack('f',value))[0]


def mean_factor(num_outputs, reduction_length):
    """FP32 host calculation in aten/native/cuda/ReduceMomentKernel.cu."""
    return f32(f32(num_outputs)/f32(num_outputs*reduction_length))


def reduction_code(cfg):
    return (int(cfg['fastest']) | (int(cfg['vectorize'])<<1)
            | ((cfg['block_width'].bit_length()-1)<<2)
            | ((cfg['reduction_height'].bit_length()-1)<<6))


@triton.jit
def source_mean(V, ROW, START:tl.constexpr, LENGTH:tl.constexpr,
                CFG:tl.constexpr, FACTOR:tl.constexpr, BASE_SHIFT:tl.constexpr=0):
    """Mean along axis 0, using the source reduction schedule and factor.

    ROW determines vectorized-input alignment: ROW*LENGTH modulo four must
    match the native source row's element offset. For a raw NKC scalar slice
    [N,1,C], pass atom*K; for a newly allocated [N,C] intermediate pass atom.
    The source reduction's output count still determines CFG separately.
    """
    return configured_m_sum(V,ROW,START,LENGTH,CFG,BASE_SHIFT)*FACTOR


@triton.jit
def source_rstd(moment, EPS:tl.constexpr):
    # CUDA pow(x,-0.5) dispatches to rsqrt_kernel_cuda -> ::rsqrt(float).
    return tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=f,f",
        [moment+EPS], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def source_row_mean(V, ROW, CHANNELS:tl.constexpr, CFG:tl.constexpr,
                    FACTOR:tl.constexpr, BASE_SHIFT:tl.constexpr=0):
    """Mean of one padded channel vector V[BC]."""
    return tl.sum(source_mean(V[:,None],ROW,0,CHANNELS,CFG,FACTOR,BASE_SHIFT),0)


@triton.jit
def source_per_degree_stats(Z,atom,START:tl.constexpr,M:tl.constexpr,C:tl.constexpr,
                            MCFG:tl.constexpr,CCFG:tl.constexpr,
                            M_FACTOR:tl.constexpr,C_FACTOR:tl.constexpr,
                            COMPONENT:tl.constexpr,EPS:tl.constexpr):
    """V3 ELN order: square -> mean/sum m -> mean C -> add eps -> rsqrt.

    Z is already scalar-centered where requested. Squaring remains FP32.
    """
    square=Z*Z
    if COMPONENT:
        per_channel=source_mean(square,atom,START,M,MCFG,M_FACTOR)
    else:
        per_channel=configured_m_sum(square,atom,START,M,MCFG)
    moment=source_row_mean(per_channel,atom,C,CCFG,C_FACTOR)
    return per_channel,moment,source_rstd(moment,EPS)


@triton.jit
def source_channel_means(V,atom,NATIVE_ROWS:tl.constexpr,K:tl.constexpr,
                         C:tl.constexpr,CFG:tl.constexpr,FACTOR:tl.constexpr,
                         START:tl.constexpr=0,NODES:tl.constexpr=1,KNC:tl.constexpr=False):
    """Per-k mean over C for a resident NKC square tile; preserve row identity.

    CFG is computed with num_outputs=N*K (m_config(N*K,C,1)), not N.
    This helper does not implement the following weighted einsum/bmm.
    """
    BK:tl.constexpr=V.shape[0]
    BC:tl.constexpr=V.shape[1]
    k=tl.arange(0,BK)
    result=tl.full((BK,),0.,tl.float32)
    if C <= 128 or (C <= 1024 and C % 4 == 0):
        # Independent channels-first reductions share one resident tile.
        # Non-vectorized means ignore ROW; divisible-by-four vector rows all
        # have zero prefix. Wider/misaligned rows retain the scalar helper.
        result=source_mean(tl.trans(V),atom*NATIVE_ROWS,0,C,CFG,FACTOR)
    else:
        for row in tl.static_range(K):
            one=tl.sum(tl.where((k==row+START)[:,None],V,0.),0)
            source_row=row*NODES+atom if KNC else atom*NATIVE_ROWS+row
            value=source_row_mean(one,source_row,C,CFG,FACTOR)
            result=tl.where(k==row+START,value,result)
    return result


@triton.jit
def source_group_stats(Z,W,atom,START:tl.constexpr,M:tl.constexpr,C:tl.constexpr,
                       KIND:tl.constexpr,COMPONENT:tl.constexpr,BALANCED:tl.constexpr,
                       MCFG:tl.constexpr,CCFG:tl.constexpr,KCFG:tl.constexpr,
                       ROW_CCFG:tl.constexpr,MF:tl.constexpr,CF:tl.constexpr,
                       KF:tl.constexpr,ROW_CF:tl.constexpr,EPS:tl.constexpr,
                       SINGLETON:tl.constexpr,NODES:tl.constexpr,KNC:tl.constexpr):
    if KIND == 1:
        _,moment,rstd=source_per_degree_stats(Z,atom,START,M,C,MCFG,CCFG,
                                              MF,CF,COMPONENT,EPS)
    elif KIND == 2:
        square=Z*Z
        if COMPONENT and BALANCED:
            channel=source_weighted_sum(square,W,START,M,SINGLETON)
        elif COMPONENT:
            channel=source_mean(square,atom,START,M,MCFG,MF)
        else:
            channel=configured_m_sum(square,atom,START,M,MCFG)
        moment=source_row_mean(channel,atom,C,CCFG,CF)
        rstd=source_rstd(moment,EPS)
    else:
        rows=source_channel_means(Z*Z,atom,M,M,C,ROW_CCFG,ROW_CF,START,NODES,KNC)
        if COMPONENT and BALANCED:
            moment=tl.sum(source_weighted_sum(rows[:,None],W,START,M,SINGLETON),0)
        elif COMPONENT:
            moment=tl.sum(source_mean(rows[:,None],atom,START,M,KCFG,KF),0)
        else:
            moment=tl.sum(configured_m_sum(rows[:,None],atom,START,M,KCFG),0)
        rstd=source_rstd(moment,EPS)
    return moment,rstd
