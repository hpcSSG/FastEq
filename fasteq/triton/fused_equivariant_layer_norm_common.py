"""Shared configuration and native FP32 reductions for equivariant LayerNorm.

Statistics use complete SO(3) degrees. Reduction helpers preserve the source
operation order while reusing values already held in the normalization tile.
"""
from dataclasses import dataclass
import math
import struct
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@dataclass(frozen=True)
class EquivariantNormSpec:
    lmax: int
    channels: int
    stats_weights: tuple[tuple[float, ...], ...]
    output_group: tuple[int, ...]
    center_scalar: bool = True
    eps: float = 1e-5

    def __post_init__(self):
        if type(self.lmax) is not int or self.lmax < 0:
            raise ValueError('lmax must be a nonnegative integer')
        if type(self.channels) is not int or self.channels < 1:
            raise ValueError('channels must be a positive integer')
        if type(self.center_scalar) is not bool:
            raise ValueError('center_scalar must be boolean')
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('eps must be finite and positive')
        weights = tuple(tuple(float(v) for v in row) for row in self.stats_weights)
        groups = tuple(self.output_group)
        if not weights or any(len(row) != self.lmax + 1 for row in weights):
            raise ValueError('stats_weights must have shape [G, lmax+1], G > 0')
        if any(not math.isfinite(v) or v < 0 for row in weights for v in row):
            raise ValueError('statistic weights must be finite and nonnegative')
        if any(not any(v > 0 for v in row) for row in weights):
            raise ValueError('each statistic must have a nonzero source')
        if len(groups) != self.lmax + 1 or any(
            type(g) is not int or not 0 <= g < len(weights) for g in groups
        ):
            raise ValueError('output_group must select one valid group per degree')
        if set(groups) != set(range(len(weights))):
            raise ValueError('every statistic must be used by an output degree')
        object.__setattr__(self, 'stats_weights', weights)
        object.__setattr__(self, 'output_group', groups)

    @property
    def components(self):
        return (self.lmax + 1) ** 2

    @property
    def num_stats(self):
        return len(self.stats_weights)

    @classmethod
    def from_preset(cls, *, lmax, channels, grouping='all',
                    weighting='degree_balanced', center_scalar=True, eps=1e-5):
        if type(lmax) is not int or lmax < 0:
            raise ValueError('lmax must be a nonnegative integer')
        if grouping == 'per_degree':
            sources = [(l,) for l in range(lmax + 1)]
        elif grouping == 'scalar_high':
            sources = [(0,)] + ([tuple(range(1, lmax + 1))] if lmax else [])
        elif grouping == 'all':
            sources = [tuple(range(lmax + 1))]
        else:
            raise ValueError('unknown grouping preset')
        if weighting not in ('component', 'degree_balanced', 'norm'):
            raise ValueError('unknown weighting preset')
        weights, output_group = [], [0] * (lmax + 1)
        for g, degrees in enumerate(sources):
            row = [0.] * (lmax + 1)
            for l in degrees:
                if weighting == 'degree_balanced':
                    row[l] = 1. / (len(degrees) * (2 * l + 1))
                elif weighting == 'component':
                    row[l] = 1. / sum(2 * j + 1 for j in degrees)
                else:
                    row[l] = 1.
                output_group[l] = g
            weights.append(tuple(row))
        return cls(lmax, channels, tuple(weights), tuple(output_group), center_scalar, eps)


class NormResult(NamedTuple):
    output: torch.Tensor
    mean: torch.Tensor | None   # [N]; absent if scalar centering is disabled
    moments: torch.Tensor | None  # [N,G]; omitted unless requested
    rstd: torch.Tensor          # [N,G]


def m_config(n, m, c, *, warp_size=32, base_shift=0):
    assert n > 0 and m > 0 and c > 0
    fastest = c == 1
    vectorize = fastest and m > 128
    vec = 1
    if not fastest:
        vec = 4
        while c % vec or base_shift % vec:
            vec //= 2
    dim0 = (m // 4 if vectorize else m) if fastest else n*c//vec
    dim1 = n if fastest else m
    mt = 512//vec
    p0 = min(1 << (dim0.bit_length()-1), mt)
    p1 = min(1 << (dim1.bit_length()-1), mt)
    bw = min(p0, warp_size)
    bh = min(p1, mt//bw)
    bw = min(p0, mt//bh)
    initial_step = bw if fastest else 1
    split_y = triton.cdiv(m, initial_step) >= min(bh*16, 256)
    h = bh if split_y else 1
    if triton.cdiv(m,initial_step*h) >= 256 and split_y:
        raise ValueError('m axis may require CTA split outside supported LayerNorm tile bound')
    return dict(fastest=fastest, vectorize=vectorize, block_width=bw,
                reduction_height=h, native_block_height=bh, output_vec=vec,
                warp_size=warp_size, base_shift=base_shift)


@triton.jit
def _take_m(V, I, START:tl.constexpr, M:tl.constexpr):
    BK:tl.constexpr = V.shape[0]
    BC:tl.constexpr = V.shape[1]
    index=tl.broadcast_to(((START+I)%BK)[:,None],(I.shape[0],BC))
    data=tl.gather(V,index,0)
    return tl.where((I[:,None]>=0)&(I[:,None]<M),data,0.)


@triton.jit
def _vec_chunk(V, chunk, R:tl.constexpr):
    BK:tl.constexpr=V.shape[0]
    NC:tl.constexpr=BK//(4*R)
    data=tl.reshape(V,(NC,R,2,2))
    selected=tl.sum(tl.where((tl.arange(0,NC)==chunk)[:,None,None,None],data,0.),0)
    even,odd=tl.split(selected)
    v0,v2=tl.split(even)
    v1,v3=tl.split(odd)
    return v0,v1,v2,v3


@triton.jit
def _large_vector_accumulate(V, ATOM, M:tl.constexpr, R:tl.constexpr,
                             BASE_SHIFT:tl.constexpr):
    """Four native streams without gathering dynamically from a huge tile."""
    BK:tl.constexpr=V.shape[0]
    row=tl.arange(0,R)
    allk=tl.arange(0,BK)
    flat=tl.reshape(V,(BK,))
    shift=(BASE_SHIFT+ATOM*M)%4
    prefix=tl.where(shift>0,4-shift,0)
    end=M-prefix
    a0=tl.full((R,),0.,tl.float32)
    a1=tl.full((R,),0.,tl.float32)
    a2=tl.full((R,),0.,tl.float32)
    a3=tl.full((R,),0.,tl.float32)
    # Head contributions reside only in the first four X lanes of Y=0.
    if shift>0:
        for j in tl.static_range(0,4):
            value=tl.sum(tl.where(allk==j-shift,flat,0.),0)
            a0=tl.where((row==j)&(j>=shift),value,a0)
    for chunk in range(0,triton.cdiv(M,4*R)):
        v0,v1,v2,v3=_vec_chunk(V,chunk,R)
        if prefix>0:
            n0,n1,n2,n3=_vec_chunk(V,chunk+1,R)
            f0=tl.sum(tl.where(row==0,n0,0.),0)
            f1=tl.sum(tl.where(row==0,n1,0.),0)
            f2=tl.sum(tl.where(row==0,n2,0.),0)
            s0=tl.where(row==R-1,f0,tl.gather(v0,(row+1)%R,0))
            s1=tl.where(row==R-1,f1,tl.gather(v1,(row+1)%R,0))
            s2=tl.where(row==R-1,f2,tl.gather(v2,(row+1)%R,0))
            w0=tl.where(prefix==1,v1,tl.where(prefix==2,v2,v3))
            w1=tl.where(prefix==1,v2,tl.where(prefix==2,v3,s0))
            w2=tl.where(prefix==1,v3,tl.where(prefix==2,s0,s1))
            w3=tl.where(prefix==1,s0,tl.where(prefix==2,s1,s2))
            v0=w0;v1=w1;v2=w2;v3=w3
        valid=(chunk*R+row)*4+3<end
        a0+=tl.where(valid,v0,0.)
        a1+=tl.where(valid,v1,0.)
        a2+=tl.where(valid,v2,0.)
        a3+=tl.where(valid,v3,0.)
    tail=prefix+end-end%4
    for j in tl.static_range(0,4):
        value=tl.sum(tl.where(allk==tail+j,flat,0.),0)
        a0+=tl.where((row==j)&(tail+j<M),value,0.)
    return a0[:,None],a1[:,None],a2[:,None],a3[:,None]


@triton.jit
def m_sum(V, ATOM, START:tl.constexpr, M:tl.constexpr,
          FASTEST:tl.constexpr, VEC:tl.constexpr, BW:tl.constexpr,
          H:tl.constexpr, WARP:tl.constexpr, BASE_SHIFT:tl.constexpr):
    """Return [BC] FP32 sums for one degree of resident V[BK,BC].

    START is the degree's l*l offset, M=2*l+1. V must already contain the
    source-equivalent FP32 product. ATOM selects native source row alignment.
    """
    BC:tl.constexpr=V.shape[1]
    BK:tl.constexpr=V.shape[0]
    # Extract the degree once with constant indices. Subsequent dynamic vector
    # head/tail gathers operate on at most next_power_of_2(M), not the whole K
    # tile, keeping compilation and intermediate shuffle sizes bounded.
    DM:tl.constexpr=triton.next_power_of_2(M)
    if BK>1024:
        # Select two fixed aligned chunks before rotating by START % DM. This
        # avoids a huge full-K gather lowering at the largest supported tiles.
        # Each reduction selects just one value and zeroes every other chunk;
        # it does not combine distinct products or alter the m sum order.
        CHUNKS:tl.constexpr=BK//DM
        chunks=tl.reshape(V,(CHUNKS,DM,BC))
        ci=tl.arange(0,CHUNKS)
        v0=tl.sum(tl.where((ci==START//DM)[:,None,None],chunks,0.),0)
        mi=tl.arange(0,DM)
        if START%DM==0:
            V=v0
        else:
            v1=tl.sum(tl.where((ci==START//DM+1)[:,None,None],chunks,0.),0)
            ix=tl.broadcast_to(((mi+START%DM)%DM)[:,None],(DM,BC))
            v0=tl.gather(v0,ix,0)
            v1=tl.gather(v1,ix,0)
            V=tl.where((mi+START%DM<DM)[:,None],v0,v1)
        V=tl.where((mi<M)[:,None],V,0.)
    else:
        V=_take_m(V,tl.arange(0,DM),START,M)
    R:tl.constexpr=BW*H if FASTEST else H
    row=tl.arange(0,R)
    a0=tl.full((R,BC),0.,tl.float32)
    a1=tl.full((R,BC),0.,tl.float32)
    a2=tl.full((R,BC),0.,tl.float32)
    a3=tl.full((R,BC),0.,tl.float32)
    if VEC and DM>1024:
        a0,a1,a2,a3=_large_vector_accumulate(V,ATOM,M,R,BASE_SHIFT)
    elif VEC:
        shift=(BASE_SHIFT+ATOM*M)%4
        prefix=tl.where(shift>0,4-shift,0)
        end=M-prefix
        head=_take_m(V,row-shift,0,M)
        a0=tl.where(((shift>0)&(row>=shift)&(row<4))[:,None],head,0.)
        for base in range(0,triton.cdiv(M,4),R):
            i=base+row
            valid=(i*4+3)<end
            a0+=tl.where(valid[:,None],_take_m(V,prefix+i*4+0,0,M),0.)
            a1+=tl.where(valid[:,None],_take_m(V,prefix+i*4+1,0,M),0.)
            a2+=tl.where(valid[:,None],_take_m(V,prefix+i*4+2,0,M),0.)
            a3+=tl.where(valid[:,None],_take_m(V,prefix+i*4+3,0,M),0.)
        tail=end-end%4+row
        a0+=tl.where((tail<end)[:,None],_take_m(V,prefix+tail,0,M),0.)
    else:
        for base in range(0,M,4*R):
            i=base+row
            a0+=_take_m(V,i+0*R,0,M)
            a1+=_take_m(V,i+1*R,0,M)
            a2+=_take_m(V,i+2*R,0,M)
            a3+=_take_m(V,i+3*R,0,M)
    a=((a0+a1)+a2)+a3
    if FASTEST:
        # Native CUDA first reduces Y independently at each x, then X.
        # Flattened thread coordinates are row = x + y * BW.
        for step in tl.static_range(H.bit_length()-2,-1,-1):
            idx=tl.broadcast_to(((row+BW*(1<<step))%R)[:,None],(R,BC))
            a+=tl.gather(a,idx,0)
        if BW>WARP:
            for step in tl.static_range(BW.bit_length()-2,WARP.bit_length()-2,-1):
                idx=tl.broadcast_to((row//BW*BW+(row+(1<<step))%BW)[:,None],(R,BC))
                a+=tl.gather(a,idx,0)
        DIMX:tl.constexpr=BW if BW<WARP else WARP
        for step in tl.static_range(0,DIMX.bit_length()-1):
            idx=tl.broadcast_to((row//BW*BW+(row+(1<<step))%BW)[:,None],(R,BC))
            a+=tl.gather(a,idx,0)
    else:
        for step in tl.static_range(R.bit_length()-2,-1,-1):
            idx=tl.broadcast_to(((row+(1<<step))%R)[:,None],(R,BC))
            a+=tl.gather(a,idx,0)
    return tl.sum(tl.where(row[:,None]==0,a,0.),0)


@triton.jit
def configured_m_sum(V, ATOM, START: tl.constexpr, M: tl.constexpr, CFG: tl.constexpr,
                     BASE_SHIFT: tl.constexpr = 0):
    FAST: tl.constexpr = CFG & 1
    VEC: tl.constexpr = (CFG >> 1) & 1
    BW: tl.constexpr = 1 << ((CFG >> 2) & 15)
    H: tl.constexpr = 1 << ((CFG >> 6) & 15)
    return m_sum(V, ATOM, START, M, FAST, VEC, BW, H, 32, BASE_SHIFT)


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
