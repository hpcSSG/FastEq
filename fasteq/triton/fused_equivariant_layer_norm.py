"""Fused FP32 equivariant LayerNorm with first-order gradients.

One program per atom shares the feature tile across statistic groups,
normalization, and affine output. Parameter gradients use deterministic
partial-buffer reductions.
"""
from dataclasses import dataclass
import math
import struct
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


# Configuration and source-compatible statistics.

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


# Forward kernels.

@triton.jit
def _moment(square, w, coefficient, C: tl.constexpr, ORDER: tl.constexpr,
            UNIFORM: tl.constexpr):
    if ORDER == 'channels_first':
        q = tl.sum(square, 1) / C
        if UNIFORM:
            # Source unbalanced component/norm path reduces before scaling.
            return tl.sum(q, 0) * coefficient
        return tl.sum(q * w, 0)
    else:
        if UNIFORM:
            q = tl.sum(square, 0) * coefficient
        else:
            q = tl.sum(square * w[:, None], 0)
        return tl.sum(q, 0) / C


@triton.jit
def _affine(z, r, k, c, valid, DEGREE, WEIGHT0, WEIGHTH, BIAS,
            HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr,
            WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr,
            BS: tl.constexpr, K: tl.constexpr):
    if HAS_WEIGHT:
        degree = tl.load(DEGREE + k, k < K, 0)
        if SPLIT:
            scalar_weight = tl.load(WEIGHT0 + c * WS0 + tl.zeros_like(k), valid & (k == 0), 0.)
            higher_weight = tl.load(WEIGHTH + (degree - 1) * WSL + c * WSC,
                                    valid & (k > 0), 0.)
            gamma = tl.where(k == 0, scalar_weight, higher_weight)
        else:
            gamma = tl.load(WEIGHT0 + degree * WSL + c * WSC, valid, 0.)
        y = z * (r * gamma)
    else:
        y = z * r
    if HAS_BIAS:
        beta = tl.load(BIAS + c * BS + tl.zeros_like(k), valid & (k == 0), 0.)
        y = tl.where(k == 0, y + beta, y)
    return y


@triton.jit
def _fused(X, Y, MU, MOMENTS, RSTD, COEFFICIENT, DEGREE,
           WEIGHT0, WEIGHTH, BIAS, K: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
           SN: tl.constexpr, SK: tl.constexpr, CENTER: tl.constexpr,
           EPS: tl.constexpr, ORDER: tl.constexpr, UNIFORM: tl.constexpr,
           HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr,
           WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr, BS: tl.constexpr,
           SAVE: tl.constexpr, SAVE_MOMENTS: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
           GROUP_BOUNDS: tl.constexpr, SOURCE: tl.constexpr, CCFG: tl.constexpr,
           CF: tl.constexpr, MCONFIG: tl.constexpr, MF: tl.constexpr, COMPONENT: tl.constexpr,
           BALANCED: tl.constexpr, KCONFIG: tl.constexpr, ROW_CCONFIG: tl.constexpr,
           KF: tl.constexpr, ROW_CF: tl.constexpr, SINGLETON: tl.constexpr,
           NODES: tl.constexpr, KNC: tl.constexpr, INPUT_SHIFT: tl.constexpr):
    n = tl.program_id(0)
    k = tl.arange(0, BK)
    c = tl.arange(0, BC)
    valid = (k[:, None] < K) & (c[None, :] < C)
    x = tl.load(X + n * SN + k[:, None] * SK + c[None, :], valid, 0.)
    if CENTER:
        scalar = tl.sum(tl.where(k[:, None] == 0, x, 0.), 0)
        if SOURCE == 1 or SOURCE == 3:
            mu = source_row_mean(scalar, n*(SN//C), C, CCFG, CF, INPUT_SHIFT)
        else:
            mu = tl.sum(scalar, 0) / C
        z = tl.where(valid, x - tl.where(k[:, None] == 0, mu, 0.), 0.)
    else:
        z = x
    w = tl.load(COEFFICIENT + k, k < K, 0.)
    square = z * z
    row_rstd = tl.full((BK,), 0., tl.float32)
    for g in tl.static_range(G):
        start = GROUP_BOUNDS[g]
        end = GROUP_BOUNDS[g + 1]
        member = (k >= start) & (k < end)
        # Select absent sources before reduction: 0 * NaN is not isolation.
        group_square = tl.where(member[:, None], square, 0.)
        group_w = tl.where(member, w, 0.)
        coefficient = tl.sum(tl.where(k == start, w, 0.), 0)
        if SOURCE != 0 and (SOURCE == 1 or SOURCE == 3 or g > 0):
            v, r = source_group_stats(z, w, n, GROUP_BOUNDS[g], GROUP_BOUNDS[g+1]-GROUP_BOUNDS[g], C,
                SOURCE, COMPONENT, BALANCED, ((MCONFIG >> (12*g)) & 4095), CCFG,
                ((KCONFIG >> (12*g)) & 4095), ((ROW_CCONFIG >> (12*g)) & 4095),
                MF[g], CF, KF[g], ROW_CF[g], EPS, SINGLETON, NODES, KNC)
        else:
            v = _moment(group_square, group_w, coefficient, C, ORDER, UNIFORM)
            r = tl.rsqrt(v + EPS)
        row_rstd = tl.where(member, r, row_rstd)
        if SAVE:
            if SAVE_MOMENTS:
                tl.store(MOMENTS + n * G + g, v)
            tl.store(RSTD + n * G + g, r)
    y = _affine(z, row_rstd[:, None], k[:, None], c[None, :], valid, DEGREE, WEIGHT0,
                WEIGHTH, BIAS, HAS_WEIGHT, SPLIT, HAS_BIAS, WS0, WSL, WSC, BS, K)
    tl.store(Y + n * K * C + k[:, None] * C + c[None, :], y, valid)
    if SAVE:
        if CENTER:
            tl.store(MU + n, mu)


# Backward kernels and gradient reductions.

# First-order FP32 gradients for disjoint equivariant normalization groups.
#
# For a component in group g, a is its statistic coefficient, z is the
# scalar-centered input, r = rsqrt(moment + eps), and h = dY * gamma:
#
#     zhat = z * r
#     Q[g] = sum_group(h * zhat)          # NOT weighted by a
#     u = r * (h - (a / C) * zhat * Q[g])
#     dX = u, except dX_scalar = u_scalar - mean_c(u_scalar)
#
# The normalized form avoids explicitly forming r**3. Parameter gradients are
# sum_(n,m)(dY * zhat) for gamma and sum_n(dY_scalar) for beta. Reductions use
# deterministic partial buffers rather than floating-point atomic additions.

# Parameter-gradient reductions retain the original CUDA summation order.

def _config(n, width, device, logical_stride, logical_ptr, logical_c_stride):
    assert n > 0 and width > 1
    vec = 4
    while (width % vec or logical_stride % vec or (logical_ptr // 4) % vec
           or (vec > 1 and logical_c_stride != 1)):
        vec //= 2
    max_threads = 512 // vec
    dim0 = width // vec
    pow0 = min(1 << (dim0.bit_length() - 1), max_threads)
    pow1 = min(1 << (n.bit_length() - 1), max_threads)
    bw = min(pow0, 32)
    bh = min(pow1, max_threads // bw)
    bw = min(pow0, max_threads // bh)
    split_y = n >= min(bh * 16, 256)
    grid_x = triton.cdiv(width // vec, bw if split_y else bw * bh)
    props = torch.cuda.get_device_properties(device)
    target = props.multi_processor_count * (props.max_threads_per_multi_processor // (bw * bh))
    height = bh if split_y else 1
    values = triton.cdiv(n, height)
    ctas = 1
    if split_y and values >= 256 and grid_x <= target:
        ctas = max(min(triton.cdiv(target, grid_x), triton.cdiv(values, 16)),
                   triton.cdiv(values, 256))
    return height, bw * vec, ctas


@triton.jit
def _tree_y(a, H: tl.constexpr, BC: tl.constexpr):
    row = tl.arange(0, H)
    for step in tl.static_range(H.bit_length() - 2, -1, -1):
        other = tl.gather(a, tl.broadcast_to(((row + (1 << step)) % H)[:, None], (H, BC)), 0)
        a = a + other
    return tl.sum(tl.where(row[:, None] == 0, a, 0.), 0)


@triton.jit
def _sum_partials(X, P, N: tl.constexpr, W: tl.constexpr, SN: tl.constexpr,
                  SG: tl.constexpr, H: tl.constexpr, BC: tl.constexpr, T: tl.constexpr):
    group = tl.program_id(2)
    col = tl.program_id(0) * BC + tl.arange(0, BC)
    row = tl.arange(0, H) + tl.program_id(1) * H
    stride: tl.constexpr = H * T
    a0 = tl.full((H, BC), 0., tl.float32)
    a1 = tl.full((H, BC), 0., tl.float32)
    a2 = tl.full((H, BC), 0., tl.float32)
    a3 = tl.full((H, BC), 0., tl.float32)
    for base in range(0, N, 4 * stride):
        n0 = base + row
        n1 = n0 + stride
        n2 = n1 + stride
        n3 = n2 + stride
        a0 += tl.load(X + group * SG + n0[:, None] * SN + col[None, :],
                      (n0[:, None] < N) & (col[None, :] < W), 0.)
        a1 += tl.load(X + group * SG + n1[:, None] * SN + col[None, :],
                      (n1[:, None] < N) & (col[None, :] < W), 0.)
        a2 += tl.load(X + group * SG + n2[:, None] * SN + col[None, :],
                      (n2[:, None] < N) & (col[None, :] < W), 0.)
        a3 += tl.load(X + group * SG + n3[:, None] * SN + col[None, :],
                      (n3[:, None] < N) & (col[None, :] < W), 0.)
    total = _tree_y(((a0 + a1) + a2) + a3, H, BC)
    tl.store(P + (group * T + tl.program_id(1)) * W + col, total, col < W)


@triton.jit
def _sum_finish(P, Y, W: tl.constexpr, H: tl.constexpr, BC: tl.constexpr, T: tl.constexpr):
    group = tl.program_id(1)
    col = tl.program_id(0) * BC + tl.arange(0, BC)
    row = tl.arange(0, H)
    a = tl.full((H, BC), 0., tl.float32)
    for base in range(0, T, H):
        idx = base + row
        a += tl.load(P + (group * T + idx[:, None]) * W + col[None, :],
                     (idx[:, None] < T) & (col[None, :] < W), 0.)
    tl.store(Y + group * W + col, _tree_y(a, H, BC), col < W)


def _sum_n_leaf(partials, width, groups=1, group_stride=0, *, logical_stride=None,
                logical_ptr=0, logical_c_stride=1):
    n = partials.shape[0]
    logical_stride = width if logical_stride is None else logical_stride
    if width > 1 and logical_stride < logical_c_stride:
        # CUDA compares input strides, so broadcast/column-major dY can
        # reduce along block.x even though PB is a compact row-major buffer.
        cfg = _fast_n_config(n, width, logical_stride, partials.device)
        ctas, lanes = cfg['ctas'], cfg['lanes']
        work = torch.empty((groups * width, ctas), device=partials.device, dtype=partials.dtype)
        _scalar_partials[(ctas, groups * width)](
            partials, work, n, partials.stride(0), lanes, cfg['vectorize'],
            (logical_ptr // 4) % 4, 32, ctas, group_stride,
            OUTPUT_WIDTH=width, LOGICAL_C_STRIDE=logical_c_stride,
            num_warps=4, enable_fp_fusion=False)
        if ctas == 1:
            return work.view(groups, width)
        result = torch.empty((groups, width), device=partials.device, dtype=partials.dtype)
        _scalar_finish[(groups * width,)](work, result, ctas, lanes, 32,
                                        num_warps=4, enable_fp_fusion=False)
        return result
    if width == 1:
        cfg = _scalar_config(n, logical_stride=logical_stride, logical_ptr=logical_ptr, device=partials.device)
        ctas, bw = cfg['ctas'], cfg['block_width']
        work = torch.empty((groups, ctas), device=partials.device, dtype=partials.dtype)
        launch = dict(num_warps=4, enable_fp_fusion=False)
        _scalar_partials[(ctas, groups)](partials, work, n, partials.stride(0), bw,
            cfg['vectorize'], cfg['shift'], cfg['warp_size'], ctas, group_stride, **launch)
        if ctas == 1:
            return work
        result = torch.empty((groups, 1), device=partials.device, dtype=partials.dtype)
        _scalar_finish[(groups,)](work, result, ctas, bw, cfg['warp_size'], **launch)
        return result
    height, block_c, ctas = _config(n, width, partials.device, logical_stride, logical_ptr, logical_c_stride)
    work = torch.empty((groups, ctas, width), device=partials.device, dtype=partials.dtype)
    launch = dict(num_warps=4, enable_fp_fusion=False)
    _sum_partials[(triton.cdiv(width, block_c), ctas, groups)](
        partials, work, n, width, partials.stride(0), group_stride,
        height, block_c, ctas, **launch)
    if ctas == 1:
        return work.view(groups, width)
    result = torch.empty((groups, width), device=partials.device, dtype=partials.dtype)
    _sum_finish[(triton.cdiv(width, block_c), groups)](
        work, result, width, height, block_c, ctas, **launch)
    return result


def _native_slices(n, width, logical_stride, logical_c_stride=1):
    """Visit FP32 TensorIterator leaves in its native front-to-back order.

    The source has one reduction over N and one contiguous output axis. Its
    input byte extent belongs to the *logical source*, not the interleaved PW
    allocation. Different degree groups are separate source reductions.
    TensorIterator.cpp checks both numel and every operand's byte extent,
    then halves the largest-extent axis, with the output axis winning ties.
    """
    limit = 2**31 - 1
    pending = [(0, n, 0, width)]
    leaves = []
    while pending:
        start_n, count_n, start_c, count_c = pending.pop()
        extent_n = (count_n - 1) * logical_stride * 4
        extent_c = (count_c - 1) * logical_c_stride * 4
        output_extent = (count_c - 1) * 4
        if (count_n * count_c <= limit
                and 1 + extent_n + extent_c <= limit
                and 1 + output_extent <= limit):
            leaves.append((start_n, count_n, start_c, count_c))
            continue
        # Ordinary packed/split gamma and NKC/KNC/strided bias split N.
        # Also handle a large column stride without recursively splitting N
        # forever when only the non-reduced axis exceeds the byte limit.
        split_c = count_c > 1 and (count_n == 1 or max(extent_c, output_extent) >= extent_n)
        if split_c:
            half = count_c // 2
            pending.append((start_n, count_n, start_c + half, count_c - half))
            pending.append((start_n, count_n, start_c, half))
        else:
            half = count_n // 2
            pending.append((start_n + half, count_n - half, start_c, count_c))
            pending.append((start_n, half, start_c, count_c))
    return leaves


@triton.jit
def _merge_n_slice(SRC, DST, W: tl.constexpr, FULL_W: tl.constexpr,
                   OFFSET: tl.constexpr, ACCUMULATE: tl.constexpr, BC: tl.constexpr):
    group = tl.program_id(1)
    col = tl.program_id(0) * BC + tl.arange(0, BC)
    value = tl.load(SRC + group * W + col, col < W, 0.)
    target = DST + group * FULL_W + OFFSET + col
    if ACCUMULATE:
        value = tl.load(target, col < W, 0.) + value
    tl.store(target, value, col < W)


def _sum_n(partials, width, groups=1, group_stride=0, *, logical_stride=None,
           logical_ptr=0, logical_c_stride=1):
    n = partials.shape[0]
    logical_stride = width if logical_stride is None else logical_stride
    leaves = _native_slices(n, width, logical_stride, logical_c_stride)
    if len(leaves) == 1:
        return _sum_n_leaf(partials, width, groups, group_stride,
                          logical_stride=logical_stride, logical_ptr=logical_ptr,
                          logical_c_stride=logical_c_stride)
    result = torch.empty((groups, width), device=partials.device, dtype=partials.dtype)
    for start_n, count_n, start_c, count_c in leaves:
        # as_strided only constructs a view. All arithmetic, including the
        # ordered FP32 accumulation between leaves, stays inside Triton.
        view = partials.as_strided((count_n, groups, count_c),
            (partials.stride(0), group_stride, 1),
            storage_offset=partials.storage_offset() + start_n * partials.stride(0) + start_c)
        leaf = _sum_n_leaf(view, count_c, groups, group_stride,
            logical_stride=logical_stride,
            logical_ptr=logical_ptr + 4 * (start_n * logical_stride + start_c * logical_c_stride),
            logical_c_stride=logical_c_stride)
        _merge_n_slice[(triton.cdiv(count_c, 128), groups)](
            leaf, result, count_c, width, start_c, start_n != 0, 128,
            num_warps=4, enable_fp_fusion=False)
    return result


@triton.jit
def _write_parameters(V, S, B, DW0, DWH, DB, C: tl.constexpr,
                       EXPANDED: tl.constexpr, SPLIT: tl.constexpr,
                       W0_GRAD: tl.constexpr, WH_GRAD: tl.constexpr,
                       BIAS_GRAD: tl.constexpr, BC: tl.constexpr):
    l = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    weight_grad = (W0_GRAD & ((l == 0) | (not SPLIT))) | (WH_GRAD & (l > 0))
    if weight_grad:
        if EXPANDED:
            if SPLIT and l == 0:
                value = tl.load(S + c, c < C, 0.)
            else:
                value = tl.full((BC,), 0., tl.float32)
                for k in range(l * l, (l + 1) * (l + 1)):
                    value += tl.load(V + (k - (1 if SPLIT else 0)) * C + c, c < C, 0.)
        else:
            value = tl.load(V + l * C + c, c < C, 0.)
        if SPLIT:
            if l == 0:
                tl.store(DW0 + c, value, c < C)
            else:
                tl.store(DWH + (l - 1) * C + c, value, c < C)
        else:
            tl.store(DW0 + l * C + c, value, c < C)
    if BIAS_GRAD:
        if l == 0:
            tl.store(DB + c, tl.load(B + c, c < C, 0.), c < C)


def reduce_parameters(op, pw, pb, dw0, dwh, db, split, needs, placeholder, dy):
    _, nw0, nwh, nb = needs
    c, lmax = op.spec.channels, op.spec.lmax
    expanded = op.parameter_order == 'expanded'
    value = scalar = bias = placeholder
    if nw0 or nwh:
        if expanded:
            if split:
                if nw0:
                    scalar = _sum_n(pw[:, 0], c)
                if nwh:
                    value = _sum_n(pw[:, 1:], (op.spec.components - 1) * c)
            else:
                value = _sum_n(pw, op.spec.components * c)
        else:
            value = _sum_n(pw, c, lmax + 1, c)
    if nb:
        bias = _sum_n(pb, c, logical_stride=dy.stride(0), logical_ptr=dy.data_ptr(),
                      logical_c_stride=dy.stride(2))
    _write_parameters[(lmax + 1, triton.cdiv(c, 32))](
        value, scalar, bias, dw0 if nw0 else placeholder, dwh if nwh else placeholder,
        db if nb else placeholder, c, expanded, split, nw0, nwh, nb, 32,
        num_warps=4, enable_fp_fusion=False)


def _fast_n_config(n, width, logical_stride, device):
    vectorize = logical_stride == 1 and n > 128
    dim0 = n // 4 if vectorize else n
    p0 = min(1 << (dim0.bit_length() - 1), 512)
    p1 = min(1 << (width.bit_length() - 1), 512)
    bw = min(p0, 32)
    bh = min(p1, 512 // bw)
    bw = min(p0, 512 // bh)
    split_y = triton.cdiv(n, bw) >= min(bh * 16, 256)
    lanes = bw * bh if split_y else bw
    values = triton.cdiv(n, lanes)
    grid = triton.cdiv(width, 1 if split_y else bh)
    ctas = 1
    if split_y and values >= 256:
        props = torch.cuda.get_device_properties(device)
        target = props.multi_processor_count * (props.max_threads_per_multi_processor // (bw * bh))
        if grid <= target:
            ctas = max(min(triton.cdiv(target, grid), triton.cdiv(values, 16)),
                       triton.cdiv(values, 256))
    return dict(vectorize=vectorize, lanes=lanes, ctas=ctas)


def _scalar_config(n, logical_stride=1, logical_ptr=0, warp_size=32, device=None):
    assert n > 0
    vectorize = logical_stride == 1 and n > 128
    dim0 = n // 4 if vectorize else n
    bw = min(1 << (dim0.bit_length() - 1), 512)
    # One output means height 1, so grid.x=1. The input-Y split flag becomes
    # true once values_per_thread>=16 despite the factor being height=1.
    values=triton.cdiv(n,bw)
    ctas=1
    if values>=256:
        props=torch.cuda.get_device_properties(device)
        target=props.multi_processor_count*(props.max_threads_per_multi_processor//bw)
        ctas=max(min(target,triton.cdiv(values,16)),triton.cdiv(values,256))
    return dict(block_width=bw, vectorize=vectorize, shift=(logical_ptr//4)%4,
                warp_size=warp_size,ctas=ctas)


@triton.jit
def _scalar_partials(X,Y,N:tl.constexpr,SN:tl.constexpr,BW:tl.constexpr,
                VEC:tl.constexpr,SHIFT:tl.constexpr,WARP:tl.constexpr,T:tl.constexpr,SG:tl.constexpr,
                OUTPUT_WIDTH:tl.constexpr=1,LOGICAL_C_STRIDE:tl.constexpr=0):
    output = tl.program_id(1)
    column = output % OUTPUT_WIDTH
    X = X + (output // OUTPUT_WIDTH) * SG + column
    Y = Y + tl.program_id(1) * T
    lane=tl.arange(0,BW)
    idx=lane+tl.program_id(0)*BW
    a0=tl.full((BW,),0.,tl.float32)
    a1=tl.full((BW,),0.,tl.float32)
    a2=tl.full((BW,),0.,tl.float32)
    a3=tl.full((BW,),0.,tl.float32)
    if VEC:
        shift=(SHIFT+column*LOGICAL_C_STRIDE)%4
        PREFIX=tl.where(shift>0,4-shift,0)
        END=N-PREFIX
        if shift>0:
            a0=tl.load(X+(lane-shift)*SN,(lane>=shift)&(lane<4)&(tl.program_id(0)==0),0.)
        for base in range(0,END//4,BW*T):
            i=base+idx
            mask=i*4+3<END
            a0+=tl.load(X+(PREFIX+i*4+0)*SN,mask,0.)
            a1+=tl.load(X+(PREFIX+i*4+1)*SN,mask,0.)
            a2+=tl.load(X+(PREFIX+i*4+2)*SN,mask,0.)
            a3+=tl.load(X+(PREFIX+i*4+3)*SN,mask,0.)
        t=END-END%4+lane
        a0+=tl.load(X+(PREFIX+t)*SN,(t<END)&(tl.program_id(0)==0),0.)
    else:
        for base in range(0,N,4*BW*T):
            i=base+idx
            a0+=tl.load(X+(i+0*BW*T)*SN,i+0*BW*T<N,0.)
            a1+=tl.load(X+(i+1*BW*T)*SN,i+1*BW*T<N,0.)
            a2+=tl.load(X+(i+2*BW*T)*SN,i+2*BW*T<N,0.)
            a3+=tl.load(X+(i+3*BW*T)*SN,i+3*BW*T<N,0.)
    a=((a0+a1)+a2)+a3
    if BW>WARP:
        for step in tl.static_range(BW.bit_length()-2,WARP.bit_length()-2,-1):
            a+=tl.gather(a,(lane+(1<<step))%BW,0)
    DIMX:tl.constexpr = BW if BW<WARP else WARP
    for step in tl.static_range(0,DIMX.bit_length()-1):
        a+=tl.gather(a,(lane+(1<<step))%BW,0)
    result=tl.sum(tl.where(lane==0,a,0.),0)
    tl.store(Y+tl.program_id(0),result)


@triton.jit
def _scalar_finish(P,Y,T:tl.constexpr,BW:tl.constexpr,WARP:tl.constexpr):
    P = P + tl.program_id(0) * T
    Y = Y + tl.program_id(0)
    lane=tl.arange(0,BW)
    a=tl.full((BW,),0.,tl.float32)
    for base in range(0,T,BW):
        idx=base+lane
        a+=tl.load(P+idx,idx<T,0.)
    if BW>WARP:
        for step in tl.static_range(BW.bit_length()-2,WARP.bit_length()-2,-1):
            a+=tl.gather(a,(lane+(1<<step))%BW,0)
    DIMX:tl.constexpr = BW if BW<WARP else WARP
    for step in tl.static_range(0,DIMX.bit_length()-1):
        a+=tl.gather(a,(lane+(1<<step))%BW,0)
    tl.store(Y,tl.sum(tl.where(lane==0,a,0.),0))


# Gradient of each reciprocal standard deviation in source operation order.

@triton.jit
def _flat_sum(value, atom, START: tl.constexpr, M: tl.constexpr,
              C: tl.constexpr, CFG: tl.constexpr):
    BK: tl.constexpr = value.shape[0]
    BC: tl.constexpr = value.shape[1]
    i = tl.arange(0, BK * BC)
    offset = (START + i // C) * BC + i % C
    selected = tl.gather(value.reshape((BK * BC,)), offset % (BK * BC), 0)
    selected = tl.where(i < M, selected, 0.)
    return tl.sum(configured_m_sum(selected[:, None], atom, 0, M, CFG), 0)


@triton.jit
def source_r_gradient(z, dy, gamma, atom, START: tl.constexpr,
                      END: tl.constexpr, C: tl.constexpr,
                      L: tl.constexpr, ORDER: tl.constexpr,
                      M_CONFIG: tl.constexpr, C_CONFIG: tl.constexpr,
                      Q_CONFIG: tl.constexpr, HAS_WEIGHT: tl.constexpr,
                      DEGREE_CONFIG: tl.constexpr):
    BK: tl.constexpr = z.shape[0]
    if ORDER == 'broadcast':
        result = tl.full((), 0., tl.float32)
        k = tl.arange(0, BK)[:, None]
        for l in tl.static_range(L, -1, -1):
            if (l * l >= START) & (l * l < END):
                if HAS_WEIGHT:
                    ds = configured_m_sum(dy * z, atom, l * l, 2 * l + 1, ((M_CONFIG >> (12 * l)) & 4095))
                    weight = tl.sum(tl.where(k == l * l, gamma, 0.), 0)
                    dr = configured_m_sum((ds * weight)[:, None], atom, 0, C, C_CONFIG)
                    result += tl.sum(dr, 0)
                else:
                    result += _flat_sum(dy * z, atom, l * l,
                        (2 * l + 1) * C, C, ((DEGREE_CONFIG >> (12 * l)) & 4095))
    else:
        result = _flat_sum((dy * z) * gamma, atom, START,
                           (END - START) * C, C, Q_CONFIG)
    return result


@triton.jit
def _gamma(k, c, valid, DEGREE, W0, WH, K: tl.constexpr,
           HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr,
           WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr):
    if HAS_WEIGHT:
        degree = tl.load(DEGREE + k, k < K, 0)
        if SPLIT:
            scalar = tl.load(W0 + c * WS0 + tl.zeros_like(k), valid & (k == 0), 0.)
            higher = tl.load(WH + (degree - 1) * WSL + c * WSC,
                             valid & (k > 0), 0.)
            return tl.where(k == 0, scalar, higher)
        return tl.load(W0 + degree * WSL + c * WSC, valid, 0.)
    return tl.full(valid.shape, 1., tl.float32)


@triton.jit
def _sum_group(value, ORDER: tl.constexpr):
    if ORDER == 'channels_first':
        return tl.sum(tl.sum(value, 1), 0)
    return tl.sum(tl.sum(value, 0), 0)


@triton.jit
def _input_gradient(h, zhat, r, q, w, valid, k, C: tl.constexpr,
                    CENTER: tl.constexpr):
    u = tl.where(valid, r * (h - (w / C) * zhat * q), 0.)
    if CENTER:
        scalar = tl.sum(tl.where(k == 0, u, 0.), 0)
        mean_u = tl.sum(scalar, 0) / C
        u = tl.where(valid, u - tl.where(k == 0, mean_u, 0.), 0.)
    return u


@triton.jit
def _fused_backward(X, DY, MU, RSTD, DX, PW, PB, COEFFICIENT, DEGREE, W0, WH,
                    K: tl.constexpr, C: tl.constexpr, G: tl.constexpr, L: tl.constexpr,
                    SN: tl.constexpr, SK: tl.constexpr,
                    DN: tl.constexpr, DK: tl.constexpr, DC: tl.constexpr,
                    CENTER: tl.constexpr, HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr,
                    WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr,
                    INPUT_GRAD: tl.constexpr, WEIGHT_GRAD: tl.constexpr,
                    BIAS_GRAD: tl.constexpr, ORDER: tl.constexpr,
                    GROUP_BOUNDS: tl.constexpr, PARAM_ORDER: tl.constexpr, M_CONFIG: tl.constexpr,
                    C_CONFIG: tl.constexpr, Q_CONFIG: tl.constexpr, DEGREE_CONFIG: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr):
    n = tl.program_id(0)
    k = tl.arange(0, BK)[:, None]
    c = tl.arange(0, BC)[None, :]
    valid = (k < K) & (c < C)
    dy = tl.load(DY + n * DN + k * DK + c * DC, valid, 0.)
    if INPUT_GRAD or WEIGHT_GRAD:
        z = tl.load(X + n * SN + k * SK + c, valid, 0.)
        if CENTER:
            mu = tl.load(MU + n)
            z = tl.where(valid, z - tl.where(k == 0, mu, 0.), 0.)
        row_rstd = tl.full((BK, 1), 0., tl.float32)
        for g in tl.static_range(G):
            member = (k >= GROUP_BOUNDS[g]) & (k < GROUP_BOUNDS[g + 1])
            r = tl.load(RSTD + n * G + g)
            row_rstd = tl.where(member, r, row_rstd)
        zhat = tl.where(valid, z * row_rstd, 0.)
    if INPUT_GRAD:
        gamma = _gamma(k, c, valid, DEGREE, W0, WH, K, HAS_WEIGHT, SPLIT, WS0, WSL, WSC)
        h = dy * gamma
        product = h * zhat
        row_q = tl.full((BK, 1), 0., tl.float32)
        for g in tl.static_range(G):
            member = (k >= GROUP_BOUNDS[g]) & (k < GROUP_BOUNDS[g + 1])
            if PARAM_ORDER == 'accurate':
                q = _sum_group(tl.where(member & valid, product, 0.), ORDER)
            else:
                r = tl.load(RSTD + n * G + g)
                q = r * source_r_gradient(z, dy, gamma, n,
                    GROUP_BOUNDS[g], GROUP_BOUNDS[g + 1], C, L, PARAM_ORDER,
                    M_CONFIG, C_CONFIG, ((Q_CONFIG >> (12 * g)) & 4095), HAS_WEIGHT, DEGREE_CONFIG)
            row_q = tl.where(member, q, row_q)
        w = tl.load(COEFFICIENT + k, k < K, 0.)
        dx = _input_gradient(h, zhat, row_rstd, row_q, w, valid, k, C, CENTER)
        tl.store(DX + n * K * C + k * C + c, dx, valid)
    if WEIGHT_GRAD:
        if PARAM_ORDER == 'expanded':
            tl.store(PW + n * K * C + k * C + c, (dy * z) * row_rstd, valid)
        else:
            product = dy * z if PARAM_ORDER == 'broadcast' else dy * zhat
            # Source broadcast: reduce m before multiplying the scale.
            for l in tl.static_range(L + 1):
                member = (k >= l * l) & (k < (l + 1) * (l + 1))
                if PARAM_ORDER == 'broadcast':
                    partial = configured_m_sum(tl.where(valid, product, 0.), n, l * l, 2 * l + 1, ((M_CONFIG >> (12 * l)) & 4095))
                    partial *= tl.sum(tl.where(k == l * l, row_rstd, 0.), 0)
                else:
                    partial = tl.sum(tl.where(member & valid, product, 0.), 0)
                tl.store(PW + (n * (L + 1) + l) * C + tl.arange(0, BC), partial,
                         tl.arange(0, BC) < C)
    if BIAS_GRAD:
        partial = tl.sum(tl.where(k == 0, dy, 0.), 0)
        tl.store(PB + n * C + tl.arange(0, BC), partial, tl.arange(0, BC) < C)


@triton.jit
def _parameter_reduce(PW, PB, DW0, DWH, DB, N: tl.constexpr, L: tl.constexpr,
                      C: tl.constexpr, SPLIT: tl.constexpr,
                      W0_GRAD: tl.constexpr, WH_GRAD: tl.constexpr,
                      BIAS_GRAD: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr):
    l = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    rows = tl.arange(0, BN)
    weight_grad = (W0_GRAD & ((l == 0) | (not SPLIT))) | (WH_GRAD & (l > 0))
    # Each lane accumulates ceil(N / BN) atom contributions. FP32 loses small
    # terms when large positive and negative partials cancel at large N.
    # Widen only the accumulator; partial buffers and returned grads stay FP32.
    if weight_grad:
        acc = tl.full((BN, BC), 0., tl.float64)
        for start in range(0, N, BN):
            n = start + rows
            value = tl.load(PW + (n[:, None] * (L + 1) + l) * C + c[None, :],
                            (n[:, None] < N) & (c[None, :] < C), 0.)
            acc += value
        total = tl.sum(acc, 0)
        if SPLIT:
            if l == 0:
                tl.store(DW0 + c, total, c < C)
            else:
                tl.store(DWH + (l - 1) * C + c, total, c < C)
        else:
            tl.store(DW0 + l * C + c, total, c < C)
    if BIAS_GRAD:
        if l == 0:
            acc = tl.full((BN, BC), 0., tl.float64)
            for start in range(0, N, BN):
                n = start + rows
                value = tl.load(PB + n[:, None] * C + c[None, :],
                                (n[:, None] < N) & (c[None, :] < C), 0.)
                acc += value
            tl.store(DB + c, tl.sum(acc, 0), c < C)


def _backward(op, x, mean, rstd, weight0, weight_high, dy, needs):
    """Allocate gradients and dispatch kernels; no PyTorch gradient arithmetic."""
    if dy.layout != torch.strided or dy.shape != x.shape:
        raise ValueError('expected a strided output gradient with the output shape')
    if dy.dtype != x.dtype or dy.device != x.device:
        raise ValueError('output gradient must match the input device and dtype')
    if any(s < 0 for s in dy.stride()) or sum(
            max(d - 1, 0) * s for d, s in zip(dy.shape, dy.stride())) > 2**31 - 1:
        raise NotImplementedError('output gradient requires nonnegative 32-bit strides')
    s = op.spec
    n, k, c = x.shape
    if n and op.parameter_order != 'accurate' and torch.version.hip is not None:
        raise NotImplementedError('source-ordered backward currently implements the PyTorch CUDA '
                                  'reduction schedule; ROCm requires its own validated schedule')
    split = weight_high is not None
    nx, nw0, nwh, nb = needs
    # The empty higher-degree parameter of an lmax=0 source is unused.
    nwh = nwh and s.lmax > 0
    nw = nw0 or nwh

    def empty(shape):
        return torch.empty(shape, device=x.device, dtype=x.dtype)

    dx = empty(x.shape) if nx else None
    dw0 = empty(weight0.shape) if nw0 else None
    dwh = empty(weight_high.shape) if nwh else None
    db = empty((c,)) if nb else None
    pw = empty((n, k if op.parameter_order == 'expanded' else s.lmax + 1, c)) if nw else None
    pb = empty((n, c)) if nb else None
    # These placeholders are eliminated by constexpr flags before dereferencing.
    def ptr(value):
        return x if value is None else value
    common = dict(K=k, C=c, G=s.num_stats, SN=x.stride(0), SK=x.stride(1),
                  DN=dy.stride(0), DK=dy.stride(1), DC=dy.stride(2),
                  CENTER=s.center_scalar, BC=op.block_c)
    affine = dict(HAS_WEIGHT=weight0 is not None, SPLIT=split,
                  WS0=weight0.stride(0) if split else 0,
                  WSL=weight_high.stride(0) if split else weight0.stride(0) if weight0 is not None else 0,
                  WSC=weight_high.stride(1) if split else weight0.stride(1) if weight0 is not None else 0)
    m_configs = c_config = q_configs = degree_configs = 0
    if n and op.parameter_order == 'broadcast':
        m_configs = sum(reduction_code(m_config(n, 2*l + 1, c)) << (12*l) for l in range(s.lmax + 1))
    if n and op.parameter_order != 'accurate' and nx:
        c_config = reduction_code(m_config(n, c, 1))
        if op.parameter_order == 'expanded':
            q_configs = sum(reduction_code(m_config(n, (b-a)*c, 1)) << (12*g)
                            for g, (a, b) in enumerate(zip(op.group_bounds, op.group_bounds[1:])))
        elif weight0 is None:
            degree_configs = sum(reduction_code(m_config(n, (2*l+1)*c, 1)) << (12*l)
                                 for l in range(s.lmax + 1))
    launch = dict(num_warps=4, enable_fp_fusion=False)
    with torch.cuda.device(x.device):
        if n and (nx or nw or nb):
            _fused_backward[(n,)](x, dy, ptr(mean), rstd, ptr(dx), ptr(pw), ptr(pb),
                op.coefficients, op.degrees, ptr(weight0), ptr(weight_high),
                **common, **affine, L=s.lmax, INPUT_GRAD=nx, WEIGHT_GRAD=nw,
                BIAS_GRAD=nb, ORDER=op.reduction_order, GROUP_BOUNDS=op.group_bounds,
                PARAM_ORDER=op.parameter_order, M_CONFIG=m_configs,
                C_CONFIG=c_config, Q_CONFIG=q_configs, DEGREE_CONFIG=degree_configs, BK=op.atom_block_k, **launch)
        if n and (nw or nb) and op.parameter_order != 'accurate':
            reduce_parameters(op, pw, pb, dw0, dwh, db, split, (nx, nw0, nwh, nb), x, dy)
        elif nw or nb:
            _parameter_reduce[(s.lmax + 1, triton.cdiv(c, 32))](ptr(pw), ptr(pb),
                ptr(dw0), ptr(dwh), ptr(db), n, s.lmax, c, split, nw0, nwh, nb,
                BN=32, BC=32, **launch)
    return dx, dw0, dwh, db


# Autograd and source adapters.

class _EquivariantNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight0, weight_high, bias, op, save_stats):
        weight = (weight0, weight_high) if weight_high is not None else weight0
        result = op._run(x, weight, bias, save_stats, training=True)
        ctx.op = op
        ctx.save_for_backward(x, result.mean, result.rstd, weight0, weight_high)
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(*(value for value in result[1:] if value is not None))
        return tuple(result)

    @staticmethod
    def backward(ctx, dy, _dmean, _dmoments, _drstd):
        if torch.is_grad_enabled():
            raise RuntimeError('equivariant norm supports first-order gradients only; '
                               'create_graph=True and double backward are not supported')
        if dy is None:
            return (None,) * 6
        x, mean, rstd, weight0, weight_high = ctx.saved_tensors
        gradients = _backward(ctx.op, x, mean, rstd, weight0, weight_high,
                              dy, ctx.needs_input_grad[:4])
        return (*gradients, None, None)


class TritonEquivariantNorm(torch.nn.Module):
    """Execution plan; affine parameters belong to the caller.

    GPU scope: FP32 forward/backward, contiguous NKC / transposed KNC inputs,
    positive-stride packed or split weights. Each statistic must describe one
    contiguous, disjoint range of full degrees and scale exactly that range.
    Broader/overlapping specs remain valid math specs but are rejected here.
    """
    def __init__(self, spec, *, reduction_order='channels_first',
                 device='cuda', parameter_order='accurate'):
        super().__init__()
        if not isinstance(spec, EquivariantNormSpec):
            raise TypeError('spec must be EquivariantNormSpec')
        if reduction_order not in ('components_first', 'channels_first'):
            raise ValueError('unknown reduction order')
        bounds, coefficients, degrees = [0], [], []
        uniform = True
        for g, row in enumerate(spec.stats_weights):
            source = [l for l, a in enumerate(row) if a > 0]
            target = [l for l, h in enumerate(spec.output_group) if h == g]
            if (source != target or source != list(range(source[0], source[-1] + 1))
                    or source[0] ** 2 != bounds[-1]):
                raise NotImplementedError('GPU groups must be ordered contiguous disjoint '
                                          'degree ranges, with identical source/output support')
            bounds.append((source[-1] + 1) ** 2)
            uniform &= len({row[l] for l in source}) == 1
            for l in source:
                coefficients.extend([row[l]] * (2*l + 1))
                degrees.extend([l] * (2*l + 1))
        if bounds[-1] != spec.components:
            raise NotImplementedError('groups must cover all degrees')
        self.spec = spec
        if parameter_order not in ('accurate', 'broadcast', 'expanded'):
            raise ValueError('unknown parameter reduction order')
        self.parameter_order = parameter_order
        self.source_kind = 0
        self.source_component = True
        self.source_balanced = False
        self.reduction_order = reduction_order
        self.uniform = uniform
        self.group_bounds = tuple(bounds)
        self.atom_block_k = triton.next_power_of_2(spec.components)
        self.block_c = triton.next_power_of_2(spec.channels)
        if self.atom_block_k * self.block_c > 65536:
            raise NotImplementedError('execution tile exceeds 65536 padded elements')
        for name, values, dtype in [('coefficients', coefficients, torch.float32),
                                    ('degrees', degrees, torch.int32)]:
            self.register_buffer(name, torch.tensor(values, dtype=dtype, device=device),
                                 persistent=False)

    def _check(self, x, weight, bias):
        s = self.spec
        if x.ndim != 3 or tuple(x.shape[1:]) != (s.components, s.channels):
            raise ValueError('expected [N,(lmax+1)^2,channels]')
        if x.device.type != 'cuda' or x.dtype != torch.float32:
            raise TypeError('Triton backend supports CUDA FP32 only')
        if x.device != self.coefficients.device or self.coefficients.dtype != torch.float32:
            raise ValueError('move the execution plan to the input device; keep it FP32')
        if not (x.is_contiguous() or x.stride() == (s.channels, x.shape[0]*s.channels, 1)):
            raise ValueError('supported input storage: NKC and KNC')
        if x.numel() > 2**31 - 1:
            raise NotImplementedError('this backend uses 32-bit tensor indexing')
        def check(t, shape):
            if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape:
                raise ValueError(f'expected parameter shape {shape}')
            if t.device != x.device or t.dtype != x.dtype:
                raise ValueError('parameter device and dtype must match input')
            if any(a <= 0 for a in t.stride()):
                raise ValueError('parameter strides must be positive')
            if sum(max(a - 1, 0) * b for a, b in zip(t.shape, t.stride())) > 2**31 - 1:
                raise NotImplementedError('parameter offsets exceed 32-bit indexing')

        split = isinstance(weight, tuple)
        if weight is not None:
            if split:
                if len(weight) != 2:
                    raise ValueError('split weight must be (scalar, higher)')
                check(weight[0], (s.channels,))
                check(weight[1], (s.lmax, s.channels))
            else:
                check(weight, (s.lmax+1, s.channels))
        if bias is not None:
            check(bias, (s.channels,))

    def _run(self, x, weight, bias, save, training=False):
        n, k, c = x.shape
        s = self.spec
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        need_stats = save or training
        mu = torch.empty(n, dtype=x.dtype, device=x.device) if need_stats and s.center_scalar else None
        moments = torch.empty((n, s.num_stats), dtype=x.dtype, device=x.device) if save else None
        rstd = torch.empty((n, s.num_stats), dtype=x.dtype, device=x.device) if need_stats else None
        if n:
            split = isinstance(weight, tuple)
            w0 = weight[0] if split else weight
            wh = weight[1] if split else None
            ws0 = w0.stride(0) if split else 0
            wsl = wh.stride(0) if split else w0.stride(0) if w0 is not None else 0
            wsc = wh.stride(1) if split else w0.stride(1) if w0 is not None else 0
            bs = bias.stride(0) if bias is not None else 0
            affine = dict(HAS_WEIGHT=weight is not None, SPLIT=split, HAS_BIAS=bias is not None,
                          WS0=ws0, WSL=wsl, WSC=wsc, BS=bs)
            common = dict(C=c, G=s.num_stats, SN=x.stride(0), SK=x.stride(1),
                          CENTER=s.center_scalar)
            native_kind = self.source_kind if torch.version.hip is None else 0
            input_shift = (x.data_ptr() // 4) % 4 if native_kind in (1, 3) else 0
            # Unused source schedules must not restrict the generic spec path
            # or a source which reduces its axes in a different order.
            # Merge's centering concatenates into NKC before square/mean;
            # without centering its square preserves the original layout.
            native = dict(SOURCE=native_kind, NODES=n,
                KNC=not x.is_contiguous() and (native_kind != 3 or not s.center_scalar),
                SINGLETON=(n*c == 1 if native_kind == 2 else n == 1),
                COMPONENT=self.source_component, BALANCED=self.source_balanced,
                CCFG=0, CF=1., MCONFIG=0, KCONFIG=0, ROW_CCONFIG=0,
                MF=(1.,)*s.num_stats, KF=(1.,)*s.num_stats, ROW_CF=(1.,)*s.num_stats)
            if native_kind:
                sizes = tuple(b-a for a,b in zip(self.group_bounds,self.group_bounds[1:]))
                native.update(CCFG=reduction_code(m_config(n,c,1)), CF=mean_factor(n,c))
                if native_kind in (1, 2):
                    native.update(
                        MCONFIG=sum(reduction_code(m_config(n,m,c)) << (12*g)
                                    for g,m in enumerate(sizes)),
                        MF=tuple(mean_factor(n*c,m) for m in sizes))
                else:
                    native.update(
                        KCONFIG=sum(reduction_code(m_config(n,m,1)) << (12*g)
                                    for g,m in enumerate(sizes)),
                        ROW_CCONFIG=sum(reduction_code(m_config(n*m,c,1)) << (12*g)
                                       for g,m in enumerate(sizes)),
                        KF=tuple(mean_factor(n,m) for m in sizes),
                        ROW_CF=tuple(mean_factor(n*m,c) for m in sizes))
            # Placeholders are never dereferenced when the corresponding flag is false.
            w0, wh, bp = w0 if w0 is not None else y, wh if wh is not None else y, bias if bias is not None else y
            with torch.cuda.device(x.device):
                _fused[(n,)](x, y, mu if mu is not None else y,
                    moments if moments is not None else y, rstd if rstd is not None else y,
                    self.coefficients, self.degrees, w0, wh, bp, K=k,
                    **common, EPS=s.eps, ORDER=self.reduction_order, UNIFORM=self.uniform,
                    BK=self.atom_block_k, BC=self.block_c,
                    GROUP_BOUNDS=self.group_bounds, **native, **affine, SAVE=need_stats, SAVE_MOMENTS=save,
                    INPUT_SHIFT=input_shift,
                    num_warps=4, enable_fp_fusion=False)
        return NormResult(y, mu, moments, rstd) if save or training else y

    def _dispatch(self, x, weight, bias, save):
        self._check(x, weight, bias)
        weight0, weight_high = weight if isinstance(weight, tuple) else (weight, None)
        tensors = (x, weight0, weight_high, bias)
        if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
            result = NormResult(*_EquivariantNormFunction.apply(*tensors, self, save))
            return result if save else result.output
        return self._run(x, weight, bias, save)

    def forward(self, x, *, weight=None, bias=None):
        return self._dispatch(x, weight, bias, False)

    def forward_with_stats(self, x, *, weight=None, bias=None):
        """Return differentiable output and detached diagnostic statistics."""
        return self._dispatch(x, weight, bias, True)


class _SourceAdapter(torch.nn.Module):
    """Keep source parameter ownership/state keys; never call its forward.

    The registered source means .to() and parameter replacement stay live.
    The adapter's state_dict is namespaced under source.; source.state_dict()
    retains the original class keys. Source parameters receive Triton gradients.
    """
    def __init__(self, source, op):
        super().__init__()
        self.source, self.op = source, op
        self.spec = op.spec

    def _params(self):
        if not self.source.affine:
            return None, None
        if hasattr(self.source, 'norm_l0'):
            return (self.source.norm_l0.weight, self.source.affine_weight), self.source.norm_l0.bias
        return self.source.affine_weight, getattr(self.source, 'affine_bias', None)

    def forward(self, x):
        weight, bias = self._params()
        return self.op(x, weight=weight, bias=bias)

    def forward_with_stats(self, x):
        weight, bias = self._params()
        return self.op.forward_with_stats(x, weight=weight, bias=bias)


def from_reference(source, *, device=None):
    """Adapt verified source variants, preserving their grouping and FP32 weights.

    Construction may read tiny coefficient buffers to CPU; forward never does.
    No model-wide automatic replacement: target class choice remains explicit.
    """
    name = type(source).__name__
    if name in ('EquivariantLayerNorm', 'EquivariantLayerNormArray'):
        grouping, order, center = 'per_degree', 'components_first', True
    elif name in ('EquivariantSeparableLayerNorm', 'EquivariantLayerNormArraySphericalHarmonics'):
        grouping, order, center = 'scalar_high', 'components_first', True
        # V3 separable actually reduces channels before weighted components.
        if name == 'EquivariantSeparableLayerNorm':
            order = 'channels_first'
    elif name == 'EquivariantMergeLayerNorm':
        grouping, order, center = 'all', 'channels_first', source.centering
    else:
        raise NotImplementedError(f'no verified source adapter for {name}')
    weighting = ('degree_balanced' if source.normalization == 'component'
                 and getattr(source, 'std_balance_degrees', False) else source.normalization)
    spec = EquivariantNormSpec.from_preset(lmax=source.lmax, channels=source.num_channels,
        grouping=grouping, weighting=weighting, center_scalar=center, eps=source.eps)
    if weighting == 'degree_balanced':
        weights = [list(row) for row in spec.stats_weights]
        cached = source.balance_degree_weight.detach().reshape(-1).cpu()
        for l in range(0 if grouping == 'all' else 1, source.lmax+1):
            offset = l*l if grouping == 'all' else l*l-1
            weights[spec.output_group[l]][l] = float(cached[offset])
        spec = EquivariantNormSpec(spec.lmax, spec.channels, tuple(map(tuple, weights)),
                                   spec.output_group, spec.center_scalar, spec.eps)
    if device is None:
        tensors = list(source.parameters()) + list(source.buffers())
        device = tensors[0].device if tensors else torch.device('cuda')
    parameter_order = ('expanded' if name in ('EquivariantMergeLayerNorm', 'EquivariantSeparableLayerNorm')
                       else 'broadcast')
    op = TritonEquivariantNorm(spec, reduction_order=order, device=device,
                              parameter_order=parameter_order)
    op.source_kind = (1 if name in ("EquivariantLayerNorm", "EquivariantLayerNormArray")
                      else 2 if name == "EquivariantLayerNormArraySphericalHarmonics"
                      else 3 if name == "EquivariantMergeLayerNorm" else 4)
    op.source_component = source.normalization == "component"
    op.source_balanced = getattr(source, "std_balance_degrees", False)
    return _SourceAdapter(source, op)
