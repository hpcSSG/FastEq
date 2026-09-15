"""Source-ordered FP32 reduction within a resident normalization tile."""

import triton
import triton.language as tl

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
