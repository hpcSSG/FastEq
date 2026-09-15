"""Triton parameter reductions preserving the original affine broadcast order.

The reduction schedule follows ATen CUDA Reduce.cuh: four independent sums,
descending block-y tree, and (when needed) the cross-block reduction. This is
FP32-source compatibility, not a high-precision summation algorithm.
"""
import torch
import triton
import triton.language as tl


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
