"""First-order FP32 gradients for disjoint equivariant normalization groups.

For a component in group g, a is its statistic coefficient, z is the
scalar-centered input, r = rsqrt(moment + eps), and h = dY * gamma:

    zhat = z * r
    Q[g] = sum_group(h * zhat)          # NOT weighted by a
    u = r * (h - (a / C) * zhat * Q[g])
    dX = u, except dX_scalar = u_scalar - mean_c(u_scalar)

The normalized form avoids explicitly forming r**3. Parameter gradients are
sum_(n,m)(dY * zhat) for gamma and sum_n(dY_scalar) for beta. Reductions use
deterministic partial buffers rather than floating-point atomic additions."""

import torch
import triton
import triton.language as tl

from .fused_equivariant_layer_norm_common import m_config, configured_m_sum, reduction_code


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


def backward(op, x, mean, rstd, weight0, weight_high, dy, needs):
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
