"""SO(3) linear forward: serial, path-parallel and group-parallel kernels.

Descriptor: uv,iu,iv (scalar path coefficients).
X: [Z, sum(i*u)], shared W: [sum(u*v)] or [1, sum(u*v)].
Y: [Z, sum(i*v)]. Segment storage order is i-major/channel-minor.

Quick use:
    layout = mace_layout('large')
    with torch.no_grad():
        y_serial = so3_linear_serial_cta(w, x, layout)
        y_parallel = so3_linear_path_parallel(w, x, layout)
        y_group = so3_linear_group_parallel(w, x, layout)
        y_auto = so3_linear_auto(w, x, layout)  # benchmarks once per descriptor/dtype

Checks:
    python so3_linear_serial_cta.py --cpu-test
    python so3_linear_serial_cta.py --gpu-test --model large --batch 2208
    python so3_linear_serial_cta.py --gpu-test --model medium --dtype float32

Requires PyTorch + Triton for GPU execution; NumPy for --cpu-test only.
Provides forward plus an explicit grad_x-only backward kernel; grad_w and
double backward are not registered. Dtypes: FP16/BF16/FP32.
The path-parallel kernel launches one logical CTA grid per path and uses atomic
adds into an FP32 workspace when paths share an output segment. Its wrapper
includes output zeroing and the final cast to the input dtype.
FP32 defaults to IEEE dot, not reduced-precision TF32. On supported NVIDIA
hardware use input_precision='tf32x3' explicitly if desired.
Revision: flat constexpr metadata avoids runtime indexing of Triton tuples.
GPU numerical execution and performance require target-device testing.
"""
from dataclasses import dataclass
from functools import cached_property
import math
import operator


# Process-local cache: (canonical descriptor, dtype) -> implementation name.
# Z is intentionally excluded because the requested policy reuses one choice
# across changing batch sizes.
_BEST_IMPL_CACHE = {}


def _offsets(shapes):
    result, total = [], 0
    for a, b in shapes:
        result.append(total)
        total += a * b
    return tuple(result), total


@dataclass(frozen=True)
class LinearLayout:
    """Immutable, cacheable metadata. All indexes are zero-based.

    input_segments:  ((i, u), ...)
    output_segments: ((i, v), ...)
    weight_segments: ((u, v), ...)
    paths:           ((weight_segment, input_segment, output_segment), ...)
    coefficients:    one finite real scalar per path; defaults to 1.

    Allows arbitrary order, repeated paths, reused segments, per-segment channel
    counts, and outputs with no paths (written as zero). Connected i/u/v must
    match. Construct once at model initialization, not on each forward call.
    """
    input_segments: tuple
    output_segments: tuple
    weight_segments: tuple
    paths: tuple
    coefficients: tuple = ()

    def __post_init__(self):
        for name in ('input_segments', 'output_segments', 'weight_segments'):
            shapes = tuple(tuple(operator.index(v) for v in s) for s in getattr(self, name))
            if not shapes or any(len(s) != 2 or min(s) <= 0 for s in shapes):
                raise ValueError(f'{name} must contain positive two-dimensional shapes')
            object.__setattr__(self, name, shapes)
        paths = tuple(tuple(operator.index(v) for v in p) for p in self.paths)
        coeff = tuple(float(c) for c in self.coefficients)
        if not coeff:
            coeff = (1.0,) * len(paths)
        if len(coeff) != len(paths) or not all(math.isfinite(c) for c in coeff):
            raise ValueError('need one finite coefficient per path')
        for p in paths:
            if len(p) != 3:
                raise ValueError('path must be (weight, input, output)')
            w, x, y = p
            if not (0 <= w < len(self.weight_segments) and
                    0 <= x < len(self.input_segments) and
                    0 <= y < len(self.output_segments)):
                raise ValueError(f'path index out of bounds: {p}')
            i, u = self.input_segments[x]
            j, v = self.output_segments[y]
            if i != j or self.weight_segments[w] != (u, v):
                raise ValueError(f'incompatible i/u/v in path {p}')
        object.__setattr__(self, 'paths', paths)
        object.__setattr__(self, 'coefficients', coeff)

    @cached_property
    def sizes(self):
        """(weight_size, input_size, output_size), as in the descriptor log."""
        return tuple(_offsets(s)[1] for s in
                     (self.weight_segments, self.input_segments, self.output_segments))

    @cached_property
    def groups(self):
        xo, _ = _offsets(self.input_segments)
        wo, _ = _offsets(self.weight_segments)
        yo, _ = _offsets(self.output_segments)
        # Each group: (i, v, output_offset, ((input_offset,u,weight_offset,c),...)).
        result = []
        for g, (i, v) in enumerate(self.output_segments):
            records = tuple((xo[x], self.input_segments[x][1], wo[w], c)
                            for (w, x, y), c in zip(self.paths, self.coefficients) if y == g)
            result.append((i, v, yo[g], records))
        return tuple(result)


    @cached_property
    def flat_metadata(self):
        """Separate flat constexpr tables: never index nested tuples in Triton."""
        gi, gv, go, gs, gc = [], [], [], [], []
        px, pu, pw, pc = [], [], [], []
        for i, v, offset, records in self.groups:
            gi.append(i)
            gv.append(v)
            go.append(offset)
            gs.append(len(px))
            gc.append(len(records))
            for xoff, u, woff, coefficient in records:
                px.append(xoff)
                pu.append(u)
                pw.append(woff)
                pc.append(coefficient)
        return tuple(tuple(t) for t in (gi, gv, go, gs, gc, px, pu, pw, pc))

    @cached_property
    def path_metadata(self):
        """Flat per-path constexpr tables for the path-parallel kernel."""
        xo, _ = _offsets(self.input_segments)
        wo, _ = _offsets(self.weight_segments)
        yo, _ = _offsets(self.output_segments)
        pi, pv, py, px, pu, pw, pc = [], [], [], [], [], [], []
        for (w, x, y), coefficient in zip(self.paths, self.coefficients):
            i, u = self.input_segments[x]
            _, v = self.output_segments[y]
            pi.append(i)
            pv.append(v)
            py.append(yo[y])
            px.append(xo[x])
            pu.append(u)
            pw.append(wo[w])
            pc.append(coefficient)
        return tuple(tuple(t) for t in (pi, pv, py, px, pu, pw, pc))

    @cached_property
    def backward_x_metadata(self):
        """Flat input-group metadata for ``grad_x = cg * grad_y @ W.T``."""
        xo, _ = _offsets(self.input_segments)
        wo, _ = _offsets(self.weight_segments)
        yo, _ = _offsets(self.output_segments)
        gi, gu, gx, gs, gc = [], [], [], [], []
        py, pv, pw, pc = [], [], [], []
        for input_id, (i, u) in enumerate(self.input_segments):
            gi.append(i)
            gu.append(u)
            gx.append(xo[input_id])
            gs.append(len(py))
            records = [((w, y), coefficient)
                       for (w, x, y), coefficient in zip(self.paths, self.coefficients)
                       if x == input_id]
            gc.append(len(records))
            for (weight_id, output_id), coefficient in records:
                _, v = self.output_segments[output_id]
                py.append(yo[output_id])
                pv.append(v)
                pw.append(wo[weight_id])
                pc.append(coefficient)
        return tuple(tuple(t) for t in (gi, gu, gx, gs, gc, py, pv, pw, pc))


def uniform_layout(u, v, input_i, output_i, path_to_output=None, coefficients=()):
    """One weight segment per input segment. Explicit mapping for ambiguous i.

    For more general connectivity use LinearLayout directly.
    """
    input_i, output_i = tuple(input_i), tuple(output_i)
    if path_to_output is None:
        if len(set(output_i)) != len(output_i):
            raise ValueError('duplicate output i: provide path_to_output explicitly')
        try:
            path_to_output = tuple(output_i.index(i) for i in input_i)
        except ValueError as exc:
            raise ValueError('an input i has no matching output') from exc
    path_to_output = tuple(path_to_output)
    if len(path_to_output) != len(input_i):
        raise ValueError('path_to_output must have one entry per input segment')
    return LinearLayout(tuple((i,u) for i in input_i), tuple((i,v) for i in output_i),
                        ((u,v),) * len(input_i),
                        tuple((p,p,g) for p,g in enumerate(path_to_output)),
                        tuple(coefficients))


def mace_layout(model='large'):
    presets = {
        'large': (224, (1,1,1,3,3,3,3,3,5,5,5,5,5,7,7,7,7)),
        'medium': (128, (1,1,3,3,3,5,5,5,7,7)),
        'small': (96, (1,3,5,7)),
    }
    try:
        c, dims = presets[model.lower()]
    except KeyError as exc:
        raise ValueError('model must be large, medium or small') from exc
    return uniform_layout(c, c, dims, (1,3,5,7))


try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    torch = triton = tl = None


if triton is not None:
    @triton.jit
    def _serial_output_group(
        W, X, Y, Z, SX0, SX1, SW,
        z0, v0,
        I: tl.constexpr, V: tl.constexpr, YO: tl.constexpr,
        START: tl.constexpr, COUNT: tl.constexpr,
        PX: tl.constexpr, PU: tl.constexpr, PW: tl.constexpr, PC: tl.constexpr,
        DY: tl.constexpr,
        BZ: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        # Inlined helper; only the entry kernel is launched.
        for row_tile in range(tl.cdiv(BZ * I, BM)):
            r = row_tile * BM + tl.arange(0, BM)
            z = z0 + r // I
            m = r % I
            v = v0 + tl.arange(0, BN)
            row_ok = (r < BZ * I) & (z < Z)
            acc = tl.zeros((BM, BN), tl.float32)
            for p in tl.static_range(START, START + COUNT):
                for kt in range(tl.cdiv(PU[p], BK)):
                    u = kt * BK + tl.arange(0, BK)
                    a = tl.load(
                        X + z[:, None].to(tl.int64) * SX0 +
                        (PX[p] + m[:, None] * PU[p] + u[None, :]).to(tl.int64) * SX1,
                        row_ok[:, None] & (u[None, :] < PU[p]), other=0)
                    b = tl.load(
                        W + (PW[p] + u[:, None].to(tl.int64) * V + v[None, :]) * SW,
                        (u[:, None] < PU[p]) & (v[None, :] < V), other=0)
                    if PC[p] == 1.0:
                        acc = tl.dot(a, b, acc, input_precision=PRECISION)
                    else:
                        product = tl.dot(a, b, input_precision=PRECISION)
                        acc = acc + product * PC[p]
            tl.store(Y + z[:, None].to(tl.int64) * DY + YO + m[:, None] * V + v[None, :],
                     acc, row_ok[:, None] & (v[None, :] < V))

    @triton.jit
    def _so3_linear_serial_cta_kernel(
        W, X, Y, Z, SX0, SX1, SW,
        GI: tl.constexpr, GV: tl.constexpr, GO: tl.constexpr,
        GS: tl.constexpr, GC: tl.constexpr,
        PX: tl.constexpr, PU: tl.constexpr, PW: tl.constexpr, PC: tl.constexpr,
        DY: tl.constexpr,
        BZ: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        z0 = tl.program_id(0) * BZ
        v0 = tl.program_id(1) * BN
        # All compiler-facing metadata consists of flat scalar tuples.
        for g in tl.static_range(len(GI)):
            if v0 < GV[g]:
                _serial_output_group(
                    W, X, Y, Z, SX0, SX1, SW, z0, v0,
                    GI[g], GV[g], GO[g], GS[g], GC[g], PX, PU, PW, PC, DY,
                    BZ, BM, BN, BK, PRECISION)

    @triton.jit
    def _so3_linear_path_parallel_kernel(
        W, X, Y, Z, SX0, SX1, SW,
        PI: tl.constexpr, PV: tl.constexpr, PY: tl.constexpr,
        PX: tl.constexpr, PU: tl.constexpr, PW: tl.constexpr,
        PC: tl.constexpr, DY: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        path_id = tl.program_id(0)
        row_tile = tl.program_id(1)
        v_tile = tl.program_id(2)
        # Static branches retain compile-time path metadata. Runtime tuple
        # indexing is deliberately avoided because constexpr tuples cannot be
        # indexed by a scalar tensor in Triton.
        for p in tl.static_range(len(PI)):
            if path_id == p:
                r = row_tile * BM + tl.arange(0, BM)
                v = v_tile * BN + tl.arange(0, BN)
                z = r // PI[p]
                m = r % PI[p]
                row_ok = r < Z * PI[p]
                acc = tl.zeros((BM, BN), tl.float32)
                for kt in range(tl.cdiv(PU[p], BK)):
                    u = kt * BK + tl.arange(0, BK)
                    a = tl.load(
                        X + z[:, None].to(tl.int64) * SX0 +
                        (PX[p] + m[:, None] * PU[p] + u[None, :]).to(tl.int64) * SX1,
                        row_ok[:, None] & (u[None, :] < PU[p]), other=0)
                    b = tl.load(
                        W + (PW[p] + u[:, None].to(tl.int64) * PV[p] + v[None, :]) * SW,
                        (u[:, None] < PU[p]) & (v[None, :] < PV[p]), other=0)
                    acc = tl.dot(a, b, acc, input_precision=PRECISION)
                if PC[p] != 1.0:
                    acc *= PC[p]
                tl.atomic_add(
                    Y + z[:, None].to(tl.int64) * DY + PY[p] +
                    m[:, None] * PV[p] + v[None, :],
                    acc,
                    mask=row_ok[:, None] & (v[None, :] < PV[p]))

    @triton.jit
    def _so3_linear_group_parallel_kernel(
        W, X, Y, Z, SX0, SX1, SW,
        GI: tl.constexpr, GV: tl.constexpr, GO: tl.constexpr,
        GS: tl.constexpr, GC: tl.constexpr,
        PX: tl.constexpr, PU: tl.constexpr, PW: tl.constexpr,
        PC: tl.constexpr, DY: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        row_tile = tl.program_id(1)
        v_tile = tl.program_id(2)
        # One CTA owns one complete output tile. Groups execute in parallel;
        # paths contributing to a group accumulate serially in registers.
        for g in tl.static_range(len(GI)):
            if group_id == g:
                r = row_tile * BM + tl.arange(0, BM)
                v = v_tile * BN + tl.arange(0, BN)
                z = r // GI[g]
                m = r % GI[g]
                row_ok = r < Z * GI[g]
                acc = tl.zeros((BM, BN), tl.float32)
                # Bounds passed to tl.static_range must be syntactically
                # constexpr. Iterate over the globally fixed path count, then
                # let compile-time group bounds prune unrelated paths.
                for p in tl.static_range(len(PX)):
                    if p >= GS[g]:
                        if p < GS[g] + GC[g]:
                            for kt in range(tl.cdiv(PU[p], BK)):
                                u = kt * BK + tl.arange(0, BK)
                                a = tl.load(
                                    X + z[:, None].to(tl.int64) * SX0 +
                                    (PX[p] + m[:, None] * PU[p] + u[None, :]).to(tl.int64) * SX1,
                                    row_ok[:, None] & (u[None, :] < PU[p]), other=0)
                                b = tl.load(
                                    W + (PW[p] + u[:, None].to(tl.int64) * GV[g] + v[None, :]) * SW,
                                    (u[:, None] < PU[p]) & (v[None, :] < GV[g]), other=0)
                                if PC[p] == 1.0:
                                    acc = tl.dot(a, b, acc, input_precision=PRECISION)
                                else:
                                    acc += tl.dot(a, b, input_precision=PRECISION) * PC[p]
                tl.store(
                    Y + z[:, None].to(tl.int64) * DY + GO[g] +
                    m[:, None] * GV[g] + v[None, :],
                    acc,
                    mask=row_ok[:, None] & (v[None, :] < GV[g]))

    @triton.jit
    def _so3_linear_backward_x_group_kernel(
        W, GY, GX, Z, SGY0, SGY1, SW,
        GI: tl.constexpr, GU: tl.constexpr, GXO: tl.constexpr,
        GS: tl.constexpr, GC: tl.constexpr,
        PY: tl.constexpr, PV: tl.constexpr, PW: tl.constexpr,
        PC: tl.constexpr, DX: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        input_id = tl.program_id(0)
        row_tile = tl.program_id(1)
        u_tile = tl.program_id(2)
        # Each CTA exclusively owns one grad_x tile. Paths targeting the same
        # input segment accumulate in FP32 registers, so no atomics are needed.
        for g in tl.static_range(len(GI)):
            if input_id == g:
                r = row_tile * BM + tl.arange(0, BM)
                u = u_tile * BN + tl.arange(0, BN)
                z = r // GI[g]
                m = r % GI[g]
                row_ok = r < Z * GI[g]
                acc = tl.zeros((BM, BN), tl.float32)
                for p in tl.static_range(len(PY)):
                    if p >= GS[g]:
                        if p < GS[g] + GC[g]:
                            for vt in range(tl.cdiv(PV[p], BK)):
                                v = vt * BK + tl.arange(0, BK)
                                dy = tl.load(
                                    GY + z[:, None].to(tl.int64) * SGY0 +
                                    (PY[p] + m[:, None] * PV[p] + v[None, :]).to(tl.int64) * SGY1,
                                    row_ok[:, None] & (v[None, :] < PV[p]), other=0)
                                # W is row-major [u, v]; this KxN tile is W.T.
                                wt = tl.load(
                                    W + (PW[p] + u[None, :] * PV[p] + v[:, None]) * SW,
                                    (v[:, None] < PV[p]) & (u[None, :] < GU[g]), other=0)
                                if PC[p] == 1.0:
                                    acc = tl.dot(dy, wt, acc, input_precision=PRECISION)
                                else:
                                    acc += tl.dot(dy, wt, input_precision=PRECISION) * PC[p]
                tl.store(
                    GX + z[:, None].to(tl.int64) * DX + GXO[g] +
                    m[:, None] * GU[g] + u[None, :],
                    acc,
                    mask=row_ok[:, None] & (u[None, :] < GU[g]))


def so3_linear_serial_cta(w, x, layout, *, block_z=16, block_m=32,
                          block_n=64, block_k=32, num_warps=4,
                          num_stages=2, input_precision='ieee'):
    """Return Y using exactly one Triton launch for nonempty X.

    Every CTA walks all output groups/path lists for its node/channel tile.
    If V differs by output segment, channel tiles outside that V skip the group.
    No physical concatenation, intermediate path outputs, atomics or split-K.
    Accepts positive-stride X and flat W; weights are shared across all Z.
    Compile-specializes to immutable layout and tile options; cache layout once.
    """
    if triton is None:
        raise RuntimeError('GPU execution requires installed torch and triton')
    if not isinstance(layout, LinearLayout):
        raise TypeError('layout must be LinearLayout')
    if not x.is_cuda or not w.is_cuda or x.device != w.device:
        raise ValueError('X and W must be on the same CUDA/HIP device')
    if x.dtype != w.dtype or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('matching FP16/BF16/FP32 tensors required; FP64 is not supported')
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        raise RuntimeError('forward-only kernel: no backward registered; use torch.no_grad() for inference')
    dw, dx, dy = layout.sizes
    if x.ndim != 2 or x.shape[1] != dx:
        raise ValueError(f'X must have shape [Z, {dx}]')
    if not ((w.ndim == 1 and w.shape[0] == dw) or
            (w.ndim == 2 and tuple(w.shape) == (1, dw))):
        raise ValueError(f'W must have shape [{dw}] or [1, {dw}]; per-node weights are unsupported')
    if any(s <= 0 for s in x.stride()) or w.stride(-1) <= 0:
        raise ValueError('positive input/weight strides required')
    for name, val in [('block_z',block_z), ('block_m',block_m), ('block_n',block_n),
                      ('block_k',block_k), ('num_warps',num_warps), ('num_stages',num_stages)]:
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if any(b < 16 or b & (b-1) for b in (block_m,block_n,block_k)):
        raise ValueError('block_m/n/k must be powers of two >= 16')
    if num_warps not in (4,8):
        raise ValueError('num_warps must be 4 or 8')
    if input_precision not in ('ieee','tf32','tf32x3'):
        raise ValueError('input_precision must be ieee, tf32 or tf32x3')
    if torch.version.hip and input_precision != 'ieee':
        raise ValueError('this wrapper uses ieee for HIP portability')
    y = torch.empty((x.shape[0], dy), device=x.device, dtype=x.dtype)
    if x.shape[0] == 0:
        return y
    grid = (triton.cdiv(x.shape[0], block_z),
            triton.cdiv(max(v for _, v in layout.output_segments), block_n))
    with torch.cuda.device(x.device):
        _so3_linear_serial_cta_kernel[grid](
            w,x,y,x.shape[0],x.stride(0),x.stride(1),w.stride(-1),
            *layout.flat_metadata,dy,block_z,block_m,block_n,block_k,input_precision,
            num_warps=num_warps,num_stages=num_stages)
    return y


def so3_linear_path_parallel(w, x, layout, *, block_m=32, block_n=64,
                             block_k=32, num_warps=4, num_stages=2,
                             input_precision='ieee'):
    """Compute Y with paths distributed across CTAs and atomic accumulation.

    Grid axes are ``(path, flattened Z*i tile, v tile)``. The output is zeroed
    before launch because multiple paths may contribute to the same segment.
    Atomic accumulation uses an FP32 workspace so FP16/BF16 paths are not
    rounded after every add. Wrapper timing includes allocation, zeroing and
    the final conversion to the input dtype.
    """
    if triton is None:
        raise RuntimeError('GPU execution requires installed torch and triton')
    if not isinstance(layout, LinearLayout):
        raise TypeError('layout must be LinearLayout')
    if not x.is_cuda or not w.is_cuda or x.device != w.device:
        raise ValueError('X and W must be on the same CUDA/HIP device')
    if x.dtype != w.dtype or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('matching FP16/BF16/FP32 tensors required; FP64 is not supported')
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        raise RuntimeError('forward-only kernel: no backward registered; use torch.no_grad() for inference')
    dw, dx, dy = layout.sizes
    if x.ndim != 2 or x.shape[1] != dx:
        raise ValueError(f'X must have shape [Z, {dx}]')
    if not ((w.ndim == 1 and w.shape[0] == dw) or
            (w.ndim == 2 and tuple(w.shape) == (1, dw))):
        raise ValueError(f'W must have shape [{dw}] or [1, {dw}]; per-node weights are unsupported')
    if any(s <= 0 for s in x.stride()) or w.stride(-1) <= 0:
        raise ValueError('positive input/weight strides required')
    for name, val in [('block_m', block_m), ('block_n', block_n), ('block_k', block_k),
                      ('num_warps', num_warps), ('num_stages', num_stages)]:
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if any(b < 16 or b & (b - 1) for b in (block_m, block_n, block_k)):
        raise ValueError('block_m/n/k must be powers of two >= 16')
    if num_warps not in (4, 8):
        raise ValueError('num_warps must be 4 or 8')
    if input_precision not in ('ieee', 'tf32', 'tf32x3'):
        raise ValueError('input_precision must be ieee, tf32 or tf32x3')
    if torch.version.hip and input_precision != 'ieee':
        raise ValueError('this wrapper uses ieee for HIP portability')
    y = torch.zeros((x.shape[0], dy), device=x.device, dtype=torch.float32)
    if x.shape[0] == 0 or not layout.paths:
        return y.to(x.dtype)
    max_i = max(layout.input_segments[xi][0] for _, xi, _ in layout.paths)
    max_v = max(layout.output_segments[yi][1] for _, _, yi in layout.paths)
    grid = (len(layout.paths), triton.cdiv(x.shape[0] * max_i, block_m),
            triton.cdiv(max_v, block_n))
    with torch.cuda.device(x.device):
        _so3_linear_path_parallel_kernel[grid](
            w, x, y, x.shape[0], x.stride(0), x.stride(1), w.stride(-1),
            *layout.path_metadata, dy, block_m, block_n, block_k, input_precision,
            num_warps=num_warps, num_stages=num_stages)
    return y.to(x.dtype)


def so3_linear_group_parallel(w, x, layout, *, block_m=32, block_n=64,
                              block_k=32, num_warps=4, num_stages=2,
                              input_precision='ieee'):
    """Compute Y with output groups parallel and paths serial within a CTA.

    Each output tile has exactly one writer, so this variant needs neither
    atomics nor a global FP32 accumulation workspace. Accumulation remains FP32
    in registers and is converted only by the final store.
    """
    if triton is None:
        raise RuntimeError('GPU execution requires installed torch and triton')
    if not isinstance(layout, LinearLayout):
        raise TypeError('layout must be LinearLayout')
    if not x.is_cuda or not w.is_cuda or x.device != w.device:
        raise ValueError('X and W must be on the same CUDA/HIP device')
    if x.dtype != w.dtype or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('matching FP16/BF16/FP32 tensors required; FP64 is not supported')
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        raise RuntimeError('forward-only kernel: no backward registered; use torch.no_grad() for inference')
    dw, dx, dy = layout.sizes
    if x.ndim != 2 or x.shape[1] != dx:
        raise ValueError(f'X must have shape [Z, {dx}]')
    if not ((w.ndim == 1 and w.shape[0] == dw) or
            (w.ndim == 2 and tuple(w.shape) == (1, dw))):
        raise ValueError(f'W must have shape [{dw}] or [1, {dw}]; per-node weights are unsupported')
    if any(s <= 0 for s in x.stride()) or w.stride(-1) <= 0:
        raise ValueError('positive input/weight strides required')
    for name, val in [('block_m', block_m), ('block_n', block_n), ('block_k', block_k),
                      ('num_warps', num_warps), ('num_stages', num_stages)]:
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if any(b < 16 or b & (b - 1) for b in (block_m, block_n, block_k)):
        raise ValueError('block_m/n/k must be powers of two >= 16')
    if num_warps not in (4, 8):
        raise ValueError('num_warps must be 4 or 8')
    if input_precision not in ('ieee', 'tf32', 'tf32x3'):
        raise ValueError('input_precision must be ieee, tf32 or tf32x3')
    if torch.version.hip and input_precision != 'ieee':
        raise ValueError('this wrapper uses ieee for HIP portability')
    y = torch.empty((x.shape[0], dy), device=x.device, dtype=x.dtype)
    if x.shape[0] == 0:
        return y
    max_i = max(i for i, _ in layout.output_segments)
    max_v = max(v for _, v in layout.output_segments)
    grid = (len(layout.output_segments), triton.cdiv(x.shape[0] * max_i, block_m),
            triton.cdiv(max_v, block_n))
    with torch.cuda.device(x.device):
        _so3_linear_group_parallel_kernel[grid](
            w, x, y, x.shape[0], x.stride(0), x.stride(1), w.stride(-1),
            *layout.flat_metadata, dy, block_m, block_n, block_k, input_precision,
            num_warps=num_warps, num_stages=num_stages)
    return y


def so3_linear_backward_x(w, grad_out, layout, *, block_m=32, block_n=64,
                          block_k=32, num_warps=4, num_stages=2,
                          input_precision='ieee'):
    """Return only grad_x using input-group parallel, path-serial accumulation.

    Computes ``sum_p coefficient[p] * grad_out[p] @ W[p].T``. Every grad_x
    element has one writer. This function does not provide grad_w or double
    backward registration.
    """
    if triton is None:
        raise RuntimeError('GPU execution requires installed torch and triton')
    if not isinstance(layout, LinearLayout):
        raise TypeError('layout must be LinearLayout')
    if not grad_out.is_cuda or not w.is_cuda or grad_out.device != w.device:
        raise ValueError('grad_out and W must be on the same CUDA/HIP device')
    if grad_out.dtype != w.dtype or grad_out.dtype not in (
            torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('matching FP16/BF16/FP32 tensors required; FP64 is not supported')
    dw, dx, dy = layout.sizes
    if grad_out.ndim != 2 or grad_out.shape[1] != dy:
        raise ValueError(f'grad_out must have shape [Z, {dy}]')
    if not ((w.ndim == 1 and w.shape[0] == dw) or
            (w.ndim == 2 and tuple(w.shape) == (1, dw))):
        raise ValueError(f'W must have shape [{dw}] or [1, {dw}]')
    if any(s <= 0 for s in grad_out.stride()) or w.stride(-1) <= 0:
        raise ValueError('positive grad_out/weight strides required')
    for name, value in [('block_m', block_m), ('block_n', block_n),
                        ('block_k', block_k), ('num_warps', num_warps),
                        ('num_stages', num_stages)]:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if any(b < 16 or b & (b - 1) for b in (block_m, block_n, block_k)):
        raise ValueError('block_m/n/k must be powers of two >= 16')
    if num_warps not in (4, 8):
        raise ValueError('num_warps must be 4 or 8')
    if input_precision not in ('ieee', 'tf32', 'tf32x3'):
        raise ValueError('input_precision must be ieee, tf32 or tf32x3')
    if torch.version.hip and input_precision != 'ieee':
        raise ValueError('this wrapper uses ieee for HIP portability')

    grad_x = torch.empty((grad_out.shape[0], dx), device=grad_out.device,
                         dtype=grad_out.dtype)
    if grad_out.shape[0] == 0:
        return grad_x
    max_i = max(i for i, _ in layout.input_segments)
    max_u = max(u for _, u in layout.input_segments)
    grid = (len(layout.input_segments),
            triton.cdiv(grad_out.shape[0] * max_i, block_m),
            triton.cdiv(max_u, block_n))
    with torch.cuda.device(grad_out.device):
        _so3_linear_backward_x_group_kernel[grid](
            w, grad_out, grad_x, grad_out.shape[0],
            grad_out.stride(0), grad_out.stride(1), w.stride(-1),
            *layout.backward_x_metadata, dx,
            block_m, block_n, block_k, input_precision,
            num_warps=num_warps, num_stages=num_stages)
    return grad_x


def best_impl_cache_key(layout, dtype):
    """Return the exact ``(descriptor, dtype)`` auto-selection cache key.

    The canonical descriptor mirrors logs such as::

        descriptor:uv,iu,iv sizes=852992,15904,3584
        num_segments=17,17,4 num_paths=17 i={1,3,5,7} u=224 v=224

    Batch size Z, tensor strides and device are deliberately not part of the
    key. Consequently, the first observed Z determines the cached winner for
    all later Z values with the same descriptor and dtype.
    """
    if not isinstance(layout, LinearLayout):
        raise TypeError('layout must be LinearLayout')
    i_values = tuple(sorted({i for i, _ in layout.input_segments} |
                            {i for i, _ in layout.output_segments}))
    u_values = tuple(sorted({u for _, u in layout.input_segments} |
                            {u for u, _ in layout.weight_segments}))
    v_values = tuple(sorted({v for _, v in layout.output_segments} |
                            {v for _, v in layout.weight_segments}))
    descriptor = (
        'uv,iu,iv',
        layout.sizes,
        (len(layout.weight_segments), len(layout.input_segments),
         len(layout.output_segments)),
        len(layout.paths),
        i_values,
        u_values,
        v_values,
    )
    return descriptor, str(dtype)


def get_best_impl_cache():
    """Return a snapshot of cached descriptor/dtype implementation choices."""
    return dict(_BEST_IMPL_CACHE)


def clear_best_impl_cache():
    """Discard all process-local implementation choices."""
    _BEST_IMPL_CACHE.clear()


def so3_linear_auto(w, x, layout, *, benchmark_warmup=10,
                    benchmark_rep=50, verbose=False):
    """Run the cached fastest Triton implementation for descriptor and dtype.

    On a cache miss, benchmarks serial CTA, path-parallel and group-parallel
    wrappers using the current inputs, caches the fastest implementation name,
    then executes it. Z affects that first measurement but is not part of the
    key, as requested. Call ``clear_best_impl_cache()`` to retune explicitly.
    """
    if torch is None or triton is None:
        raise RuntimeError('automatic selection requires installed torch and triton')
    if not isinstance(benchmark_warmup, int) or isinstance(benchmark_warmup, bool) or benchmark_warmup < 0:
        raise ValueError('benchmark_warmup must be a nonnegative integer')
    if not isinstance(benchmark_rep, int) or isinstance(benchmark_rep, bool) or benchmark_rep <= 0:
        raise ValueError('benchmark_rep must be a positive integer')
    key = best_impl_cache_key(layout, x.dtype)
    implementations = {
        'serial': so3_linear_serial_cta,
        'path_parallel': so3_linear_path_parallel,
        'group_parallel': so3_linear_group_parallel,
    }
    selected = _BEST_IMPL_CACHE.get(key)
    if selected is None:
        timings = {}
        for name, implementation in implementations.items():
            timings[name] = _bench(
                lambda implementation=implementation: implementation(w, x, layout),
                warmup=benchmark_warmup,
                rep=benchmark_rep)
        selected = min(timings, key=timings.get)
        _BEST_IMPL_CACHE[key] = selected
        if verbose:
            timing_text = ', '.join(f'{name}={ms:.6f} ms'
                                    for name, ms in timings.items())
            print(f'SO3 linear autotune: {timing_text}; selected={selected}')
    elif verbose:
        print(f'SO3 linear cache hit: selected={selected}')
    return implementations[selected](w, x, layout)


def torch_reference(w, x, layout):
    """Independent path-wise reference. Accumulate low-precision inputs in FP32."""
    xo, _ = _offsets(layout.input_segments)
    wo, _ = _offsets(layout.weight_segments)
    yo, dy = _offsets(layout.output_segments)
    wf = w.reshape(-1).float()
    result = torch.zeros((x.shape[0],dy), device=x.device, dtype=torch.float32)
    for (wi,xi,yi), c in zip(layout.paths,layout.coefficients):
        i,u = layout.input_segments[xi]
        _,v = layout.output_segments[yi]
        a = x[:,xo[xi]:xo[xi]+i*u].float().reshape(-1,i,u)
        b = wf[wo[wi]:wo[wi]+u*v].reshape(u,v)
        result[:,yo[yi]:yo[yi]+i*v] += (a @ b).reshape(x.shape[0],i*v) * c
    return result.to(x.dtype)


def torch_backward_x_reference(w, grad_out, layout):
    """Independent path-wise reference for grad_x only."""
    xo, dx = _offsets(layout.input_segments)
    wo, _ = _offsets(layout.weight_segments)
    yo, _ = _offsets(layout.output_segments)
    wf = w.reshape(-1).float()
    result = torch.zeros((grad_out.shape[0], dx), device=grad_out.device,
                         dtype=torch.float32)
    for (wi, xi, yi), coefficient in zip(layout.paths, layout.coefficients):
        i, u = layout.input_segments[xi]
        _, v = layout.output_segments[yi]
        gy = grad_out[:, yo[yi]:yo[yi] + i * v].float().reshape(-1, i, v)
        weight = wf[wo[wi]:wo[wi] + u * v].reshape(u, v)
        result[:, xo[xi]:xo[xi] + i * u] += (
            gy @ weight.T).reshape(grad_out.shape[0], i * u) * coefficient
    return result.to(grad_out.dtype)


def _irregular_layout():
    return LinearLayout(((3,7),(1,5)), ((1,11),(3,9),(5,2)),
                        ((7,9),(5,11)), ((1,1,0),(0,0,1),(0,0,1)), (0.5,-1.,2.))


def cpu_test():
    """Validate metadata/address mapping against an independent dense operator.

    This is CPU emulation, not execution or compilation of the Triton kernel.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    for name, expected in [('large',(852992,15904,3584)),
                           ('medium',(163840,5120,2048)), ('small',(36864,1536,1536))]:
        assert mace_layout(name).sizes == expected
    cases = [uniform_layout(5,7,(1,1,3,3,3,5,5,5,7,7),(1,3,5,7)),
             _irregular_layout(),
             LinearLayout(((1,3),),((1,2),),((3,2),),())]
    for layout in cases:
        dw,dx,dy = layout.sizes
        zcount, bz, bm, bn, bk = 7,3,4,5,3
        x = rng.normal(size=(zcount,dx))
        w = rng.normal(size=dw)
        xo,_ = _offsets(layout.input_segments)
        wo,_ = _offsets(layout.weight_segments)
        yo,_ = _offsets(layout.output_segments)
        dense = np.zeros((dx,dy))
        for (wi,xi,yi),c in zip(layout.paths,layout.coefficients):
            i,u = layout.input_segments[xi]
            _,v = layout.output_segments[yi]
            b = w[wo[wi]:wo[wi]+u*v].reshape(u,v)
            for m in range(i):
                dense[xo[xi]+m*u:xo[xi]+(m+1)*u,
                      yo[yi]+m*v:yo[yi]+(m+1)*v] += c*b
        reference = x @ dense
        actual = np.full((zcount,dy), np.nan)
        writes = np.zeros((zcount,dy), dtype=int)
        for z0 in range(0,zcount,bz):
            for v0 in range(0,max(v for _,v in layout.output_segments),bn):
                for i,v,yoff,records in layout.groups:
                    if v0 >= v:
                        continue
                    vs = np.arange(v0,min(v,v0+bn))
                    for r0 in range(0,bz*i,bm):
                        r = np.arange(r0,min(bz*i,r0+bm))
                        zs,ms = z0+r//i,r%i
                        valid = zs < zcount
                        zs,ms = zs[valid],ms[valid]
                        acc = np.zeros((len(zs),len(vs)))
                        for xoff,u,woff,c in records:
                            for k0 in range(0,u,bk):
                                us = np.arange(k0,min(u,k0+bk))
                                a = x[zs[:,None],xoff+ms[:,None]*u+us]
                                b = w[woff+us[:,None]*v+vs]
                                acc += (a@b)*c
                        actual[zs[:,None],yoff+ms[:,None]*v+vs] = acc
                        writes[zs[:,None],yoff+ms[:,None]*v+vs] += 1
        np.testing.assert_allclose(actual,reference,rtol=1e-12,atol=1e-12)
        assert np.all(writes == 1), 'every output must have exactly one writer'
    print('CPU PASS: preset sizes, arbitrary paths/coefficients, tails, zero outputs, single-writer address mapping')
    print('CPU checks do NOT validate Triton compilation or GPU execution.')


def _bench(fn, warmup=10, rep=50):
    """Explicit warmup, then measure with do_bench; returns mean latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=0, rep=rep)


def gpu_test(model, batch, dtype_name):
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError('GPU test requires torch, triton and an available GPU')
    dtype = getattr(torch,dtype_name)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    layouts = [mace_layout(model), _irregular_layout(),
               LinearLayout(((1,3),),((1,2),),((3,2),),())]
    atol,rtol = {'float32':(2e-4,2e-4), 'float16':(2e-3,2e-3),
                 'bfloat16':(2e-2,2e-2)}[dtype_name]
    with torch.no_grad():
        for idx,layout in enumerate(layouts):
            dw,dx,dy = layout.sizes
            z = batch if idx == 0 else 19
            # Test noncontiguous X and W in the irregular case.
            if idx == 1:
                x = torch.randn((z,dx*2),device='cuda',dtype=dtype)[:,::2]
                w = torch.randn((1,dw*2),device='cuda',dtype=dtype)[:,::2]
            else:
                x = torch.randn((z,dx),device='cuda',dtype=dtype)
                w = torch.randn((1,dw),device='cuda',dtype=dtype)
            w.mul_(0.1)
            actual = so3_linear_serial_cta(w,x,layout)
            parallel = so3_linear_path_parallel(w,x,layout)
            grouped = so3_linear_group_parallel(w,x,layout)
            expected = torch_reference(w,x,layout)
            grad_out = torch.randn_like(expected)
            grad_x = so3_linear_backward_x(w, grad_out, layout)
            expected_grad_x = torch_backward_x_reference(w, grad_out, layout)
            torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
            torch.testing.assert_close(parallel,expected,atol=atol,rtol=rtol)
            torch.testing.assert_close(grouped,expected,atol=atol,rtol=rtol)
            torch.testing.assert_close(grad_x,expected_grad_x,atol=atol,rtol=rtol)
            print(f'GPU PASS case={idx}, shape={tuple(actual.shape)}, '
                  f'serial_max_abs={(actual.float()-expected.float()).abs().max().item():.6g}, '
                  f'path_max_abs={(parallel.float()-expected.float()).abs().max().item():.6g}, '
                  f'group_max_abs={(grouped.float()-expected.float()).abs().max().item():.6g}, '
                  f'grad_x_max_abs={(grad_x.float()-expected_grad_x.float()).abs().max().item():.6g}')
            if idx == 0:
                clear_best_impl_cache()
                automatic = so3_linear_auto(w, x, layout, verbose=True)
                torch.testing.assert_close(automatic,expected,atol=atol,rtol=rtol)
                cache_key = best_impl_cache_key(layout, dtype)
                print(f'auto_cache_selected={get_best_impl_cache()[cache_key]}')
                ms_serial = _bench(lambda: so3_linear_serial_cta(w,x,layout))
                ms_parallel = _bench(lambda: so3_linear_path_parallel(w,x,layout))
                ms_group = _bench(lambda: so3_linear_group_parallel(w,x,layout))
                ms_torch = _bench(lambda: torch_reference(w,x,layout))
                ms_grad_x = _bench(lambda: so3_linear_backward_x(w,grad_out,layout))
                ms_grad_x_torch = _bench(
                    lambda: torch_backward_x_reference(w,grad_out,layout))
                print(f'{model} {dtype_name} Z={z}: '
                      f'serial={ms_serial:.6f} ms, '
                      f'path_parallel={ms_parallel:.6f} ms, '
                      f'group_parallel={ms_group:.6f} ms, torch={ms_torch:.6f} ms')
                print(f'speedup_vs_torch: serial={ms_torch/ms_serial:.2f}x, '
                      f'path_parallel={ms_torch/ms_parallel:.2f}x, '
                      f'group_parallel={ms_torch/ms_group:.2f}x; '
                      f'parallel_vs_serial={ms_serial/ms_parallel:.2f}x '
                      f'group_vs_serial={ms_serial/ms_group:.2f}x '
                      f'(wrapper allocation included; path-parallel also includes '
                      f'zeroing/cast; not a cuEq comparison)')
                print(f'backward_x: triton={ms_grad_x:.6f} ms, '
                      f'torch={ms_grad_x_torch:.6f} ms, '
                      f'speedup={ms_grad_x_torch/ms_grad_x:.2f}x')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cpu-test',action='store_true')
    p.add_argument('--gpu-test',action='store_true')
    p.add_argument('--model',choices=('large','medium','small'),default='large')
    p.add_argument('--batch',type=int,default=2208)
    p.add_argument('--dtype',choices=('float16','bfloat16','float32'),default='float32')
    args = p.parse_args()
    if args.cpu_test:
        cpu_test()
    if args.gpu_test:
        if args.batch <= 0:
            p.error('--batch must be positive for the benchmark')
        gpu_test(args.model,args.batch,args.dtype)
    if not args.cpu_test and not args.gpu_test:
        p.print_help()
