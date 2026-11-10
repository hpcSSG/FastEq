"""CUTLASS-backed eqlinear: forward and grad_x for the uniform uv/iu/iv layout.

The public entry point keeps fasteq's historical signature
``fast_equi_linear(descriptor, w, x)`` and adds the optional, keyword-only
``compute_precision`` switch.  FP64 inputs always use the FP64 kernel; FP32 inputs use
``"tf32"`` (the default) or ``"fp32"``.

Prepared plans are cached by tensor identity/version/shape/stream, so the B-independent
``prepare_weight`` work is hoisted out of the steady-state call.  Activation preparation still
runs on every call because ``x`` changes.
"""

from __future__ import annotations

import operator
import weakref
from collections import OrderedDict
from functools import lru_cache

import torch

try:
    from cuequivariance import SegmentedOperand as _CueOperand
    from cuequivariance import SegmentedTensorProduct as _CueDescriptor
    from cuequivariance.segmented_polynomials import Path as _CuePath
except ImportError:
    # Different cue versions may still implement the public descriptor protocol.
    # Keep full parsing available, but disable structural caching.
    _CueOperand = _CueDescriptor = _CuePath = None

_TF32 = 0
_FP32 = 1
_FP64 = 2

# Prepared-plan LRU.  Values keep a weak reference to w so allocator pointer reuse cannot produce a
# false cache hit; the C++ handle is destroyed when the entry is evicted.
_PLAN_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_PLAN_CACHE_BYTES = 0
_PLAN_CACHE_MAX_BYTES = 2 << 30
_PLAN_CACHE_LAST_ENTRY = None
_GET_CURRENT_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


class _DescriptorGeometryCache:
    """Reuse geometry only for canonical descriptors with unchanged immutable sources."""

    def __init__(self):
        self._entry = None

    @staticmethod
    def _coefficient_signature(paths):
        return tuple(path.coefficients.tobytes() for path in paths)

    @staticmethod
    def _dimension_state(dimensions, path_count):
        # cuEquivariance exposes its cached dict/sets directly. Unknown mutable
        # representations must use the full parser rather than become cache keys.
        if type(dimensions) is not dict:
            return None
        names = ("i", "u", "v") if path_count <= 4 else ("u", "v")
        values = []
        for name in names:
            dims = dimensions.get(name)
            if type(dims) is not set or any(type(value) is not int for value in dims):
                return None
            values.append(tuple(dims))
        return tuple(values)

    @staticmethod
    def _is_supported(descriptor):
        return type(descriptor) is _CueDescriptor

    def get(self, descriptor):
        entry = self._entry
        if entry is None or entry["descriptor"]() is not descriptor:
            return None
        if not self._is_supported(descriptor):
            return None
        operands = descriptor.operands_and_subscripts
        paths = descriptor.paths
        if operands is not entry["operands"] or paths is not entry["paths"]:
            return None
        if descriptor.coefficient_subscripts != entry["coefficient_subscripts"]:
            return None
        for (operand, _), old_segments in zip(operands, entry["segments"]):
            if type(operand) is not _CueOperand or operand.segments is not old_segments:
                return None
        if any(type(path) is not _CuePath for path in paths):
            return None

        # Always retrieve the current object: functools.cache.cache_clear() can
        # replace this dict without changing any descriptor/coefficient fields.
        dimensions = descriptor.get_dimensions_dict()
        dimension_state = self._dimension_state(dimensions, len(paths))
        if dimension_state is None or dimension_state != entry["dimension_state"]:
            # Keep the last validated snapshot while this call takes the full parser.
            # A rejected temporary mutation can then be restored without rebuilding;
            # every lookup still verifies the current dict and its values first.
            return None
        coefficient_signature = self._coefficient_signature(paths)
        if coefficient_signature == entry["coefficient_signature"]:
            if dimensions is not entry["dimensions"]:
                self.clear()
                return None
        else:
            # Coefficients participate in cue's hash, so legal coefficient-only
            # changes produce a fresh dict too. Equal geometry remains reusable.
            entry["dimensions"] = dimensions
            entry["coefficient_signature"] = coefficient_signature
        return entry["geometry"]

    def put(self, descriptor, geometry):
        self.clear()
        if not self._is_supported(descriptor):
            return
        operands = descriptor.operands_and_subscripts
        paths = descriptor.paths
        if type(operands) is not tuple or type(paths) is not tuple:
            return
        if any(type(path) is not _CuePath
               or not hasattr(path.coefficients, "tobytes") for path in paths):
            return
        segments = []
        for pair in operands:
            if type(pair) is not tuple or len(pair) != 2:
                return
            operand, subscripts = pair
            if (type(operand) is not _CueOperand or type(subscripts) is not str
                    or type(operand.segments) is not tuple
                    or any(type(segment) is not tuple
                           or any(type(value) is not int for value in segment)
                           for segment in operand.segments)):
                return
            segments.append(operand.segments)
        dimensions = descriptor.get_dimensions_dict()
        dimension_state = self._dimension_state(dimensions, len(paths))
        if dimension_state is None:
            return
        try:
            descriptor_ref = weakref.ref(descriptor, self._release)
        except TypeError:
            return
        self._entry = {"descriptor": descriptor_ref, "operands": operands, "paths": paths,
                       "segments": tuple(segments), "geometry": geometry,
                       "dimensions": dimensions, "dimension_state": dimension_state,
                       "coefficient_signature": self._coefficient_signature(paths),
                       "coefficient_subscripts": descriptor.coefficient_subscripts}

    def clear(self):
        self._entry = None

    def _release(self, descriptor_ref):
        if self._entry is not None and self._entry["descriptor"] is descriptor_ref:
            self._entry = None


_DESCRIPTOR_GEOMETRY_CACHE = _DescriptorGeometryCache()


def _current_stream_ptr(device):
    if _GET_CURRENT_RAW_STREAM is None or device.index is None:
        return int(torch.cuda.current_stream(device).cuda_stream)
    return int(_GET_CURRENT_RAW_STREAM(device.index))


def _operand_segments(operand):
    """Convert a cuequivariance operand's segments to immutable int tuples."""
    segments = getattr(operand, "segments", operand)
    if type(segments) is tuple and all(
            type(segment) is tuple and all(type(value) is int for value in segment)
            for segment in segments):
        return segments
    return tuple(tuple(operator.index(value) for value in segment) for segment in operand)


def _single_dimension(descriptor, name):
    first = None
    found = False
    unique = None
    for raw_value in descriptor.get_dims(name):
        value = operator.index(raw_value)
        if not found:
            first = value
            found = True
        elif value != first:
            if unique is None:
                unique = [first]
            if value not in unique:
                unique.append(value)
    if unique is not None:
        raise ValueError(f"eqlinear requires uniform {name}; got {tuple(unique)}")
    if not found:
        raise ValueError(f"eqlinear requires uniform {name}; got ()")
    return first


@lru_cache(maxsize=1024)
def _cached_out_rows(i_dims):
    """Cache output grouping by immutable dimensions, not mutable descriptors."""
    return int(torch.ops.eqlinear.out_rows(i_dims))


def _descriptor_geometry_uncached(descriptor):
    """Validate and translate a ``uv,iu,iv`` descriptor's geometry."""
    if len(descriptor.operands) != 3:
        raise ValueError("eqlinear requires exactly three operands: uv, iu, iv")

    input_segments = _operand_segments(descriptor.operands[1])
    output_segments = _operand_segments(descriptor.operands[2])
    paths = len(descriptor.paths)
    if paths == 0:
        raise ValueError("eqlinear descriptor must contain at least one path")

    if paths <= 4:
        i_dims = tuple(operator.index(v) for v in descriptor.get_dimensions_dict()["i"])
    else:
        i_dims = tuple(segment[0] for segment in input_segments)

    if len(i_dims) != paths:
        raise ValueError(f"eqlinear: derived {len(i_dims)} i_dims for {paths} paths")
    if sum(i_dims) != sum(int(segment[0]) for segment in input_segments):
        raise ValueError("eqlinear: descriptor i_dims disagree with the input operand segments")

    u = _single_dimension(descriptor, "u")
    v = _single_dimension(descriptor, "v")
    j_rows = (_cached_out_rows(i_dims) if len(i_dims) <= 32
              else int(torch.ops.eqlinear.out_rows(i_dims)))
    expected_j = sum(segment[0] for segment in output_segments)
    if j_rows != expected_j:
        raise NotImplementedError(
            f"eqlinear: output segments imply {expected_j} rows, kernel grouping implies {j_rows}"
        )
    return i_dims, u, v, j_rows


def _descriptor_problem(descriptor):
    """Translate a descriptor, reusing geometry only while its immutable sources match."""
    geometry = _DESCRIPTOR_GEOMETRY_CACHE.get(descriptor)
    if geometry is None:
        geometry = _descriptor_geometry_uncached(descriptor)
        _DESCRIPTOR_GEOMETRY_CACHE.put(descriptor, geometry)
    i_dims, u, v, j_rows = geometry
    cg = tuple(float(path.coefficients) for path in descriptor.paths)
    return i_dims, cg, u, v, j_rows


def _resolve_precision(x, w, compute_precision):
    if x.dtype != w.dtype:
        raise ValueError(f"eqlinear: x and w must have the same dtype; got {x.dtype} and {w.dtype}")
    if x.dtype == torch.float64:
        return _FP64
    if x.dtype != torch.float32:
        raise ValueError(f"eqlinear: only float32 and float64 are supported; got {x.dtype}")

    if compute_precision is None:
        return _TF32
    value = compute_precision.lower() if isinstance(compute_precision, str) else compute_precision
    if value in ("tf32", "tf32x3"):
        return _TF32
    if value == "fp32":
        return _FP32
    raise ValueError(
        f"eqlinear: compute_precision must be None, 'tf32', or 'fp32'; got {compute_precision!r}"
    )


def _make_key(kind, ref_tensor, w3d, shape, i_dims, cg, precision):
    """Identify prepared weights by their original tensor layout and version."""
    return _make_key_view(
        kind, ref_tensor, tuple(ref_tensor.shape), tuple(ref_tensor.stride()), ref_tensor.dtype,
        shape, i_dims, cg, precision
    )


def _make_key_view(kind, ref_tensor, w_shape, w_stride, w_dtype, shape, i_dims, cg, precision,
                   stream=None):
    if stream is None:
        stream = _current_stream_ptr(ref_tensor.device)
    return (
        kind,
        ref_tensor.device.index,
        precision,
        int(shape[0]),
        int(shape[1]),
        int(shape[2]),
        tuple(i_dims),
        tuple(cg),
        ref_tensor.data_ptr(),
        ref_tensor._version,
        tuple(w_shape),
        tuple(w_stride),
        w_dtype,
        stream,
    )


def _cache_entry_matches(entry, kind, ref_tensor, w_shape, w_stride, w_dtype, shape, i_dims,
                         cg, precision, stream):
    key = entry["key"]
    return (
        entry["ref"]() is ref_tensor
        and key[0] == kind
        and key[1] == ref_tensor.device.index
        and key[2] == precision
        and key[3] == int(shape[0])
        and key[4] == int(shape[1])
        and key[5] == int(shape[2])
        and key[6] == tuple(i_dims)
        and key[7] == tuple(cg)
        and key[8] == ref_tensor.data_ptr()
        and key[9] == ref_tensor._version
        and key[10] == tuple(w_shape)
        and key[11] == tuple(w_stride)
        and key[12] == w_dtype
        and key[13] == stream
    )


def _destroy_entry(entry):
    handle = entry["handle"]
    if entry["kind"] == "f":
        torch.ops.eqlinear.forward_destroy(handle)
    else:
        torch.ops.eqlinear.dgrad_destroy(handle)


def _estimate_bytes(w3d, shape, j_rows, v):
    elements = int(shape[0]) * int(shape[1]) * int(shape[2]) + int(w3d.numel())
    elements += int(shape[0]) * j_rows * v
    return elements * w3d.element_size() * 2


def _cache_lookup(kind, ref_tensor, w3d, shape, i_dims, cg, precision):
    return _cache_lookup_view(
        kind, ref_tensor, tuple(ref_tensor.shape), tuple(ref_tensor.stride()), ref_tensor.dtype,
        shape, i_dims, cg, precision
    )


def _cache_lookup_view(kind, ref_tensor, w_shape, w_stride, w_dtype, shape, i_dims, cg, precision):
    global _PLAN_CACHE_BYTES, _PLAN_CACHE_LAST_ENTRY
    if _PLAN_CACHE_MAX_BYTES <= 0:
        return None
    stream = _current_stream_ptr(ref_tensor.device)
    entry = _PLAN_CACHE_LAST_ENTRY
    if entry is not None and _cache_entry_matches(
            entry, kind, ref_tensor, w_shape, w_stride, w_dtype, shape, i_dims, cg,
            precision, stream):
        return entry["handle"]
    _PLAN_CACHE_LAST_ENTRY = None
    key = _make_key_view(kind, ref_tensor, w_shape, w_stride, w_dtype, shape, i_dims, cg,
                         precision, stream)
    entry = _PLAN_CACHE.get(key)
    if entry is not None:
        if entry["ref"]() is ref_tensor:
            _PLAN_CACHE.move_to_end(key)
            _PLAN_CACHE_LAST_ENTRY = entry
            return entry["handle"]
        # The tensor died and a new one reused the key; drop the stale plan.
        _PLAN_CACHE.pop(key, None)
        _PLAN_CACHE_BYTES -= entry["bytes"]
        if _PLAN_CACHE_LAST_ENTRY is entry:
            _PLAN_CACHE_LAST_ENTRY = None
        _destroy_entry(entry)
    return None


def _cache_insert(kind, handle, ref_tensor, w3d, shape, i_dims, cg, precision, j_rows, v):
    global _PLAN_CACHE_BYTES, _PLAN_CACHE_LAST_ENTRY
    if _PLAN_CACHE_MAX_BYTES <= 0:
        return
    key = _make_key(kind, ref_tensor, w3d, shape, i_dims, cg, precision)
    size = _estimate_bytes(w3d, shape, j_rows, v)
    entry = {"kind": kind, "handle": handle, "ref": weakref.ref(ref_tensor), "bytes": size,
             "key": key}
    _PLAN_CACHE[key] = entry
    _PLAN_CACHE.move_to_end(key)
    _PLAN_CACHE_BYTES += size
    while _PLAN_CACHE_BYTES > _PLAN_CACHE_MAX_BYTES and len(_PLAN_CACHE) > 1:
        _, old = _PLAN_CACHE.popitem(last=False)
        _PLAN_CACHE_BYTES -= old["bytes"]
        if _PLAN_CACHE_LAST_ENTRY is old:
            _PLAN_CACHE_LAST_ENTRY = None
        _destroy_entry(old)
    _PLAN_CACHE_LAST_ENTRY = entry


def _forward_handle(w_ref, w3d, x3d, i_dims, cg, precision, j_rows, v):
    shape = (int(x3d.shape[0]), int(x3d.shape[1]), int(x3d.shape[2]))
    return _forward_handle_shape(w_ref, w3d, shape, i_dims, cg, precision, j_rows, v)


def _forward_handle_shape(w_ref, w3d, shape, i_dims, cg, precision, j_rows, v):
    handle = _cache_lookup("f", w_ref, w3d, shape, i_dims, cg, precision)
    if handle is not None:
        return handle
    handle = torch.ops.eqlinear.forward_create(
        w3d, i_dims, cg, shape[0], shape[2], len(i_dims), v, precision
    )
    _cache_insert("f", handle, w_ref, w3d, shape, i_dims, cg, precision, j_rows, v)
    return handle


def _forward_handle_flat(w_ref, shape, i_dims, cg, precision, j_rows, v):
    u = int(shape[2])
    w_shape = (len(i_dims), u, v)
    handle = _cache_lookup_view(
        "f", w_ref, tuple(w_ref.shape), tuple(w_ref.stride()), w_ref.dtype,
        shape, i_dims, cg, precision
    )
    if handle is not None:
        return handle
    w3d = w_ref.reshape(*w_shape).contiguous()
    handle = torch.ops.eqlinear.forward_create(
        w3d, i_dims, cg, shape[0], u, len(i_dims), v, precision
    )
    _cache_insert("f", handle, w_ref, w3d, shape, i_dims, cg, precision, j_rows, v)
    return handle


def _dgrad_handle(w_ref, w3d, shape, i_dims, cg, precision, j_rows, v):
    handle = _cache_lookup("d", w_ref, w3d, shape, i_dims, cg, precision)
    if handle is not None:
        return handle
    handle = torch.ops.eqlinear.dgrad_create(
        w3d, i_dims, cg, shape[0], shape[2], len(i_dims), v, precision
    )
    _cache_insert("d", handle, w_ref, w3d, shape, i_dims, cg, precision, j_rows, v)
    return handle


def _forward(w, x, descriptor, precision, ctx=None):
    i_dims, cg, u, v, j_rows = _descriptor_problem(descriptor)
    batch = x.shape[0]
    if x.ndim != 2:
        raise ValueError(f"eqlinear: x must be [B, sum(i*u)]; got {tuple(x.shape)}")
    if x.shape[1] != sum(i_dims) * u:
        raise ValueError(f"eqlinear: x.shape[1] must be {sum(i_dims) * u}; got {x.shape[1]}")
    if w.numel() != len(i_dims) * u * v:
        raise ValueError(
            f"eqlinear: w must contain {len(i_dims) * u * v} elements; got {w.numel()}"
        )

    flat_path = (_PLAN_CACHE_MAX_BYTES > 0 and type(x) is torch.Tensor
                 and x.is_contiguous() and w.is_contiguous())
    if flat_path and ctx is None:
        shape = (int(batch), sum(i_dims), u)
        out = torch.empty((batch, j_rows * v), dtype=x.dtype, device=x.device)
        handle = _forward_handle_flat(w, shape, i_dims, cg, precision, j_rows, v)
        torch.ops.eqlinear.forward_run_flat(handle, x, out)
    else:
        w3d = w.reshape(len(i_dims), u, v).contiguous()
        x3d = x.reshape(batch, sum(i_dims), u).contiguous()
        out3d = torch.empty((batch, j_rows, v), dtype=x.dtype, device=x.device)
        if _PLAN_CACHE_MAX_BYTES > 0:
            handle = _forward_handle(w, w3d, x3d, i_dims, cg, precision, j_rows, v)
            torch.ops.eqlinear.forward_run(handle, x3d, out3d)
        else:
            out3d = torch.ops.eqlinear.forward(x3d, w3d, i_dims, cg, precision)
        if out3d.shape != (batch, j_rows, v):
            raise RuntimeError(f"eqlinear: kernel returned {tuple(out3d.shape)}, expected {(batch, j_rows, v)}")
        out = out3d.reshape(batch, j_rows * v)

    if ctx is not None:
        ctx.save_for_backward(w3d)
        ctx.w_ref = w
        ctx.i_dims = i_dims
        ctx.cg = cg
        ctx.precision = precision
        ctx.batch = batch
        ctx.i_total = sum(i_dims)
        ctx.j_rows = j_rows
        ctx.u = u
        ctx.v = v
    return out


class _EqLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, descriptor, precision):
        return _forward(w, x, descriptor, precision, ctx)

    @staticmethod
    def backward(ctx, grad_out):
        (w3d,) = ctx.saved_tensors
        grad3d = grad_out.reshape(ctx.batch, ctx.j_rows, ctx.v).contiguous()
        dx3d = torch.empty((ctx.batch, ctx.i_total, ctx.u), dtype=grad3d.dtype, device=grad3d.device)
        if _PLAN_CACHE_MAX_BYTES > 0:
            handle = _dgrad_handle(
                ctx.w_ref, w3d, (ctx.batch, ctx.i_total, ctx.u), ctx.i_dims, ctx.cg,
                ctx.precision, ctx.j_rows, ctx.v
            )
            torch.ops.eqlinear.dgrad_run(handle, grad3d, dx3d)
        else:
            dx3d = torch.ops.eqlinear.dgrad(grad3d, w3d, ctx.i_dims, ctx.cg, ctx.precision)
        return None, dx3d.reshape(ctx.batch, ctx.i_total * ctx.u), None, None


def fast_equi_linear(descriptor, w, x, *, compute_precision=None):
    """Forward and grad_x for the uniform uv/iu/iv eqlinear operator."""
    precision = _resolve_precision(x, w, compute_precision)
    if (type(w) is torch.Tensor and type(x) is torch.Tensor
            and not w.requires_grad and not x.requires_grad
            and torch.autograd.forward_ad._current_level < 0
            and not torch._C._are_functorch_transforms_active()):
        return _forward(w, x, descriptor, precision)
    return _EqLinearFn.apply(w, x, descriptor, precision)


def eqlinear_clear_plan_cache() -> None:
    global _PLAN_CACHE_BYTES, _PLAN_CACHE_LAST_ENTRY
    for entry in _PLAN_CACHE.values():
        _destroy_entry(entry)
    _PLAN_CACHE.clear()
    _PLAN_CACHE_BYTES = 0
    _PLAN_CACHE_LAST_ENTRY = None


def eqlinear_set_plan_cache(max_bytes=None) -> None:
    """Set the prepared-plan budget in bytes; 0 disables caching."""
    global _PLAN_CACHE_MAX_BYTES
    eqlinear_clear_plan_cache()
    _PLAN_CACHE_MAX_BYTES = 2 << 30 if max_bytes is None else max(0, int(max_bytes))


def eqlinear_debug_info(descriptor, w, x, *, compute_precision=None) -> dict:
    """Return the selected precision/impl for the given call without keeping a cache entry."""
    i_dims, cg, u, v, j_rows = _descriptor_problem(descriptor)
    precision = _resolve_precision(x, w, compute_precision)
    w3d = w.reshape(len(i_dims), u, v).contiguous()
    x3d = x.reshape(x.shape[0], sum(i_dims), u).contiguous()
    handle = torch.ops.eqlinear.forward_create(
        w3d, i_dims, cg, int(x3d.shape[0]), u, len(i_dims), v, precision
    )
    try:
        info = torch.ops.eqlinear.forward_plan_info(handle)
    finally:
        torch.ops.eqlinear.forward_destroy(handle)
    return {
        "precision": int(info[0]),
        "forward_impl": int(info[1]),
        "a_view": int(info[2]),
        "j_rows": int(info[3]),
    }
