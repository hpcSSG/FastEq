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

import torch

_TF32 = 0
_FP32 = 1
_FP64 = 2

# Prepared-plan LRU.  Values keep a weak reference to w so allocator pointer reuse cannot produce a
# false cache hit; the C++ handle is destroyed when the entry is evicted.
_PLAN_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_PLAN_CACHE_BYTES = 0
_PLAN_CACHE_MAX_BYTES = 2 << 30


def _operand_segments(operand):
    """Convert a cuequivariance operand's segments to immutable int tuples."""
    return tuple(tuple(operator.index(value) for value in segment) for segment in operand)


def _single_dimension(descriptor, name):
    values = tuple(operator.index(value) for value in descriptor.get_dims(name))
    unique = tuple(dict.fromkeys(values))
    if len(unique) != 1:
        raise ValueError(f"eqlinear requires uniform {name}; got {unique}")
    return unique[0]


def _descriptor_problem(descriptor):
    """Translate a ``uv,iu,iv`` descriptor into the kernel's path geometry."""
    if len(descriptor.operands) != 3:
        raise ValueError("eqlinear requires exactly three operands: uv, iu, iv")

    input_segments = _operand_segments(descriptor.operands[1])
    output_segments = _operand_segments(descriptor.operands[2])
    paths = len(descriptor.paths)
    if paths == 0:
        raise ValueError("eqlinear descriptor must contain at least one path")

    if paths <= 4:
        i_dims = [operator.index(v) for v in descriptor.get_dimensions_dict()["i"]]
    else:
        i_dims = [int(segment[0]) for segment in input_segments]

    if len(i_dims) != paths:
        raise ValueError(f"eqlinear: derived {len(i_dims)} i_dims for {paths} paths")
    if sum(i_dims) != sum(int(segment[0]) for segment in input_segments):
        raise ValueError("eqlinear: descriptor i_dims disagree with the input operand segments")

    u = _single_dimension(descriptor, "u")
    v = _single_dimension(descriptor, "v")
    cg = [float(path.coefficients) for path in descriptor.paths]
    j_rows = int(torch.ops.eqlinear.out_rows(i_dims))
    expected_j = sum(int(segment[0]) for segment in output_segments)
    if j_rows != expected_j:
        raise NotImplementedError(
            f"eqlinear: output segments imply {expected_j} rows, kernel grouping implies {j_rows}"
        )
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
    return (
        kind,
        ref_tensor.device.index,
        precision,
        int(shape[0]),
        int(shape[1]),
        int(shape[2]),
        tuple(int(v) for v in i_dims),
        tuple(cg),
        ref_tensor.data_ptr(),
        ref_tensor._version,
        tuple(w3d.shape),
        tuple(w3d.stride()),
        w3d.dtype,
        int(torch.cuda.current_stream(ref_tensor.device).cuda_stream),
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
    global _PLAN_CACHE_BYTES
    if _PLAN_CACHE_MAX_BYTES <= 0:
        return None
    key = _make_key(kind, ref_tensor, w3d, shape, i_dims, cg, precision)
    entry = _PLAN_CACHE.get(key)
    if entry is not None:
        if entry["ref"]() is ref_tensor:
            _PLAN_CACHE.move_to_end(key)
            return entry["handle"]
        # The tensor died and a new one reused the key; drop the stale plan.
        _PLAN_CACHE.pop(key, None)
        _PLAN_CACHE_BYTES -= entry["bytes"]
        _destroy_entry(entry)
    return None


def _cache_insert(kind, handle, ref_tensor, w3d, shape, i_dims, cg, precision, j_rows, v):
    global _PLAN_CACHE_BYTES
    if _PLAN_CACHE_MAX_BYTES <= 0:
        return
    key = _make_key(kind, ref_tensor, w3d, shape, i_dims, cg, precision)
    size = _estimate_bytes(w3d, shape, j_rows, v)
    entry = {"kind": kind, "handle": handle, "ref": weakref.ref(ref_tensor), "bytes": size}
    _PLAN_CACHE[key] = entry
    _PLAN_CACHE_BYTES += size
    while _PLAN_CACHE_BYTES > _PLAN_CACHE_MAX_BYTES and len(_PLAN_CACHE) > 1:
        _, old = _PLAN_CACHE.popitem(last=False)
        _PLAN_CACHE_BYTES -= old["bytes"]
        _destroy_entry(old)


def _forward_handle(w_ref, w3d, x3d, i_dims, cg, precision, j_rows, v):
    shape = (int(x3d.shape[0]), int(x3d.shape[1]), int(x3d.shape[2]))
    handle = _cache_lookup("f", w_ref, w3d, shape, i_dims, cg, precision)
    if handle is not None:
        return handle
    handle = torch.ops.eqlinear.forward_create(
        w3d, i_dims, cg, shape[0], shape[2], len(i_dims), v, precision
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


class _EqLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, descriptor, precision):
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
        return out3d.reshape(batch, j_rows * v)

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
    """Forward and grad_x for the uniform uv/iu/iv equi-linear operator."""
    precision = _resolve_precision(x, w, compute_precision)
    return _EqLinearFn.apply(w, x, descriptor, precision)


def eqlinear_clear_plan_cache() -> None:
    global _PLAN_CACHE_BYTES
    for entry in _PLAN_CACHE.values():
        _destroy_entry(entry)
    _PLAN_CACHE.clear()
    _PLAN_CACHE_BYTES = 0


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
