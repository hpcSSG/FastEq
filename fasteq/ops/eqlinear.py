"""CUTLASS eqlinear public API: forward, grad_x, precision selection and cache controls.

FP64 tensors use FP64 kernels. FP32 tensors default to TF32x3; pass
``compute_precision="fp32"`` for strict FP32 computation.
"""
from __future__ import annotations

import torch

from .auto_schedule.eqlinear_descriptor import _descriptor_problem
from .auto_schedule.eqlinear_plan import PLAN_CACHE, _create_handle
from .auto_schedule.eqlinear_runtime import _EqLinearFn, _forward, _forward_single_path

__all__ = [
    "fast_equi_linear", "eqlinear_clear_plan_cache",
    "eqlinear_set_plan_cache", "eqlinear_debug_info",
]

_TF32 = 0
_FP32 = 1
_FP64 = 2


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


def fast_equi_linear(descriptor, w, x, *, compute_precision=None):
    """Forward and grad_x for the uniform uv/iu/iv eqlinear operator."""
    entry = PLAN_CACHE.last_entry
    if entry is not None:
        call = entry.get("call")
        if call is not None:
            result = call(descriptor, w, x, compute_precision)
            if result is not None:
                return result
    precision = _resolve_precision(x, w, compute_precision)
    if (type(w) is torch.Tensor and type(x) is torch.Tensor
            and len(getattr(descriptor, "paths", ())) == 1
            and x.is_contiguous() and w.is_contiguous()
            and torch.autograd.forward_ad._current_level < 0
            and not torch._C._are_functorch_transforms_active()):
        return _forward_single_path(w, x, descriptor, precision)
    if (type(w) is torch.Tensor and type(x) is torch.Tensor
            and not w.requires_grad and not x.requires_grad
            and torch.autograd.forward_ad._current_level < 0
            and not torch._C._are_functorch_transforms_active()):
        return _forward(w, x, descriptor, precision)
    return _EqLinearFn.apply(w, x, descriptor, precision)


def eqlinear_clear_plan_cache() -> None:
    """Release cached prepared plans."""
    PLAN_CACHE.clear()


def eqlinear_set_plan_cache(max_bytes=None) -> None:
    """Set the prepared-plan budget in bytes; 0 disables caching."""
    PLAN_CACHE.set_budget(max_bytes)


def eqlinear_debug_info(descriptor, w, x, *, compute_precision=None) -> dict:
    """Return the selected precision/impl for the given call without keeping a cache entry."""
    i_dims, cg, u, v, j_rows = _descriptor_problem(descriptor)
    precision = _resolve_precision(x, w, compute_precision)
    w3d = w.reshape(len(i_dims), u, v).contiguous()
    handle = _create_handle("f", w3d, i_dims, cg, int(x.shape[0]), u, v, precision, j_rows)
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
