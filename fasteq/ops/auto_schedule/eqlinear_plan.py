"""Prepared-plan ownership, cache lookup and optional native host dispatch for eqlinear."""
from __future__ import annotations

import importlib
import sys
import weakref
from collections import OrderedDict

import torch

from .eqlinear_descriptor import (
    _PathDimensions, _geometry_key, _DESCRIPTOR_GEOMETRY_CACHE,
    _CueOperand, _CuePath, _CoefficientArray,
)

_GET_CURRENT_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _current_stream_ptr(device):
    if _GET_CURRENT_RAW_STREAM is None or device.index is None:
        return int(torch.cuda.current_stream(device).cuda_stream)
    return int(_GET_CURRENT_RAW_STREAM(device.index))


def _create_handle(kind, w3d, dims, cg, batch, u, v, precision, j_rows):
    if isinstance(dims, _PathDimensions):
        create = (torch.ops.eqlinear.forward_create_geometry if kind == "f"
                  else torch.ops.eqlinear.dgrad_create_geometry)
        return create(w3d, dims, cg, batch, u, len(dims), v, precision,
                      dims.input_rows, j_rows, dims.mapping)
    create = torch.ops.eqlinear.forward_create if kind == "f" else torch.ops.eqlinear.dgrad_create
    return create(w3d, dims, cg, batch, u, len(dims), v, precision)


class PlanCache:
    """Own prepared plans, LRU accounting and guarded host calls as one lifetime."""

    def __init__(self):
        self.entries = OrderedDict()
        self.bytes = 0
        self.max_bytes = 2 << 30
        self.last_entry = None
        self.host_factory = None
        self.host_probed = False

    def install_host_frontend(self, factory):
        """Optional parent loader hook; LRU entries remain the only callable index."""
        self.host_factory = factory
        self.host_probed = True
        for entry in self.entries.values():
            entry["call"] = None

    def bind_host_frontend(self, handle, descriptor, w):
        if (not self.host_probed and self.host_factory is None
                and "fasteq.cuda" in sys.modules):
            self.host_probed = True
            try:
                frontend = importlib.import_module("fasteq.cuda._cuda")
            except ImportError:
                # Older DSOs only register torch.ops and have no optional PyInit.
                pass
            else:
                self.install_host_frontend(frontend.make_guarded_call)
        if self.host_factory is None:
            return
        entry = self.last_entry
        snapshot = _DESCRIPTOR_GEOMETRY_CACHE._entry
        if (entry is not None and entry["handle"] == handle and entry["ref"]() is w
                and snapshot is not None and snapshot["descriptor"]() is descriptor
                and len(snapshot["paths"]) == 1):
            entry["call"] = self.host_factory(
                handle, descriptor, snapshot, w, _CueOperand, _CuePath, _CoefficientArray)

    def _make_key(self, kind, ref_tensor, w3d, shape, i_dims, cg, precision):
        """Identify prepared weights by their original tensor layout and version."""
        return self._make_key_view(
            kind, ref_tensor, tuple(ref_tensor.shape), tuple(ref_tensor.stride()), ref_tensor.dtype,
            shape, i_dims, cg, precision
        )

    def _make_key_view(self, kind, ref_tensor, w_shape, w_stride, w_dtype, shape, i_dims, cg, precision,
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
            _geometry_key(i_dims),
            tuple(cg),
            ref_tensor.data_ptr(),
            ref_tensor._version,
            tuple(w_shape),
            tuple(w_stride),
            w_dtype,
            stream,
        )

    def _cache_entry_matches(self, entry, kind, ref_tensor, w_shape, w_stride, w_dtype, shape, i_dims,
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
            and key[6] == _geometry_key(i_dims)
            and key[7] == tuple(cg)
            and key[8] == ref_tensor.data_ptr()
            and key[9] == ref_tensor._version
            and key[10] == tuple(w_shape)
            and key[11] == tuple(w_stride)
            and key[12] == w_dtype
            and key[13] == stream
        )

    def _destroy_entry(self, entry):
        handle = entry["handle"]
        if entry["kind"] == "f":
            torch.ops.eqlinear.forward_destroy(handle)
        else:
            torch.ops.eqlinear.dgrad_destroy(handle)

    def _estimate_bytes(self, w3d, shape, j_rows, v):
        elements = int(shape[0]) * int(shape[1]) * int(shape[2]) + int(w3d.numel())
        elements += int(shape[0]) * j_rows * v
        return elements * w3d.element_size() * 2

    def _cache_lookup(self, kind, ref_tensor, w3d, shape, i_dims, cg, precision):
        return self._cache_lookup_view(
            kind, ref_tensor, tuple(ref_tensor.shape), tuple(ref_tensor.stride()), ref_tensor.dtype,
            shape, i_dims, cg, precision
        )

    def _cache_lookup_view(self, kind, ref_tensor, w_shape, w_stride, w_dtype, shape, i_dims, cg, precision):
        if self.max_bytes <= 0:
            return None
        stream = _current_stream_ptr(ref_tensor.device)
        entry = self.last_entry
        if entry is not None and self._cache_entry_matches(
                entry, kind, ref_tensor, w_shape, w_stride, w_dtype, shape, i_dims, cg,
                precision, stream):
            return entry["handle"]
        self.last_entry = None
        key = self._make_key_view(kind, ref_tensor, w_shape, w_stride, w_dtype, shape, i_dims, cg,
                             precision, stream)
        entry = self.entries.get(key)
        if entry is not None:
            if entry["ref"]() is ref_tensor:
                self.entries.move_to_end(key)
                self.last_entry = entry
                return entry["handle"]
            # The tensor died and a new one reused the key; drop the stale plan.
            self.entries.pop(key, None)
            self.bytes -= entry["bytes"]
            if self.last_entry is entry:
                self.last_entry = None
            self._destroy_entry(entry)
        return None

    def _cache_insert(self, kind, handle, ref_tensor, w3d, shape, i_dims, cg, precision, j_rows, v):
        if self.max_bytes <= 0:
            return
        key = self._make_key(kind, ref_tensor, w3d, shape, i_dims, cg, precision)
        size = self._estimate_bytes(w3d, shape, j_rows, v)
        if kind == "f" and len(i_dims) == 1:
            # Reserve the lazy C++ dgrad and its immutable weight snapshot up front.
            # It shares the forward entry's budget/eviction, with no new cache index.
            size = size * 2 + w3d.numel() * w3d.element_size()
        entry = {"kind": kind, "handle": handle, "ref": weakref.ref(ref_tensor), "bytes": size,
                 "key": key}
        self.entries[key] = entry
        self.entries.move_to_end(key)
        self.bytes += size
        while self.bytes > self.max_bytes and len(self.entries) > 1:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old["bytes"]
            if self.last_entry is old:
                self.last_entry = None
            self._destroy_entry(old)
        self.last_entry = entry

    def forward_handle(self, w_ref, w3d, x3d, i_dims, cg, precision, j_rows, v):
        shape = (int(x3d.shape[0]), int(x3d.shape[1]), int(x3d.shape[2]))
        return self._forward_handle_shape(w_ref, w3d, shape, i_dims, cg, precision, j_rows, v)

    def _forward_handle_shape(self, w_ref, w3d, shape, i_dims, cg, precision, j_rows, v):
        handle = self._cache_lookup("f", w_ref, w3d, shape, i_dims, cg, precision)
        if handle is not None:
            return handle
        handle = _create_handle("f", w3d, i_dims, cg, shape[0], shape[2], v, precision, j_rows)
        self._cache_insert("f", handle, w_ref, w3d, shape, i_dims, cg, precision, j_rows, v)
        return handle

    def forward_handle_flat(self, w_ref, shape, i_dims, cg, precision, j_rows, v):
        u = int(shape[2])
        w_shape = (len(i_dims), u, v)
        handle = self._cache_lookup_view(
            "f", w_ref, tuple(w_ref.shape), tuple(w_ref.stride()), w_ref.dtype,
            shape, i_dims, cg, precision
        )
        if handle is not None:
            return handle
        w3d = w_ref.reshape(*w_shape).contiguous()
        handle = _create_handle("f", w3d, i_dims, cg, shape[0], u, v, precision, j_rows)
        self._cache_insert("f", handle, w_ref, w3d, shape, i_dims, cg, precision, j_rows, v)
        return handle

    def dgrad_handle(self, w_ref, w3d, shape, i_dims, cg, precision, j_rows, v):
        handle = self._cache_lookup("d", w_ref, w3d, shape, i_dims, cg, precision)
        if handle is not None:
            return handle
        handle = _create_handle("d", w3d, i_dims, cg, shape[0], shape[2], v, precision, j_rows)
        self._cache_insert("d", handle, w_ref, w3d, shape, i_dims, cg, precision, j_rows, v)
        return handle

    def clear(self) -> None:
        for entry in self.entries.values():
            self._destroy_entry(entry)
        self.entries.clear()
        self.bytes = 0
        self.last_entry = None

    def set_budget(self, max_bytes=None) -> None:
        """Set the prepared-plan budget in bytes; 0 disables caching."""
        self.clear()
        self.max_bytes = 2 << 30 if max_bytes is None else max(0, int(max_bytes))


PLAN_CACHE = PlanCache()
