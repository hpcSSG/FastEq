"""Tensor layout routing and autograd execution for eqlinear."""
from __future__ import annotations

import torch

from .eqlinear_descriptor import _descriptor_problem, _input_rows
from .eqlinear_plan import PLAN_CACHE, _create_handle


def _validate_flat_problem(w, x, i_dims, u, v):
    if x.ndim != 2:
        raise ValueError(f"eqlinear: x must be [B, sum(i*u)]; got {tuple(x.shape)}")
    i_total = _input_rows(i_dims)
    if x.shape[1] != i_total * u:
        raise ValueError(f"eqlinear: x.shape[1] must be {i_total * u}; got {x.shape[1]}")
    if w.numel() != len(i_dims) * u * v:
        raise ValueError(
            f"eqlinear: w must contain {len(i_dims) * u * v} elements; got {w.numel()}"
        )
    return (int(x.shape[0]), i_total, u)


def _forward_single_path(w, x, descriptor, precision):
    i_dims, cg, u, v, j_rows = _descriptor_problem(descriptor)
    shape = _validate_flat_problem(w, x, i_dims, u, v)
    if PLAN_CACHE.max_bytes > 0:
        handle = PLAN_CACHE.forward_handle_flat(w, shape, i_dims, cg, precision, j_rows, v)
        PLAN_CACHE.bind_host_frontend(handle, descriptor, w)
        return torch.ops.eqlinear._forward_allocate(handle, x, w)
    w3d = w.reshape(1, u, v).contiguous()
    handle = _create_handle("f", w3d, i_dims, cg, shape[0], u, v, precision, j_rows)
    try:
        return torch.ops.eqlinear._forward_allocate(handle, x, w)
    finally:
        # The C++ autograd node holds a shared owner until its graph is released.
        torch.ops.eqlinear.forward_destroy(handle)


def _forward(w, x, descriptor, precision, ctx=None):
    i_dims, cg, u, v, j_rows = _descriptor_problem(descriptor)
    batch, i_total, _ = _validate_flat_problem(w, x, i_dims, u, v)

    flat_path = (PLAN_CACHE.max_bytes > 0 and type(x) is torch.Tensor
                 and x.is_contiguous() and w.is_contiguous())
    if flat_path and ctx is None:
        shape = (int(batch), i_total, u)
        out = torch.empty((batch, j_rows * v), dtype=x.dtype, device=x.device)
        handle = PLAN_CACHE.forward_handle_flat(w, shape, i_dims, cg, precision, j_rows, v)
        torch.ops.eqlinear.forward_run_flat(handle, x, out)
    else:
        w3d = w.reshape(len(i_dims), u, v).contiguous()
        x3d = x.reshape(batch, i_total, u).contiguous()
        out3d = torch.empty((batch, j_rows, v), dtype=x.dtype, device=x.device)
        if PLAN_CACHE.max_bytes > 0:
            handle = PLAN_CACHE.forward_handle(w, w3d, x3d, i_dims, cg, precision, j_rows, v)
            torch.ops.eqlinear.forward_run(handle, x3d, out3d)
        else:
            handle = _create_handle("f", w3d, i_dims, cg, int(batch), u, v, precision, j_rows)
            try:
                torch.ops.eqlinear.forward_run(handle, x3d, out3d)
            finally:
                torch.ops.eqlinear.forward_destroy(handle)
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
        ctx.i_total = i_total
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
        if PLAN_CACHE.max_bytes > 0:
            handle = PLAN_CACHE.dgrad_handle(
                ctx.w_ref, w3d, (ctx.batch, ctx.i_total, ctx.u), ctx.i_dims, ctx.cg,
                ctx.precision, ctx.j_rows, ctx.v
            )
            torch.ops.eqlinear.dgrad_run(handle, grad3d, dx3d)
        else:
            handle = _create_handle("d", w3d, ctx.i_dims, ctx.cg, int(ctx.batch), ctx.u,
                                    ctx.v, ctx.precision, ctx.j_rows)
            try:
                torch.ops.eqlinear.dgrad_run(handle, grad3d, dx3d)
            finally:
                torch.ops.eqlinear.dgrad_destroy(handle)
        return None, dx3d.reshape(ctx.batch, ctx.i_total * ctx.u), None, None
