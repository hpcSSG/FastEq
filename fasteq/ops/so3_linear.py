"""Autograd adapter that dispatches equi-linear forward through SO3 autotune."""

from functools import lru_cache
import operator

import torch

from ..triton.so3_linear import (
    LinearLayout,
    so3_linear_auto,
    so3_linear_backward_x,
)


def _operand_segments(operand):
    """Convert a cuequivariance operand's segments to immutable int pairs."""
    return tuple(tuple(operator.index(value) for value in segment)
                 for segment in operand)


@lru_cache(maxsize=None)
def _build_linear_layout(weight_segments, input_segments, output_segments,
                         paths, coefficients):
    """Cache immutable LinearLayout instances by descriptor contents."""
    return LinearLayout(
        input_segments=input_segments,
        output_segments=output_segments,
        weight_segments=weight_segments,
        paths=paths,
        coefficients=coefficients,
    )


def _descriptor_to_linear_layout(descriptor):
    """Translate a ``uv,iu,iv`` descriptor into the Triton layout."""
    if len(descriptor.operands) != 3:
        raise ValueError('equi-linear requires exactly three operands: uv, iu, iv')

    weight_segments = _operand_segments(descriptor.operands[0])
    input_segments = _operand_segments(descriptor.operands[1])
    output_segments = _operand_segments(descriptor.operands[2])
    paths = tuple(tuple(operator.index(index) for index in path.indices)
                  for path in descriptor.paths)
    coefficients = tuple(float(path.coefficients) for path in descriptor.paths)

    return _build_linear_layout(
        weight_segments,
        input_segments,
        output_segments,
        paths,
        coefficients,
    )


def _single_dimension(descriptor, name):
    values = tuple(operator.index(value) for value in descriptor.get_dims(name))
    unique = tuple(dict.fromkeys(values))
    if len(unique) != 1:
        raise ValueError(f'equi-linear backward requires uniform {name}; got {unique}')
    return unique[0]


class _FastEquiLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, descriptor):
        layout = _descriptor_to_linear_layout(descriptor)
        num_paths = len(descriptor.paths)
        if num_paths == 0:
            raise ValueError('equi-linear descriptor must contain at least one path')

        input_segments = layout.input_segments
        output_segments = layout.output_segments
        i_list = [i for i, _ in input_segments]
        cg_list = [float(path.coefficients) for path in descriptor.paths]
        input_i_total = sum(i for i, _ in input_segments)
        output_i_total = sum(i for i, _ in output_segments)
        u = _single_dimension(descriptor, 'u')
        v = _single_dimension(descriptor, 'v')

        if x.ndim != 2:
            raise ValueError(f'x must be two-dimensional [B, sum(i*u)]; got {tuple(x.shape)}')
        batch = x.shape[0]
        expected_x = input_i_total * u
        expected_w = num_paths * u * v
        if x.shape[1] != expected_x:
            raise ValueError(f'x.shape[1] must be {expected_x}; got {x.shape[1]}')
        if w.numel() != expected_w:
            raise ValueError(f'w must contain {expected_w} elements; got {w.numel()}')
        if layout.sizes != (expected_w, expected_x, output_i_total * v):
            raise ValueError(
                'descriptor storage does not match the expected one-weight-segment-per-path '
                'uniform equi-linear layout')

        # Keep the old three-dimensional weight view for the existing backward,
        # while the new forward consumes flat W and two-dimensional X.
        w_3d = w.reshape(num_paths, u, v).contiguous()
        w_flat = w_3d.reshape(-1)
        x_2d = x.contiguous()
        out = so3_linear_auto(w_flat, x_2d, layout)

        ctx.save_for_backward(w_3d)
        ctx.B = batch
        ctx.I_list = i_list
        ctx.input_i_total = input_i_total
        ctx.output_i_total = output_i_total
        ctx.u = u
        ctx.v = v
        ctx.cg_list = cg_list
        ctx.layout = layout
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (w_3d,) = ctx.saved_tensors
        grad_out_2d = grad_out.reshape(
            ctx.B, ctx.output_i_total * ctx.v).contiguous()
        grad_x = so3_linear_backward_x(
            w_3d.reshape(-1), grad_out_2d, ctx.layout)

        # Inputs to forward are (w, x, descriptor). Weight gradients remain
        # disabled, matching the original adapter.
        return None, grad_x, None


def fast_equi_linear(descriptor, w, x):
    return _FastEquiLinearFn.apply(w, x, descriptor)
