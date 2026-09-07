"""Triton implementation compatible with :class:`e3nn.nn.Gate`.

Supported pointwise activations are identity/None, SiLU, sigmoid, and tanh.
Like e3nn.nn.Activation, non-identity activations are normalized with
e3nn.math.normalize2mom before being evaluated by the Triton kernels.

The module implements forward, backward, and double backward.  Metadata is
constructed once in ``__init__`` and registered as buffers, so moving the
module to CUDA also moves all index maps.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

import triton
import triton.language as tl


ACT_ID = 0
ACT_SILU = 1
ACT_SIGMOID = 2
ACT_TANH = 3


def act_code(fn: Optional[Callable]) -> int:
    """Return the Triton activation opcode, rejecting unsupported functions."""
    if fn is None:
        return ACT_ID

    name = f"{getattr(fn, '__name__', '')} {fn.__class__.__name__} {fn}".lower()
    if "silu" in name or "swish" in name:
        return ACT_SILU
    if "sigmoid" in name:
        return ACT_SIGMOID
    if "tanh" in name:
        return ACT_TANH
    if "identity" in name:
        return ACT_ID

    raise NotImplementedError(
        "FastEquivariantGate only supports None/identity, SiLU, sigmoid, "
        f"and tanh; received {fn!r}"
    )


@torch.no_grad()
def _normalized_scale(fn: Optional[Callable]) -> float:
    """Get the same second-moment scale used by e3nn.nn.Activation."""
    if fn is None:
        return 1.0

    from e3nn.math import normalize2mom

    normalized = normalize2mom(fn)
    # Deriving the ratio via evaluation is stable across e3nn versions and
    # avoids relying on a private attribute name of normalize2mom's wrapper.
    probe = torch.tensor([-1.173, -0.371, 0.619, 1.287], dtype=torch.float64)
    try:
        raw = fn(probe)
        norm = normalized(probe)
    except (RuntimeError, TypeError):
        probe = probe.float()
        raw = fn(probe)
        norm = normalized(probe)

    raw = torch.as_tensor(raw, device="cpu").reshape(-1)
    norm = torch.as_tensor(norm, device="cpu").reshape(-1)
    valid = raw.abs() > 1.0e-12
    if not bool(valid.any()):
        raise ValueError(f"Cannot determine e3nn activation normalization for {fn!r}")

    ratios = (norm[valid] / raw[valid]).double()
    scale = ratios[0]
    if not torch.allclose(ratios, scale.expand_as(ratios), rtol=1e-6, atol=1e-8):
        raise ValueError(f"Activation {fn!r} is not a supported pointwise activation")
    return float(scale)


def _physical_copy_starts(combined_blocks, irreps_in):
    """Map every conceptual irrep copy to its offset in ``Gate.irreps_in``.

    e3nn's ``_Sortcut`` sorts the concatenated input irreps and may simplify
    adjacent equal irreps.  Consequently an original ``3x0e`` block and an
    original ``2x0e`` block can appear as one physical ``5x0e`` block.  Mapping
    whole blocks by multiplicity is therefore incorrect; copies are the stable
    unit that survives both sorting and simplification.
    """
    conceptual_copies = []
    copy_starts = {
        (kind, block_id): []
        for kind, block_id, _, _ in combined_blocks
    }
    for kind, block_id, mul, ir in combined_blocks:
        for copy in range(mul):
            conceptual_copies.append((kind, block_id, copy, ir))

    used = [False] * len(conceptual_copies)
    physical_offset = 0

    # Sorting is stable, so first-unused matching also disambiguates equal
    # irreps originating from scalars, gates, and gated blocks.
    for physical_mul, physical_ir in irreps_in:
        for physical_copy in range(physical_mul):
            found = None
            for i, (_, _, _, conceptual_ir) in enumerate(conceptual_copies):
                if not used[i] and conceptual_ir == physical_ir:
                    found = i
                    break
            if found is None:
                raise RuntimeError(
                    "Could not reproduce e3nn Gate input permutation at "
                    f"physical irrep {physical_ir} copy {physical_copy}"
                )

            used[found] = True
            kind, block_id, _, _ = conceptual_copies[found]
            copy_starts[(kind, block_id)].append(
                physical_offset + physical_copy * physical_ir.dim
            )

        physical_offset += physical_mul * physical_ir.dim

    if not all(used):
        missing = [
            (kind, block_id, copy, str(ir))
            for is_used, (kind, block_id, copy, ir) in zip(
                used, conceptual_copies
            )
            if not is_used
        ]
        raise RuntimeError(
            "Incomplete e3nn Gate input permutation; unmatched copies: "
            f"{missing}"
        )
    return copy_starts


@torch.jit.ignore
def build_gate_metadata(
    irreps_scalars,
    act_scalars: Sequence[Optional[Callable]],
    irreps_gates,
    act_gates: Sequence[Optional[Callable]],
    irreps_gated,
):
    """Compile e3nn Gate descriptions into forward/backward Triton maps."""
    from e3nn.nn import Gate
    from e3nn.o3 import Irreps

    irreps_scalars = Irreps(irreps_scalars)
    irreps_gates = Irreps(irreps_gates)
    irreps_gated = Irreps(irreps_gated)
    act_scalars = list(act_scalars)
    act_gates = list(act_gates)

    if len(act_scalars) != len(irreps_scalars):
        raise ValueError("len(act_scalars) must equal len(irreps_scalars)")
    if len(act_gates) != len(irreps_gates):
        raise ValueError("len(act_gates) must equal len(irreps_gates)")

    # Let e3nn perform all parity, multiplicity, and representation checks and
    # use its public irreps_in/irreps_out as the compatibility contract.
    reference = Gate(
        irreps_scalars,
        act_scalars,
        irreps_gates,
        act_gates,
        irreps_gated,
    )
    irreps_in = reference.irreps_in
    irreps_out = reference.irreps_out

    combined = []
    for kind, irreps in (
        ("scalar", irreps_scalars),
        ("gate", irreps_gates),
        ("gated", irreps_gated),
    ):
        for block_id, (mul, ir) in enumerate(irreps):
            combined.append((kind, block_id, mul, ir))

    copy_starts = _physical_copy_starts(combined, irreps_in)

    out_src = []
    out_gate = []
    out_act = []
    out_scale = []

    # Scalar output remains in irreps_scalars order.
    for block_id, ((mul, ir), fn) in enumerate(zip(irreps_scalars, act_scalars)):
        starts = copy_starts[("scalar", block_id)]
        if len(starts) != mul:
            raise RuntimeError("Incorrect scalar-copy metadata")
        code = act_code(fn)
        scale = _normalized_scale(fn)
        for copy in range(mul):
            for m in range(ir.dim):
                out_src.append(starts[copy] + m)
                out_gate.append(-1)
                out_act.append(code)
                out_scale.append(scale)

    # Flatten gate copies in representation order.  Each scalar gate controls
    # one irrep copy, not one magnetic (m) component.
    gate_inputs = []
    gate_codes = []
    gate_scales = []
    for block_id, ((mul, ir), fn) in enumerate(zip(irreps_gates, act_gates)):
        if ir.l != 0:
            raise ValueError(f"Gate irreps must be scalars; received {ir}")
        starts = copy_starts[("gate", block_id)]
        if len(starts) != mul:
            raise RuntimeError("Incorrect gate-copy metadata")
        code = act_code(fn)
        scale = _normalized_scale(fn)
        for copy in range(mul):
            gate_inputs.append(starts[copy])
            gate_codes.append(code)
            gate_scales.append(scale)

    if len(gate_inputs) != irreps_gated.num_irreps:
        raise ValueError(
            "The number of scalar gates must equal irreps_gated.num_irreps: "
            f"{len(gate_inputs)} != {irreps_gated.num_irreps}"
        )

    gate_copy = 0
    for block_id, (mul, ir) in enumerate(irreps_gated):
        starts = copy_starts[("gated", block_id)]
        if len(starts) != mul:
            raise RuntimeError("Incorrect gated-copy metadata")
        # e3nn stores a mul x ir.dim block copy-major:
        # [copy0(m=-l..l), copy1(m=-l..l), ...].
        for copy in range(mul):
            gate_input = gate_inputs[gate_copy]
            code = gate_codes[gate_copy]
            scale = gate_scales[gate_copy]
            for m in range(ir.dim):
                out_src.append(starts[copy] + m)
                out_gate.append(gate_input)
                out_act.append(code)
                out_scale.append(scale)
            gate_copy += 1

    in_dim = irreps_in.dim
    out_dim = irreps_out.dim
    if len(out_src) != out_dim:
        raise RuntimeError(f"Metadata output dim {len(out_src)} != e3nn output dim {out_dim}")

    # Input-centric contribution table.  It gives deterministic reductions in
    # backward and double backward and avoids atomic accumulation at gates.
    # role: 0=scalar, 1=gated source, 2=gate scalar.
    contributions = [[] for _ in range(in_dim)]
    for out, (src, gate) in enumerate(zip(out_src, out_gate)):
        if gate < 0:
            contributions[src].append((out, 0))
        else:
            contributions[src].append((out, 1))
            contributions[gate].append((out, 2))

    max_fan = max((len(v) for v in contributions), default=0)
    max_fan = max(max_fan, 1)
    in_out = torch.full((in_dim, max_fan), -1, dtype=torch.int32)
    in_role = torch.full((in_dim, max_fan), -1, dtype=torch.int8)
    for input_index, items in enumerate(contributions):
        for fan, (out, role) in enumerate(items):
            in_out[input_index, fan] = out
            in_role[input_index, fan] = role

    return {
        "irreps_in": irreps_in,
        "irreps_out": irreps_out,
        "out_src": torch.tensor(out_src, dtype=torch.int32),
        "out_gate": torch.tensor(out_gate, dtype=torch.int32),
        "out_act": torch.tensor(out_act, dtype=torch.int8),
        "out_scale": torch.tensor(out_scale, dtype=torch.float32),
        "in_out": in_out,
        "in_role": in_role,
    }


@triton.jit
def _activation(z, code, scale):
    # Triton >= 3.x forbids ordinary Python globals inside @jit functions.
    act_silu = tl.constexpr(1)
    act_sigmoid = tl.constexpr(2)
    act_tanh = tl.constexpr(3)
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    tanh = 2.0 / (1.0 + tl.exp(-2.0 * z)) - 1.0
    value = z
    value = tl.where(code == act_silu, z * sigmoid, value)
    value = tl.where(code == act_sigmoid, sigmoid, value)
    value = tl.where(code == act_tanh, tanh, value)
    return scale * value


@triton.jit
def _activation_d1(z, code, scale):
    act_silu = tl.constexpr(1)
    act_sigmoid = tl.constexpr(2)
    act_tanh = tl.constexpr(3)
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    ds = sigmoid * (1.0 - sigmoid)
    tanh = 2.0 / (1.0 + tl.exp(-2.0 * z)) - 1.0
    value = tl.full(z.shape, 1.0, z.dtype)
    value = tl.where(code == act_silu, sigmoid + z * ds, value)
    value = tl.where(code == act_sigmoid, ds, value)
    value = tl.where(code == act_tanh, 1.0 - tanh * tanh, value)
    return scale * value


@triton.jit
def _activation_d2(z, code, scale):
    act_silu = tl.constexpr(1)
    act_sigmoid = tl.constexpr(2)
    act_tanh = tl.constexpr(3)
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    ds = sigmoid * (1.0 - sigmoid)
    tanh = 2.0 / (1.0 + tl.exp(-2.0 * z)) - 1.0
    value = tl.zeros(z.shape, z.dtype)
    value = tl.where(code == act_silu, ds * (2.0 + z * (1.0 - 2.0 * sigmoid)), value)
    value = tl.where(code == act_sigmoid, ds * (1.0 - 2.0 * sigmoid), value)
    value = tl.where(code == act_tanh, -2.0 * tanh * (1.0 - tanh * tanh), value)
    return scale * value


@triton.jit
def gate_forward_kernel(
    x_ptr,
    y_ptr,
    src_ptr,
    gate_ptr,
    act_ptr,
    scale_ptr,
    stride_x,
    stride_y,
    OUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < OUT

    src = tl.load(src_ptr + cols, mask=mask, other=0)
    gate = tl.load(gate_ptr + cols, mask=mask, other=-1)
    code = tl.load(act_ptr + cols, mask=mask, other=0).to(tl.int32)
    scale = tl.load(scale_ptr + cols, mask=mask, other=1.0)
    source_value = tl.load(x_ptr + row * stride_x + src, mask=mask, other=0.0)

    is_gated = gate >= 0
    gate_safe = tl.where(is_gated, gate, src)
    z = tl.load(x_ptr + row * stride_x + gate_safe, mask=mask, other=0.0)
    activated = _activation(z, code, scale)
    scalar_value = _activation(source_value, code, scale)
    y = tl.where(is_gated, source_value * activated, scalar_value)
    tl.store(y_ptr + row * stride_y + cols, y, mask=mask)


@triton.jit
def gate_backward_kernel(
    x_ptr,
    gy_ptr,
    dx_ptr,
    src_ptr,
    gate_ptr,
    act_ptr,
    scale_ptr,
    in_out_ptr,
    in_role_ptr,
    stride_x,
    stride_gy,
    stride_dx,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    MAX_FAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    inputs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    input_mask = inputs < IN
    # Seed the accumulator from x so FP64 inputs retain FP64 accumulation;
    # Triton promotes FP16/BF16 arithmetic to its normal FP32 compute type.
    total = tl.load(
        x_ptr + row * stride_x + inputs,
        mask=input_mask,
        other=0.0,
    ) * 0.0

    for fan in tl.static_range(0, MAX_FAN):
        slot = inputs * MAX_FAN + fan
        out = tl.load(in_out_ptr + slot, mask=input_mask, other=-1)
        role = tl.load(in_role_ptr + slot, mask=input_mask, other=-1).to(tl.int32)
        valid = input_mask & (out >= 0)
        out_safe = tl.where(valid, out, 0)

        src = tl.load(src_ptr + out_safe, mask=valid, other=0)
        gate = tl.load(gate_ptr + out_safe, mask=valid, other=-1)
        code = tl.load(act_ptr + out_safe, mask=valid, other=0).to(tl.int32)
        scale = tl.load(scale_ptr + out_safe, mask=valid, other=1.0)
        gy = tl.load(gy_ptr + row * stride_gy + out_safe, mask=valid, other=0.0)

        gate_safe = tl.where(gate >= 0, gate, src)
        z = tl.load(x_ptr + row * stride_x + gate_safe, mask=valid, other=0.0)
        source_value = tl.load(x_ptr + row * stride_x + src, mask=valid, other=0.0)
        a = _activation(z, code, scale)
        d1 = _activation_d1(z, code, scale)

        contribution = gy * d1
        contribution = tl.where(role == 1, gy * a, contribution)
        contribution = tl.where(role == 2, gy * source_value * d1, contribution)
        total += tl.where(valid, contribution, 0.0)

    tl.store(dx_ptr + row * stride_dx + inputs, total, mask=input_mask)


@triton.jit
def gate_double_backward_x_kernel(
    x_ptr,
    gy_ptr,
    u_ptr,
    grad_x_ptr,
    src_ptr,
    gate_ptr,
    act_ptr,
    scale_ptr,
    in_out_ptr,
    in_role_ptr,
    stride_x,
    stride_gy,
    stride_u,
    stride_grad_x,
    IN: tl.constexpr,
    MAX_FAN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    inputs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    input_mask = inputs < IN
    total = tl.load(
        x_ptr + row * stride_x + inputs,
        mask=input_mask,
        other=0.0,
    ) * 0.0

    for fan in tl.static_range(0, MAX_FAN):
        slot = inputs * MAX_FAN + fan
        out = tl.load(in_out_ptr + slot, mask=input_mask, other=-1)
        role = tl.load(in_role_ptr + slot, mask=input_mask, other=-1).to(tl.int32)
        valid = input_mask & (out >= 0)
        out_safe = tl.where(valid, out, 0)

        src = tl.load(src_ptr + out_safe, mask=valid, other=0)
        gate = tl.load(gate_ptr + out_safe, mask=valid, other=-1)
        code = tl.load(act_ptr + out_safe, mask=valid, other=0).to(tl.int32)
        scale = tl.load(scale_ptr + out_safe, mask=valid, other=1.0)
        gy = tl.load(gy_ptr + row * stride_gy + out_safe, mask=valid, other=0.0)
        u_input = tl.load(u_ptr + row * stride_u + inputs, mask=input_mask, other=0.0)

        gate_safe = tl.where(gate >= 0, gate, src)
        z = tl.load(x_ptr + row * stride_x + gate_safe, mask=valid, other=0.0)
        source_value = tl.load(x_ptr + row * stride_x + src, mask=valid, other=0.0)
        u_src = tl.load(u_ptr + row * stride_u + src, mask=valid, other=0.0)
        u_gate = tl.load(u_ptr + row * stride_u + gate_safe, mask=valid, other=0.0)
        d1 = _activation_d1(z, code, scale)
        d2 = _activation_d2(z, code, scale)

        # scalar input; gated source input; gate input, respectively.
        contribution = u_input * gy * d2
        contribution = tl.where(role == 1, u_gate * gy * d1, contribution)
        gate_term = gy * (u_src * d1 + u_gate * source_value * d2)
        contribution = tl.where(role == 2, gate_term, contribution)
        total += tl.where(valid, contribution, 0.0)

    tl.store(grad_x_ptr + row * stride_grad_x + inputs, total, mask=input_mask)


@triton.jit
def gate_double_backward_gy_kernel(
    x_ptr,
    u_ptr,
    grad_gy_ptr,
    src_ptr,
    gate_ptr,
    act_ptr,
    scale_ptr,
    stride_x,
    stride_u,
    stride_grad_gy,
    OUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    outputs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = outputs < OUT

    src = tl.load(src_ptr + outputs, mask=mask, other=0)
    gate = tl.load(gate_ptr + outputs, mask=mask, other=-1)
    code = tl.load(act_ptr + outputs, mask=mask, other=0).to(tl.int32)
    scale = tl.load(scale_ptr + outputs, mask=mask, other=1.0)
    is_gated = gate >= 0
    gate_safe = tl.where(is_gated, gate, src)

    z = tl.load(x_ptr + row * stride_x + gate_safe, mask=mask, other=0.0)
    source_value = tl.load(x_ptr + row * stride_x + src, mask=mask, other=0.0)
    u_src = tl.load(u_ptr + row * stride_u + src, mask=mask, other=0.0)
    u_gate = tl.load(u_ptr + row * stride_u + gate_safe, mask=mask, other=0.0)
    a = _activation(z, code, scale)
    d1 = _activation_d1(z, code, scale)

    scalar = u_src * d1
    gated = u_src * a + u_gate * source_value * d1
    grad_gy = tl.where(is_gated, gated, scalar)
    tl.store(grad_gy_ptr + row * stride_grad_gy + outputs, grad_gy, mask=mask)


def _launch_grid(rows: int, columns: int, block: int):
    return (rows, triton.cdiv(columns, block))


class FastGateBackwardFunction(torch.autograd.Function):
    """Differentiable first backward; its backward is Gate's second derivative."""

    @staticmethod
    def forward(ctx, x, gy, out_src, out_gate, out_act, out_scale, in_out, in_role):
        x2 = x.reshape(-1, x.shape[-1])
        gy2 = gy.contiguous().reshape(-1, gy.shape[-1])
        dx2 = torch.empty_like(x2)
        block = 128

        if x2.numel() != 0:
            gate_backward_kernel[_launch_grid(x2.shape[0], x2.shape[1], block)](
                x2,
                gy2,
                dx2,
                out_src,
                out_gate,
                out_act,
                out_scale,
                in_out,
                in_role,
                x2.stride(0),
                gy2.stride(0),
                dx2.stride(0),
                IN=x2.shape[1],
                OUT=gy2.shape[1],
                MAX_FAN=in_out.shape[1],
                BLOCK=block,
            )

        ctx.save_for_backward(x2, gy2, out_src, out_gate, out_act, out_scale, in_out, in_role)
        ctx.input_shape = x.shape
        ctx.gy_shape = gy.shape
        return dx2.reshape(x.shape)

    @staticmethod
    def backward(ctx, u):
        x, gy, out_src, out_gate, out_act, out_scale, in_out, in_role = ctx.saved_tensors
        u2 = u.contiguous().reshape_as(x)
        grad_x = torch.empty_like(x)
        grad_gy = torch.empty_like(gy)
        block = 128

        if x.numel() != 0:
            gate_double_backward_x_kernel[_launch_grid(x.shape[0], x.shape[1], block)](
                x,
                gy,
                u2,
                grad_x,
                out_src,
                out_gate,
                out_act,
                out_scale,
                in_out,
                in_role,
                x.stride(0),
                gy.stride(0),
                u2.stride(0),
                grad_x.stride(0),
                IN=x.shape[1],
                MAX_FAN=in_out.shape[1],
                BLOCK=block,
            )
            gate_double_backward_gy_kernel[_launch_grid(x.shape[0], gy.shape[1], block)](
                x,
                u2,
                grad_gy,
                out_src,
                out_gate,
                out_act,
                out_scale,
                x.stride(0),
                u2.stride(0),
                grad_gy.stride(0),
                OUT=gy.shape[1],
                BLOCK=block,
            )

        return (
            grad_x.reshape(ctx.input_shape),
            grad_gy.reshape(ctx.gy_shape),
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FastGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_src, out_gate, out_act, out_scale, in_out, in_role):
        x2 = x.reshape(-1, x.shape[-1])
        out_dim = out_src.numel()
        y2 = torch.empty((x2.shape[0], out_dim), device=x.device, dtype=x.dtype)
        block = 128

        if y2.numel() != 0:
            gate_forward_kernel[_launch_grid(x2.shape[0], out_dim, block)](
                x2,
                y2,
                out_src,
                out_gate,
                out_act,
                out_scale,
                x2.stride(0),
                y2.stride(0),
                OUT=out_dim,
                BLOCK=block,
            )

        ctx.save_for_backward(x2, out_src, out_gate, out_act, out_scale, in_out, in_role)
        ctx.input_shape = x.shape
        return y2.reshape(*x.shape[:-1], out_dim)

    @staticmethod
    def backward(ctx, gy):
        x, out_src, out_gate, out_act, out_scale, in_out, in_role = ctx.saved_tensors
        dx = FastGateBackwardFunction.apply(
            x,
            gy,
            out_src,
            out_gate,
            out_act,
            out_scale,
            in_out,
            in_role,
        )
        return dx.reshape(ctx.input_shape), None, None, None, None, None, None


class FastEquivariantGate(nn.Module):
    """Drop-in constructor-compatible replacement for ``e3nn.nn.Gate``."""

    def __init__(
        self,
        irreps_scalars,
        act_scalars,
        irreps_gates,
        act_gates,
        irreps_gated,
    ) -> None:
        super().__init__()
        metadata = build_gate_metadata(
            irreps_scalars,
            act_scalars,
            irreps_gates,
            act_gates,
            irreps_gated,
        )
        self.irreps_in = metadata["irreps_in"]
        self.irreps_out = metadata["irreps_out"]
        self.register_buffer("out_src", metadata["out_src"], persistent=False)
        self.register_buffer("out_gate", metadata["out_gate"], persistent=False)
        self.register_buffer("out_act", metadata["out_act"], persistent=False)
        self.register_buffer("out_scale", metadata["out_scale"], persistent=False)
        self.register_buffer("in_out", metadata["in_out"], persistent=False)
        self.register_buffer("in_role", metadata["in_role"], persistent=False)

    @torch.jit.ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"Expected input last dimension {self.irreps_in.dim}, got {x.shape[-1]}"
            )
        if not x.is_cuda:
            raise RuntimeError("FastEquivariantGate requires a CUDA tensor")
        # The kernels treat all leading dimensions as a flattened batch.
        return FastGateFunction.apply(
            x.contiguous(),
            self.out_src,
            self.out_gate,
            self.out_act,
            self.out_scale,
            self.in_out,
            self.in_role,
        )
