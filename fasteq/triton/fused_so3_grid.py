import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


# ============================================================
# PyTorch Reference Implementation
"""
# 4. Sigmoid for gating
gate_scalars = self.gate_act(gate_scalars)          # [N, 1, num_channels]

# 5. Project to S2 grid signals, perform nonlinear gating, perform elementwise multiplication, 
#    optionally perform dropout, and project back
x_grid = self.so3_grid.to_grid(inputs)

x_grid_1, x_grid_2 = torch.chunk(x_grid, chunks=2, dim=-1)
#x_grid_1 = x_grid_1 * gate_scalars
x_grid = x_grid_1 * x_grid_2
x_grid = self.grid_drop(x_grid)

output_vectors = self.so3_grid.from_grid(x_grid)

output_vectors = output_vectors * gate_scalars 
"""
# ============================================================

@triton.jit
def _project_input_pair(
    X,
    T,
    n,
    c,
    a,
    mask_c,
    mask_a,
    J: tl.constexpr,
    C: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    u = tl.zeros((BLOCK_A, BLOCK_C), tl.float32)
    v = tl.zeros((BLOCK_A, BLOCK_C), tl.float32)

    base = X + n * J * (2 * C)

    # x_grid = torch.einsum("aj,njc->nac", T, inputs)
    # u, v = torch.chunk(x_grid, 2, dim=-1)
    #
    for p in tl.static_range(0, J):
        t = tl.load(
            T + a * J + p,
            mask=mask_a,
            other=0.0,
        )

        x1 = tl.load(
            base + p * (2 * C) + c,
            mask=mask_c,
            other=0.0,
        )
        x2 = tl.load(
            base + p * (2 * C) + C + c,
            mask=mask_c,
            other=0.0,
        )

        u = u + t[:, None] * x1[None, :]
        v = v + t[:, None] * x2[None, :]

    return u, v


@triton.jit
def _gate_value_derivative(s, GATE_ACT: tl.constexpr):
    if GATE_ACT == 0:
        # torch.nn.functional.silu(s)
        sig = tl.sigmoid(s)
        gate = s * sig

        # d SiLU(s) / ds
        derivative = sig + s * sig * (1.0 - sig)

    elif GATE_ACT == 1:
        # torch.sigmoid(s)
        sig = tl.sigmoid(s)
        gate = sig
        derivative = sig * (1.0 - sig)

    else:
        # Identity
        gate = s
        derivative = tl.full(s.shape, 1.0, tl.float32)

    return gate, derivative


def _s2_configs():
    return [
        triton.Config({"BLOCK_C": 4}, num_warps=4),
        triton.Config({"BLOCK_C": 8}, num_warps=4),
        triton.Config({"BLOCK_C": 16}, num_warps=4),
        triton.Config({"BLOCK_C": 32}, num_warps=4),
        triton.Config({"BLOCK_C": 16}, num_warps=8),
    ]


# ============================================================
# Forward kernel
# ============================================================

@triton.autotune(
    configs=_s2_configs(),
    key=["J", "A", "C", "GATE_ACT"],
)
@triton.jit(do_not_specialize=["N"])
def _s2_forward_kernel(
    X,
    S,
    T,
    FM,
    Y,
    N,
    J: tl.constexpr,
    A: tl.constexpr,
    C: tl.constexpr,
    GATE_ACT: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    a = tl.arange(0, BLOCK_A)

    mask_c = (n < N) & (c < C)
    mask_a = a < A

    # PyTorch:
    # x_grid = torch.einsum("aj,njc->nac", T, X)
    # u, v = torch.chunk(x_grid, 2, dim=-1)
    u, v = _project_input_pair(
        X, T, n, c, a, mask_c, mask_a,
        J, C, BLOCK_A, BLOCK_C,
    )

    # PyTorch:
    # product = u * v
    # product = Identity(product)
    product = u * v

    # PyTorch:
    # s2 = scalars[:, 2*C:3*C]
    # gate = act(s2)
    s2 = tl.load(
        S + n * (3 * C) + 2 * C + c,
        mask=mask_c,
        other=0.0,
    )
    gate, _ = _gate_value_derivative(s2, GATE_ACT)

    # PyTorch:
    # z = torch.einsum("ja,nac->njc", FM, product)
    # output = z * gate[:, None, :]
    #
    for j in tl.static_range(0, J):
        f = tl.load(
            FM + j * A + a,
            mask=mask_a,
            other=0.0,
        )

        z = tl.sum(f[:, None] * product, axis=0)
        y = z * gate

        tl.store(
            Y + n * J * C + j * C + c,
            y,
            mask=mask_c,
        )


# ============================================================
# Backward kernel
# ============================================================

@triton.autotune(
    configs=_s2_configs(),
    key=["J", "A", "C", "GATE_ACT", "NEED_DX", "NEED_DS"],
)
@triton.jit(do_not_specialize=["N"])
def _s2_backward_kernel(
    X,
    S,
    T,
    FM,
    DY,
    DX,
    DS,
    N,
    J: tl.constexpr,
    A: tl.constexpr,
    C: tl.constexpr,
    GATE_ACT: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DS: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    a = tl.arange(0, BLOCK_A)

    mask_c = (n < N) & (c < C)
    mask_a = a < A

    # x_grid = torch.einsum("aj,njc->nac", T, X)
    # u, v = torch.chunk(x_grid, 2, dim=-1)
    u, v = _project_input_pair(
        X, T, n, c, a, mask_c, mask_a,
        J, C, BLOCK_A, BLOCK_C,
    )

    s2 = tl.load(
        S + n * (3 * C) + 2 * C + c,
        mask=mask_c,
        other=0.0,
    )
    gate, gate_prime = _gate_value_derivative(s2, GATE_ACT)

    # PyTorch backward:
    # r = torch.einsum("ja,njc->nac", FM, grad_output)
    #
    r = tl.zeros((BLOCK_A, BLOCK_C), tl.float32)

    for j in tl.static_range(0, J):
        f = tl.load(
            FM + j * A + a,
            mask=mask_a,
            other=0.0,
        )
        dy = tl.load(
            DY + n * J * C + j * C + c,
            mask=mask_c,
            other=0.0,
        )
        r = r + f[:, None] * dy[None, :]

    if NEED_DS:
        # z = einsum("ja,nac->njc", FM, u * v)
        # dgate = (grad_output * z).sum(dim=1)
        #
        # reuse r：
        # dgate = (r * u * v).sum(dim=1)
        # ds2 = dgate * act_prime(s2)
        dgate = tl.sum(r * (u * v), axis=0)
        ds2 = dgate * gate_prime

        s_base = DS + n * (3 * C) + c

        tl.store(s_base, 0.0, mask=mask_c)
        tl.store(s_base + C, 0.0, mask=mask_c)
        tl.store(s_base + 2 * C, ds2, mask=mask_c)

    if NEED_DX:
        # PyTorch:
        # dproduct = r * gate[:, None, :]
        # du = dproduct * v
        # dv = dproduct * u
        dproduct = r * gate[None, :]
        du = dproduct * v
        dv = dproduct * u

        # PyTorch:
        # dx1 = torch.einsum("aj,nac->njc", T, du)
        # dx2 = torch.einsum("aj,nac->njc", T, dv)
        # dx = torch.cat((dx1, dx2), dim=-1)
        #
        for p in tl.static_range(0, J):
            t = tl.load(
                T + a * J + p,
                mask=mask_a,
                other=0.0,
            )

            dx1 = tl.sum(t[:, None] * du, axis=0)
            dx2 = tl.sum(t[:, None] * dv, axis=0)

            dx_base = DX + n * J * (2 * C) + p * (2 * C) + c

            tl.store(dx_base, dx1, mask=mask_c)
            tl.store(dx_base + C, dx2, mask=mask_c)


# ============================================================
# PyTorch autograd
# ============================================================

class _S2ProductGateFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, scalars, to_grid_mat, from_grid_mat, act_code):
        N, J, two_c = inputs.shape
        C = two_c // 2
        A = to_grid_mat.shape[0]

        output = torch.empty(
            (N, J, C),
            device=inputs.device,
            dtype=inputs.dtype,
        )

        ctx.save_for_backward(
            inputs, scalars, to_grid_mat, from_grid_mat
        )
        ctx.act_code = act_code

        if N > 0:
            grid = lambda meta: (
                N,
                triton.cdiv(C, meta["BLOCK_C"]),
            )

            with torch.cuda.device(inputs.device):
                _s2_forward_kernel[grid](
                    inputs,
                    scalars,
                    to_grid_mat,
                    from_grid_mat,
                    output,
                    N=N,
                    J=J,
                    A=A,
                    C=C,
                    GATE_ACT=act_code,
                    BLOCK_A=triton.next_power_of_2(A),
                )

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        inputs, scalars, to_grid_mat, from_grid_mat = ctx.saved_tensors

        need_dx = ctx.needs_input_grad[0]
        need_ds = ctx.needs_input_grad[1]

        if not need_dx and not need_ds:
            return None, None, None, None, None

        N, J, two_c = inputs.shape
        C = two_c // 2
        A = to_grid_mat.shape[0]

        grad_output = grad_output.contiguous()

        dx = torch.empty_like(inputs) if need_dx else None
        ds = torch.empty_like(scalars) if need_ds else None

        if N > 0:
            grid = lambda meta: (
                N,
                triton.cdiv(C, meta["BLOCK_C"]),
            )

            dx_ptr = dx if need_dx else inputs
            ds_ptr = ds if need_ds else scalars

            with torch.cuda.device(inputs.device):
                _s2_backward_kernel[grid](
                    inputs,
                    scalars,
                    to_grid_mat,
                    from_grid_mat,
                    grad_output,
                    dx_ptr,
                    ds_ptr,
                    N=N,
                    J=J,
                    A=A,
                    C=C,
                    GATE_ACT=ctx.act_code,
                    NEED_DX=need_dx,
                    NEED_DS=need_ds,
                    BLOCK_A=triton.next_power_of_2(A),
                )
        # inputs, scalars, to_grid_mat, from_grid_mat, act_code
        return dx, ds, None, None, None


# ============================================================
# Interface
# ============================================================

def fused_s2_product_gate(
    inputs,
    scalars,
    to_grid_mat,
    from_grid_mat,
    gate_act="silu",
    lmax=None,
):
    """
    inputs:        [N, J, 2*C], float32
    scalars:       [N, 3*C] / [N, 1, 3*C], float32
    to_grid_mat:   [A, J], float32
    from_grid_mat: [J, A], float32

    return: 
        [N, J, C]

    """
    tensors = (inputs, scalars, to_grid_mat, from_grid_mat)

    if not all(t.is_cuda for t in tensors):
        raise ValueError("All tensors must be CUDA tensors.")

    if not all(t.device == inputs.device for t in tensors):
        raise ValueError("All tensors must be on the same device.")

    if not all(t.dtype == torch.float32 for t in tensors):
        raise TypeError("Only float32 is supported.")

    if to_grid_mat.requires_grad or from_grid_mat.requires_grad:
        raise ValueError(
            "Gradients for to_grid_mat/from_grid_mat are not implemented."
        )

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape [N, J, 2*C].")

    N, J, two_c = inputs.shape

    if J <= 0 or two_c <= 0 or two_c % 2 != 0:
        raise ValueError("J must be positive and 2*C must be positive/even.")

    C = two_c // 2

    if lmax is not None:
        if not isinstance(lmax, int) or lmax < 0:
            raise ValueError("lmax must be a nonnegative integer.")
        if J != (lmax + 1) ** 2:
            raise ValueError(
                f"lmax={lmax} requires J={(lmax + 1)**2}, got {J}."
            )

    if scalars.ndim == 3:
        if tuple(scalars.shape) != (N, 1, 3 * C):
            raise ValueError(
                f"Expected scalars {(N, 1, 3*C)}, got {tuple(scalars.shape)}."
            )
        scalars_2d = scalars[:, 0, :]
    elif scalars.ndim == 2:
        if tuple(scalars.shape) != (N, 3 * C):
            raise ValueError(
                f"Expected scalars {(N, 3*C)}, got {tuple(scalars.shape)}."
            )
        scalars_2d = scalars
    else:
        raise ValueError("scalars must have 2 or 3 dimensions.")

    if to_grid_mat.ndim != 2 or to_grid_mat.shape[1] != J:
        raise ValueError(f"to_grid_mat must have shape [A, {J}].")

    A = to_grid_mat.shape[0]

    if A <= 0 or tuple(from_grid_mat.shape) != (J, A):
        raise ValueError(f"from_grid_mat must have shape {(J, A)}, A > 0.")

    act_codes = {
        "silu": 0,
        "sigmoid": 1,
        "identity": 2,
    }
    if gate_act not in act_codes:
        raise ValueError(f"Unsupported gate_act: {gate_act!r}")

    return _S2ProductGateFunction.apply(
        inputs.contiguous(),
        scalars_2d.contiguous(),
        to_grid_mat.contiguous(),
        from_grid_mat.contiguous(),
        act_codes[gate_act],
    )