"""Batched rotation conversions and quaternion arithmetic in one launch each.

The public wrappers preserve the upstream shapes. Kernel IDs are internal.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .._common import check_gpu, grid


@triton.jit
def _rotation_kernel(A, B, C, D, OUT, N: tl.constexpr, OP: tl.constexpr,
                     BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    if OP == 0:  # identity angles
        tl.store(OUT + i * 3, 0., valid)
        tl.store(OUT + i * 3 + 1, 0., valid)
        tl.store(OUT + i * 3 + 2, 0., valid)
    elif OP == 1:  # inverse angles
        tl.store(OUT + i * 3, -tl.load(C + i, valid, other=0), valid)
        tl.store(OUT + i * 3 + 1, -tl.load(B + i, valid, other=0), valid)
        tl.store(OUT + i * 3 + 2, -tl.load(A + i, valid, other=0), valid)
    elif OP == 2:  # identity quaternion
        for j in tl.static_range(4):
            tl.store(OUT + i * 4 + j, 1. if j == 0 else 0., valid)
    elif OP == 3:  # compose quaternion
        w = tl.load(A + i * 4, valid, other=0)
        x = tl.load(A + i * 4 + 1, valid, other=0)
        y = tl.load(A + i * 4 + 2, valid, other=0)
        z = tl.load(A + i * 4 + 3, valid, other=0)
        v = tl.load(B + i * 4, valid, other=0)
        a = tl.load(B + i * 4 + 1, valid, other=0)
        b = tl.load(B + i * 4 + 2, valid, other=0)
        c = tl.load(B + i * 4 + 3, valid, other=0)
        tl.store(OUT + i * 4, w*v-x*a-y*b-z*c, valid)
        tl.store(OUT + i * 4 + 1, x*v+w*a+y*c-z*b, valid)
        tl.store(OUT + i * 4 + 2, w*b-x*c+y*v+z*a, valid)
        tl.store(OUT + i * 4 + 3, w*c+x*b-y*a+z*v, valid)
    elif OP == 4:  # inverse unit quaternion
        for j in tl.static_range(4):
            v = tl.load(A + i * 4 + j, valid, other=0)
            tl.store(OUT + i * 4 + j, v if j == 0 else -v, valid)
    elif OP == 5 or OP == 6 or OP == 7:  # x / y / z axis matrices
        angle = tl.load(A + i, valid, other=0)
        c, s = tl.cos(angle), tl.sin(angle)
        if OP == 5:
            v0, v1, v2, v3, v4, v5, v6, v7, v8 = 1.,0.,0.,0.,c,-s,0.,s,c
        elif OP == 6:
            v0, v1, v2, v3, v4, v5, v6, v7, v8 = c,0.,s,0.,1.,0.,-s,0.,c
        else:
            v0, v1, v2, v3, v4, v5, v6, v7, v8 = c,-s,0.,s,c,0.,0.,0.,1.
        tl.store(OUT + i*9, v0, valid); tl.store(OUT + i*9+1, v1, valid)
        tl.store(OUT + i*9+2, v2, valid); tl.store(OUT + i*9+3, v3, valid)
        tl.store(OUT + i*9+4, v4, valid); tl.store(OUT + i*9+5, v5, valid)
        tl.store(OUT + i*9+6, v6, valid); tl.store(OUT + i*9+7, v7, valid)
        tl.store(OUT + i*9+8, v8, valid)
    elif OP == 8:  # Y(alpha) X(beta) Y(gamma)
        a = tl.load(A+i, valid, other=0)
        b = tl.load(B+i, valid, other=0)
        g = tl.load(C+i, valid, other=0)
        ca, sa, cb, sb, cg, sg = tl.cos(a), tl.sin(a), tl.cos(b), tl.sin(b), tl.cos(g), tl.sin(g)
        tl.store(OUT+i*9, ca*cg-sa*cb*sg, valid)
        tl.store(OUT+i*9+1, sa*sb, valid)
        tl.store(OUT+i*9+2, ca*sg+sa*cb*cg, valid)
        tl.store(OUT+i*9+3, sb*sg, valid)
        tl.store(OUT+i*9+4, cb, valid)
        tl.store(OUT+i*9+5, -sb*cg, valid)
        tl.store(OUT+i*9+6, -sa*cg-ca*cb*sg, valid)
        tl.store(OUT+i*9+7, ca*sb, valid)
        tl.store(OUT+i*9+8, -sa*sg+ca*cb*cg, valid)
    elif OP == 9:  # angles to quaternion
        a = tl.load(A+i, valid, other=0) / 2
        b = tl.load(B+i, valid, other=0) / 2
        g = tl.load(C+i, valid, other=0) / 2
        ca, sa, cb, sb, cg, sg = tl.cos(a), tl.sin(a), tl.cos(b), tl.sin(b), tl.cos(g), tl.sin(g)
        tl.store(OUT+i*4, cb*(ca*cg-sa*sg), valid)
        tl.store(OUT+i*4+1, sb*(ca*cg+sa*sg), valid)
        tl.store(OUT+i*4+2, cb*(ca*sg+sa*cg), valid)
        tl.store(OUT+i*4+3, sb*(ca*sg-sa*cg), valid)
    elif OP == 10:  # axis angle to quaternion
        x = tl.load(A+i*3, valid, other=0)
        y = tl.load(A+i*3+1, valid, other=0)
        z = tl.load(A+i*3+2, valid, other=0)
        angle = tl.load(B+i, valid, other=0) / 2
        inv = 1. / tl.maximum(tl.sqrt(x*x+y*y+z*z), 1.e-12)
        s = tl.sin(angle)
        tl.store(OUT+i*4, tl.cos(angle), valid)
        tl.store(OUT+i*4+1, x*inv*s, valid)
        tl.store(OUT+i*4+2, y*inv*s, valid)
        tl.store(OUT+i*4+3, z*inv*s, valid)
    elif OP == 11:  # quaternion to axis angle (axis, angle)
        w = tl.load(A+i*4, valid, other=0)
        x = tl.load(A+i*4+1, valid, other=0)
        y = tl.load(A+i*4+2, valid, other=0)
        z = tl.load(A+i*4+3, valid, other=0)
        inv = 1. / tl.maximum(tl.sqrt(x*x+y*y+z*z), 1.e-12)
        tl.store(OUT+i*4, x*inv, valid)
        tl.store(OUT+i*4+1, y*inv, valid)
        tl.store(OUT+i*4+2, z*inv, valid)
        tl.store(OUT+i*4+3, 2.*libdevice.acos(tl.minimum(1., tl.maximum(-1., w))), valid)
    elif OP == 12:  # axis angle to matrix (Rodrigues, zero axis -> z)
        x = tl.load(A+i*3, valid, other=0)
        y = tl.load(A+i*3+1, valid, other=0)
        z = tl.load(A+i*3+2, valid, other=0)
        angle = tl.load(B+i, valid, other=0)
        norm = tl.sqrt(x*x+y*y+z*z)
        x = tl.where(norm > 0, x / tl.maximum(norm, 1.e-12), 0.)
        y = tl.where(norm > 0, y / tl.maximum(norm, 1.e-12), 0.)
        z = tl.where(norm > 0, z / tl.maximum(norm, 1.e-12), 1.)
        c, s = tl.cos(angle), tl.sin(angle)
        t = 1.-c
        tl.store(OUT+i*9, c+t*x*x, valid)
        tl.store(OUT+i*9+1, t*x*y-s*z, valid)
        tl.store(OUT+i*9+2, t*x*z+s*y, valid)
        tl.store(OUT+i*9+3, t*y*x+s*z, valid)
        tl.store(OUT+i*9+4, c+t*y*y, valid)
        tl.store(OUT+i*9+5, t*y*z-s*x, valid)
        tl.store(OUT+i*9+6, t*z*x-s*y, valid)
        tl.store(OUT+i*9+7, t*z*y+s*x, valid)
        tl.store(OUT+i*9+8, c+t*z*z, valid)
    elif OP == 13:  # quaternion to matrix, unit quaternions
        w = tl.load(A+i*4, valid, other=0)
        x = tl.load(A+i*4+1, valid, other=0)
        y = tl.load(A+i*4+2, valid, other=0)
        z = tl.load(A+i*4+3, valid, other=0)
        vnorm = tl.sqrt(x*x+y*y+z*z)
        axisnorm = tl.maximum(vnorm, 1.e-12)
        x, y, z = tl.where(vnorm > 0, x/axisnorm, 0.), tl.where(vnorm > 0, y/axisnorm, 0.), tl.where(vnorm > 0, z/axisnorm, 1.)
        angle = 2.*libdevice.acos(tl.minimum(1., tl.maximum(-1., w)))
        c, s = tl.cos(angle), tl.sin(angle)
        t = 1.-c
        tl.store(OUT+i*9, c+t*x*x, valid)
        tl.store(OUT+i*9+1, t*x*y-s*z, valid)
        tl.store(OUT+i*9+2, t*x*z+s*y, valid)
        tl.store(OUT+i*9+3, t*y*x+s*z, valid)
        tl.store(OUT+i*9+4, c+t*y*y, valid)
        tl.store(OUT+i*9+5, t*y*z-s*x, valid)
        tl.store(OUT+i*9+6, t*z*x-s*y, valid)
        tl.store(OUT+i*9+7, t*z*y+s*x, valid)
        tl.store(OUT+i*9+8, c+t*z*z, valid)
    elif OP == 14:  # angles to xyz
        a = tl.load(A+i, valid, other=0)
        b = tl.load(B+i, valid, other=0)
        tl.store(OUT+i*3, tl.sin(b)*tl.sin(a), valid)
        tl.store(OUT+i*3+1, tl.cos(b), valid)
        tl.store(OUT+i*3+2, tl.sin(b)*tl.cos(a), valid)
    elif OP == 15:  # xyz to angles
        x = tl.load(A+i*3, valid, other=0)
        y = tl.load(A+i*3+1, valid, other=0)
        z = tl.load(A+i*3+2, valid, other=0)
        inv = 1. / tl.maximum(tl.sqrt(x*x+y*y+z*z), 1.e-12)
        x = tl.minimum(1., tl.maximum(-1., x*inv))
        y = tl.minimum(1., tl.maximum(-1., y*inv))
        z = tl.minimum(1., tl.maximum(-1., z*inv))
        tl.store(OUT+i*2, libdevice.atan2(x, z), valid)
        tl.store(OUT+i*2+1, libdevice.acos(y), valid)
    elif OP == 16:  # compose two axis-angle pairs
        x1 = tl.load(A+i*3, valid, other=0)
        y1 = tl.load(A+i*3+1, valid, other=0)
        z1 = tl.load(A+i*3+2, valid, other=0)
        x2 = tl.load(C+i*3, valid, other=0)
        y2 = tl.load(C+i*3+1, valid, other=0)
        z2 = tl.load(C+i*3+2, valid, other=0)
        s1 = tl.sin(tl.load(B+i, valid, other=0)/2)
        s2 = tl.sin(tl.load(D+i, valid, other=0)/2)
        w1 = tl.cos(tl.load(B+i, valid, other=0)/2)
        w2 = tl.cos(tl.load(D+i, valid, other=0)/2)
        n1 = 1./tl.maximum(tl.sqrt(x1*x1+y1*y1+z1*z1), 1.e-12)
        n2 = 1./tl.maximum(tl.sqrt(x2*x2+y2*y2+z2*z2), 1.e-12)
        x1, y1, z1 = x1*n1*s1, y1*n1*s1, z1*n1*s1
        x2, y2, z2 = x2*n2*s2, y2*n2*s2, z2*n2*s2
        w = w1*w2-x1*x2-y1*y2-z1*z2
        x = x1*w2+w1*x2+y1*z2-z1*y2
        y = w1*y2-x1*z2+y1*w2+z1*x2
        z = w1*z2+x1*y2-y1*x2+z1*w2
        inv = 1./tl.maximum(tl.sqrt(x*x+y*y+z*z), 1.e-12)
        tl.store(OUT+i*4, x*inv, valid)
        tl.store(OUT+i*4+1, y*inv, valid)
        tl.store(OUT+i*4+2, z*inv, valid)
        tl.store(OUT+i*4+3, 2.*libdevice.acos(tl.minimum(1., tl.maximum(-1., w))), valid)


def _run(op, shape, width, dtype, device, *args):
    inputs = [x.contiguous() for x in args]
    check_gpu(*inputs)
    n = int(torch.tensor(shape).prod()) if shape else 1
    out = torch.empty((*shape, width), dtype=dtype, device=device)
    sentinel = inputs[0] if inputs else out
    padded = (inputs + [sentinel] * 4)[:4]
    _rotation_kernel[grid(n)](*padded, out, n, op, 256)
    return out


def _broadcast(*args):
    check_gpu(*args)
    return tuple(x.contiguous() for x in torch.broadcast_tensors(*args))


@triton.jit
def _inverse_angles_kernel(A, B, C, OA, OB, OC, NA: tl.constexpr,
                           NB: tl.constexpr, NC: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OA + i, -tl.load(C + i, i < NC, other=0), i < NC)
    tl.store(OB + i, -tl.load(B + i, i < NB, other=0), i < NB)
    tl.store(OC + i, -tl.load(A + i, i < NA, other=0), i < NA)


@triton.jit
def _matrix_to_angles_kernel(R, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    r01 = tl.load(R + 9 * i + 1, valid, other=0)
    r21 = tl.load(R + 9 * i + 7, valid, other=0)
    r11 = tl.load(R + 9 * i + 4, valid, other=1)
    a = libdevice.atan2(r01, r21)
    b = libdevice.acos(tl.minimum(1., tl.maximum(-1., r11)))
    ca, sa = tl.cos(a), tl.sin(a)
    r02 = tl.load(R + 9 * i + 2, valid, other=0)
    r22 = tl.load(R + 9 * i + 8, valid, other=0)
    r00 = tl.load(R + 9 * i, valid, other=1)
    r20 = tl.load(R + 9 * i + 6, valid, other=0)
    c = libdevice.atan2(ca * r02 - sa * r22, ca * r00 - sa * r20)
    tl.store(OUT + i * 3, a, valid)
    tl.store(OUT + i * 3 + 1, b, valid)
    tl.store(OUT + i * 3 + 2, c, valid)


@triton.jit
def _matrix_to_axis_angle_kernel(R, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    r00 = tl.load(R + 9 * i, valid, other=1)
    r11 = tl.load(R + 9 * i + 4, valid, other=1)
    r22 = tl.load(R + 9 * i + 8, valid, other=1)
    x = tl.load(R + 9 * i + 7, valid, other=0) - tl.load(R + 9 * i + 5, valid, other=0)
    y = tl.load(R + 9 * i + 2, valid, other=0) - tl.load(R + 9 * i + 6, valid, other=0)
    z = tl.load(R + 9 * i + 3, valid, other=0) - tl.load(R + 9 * i + 1, valid, other=0)
    norm = tl.maximum(tl.sqrt(x*x+y*y+z*z), 1.e-12)
    angle = libdevice.acos(tl.minimum(1., tl.maximum(-1., (r00+r11+r22-1.)/2.)))
    tl.store(OUT + i * 4, x / norm, valid)
    tl.store(OUT + i * 4 + 1, y / norm, valid)
    tl.store(OUT + i * 4 + 2, z / norm, valid)
    tl.store(OUT + i * 4 + 3, angle, valid)


@triton.jit
def _matrix_mul3_kernel(A, B, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    for row in tl.static_range(3):
        for col in tl.static_range(3):
            value = tl.load(A + i*9 + row*3, valid, other=0) * tl.load(B + i*9 + col, valid, other=0)
            for inner in tl.static_range(1, 3):
                value += (tl.load(A + i*9 + row*3 + inner, valid, other=0)
                          * tl.load(B + i*9 + inner*3 + col, valid, other=0))
            tl.store(OUT + i*9 + row*3 + col, value, valid)


def identity_angles(*shape, requires_grad=False, dtype=None, device=None):
    ref = torch.empty((), dtype=dtype, device=device)
    if not ref.is_cuda:
        raise ValueError("Triton implementation requires a GPU device")
    out = _run(0, shape, 3, ref.dtype, ref.device)
    return tuple(x.requires_grad_(requires_grad) for x in out.unbind(-1))


@triton.jit
def _rand_angles_transform_kernel(U_AG, U_B, OUT, N: tl.constexpr,
                                  BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    u_alpha = tl.load(U_AG + i, valid, other=0)
    u_gamma = tl.load(U_AG + N + i, valid, other=0)
    u_beta = tl.load(U_B + i, valid, other=0)
    tl.store(OUT + i, u_alpha * 6.283185307179586, valid)
    tl.store(OUT + N + i, libdevice.acos((u_beta * 2.0) - 1.0), valid)
    tl.store(OUT + 2 * N + i, u_gamma * 6.283185307179586, valid)


def rand_angles(*shape, requires_grad=False, dtype=None, device=None):
    """Random Euler angles with the same PyTorch RNG draws as e3nn."""
    # Keep the two calls, their shapes, and their order identical to e3nn.
    u_alpha_gamma = torch.rand(2, *shape, dtype=dtype, device=device)
    u_beta = torch.rand(shape, dtype=dtype, device=device)
    check_gpu(u_alpha_gamma, u_beta)
    n = u_beta.numel()
    angles = torch.empty((3, *shape), dtype=u_beta.dtype, device=u_beta.device)
    if n:
        _rand_angles_transform_kernel[grid(n)](
            u_alpha_gamma, u_beta, angles, n, 256, enable_fp_fusion=False
        )
    return tuple(angles[i].detach().requires_grad_(requires_grad) for i in range(3))


def rand_matrix(*shape, requires_grad=False, dtype=None, device=None):
    matrix = angles_to_matrix(*rand_angles(*shape, dtype=dtype, device=device))
    return matrix.detach().requires_grad_(requires_grad)


def rand_quaternion(*shape, requires_grad=False, dtype=None, device=None):
    quaternion = angles_to_quaternion(*rand_angles(*shape, dtype=dtype, device=device))
    return quaternion.detach().requires_grad_(requires_grad)


def rand_axis_angle(*shape, requires_grad=False, dtype=None, device=None):
    axis, angle = angles_to_axis_angle(*rand_angles(*shape, dtype=dtype, device=device))
    return axis.detach().requires_grad_(requires_grad), angle.detach().requires_grad_(requires_grad)


def matrix_to_angles(matrix):
    check_gpu(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("expected (..., 3, 3) matrices")
    out = matrix.new_empty((*matrix.shape[:-2], 3))
    if out.numel():
        _matrix_to_angles_kernel[grid(matrix.numel() // 9)](
            matrix.contiguous(), out, matrix.numel() // 9, 256
        )
    return out.unbind(-1)


def matrix_to_axis_angle(matrix):
    check_gpu(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("expected (..., 3, 3) matrices")
    out = matrix.new_empty((*matrix.shape[:-2], 4))
    if out.numel():
        _matrix_to_axis_angle_kernel[grid(matrix.numel() // 9)](
            matrix.contiguous(), out, matrix.numel() // 9, 256
        )
    return out[..., :3], out[..., 3]


def compose_angles(a1, b1, c1, a2, b2, c2):
    a1, b1, c1, a2, b2, c2 = _broadcast(a1, b1, c1, a2, b2, c2)
    left = angles_to_matrix(a1, b1, c1)
    right = angles_to_matrix(a2, b2, c2)
    out = torch.empty_like(left)
    if out.numel():
        _matrix_mul3_kernel[grid(out.numel() // 9)](
            left, right, out, out.numel() // 9, 256
        )
    return matrix_to_angles(out)


def matrix_to_quaternion(matrix):
    return axis_angle_to_quaternion(*matrix_to_axis_angle(matrix))


def angles_to_axis_angle(alpha, beta, gamma):
    return matrix_to_axis_angle(angles_to_matrix(alpha, beta, gamma))


def quaternion_to_angles(quaternion):
    return matrix_to_angles(quaternion_to_matrix(quaternion))


def axis_angle_to_angles(axis, angle):
    return matrix_to_angles(axis_angle_to_matrix(axis, angle))


def inverse_angles(a, b, c):
    check_gpu(a, b, c)
    oa, ob, oc = (torch.empty_like(x, memory_format=torch.contiguous_format) for x in (c, b, a))
    _inverse_angles_kernel[grid(max(a.numel(), b.numel(), c.numel()))](
        a.contiguous(), b.contiguous(), c.contiguous(), oa, ob, oc,
        a.numel(), b.numel(), c.numel(), 256)
    return oa, ob, oc


def identity_quaternion(*shape, requires_grad=False, dtype=None, device=None):
    ref = torch.empty((), dtype=dtype, device=device)
    if not ref.is_cuda:
        raise ValueError("Triton implementation requires a GPU device")
    return _run(2, shape, 4, ref.dtype, ref.device).requires_grad_(requires_grad)


def compose_quaternion(q1, q2):
    q1, q2 = _broadcast(q1, q2)
    return _run(3, q1.shape[:-1], 4, q1.dtype, q1.device, q1, q2)


def inverse_quaternion(q):
    return _run(4, q.shape[:-1], 4, q.dtype, q.device, q)


def compose_axis_angle(axis1, angle1, axis2, angle2):
    axis1, axis2 = _broadcast(axis1, axis2)
    angle1, angle2 = _broadcast(angle1, angle2)
    out = _run(16, axis1.shape[:-1], 4, axis1.dtype, axis1.device, axis1, angle1, axis2, angle2)
    return out[..., :3], out[..., 3]


def matrix_x(angle):
    return _run(5, angle.shape, 9, angle.dtype, angle.device, angle).reshape(*angle.shape, 3, 3)


def matrix_y(angle):
    return _run(6, angle.shape, 9, angle.dtype, angle.device, angle).reshape(*angle.shape, 3, 3)


def matrix_z(angle):
    return _run(7, angle.shape, 9, angle.dtype, angle.device, angle).reshape(*angle.shape, 3, 3)


def angles_to_matrix(alpha, beta, gamma):
    alpha, beta, gamma = _broadcast(alpha, beta, gamma)
    return _run(8, alpha.shape, 9, alpha.dtype, alpha.device, alpha, beta, gamma).reshape(*alpha.shape, 3, 3)


def angles_to_quaternion(alpha, beta, gamma):
    alpha, beta, gamma = _broadcast(alpha, beta, gamma)
    return _run(9, alpha.shape, 4, alpha.dtype, alpha.device, alpha, beta, gamma)


def axis_angle_to_quaternion(xyz, angle):
    xyz, angle = _broadcast(xyz, angle[..., None])
    return _run(10, xyz.shape[:-1], 4, xyz.dtype, xyz.device, xyz, angle[..., 0])


def quaternion_to_axis_angle(q):
    out = _run(11, q.shape[:-1], 4, q.dtype, q.device, q)
    return out[..., :3], out[..., 3]


def axis_angle_to_matrix(axis, angle):
    axis, angle = _broadcast(axis, angle[..., None])
    return _run(12, axis.shape[:-1], 9, axis.dtype, axis.device, axis, angle[..., 0]).reshape(*axis.shape[:-1], 3, 3)


def quaternion_to_matrix(q):
    return _run(13, q.shape[:-1], 9, q.dtype, q.device, q).reshape(*q.shape[:-1], 3, 3)


def angles_to_xyz(alpha, beta):
    alpha, beta = _broadcast(alpha, beta)
    return _run(14, alpha.shape, 3, alpha.dtype, alpha.device, alpha, beta)


def xyz_to_angles(xyz):
    return _run(15, xyz.shape[:-1], 2, xyz.dtype, xyz.device, xyz).unbind(-1)
