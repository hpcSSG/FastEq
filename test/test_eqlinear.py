"""Correctness and dispatch tests for the CUTLASS-backed eqlinear operator.

Run with: /app/work/fasteq_venv/bin/python -m pytest test/test_eqlinear.py -q
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

CONFIGS = {
    "SMALL_P1": (96, [1, 3, 5, 7]),
    "MEDIUM_P1": (128, [1, 3, 5, 7]),
    "MEDIUM_P2": (128, [1, 1, 3, 3, 3, 5, 5, 5, 7, 7]),
    "LARGE_P1": (224, [1, 3, 5, 7]),
    "LARGE_P3": (224, [1, 1, 1, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 7]),
}
BATCHES = [512, 1024, 2048, 2944, 5888]


def _out_rows(i_dims):
    total = 0
    p = 0
    while p < len(i_dims):
        i = i_dims[p]
        q = p + 1
        while q < len(i_dims) and i_dims[q] == i:
            q += 1
        total += i
        p = q
    return total


def _reference(x, w, i_dims, cg, dtype):
    """Direct loop implementation of the uniform uv/iu/iv eqlinear."""
    if x.ndim == 2:
        x = x.reshape(x.shape[0], sum(i_dims), w.shape[1])
    x = x.to(dtype)
    w = w.to(dtype)
    batch, _, u = x.shape
    v = w.shape[2]
    runs = []
    p = 0
    while p < len(i_dims):
        i = i_dims[p]
        q = p + 1
        while q < len(i_dims) and i_dims[q] == i:
            q += 1
        runs.append((i, p, q - p))
        p = q
    in_prefix = [0]
    for i in i_dims[:-1]:
        in_prefix.append(in_prefix[-1] + i)
    out = torch.zeros(batch, _out_rows(i_dims), v, dtype=dtype, device=x.device)
    out_prefix = 0
    for i, p0, count in runs:
        for t in range(i):
            acc = None
            for q in range(p0, p0 + count):
                row = in_prefix[q] + t
                term = x[:, row, :] @ w[q] * float(cg[q])
                acc = term if acc is None else acc + term
            out[:, out_prefix + t, :] = acc
        out_prefix += i
    return out


@dataclass
class _FakePath:
    coefficients: float


class _FakeDescriptor:
    def __init__(self, i_dims, u, v, cg):
        self.paths = [_FakePath(c) for c in cg]
        runs = []
        p = 0
        while p < len(i_dims):
            runs.append(i_dims[p])
            q = p + 1
            while q < len(i_dims) and i_dims[q] == i_dims[p]:
                q += 1
            p = q
        self.operands = [
            tuple((u * v,) for _ in i_dims),
            tuple((i,) for i in i_dims),
            tuple((i,) for i in runs),
        ]
        self._u = u
        self._v = v
        self._i = list(i_dims)

    def get_dims(self, name):
        return [self._u] if name == "u" else [self._v]

    def get_dimensions_dict(self):
        return {"i": list(self._i)}


def _make(shape, batch, dtype, seed=0):
    u, i_dims = CONFIGS[shape]
    torch.manual_seed(seed)
    x = torch.randn(batch, sum(i_dims) * u, device="cuda", dtype=dtype)
    w = torch.randn(len(i_dims) * u * u, device="cuda", dtype=dtype)
    cg = [float(0.01 + 0.001 * p) for p in range(len(i_dims))]
    return x, w, cg, i_dims, u


def _rel_fro(a, b):
    return (a - b).norm().item() / b.norm().clamp_min(1e-30).item()


@pytest.mark.parametrize("shape", list(CONFIGS))
@pytest.mark.parametrize("batch", BATCHES)
def test_fp64_forward_and_grad(shape, batch):
    x, w, cg, i_dims, u = _make(shape, batch, torch.float64)
    desc = _FakeDescriptor(i_dims, u, u, cg)
    from fasteq.ops.eqlinear import fast_equi_linear

    x.requires_grad_(True)
    out = fast_equi_linear(desc, w, x)
    ref = _reference(x, w.reshape(len(i_dims), u, u), i_dims, cg, torch.float64)
    assert out.shape == (batch, _out_rows(i_dims) * u)
    assert _rel_fro(out, ref.reshape(batch, -1)) <= 1e-12

    grad_out = torch.randn_like(out)
    (grad_x,) = torch.autograd.grad(out, x, grad_out)
    ref_x = x.detach().clone().requires_grad_(True)
    ref_out = _reference(ref_x, w.reshape(len(i_dims), u, u), i_dims, cg, torch.float64)
    (ref_grad_x,) = torch.autograd.grad(ref_out, ref_x, grad_out.reshape(ref_out.shape))
    assert _rel_fro(grad_x, ref_grad_x) <= 1e-12


@pytest.mark.parametrize("shape", list(CONFIGS))
@pytest.mark.parametrize("batch", BATCHES)
def test_fp32_forward_and_grad(shape, batch):
    torch.backends.cuda.matmul.allow_tf32 = False
    x, w, cg, i_dims, u = _make(shape, batch, torch.float32)
    desc = _FakeDescriptor(i_dims, u, u, cg)
    from fasteq.ops.eqlinear import fast_equi_linear

    out = fast_equi_linear(desc, w, x, compute_precision="fp32")
    ref = _reference(x, w.reshape(len(i_dims), u, u), i_dims, cg, torch.float64).reshape(batch, -1)
    ref_fp32 = _reference(x, w.reshape(len(i_dims), u, u), i_dims, cg, torch.float32).reshape(batch, -1)
    ours = _rel_fro(out.float(), ref)
    torch_rel = _rel_fro(ref_fp32, ref)
    assert ours <= 1.05 * torch_rel + 1e-7, (ours, torch_rel)

    x = x.detach().requires_grad_(True)
    out = fast_equi_linear(desc, w, x, compute_precision="fp32")
    grad_out = torch.randn_like(out)
    (grad_x,) = torch.autograd.grad(out, x, grad_out)
    ref_x = x.detach().clone().requires_grad_(True)
    ref_out = _reference(ref_x, w.reshape(len(i_dims), u, u), i_dims, cg, torch.float32)
    (ref_grad_x,) = torch.autograd.grad(ref_out, ref_x, grad_out.reshape(ref_out.shape))
    assert _rel_fro(grad_x, ref_grad_x) <= 1.05 * torch_rel + 1e-7


@pytest.mark.parametrize("shape", list(CONFIGS))
def test_tf32_forward(shape):
    x, w, cg, i_dims, u = _make(shape, 1024, torch.float32)
    desc = _FakeDescriptor(i_dims, u, u, cg)
    from fasteq.ops.eqlinear import fast_equi_linear

    out = fast_equi_linear(desc, w, x)
    ref = _reference(x, w.reshape(len(i_dims), u, u), i_dims, cg, torch.float64).reshape(x.shape[0], -1)
    assert _rel_fro(out, ref) < 2e-5


def test_dispatch_and_determinism():
    from fasteq.ops.eqlinear import fast_equi_linear

    x, w, cg, i_dims, u = _make("SMALL_P1", 512, torch.float32)
    desc = _FakeDescriptor(i_dims, u, u, cg)
    a = fast_equi_linear(desc, w, x)
    b = fast_equi_linear(desc, w, x)
    assert torch.equal(a, b)

    w3 = w.reshape(len(i_dims), u, u).contiguous()
    h = torch.ops.eqlinear.forward_create(w3, i_dims, cg, 512, u, len(i_dims), u, 0)
    info = torch.ops.eqlinear.forward_plan_info(h)
    torch.ops.eqlinear.forward_destroy(h)
    assert info[1] == 0  # tf32 small-batch branch

    h = torch.ops.eqlinear.forward_create(w3, i_dims, cg, 2048, u, len(i_dims), u, 0)
    info = torch.ops.eqlinear.forward_plan_info(h)
    torch.ops.eqlinear.forward_destroy(h)
    assert info[1] == 5


def test_plan_cache_invalidation():
    from fasteq.ops.eqlinear import (
        eqlinear_clear_plan_cache,
        eqlinear_set_plan_cache,
        fast_equi_linear,
    )

    x, w, cg, i_dims, u = _make("SMALL_P1", 512, torch.float32)
    desc = _FakeDescriptor(i_dims, u, u, cg)
    eqlinear_clear_plan_cache()
    first = fast_equi_linear(desc, w, x)
    second = fast_equi_linear(desc, w, x)
    assert torch.equal(first, second)

    w = w.clone()
    w.add_(1.0)
    third = fast_equi_linear(desc, w, x)
    ref = _reference(x, w.reshape(len(i_dims), u, u), i_dims, cg, torch.float32)
    assert _rel_fro(third, ref.reshape(512, -1)) < 1e-6

    eqlinear_set_plan_cache(0)
    fourth = fast_equi_linear(desc, w, x)
    assert torch.equal(third, fourth)
    eqlinear_set_plan_cache()


def test_strict_errors():
    x, w, cg, i_dims, u = _make("SMALL_P1", 512, torch.float32)
    desc = _FakeDescriptor(i_dims, u, u, cg)
    from fasteq.ops.eqlinear import fast_equi_linear

    with pytest.raises(ValueError):
        fast_equi_linear(desc, w, x, compute_precision="bf16")
    with pytest.raises(ValueError):
        fast_equi_linear(desc, w, x.double())
    with pytest.raises(ValueError):
        fast_equi_linear(desc, w, x.half())
