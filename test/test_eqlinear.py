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


def test_plan_cache_reuses_hot_entry_without_rebuilding_key(monkeypatch):
    from fasteq.ops import eqlinear

    x, w, cg, i_dims, u = _make("SMALL_P1", 512, torch.float32)
    desc = _FakeDescriptor(i_dims, u, u, cg)
    eqlinear.eqlinear_clear_plan_cache()
    try:
        first = eqlinear.fast_equi_linear(desc, w, x)

        def unexpected_key_build(*args, **kwargs):
            raise AssertionError("hot plan-cache lookup rebuilt the full key")

        monkeypatch.setattr(eqlinear, "_make_key", unexpected_key_build)
        second = eqlinear.fast_equi_linear(desc, w, x)
        assert torch.equal(first, second)
    finally:
        eqlinear.eqlinear_clear_plan_cache()


def test_flat_cache_hit_skips_weight_view_creation(monkeypatch):
    from fasteq.ops import eqlinear

    x, w, cg, i_dims, u = _make("SMALL_P1", 512, torch.float32)
    desc = _FakeDescriptor(i_dims, u, u, cg)
    eqlinear.eqlinear_clear_plan_cache()
    try:
        first = eqlinear.fast_equi_linear(desc, w, x)
        original_reshape = torch.Tensor.reshape

        def reject_weight_reshape(tensor, *shape):
            if tensor is w:
                raise AssertionError("warm flat call reshaped the cached weight")
            return original_reshape(tensor, *shape)

        monkeypatch.setattr(torch.Tensor, "reshape", reject_weight_reshape)
        second = eqlinear.fast_equi_linear(desc, w, x)
        assert torch.equal(first, second)
    finally:
        eqlinear.eqlinear_clear_plan_cache()


def test_descriptor_metadata_fast_path_preserves_validation():
    from fasteq.ops import eqlinear

    segments = ((1, 96), (3, 96))
    assert eqlinear._operand_segments(segments) is segments

    class IndexLike:
        def __index__(self):
            return 7

    assert eqlinear._operand_segments(((IndexLike(),),)) == ((7,),)

    class DimensionDescriptor:
        def __init__(self, values):
            self.values = values

        def get_dims(self, name):
            return self.values

    assert eqlinear._single_dimension(DimensionDescriptor([96, 96]), "u") == 96
    with pytest.raises(ValueError, match=r"got \(96, 128\)"):
        eqlinear._single_dimension(DimensionDescriptor([96, 128]), "u")
    with pytest.raises(ValueError, match=r"got \(\)"):
        eqlinear._single_dimension(DimensionDescriptor([]), "u")


def test_plan_cache_stream_pointer_matches_pytorch_current_stream():
    from fasteq.ops import eqlinear

    device = torch.device("cuda", torch.cuda.current_device())
    default = torch.cuda.current_stream(device)
    assert eqlinear._current_stream_ptr(device) == default.cuda_stream

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        assert eqlinear._current_stream_ptr(device) == torch.cuda.current_stream(device).cuda_stream
        assert eqlinear._current_stream_ptr(device) == stream.cuda_stream


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


@pytest.mark.parametrize("shape", ["SMALL_P1", "LARGE_P3"])
def test_tf32_queued_distinct_outputs(shape):
    """Queued calls must not overwrite an earlier call's output pointer table."""
    import fasteq  # noqa: F401

    x, w, cg, i_dims, u = _make(shape, 512, torch.float32)
    x3 = x.reshape(512, sum(i_dims), u)
    w3 = w.reshape(len(i_dims), u, u)
    j = _out_rows(i_dims)
    handle = torch.ops.eqlinear.forward_create(w3, i_dims, cg, 512, u, len(i_dims), u, 0)
    try:
        first = torch.full((512, j, u), float("nan"), device="cuda")
        second = torch.full_like(first, float("nan"))
        torch.cuda._sleep(20_000_000)
        torch.ops.eqlinear.forward_run(handle, x3, first)
        torch.ops.eqlinear.forward_run(handle, x3, second)
        torch.cuda.synchronize()
        ref = _reference(x3, w3, i_dims, cg, torch.float64)
        assert torch.equal(first, second)
        assert _rel_fro(first, ref) < 2e-5
    finally:
        torch.ops.eqlinear.forward_destroy(handle)


def test_fp64_dgrad_queued_distinct_outputs():
    """Back-to-back gradient launches must each write their own output."""
    import fasteq  # noqa: F401

    x, w, cg, i_dims, u = _make("SMALL_P1", 512, torch.float64)
    w3 = w.reshape(len(i_dims), u, u)
    grad = torch.randn(512, _out_rows(i_dims), u, dtype=x.dtype, device=x.device)
    handle = torch.ops.eqlinear.dgrad_create(w3, i_dims, cg, 512, u, len(i_dims), u, 2)
    try:
        first = torch.full((512, sum(i_dims), u), float("nan"), dtype=x.dtype, device=x.device)
        second = torch.full_like(first, float("nan"))
        torch.cuda._sleep(20_000_000)
        torch.ops.eqlinear.dgrad_run(handle, grad, first)
        torch.ops.eqlinear.dgrad_run(handle, grad, second)
        torch.cuda.synchronize()
        assert torch.equal(first, second)
    finally:
        torch.ops.eqlinear.dgrad_destroy(handle)


@pytest.mark.parametrize("dtype,precision", [(torch.float32, 0), (torch.float64, 2)])
def test_dgrad_queued_distinct_inputs(dtype, precision):
    """Each queued gradient launch must use its own activation pointer table."""
    import fasteq  # noqa: F401

    x, w, cg, i_dims, u = _make("SMALL_P1", 512, dtype)
    w3 = w.reshape(len(i_dims), u, u)
    grads = [torch.randn(512, _out_rows(i_dims), u, dtype=dtype, device=x.device) for _ in range(2)]
    handle = torch.ops.eqlinear.dgrad_create(w3, i_dims, cg, 512, u, len(i_dims), u, precision)
    try:
        reference = [torch.empty(512, sum(i_dims), u, dtype=dtype, device=x.device) for _ in grads]
        for grad, out in zip(grads, reference):
            torch.ops.eqlinear.dgrad_run(handle, grad, out)
            torch.cuda.synchronize()
        actual = [torch.empty_like(out) for out in reference]
        torch.cuda._sleep(20_000_000)
        for grad, out in zip(grads, actual):
            torch.ops.eqlinear.dgrad_run(handle, grad, out)
        torch.cuda.synchronize()
        assert all(torch.equal(out, ref) for out, ref in zip(actual, reference))
    finally:
        torch.ops.eqlinear.dgrad_destroy(handle)


@pytest.mark.parametrize("precision,dtype", [("fp32", torch.float32), ("tf32", torch.float32), ("fp64", torch.float64)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_no_grad_forward_matches_autograd(precision, dtype, noncontiguous):
    from fasteq.ops.eqlinear import fast_equi_linear

    x, w, cg, i_dims, u = _make("SMALL_P1", 512, dtype)
    if noncontiguous:
        x_view = torch.empty((x.shape[0], x.shape[1] * 2), dtype=dtype, device=x.device)[:, ::2]
        w_view = torch.empty(w.numel() * 2, dtype=dtype, device=w.device)[::2]
        x_view.copy_(x)
        w_view.copy_(w)
        x, w = x_view, w_view
    desc = _FakeDescriptor(i_dims, u, u, cg)
    kwargs = {"compute_precision": precision} if precision != "fp64" else {}
    without_grad = fast_equi_linear(desc, w, x, **kwargs)
    x_grad = x.detach().requires_grad_(True)
    with_grad = fast_equi_linear(desc, w, x_grad, **kwargs)
    assert not without_grad.requires_grad
    assert with_grad.requires_grad
    assert torch.equal(without_grad, with_grad)
    grad = torch.randn_like(with_grad)
    dx, = torch.autograd.grad(with_grad, x_grad, grad)
    x_ref = x.detach().clone().requires_grad_(True)
    ref = _reference(x_ref, w.reshape(len(i_dims), u, u), i_dims, cg, dtype)
    dx_ref, = torch.autograd.grad(ref, x_ref, grad.reshape(ref.shape))
    tolerance = 1e-12 if dtype == torch.float64 else 2e-5 if precision == "tf32" else 1e-6
    assert _rel_fro(dx, dx_ref) < tolerance


def test_no_grad_fast_path_preserves_autograd_dispatch():
    from fasteq.ops.eqlinear import fast_equi_linear

    x, w, cg, i_dims, u = _make("SMALL_P1", 512, torch.float32)
    desc = _FakeDescriptor(i_dims, u, u, cg)
    w.requires_grad_(True)
    out = fast_equi_linear(desc, w, x)
    assert out.requires_grad
    w.requires_grad_(False)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, torch.ones_like(x))
        with pytest.raises(NotImplementedError, match="jvp"):
            fast_equi_linear(desc, w, dual)


@pytest.mark.parametrize("precision,dtype", [(0, torch.float32), (1, torch.float32), (2, torch.float64)])
def test_flat_forward_matches_3d_and_checks_shape(precision, dtype):
    import fasteq  # noqa: F401

    x, w, cg, i_dims, u = _make("SMALL_P1", 512, dtype)
    x3 = x.reshape(512, sum(i_dims), u)
    w3 = w.reshape(len(i_dims), u, u)
    j = _out_rows(i_dims)
    handle = torch.ops.eqlinear.forward_create(w3, i_dims, cg, 512, u, len(i_dims), u, precision)
    try:
        flat = torch.empty(512, j * u, dtype=dtype, device=x.device)
        shaped = torch.empty(512, j, u, dtype=dtype, device=x.device)
        torch.ops.eqlinear.forward_run_flat(handle, x, flat)
        torch.ops.eqlinear.forward_run(handle, x3, shaped)
        assert torch.equal(flat, shaped.reshape(512, -1))
        with pytest.raises(RuntimeError, match="flat x shape"):
            torch.ops.eqlinear.forward_run_flat(handle, x[:, :-u].contiguous(), flat)
        with pytest.raises(RuntimeError, match="flat out shape"):
            torch.ops.eqlinear.forward_run_flat(handle, x, flat[:, :-u].contiguous())
        with pytest.raises(RuntimeError, match="contiguous"):
            torch.ops.eqlinear.forward_run_flat(handle, x.t(), flat)
    finally:
        torch.ops.eqlinear.forward_destroy(handle)


def test_tf32_grouped_activation_view_rebind_and_fallback():
    import fasteq  # noqa: F401

    x, w, cg, dims, u = _make("SMALL_P1", 512, torch.float32)
    w3 = w.reshape(len(dims), u, u)
    inputs = [x.reshape(512, sum(dims), u), torch.randn(512, sum(dims), u, device=x.device)]
    handle = torch.ops.eqlinear.forward_create(w3, dims, cg, 512, u, len(dims), u, 0)
    try:
        assert torch.ops.eqlinear.forward_plan_info(handle)[1:3] == (0, 1)
        refs = [torch.empty(512, _out_rows(dims), u, device=x.device) for _ in inputs]
        for source, out in zip(inputs, refs):
            torch.ops.eqlinear.forward_run(handle, source, out)
            torch.cuda.synchronize()
        outs = [torch.full_like(out, float("nan")) for out in refs]
        torch.cuda._sleep(20_000_000)
        for source, out in zip(inputs, outs):
            torch.ops.eqlinear.forward_run(handle, source, out)
        torch.cuda.synchronize()
        assert all(torch.equal(out, ref) for out, ref in zip(outs, refs))
    finally:
        torch.ops.eqlinear.forward_destroy(handle)

    mixed = [1, 1, 3, 7]
    fallback = torch.ops.eqlinear.forward_create(w3, mixed, cg, 512, u, len(mixed), u, 0)
    try:
        assert torch.ops.eqlinear.forward_plan_info(fallback)[1:3] == (0, 0)
        source = torch.randn(512, sum(mixed), u, device=x.device)
        actual = torch.empty(512, _out_rows(mixed), u, device=x.device)
        torch.ops.eqlinear.forward_run(fallback, source, actual)
        expected = _reference(source, w3, mixed, cg, torch.float64)
        assert _rel_fro(actual, expected) < 2e-5
    finally:
        torch.ops.eqlinear.forward_destroy(fallback)


def test_tf32_grouped_activation_view_on_nondefault_stream():
    import fasteq  # noqa: F401

    x, w, cg, dims, u = _make("SMALL_P1", 512, torch.float32)
    x3 = x.reshape(512, sum(dims), u)
    w3 = w.reshape(len(dims), u, u)
    stream = torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(stream):
        handle = torch.ops.eqlinear.forward_create(w3, dims, cg, 512, u, len(dims), u, 0)
        try:
            assert torch.ops.eqlinear.forward_plan_info(handle)[1:3] == (0, 1)
            out = torch.empty(512, _out_rows(dims), u, device=x.device)
            torch.ops.eqlinear.forward_run(handle, x3, out)
            stream.synchronize()
        finally:
            torch.ops.eqlinear.forward_destroy(handle)
    expected = _reference(x3, w3, dims, cg, torch.float64)
    assert _rel_fro(out, expected) < 2e-5


def test_tf32_grouped_activation_view_concurrent_streams():
    from fasteq.ops.eqlinear import eqlinear_clear_plan_cache, fast_equi_linear

    x, w, cg, dims, u = _make("SMALL_P1", 512, torch.float32)
    inputs = [x, torch.randn_like(x)]
    desc = _FakeDescriptor(dims, u, u, cg)
    streams = [torch.cuda.Stream(device=x.device) for _ in inputs]
    default = torch.cuda.current_stream(x.device)
    outputs = []
    eqlinear_clear_plan_cache()
    for stream, source in zip(streams, inputs):
        stream.wait_stream(default)
        with torch.cuda.stream(stream):
            outputs.append(fast_equi_linear(desc, w, source))
    torch.cuda.synchronize()
    for source, out in zip(inputs, outputs):
        ref = _reference(source, w.reshape(len(dims), u, u), dims, cg, torch.float64)
        assert _rel_fro(out, ref.reshape(512, -1)) < 2e-5
    eqlinear_clear_plan_cache()


def test_output_row_cache_tracks_descriptor_mutations(monkeypatch):
    from fasteq.ops import eqlinear

    original = torch.ops.eqlinear.out_rows
    calls = []

    def record(dims):
        calls.append(tuple(dims))
        return original(dims)

    monkeypatch.setattr(torch.ops.eqlinear, "out_rows", record)
    eqlinear._cached_out_rows.cache_clear()
    try:
        dims = [1, 3, 5, 7]
        desc = _FakeDescriptor(dims, 96, 96, [.01, .011, .012, .013])
        first = eqlinear._descriptor_problem(desc)
        desc.paths[0].coefficients = .125
        second = eqlinear._descriptor_problem(desc)
        assert isinstance(first[0], tuple) and isinstance(first[1], tuple)
        assert first[4] == second[4] == 16
        assert first[1][0] == .01 and second[1][0] == .125
        assert calls == [tuple(dims)]

        changed = _FakeDescriptor([1, 3, 3, 7], 96, 96, [.02] * 4)
        desc.__dict__.update(changed.__dict__)
        assert eqlinear._descriptor_problem(desc)[4] == 11
        assert calls[-1] == (1, 3, 3, 7)

        desc.operands[2] = ((1,),)
        with pytest.raises(NotImplementedError):
            eqlinear._descriptor_problem(desc)

        oversized = _FakeDescriptor([1] * 33, 96, 96, [.01] * 33)
        eqlinear._descriptor_problem(oversized)
        eqlinear._descriptor_problem(oversized)
        assert calls[-2:] == [(1,) * 33, (1,) * 33]
    finally:
        eqlinear._cached_out_rows.cache_clear()
