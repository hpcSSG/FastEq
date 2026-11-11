"""Weight-layout routing/cache regressions; CPU checks need no CUDA extension."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import statistics
import time

import pytest
import torch


class WeightLayoutCase:
    DIMS = (1, 3)
    CG = (0.01, 0.03)

    @staticmethod
    def source():
        path = Path(__file__).resolve().parents[1] / 'fasteq/ops/eqlinear.py'
        spec = importlib.util.spec_from_file_location('eqlinear_layout_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def weight(layout, u, dtype=torch.float32, device='cpu'):
        weight = torch.randn(2, u, u, dtype=dtype, device=device)
        if layout == 'flat':
            return weight.flatten()
        if layout == 'transpose':
            return weight.transpose(1, 2)
        if layout == 'slice':
            storage = torch.empty(weight.numel() * 2, dtype=dtype, device=device)
            view = storage[::2]
            view.copy_(weight.flatten())
            return view
        return weight

    @classmethod
    def reference(cls, x, weight):
        u = x.shape[1] // sum(cls.DIMS)
        shaped = weight.reshape(2, u, u)
        chunks = x.reshape(x.shape[0], sum(cls.DIMS), u).split(cls.DIMS, dim=1)
        return torch.cat([chunk @ shaped[p] * cls.CG[p]
                          for p, chunk in enumerate(chunks)], dim=1).flatten(1)

    @staticmethod
    def relative_error(actual, expected):
        return (actual.double() - expected.double()).norm() / expected.double().norm()


class CpuPlanRuntime:
    """Small execution double: exercises real Python dispatch and plan-cache logic."""

    def __init__(self):
        self.plans = {}
        self.creates = {'f': 0, 'd': 0}
        self.runs = {'flat': 0, 'shaped': 0}

    def forward_create(self, weight, *args):
        self.creates['f'] += 1
        handle = len(self.plans) + 1
        self.plans[handle] = weight.clone()
        return handle

    def dgrad_create(self, weight, *args):
        self.creates['d'] += 1
        return -self.creates['d']

    def forward_run_flat(self, handle, x, out):
        self.runs['flat'] += 1
        out.copy_(WeightLayoutCase.reference(x, self.plans[handle]))

    def forward_run(self, handle, x, out):
        self.runs['shaped'] += 1
        value = WeightLayoutCase.reference(x.flatten(1), self.plans[handle])
        out.copy_(value.reshape_as(out))


class TestWeightLayout:
    @pytest.fixture
    def cpu_runtime(self, monkeypatch):
        module = WeightLayoutCase.source()
        native = CpuPlanRuntime()
        monkeypatch.setattr(module, '_current_stream_ptr', lambda device: 17)
        monkeypatch.setattr(module, '_descriptor_problem',
                            lambda descriptor: (WeightLayoutCase.DIMS, WeightLayoutCase.CG, 2, 2, 4))
        for name in ('forward_create', 'forward_run', 'forward_run_flat', 'dgrad_create'):
            monkeypatch.setattr(torch.ops.eqlinear, name, getattr(native, name), raising=False)
        monkeypatch.setattr(torch.ops.eqlinear, 'forward_destroy', lambda handle: None, raising=False)
        monkeypatch.setattr(torch.ops.eqlinear, 'dgrad_destroy', lambda handle: None, raising=False)
        return module, native

    @pytest.mark.parametrize('layout', ['flat', 'shaped', 'transpose', 'slice'])
    def test_source_layout_keys_and_fallback(self, cpu_runtime, layout):
        module, native = cpu_runtime
        weight = WeightLayoutCase.weight(layout, 2)
        x = torch.randn(2, 8)
        expected = WeightLayoutCase.reference(x, weight)
        for _ in range(2):
            torch.testing.assert_close(module._forward(weight, x, object(), 1), expected)
        assert native.creates['f'] == 1
        assert native.runs['flat'] == (2 if weight.is_contiguous() else 0)
        assert native.runs['shaped'] == (0 if weight.is_contiguous() else 2)
        entry = module._PLAN_CACHE_LAST_ENTRY
        assert entry['key'][10:13] == (tuple(weight.shape), tuple(weight.stride()), weight.dtype)
        assert module._PLAN_CACHE_BYTES == entry['bytes']

        # Changing x layout must reuse the same weight plan in the 3D fallback.
        strided_x = torch.empty(2, 16)[:, ::2]
        strided_x.copy_(x)
        torch.testing.assert_close(module._forward(weight, strided_x, object(), 1), expected)
        assert native.creates['f'] == 1
        assert native.runs['shaped'] >= 1

        # A source mutation must invalidate prepared weights even when storage is shared.
        weight.add_(0.5)
        torch.testing.assert_close(module._forward(weight, x, object(), 1),
                                   WeightLayoutCase.reference(x, weight))
        assert native.creates['f'] == 2

    @pytest.mark.parametrize('layout', ['flat', 'transpose', 'slice'])
    def test_dgrad_uses_same_source_metadata(self, cpu_runtime, layout):
        module, native = cpu_runtime
        weight = WeightLayoutCase.weight(layout, 2)
        prepared = weight.reshape(2, 2, 2).contiguous()
        args = (weight, prepared, (2, 4, 2), WeightLayoutCase.DIMS, WeightLayoutCase.CG, 1, 4, 2)
        first = module._dgrad_handle(*args)
        assert module._dgrad_handle(*args) == first
        assert native.creates['d'] == 1
        assert module._PLAN_CACHE_LAST_ENTRY['key'][10:13] == (
            tuple(weight.shape), tuple(weight.stride()), weight.dtype)

    @pytest.mark.parametrize('layout', ['flat', 'shaped', 'transpose', 'slice'])
    @pytest.mark.parametrize('precision,dtype', [('fp32', torch.float32),
                                               ('tf32', torch.float32), ('fp64', torch.float64)])
    def test_cuda_outputs_gradients_and_route(self, monkeypatch, layout, precision, dtype):
        if not torch.cuda.is_available():
            pytest.skip('CUDA is required')
        import fasteq  # noqa: F401 -- registers native operators
        import cuequivariance as cue

        module = WeightLayoutCase.source()
        irreps = cue.Irreps('O3', '96x0e+96x1o')
        original = cue.descriptors.linear(irreps, irreps).polynomial.operations[0][1]
        descriptor = cue.SegmentedTensorProduct.from_dict(original.to_dict())
        # Use explicit coefficients so the reference is independent of factory normalization.
        for path, coefficient in zip(descriptor.paths, WeightLayoutCase.CG):
            path.coefficients[...] = coefficient
        weight = WeightLayoutCase.weight(layout, 96, dtype, 'cuda')
        x = torch.randn(512, 4 * 96, dtype=dtype, device='cuda')
        runs = {'flat': 0, 'shaped': 0}
        original_flat = torch.ops.eqlinear.forward_run_flat
        original_shaped = torch.ops.eqlinear.forward_run

        def run_flat(*args):
            runs['flat'] += 1
            return original_flat(*args)

        def run_shaped(*args):
            runs['shaped'] += 1
            return original_shaped(*args)

        monkeypatch.setattr(torch.ops.eqlinear, 'forward_run_flat', run_flat)
        monkeypatch.setattr(torch.ops.eqlinear, 'forward_run', run_shaped)
        kwargs = {} if precision == 'fp64' else {'compute_precision': precision}
        tolerance = 1e-12 if dtype == torch.float64 else 2e-5 if precision == 'tf32' else 1e-6
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            first = module.fast_equi_linear(descriptor, weight, x, **kwargs)
            second = module.fast_equi_linear(descriptor, weight, x, **kwargs)
            assert torch.equal(first, second)
            assert runs['flat'] == (2 if weight.is_contiguous() else 0)
            assert runs['shaped'] == (0 if weight.is_contiguous() else 2)
            expected = WeightLayoutCase.reference(x.double(), weight.double())
            assert WeightLayoutCase.relative_error(first, expected) < tolerance

            grad_x = x.detach().requires_grad_(True)
            actual = module.fast_equi_linear(descriptor, weight, grad_x, **kwargs)
            grad = torch.randn_like(actual)
            dx, = torch.autograd.grad(actual, grad_x, grad)
            ref_x = x.detach().clone().requires_grad_(True)
            reference = WeightLayoutCase.reference(ref_x, weight)
            dx_ref, = torch.autograd.grad(reference, ref_x, grad)
            assert WeightLayoutCase.relative_error(dx, dx_ref) < tolerance
            assert torch.equal(actual.detach(), first)

            weight.add_(0.125)
            changed = module.fast_equi_linear(descriptor, weight, x, **kwargs)
            expected = WeightLayoutCase.reference(x.double(), weight.double())
            assert WeightLayoutCase.relative_error(changed, expected) < tolerance
        finally:
            module.eqlinear_clear_plan_cache()
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


class WeightLayoutBenchmark:
    """Optional measurement interface; run only after selecting an idle GPU.

    ``measure(descriptor, weight, x, precision='fp32', repeats=20)`` reports
    public enqueue+completion wall time and the fallback weight-copy GPU time.
    Call separately with contiguous, transposed and sliced weights of equal values.
    """

    @staticmethod
    def measure(descriptor, weight, x, precision='fp32', repeats=20):
        from fasteq.ops import eqlinear

        kwargs = {} if weight.dtype == torch.float64 else {'compute_precision': precision}
        dims, _, u, v, _ = eqlinear._descriptor_problem(descriptor)
        prepared = weight.reshape(len(dims), u, v).contiguous()
        copied = prepared.data_ptr() != weight.data_ptr()
        for _ in range(5):
            eqlinear.fast_equi_linear(descriptor, weight, x, **kwargs)
        timings = []
        for _ in range(5):
            torch.cuda.synchronize(weight.device)
            start = time.perf_counter_ns()
            for _ in range(repeats):
                eqlinear.fast_equi_linear(descriptor, weight, x, **kwargs)
            torch.cuda.synchronize(weight.device)
            timings.append((time.perf_counter_ns() - start) / (1000 * repeats))
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(repeats):
            weight.reshape(len(dims), u, v).contiguous()
        end.record()
        end.synchronize()
        return {'public_us': statistics.median(timings),
                'public_groups_us': timings,
                'weight_prepare_gpu_us': begin.elapsed_time(end) * 1000 / repeats,
                'weight_copy': copied,
                'copied_bytes_per_call': weight.numel() * weight.element_size() if copied else 0,
                'flat_eligible': x.is_contiguous() and weight.is_contiguous()}
