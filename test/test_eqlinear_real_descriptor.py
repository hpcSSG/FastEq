"""Public eqlinear smoke tests using a real cuEquivariance descriptor."""
import cuequivariance as cue
import pytest
import torch

from test_eqlinear import _reference, _rel_fro

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')


class TestRealDescriptor:
    @staticmethod
    def descriptor():
        irreps = cue.Irreps('O3', '96x0e+96x1o+96x2e+96x3o')
        return cue.descriptors.linear(irreps, irreps).polynomial.operations[0][1]

    def test_geometry_cache_reuses_only_unchanged_descriptor_structure(self, monkeypatch):
        from fasteq.ops import eqlinear

        eqlinear._DESCRIPTOR_GEOMETRY_CACHE.clear()
        descriptor = self.descriptor()
        original = eqlinear._descriptor_geometry_uncached
        calls = []

        def count_geometry(value):
            calls.append(value)
            return original(value)

        monkeypatch.setattr(eqlinear, "_descriptor_geometry_uncached", count_geometry)
        first = eqlinear._descriptor_problem(descriptor)
        descriptor.paths[0].coefficients[...] *= 1.5
        second = eqlinear._descriptor_problem(descriptor)
        assert len(calls) == 1
        assert first[0] == second[0] and first[2:] == second[2:]
        assert first[1] != second[1]

        dimensions = descriptor.get_dimensions_dict()
        u_dimensions = dimensions["u"]
        dimensions["u"] = set()
        with pytest.raises(ValueError, match="uniform u"):
            eqlinear._descriptor_problem(descriptor)
        dimensions["u"] = u_dimensions
        assert len(calls) == 2
        eqlinear._descriptor_problem(descriptor)
        assert len(calls) == 2

        descriptor.set_paths(list(descriptor.paths))
        assert eqlinear._descriptor_problem(descriptor)[0] == first[0]
        assert len(calls) == 3

        descriptor.operands[1].add_segment(descriptor.operands[1].segments[-1])
        with pytest.raises(ValueError, match="i_dims disagree"):
            eqlinear._descriptor_problem(descriptor)
        assert len(calls) == 4
        eqlinear._DESCRIPTOR_GEOMETRY_CACHE.clear()

    def test_geometry_cache_checks_cached_mutable_dimension_sets(self):
        from fasteq.ops import eqlinear

        eqlinear._DESCRIPTOR_GEOMETRY_CACHE.clear()
        descriptor = self.descriptor()
        eqlinear._descriptor_problem(descriptor)
        u_dimensions = descriptor.get_dimensions_dict()["u"]
        original_dimensions = tuple(u_dimensions)
        u_dimensions.clear()
        try:
            with pytest.raises(ValueError, match="uniform u"):
                eqlinear._descriptor_problem(descriptor)
        finally:
            u_dimensions.update(original_dimensions)
        eqlinear._DESCRIPTOR_GEOMETRY_CACHE.clear()

    @pytest.mark.parametrize('precision,dtype', [('fp32', torch.float32),
                                                  ('tf32', torch.float32),
                                                  ('fp64', torch.float64)])
    def test_forward_backward_and_mutation(self, precision, dtype, monkeypatch):
        from fasteq.ops.eqlinear import _descriptor_problem, eqlinear_clear_plan_cache, fast_equi_linear

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(407)
        desc = self.descriptor()
        dims, cg, u, v, j = _descriptor_problem(desc)
        batch = 512
        x = torch.randn(batch, sum(dims) * u, dtype=dtype, device='cuda', requires_grad=True)
        w = torch.randn(len(dims) * u * v, dtype=dtype, device='cuda')
        kwargs = {'compute_precision': precision} if precision != 'fp64' else {}
        out = fast_equi_linear(desc, w, x, **kwargs)
        ref = _reference(x, w.reshape(len(dims), u, v), dims, cg, torch.float64).reshape(batch, -1)
        threshold = 1e-12 if precision == 'fp64' else 2e-5 if precision == 'tf32' else 1e-6
        assert out.shape == (batch, j * v)
        error = _rel_fro(out, ref)
        assert error < threshold
        if precision == 'fp32':
            ref_fp32 = _reference(x, w.reshape(len(dims), u, v), dims, cg, dtype).reshape(batch, -1)
            assert error <= 1.05 * _rel_fro(ref_fp32, ref)
        grad = torch.randn_like(out)
        dx, = torch.autograd.grad(out, x, grad)
        x_ref = x.detach().clone().requires_grad_(True)
        ref_grad = _reference(x_ref, w.reshape(len(dims), u, v), dims, cg, dtype)
        dx_ref, = torch.autograd.grad(ref_grad, x_ref, grad.reshape(ref_grad.shape))
        assert _rel_fro(dx, dx_ref) < threshold

        eqlinear_clear_plan_cache()
        recreated = fast_equi_linear(desc, w, x.detach(), **kwargs)
        assert torch.equal(out.detach(), recreated)
        eqlinear_clear_plan_cache()

        desc.paths[0].coefficients[...] *= 1.5
        dims2, cg2, _, _, _ = _descriptor_problem(desc)
        assert dims2 == dims and cg2 != cg
        changed = fast_equi_linear(desc, w, x.detach(), **kwargs)
        changed_ref = _reference(x.detach(), w.reshape(len(dims), u, v), dims, cg2, torch.float64)
        assert _rel_fro(changed, changed_ref.reshape(batch, -1)) < threshold
