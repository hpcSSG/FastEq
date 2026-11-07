"""Real cue 0.8 cache replacement/mutation regressions; no GPU is required."""
from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import cuequivariance as cue
import pytest


class DescriptorProxy:
    """A compatible descriptor is parseable without promising canonical mutability."""

    def __init__(self, descriptor):
        self._descriptor = descriptor

    def __getattr__(self, name):
        return getattr(self._descriptor, name)


class PrivateValueProxy:
    def __init__(self, value):
        self._value = value

    def __getattr__(self, name):
        return getattr(self._value, name)

    def __iter__(self):
        return iter(self._value)

    def __hash__(self):
        return hash(self._value)


class TestDescriptorMutation:
    @staticmethod
    def source():
        source = Path(__file__).resolve().parents[1] / 'fasteq/ops/eqlinear.py'
        spec = importlib.util.spec_from_file_location('eqlinear_descriptor_under_test', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @pytest.fixture
    def problem(self, monkeypatch):
        module = self.source()
        irreps = cue.Irreps('O3', '96x0e+96x1o+96x2e+96x3o')
        original = cue.descriptors.linear(irreps, irreps).polynomial.operations[0][1]
        descriptor = cue.SegmentedTensorProduct.from_dict(original.to_dict())
        cue.SegmentedTensorProduct.get_dimensions_dict.cache_clear()
        # These tests exercise the actual geometry parser. Only the CUDA row-count
        # primitive is replaced by its deterministic CPU grouping formula.
        monkeypatch.setattr(module, '_cached_out_rows', self.output_rows)
        calls = []
        parse = module._descriptor_geometry_uncached

        def record_parse(value):
            calls.append(value)
            return parse(value)

        monkeypatch.setattr(module, '_descriptor_geometry_uncached', record_parse)
        yield module, descriptor, calls
        module._DESCRIPTOR_GEOMETRY_CACHE.clear()
        cue.SegmentedTensorProduct.get_dimensions_dict.cache_clear()

    @staticmethod
    def output_rows(dims):
        return sum(value for index, value in enumerate(dims)
                   if index == 0 or value != dims[index - 1])

    def test_real_cache_clear_replaces_returned_dict(self, problem):
        module, descriptor, calls = problem
        expected = module._descriptor_problem(descriptor)
        old_dimensions = descriptor.get_dimensions_dict()
        # This uses the real functools.cache API, not a mocked getter.
        type(descriptor).get_dimensions_dict.cache_clear()
        new_dimensions = descriptor.get_dimensions_dict()
        assert new_dimensions is not old_dimensions
        assert new_dimensions == old_dimensions
        assert module._descriptor_problem(descriptor) == expected
        assert len(calls) == 2
        new_dimensions['u'] = set()
        with pytest.raises(ValueError, match='uniform u'):
            module._descriptor_problem(descriptor)
        assert len(calls) == 3

    def test_changed_replacement_dict_cannot_hide_behind_old_dict(self, problem):
        module, descriptor, calls = problem
        module._descriptor_problem(descriptor)
        old_dimensions = descriptor.get_dimensions_dict()
        type(descriptor).get_dimensions_dict.cache_clear()
        new_dimensions = descriptor.get_dimensions_dict()
        new_dimensions['u'] = set()
        assert old_dimensions['u'] == {96}
        with pytest.raises(ValueError, match='uniform u'):
            module._descriptor_problem(descriptor)
        assert len(calls) == 2

    @pytest.mark.parametrize('replace_set', [False, True])
    def test_dimension_set_mutations_are_reparsed(self, problem, replace_set):
        module, descriptor, calls = problem
        module._descriptor_problem(descriptor)
        dimensions = descriptor.get_dimensions_dict()
        if replace_set:
            dimensions['v'] = set()
        else:
            dimensions['v'].clear()
        with pytest.raises(ValueError, match='uniform v'):
            module._descriptor_problem(descriptor)
        assert len(calls) == 2

    def test_coefficient_only_change_reuses_geometry_but_updates_cg(self, problem):
        module, descriptor, calls = problem
        before = module._descriptor_problem(descriptor)
        old_dimensions = descriptor.get_dimensions_dict()
        descriptor.paths[0].coefficients[...] *= 1.5
        new_dimensions = descriptor.get_dimensions_dict()
        assert new_dimensions is not old_dimensions
        assert new_dimensions == old_dimensions
        after = module._descriptor_problem(descriptor)
        assert before[0] == after[0] and before[2:] == after[2:]
        assert before[1] != after[1]
        assert len(calls) == 1

    @pytest.mark.parametrize('structure', ['operands', 'paths', 'segments'])
    def test_real_structural_mutators_invalidate(self, problem, structure):
        module, descriptor, calls = problem
        before = module._descriptor_problem(descriptor)
        if structure == 'operands':
            descriptor.set_operand(1, descriptor.operands[1])
        elif structure == 'paths':
            descriptor.set_paths(list(descriptor.paths))
        else:
            descriptor.operands[1].add_segment(descriptor.operands[1].segments[-1])
        if structure == 'segments':
            with pytest.raises(ValueError, match='i_dims disagree'):
                module._descriptor_problem(descriptor)
        else:
            assert module._descriptor_problem(descriptor) == before
        assert len(calls) == 2

    @pytest.mark.parametrize('spoof_name', [False, True])
    def test_unknown_descriptor_never_uses_geometry_cache(self, problem, spoof_name):
        module, descriptor, calls = problem
        proxy_type = DescriptorProxy
        if spoof_name:
            proxy_type = type('SegmentedTensorProduct', (DescriptorProxy,),
                              {'__module__': type(descriptor).__module__})
        proxy = proxy_type(descriptor)
        first = module._descriptor_problem(proxy)
        second = module._descriptor_problem(proxy)
        assert first == second
        assert len(calls) == 2
        descriptor.paths[0].coefficients[...] *= 1.5
        assert module._descriptor_problem(proxy)[1] != first[1]
        assert len(calls) == 3
        assert module._DESCRIPTOR_GEOMETRY_CACHE._entry is None

    @pytest.mark.parametrize('private_type', ['operand', 'path'])
    def test_spoofed_private_types_always_fall_back(self, problem, private_type):
        module, descriptor, calls = problem
        if private_type == 'operand':
            value = descriptor.operands[1]
            proxy_type = type('SegmentedOperand', (PrivateValueProxy,),
                              {'__module__': type(value).__module__})
            operands = list(descriptor.operands_and_subscripts)
            operands[1] = (proxy_type(value), operands[1][1])
            object.__setattr__(descriptor, 'operands_and_subscripts', tuple(operands))
        else:
            value = descriptor.paths[0]
            proxy_type = type('Path', (PrivateValueProxy,),
                              {'__module__': type(value).__module__})
            descriptor.set_paths([proxy_type(value), *descriptor.paths[1:]])
        type(descriptor).get_dimensions_dict.cache_clear()
        assert module._descriptor_problem(descriptor) == module._descriptor_problem(descriptor)
        assert len(calls) == 2
        assert module._DESCRIPTOR_GEOMETRY_CACHE._entry is None

    def test_noncanonical_dimension_container_is_not_cached(self, problem):
        module, descriptor, calls = problem
        dimensions = descriptor.get_dimensions_dict()
        dimensions['u'] = (96,)  # Full parsing supports iterable integer dimensions.
        before = module._descriptor_problem(descriptor)
        assert module._descriptor_problem(descriptor) == before
        assert len(calls) == 2
        assert module._DESCRIPTOR_GEOMETRY_CACHE._entry is None

    def test_missing_canonical_import_keeps_full_parser_available(self, problem, monkeypatch):
        _, descriptor, _ = problem
        original_import = builtins.__import__

        def missing_path(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'cuequivariance.segmented_polynomials' and 'Path' in fromlist:
                raise ImportError('Path is unavailable in this cue version')
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, '__import__', missing_path)
        module = self.source()
        monkeypatch.setattr(module, '_cached_out_rows', self.output_rows)
        assert module._CueDescriptor is None
        first = module._descriptor_problem(descriptor)
        assert module._descriptor_problem(descriptor) == first
        assert module._DESCRIPTOR_GEOMETRY_CACHE._entry is None
