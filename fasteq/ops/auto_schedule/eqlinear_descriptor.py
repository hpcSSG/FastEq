"""Descriptor validation, explicit path geometry and mutation-aware caching for eqlinear."""
from __future__ import annotations

import operator
import weakref
from functools import lru_cache

import torch

try:
    from cuequivariance import SegmentedOperand as _CueOperand
    from cuequivariance import SegmentedTensorProduct as _CueDescriptor
    from cuequivariance.segmented_polynomials import Path as _CuePath
    from numpy import ndarray as _CoefficientArray
except ImportError:
    # Different cue versions may still implement the public descriptor protocol.
    # Keep full parsing available, but disable structural caching.
    _CueOperand = _CueDescriptor = _CuePath = _CoefficientArray = None


class _DescriptorGeometryCache:
    """Reuse geometry only for canonical descriptors with unchanged immutable sources."""

    def __init__(self):
        self._entry = None

    @staticmethod
    def _coefficient_signature(paths):
        if len(paths) == 1:
            coefficient = paths[0].coefficients
            if type(coefficient) is _CoefficientArray:
                # dtype can change in-place without changing bytes or cue's hash.
                return (coefficient.dtype.str, coefficient.shape, coefficient.tobytes())
            return (coefficient.tobytes(),)
        return tuple(path.coefficients.tobytes() for path in paths)

    @staticmethod
    def _single_coefficient(paths):
        if len(paths) == 1 and type(paths[0].coefficients) is _CoefficientArray:
            return (float(paths[0].coefficients),)
        return None

    @staticmethod
    def _dimension_state(dimensions, path_count):
        # cuEquivariance exposes its cached dict/sets directly. Unknown mutable
        # representations must use the full parser rather than become cache keys.
        if type(dimensions) is not dict:
            return None
        names = ("i", "u", "v") if path_count <= 4 else ("u", "v")
        values = []
        for name in names:
            dims = dimensions.get(name)
            if type(dims) is not set:
                return None
            if len(dims) == 1:
                value = next(iter(dims))
                if type(value) is not int:
                    return None
                values.append((value,))
            else:
                if any(type(value) is not int for value in dims):
                    return None
                values.append(tuple(dims))
        return tuple(values)

    @staticmethod
    def _is_supported(descriptor):
        return type(descriptor) is _CueDescriptor

    def get(self, descriptor):
        entry = self._entry
        if entry is None or entry["descriptor"]() is not descriptor:
            return None
        if not self._is_supported(descriptor):
            return None
        operands = descriptor.operands_and_subscripts
        paths = descriptor.paths
        if operands is not entry["operands"] or paths is not entry["paths"]:
            return None
        if descriptor.coefficient_subscripts != entry["coefficient_subscripts"]:
            return None
        for (operand, _), old_segments in zip(operands, entry["segments"]):
            if type(operand) is not _CueOperand or operand.segments is not old_segments:
                return None
        if any(type(path) is not _CuePath for path in paths):
            return None

        # Always retrieve the current object: functools.cache.cache_clear() can
        # replace this dict without changing any descriptor/coefficient fields.
        dimensions = descriptor.get_dimensions_dict()
        dimension_state = self._dimension_state(dimensions, len(paths))
        if dimension_state is None or dimension_state != entry["dimension_state"]:
            # Keep the last validated snapshot while this call takes the full parser.
            # A rejected temporary mutation can then be restored without rebuilding;
            # every lookup still verifies the current dict and its values first.
            return None
        if tuple(getattr(path, "indices", None) for path in paths) != entry["path_indices"]:
            return None
        coefficient_signature = self._coefficient_signature(paths)
        if coefficient_signature == entry["coefficient_signature"]:
            if dimensions is not entry["dimensions"]:
                self.clear()
                return None
        else:
            # Coefficients participate in cue's hash, so legal coefficient-only
            # changes produce a fresh dict too. Equal geometry remains reusable.
            entry["dimensions"] = dimensions
            entry["coefficient_signature"] = coefficient_signature
            entry["cg"] = self._single_coefficient(paths)
        return entry["geometry"]

    def put(self, descriptor, geometry):
        self.clear()
        if not self._is_supported(descriptor):
            return
        operands = descriptor.operands_and_subscripts
        paths = descriptor.paths
        if type(operands) is not tuple or type(paths) is not tuple:
            return
        if any(type(path) is not _CuePath
               or not hasattr(path.coefficients, "tobytes") for path in paths):
            return
        segments = []
        for pair in operands:
            if type(pair) is not tuple or len(pair) != 2:
                return
            operand, subscripts = pair
            if (type(operand) is not _CueOperand or type(subscripts) is not str
                    or type(operand.segments) is not tuple
                    or any(type(segment) is not tuple
                           or any(type(value) is not int for value in segment)
                           for segment in operand.segments)):
                return
            segments.append(operand.segments)
        dimensions = descriptor.get_dimensions_dict()
        dimension_state = self._dimension_state(dimensions, len(paths))
        if dimension_state is None:
            return
        try:
            descriptor_ref = weakref.ref(descriptor, self._release)
        except TypeError:
            return
        self._entry = {"descriptor": descriptor_ref, "operands": operands, "paths": paths,
                       "segments": tuple(segments), "geometry": geometry,
                       "dimensions": dimensions, "dimension_state": dimension_state,
                       "coefficient_signature": self._coefficient_signature(paths),
                       "cg": self._single_coefficient(paths),
                       "coefficient_subscripts": descriptor.coefficient_subscripts,
                       "path_indices": tuple(getattr(path, "indices", None) for path in paths)}

    def problem(self, descriptor):
        geometry = self.get(descriptor)
        if geometry is None:
            geometry = _descriptor_geometry_uncached(descriptor)
            self.put(descriptor, geometry)
        entry = self._entry
        # Reuse the scalar conversion only after get() has checked the current
        # coefficient bytes AND the live dimension dict, including cache_clear.
        cg = (entry["cg"] if entry is not None and entry["cg"] is not None
              else tuple(float(path.coefficients) for path in descriptor.paths))
        i_dims, u, v, j_rows = geometry
        return i_dims, cg, u, v, j_rows

    def clear(self):
        self._entry = None

    def _release(self, descriptor_ref):
        if self._entry is not None and self._entry["descriptor"] is descriptor_ref:
            self._entry = None


def _operand_segments(operand):
    """Convert a cuequivariance operand's segments to immutable int tuples."""
    segments = getattr(operand, "segments", operand)
    if type(segments) is tuple and all(
            type(segment) is tuple and all(type(value) is int for value in segment)
            for segment in segments):
        return segments
    return tuple(tuple(operator.index(value) for value in segment) for segment in operand)


def _single_dimension(descriptor, name):
    first = None
    found = False
    unique = None
    for raw_value in descriptor.get_dims(name):
        value = operator.index(raw_value)
        if not found:
            first = value
            found = True
        elif value != first:
            if unique is None:
                unique = [first]
            if value not in unique:
                unique.append(value)
    if unique is not None:
        raise ValueError(f"eqlinear requires uniform {name}; got {tuple(unique)}")
    if not found:
        raise ValueError(f"eqlinear requires uniform {name}; got ()")
    return first

@lru_cache(maxsize=1024)
def _cached_out_rows(i_dims):
    """Cache output grouping by immutable dimensions, not mutable descriptors."""
    return int(torch.ops.eqlinear.out_rows(i_dims))


class _PathDimensions(tuple):
    """Immutable path lengths plus full operand extents and explicit row mapping."""

    def __new__(cls, dims, input_rows, output_rows, mapping):
        value = super().__new__(cls, dims)
        value.input_rows = input_rows
        value.output_rows = output_rows
        value.mapping = tuple(mapping)
        value.cache_key = (tuple(dims), input_rows, output_rows, value.mapping)
        return value


def _geometry_key(dims):
    return dims.cache_key if isinstance(dims, _PathDimensions) else tuple(dims)


def _input_rows(dims):
    return dims.input_rows if isinstance(dims, _PathDimensions) else sum(dims)


def _mapped_geometry(descriptor, input_segments, output_segments, u, v):
    subscripts = tuple(subscripts for _, subscripts in descriptor.operands_and_subscripts)
    if subscripts != ("uv", "iu", "iv") or descriptor.coefficient_subscripts:
        raise NotImplementedError("eqlinear requires scalar coefficients and uv,iu,iv subscripts")
    weight_segments = _operand_segments(descriptor.operands[0])
    paths = descriptor.paths
    if len(weight_segments) != len(paths):
        raise NotImplementedError("eqlinear requires one weight block per path")
    if (any(len(seg) != 2 or seg[1] != u for seg in input_segments)
            or any(len(seg) != 2 or seg[1] != v for seg in output_segments)
            or any(tuple(seg) != (u, v) for seg in weight_segments)):
        raise ValueError("eqlinear: inconsistent uniform operand segment dimensions")
    dimensions = descriptor.get_dimensions_dict()
    if set(dimensions["i"]) != {seg[0] for seg in input_segments + output_segments}:
        raise ValueError("eqlinear: descriptor i_dims disagree with the operand segments")
    in_offsets, out_offsets = [], []
    I = J = 0
    for seg in input_segments:
        in_offsets.append(I)
        I += seg[0]
    for seg in output_segments:
        out_offsets.append(J)
        J += seg[0]
    dims, mapping = [], []
    for path in paths:
        wp, ip, op = (operator.index(index) for index in path.indices)
        if not (0 <= wp < len(weight_segments) and 0 <= ip < len(input_segments)
                and 0 <= op < len(output_segments)):
            raise ValueError("eqlinear: invalid descriptor path index")
        i = input_segments[ip][0]
        if i <= 0 or i != output_segments[op][0]:
            raise ValueError("eqlinear: path i dimensions disagree")
        dims.append(i)
        mapping.extend((in_offsets[ip], out_offsets[op], wp, i))
    expected, input_row, output_row, previous = [], 0, 0, None
    for q, i in enumerate(dims):
        if previous is not None and previous != i:
            output_row += previous
        expected.extend((input_row, output_row, q, i))
        input_row += i
        previous = i
    dense_j = output_row + dims[-1]
    if input_row == I and dense_j == J and mapping == expected:
        return tuple(dims), u, v, J
    return _PathDimensions(dims, I, J, mapping), u, v, J


def _descriptor_geometry_uncached(descriptor):
    """Validate and translate a ``uv,iu,iv`` descriptor's geometry."""
    if len(descriptor.operands) != 3:
        raise ValueError("eqlinear requires exactly three operands: uv, iu, iv")

    input_segments = _operand_segments(descriptor.operands[1])
    output_segments = _operand_segments(descriptor.operands[2])
    paths = len(descriptor.paths)
    if paths == 0:
        raise ValueError("eqlinear descriptor must contain at least one path")

    if all(hasattr(path, "indices") for path in descriptor.paths):
        u = _single_dimension(descriptor, "u")
        v = _single_dimension(descriptor, "v")
        return _mapped_geometry(descriptor, input_segments, output_segments, u, v)

    if paths <= 4:
        i_dims = tuple(operator.index(v) for v in descriptor.get_dimensions_dict()["i"])
    else:
        i_dims = tuple(segment[0] for segment in input_segments)

    if len(i_dims) != paths:
        raise ValueError(f"eqlinear: derived {len(i_dims)} i_dims for {paths} paths")
    if sum(i_dims) != sum(int(segment[0]) for segment in input_segments):
        raise ValueError("eqlinear: descriptor i_dims disagree with the input operand segments")

    u = _single_dimension(descriptor, "u")
    v = _single_dimension(descriptor, "v")
    j_rows = (_cached_out_rows(i_dims) if len(i_dims) <= 32
              else int(torch.ops.eqlinear.out_rows(i_dims)))
    expected_j = sum(segment[0] for segment in output_segments)
    if j_rows != expected_j:
        raise NotImplementedError(
            f"eqlinear: output segments imply {expected_j} rows, kernel grouping implies {j_rows}"
        )
    return i_dims, u, v, j_rows


def _descriptor_problem(descriptor):
    """Translate a descriptor, reusing geometry only while its immutable sources match."""
    return _DESCRIPTOR_GEOMETRY_CACHE.problem(descriptor)


_DESCRIPTOR_GEOMETRY_CACHE = _DescriptorGeometryCache()
