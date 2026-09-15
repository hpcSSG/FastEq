"""Portable normalization tests: torch + triton + pytest, no model checkout.

Run from the checkout with::

    FASTEQ_BACKEND=cpu CUDA_VISIBLE_DEVICES=1 python -m pytest -q \
        test/test_triton_equivariant_layer_norm.py

FASTEQ_BACKEND selects FastEq's native extension, not the Torch/Triton device.
GPU tests skip when CUDA is unavailable. Optional original-source checks use
FASTEQ_EQUIFORMER_V3_LAYER_NORM and FASTEQ_EQUIFORMER_V2_LAYER_NORM file paths.
The independent oracle below follows the source equations directly; it does
not use the shared test reference to determine expected results.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from fasteq.triton.fused_equivariant_layer_norm import (
    EquivariantNormSpec,
    TritonEquivariantNorm,
    from_reference,
)


def _reference_forward():
    # Load the sibling by path so both pytest import modes work.
    path = Path(__file__).with_name("equivariant_layer_norm_reference.py")
    descriptor = importlib.util.spec_from_file_location("_fasteq_norm_math_reference", path)
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    return module.reference_forward


reference_forward = _reference_forward()


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
GROUPINGS = ("per_degree", "scalar_high")


def _layout(x, layout):
    return x.contiguous() if layout == "NKC" else x.transpose(0, 1).contiguous().transpose(0, 1)


def _double(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(v.double() for v in value)
    return value.double()


def _groups(lmax, grouping):
    if grouping == "per_degree":
        return [(l,) for l in range(lmax + 1)]
    if grouping == "scalar_high":
        return [(0,)] + ([tuple(range(1, lmax + 1))] if lmax else [])
    raise ValueError(f"unsupported grouping: {grouping}")


def _oracle(x, grouping, weighting, center=True, weight=None, bias=None, eps=1e-5):
    """Direct slice/reduction equations, independent of the plan's metadata.

    Per-degree and SH reduce magnetic components before channels.
    SH's scalar output uses
    native LayerNorm, like EquiformerV2, for source-compatibility checks.
    """
    lmax = int(x.shape[1] ** 0.5) - 1
    mean = x[:, 0].mean(-1) if center else None
    z = x.clone()
    if center:
        z[:, 0] -= mean[:, None]
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    moments = []
    for degrees in _groups(lmax, grouping):
        start, stop = degrees[0] ** 2, (degrees[-1] + 1) ** 2
        square = z[:, start:stop].square()
        if weighting == "degree_balanced":
            coefficients = torch.cat([
                torch.full((2 * l + 1,), 1 / (len(degrees) * (2 * l + 1)),
                           dtype=x.dtype, device=x.device) for l in degrees
            ])
            moment = (square * coefficients[None, :, None]).sum(1).mean(-1)
        else:
            per_channel = square.mean(1) if weighting == "component" else square.sum(1)
            moment = per_channel.mean(-1)
        moments.append(moment)
        rstd = torch.rsqrt(moment + eps)
        for l in degrees:
            gamma = None if weight is None else (
                (weight[0] if l == 0 else weight[1][l - 1]) if isinstance(weight, tuple) else weight[l]
            )
            scale = rstd[:, None, None]
            if gamma is not None:
                scale = scale * gamma[None, None]
            y = z[:, l * l:(l + 1) ** 2] * scale
            if l == 0 and bias is not None:
                y = y + bias[None, None]
            if grouping == "scalar_high" and l == 0 and center:
                y = F.layer_norm(x[:, :1], (x.shape[-1],), gamma, bias, eps)
            output[:, l * l:(l + 1) ** 2] = y
    moments = torch.stack(moments, dim=-1)
    return output, mean, moments, torch.rsqrt(moments + eps)


def _plan(lmax, channels, grouping, weighting="degree_balanced", center=True):
    spec = EquivariantNormSpec.from_preset(
        lmax=lmax, channels=channels, grouping=grouping,
        weighting=weighting, center_scalar=center,
    )
    op = TritonEquivariantNorm(
        spec, device="cuda",
        reduction_order="components_first",
    )
    return spec, op


def _assert_close(actual, expected, *, atol=3e-5, rtol=3e-5):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol,
                               check_dtype=False, equal_nan=True)


def _check_case(x, grouping, weighting="degree_balanced", center=True,
                weight=None, bias=None):
    lmax, channels = int(x.shape[1] ** 0.5) - 1, x.shape[-1]
    spec, op = _plan(lmax, channels, grouping, weighting, center)
    before = x.clone()
    expected = _oracle(x, grouping, weighting, center, weight, bias)
    expected64 = _oracle(x.double(), grouping, weighting, center,
                         _double(weight), _double(bias))
    with torch.inference_mode():
        actual = op(x, weight=weight, bias=bias)
        debug = op.forward_with_stats(x, weight=weight, bias=bias)
    assert actual.shape == x.shape and actual.dtype == torch.float32
    assert actual.is_contiguous()
    _assert_close(actual, expected[0])
    _assert_close(actual, expected64[0])
    _assert_close(debug.output, actual, atol=0, rtol=0)
    assert debug.moments.shape == debug.rstd.shape == (x.shape[0], spec.num_stats)
    if center:
        _assert_close(debug.mean, expected64[1], atol=2e-6, rtol=3e-6)
    else:
        assert debug.mean is None
    _assert_close(debug.moments, expected64[2])
    _assert_close(debug.rstd, expected64[3])
    _assert_close(x, before, atol=0, rtol=0)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(20260909)


@GPU
@pytest.mark.parametrize("layout", ("NKC", "KNC"))
@pytest.mark.parametrize("grouping", GROUPINGS)
@pytest.mark.parametrize("shape", ((0, 1), (2, 7), (4, 128)))
def test_default_source_equations(layout, grouping, shape):
    lmax, channels = shape
    x = torch.randn(3, (lmax + 1) ** 2, channels, device="cuda")
    x[:, 0] += 1.5
    packed = torch.randn(lmax + 1, channels, device="cuda")
    weight = (packed[0], packed[1:]) if grouping == "scalar_high" else packed
    bias = torch.randn(channels, device="cuda")
    _check_case(_layout(x, layout), grouping, weight=weight, bias=bias)


NONDEFAULTS = (
    ("per_degree", "norm", True),
    ("scalar_high", "component", True),
    ("scalar_high", "norm", True),
)


@GPU
@pytest.mark.parametrize("layout", ("NKC", "KNC"))
@pytest.mark.parametrize("grouping,weighting,center", NONDEFAULTS)
def test_nondefault_source_options(layout, grouping, weighting, center):
    x = torch.randn(3, 9, 7, device="cuda")
    x[:, 0] += 2
    _check_case(_layout(x, layout), grouping, weighting, center)


@GPU
@pytest.mark.parametrize("layout", ("NKC", "KNC"))
@pytest.mark.parametrize("parameters", ("packed", "split", "bias_only"))
def test_strided_affine_parameters(layout, parameters):
    x = _layout(torch.randn(3, 9, 7, device="cuda"), layout)
    packed = torch.randn(6, 14, device="cuda")[::2, ::2]
    bias = torch.randn(14, device="cuda")[::2]
    weight = {"packed": packed, "split": (packed[0], packed[1:]), "bias_only": None}[parameters]
    _check_case(x, "per_degree", weight=weight, bias=bias)


@GPU
@pytest.mark.parametrize("grouping", GROUPINGS)
def test_finite_pathologies(grouping):
    spec, op = _plan(2, 7, grouping)
    random = torch.randn(3, 9, 7, device="cuda")
    for x in (torch.zeros_like(random), torch.full_like(random, 4.),
              random * 1e-6, random * 1e6, 1. + random * 1e-4):
        with torch.inference_mode():
            actual = op(x)
        expected = _oracle(x, grouping, "degree_balanced")[0]
        # Near-constant native LayerNorm uses a different FP32 mean reduction.
        _assert_close(actual, expected, atol=3e-4)
        assert torch.isfinite(actual).all()


@GPU
@pytest.mark.parametrize("grouping", GROUPINGS)
@pytest.mark.parametrize("magnitude", (float("nan"), 1e20), ids=("nan", "square_overflow"))
def test_nonfinite_values_do_not_leak_between_groups(grouping, magnitude):
    spec, op = _plan(2, 7, grouping)
    for corrupted_degree in (0, 1):
        x = torch.ones(2, 9, 7, device="cuda")
        x[:, 0] = torch.arange(-3, 4, device="cuda", dtype=torch.float32)
        x[:, corrupted_degree ** 2:(corrupted_degree + 1) ** 2] *= magnitude
        with torch.inference_mode():
            debug = op.forward_with_stats(x)
        expected = _oracle(x, grouping, "degree_balanced")
        _assert_close(debug.output, expected[0])
        _assert_close(debug.moments, expected[2])
        _assert_close(debug.rstd, expected[3])
        for l, group in enumerate(spec.output_group):
            if spec.stats_weights[group][corrupted_degree] == 0:
                assert torch.isfinite(debug.output[:, l * l:(l + 1) ** 2]).all()


@GPU
@pytest.mark.parametrize("order", ("channels_first", "components_first"))
@pytest.mark.parametrize("layout", ("NKC", "KNC"))
def test_custom_partition_uses_shared_kernel(order, layout):
    spec = EquivariantNormSpec(3, 7, ((1., 1 / 3, 0., 0.), (0., 0., 1 / 10, 1 / 14)),
                               (0, 0, 1, 1), center_scalar=False)
    op = TritonEquivariantNorm(spec, reduction_order=order, device="cuda")
    x = _layout(torch.randn(3, 16, 7, device="cuda"), layout)
    # Independent explicit equations for {l=0,1} and {l=2,3}; no model preset.
    q = x.double().square().mean(-1)
    moments = torch.stack((q[:, 0] + q[:, 1:4].sum(-1) / 3,
                           q[:, 4:9].sum(-1) / 10 + q[:, 9:16].sum(-1) / 14), dim=-1)
    rstd = torch.rsqrt(moments + spec.eps)
    expected = torch.cat((x[:, :4] * rstd[:, :1, None],
                          x[:, 4:] * rstd[:, 1:, None]), dim=1)
    with torch.inference_mode():
        debug = op.forward_with_stats(x)
        actual = op(x)
    assert debug.mean is None
    _assert_close(debug.moments, moments)
    _assert_close(debug.rstd, rstd)
    _assert_close(actual, expected)
    _assert_close(actual, debug.output, atol=0, rtol=0)


@GPU
@pytest.mark.parametrize("grouping", GROUPINGS)
def test_empty_batch_and_live_weights(grouping):
    spec, op = _plan(2, 7, grouping)
    with torch.inference_mode():
        for layout in ("NKC", "KNC"):
            x = _layout(torch.empty(0, 9, 7, device="cuda"), layout)
            output, debug = op(x), op.forward_with_stats(x)
            assert output.shape == debug.output.shape == (0, 9, 7)
            assert debug.mean.shape == (0,)
            assert debug.moments.shape == debug.rstd.shape == (0, spec.num_stats)
        x = torch.randn(3, 9, 7, device="cuda")
        weight = torch.ones(3, 7, device="cuda")
        before = op(x, weight=weight)
        weight.mul_(2)
        after = op(x, weight=weight)
        _assert_close(after, before * 2, atol=0, rtol=0)


@GPU
@pytest.mark.parametrize("grouping", GROUPINGS)
def test_orthogonal_equivariance(grouping):
    _, op = _plan(2, 7, grouping)
    x = torch.randn(3, 9, 7, device="cuda")
    weight, bias = torch.randn(3, 7, device="cuda"), torch.randn(7, device="cuda")
    transforms = [torch.linalg.qr(torch.randn(2 * l + 1, 2 * l + 1,
                                             dtype=torch.float64))[0].float().cuda()
                  for l in (1, 2)]

    def transform(tensor):
        result = tensor.clone()
        for l, q in zip((1, 2), transforms):
            result[:, l * l:(l + 1) ** 2] = torch.einsum(
                "ij,njc->nic", q, tensor[:, l * l:(l + 1) ** 2])
        return result

    with torch.inference_mode():
        original = op.forward_with_stats(x, weight=weight, bias=bias)
        rotated = op.forward_with_stats(transform(x), weight=weight, bias=bias)
    _assert_close(rotated.output, transform(original.output), atol=4e-5, rtol=4e-5)
    _assert_close(rotated.moments, original.moments, atol=4e-5, rtol=4e-5)


@GPU
def test_input_guards_and_grad_dispatch():
    _, op = _plan(2, 7, "per_degree")
    x = torch.randn(3, 9, 7, device="cuda")
    with pytest.raises(TypeError, match="FP32"):
        op(x.half())
    with pytest.raises(ValueError, match="expected"):
        op(x[:, :4])
    with pytest.raises(ValueError, match="storage"):
        op(torch.randn(3, 9, 14, device="cuda")[:, :, ::2])
    with pytest.raises(ValueError, match="shape"):
        op(x, weight=torch.ones(7, device="cuda"))
    with pytest.raises(ValueError, match="positive"):
        op(x, weight=torch.ones(1, 7, device="cuda").expand(3, 7))
    with pytest.raises(ValueError, match="dtype"):
        op(x, bias=torch.zeros(7, device="cuda", dtype=torch.float64))
    with torch.enable_grad():
        assert op(x.clone().requires_grad_()).requires_grad
        assert op(x, weight=torch.ones(3, 7, device="cuda", requires_grad=True)).requires_grad
    with torch.no_grad():
        assert not op(x.clone().requires_grad_()).requires_grad


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_cpu_reference_and_spec(grouping):
    spec = EquivariantNormSpec.from_preset(lmax=2, channels=7, grouping=grouping)
    assert spec.components == 9
    assert spec.num_stats == {"per_degree": 3, "scalar_high": 2}[grouping]
    x = torch.randn(3, 9, 7, dtype=torch.float64)
    weight, bias = torch.randn(3, 7, dtype=torch.float64), torch.randn(7, dtype=torch.float64)
    expected = _oracle(x, grouping, "degree_balanced", weight=weight, bias=bias)
    actual = reference_forward(x, spec, weight=weight, bias=bias)
    for value, oracle in zip(actual, expected):
        _assert_close(value, oracle, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kwargs", (
    {"lmax": -1}, {"channels": 0}, {"eps": 0.}, {"eps": float("nan")},
    {"center_scalar": 1}, {"grouping": "unknown"}, {"grouping": "all"},
    {"weighting": "unknown"},
))
def test_invalid_spec(kwargs):
    options = dict(lmax=2, channels=7)
    options.update(kwargs)
    with pytest.raises(ValueError):
        EquivariantNormSpec.from_preset(**options)


def test_unsupported_gpu_partition_is_rejected_before_allocation():
    unsupported = (
        EquivariantNormSpec(2, 7, ((1., 0., 0.), (1 / 3, 1 / 9, 1 / 15)), (0, 1, 1)),
        EquivariantNormSpec(2, 7, ((1., 0., 1 / 5), (0., 1 / 3, 0.)), (0, 1, 0)),
        EquivariantNormSpec(2, 7, ((1., 1 / 3, 0.), (0., 0., 1 / 5)), (0, 1, 1)),
    )
    for spec in unsupported:
        with pytest.raises(NotImplementedError, match="groups"):
            TritonEquivariantNorm(spec, device="cpu")


def test_public_package_exports():
    from fasteq import triton as public

    assert public.EquivariantNormSpec is EquivariantNormSpec
    assert public.TritonEquivariantNorm is TritonEquivariantNorm
    assert public.from_reference is from_reference


SOURCE_CLASSES = (
    ("V3", "EquivariantLayerNorm"),
    ("V3", "EquivariantSeparableLayerNorm"),
    ("V2", "EquivariantLayerNormArraySphericalHarmonics"),
)


@GPU
@pytest.mark.parametrize("source_family,class_name", SOURCE_CLASSES)
def test_optional_original_source(source_family, class_name):
    """Extended check against user-provided, unmodified model source files."""
    variable = f"FASTEQ_EQUIFORMER_{source_family}_LAYER_NORM"
    source_path = os.environ.get(variable)
    if not source_path:
        pytest.skip(f"set {variable} to enable original-source validation")
    path = Path(source_path)
    assert path.is_file(), f"{variable} is not a source file: {path}"
    descriptor = importlib.util.spec_from_file_location("_fasteq_norm_test_" + source_family, path)
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    with torch.inference_mode():
        source = getattr(module, class_name)(2, 7).cuda().eval()
        for parameter in source.parameters():
            parameter.normal_(mean=.1, std=.9)
        adapter = from_reference(source)
        for layout in ("NKC", "KNC"):
            x = _layout(torch.randn(3, 9, 7, device="cuda"), layout)
            _assert_close(adapter(x), source(x))
        # Keep parameter ownership live through reassignment and state_dict.
        source.affine_weight = torch.nn.Parameter(torch.randn_like(source.affine_weight))
        if hasattr(source, "norm_l0"):
            source.norm_l0.weight = torch.nn.Parameter(torch.randn_like(source.norm_l0.weight))
            source.norm_l0.bias = torch.nn.Parameter(torch.randn_like(source.norm_l0.bias))
        else:
            source.affine_bias = torch.nn.Parameter(torch.randn_like(source.affine_bias))
        _assert_close(adapter(x), source(x))
        assert any(key.startswith("source.") for key in adapter.state_dict())


def test_default_preset_is_per_degree():
    default = EquivariantNormSpec.from_preset(lmax=2, channels=7)
    explicit = EquivariantNormSpec.from_preset(lmax=2, channels=7, grouping="per_degree")
    assert default == explicit


def test_out_of_scope_source_is_rejected_before_allocation():
    class EquivariantMergeLayerNorm:
        pass

    with pytest.raises(NotImplementedError, match="no verified source adapter"):
        from_reference(EquivariantMergeLayerNorm(), device="cpu")
