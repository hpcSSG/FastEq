"""First-order gradient checks for the shared equivariant normalization kernels.

Run with FASTEQ_BACKEND=cpu; this selects the native extension and does not
disable CUDA/Triton. Original-source checks are optional and use the same
FASTEQ_EQUIFORMER_{V2,V3}_LAYER_NORM paths as the forward tests. Use
``--junitxml=backward.xml`` to retain per-gradient maximum absolute and scaled
errors from the accuracy tests. There are no timing or performance assertions.

The mathematical oracle below builds an ordinary autograd graph from slices,
reductions, and concatenation, independently of the kernel metadata and shipped
reference_forward. Its FP64 inputs are the exact tested FP32 values converted
to double, so errors do not include a different random-input quantization.
"""

import copy
import importlib.util
import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")

from fasteq.triton.fused_equivariant_layer_norm import (
    EquivariantNormSpec,
    TritonEquivariantNorm,
    from_reference,
)


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
VERSIONS = ("v0", "v1")
GROUPINGS = ("per_degree", "all", "scalar_high")


def _layout(x, layout):
    if layout == "NKC":
        return x.contiguous()
    return x.transpose(0, 1).contiguous().transpose(0, 1)


def _groups(lmax, grouping):
    if grouping == "per_degree":
        return [(degree,) for degree in range(lmax + 1)]
    if grouping == "scalar_high":
        return [(0,)] + ([tuple(range(1, lmax + 1))] if lmax else [])
    return [tuple(range(lmax + 1))]


def _math_output(x, grouping, weighting, center, weight, bias, eps=1e-5):
    """Differentiable, direct equations; never reads a Triton plan or spec."""
    lmax = int(x.shape[1] ** 0.5) - 1
    blocks = [x[:, degree**2:(degree + 1)**2] for degree in range(lmax + 1)]
    if center:
        blocks[0] = blocks[0] - blocks[0].mean(dim=-1, keepdim=True)
    outputs = []
    for degrees in _groups(lmax, grouping):
        if weighting == "degree_balanced":
            moment = torch.stack(
                [blocks[degree].square().mean(dim=(1, 2)) for degree in degrees], dim=1
            ).mean(dim=1)
        else:
            square = torch.cat([blocks[degree] for degree in degrees], dim=1).square()
            moment = (square.mean(dim=(1, 2)) if weighting == "component"
                      else square.sum(dim=1).mean(dim=1))
        scale = torch.rsqrt(moment + eps)[:, None, None]
        for degree in degrees:
            if weight is None:
                gamma = None
            elif isinstance(weight, tuple):
                gamma = weight[0] if degree == 0 else weight[1][degree - 1]
            else:
                gamma = weight[degree]
            degree_scale = scale if gamma is None else scale * gamma[None, None]
            output = blocks[degree] * degree_scale
            if degree == 0 and bias is not None:
                output = output + bias[None, None]
            outputs.append(output)
    return torch.cat(outputs, dim=1)


def _plan(lmax, channels, grouping, version, weighting="degree_balanced", center=True):
    spec = EquivariantNormSpec.from_preset(
        lmax=lmax, channels=channels, grouping=grouping, weighting=weighting,
        center_scalar=center,
    )
    return TritonEquivariantNorm(
        spec, version=version, device="cuda",
        reduction_order="channels_first" if grouping == "all" else "components_first",
    )


def _clone_leaf(tensor, dtype=None):
    if tensor is None:
        return None
    return tensor.detach().to(dtype=dtype or tensor.dtype).clone().requires_grad_(
        tensor.requires_grad
    )


def _clone_arguments(x, weight, bias, dtype):
    cloned_weight = (tuple(_clone_leaf(value, dtype) for value in weight)
                     if isinstance(weight, tuple) else _clone_leaf(weight, dtype))
    return _clone_leaf(x, dtype), cloned_weight, _clone_leaf(bias, dtype)


def _arguments(x, weight, bias):
    values = {"dX": x}
    if isinstance(weight, tuple):
        values.update(dgamma_scalar=weight[0], dgamma_high=weight[1])
    elif weight is not None:
        values["dgamma"] = weight
    if bias is not None:
        values["dbeta"] = bias
    return {name: tensor for name, tensor in values.items() if tensor.requires_grad}


def _gradients(output, x, weight, bias, grad_output):
    arguments = _arguments(x, weight, bias)
    # At lmax=0, split gamma_high is empty and absent from the math graph.
    values = torch.autograd.grad(
        output, tuple(arguments.values()), grad_outputs=grad_output,
        allow_unused=True,
    )
    return dict(zip(arguments, values))


def _tolerance(name, profile):
    if name == "dbeta":
        return 2e-6, 1e-5
    if profile in ("tiny", "near_constant", "constant", "zero"):
        # eps=1e-5 makes dX as large as O(1/sqrt(eps)) on these inputs.
        if name == "dX":
            return 3e-4, 5e-6
        # Only the near-constant gamma sum needs a larger absolute budget:
        # its centered input includes FP32 scalar-mean rounding near 1.0.
        return (3e-4, 3e-5) if profile == "near_constant" else (2e-5, 3e-5)
    return (3e-6, 3e-5) if name == "dX" else (2e-5, 3e-5)


def _assert_gradients(actual, expected, *, profile="random", record_property=None,
                      prefix=""):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        reference = expected[name]
        if value is None or reference is None:
            assert value is reference is None, f"{prefix}/{name}: unused gradient mismatch"
            continue
        assert torch.isfinite(value).all(), f"{prefix}/{name}: non-finite gradient"
        if value.numel():
            error = (value.double() - reference.double()).abs()
            maximum = error.max().item()
            scaled = maximum / max(reference.double().abs().max().item(), 1e-30)
        else:
            maximum = scaled = 0.
        if record_property is not None:
            record_property(f"{prefix}/{name}/max_abs", maximum)
            record_property(f"{prefix}/{name}/max_abs_over_ref_max", scaled)
        atol, rtol = _tolerance(name, profile)
        torch.testing.assert_close(
            value, reference, atol=atol, rtol=rtol, check_dtype=False,
            msg=lambda message: (f"{prefix}/{name}: max_abs={maximum:.9g}, "
                                 f"scaled_max={scaled:.9g}, atol={atol}, rtol={rtol}\n"
                                 + message),
        )


def _compare_math(op, x, weight, bias, grad_output, *, grouping,
                  weighting="degree_balanced", center=True, profile="random",
                  record_property=None, prefix="", with_stats=False):
    before = [tensor.detach().clone() for tensor in _arguments(x, weight, bias).values()]
    result = (op.forward_with_stats(x, weight=weight, bias=bias) if with_stats
              else op(x, weight=weight, bias=bias))
    output = result.output if with_stats else result
    assert output.requires_grad and output.grad_fn is not None
    actual = _gradients(output, x, weight, bias, grad_output)
    for dtype in (torch.float64, torch.float32):
        rx, rw, rb = _clone_arguments(x, weight, bias, dtype)
        reference = _math_output(rx, grouping, weighting, center, rw, rb)
        expected = _gradients(reference, rx, rw, rb, grad_output.to(dtype))
        _assert_gradients(actual, expected, profile=profile, record_property=record_property,
                          prefix=f"{prefix}/{str(dtype).split('.')[-1]}")
    for value, original in zip(_arguments(x, weight, bias).values(), before):
        torch.testing.assert_close(value.detach(), original, atol=0, rtol=0)
    return result, actual


def _inputs(lmax=2, channels=7, n=5, layout="NKC", parameters="packed",
            profile="random", grad_mode="all"):
    x = torch.randn(n, (lmax + 1)**2, channels, device="cuda")
    if profile == "tiny":
        x = x * 1e-6
    elif profile == "near_constant":
        x = x * 1e-4
        x[:, 0] += 1.
    elif profile == "constant":
        x.fill_(1.)
    elif profile == "zero":
        x.zero_()
    else:
        x[:, 0] += 1.5
    x = _layout(x, layout).detach().requires_grad_(grad_mode in ("all", "input"))
    # Positive, non-unit strides exercise both parameter reads and gradients.
    packed = torch.randn(2 * (lmax + 1), 2 * channels, device="cuda")[::2, ::2]
    requires_weight = grad_mode in ("all", "weight", "params")
    if parameters in ("none", "bias_only"):
        weight = None
    elif parameters == "split":
        weight = (packed[0].detach().requires_grad_(requires_weight),
                  packed[1:].detach().requires_grad_(requires_weight))
    else:
        weight = packed.detach().requires_grad_(requires_weight)
    bias = None if parameters in ("none", "weight_only") else torch.randn(
        2 * channels, device="cuda"
    )[::2].detach().requires_grad_(grad_mode in ("all", "bias", "params"))
    grad_output = torch.randn(x.shape, device="cuda")
    return x, weight, bias, grad_output


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(20260914)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("layout", ("NKC", "KNC"))
@pytest.mark.parametrize("grouping", GROUPINGS)
@pytest.mark.parametrize("lmax,channels", ((0, 1), (2, 7), (4, 33)))
@pytest.mark.parametrize("parameters", ("packed", "split"))
def test_backward_matches_fp64_and_fp32_math(
        version, layout, grouping, lmax, channels, parameters, record_property):
    op = _plan(lmax, channels, grouping, version)
    for seed in (17, 821):
        torch.manual_seed(seed)
        x, weight, bias, grad_output = _inputs(
            lmax, channels, n=37 if lmax == 4 else 5, layout=layout,
            parameters=parameters,
        )
        _compare_math(op, x, weight, bias, grad_output, grouping=grouping,
                      record_property=record_property, prefix=f"seed{seed}")


NONDEFAULTS = (
    ("per_degree", "norm", True),
    ("all", "component", True),
    ("all", "norm", True),
    ("all", "degree_balanced", False),
    ("all", "component", False),
    ("all", "norm", False),
    ("scalar_high", "component", True),
    ("scalar_high", "norm", True),
)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("layout", ("NKC", "KNC"))
@pytest.mark.parametrize("grouping,weighting,center", NONDEFAULTS)
def test_backward_nondefault_statistics(version, layout, grouping, weighting, center):
    op = _plan(2, 7, grouping, version, weighting, center)
    x, weight, bias, grad_output = _inputs(layout=layout)
    # The source merge class omits beta when scalar centering is disabled.
    bias = bias if center else None
    _compare_math(op, x, weight, bias, grad_output, grouping=grouping,
                  weighting=weighting, center=center)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
@pytest.mark.parametrize("parameters", ("none", "bias_only", "weight_only"))
def test_backward_optional_affine(version, grouping, parameters):
    op = _plan(2, 7, grouping, version)
    x, weight, bias, grad_output = _inputs(layout="KNC", parameters=parameters)
    _compare_math(op, x, weight, bias, grad_output, grouping=grouping)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
@pytest.mark.parametrize("grad_mode", ("input", "weight", "bias", "params"))
def test_backward_only_requested_leaves(version, grouping, grad_mode):
    op = _plan(2, 7, grouping, version)
    x, weight, bias, grad_output = _inputs(parameters="split", grad_mode=grad_mode)
    _, actual = _compare_math(op, x, weight, bias, grad_output, grouping=grouping)
    assert ("dX" in actual) == (grad_mode == "input")
    assert ("dgamma_scalar" in actual) == (grad_mode in ("weight", "params"))
    assert ("dbeta" in actual) == (grad_mode in ("bias", "params"))


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
@pytest.mark.parametrize("profile", ("tiny", "near_constant", "constant", "zero"))
def test_backward_small_variance(version, grouping, profile, record_property):
    op = _plan(2, 7, grouping, version)
    x, weight, bias, grad_output = _inputs(parameters="split", profile=profile)
    _compare_math(op, x, weight, bias, grad_output, grouping=grouping, profile=profile,
                  record_property=record_property, prefix=profile)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
@pytest.mark.parametrize("gradient_layout", ("KNC", "strided", "broadcast"))
def test_backward_noncontiguous_upstream_gradient(version, grouping, gradient_layout):
    op = _plan(2, 7, grouping, version)
    x, weight, bias, grad_output = _inputs(layout="KNC")
    if gradient_layout == "KNC":
        grad_output = _layout(grad_output, "KNC")
    elif gradient_layout == "strided":
        grad_output = torch.randn(5, 9, 14, device="cuda")[:, :, ::2]
    else:
        grad_output = torch.randn(1, 1, 7, device="cuda").expand_as(x)
    assert not grad_output.is_contiguous()
    _compare_math(op, x, weight, bias, grad_output, grouping=grouping)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
@pytest.mark.parametrize("layout", ("NKC", "KNC"))
def test_backward_empty_batch(version, grouping, layout):
    op = _plan(2, 7, grouping, version)
    x, weight, bias, grad_output = _inputs(n=0, layout=layout, parameters="split")
    _, actual = _compare_math(op, x, weight, bias, grad_output, grouping=grouping)
    for value in actual.values():
        if value is not None:
            assert torch.count_nonzero(value).item() == 0


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
def test_backward_forward_with_stats_marks_auxiliary_outputs_nondifferentiable(version, grouping):
    op = _plan(2, 7, grouping, version)
    x, weight, bias, grad_output = _inputs()
    result, _ = _compare_math(op, x, weight, bias, grad_output, grouping=grouping,
                              with_stats=True)
    assert result.output.requires_grad
    for statistic in (result.mean, result.moments, result.rstd):
        assert not statistic.requires_grad
        assert statistic.grad_fn is None


@GPU
@pytest.mark.parametrize("version", VERSIONS)
def test_backward_uncentered_stats_have_no_mean(version):
    op = _plan(2, 7, "all", version, center=False)
    x, weight, _, grad_output = _inputs()
    result, _ = _compare_math(op, x, weight, None, grad_output, grouping="all",
                              center=False, with_stats=True)
    assert result.mean is None
    assert not result.moments.requires_grad and not result.rstd.requires_grad


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
def test_backward_rejects_higher_order_graphs_explicitly(version, grouping):
    op = _plan(2, 7, grouping, version)
    x, weight, bias, grad_output = _inputs()
    output = op(x, weight=weight, bias=bias)
    # First-order-only means create_graph must fail immediately and clearly;
    # silently returning a detached dX would produce an incorrect Hessian.
    with pytest.raises((RuntimeError, NotImplementedError),
                       match=r"(?i)(first.order|higher.order|second.order|double.backward)"):
        torch.autograd.grad(output, x, grad_outputs=grad_output, create_graph=True)


SOURCE_CLASSES = (
    ("V3", "EquivariantLayerNorm"),
    ("V3", "EquivariantMergeLayerNorm"),
    ("V2", "EquivariantLayerNormArraySphericalHarmonics"),
)


def _load_source(source_family, class_name):
    variable = f"FASTEQ_EQUIFORMER_{source_family}_LAYER_NORM"
    source_path = os.environ.get(variable)
    if not source_path:
        pytest.skip(f"set {variable} to enable original-source validation")
    path = Path(source_path)
    assert path.is_file(), f"{variable} is not a source file: {path}"
    descriptor = importlib.util.spec_from_file_location(
        "_fasteq_norm_backward_source_" + source_family, path
    )
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    return getattr(module, class_name)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("layout", ("NKC", "KNC"))
@pytest.mark.parametrize("source_family,class_name", SOURCE_CLASSES)
@pytest.mark.parametrize("profile", ("random", "near_constant"))
def test_backward_optional_original_source(
        version, layout, source_family, class_name, profile, record_property):
    source = _load_source(source_family, class_name)(2, 7).cuda().train()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(mean=.1, std=.9)
    adapter = from_reference(source, version=version)
    x, _, _, grad_output = _inputs(layout=layout, profile=profile)
    rx = _clone_leaf(x)
    named = dict(source.named_parameters())
    parameters = tuple(named.values())
    actual = torch.autograd.grad(adapter(x), (x, *parameters), grad_outputs=grad_output)
    expected = torch.autograd.grad(source(rx), (rx, *parameters), grad_outputs=grad_output)
    # Keep original parameter names in the report to identify scalar/higher
    # weights, while using the same per-gradient tolerance as the math tests.
    for name, value, reference in zip(("dX", *named), actual, expected):
        kind = "dX" if name == "dX" else "dbeta" if "bias" in name else "dgamma"
        _assert_gradients({kind: value}, {kind: reference}, profile=profile,
                          record_property=record_property, prefix=f"source/{name}")
    assert {id(parameter) for parameter in adapter.parameters()} == {
        id(parameter) for parameter in source.parameters()
    }


def _custom_partition_output(x, weight, bias, center):
    """Explicit {l=0,1}/{l=2,3} equations; coefficients do not sum to one."""
    scalar = x[:, :1]
    if center:
        scalar = scalar - scalar.mean(dim=-1, keepdim=True)
    z = torch.cat((scalar, x[:, 1:]), dim=1)
    q = z.square().mean(dim=-1)
    first = .7 * q[:, 0] + .2 * q[:, 1:4].sum(dim=1)
    second = .3 * q[:, 4:9].sum(dim=1) + .05 * q[:, 9:16].sum(dim=1)
    first_scale = torch.rsqrt(first + .17)[:, None, None]
    second_scale = torch.rsqrt(second + .17)[:, None, None]
    return torch.cat((
        z[:, :1] * (first_scale * weight[0]) + bias,
        z[:, 1:4] * (first_scale * weight[1]),
        z[:, 4:9] * (second_scale * weight[2]),
        z[:, 9:16] * (second_scale * weight[3]),
    ), dim=1)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("center", (True, False))
@pytest.mark.parametrize("order", ("channels_first", "components_first"))
def test_backward_custom_nonuniform_partition(version, center, order, record_property):
    spec = EquivariantNormSpec(
        3, 7, ((.7, .2, 0., 0.), (0., 0., .3, .05)), (0, 0, 1, 1),
        center_scalar=center, eps=.17,
    )
    op = TritonEquivariantNorm(spec, version=version, reduction_order=order, device="cuda")
    x, weight, bias, grad_output = _inputs(lmax=3, layout="KNC")
    actual = _gradients(op(x, weight=weight, bias=bias), x, weight, bias, grad_output)
    for dtype in (torch.float64, torch.float32):
        rx, rw, rb = _clone_arguments(x, weight, bias, dtype)
        output = _custom_partition_output(rx, rw, rb, center)
        expected = _gradients(output, rx, rw, rb, grad_output.to(dtype))
        _assert_gradients(actual, expected, record_property=record_property,
                          prefix=f"custom/{str(dtype).split('.')[-1]}")


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
@pytest.mark.parametrize("trainable", ("scalar", "higher"))
def test_backward_split_only_one_weight_requires_grad(version, grouping, trainable):
    op = _plan(2, 7, grouping, version)
    x, weight, bias, grad_output = _inputs(parameters="split", grad_mode="weight")
    weight[1 if trainable == "scalar" else 0].requires_grad_(False)
    _, actual = _compare_math(op, x, weight, bias, grad_output, grouping=grouping)
    assert set(actual) == {"dgamma_scalar" if trainable == "scalar" else "dgamma_high"}
    assert not x.requires_grad and not bias.requires_grad


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", ("per_degree", "scalar_high"))
@pytest.mark.parametrize("corrupted_tensor", ("X", "dY"))
@pytest.mark.parametrize("corrupted_degree", (0, 1))
def test_backward_nan_isolation(version, grouping, corrupted_tensor, corrupted_degree):
    op = _plan(2, 7, grouping, version)
    x, weight, bias, grad_output = _inputs(layout="KNC")
    rx, rw, rb = _clone_arguments(x, weight, bias, torch.float64)
    clean_output = _math_output(rx, grouping, "degree_balanced", True, rw, rb)
    clean = _gradients(clean_output, rx, rw, rb, grad_output.double())
    # Corrupt one channel of one atom. Other atoms, as well as other groups
    # in this atom, must retain their clean gradients.
    with torch.no_grad():
        target = x if corrupted_tensor == "X" else grad_output
        target[1, corrupted_degree**2, 2] = float("nan")
    actual = _gradients(op(x, weight=weight, bias=bias), x, weight, bias, grad_output)
    bad_group = next(degrees for degrees in _groups(2, grouping)
                     if corrupted_degree in degrees)
    bad_start, bad_end = bad_group[0]**2, (bad_group[-1] + 1)**2
    assert torch.isnan(actual["dX"][1, bad_start:bad_end]).any()
    for degree in range(3):
        if degree in bad_group:
            continue
        span = slice(degree**2, (degree + 1)**2)
        # These subsets must all be finite; contaminated subsets are never
        # passed to the helper's all-finite assertion.
        _assert_gradients(
            {"dX": actual["dX"][:, span], "dgamma": actual["dgamma"][degree]},
            {"dX": clean["dX"][:, span], "dgamma": clean["dgamma"][degree]},
            prefix=f"unaffected_degree{degree}",
        )
    clean_atoms = torch.tensor([0, 2, 3, 4], device="cuda")
    _assert_gradients({"dX": actual["dX"][clean_atoms]},
                      {"dX": clean["dX"][clean_atoms]}, prefix="unaffected_atoms")
    if corrupted_tensor == "X" or corrupted_degree != 0:
        _assert_gradients({"dbeta": actual["dbeta"]}, {"dbeta": clean["dbeta"]},
                          prefix="independent_bias")


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
def test_backward_model_shape_l4_c128_n108(version, grouping, record_property):
    op = _plan(4, 128, grouping, version)
    x, weight, bias, grad_output = _inputs(
        lmax=4, channels=128, n=108, layout="KNC",
        parameters="split" if grouping == "scalar_high" else "packed",
    )
    _compare_math(op, x, weight, bias, grad_output, grouping=grouping,
                  record_property=record_property, prefix="l4_c128_n108")


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
def test_backward_tiny_epsilon_zero_input_stays_finite(version, grouping, record_property):
    eps = 1e-30
    spec = EquivariantNormSpec.from_preset(lmax=2, channels=7, grouping=grouping, eps=eps)
    op = TritonEquivariantNorm(spec, version=version, device="cuda")
    x, weight, bias, grad_output = _inputs(profile="zero")
    actual = _gradients(op(x, weight=weight, bias=bias), x, weight, bias, grad_output)
    rx, rw, rb = _clone_arguments(x, weight, bias, torch.float64)
    reference = _math_output(rx, grouping, "degree_balanced", True, rw, rb, eps=eps)
    expected = _gradients(reference, rx, rw, rb, grad_output.double())
    for name, value in actual.items():
        assert torch.isfinite(value).all(), f"tiny epsilon produced non-finite {name}"
    # dX is O(1e15). Compare in units of 1/sqrt(eps), retaining an explicit
    # 3e-7 absolute / 3e-6 relative FP32 error budget in those units. An r**3
    # intermediate would overflow FP32 even though the final derivative fits.
    scale = eps**-.5
    normalized_error = ((actual["dX"].double() - expected["dX"]) / scale).abs().max().item()
    record_property("tiny_eps/dX/max_abs_over_rstd", normalized_error)
    torch.testing.assert_close(actual["dX"].double() / scale, expected["dX"] / scale,
                               atol=3e-7, rtol=3e-6)
    _assert_gradients({name: value for name, value in actual.items() if name != "dX"},
                      {name: value for name, value in expected.items() if name != "dX"},
                      record_property=record_property, prefix="tiny_eps")


@GPU
@pytest.mark.parametrize("version", VERSIONS)
def test_backward_optional_original_source_chain_two_sgd_steps(version, record_property):
    originals = torch.nn.Sequential(*[
        _load_source(family, class_name)(2, 7) for family, class_name in SOURCE_CLASSES
    ]).cuda().train()
    with torch.no_grad():
        for name, parameter in originals.named_parameters():
            if "weight" in name:
                parameter.uniform_(.7, 1.3)
            else:
                parameter.normal_(mean=0., std=.1)
    reference = copy.deepcopy(originals)
    adapted = torch.nn.Sequential(*[from_reference(module, version=version)
                                    for module in originals]).train()
    optimizer = torch.optim.SGD(adapted.parameters(), lr=.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.05)
    actual_parameters = dict(originals.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    assert actual_parameters.keys() == reference_parameters.keys()
    initial = [parameter.detach().clone() for parameter in actual_parameters.values()]
    for step, layout in enumerate(("NKC", "KNC")):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        x, _, _, upstream = _inputs(layout=layout)
        rx = _clone_leaf(x)
        output, reference_output = adapted(x), reference(rx)
        loss = (output * upstream).mean() + .05 * output.square().mean()
        reference_loss = ((reference_output * upstream).mean()
                          + .05 * reference_output.square().mean())
        torch.testing.assert_close(loss, reference_loss, atol=3e-6, rtol=3e-5)
        loss.backward()
        reference_loss.backward()
        _assert_gradients({"dX": x.grad}, {"dX": rx.grad},
                          record_property=record_property, prefix=f"sgd_step{step}")
        for name, parameter in actual_parameters.items():
            expected = reference_parameters[name]
            assert parameter.grad is not None and expected.grad is not None
            kind = "dbeta" if "bias" in name else "dgamma"
            _assert_gradients({kind: parameter.grad}, {kind: expected.grad},
                              record_property=record_property, prefix=f"sgd_step{step}/{name}")
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in actual_parameters.items():
            torch.testing.assert_close(parameter, reference_parameters[name],
                                       atol=3e-6, rtol=3e-6,
                                       msg=f"parameter update differs at step {step}: {name}")
    assert any(not torch.equal(parameter.detach(), before)
               for parameter, before in zip(actual_parameters.values(), initial))
