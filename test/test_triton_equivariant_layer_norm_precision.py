"""Large-batch FP32 parameter-gradient regression tests.

Run with FASTEQ_BACKEND=cpu and a CUDA device. The cancellation cases have
closed-form FP64 answers; the random case reuses the independent equations
from the backward tests. Optional source comparisons use the same V2/V3
source-file environment variables as the existing backward suite.
"""

import hashlib
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
    reference_forward,
)


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
VERSIONS = ("v0", "v1")
GROUPINGS = ("per_degree", "all", "scalar_high")


def _plan(grouping, version, channels):
    spec = EquivariantNormSpec.from_preset(lmax=3, channels=channels, grouping=grouping)
    return TritonEquivariantNorm(
        spec, version=version, device="cuda",
        reduction_order="channels_first" if grouping == "all" else "components_first",
    )


def _record_error(record_property, name, actual, expected):
    error = (actual.double() - expected.double()).abs().max().item()
    record_property(f"{name}/max_abs", error)


def _check_cancellation(n, grouping, version, record_property):
    channels = 7
    op = _plan(grouping, version, channels)
    # Every higher component has z=1; the centered scalar has z=0.
    x = torch.ones(n, 16, channels, device="cuda")
    x[:, 0] = 0.
    weight = torch.ones(4, channels, device="cuda", requires_grad=True)
    bias = torch.zeros(channels, device="cuda", requires_grad=True)
    dy = torch.full_like(x, 1e-8)
    dy[:32] = 1.
    dy[-32:] = -1.
    dgamma, dbeta = torch.autograd.grad(op(x, weight=weight, bias=bias),
                                       (weight, bias), grad_outputs=dy)

    # The endpoint blocks cancel exactly. Convert the actual FP32 small term
    # to FP64 before multiplying, so input quantization is not counted as error.
    total = dy[32, 0, 0].double() * (n - 64)
    moment = .75 if grouping == "all" else 1.
    rstd = torch.rsqrt(torch.tensor(moment + 1e-5, device="cuda", dtype=torch.float64))
    expected_gamma = torch.zeros(4, channels, device="cuda", dtype=torch.float64)
    for degree in (1, 2, 3):
        expected_gamma[degree] = (2 * degree + 1) * rstd * total
    expected_beta = total.expand(channels)
    _record_error(record_property, "dgamma", dgamma, expected_gamma)
    _record_error(record_property, "dbeta", dbeta, expected_beta)
    # These are the existing backward suite's random parameter tolerances.
    torch.testing.assert_close(dgamma, expected_gamma, atol=2e-5, rtol=3e-5,
                               check_dtype=False)
    torch.testing.assert_close(dbeta, expected_beta, atol=2e-6, rtol=1e-5,
                               check_dtype=False)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
def test_large_parameter_reduction_cancellation(version, grouping, record_property):
    _check_cancellation(27648, grouping, version, record_property)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
def test_large_parameter_reduction_partial_final_tile(version, record_property):
    # One extra atom exercises masking in the last, incomplete reduction tile.
    _check_cancellation(27649, "all", version, record_property)


def _backward_helpers():
    # Resolve the sibling by file path so both pytest import modes work and
    # callers need not add the test directory to PYTHONPATH.
    path = Path(__file__).with_name("test_triton_equivariant_layer_norm_backward.py")
    descriptor = importlib.util.spec_from_file_location("_fasteq_precision_math_oracle", path)
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    return module


def _math_oracle():
    return _backward_helpers()._math_output


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("grouping", GROUPINGS)
def test_large_random_gradients_match_fp64(version, grouping, record_property):
    torch.manual_seed(20260914)
    op = _plan(grouping, version, 128)
    weight = torch.randn(4, 128, device="cuda", requires_grad=True)
    bias = torch.randn(128, device="cuda", requires_grad=True)
    x = torch.randn(4096, 16, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    actual = torch.autograd.grad(op(x, weight=weight, bias=bias), (x, weight, bias),
                                 grad_outputs=dy)
    reference_inputs = tuple(value.detach().double().requires_grad_()
                             for value in (x, weight, bias))
    rx, rw, rb = reference_inputs
    output = _math_oracle()(rx, grouping, "degree_balanced", True, rw, rb)
    expected = torch.autograd.grad(output, reference_inputs, grad_outputs=dy.double())
    for name, value, reference in zip(("dX", "dgamma", "dbeta"), actual, expected):
        _record_error(record_property, name, value, reference)
        # Keep the user's reported elementwise tolerance unchanged, including
        # for near-zero parameter gradients after cancellation across atoms.
        torch.testing.assert_close(value, reference, atol=5e-5, rtol=5e-4,
                                   check_dtype=False, msg=f"{name} differs from FP64")


NATIVE_SOURCES = (
    ("V3", "EquivariantLayerNorm"),
    ("V3", "EquivariantMergeLayerNorm"),
    ("V2", "EquivariantLayerNormArraySphericalHarmonics"),
    ("V3", "EquivariantSeparableLayerNorm"),
)


def _native_comparison_metrics(actual, expected):
    actual64, expected64 = actual.double(), expected.double()
    error = (actual64 - expected64).abs()
    tolerance = 5e-5 + 5e-4 * expected64.abs()
    failed = ((error > tolerance) | ~torch.isfinite(actual64)
              | ~torch.isfinite(expected64))
    return {
        "max_abs": error.max().item(),
        "relative_l2": (error.norm() / expected64.norm().clamp_min(1e-300)).item(),
        "failures": failed.sum().item(),
        "max_tolerance_ratio": (error / tolerance).max().item(),
    }


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("layout", ("NKC", "KNC"))
@pytest.mark.parametrize("n", (1, 17, 257, 4096, 27648))
@pytest.mark.parametrize("source_family,class_name", NATIVE_SOURCES)
def test_optional_original_source_large_batch_acceptance(
        version, layout, n, source_family, class_name, record_property):
    source_class = _backward_helpers()._load_source(source_family, class_name)
    source_path = Path(os.environ[f"FASTEQ_EQUIFORMER_{source_family}_LAYER_NORM"])
    record_property("source/path", str(source_path.resolve()))
    record_property("source/sha256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    record_property("source/class", class_name)
    record_property("reference", "original FP32 source forward and autograd")

    # Match diagnose.py: construct the module, randomize every parameter, draw
    # the input, run both forwards, then draw one shared upstream gradient.
    torch.manual_seed(20260914)
    source = source_class(3, 128).cuda().train()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.randn_like(parameter))
    x = torch.randn(n, 16, 128, device="cuda", requires_grad=True)
    if layout == "KNC":
        x = x.transpose(0, 1).contiguous().transpose(0, 1).detach().requires_grad_()
    rx = x.detach().clone(memory_format=torch.preserve_format).requires_grad_()
    adapter = from_reference(source, version=version)
    actual_output = adapter(x)
    expected_output = source(rx)
    dy = torch.randn_like(actual_output)
    named = dict(source.named_parameters())
    parameters = tuple(named.values())
    actual_gradients = torch.autograd.grad(actual_output, (x, *parameters), grad_outputs=dy)
    expected_gradients = torch.autograd.grad(expected_output, (rx, *parameters), grad_outputs=dy)

    # Record every output/gradient before asserting, so a forward failure does
    # not hide the input or parameter-gradient evidence in the XML report.
    comparisons = [("forward", actual_output, expected_output)]
    comparisons.extend(zip(("dX", *(f"parameter/{name}" for name in named)),
                           actual_gradients, expected_gradients))
    failures = []
    for name, actual, expected in comparisons:
        statistics = _native_comparison_metrics(actual, expected)
        for metric, value in statistics.items():
            record_property(f"{name}/{metric}", value)
        if statistics["failures"]:
            failures.append(f"{name}: {statistics}")
        if actual.dtype != torch.float32 or expected.dtype != torch.float32:
            failures.append(f"{name}: expected FP32 tensors, got {actual.dtype}/{expected.dtype}")
    assert not failures, "\n".join(failures)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("n", (27648, 27649))
@pytest.mark.parametrize("source_family,class_name", NATIVE_SOURCES)
def test_optional_original_source_cancellation_acceptance(
        version, n, source_family, class_name, record_property):
    source_class = _backward_helpers()._load_source(source_family, class_name)
    source_path = Path(os.environ[f"FASTEQ_EQUIFORMER_{source_family}_LAYER_NORM"])
    record_property("source/path", str(source_path.resolve()))
    record_property("source/sha256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    record_property("source/class", class_name)
    record_property("reference", "original FP32 source forward and autograd")
    source = source_class(3, 7).cuda().train()
    x = torch.ones(n, 16, 7, device="cuda")
    x[:, 0] = 0.
    x.requires_grad_()
    rx = x.detach().clone().requires_grad_()
    dy = torch.full_like(x, 1e-8)
    dy[:32] = 1.
    dy[-32:] = -1.
    adapter = from_reference(source, version=version)
    actual_output = adapter(x)
    expected_output = source(rx)
    named = dict(source.named_parameters())
    parameters = tuple(named.values())
    actual_gradients = torch.autograd.grad(actual_output, (x, *parameters), dy)
    expected_gradients = torch.autograd.grad(expected_output, (rx, *parameters), dy)
    comparisons = [("forward", actual_output, expected_output)]
    comparisons.extend(zip(("dX", *(f"parameter/{name}" for name in named)),
                           actual_gradients, expected_gradients))
    failures = []
    for name, actual, expected in comparisons:
        statistics = _native_comparison_metrics(actual, expected)
        for metric, value in statistics.items():
            record_property(f"{name}/{metric}", value)
        if statistics["failures"]:
            failures.append(f"{name}: {statistics}")
    assert not failures, "\n".join(failures)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("terms", ((1e8, -1e8, 1.), (1e8, 1., -1e8)),
                         ids=("large_cancel_then_small", "small_between_large"))
@pytest.mark.parametrize("source_family,class_name", NATIVE_SOURCES)
def test_optional_original_source_component_cancellation_acceptance(
        version, terms, source_family, class_name, record_property):
    source_class = _backward_helpers()._load_source(source_family, class_name)
    source_path = Path(os.environ[f"FASTEQ_EQUIFORMER_{source_family}_LAYER_NORM"])
    record_property("source/path", str(source_path.resolve()))
    record_property("source/sha256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    record_property("source/class", class_name)
    record_property("reference", "original FP32 source forward and autograd")
    record_property("upstream/degree1_channel0", str(terms))
    source = source_class(3, 7).cuda().train()
    x = torch.ones(1, 16, 7, device="cuda")
    x[:, 0] = 0.
    x.requires_grad_()
    rx = x.detach().clone().requires_grad_()
    dy = torch.zeros_like(x)
    dy[0, 1:4, 0] = torch.tensor(terms, device="cuda", dtype=torch.float32)
    adapter = from_reference(source, version=version)
    actual_output, expected_output = adapter(x), source(rx)
    named = dict(source.named_parameters())
    parameters = tuple(named.values())
    actual_gradients = torch.autograd.grad(actual_output, (x, *parameters), dy)
    expected_gradients = torch.autograd.grad(expected_output, (rx, *parameters), dy)
    comparisons = [("forward", actual_output, expected_output)]
    comparisons.extend(zip(("dX", *(f"parameter/{name}" for name in named)),
                           actual_gradients, expected_gradients))
    failures = []
    # Keep dX diagnostics separate from dgamma: this large upstream gradient
    # also stresses cancellation in the normalization derivative itself.
    # Record every metric before enforcing the unchanged native tolerance.
    for name, actual, expected in comparisons:
        statistics = _native_comparison_metrics(actual, expected)
        for metric, value in statistics.items():
            record_property(f"{name}/{metric}", value)
        if statistics["failures"]:
            failures.append(f"{name}: {statistics}")
    assert not failures, "\n".join(failures)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("source_family,class_name", NATIVE_SOURCES)
def test_optional_original_source_unweighted_component_cancellation(
        version, source_family, class_name, record_property):
    source_class = _backward_helpers()._load_source(source_family, class_name)
    source_path = Path(os.environ[f"FASTEQ_EQUIFORMER_{source_family}_LAYER_NORM"])
    record_property("source/path", str(source_path.resolve()))
    record_property("source/sha256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    record_property("source/class", class_name)
    record_property("source/affine", False)
    record_property("reference", "original FP32 source forward and autograd")
    source = source_class(3, 7, affine=False).cuda().train()
    x = torch.ones(1, 16, 7, device="cuda")
    x[:, 0] = 0.
    x.requires_grad_()
    rx = x.detach().clone().requires_grad_()
    dy = torch.zeros_like(x)
    dy[0, 1:4, 0] = torch.tensor((1e8, -1e8, 1.), device="cuda")
    actual_output = from_reference(source, version=version)(x)
    expected_output = source(rx)
    actual_dx = torch.autograd.grad(actual_output, x, dy)[0]
    expected_dx = torch.autograd.grad(expected_output, rx, dy)[0]
    failures = []
    # Without the affine multiplication node, source dR reduces m and c
    # together instead of following the weighted broadcast graph.
    for name, actual, expected in (("forward", actual_output, expected_output),
                                   ("dX", actual_dx, expected_dx)):
        statistics = _native_comparison_metrics(actual, expected)
        for metric, value in statistics.items():
            record_property(f"{name}/{metric}", value)
        if statistics["failures"]:
            failures.append(f"{name}: {statistics}")
    assert not failures, "\n".join(failures)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("terms", ((1e8, -1e8, 1.), (1e8, 1., -1e8)),
                         ids=("large_cancel_then_small", "small_between_large"))
def test_optional_original_source_shared_scale_degree_accumulation(
        version, terms, record_property):
    class_name = "EquivariantLayerNormArraySphericalHarmonics"
    source_class = _backward_helpers()._load_source("V2", class_name)
    source_path = Path(os.environ["FASTEQ_EQUIFORMER_V2_LAYER_NORM"])
    record_property("source/path", str(source_path.resolve()))
    record_property("source/sha256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    record_property("source/class", class_name)
    record_property("reference", "original FP32 source forward and autograd")
    record_property("upstream/degrees1_2_3", str(terms))
    source = source_class(3, 7).cuda().train()
    x = torch.ones(1, 16, 7, device="cuda")
    x[:, 0] = 0.
    x.requires_grad_()
    rx = x.detach().clone().requires_grad_()
    dy = torch.zeros_like(x)
    # Only the first component/channel of each degree contributes. The source
    # graph accumulates the shared scale's branches in reverse degree order.
    for degree, term in zip((1, 2, 3), terms):
        dy[0, degree**2, 0] = term
    actual_output = from_reference(source, version=version)(x)
    expected_output = source(rx)
    named = dict(source.named_parameters())
    parameters = tuple(named.values())
    actual_gradients = torch.autograd.grad(actual_output, (x, *parameters), dy)
    expected_gradients = torch.autograd.grad(expected_output, (rx, *parameters), dy)
    comparisons = [("forward", actual_output, expected_output)]
    comparisons.extend(zip(("dX", *(f"parameter/{name}" for name in named)),
                           actual_gradients, expected_gradients))
    failures = []
    for name, actual, expected in comparisons:
        statistics = _native_comparison_metrics(actual, expected)
        for metric, value in statistics.items():
            record_property(f"{name}/{metric}", value)
        if statistics["failures"]:
            failures.append(f"{name}: {statistics}")
    assert not failures, "\n".join(failures)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("source_family,class_name,n,lmax,channels", (
    ("V3", "EquivariantMergeLayerNorm", 17, 4, 512),
    ("V3", "EquivariantSeparableLayerNorm", 17, 5, 256),
    ("V3", "EquivariantLayerNorm", 108, 0, 8192),
    ("V2", "EquivariantLayerNormArraySphericalHarmonics", 108, 0, 8192),
))
def test_optional_original_source_wide_reduction_acceptance(
        version, source_family, class_name, n, lmax, channels, record_property):
    source_class = _backward_helpers()._load_source(source_family, class_name)
    source_path = Path(os.environ[f"FASTEQ_EQUIFORMER_{source_family}_LAYER_NORM"])
    record_property("source/path", str(source_path.resolve()))
    record_property("source/sha256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    record_property("source/class", class_name)
    record_property("reference", "original FP32 source forward and autograd")
    # These shapes fit the existing execution-tile limit but need native
    # block-y cooperation when reducing a contiguous component/channel axis.
    torch.manual_seed(20260914)
    source = source_class(lmax, channels).cuda().train()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.randn_like(parameter))
    x = torch.randn(n, (lmax + 1)**2, channels, device="cuda", requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    actual_output = from_reference(source, version=version)(x)
    expected_output = source(rx)
    dy = torch.randn_like(actual_output)
    # A split-affine lmax=0 source can declare an empty higher-degree weight.
    # It is absent from the source graph; compare every nonempty parameter.
    all_named = dict(source.named_parameters())
    named = {name: parameter for name, parameter in all_named.items()
             if parameter.numel()}
    record_property("source/empty_parameters", ",".join(
        name for name, parameter in all_named.items() if not parameter.numel()))
    parameters = tuple(named.values())
    actual_gradients = torch.autograd.grad(actual_output, (x, *parameters), dy)
    expected_gradients = torch.autograd.grad(expected_output, (rx, *parameters), dy)
    comparisons = [("forward", actual_output, expected_output)]
    comparisons.extend(zip(("dX", *(f"parameter/{name}" for name in named)),
                           actual_gradients, expected_gradients))
    failures = []
    for name, actual, expected in comparisons:
        statistics = _native_comparison_metrics(actual, expected)
        for metric, value in statistics.items():
            record_property(f"{name}/{metric}", value)
        if statistics["failures"]:
            failures.append(f"{name}: {statistics}")
        if actual.dtype != torch.float32 or expected.dtype != torch.float32:
            failures.append(f"{name}: expected FP32 tensors, got {actual.dtype}/{expected.dtype}")
    assert not failures, "\n".join(failures)


@torch.no_grad()
def _native_comparison_metrics_chunked(actual, expected, chunk_elements=2**22):
    """Compare every FP32 value without materializing full FP64 tensors."""
    assert actual.shape == expected.shape
    assert 0 < chunk_elements <= 2**22
    actual_flat = actual.detach().reshape(-1)
    expected_flat = expected.detach().reshape(-1)
    max_abs = torch.zeros((), device=actual.device, dtype=torch.float64)
    max_ratio = max_abs.clone()
    squared_error = max_abs.clone()
    squared_reference = max_abs.clone()
    failures = torch.zeros((), device=actual.device, dtype=torch.int64)
    differing_elements = failures.clone()
    for start in range(0, actual_flat.numel(), chunk_elements):
        actual64 = actual_flat[start:start + chunk_elements].double()
        expected64 = expected_flat[start:start + chunk_elements].double()
        error = (actual64 - expected64).abs()
        tolerance = 5e-5 + 5e-4 * expected64.abs()
        failed = ((error > tolerance) | ~torch.isfinite(actual64)
                  | ~torch.isfinite(expected64))
        max_abs = torch.maximum(max_abs, error.max())
        max_ratio = torch.maximum(max_ratio, (error / tolerance).max())
        failures += failed.sum()
        differing_elements += (actual64 != expected64).sum()
        squared_error += error.square().sum()
        squared_reference += expected64.square().sum()
    return {
        "numel": actual.numel(),
        "max_abs": max_abs.item(),
        "relative_l2": (squared_error
                        / squared_reference.clamp_min(1e-300)).sqrt().item(),
        "failures": failures.item(),
        "differing_elements": differing_elements.item(),
        "max_tolerance_ratio": max_ratio.item(),
    }


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("class_name,n", (
    pytest.param("EquivariantLayerNorm", 32768, id="norm-32768"),
    pytest.param("EquivariantLayerNorm", 131072, id="norm-131072"),
    pytest.param("EquivariantMergeLayerNorm", 262144, id="merge-262144"),
    pytest.param("EquivariantSeparableLayerNorm", 524288, id="separable-524288"),
))
def test_optional_original_source_scaling_precision_regressions(
        version, class_name, n, record_property):
    """Reproduce the four native-FP32 failures from the scaling benchmark."""
    source_class = _backward_helpers()._load_source("V3", class_name)
    source_path = Path(os.environ["FASTEQ_EQUIFORMER_V3_LAYER_NORM"])
    record_property("source/path", str(source_path.resolve()))
    record_property("source/sha256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    record_property("source/class", class_name)
    record_property("reference", "original FP32 source forward and autograd")
    record_property("seed", 20260914)
    record_property("layout", "NKC")
    record_property("shape", str((n, 16, 128)))
    record_property("full_tensor_comparison", True)
    record_property("comparison/chunk_elements", 2**22)
    record_property("comparison/atol", 5e-5)
    record_property("comparison/rtol", 5e-4)

    # Preserve the benchmark RNG order: construct the native module,
    # randomize its parameters, draw x, then draw dY before either forward.
    torch.manual_seed(20260914)
    source = source_class(3, 128).cuda().train()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.randn_like(parameter))
    x = torch.randn(n, 16, 128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    named = dict(source.named_parameters())
    parameters = tuple(named.values())
    adapter = from_reference(source, version=version)

    # Share the exact input and parameters. Finish one backward before the
    # next forward so both large sets of saved activations are not live together.
    actual_output = adapter(x)
    actual_gradients = torch.autograd.grad(actual_output, (x, *parameters), dy)
    expected_output = source(x)
    expected_gradients = torch.autograd.grad(expected_output, (x, *parameters), dy)

    comparisons = [("forward", actual_output, expected_output)]
    comparisons.extend(zip(("dX", *(f"parameter/{name}" for name in named)),
                           actual_gradients, expected_gradients))
    failures = []
    for name, actual, expected in comparisons:
        statistics = _native_comparison_metrics_chunked(actual, expected)
        for metric, value in statistics.items():
            record_property(f"{name}/{metric}", value)
        if statistics["failures"]:
            failures.append(f"{name}: {statistics}")
        if actual.dtype != torch.float32 or expected.dtype != torch.float32:
            failures.append(f"{name}: expected FP32 tensors, got {actual.dtype}/{expected.dtype}")
    assert not failures, "\n".join(failures)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
def test_generic_large_tile_skips_unused_source_schedules(version, record_property):
    """A supported generic tile must not require an unused native m schedule."""
    record_property("reference", "independent FP32 reference_forward equations")
    torch.manual_seed(20260915)
    n, lmax, channels = 32, 180, 2
    spec = EquivariantNormSpec.from_preset(
        lmax=lmax, channels=channels, grouping="all")
    op = TritonEquivariantNorm(spec, version=version, device="cuda")
    x = torch.randn(n, (lmax + 1)**2, channels, device="cuda")
    weight = torch.randn(lmax + 1, channels, device="cuda")
    bias = torch.randn(channels, device="cuda")
    with torch.no_grad():
        actual = op(x, weight=weight, bias=bias)
        expected = reference_forward(x, spec, weight=weight, bias=bias).output
    statistics = _native_comparison_metrics(actual, expected)
    for metric, value in statistics.items():
        record_property(f"forward/{metric}", value)
    assert not statistics["failures"], statistics


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("class_name", (
    "EquivariantLayerNorm", "EquivariantMergeLayerNorm"))
@pytest.mark.parametrize("offset", (1, 2, 3))
def test_optional_original_source_scalar_pointer_alignment(
        version, class_name, offset, record_property):
    """Preserve the original scalar slice's address in its native mean order."""
    source_class = _backward_helpers()._load_source("V3", class_name)
    source_path = Path(os.environ["FASTEQ_EQUIFORMER_V3_LAYER_NORM"])
    record_property("source/path", str(source_path.resolve()))
    record_property("source/sha256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    record_property("source/class", class_name)
    record_property("reference", "original FP32 source forward and autograd")
    source_path = Path(os.environ["FASTEQ_EQUIFORMER_V3_LAYER_NORM"])
    record_property("source/path", str(source_path.resolve()))
    record_property("source/sha256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    record_property("source/class", class_name)
    record_property("reference", "original FP32 source forward and autograd")
    torch.manual_seed(20260915)
    n, k, channels = 17, 16, 129
    source = source_class(3, channels).cuda().train()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.randn_like(parameter))
    # This seed exposes the old alignment error for each of the three offsets.
    torch.manual_seed(1601)
    storage = torch.randn(n * k * channels + offset, device="cuda")
    x = storage[offset:].view(n, k, channels).requires_grad_()
    dy = torch.randn_like(x)
    named = dict(source.named_parameters())
    parameters = tuple(named.values())
    adapter = from_reference(source, version=version)
    actual = adapter.forward_with_stats(x)
    with torch.no_grad():
        expected_mean = x[:, 0:1, :].mean(dim=2).flatten()
    record_property("input/pointer_mod16", x.data_ptr() % 16)
    torch.testing.assert_close(actual.mean, expected_mean, atol=0, rtol=0)

    actual_gradients = torch.autograd.grad(actual.output, (x, *parameters), dy)
    expected_output = source(x)
    expected_gradients = torch.autograd.grad(expected_output, (x, *parameters), dy)
    comparisons = [("forward", actual.output, expected_output)]
    comparisons.extend(zip(("dX", *(f"parameter/{name}" for name in named)),
                           actual_gradients, expected_gradients))
    failures = []
    for name, actual_value, expected_value in comparisons:
        statistics = _native_comparison_metrics(actual_value, expected_value)
        for metric, value in statistics.items():
            record_property(f"{name}/{metric}", value)
        if statistics["failures"]:
            failures.append(f"{name}: {statistics}")
    assert not failures, "\n".join(failures)


@GPU
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("channels", (129, 257))
def test_optional_original_source_uncentered_merge_knc_alignment(
        version, channels, record_property):
    """Without scalar centering, Merge's squared input retains KNC row order."""
    source_class = _backward_helpers()._load_source("V3", "EquivariantMergeLayerNorm")
    source_path = Path(os.environ["FASTEQ_EQUIFORMER_V3_LAYER_NORM"])
    record_property("source/path", str(source_path.resolve()))
    record_property("source/sha256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    record_property("source/class", "EquivariantMergeLayerNorm")
    record_property("reference", "original FP32 source forward and autograd")
    source_path = Path(os.environ["FASTEQ_EQUIFORMER_V3_LAYER_NORM"])
    record_property("source/path", str(source_path.resolve()))
    record_property("source/sha256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    record_property("source/class", "EquivariantMergeLayerNorm")
    record_property("reference", "original FP32 source forward and autograd")
    torch.manual_seed(20260914)
    source = source_class(3, channels, centering=False).cuda().train()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.randn_like(parameter))
    x = torch.randn(17, 16, channels, device="cuda")
    x = x.transpose(0, 1).contiguous().transpose(0, 1).detach().requires_grad_()
    dy = torch.randn_like(x)
    named = dict(source.named_parameters())
    parameters = tuple(named.values())
    actual = from_reference(source, version=version).forward_with_stats(x)
    with torch.no_grad():
        rows = x.square().mean(dim=2, keepdim=True)
        moment = torch.einsum("ai,nic->nac", source.balance_degree_weight, rows).reshape(17, 1)
        rstd = (moment + source.eps).pow(-0.5)
    torch.testing.assert_close(actual.moments, moment, atol=0, rtol=0)
    torch.testing.assert_close(actual.rstd, rstd, atol=0, rtol=0)

    actual_gradients = torch.autograd.grad(actual.output, (x, *parameters), dy)
    expected_output = source(x)
    expected_gradients = torch.autograd.grad(expected_output, (x, *parameters), dy)
    comparisons = [("forward", actual.output, expected_output)]
    comparisons.extend(zip(("dX", *(f"parameter/{name}" for name in named)),
                           actual_gradients, expected_gradients))
    failures = []
    for name, actual_value, expected_value in comparisons:
        statistics = _native_comparison_metrics(actual_value, expected_value)
        for metric, value in statistics.items():
            record_property(f"{name}/{metric}", value)
        if statistics["failures"]:
            failures.append(f"{name}: {statistics}")
    assert not failures, "\n".join(failures)
