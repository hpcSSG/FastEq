"""GPU cases adapted from tests/nn, excluding models and tensor products."""

import importlib
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from e3nn import nn as reference

optimized = importlib.import_module(os.environ.get("E3NN_TRITON_PACKAGE", "fasteq.e3nn.triton") + ".nn")


def close(actual, expected, tol=3e-5):
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            close(a, b, tol)
    else:
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol, equal_nan=True)


@pytest.mark.parametrize("affine,reduce,instance", [
    (True, "mean", False), (False, "max", False), (True, "mean", True),
])
def test_batchnorm_train_updates_and_eval(affine, reduce, instance):
    kwargs = dict(affine=affine, reduce=reduce, instance=instance)
    actual = optimized.BatchNorm("2x0e+1x1o", **kwargs).cuda()
    expected = reference.BatchNorm("2x0e+1x1o", **kwargs).cuda()
    expected.load_state_dict(actual.state_dict())
    x = torch.randn(4, 3, 5, device="cuda")
    with torch.no_grad():
        close(actual(x), expected(x))
        if not instance:
            close(actual.running_mean, expected.running_mean)
            close(actual.running_var, expected.running_var)
        actual.eval()
        expected.eval()
        close(actual(x), expected(x))


def test_batchnorm_affine_gradient_fallback():
    actual = optimized.BatchNorm("2x0e+1x1o").cuda().eval()
    expected = reference.BatchNorm("2x0e+1x1o").cuda().eval()
    expected.load_state_dict(actual.state_dict())
    x = torch.randn(2, 5, device="cuda", requires_grad=True)
    a = actual(x).sum()
    b = expected(x).sum()
    ga = torch.autograd.grad(a, (x, actual.weight, actual.bias), retain_graph=True)
    gb = torch.autograd.grad(b, (x, expected.weight, expected.bias))
    for lhs, rhs in zip(ga, gb):
        close(lhs, rhs)

