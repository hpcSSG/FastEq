"""Use the Triton norm on supported inference calls."""

from e3nn.nn._normact import NormActivation as _ReferenceNormActivation
from ..o3._norm import Norm


class NormActivation(_ReferenceNormActivation):
    def __init__(self, irreps_in, scalar_nonlinearity, normalize=True, epsilon=None, bias=False):
        super().__init__(irreps_in, scalar_nonlinearity, normalize, epsilon, bias)
        self.norm = Norm(self.irreps_in, squared=(self.epsilon is not None))


__all__ = ["NormActivation"]
