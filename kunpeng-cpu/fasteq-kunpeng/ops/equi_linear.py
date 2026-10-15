from fasteq.cpu import load_extension
from ._autograd import NativeLinearFunction, PackedLinearFunction

def fast_equi_linear(x, weight, metadata):
    """Forward and input gradient only, without bias instructions."""
    return NativeLinearFunction.apply(x, weight, load_extension(), metadata)

def fast_equi_linear_packed(x, metadata):
    return PackedLinearFunction.apply(x, metadata["packed_weight"], load_extension(), metadata)
