"""e3nn uvu tensor product; not a cuEquivariance descriptor adapter."""
from fasteq.cpu import load_extension
from ._autograd import NativeTPUFunction

def fast_cwtp(x1, x2, weight, metadata):
    return NativeTPUFunction.apply(x1, x2, weight, load_extension(), metadata)
