from fasteq.cpu import load_extension
from ._autograd import NativeFCTPFunction, PackedFCTPFunction

def fast_fctp(x1, x2, weight, metadata, one_hot=False):
    """Forward and grad_x1 only; x2 and weight must be fixed."""
    return NativeFCTPFunction.apply(x1, x2, weight, load_extension(), metadata, one_hot)

def fast_fctp_packed(x, element, metadata):
    """Fixed one-hot scalar attributes and prepacked shared weights."""
    return PackedFCTPFunction.apply(x, metadata["packed_weight"], element, load_extension(), metadata)
