from fasteq.cpu import load_extension
from ._autograd import NativeSymmetricContractionFunction, NativeSymmetricContractionFusedFunction

def fast_stc(U, weight, x, element):
    """One contraction step; not the full generic MACE contraction recursion."""
    return NativeSymmetricContractionFunction.apply(U, weight, x, element, load_extension())

def fast_stc_scalar_fused(x, element, metadata):
    """Complete packed correlation=3 scalar-output contraction."""
    return NativeSymmetricContractionFusedFunction.apply(x, element, load_extension(), metadata)
