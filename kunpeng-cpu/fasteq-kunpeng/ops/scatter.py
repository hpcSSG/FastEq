from fasteq.cpu import load_extension
from ._geometry_autograd import NativeScatterSumFunction

def fast_scatter_sum(src, index, dim_size):
    return NativeScatterSumFunction.apply(src, index, dim_size, load_extension())
