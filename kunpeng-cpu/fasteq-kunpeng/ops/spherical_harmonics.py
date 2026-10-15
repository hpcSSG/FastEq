import math
from fasteq.cpu import load_extension
from ._autograd import NativeSHFunction

def fast_spherical_harmonics(x, normalization="integral"):
    """Normalized Cartesian SH for l=0..3; integral or component output."""
    if normalization not in ("integral", "component"):
        raise ValueError("normalization must be integral or component")
    out = NativeSHFunction.apply(x, load_extension())
    return out if normalization == "integral" else out * math.sqrt(4 * math.pi)
