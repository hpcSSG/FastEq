from fasteq.cpu import load_extension
from ._geometry_autograd import NativeEdgeGeometryFunction

def fast_edge_geometry(positions, edge_index, shifts, normalize=False, eps=1e-9):
    return NativeEdgeGeometryFunction.apply(positions, edge_index, shifts, load_extension(), normalize, eps)
