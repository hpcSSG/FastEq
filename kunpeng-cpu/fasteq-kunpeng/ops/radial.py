import torch
from fasteq.cpu import load_extension
from ._autograd import NativeRadialMLPFunction
from ._geometry_autograd import NativeRadialEmbeddingFunction

def fast_radial_mlp(x, metadata):
    return NativeRadialMLPFunction.apply(x, load_extension(), metadata)

def fast_radial_embedding(lengths, bessel_weights, r_max, cutoff_p):
    # The original C++ ABI reads r_max through data_ptr<double>().
    radius = torch.as_tensor(r_max, dtype=torch.float64, device="cpu")
    if radius.numel() != 1 or not torch.isfinite(radius).all() or radius.item() <= 0:
        raise ValueError("r_max must be a finite positive scalar")
    if radius.requires_grad:
        raise ValueError("r_max gradients are not supported")
    return NativeRadialEmbeddingFunction.apply(lengths, bessel_weights, radius.contiguous(), cutoff_p, load_extension())
