"""Dense training layers keep PyTorch GEMM and arbitrary activation support."""

from e3nn.nn._fc import _Layer, FullyConnectedNet


__all__ = ["_Layer", "FullyConnectedNet"]
