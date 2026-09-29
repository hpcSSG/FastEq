"""SO3 activation using Triton grid contractions on CUDA inputs."""

import torch

from e3nn.nn._so3act import SO3Activation as _ReferenceSO3Activation
from ..o3._so3grid import SO3Grid


class SO3Activation(_ReferenceSO3Activation):
    def __init__(self, lmax_in, lmax_out, act, resolution, *, normalization="component", aspect_ratio=2):
        super().__init__(lmax_in, lmax_out, act, resolution,
                         normalization=normalization, aspect_ratio=aspect_ratio)
        self._triton_resolution = resolution
        self._triton_normalization = normalization
        self._triton_aspect_ratio = aspect_ratio
        self._triton_grid_key = None

    def forward(self, features):
        if (not features.is_cuda or features.requires_grad
                or (torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()))
                or features.dtype not in (torch.float32, torch.float64)):
            return super().forward(features)
        key = (features.device, features.dtype)
        if (self._triton_grid_key != key
                or self._cuda_grid_in.D.device != features.device
                or self._cuda_grid_in.D.dtype != features.dtype):
            self._cuda_grid_in = SO3Grid(self.lmax_in, self._triton_resolution,
                                        normalization=self._triton_normalization,
                                        aspect_ratio=self._triton_aspect_ratio).to(
                                            device=features.device, dtype=features.dtype)
            self._cuda_grid_out = SO3Grid(self.lmax_out, self._triton_resolution,
                                         normalization=self._triton_normalization,
                                         aspect_ratio=self._triton_aspect_ratio).to(
                                             device=features.device, dtype=features.dtype)
            self._triton_grid_key = key
        return self._cuda_grid_out.from_grid(self.act(self._cuda_grid_in.to_grid(features)))


__all__ = ["SO3Activation"]
