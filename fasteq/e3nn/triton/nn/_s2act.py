"""S2 activation using Triton grid projection on CUDA inputs."""

import torch

from e3nn.nn._s2act import S2Activation as _ReferenceS2Activation
from ..o3._s2grid import ToS2Grid, FromS2Grid
from ..o3._rotation import rand_angles


class S2Activation(_ReferenceS2Activation):
    def __init__(self, irreps, act, res, normalization="component", lmax_out=None, random_rot=False):
        super().__init__(irreps, act, res, normalization, lmax_out, random_rot)
        self._triton_res = res
        self._triton_normalization = normalization
        self._triton_lmax_in = self.irreps_in.lmax
        self._triton_lmax_out = self.irreps_out.lmax
        self._triton_grid_key = None

    def forward(self, features):
        if (not features.is_cuda or features.requires_grad
                or (torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()))
                or features.dtype not in (torch.float32, torch.float64)):
            return super().forward(features)
        key = (features.device, features.dtype)
        if (self._triton_grid_key != key
                or self._cuda_to_s2.betas.device != features.device
                or self._cuda_to_s2.betas.dtype != features.dtype):
            self._cuda_to_s2 = ToS2Grid(self._triton_lmax_in, self._triton_res,
                                        normalization=self._triton_normalization,
                                        dtype=features.dtype, device=features.device)
            self._cuda_from_s2 = FromS2Grid(self._triton_res, self._triton_lmax_out,
                                            normalization=self._triton_normalization,
                                            lmax_in=self._triton_lmax_in,
                                            dtype=features.dtype, device=features.device)
            self._triton_grid_key = key
        if self.random_rot:
            abc = rand_angles(dtype=features.dtype, device=features.device)
            features = torch.einsum("ij,...j->...i", self.irreps_in.D_from_angles(*abc), features)
        features = self._cuda_from_s2(self.act(self._cuda_to_s2(features)))
        if self.random_rot:
            features = torch.einsum("ij,...j->...i", self.irreps_out.D_from_angles(*abc).T, features)
        return features


__all__ = ["S2Activation"]
