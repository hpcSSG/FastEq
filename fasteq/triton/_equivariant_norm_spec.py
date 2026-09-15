"""Semantic configuration and PyTorch reference for equivariant normalization.

stats_weights[g][l] multiplies EVERY m component in degree l, after
channel-mean squared magnitude. output_group[l] selects that degree's rstd.
"""
from dataclasses import dataclass
import math
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class EquivariantNormSpec:
    lmax: int
    channels: int
    stats_weights: tuple[tuple[float, ...], ...]
    output_group: tuple[int, ...]
    center_scalar: bool = True
    eps: float = 1e-5

    def __post_init__(self):
        if type(self.lmax) is not int or self.lmax < 0:
            raise ValueError('lmax must be a nonnegative integer')
        if type(self.channels) is not int or self.channels < 1:
            raise ValueError('channels must be a positive integer')
        if type(self.center_scalar) is not bool:
            raise ValueError('center_scalar must be boolean')
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('eps must be finite and positive')
        weights = tuple(tuple(float(v) for v in row) for row in self.stats_weights)
        groups = tuple(self.output_group)
        if not weights or any(len(row) != self.lmax + 1 for row in weights):
            raise ValueError('stats_weights must have shape [G, lmax+1], G > 0')
        if any(not math.isfinite(v) or v < 0 for row in weights for v in row):
            raise ValueError('statistic weights must be finite and nonnegative')
        if any(not any(v > 0 for v in row) for row in weights):
            raise ValueError('each statistic must have a nonzero source')
        if len(groups) != self.lmax + 1 or any(
            type(g) is not int or not 0 <= g < len(weights) for g in groups
        ):
            raise ValueError('output_group must select one valid group per degree')
        if set(groups) != set(range(len(weights))):
            raise ValueError('every statistic must be used by an output degree')
        object.__setattr__(self, 'stats_weights', weights)
        object.__setattr__(self, 'output_group', groups)

    @property
    def components(self):
        return (self.lmax + 1) ** 2

    @property
    def num_stats(self):
        return len(self.stats_weights)

    @classmethod
    def from_preset(cls, *, lmax, channels, grouping='all',
                    weighting='degree_balanced', center_scalar=True, eps=1e-5):
        if type(lmax) is not int or lmax < 0:
            raise ValueError('lmax must be a nonnegative integer')
        if grouping == 'per_degree':
            sources = [(l,) for l in range(lmax + 1)]
        elif grouping == 'scalar_high':
            sources = [(0,)] + ([tuple(range(1, lmax + 1))] if lmax else [])
        elif grouping == 'all':
            sources = [tuple(range(lmax + 1))]
        else:
            raise ValueError('unknown grouping preset')
        if weighting not in ('component', 'degree_balanced', 'norm'):
            raise ValueError('unknown weighting preset')
        weights, output_group = [], [0] * (lmax + 1)
        for g, degrees in enumerate(sources):
            row = [0.] * (lmax + 1)
            for l in degrees:
                if weighting == 'degree_balanced':
                    row[l] = 1. / (len(degrees) * (2 * l + 1))
                elif weighting == 'component':
                    row[l] = 1. / sum(2 * j + 1 for j in degrees)
                else:
                    row[l] = 1.
                output_group[l] = g
            weights.append(tuple(row))
        return cls(lmax, channels, tuple(weights), tuple(output_group), center_scalar, eps)


class NormResult(NamedTuple):
    output: torch.Tensor
    mean: torch.Tensor | None   # [N]; absent if scalar centering is disabled
    moments: torch.Tensor       # [N,G]; not generally ordinary variances
    rstd: torch.Tensor          # [N,G]


def reference_forward(x, spec, *, weight=None, bias=None):
    """Readable PyTorch math oracle, NOT a performance/fused implementation.

    weight: None, packed [L+1,C], or (scalar [C], higher [L,C]).
    bias: None or scalar-only [C]. Split weights need no per-call concatenation.
    Reference calculations support float32/float64; GPU forward/backward uses FP32.
    """
    if x.ndim != 3 or tuple(x.shape[1:]) != (spec.components, spec.channels):
        raise ValueError('expected [N,(lmax+1)^2,channels]')
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError('the math reference supports float32 and float64')
    n, k, c = x.shape
    if not (x.is_contiguous() or x.stride() == (c, n*c, 1)):
        raise ValueError('initial contract supports NKC and KNC storage')

    def check_param(tensor, shape):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f'expected parameter shape {shape}')
        if tensor.device != x.device or tensor.dtype != x.dtype:
            raise ValueError('parameters must match input device and dtype')

    if weight is not None:
        if isinstance(weight, tuple):
            if len(weight) != 2:
                raise ValueError('split weight must contain scalar and higher tensors')
            check_param(weight[0], (c,))
            check_param(weight[1], (spec.lmax, c))
        else:
            check_param(weight, (spec.lmax + 1, c))
    if bias is not None:
        check_param(bias, (c,))

    mean = x[:, 0, :].mean(-1) if spec.center_scalar else None
    blocks = []
    for l in range(spec.lmax + 1):
        z = x[:, l*l:(l+1)**2, :]
        if l == 0 and mean is not None:
            z = z - mean[:, None, None]
        blocks.append(z)
    # s[n,l] = sum_m mean_c Z[n,l,m,c]^2; no division by (2l+1) here.
    s = torch.stack([z.square().sum(1).mean(-1) for z in blocks], dim=1)
    # Zero means an absent source, not 0 * an already-overflowed/NaN value.
    moments = torch.stack([
        torch.stack([s[:, l] * coefficient for l, coefficient in enumerate(row)
                     if coefficient > 0], dim=1).sum(1)
        for row in spec.stats_weights
    ], dim=1)
    rstd = torch.rsqrt(moments + spec.eps)
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    for l, z in enumerate(blocks):
        scale = rstd[:, spec.output_group[l], None, None]
        if weight is not None:
            gamma = (weight[0] if l == 0 else weight[1][l-1]) if isinstance(weight, tuple) else weight[l]
            scale = scale * gamma[None, None, :]
        y = z * scale
        if l == 0 and bias is not None:
            y = y + bias[None, None, :]
        output[:, l*l:(l+1)**2, :] = y
    return NormResult(output, mean, moments, rstd)
