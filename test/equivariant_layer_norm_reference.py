"""Torch equations used only by the equivariant LayerNorm tests."""

import torch

from fasteq.triton.fused_equivariant_layer_norm_common import NormResult


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
