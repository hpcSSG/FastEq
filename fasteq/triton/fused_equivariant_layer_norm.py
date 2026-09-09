"""Shared FP32 inference kernels for disjoint equivariant norm groups.

v0: scalar mean -> group statistics -> affine output (separate launches).
v1: one program per atom, sharing the input tile across statistic groups.
Both use the same statistics and affine helpers; no model/shape-specific tuning.
The version flag changes scheduling; both versions use the same math contract.
"""
import torch
import triton
import triton.language as tl

from ._equivariant_norm_spec import EquivariantNormSpec, NormResult, reference_forward


@triton.jit
def _scalar_mean(X, MU, C: tl.constexpr, SN: tl.constexpr, BC: tl.constexpr):
    n = tl.program_id(0)
    c = tl.arange(0, BC)
    x = tl.load(X + n * SN + c, c < C, 0.)
    tl.store(MU + n, tl.sum(x, 0) / C)


@triton.jit
def _moment(square, w, coefficient, C: tl.constexpr, ORDER: tl.constexpr,
            UNIFORM: tl.constexpr):
    if ORDER == 'channels_first':
        q = tl.sum(square, 1) / C
        if UNIFORM:
            # Source unbalanced component/norm path reduces before scaling.
            return tl.sum(q, 0) * coefficient
        return tl.sum(q * w, 0)
    else:
        if UNIFORM:
            q = tl.sum(square, 0) * coefficient
        else:
            q = tl.sum(square * w[:, None], 0)
        return tl.sum(q, 0) / C


@triton.jit
def _affine(z, r, k, c, valid, DEGREE, WEIGHT0, WEIGHTH, BIAS,
            HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr,
            WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr,
            BS: tl.constexpr, K: tl.constexpr):
    if HAS_WEIGHT:
        degree = tl.load(DEGREE + k, k < K, 0)
        if SPLIT:
            scalar_weight = tl.load(WEIGHT0 + c * WS0 + tl.zeros_like(k), valid & (k == 0), 0.)
            higher_weight = tl.load(WEIGHTH + (degree - 1) * WSL + c * WSC,
                                    valid & (k > 0), 0.)
            gamma = tl.where(k == 0, scalar_weight, higher_weight)
        else:
            gamma = tl.load(WEIGHT0 + degree * WSL + c * WSC, valid, 0.)
        y = z * (r * gamma)
    else:
        y = z * r
    if HAS_BIAS:
        beta = tl.load(BIAS + c * BS + tl.zeros_like(k), valid & (k == 0), 0.)
        y = tl.where(k == 0, y + beta, y)
    return y


@triton.jit
def _statistics(X, MU, MOMENTS, RSTD, BOUNDS, COEFFICIENT,
                C: tl.constexpr, G: tl.constexpr, SN: tl.constexpr, SK: tl.constexpr,
                CENTER: tl.constexpr, EPS: tl.constexpr,
                ORDER: tl.constexpr, UNIFORM: tl.constexpr,
                BK: tl.constexpr, BC: tl.constexpr):
    n, g = tl.program_id(0), tl.program_id(1)
    start = tl.load(BOUNDS + g)
    end = tl.load(BOUNDS + g + 1)
    k = start + tl.arange(0, BK)
    c = tl.arange(0, BC)
    valid = (k[:, None] < end) & (c[None, :] < C)
    x = tl.load(X + n * SN + k[:, None] * SK + c[None, :], valid, 0.)
    if CENTER:
        mu = tl.load(MU + n)
        z = tl.where(valid, x - tl.where(k[:, None] == 0, mu, 0.), 0.)
    else:
        z = x
    w = tl.load(COEFFICIENT + k, k < end, 0.)
    coefficient = tl.sum(tl.where(tl.arange(0, BK) == 0, w, 0.), 0)
    v = _moment(z * z, w, coefficient, C, ORDER, UNIFORM)
    r = tl.rsqrt(v + EPS)
    tl.store(MOMENTS + n * G + g, v)
    tl.store(RSTD + n * G + g, r)


@triton.jit
def _output(X, MU, RSTD, Y, DEGREE, GROUP, WEIGHT0, WEIGHTH, BIAS,
            N: tl.constexpr, K: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
            SN: tl.constexpr, SK: tl.constexpr, CENTER: tl.constexpr,
            HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr,
            WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr,
            BS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = i // (K * C)
    k = i // C % K
    c = i % C
    valid = i < N * K * C
    x = tl.load(X + n * SN + k * SK + c, valid, 0.)
    if CENTER:
        mu = tl.load(MU + n, valid, 0.)
        z = x - tl.where(k == 0, mu, 0.)
    else:
        z = x
    group = tl.load(GROUP + k, valid, 0)
    r = tl.load(RSTD + n * G + group, valid, 0.)
    y = _affine(z, r, k, c, valid, DEGREE, WEIGHT0, WEIGHTH, BIAS,
                HAS_WEIGHT, SPLIT, HAS_BIAS, WS0, WSL, WSC, BS, K)
    tl.store(Y + i, y, valid)


@triton.jit
def _fused(X, Y, MU, MOMENTS, RSTD, COEFFICIENT, DEGREE,
           WEIGHT0, WEIGHTH, BIAS, K: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
           SN: tl.constexpr, SK: tl.constexpr, CENTER: tl.constexpr,
           EPS: tl.constexpr, ORDER: tl.constexpr, UNIFORM: tl.constexpr,
           HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr,
           WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr, BS: tl.constexpr,
           SAVE: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
           GROUP_BOUNDS: tl.constexpr):
    n = tl.program_id(0)
    k = tl.arange(0, BK)
    c = tl.arange(0, BC)
    valid = (k[:, None] < K) & (c[None, :] < C)
    x = tl.load(X + n * SN + k[:, None] * SK + c[None, :], valid, 0.)
    if CENTER:
        scalar = tl.sum(tl.where(k[:, None] == 0, x, 0.), 0)
        mu = tl.sum(scalar, 0) / C
        z = tl.where(valid, x - tl.where(k[:, None] == 0, mu, 0.), 0.)
    else:
        z = x
    w = tl.load(COEFFICIENT + k, k < K, 0.)
    square = z * z
    row_rstd = tl.full((BK,), 0., tl.float32)
    for g in tl.static_range(G):
        start = GROUP_BOUNDS[g]
        end = GROUP_BOUNDS[g + 1]
        member = (k >= start) & (k < end)
        # Select absent sources before reduction: 0 * NaN is not isolation.
        group_square = tl.where(member[:, None], square, 0.)
        group_w = tl.where(member, w, 0.)
        coefficient = tl.sum(tl.where(k == start, w, 0.), 0)
        v = _moment(group_square, group_w, coefficient, C, ORDER, UNIFORM)
        r = tl.rsqrt(v + EPS)
        row_rstd = tl.where(member, r, row_rstd)
        if SAVE:
            tl.store(MOMENTS + n * G + g, v)
            tl.store(RSTD + n * G + g, r)
    y = _affine(z, row_rstd[:, None], k[:, None], c[None, :], valid, DEGREE, WEIGHT0,
                WEIGHTH, BIAS, HAS_WEIGHT, SPLIT, HAS_BIAS, WS0, WSL, WSC, BS, K)
    tl.store(Y + n * K * C + k[:, None] * C + c[None, :], y, valid)
    if SAVE:
        if CENTER:
            tl.store(MU + n, mu)


class TritonEquivariantNorm(torch.nn.Module):
    """Execution plan; affine parameters belong to the caller.

    GPU scope: FP32 inference, contiguous NKC / transposed KNC inputs,
    positive-stride packed or split weights. Each statistic must describe one
    contiguous, disjoint range of full degrees and scale exactly that range.
    Broader/overlapping specs remain valid math specs but are rejected here.
    """
    def __init__(self, spec, *, version='v1', reduction_order='channels_first',
                 device='cuda'):
        super().__init__()
        if not isinstance(spec, EquivariantNormSpec):
            raise TypeError('spec must be EquivariantNormSpec')
        if version not in ('v0', 'v1'):
            raise ValueError('version must be v0 or v1')
        if reduction_order not in ('components_first', 'channels_first'):
            raise ValueError('unknown reduction order')
        bounds, coefficients, degrees, groups = [0], [], [], []
        uniform = True
        for g, row in enumerate(spec.stats_weights):
            source = [l for l, a in enumerate(row) if a > 0]
            target = [l for l, h in enumerate(spec.output_group) if h == g]
            if (source != target or source != list(range(source[0], source[-1] + 1))
                    or source[0] ** 2 != bounds[-1]):
                raise NotImplementedError('GPU groups must be ordered contiguous disjoint '
                                          'degree ranges, with identical source/output support')
            bounds.append((source[-1] + 1) ** 2)
            uniform &= len({row[l] for l in source}) == 1
            for l in source:
                coefficients.extend([row[l]] * (2*l + 1))
                degrees.extend([l] * (2*l + 1))
                groups.extend([g] * (2*l + 1))
        if bounds[-1] != spec.components:
            raise NotImplementedError('groups must cover all degrees')
        self.spec, self.version = spec, version
        self.reduction_order = reduction_order
        self.uniform = uniform
        self.group_bounds = tuple(bounds)
        self.block_k = triton.next_power_of_2(max(b-a for a, b in zip(bounds, bounds[1:])))
        self.atom_block_k = triton.next_power_of_2(spec.components)
        self.block_c = triton.next_power_of_2(spec.channels)
        tile_k = self.block_k if version == 'v0' else self.atom_block_k
        if tile_k * self.block_c > 65536:
            raise NotImplementedError('execution tile exceeds 65536 padded elements')
        for name, values, dtype in [('bounds', bounds, torch.int32),
                                    ('coefficients', coefficients, torch.float32),
                                    ('degrees', degrees, torch.int32),
                                    ('groups', groups, torch.int32)]:
            self.register_buffer(name, torch.tensor(values, dtype=dtype, device=device),
                                 persistent=False)

    def _check(self, x, weight, bias):
        s = self.spec
        if x.ndim != 3 or tuple(x.shape[1:]) != (s.components, s.channels):
            raise ValueError('expected [N,(lmax+1)^2,channels]')
        if x.device.type != 'cuda' or x.dtype != torch.float32:
            raise TypeError('Triton backend supports CUDA FP32 inference only')
        if x.device != self.bounds.device or self.coefficients.dtype != torch.float32:
            raise ValueError('move the execution plan to the input device; keep it FP32')
        if not (x.is_contiguous() or x.stride() == (s.channels, x.shape[0]*s.channels, 1)):
            raise ValueError('supported input storage: NKC and KNC')
        if x.numel() > 2**31 - 1:
            raise NotImplementedError('this backend uses 32-bit tensor indexing')
        tensors = [x]

        def check(t, shape):
            if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape:
                raise ValueError(f'expected parameter shape {shape}')
            if t.device != x.device or t.dtype != x.dtype:
                raise ValueError('parameter device and dtype must match input')
            if any(a <= 0 for a in t.stride()):
                raise ValueError('parameter strides must be positive')
            if sum(max(a - 1, 0) * b for a, b in zip(t.shape, t.stride())) > 2**31 - 1:
                raise NotImplementedError('parameter offsets exceed 32-bit indexing')
            tensors.append(t)

        split = isinstance(weight, tuple)
        if weight is not None:
            if split:
                if len(weight) != 2:
                    raise ValueError('split weight must be (scalar, higher)')
                check(weight[0], (s.channels,))
                check(weight[1], (s.lmax, s.channels))
            else:
                check(weight, (s.lmax+1, s.channels))
        if bias is not None:
            check(bias, (s.channels,))
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            raise RuntimeError('inference only: use torch.no_grad() or torch.inference_mode()')

    def _run(self, x, weight, bias, save):
        self._check(x, weight, bias)
        n, k, c = x.shape
        s = self.spec
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        need_stats = save or self.version == 'v0'
        mu = torch.empty(n, dtype=x.dtype, device=x.device) if need_stats and s.center_scalar else None
        moments = torch.empty((n, s.num_stats), dtype=x.dtype, device=x.device) if need_stats else None
        rstd = torch.empty_like(moments) if need_stats else None
        if n:
            split = isinstance(weight, tuple)
            w0 = weight[0] if split else weight
            wh = weight[1] if split else None
            ws0 = w0.stride(0) if split else 0
            wsl = wh.stride(0) if split else w0.stride(0) if w0 is not None else 0
            wsc = wh.stride(1) if split else w0.stride(1) if w0 is not None else 0
            bs = bias.stride(0) if bias is not None else 0
            affine = dict(HAS_WEIGHT=weight is not None, SPLIT=split, HAS_BIAS=bias is not None,
                          WS0=ws0, WSL=wsl, WSC=wsc, BS=bs)
            common = dict(C=c, G=s.num_stats, SN=x.stride(0), SK=x.stride(1),
                          CENTER=s.center_scalar)
            stats = dict(EPS=s.eps, ORDER=self.reduction_order, UNIFORM=self.uniform,
                         BK=self.block_k, BC=self.block_c)
            # Placeholders are never dereferenced when the corresponding flag is false.
            w0, wh, bp = w0 if w0 is not None else y, wh if wh is not None else y, bias if bias is not None else y
            with torch.cuda.device(x.device):
                if self.version == 'v0':
                    if s.center_scalar:
                        _scalar_mean[(n,)](x, mu, c, x.stride(0), self.block_c,
                                            num_warps=4, enable_fp_fusion=False)
                    _statistics[(n, s.num_stats)](x, mu if mu is not None else y,
                        moments, rstd, self.bounds, self.coefficients, **common, **stats,
                        num_warps=4, enable_fp_fusion=False)
                    _output[(triton.cdiv(n*k*c, 256),)](x, mu if mu is not None else y,
                        rstd, y, self.degrees, self.groups, w0, wh, bp, N=n, K=k,
                        **common, **affine, BLOCK=256, num_warps=4, enable_fp_fusion=False)
                else:
                    _fused[(n,)](x, y, mu if mu is not None else y,
                        moments if moments is not None else y, rstd if rstd is not None else y,
                        self.coefficients, self.degrees, w0, wh, bp, K=k,
                        **common, EPS=s.eps, ORDER=self.reduction_order, UNIFORM=self.uniform,
                        BK=self.atom_block_k, BC=self.block_c,
                        GROUP_BOUNDS=self.group_bounds, **affine, SAVE=save,
                        num_warps=4, enable_fp_fusion=False)
        return NormResult(y, mu, moments, rstd) if save else y

    def forward(self, x, *, weight=None, bias=None):
        return self._run(x, weight, bias, False)

    def forward_with_stats(self, x, *, weight=None, bias=None):
        return self._run(x, weight, bias, True)


class _SourceAdapter(torch.nn.Module):
    """Keep source parameter ownership/state keys; never call its forward.

    The registered source means .to() and parameter replacement stay live.
    The adapter's state_dict is namespaced under source.; source.state_dict()
    retains the original class keys. This is a standalone inference adapter.
    """
    def __init__(self, source, op):
        super().__init__()
        self.source, self.op = source, op
        self.spec = op.spec
        self.version = op.version

    def _params(self):
        if not self.source.affine:
            return None, None
        if hasattr(self.source, 'norm_l0'):
            return (self.source.norm_l0.weight, self.source.affine_weight), self.source.norm_l0.bias
        return self.source.affine_weight, getattr(self.source, 'affine_bias', None)

    def forward(self, x):
        weight, bias = self._params()
        return self.op(x, weight=weight, bias=bias)

    def forward_with_stats(self, x):
        weight, bias = self._params()
        return self.op.forward_with_stats(x, weight=weight, bias=bias)


def from_reference(source, *, version='v1', device=None):
    """Adapt verified source variants, preserving their grouping and FP32 weights.

    Construction may read tiny coefficient buffers to CPU; forward never does.
    No model-wide automatic replacement: target class choice remains explicit.
    """
    name = type(source).__name__
    if name in ('EquivariantLayerNorm', 'EquivariantLayerNormArray'):
        grouping, order, center = 'per_degree', 'components_first', True
    elif name in ('EquivariantSeparableLayerNorm', 'EquivariantLayerNormArraySphericalHarmonics'):
        grouping, order, center = 'scalar_high', 'components_first', True
        # V3 separable actually reduces channels before weighted components.
        if name == 'EquivariantSeparableLayerNorm':
            order = 'channels_first'
    elif name == 'EquivariantMergeLayerNorm':
        grouping, order, center = 'all', 'channels_first', source.centering
    else:
        raise NotImplementedError(f'no verified source adapter for {name}')
    weighting = ('degree_balanced' if source.normalization == 'component'
                 and getattr(source, 'std_balance_degrees', False) else source.normalization)
    spec = EquivariantNormSpec.from_preset(lmax=source.lmax, channels=source.num_channels,
        grouping=grouping, weighting=weighting, center_scalar=center, eps=source.eps)
    if weighting == 'degree_balanced':
        weights = [list(row) for row in spec.stats_weights]
        cached = source.balance_degree_weight.detach().reshape(-1).cpu()
        for l in range(0 if grouping == 'all' else 1, source.lmax+1):
            offset = l*l if grouping == 'all' else l*l-1
            weights[spec.output_group[l]][l] = float(cached[offset])
        spec = EquivariantNormSpec(spec.lmax, spec.channels, tuple(map(tuple, weights)),
                                   spec.output_group, spec.center_scalar, spec.eps)
    if device is None:
        tensors = list(source.parameters()) + list(source.buffers())
        device = tensors[0].device if tensors else torch.device('cuda')
    op = TritonEquivariantNorm(spec, version=version, reduction_order=order, device=device)
    return _SourceAdapter(source, op)
