"""Fused FP32 equivariant LayerNorm with first-order gradients.

One program per atom shares the feature tile across statistic groups,
normalization, and affine output.
"""
import torch
import triton
import triton.language as tl

from .fused_equivariant_layer_norm_backward import backward as _backward
from .fused_equivariant_layer_norm_common import (
    EquivariantNormSpec, NormResult, source_row_mean, source_group_stats,
    mean_factor, reduction_code, m_config,
)


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
def _fused(X, Y, MU, MOMENTS, RSTD, COEFFICIENT, DEGREE,
           WEIGHT0, WEIGHTH, BIAS, K: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
           SN: tl.constexpr, SK: tl.constexpr, CENTER: tl.constexpr,
           EPS: tl.constexpr, ORDER: tl.constexpr, UNIFORM: tl.constexpr,
           HAS_WEIGHT: tl.constexpr, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr,
           WS0: tl.constexpr, WSL: tl.constexpr, WSC: tl.constexpr, BS: tl.constexpr,
           SAVE: tl.constexpr, SAVE_MOMENTS: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
           GROUP_BOUNDS: tl.constexpr, SOURCE: tl.constexpr, CCFG: tl.constexpr,
           CF: tl.constexpr, MCONFIG: tl.constexpr, MF: tl.constexpr, COMPONENT: tl.constexpr,
           BALANCED: tl.constexpr, KCONFIG: tl.constexpr, ROW_CCONFIG: tl.constexpr,
           KF: tl.constexpr, ROW_CF: tl.constexpr, SINGLETON: tl.constexpr,
           NODES: tl.constexpr, KNC: tl.constexpr, INPUT_SHIFT: tl.constexpr):
    n = tl.program_id(0)
    k = tl.arange(0, BK)
    c = tl.arange(0, BC)
    valid = (k[:, None] < K) & (c[None, :] < C)
    x = tl.load(X + n * SN + k[:, None] * SK + c[None, :], valid, 0.)
    if CENTER:
        scalar = tl.sum(tl.where(k[:, None] == 0, x, 0.), 0)
        if SOURCE == 1 or SOURCE == 3:
            mu = source_row_mean(scalar, n*(SN//C), C, CCFG, CF, INPUT_SHIFT)
        else:
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
        if SOURCE != 0 and (SOURCE == 1 or SOURCE == 3 or g > 0):
            v, r = source_group_stats(z, w, n, GROUP_BOUNDS[g], GROUP_BOUNDS[g+1]-GROUP_BOUNDS[g], C,
                SOURCE, COMPONENT, BALANCED, ((MCONFIG >> (12*g)) & 4095), CCFG,
                ((KCONFIG >> (12*g)) & 4095), ((ROW_CCONFIG >> (12*g)) & 4095),
                MF[g], CF, KF[g], ROW_CF[g], EPS, SINGLETON, NODES, KNC)
        else:
            v = _moment(group_square, group_w, coefficient, C, ORDER, UNIFORM)
            r = tl.rsqrt(v + EPS)
        row_rstd = tl.where(member, r, row_rstd)
        if SAVE:
            if SAVE_MOMENTS:
                tl.store(MOMENTS + n * G + g, v)
            tl.store(RSTD + n * G + g, r)
    y = _affine(z, row_rstd[:, None], k[:, None], c[None, :], valid, DEGREE, WEIGHT0,
                WEIGHTH, BIAS, HAS_WEIGHT, SPLIT, HAS_BIAS, WS0, WSL, WSC, BS, K)
    tl.store(Y + n * K * C + k[:, None] * C + c[None, :], y, valid)
    if SAVE:
        if CENTER:
            tl.store(MU + n, mu)


class _EquivariantNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight0, weight_high, bias, op, save_stats):
        weight = (weight0, weight_high) if weight_high is not None else weight0
        result = op._run(x, weight, bias, save_stats, training=True)
        ctx.op = op
        ctx.save_for_backward(x, result.mean, result.rstd, weight0, weight_high)
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(*(value for value in result[1:] if value is not None))
        return tuple(result)

    @staticmethod
    def backward(ctx, dy, _dmean, _dmoments, _drstd):
        if torch.is_grad_enabled():
            raise RuntimeError('equivariant norm supports first-order gradients only; '
                               'create_graph=True and double backward are not supported')
        if dy is None:
            return (None,) * 6
        x, mean, rstd, weight0, weight_high = ctx.saved_tensors
        gradients = _backward(ctx.op, x, mean, rstd, weight0, weight_high,
                              dy, ctx.needs_input_grad[:4])
        return (*gradients, None, None)


class TritonEquivariantNorm(torch.nn.Module):
    """Execution plan; affine parameters belong to the caller.

    GPU scope: FP32 forward/backward, contiguous NKC / transposed KNC inputs,
    positive-stride packed or split weights. Each statistic must describe one
    contiguous, disjoint range of full degrees and scale exactly that range.
    Broader/overlapping specs remain valid math specs but are rejected here.
    """
    def __init__(self, spec, *, reduction_order='channels_first',
                 device='cuda', parameter_order='accurate'):
        super().__init__()
        if not isinstance(spec, EquivariantNormSpec):
            raise TypeError('spec must be EquivariantNormSpec')
        if reduction_order not in ('components_first', 'channels_first'):
            raise ValueError('unknown reduction order')
        bounds, coefficients, degrees = [0], [], []
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
        if bounds[-1] != spec.components:
            raise NotImplementedError('groups must cover all degrees')
        self.spec = spec
        if parameter_order not in ('accurate', 'broadcast', 'expanded'):
            raise ValueError('unknown parameter reduction order')
        self.parameter_order = parameter_order
        self.source_kind = 0
        self.source_component = True
        self.source_balanced = False
        self.reduction_order = reduction_order
        self.uniform = uniform
        self.group_bounds = tuple(bounds)
        self.atom_block_k = triton.next_power_of_2(spec.components)
        self.block_c = triton.next_power_of_2(spec.channels)
        if self.atom_block_k * self.block_c > 65536:
            raise NotImplementedError('execution tile exceeds 65536 padded elements')
        for name, values, dtype in [('coefficients', coefficients, torch.float32),
                                    ('degrees', degrees, torch.int32)]:
            self.register_buffer(name, torch.tensor(values, dtype=dtype, device=device),
                                 persistent=False)

    def _check(self, x, weight, bias):
        s = self.spec
        if x.ndim != 3 or tuple(x.shape[1:]) != (s.components, s.channels):
            raise ValueError('expected [N,(lmax+1)^2,channels]')
        if x.device.type != 'cuda' or x.dtype != torch.float32:
            raise TypeError('Triton backend supports CUDA FP32 only')
        if x.device != self.coefficients.device or self.coefficients.dtype != torch.float32:
            raise ValueError('move the execution plan to the input device; keep it FP32')
        if not (x.is_contiguous() or x.stride() == (s.channels, x.shape[0]*s.channels, 1)):
            raise ValueError('supported input storage: NKC and KNC')
        if x.numel() > 2**31 - 1:
            raise NotImplementedError('this backend uses 32-bit tensor indexing')
        def check(t, shape):
            if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape:
                raise ValueError(f'expected parameter shape {shape}')
            if t.device != x.device or t.dtype != x.dtype:
                raise ValueError('parameter device and dtype must match input')
            if any(a <= 0 for a in t.stride()):
                raise ValueError('parameter strides must be positive')
            if sum(max(a - 1, 0) * b for a, b in zip(t.shape, t.stride())) > 2**31 - 1:
                raise NotImplementedError('parameter offsets exceed 32-bit indexing')

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

    def _run(self, x, weight, bias, save, training=False):
        n, k, c = x.shape
        s = self.spec
        y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        need_stats = save or training
        mu = torch.empty(n, dtype=x.dtype, device=x.device) if need_stats and s.center_scalar else None
        moments = torch.empty((n, s.num_stats), dtype=x.dtype, device=x.device) if save else None
        rstd = torch.empty((n, s.num_stats), dtype=x.dtype, device=x.device) if need_stats else None
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
            native_kind = self.source_kind if torch.version.hip is None else 0
            input_shift = (x.data_ptr() // 4) % 4 if native_kind in (1, 3) else 0
            # Unused source schedules must not restrict the generic spec path
            # or a source which reduces its axes in a different order.
            # Merge's centering concatenates into NKC before square/mean;
            # without centering its square preserves the original layout.
            native = dict(SOURCE=native_kind, NODES=n,
                KNC=not x.is_contiguous() and (native_kind != 3 or not s.center_scalar),
                SINGLETON=(n*c == 1 if native_kind == 2 else n == 1),
                COMPONENT=self.source_component, BALANCED=self.source_balanced,
                CCFG=0, CF=1., MCONFIG=0, KCONFIG=0, ROW_CCONFIG=0,
                MF=(1.,)*s.num_stats, KF=(1.,)*s.num_stats, ROW_CF=(1.,)*s.num_stats)
            if native_kind:
                sizes = tuple(b-a for a,b in zip(self.group_bounds,self.group_bounds[1:]))
                native.update(CCFG=reduction_code(m_config(n,c,1)), CF=mean_factor(n,c))
                if native_kind in (1, 2):
                    native.update(
                        MCONFIG=sum(reduction_code(m_config(n,m,c)) << (12*g)
                                    for g,m in enumerate(sizes)),
                        MF=tuple(mean_factor(n*c,m) for m in sizes))
                else:
                    native.update(
                        KCONFIG=sum(reduction_code(m_config(n,m,1)) << (12*g)
                                    for g,m in enumerate(sizes)),
                        ROW_CCONFIG=sum(reduction_code(m_config(n*m,c,1)) << (12*g)
                                       for g,m in enumerate(sizes)),
                        KF=tuple(mean_factor(n,m) for m in sizes),
                        ROW_CF=tuple(mean_factor(n*m,c) for m in sizes))
            # Placeholders are never dereferenced when the corresponding flag is false.
            w0, wh, bp = w0 if w0 is not None else y, wh if wh is not None else y, bias if bias is not None else y
            with torch.cuda.device(x.device):
                _fused[(n,)](x, y, mu if mu is not None else y,
                    moments if moments is not None else y, rstd if rstd is not None else y,
                    self.coefficients, self.degrees, w0, wh, bp, K=k,
                    **common, EPS=s.eps, ORDER=self.reduction_order, UNIFORM=self.uniform,
                    BK=self.atom_block_k, BC=self.block_c,
                    GROUP_BOUNDS=self.group_bounds, **native, **affine, SAVE=need_stats, SAVE_MOMENTS=save,
                    INPUT_SHIFT=input_shift,
                    num_warps=4, enable_fp_fusion=False)
        return NormResult(y, mu, moments, rstd) if save or training else y

    def _dispatch(self, x, weight, bias, save):
        self._check(x, weight, bias)
        weight0, weight_high = weight if isinstance(weight, tuple) else (weight, None)
        tensors = (x, weight0, weight_high, bias)
        if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
            result = NormResult(*_EquivariantNormFunction.apply(*tensors, self, save))
            return result if save else result.output
        return self._run(x, weight, bias, save)

    def forward(self, x, *, weight=None, bias=None):
        return self._dispatch(x, weight, bias, False)

    def forward_with_stats(self, x, *, weight=None, bias=None):
        """Return differentiable output and detached diagnostic statistics."""
        return self._dispatch(x, weight, bias, True)


class _SourceAdapter(torch.nn.Module):
    """Keep source parameter ownership/state keys; never call its forward.

    The registered source means .to() and parameter replacement stay live.
    The adapter's state_dict is namespaced under source.; source.state_dict()
    retains the original class keys. Source parameters receive Triton gradients.
    """
    def __init__(self, source, op):
        super().__init__()
        self.source, self.op = source, op
        self.spec = op.spec

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


def from_reference(source, *, device=None):
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
    parameter_order = ('expanded' if name in ('EquivariantMergeLayerNorm', 'EquivariantSeparableLayerNorm')
                       else 'broadcast')
    op = TritonEquivariantNorm(spec, reduction_order=order, device=device,
                              parameter_order=parameter_order)
    op.source_kind = (1 if name in ("EquivariantLayerNorm", "EquivariantLayerNormArray")
                      else 2 if name == "EquivariantLayerNormArraySphericalHarmonics"
                      else 3 if name == "EquivariantMergeLayerNorm" else 4)
    op.source_component = source.normalization == "component"
    op.source_balanced = getattr(source, "std_balance_degrees", False)
    return _SourceAdapter(source, op)
