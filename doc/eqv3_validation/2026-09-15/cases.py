"""Original Torch references and unchanged FastEq operators for this run."""
import ast
import gc
import hashlib
import importlib
import importlib.util
from pathlib import Path
import time

import torch

REPO = Path('/public-data/zhouxibo/zxb/FastEq-fork-layernorm')
BASE = Path('/public-data/zhouxibo/zxb/equiformer_v3_baseline')
ORIGINAL = BASE / 'repo/experimental/models/equiformer_v3'
NORMS = Path('/public-data/zhouxibo/zxb/fasteq_unified_layernorm_20260915/reference')
OPS = ('softmax', 'alpha', 'gate', 'norm', 'separable', 'dropout')


def load_file(path, name):
    desc = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(desc)
    desc.loader.exec_module(module)
    return module


def source_class(path, name):
    """Load an unchanged class body without unrelated model imports."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = {'torch': torch, '__name__': 'original_eqv3_class'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def sample(t, count=32):
    flat = t.detach().reshape(-1)
    count = min(count, flat.numel())
    if not count:
        return []
    ix = torch.arange(count, device=flat.device, dtype=torch.int64)
    ix = ix * (flat.numel() - 1) // max(count - 1, 1)
    return flat[ix].cpu().tolist()


class ReplayMask(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return self.mask


class Case:
    def __init__(self, op, n, mode, backend='both', channels=128, lmax=3,
                 layout='NKC', heads=8, degree=32, alpha_channels=32):
        torch.manual_seed(20260914)
        self.op, self.n, self.mode = op, n, mode
        self.inputs, self.names = [], []
        self.meta = dict(operator=op, atoms=n, lmax=lmax, channels=channels,
                         dtype='float32', layout=layout, seed=20260914)
        self.setup = {}
        train = mode == 'fwd_bwd'
        k = (lmax + 1)**2

        def input_tensor(shape, name):
            x = torch.randn(shape, device='cuda')
            if layout == 'KNC' and len(shape) == 3:
                x = x.transpose(0, 1).contiguous().transpose(0, 1)
            elif layout == 'KNC' and len(shape) == 2:
                x = x.t().contiguous().t()
            x.requires_grad_(train)
            self.inputs.append(x)
            self.names.append(name)
            return x

        def parameter(p, name):
            with torch.no_grad():
                p.copy_(torch.randn_like(p))
            self.inputs.append(p)
            self.names.append(name)

        if op in ('norm', 'separable'):
            from fasteq.triton import from_reference
            name = {'norm': 'EquivariantLayerNorm',
                    'separable': 'EquivariantSeparableLayerNorm'}[op]
            source = load_file(NORMS/'layer_norm.py', 'original_norm')
            self.source = getattr(source, name)(lmax, channels).cuda().train()
            for name, p in self.source.named_parameters():
                parameter(p, name)
            x = input_tensor((n, k, channels), 'x')
            self.reference = lambda: self.source(x)
            if backend != 'torch':
                self.fused = from_reference(self.source)
                self.fast = lambda: self.fused(x)
            self.meta.update(reference=str(NORMS/'layer_norm.py'),
                             parameter_reduction='native', input_shape=list(x.shape))
        elif op == 'alpha':
            from fasteq.triton.fused_attention_alpha import fused_atten_alpha
            c = alpha_channels
            self.norm = torch.nn.LayerNorm(c).cuda()
            self.act = source_class(ORIGINAL/'activation.py', 'SmoothLeakyReLU')().cuda()
            self.drop = torch.nn.Identity().cuda()
            w = torch.nn.Parameter(torch.empty(heads, c, device='cuda'))
            parameter(w, 'alpha_dot')
            parameter(self.norm.weight, 'norm_weight')
            parameter(self.norm.bias, 'norm_bias')
            x = input_tensor((n*degree, heads, c), 'x')
            # Exact four-operation block in EquivariantGraphAttention.forward.
            self.reference = lambda: torch.einsum('bik, ik -> bi',
                self.drop(self.act(self.norm(x))), w)
            self.fast = lambda: fused_atten_alpha(x, w, self.norm, self.act, self.drop)
            self.meta.update(reference=str(ORIGINAL/'transformer_block.py'),
                reference_lines='312-316', input_shape=list(x.shape), edges=n*degree,
                heads=heads, alpha_channels=c, activation='SmoothLeakyReLU', dropout_p=0.)
        elif op == 'softmax':
            from fasteq.triton.graph_softmax import FusedGraphSoftmax, prepare_graph_softmax
            source = load_file(ORIGINAL/'softmax.py', 'original_softmax')
            self.source = source.GraphSoftmax(softcap=3., eps=1e-16).cuda().train()
            # Balanced, unsorted destination list: every atom has degree neighbors.
            index = torch.arange(n*degree, device='cuda', dtype=torch.int64) % n
            x = input_tensor((n*degree, heads), 'x')
            rescale = torch.rand(n*degree, 1, device='cuda').requires_grad_(train)
            self.inputs.append(rescale)
            self.names.append('exp_rescale')
            self.reference = lambda: self.source(x, index, num_nodes=n, exp_rescale=rescale)
            if backend != 'torch':
                self.fused = FusedGraphSoftmax(softcap=3., eps=1e-16).cuda().train()
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); end.record(); end.synchronize()
                wall = time.perf_counter_ns()
                start.record()
                self.csr = prepare_graph_softmax(index=index, num_nodes=n)
                end.record(); end.synchronize()
                self.setup = dict(csr_gpu_ms=start.elapsed_time(end),
                                 csr_wall_ms=(time.perf_counter_ns()-wall)/1e6)
                self.fast = lambda: self.fused(x, index, num_nodes=n,
                                               exp_rescale=rescale, csr=self.csr)
                self.cold = lambda: self.fused(x, index, num_nodes=n, exp_rescale=rescale)
            self.meta.update(reference=str(ORIGINAL/'softmax.py'), input_shape=list(x.shape),
                edges=n*degree, degree=degree, heads=heads, softcap=3., exp_dropout=0.,
                graph='unsorted repeated destination sequence; fixed degree',
                csr_timing='prepared reused CSR; preprocessing recorded separately')
        elif op == 'gate':
            # Installed e3nn's immutable constants contain Python slice objects.
            # Keep weights_only loading and allow only this builtin data type.
            with torch.serialization.safe_globals([slice]):
                from e3nn.nn import Gate
                from fasteq.triton.fused_equivariant_gate import FastEquivariantGate
            args = (f'{channels}x0e', [torch.nn.functional.silu],
                    f'{lmax*channels}x0e', [torch.sigmoid],
                    ' + '.join(f'{channels}x{l}{"e" if l%2==0 else "o"}' for l in range(1,lmax+1)))
            self.source = Gate(*args).cuda()
            x = input_tensor((n, self.source.irreps_in.dim), 'x')
            self.reference = lambda: self.source(x)
            if backend != 'torch':
                self.fused = FastEquivariantGate(*args).cuda()
                self.fast = lambda: self.fused(x)
            import inspect
            self.meta.update(reference=inspect.getfile(Gate), input_shape=list(x.shape),
                irreps_in=str(self.source.irreps_in), irreps_out=str(self.source.irreps_out),
                scope='catalog FastEquivariantGate vs e3nn.nn.Gate; not EQv3 GateActivation')
        elif op == 'dropout':
            from fasteq.triton.fused_equivariant_dropout import EquivariantDropout
            source = load_file(ORIGINAL/'drop.py', 'original_drop')
            self.source = source.EquivariantDropout(lmax, lmax, .3).cuda().train()
            x = input_tensor((n,k,channels), 'x')
            self.reference = lambda: self.source(x)
            if backend != 'torch':
                self.fused = EquivariantDropout(lmax,lmax,.3).cuda().train()
                self.fast = lambda: self.fused(x)
                self.seed = torch.tensor(12345,device='cuda',dtype=torch.int64)
                self.fixed_fast = lambda: self.fused(x,seed=self.seed)
            self.meta.update(reference=str(ORIGINAL/'drop.py'), input_shape=list(x.shape),
                dropout_p=.3, training=True, fwd_scope='active dropout under no_grad; eval is identity')
        else:
            raise ValueError(op)
        self.meta['reference_sha256'] = hashlib.sha256(Path(self.meta['reference']).read_bytes()).hexdigest()
        output_shape = (n*degree,heads) if op in ('alpha','softmax') else ((n,k,channels) if op!='gate' else (n,self.source.irreps_out.dim))
        self.dy = torch.randn(output_shape,device='cuda') if train else None

    def invoke(self, backend, *, fixed=False, cold=False):
        function = self.reference if backend=='torch' else (self.cold if cold else
                    self.fixed_fast if fixed and self.op=='dropout' else self.fast)
        if self.mode == 'fwd':
            with torch.no_grad():
                return (function(),)
        y=function()
        grad=torch.autograd.grad(y,self.inputs,self.dy)
        return (y,*grad)

    def install_replayed_dropout(self, y):
        if self.op != 'dropout':
            return
        # Generate the compact mask directly: inferring it from y != 0 would
        # incorrectly mark a kept, exactly-zero input as dropped at large N.
        from fasteq.triton.fused_equivariant_dropout import fused_equivariant_dropout
        levels=self.meta['lmax']+1
        index=torch.arange(levels,device=y.device,dtype=torch.int64)
        ones=y.new_ones(()).expand(self.n,levels,self.meta['channels'])
        with torch.no_grad():
            mask=fused_equivariant_dropout(ones,index,levels-1,
                self.meta['dropout_p'],True,seed=self.seed)
        self.source.drop=ReplayMask(mask)


@torch.no_grad()
def compare_cpu(actual, expected, atol=5e-5, rtol=5e-4):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    a=actual.reshape(-1)
    # References may be views; transfer to CPU before flattening to bound GPU memory.
    b=expected.detach().cpu().reshape(-1)
    out=dict(numel=a.numel(),shape=list(actual.shape),atol=atol,rtol=rtol,
             max_abs=0.,max_tolerance_ratio=0.,failures=0.,squared_error=0.,squared_reference=0.,examples=[])
    for start in range(0,a.numel(),2**20):
        av=a[start:start+2**20].double(); bv=b[start:start+2**20].double()
        error=(av-bv).abs(); tol=atol+rtol*bv.abs()
        bad=(error>tol)|~torch.isfinite(av)|~torch.isfinite(bv)
        out['max_abs']=max(out['max_abs'],error.max().item())
        out['max_tolerance_ratio']=max(out['max_tolerance_ratio'],(error/tol.clamp_min(1e-300)).max().item())
        out['failures']+=int(bad.sum())
        out['squared_error']+=error.square().sum().item()
        out['squared_reference']+=bv.square().sum().item()
        if len(out['examples'])<4 and bad.any():
            for i in bad.nonzero().flatten()[:4-len(out['examples'])].tolist():
                out['examples'].append(dict(flat_index=start+i,actual=av[i].item(),expected=bv[i].item()))
    out['relative_l2']=(out.pop('squared_error')/max(out.pop('squared_reference'),1e-300))**.5
    out['actual_sample']=sample(actual)
    out['expected_sample']=sample(b)
    return out


def check(case):
    actual=case.invoke('fused',fixed=True)
    case.install_replayed_dropout(actual[0])
    saved=[v.detach().cpu() for v in actual]
    del actual
    gc.collect(); torch.cuda.empty_cache()
    expected=case.invoke('torch')
    names=('output',*case.names) if case.mode=='fwd_bwd' else ('output',)
    checks={}
    for name,a,b in zip(names,saved,expected):
        if case.op=='softmax':
            atol,rtol=(3e-6,3e-5) if name=='output' else (3e-5,3e-4)
        elif case.op=='dropout':
            atol,rtol=0.,0.
        else:
            atol,rtol=5e-5,5e-4
        checks[name]=compare_cpu(a,b,atol,rtol)
    return dict(status='PASS' if not any(c['failures'] for c in checks.values()) else 'FAIL',
                full_tensor_comparison=True,checks=checks,**case.meta)
