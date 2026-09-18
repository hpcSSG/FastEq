"""Validate and time resident node order and explicit pack/restore integration."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import statistics
import time

import torch
import triton
from fasteq.triton import graph_softmax as s
from fasteq.triton import graph_edge_layout as layout_module

p = argparse.ArgumentParser()
p.add_argument('--n', type=int, required=True)
p.add_argument('--topology', choices=['interleaved', 'random'], default='interleaved')
p.add_argument('--out', required=True)
args = p.parse_args()
native_path = Path(os.environ['FASTEQ_EQUIFORMER_V3_SOURCE_DIR'])/'softmax.py'
spec = importlib.util.spec_from_file_location('native_layout_check', native_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
native = m.GraphSoftmax(eps=1e-16, softcap=3., exp_dropout=0.)
torch.manual_seed(20260914)
n, h, e = args.n, 8, args.n*32
idx = (torch.arange(e, device='cuda', dtype=torch.int64) % n if args.topology == 'interleaved'
       else torch.randint(n, (e,), device='cuda'))
x = torch.randn(e, h, device='cuda', requires_grad=True)
r = torch.rand(e, 1, device='cuda', requires_grad=True)
g = torch.randn_like(x)
csr = s.prepare_graph_softmax(idx, num_nodes=n)
plan = layout_module.prepare_graph_softmax_layout(idx, num_nodes=n)
xs = plan.pack(x.detach()).requires_grad_()
rs = plan.pack_rescale(r.detach()).requires_grad_()
gs = plan.pack(g)


def roundtrip(plan):
    return plan.unpack(plan.softmax(plan.pack(x), exp_rescale=plan.pack_rescale(r), softcap=3.))


cases = {
    'torch': (lambda: native(x, idx, num_nodes=n, exp_rescale=r), (x, r), g),
    'unsorted_triton': (lambda: s.fused_graph_softmax(x, idx, num_nodes=n,
                        exp_rescale=r, csr=csr, softcap=3.), (x, r), g),
    'sorted_resident': (lambda: plan.softmax(xs, exp_rescale=rs, softcap=3.), (xs, rs), gs),
    'pack_restore_cached': (lambda: roundtrip(plan), (x, r), g),
}
result = {
    'host': socket.gethostname(), 'device': torch.cuda.get_device_name(),
    'torch': torch.__version__, 'triton': triton.__version__,
    'target': str(triton.runtime.driver.active.get_current_target()),
    'sources': {Path(mod.__file__).name: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                for mod in [s, layout_module]},
    'native_sha256': hashlib.sha256(native_path.read_bytes()).hexdigest(),
    'N': n, 'E': e, 'H': h, 'topology': args.topology, 'max_degree': csr.max_degree,
    'settings': {'dtype': 'FP32', 'rescale': '[E,1]', 'softcap': 3., 'eps': 1e-16,
                 'warmup': 5, 'repeat': 20, 'statistic': 'median GPU events',
                 'sorted_resident': 'upstream already keeps all features and gradients in node order; excludes feature permutation',
                 'pack_restore_cached': 'includes feature packing, result restoration and all inverse gradient permutations; cached integer topology only'},
    'checks': {}, 'timings': {},
}


def save():
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')


def metric(a, b, atol, rtol):
    a, b = a.detach().reshape(-1), b.detach().reshape(-1)
    bad, maximum, maxabs = 0, 0., 0.
    for start in range(0, a.numel(), 1048576):
        aa, bb = a[start:start+1048576], b[start:start+1048576]
        delta = (aa-bb).abs()
        q = delta/(atol+rtol*bb.abs())
        bad += int(((q > 1) | ~torch.isfinite(q)).sum())
        maximum = max(maximum, float(q.max()))
        maxabs = max(maxabs, float(delta.max()))
    return {'bad': bad, 'max_tolerance_ratio': maximum, 'max_abs': maxabs}


ref = cases['torch'][0]()
ref_grads = torch.autograd.grad(ref, (x, r), g)
for name in ['unsorted_triton', 'sorted_resident', 'pack_restore_cached']:
    fn, inputs, up = cases[name]
    y = fn()
    grads = torch.autograd.grad(y, inputs, up)
    if name == 'sorted_resident':
        y = plan.unpack(y.detach())
        grads = tuple(plan.unpack(v) for v in grads)
    rec = {'output': metric(y, ref, 3e-6, 3e-5),
           'dx': metric(grads[0], ref_grads[0], 3e-5, 3e-4),
           'dr': metric(grads[1], ref_grads[1], 3e-5, 3e-4)}
    result['checks'][name] = rec
    save()
    assert all(v['bad'] == 0 for v in rec.values()), (name, rec)
    del y, grads
del ref, ref_grads
gc.collect()
torch.cuda.empty_cache()


def invoke(name, mode):
    fn, inputs, up = cases[name]
    if mode == 'fwd':
        with torch.no_grad():
            return fn()
    return torch.autograd.grad(fn(), inputs, up)


for mode in ['fwd', 'fwd_bwd']:
    names = list(cases)
    samples = {k: [] for k in names}
    for name in names:
        for _ in range(5):
            value = invoke(name, mode)
            del value
        torch.cuda.synchronize()
    for rep in range(20):
        order = names[rep % len(names):] + names[:rep % len(names)]
        for name in order:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            value = invoke(name, mode)
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
            del value
    for name in names:
        result['timings'].setdefault(name, {})[mode] = {
            'gpu_ms': statistics.median(samples[name]),
            'min_ms': min(samples[name]), 'max_ms': max(samples[name]),
        }
    save()


def wall(fn):
    value = fn()
    del value
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
        del value
    return statistics.median(times)


result['prepare_layout_wall_ms'] = wall(lambda: layout_module.prepare_graph_softmax_layout(idx, num_nodes=n))
result['prepare_and_pack_restore_fwd_bwd_wall_ms'] = wall(lambda: torch.autograd.grad(
    roundtrip(layout_module.prepare_graph_softmax_layout(idx, num_nodes=n)), (x, r), g))
save()
print(json.dumps(result), flush=True)
