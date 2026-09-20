"""Compare raw-index fusion with native Torch and CSR rebuilt on every call."""
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
from fasteq.triton import graph_softmax as csr_module
from fasteq.triton import graph_softmax_index as index_module
from fasteq.triton import graph_edge_layout as layout_module


def metric(a, b, atol, rtol):
    a, b = a.detach().reshape(-1), b.reshape(-1)
    bad, maximum, maxabs = 0, 0., 0.
    for start in range(0, a.numel(), 1048576):
        aa, bb = a[start:start + 1048576], b[start:start + 1048576].to(a.device)
        error = (aa - bb).abs()
        ratio = error / (atol + rtol * bb.abs())
        bad += int(((ratio > 1) | ~torch.isfinite(ratio)).sum())
        maximum = max(maximum, float(ratio.max()))
        maxabs = max(maxabs, float(error.max()))
    return dict(bad=bad, max_tolerance_ratio=maximum, max_abs=maxabs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--topology', choices=['interleaved', 'random', 'contiguous'], default='random')
    parser.add_argument('--out', required=True)
    parser.add_argument('--repeat', type=int, default=20)
    args = parser.parse_args()
    native_path = Path(os.environ['FASTEQ_EQUIFORMER_V3_SOURCE_DIR']) / 'softmax.py'
    spec = importlib.util.spec_from_file_location('native_index_benchmark', native_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    native = reference.GraphSoftmax(eps=1e-16, softcap=3., exp_dropout=0.)
    torch.set_num_threads(8)
    torch.manual_seed(20260914)
    n, h, e = args.n, 8, args.n * 32
    if args.topology == 'random':
        index = torch.randint(n, (e,), device='cuda')
    else:
        index = torch.arange(e, device='cuda')
        index = index % n if args.topology == 'interleaved' else index // 32
    x = torch.randn(e, h, device='cuda', requires_grad=True)
    r = torch.rand(e, 1, device='cuda', requires_grad=True)
    g = torch.randn_like(x)

    def packed_rebuild():
        layout = layout_module.prepare_graph_softmax_layout(index, num_nodes=n)
        return layout.unpack(layout.softmax(layout.pack(x),
            exp_rescale=layout.pack_rescale(r), softcap=3.))

    providers = {
        'torch': lambda: native(x, index, num_nodes=n, exp_rescale=r),
        'index_fused': lambda: index_module.fused_graph_softmax_index(
            x, index, num_nodes=n, exp_rescale=r, softcap=3.),
        'csr_rebuild': lambda: csr_module.fused_graph_softmax(
            x, index, num_nodes=n, exp_rescale=r, softcap=3.),
        'packed_rebuild': packed_rebuild,
    }

    def train(fn):
        y = fn()
        return y, *torch.autograd.grad(y, (x, r), g)

    result = {
        'host': socket.gethostname(), 'device': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'triton': triton.__version__,
        'target': str(triton.runtime.driver.active.get_current_target()),
        'N': n, 'E': e, 'H': h, 'topology': args.topology,
        'sources': {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                    for m in (csr_module, index_module, layout_module)},
        'native_sha256': hashlib.sha256(native_path.read_bytes()).hexdigest(),
        'settings': {'dtype': 'FP32', 'rescale': '[E,1]', 'softcap': 3., 'eps': 1e-16,
                     'warmup': 5, 'repeat': args.repeat, 'order': 'rotated',
                     'topology_reuse': False, 'statistic': 'median synchronized wall and GPU event ms'},
        'checks': {}, 'timings': {},
    }

    def save():
        Path(args.out).write_text(json.dumps(result, indent=2) + '\n')

    ref = tuple(t.detach().cpu() for t in train(providers['torch']))
    gc.collect()
    torch.cuda.empty_cache()
    for name, fn in providers.items():
        if name == 'torch':
            continue
        values = train(fn)
        checks = {component: metric(a, b, atol, rtol)
                  for component, a, b, atol, rtol in zip(
                      ('output', 'dx', 'dr'), values, ref, (3e-6, 3e-5, 3e-5), (3e-5, 3e-4, 3e-4))}
        result['checks'][name] = checks
        save()
        assert all(v['bad'] == 0 for v in checks.values()), (name, checks)
        del values
        gc.collect()
        torch.cuda.empty_cache()
    del ref
    names = list(providers)
    for mode in ('fwd', 'fwd_bwd'):
        def invoke(name):
            if mode == 'fwd':
                with torch.no_grad():
                    return providers[name]()
            return train(providers[name])
        for name in names:
            for _ in range(5):
                values = invoke(name)
                del values
        torch.cuda.synchronize()
        samples = {name: dict(wall_ms=[], gpu_ms=[]) for name in names}
        for i in range(args.repeat):
            offset = i % len(names)
            for name in names[offset:] + names[:offset]:
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                values = invoke(name)
                end.record()
                end.synchronize()
                samples[name]['wall_ms'].append((time.perf_counter() - wall_start) * 1000)
                samples[name]['gpu_ms'].append(start.elapsed_time(end))
                del values
        result['timings'][mode] = {
            name: {key: {'median': statistics.median(s), 'samples': s} for key, s in v.items()}
            for name, v in samples.items()
        }
        save()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
