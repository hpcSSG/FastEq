"""Matched FP32 accuracy and timing with reference tensors staged on the CPU."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import statistics

import torch
import triton


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric(actual, reference):
    flat, ref = actual.detach().reshape(-1), reference.reshape(-1)
    bad, max_ratio, max_abs = 0, 0.0, 0.0
    for start in range(0, flat.numel(), 1048576):
        a = flat[start:start + 1048576]
        b = ref[start:start + 1048576].to(a.device)
        error = (a - b).abs()
        ratio = error / (5e-5 + 5e-4 * b.abs())
        bad += int(((ratio > 1) | ~torch.isfinite(ratio)).sum())
        max_ratio = max(max_ratio, float(ratio.max()))
        max_abs = max(max_abs, float(error.max()))
    return dict(bad=bad, max_ratio=max_ratio, max_abs=max_abs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--module', type=Path, default=(
        Path(__file__).resolve().parents[1] / 'fasteq/triton/fused_attention_alpha.py'))
    parser.add_argument('--control-module', type=Path,
                        help='Optional independent control between Torch and the current operator.')
    args = parser.parse_args()
    if args.n < 1:
        parser.error('--n must be positive')

    current = load(args.module, 'alpha_current')
    control = load(args.control_module, 'alpha_control') if args.control_module else None
    torch.set_num_threads(8)
    torch.manual_seed(20260914)
    e, h, c = args.n * 32, 8, 32
    w = torch.randn(h, c, device='cuda', requires_grad=True)
    g = torch.randn(c, device='cuda', requires_grad=True)
    b = torch.randn(c, device='cuda', requires_grad=True)
    x = torch.randn(e, h, c, device='cuda', requires_grad=True)
    dy = torch.randn(e, h, device='cuda')
    inputs = (x, w, g, b)

    def native():
        z = torch.nn.functional.layer_norm(x, (c,), g, b, 1e-5)
        a = .6 * z + .4 * z * (2 * torch.sigmoid(z) - 1)
        return torch.einsum('ehc,hc->eh', a, w)

    providers = {'torch': native}
    if control is not None:
        providers['control'] = lambda: control.fused_attention_alpha(
            x, w, g, b, activation='smooth_leaky_relu')
    providers['triton'] = lambda: current.fused_attention_alpha(
        x, w, g, b, activation='smooth_leaky_relu')

    def train(fn):
        y = fn()
        return (y, *torch.autograd.grad(y, inputs, dy))

    result = {
        'host': socket.gethostname(), 'device': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'triton': triton.__version__,
        'N': args.n, 'E': e, 'H': h, 'C': c,
        'module_sha256': hashlib.sha256(args.module.read_bytes()).hexdigest(),
        'settings': {
            'dtype': 'float32', 'warmups': 5, 'samples': 20,
            'provider_order': 'rotated', 'providers': list(providers),
            'dropout': 0, 'activation': 'smooth_leaky_relu',
            'atol': 5e-5, 'rtol': 5e-4,
        },
        'checks': {}, 'timings': {},
    }
    if args.control_module:
        result['control_sha256'] = hashlib.sha256(args.control_module.read_bytes()).hexdigest()

    def save():
        Path(args.out).write_text(json.dumps(result, indent=2) + '\n')

    # Avoid holding two implementations' full GPU outputs/graphs at once.
    reference = tuple(t.detach().cpu() for t in train(native))
    gc.collect()
    torch.cuda.empty_cache()
    components = ('output', 'x', 'alpha_dot', 'norm_weight', 'norm_bias')
    for name, fn in providers.items():
        if name == 'torch':
            continue
        actual = train(fn)
        result['checks'][name] = {
            component: metric(a, r)
            for component, a, r in zip(components, actual, reference)
        }
        save()
        assert all(m['bad'] == 0 for m in result['checks'][name].values()), result['checks'][name]
        del actual
        gc.collect()
        torch.cuda.empty_cache()
    del reference
    gc.collect()

    names = list(providers)
    for mode in ('fwd', 'fwd_bwd'):
        def run(name):
            if mode == 'fwd':
                with torch.no_grad():
                    return providers[name]()
            return train(providers[name])

        for _ in range(5):
            for name in names:
                run(name)
        torch.cuda.synchronize()
        samples = {name: [] for name in names}
        for i in range(20):
            offset = i % len(names)
            for name in names[offset:] + names[:offset]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                run(name)
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
        result['timings'][mode] = {
            name: dict(median_ms=statistics.median(values), samples_ms=values)
            for name, values in samples.items()
        }
        save()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
