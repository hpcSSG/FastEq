"""Standalone FP32 inference benchmark for shared equivariant LayerNorm v0/v1.

From the repository root (CUDA PyTorch and Triton required):
    PYTHONPATH=. FASTEQ_BACKEND=cpu python test/benchmark_triton_equivariant_layer_norm.py \
        --output equivariant_layer_norm_benchmark.json

FASTEQ_BACKEND=cpu bypasses FastEq's optional native extension; these operators
still run on the GPU. The mathematical reference checks correctness only. It is
not timed or presented as original model source performance. CUDA Graph times
are repeated-call, warm-cache GPU times, not model latency or DRAM counters.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time

import torch
import triton

from fasteq.triton.fused_equivariant_layer_norm import (
    EquivariantNormSpec, TritonEquivariantNorm, reference_forward,
)

VERSIONS = ("v0", "v1")
# Presets match the statistics and reduction order of EqLN, Merge, and SH.
PRESETS = {
    "per_degree": ("component", "components_first"),
    "all": ("degree_balanced", "channels_first"),
    "scalar_high": ("degree_balanced", "components_first"),
}


def summary(samples):
    return {"samples": samples, "median": statistics.median(samples),
            "min": min(samples), "max": max(samples),
            "std": statistics.stdev(samples)}


def capture(fn, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            output = fn()
    return graph, output


def measure(functions, expected, args, case_index):
    graphs = {name: capture(fn, args.graph_repeats) for name, fn in functions.items()}
    for graph, _ in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    starts = {name: torch.cuda.Event(enable_timing=True) for name in VERSIONS}
    stops = {name: torch.cuda.Event(enable_timing=True) for name in VERSIONS}
    modes = {"graph_gpu_us": {name: [] for name in VERSIONS}}
    if args.eager:
        modes["eager_wall_us"] = {name: [] for name in VERSIONS}
    orders = []
    for sample in range(args.samples):
        order = VERSIONS if (sample + case_index) % 2 == 0 else VERSIONS[::-1]
        orders.append(order)
        for name in order:
            starts[name].record()
            graphs[name][0].replay()
            stops[name].record()
            stops[name].synchronize()
            elapsed = starts[name].elapsed_time(stops[name]) * 1000 / args.graph_repeats
            modes["graph_gpu_us"][name].append(elapsed)
    # Validate actual replay outputs outside all timed intervals.
    for _, output in graphs.values():
        torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)
    if args.eager:
        for order in orders:
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                output = functions[name]()
                torch.cuda.synchronize()
                modes["eager_wall_us"][name].append((time.perf_counter_ns() - start) / 1000)
                del output
    results = {}
    for mode, values in modes.items():
        results[mode] = {name: summary(samples) for name, samples in values.items()}
        ratios = [a / b for a, b in zip(values["v0"], values["v1"])]
        results[mode]["v0_over_v1"] = summary(ratios)
        results[mode]["v0_over_v1"]["ratio_of_medians"] = (
            statistics.median(values["v0"]) / statistics.median(values["v1"]))
    return {"timings": results, "paired_sample_orders": orders}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", nargs="+", type=int, default=[108, 3456])
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--graph-repeats", type=int, default=32)
    parser.add_argument("--eager", action="store_true",
                        help="Also time dispatch/allocation and GPU completion using wall time")
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if min(args.nodes) < 1 or args.channels < 1 or args.lmax < 0:
        parser.error("nodes/channels must be positive and lmax nonnegative")
    if args.samples < 3 or args.graph_repeats < 1:
        parser.error("samples must be at least 3 and graph-repeats positive")
    return args


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    source_paths = [Path(__file__).resolve(), Path(inspect.getfile(TritonEquivariantNorm)),
                    Path(inspect.getfile(EquivariantNormSpec))]
    report = {
        "metadata": {
            "utc": datetime.now(timezone.utc).isoformat(), "gpu": props.name,
            "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
            "compute_capability": [props.major, props.minor],
            "gpu_total_memory_bytes": props.total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__, "triton": triton.__version__,
            "cuda": torch.version.cuda, "dtype": "float32", "tf32": False,
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in source_paths},
            "arguments": {k: str(v) if isinstance(v, Path) else v
                          for k, v in vars(args).items()},
            "reference_role": "mathematical correctness oracle only; not timed",
            "timing_scope": "standalone inference; warm-cache CUDA Graph; no model or HBM counters",
        }, "cases": [],
    }
    for grouping, (weighting, order) in PRESETS.items():
        spec = EquivariantNormSpec.from_preset(lmax=args.lmax, channels=args.channels,
                                              grouping=grouping, weighting=weighting)
        modules = {v: TritonEquivariantNorm(spec, version=v, reduction_order=order)
                   for v in VERSIONS}
        weight = torch.randn(args.lmax + 1, args.channels, device="cuda")
        bias = torch.randn(args.channels, device="cuda")
        for nodes in args.nodes:
            contiguous = torch.randn(nodes, spec.components, args.channels, device="cuda")
            for layout in ("NKC", "KNC"):
                x = contiguous if layout == "NKC" else contiguous.transpose(0, 1).contiguous().transpose(0, 1)
                expected = reference_forward(x, spec, weight=weight, bias=bias).output
                functions = {v: (lambda op=op: op(x, weight=weight, bias=bias))
                             for v, op in modules.items()}
                errors = {}
                for name, fn in functions.items():
                    actual = fn()
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                    if not actual.is_contiguous():
                        raise AssertionError("Expected contiguous NKC output")
                    errors[name] = {"max_abs_error": (actual - expected).abs().max().item(),
                                    "output_sum_float64": actual.double().sum().item()}
                case = {"grouping": grouping, "weighting": weighting, "reduction_order": order,
                        "nodes": nodes, "lmax": args.lmax, "channels": args.channels,
                        "layout": layout, "input_stride": list(x.stride()), "correctness": errors}
                case.update(measure(functions, expected, args, len(report["cases"])))
                report["cases"].append(case)
                medians = case["timings"]["graph_gpu_us"]
                print(f"{grouping:12} N={nodes:<6} {layout}: "
                      f"v0={medians['v0']['median']:.3f} us, v1={medians['v1']['median']:.3f} us", flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
