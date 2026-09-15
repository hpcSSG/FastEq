r"""Benchmark original Torch LayerNorm classes against the fused Triton adapter.

    PYTHONPATH=. FASTEQ_BACKEND=cpu python test/benchmark_triton_equivariant_layer_norm.py \
        --v3-source /path/to/equiformer_v3/layer_norm.py --operators norm merge \
        --mode fwd_bwd --nodes 108 4096 --output layernorm_benchmark.json

V2/V3 paths also accept FASTEQ_EQUIFORMER_{V2,V3}_LAYER_NORM. Each case checks
all outputs and requested gradients before timing. GPU events and synchronized
wall times measure eager standalone calls; fwd_bwd includes forward and backward.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import statistics
import time

import torch
import triton

from fasteq.triton.fused_equivariant_layer_norm import from_reference


OPERATORS = {
    "norm": ("V3", "EquivariantLayerNorm"),
    "merge": ("V3", "EquivariantMergeLayerNorm"),
    "separable": ("V3", "EquivariantSeparableLayerNorm"),
    "sh": ("V2", "EquivariantLayerNormArraySphericalHarmonics"),
}
BACKENDS = ("torch", "fused")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for family in ("v2", "v3"):
        parser.add_argument(f"--{family}-source", type=Path,
                            default=os.environ.get(f"FASTEQ_EQUIFORMER_{family.upper()}_LAYER_NORM"))
    parser.add_argument("--operators", nargs="+", choices=OPERATORS, default=list(OPERATORS))
    parser.add_argument("--mode", choices=("fwd", "fwd_bwd"), default="fwd_bwd")
    parser.add_argument("--nodes", nargs="+", type=int, default=[108, 4096])
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--layout", choices=("NKC", "KNC"), default="NKC")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.nodes) < 1 or args.channels < 1 or args.lmax < 0:
        parser.error("nodes/channels must be positive and lmax nonnegative")
    if args.warmup < 1 or args.samples < 1:
        parser.error("warmup and samples must be positive")
    for family in {OPERATORS[name][0] for name in args.operators}:
        path = getattr(args, family.lower() + "_source")
        if path is None or not path.is_file():
            parser.error(f"provide --{family.lower()}-source or FASTEQ_EQUIFORMER_{family}_LAYER_NORM")
    return args


def load_source(path, class_name):
    descriptor = importlib.util.spec_from_file_location("_layernorm_benchmark_source", path)
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    return getattr(module, class_name)


@torch.no_grad()
def compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("source and fused shapes/dtypes differ")
    a, b = actual.detach().reshape(-1), expected.detach().reshape(-1)
    maximum = torch.zeros((), dtype=torch.float64, device=a.device)
    ratio = maximum.clone()
    failures = torch.zeros((), dtype=torch.int64, device=a.device)
    for start in range(0, a.numel(), 2**22):
        av, bv = a[start:start + 2**22].double(), b[start:start + 2**22].double()
        error = (av - bv).abs()
        tolerance = 5e-5 + 5e-4 * bv.abs()
        failures += ((error > tolerance) | ~torch.isfinite(av) | ~torch.isfinite(bv)).sum()
        maximum = torch.maximum(maximum, error.max())
        ratio = torch.maximum(ratio, (error / tolerance).max())
    return {"numel": a.numel(), "max_abs": maximum.item(),
            "max_tolerance_ratio": ratio.item(), "failures": failures.item()}


def summary(values):
    return {"samples": values, "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def measure(functions, args):
    for name in BACKENDS:
        for _ in range(args.warmup):
            result = functions[name]()
            del result
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Materialize both CUDA events before collecting the first wall interval.
    start.record()
    end.record()
    end.synchronize()
    samples = {name: {"gpu_ms": [], "wall_ms": []} for name in BACKENDS}
    orders = []
    for index in range(args.samples):
        order = BACKENDS if index % 2 == 0 else BACKENDS[::-1]
        orders.append(order)
        for name in order:
            torch.cuda.synchronize()
            wall_start = time.perf_counter_ns()
            start.record()
            result = functions[name]()
            end.record()
            end.synchronize()
            samples[name]["wall_ms"].append((time.perf_counter_ns() - wall_start) / 1e6)
            samples[name]["gpu_ms"].append(start.elapsed_time(end))
            del result
    times = {name: {kind: summary(values) for kind, values in data.items()}
             for name, data in samples.items()}
    speedup = {kind: times["torch"][kind]["median"] / times["fused"][kind]["median"]
               for kind in ("gpu_ms", "wall_ms")}
    return {"timings": times, "torch_over_fused": speedup, "paired_sample_orders": orders}


def run_case(args, operator, nodes):
    family, class_name = OPERATORS[operator]
    source_path = getattr(args, family.lower() + "_source")
    torch.manual_seed(args.seed)
    source = load_source(source_path, class_name)(args.lmax, args.channels).cuda().train()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.randn_like(parameter))
    x = torch.randn(nodes, (args.lmax + 1)**2, args.channels, device="cuda")
    if args.layout == "KNC":
        x = x.transpose(0, 1).contiguous().transpose(0, 1)
    x.requires_grad_(args.mode == "fwd_bwd")
    dy = torch.randn_like(x) if args.mode == "fwd_bwd" else None
    # A split-affine lmax=0 module may declare an unused empty higher weight.
    named = {name: value for name, value in source.named_parameters() if value.numel()}
    parameters = tuple(named.values())
    adapter = from_reference(source)

    def invoke(op):
        if args.mode == "fwd":
            with torch.no_grad():
                return (op(x),)
        output = op(x)
        return (output, *torch.autograd.grad(output, (x, *parameters), dy))

    functions = {"torch": lambda: invoke(source), "fused": lambda: invoke(adapter)}
    expected, actual = functions["torch"](), functions["fused"]()
    names = ("forward",) if args.mode == "fwd" else ("forward", "dX", *named)
    checks = {name: compare(a, b) for name, a, b in zip(names, actual, expected)}
    del expected, actual
    failed = {name: result for name, result in checks.items() if result["failures"]}
    if failed:
        raise AssertionError(f"{operator} N={nodes} failed original-source precision: {failed}")
    row = {"operator": operator, "class_name": class_name, "mode": args.mode,
           "nodes": nodes, "shape": list(x.shape), "layout": args.layout,
           "input_stride": list(x.stride()), "source_path": str(source_path.resolve()),
           "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
           "full_tensor_comparison": True, "correctness": checks}
    row.update(measure(functions, args))
    return row


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    folder = Path(inspect.getfile(from_reference)).parent
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = {"metadata": {
        "utc": datetime.now(timezone.utc).isoformat(), "gpu": props.name,
        "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
        "dtype": "float32", "atol": 5e-5, "rtol": 5e-4,
        "timing_scope": "eager standalone calls; fwd_bwd includes forward and first-order gradients",
        "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in folder.glob("*norm*.py")},
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
    }, "cases": []}
    for operator in args.operators:
        for nodes in args.nodes:
            row = run_case(args, operator, nodes)
            report["cases"].append(row)
            timing = row["timings"]
            print(f"{operator} N={nodes} {args.mode}: "
                  f"Torch {timing['torch']['gpu_ms']['median']:.4f} ms, "
                  f"fused {timing['fused']['gpu_ms']['median']:.4f} ms, "
                  f"{row['torch_over_fused']['gpu_ms']:.3f}x", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
