#!/usr/bin/env python3
"""cuEq uniform_1d GPU benchmarks for CWTP and native STC.
Requires numpy, CUDA-enabled torch, cuequivariance, cuequivariance-torch,
and its matching CUDA ops package (or a compatible local uniform_1d build).
Examples:
  python benchmark_cueq_uniform1d.py
  python benchmark_cueq_uniform1d.py --operators cwtp
  python benchmark_cueq_uniform1d.py --phases backward --dtype float64
CWTP uses the supplied MACE-MP medium and MACE-OFF large/medium/small
TensorProduct configurations (128/224/128/96 channels), with SH lmax=3.
All models and layers 0/1 are selected by default. MACE-OFF small has only
one supplied CWTP configuration (layer 0); its CWTP layer 1 is skipped.
CWTP includes per-edge path weights. Index modes exercise node-to-edge gather
and edge-to-node scatter within SegmentedPolynomial; radial MLP is excluded.
Default index modes: none and both; select input/output for isolated indexing.
Example matching the supplied shapes:
  python benchmark_cueq_uniform1d.py --models mace_mp_medium --operators cwtp --layers 1 --atoms 2208 --edges 79020 --index-modes both --check-indexing
This uses x=(2208,4,128), y=(79020,16,1), w=(79020,10,128),
and out=(2208,40,128). Indices are independent random int32 tensors.
--check-indexing compares outputs and all floating-input gradients against
explicit GPU gather + unindexed operator + scatter outside timing.
STC follows each model's channel width and hidden L using the existing
reduced symmetric basis recipe, species selection and correlation order 3.
These STC configurations are inferred, not verified against model logs.
Inputs are synthetic; no trained checkpoint or model inference is measured.
Default atoms: [256, 1024, 4096, 8192, 12288]. CWTP rows=atoms*neighbors.
The method is fixed to uniform_1d and checked after module construction.
OOM during construction, allocation, compile, warmup or timing is recorded as
status=oom and skipped. Other errors are raised.
Timing is synchronized host-observed latency (dispatch + execution + sync).
Compilation/init/transfers are excluded. Backward measures only gradients
of all floating inputs/weights: each fresh forward graph is prepared and
synchronized BEFORE the timer starts. forward_backward aliases backward.
The operator callable runs in PyTorch eager mode.
GPU is required. Optional indexing reference checks also run on GPU.
"""
from __future__ import annotations
import argparse
import csv
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
# CWTP shapes transcribed from the supplied TensorProduct logs.
# STC uses each model's channel width / hidden L with the existing STC recipe;
# these STC shapes are not verified against model logs or checkpoints.
PRESETS = {
    "mace_mp_medium": (128, 1),
    "mace_off_large": (224, 2),
    "mace_off_medium": (128, 1),
    "mace_off_small": (96, 0),
}
CWTP_SH_LMAX = 3
CWTP_LAYERS = {
    "mace_mp_medium": (0, 1),
    "mace_off_large": (0, 1),
    "mace_off_medium": (0, 1),
    "mace_off_small": (0,),
}
# Separate output blocks for each path, ordered exactly as in the logs.
CWTP_OUTPUT_BLOCKS = {
    0: (0, 1, 2, 3),
    1: (0, 0, 1, 1, 1, 2, 2, 2, 3, 3),
    2: (0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3),
}
def irrep_string(lmax, channels=1):
    return "+".join(f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1))
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=list(PRESETS), default=list(PRESETS))
    p.add_argument("--operators", nargs="+", choices=["cwtp", "stc"], default=["cwtp", "stc"])
    p.add_argument("--layers", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    p.add_argument("--atoms", nargs="+", type=int, default=[256, 1024, 4096, 8192, 12288])
    p.add_argument("--edges", type=int, default=None,
                   help="Exact CWTP edge count; overrides atoms * neighbors")
    p.add_argument("--index-modes", nargs="+", choices=["none", "input", "output", "both"],
                   default=["none", "both"], help="CWTP indexing modes; STC is unchanged")
    p.add_argument("--check-indexing", action="store_true",
                   help="Check indexed CWTP forward/gradients against explicit GPU gather/scatter")
    p.add_argument("--neighbors", type=int, default=32, help="CWTP rows = atoms * neighbors")
    p.add_argument("--lmax", type=int, default=3, help="STC input lmax; CWTP is fixed at SH lmax=3")
    p.add_argument("--correlation", type=int, default=3)
    p.add_argument("--species", type=int, default=10)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--phases", nargs="+", choices=["forward", "backward", "forward_backward"], default=["forward", "backward"],
                   help="forward or backward-only; forward_backward is a legacy alias for backward")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", default="cueq_uniform1d.csv")
    a = p.parse_args()
    a.index_modes = list(dict.fromkeys(a.index_modes))
    a.phases = list(dict.fromkeys("backward" if phase == "forward_backward" else phase for phase in a.phases))
    if min(a.atoms) < 1 or min(a.neighbors, a.species, a.iterations) < 1 or a.warmup < 0:
        p.error("sizes and iterations must be positive; warmup must be nonnegative")
    if a.edges is not None and a.edges < 1:
        p.error("edges must be positive")
    if a.lmax < 0 or not 1 <= a.correlation <= 3:
        p.error("lmax must be nonnegative, and correlation must be in [1,3]")
    if "stc" in a.operators and a.lmax < max(PRESETS[m][1] for m in a.models):
        p.error("STC lmax must cover the selected models' hidden L")
    return a
def imports():
    global np, cue, cuet, torch
    import numpy as np
    import cuequivariance as cue
    import torch
    import cuequivariance_torch as cuet
    if not torch.cuda.is_available():
        raise RuntimeError("cuEq requires PyTorch with CUDA GPU support.")
    torch.cuda.set_device(0)
def cue_module(poly, a, op_name):
    module = cuet.SegmentedPolynomial(poly, op_name=op_name, method="uniform_1d", math_dtype=getattr(torch, a.dtype)).to("cuda:0")
    if module.method != "uniform_1d":
        raise RuntimeError(f"cuEq switched to {module.method}; install its CUDA ops package.")
    return module
def gpu_inputs(arrays):
    # Transfer inputs before timing.
    return tuple(torch.as_tensor(np.array(x, copy=True), device="cuda:0") for x in arrays)
def make_cwtp(a, channels, hidden_l, layer, index_mode="none"):
    in_l = 0 if layer == 0 else hidden_l
    base1, base2 = irrep_string(in_l), irrep_string(CWTP_SH_LMAX)
    filters = [cue.O3(l, (-1)**l) for l in range(CWTP_SH_LMAX + 1)]
    small = cue.descriptors.channelwise_tensor_product(cue.Irreps("O3", base1), cue.Irreps("O3", base2), filters)
    shape = (small.outputs[0].dim, small.inputs[1].dim, small.inputs[2].dim)
    nw, dout = small.inputs[0].dim, shape[0]
    full = cue.descriptors.channelwise_tensor_product(cue.Irreps("O3", irrep_string(in_l, channels)), cue.Irreps("O3", base2), filters)
    expected_out = "+".join(f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}"
                            for l in CWTP_OUTPUT_BLOCKS[in_l])
    actual_out = str(full.outputs[0].irreps).replace(" ", "")
    if actual_out != expected_out:
        raise ValueError(f"CWTP output irreps differ from supplied log: {actual_out} != {expected_out}")
    module = cue_module(full.polynomial, a, "cwtp")
    gather = index_mode in ("input", "both")
    scatter = index_mode in ("output", "both")

    def fn(x, y, w, input_index=None, output_index=None, output_shape=None):
        # Inputs are [w, x, y]: x is input operand 1, output is operand 0.
        buffers = [w.reshape(w.shape[0], -1), x.reshape(x.shape[0], -1),
                   y.reshape(y.shape[0], -1)]
        kwargs = {}
        if gather:
            kwargs["input_indices"] = {1: input_index}
        if scatter:
            # Only shape[0] is read. No output allocation or GPU scalar read needed.
            kwargs["output_shapes"] = {0: output_shape}
            kwargs["output_indices"] = {0: output_index}
        out = module(buffers, **kwargs)[0]
        return out.reshape(out.shape[0], dout, channels)

    def reference(x, y, w, input_index=None, output_index=None, output_shape=None):
        edge_x = x.index_select(0, input_index.long()) if gather else x
        edge_out = module([w.reshape(w.shape[0], -1),
                           edge_x.reshape(edge_x.shape[0], -1),
                           y.reshape(y.shape[0], -1)])[0]
        if scatter:
            edge_out = edge_out.new_zeros((output_shape.shape[0], edge_out.shape[1])).index_add(
                0, output_index.long(), edge_out)
        return edge_out.reshape(edge_out.shape[0], dout, channels)

    def inputs(rows, rng, atoms):
        x_rows = atoms if gather else rows
        shapes = [(x_rows, shape[1], channels), (rows, shape[2], 1), (rows, nw, channels)]
        arrays = [rng.normal(scale=0.3, size=sh).astype(a.dtype) for sh in shapes]
        if gather or scatter:
            # Always pass both integer arguments; an unused mode ignores its index.
            arrays.extend(rng.integers(0, atoms, size=rows, dtype=np.int32) for _ in range(2))
            arrays.append(np.empty((atoms, 0), dtype=np.int32))
        return gpu_inputs(arrays)

    name = "cwtp" if index_mode == "none" else f"cwtp_index_{index_mode}"
    return {"name": name, "fn": fn, "reference": reference, "inputs": inputs,
            "meta": {"in_l": in_l, "sh_lmax": CWTP_SH_LMAX, "irreps_in1": irrep_string(in_l, channels),
                     "irreps_in2": base2, "irreps_out": actual_out, "index_mode": index_mode,
                     "out_dim": dout * channels, "weights": nw * channels}}


def make_stc(a, channels, hidden_l, layer):
    out_l = hidden_l if layer == 0 else 0
    meta = {"in_l": a.lmax, "out_l": out_l, "correlation": a.correlation, "species": a.species,
            "out_dim": (out_l+1)**2 * channels}
    native = cue.descriptors.symmetric_contraction(cue.Irreps("O3", irrep_string(a.lmax, channels)), cue.Irreps("O3", irrep_string(out_l, channels)), tuple(range(1, a.correlation+1)))
    module = cue_module(native.polynomial, a, "stc")
    meta.update(weights=a.species * native.inputs[0].dim)
    def fn(x, species, w):
        return module([w, x.reshape(x.shape[0], -1)], input_indices={0: species})[0]
    def inputs(rows, rng, atoms):
        x = rng.normal(scale=0.3, size=(rows, (a.lmax+1)**2, channels)).astype(a.dtype)
        species = rng.integers(0, a.species, rows, dtype=np.int32)
        weights = (rng.normal(scale=0.1, size=(a.species, native.inputs[0].dim)).astype(a.dtype),)
        return gpu_inputs((x, species, *weights))
    return {"name": "stc_native", "fn": fn, "inputs": inputs, "meta": meta}
def measure(fn, args, a, phase):
    """Measure synchronized latency; prepare each backward outside the timer."""
    phase = "backward" if phase == "forward_backward" else phase
    t0 = time.perf_counter()
    prepare = None
    args = tuple(x.detach().requires_grad_(True)
                 if phase == "backward" and x.is_floating_point()
                 else x.detach() for x in args)
    if phase == "forward":
        def run(state):
            with torch.no_grad():
                return fn(*args)
    else:
        variables = tuple(x for x in args if x.is_floating_point())
        with torch.no_grad():
            cotangent = torch.full_like(fn(*args), 0.2)
        torch.cuda.synchronize(0)
        def prepare():
            y = fn(*args)
            torch.cuda.synchronize(0)
            return y
        def run(state):
            return torch.autograd.grad(state, variables, grad_outputs=cotangent,
                                       allow_unused=True, materialize_grads=True)
    def sync(result):
        torch.cuda.synchronize(0)
    execution = "torch_eager"
    # Initialize backward paths before collecting any samples.
    def untimed_step():
        state = prepare() if prepare is not None else None
        result = run(state)
        sync(result)
    untimed_step()
    setup_seconds = time.perf_counter() - t0
    for _ in range(a.warmup):
        untimed_step()
    samples = []
    for _ in range(a.iterations):
        state = prepare() if prepare is not None else None
        start = time.perf_counter_ns()
        result = run(state)
        sync(result)
        elapsed_us = (time.perf_counter_ns() - start) / 1000
        samples.append(elapsed_us)
        # Release graphs, residuals and gradients outside the measured region.
        del state, result
    return {"median_us": float(np.median(samples)), "p10_us": float(np.percentile(samples, 10)),
            "p90_us": float(np.percentile(samples, 90)), "setup_s": setup_seconds, "execution": execution}
def is_oom(exc):
    """Recognize allocation failures without swallowing other runtime errors."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, MemoryError):
            return True
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        if "outofmemory" in name or any(token in message for token in (
            "out of memory", "out_of_memory", "cuda_error_out_of_memory", "std::bad_alloc",
        )):
            return True
        if "resource_exhausted" in message and ("memory" in message or "alloc" in message):
            return True
        exc = exc.__cause__ or exc.__context__
    return False
def oom_message(exc):
    message = f"{type(exc).__name__}: {exc}"[:1500]
    # Failed allocation/compilation frames can retain large GPU input buffers.
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        exc.__traceback__ = None
        exc = exc.__cause__ or exc.__context__
    return message
def cleanup():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        if not is_oom(exc):
            raise
def check_indexing(track, args, a):
    """Check indexing semantics and VJPs; never executed inside the timer."""
    values = tuple(v.detach().requires_grad_(True) if v.is_floating_point()
                   else v.detach() for v in args)
    floats = tuple(v for v in values if v.is_floating_point())
    actual = track["fn"](*values)
    expected = track["reference"](*values)
    generator = torch.Generator(device=actual.device).manual_seed(a.seed + 1)
    cotangent = torch.randn(actual.shape, dtype=actual.dtype, device=actual.device,
                            generator=generator)
    # Atomic scatter changes addition order. Float64 uses tighter tolerances.
    rtol, atol = (2e-4, 2e-4) if a.dtype == "float32" else (1e-9, 1e-9)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    actual_grads = torch.autograd.grad(actual, floats, cotangent)
    expected_grads = torch.autograd.grad(expected, floats, cotangent)
    for grad, ref_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=rtol, atol=atol)
    torch.cuda.synchronize(0)
    print("PASS indexed CWTP output and gradients (x/y/w)", flush=True)


def attempt_case(track, rows, atoms, a, phase):
    """Allocation, compilation, warmup and timing are all covered by OOM skip."""
    try:
        args = track["inputs"](rows, np.random.default_rng(a.seed), atoms)
        if track["name"].startswith("cwtp"):
            out_rows = atoms if track["meta"]["index_mode"] in ("output", "both") else rows
            out_shape = (out_rows, track["meta"]["out_dim"] // args[0].shape[-1], args[0].shape[-1])
            print(f"x shape:{args[0].shape}, y shape:{args[1].shape}, w shape:{args[2].shape}, "
                  f"out shape:torch.Size({list(out_shape)})", flush=True)
            if len(args) == 6:
                print(f"input_index shape:{args[3].shape}, output_index shape:{args[4].shape}", flush=True)
            if a.check_indexing and track["meta"]["index_mode"] != "none":
                check_indexing(track, args, a)
        torch.cuda.synchronize(0)
        return "ok", "", measure(track["fn"], args, a, phase)
    except Exception as exc:
        if not is_oom(exc):
            raise
        return "oom", oom_message(exc), {}
def write_rows(out, rows):
    with out.open("w", newline="") as f:
        fieldnames = list(rows[0])
        for row in rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
def main():
    a = parse_args()
    imports()
    packages = ["numpy", "torch", "cuequivariance", "cuequivariance-torch",
                "cuequivariance-ops-torch-cu12", "cuequivariance-ops-torch-cu13",
                "cuequivariance-ops-cu12", "cuequivariance-ops-cu13"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    environment = {
        "args": vars(a), "versions": versions, "python": sys.version,
        "platform": platform.platform(), "backend": "cueq", "method": "uniform_1d",
        "device": {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda},
        "timing": "synchronized_host_operator_latency",
        "backward_timing": "forward_and_residual_preparation_excluded",
        "presets": PRESETS,
        "cwtp_configs": {"sh_lmax": CWTP_SH_LMAX, "layers": CWTP_LAYERS,
                         "output_blocks": CWTP_OUTPUT_BLOCKS},
        "stc_config_source": "inferred_from_model_width_and_hidden_l_not_checkpoint_verified",
        "framework": "PyTorch eager",
    }
    print(json.dumps(environment, indent=2))
    rows = []
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".metadata.json").write_text(json.dumps(environment, indent=2))
    for model in a.models:
        model_channels, model_hidden_l = PRESETS[model]
        for layer in a.layers:
            for operator, index_mode in [(op, mode) for op in a.operators
                                         for mode in (a.index_modes if op == "cwtp" else ["none"])]:
                if operator == "cwtp" and layer not in CWTP_LAYERS[model]:
                    print(f"SKIP {model} CWTP layer {layer}: no configuration supplied.", flush=True)
                    continue
                channels, hidden_l = model_channels, model_hidden_l
                factory = make_cwtp if operator == "cwtp" else make_stc
                label = ("cwtp" if index_mode == "none" else f"cwtp_index_{index_mode}") if operator == "cwtp" else "stc_native"
                track, setup_error = None, ""
                try:
                    track = (factory(a, channels, hidden_l, layer, index_mode)
                             if operator == "cwtp" else factory(a, channels, hidden_l, layer))
                except Exception as exc:
                    if not is_oom(exc):
                        raise
                    setup_error = oom_message(exc)
                if setup_error:
                    cleanup()
                for phase in a.phases:
                    for atoms in a.atoms:
                        n = (a.edges if a.edges is not None else atoms*a.neighbors) if operator == "cwtp" else atoms
                        name = f"{model}_l{layer}_{label}_{phase}_{atoms}_cueq_uniform1d"
                        if setup_error:
                            status, error, result = "oom", setup_error, {}
                        else:
                            status, error, result = attempt_case(track, n, atoms, a, phase)
                        row = {"model": model, "layer": layer, "operator": label, "phase": phase,
                               "atoms": atoms, "rows": n, "channels": channels,
                               "index_mode": index_mode if operator == "cwtp" else "species",
                               "status": status, "error": error,
                               "dtype": a.dtype, 
                               "config": json.dumps(track["meta"], sort_keys=True) if track else "",
                               "median_us": ""}
                        row.update(result)
                        rows.append(row)
                        write_rows(out, rows)
                        if status == "oom":
                            print(f"SKIP OOM {name}: {error}", flush=True)
                        else:
                            print(f"{name}: {result['median_us']:.2f} us", flush=True)
                        cleanup()
                track = None
                cleanup()
if __name__ == "__main__":
    main()
