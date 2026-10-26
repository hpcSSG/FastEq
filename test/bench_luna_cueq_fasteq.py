#!/usr/bin/env python3
"""FastEq/cuEq uniform_1d and LunaEquivariance CWTP/STC/SevenNet UUU benchmark.

Requires CUDA-enabled torch, numpy and cuequivariance. Selected backends require
cuequivariance-torch (+ matching CUDA ops), the FastEq-enabled cuEq fork, and/or
LunaEquivariance. Install Luna after torch: pip install LunaEquivariance.

Examples:
  python benchmark_cueq_luna.py --operators cwtp
  python benchmark_cueq_luna.py --backends cueq luna --operators cwtp
  python benchmark_cueq_luna.py --backends cueq fasteq luna --operators cwtp
  python benchmark_cueq_luna.py --operators uuu --backends cueq fasteq luna --atoms 256 1024 4096
  python benchmark_cueq_luna.py --backends cueq luna --operators cwtp --models mace_mp_medium --layers 1 --atoms 32 --edges 127 --index-modes none input output both --check-backends --check-indexing

Defaults preserve the supplied FastEq implementation (use_fasteq=True), alongside
Luna. Select cueq for stock cuEq (no FastEq keyword). CSV backend labels distinguish
all three. Float32/float64, forward/backward, all original presets and OOM skipping
are retained. forward_backward aliases backward; second derivatives are not timed.
STC is unchanged for cuEq/FastEq; Luna STC records status=unsupported.

CWTP configurations: MACE-MP medium and MACE-OFF large/medium/small, channels
128/224/128/96, SH lmax=3. OFF-small CWTP has only layer 0. STC is inferred from
width/hidden L, not checkpoint verified. Default atoms: 4096, 8192, 12288.

--operators uuu selects SevenNet-Omni by default, with five zero-based layers.
--layers can select any subset of 0,1,2,3,4; omitting it runs all five.
These tests reproduce the supplied scalar-expanded polynomial dimensions:
layer | weights | node input | SH | output | uniform u | scalar paths
    0 |     512 |        128 | 16 |   2048 |       128 |           16
    1 |    1760 |        704 | 16 |   7776 |        32 |          777
    2 |    3008 |       1280 | 16 |  13504 |        32 |         1490
    3 |    3520 |       1408 | 16 |  15552 |        32 |         1554
    4 |     256 |        704 | 16 |    256 |        32 |           22
cuEq/FastEq use explicit flattening and channel splitting to the logged u; this
avoids the naive path for mixed multiplicities. Dimensions and scalar path counts
are checked at construction. Connectivity is reconstructed from standard SevenNet
paths and the supplied summaries, not loaded from a checkpoint; summaries alone
do not uniquely specify every coefficient or segment index. CSV records this.
The logged a[u]*b[u]*c[]->D[u] contraction is called uuu here, while its angular
TP formulation uses UVU with SH multiplicity one. Luna uses UVU/UVUConv for the
same computation; SH is never replicated across channels. CSV channels is the
uniform tile u, and config also lists the original mixed channel multiplicities.
All four index modes, forward/backward, source export and GPU checks also apply.

Luna uses UVU TensorProduct without indices; both indices use TensorProductConv.
Input-only/output-only use TP + explicit torch gather/scatter inside the timer.
--luna-indexing explicit also uses explicit gather/scatter for both indices.
Luna fused graphs require destination-sorted edges. In per_iteration mode each
fused call follows the official three expressions on original unsorted inputs:
  order = dst.argsort(stable=True)
  graph = conv.create_graph(src[order], dst[order], num_nodes)
  out = conv(x, y[order], w[order], graph=graph)
CSV create_graph_us includes the graph expression's src/dst indexing.
The end-to-end breakdown is create_graph_us + forward_us + backward_us + prepare_us.
For fused Luna, forward_us measures conv after floating edge indexing. backward_us
measures the convolution VJP, stopping at the sorted edge tensors. A second VJP
maps those gradients back to original edge order. prepare_us is the remaining
end-to-end host interval: argsort, external floating gathers, index backward and
residual dispatch/statistics overhead. Internal Luna gather/reduction stays in
the convolution interval. argsort_us is a subset of prepare_us; do not add twice.
Two VJP calls and stage synchronization add instrumentation overhead. All backends
use the same outer host timer; non-fused backends retain complete fn/VJP boundaries.
Legacy operator_us includes floating indexing/its gradients, and graph_preprocess_us
remains an alias for create_graph_us. Per-step components add to end_to_end_us;
independent medians in CSV need not add exactly. No pure kernel times are claimed.
cuEq/FastEq always sort scatter cases (output/both) by destination during setup,
independently of selected backends. Explicit Luna indexing retains setup sorting.
--luna-graph-mode per_iteration (default) repeats sorting and graph construction
on EVERY fused forward, including setup, warmup and measured iterations.
Forward operator_us subtracts argsort and graph from the contiguous total.
Backward operator_us includes gradients through floating edge indexing.
Backward total_us/median_us summarize argsort + graph + backward; preparation
forward is separate in backward_prepare_forward_us and included in end_to_end_us.
No extra argsort or graph build occurs during backward itself.
Medians are independent, so component medians may not sum to the total median.
--luna-graph-mode once sorts/reorders/builds once during setup and reports zero
per-iteration argsort and graph cost. Inputs/indices stay fixed in either mode.
Initial transfers, layout conversion and neighbor search are excluded.
Stage timings include host dispatch and synchronization overhead. Setup/cold and
warmup samples are excluded from CSV statistics. Full models can reuse graphs.
cuEq/FastEq use ir_mul and Luna uses mul_ir. Outputs remain in native layout.
This measures native-layout operator latency, not conversion-inclusive latency.
Weights and each output block follow the same sorted CWTP path order.
Generated Luna CUDA/C++ sources are saved by default under luna_cuda_sources/.
--luna-dump-dir PATH changes the directory; --luna-dump-dir "" disables export.
Each source/options variant has its own subdirectory containing cuda.cu,
main.cpp and build_options.json. Export runs before the compiled-library cache
lookup, so cached builds are also captured. Only exercised specializations are
exported. Sources precede jit.py's implicit header additions; these are recorded
in the manifest via no_implicit_headers when explicitly supplied.
--check-backends validates Luna outputs and all x/y/w VJPs against native cuEq,
including differentiable layout conversion, outside timing. It requires cuEq ops.
--check-indexing checks against explicit gather + TP + scatter outside timing.

Timing: synchronized host latency (dispatch + execution + synchronization), eager
PyTorch, no torch.compile. Every backward forward/residual preparation occurs
outside the operator-only component. All backward floating inputs require
gradients. No radial MLP or trained model is included. Unknown runtime errors are raised, not silently skipped.
Source interface reviewed at Luna commit 5a24ffeb74ecf74de95211c72acfaa1f342126c3;
this records the reviewed API, not the installed package's Git commit.
"""
from __future__ import annotations
import argparse
import csv
import gc
import importlib.metadata
import inspect
import hashlib
import importlib
import re
from functools import wraps
import json
from pathlib import Path
import platform
import sys
import time
import ctypes


cupti = ctypes.CDLL(
    "/home/malixian/miniconda3/envs/fasteq/lib/"
    "python3.10/site-packages/nvidia/cu13/lib/libcupti.so.13"
)
cupti.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
cupti.cuptiActivityFlushAll.restype = ctypes.c_int

PRESETS = {
    "mace_mp_medium": (128, 1),
    "mace_off_large": (224, 2),
    "mace_off_medium": (128, 1),
    "mace_off_small": (96, 0),
}
SEVENNET_MODEL = "sevennet_omni"
# (multiplicity, degree, parity). Reconstructed standard SevenNet connectivity;
# runtime descriptor checks below require every supplied dimension/path count.
SEVENNET_INPUTS = (
    ((128, 0, 1),),
    ((128, 0, 1), (64, 1, -1), (32, 2, 1), (32, 3, -1)),
    ((128, 0, 1), (64, 1, -1), (64, 1, 1), (32, 2, -1),
     (32, 2, 1), (32, 3, -1), (32, 3, 1)),
    ((128, 0, -1), (128, 0, 1), (64, 1, -1), (64, 1, 1),
     (32, 2, -1), (32, 2, 1), (32, 3, -1), (32, 3, 1)),
    ((128, 0, 1), (64, 1, -1), (32, 2, 1), (32, 3, -1)),
)
SEVENNET_LOGS = (
    (512, 128, 16, 2048, 128, 16),
    (1760, 704, 16, 7776, 32, 777),
    (3008, 1280, 16, 13504, 32, 1490),
    (3520, 1408, 16, 15552, 32, 1554),
    (256, 704, 16, 256, 32, 22),
)


def blocks_string(blocks):
    return "+".join(f"{mul}x{l}{'e' if parity == 1 else 'o'}"
                    for mul, l, parity in blocks)


def sevennet_plan(layer):
    blocks = SEVENNET_INPUTS[layer]
    paths = []
    for i1, (mul, l1, p1) in enumerate(blocks):
        for l2 in range(4):
            for l3 in range(abs(l1-l2), min(l1+l2, 3)+1):
                parity = p1 * (-1)**l2
                if layer == 4 and (l3, parity) != (0, 1):
                    continue
                paths.append(dict(i1=i1, i2=l2, l1=l1, l2=l2, l3=l3,
                                  parity=parity, channels=mul))
    # cuEq orders natural parity before opposite parity for each degree.
    paths.sort(key=lambda p: (p['l3'], -p['parity']*(-1)**p['l3']))
    output_blocks = [(p['channels'], p['l3'], p['parity']) for p in paths]
    w, x, y, out, u, nnz = SEVENNET_LOGS[layer]
    assert sum(mul*(2*l+1) for mul, l, _ in blocks) == x
    assert sum(p['channels'] for p in paths) == w
    assert sum(mul*(2*l+1) for mul, l, _ in output_blocks) == out
    return dict(layer=layer, input_blocks=blocks, output_blocks=output_blocks,
                paths=paths, irreps_in1=blocks_string(blocks),
                irreps_in2="1x0e+1x1o+1x2e+1x3o", irreps_out=blocks_string(output_blocks),
                weight_dim=w, in_dim=x, sh_dim=y, out_dim=out, u=u, nnz_paths=nnz)


def irreps_layout_permutation(blocks):
    indices, offset = [], 0
    for channels, l, _ in blocks:
        dim = 2*l+1
        indices.extend(offset+m*channels+c
                       for c in range(channels) for m in range(dim))
        offset += channels*dim
    return indices
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
LUNA_SOURCE_REF = "5a24ffeb74ecf74de95211c72acfaa1f342126c3"


def cwtp_paths(in_l):
    """cuEq channelwise path order: input pair order, then stable output-irrep sort."""
    paths = [(l1, l2, l3)
             for l1 in range(in_l + 1)
             for l2 in range(CWTP_SH_LMAX + 1)
             for l3 in range(abs(l1-l2), min(l1+l2, CWTP_SH_LMAX) + 1)
             if (l1+l2+l3) % 2 == 0]
    return sorted(paths, key=lambda path: path[2])


def layout_permutation(block_ls, channels):
    """Indices mapping flattened ir_mul to e3nn/Luna mul_ir, block by block."""
    indices, offset = [], 0
    for l in block_ls:
        dim = 2*l + 1
        indices.extend(offset + m*channels + c
                       for c in range(channels) for m in range(dim))
        offset += channels*dim
    return indices


def irrep_string(lmax, channels=1):
    return "+".join(f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1))


def install_luna_source_dump(directory):
    """Export generated sources before Luna checks its compiled-library cache.

    Sources are captured at load_extension entry, before jit.py adds implicit
    headers. The manifest retains explicit compiler options. Dump hashes identify
    source/options, and are separate from Luna's toolchain-dependent cache hashes.
    """
    jit = importlib.import_module("LunaEquivariance.utils.jit")
    original = jit.load_extension
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dumped = set()

    @wraps(original)
    def load_and_dump(name, **kwargs):
        manifest = {"name": name, "source_stage": "before_jit_implicit_headers",
                    "options": {k: v for k, v in kwargs.items()
                                if k not in ("cpp_sources", "cuda_sources")}}
        cuda_source = kwargs.get("cuda_sources", "")
        cpp_source = kwargs.get("cpp_sources", "")
        fingerprint = json.dumps([manifest, cpp_source, cuda_source],
                                 sort_keys=True, default=str)
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:24]
        if digest not in dumped:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name)).strip(".") or "extension"
            destination = root / f"{safe_name}_{digest}"
            destination.mkdir(parents=True, exist_ok=True)
            if cuda_source:
                (destination / "cuda.cu").write_text(cuda_source, encoding="utf-8")
            (destination / "main.cpp").write_text(cpp_source, encoding="utf-8")
            (destination / "build_options.json").write_text(
                json.dumps(manifest, indent=2, default=str), encoding="utf-8")
            dumped.add(digest)
            print(f"Luna generated source: {destination}", flush=True)
        return original(name, **kwargs)

    # Several modules use `from .jit import load_extension`. Replace their existing
    # aliases too; future imports see the wrapper on jit.load_extension.
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("LunaEquivariance.") and module is not None:
            if vars(module).get("load_extension") is original:
                module.load_extension = load_and_dump
    print(f"Luna CUDA source export directory: {root}", flush=True)


class LunaGraphTiming:
    """Time official argsort and graph expressions as separate host intervals."""

    def __init__(self):
        self.records = []
        self.sort_records = []
        self.forward_records = []
        self.operator_inputs = None

    def argsort(self, dst):
        torch.cuda.synchronize(0)
        start_ns = time.perf_counter_ns()
        order = dst.argsort(stable=True)
        torch.cuda.synchronize(0)
        self.sort_records.append((time.perf_counter_ns() - start_ns) / 1000.0)
        return order

    def argsort_us(self):
        return sum(self.sort_records)

    def build(self, conv, src, dst, num_nodes, *, order=None):
        torch.cuda.synchronize(0)
        start_ns = time.perf_counter_ns()
        # Include argument indexing, matching the official complete expression.
        if order is not None:
            graph = conv.create_graph(src[order], dst[order], num_nodes)
        else:
            graph = conv.create_graph(src, dst, num_nodes)
        torch.cuda.synchronize(0)
        self.records.append((time.perf_counter_ns() - start_ns) / 1000.0)
        return graph

    def forward(self, conv, x, y, w, *, graph):
        # Arguments have already been indexed before entering this timer.
        # Keep the boundary tensors to stop the first VJP before index backward.
        self.operator_inputs = (x, y, w) if torch.is_grad_enabled() else None
        torch.cuda.synchronize(0)
        start_ns = time.perf_counter_ns()
        out = conv(x, y, w, graph=graph)
        torch.cuda.synchronize(0)
        self.forward_records.append((time.perf_counter_ns() - start_ns) / 1000.0)
        return out

    def forward_us(self):
        return sum(self.forward_records)

    def backward(self, state, variables, cotangent):
        """Compute convolution VJP, then the external edge permutation VJP.

        The returned time covers only the first VJP; the enclosing measurement
        still includes both calls, synchronization and Python engine overhead.
        Internal Luna gathers/reductions remain inside the convolution VJP.
        """
        boundary = self.operator_inputs
        if boundary is None or len(boundary) != len(variables):
            raise RuntimeError("Missing Luna convolution input boundary for backward timing")
        torch.cuda.synchronize(0)
        start_ns = time.perf_counter_ns()
        gradients = torch.autograd.grad(
            state, boundary, grad_outputs=cotangent,
            allow_unused=True, materialize_grads=True)
        torch.cuda.synchronize(0)
        backward_us = (time.perf_counter_ns() - start_ns) / 1000.0

        # Stop above at the sorted tensors, then propagate their gradients back
        # to the original edge order. Never detach or omit the index gradients.
        changed = [i for i, (prepared, original) in enumerate(zip(boundary, variables))
                   if prepared is not original]
        result = list(gradients)
        if changed:
            remapped = torch.autograd.grad(
                tuple(boundary[i] for i in changed),
                tuple(variables[i] for i in changed),
                grad_outputs=tuple(gradients[i] for i in changed),
                allow_unused=True, materialize_grads=True)
            for i, grad in zip(changed, remapped):
                result[i] = grad
        self.operator_inputs = None
        return tuple(result), backward_us

    def reset(self):
        self.records.clear()
        self.sort_records.clear()
        self.forward_records.clear()
        self.operator_inputs = None

    def elapsed_us(self):
        return sum(self.records)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", nargs="+", choices=["cueq", "fasteq", "luna"],
                   default=["fasteq", "luna"],
                   help="fasteq preserves the supplied use_fasteq=True path; cueq is native")
    p.add_argument("--check-backends", action="store_true",
                   help="Check Luna outputs and x/y/w VJPs against native cuEq, outside timing")
    p.add_argument("--luna-indexing", choices=["fused", "explicit"], default="fused",
                   help="Use TensorProductConv for both indices, or time TP + torch gather/scatter")
    p.add_argument("--luna-graph-mode", choices=["per_iteration", "once"],
                   default="per_iteration",
                   help="Rebuild fused Luna graph on every forward (including warmup); "
                        "total includes argsort + graph + forward/backward; "
                        "preparation forward is separate. "
                        "Use once for graph reuse")
    # Accept previous diagnostic flags for command compatibility; no per-call logs.
    p.add_argument("--luna-log-graph-times", action=argparse.BooleanOptionalAction,
                   default=False, help=argparse.SUPPRESS)
    p.add_argument("--luna-dump-dir", default="luna_cuda_sources",
                   help="Save Luna generated CUDA/C++ and compiler options here; empty string disables")
    p.add_argument("--models", nargs="+", choices=[*PRESETS, SEVENNET_MODEL], default=None)
    p.add_argument("--operators", nargs="+", choices=["cwtp", "stc", "uuu"], default=["cwtp", "stc"])
    p.add_argument("--layers", nargs="+", type=int, choices=range(5), default=None,
                   help="Zero-based layers; default MACE=0,1; SevenNet=0,1,2,3,4")
    p.add_argument("--atoms", nargs="+", type=int, default=[4096, 8192, 12288])
    p.add_argument("--edges", type=int, default=None,
                   help="Exact CWTP/SevenNet UUU edge count; overrides atoms * neighbors")
    p.add_argument("--index-modes", nargs="+", choices=["none", "input", "output", "both"],
                   default=["none", "both"], help="CWTP/SevenNet UUU indexing modes; STC is unchanged")
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
    p.add_argument("--output", default="cueq_luna.csv")
    a = p.parse_args()
    if a.models is None:
        a.models = ([] if a.operators == ["uuu"] else list(PRESETS))
        if "uuu" in a.operators:
            a.models.append(SEVENNET_MODEL)
    a.backends = list(dict.fromkeys(a.backends))
    a.index_modes = list(dict.fromkeys(a.index_modes))
    a.phases = list(dict.fromkeys("backward" if phase == "forward_backward" else phase for phase in a.phases))
    if min(a.atoms) < 1 or min(a.neighbors, a.species, a.iterations) < 1 or a.warmup < 0:
        p.error("sizes and iterations must be positive; warmup must be nonnegative")
    if a.edges is not None and a.edges < 1:
        p.error("edges must be positive")
    if a.lmax < 0 or not 1 <= a.correlation <= 3:
        p.error("lmax must be nonnegative, and correlation must be in [1,3]")
    if "stc" in a.operators and any(a.lmax < PRESETS[m][1] for m in a.models if m in PRESETS):
        p.error("STC lmax must cover the selected models' hidden L")
    if not any((m == SEVENNET_MODEL and "uuu" in a.operators) or
               (m in PRESETS and any(op != "uuu" for op in a.operators)) for m in a.models):
        p.error("uuu requires --models sevennet_omni; cwtp/stc require a MACE preset")
    return a
def imports(a):
    global np, cue, cuet, torch, luna
    import numpy as np
    import cuequivariance as cue
    import torch
    if any(b in a.backends for b in ("cueq", "fasteq")) or a.check_backends:
        import cuequivariance_torch as cuet
    if "luna" in a.backends:
        import LunaEquivariance as luna
        if a.luna_dump_dir:
            install_luna_source_dump(a.luna_dump_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmarks require PyTorch with an NVIDIA CUDA GPU.")
    torch.cuda.set_device(0)
def cue_module(poly, a, op_name, backend="fasteq"):
    parameters = inspect.signature(cuet.SegmentedPolynomial).parameters
    kwargs = dict(method="uniform_1d", math_dtype=getattr(torch, a.dtype))
    # The supplied local fork uses op_name; upstream cuEq uses name.
    name_key = "op_name" if "op_name" in parameters else "name" if "name" in parameters else None
    if name_key is not None:
        kwargs[name_key] = op_name
    if "use_fasteq" in parameters:
        kwargs["use_fasteq"] = backend == "fasteq"
    elif backend == "fasteq":
        raise RuntimeError("--backends fasteq requires the supplied FastEq-enabled cuEq fork; "
                           "select --backends cueq luna for stock cuEq")
    module = cuet.SegmentedPolynomial(poly, **kwargs).to("cuda:0")
    if module.method != "uniform_1d" or "naive" in type(getattr(module, "m", None)).__name__.lower():
        raise RuntimeError(f"cuEq switched to {module.method}; install its CUDA ops package.")
    return module
def gpu_inputs(arrays):
    # Transfer inputs before timing.
    return tuple(torch.as_tensor(np.array(x, copy=True), device="cuda:0") for x in arrays)
def make_cwtp(a, channels, hidden_l, layer, index_mode="none", backend="fasteq"):
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
    if backend == "luna":
        return make_luna_cwtp(a, channels, in_l, index_mode, full, nw, dout)
    module = cue_module(full.polynomial, a, "cwtp", backend)
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
        if scatter:
            # Destination order depends only on index mode, never on selected backends.
            order = np.argsort(arrays[4], kind="stable")
            for i in (1, 2, 3, 4):
                arrays[i] = arrays[i][order]
            if not gather:
                arrays[0] = arrays[0][order]
        return gpu_inputs(arrays)

    name = "cwtp" if index_mode == "none" else f"cwtp_index_{index_mode}"
    return {"name": name, "fn": fn, "reference": reference, "inputs": inputs,
            "meta": {"in_l": in_l, "sh_lmax": CWTP_SH_LMAX, "irreps_in1": irrep_string(in_l, channels),
                     "irreps_in2": base2, "irreps_out": actual_out, "index_mode": index_mode,
                     "out_dim": dout * channels, "weights": nw * channels,
                     "channels": channels, "layout": "ir_mul",
                     "implementation": "uniform_1d", "backend": backend,
                     "edge_order": "destination_sorted" if scatter else "random"}}


def make_luna_cwtp(a, channels, in_l, index_mode, descriptor, nw, dout,
                   *, plan=None, polynomial=None):
    """Benchmark native layouts with optional per-iteration graph construction."""
    gather = index_mode in ("input", "both")
    scatter = index_mode in ("output", "both")
    fused = index_mode == "both" and a.luna_indexing == "fused"
    if plan is None:
        paths = cwtp_paths(in_l)
        out_ls = [path[2] for path in paths]
        if tuple(out_ls) != CWTP_OUTPUT_BLOCKS[in_l] or nw != len(paths):
            raise ValueError("cuEq channelwise descriptor does not match the reviewed Luna path mapping")
        input_blocks = [(channels, l, (-1)**l) for l in range(in_l+1)]
        output_blocks = [(channels, l, (-1)**l) for l in out_ls]
        irreps_in1, irreps_in2 = irrep_string(in_l, channels), irrep_string(CWTP_SH_LMAX)
        irreps_out = blocks_string(output_blocks)
        instructions = [(l1, l2, i, "uvu", True) for i, (l1, l2, _) in enumerate(paths)]
    else:
        input_blocks, output_blocks = plan['input_blocks'], plan['output_blocks']
        irreps_in1, irreps_in2, irreps_out = (plan[k] for k in ('irreps_in1', 'irreps_in2', 'irreps_out'))
        instructions = [(p['i1'], p['i2'], i, "uvu", True) for i, p in enumerate(plan['paths'])]
    input_dim = sum(mul*(2*l+1) for mul, l, _ in input_blocks)
    if polynomial is None:
        polynomial = descriptor.polynomial
    if irreps_out != str(descriptor.outputs[0].irreps).replace(" ", ""):
        raise ValueError("Luna/cuEq output block ordering differs")
    # Every CWTP output block has exactly one weighted path, with mul_in2=1.
    # component + no path normalization gives sqrt(2*l_out+1) * Wigner3j,
    # matching cuEq's normalized Clebsch-Gordan coefficient for that block.
    options = dict(internal_weights=False, shared_weights=False,
                   irrep_normalization="component", path_normalization="none",
                   dtype=getattr(torch, a.dtype))
    tp = luna.TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions, **options).to("cuda:0")
    conv = (luna.TensorProductConv(irreps_in1, irreps_in2, irreps_out,
                                   instructions, **options).to("cuda:0") if fused else None)
    if tp.weight_numel != nw*channels:
        raise ValueError("Luna/cuEq per-edge weight sizes differ")
    x_perm = torch.tensor(irreps_layout_permutation(input_blocks), device="cuda:0")
    out_perm = torch.tensor(irreps_layout_permutation(output_blocks), device="cuda:0")
    x_inverse = torch.argsort(x_perm)
    graph = None
    graph_timing = LunaGraphTiming()
    cue_ref_module = cue_module(polynomial, a, "cwtp", "cueq") if a.check_backends else None

    def fn(x, y, w, input_index=None, output_index=None, output_shape=None):
        if fused:
            if a.luna_graph_mode == "per_iteration":
                # Each call starts from the same original unsorted edge data.
                order = graph_timing.argsort(output_index)
                active_graph = graph_timing.build(
                    conv, input_index, output_index, output_shape.shape[0], order=order)
                # Indexing stays differentiable; the inner forward timer starts
                # after indexing, so the four-way breakdown puts it in prepare.
                return graph_timing.forward(conv, x, y[order], w[order], graph=active_graph)
            else:
                active_graph = graph
            return graph_timing.forward(conv, x, y, w, graph=active_graph)
        edge_x = x.index_select(0, input_index.long()) if gather else x
        out = tp(edge_x, y, w)
        if scatter:
            out = out.new_zeros((output_shape.shape[0], out.shape[1])).index_add(
                0, output_index.long(), out)
        return out

    fn.graph_timing = graph_timing if fused else None

    def reference(x, y, w, input_index=None, output_index=None, output_shape=None):
        # Independent unindexed TP + explicit gather/scatter checks fused graph semantics.
        edge_x = x.index_select(0, input_index.long()) if gather else x
        out = tp(edge_x, y, w)
        if scatter:
            out = out.new_zeros((output_shape.shape[0], out.shape[1])).index_add(
                0, output_index.long(), out)
        return out

    def cue_reference(x, y, w, input_index=None, output_index=None, output_shape=None):
        # Convert only for validation, outside all timed regions. Autograd traverses
        # these permutations so VJPs are compared in the same native Luna layout.
        canonical_x = x.index_select(1, x_inverse)
        edge_x = canonical_x.index_select(0, input_index.long()) if gather else canonical_x
        out = cue_ref_module([w, edge_x, y])[0]
        if scatter:
            out = out.new_zeros((output_shape.shape[0], out.shape[1])).index_add(
                0, output_index.long(), out)
        return out.index_select(1, out_perm)

    def inputs(rows, rng, atoms):
        nonlocal graph
        x_rows = atoms if gather else rows
        shapes = [(x_rows, input_dim // channels, channels),
                  (rows, (CWTP_SH_LMAX+1)**2, 1), (rows, nw, channels)]
        arrays = [rng.normal(scale=0.3, size=sh).astype(a.dtype) for sh in shapes]
        if gather or scatter:
            arrays.extend(rng.integers(0, atoms, size=rows, dtype=np.int32) for _ in range(2))
            arrays.append(np.empty((atoms, 0), dtype=np.int32))
        if scatter and not fused:
            order = np.argsort(arrays[4], kind="stable")
            for i in (1, 2, 3, 4):
                arrays[i] = arrays[i][order]
            if not gather:
                arrays[0] = arrays[0][order]
        values = list(gpu_inputs(arrays))
        values[0] = values[0].reshape(x_rows, -1).index_select(1, x_perm).contiguous()
        values[1] = values[1].reshape(rows, -1)
        values[2] = values[2].reshape(rows, -1)
        graph = None
        if fused and a.luna_graph_mode == "once":
            # Reuse mode: GPU sort, permutation and graph build are setup only.
            order = graph_timing.argsort(values[4])
            graph = graph_timing.build(conv, values[3], values[4], atoms, order=order)
            for i in (1, 2, 3, 4):
                values[i] = values[i][order]
            torch.cuda.synchronize(values[0].device)
            graph_timing.reset()
        return tuple(values)

    implementation = "uvu_conv" if fused else "uvu" if index_mode == "none" else "uvu_torch_indexing"
    name = "cwtp" if index_mode == "none" else f"cwtp_index_{index_mode}"
    return {"name": name, "fn": fn, "reference": reference, "cue_reference": cue_reference,
            "inputs": inputs,
            "meta": {"backend": "luna", "implementation": implementation,
                     "layout": "mul_ir", "in_l": in_l, "sh_lmax": CWTP_SH_LMAX,
                     "irreps_in1": irreps_in1, "irreps_in2": irreps_in2,
                     "irreps_out": irreps_out, "index_mode": index_mode,
                     "channels": channels, "out_dim": dout*channels, "weights": nw*channels,
                     "edge_order": "random_input_sorted_per_call" if fused and a.luna_graph_mode == "per_iteration"
                                   else "destination_sorted" if scatter else "random",
                     "argsort_in_timing": fused and a.luna_graph_mode == "per_iteration",
                     "graph_mode": a.luna_graph_mode if fused else "not_applicable",
                     "other_preprocessing_in_timing": fused and a.luna_graph_mode == "per_iteration",
                     "graph_construction_in_forward_timing": fused and a.luna_graph_mode == "per_iteration",
                     "graph_construction_in_backward_timing": fused and a.luna_graph_mode == "per_iteration"}}


def make_sevennet_uuu(a, layer, index_mode, backend):
    """Reconstruct the supplied scalar-expanded UUU polynomial from SevenNet TP paths.

    Its angular formulation is UVU (edge multiplicity one); Luna runs that same
    formulation without expanding spherical harmonics along the channel axis.
    """
    plan = sevennet_plan(layer)
    filters = ([cue.O3(0, 1)] if layer == 4 else
               [cue.O3(l, p) for l in range(4) for p in (-1, 1)])
    full = cue.descriptors.channelwise_tensor_product(
        cue.Irreps("O3", plan['irreps_in1']), cue.Irreps("O3", plan['irreps_in2']), filters)
    actual_dims = tuple(operand.dim for operand in (*full.inputs, *full.outputs))
    expected_dims = tuple(plan[k] for k in ('weight_dim', 'in_dim', 'sh_dim', 'out_dim'))
    if actual_dims != expected_dims:
        raise ValueError(f"SevenNet layer {layer} descriptor dimensions {actual_dims} != {expected_dims}")
    if str(full.outputs[0].irreps).replace(' ', '') != plan['irreps_out']:
        raise ValueError(f"SevenNet layer {layer}: reconstructed output block order differs")
    # First remove v=1; flatten i,j,k coefficients; then split all channel
    # multiplicities into the exact common u from the user's polynomial dump.
    poly = full.polynomial.squeeze_modes().flatten_coefficient_modes().split_mode('u', plan['u'])
    actual_paths = sum(len(stp.paths) for _, stp in poly.operations)
    if actual_paths != plan['nnz_paths']:
        raise ValueError(f"SevenNet layer {layer}: expanded path count {actual_paths} != "
                         f"supplied {plan['nnz_paths']}; exact connectivity must be checked")
    expected_segments = [plan['weight_dim']//plan['u'], plan['in_dim']//plan['u'],
                         16, plan['out_dim']//plan['u']]
    actual_segments = [len(operand.segments) for operand in poly.operands]
    if actual_segments != expected_segments:
        raise ValueError(f"SevenNet layer {layer}: segment counts {actual_segments} != {expected_segments}")
    print(f"SevenNet layer {layer}: {poly}", flush=True)
    meta = {k: plan[k] for k in ('layer', 'irreps_in1', 'irreps_in2', 'irreps_out', 'nnz_paths')}
    meta.update(channels=plan['u'], uniform_u=plan['u'], out_dim=plan['out_dim'],
                weights=plan['weight_dim'], input_dim=plan['in_dim'], sh_dim=16,
                operand_order=['weights', 'node_features', 'spherical_harmonics'],
                segment_counts=[plan['weight_dim']//plan['u'], plan['in_dim']//plan['u'],
                                16, plan['out_dim']//plan['u']],
                channel_multiplicities=[mul for mul, _, _ in plan['input_blocks']],
                config_source='standard_SevenNet_connectivity_reconstructed_and_log_checked',
                checkpoint_verified=False, polynomial=str(poly), index_mode=index_mode)
    name = 'sevennet_uuu' if index_mode == 'none' else f'sevennet_uuu_index_{index_mode}'
    if backend == 'luna':
        track = make_luna_cwtp(a, plan['u'], 3, index_mode, full,
                               plan['weight_dim']//plan['u'], plan['out_dim']//plan['u'],
                               plan=plan, polynomial=poly)
        track['name'] = name
        track['meta'].update(meta, luna_connection_mode='uvu',
                             benchmark_connection='scalar_expanded_uuu')
        return track
    # Preserve the supplied FastEq fork's CWTP dispatch selector: the scalar
    # expanded UUU polynomial here is still a weighted channelwise TP.
    module = cue_module(poly, a, 'cwtp', backend)
    gather = index_mode in ('input', 'both')
    scatter = index_mode in ('output', 'both')

    def fn(x, y, w, input_index=None, output_index=None, output_shape=None):
        buffers = [w.reshape(w.shape[0], -1), x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)]
        kwargs = {}
        if gather:
            kwargs['input_indices'] = {1: input_index}
        if scatter:
            kwargs.update(output_indices={0: output_index}, output_shapes={0: output_shape})
        out = module(buffers, **kwargs)[0]
        return out.reshape(out.shape[0], plan['out_dim']//plan['u'], plan['u'])

    def reference(x, y, w, input_index=None, output_index=None, output_shape=None):
        edge_x = x.index_select(0, input_index.long()) if gather else x
        out = module([w.reshape(w.shape[0], -1), edge_x.reshape(edge_x.shape[0], -1),
                      y.reshape(y.shape[0], -1)])[0]
        if scatter:
            out = out.new_zeros((output_shape.shape[0], out.shape[1])).index_add(0, output_index.long(), out)
        return out.reshape(out.shape[0], plan['out_dim']//plan['u'], plan['u'])

    def inputs(rows, rng, atoms):
        x_rows = atoms if gather else rows
        u = plan['u']
        shapes = [(x_rows, plan['in_dim']//u, u), (rows, 16, 1),
                  (rows, plan['weight_dim']//u, u)]
        arrays = [rng.normal(scale=0.3, size=sh).astype(a.dtype) for sh in shapes]
        if gather or scatter:
            arrays.extend(rng.integers(0, atoms, size=rows, dtype=np.int32) for _ in range(2))
            arrays.append(np.empty((atoms, 0), dtype=np.int32))
        if scatter:
            order = np.argsort(arrays[4], kind='stable')
            for i in (1, 2, 3, 4):
                arrays[i] = arrays[i][order]
            if not gather:
                arrays[0] = arrays[0][order]
        return gpu_inputs(arrays)

    meta.update(backend=backend, implementation='uniform_1d', layout='ir_mul',
                edge_order='destination_sorted' if scatter else 'random')
    return dict(name=name, fn=fn, reference=reference, inputs=inputs, meta=meta)


def make_stc(a, channels, hidden_l, layer, backend="fasteq"):
    if backend == "luna":
        raise NotImplementedError("LunaEquivariance has no native symmetric contraction (STC) API")
    out_l = hidden_l if layer == 0 else 0
    meta = {"in_l": a.lmax, "out_l": out_l, "correlation": a.correlation, "species": a.species,
            "out_dim": (out_l+1)**2 * channels}
    native = cue.descriptors.symmetric_contraction(cue.Irreps("O3", irrep_string(a.lmax, channels)), cue.Irreps("O3", irrep_string(out_l, channels)), tuple(range(1, a.correlation+1)))
    module = cue_module(native.polynomial, a, "stc", backend)
    meta.update(weights=a.species * native.inputs[0].dim, backend=backend, layout="ir_mul", implementation="uniform_1d")
    def fn(x, species, w):
        return module([w, x.reshape(x.shape[0], -1)], input_indices={0: species})[0]
    def inputs(rows, rng, atoms):
        x = rng.normal(scale=0.3, size=(rows, (a.lmax+1)**2, channels)).astype(a.dtype)
        species = rng.integers(0, a.species, rows, dtype=np.int32)
        weights = (rng.normal(scale=0.1, size=(a.species, native.inputs[0].dim)).astype(a.dtype),)
        return gpu_inputs((x, species, *weights))
    return {"name": "stc_native", "fn": fn, "inputs": inputs, "meta": meta}
def measure(fn, args, a, phase):
    """Measure four disjoint end-to-end components on each measured step.

    Fused Luna forward_us is conv only; backward_us is conv VJP only. Its
    external permutation VJP is executed separately, preserving input gradients.
    prepare_us is the end-to-end residual after graph + forward + backward:
    sorting, floating gathers, index backward and residual host/timing overhead.
    argsort_us is a subset of prepare_us, not a fifth additive component.
    Other backends use their complete fn/VJP as the forward/backward boundary.
    Legacy operator_us/total_us retain their combined indexing semantics.
    All medians/percentiles exclude setup and warmup.
    """
    phase = "backward" if phase == "forward_backward" else phase
    graph_timing = getattr(fn, "graph_timing", None)

    def reset_graph():
        if graph_timing is not None:
            graph_timing.reset()

    def graph_us():
        return graph_timing.elapsed_us() if graph_timing is not None else 0.0

    t0 = time.perf_counter()
    args = tuple(x.detach().requires_grad_(True)
                 if phase == "backward" and x.is_floating_point()
                 else x.detach() for x in args)
    variables = tuple(x for x in args if x.is_floating_point())
    if phase == "backward":
        # Shape discovery is setup only; do not retain its graph timing.
        with torch.no_grad():
            cotangent = torch.full_like(fn(*args), 0.2)
        torch.cuda.synchronize(0)

    def step():
        reset_graph()
        torch.cuda.synchronize(0)
        full_start = time.perf_counter_ns()
        if phase == "forward":
            with torch.no_grad():
                result = fn(*args)
            torch.cuda.synchronize(0)
            end_to_end = (time.perf_counter_ns() - full_start) / 1000.0
            construction = graph_us()
            sorting = graph_timing.argsort_us() if graph_timing is not None else 0.0
            preprocessing = sorting + construction
            operator = max(0.0, end_to_end - preprocessing)
            forward = graph_timing.forward_us() if graph_timing is not None else operator
            backward = 0.0
            prepare_forward = 0.0
            total = end_to_end
            state = None
        else:
            state = fn(*args)  # sorting + reordering + graph + preparation forward
            torch.cuda.synchronize(0)
            prepare_elapsed = (time.perf_counter_ns() - full_start) / 1000.0
            construction = graph_us()
            sorting = graph_timing.argsort_us() if graph_timing is not None else 0.0
            preprocessing = sorting + construction
            prepare_forward = max(0.0, prepare_elapsed - preprocessing)
            forward = graph_timing.forward_us() if graph_timing is not None else prepare_forward
            backward_start = time.perf_counter_ns()
            if graph_timing is not None:
                result, backward = graph_timing.backward(state, variables, cotangent)
            else:
                result = torch.autograd.grad(state, variables, grad_outputs=cotangent,
                                             allow_unused=True, materialize_grads=True)
            torch.cuda.synchronize(0)
            end_ns = time.perf_counter_ns()
            operator = (end_ns - backward_start) / 1000.0
            if graph_timing is None:
                backward = operator
            total = preprocessing + operator
            end_to_end = (end_ns - full_start) / 1000.0
        # These intervals are disjoint. The residual includes argsort, floating
        # edge indexing, index backward, and host gaps between timed stages.
        prepare = max(0.0, end_to_end - construction - forward - backward)
        del state, result
        if graph_timing is not None:
            graph_timing.operator_inputs = None
        torch.cuda.synchronize(0)
        return construction, sorting, operator, total, prepare_forward, end_to_end, forward, backward, prepare

    step()  # Initialize execution paths, excluded from samples.
    setup_seconds = time.perf_counter() - t0
    for _ in range(a.warmup):
        step()
    samples = [step() for _ in range(a.iterations)]
    graphs, sorts, operators, totals, prepares, full, forwards, backwards, preparation = zip(*samples)
    median = lambda values: float(np.median(values))
    return {"median_us": median(totals),
            "p10_us": float(np.percentile(totals, 10)),
            "p90_us": float(np.percentile(totals, 90)),
            "argsort_us": median(sorts),
            "create_graph_us": median(graphs),
            "forward_us": median(forwards),
            "backward_us": median(backwards),
            "prepare_us": median(preparation),
            "graph_preprocess_us": median(graphs),
            "operator_us": median(operators), "total_us": median(totals),
            "backward_prepare_forward_us": median(prepares),
            "end_to_end_us": median(full),
            "setup_s": setup_seconds, "execution": "torch_eager"}

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
    timer = getattr(track["fn"], "graph_timing", None)
    if timer is not None:
        # Verify the timed two-stage VJP against the same independent reference.
        # Use a fresh forward because the preceding VJP has freed its graph.
        split_out = track["fn"](*values)
        split_grads, _ = timer.backward(split_out, floats, cotangent)
        for grad, ref_grad in zip(split_grads, expected_grads):
            torch.testing.assert_close(grad, ref_grad, rtol=rtol, atol=atol)
        print("PASS split convolution/index backward gradients against reference", flush=True)
    torch.cuda.synchronize(0)
    print("PASS TP output and gradients (x/y/w) against reference", flush=True)


def attempt_case(track, rows, atoms, a, phase):
    """Allocation, compilation, warmup and timing are all covered by OOM skip."""
    try:
        args = track["inputs"](rows, np.random.default_rng(a.seed), atoms)
        if track["name"].startswith(("cwtp", "sevennet_uuu")):
            out_rows = atoms if track["meta"]["index_mode"] in ("output", "both") else rows
            out_shape = (out_rows, track["meta"]["out_dim"] // track["meta"]["channels"], track["meta"]["channels"])
            print(f"x shape:{args[0].shape}, y shape:{args[1].shape}, w shape:{args[2].shape}, "
                  f"out shape:torch.Size({list(out_shape)})", flush=True)
            if len(args) == 6:
                print(f"input_index shape:{args[3].shape}, output_index shape:{args[4].shape}", flush=True)
            if a.check_indexing and track["meta"]["index_mode"] != "none":
                check_indexing(track, args, a)
        if a.check_backends and track["meta"].get("backend") == "luna":
            check_indexing(dict(track, reference=track["cue_reference"]), args, a)
        torch.cuda.synchronize(0)
        result = measure(track["fn"], args, a, phase)
        return "ok", "", result
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
    imports(a)
    packages = ["numpy", "torch", "cuequivariance", "cuequivariance-torch",
                "LunaEquivariance", "e3nn", "fasteq",
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
        "platform": platform.platform(), "backends": a.backends,
        "device": {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda},
        "timing": "synchronized_host_operator_latency",
        "luna_graph_mode": a.luna_graph_mode,
        "luna_graph_timing_instrumented": True,
        "luna_forward_timing": "argsort_plus_create_graph_plus_convolution" if a.luna_graph_mode == "per_iteration" else "convolution_with_reused_graph_and_sorted_inputs",
        "repeated_graph_inputs": "fixed original unsorted indices; GPU argsort each fused Luna call in per_iteration mode; no neighbor search",
        "argsort_timing": "stable torch.argsort on GPU; synchronized host latency; fused Luna per_iteration only",
        "indexing_timing": "src[order]/dst[order] included in create_graph_us; y[order]/w[order] included in forward operator_us",
        "backward_timing": "fused Luna: two-stage VJP, convolution boundary then external edge permutation; end_to_end includes both calls and their synchronization overhead",
        "end_to_end_breakdown": {
            "create_graph_us": "graph expression including src/dst integer indexing",
            "forward_us": "fused Luna convolution call only, excluding external floating gathers",
            "backward_us": "fused Luna convolution VJP only, including internal gathers/reductions",
            "prepare_us": "end_to_end minus graph/forward/backward per step; includes argsort, external floating gathers, index backward and residual host overhead",
            "argsort_us": "subset of prepare_us; do not add again",
            "non_fused_scope": "complete fn and VJP boundaries; no separation of internal indexing",
            "statistics": "per-step additive; independent CSV medians need not add exactly",
            "instrumentation": "stage synchronization and two autograd engine calls add overhead; compare consistently instrumented runs",
        },
        "backward_edge_reorder_gradients": "included in legacy operator_us and new prepare_us; excluded from new backward_us; gradients returned in original unsorted edge order",
        "graph_preprocessing_timing": {
            "csv_field": "graph_preprocess_us",
            "scope": "Luna fused conv.create_graph(src[order], dst[order], num_nodes), including index permutation; synchronized host latency",
            "sampling": "median of measured iterations; setup and warmup excluded",
            "first_use_initialization_included": False,
            "non_fused_value": 0,
            "aliases": ["create_graph_us", "graph_preprocess_us"],
        },
        "presets": PRESETS,
        "sevennet_configs": [sevennet_plan(layer) for layer in range(5)] if "uuu" in a.operators else [],
        "cwtp_configs": {"sh_lmax": CWTP_SH_LMAX, "layers": CWTP_LAYERS,
                         "output_blocks": CWTP_OUTPUT_BLOCKS},
        "stc_config_source": "inferred_from_model_width_and_hidden_l_not_checkpoint_verified",
        "framework": "PyTorch eager",
        "edge_order": "fused Luna per_iteration receives random edges and sorts in timer; cuEq/FastEq and explicit Luna scatter cases always sort in setup, independently of selected backends",
        "luna_source_reviewed": LUNA_SOURCE_REF,
        "luna_source_export": str(Path(a.luna_dump_dir).expanduser().resolve())
                              if "luna" in a.backends and a.luna_dump_dir else None,
        "preprocessing_excluded": ["initial_transfers", "layout_conversion", "neighbor_search"],
        "non_fused_and_once_preprocessing": "setup only; argsort_us/create_graph_us are zero",
        "graph_construction_excluded": "operator_us only; included in total_us for per_iteration fused Luna",
        "total_us_semantics": "per-step argsort+graph+operator, then median; backward excludes preparation forward",
        "end_to_end_us_semantics": "contiguous interval, includes sorting, edge permutation, graph and backward preparation forward when applicable",
        "forward_operator_us_semantics": "synchronized forward total minus argsort and graph intervals; includes floating edge indexing",
        "comparison": "native_layout_per_backend; explicit indexing operations included when used",
    }
    print(json.dumps(environment, indent=2))
    rows = []
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    #out.with_suffix(".metadata.json").write_text(json.dumps(environment, indent=2))
    for model in a.models:
        model_layers = a.layers if a.layers is not None else (range(5) if model == SEVENNET_MODEL else [0, 1])
        for layer in model_layers:
            for operator in a.operators:
                if (model == SEVENNET_MODEL) != (operator == "uuu"):
                    continue
                if model != SEVENNET_MODEL and layer not in (0, 1):
                    continue
                channels, hidden_l = ((sevennet_plan(layer)['u'], 3) if operator == 'uuu' else PRESETS[model])
                if operator == "cwtp" and layer not in CWTP_LAYERS[model]:
                    print(f"SKIP {model} CWTP layer {layer}: no configuration supplied.", flush=True)
                    continue
                for index_mode in a.index_modes if operator in ("cwtp", "uuu") else ["none"]:
                    label = ("cwtp" if index_mode == "none" else f"cwtp_index_{index_mode}") if operator == "cwtp" else "stc_native"
                    if operator == "uuu":
                        label = 'sevennet_uuu' if index_mode == 'none' else f'sevennet_uuu_index_{index_mode}'
                    for backend in a.backends:
                        track, setup_error, setup_status = None, "", "ok"
                        try:
                            if operator == "uuu":
                                track = make_sevennet_uuu(a, layer, index_mode, backend)
                            elif operator == "cwtp":
                                track = make_cwtp(a, channels, hidden_l, layer, index_mode, backend)
                            else:
                                track = make_stc(a, channels, hidden_l, layer, backend)
                        except NotImplementedError as exc:
                            setup_status, setup_error = "unsupported", str(exc)
                        except Exception as exc:
                            if not is_oom(exc):
                                raise
                            setup_status, setup_error = "oom", oom_message(exc)
                        for phase in a.phases:
                            #from torch.profiler import profile, record_function, ProfilerActivity
                            #activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                            #with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
                            for atoms in a.atoms:
                                n = (a.edges if a.edges is not None else atoms*a.neighbors) if operator in ("cwtp", "uuu") else atoms
                                name = f"{model}_l{layer}_{label}_{phase}_{atoms}_{backend}"
                                if setup_error:
                                    status, error, result = setup_status, setup_error, {}
                                else:
                                    status, error, result = attempt_case(track, n, atoms, a, phase)
                                row = {"model": model, "layer": layer, "operator": label, "phase": phase,
                                    "backend": backend, "atoms": atoms, "rows": n, "channels": channels,
                                    "index_mode": index_mode if operator in ("cwtp", "uuu") else "species",
                                    "status": status, "error": error, "dtype": a.dtype,
                                    "config": json.dumps(track["meta"], sort_keys=True) if track else "",
                                    "median_us": "", "graph_preprocess_us": "",
                                    "argsort_us": "",
                                    "create_graph_us": "", "forward_us": "", "backward_us": "", "prepare_us": "",
                                    "operator_us": "", "total_us": "",
                                    "backward_prepare_forward_us": "", "end_to_end_us": "",
                                    "luna_graph_mode": track["meta"].get("graph_mode", "not_applicable") if track else "not_applicable",
                                    "graph_in_median": bool(track and track["meta"].get("graph_construction_in_forward_timing", False))}
                                row.update(result)
                                rows.append(row)
                                write_rows(out, rows)
                                if status != "ok":
                                    print(f"SKIP {status.upper()} {name}: {error}", flush=True)
                                else:
                                    print(f"{name}: total={result['total_us']:.2f} us, "
                                        f"argsort={result['argsort_us']:.2f} us, "
                                        f"create_graph={result['create_graph_us']:.2f} us, "
                                        f"forward={result['forward_us']:.2f} us, "
                                        f"backward={result['backward_us']:.2f} us, "
                                        f"prepare={result['prepare_us']:.2f} us, "
                                        f"{phase}={result['operator_us']:.2f} us", flush=True)
                                cleanup()
                                #status = cupti.cuptiActivityFlushAll(1)
                                #print("CUPTI flush status:", status)  # 0 表示 API 调用成功
                            
                            #prof.export_chrome_trace("fasteq-triton-trace.json")
                        track = None
                        cleanup()
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()


