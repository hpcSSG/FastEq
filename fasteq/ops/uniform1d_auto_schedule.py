from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Union, Iterable, Set, Sequence
from time import perf_counter
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import torch


# =============================================================================
# STC LARS scheduler and CUDA code generation
# Moved from stc_uniform1d_jit.py so the STC runtime wrapper only keeps
# forward/backward dispatch and file-based JIT loading.
# =============================================================================

Label = Tuple[str, int]  # ('x0', i), ('x1', j), ('o', v)
STC_PAD_VALUE = -1



# =============================================================================
# Unified schedule statistics: Uniform1D/STC forward and backward
# =============================================================================
#
# Per-label metrics:
#   count    : number of scheduled paths that reference the label.  Repeated
#              occurrences of the same label inside one path are counted once,
#              matching one live value/register for that path.
#   lifetime : inclusive scheduled-path span from the label's first use to its
#              last use: last_use - first_use + 1.
#   max_live : maximum number of simultaneously-live labels while this label is
#              live.  All input and accumulator labels participate.
#
# The report is intentionally independent of LARSPlacementConfig and is emitted
# from the original path order selected by the scheduler, before placement
# rewrites the instruction stream.
_SCHEDULE_STATS_DUMPED: Set[Tuple[str, str, str]] = set()
_DEFAULT_SCHEDULE_STATS_ENABLED = False


def _schedule_stats_enabled() -> bool:
    raw = os.environ.get(
        "FASTEQ_SCHEDULE_STATS",
        os.environ.get("FASTEQ_UNIFORM1D_FWD_SCHEDULE_STATS"),
    )
    if raw is None:
        return bool(_DEFAULT_SCHEDULE_STATS_ENABLED)
    return str(raw).strip() not in (
        "0", "false", "False", "OFF", "off", "no", "No"
    )


def _dump_schedule_stats_dir() -> Path:
    """Return the common dump directory for all four scheduler directions."""
    raw = os.environ.get("FASTEQ_SCHEDULE_STATS_DIR", "").strip()
    if not raw:
        # Backward compatibility with the old Uniform1D-forward-only switch.
        raw = os.environ.get(
            "FASTEQ_UNIFORM1D_FWD_SCHEDULE_STATS_DIR", ""
        ).strip()

    if raw:
        path = Path(raw).expanduser().resolve()
    else:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
        cache_root = (
            Path(xdg_cache_home).expanduser()
            if xdg_cache_home
            else Path.home() / ".cache"
        ) / "fasteq"
        path = cache_root / "schedule_stats"

    path.mkdir(parents=True, exist_ok=True)
    return path


def _schedule_stats_safe_tag(value: Any) -> str:
    tag = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_.-")
    return (tag or "schedule")[:96]


def _schedule_stats_sha1(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _build_schedule_label_stats(
    *,
    path_order: Sequence[int],
    labels_by_path: Sequence[Sequence[str]],
) -> List[Dict[str, Any]]:
    """Return label/count/lifetime/max_live rows for one scheduled path stream."""
    path_count = len(labels_by_path)
    order = [int(pid) for pid in path_order]
    if len(order) != path_count:
        raise RuntimeError(
            f"schedule path_order has {len(order)} entries, expected {path_count}"
        )
    if sorted(order) != list(range(path_count)):
        raise RuntimeError(
            "schedule path_order must contain every dense path id exactly once"
        )

    positions: Dict[str, List[int]] = defaultdict(list)
    for scheduled_pos, pid in enumerate(order):
        # A label is one live state even if STC uses it multiple times in the
        # same product, e.g. x1[i] * x1[i].
        seen: Set[str] = set()
        for label in labels_by_path[pid]:
            label = str(label)
            if label in seen:
                continue
            seen.add(label)
            positions[label].append(int(scheduled_pos))

    if not positions:
        return []

    intervals: Dict[str, Tuple[int, int]] = {
        label: (int(pos[0]), int(pos[-1]))
        for label, pos in positions.items()
    }

    # Number of all live labels at each scheduled path position.
    diff = [0] * (path_count + 1)
    for first, last in intervals.values():
        diff[first] += 1
        if last + 1 < len(diff):
            diff[last + 1] -= 1

    live_by_pos: List[int] = []
    live = 0
    for pos in range(path_count):
        live += diff[pos]
        live_by_pos.append(int(live))

    def label_sort_key(label: str) -> Tuple[str, int, str]:
        match = re.match(r"^([^\[]+)\[(-?\d+)\]$", label)
        if match:
            return (match.group(1), int(match.group(2)), label)
        return (label, -1, label)

    rows: List[Dict[str, Any]] = []
    for label in sorted(positions, key=label_sort_key):
        pos_list = positions[label]
        first, last = intervals[label]
        lifetime = int(last - first + 1)
        count = int(len(pos_list))
        rows.append({
            "label": label,
            "count": count,
            "lifetime": lifetime,
            "max_live": int(max(live_by_pos[first:last + 1], default=0)),
        })
    return rows


def _dump_and_print_schedule_label_stats(
    *,
    operation: str,
    candidate_tag: str,
    path_order: Sequence[int],
    labels_by_path: Sequence[Sequence[str]],
) -> None:
    """Dump compact per-label schedule statistics as JSON and CSV."""
    if not _schedule_stats_enabled():
        return

    operation = _schedule_stats_safe_tag(operation).lower()
    candidate = _schedule_stats_safe_tag(candidate_tag)
    order = [int(pid) for pid in path_order]
    labels_normalized = [
        [str(label) for label in labels] for labels in labels_by_path
    ]
    signature = _schedule_stats_sha1({
        "operation": operation,
        "candidate": candidate,
        "path_order": order,
        "labels_by_path": labels_normalized,
    })
    dump_key = (operation, candidate, signature)
    if dump_key in _SCHEDULE_STATS_DUMPED:
        return

    rows = _build_schedule_label_stats(
        path_order=order,
        labels_by_path=labels_normalized,
    )
    payload = {
        "operation": operation,
        "candidate": candidate,
        "path_count": len(order),
        "definitions": {
            "count": "number of scheduled paths referencing the label",
            "lifetime": (
                "last scheduled use - first scheduled use + 1; "
                "scheduled positions are zero-based"
            ),
            "max_live": (
                "maximum number of simultaneously-live input/accumulator labels "
                "during this label's live interval"
            ),
        },
        "variables": rows,
    }

    dump_dir = _dump_schedule_stats_dir()
    stem = f"{operation}_{candidate}_{signature}"
    json_path = dump_dir / f"{stem}.json"
    csv_path = dump_dir / f"{stem}.csv"

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label", "count", "lifetime", "max_live"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[ScheduleStats][{operation}] candidate={candidate} "
        f"paths={len(order)} labels={len(rows)}",
        flush=True,
    )
    print(f"[ScheduleStats][{operation}] JSON: {json_path}", flush=True)
    print(f"[ScheduleStats][{operation}] CSV : {csv_path}", flush=True)
    print(
        f"[ScheduleStats][{operation}] "
        f"{'label':>20s} {'count':>8s} {'lifetime':>10s} "
        f"{'max_live':>10s}",
        flush=True,
    )
    for row in rows:
        print(
            f"[ScheduleStats][{operation}] "
            f"{row['label']:>20s} {row['count']:8d} "
            f"{row['lifetime']:10d} {row['max_live']:10d} ",
            flush=True,
        )

    _SCHEDULE_STATS_DUMPED.add(dump_key)


def _stats_labels_stc_fwd(paths: Sequence[Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    for path in paths:
        labels = [f"x1[{int(idx)}]" for idx in path.x1_indices]
        labels.extend([
            f"x0[{int(path.x0_index)}]",
            f"out[{int(path.v)}]",
        ])
        rows.append(labels)
    return rows


def _stats_labels_stc_bwd(
    paths: Sequence[Any], *, need_grad_x0: bool = True
) -> List[List[str]]:
    rows: List[List[str]] = []
    for path in paths:
        labels = [
            f"grad_out[{int(path.v)}]",
            f"x0[{int(path.x0_index)}]",
        ]
        labels.extend(f"x1[{int(idx)}]" for idx in path.x1_indices)
        labels.extend(f"grad_x1[{int(idx)}]" for idx in path.x1_indices)
        if need_grad_x0:
            labels.append(f"grad_x0[{int(path.x0_index)}]")
        rows.append(labels)
    return rows


def _stats_labels_uniform1d_fwd(paths: Sequence[Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    for path in paths:
        rows.append([
            f"x[{int(path.i)}]",
            f"y[{int(path.j)}]",
            f"w[{int(path.k)}]",
            f"out[{int(path.v)}]",
        ])
    return rows


def _stats_labels_uniform1d_bwd_fused(
    paths: Sequence[Any],
    *,
    need_grad_w: bool,
) -> List[List[str]]:
    rows: List[List[str]] = []
    for path in paths:
        labels = [
            f"w[{int(path.i)}]",
            f"x[{int(path.j)}]",
            f"y[{int(path.k)}]",
            f"grad_out[{int(path.v)}]",
        ]
        if need_grad_w:
            labels.append(f"grad_w[{int(path.i)}]")
        labels.extend([
            f"grad_x[{int(path.j)}]",
            f"grad_y[{int(path.k)}]",
        ])
        rows.append(labels)
    return rows


def _stats_labels_uniform1d_bwd_split(
    paths: Sequence[Any],
    *,
    grad_kind: str,
) -> List[List[str]]:
    target_name = {
        "gw": "grad_w",
        "gx": "grad_x",
        "gy": "grad_y",
    }[str(grad_kind)]

    rows: List[List[str]] = []
    for path in paths:
        labels: List[str] = []
        for kind, idx in path.labels:
            name = "grad_out" if str(kind) == "go" else str(kind)
            labels.append(f"{name}[{int(idx)}]")
        labels.append(f"{target_name}[{int(path.target_index)}]")
        rows.append(labels)
    return rows




def _to_int_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]


def _to_float_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]


@dataclass
class Inst:
    op: str
    args: Tuple[Any, ...]
    comment: str = ""


@dataclass
class ScheduleResult:
    instructions: List[Inst]
    path_order: List[int]
    max_live: int
    spills: int
    reloads: int
    final_reg_map: Dict[Label, str]
    profile: Dict[str, Any] = field(default_factory=dict)
    variable_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    placement_summary: Dict[str, Any] = field(default_factory=dict)
    shared_slots: int = 0


@dataclass(frozen=True)
class STCPath:
    """One STC contraction path.

    Padded STC metadata uses baseline-compatible row semantics:

        len == 3: [x1_a,                 x0_d, out_v, pad, ...]
        len == 4: [x1_a, x1_b,          x0_d, out_v, pad, ...]
        len == 5: [x1_a, x1_b, x1_c,   x0_d, out_v, pad, ...]

    Only the first ``path_lens[p]`` entries are semantic; right-side padding is
    ignored.  The generated expression is therefore:

        out[v] += c * prod_i x1[x1_indices[i]] * x0[x0_index]

    Duplicate indices are intentionally preserved in ``product_labels`` so that
    x1[i] * x1[i] is emitted as a square/cube instead of being collapsed.
    """

    pid: int
    x1_indices: Tuple[int, ...]
    x0_index: int
    v: int
    c: float

    @property
    def product_labels(self) -> Tuple[Label, ...]:
        labels: List[Label] = [("x1", int(i)) for i in self.x1_indices]
        labels.append(("x0", int(self.x0_index)))
        return tuple(labels)

    @property
    def labels(self) -> Tuple[Label, ...]:
        # Remove duplicates only for liveness/register allocation.  The product
        # expression still uses product_labels, so x1[i] * x1[i] remains squared.
        seen: Set[Label] = set()
        out: List[Label] = []
        for lab in self.product_labels:
            if lab not in seen:
                seen.add(lab)
                out.append(lab)
        return tuple(out)


def _lars_reg_id(reg: str) -> int:
    if not isinstance(reg, str) or not reg.startswith("r"):
        raise ValueError(f"Bad register name: {reg!r}")
    return int(reg[1:])


def _max_reg_count_any(schedule_result: ScheduleResult) -> int:
    max_id = -1

    def scan(arg: Any) -> None:
        nonlocal max_id
        if isinstance(arg, str) and arg.startswith("r"):
            max_id = max(max_id, _lars_reg_id(arg))
        elif isinstance(arg, (tuple, list)):
            for a in arg:
                scan(a)

    for inst in schedule_result.instructions:
        for arg in inst.args:
            scan(arg)
    return max_id + 1


def _parse_label_ref(ref: str) -> Label:
    if not isinstance(ref, str) or "[" not in ref or not ref.endswith("]"):
        raise ValueError(f"Bad label ref: {ref!r}")
    kind, rest = ref.split("[", 1)
    idx = int(rest[:-1])
    if kind == "out":
        kind = "o"
    if kind not in ("x0", "x1", "o"):
        raise ValueError(f"Bad STC label kind: {kind!r}")
    return kind, idx


def _parse_stc_placement_ref(ref: str) -> Tuple[str, int]:
    """Parse STC placement refs used by forward and full x1/x0 backward."""
    if not isinstance(ref, str) or "[" not in ref or not ref.endswith("]"):
        raise ValueError(f"Bad STC placement ref: {ref!r}")
    kind, rest = ref.split("[", 1)
    idx = int(rest[:-1])
    aliases = {
        "out": "o",
        "grad_out": "go",
        "grad_x1": "gx1",
        "grad_x0": "gx0",
    }
    kind = aliases.get(kind, kind)
    if kind not in ("x0", "x1", "o", "go", "gx1", "gx0"):
        raise ValueError(f"Bad STC placement kind: {kind!r}")
    return kind, idx


def _fmt_float(x: float) -> str:
    return repr(float(x))


def _sanitize_cuda_comment(s: str) -> str:
    return str(s).replace("\n", " ").replace("\r", " ").replace("*/", "* /")


def _product_expr(regs: Sequence[str], coeff: float) -> str:
    if not regs:
        return f"scalar_t({_fmt_float(coeff)})"
    expr = str(regs[0])
    for r in regs[1:]:
        expr = f"({expr} * {r})"
    return f"scalar_t({_fmt_float(coeff)}) * {expr}"


def _index_expr(kind: str, idx: int, *, u_dim: Optional[int], x0_dim: Optional[int], x1_dim: Optional[int], v_dim: Optional[int]) -> str:
    if kind == "x0":
        if x0_dim is not None and idx >= x0_dim:
            raise ValueError(f"x0 index {idx} out of x0_dim={x0_dim}")
        if u_dim is not None:
            return f"x0_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"x0_base + (index_t){idx} * (index_t)U + (index_t)u"
    if kind == "x1":
        if x1_dim is not None and idx >= x1_dim:
            raise ValueError(f"x1 index {idx} out of x1_dim={x1_dim}")
        if u_dim is not None:
            return f"x1_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"x1_base + (index_t){idx} * (index_t)U + (index_t)u"
    if kind == "o":
        if v_dim is not None and idx >= v_dim:
            raise ValueError(f"out index {idx} out of v_dim={v_dim}")
        if u_dim is not None:
            return f"out_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"out_base + (index_t){idx} * (index_t)U + (index_t)u"
    raise ValueError(f"Bad kind: {kind}")


def emit_stc_fwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    u_dim: Optional[int] = None,
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """Emit STC forward code from either a legacy or lifetime-placed schedule."""
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")

    reg_count = _max_reg_count_any(schedule_result)
    shared_slots = int(getattr(schedule_result, "shared_slots", 0) or 0)
    has_placed = any(
        inst.op == "mul_stc_placed" for inst in schedule_result.instructions
    )

    resident_out_indices = [] if has_placed else sorted({
        int(inst.args[0])
        for inst in schedule_result.instructions
        if inst.op == "mul_stc_resident"
    })
    direct_out_indices = [] if has_placed else sorted({
        int(inst.args[0])
        for inst in schedule_result.instructions
        if inst.op == "mul_stc_direct"
    })

    lines: List[str] = []

    def ap(line: str = "") -> None:
        lines.append(line)

    def shared_expr(token: str) -> str:
        if not isinstance(token, str) or not token.startswith("s"):
            raise ValueError(f"Bad STC shared token: {token!r}")
        slot = int(token[1:])
        return (
            f"lars_smem[(size_t){slot} * (size_t)blockDim.x + "
            f"(size_t)tid]"
        )

    def direct_input_expr(ref: str) -> str:
        kind, idx = _parse_stc_placement_ref(ref)
        if kind not in ("x0", "x1"):
            raise ValueError(f"STC forward direct input expects x0/x1, got {ref!r}")
        expr = _index_expr(
            kind, idx,
            u_dim=u_dim,
            x0_dim=x0_dim,
            x1_dim=x1_dim,
            v_dim=v_dim,
        )
        return f"{kind}[{expr}]"

    def operand_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        if token.startswith("d:"):
            return direct_input_expr(token[2:])
        raise ValueError(f"Bad STC forward operand token: {token!r}")

    def acc_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        raise ValueError(f"Bad STC forward accumulator token: {token!r}")

    def emit_out_write(idx: int, value_expr: str) -> None:
        expr = _index_expr(
            "o", int(idx),
            u_dim=u_dim,
            x0_dim=x0_dim,
            x1_dim=x1_dim,
            v_dim=v_dim,
        )
        ap(f"            out[{expr}] += {value_expr};")

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ x1,")
    ap("    const scalar_t* __restrict__ x0,")
    ap("    scalar_t* __restrict__ out,")
    ap("    int B, int X1, int X0, int V, int U)")
    ap("{")
    ap("    const int b = (int)blockIdx.x;")
    ap("    if (b >= B) return;")
    ap("    const int tid = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    if shared_slots:
        ap("    extern __shared__ unsigned char lars_smem_raw[];")
        ap("    scalar_t* lars_smem = reinterpret_cast<scalar_t*>(lars_smem_raw);")
    ap("    const index_t x1_base = (index_t)b * (index_t)X1 * (index_t)U;")
    ap("    const index_t x0_base = (index_t)b * (index_t)X0 * (index_t)U;")
    ap("    const index_t out_base = (index_t)b * (index_t)V * (index_t)U;")
    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    for out_idx in resident_out_indices:
        ap(f"            scalar_t out_acc_v_{out_idx} = scalar_t(0);")
    if resident_out_indices:
        ap("")
    if direct_out_indices:
        ap(
            f"            // direct single-use output writeback enabled for "
            f"{len(direct_out_indices)} out accumulator(s)"
        )
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op in ("load", "load_shared"):
            token, ref = inst.args
            kind, idx = _parse_stc_placement_ref(str(ref))
            if kind not in ("x0", "x1"):
                raise ValueError("STC forward input load must target x0/x1")
            expr = _index_expr(
                kind, idx,
                u_dim=u_dim,
                x0_dim=x0_dim,
                x1_dim=x1_dim,
                v_dim=v_dim,
            )
            lhs = str(token) if inst.op == "load" else shared_expr(str(token))
            ap(f"            {lhs} = {kind}[{expr}];")

        elif inst.op == "init_acc":
            token, ref = inst.args
            kind, _idx = _parse_stc_placement_ref(str(ref))
            if kind != "o":
                raise ValueError("STC forward init_acc expects out[]")
            ap(f"            {acc_expr(str(token))} = scalar_t(0);")

        elif inst.op == "mul_stc_placed":
            out_idx, out_token, operand_tokens, coeff = inst.args
            product = _product_expr(
                tuple(operand_expr(str(tok)) for tok in operand_tokens),
                float(coeff),
            )
            if str(out_token).startswith("d:"):
                emit_out_write(int(out_idx), product)
            else:
                ap(f"            {acc_expr(str(out_token))} += {product};")

        elif inst.op == "store_acc_placed":
            ref, token = inst.args
            kind, idx = _parse_stc_placement_ref(str(ref))
            if kind != "o":
                raise ValueError("STC forward store_acc_placed expects out[]")
            emit_out_write(int(idx), acc_expr(str(token)))

        elif inst.op == "mul_stc_resident":
            out_idx, regs, coeff = inst.args
            ap(
                f"            out_acc_v_{int(out_idx)} += "
                f"{_product_expr(tuple(regs), float(coeff))};"
            )

        elif inst.op == "mul_stc_direct":
            out_idx, regs, coeff = inst.args
            emit_out_write(
                int(out_idx), _product_expr(tuple(regs), float(coeff))
            )

        elif inst.op in ("release", "release_shared"):
            pass
        else:
            raise ValueError(f"Unsupported STC forward instruction op: {inst.op}")

    if resident_out_indices:
        ap("")
        ap("            // resident output accumulator writeback")
    for out_idx in resident_out_indices:
        emit_out_write(int(out_idx), f"out_acc_v_{out_idx}")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("static inline bool mul3_fits_int32(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32(a, b)) return false;")
    ap("    return mul_fits_int32(a * b, c);")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* x1, const scalar_t* x0, scalar_t* out,")
    ap("    int B, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(
        f"    size_t lars_shared_bytes = (size_t){shared_slots} * "
        f"(size_t)block.x * sizeof(scalar_t);"
    )
    ap(
        f"    {kernel_name}<scalar_t, index_t><<<grid, block, "
        f"lars_shared_bytes, stream>>>(x1, x0, out, B, X1, X0, V, U);"
    )
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* x1, const scalar_t* x0, scalar_t* out,")
    ap("    int B, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap("    bool use_i32 = mul3_fits_int32((int64_t)B, (int64_t)X1, (int64_t)U) &&")
    ap("                   mul3_fits_int32((int64_t)B, (int64_t)X0, (int64_t)U) &&")
    ap("                   mul3_fits_int32((int64_t)B, (int64_t)V,  (int64_t)U);")
    ap("    if (use_i32) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(x1, x0, out, B, X1, X0, V, U, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(x1, x0, out, B, X1, X0, V, U, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap(f"torch::Tensor launcher_{kernel_name}(torch::Tensor x1, torch::Tensor x0, int64_t V64) {{")
    ap("    TORCH_CHECK(x1.is_cuda() && x0.is_cuda(), \"x1/x0 must be CUDA/HIP\");")
    ap("    TORCH_CHECK(x1.is_contiguous() && x0.is_contiguous(), \"x1/x0 must be contiguous\");")
    ap("    TORCH_CHECK(x1.dim() == 3 && x0.dim() == 3, \"x1/x0 must be [B,S,U]\");")
    ap("    TORCH_CHECK(x1.scalar_type() == x0.scalar_type(), \"x1/x0 dtype mismatch\");")
    ap("    int B = (int)x1.size(0);")
    ap("    int X1 = (int)x1.size(1);")
    ap("    int U = (int)x1.size(2);")
    ap("    int X0 = (int)x0.size(1);")
    ap("    int V = (int)V64;")
    ap("    TORCH_CHECK((int)x0.size(0) == B, \"x0 batch mismatch\");")
    ap("    TORCH_CHECK((int)x0.size(2) == U, \"x0 U mismatch\");")
    ap("    TORCH_CHECK(V > 0, \"V must be > 0\");")
    ap("    auto out = torch::zeros({B, V, U}, x1.options());")
    ap("    GPU_Guard device_guard(x1.device());")
    ap("    gpuStream_t stream = getCurrentGPUStream(x1.device().index());")
    ap(f"    AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), \"{kernel_name}\", [&] {{")
    ap(f"        launch_{kernel_name}<scalar_t>((const scalar_t*)x1.data_ptr<scalar_t>(),")
    ap("            (const scalar_t*)x0.data_ptr<scalar_t>(),")
    ap("            (scalar_t*)out.data_ptr<scalar_t>(), B, X1, X0, V, U, stream);")
    ap("    });")
    ap("    GPU_KERNEL_LAUNCH_CHECK();")
    ap("    return out;")
    ap("}")
    ap("")
    ap("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    ap(f"    m.def(\"run\", &launcher_{kernel_name}, \"{kernel_name} STC forward jit impl\");")
    ap("}")

    return "\n".join(lines)


def _normalize_stc_padded_paths(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: Optional[torch.Tensor] = None,
    path_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return padded STC paths as a CPU int64 tensor with shape [max_len, P].

    Accepted inputs:
      * ``idx_lists_tensor`` style: [max_len, P]
      * ``paths_tensor`` style:    [P, max_len]
      * Python sequence of max_len tensors, each [P]

    Without ``path_lens``, the path dimension is inferred from
    ``coeff_list.numel()``.  This allows dropping ``path_lens_tensor`` from the
    public API as long as padding uses a sentinel value such as -1.
    """
    if path_lens is not None:
        P = int(path_lens.detach().cpu().reshape(-1).numel())
    elif coeff_list is not None:
        P = int(coeff_list.detach().cpu().reshape(-1).numel())
    else:
        raise ValueError("Either coeff_list or path_lens is required to infer the path dimension")

    if isinstance(idx_lists, torch.Tensor):
        idx = idx_lists.detach().cpu().to(torch.int64)
    else:
        idx = torch.stack([t.detach().cpu().to(torch.int64).reshape(-1) for t in idx_lists], dim=0)

    if idx.dim() != 2:
        raise ValueError(f"STC padded paths must be 2D, got shape={tuple(idx.shape)}")

    # Already [max_len, P], as in stc_meta['idx_lists_tensor'].
    if idx.shape[1] == P and idx.shape[0] != P:
        return idx.contiguous()

    # Row-major [P, max_len], as in stc_meta['paths'].
    if idx.shape[0] == P and idx.shape[1] != P:
        return idx.t().contiguous()

    # Ambiguous square case. Prefer the STC convention max_len <= 5 when possible.
    if idx.shape[0] == P and idx.shape[1] == P:
        if idx.shape[0] <= 8:
            raise ValueError(
                f"Ambiguous square STC padded path shape={tuple(idx.shape)}. "
                "Pass a non-square [P,max_len]/[max_len,P] tensor or keep path_lens."
            )
        return idx.t().contiguous()

    raise ValueError(
        f"Cannot infer STC path layout from shape={tuple(idx.shape)} and "
        f"P={P}; expected [max_len, P] or [P, max_len]"
    )


def infer_stc_path_lens_from_padded(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    pad_value: int = STC_PAD_VALUE,
) -> torch.Tensor:
    """Infer [P] int64 path lengths from sentinel-padded STC paths.

    Padding must be a suffix and must use ``pad_value``.  Valid indices are
    expected to be non-negative, so -1 is the recommended sentinel.
    """
    idx = _normalize_stc_padded_paths(idx_lists, coeff_list=coeff_list)
    valid = idx.ne(int(pad_value))
    lens = valid.to(torch.int64).sum(dim=0)
    max_len, P = idx.shape

    for p in range(P):
        L = int(lens[p].item())
        if L < 3 or L > max_len:
            raise ValueError(
                f"Bad inferred STC path length {L} for path {p}; "
                f"expected 3..{max_len}. Did you use pad_value={pad_value}?"
            )
        # Enforce suffix padding: valid entries must be exactly [:L].
        if not bool(valid[:L, p].all().item()) or bool(valid[L:, p].any().item()):
            vals = [int(v) for v in idx[:, p].tolist()]
            raise ValueError(
                f"Non-suffix padding in STC path {p}: {vals}. "
                f"Use valid entries first, then pad with {pad_value}."
            )
    return lens.contiguous()


def _infer_stc_path_lens_tensor_from_padded(
    padded_paths: torch.Tensor,
    coeff_list: torch.Tensor,
    *,
    pad_value: int = STC_PAD_VALUE,
) -> torch.Tensor:
    """Infer CUDA/CPU [P] int32 path lengths from sentinel-padded paths.

    This is used only to call the existing baseline backward kernel, whose API
    still expects path_lens_tensor.
    """
    if not isinstance(padded_paths, torch.Tensor) or padded_paths.dim() != 2:
        raise ValueError(f"padded_paths must be a 2D tensor, got {type(padded_paths)}")
    P = int(coeff_list.detach().reshape(-1).numel())

    if int(padded_paths.size(0)) == P and int(padded_paths.size(1)) != P:
        lens = padded_paths.ne(int(pad_value)).to(torch.int32).sum(dim=1)
    elif int(padded_paths.size(1)) == P and int(padded_paths.size(0)) != P:
        lens = padded_paths.ne(int(pad_value)).to(torch.int32).sum(dim=0)
    elif int(padded_paths.size(0)) == P and int(padded_paths.size(1)) == P:
        raise ValueError(
            f"Ambiguous square STC padded path shape={tuple(padded_paths.shape)}; "
            "cannot infer path dimension without path_lens_tensor."
        )
    else:
        raise ValueError(
            f"Cannot infer STC path layout from shape={tuple(padded_paths.shape)} and P={P}"
        )

    return lens.to(device=padded_paths.device, dtype=torch.int32).contiguous()


def make_stc_paths_from_padded_lists(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
) -> List[STCPath]:
    """Convert padded STC metadata into scheduler-native paths.

    If ``path_lens`` is omitted, lengths are inferred from sentinel padding.
    Recommended padded format is:

        len == 3: [x1_a,                 x0_d, out_v, -1, -1]
        len == 4: [x1_a, x1_b,          x0_d, out_v, -1]
        len == 5: [x1_a, x1_b, x1_c,   x0_d, out_v]

    Baseline-compatible semantics are:

        x1_indices = idx[0 : L-2, p]
        x0_index   = idx[L-2, p]
        v          = idx[L-1, p]
    """
    idx = _normalize_stc_padded_paths(idx_lists, coeff_list=coeff_list, path_lens=path_lens)
    if path_lens is None:
        lens = infer_stc_path_lens_from_padded(idx, coeff_list, pad_value=pad_value)
    else:
        lens = path_lens.detach().cpu().to(torch.int64).reshape(-1).contiguous()

    coeff = coeff_list.detach().cpu().reshape(-1)
    max_len, P = idx.shape

    if lens.numel() != P:
        raise ValueError(f"path_lens numel {lens.numel()} does not match P={P}")
    if coeff.numel() != P:
        raise ValueError(f"coeff_list numel {coeff.numel()} does not match P={P}")

    paths: List[STCPath] = []
    for p in range(P):
        L = int(lens[p].item())
        if L < 3 or L > max_len:
            raise ValueError(f"Bad STC path length {L} for path {p}; expected 3..{max_len}")

        vals = [int(idx[t, p].item()) for t in range(L)]
        x1_indices = tuple(vals[: L - 2])
        x0_index = vals[L - 2]
        v = vals[L - 1]

        paths.append(
            STCPath(
                pid=p,
                x1_indices=x1_indices,
                x0_index=x0_index,
                v=v,
                c=float(coeff[p].item()),
            )
        )
    return paths


def generate_code_stc_fwd_with_scheduler(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    num_out_segments: int,
    u_dim: int,
    out_path: str = "generated_stc_fwd_lars.cu",
    kernel_name: str = "stc_lars_fwd",
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    return_schedule: bool = False,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    block_size: int = 32,
    placement_config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]] = None,
) -> Any:
    paths = make_stc_paths_from_padded_lists(
        idx_lists, coeff_list, path_lens=path_lens, pad_value=pad_value
    )
    scheduler = LARSUniform1DScheduler(
        paths,
        path_kind="stc",
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name="stc_lars_all_inputs",
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    schedule_result = scheduler.schedule()
    _dump_and_print_schedule_label_stats(
        operation="stc_fwd",
        candidate_tag="stc_lars_all_inputs",
        path_order=schedule_result.path_order,
        labels_by_path=_stats_labels_stc_fwd(paths),
    )
    resolved_placement = _resolve_lars_placement_config(placement_config)
    schedule_result = apply_lars_lifetime_placement(
        schedule_result,
        schedule_kind="stc_fwd",
        config=resolved_placement,
    )
    dump_lars_variable_stats(
        schedule_result,
        print_stats=resolved_placement.stats_print,
        out_path=resolved_placement.stats_path,
    )

    idx_norm = _normalize_stc_padded_paths(
        idx_lists, coeff_list=coeff_list, path_lens=path_lens
    )
    if path_lens is None:
        inferred_lens = infer_stc_path_lens_from_padded(
            idx_norm, coeff_list, pad_value=pad_value
        )
    else:
        inferred_lens = path_lens.detach().cpu().to(torch.int64).reshape(-1)
    base_kernel_name = (
        f"{kernel_name}_u{int(u_dim)}_path{len(paths)}_"
        f"maxlen{int(inferred_lens.max().item())}"
    )
    code = emit_stc_fwd_kernel_from_lars_schedule(
        schedule_result,
        kernel_name=base_kernel_name,
        u_dim=int(u_dim),
        x0_dim=x0_dim,
        x1_dim=x1_dim,
        v_dim=int(num_out_segments),
        block_size=int(block_size),
    )
    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")
    if return_schedule:
        return {
            "name": "stc_lars_all_inputs",
            "code": code,
            "kernel_name": base_kernel_name,
            "schedule": schedule_result,
            "num_paths": len(paths),
            "max_live": schedule_result.max_live,
            "profile": schedule_result.profile,
        }
    return code


def _check_stc_path_bounds_for_bwd(
    paths: Sequence[STCPath],
    *,
    x0_dim: Optional[int],
    x1_dim: Optional[int],
    v_dim: Optional[int],
) -> None:
    for p in paths:
        if x0_dim is not None and int(p.x0_index) >= int(x0_dim):
            raise ValueError(f"x0 index {p.x0_index} out of x0_dim={x0_dim} in STC path {p.pid}")
        if v_dim is not None and int(p.v) >= int(v_dim):
            raise ValueError(f"out index {p.v} out of v_dim={v_dim} in STC path {p.pid}")
        if int(p.x0_index) < 0 or int(p.v) < 0:
            raise ValueError(f"negative x0/out index in STC path {p.pid}: x0={p.x0_index}, out={p.v}")
        for x1_idx in p.x1_indices:
            if int(x1_idx) < 0:
                raise ValueError(f"negative x1 index {x1_idx} in STC path {p.pid}")
            if x1_dim is not None and int(x1_idx) >= int(x1_dim):
                raise ValueError(f"x1 index {x1_idx} out of x1_dim={x1_dim} in STC path {p.pid}")


def _stc_x1_global_expr(x1_idx: int) -> str:
    return f"x1[((index_t)b * (index_t)X1 + (index_t){int(x1_idx)}) * (index_t)U + (index_t)u]"


def _stc_grad_x1_acc_name(x1_idx: int) -> str:
    return f"gx1_acc_a_{int(x1_idx)}"


def _stc_grad_x0_acc_name(x0_idx: int) -> str:
    return f"gx0_acc_d_{int(x0_idx)}"


def _stc_bwd_grad_expr(base_expr: str, other_x1_indices: Sequence[int]) -> str:
    expr = base_expr
    for x1_idx in other_x1_indices:
        expr = f"({expr} * {_stc_x1_global_expr(int(x1_idx))})"
    return expr


def _make_stc_bwd_logical_schedule(
    paths: Sequence[STCPath],
    base_schedule: ScheduleResult,
    *,
    need_grad_x0: bool = True,
) -> ScheduleResult:
    """Convert the STC LARS path order into backward placement records."""
    by_pid = {int(p.pid): p for p in paths}
    instructions: List[Inst] = []
    for pid in base_schedule.path_order:
        p = by_pid[int(pid)]
        instructions.append(
            Inst(
                "stc_bwd_path",
                (
                    tuple(int(v) for v in p.x1_indices),
                    int(p.x0_index),
                    int(p.v),
                    float(p.c),
                    bool(need_grad_x0),
                ),
                f"path#{p.pid}: STC grad_x1",
            )
        )
    return ScheduleResult(
        instructions=instructions,
        path_order=list(base_schedule.path_order),
        max_live=int(base_schedule.max_live),
        spills=int(base_schedule.spills),
        reloads=int(base_schedule.reloads),
        final_reg_map={},
        profile=dict(base_schedule.profile),
    )


def emit_stc_bwd_kernel_from_paths(
    paths: Sequence[STCPath],
    *,
    kernel_name: str,
    path_order: Optional[Sequence[int]] = None,
    schedule_result: Optional[ScheduleResult] = None,
    u_dim: Optional[int] = None,
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    tile_u: int = 32,
    block_size: int = 32,
    need_grad_x0: bool = True,
) -> str:
    """Emit STC backward for x1 and, when requested, gathered x0.

    When ``schedule_result`` is supplied, it may contain ``stc_bwd_placed``
    instructions produced by :func:`apply_lars_lifetime_placement`.  Without a
    placed schedule, the function preserves the previous all-accumulator-local
    implementation.
    """
    if int(tile_u) != 32:
        raise ValueError(f"STC backward currently assumes tile_u=32, got {tile_u}")
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")

    path_list = list(paths)
    by_pid = {int(p.pid): p for p in path_list}
    if schedule_result is None:
        order = list(path_order) if path_order is not None else [int(p.pid) for p in path_list]
        ordered_paths = [by_pid[int(pid)] for pid in order]
        legacy_insts = [
            Inst(
                "stc_bwd_path",
                (
                    tuple(int(v) for v in p.x1_indices),
                    int(p.x0_index),
                    int(p.v),
                    float(p.c),
                    bool(need_grad_x0),
                ),
                f"path#{p.pid}: STC grad_x1",
            )
            for p in ordered_paths
        ]
        schedule_result = ScheduleResult(
            instructions=legacy_insts,
            path_order=order,
            max_live=0,
            spills=0,
            reloads=0,
            final_reg_map={},
        )
    else:
        ordered_paths = [by_pid[int(pid)] for pid in schedule_result.path_order]

    _check_stc_path_bounds_for_bwd(
        ordered_paths, x0_dim=x0_dim, x1_dim=x1_dim, v_dim=v_dim
    )

    has_placed = any(
        inst.op == "stc_bwd_placed" for inst in schedule_result.instructions
    )
    reg_count = _max_reg_count_any(schedule_result)
    shared_slots = int(getattr(schedule_result, "shared_slots", 0) or 0)
    touched_x1_indices = sorted({
        int(idx) for p in ordered_paths for idx in p.x1_indices
    })
    touched_x0_indices = (
        sorted({int(p.x0_index) for p in ordered_paths})
        if need_grad_x0 else []
    )

    lines: List[str] = []

    def ap(line: str = "") -> None:
        lines.append(line)

    def shared_expr(token: str) -> str:
        if not isinstance(token, str) or not token.startswith("s"):
            raise ValueError(f"Bad STC backward shared token: {token!r}")
        slot = int(token[1:])
        return (
            f"lars_smem[(size_t){slot} * (size_t)blockDim.x + "
            f"(size_t)tid]"
        )

    def input_index_expr(kind: str, idx: int) -> str:
        if kind == "x1":
            return (
                f"((index_t)b * (index_t)X1 + (index_t){int(idx)}) * "
                f"(index_t)U + (index_t)u"
            )
        if kind == "x0":
            return (
                f"((index_t)b * (index_t)X0 + (index_t){int(idx)}) * "
                f"(index_t)U + (index_t)u"
            )
        if kind == "go":
            return (
                f"((index_t)b * (index_t)V + (index_t){int(idx)}) * "
                f"(index_t)U + (index_t)u"
            )
        raise ValueError(f"Bad STC backward input kind: {kind}")

    def direct_input_expr(ref: str) -> str:
        kind, idx = _parse_stc_placement_ref(ref)
        if kind not in ("x1", "x0", "go"):
            raise ValueError(f"Bad STC backward direct input ref: {ref!r}")
        arr = "grad_out" if kind == "go" else kind
        return f"{arr}[{input_index_expr(kind, idx)}]"

    def operand_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        if token.startswith("d:"):
            return direct_input_expr(token[2:])
        raise ValueError(f"Bad STC backward operand token: {token!r}")

    def acc_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        raise ValueError(f"Bad STC backward accumulator token: {token!r}")

    def grad_x1_index_expr(idx: int) -> str:
        return (
            f"((index_t)b * (index_t)X1 + (index_t){int(idx)}) * "
            f"(index_t)U + (index_t)u"
        )

    def grad_x0_index_expr(idx: int) -> str:
        return (
            f"((index_t)b * (index_t)X0 + (index_t){int(idx)}) * "
            f"(index_t)U + (index_t)u"
        )

    def emit_grad_x1_write(idx: int, value_expr: str) -> None:
        ap(f"            grad_x1[{grad_x1_index_expr(idx)}] = {value_expr};")

    def emit_grad_x0_write(idx: int, value_expr: str) -> None:
        ap(f"            grad_x0[{grad_x0_index_expr(idx)}] = {value_expr};")

    def product_from_tokens(base_expr: str, tokens: Sequence[str]) -> str:
        expr = str(base_expr)
        for token in tokens:
            expr = f"({expr} * {operand_expr(str(token))})"
        return expr

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")
    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ grad_out,")
    ap("    const scalar_t* __restrict__ x1,")
    ap("    const scalar_t* __restrict__ x0,")
    ap("    scalar_t* __restrict__ grad_x1,")
    ap("    scalar_t* __restrict__ grad_x0,")
    ap("    int B, int X1, int X0, int V, int U)")
    ap("{")
    ap("    const int b = (int)blockIdx.x;")
    ap("    if (b >= B) return;")
    ap("")
    ap("    const int tid = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    if shared_slots:
        ap("    extern __shared__ unsigned char lars_smem_raw[];")
        ap("    scalar_t* lars_smem = reinterpret_cast<scalar_t*>(lars_smem_raw);")
    ap("")
    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    if not has_placed:
        for x1_idx in touched_x1_indices:
            ap(f"            scalar_t {_stc_grad_x1_acc_name(x1_idx)} = scalar_t(0);")
        for x0_idx in touched_x0_indices:
            ap(f"            scalar_t {_stc_grad_x0_acc_name(x0_idx)} = scalar_t(0);")
        if touched_x1_indices or touched_x0_indices:
            ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // bwd inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op in ("load", "load_shared"):
            token, ref = inst.args
            kind, idx = _parse_stc_placement_ref(str(ref))
            if kind not in ("x1", "x0", "go"):
                raise ValueError("STC backward input load expects x1/x0/grad_out")
            arr = "grad_out" if kind == "go" else kind
            lhs = str(token) if inst.op == "load" else shared_expr(str(token))
            ap(f"            {lhs} = {arr}[{input_index_expr(kind, idx)}];")

        elif inst.op == "init_acc":
            token, ref = inst.args
            kind, _idx = _parse_stc_placement_ref(str(ref))
            if kind not in ("gx1", "gx0"):
                raise ValueError("STC backward init_acc expects grad_x1[]/grad_x0[]")
            ap(f"            {acc_expr(str(token))} = scalar_t(0);")

        elif inst.op == "stc_bwd_placed":
            (
                x1_indices,
                target_tokens,
                x0_target_token,
                go_token,
                x0_token,
                x1_tokens,
                coeff,
            ) = inst.args
            x1_indices = tuple(int(v) for v in x1_indices)
            target_tokens = tuple(str(v) for v in target_tokens)
            x1_tokens = tuple(str(v) for v in x1_tokens)
            base_expr = (
                f"scalar_t({_fmt_float(float(coeff))}) * "
                f"{operand_expr(str(go_token))} * {operand_expr(str(x0_token))}"
            )
            for pos, target_idx in enumerate(x1_indices):
                other_tokens = [
                    x1_tokens[q]
                    for q in range(len(x1_indices))
                    if q != pos
                ]
                value = product_from_tokens(base_expr, other_tokens)
                target_token = target_tokens[pos]
                if target_token.startswith("d:"):
                    emit_grad_x1_write(int(target_idx), value)
                else:
                    ap(f"            {acc_expr(target_token)} += {value};")
            if need_grad_x0:
                gx0_value = product_from_tokens(
                    f"scalar_t({_fmt_float(float(coeff))}) * {operand_expr(str(go_token))}",
                    x1_tokens,
                )
                if str(x0_target_token).startswith("d:"):
                    emit_grad_x0_write(int(_parse_stc_placement_ref(str(x0_target_token)[2:])[1]), gx0_value)
                else:
                    ap(f"            {acc_expr(str(x0_target_token))} += {gx0_value};")

        elif inst.op == "store_acc_placed":
            ref, token = inst.args
            kind, idx = _parse_stc_placement_ref(str(ref))
            if kind == "gx1":
                emit_grad_x1_write(int(idx), acc_expr(str(token)))
            elif kind == "gx0":
                emit_grad_x0_write(int(idx), acc_expr(str(token)))
            else:
                raise ValueError("STC backward store_acc_placed expects grad_x1[]/grad_x0[]")

        elif inst.op == "stc_bwd_path":
            x1_indices, x0_idx, out_v, coeff, path_need_grad_x0 = inst.args
            x1_indices = tuple(int(v) for v in x1_indices)
            base_name = f"base_{inst_id}"
            ap(
                f"            const scalar_t {base_name} = "
                f"grad_out[{input_index_expr('go', int(out_v))}] * "
                f"scalar_t({_fmt_float(float(coeff))}) * "
                f"x0[{input_index_expr('x0', int(x0_idx))}];"
            )
            for pos, target_x1 in enumerate(x1_indices):
                other = [idx for q, idx in enumerate(x1_indices) if q != pos]
                grad_expr = _stc_bwd_grad_expr(base_name, other)
                ap(
                    f"            {_stc_grad_x1_acc_name(target_x1)} += "
                    f"{grad_expr};"
                )
            if bool(path_need_grad_x0):
                gx0_expr = _stc_bwd_grad_expr(
                    f"grad_out[{input_index_expr('go', int(out_v))}] * scalar_t({_fmt_float(float(coeff))})",
                    x1_indices,
                )
                ap(
                    f"            {_stc_grad_x0_acc_name(int(x0_idx))} += "
                    f"{gx0_expr};"
                )

        elif inst.op in ("release", "release_shared"):
            pass
        else:
            raise ValueError(f"Unsupported STC backward instruction op: {inst.op}")

    if not has_placed:
        for x1_idx in touched_x1_indices:
            emit_grad_x1_write(
                int(x1_idx), _stc_grad_x1_acc_name(int(x1_idx))
            )
        for x0_idx in touched_x0_indices:
            emit_grad_x0_write(
                int(x0_idx), _stc_grad_x0_acc_name(int(x0_idx))
            )

    ap("        }")
    ap("    }")
    ap("}")
    ap("")
    ap("static inline bool mul_fits_int32(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("static inline bool mul3_fits_int32(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32(a, b)) return false;")
    ap("    return mul_fits_int32(a * b, c);")
    ap("}")
    ap("")
    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, scalar_t* grad_x1, scalar_t* grad_x0,")
    ap("    int B, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(
        f"    size_t lars_shared_bytes = (size_t){shared_slots} * "
        f"(size_t)block.x * sizeof(scalar_t);"
    )
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, lars_shared_bytes, stream>>>(")
    ap("        grad_out, x1, x0, grad_x1, grad_x0, B, X1, X0, V, U);")
    ap("}")
    ap("")
    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, scalar_t* grad_x1, scalar_t* grad_x0,")
    ap("    int B, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap("    bool use_i32 = mul3_fits_int32((int64_t)B, (int64_t)X1, (int64_t)U) &&")
    ap("                   mul3_fits_int32((int64_t)B, (int64_t)X0, (int64_t)U) &&")
    ap("                   mul3_fits_int32((int64_t)B, (int64_t)V,  (int64_t)U);")
    ap("    if (use_i32) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(grad_out, x1, x0, grad_x1, grad_x0, B, X1, X0, V, U, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(grad_out, x1, x0, grad_x1, grad_x0, B, X1, X0, V, U, stream);")
    ap("    }")
    ap("}")
    ap("")
    ap(f"std::vector<torch::Tensor> launcher_{kernel_name}(torch::Tensor grad_out, torch::Tensor x1, torch::Tensor x0, int64_t V64) {{")
    ap("    TORCH_CHECK(grad_out.is_cuda() && x1.is_cuda() && x0.is_cuda(), \"grad_out/x1/x0 must be CUDA/HIP\");")
    ap("    TORCH_CHECK(grad_out.is_contiguous() && x1.is_contiguous() && x0.is_contiguous(), \"grad_out/x1/x0 must be contiguous\");")
    ap("    TORCH_CHECK(x1.dim() == 3 && x0.dim() == 3, \"x1/x0 must be [B,S,U]\");")
    ap("    TORCH_CHECK(grad_out.scalar_type() == x1.scalar_type() && x1.scalar_type() == x0.scalar_type(), \"grad_out/x1/x0 dtype mismatch\");")
    ap("    int B = (int)x1.size(0);")
    ap("    int X1 = (int)x1.size(1);")
    ap("    int U = (int)x1.size(2);")
    ap("    int X0 = (int)x0.size(1);")
    ap("    int V = (int)V64;")
    ap("    TORCH_CHECK((int)x0.size(0) == B, \"x0 batch mismatch\");")
    ap("    TORCH_CHECK((int)x0.size(2) == U, \"x0 U mismatch\");")
    ap("    TORCH_CHECK(V > 0, \"V must be > 0\");")
    ap("    TORCH_CHECK(grad_out.numel() == (int64_t)B * (int64_t)V * (int64_t)U, \"grad_out numel mismatch; expected B*V*U\");")
    if u_dim is not None:
        ap(f"    TORCH_CHECK(U == {int(u_dim)}, \"U mismatch for generated STC backward kernel\");")
    if x0_dim is not None:
        ap(f"    TORCH_CHECK(X0 == {int(x0_dim)}, \"X0 mismatch for generated STC backward kernel\");")
    if x1_dim is not None:
        ap(f"    TORCH_CHECK(X1 == {int(x1_dim)}, \"X1 mismatch for generated STC backward kernel\");")
    if v_dim is not None:
        ap(f"    TORCH_CHECK(V == {int(v_dim)}, \"V mismatch for generated STC backward kernel\");")
    ap("    auto grad_x1 = torch::zeros_like(x1);")
    if need_grad_x0:
        ap("    auto grad_x0 = torch::zeros_like(x0);")
    ap("    GPU_Guard device_guard(x1.device());")
    ap("    gpuStream_t stream = getCurrentGPUStream(x1.device().index());")
    ap(f"    AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), \"{kernel_name}\", [&] {{")
    ap(f"        launch_{kernel_name}<scalar_t>((const scalar_t*)grad_out.data_ptr<scalar_t>(),")
    ap("            (const scalar_t*)x1.data_ptr<scalar_t>(),")
    ap("            (const scalar_t*)x0.data_ptr<scalar_t>(),")
    ap("            (scalar_t*)grad_x1.data_ptr<scalar_t>(),")
    if need_grad_x0:
        ap("            (scalar_t*)grad_x0.data_ptr<scalar_t>(), B, X1, X0, V, U, stream);")
    else:
        ap("            nullptr, B, X1, X0, V, U, stream);")
    ap("    });")
    ap("    GPU_KERNEL_LAUNCH_CHECK();")
    if need_grad_x0:
        ap("    return {grad_x1, grad_x0};")
    else:
        ap("    return {grad_x1};")
    ap("}")
    ap("")
    ap("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    ap(f"    m.def(\"run\", &launcher_{kernel_name}, \"{kernel_name} STC backward x1/x0 jit impl\");")
    ap("}")
    return "\n".join(lines)


def generate_code_stc_bwd_with_scheduler(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    num_out_segments: int,
    u_dim: int,
    out_path: str = "generated_stc_bwd_lars.cu",
    kernel_name: str = "stc_lars_bwd",
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    return_schedule: bool = False,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    tile_u: int = 32,
    block_size: int = 32,
    placement_config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]] = None,
    need_grad_x0: bool = True,
) -> Any:
    """Generate STC backward for x1 and optionally gathered x0."""
    paths = make_stc_paths_from_padded_lists(
        idx_lists, coeff_list, path_lens=path_lens, pad_value=pad_value
    )
    scheduler = LARSUniform1DScheduler(
        paths,
        path_kind="stc",
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name=("stc_lars_bwd_x1_x0" if need_grad_x0 else "stc_lars_bwd_x1_only"),
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    path_schedule = scheduler.schedule()
    _dump_and_print_schedule_label_stats(
        operation="stc_bwd",
        candidate_tag=("stc_lars_bwd_x1_x0" if need_grad_x0 else "stc_lars_bwd_x1_only"),
        path_order=path_schedule.path_order,
        labels_by_path=_stats_labels_stc_bwd(paths, need_grad_x0=need_grad_x0),
    )
    schedule_result = _make_stc_bwd_logical_schedule(
        paths, path_schedule, need_grad_x0=need_grad_x0
    )
    resolved_placement = _resolve_lars_placement_config(placement_config)
    schedule_result = apply_lars_lifetime_placement(
        schedule_result,
        schedule_kind="stc_bwd",
        config=resolved_placement,
    )
    dump_lars_variable_stats(
        schedule_result,
        print_stats=resolved_placement.stats_print,
        out_path=resolved_placement.stats_path,
    )

    idx_norm = _normalize_stc_padded_paths(
        idx_lists, coeff_list=coeff_list, path_lens=path_lens
    )
    if path_lens is None:
        inferred_lens = infer_stc_path_lens_from_padded(
            idx_norm, coeff_list, pad_value=pad_value
        )
    else:
        inferred_lens = path_lens.detach().cpu().to(torch.int64).reshape(-1)
    base_kernel_name = (
        f"{kernel_name}_u{int(u_dim)}_path{len(paths)}_"
        f"maxlen{int(inferred_lens.max().item())}"
    )

    code = emit_stc_bwd_kernel_from_paths(
        paths,
        schedule_result=schedule_result,
        kernel_name=base_kernel_name,
        u_dim=int(u_dim),
        x0_dim=x0_dim,
        x1_dim=x1_dim,
        v_dim=int(num_out_segments),
        tile_u=int(tile_u),
        block_size=int(block_size),
        need_grad_x0=bool(need_grad_x0),
    )
    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")
    if return_schedule:
        return {
            "name": ("stc_lars_bwd_x1_x0" if need_grad_x0 else "stc_lars_bwd_x1_only"),
            "code": code,
            "kernel_name": base_kernel_name,
            "schedule": schedule_result,
            "num_paths": len(paths),
            "max_live": schedule_result.max_live,
            "profile": schedule_result.profile,
        }
    return code


def _to_int_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().tolist()]


def _to_float_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]

@dataclass(frozen=True)
class U1DPath:
    pid: int
    i: int
    j: int
    k: int
    v: int
    c: float

    @property
    def labels(self) -> Tuple[Label, Label, Label]:
        # LARS schedules only input operands.  Output accumulators are emitted
        # as full-resident local variables and are not reload-managed candidates.
        return (("x", self.i), ("y", self.j), ("w", self.k))

    @property
    def product_labels(self) -> Tuple[Label, Label, Label]:
        # Uniform1D has no duplicate-input multiplicity beyond x/y/w.  This
        # property lets the common scheduler handle both fixed-arity Uniform1D
        # and variable-arity STC paths through the same emission path.
        return self.labels


@dataclass
class Inst:
    op: str
    args: Tuple[Any, ...]
    comment: str = ""

    def __str__(self) -> str:
        body = f"{self.op} " + ", ".join(map(str, self.args))
        return f"{body:<48} # {self.comment}" if self.comment else body


@dataclass
class ScheduleResult:
    instructions: List[Inst]
    path_order: List[int]
    max_live: int
    spills: int
    reloads: int
    final_reg_map: Dict[Label, str]
    profile: Dict[str, Any] = field(default_factory=dict)
    variable_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    placement_summary: Dict[str, Any] = field(default_factory=dict)
    shared_slots: int = 0


@dataclass(frozen=True)
class LARSPlacementConfig:
    """Post-schedule storage placement policy.

    The defaults intentionally target only clearly low-frequency, long-lived
    values.  They are conservative starting points and are exposed so JIT
    candidate generation can tune them later.
    """

    enabled: bool = True
    direct_use_threshold: int = 1
    input_low_freq_threshold: int = 1000
    input_long_lifetime_threshold: int = 10
    accumulator_low_freq_threshold: int = 2
    accumulator_long_lifetime_threshold: int = 32
    max_shared_slots: Optional[int] = 16
    stats_print: bool = False
    stats_path: Optional[str] = None


def _resolve_lars_placement_config(
    config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]],
) -> LARSPlacementConfig:
    if config is None:
        return LARSPlacementConfig()
    if isinstance(config, LARSPlacementConfig):
        return config
    if isinstance(config, dict):
        return LARSPlacementConfig(**dict(config))
    raise TypeError(
        "placement_config must be None, LARSPlacementConfig, or a dict"
    )


def _placement_ref_parts(ref: str) -> Tuple[str, int]:
    if not isinstance(ref, str) or "[" not in ref or not ref.endswith("]"):
        raise ValueError(f"Bad placement reference: {ref!r}")
    kind, tail = ref.split("[", 1)
    return kind, int(tail[:-1])


def _placement_peak_live(intervals: Sequence[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    events: Dict[int, int] = defaultdict(int)
    for first, last in intervals:
        events[int(first)] += 1
        events[int(last) + 1] -= 1
    live = 0
    peak = 0
    for pos in sorted(events):
        live += events[pos]
        peak = max(peak, live)
    return int(peak)


def _placement_assign_tokens(
    variables: List[Dict[str, Any]],
    *,
    placement: str,
    prefix: str,
) -> int:
    """Linear-scan color intervals and return the number of required tokens."""
    selected = [v for v in variables if v["placement"] == placement]
    selected.sort(key=lambda v: (v["first_use"], v["last_use"], v["name"]))
    active: List[Tuple[int, int]] = []  # (last_use, token_id)
    free_ids: List[int] = []
    next_id = 0
    for var in selected:
        first = int(var["first_use"])
        still_active: List[Tuple[int, int]] = []
        for last, token_id in active:
            if int(last) < first:
                free_ids.append(int(token_id))
            else:
                still_active.append((int(last), int(token_id)))
        active = still_active
        free_ids.sort()
        if free_ids:
            token_id = free_ids.pop(0)
        else:
            token_id = next_id
            next_id += 1
        var["token"] = f"{prefix}{token_id}"
        active.append((int(var["last_use"]), int(token_id)))
    return int(next_id)


def _extract_lars_compute_records(
    schedule_result: ScheduleResult,
    *,
    schedule_kind: str,
) -> List[Dict[str, Any]]:
    """Recover label-level operands from the original logical instruction stream."""
    reg_to_ref: Dict[str, str] = {}
    records: List[Dict[str, Any]] = []

    def ref_of(reg: str) -> str:
        if not reg:
            return ""
        if reg not in reg_to_ref:
            raise RuntimeError(
                f"Cannot recover label for register {reg!r}; schedule_kind={schedule_kind}"
            )
        return reg_to_ref[reg]

    for inst in schedule_result.instructions:
        if inst.op == "load":
            reg, ref = inst.args
            reg_to_ref[str(reg)] = str(ref)
            continue
        if inst.op == "release":
            reg = str(inst.args[0])
            reg_to_ref.pop(reg, None)
            continue

        if schedule_kind == "fwd" and inst.op in (
            "fma_u1d_resident", "fma_u1d_direct"
        ):
            out_idx, rx, ry, rw, coeff = inst.args
            records.append({
                "kind": "fwd",
                "input_refs": [ref_of(str(rx)), ref_of(str(ry)), ref_of(str(rw))],
                "input_weights": [1, 1, 1],
                "acc_refs": [f"out[{int(out_idx)}]"],
                "out_idx": int(out_idx),
                "coeff": float(coeff),
                "comment": inst.comment,
            })
            continue

        if schedule_kind == "stc_fwd" and inst.op in (
            "mul_stc_resident", "mul_stc_direct"
        ):
            out_idx, regs, coeff = inst.args
            refs = [ref_of(str(reg)) for reg in regs]
            records.append({
                "kind": "stc_fwd",
                # Duplicate refs intentionally preserve squares/cubes.  The
                # placement statistics count each expression occurrence while
                # path_use_count is deduplicated later per path.
                "input_refs": refs,
                "input_weights": [1] * len(refs),
                "acc_refs": [f"out[{int(out_idx)}]"],
                "out_idx": int(out_idx),
                "coeff": float(coeff),
                "comment": inst.comment,
            })
            continue

        if schedule_kind == "stc_bwd" and inst.op == "stc_bwd_path":
            x1_indices, x0_idx, out_v, coeff, need_grad_x0 = inst.args
            x1_indices = tuple(int(v) for v in x1_indices)
            arity = len(x1_indices)
            input_refs: List[str] = [
                f"grad_out[{int(out_v)}]",
                f"x0[{int(x0_idx)}]",
            ]
            # grad_out/x0 are used by every gx1 derivative.  gx0 adds one
            # grad_out use and one use of every x1 occurrence.
            input_weights: List[int] = [
                arity + int(bool(need_grad_x0)), arity
            ]
            multiplicity = Counter(x1_indices)
            for idx in sorted(multiplicity):
                weight = int(multiplicity[idx]) * (
                    arity - 1 + int(bool(need_grad_x0))
                )
                if weight > 0:
                    input_refs.append(f"x1[{int(idx)}]")
                    input_weights.append(weight)
            acc_refs = [f"grad_x1[{int(idx)}]" for idx in x1_indices]
            if need_grad_x0:
                acc_refs.append(f"grad_x0[{int(x0_idx)}]")
            records.append({
                "kind": "stc_bwd",
                "input_refs": input_refs,
                "input_weights": input_weights,
                # Keep one accumulator occurrence per derivative position so
                # duplicate x1 indices correctly increase accumulator use_count.
                "acc_refs": acc_refs,
                "x1_indices": x1_indices,
                "x0_idx": int(x0_idx),
                "out_v": int(out_v),
                "coeff": float(coeff),
                "need_grad_x0": bool(need_grad_x0),
                "comment": inst.comment,
            })
            continue

        if schedule_kind == "bwd_fused" and inst.op == "bwd_fma_resident":
            wi, xj, yk, rw, rx, ry, rgo, coeff, need_grad_w = inst.args
            acc_refs = []
            if bool(need_grad_w):
                acc_refs.append(f"grad_w[{int(wi)}]")
            acc_refs.extend([f"grad_x[{int(xj)}]", f"grad_y[{int(yk)}]"])
            records.append({
                "kind": "bwd_fused",
                "input_refs": [
                    ref_of(str(rw)), ref_of(str(rx)), ref_of(str(ry)), ref_of(str(rgo))
                ],
                "input_weights": (
                    [2, 2, 2, 3] if bool(need_grad_w) else [2, 1, 1, 2]
                ),
                "acc_refs": acc_refs,
                "wi": int(wi),
                "xj": int(xj),
                "yk": int(yk),
                "coeff": float(coeff),
                "need_grad_w": bool(need_grad_w),
                "comment": inst.comment,
            })
            continue

        if schedule_kind.startswith("bwd_split:") and inst.op == "bwd_split_fma_resident":
            grad_kind, target_idx, rw, rx, ry, rgo, coeff = inst.args
            expected_kind = schedule_kind.split(":", 1)[1]
            if str(grad_kind) != expected_kind:
                raise ValueError(
                    f"split schedule kind mismatch: {grad_kind!r} vs {expected_kind!r}"
                )
            kind_to_ref = {"gw": "grad_w", "gx": "grad_x", "gy": "grad_y"}
            refs = [ref_of(str(r)) for r in (rw, rx, ry, rgo) if str(r)]
            records.append({
                "kind": "bwd_split",
                "grad_kind": str(grad_kind),
                "input_refs": refs,
                "input_weights": [1] * len(refs),
                "input_regs_present": tuple(bool(str(r)) for r in (rw, rx, ry, rgo)),
                "acc_refs": [f"{kind_to_ref[str(grad_kind)]}[{int(target_idx)}]"],
                "target_idx": int(target_idx),
                "coeff": float(coeff),
                "comment": inst.comment,
            })
            continue

    if len(records) != len(schedule_result.path_order):
        raise RuntimeError(
            f"Recovered {len(records)} compute records for {len(schedule_result.path_order)} "
            f"scheduled paths ({schedule_kind})"
        )
    return records


def apply_lars_lifetime_placement(
    schedule_result: ScheduleResult,
    *,
    schedule_kind: str,
    config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]] = None,
) -> ScheduleResult:
    """Analyze use/lifetime and rewrite a LARS schedule with tiered placement.

    Policy:
      * use_count <= direct_use_threshold: inline global read/direct writeback;
      * low-frequency and long-lived: one private shared-memory slot per thread;
      * everything else: a linear-scan reused scalar register.
    """
    cfg = _resolve_lars_placement_config(config)
    if not cfg.enabled:
        return schedule_result

    records = _extract_lars_compute_records(
        schedule_result, schedule_kind=schedule_kind
    )
    variables_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for pos, rec in enumerate(records):
        input_weights = rec.get("input_weights", [1] * len(rec["input_refs"]))
        seen_input_refs: Set[str] = set()
        for ref, weight in zip(rec["input_refs"], input_weights):
            ref = str(ref)
            key = ("input", ref)
            var = variables_by_key.setdefault(key, {
                "role": "input",
                "name": ref,
                "use_count": 0,
                "path_use_count": 0,
                "first_use": int(pos),
                "last_use": int(pos),
            })
            var["use_count"] += int(weight)
            if ref not in seen_input_refs:
                var["path_use_count"] += 1
                seen_input_refs.add(ref)
            var["last_use"] = int(pos)
        seen_acc_refs: Set[str] = set()
        for ref in rec["acc_refs"]:
            ref = str(ref)
            key = ("accumulator", ref)
            var = variables_by_key.setdefault(key, {
                "role": "accumulator",
                "name": ref,
                "use_count": 0,
                "path_use_count": 0,
                "first_use": int(pos),
                "last_use": int(pos),
            })
            var["use_count"] += 1
            if ref not in seen_acc_refs:
                var["path_use_count"] += 1
                seen_acc_refs.add(ref)
            var["last_use"] = int(pos)

    variables = list(variables_by_key.values())
    for var in variables:
        var["lifetime"] = int(var["last_use"] - var["first_use"] + 1)
        kind, idx = _placement_ref_parts(var["name"])
        var["kind"] = kind
        var["index"] = int(idx)
        accesses = int(var["use_count"])
        path_uses = int(var.get("path_use_count", accesses))
        lifetime = int(var["lifetime"])
        # Direct placement is based on actual expression references.  Shared
        # classification uses scheduled path frequency, because one fused
        # backward path may reference the same loaded operand in 2-3 gradient
        # expressions without requiring additional lifetime state.
        if accesses <= int(cfg.direct_use_threshold):
            var["placement"] = "direct"
        else:
            if var["role"] == "input":
                low_freq = path_uses <= int(cfg.input_low_freq_threshold)
                long_lived = lifetime >= int(cfg.input_long_lifetime_threshold)
            else:
                low_freq = path_uses <= int(cfg.accumulator_low_freq_threshold)
                long_lived = lifetime >= int(cfg.accumulator_long_lifetime_threshold)
            var["placement"] = "shared" if (low_freq and long_lived) else "register"

    # Respect the shared-slot cap by selecting the highest pressure-relief value
    # first.  A candidate is retained only when the peak interval overlap remains
    # within the configured cap.
    shared_candidates = [v for v in variables if v["placement"] == "shared"]
    cap = cfg.max_shared_slots
    if cap is not None:
        cap = max(0, int(cap))
        selected: List[Dict[str, Any]] = []
        shared_candidates.sort(
            key=lambda v: (
                float(v["lifetime"]) / max(1, int(v.get("path_use_count", v["use_count"]))),
                int(v["lifetime"]),
                1 if v["role"] == "accumulator" else 0,
                -int(v.get("path_use_count", v["use_count"])),
                v["name"],
            ),
            reverse=True,
        )
        for cand in shared_candidates:
            trial = selected + [cand]
            peak = _placement_peak_live([
                (int(v["first_use"]), int(v["last_use"])) for v in trial
            ])
            if peak <= cap:
                selected.append(cand)
            else:
                cand["placement"] = "register"

    shared_slots = _placement_assign_tokens(
        variables, placement="shared", prefix="s"
    )
    register_slots = _placement_assign_tokens(
        variables, placement="register", prefix="r"
    )
    for var in variables:
        if var["placement"] == "direct":
            var["token"] = f"d:{var['name']}"

    input_by_ref = {
        v["name"]: v for v in variables if v["role"] == "input"
    }
    acc_by_ref = {
        v["name"]: v for v in variables if v["role"] == "accumulator"
    }
    input_first: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    input_last: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    acc_first: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    acc_last: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for var in variables:
        if var["role"] == "input":
            input_first[int(var["first_use"])].append(var)
            input_last[int(var["last_use"])].append(var)
        else:
            acc_first[int(var["first_use"])].append(var)
            acc_last[int(var["last_use"])].append(var)

    rewritten: List[Inst] = []
    for pos, rec in enumerate(records):
        for var in sorted(input_first[pos], key=lambda v: v["name"]):
            if var["placement"] == "register":
                rewritten.append(Inst(
                    "load", (var["token"], var["name"]),
                    f"placement=register accesses={var['use_count']} paths={var.get('path_use_count', var['use_count'])} lifetime={var['lifetime']}",
                ))
            elif var["placement"] == "shared":
                rewritten.append(Inst(
                    "load_shared", (var["token"], var["name"]),
                    f"placement=shared accesses={var['use_count']} paths={var.get('path_use_count', var['use_count'])} lifetime={var['lifetime']}",
                ))
        for var in sorted(acc_first[pos], key=lambda v: v["name"]):
            if var["placement"] in ("register", "shared"):
                rewritten.append(Inst(
                    "init_acc", (var["token"], var["name"]),
                    f"placement={var['placement']} accesses={var['use_count']} paths={var.get('path_use_count', var['use_count'])} lifetime={var['lifetime']}",
                ))

        input_tokens = [input_by_ref[ref]["token"] for ref in rec["input_refs"]]
        acc_tokens = [acc_by_ref[ref]["token"] for ref in rec["acc_refs"]]
        if rec["kind"] == "fwd":
            rewritten.append(Inst(
                "fma_u1d_placed",
                (
                    int(rec["out_idx"]), acc_tokens[0],
                    input_tokens[0], input_tokens[1], input_tokens[2],
                    float(rec["coeff"]),
                ),
                rec["comment"],
            ))
        elif rec["kind"] == "stc_fwd":
            rewritten.append(Inst(
                "mul_stc_placed",
                (
                    int(rec["out_idx"]), acc_tokens[0],
                    tuple(input_tokens), float(rec["coeff"]),
                ),
                rec["comment"],
            ))
        elif rec["kind"] == "stc_bwd":
            token_by_ref = {
                str(ref): str(token)
                for ref, token in zip(rec["input_refs"], input_tokens)
            }
            x1_indices = tuple(int(v) for v in rec["x1_indices"])
            x1_tokens = tuple(
                token_by_ref.get(f"x1[{int(idx)}]", "")
                for idx in x1_indices
            )
            need_grad_x0 = bool(rec["need_grad_x0"])
            if need_grad_x0:
                gx1_tokens = tuple(acc_tokens[:-1])
                gx0_token = acc_tokens[-1]
            else:
                gx1_tokens = tuple(acc_tokens)
                gx0_token = ""
            rewritten.append(Inst(
                "stc_bwd_placed",
                (
                    x1_indices, gx1_tokens, gx0_token,
                    token_by_ref[f"grad_out[{int(rec['out_v'])}]"],
                    token_by_ref[f"x0[{int(rec['x0_idx'])}]"],
                    x1_tokens, float(rec["coeff"]),
                ),
                rec["comment"],
            ))
        elif rec["kind"] == "bwd_fused":
            acc_iter = iter(acc_tokens)
            gw_token = next(acc_iter) if rec["need_grad_w"] else ""
            gx_token = next(acc_iter)
            gy_token = next(acc_iter)
            rewritten.append(Inst(
                "bwd_fma_placed",
                (
                    int(rec["wi"]), int(rec["xj"]), int(rec["yk"]),
                    gw_token, gx_token, gy_token,
                    input_tokens[0], input_tokens[1], input_tokens[2], input_tokens[3],
                    float(rec["coeff"]), bool(rec["need_grad_w"]),
                ),
                rec["comment"],
            ))
        elif rec["kind"] == "bwd_split":
            present = rec["input_regs_present"]
            token_iter = iter(input_tokens)
            operand_tokens = [next(token_iter) if flag else "" for flag in present]
            rewritten.append(Inst(
                "bwd_split_fma_placed",
                (
                    rec["grad_kind"], int(rec["target_idx"]), acc_tokens[0],
                    operand_tokens[0], operand_tokens[1], operand_tokens[2], operand_tokens[3],
                    float(rec["coeff"]),
                ),
                rec["comment"],
            ))
        else:
            raise ValueError(f"Unsupported placement record kind: {rec['kind']}")

        for var in sorted(acc_last[pos], key=lambda v: v["name"]):
            if var["placement"] in ("register", "shared"):
                rewritten.append(Inst(
                    "store_acc_placed", (var["name"], var["token"]),
                    f"last use at scheduled path position {pos}",
                ))
        for var in sorted(input_last[pos], key=lambda v: v["name"]):
            if var["placement"] == "register":
                rewritten.append(Inst(
                    "release", (var["token"], var["name"]),
                    f"last use at scheduled path position {pos}",
                ))
            elif var["placement"] == "shared":
                rewritten.append(Inst(
                    "release_shared", (var["token"], var["name"]),
                    f"last use at scheduled path position {pos}",
                ))

    reg_intervals = [
        (int(v["first_use"]), int(v["last_use"]))
        for v in variables if v["placement"] == "register"
    ]
    shared_intervals = [
        (int(v["first_use"]), int(v["last_use"]))
        for v in variables if v["placement"] == "shared"
    ]
    state_intervals = reg_intervals + shared_intervals
    reg_peak = _placement_peak_live(reg_intervals)
    shared_peak = _placement_peak_live(shared_intervals)
    total_peak = _placement_peak_live(state_intervals)

    reg_live_by_pos = [
        sum(
            1 for v in variables
            if v["placement"] == "register"
            and int(v["first_use"]) <= pos <= int(v["last_use"])
        )
        for pos in range(len(records))
    ]
    shared_live_by_pos = [
        sum(
            1 for v in variables
            if v["placement"] == "shared"
            and int(v["first_use"]) <= pos <= int(v["last_use"])
        )
        for pos in range(len(records))
    ]

    variable_stats: Dict[str, Dict[str, Any]] = {}
    role_counts: Dict[str, Counter[str]] = {
        "input": Counter(), "accumulator": Counter()
    }
    for var in sorted(variables, key=lambda v: (v["role"], v["kind"], v["index"])):
        role_counts[var["role"]][var["placement"]] += 1
        variable_stats[f"{var['role']}:{var['name']}"] = {
            "role": var["role"],
            "name": var["name"],
            "kind": var["kind"],
            "index": int(var["index"]),
            "use_count": int(var["use_count"]),
            "path_use_count": int(var.get("path_use_count", var["use_count"])),
            "first_use": int(var["first_use"]),
            "last_use": int(var["last_use"]),
            "lifetime": int(var["lifetime"]),
            "max_live_register_during_lifetime": int(max(
                reg_live_by_pos[int(var["first_use"]): int(var["last_use"]) + 1],
                default=0,
            )),
            "max_live_shared_during_lifetime": int(max(
                shared_live_by_pos[int(var["first_use"]): int(var["last_use"]) + 1],
                default=0,
            )),
            "max_live_total_during_lifetime": int(max(
                [
                    reg_live_by_pos[pos] + shared_live_by_pos[pos]
                    for pos in range(int(var["first_use"]), int(var["last_use"]) + 1)
                ],
                default=0,
            )),
            "placement": var["placement"],
            "token": var["token"],
        }

    summary = {
        "schedule_kind": schedule_kind,
        "path_count": len(records),
        "original_max_live_inputs": int(schedule_result.max_live),
        "max_live_register_state": int(reg_peak),
        "max_live_shared_state": int(shared_peak),
        "max_live_total_state": int(total_peak),
        "register_slots": int(register_slots),
        "shared_slots": int(shared_slots),
        "input_counts": dict(role_counts["input"]),
        "accumulator_counts": dict(role_counts["accumulator"]),
        "config": {
            "direct_use_threshold": int(cfg.direct_use_threshold),
            "input_low_freq_threshold": int(cfg.input_low_freq_threshold),
            "input_long_lifetime_threshold": int(cfg.input_long_lifetime_threshold),
            "accumulator_low_freq_threshold": int(cfg.accumulator_low_freq_threshold),
            "accumulator_long_lifetime_threshold": int(cfg.accumulator_long_lifetime_threshold),
            "max_shared_slots": cfg.max_shared_slots,
        },
    }
    profile = dict(schedule_result.profile)
    profile["placement"] = summary
    profile["variable_stats"] = variable_stats

    return ScheduleResult(
        instructions=rewritten,
        path_order=list(schedule_result.path_order),
        max_live=int(reg_peak),
        spills=int(schedule_result.spills),
        reloads=sum(1 for inst in rewritten if inst.op in ("load", "load_shared")),
        final_reg_map={},
        profile=profile,
        variable_stats=variable_stats,
        placement_summary=summary,
        shared_slots=int(shared_slots),
    )


def dump_lars_variable_stats(
    schedule_result: ScheduleResult,
    *,
    print_stats: bool = False,
    out_path: Optional[str] = None,
) -> None:
    """Print and/or dump per-variable use count, lifetime and placement."""
    summary = dict(getattr(schedule_result, "placement_summary", {}) or {})
    stats = dict(getattr(schedule_result, "variable_stats", {}) or {})
    if print_stats:
        print(
            "[LARS][placement] "
            f"kind={summary.get('schedule_kind')} paths={summary.get('path_count')} "
            f"max_live(original/register/shared/total)="
            f"{summary.get('original_max_live_inputs')}/"
            f"{summary.get('max_live_register_state')}/"
            f"{summary.get('max_live_shared_state')}/"
            f"{summary.get('max_live_total_state')} "
            f"slots(reg/shared)={summary.get('register_slots')}/"
            f"{summary.get('shared_slots')}",
            flush=True,
        )
        for item in stats.values():
            print(
                "[LARS][variable] "
                f"role={item['role']:<11s} name={item['name']:<18s} "
                f"uses={item['use_count']:<4d} paths={item['path_use_count']:<4d} "
                f"first={item['first_use']:<4d} last={item['last_use']:<4d} "
                f"lifetime={item['lifetime']:<4d} "
                f"max_live(reg/shared/total)="
                f"{item['max_live_register_during_lifetime']}/"
                f"{item['max_live_shared_during_lifetime']}/"
                f"{item['max_live_total_during_lifetime']} "
                f"placement={item['placement']:<8s} token={item['token']}",
                flush=True,
            )
    if out_path:
        payload = {"summary": summary, "variables": list(stats.values())}
        Path(out_path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


class LARSUniform1DScheduler:
    """
    All-input-resident LARS-style scheduler for Uniform1D paths:

        out[v] += x[i] * y[j] * w[k] * c

    This scheduler no longer implements spill/victim selection.  It allocates
    one virtual register for every distinct input label, chooses labels by the
    LARS score, and fires paths as soon as their input labels are live.

    Notes:
      - ScheduleResult.spills is kept for compatibility and is always zero.
      - Output accumulators are handled by the emitter, not by the LARS label
        allocator.
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        *,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        debug: bool = False,
        path_kind: str = "u1d",
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        self.path_kind = str(path_kind)
        if self.path_kind == "u1d":
            self.paths: List[Any] = [
                U1DPath(pid=p, i=i, j=j, k=k, v=v, c=c)
                for p, (i, j, k, v, c) in enumerate(paths)
            ]
        elif self.path_kind == "stc":
            # STC paths are already materialized as STCPath objects.  They expose
            # labels for live-set management and product_labels for multiplicity-
            # preserving emission, e.g. x1[i] * x1[i].
            self.paths = list(paths)
            for expected_pid, path in enumerate(self.paths):
                if int(path.pid) != expected_pid:
                    raise ValueError(
                        f"STC paths must be dense pid-ordered; "
                        f"got path.pid={path.pid} at position {expected_pid}"
                    )
        else:
            raise ValueError(f"Unsupported LARSUniform1DScheduler path_kind={self.path_kind!r}")

        # Input labels are unbounded: virtual registers are allocated on demand.

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
        self.debug = bool(debug)

        self.label_to_paths: Dict[Label, Set[int]] = defaultdict(set)
        for path in self.paths:
            for lab in path.labels:
                self.label_to_paths[lab].add(path.pid)

        self.path_label_sets: List[Set[Label]] = [set(p.labels) for p in self.paths]
        self.unscheduled: Set[int] = set(p.pid for p in self.paths)

        self.remaining_uses: Counter[Label] = Counter()
        for p in self.paths:
            for lab in p.labels:
                self.remaining_uses[lab] += 1

        # Forward output accumulator use counts.  out[v] with exactly one path
        # does not need a full-resident accumulator: emit one direct writeback
        # immediately after its computation instead of keeping out_acc_v alive
        # until the end of the kernel.
        self.output_uses: Counter[int] = Counter(int(p.v) for p in self.paths)

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = []
        self._next_reg_id = 0

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        # max_live is kept for backward compatibility. It records the maximum
        # number of live normal input labels in the logical schedule.
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}

        # Optional profile / progress reporting.  Kept disabled by default so
        # existing code generation remains silent unless requested.
        self.profile_enabled = bool(profile)
        self.profile_name = str(profile_name or self.__class__.__name__)
        self.profile_interval = int(profile_interval)
        self.profile_seconds = float(profile_seconds)
        self.profile_print = bool(profile_print)
        self._profile_t0 = perf_counter()
        self._profile_last_print_t = self._profile_t0
        self._profile_last_print_done = 0
        self._profile_loop_iter = 0
        self._profile_select_rounds = 0
        self._profile_fire_rounds = 0
        self._profile_load_rounds = 0
        self._profile_records: List[Dict[str, Any]] = []
        self._profile_last_event = ""
        self._profile_last_reason = ""

    # ------------------------------------------------------------------
    # Profile helpers
    # ------------------------------------------------------------------
    def _profile_total_live_regs(self) -> int:
        if hasattr(self, "_total_live_regs"):
            try:
                return int(self._total_live_regs())
            except Exception:
                pass
        return len(self.live)

    def _fmt_profile_eta(self, seconds: float) -> str:
        if seconds == float("inf"):
            return "inf"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"

    def _profile_candidate_count_safe(self) -> int:
        try:
            return len(self._candidate_labels())
        except Exception:
            return 0


    def _profile_snapshot(
        self,
        *,
        event: str,
        reason: str = "",
        fireable_n: Optional[int] = None,
        candidate_n: Optional[int] = None,
        select_ms: Optional[float] = None,
        lab: Optional[Label] = None,
    ) -> Dict[str, Any]:
        done = len(self.path_order)
        remain = len(self.unscheduled)
        total = len(self.paths)
        elapsed = max(perf_counter() - self._profile_t0, 1e-12)
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (remain / rate) if rate > 0 else float("inf")
        snap: Dict[str, Any] = {
            "name": self.profile_name,
            "event": event,
            "iter": int(self._profile_loop_iter),
            "done_paths": int(done),
            "remaining_paths": int(remain),
            "total_paths": int(total),
            "progress_pct": float(100.0 * done / max(1, total)),
            "live_labels": int(len(self.live)),
            "live_regs_total": int(self._profile_total_live_regs()),
            "max_live": int(self.max_live),
            "max_live_labels": int(getattr(self, "max_live_labels", self.max_live)),
            "max_live_total": int(getattr(self, "max_live_total", self.max_live)),
            "fireable_paths": int(fireable_n if fireable_n is not None else 0),
            "candidate_labels": int(candidate_n if candidate_n is not None else 0),
            "select_ms": float(select_ms or 0.0),
            "spills": int(self.spills),
            "reloads": int(self.reloads),
            "instructions": int(len(self.instructions)),
            "select_rounds": int(self._profile_select_rounds),
            "fire_rounds": int(self._profile_fire_rounds),
            "load_rounds": int(self._profile_load_rounds),
            "elapsed_s": float(elapsed),
            "rate_paths_per_s": float(rate),
            "eta_s": float(eta),
            "reason": str(reason),
            "label": None if lab is None else self._label_name(lab),
        }
        return snap

    def _profile_emit(
        self,
        *,
        event: str,
        force: bool = False,
        reason: str = "",
        fireable_n: Optional[int] = None,
        candidate_n: Optional[int] = None,
        select_ms: Optional[float] = None,
        lab: Optional[Label] = None,
    ) -> None:
        if not self.profile_enabled:
            return

        now = perf_counter()
        done = len(self.path_order)
        enough_paths = (
            self.profile_interval > 0
            and done > 0
            and done != self._profile_last_print_done
            and done % self.profile_interval == 0
        )
        enough_time = (now - self._profile_last_print_t) >= self.profile_seconds
        if not (force or enough_paths or enough_time):
            return

        snap = self._profile_snapshot(
            event=event,
            reason=reason,
            fireable_n=fireable_n,
            candidate_n=candidate_n,
            select_ms=select_ms,
            lab=lab,
        )
        self._profile_records.append(snap)
        self._profile_last_event = event
        self._profile_last_reason = reason

        if self.profile_print:
            eta = self._fmt_profile_eta(snap["eta_s"])
            elapsed = self._fmt_profile_eta(snap["elapsed_s"])
            print(
                f"[LARS][profile] {snap['name']} {event:>8s} "
                f"iter={snap['iter']} "
                f"done={snap['done_paths']}/{snap['total_paths']} "
                f"remain={snap['remaining_paths']} "
                f"max=label/total "
                f"{snap['max_live_labels']}/{snap['max_live_total']} "
                f"fireable={snap['fireable_paths']} "
                f"candidates={snap['candidate_labels']} "
                f"label={snap['label']} "
                f"select={snap['select_ms']:.2f}ms "
                f"spills={snap['spills']} reloads={snap['reloads']} "
                f"insts={snap['instructions']} "
                f"rate={snap['rate_paths_per_s']:.1f} path/s "
                f"elapsed={elapsed} eta={eta} "
                f"reason={snap['reason']}",
                flush=True,
            )

        self._profile_last_print_t = now
        self._profile_last_print_done = done

    def _build_profile_summary(self) -> Dict[str, Any]:
        elapsed = max(perf_counter() - self._profile_t0, 1e-12)
        total = len(self.paths)
        done = len(self.path_order)
        remain = len(self.unscheduled)
        summary: Dict[str, Any] = {
            "name": self.profile_name,
            "total_paths": int(total),
            "done_paths": int(done),
            "remaining_paths": int(remain),
            "max_live": int(self.max_live),
            "max_live_labels": int(getattr(self, "max_live_labels", self.max_live)),
            "max_live_total": int(getattr(self, "max_live_total", self.max_live)),
            "spills": int(self.spills),
            "reloads": int(self.reloads),
            "instructions": int(len(self.instructions)),
            "select_rounds": int(self._profile_select_rounds),
            "fire_rounds": int(self._profile_fire_rounds),
            "load_rounds": int(self._profile_load_rounds),
            "elapsed_s": float(elapsed),
            "rate_paths_per_s": float(done / elapsed if elapsed > 0 else 0.0),
            "records": list(self._profile_records),
        }
        return summary

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _path(self, pid: int) -> U1DPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        if kind == "o":
            return f"out[{idx}]"
        return f"{kind}[{idx}]"

    def _invalidate_score_cache(self) -> None:
        self._score_cache.clear()

    def _alloc_reg_no_spill(self, lab: Label) -> str:
        if self.free_regs:
            reg = self.free_regs.pop(0)
        else:
            reg = f"r{self._next_reg_id}"
            self._next_reg_id += 1

        self.reg_of[lab] = reg
        self.live.add(lab)
        self.max_live_labels = max(self.max_live_labels, len(self.live))
        self.max_live_total = max(self.max_live_total, len(self.live))
        self.max_live = max(self.max_live, len(self.live))
        self._invalidate_score_cache()
        return reg

    def _emit_load_inst(self, lab: Label, reg: str, reason: str) -> None:
        if lab[0] == "o":
            self.instructions.append(
                Inst("load_acc", (reg, self._label_name(lab)), reason)
            )
        else:
            self.instructions.append(
                Inst("load", (reg, self._label_name(lab)), reason)
            )
        self.reloads += 1

    def _store_and_release(self, lab: Label, reason: str = "") -> None:
        if lab not in self.live:
            return

        reg = self.reg_of[lab]

        if lab[0] == "o":
            # If the output accumulator was never modified, a store_acc is not
            # needed. Emit a plain release comment instead. The emitter can
            # ignore release instructions.
            if lab in self.dirty_outputs:
                self.instructions.append(
                    Inst("store_acc", (self._label_name(lab), reg), reason)
                )
                self.dirty_outputs.discard(lab)
            else:
                self.instructions.append(
                    Inst("release", (reg, self._label_name(lab)), reason)
                )
        else:
            self.instructions.append(
                Inst("release", (reg, self._label_name(lab)), reason)
            )

        self.live.remove(lab)
        self.reg_of.pop(lab, None)
        self.free_regs.insert(0, reg)
        self._invalidate_score_cache()

    # ------------------------------------------------------------------
    # Fireability
    # ------------------------------------------------------------------
    def _is_fireable(self, pid: int) -> bool:
        return self.path_label_sets[pid] <= self.live

    def _fireable_paths(self) -> List[int]:
        return [pid for pid in self.unscheduled if self.path_label_sets[pid] <= self.live]

    def _fireable_count_after_load(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
        count = 0
        for pid in self.unscheduled:
            if self.path_label_sets[pid] <= tmp_live:
                count += 1
        return count

    def _best_fireable_release_after_load(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
        best = 0
        for pid in self.unscheduled:
            labels = self.path_label_sets[pid]
            if labels <= tmp_live:
                release_now = sum(1 for l in labels if self.remaining_uses[l] == 1)
                best = max(best, release_now)
        return best

    # ------------------------------------------------------------------
    # LARS score
    # ------------------------------------------------------------------
    def _release_potential_if_add(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
        score = 0

        for pid in self.unscheduled:
            labels = self.path_label_sets[pid]
            if labels <= tmp_live:
                for l in labels:
                    if self.remaining_uses[l] == 1:
                        score += 2 if l in self.live else 1

        return score

    def _fire_potential_if_add(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
        count = 0

        for pid in self.label_to_paths[lab] & self.unscheduled:
            if self.path_label_sets[pid] <= tmp_live:
                count += 1

        return count

    def _primary_affinity(self, lab: Label) -> int:
        score = 0
        for pid in self.label_to_paths[lab] & self.unscheduled:
            score += len(self.path_label_sets[pid] & self.live)
        return score

    def _secondary_affinity(self, lab: Label) -> int:
        if not self.enable_secondary_affinity:
            return 0

        neighbor_labels: Set[Label] = set()
        for pid in self.unscheduled:
            labels = self.path_label_sets[pid]
            if labels & self.live:
                neighbor_labels |= labels

        score = 0
        for pid in self.label_to_paths[lab] & self.unscheduled:
            labels = self.path_label_sets[pid]
            score += len((labels & neighbor_labels) - self.live - {lab})
        return score

    def _non_live_affinity_penalty(self, lab: Label) -> int:
        neighbors: Set[Label] = set()
        for pid in self.label_to_paths[lab] & self.unscheduled:
            neighbors |= self.path_label_sets[pid]
        return len(neighbors - self.live - {lab})

    def _priority(self, lab: Label) -> int:
        acc_bonus = 1 if lab[0] == "o" else 0
        return self.remaining_uses[lab] + acc_bonus

    def _label_score(self, lab: Label) -> Tuple[int, int, int, int, int, int]:
        cached = self._score_cache.get(lab)
        if cached is not None:
            return cached

        score = (
            self._release_potential_if_add(lab),
            self._fire_potential_if_add(lab),
            self._primary_affinity(lab),
            self._secondary_affinity(lab),
            -self._non_live_affinity_penalty(lab),
            self._priority(lab),
        )
        self._score_cache[lab] = score
        return score

    def _candidate_labels(self) -> List[Label]:
        candidates = {
            lab
            for pid in self.unscheduled
            for lab in self.path_label_sets[pid]
            if lab not in self.live
        }

        if not candidates:
            raise RuntimeError("No candidate label but no path is fireable.")

        # Optional cheap prefilter. This is useful for large path lists, because
        # full LARS score repeatedly scans unscheduled paths.
        if self.topk_candidates is not None and len(candidates) > self.topk_candidates:
            ranked = sorted(
                candidates,
                key=lambda lab: (
                    self.remaining_uses[lab],
                    len(self.label_to_paths[lab] & self.unscheduled),
                    1 if lab[0] == "o" else 0,
                    str(lab),
                ),
                reverse=True,
            )
            return ranked[: int(self.topk_candidates)]

        return list(candidates)

    # ------------------------------------------------------------------
    # Label selection / no-spill load
    # ------------------------------------------------------------------
    def _select_next_label_global(self) -> Tuple[Label, str]:
        candidates = self._candidate_labels()

        best_lab: Optional[Label] = None
        best_key = None

        for lab in candidates:
            lab_score = self._label_score(lab)
            fire_after = self._fireable_count_after_load(lab)
            release_after = self._best_fireable_release_after_load(lab)
            key = (
                fire_after,
                lab_score,
                #release_after,
                str(lab),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_lab = lab

        if best_lab is None:
            raise RuntimeError("No candidate label but no path is fireable.")
        return best_lab, f"global key={best_key}"

    def _load_label_no_spill(self, lab: Label, reason: str = "") -> None:
        if lab in self.live:
            return
        reg = self._alloc_reg_no_spill(lab)
        self._emit_load_inst(lab, reg, reason)

    # ------------------------------------------------------------------
    # Path firing
    # ------------------------------------------------------------------
    def _choose_fireable_path(self, fireable: List[int]) -> int:
        def key(pid: int):
            labels = self.path_label_sets[pid]
            p = self._path(pid)
            release_now = sum(1 for l in labels if self.remaining_uses[l] == 1)
            reuse_score = sum(self.remaining_uses[l] for l in labels)

            if getattr(self, "path_kind", "u1d") == "stc":
                # Prefer shorter products as a final tie-breaker; this keeps the
                # emitted STC expression compact when all other reuse signals tie.
                arity = len(getattr(p, "product_labels", p.labels))
                return (release_now, reuse_score, -arity, -pid)

            out_dirty = 1 if ("o", p.v) in self.dirty_outputs else 0
            return (release_now, out_dirty, reuse_score, -pid)

        return max(fireable, key=key)

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)

        if getattr(self, "path_kind", "u1d") == "stc":
            product_labels = tuple(getattr(p, "product_labels", p.labels))
            regs = tuple(self.reg_of[lab] for lab in product_labels)
            if self.output_uses[int(p.v)] <= 1:
                self.instructions.append(
                    Inst(
                        "mul_stc_direct",
                        (int(p.v), regs, float(p.c)),
                        f"path#{pid}: direct out[{p.v}] += c * product",
                    )
                )
            else:
                self.instructions.append(
                    Inst(
                        "mul_stc_resident",
                        (int(p.v), regs, float(p.c)),
                        f"path#{pid}: out[{p.v}] += c * product",
                    )
                )
        else:
            lx, ly, lw = p.labels

            rx = self.reg_of[lx]
            ry = self.reg_of[ly]
            rw = self.reg_of[lw]

            if self.output_uses[int(p.v)] <= 1:
                self.instructions.append(
                    Inst(
                        "fma_u1d_direct",
                        (p.v, rx, ry, rw, p.c),
                        (
                            f"path#{pid}: direct out[{p.v}] += "
                            f"x[{p.i}] * y[{p.j}] * w[{p.k}] * {p.c}"
                        ),
                    )
                )
            else:
                self.instructions.append(
                    Inst(
                        "fma_u1d_resident",
                        (p.v, rx, ry, rw, p.c),
                        f"path#{pid}: out[{p.v}] += x[{p.i}] * y[{p.j}] * w[{p.k}] * {p.c}",
                    )
                )

        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        for lab in p.labels:
            self.remaining_uses[lab] -= 1

        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")

    # ------------------------------------------------------------------
    # Public scheduling entry
    # ------------------------------------------------------------------
    def schedule(self) -> ScheduleResult:
        self._profile_emit(event="start", force=True, reason="schedule start")

        while self.unscheduled:
            self._profile_loop_iter += 1

            fireable = self._fireable_paths()
            fireable_n = len(fireable)
            if fireable:
                pid = self._choose_fireable_path(fireable)
                self._emit_path_compute(pid)
                self._profile_fire_rounds += 1
                self._profile_emit(
                    event="fire",
                    reason=f"path#{pid}",
                    fireable_n=fireable_n,
                    candidate_n=0,
                )
                continue

            select_t0 = perf_counter()
            candidate_n = self._profile_candidate_count_safe()
            lab, reason = self._select_next_label_global()
            select_ms = (perf_counter() - select_t0) * 1000.0
            self._profile_select_rounds += 1

            self._load_label_no_spill(lab, reason=reason)
            self._profile_load_rounds += 1
            self._profile_emit(
                event="load",
                reason=reason,
                fireable_n=fireable_n,
                candidate_n=candidate_n,
                select_ms=select_ms,
                lab=lab,
            )

        for lab in list(self.live):
            self._store_and_release(lab, reason="end of schedule")

        self._profile_emit(event="end", force=True, reason="schedule end")
        profile_summary = self._build_profile_summary()

        return ScheduleResult(
            instructions=self.instructions,
            path_order=self.path_order,
            max_live=self.max_live,
            spills=0,
            reloads=self.reloads,
            final_reg_map=dict(self.reg_of),
            profile=profile_summary,
        )


PairKey = Tuple[Label, Label]  # (w-label, x-label)




class ProgressLARSUniform1DScheduler(LARSUniform1DScheduler):
    """Progress-printing wrapper for the no-spill all-input-resident scheduler."""

    def __init__(
        self,
        *args,
        progress_interval: int = 1000,
        progress_seconds: float = 2.0,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.progress_interval = int(progress_interval)
        self.progress_seconds = float(progress_seconds)
        self.verbose = bool(verbose)

        self._total_paths = len(self.paths)
        self._loop_iter = 0
        self._select_rounds = 0
        self._fire_rounds = 0
        self._load_rounds = 0

        self._t0 = perf_counter()
        self._last_print_t = self._t0
        self._last_print_done = 0

        self._last_fireable_n = 0
        self._last_candidate_n = 0
        self._last_best_lab = None
        self._last_best_score = None
        self._last_select_ms = 0.0
        self._last_select_mode = ""
        self._last_select_reason = ""

    def _fmt_lab(self, lab) -> str:
        if lab is None:
            return "None"
        kind, idx = lab
        return f"{kind}[{idx}]"

    def _fmt_eta(self, seconds: float) -> str:
        if seconds == float("inf"):
            return "inf"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"

    def _candidate_count(self) -> int:
        try:
            return len(self._candidate_labels())
        except Exception:
            return 0

    def _print_progress(self, *, force: bool = False, event: str = "") -> None:
        if not self.verbose:
            return

        now = perf_counter()
        done = len(self.path_order)
        remain = len(self.unscheduled)

        enough_paths = (
            self.progress_interval > 0
            and done > 0
            and done != self._last_print_done
            and done % self.progress_interval == 0
        )
        enough_time = (now - self._last_print_t) >= self.progress_seconds
        if not (force or enough_paths or enough_time):
            return

        elapsed = max(now - self._t0, 1e-12)
        rate = done / elapsed
        eta = (remain / rate) if rate > 0 else float("inf")
        pct = 100.0 * done / max(1, self._total_paths)

        print(
            f"[LARS] {event:>8s} "
            f"iter={self._loop_iter} "
            f"done={done}/{self._total_paths} ({pct:.2f}%) "
            f"remain={remain} "
            f"live={len(self.live)} "
            f"fireable={self._last_fireable_n} "
            f"candidates={self._last_candidate_n} "
            f"mode={self._last_select_mode} "
            f"best={self._fmt_lab(self._last_best_lab)} "
            f"score={self._last_best_score} "
            f"select={self._last_select_ms:.2f}ms "
            f"reloads={self.reloads} "
            f"insts={len(self.instructions)} "
            f"rate={rate:.1f} path/s "
            f"elapsed={self._fmt_eta(elapsed)} eta={self._fmt_eta(eta)} "
            f"reason={self._last_select_reason}",
            flush=True,
        )

        self._last_print_t = now
        self._last_print_done = done

    def _select_label_with_progress(self):
        t0 = perf_counter()
        self._last_candidate_n = self._candidate_count()
        lab, reason = self._select_next_label_global()
        self._select_rounds += 1
        self._last_select_ms = (perf_counter() - t0) * 1000.0
        self._last_best_lab = lab
        self._last_select_mode = "global"
        self._last_select_reason = str(reason)
        try:
            self._last_best_score = self._label_score(lab)
        except Exception:
            self._last_best_score = None
        return lab, reason

    def schedule(self) -> ScheduleResult:
        self._print_progress(force=True, event="start")

        while self.unscheduled:
            self._loop_iter += 1
            fireable = self._fireable_paths()
            self._last_fireable_n = len(fireable)

            if fireable:
                pid = self._choose_fireable_path(fireable)
                self._emit_path_compute(pid)
                self._fire_rounds += 1
                self._last_select_mode = "fire"
                self._last_best_lab = None
                self._last_best_score = None
                self._last_select_reason = f"path#{pid}"
                self._print_progress(event="fire")
                continue

            lab, reason = self._select_label_with_progress()
            self._load_rounds += 1
            self._load_label_no_spill(
                lab,
                reason=(
                    f"LARS-score={self._last_best_score}, "
                    f"mode={self._last_select_mode}, "
                    f"select_reason={reason}"
                ),
            )
            self._print_progress(event="load")

        for lab in list(self.live):
            self._store_and_release(lab, reason="end of schedule")

        self._print_progress(force=True, event="end")

        if self.verbose:
            print(
                f"[LARS] summary "
                f"select_rounds={self._select_rounds} "
                f"fire_rounds={self._fire_rounds} "
                f"load_rounds={self._load_rounds} "
                f"max_live={self.max_live} "
                f"spills=0 "
                f"reloads={self.reloads}",
                flush=True,
            )

        return ScheduleResult(
            instructions=self.instructions,
            path_order=self.path_order,
            max_live=self.max_live,
            spills=0,
            reloads=self.reloads,
            final_reg_map=dict(self.reg_of),
            profile=self._build_profile_summary(),
        )


# =============================================================================
# CUDA codegen from LARSUniform1DScheduler ScheduleResult
# =============================================================================

def _lars_reg_id(reg: str) -> int:
    if not isinstance(reg, str) or not reg.startswith("r"):
        raise ValueError(f"Bad LARS register name: {reg!r}")
    return int(reg[1:])


def _parse_lars_label_ref(ref: str) -> Tuple[str, int]:
    """Parse refs emitted by LARSUniform1DScheduler: x[0], y[1], w[2], out[3]."""
    if not isinstance(ref, str) or "[" not in ref or not ref.endswith("]"):
        raise ValueError(f"Bad LARS label ref: {ref!r}")
    kind, rest = ref.split("[", 1)
    idx = int(rest[:-1])
    if kind == "out":
        kind = "o"
    if kind not in ("x", "y", "w", "o"):
        raise ValueError(f"Bad LARS label kind in ref: {ref!r}")
    return kind, idx


def _fmt_lars_float(x: float) -> str:
    # repr(float) is accepted by CUDA for normal finite constants.
    return repr(float(x))


def _sanitize_cuda_comment(s: str) -> str:
    return str(s).replace("\n", " ").replace("\r", " ").replace("*/", "* /")




def _lars_has_output_reload_after_store(schedule_result: ScheduleResult) -> bool:
    """
    Scatter mode cannot safely use out[] as an accumulator temporary buffer, because
    another block may atomically update the same out element between a temporary store
    and a later reload. Dense mode is safe because one block owns one output row.
    """
    stored = set()
    for inst in schedule_result.instructions:
        if inst.op == "store_acc":
            lab = _parse_lars_label_ref(inst.args[0])
            stored.add(lab)
        elif inst.op == "load_acc":
            lab = _parse_lars_label_ref(inst.args[1])
            if lab in stored:
                return True
    return False


def _lars_label_index_expr(
    kind: str,
    idx: int,
    *,
    mode_scalar_y: bool,
    u_dim: Optional[int],
    x_dim: Optional[int],
    y_dim: Optional[int],
    w_dim: Optional[int],
    v_dim: Optional[int],
) -> str:
    if kind == "x":
        if x_dim is not None and idx >= x_dim:
            raise ValueError(f"x index {idx} out of x_dim={x_dim}")
        if u_dim is not None:
            return f"x_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"x_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "w":
        if w_dim is not None and idx >= w_dim:
            raise ValueError(f"w index {idx} out of w_dim={w_dim}")
        if u_dim is not None:
            return f"w_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"w_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "y":
        if y_dim is not None and idx >= y_dim:
            raise ValueError(f"y index {idx} out of y_dim={y_dim}")
        if mode_scalar_y:
            return f"y_base + (index_t){idx}"
        if u_dim is not None:
            return f"y_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"y_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "o":
        if v_dim is not None and idx >= v_dim:
            raise ValueError(f"out index {idx} out of v_dim={v_dim}")
        if u_dim is not None:
            return f"out_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"out_base + (index_t){idx} * (index_t)U + (index_t)u"

    raise ValueError(f"Bad label kind: {kind}")



def emit_launcher(
    bundle_name: str,
    mode: str = "u,u,,u",
    *,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
    if mode == "u,u,,u":
        y_comment = "[B,Ky,1] or [S,Ky,1]"
        y_check_u = (
            'TORCH_CHECK((int)y.size(2) == 1, '
            '"y.size(2) must be 1 for mode u,u,,u");'
        )
    elif mode == "u,u,u,u":
        y_comment = "[B,Ky,U] or [S,Ky,U]"
        y_check_u = (
            'TORCH_CHECK((int)y.size(2) == U, '
            '"y.size(2) must equal U for mode u,u,u,u");'
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    param_lines = [
        "    torch::Tensor w",
        "    torch::Tensor x_all",
        "    torch::Tensor y",
    ]

    if use_x_src or use_y_src:
        param_lines.append("    torch::Tensor src_idx")

    if use_scatter:
        param_lines.append("    torch::Tensor dst_idx")

    param_lines.append("    int64_t V64")

    params = ",\n".join(param_lines)

    src_check = ""
    if use_x_src or use_y_src:
        src_check = r'''
    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''

    dst_check = ""
    if use_scatter:
        dst_check = r'''
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''

    if use_scatter:
        out_alloc = "    auto out = torch::zeros({S, V, U}, w.options());"
    else:
        out_alloc = "    auto out = torch::zeros({B, V, U}, w.options());"

    blist_logic = ""

    src_numel_check = ""
    if use_x_src or use_y_src:
        src_numel_check = (
            '    TORCH_CHECK(src_idx.numel() >= B, '
            '"src_idx numel must be >= B");'
        )

    dst_numel_check = ""
    if use_scatter:
        dst_numel_check = (
            '    TORCH_CHECK(dst_idx.numel() >= B, '
            '"dst_idx numel must be >= B");'
        )

    launch_src_arg = (
        "(const int32_t*)src_idx.data_ptr<int32_t>(),"
        if (use_x_src or use_y_src)
        else "nullptr,"
    )

    launch_dst_arg = (
        "(const int32_t*)dst_idx.data_ptr<int32_t>(),"
        if use_scatter
        else "nullptr,"
    )

    # ------------------------------------------------------------------
    # Select B source according to execution mode.
    #
    # scatter mode:
    #   B is number of destination/scatter entries.
    #
    # src-index mode:
    #   B is number of source-index entries.
    #
    # dense mode:
    #   B comes from w.size(0), where w is either [1,Iw,U] or [B,Iw,U].
    # ------------------------------------------------------------------
    if use_scatter:
        b_expr = "dst_idx.size(0)"
    elif use_x_src or use_y_src:
        b_expr = "src_idx.size(0)"
    else:
        b_expr = "w.size(0)"

    return rf'''

torch::Tensor launcher_{bundle_name}(
{params})
{{
    // Expected tensors:
    //   w      : [WB, Iw, U], WB can be 1 or B
    //   x_all  : [S, Ix, U] or [B, Ix, U]
    //   y      : {y_comment}
    // Optional:
    //   src_idx: [?] int32, enabled when x/y source indirection is used
    //   dst_idx: [?] int32, enabled when scatter is used

    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(),
                "w/x_all/y must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(),
                "w/x_all/y must be contiguous");
{src_check}{dst_check}

    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x_all.dim() == 3, "x_all must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");

    int B  = (int){b_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x_all.size(0);
    int Ix = (int)x_all.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");

    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(1) > 0, "Iw must be > 0");

    TORCH_CHECK((int)x_all.size(1) > 0, "Ix must be > 0");
    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");

    {y_check_u}

{src_numel_check}
{dst_numel_check}

{blist_logic}
{out_alloc}

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{

        TORCH_CHECK(x_all.scalar_type() == w.scalar_type(),
                    "x_all dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(),
                    "y dtype must match w");

        launch_{bundle_name}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    }});

    GPU_KERNEL_LAUNCH_CHECK();

    return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} forward jit impl");
}}
'''



def _max_lars_reg_count_any(schedule_result: ScheduleResult) -> int:
    """
    Return max virtual register id + 1 by scanning string register arguments
    in the logical instruction stream.
    """
    max_id = -1
    for inst in schedule_result.instructions:
        for arg in inst.args:
            if isinstance(arg, str) and arg.startswith("r"):
                try:
                    max_id = max(max_id, _lars_reg_id(arg))
                except ValueError:
                    pass
    return max_id + 1






def emit_fused_fwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    w_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """Emit forward code from either a legacy or lifetime-placed schedule."""
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    reg_count = _max_lars_reg_count_any(schedule_result)
    shared_slots = int(getattr(schedule_result, "shared_slots", 0) or 0)
    has_placed = any(
        inst.op == "fma_u1d_placed" for inst in schedule_result.instructions
    )

    resident_out_indices = [] if has_placed else sorted({
        int(inst.args[0])
        for inst in schedule_result.instructions
        if inst.op == "fma_u1d_resident"
    })
    direct_out_indices = [] if has_placed else sorted({
        int(inst.args[0])
        for inst in schedule_result.instructions
        if inst.op == "fma_u1d_direct"
    })

    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    def shared_expr(token: str) -> str:
        if not isinstance(token, str) or not token.startswith("s"):
            raise ValueError(f"Bad shared token: {token!r}")
        slot = int(token[1:])
        return f"lars_smem[(size_t){slot} * (size_t)blockDim.x + (size_t)tid]"

    def direct_input_expr(ref: str) -> str:
        kind, idx = _parse_lars_label_ref(ref)
        if kind == "o":
            raise ValueError("output cannot be used as a direct input")
        expr = _lars_label_index_expr(
            kind, idx,
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            w_dim=w_dim,
            v_dim=v_dim,
        )
        return f"{kind}[{expr}]"

    def operand_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        if token.startswith("d:"):
            return direct_input_expr(token[2:])
        raise ValueError(f"Bad forward operand token: {token!r}")

    def acc_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        raise ValueError(f"Bad forward accumulator token: {token!r}")

    def emit_out_write(idx: int, value_expr: str) -> None:
        expr = _lars_label_index_expr(
            "o", int(idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            w_dim=w_dim,
            v_dim=v_dim,
        )
        if use_scatter:
            ap(f"            atomicAdd(&out[{expr}], {value_expr});")
        else:
            ap(f"            out[{expr}] += {value_expr};")

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    scalar_t* __restrict__ out,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    if shared_slots:
        ap("    extern __shared__ unsigned char lars_smem_raw[];")
        ap("    scalar_t* lars_smem = reinterpret_cast<scalar_t*>(lars_smem_raw);")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
        ap("")
    ap("    const int e_orig = e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    ap("    const int x_row = " + ("src_idx[e_orig];" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src_idx[e_orig];" if use_y_src else "e_orig;"))
    ap("    const int out_row = " + ("dst_idx[e_orig];" if use_scatter else "e_orig;"))
    ap("")
    ap("    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;")
    ap("    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;")
    if mode_scalar_y:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky;")
    else:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky * (index_t)U;")
    ap("    const index_t out_base = (index_t)out_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    for out_idx in resident_out_indices:
        ap(f"            scalar_t out_acc_v_{out_idx} = scalar_t(0);")
    if resident_out_indices:
        ap("")
    if direct_out_indices:
        ap(f"            // direct single-use output writeback enabled for {len(direct_out_indices)} out accumulator(s)")
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op in ("load", "load_shared"):
            token, ref = inst.args
            kind, idx = _parse_lars_label_ref(ref)
            if kind == "o":
                raise ValueError("input load must not target output labels")
            expr = _lars_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                w_dim=w_dim,
                v_dim=v_dim,
            )
            lhs = str(token) if inst.op == "load" else shared_expr(str(token))
            ap(f"            {lhs} = {kind}[{expr}];")

        elif inst.op == "init_acc":
            token, ref = inst.args
            kind, _idx = _parse_lars_label_ref(ref)
            if kind != "o":
                raise ValueError("forward init_acc expects out[]")
            ap(f"            {acc_expr(str(token))} = scalar_t(0);")

        elif inst.op == "fma_u1d_placed":
            out_idx, out_token, rx, ry, rw, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            value = (
                f"scalar_t({c}) * ({operand_expr(str(rw))} * "
                f"{operand_expr(str(rx))}) * {operand_expr(str(ry))}"
            )
            if str(out_token).startswith("d:"):
                emit_out_write(int(out_idx), value)
            else:
                ap(f"            {acc_expr(str(out_token))} += {value};")

        elif inst.op == "store_acc_placed":
            ref, token = inst.args
            kind, idx = _parse_lars_label_ref(ref)
            if kind != "o":
                raise ValueError("forward store_acc_placed expects out[]")
            emit_out_write(int(idx), acc_expr(str(token)))

        elif inst.op == "load_acc":
            reg, ref = inst.args
            kind, _idx = _parse_lars_label_ref(ref)
            if kind != "o":
                raise ValueError("load_acc expects an output label")
            ap(f"            {reg} = scalar_t(0);")

        elif inst.op == "fma_u1d_resident":
            out_idx, rx, ry, rw, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            ap(f"            out_acc_v_{int(out_idx)} += scalar_t({c}) * ({rw} * {rx}) * {ry};")

        elif inst.op == "fma_u1d_direct":
            out_idx, rx, ry, rw, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            emit_out_write(int(out_idx), f"scalar_t({c}) * ({rw} * {rx}) * {ry}")

        elif inst.op == "fma_u1d":
            ro, rx, ry, rw, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            ap(f"            {ro} += scalar_t({c}) * ({rw} * {rx}) * {ry};")

        elif inst.op == "store_acc":
            ref, reg = inst.args
            kind, idx = _parse_lars_label_ref(ref)
            if kind != "o":
                raise ValueError("store_acc expects an output label")
            emit_out_write(int(idx), str(reg))

        elif inst.op in ("release", "release_shared"):
            pass
        else:
            raise ValueError(f"Unsupported LARS instruction op: {inst.op}")

    if resident_out_indices:
        ap("")
        ap("            // resident output accumulator writeback")
    for out_idx in resident_out_indices:
        emit_out_write(int(out_idx), f"out_acc_v_{out_idx}")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32(a, b)) return false;")
    ap("    return mul_fits_int32(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_fwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t out_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool out_ok = mul3_fits_int32(out_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && out_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    size_t lars_shared_bytes = (size_t){shared_slots} * (size_t)block.x * sizeof(scalar_t);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, lars_shared_bytes, stream>>>(")
    ap("        w, x, y, out,")
    ap("        src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_fwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    ap("            w, x, y, out, src_idx, dst_idx,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    ap("            w, x, y, out, src_idx, dst_idx,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    ap("        w, x, y, out, src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def _make_lars_paths_from_uniform1d_lists(
    i_list: List[int],
    j_list: List[int],
    k_list: List[int],
    v_list: List[int],
    coeff_list: List[float],
    *,
    path_semantics: str,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Convert external Uniform1D path indices to LARS-native tuples.

    path_semantics="wxy" keeps compatibility with the previous emitter in this
    file, where external (i,j,k) means w[i], x[j], y[k]. Since LARS expects
    x[i], y[j], w[k], we feed it as (j,k,i,v,c).

    path_semantics="xyw" uses LARS-native semantics directly: x[i], y[j], w[k].
    """
    if path_semantics not in ("wxy", "xyw"):
        raise ValueError("path_semantics must be 'wxy' or 'xyw'")
    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

    paths: List[Tuple[int, int, int, int, float]] = []
    for i, j, k, v, c in zip(i_list, j_list, k_list, v_list, coeff_list):
        i = int(i); j = int(j); k = int(k); v = int(v); c = float(c)
        if path_semantics == "wxy":
            paths.append((j, k, i, v, c))  # LARS native: x[j], y[k], w[i]
        else:
            paths.append((i, j, k, v, c))  # LARS native: x[i], y[j], w[k]
    return paths

def generate_code_uniform1d_fwd_with_scheduler(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    mode: str = "u,u,,u",
    out_path: str = "generated_uniform1d_fwd_lars.cu",
    kernel_name: str = "stp_codegen_lars",
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    block_size: int = 32,
    placement_config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]] = None,
):
    """
    Generate one LARS forward CUDA implementation.

    API compatibility, but this slim scheduler derives an unbounded virtual
    input-register file from all distinct x/y/w labels and emits a single
    single candidate.
    """
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    lars_paths = _make_lars_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )

    cand_name = "lars_all_inputs"

    scheduler = LARSUniform1DScheduler(
        paths=lars_paths,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name=cand_name,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    schedule_result = scheduler.schedule()
    _dump_and_print_schedule_label_stats(
        operation="uniform1d_fwd",
        candidate_tag=cand_name,
        path_order=schedule_result.path_order,
        labels_by_path=_stats_labels_uniform1d_fwd(scheduler.paths),
    )
    resolved_placement = _resolve_lars_placement_config(placement_config)
    schedule_result = apply_lars_lifetime_placement(
        schedule_result,
        schedule_kind="fwd",
        config=resolved_placement,
    )
    dump_lars_variable_stats(
        schedule_result,
        print_stats=resolved_placement.stats_print,
        out_path=resolved_placement.stats_path,
    )

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"

    if kernel_name and kernel_name != "stp_codegen_lars":
        base_kernel_name = f"{kernel_name}_{cand_name}"
    else:
        safe_cand = cand_name.replace("-", "_").replace(".", "_")
        base_kernel_name = (
            f"uniform1d_{safe_cand}_u{u_dim}_path{P}_"
            f"{mode_str}_{layout_tag}_fwd"
        )

    code = emit_fused_fwd_kernel_from_lars_schedule(
        schedule_result,
        kernel_name=base_kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        u_dim=u_dim,
        x_dim=None,
        y_dim=None,
        w_dim=None,
        v_dim=None,
        block_size=int(block_size),
    )

    code = code + "\n" + emit_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")

    if return_schedule:
        return {
            "name": cand_name,
            "code": code,
            "schedule": schedule_result,
            "config": {
                "name": cand_name,
                "auto_fallback_when_full": False,
            },
            "profile": schedule_result.profile,
        }

    return code

# =============================================================================
# LARS backward scheduler/codegen
# =============================================================================

BWD_OUTPUT_KINDS = {"gw", "gx", "gy"}

@dataclass(frozen=True)
class U1DBwdPath:
    """
    Backward path for forward expression:

        out[v] += w[i] * x[j] * y[k] * c

    Per path, backward contributes:
        grad_w[i] += c * grad_out[v] * x[j] * y[k]
        grad_x[j] += c * grad_out[v] * w[i] * y[k]
        grad_y[k] += c * grad_out[v] * w[i] * x[j]
    """
    pid: int
    i: int  # w index
    j: int  # x index
    k: int  # y index
    v: int  # grad_out/out index
    c: float
    need_grad_w: bool = True

    @property
    def labels(self) -> Tuple[Label, ...]:
        # LARS schedules only input operands.  Backward accumulators
        # (gw/gx/gy) are full-resident emitter variables, so they are never
        # reload-managed candidates.
        return (
            ("w", self.i),
            ("x", self.j),
            ("y", self.k),
            ("go", self.v),
        )


class LARSUniform1DBwdScheduler(LARSUniform1DScheduler):
    """
    LARS scheduler for fused Uniform1D backward.

    It reuses the forward LARS machinery, but a path now has four input labels
    (w, x, y, grad_out) and two or three output accumulators
    (grad_x, grad_y, and optionally grad_w).
    """

    output_kinds = BWD_OUTPUT_KINDS

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        *,
        need_grad_w: bool = True,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        # Backward reuses the generic forward scheduler helpers such as
        # _choose_fireable_path().  Those helpers now branch on path_kind for
        # STC, so make the backward kind explicit and keep it on the normal
        # Uniform1D branch.
        self.path_kind = "u1d_bwd"
        self.need_grad_w = bool(need_grad_w)
        self.paths: List[U1DBwdPath] = [
            U1DBwdPath(pid=p, i=i, j=j, k=k, v=v, c=c, need_grad_w=self.need_grad_w)
            for p, (i, j, k, v, c) in enumerate(paths)
        ]

        # Backward input labels are unbounded: virtual registers are allocated on demand.

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
        self.debug = bool(debug)

        self.label_to_paths: Dict[Label, Set[int]] = defaultdict(set)
        for path in self.paths:
            for lab in path.labels:
                self.label_to_paths[lab].add(path.pid)

        self.path_label_sets: List[Set[Label]] = [set(p.labels) for p in self.paths]
        self.unscheduled: Set[int] = set(p.pid for p in self.paths)

        self.remaining_uses: Counter[Label] = Counter()
        for p in self.paths:
            for lab in p.labels:
                self.remaining_uses[lab] += 1

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = []
        self._next_reg_id = 0

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}

        self.profile_enabled = bool(profile)
        self.profile_name = str(profile_name or self.__class__.__name__)
        self.profile_interval = int(profile_interval)
        self.profile_seconds = float(profile_seconds)
        self.profile_print = bool(profile_print)
        self._profile_t0 = perf_counter()
        self._profile_last_print_t = self._profile_t0
        self._profile_last_print_done = 0
        self._profile_loop_iter = 0
        self._profile_select_rounds = 0
        self._profile_fire_rounds = 0
        self._profile_load_rounds = 0
        self._profile_records: List[Dict[str, Any]] = []
        self._profile_last_event = ""
        self._profile_last_reason = ""

    def _path(self, pid: int) -> U1DBwdPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        if kind == "go":
            return f"grad_out[{idx}]"
        if kind == "gw":
            return f"grad_w[{idx}]"
        if kind == "gx":
            return f"grad_x[{idx}]"
        if kind == "gy":
            return f"grad_y[{idx}]"
        return f"{kind}[{idx}]"

    def _is_output_label(self, lab: Label) -> bool:
        return lab[0] in self.output_kinds

    def _emit_load_inst(self, lab: Label, reg: str, reason: str) -> None:
        if self._is_output_label(lab):
            self.instructions.append(
                Inst("load_acc", (reg, self._label_name(lab)), reason)
            )
        else:
            self.instructions.append(
                Inst("load", (reg, self._label_name(lab)), reason)
            )
        self.reloads += 1

    def _store_and_release(self, lab: Label, reason: str = "") -> None:
        if lab not in self.live:
            return

        reg = self.reg_of[lab]
        if self._is_output_label(lab):
            if lab in self.dirty_outputs:
                self.instructions.append(
                    Inst("store_acc", (self._label_name(lab), reg), reason)
                )
                self.dirty_outputs.discard(lab)
            else:
                self.instructions.append(
                    Inst("release", (reg, self._label_name(lab)), reason)
                )
        else:
            self.instructions.append(
                Inst("release", (reg, self._label_name(lab)), reason)
            )

        self.live.remove(lab)
        self.reg_of.pop(lab, None)
        self.free_regs.insert(0, reg)
        self._invalidate_score_cache()


    def _priority(self, lab: Label) -> int:
        acc_bonus = 1 if self._is_output_label(lab) else 0
        return self.remaining_uses[lab] + acc_bonus

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)

        lw = ("w", p.i)
        lx = ("x", p.j)
        ly = ("y", p.k)
        lgo = ("go", p.v)

        rw = self.reg_of[lw]
        rx = self.reg_of[lx]
        ry = self.reg_of[ly]
        rgo = self.reg_of[lgo]

        self.instructions.append(
            Inst(
                "bwd_fma_resident",
                (p.i, p.j, p.k, rw, rx, ry, rgo, p.c, self.need_grad_w),
                (
                    f"path#{pid}: "
                    f"gw[{p.i}] += go[{p.v}]*x[{p.j}]*y[{p.k}], "
                    f"gx[{p.j}] += w[{p.i}]*go[{p.v}]*y[{p.k}], "
                    f"gy[{p.k}] += w[{p.i}]*go[{p.v}]*x[{p.j}]"
                ),
            )
        )

        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        for lab in p.labels:
            self.remaining_uses[lab] -= 1

        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")


BwdPairKey = Tuple[Label, Label]  # (w-label, grad_out-label)




def _parse_bwd_lars_label_ref(ref: str) -> Tuple[str, int]:
    if not isinstance(ref, str) or "[" not in ref or not ref.endswith("]"):
        raise ValueError(f"Bad LARS backward label ref: {ref!r}")
    kind, rest = ref.split("[", 1)
    idx = int(rest[:-1])
    aliases = {
        "grad_out": "go",
        "grad_w": "gw",
        "grad_x": "gx",
        "grad_y": "gy",
    }
    kind = aliases.get(kind, kind)
    if kind not in ("w", "x", "y", "go", "gw", "gx", "gy"):
        raise ValueError(f"Bad backward label kind in ref: {ref!r}")
    return kind, idx


def _bwd_label_index_expr(
    kind: str,
    idx: int,
    *,
    mode_scalar_y: bool,
    u_dim: Optional[int],
    iw_dim: Optional[int],
    ix_dim: Optional[int],
    ky_dim: Optional[int],
    v_dim: Optional[int],
) -> str:
    if kind == "w":
        if iw_dim is not None and idx >= iw_dim:
            raise ValueError(f"w index {idx} out of iw_dim={iw_dim}")
        if u_dim is not None:
            return f"w_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"w_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "gw":
        if iw_dim is not None and idx >= iw_dim:
            raise ValueError(f"grad_w index {idx} out of iw_dim={iw_dim}")
        if u_dim is not None:
            return f"gw_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"gw_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "x":
        if ix_dim is not None and idx >= ix_dim:
            raise ValueError(f"x index {idx} out of ix_dim={ix_dim}")
        if u_dim is not None:
            return f"x_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"x_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "gx":
        if ix_dim is not None and idx >= ix_dim:
            raise ValueError(f"grad_x index {idx} out of ix_dim={ix_dim}")
        if u_dim is not None:
            return f"gx_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"gx_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "y":
        if ky_dim is not None and idx >= ky_dim:
            raise ValueError(f"y index {idx} out of ky_dim={ky_dim}")
        if mode_scalar_y:
            return f"y_base + (index_t){idx}"
        if u_dim is not None:
            return f"y_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"y_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "gy":
        if ky_dim is not None and idx >= ky_dim:
            raise ValueError(f"grad_y index {idx} out of ky_dim={ky_dim}")
        if mode_scalar_y:
            return f"gy_base + (index_t){idx}"
        if u_dim is not None:
            return f"gy_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"gy_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "go":
        if v_dim is not None and idx >= v_dim:
            raise ValueError(f"grad_out index {idx} out of v_dim={v_dim}")
        if u_dim is not None:
            return f"go_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"go_base + (index_t){idx} * (index_t)U + (index_t)u"

    raise ValueError(f"Bad backward label kind: {kind}")


def emit_fused_bwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """
    Emit CUDA/HIP-compatible fused backward code from a backward LARS schedule.

    Supported instruction ops:
      - load/load_acc/release/store_acc
      - bwd_fma
    """
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    reg_count = _max_lars_reg_count_any(schedule_result)
    shared_slots = int(getattr(schedule_result, "shared_slots", 0) or 0)

    # ------------------------------------------------------------------
    # Mixed accumulator placement for backward gradients.
    #                        as a scalar local variable/register.
    #                        registers and demote the rest to per-thread
    #                        write back once at the end, preserving atomic
    #                        coalescing compared with path-level atomicAdd.
    # ------------------------------------------------------------------
    gw_use: Counter[int] = Counter()
    gx_use: Counter[int] = Counter()
    gy_use: Counter[int] = Counter()
    first_seen: Dict[Tuple[str, int], int] = {}

    def _note_acc(kind: str, idx: int, inst_id: int) -> None:
        first_seen.setdefault((kind, int(idx)), int(inst_id))

    for inst_id, inst in enumerate(schedule_result.instructions):
        if inst.op == "bwd_fma_resident":
            wi, xj, yk = map(int, inst.args[:3])
            if need_grad_w:
                gw_use[wi] += 1
                _note_acc("gw", wi, inst_id)
            gx_use[xj] += 1
            gy_use[yk] += 1
            _note_acc("gx", xj, inst_id)
            _note_acc("gy", yk, inst_id)

    all_accs: Set[Tuple[str, int]] = set()
    if need_grad_w:
        all_accs |= {("gw", int(i)) for i in gw_use}
    all_accs |= {("gx", int(j)) for j in gx_use}
    all_accs |= {("gy", int(k)) for k in gy_use}

    def _acc_use_count(acc: Tuple[str, int]) -> int:
        kind, idx = acc
        if kind == "gw":
            return int(gw_use.get(idx, 0))
        if kind == "gx":
            return int(gx_use.get(idx, 0))
        if kind == "gy":
            return int(gy_use.get(idx, 0))
        return 0

    def _acc_score(acc: Tuple[str, int]) -> Tuple[float, int, int, int, str, int]:
        # gx/gy have high value because local accumulation reduces
        # global atomicAdd count.  gy is slightly favored in scalar-y mode
        # because writeback also needs a warp reduction.
        kind, idx = acc
        cnt = _acc_use_count(acc)
        if kind == "gy":
            weight = 5.0 if mode_scalar_y else 4.0
            kind_rank = 3
        elif kind == "gx":
            weight = 4.0
            kind_rank = 2
        else:  # gw uses non-atomic writeback, so its register value is lower.
            weight = 2.0
            kind_rank = 1
        return (float(cnt) * weight, cnt, -int(first_seen.get(acc, 10**9)), kind_rank, kind, -idx)

    # All backward accumulators are resident scalar locals/registers.  Shared-memory
    # accumulator demotion has been removed, but the emitter still needs stable
    # sorted index lists for declaring and writing back those scalar locals.
    reg_accs: Set[Tuple[str, int]] = set(all_accs)
    resident_gw_indices_sorted = sorted(idx for kind, idx in reg_accs if kind == "gw")
    resident_gx_indices_sorted = sorted(idx for kind, idx in reg_accs if kind == "gx")
    resident_gy_indices_sorted = sorted(idx for kind, idx in reg_accs if kind == "gy")

    def _is_reg_acc(kind: str, idx: int) -> bool:
        return (kind, int(idx)) in reg_accs

    def _reg_acc_name(kind: str, idx: int) -> str:
        if kind == "gw":
            return f"gw_acc_i_{int(idx)}"
        if kind == "gx":
            return f"gx_acc_j_{int(idx)}"
        if kind == "gy":
            return f"gy_acc_k_{int(idx)}"
        raise ValueError(f"Bad accumulator kind: {kind}")

    def _emit_bwd_acc_update(kind: str, idx: int, value_expr: str) -> None:
        if _is_reg_acc(kind, idx):
            ap(f"            {_reg_acc_name(kind, idx)} += {value_expr};")
        else:
            # This path is reachable only if need_grad_w=False and kind==gw.
            pass

    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    def shared_expr(token: str) -> str:
        if not isinstance(token, str) or not token.startswith("s"):
            raise ValueError(f"Bad shared token: {token!r}")
        slot = int(token[1:])
        return f"lars_smem[(size_t){slot} * (size_t)blockDim.x + (size_t)tid]"

    def operand_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        if token.startswith("d:"):
            kind, idx = _parse_bwd_lars_label_ref(token[2:])
            if kind in BWD_OUTPUT_KINDS:
                raise ValueError("backward output cannot be a direct input")
            expr = _bwd_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            arr = "grad_out" if kind == "go" else kind
            return f"{arr}[{expr}]"
        raise ValueError(f"Bad backward operand token: {token!r}")

    def acc_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        raise ValueError(f"Bad backward accumulator token: {token!r}")

    def emit_grad_writeback(kind: str, idx: int, value_expr: str, suffix: str) -> None:
        expr = _bwd_label_index_expr(
            kind, int(idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        if kind == "gw":
            if not need_grad_w:
                raise ValueError("grad_w writeback requested with need_grad_w=False")
            ap(f"            grad_w[{expr}] = {value_expr};")
        elif kind == "gx":
            ap(f"            atomicAdd(&grad_x[{expr}], {value_expr});")
        elif kind == "gy":
            # Portable correctness path for scalar-y and vector-y:
            # each valid u lane contributes directly.  This deliberately avoids
            # warp/wavefront shuffle semantics (CUDA warp32 vs HIP wave32/wave64)
            # and is also correct for a partial final 32-lane tile.
            ap(f"            atomicAdd(&grad_y[{expr}], {value_expr});")
        else:
            raise ValueError(f"Bad backward accumulator kind: {kind}")

    def emit_placed_update(kind: str, idx: int, token: str, value_expr: str, suffix: str) -> None:
        if str(token).startswith("d:"):
            emit_grad_writeback(kind, idx, value_expr, suffix)
        else:
            ap(f"            {acc_expr(str(token))} += {value_expr};")

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")
    ap("// grad_y scalar reduction intentionally uses per-lane atomicAdd.")
    ap("// No warp/wavefront shuffle is used, so CUDA and HIP share one path.")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    if need_grad_w:
        ap("    scalar_t* __restrict__ grad_w,")
    ap("    scalar_t* __restrict__ grad_x,")
    ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    if shared_slots:
        ap("    extern __shared__ unsigned char lars_smem_raw[];")
        ap("    scalar_t* lars_smem = reinterpret_cast<scalar_t*>(lars_smem_raw);")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
    ap("    const int e_orig = e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e_orig];")
    else:
        ap("    const int src = e_orig;")
    if use_scatter:
        ap("    const int dst = dst_idx[e_orig];")
    else:
        ap("    const int dst = e_orig;")
    ap("")
    ap("    const int x_row = " + ("src;" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src;" if use_y_src else "e_orig;"))
    ap("    const int go_row = " + ("dst;" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base  = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
        if need_grad_w:
            ap(f"    const index_t gw_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base  = (index_t)w_row * (index_t)Iw * (index_t)U;")
        if need_grad_w:
            ap("    const index_t gw_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base  = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
        ap(f"    const index_t gx_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base  = (index_t)x_row * (index_t)Ix * (index_t)U;")
        ap("    const index_t gx_base = (index_t)x_row * (index_t)Ix * (index_t)U;")

    if mode_scalar_y:
        ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky;")
        ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base  = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
            ap(f"    const index_t gy_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky * (index_t)U;")
            ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky * (index_t)U;")

    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t go_base = (index_t)go_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t go_base = (index_t)go_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    for gw_idx in resident_gw_indices_sorted:
        ap(f"            scalar_t gw_acc_i_{gw_idx} = scalar_t(0);")
    for gx_idx in resident_gx_indices_sorted:
        ap(f"            scalar_t gx_acc_j_{gx_idx} = scalar_t(0);")
    for gy_idx in resident_gy_indices_sorted:
        ap(f"            scalar_t gy_acc_k_{gy_idx} = scalar_t(0);")
    if resident_gw_indices_sorted or resident_gx_indices_sorted or resident_gy_indices_sorted:
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op in ("load", "load_shared"):
            token, ref = inst.args
            kind, idx = _parse_bwd_lars_label_ref(ref)
            if kind in BWD_OUTPUT_KINDS:
                raise ValueError("input load must not target backward output labels")
            expr = _bwd_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            arr = "grad_out" if kind == "go" else kind
            lhs = str(token) if inst.op == "load" else shared_expr(str(token))
            ap(f"            {lhs} = {arr}[{expr}];")

        elif inst.op == "init_acc":
            token, ref = inst.args
            kind, _idx = _parse_bwd_lars_label_ref(ref)
            if kind not in BWD_OUTPUT_KINDS:
                raise ValueError("backward init_acc expects grad_w/grad_x/grad_y")
            ap(f"            {acc_expr(str(token))} = scalar_t(0);")

        elif inst.op == "bwd_fma_placed":
            wi, xj, yk, gw_token, gx_token, gy_token, rw, rx, ry, rgo, coeff, inst_need_grad_w = inst.args
            c = _fmt_lars_float(float(coeff))
            rw_e = operand_expr(str(rw))
            rx_e = operand_expr(str(rx))
            ry_e = operand_expr(str(ry))
            rgo_e = operand_expr(str(rgo))
            if bool(inst_need_grad_w):
                emit_placed_update(
                    "gw", int(wi), str(gw_token),
                    f"scalar_t({c}) * {rgo_e} * {rx_e} * {ry_e}",
                    f"direct_gw_{inst_id}",
                )
            emit_placed_update(
                "gx", int(xj), str(gx_token),
                f"scalar_t({c}) * ({rw_e} * {rgo_e}) * {ry_e}",
                f"direct_gx_{inst_id}",
            )
            emit_placed_update(
                "gy", int(yk), str(gy_token),
                f"scalar_t({c}) * ({rw_e} * {rgo_e}) * {rx_e}",
                f"direct_gy_{inst_id}",
            )

        elif inst.op == "store_acc_placed":
            ref, token = inst.args
            kind, idx = _parse_bwd_lars_label_ref(ref)
            if kind not in BWD_OUTPUT_KINDS:
                raise ValueError("store_acc_placed expects a backward output label")
            emit_grad_writeback(
                kind, int(idx), acc_expr(str(token)), f"placed_{kind}_{idx}_{inst_id}"
            )

        elif inst.op == "load_acc":
            reg, ref = inst.args
            kind, _idx = _parse_bwd_lars_label_ref(ref)
            if kind not in BWD_OUTPUT_KINDS:
                raise ValueError("load_acc expects a backward output label")
            ap(f"            {reg} = scalar_t(0);")

        elif inst.op == "bwd_fma_resident":
            wi, xj, yk, rw, rx, ry, rgo, coeff, inst_need_grad_w = inst.args
            c = _fmt_lars_float(float(coeff))
            if bool(inst_need_grad_w):
                _emit_bwd_acc_update("gw", int(wi), f"scalar_t({c}) * {rgo} * {rx} * {ry}")
            _emit_bwd_acc_update("gx", int(xj), f"scalar_t({c}) * ({rw} * {rgo}) * {ry}")
            _emit_bwd_acc_update("gy", int(yk), f"scalar_t({c}) * ({rw} * {rgo}) * {rx}")

        elif inst.op == "bwd_fma":
            # Backward-compatible support for older schedules that kept
            # output accumulators inside the LARS register file.
            rgw, rgx, rgy, rw, rx, ry, rgo, coeff, inst_need_grad_w = inst.args
            c = _fmt_lars_float(float(coeff))
            if bool(inst_need_grad_w):
                ap(f"            {rgw} += scalar_t({c}) * {rgo} * {rx} * {ry};")
            ap(f"            {rgx} += scalar_t({c}) * ({rw} * {rgo}) * {ry};")
            ap(f"            {rgy} += scalar_t({c}) * ({rw} * {rgo}) * {rx};")

        elif inst.op == "store_acc":
            ref, reg = inst.args
            kind, idx = _parse_bwd_lars_label_ref(ref)
            expr = _bwd_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            if kind == "gw":
                if not need_grad_w:
                    raise ValueError("schedule stores grad_w but need_grad_w=False")
                # grad_w is owned by the current w_row in this generated
                # backward path, matching the baseline fused implementation:
                # use a plain store rather than atomicAdd.
                ap(f"            grad_w[{expr}] = {reg};")
            elif kind == "gx":
                ap(f"            atomicAdd(&grad_x[{expr}], {reg});")
            elif kind == "gy":
                # One atomic contribution per valid u lane.  Do not reduce with
                # hardware warp/wavefront shuffle here.
                ap(f"            atomicAdd(&grad_y[{expr}], {reg});")
            else:
                raise ValueError("store_acc expects a backward output label")

        elif inst.op in ("release", "release_shared"):
            pass
        else:
            raise ValueError(f"Unsupported backward LARS instruction op: {inst.op}")

    if resident_gw_indices_sorted or resident_gx_indices_sorted or resident_gy_indices_sorted:
        ap("")
        ap("            // register-resident backward accumulator writeback")

    # Register-resident writeback.
    for gw_idx in resident_gw_indices_sorted:
        expr = _bwd_label_index_expr(
            "gw", int(gw_idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        ap(f"            grad_w[{expr}] = gw_acc_i_{gw_idx};")
    for gx_idx in resident_gx_indices_sorted:
        expr = _bwd_label_index_expr(
            "gx", int(gx_idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        ap(f"            atomicAdd(&grad_x[{expr}], gx_acc_j_{gx_idx});")
    for gy_idx in resident_gy_indices_sorted:
        expr = _bwd_label_index_expr(
            "gy", int(gy_idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        # Portable per-lane accumulation; valid for both scalar-y and vector-y.
        ap(f"            atomicAdd(&grad_y[{expr}], gy_acc_k_{gy_idx});")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32_bwd(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32_bwd(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32_bwd(a, b)) return false;")
    ap("    return mul_fits_int32_bwd(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_bwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32_bwd((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32_bwd(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32_bwd(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32_bwd(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool go_ok = mul3_fits_int32_bwd(go_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && go_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    size_t lars_shared_bytes = (size_t){shared_slots} * (size_t)block.x * sizeof(scalar_t);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, lars_shared_bytes, stream>>>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_bwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def emit_lars_bwd_launcher(
    bundle_name: str,
    mode: str,
    *,
    need_grad_w: bool,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
    if mode == "u,u,,u":
        y_comment = "[B,Ky,1] or [S,Ky,1]"
        gy_alloc = "    auto grad_y = torch::zeros({y.size(0), Ky, 1}, y.options());"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");'
    elif mode == "u,u,u,u":
        y_comment = "[B,Ky,U] or [S,Ky,U]"
        gy_alloc = "    auto grad_y = torch::zeros_like(y);"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == U, "y.size(2) must equal U for mode u,u,u,u");'
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    param_lines = [
        "    torch::Tensor w",
        "    torch::Tensor x",
        "    torch::Tensor y",
        "    torch::Tensor grad_out",
    ]
    if use_x_src or use_y_src:
        param_lines.append("    torch::Tensor src_idx")
    if use_scatter:
        param_lines.append("    torch::Tensor dst_idx")
    param_lines.append("    int64_t V64")
    params = ",\n".join(param_lines)

    src_check = ""
    if use_x_src or use_y_src:
        src_check = r'''
    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''
    dst_check = ""
    if use_scatter:
        dst_check = r'''
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''

    if use_x_src or use_y_src:
        b_expr = "src_idx.size(0)"
        src_numel_check = '    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");'
    elif use_scatter:
        b_expr = "dst_idx.size(0)"
        src_numel_check = ""
    else:
        b_expr = "grad_out.size(0)"
        src_numel_check = ""

    dst_numel_check = '    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");' if use_scatter else ""

    blist_logic = ""

    launch_src_arg = "(const int32_t*)src_idx.data_ptr<int32_t>()," if (use_x_src or use_y_src) else "nullptr,"
    launch_dst_arg = "(const int32_t*)dst_idx.data_ptr<int32_t>()," if use_scatter else "nullptr,"

    grad_w_alloc = "    auto grad_w = torch::zeros_like(w);\n" if need_grad_w else ""
    grad_w_launch_arg = "                (scalar_t*)grad_w.data_ptr<scalar_t>(),\n" if need_grad_w else ""
    ret_expr = "return {grad_w, grad_x, grad_y};" if need_grad_w else "return {grad_x, grad_y};"

    return rf'''

std::vector<torch::Tensor> launcher_{bundle_name}(
{params})
{{
    // Expected tensors:
    //   w        : [WB,Iw,U], WB can be 1 or B
    //   x        : [S,Ix,U] or [B,Ix,U]
    //   y        : {y_comment}
    //   grad_out : [B,V,U] or [S,V,U], matching forward output layout

    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x/y/grad_out must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");
{src_check}{dst_check}

    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x.dim() == 3, "x must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");

    int B  = (int){b_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x.size(0);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
    {y_check_u}
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
{src_numel_check}
{dst_numel_check}

{blist_logic}
{grad_w_alloc}    auto grad_x = torch::zeros_like(x);
{gy_alloc}

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{

        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");

        launch_{bundle_name}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
{grad_w_launch_arg}                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    }});

    GPU_KERNEL_LAUNCH_CHECK();

    {ret_expr}
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} backward LARS fused jit impl");
}}
'''


def _make_lars_bwd_paths_from_uniform1d_lists(
    i_list: List[int],
    j_list: List[int],
    k_list: List[int],
    v_list: List[int],
    coeff_list: List[float],
    *,
    path_semantics: str,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Convert external Uniform1D path indices to backward-LARS tuples.

    Backward LARS native path tuple is:
        (w_index, x_index, y_index, out_index, coeff)

    path_semantics="wxy":
        external (i,j,k) means w[i], x[j], y[k].
    path_semantics="xyw":
        external (i,j,k) means x[i], y[j], w[k].
    """
    if path_semantics not in ("wxy", "xyw"):
        raise ValueError("path_semantics must be 'wxy' or 'xyw'")
    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

    paths: List[Tuple[int, int, int, int, float]] = []
    for i, j, k, v, c in zip(i_list, j_list, k_list, v_list, coeff_list):
        i = int(i); j = int(j); k = int(k); v = int(v); c = float(c)
        if path_semantics == "wxy":
            paths.append((i, j, k, v, c))
        else:
            paths.append((k, i, j, v, c))
    return paths








# =============================================================================
# Split backward LARS codegen: grad_w / grad_x / grad_y in separate kernels
# =============================================================================

BWD_SPLIT_KINDS = {"gw", "gx", "gy"}


@dataclass(frozen=True)
class U1DBwdSplitPath:
    """
    Backward path used by split-gradient LARS scheduling.

    grad_kind="gw": grad_w[i] += c * grad_out[v] * x[j] * y[k]
    grad_kind="gx": grad_x[j] += c * w[i]        * grad_out[v] * y[k]
    grad_kind="gy": grad_y[k] += c * w[i]        * grad_out[v] * x[j]
    """
    pid: int
    i: int
    j: int
    k: int
    v: int
    c: float
    grad_kind: str

    @property
    def target_index(self) -> int:
        if self.grad_kind == "gw":
            return int(self.i)
        if self.grad_kind == "gx":
            return int(self.j)
        if self.grad_kind == "gy":
            return int(self.k)
        raise ValueError(f"Bad grad_kind: {self.grad_kind}")

    @property
    def labels(self) -> Tuple[Label, ...]:
        if self.grad_kind == "gw":
            return (("x", self.j), ("y", self.k), ("go", self.v))
        if self.grad_kind == "gx":
            return (("w", self.i), ("y", self.k), ("go", self.v))
        if self.grad_kind == "gy":
            return (("w", self.i), ("x", self.j), ("go", self.v))
        raise ValueError(f"Bad grad_kind: {self.grad_kind}")


class LARSUniform1DBwdSplitScheduler(LARSUniform1DScheduler):
    """
    Per-gradient backward scheduler.  It keeps the no-spill LARS label
    selection policy, but builds labels for a single gradient target only.
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        *,
        grad_kind: str,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        if grad_kind not in BWD_SPLIT_KINDS:
            raise ValueError(f"grad_kind must be one of {sorted(BWD_SPLIT_KINDS)}, got {grad_kind!r}")
        self.grad_kind = str(grad_kind)
        self.paths: List[U1DBwdSplitPath] = [
            U1DBwdSplitPath(pid=p, i=i, j=j, k=k, v=v, c=c, grad_kind=self.grad_kind)
            for p, (i, j, k, v, c) in enumerate(paths)
        ]

        # Split-backward input labels are unbounded: virtual registers are allocated on demand.

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
        self.debug = bool(debug)

        self.label_to_paths: Dict[Label, Set[int]] = defaultdict(set)
        for path in self.paths:
            for lab in path.labels:
                self.label_to_paths[lab].add(path.pid)

        self.path_label_sets: List[Set[Label]] = [set(p.labels) for p in self.paths]
        self.unscheduled: Set[int] = set(p.pid for p in self.paths)

        self.remaining_uses: Counter[Label] = Counter()
        for p in self.paths:
            for lab in p.labels:
                self.remaining_uses[lab] += 1

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = []
        self._next_reg_id = 0

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}

        self.profile_enabled = bool(profile)
        self.profile_name = str(profile_name or f"{self.__class__.__name__}_{self.grad_kind}")
        self.profile_interval = int(profile_interval)
        self.profile_seconds = float(profile_seconds)
        self.profile_print = bool(profile_print)
        self._profile_t0 = perf_counter()
        self._profile_last_print_t = self._profile_t0
        self._profile_last_print_done = 0
        self._profile_loop_iter = 0
        self._profile_select_rounds = 0
        self._profile_fire_rounds = 0
        self._profile_load_rounds = 0
        self._profile_records: List[Dict[str, Any]] = []
        self._profile_last_event = ""
        self._profile_last_reason = ""

    def _path(self, pid: int) -> U1DBwdSplitPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        if kind == "go":
            return f"grad_out[{idx}]"
        return f"{kind}[{idx}]"

    def _priority(self, lab: Label) -> int:
        return self.remaining_uses[lab]

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)
        regs: Dict[str, str] = {}
        for kind, idx in p.labels:
            regs[kind] = self.reg_of[(kind, idx)]

        self.instructions.append(
            Inst(
                "bwd_split_fma_resident",
                (
                    self.grad_kind,
                    p.target_index,
                    regs.get("w", ""),
                    regs.get("x", ""),
                    regs.get("y", ""),
                    regs.get("go", ""),
                    p.c,
                ),
                (
                    f"path#{pid}: split {self.grad_kind}; "
                    f"w[{p.i}], x[{p.j}], y[{p.k}], go[{p.v}], c={p.c}"
                ),
            )
        )

        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        for lab in p.labels:
            self.remaining_uses[lab] -= 1

        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")


def _split_grad_acc_name(grad_kind: str, idx: int) -> str:
    if grad_kind == "gw":
        return f"gw_acc_i_{int(idx)}"
    if grad_kind == "gx":
        return f"gx_acc_j_{int(idx)}"
    if grad_kind == "gy":
        return f"gy_acc_k_{int(idx)}"
    raise ValueError(f"Bad grad_kind: {grad_kind}")


def emit_lars_bwd_split_preamble() -> str:
    return r'''
#include <stdint.h>
#include <torch/extension.h>
#include <vector>
#include <cstdint>

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #include <ATen/hip/HIPContext.h>
  #include <c10/hip/HIPGuard.h>
  using gpuStream_t = hipStream_t;
  #define getCurrentGPUStream at::hip::getCurrentHIPStream
  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()
#else
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <ATen/cuda/CUDAContext.h>
  using gpuStream_t = cudaStream_t;
  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream
  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

using GPU_Guard = c10::DeviceGuard;

// grad_y scalar reduction intentionally uses per-lane atomicAdd.
// This avoids any dependency on CUDA warp32 / HIP wave32-or-wave64 semantics.

static inline bool mul_fits_int32_bwd_split(int64_t a, int64_t b) {
    if (a < 0 || b < 0) return false;
    constexpr int64_t LIM = 2147483647LL;
    if (a == 0 || b == 0) return true;
    return a <= LIM / b;
}

static inline bool mul3_fits_int32_bwd_split(int64_t a, int64_t b, int64_t c) {
    if (!mul_fits_int32_bwd_split(a, b)) return false;
    return mul_fits_int32_bwd_split(a * b, c);
}

static inline bool should_use_int32_index_bwd_split(
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)
{
    bool w_ok = mul3_fits_int32_bwd_split((int64_t)WB, (int64_t)Iw, (int64_t)U);
    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;
    bool x_ok = mul3_fits_int32_bwd_split(x_dim0, (int64_t)Ix, (int64_t)U);
    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;
    bool y_ok = mode_scalar_y ? mul_fits_int32_bwd_split(y_dim0, (int64_t)Ky)
                              : mul3_fits_int32_bwd_split(y_dim0, (int64_t)Ky, (int64_t)U);
    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;
    bool go_ok = mul3_fits_int32_bwd_split(go_dim0, (int64_t)V, (int64_t)U);
    return w_ok && x_ok && y_ok && go_ok;
}
'''


def emit_lars_bwd_split_kernel_from_schedule(
    schedule_result: ScheduleResult,
    *,
    grad_kind: str,
    kernel_name: str,
    mode: str = "u,u,,u",
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    if grad_kind not in BWD_SPLIT_KINDS:
        raise ValueError(f"grad_kind must be one of {sorted(BWD_SPLIT_KINDS)}, got {grad_kind!r}")
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    reg_count = _max_lars_reg_count_any(schedule_result)
    shared_slots = int(getattr(schedule_result, "shared_slots", 0) or 0)

    target_use: Counter[int] = Counter()
    first_seen: Dict[int, int] = {}
    for inst_id, inst in enumerate(schedule_result.instructions):
        if inst.op == "bwd_split_fma_resident":
            ikind, target_idx = inst.args[0], int(inst.args[1])
            if ikind != grad_kind:
                raise ValueError(f"schedule contains {ikind}, but emitter grad_kind is {grad_kind}")
            target_use[target_idx] += 1
            first_seen.setdefault(target_idx, inst_id)

    all_targets: Set[int] = set(target_use)
    # target is register-resident.
    reg_targets: Set[int] = set(all_targets)

    reg_targets_sorted = sorted(reg_targets)

    def _is_reg_target(idx: int) -> bool:
        return int(idx) in reg_targets

    def _target_acc_name(idx: int) -> str:
        return _split_grad_acc_name(grad_kind, int(idx))

    def _emit_acc_update(idx: int, value_expr: str) -> None:
        if _is_reg_target(idx):
            ap(f"            {_target_acc_name(idx)} += {value_expr};")

    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    def shared_expr(token: str) -> str:
        if not isinstance(token, str) or not token.startswith("s"):
            raise ValueError(f"Bad shared token: {token!r}")
        slot = int(token[1:])
        return f"lars_smem[(size_t){slot} * (size_t)blockDim.x + (size_t)tid]"

    def operand_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        if token.startswith("d:"):
            kind, idx = _parse_bwd_lars_label_ref(token[2:])
            if kind in BWD_OUTPUT_KINDS:
                raise ValueError("split backward output cannot be a direct input")
            expr = _bwd_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            arr = "grad_out" if kind == "go" else kind
            return f"{arr}[{expr}]"
        raise ValueError(f"Bad split backward operand token: {token!r}")

    def acc_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        raise ValueError(f"Bad split backward accumulator token: {token!r}")

    def emit_placed_writeback(idx: int, value_expr: str, suffix: str) -> None:
        out_kind = {"gw": "gw", "gx": "gx", "gy": "gy"}[grad_kind]
        expr = _bwd_label_index_expr(
            out_kind, int(idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        if grad_kind == "gw":
            ap(f"            grad_w[{expr}] = {value_expr};")
        elif grad_kind == "gx":
            ap(f"            atomicAdd(&grad_x[{expr}], {value_expr});")
        else:
            # Portable correctness path: one atomic contribution per valid u lane.
            ap(f"            atomicAdd(&grad_y[{expr}], {value_expr});")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* __restrict__ grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* __restrict__ grad_x,")
    else:
        ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    if shared_slots:
        ap("    extern __shared__ unsigned char lars_smem_raw[];")
        ap("    scalar_t* lars_smem = reinterpret_cast<scalar_t*>(lars_smem_raw);")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
    ap("    const int e_orig = e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e_orig];")
    else:
        ap("    const int src = e_orig;")
    if use_scatter:
        ap("    const int dst = dst_idx[e_orig];")
    else:
        ap("    const int dst = e_orig;")
    ap("")
    ap("    const int x_row = " + ("src;" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src;" if use_y_src else "e_orig;"))
    ap("    const int go_row = " + ("dst;" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base  = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
        if grad_kind == "gw":
            ap(f"    const index_t gw_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base  = (index_t)w_row * (index_t)Iw * (index_t)U;")
        if grad_kind == "gw":
            ap("    const index_t gw_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base  = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
        if grad_kind == "gx":
            ap(f"    const index_t gx_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base  = (index_t)x_row * (index_t)Ix * (index_t)U;")
        if grad_kind == "gx":
            ap("    const index_t gx_base = (index_t)x_row * (index_t)Ix * (index_t)U;")

    if mode_scalar_y:
        ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky;")
        if grad_kind == "gy":
            ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base  = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
            if grad_kind == "gy":
                ap(f"    const index_t gy_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky * (index_t)U;")
            if grad_kind == "gy":
                ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky * (index_t)U;")

    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t go_base = (index_t)go_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t go_base = (index_t)go_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    for idx in reg_targets_sorted:
        ap(f"            scalar_t {_target_acc_name(idx)} = scalar_t(0);")
    if reg_targets_sorted:
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op in ("load", "load_shared"):
            token, ref = inst.args
            kind, idx = _parse_bwd_lars_label_ref(ref)
            if kind in BWD_OUTPUT_KINDS:
                raise ValueError("input load must not target backward output labels")
            expr = _bwd_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            arr = "grad_out" if kind == "go" else kind
            lhs = str(token) if inst.op == "load" else shared_expr(str(token))
            ap(f"            {lhs} = {arr}[{expr}];")

        elif inst.op == "init_acc":
            token, ref = inst.args
            kind, _idx = _parse_bwd_lars_label_ref(ref)
            expected = {"gw": "gw", "gx": "gx", "gy": "gy"}[grad_kind]
            if kind != expected:
                raise ValueError(f"split init_acc expects {expected}, got {kind}")
            ap(f"            {acc_expr(str(token))} = scalar_t(0);")

        elif inst.op == "bwd_split_fma_placed":
            ikind, target_idx, target_token, rw, rx, ry, rgo, coeff = inst.args
            if ikind != grad_kind:
                raise ValueError(f"schedule contains {ikind}, but emitter grad_kind is {grad_kind}")
            c = _fmt_lars_float(float(coeff))
            if grad_kind == "gw":
                value = f"scalar_t({c}) * {operand_expr(str(rgo))} * {operand_expr(str(rx))} * {operand_expr(str(ry))}"
            elif grad_kind == "gx":
                value = f"scalar_t({c}) * ({operand_expr(str(rw))} * {operand_expr(str(rgo))}) * {operand_expr(str(ry))}"
            else:
                value = f"scalar_t({c}) * ({operand_expr(str(rw))} * {operand_expr(str(rgo))}) * {operand_expr(str(rx))}"
            if str(target_token).startswith("d:"):
                emit_placed_writeback(int(target_idx), value, f"direct_{grad_kind}_{inst_id}")
            else:
                ap(f"            {acc_expr(str(target_token))} += {value};")

        elif inst.op == "store_acc_placed":
            ref, token = inst.args
            kind, idx = _parse_bwd_lars_label_ref(ref)
            expected = {"gw": "gw", "gx": "gx", "gy": "gy"}[grad_kind]
            if kind != expected:
                raise ValueError(f"split store expects {expected}, got {kind}")
            emit_placed_writeback(
                int(idx), acc_expr(str(token)), f"placed_{grad_kind}_{idx}_{inst_id}"
            )

        elif inst.op == "bwd_split_fma_resident":
            ikind, target_idx, rw, rx, ry, rgo, coeff = inst.args
            if ikind != grad_kind:
                raise ValueError(f"schedule contains {ikind}, but emitter grad_kind is {grad_kind}")
            c = _fmt_lars_float(float(coeff))
            target_idx = int(target_idx)
            if grad_kind == "gw":
                _emit_acc_update(target_idx, f"scalar_t({c}) * {rgo} * {rx} * {ry}")
            elif grad_kind == "gx":
                _emit_acc_update(target_idx, f"scalar_t({c}) * ({rw} * {rgo}) * {ry}")
            elif grad_kind == "gy":
                _emit_acc_update(target_idx, f"scalar_t({c}) * ({rw} * {rgo}) * {rx}")
            else:
                raise ValueError(f"Bad grad_kind: {grad_kind}")

        elif inst.op in ("release", "release_shared"):
            pass
        else:
            raise ValueError(f"Unsupported split backward LARS instruction op: {inst.op}")

    if reg_targets_sorted:
        ap("")
        ap("            // split backward accumulator writeback")

    def _emit_writeback(idx: int, acc_expr: str) -> None:
        if grad_kind == "gw":
            expr = _bwd_label_index_expr(
                "gw", int(idx),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            ap(f"            grad_w[{expr}] = {acc_expr};")
        elif grad_kind == "gx":
            expr = _bwd_label_index_expr(
                "gx", int(idx),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            ap(f"            atomicAdd(&grad_x[{expr}], {acc_expr});")
        elif grad_kind == "gy":
            expr = _bwd_label_index_expr(
                "gy", int(idx),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            # Portable correctness path: one atomic contribution per valid u lane.
            ap(f"            atomicAdd(&grad_y[{expr}], {acc_expr});")
        else:
            raise ValueError(f"Bad grad_kind: {grad_kind}")

    for idx in reg_targets_sorted:
        _emit_writeback(int(idx), _target_acc_name(int(idx)))
    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* grad_x,")
    else:
        ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    size_t lars_shared_bytes = (size_t){shared_slots} * (size_t)block.x * sizeof(scalar_t);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, lars_shared_bytes, stream>>>(")
    if grad_kind == "gw":
        ap("        w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("        w, x, y, grad_out, grad_x,")
    else:
        ap("        w, x, y, grad_out, grad_y,")
    ap("        src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* grad_x,")
    else:
        ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_bwd_split(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                       kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    if grad_kind == "gw":
        ap("            w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("            w, x, y, grad_out, grad_x,")
    else:
        ap("            w, x, y, grad_out, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    if grad_kind == "gw":
        ap("            w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("            w, x, y, grad_out, grad_x,")
    else:
        ap("            w, x, y, grad_out, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* grad_x,")
    else:
        ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    if grad_kind == "gw":
        ap("        w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("        w, x, y, grad_out, grad_x,")
    else:
        ap("        w, x, y, grad_out, grad_y,")
    ap("        src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def emit_lars_bwd_split_launcher(
    bundle_name: str,
    mode: str,
    *,
    need_grad_w: bool,
    need_grad_x: bool,
    need_grad_y: bool,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
    gradw_kernel: Optional[str],
    gradx_kernel: Optional[str],
    grady_kernel: Optional[str],
) -> str:
    if mode == "u,u,,u":
        y_comment = "[B,Ky,1] or [S,Ky,1]"
        gy_alloc = "    auto grad_y = torch::zeros({y.size(0), Ky, 1}, y.options());"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");'
    elif mode == "u,u,u,u":
        y_comment = "[B,Ky,U] or [S,Ky,U]"
        gy_alloc = "    auto grad_y = torch::zeros_like(y);"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == U, "y.size(2) must equal U for mode u,u,u,u");'
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    param_lines = [
        "    torch::Tensor w",
        "    torch::Tensor x",
        "    torch::Tensor y",
        "    torch::Tensor grad_out",
    ]
    if use_x_src or use_y_src:
        param_lines.append("    torch::Tensor src_idx")
    if use_scatter:
        param_lines.append("    torch::Tensor dst_idx")
    param_lines.append("    int64_t V64")
    params = ",\n".join(param_lines)

    src_check = ""
    if use_x_src or use_y_src:
        src_check = r'''
    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''
    dst_check = ""
    if use_scatter:
        dst_check = r'''
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''

    if use_x_src or use_y_src:
        b_expr = "src_idx.size(0)"
        src_numel_check = '    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");'
    elif use_scatter:
        b_expr = "dst_idx.size(0)"
        src_numel_check = ""
    else:
        b_expr = "grad_out.size(0)"
        src_numel_check = ""

    dst_numel_check = '    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");' if use_scatter else ""

    blist_logic = ""

    launch_src_arg = "(const int32_t*)src_idx.data_ptr<int32_t>()," if (use_x_src or use_y_src) else "nullptr,"
    launch_dst_arg = "(const int32_t*)dst_idx.data_ptr<int32_t>()," if use_scatter else "nullptr,"

    grad_w_alloc = "    auto grad_w = torch::zeros_like(w);\n" if need_grad_w else ""
    grad_w_launch = ""
    if need_grad_w:
        assert gradw_kernel is not None
        grad_w_launch = f'''
        launch_{gradw_kernel}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                B, WB, Iw, Ix, Ky, V, U, S, stream);
'''
    grad_y_alloc = f"{gy_alloc}\n" if need_grad_y else ""
    grad_y_launch = ""
    if need_grad_y:
        assert grady_kernel is not None
        grad_y_launch = f'''
        launch_{grady_kernel}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                B, WB, Iw, Ix, Ky, V, U, S, stream);
'''
    grad_x_alloc = "    auto grad_x = torch::zeros_like(x);\n" if need_grad_x else ""
    grad_x_launch = ""
    if need_grad_x:
        assert gradx_kernel is not None
        grad_x_launch = f'''
        launch_{gradx_kernel}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                B, WB, Iw, Ix, Ky, V, U, S, stream);
'''
    result_names = []
    if need_grad_w:
        result_names.append("grad_w")
    if need_grad_x:
        result_names.append("grad_x")
    if need_grad_y:
        result_names.append("grad_y")
    if not result_names:
        raise ValueError("at least one backward gradient must be requested")
    ret_expr = "return {" + ", ".join(result_names) + "};"

    return rf'''

std::vector<torch::Tensor> launcher_{bundle_name}(
{params})
{{
    // Split backward: grad_w / grad_x / grad_y are computed by independent
    // LARS-scheduled kernels to reduce register pressure for large path lists.
    // Expected tensors:
    //   w        : [WB,Iw,U], WB can be 1 or B
    //   x        : [S,Ix,U] or [B,Ix,U]
    //   y        : {y_comment}
    //   grad_out : [B,V,U] or [S,V,U], matching forward output layout

    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x/y/grad_out must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");
{src_check}{dst_check}

    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x.dim() == 3, "x must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");

    int B  = (int){b_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x.size(0);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
    {y_check_u}
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
{src_numel_check}
{dst_numel_check}

{blist_logic}
{grad_w_alloc}{grad_x_alloc}
{grad_y_alloc}

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{

        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");
{grad_w_launch}
{grad_x_launch}

{grad_y_launch}
    }});

    GPU_KERNEL_LAUNCH_CHECK();

    {ret_expr}
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} backward LARS split jit impl");
}}
'''



@dataclass(frozen=True)
class BwdCodegenContext:
    """Pre-parsed Uniform1D backward codegen inputs used by fused/split paths."""
    input_indices: Dict[int, Any]
    output_indices: Dict[int, Any]
    use_x_src: bool
    use_y_src: bool
    use_scatter: bool
    P: int
    i_cpu: List[int]
    j_cpu: List[int]
    k_cpu: List[int]
    v_cpu: List[int]
    c_cpu: List[float]
    bwd_paths: List[Tuple[int, int, int, int, float]]
    mode_str: str
    layout_tag: str


def _prepare_uniform1d_bwd_codegen_context(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]],
    output_indices: Optional[Dict[int, Any]],
    *,
    mode: str,
    path_semantics: str,
) -> BwdCodegenContext:
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = int(i_list.numel())
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    bwd_paths = _make_lars_bwd_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )

    return BwdCodegenContext(
        input_indices=input_indices,
        output_indices=output_indices,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        P=P,
        i_cpu=i_cpu,
        j_cpu=j_cpu,
        k_cpu=k_cpu,
        v_cpu=v_cpu,
        c_cpu=c_cpu,
        bwd_paths=bwd_paths,
        mode_str="uu_u" if mode == "u,u,,u" else "uuuu",
        layout_tag=f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}",
    )


def _resolve_split_backward(
    split_backward: Union[bool, str],
    *,
    path_count: int,
    split_path_threshold: int,
) -> bool:
    if isinstance(split_backward, str):
        split_mode = split_backward.lower()
        if split_mode in ("auto",):
            return int(path_count) > int(split_path_threshold)
        if split_mode in ("split", "true", "always", "on", "yes"):
            return True
        if split_mode in ("fused", "false", "never", "off", "no"):
            return False
        raise ValueError(
            "split_backward must be bool or one of "
            "'auto', 'split'/'always'/'on', 'fused'/'never'/'off'"
        )
    return bool(split_backward)


def _write_codegen_candidate(
    *,
    out_path: str,
    candidate_count: int,
    cand_name: str,
    code: str,
) -> None:
    if not out_path:
        return
    out_p = Path(out_path)
    if candidate_count > 1:
        stem = out_p.stem
        suffix = out_p.suffix or ".cu"
        out_p.with_name(f"{stem}_{cand_name}{suffix}").write_text(code, encoding="utf-8")
    else:
        out_p.write_text(code, encoding="utf-8")


def _finalize_codegen_candidates(candidates: List[Any], *, return_schedule: bool):
    if return_schedule:
        return candidates if len(candidates) != 1 else candidates[0]
    if len(candidates) == 1:
        return candidates[0][1]
    return candidates


def _bwd_base_kernel_name(
    *,
    kernel_name: str,
    cand_name: str,
    u_dim: int,
    path_count: int,
    mode_str: str,
    layout_tag: str,
    grad_tag: str,
) -> str:
    safe_cand = cand_name.replace("-", "_").replace(".", "_")
    if kernel_name and kernel_name != "uniform1d_bwd_lars":
        return f"{kernel_name}_{safe_cand}"
    return (
        f"uniform1d_{safe_cand}_u{u_dim}_path{path_count}_"
        f"{mode_str}_{layout_tag}_{grad_tag}_bwd"
    )


def _generate_code_uniform1d_bwd_split_from_context(
    ctx: BwdCodegenContext,
    *,
    u_dim: int,
    iw_dim: Optional[int],
    ix_dim: Optional[int],
    ky_dim: Optional[int],
    v_dim: Optional[int],
    mode: str,
    need_grad_w: bool,
    need_grad_x: bool,
    need_grad_y: bool,
    out_path: str,
    kernel_name: str,
    return_schedule: bool,
    profile: bool,
    profile_interval: int,
    profile_seconds: float,
    profile_print: bool,
    enable_secondary_affinity: bool,
    topk_candidates: Optional[int],
    block_size: int = 32,
    placement_config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]] = None,
):
    """Internal split-backward implementation.  The public entry is the unified wrapper."""
    resolved_placement = _resolve_lars_placement_config(placement_config)
    # One split candidate is emitted.  Lifetime placement is applied independently
    # to each gradient kernel after LARS chooses its path order.
    configs: List[Dict[str, Any]] = [{
        "name": "lars_bwd_split_all_inputs_accall",
    }]
    max_effective_need = 0

    grad_tag = f"split_gw{int(bool(need_grad_w))}_gx{int(bool(need_grad_x))}_gy{int(bool(need_grad_y))}"
    candidates = []
    split_kinds = ((["gw"] if need_grad_w else []) +
                   (["gx"] if need_grad_x else []) +
                   (["gy"] if need_grad_y else []))
    if not split_kinds:
        raise ValueError("at least one backward gradient must be requested")

    for cfg_in in configs:
        cfg = dict(cfg_in)
        cand_name = str(cfg["name"])
        cfg["auto_fallback_when_full"] = False
        cfg["effective_live_need"] = int(max_effective_need)
        cfg["split_backward"] = True

        base_kernel_name = _bwd_base_kernel_name(
            kernel_name=kernel_name,
            cand_name=cand_name,
            u_dim=u_dim,
            path_count=ctx.P,
            mode_str=ctx.mode_str,
            layout_tag=ctx.layout_tag,
            grad_tag=grad_tag,
        )

        schedules: Dict[str, ScheduleResult] = {}
        profiles: Dict[str, Any] = {}
        code_parts: List[str] = [emit_lars_bwd_split_preamble()]
        kernel_names: Dict[str, str] = {}

        for gkind in split_kinds:
            scheduler = LARSUniform1DBwdSplitScheduler(
                paths=ctx.bwd_paths,
                grad_kind=gkind,
                enable_secondary_affinity=bool(cfg.get("enable_secondary_affinity", enable_secondary_affinity)),
                topk_candidates=cfg.get("topk_candidates", topk_candidates),
                profile=bool(cfg.get("profile", profile)),
                profile_name=str(cfg.get("profile_name", f"{cand_name}_{gkind}")),
                profile_interval=int(cfg.get("profile_interval", profile_interval)),
                profile_seconds=float(cfg.get("profile_seconds", profile_seconds)),
                profile_print=bool(cfg.get("profile_print", profile_print)),
            )
            schedule_result = scheduler.schedule()
            _dump_and_print_schedule_label_stats(
                operation=f"uniform1d_bwd_split_{gkind}",
                candidate_tag=cand_name,
                path_order=schedule_result.path_order,
                labels_by_path=_stats_labels_uniform1d_bwd_split(
                    scheduler.paths,
                    grad_kind=gkind,
                ),
            )
            schedule_result = apply_lars_lifetime_placement(
                schedule_result,
                schedule_kind=f"bwd_split:{gkind}",
                config=resolved_placement,
            )
            stats_path = None
            if resolved_placement.stats_path:
                stats_p = Path(resolved_placement.stats_path)
                suffix = stats_p.suffix or ".json"
                stats_path = str(stats_p.with_name(f"{stats_p.stem}_{gkind}{suffix}"))
            dump_lars_variable_stats(
                schedule_result,
                print_stats=resolved_placement.stats_print,
                out_path=stats_path,
            )
            schedules[gkind] = schedule_result
            profiles[gkind] = schedule_result.profile
            split_kernel_name = f"{base_kernel_name}_{gkind}"
            kernel_names[gkind] = split_kernel_name
            code_parts.append(
                emit_lars_bwd_split_kernel_from_schedule(
                    schedule_result,
                    grad_kind=gkind,
                    kernel_name=split_kernel_name,
                    mode=mode,
                    use_x_src=ctx.use_x_src,
                    use_y_src=ctx.use_y_src,
                    use_scatter=ctx.use_scatter,
                    u_dim=u_dim,
                    iw_dim=iw_dim,
                    ix_dim=ix_dim,
                    ky_dim=ky_dim,
                    v_dim=v_dim,
                    block_size=int(block_size),
                )
            )

        code_parts.append(
            emit_lars_bwd_split_launcher(
                bundle_name=base_kernel_name,
                mode=mode,
                need_grad_w=need_grad_w,
                need_grad_x=need_grad_x,
                need_grad_y=need_grad_y,
                use_x_src=ctx.use_x_src,
                use_y_src=ctx.use_y_src,
                use_scatter=ctx.use_scatter,
                gradw_kernel=kernel_names.get("gw"),
                gradx_kernel=kernel_names.get("gx"),
                grady_kernel=kernel_names.get("gy"),
            )
        )
        code = "\n".join(code_parts)

        _write_codegen_candidate(
            out_path=out_path,
            candidate_count=len(configs),
            cand_name=cand_name,
            code=code,
        )

        if return_schedule:
            candidates.append({
                "name": cand_name,
                "code": code,
                "schedule": schedules,
                "config": cfg,
                "profiles": profiles,
                "auto_fallback_when_full": False,
            })
        else:
            candidates.append((cand_name, code))

    return _finalize_codegen_candidates(candidates, return_schedule=return_schedule)



def _generate_code_uniform1d_bwd_fused_from_context(
    ctx: BwdCodegenContext,
    *,
    u_dim: int,
    iw_dim: Optional[int],
    ix_dim: Optional[int],
    ky_dim: Optional[int],
    v_dim: Optional[int],
    mode: str,
    need_grad_w: bool,
    need_grad_x: bool,
    need_grad_y: bool,
    out_path: str,
    kernel_name: str,
    return_schedule: bool,
    profile: bool,
    profile_interval: int,
    profile_seconds: float,
    profile_print: bool,
    enable_secondary_affinity: bool,
    topk_candidates: Optional[int],
    block_size: int = 32,
    placement_config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]] = None,
):
    """
    Generate one fused backward LARS candidate.

    input labels are virtually unbounded and all gw/gx/gy accumulators stay
    register-resident in the emitter.
    """

    if not need_grad_x or not need_grad_y:
        raise ValueError("partial gx/gy backward must use split codegen")
    grad_tag = "full" if need_grad_w else "nogradw"
    cand_name = "lars_bwd_all_inputs_accall"

    scheduler = LARSUniform1DBwdScheduler(
        paths=ctx.bwd_paths,
        need_grad_w=need_grad_w,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name=cand_name,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    schedule_result = scheduler.schedule()
    _dump_and_print_schedule_label_stats(
        operation="uniform1d_bwd_fused",
        candidate_tag=cand_name,
        path_order=schedule_result.path_order,
        labels_by_path=_stats_labels_uniform1d_bwd_fused(
            scheduler.paths,
            need_grad_w=need_grad_w,
        ),
    )
    resolved_placement = _resolve_lars_placement_config(placement_config)
    schedule_result = apply_lars_lifetime_placement(
        schedule_result,
        schedule_kind="bwd_fused",
        config=resolved_placement,
    )
    dump_lars_variable_stats(
        schedule_result,
        print_stats=resolved_placement.stats_print,
        out_path=resolved_placement.stats_path,
    )

    base_kernel_name = _bwd_base_kernel_name(
        kernel_name=kernel_name,
        cand_name=cand_name,
        u_dim=u_dim,
        path_count=ctx.P,
        mode_str=ctx.mode_str,
        layout_tag=ctx.layout_tag,
        grad_tag=grad_tag,
    )

    code = emit_fused_bwd_kernel_from_lars_schedule(
        schedule_result,
        kernel_name=base_kernel_name,
        mode=mode,
        need_grad_w=need_grad_w,
        use_x_src=ctx.use_x_src,
        use_y_src=ctx.use_y_src,
        use_scatter=ctx.use_scatter,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        block_size=int(block_size),
    )

    code = code + "\n" + emit_lars_bwd_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        need_grad_w=need_grad_w,
        use_x_src=ctx.use_x_src,
        use_y_src=ctx.use_y_src,
        use_scatter=ctx.use_scatter,
    )

    _write_codegen_candidate(
        out_path=out_path,
        candidate_count=1,
        cand_name=cand_name,
        code=code,
    )

    if return_schedule:
        return {
            "name": cand_name,
            "code": code,
            "schedule": schedule_result,
            "config": {
                "name": cand_name,
                "auto_fallback_when_full": False,
                "need_grad_w": bool(need_grad_w),
                "split_backward": False,
            },
            "profile": schedule_result.profile,
        }

    return code


def generate_code_uniform1d_bwd_split_with_scheduler(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    need_grad_x: bool = True,
    need_grad_y: bool = True,
    out_path: str = "generated_uniform1d_bwd_lars_split.cu",
    kernel_name: str = "uniform1d_bwd_lars",
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    profile: bool = True,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = True,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    placement_config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]] = None,
):
    """
    Backward-compatible wrapper.  New code should call
    generate_code_uniform1d_bwd_with_scheduler(..., split_backward=True).
    """
    return generate_code_uniform1d_bwd_with_scheduler(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
        input_indices=input_indices,
        output_indices=output_indices,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        mode=mode,
        need_grad_w=need_grad_w,
        need_grad_x=need_grad_x,
        need_grad_y=need_grad_y,
        out_path=out_path,
        kernel_name=kernel_name,
        # Keep legacy parameters in the public signature, but force the new
        # policy: input labels unbounded, accumulators all register-resident.
        path_semantics=path_semantics,
        return_schedule=return_schedule,
        profile=profile,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        split_backward=True,
        placement_config=placement_config,
    )



def generate_code_uniform1d_bwd_with_scheduler(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    need_grad_x: bool = True,
    need_grad_y: bool = True,
    out_path: str = "generated_uniform1d_bwd_lars.cu",
    kernel_name: str = "uniform1d_bwd_lars",
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    profile: bool = True,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = True,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    split_backward: Union[bool, str] = "auto",
    split_path_threshold: int = 256,
    block_size: int = 32,
    placement_config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]] = None,
):
    """
    Unified LARS backward code generator.

    split_backward controls implementation strategy:
      - False / "fused": one fused LARS backward kernel.
      - True / "split": independent grad_w, grad_x and grad_y kernels.
      - "auto": split when path_count > split_path_threshold.
    """
    ctx = _prepare_uniform1d_bwd_codegen_context(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
        input_indices=input_indices,
        output_indices=output_indices,
        mode=mode,
        path_semantics=path_semantics,
    )

    use_split_backward = _resolve_split_backward(
        split_backward,
        path_count=ctx.P,
        split_path_threshold=split_path_threshold,
    )
    # Partial gx/gy specializations omit their kernel, allocation, and atomic
    # reduction. Route them through split codegen so the fused all-gradient
    # schedule cannot retain hidden gradient work.
    if not need_grad_x or not need_grad_y:
        use_split_backward = True

    common_kwargs = dict(
        ctx=ctx,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        mode=mode,
        need_grad_w=need_grad_w,
        need_grad_x=need_grad_x,
        need_grad_y=need_grad_y,
        out_path=out_path,
        kernel_name=kernel_name,
        return_schedule=return_schedule,
        profile=profile,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        block_size=int(block_size),
        placement_config=placement_config,
    )

    if use_split_backward:
        return _generate_code_uniform1d_bwd_split_from_context(**common_kwargs)

    return _generate_code_uniform1d_bwd_fused_from_context(**common_kwargs)


# =============================================================================
# Uniform1D training double backward: fully-unrolled LARS CUDA/HIP codegen
# =============================================================================

@dataclass(frozen=True)
class U1DDoubleBwdPath:
    """One path in the VJP of the Uniform1D first backward.

    Forward path:
        out[v] += c * w[i] * x[j] * y[k]

    First backward upstream is ``go = grad_out``.  Double backward receives
    upstream gradients of (grad_w, grad_x, grad_y), denoted ggw/ggx/ggy, and
    returns gradients with respect to (go, w, x, y).
    """
    pid: int
    i: int
    j: int
    k: int
    v: int
    c: float
    need_grad_w: bool = True
    need_grad_x: bool = True
    need_grad_y: bool = True

    @property
    def labels(self) -> Tuple[Label, ...]:
        labs: List[Label] = [
            ("w", self.i),
            ("x", self.j),
            ("y", self.k),
            ("go", self.v),
        ]
        if self.need_grad_w:
            labs.append(("ggw", self.i))
        if self.need_grad_x:
            labs.append(("ggx", self.j))
        if self.need_grad_y:
            labs.append(("ggy", self.k))
        return tuple(labs)


class LARSUniform1DDoubleBwdScheduler(LARSUniform1DScheduler):
    """LARS scheduler for the fully-unrolled Uniform1D double backward.

    Only read-only operands participate in the LARS live set.  The four output
    gradients are emitted as atomic accumulations, which is required for all
    supported layouts (source indirection, scatter, WB==1, and scalar-y).
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        *,
        need_grad_w: bool = True,
        need_grad_x: bool = True,
        need_grad_y: bool = True,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        self.path_kind = "u1d_double_bwd"
        self.need_grad_w = bool(need_grad_w)
        self.need_grad_x = bool(need_grad_x)
        self.need_grad_y = bool(need_grad_y)
        self.paths: List[U1DDoubleBwdPath] = [
            U1DDoubleBwdPath(
                pid=p, i=i, j=j, k=k, v=v, c=c,
                need_grad_w=self.need_grad_w,
                need_grad_x=self.need_grad_x,
                need_grad_y=self.need_grad_y,
            )
            for p, (i, j, k, v, c) in enumerate(paths)
        ]

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
        self.debug = bool(debug)

        self.label_to_paths: Dict[Label, Set[int]] = defaultdict(set)
        for path in self.paths:
            for lab in path.labels:
                self.label_to_paths[lab].add(path.pid)

        self.path_label_sets: List[Set[Label]] = [set(p.labels) for p in self.paths]
        self.unscheduled: Set[int] = set(p.pid for p in self.paths)
        self.remaining_uses: Counter[Label] = Counter()
        for path in self.paths:
            for lab in path.labels:
                self.remaining_uses[lab] += 1

        # The generic scheduler expects this attribute for forward bookkeeping.
        self.output_uses: Counter[int] = Counter(int(p.v) for p in self.paths)

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = []
        self._next_reg_id = 0
        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0
        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}

        self.profile_enabled = bool(profile)
        self.profile_name = str(profile_name or self.__class__.__name__)
        self.profile_interval = int(profile_interval)
        self.profile_seconds = float(profile_seconds)
        self.profile_print = bool(profile_print)
        self._profile_t0 = perf_counter()
        self._profile_last_print_t = self._profile_t0
        self._profile_last_print_done = 0
        self._profile_loop_iter = 0
        self._profile_select_rounds = 0
        self._profile_fire_rounds = 0
        self._profile_load_rounds = 0
        self._profile_records: List[Dict[str, Any]] = []
        self._profile_last_event = ""
        self._profile_last_reason = ""

    def _path(self, pid: int) -> U1DDoubleBwdPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        names = {
            "go": "grad_out",
            "ggw": "grad_grad_w",
            "ggx": "grad_grad_x",
            "ggy": "grad_grad_y",
        }
        return f"{names.get(kind, kind)}[{idx}]"

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)
        reg = self.reg_of
        rw = reg[("w", p.i)]
        rx = reg[("x", p.j)]
        ry = reg[("y", p.k)]
        rgo = reg[("go", p.v)]
        rggw = reg[("ggw", p.i)] if self.need_grad_w else ""
        rggx = reg[("ggx", p.j)] if self.need_grad_x else ""
        rggy = reg[("ggy", p.k)] if self.need_grad_y else ""

        self.instructions.append(Inst(
            "double_bwd_fma_resident",
            (
                p.i, p.j, p.k, p.v,
                rw, rx, ry, rgo, rggw, rggx, rggy,
                p.c, self.need_grad_w, self.need_grad_x, self.need_grad_y,
            ),
            (
                f"path#{pid}: double backward; "
                f"w[{p.i}], x[{p.j}], y[{p.k}], go[{p.v}], c={p.c}"
            ),
        ))
        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        for lab in p.labels:
            self.remaining_uses[lab] -= 1
        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")


def _stats_labels_uniform1d_double_bwd(
    paths: Sequence[U1DDoubleBwdPath],
    *,
    need_grad_w: bool,
    need_grad_x: bool,
    need_grad_y: bool,
) -> List[List[str]]:
    rows: List[List[str]] = []
    for p in paths:
        labels = [
            f"w[{p.i}]", f"x[{p.j}]", f"y[{p.k}]", f"grad_out[{p.v}]",
        ]
        if need_grad_w:
            labels.append(f"grad_grad_w[{p.i}]")
        if need_grad_x:
            labels.append(f"grad_grad_x[{p.j}]")
        if need_grad_y:
            labels.append(f"grad_grad_y[{p.k}]")
        rows.append(labels)
    return rows


def emit_uniform1d_double_bwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    mode: str,
    need_grad_w: bool,
    need_grad_x: bool,
    need_grad_y: bool,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """Emit a fully-unrolled fused double-backward kernel from a LARS schedule."""
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")
    mode_scalar_y = mode == "u,u,,u"
    if not (need_grad_w or need_grad_x or need_grad_y):
        raise ValueError("at least one first-gradient output must be requested")
    reg_count = _max_lars_reg_count_any(schedule_result)

    lines: List[str] = []
    def ap(line: str = "") -> None:
        lines.append(line)

    def parse_ref(ref: str) -> Tuple[str, int]:
        if "[" not in ref or not ref.endswith("]"):
            raise ValueError(f"Bad double backward ref: {ref!r}")
        kind, tail = ref.split("[", 1)
        idx = int(tail[:-1])
        aliases = {
            "grad_out": "go",
            "grad_grad_w": "ggw",
            "grad_grad_x": "ggx",
            "grad_grad_y": "ggy",
        }
        return aliases.get(kind, kind), idx

    def input_expr(kind: str, idx: int) -> str:
        if kind == "w":
            return f"w[w_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "x":
            return f"x[x_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "y":
            if mode_scalar_y:
                return f"y[y_base + (index_t){idx}]"
            return f"y[y_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "go":
            return f"grad_out[go_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "ggw":
            return f"grad_grad_w[ggw_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "ggx":
            return f"grad_grad_x[ggx_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "ggy":
            if mode_scalar_y:
                return f"grad_grad_y[ggy_base + (index_t){idx}]"
            return f"grad_grad_y[ggy_base + (index_t){idx} * (index_t)U + (index_t)u]"
        raise ValueError(f"Bad double backward input kind: {kind}")

    def output_index(kind: str, idx: int) -> str:
        if kind == "dgo":
            return f"dgo_base + (index_t){idx} * (index_t)U + (index_t)u"
        if kind == "dw":
            return f"dw_base + (index_t){idx} * (index_t)U + (index_t)u"
        if kind == "dx":
            return f"dx_base + (index_t){idx} * (index_t)U + (index_t)u"
        if kind == "dy":
            if mode_scalar_y:
                return f"dy_base + (index_t){idx}"
            return f"dy_base + (index_t){idx} * (index_t)U + (index_t)u"
        raise ValueError(kind)

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")
    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    if need_grad_w:
        ap("    const scalar_t* __restrict__ grad_grad_w,")
    if need_grad_x:
        ap("    const scalar_t* __restrict__ grad_grad_x,")
    if need_grad_y:
        ap("    const scalar_t* __restrict__ grad_grad_y,")
    ap("    scalar_t* __restrict__ d_grad_out,")
    if need_grad_w:
        ap("    scalar_t* __restrict__ d_w,")
    if need_grad_x:
        ap("    scalar_t* __restrict__ d_x,")
    if need_grad_y:
        ap("    scalar_t* __restrict__ d_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U)")
    ap("{")
    ap("    const int e = (int)blockIdx.x;")
    ap("    if (e >= B) return;")
    ap("    const int tid = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e];")
    else:
        ap("    const int src = e;")
    if use_scatter:
        ap("    const int dst = dst_idx[e];")
    else:
        ap("    const int dst = e;")
    ap("    const int w_row = (WB == 1 ? 0 : e);")
    ap(f"    const int x_row = {'src' if use_x_src else 'e'};")
    ap(f"    const int y_row = {'src' if use_y_src else 'e'};")
    ap(f"    const int go_row = {'dst' if use_scatter else 'e'};")
    ap("    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;")
    ap("    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;")
    if mode_scalar_y:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky;")
    else:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky * (index_t)U;")
    ap("    const index_t go_base = (index_t)go_row * (index_t)V * (index_t)U;")
    if need_grad_w:
        ap("    const index_t ggw_base = w_base;")
    if need_grad_x:
        ap("    const index_t ggx_base = x_base;")
    if need_grad_y:
        ap("    const index_t ggy_base = y_base;")
    ap("    const index_t dgo_base = go_base;")
    if need_grad_w:
        ap("    const index_t dw_base = w_base;")
    if need_grad_x:
        ap("    const index_t dx_base = x_base;")
    if need_grad_y:
        ap("    const index_t dy_base = y_base;")
    ap("")
    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        ap(f"            // double-bwd inst {inst_id}: {inst.op}" + (f" | {comment}" if comment else ""))
        if inst.op == "load":
            reg, ref = inst.args
            kind, idx = parse_ref(str(ref))
            ap(f"            {reg} = {input_expr(kind, idx)};")
        elif inst.op == "double_bwd_fma_resident":
            wi, xj, yk, ov, rw, rx, ry, rgo, rggw, rggx, rggy, coeff, inst_need_gw, inst_need_gx, inst_need_gy = inst.args
            c = _fmt_lars_float(float(coeff))
            t_dgo: List[str] = []
            if bool(inst_need_gw):
                t_dgo.append(f"({rggw} * {rx} * {ry})")
            if bool(inst_need_gx):
                t_dgo.append(f"({rggx} * {rw} * {ry})")
            if bool(inst_need_gy):
                t_dgo.append(f"({rggy} * {rw} * {rx})")
            t_dw: List[str] = []
            if bool(inst_need_gx):
                t_dw.append(f"({rggx} * {rgo} * {ry})")
            if bool(inst_need_gy):
                t_dw.append(f"({rggy} * {rgo} * {rx})")
            t_dx: List[str] = []
            if bool(inst_need_gw):
                t_dx.append(f"({rggw} * {rgo} * {ry})")
            if bool(inst_need_gy):
                t_dx.append(f"({rggy} * {rgo} * {rw})")
            ap(f"            atomicAdd(&d_grad_out[{output_index('dgo', int(ov))}], scalar_t({c}) * ({' + '.join(t_dgo)}));")
            if bool(inst_need_gw) and t_dw:
                ap(f"            atomicAdd(&d_w[{output_index('dw', int(wi))}], scalar_t({c}) * ({' + '.join(t_dw)}));")
            if bool(inst_need_gx) and t_dx:
                ap(f"            atomicAdd(&d_x[{output_index('dx', int(xj))}], scalar_t({c}) * ({' + '.join(t_dx)}));")
            if bool(inst_need_gy):
                t_dy: List[str] = []
                if bool(inst_need_gw):
                    t_dy.append(f"({rggw} * {rgo} * {rx})")
                if bool(inst_need_gx):
                    t_dy.append(f"({rggx} * {rgo} * {rw})")
                if t_dy:
                    ap(f"            atomicAdd(&d_y[{output_index('dy', int(yk))}], scalar_t({c}) * ({' + '.join(t_dy)}));")
        elif inst.op == "release":
            pass
        else:
            raise ValueError(f"Unsupported double backward instruction op: {inst.op}")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")
    grad_grad_names = []
    if need_grad_w:
        grad_grad_names.append("grad_grad_w")
    if need_grad_x:
        grad_grad_names.append("grad_grad_x")
    if need_grad_y:
        grad_grad_names.append("grad_grad_y")
    double_output_names = ["d_grad_out"]
    if need_grad_w:
        double_output_names.append("d_w")
    if need_grad_x:
        double_output_names.append("d_x")
    if need_grad_y:
        double_output_names.append("d_y")
    kernel_call_args = (["w", "x", "y", "grad_out"] + grad_grad_names +
                        double_output_names + ["src_idx", "dst_idx", "B", "WB",
                        "Iw", "Ix", "Ky", "V", "U"])

    def emit_double_launch_signature(template_line: str, function_line: str) -> None:
        ap(template_line)
        ap(function_line)
        ap("    const scalar_t* w, const scalar_t* x, const scalar_t* y, const scalar_t* grad_out,")
        for name in grad_grad_names:
            ap(f"    const scalar_t* {name},")
        for name in double_output_names:
            ap(f"    scalar_t* {name},")
        ap("    const int32_t* src_idx, const int32_t* dst_idx,")
        ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, gpuStream_t stream)")

    emit_double_launch_signature(
        "template <typename scalar_t, typename index_t>",
        f"void launch_{kernel_name}_typed(",
    )
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        " + ", ".join(kernel_call_args) + ");")
    ap("}")
    ap("")

    emit_double_launch_signature(
        "template <typename scalar_t>",
        f"void launch_{kernel_name}(",
    )
    ap("{")
    ap("    bool use_i32 = ((int64_t)WB * Iw * U <= 2147483647LL) &&")
    ap("                   ((int64_t)B * V * U <= 2147483647LL);")
    ap("    if (use_i32) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(" + ", ".join(kernel_call_args) + ", stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(" + ", ".join(kernel_call_args) + ", stream);")
    ap("    }")
    ap("}")
    return "\n".join(lines)


def emit_uniform1d_double_bwd_launcher(
    bundle_name: str,
    mode: str,
    *,
    need_grad_w: bool,
    need_grad_x: bool,
    need_grad_y: bool,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
    params = [
        "    torch::Tensor w",
        "    torch::Tensor x",
        "    torch::Tensor y",
        "    torch::Tensor grad_out",
    ]
    if need_grad_w:
        params.append("    torch::Tensor grad_grad_w")
    if need_grad_x:
        params.append("    torch::Tensor grad_grad_x")
    if need_grad_y:
        params.append("    torch::Tensor grad_grad_y")
    if use_x_src or use_y_src:
        params.append("    torch::Tensor src_idx")
    if use_scatter:
        params.append("    torch::Tensor dst_idx")
    params.append("    int64_t V64")
    params_s = ",\n".join(params)

    b_expr = "src_idx.size(0)" if (use_x_src or use_y_src) else ("dst_idx.size(0)" if use_scatter else "grad_out.size(0)")
    src_checks = ""
    if use_x_src or use_y_src:
        src_checks = '''
    TORCH_CHECK(src_idx.is_cuda() && src_idx.is_contiguous(), "src_idx must be contiguous CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''
    dst_checks = ""
    if use_scatter:
        dst_checks = '''
    TORCH_CHECK(dst_idx.is_cuda() && dst_idx.is_contiguous(), "dst_idx must be contiguous CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''
    ggw_checks = ""
    if need_grad_w:
        ggw_checks = '''
    TORCH_CHECK(grad_grad_w.is_cuda() && grad_grad_w.is_contiguous(), "grad_grad_w must be contiguous CUDA/HIP");
    TORCH_CHECK(grad_grad_w.sizes() == w.sizes(), "grad_grad_w shape must match w");
    TORCH_CHECK(grad_grad_w.scalar_type() == w.scalar_type(), "grad_grad_w dtype mismatch");
'''
    ggx_checks = ""
    if need_grad_x:
        ggx_checks = '''
    TORCH_CHECK(grad_grad_x.is_cuda() && grad_grad_x.is_contiguous(), "grad_grad_x must be contiguous CUDA/HIP");
    TORCH_CHECK(grad_grad_x.sizes() == x.sizes(), "grad_grad_x shape must match x");
    TORCH_CHECK(grad_grad_x.scalar_type() == w.scalar_type(), "grad_grad_x dtype mismatch");
'''
    y_shape_check = (
        '    TORCH_CHECK((int)y.size(2) == 1, "scalar-y mode expects y[...,1]");'
        if mode == "u,u,,u" else
        '    TORCH_CHECK((int)y.size(2) == U, "vector-y mode expects y[...,U]");'
    )
    ggy_checks = ""
    if need_grad_y:
        ggy_checks = (
            '    TORCH_CHECK(grad_grad_y.is_cuda() && grad_grad_y.is_contiguous(), "grad_grad_y must be contiguous CUDA/HIP");\n'
            '    TORCH_CHECK(grad_grad_y.sizes() == y.sizes(), "grad_grad_y shape must match y");\n'
            '    TORCH_CHECK(grad_grad_y.scalar_type() == w.scalar_type(), "grad_grad_y dtype mismatch");'
        )
    src_arg = "(const int32_t*)src_idx.data_ptr<int32_t>()" if (use_x_src or use_y_src) else "nullptr"
    dst_arg = "(const int32_t*)dst_idx.data_ptr<int32_t>()" if use_scatter else "nullptr"
    ggw_launch_decl = "        (const scalar_t*)grad_grad_w.data_ptr<scalar_t>(),\n" if need_grad_w else ""
    ggx_launch_decl = "            (const scalar_t*)grad_grad_x.data_ptr<scalar_t>(),\n" if need_grad_x else ""
    ggy_launch_decl = "            (const scalar_t*)grad_grad_y.data_ptr<scalar_t>(),\n" if need_grad_y else ""
    dw_alloc = "    auto d_w = torch::zeros_like(w);\n" if need_grad_w else ""
    dx_alloc = "    auto d_x = torch::zeros_like(x);\n" if need_grad_x else ""
    dy_alloc = "    auto d_y = torch::zeros_like(y);\n" if need_grad_y else ""
    dw_launch_decl = "            (scalar_t*)d_w.data_ptr<scalar_t>(),\n" if need_grad_w else ""
    dx_launch_decl = "            (scalar_t*)d_x.data_ptr<scalar_t>(),\n" if need_grad_x else ""
    dy_launch_decl = "            (scalar_t*)d_y.data_ptr<scalar_t>(),\n" if need_grad_y else ""
    double_result_names = ["d_grad_out"]
    if need_grad_w:
        double_result_names.append("d_w")
    if need_grad_x:
        double_result_names.append("d_x")
    if need_grad_y:
        double_result_names.append("d_y")
    ret_expr = "return {" + ", ".join(double_result_names) + "};"

    return f'''

torch::Tensor launcher_dummy_{bundle_name}();

std::vector<torch::Tensor> launcher_{bundle_name}(
{params_s})
{{
    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(), "w/x/y/grad_out must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(), "w/x/y/grad_out must be contiguous");
{src_checks}{dst_checks}{ggw_checks}{ggx_checks}{ggy_checks}
    TORCH_CHECK(w.dim() == 3 && x.dim() == 3 && y.dim() == 3 && grad_out.dim() == 3, "all primal tensors must be 3D");
    int B = (int){b_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U = (int)w.size(2);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V = (int)V64;
    TORCH_CHECK(B > 0 && V > 0, "B/V must be > 0");
    TORCH_CHECK(WB == 1 || WB == B, "w.size(0) must be 1 or B");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
{y_shape_check}
    TORCH_CHECK((int)grad_out.size(1) == V && (int)grad_out.size(2) == U, "grad_out shape mismatch");

    auto d_grad_out = torch::zeros_like(grad_out);
{dw_alloc}{dx_alloc}{dy_alloc}

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());
    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{
        launch_{bundle_name}<scalar_t>(
            (const scalar_t*)w.data_ptr<scalar_t>(),
            (const scalar_t*)x.data_ptr<scalar_t>(),
            (const scalar_t*)y.data_ptr<scalar_t>(),
            (const scalar_t*)grad_out.data_ptr<scalar_t>(),
{ggw_launch_decl}{ggx_launch_decl}{ggy_launch_decl}
            (scalar_t*)d_grad_out.data_ptr<scalar_t>(),
{dw_launch_decl}{dx_launch_decl}{dy_launch_decl}
            {src_arg}, {dst_arg}, B, WB, Iw, Ix, Ky, V, U, stream);
    }});
    GPU_KERNEL_LAUNCH_CHECK();
    {ret_expr}
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} Uniform1D double backward LARS jit impl");
}}
'''


def generate_code_uniform1d_double_bwd_with_scheduler(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    need_grad_x: bool = True,
    need_grad_y: bool = True,
    out_path: str = "generated_uniform1d_double_bwd_lars.cu",
    kernel_name: str = "uniform1d_double_bwd_lars",
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    block_size: int = 32,
) -> Any:
    """Generate the fused fully-unrolled LARS Uniform1D double backward."""
    ctx = _prepare_uniform1d_bwd_codegen_context(
        i_list=i_list, j_list=j_list, k_list=k_list, v_list=v_list,
        coeff_list=coeff_list, input_indices=input_indices,
        output_indices=output_indices, mode=mode,
        path_semantics=path_semantics,
    )
    cand_name = "lars_double_bwd_all_inputs"
    scheduler = LARSUniform1DDoubleBwdScheduler(
        paths=ctx.bwd_paths,
        need_grad_w=bool(need_grad_w),
        need_grad_x=bool(need_grad_x),
        need_grad_y=bool(need_grad_y),
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name=cand_name,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    schedule_result = scheduler.schedule()
    _dump_and_print_schedule_label_stats(
        operation="uniform1d_double_bwd",
        candidate_tag=cand_name,
        path_order=schedule_result.path_order,
        labels_by_path=_stats_labels_uniform1d_double_bwd(
            scheduler.paths,
            need_grad_w=bool(need_grad_w),
            need_grad_x=bool(need_grad_x),
            need_grad_y=bool(need_grad_y),
        ),
    )

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    grad_tag = f"gw{int(bool(need_grad_w))}_gx{int(bool(need_grad_x))}_gy{int(bool(need_grad_y))}"
    base_kernel_name = (
        f"{kernel_name}_{cand_name}_u{int(u_dim)}_path{ctx.P}_"
        f"{mode_str}_{ctx.layout_tag}_{grad_tag}"
    )
    code = emit_uniform1d_double_bwd_kernel_from_lars_schedule(
        schedule_result,
        kernel_name=base_kernel_name,
        mode=mode,
        need_grad_w=bool(need_grad_w),
        need_grad_x=bool(need_grad_x),
        need_grad_y=bool(need_grad_y),
        use_x_src=ctx.use_x_src,
        use_y_src=ctx.use_y_src,
        use_scatter=ctx.use_scatter,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        block_size=int(block_size),
    )
    code += "\n" + emit_uniform1d_double_bwd_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        need_grad_w=bool(need_grad_w),
        need_grad_x=bool(need_grad_x),
        need_grad_y=bool(need_grad_y),
        use_x_src=ctx.use_x_src,
        use_y_src=ctx.use_y_src,
        use_scatter=ctx.use_scatter,
    )
    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")
    if return_schedule:
        return {
            "name": cand_name,
            "code": code,
            "kernel_name": base_kernel_name,
            "schedule": schedule_result,
            "profile": schedule_result.profile,
        }
    return code


# =============================================================================
# STC fully-unrolled LARS double backward
# =============================================================================

@dataclass(frozen=True)
class STCDoubleBwdPath:
    """One logical STC double-backward path.

    First backward computes grad_x1 and grad_x0.  Given upstream gradients
    grad_grad_x1 and grad_grad_x0, double backward returns gradients with
    respect to (grad_out, x1, x0).

    Accumulator labels (dgo/dx1/dx0) are part of the LARS live set, so their
    lifetimes are scheduled together with read-only operands instead of being
    emitted as per-path global atomics.
    """
    pid: int
    x1_indices: Tuple[int, ...]
    x0_index: int
    v: int
    c: float

    @property
    def labels(self) -> Tuple[Label, ...]:
        labs: List[Label] = [
            ("go", int(self.v)),
            ("x0", int(self.x0_index)),
            ("ggx0", int(self.x0_index)),
        ]
        seen: Set[Label] = set(labs)
        for idx in self.x1_indices:
            lab = ("x1", int(idx))
            if lab not in seen:
                labs.append(lab); seen.add(lab)
        for idx in self.x1_indices:
            lab = ("ggx1", int(idx))
            if lab not in seen:
                labs.append(lab); seen.add(lab)
        for lab in (("dgo", int(self.v)), ("dx0", int(self.x0_index))):
            if lab not in seen:
                labs.append(lab); seen.add(lab)
        # grad_x0 depends on every x1 factor, so its upstream gradient creates
        # a d_x1 contribution even for degree-1 paths.
        for idx in self.x1_indices:
            lab = ("dx1", int(idx))
            if lab not in seen:
                labs.append(lab); seen.add(lab)
        return tuple(labs)


class LARSSTCDoubleBwdScheduler(LARSUniform1DScheduler):
    """LARS scheduler for STC double backward, including output accumulators."""

    _ACC_KINDS = {"dgo", "dx0", "dx1"}

    def __init__(
        self,
        paths: Sequence[STCPath],
        *,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "stc_double_bwd_lars",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = False,
    ):
        double_paths = [
            STCDoubleBwdPath(
                pid=int(p.pid),
                x1_indices=tuple(int(v) for v in p.x1_indices),
                x0_index=int(p.x0_index),
                v=int(p.v),
                c=float(p.c),
            )
            for p in paths
        ]
        # Reuse the mature STC LARS selection logic.  We override path emission,
        # accumulator initialization, and accumulator writeback below.
        super().__init__(
            double_paths,
            path_kind="stc",
            enable_secondary_affinity=enable_secondary_affinity,
            topk_candidates=topk_candidates,
            debug=debug,
            profile=profile,
            profile_name=profile_name,
            profile_interval=profile_interval,
            profile_seconds=profile_seconds,
            profile_print=profile_print,
        )

    def _path(self, pid: int) -> STCDoubleBwdPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        names = {
            "go": "grad_out",
            "ggx0": "grad_grad_x0",
            "ggx1": "grad_grad_x1",
            "dgo": "d_grad_out",
            "dx0": "d_x0",
            "dx1": "d_x1",
        }
        return f"{names.get(kind, kind)}[{idx}]"

    def _priority(self, lab: Label) -> int:
        # Accumulator labels benefit from staying resident across neighboring
        # paths just like forward output accumulators.
        return int(self.remaining_uses[lab]) + (1 if lab[0] in self._ACC_KINDS else 0)

    def _candidate_labels(self) -> List[Label]:
        candidates = {
            lab for pid in self.unscheduled
            for lab in self.path_label_sets[pid]
            if lab not in self.live
        }
        if not candidates:
            raise RuntimeError("No candidate label but no path is fireable.")
        if self.topk_candidates is not None and len(candidates) > self.topk_candidates:
            ranked = sorted(
                candidates,
                key=lambda lab: (
                    int(self.remaining_uses[lab]),
                    len(self.label_to_paths[lab] & self.unscheduled),
                    1 if lab[0] in self._ACC_KINDS else 0,
                    str(lab),
                ),
                reverse=True,
            )
            return ranked[: int(self.topk_candidates)]
        return list(candidates)

    def _emit_load_inst(self, lab: Label, reg: str, reason: str) -> None:
        if lab[0] in self._ACC_KINDS:
            self.instructions.append(
                Inst("init_acc", (reg, self._label_name(lab)), reason)
            )
        else:
            self.instructions.append(
                Inst("load", (reg, self._label_name(lab)), reason)
            )
            self.reloads += 1

    def _store_and_release(self, lab: Label, reason: str = "") -> None:
        if lab not in self.live:
            return
        reg = self.reg_of[lab]
        if lab[0] in self._ACC_KINDS:
            # All accumulator labels are initialized to zero and every path that
            # references one contributes before its final remaining_use reaches 0.
            self.instructions.append(
                Inst("store_acc", (self._label_name(lab), reg), reason)
            )
            self.dirty_outputs.discard(lab)
        else:
            self.instructions.append(
                Inst("release", (reg, self._label_name(lab)), reason)
            )
        self.live.remove(lab)
        self.reg_of.pop(lab, None)
        self.free_regs.append(reg)
        self.free_regs.sort(key=_lars_reg_id)
        self._invalidate_score_cache()

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)
        reg = self.reg_of
        rgo = reg[("go", int(p.v))]
        rx0 = reg[("x0", int(p.x0_index))]
        rggx0 = reg[("ggx0", int(p.x0_index))]
        rx1 = tuple(reg[("x1", int(idx))] for idx in p.x1_indices)
        rgg = tuple(reg[("ggx1", int(idx))] for idx in p.x1_indices)
        rdgo = reg[("dgo", int(p.v))]
        rdx0 = reg[("dx0", int(p.x0_index))]
        rdx1 = tuple(reg[("dx1", int(idx))] for idx in p.x1_indices)
        self.instructions.append(Inst(
            "stc_double_bwd_path",
            (
                tuple(int(v) for v in p.x1_indices),
                int(p.x0_index), int(p.v), float(p.c),
                rgo, rx0, rggx0, rx1, rgg, rdgo, rdx0, rdx1,
            ),
            f"path#{pid}: STC double backward",
        ))
        for lab in p.labels:
            if lab[0] in self._ACC_KINDS:
                self.dirty_outputs.add(lab)
        self.path_order.append(pid)
        self.unscheduled.remove(pid)
        for lab in p.labels:
            self.remaining_uses[lab] -= 1
        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")


def _stats_labels_stc_double_bwd(paths: Sequence[STCDoubleBwdPath]) -> List[List[str]]:
    rows: List[List[str]] = []
    for p in paths:
        labels = [
            f"grad_out[{p.v}]",
            f"x0[{p.x0_index}]",
            f"grad_grad_x0[{p.x0_index}]",
        ]
        labels.extend(f"x1[{int(idx)}]" for idx in p.x1_indices)
        labels.extend(f"grad_grad_x1[{int(idx)}]" for idx in p.x1_indices)
        labels.extend([f"d_grad_out[{p.v}]", f"d_x0[{p.x0_index}]"])
        labels.extend(f"d_x1[{int(idx)}]" for idx in p.x1_indices)
        rows.append(labels)
    return rows


def emit_stc_double_bwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    u_dim: Optional[int] = None,
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """Emit fully-unrolled STC double backward from a LARS schedule."""
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")
    reg_count = _max_lars_reg_count_any(schedule_result)

    lines: List[str] = []
    def ap(line: str = "") -> None:
        lines.append(line)

    def parse_ref(ref: str) -> Tuple[str, int]:
        if "[" not in ref or not ref.endswith("]"):
            raise ValueError(f"Bad STC double backward ref: {ref!r}")
        kind, tail = ref.split("[", 1)
        idx = int(tail[:-1])
        aliases = {
            "grad_out": "go", "grad_grad_x0": "ggx0",
            "grad_grad_x1": "ggx1",
            "d_grad_out": "dgo", "d_x0": "dx0", "d_x1": "dx1",
        }
        return aliases.get(kind, kind), idx

    def base_index(dim_name: str, idx: int) -> str:
        return f"((index_t)b * (index_t){dim_name} + (index_t){int(idx)}) * (index_t)U + (index_t)u"

    def x0_base_index(idx: int) -> str:
        # x0 is the original ungathered tensor.  STC uses x0_g=x0[i0], so
        # double backward must read and reduce gradients through i0[b].
        return f"((index_t)x0_b * (index_t)X0 + (index_t){int(idx)}) * (index_t)U + (index_t)u"

    def input_expr(kind: str, idx: int) -> str:
        if kind == "go": return f"grad_out[{base_index('V', idx)}]"
        if kind == "x0": return f"x0[{x0_base_index(idx)}]"
        if kind == "x1": return f"x1[{base_index('X1', idx)}]"
        if kind == "ggx0": return f"grad_grad_x0[{x0_base_index(idx)}]"
        if kind == "ggx1": return f"grad_grad_x1[{base_index('X1', idx)}]"
        raise ValueError(f"Bad STC double backward input kind: {kind}")

    def output_lhs(kind: str, idx: int) -> str:
        if kind == "dgo": return f"d_grad_out[{base_index('V', idx)}]"
        if kind == "dx0": return f"d_x0[{x0_base_index(idx)}]"
        if kind == "dx1": return f"d_x1[{base_index('X1', idx)}]"
        raise ValueError(f"Bad STC double backward output kind: {kind}")

    def mul_terms(terms: Sequence[str]) -> str:
        if not terms:
            return "scalar_t(1)"
        expr = str(terms[0])
        for term in terms[1:]:
            expr = f"({expr} * {term})"
        return expr

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")
    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ grad_out,")
    ap("    const scalar_t* __restrict__ x1,")
    ap("    const scalar_t* __restrict__ x0,")
    ap("    const int64_t* __restrict__ i0,")
    ap("    const scalar_t* __restrict__ grad_grad_x0,")
    ap("    const scalar_t* __restrict__ grad_grad_x1,")
    ap("    scalar_t* __restrict__ d_grad_out,")
    ap("    scalar_t* __restrict__ d_x1,")
    ap("    scalar_t* __restrict__ d_x0,")
    ap("    int B, int B0, int X1, int X0, int V, int U)")
    ap("{")
    ap("    const int b = (int)blockIdx.x;")
    ap("    if (b >= B) return;")
    ap("    const int64_t x0_b64 = i0[b];")
    ap("    if (x0_b64 < 0 || x0_b64 >= (int64_t)B0) return;")
    ap("    const index_t x0_b = (index_t)x0_b64;")
    ap("    const int tid = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        ap(f"            // stc-double-bwd inst {inst_id}: {inst.op}" + (f" | {comment}" if comment else ""))
        if inst.op == "load":
            reg, ref = inst.args
            kind, idx = parse_ref(str(ref))
            ap(f"            {reg} = {input_expr(kind, idx)};")
        elif inst.op == "init_acc":
            reg, _ref = inst.args
            ap(f"            {reg} = scalar_t(0);")
        elif inst.op == "stc_double_bwd_path":
            x1_indices, x0_idx, ov, coeff, rgo, rx0, rggx0, rx1, rgg, rdgo, rdx0, rdx1 = inst.args
            xs = tuple(int(v) for v in x1_indices)
            rx1 = tuple(str(v) for v in rx1)
            rgg = tuple(str(v) for v in rgg)
            rdx1 = tuple(str(v) for v in rdx1)
            c = _fmt_lars_float(float(coeff))
            prod_all = mul_terms(rx1)
            ap(f"            {rdgo} += scalar_t({c}) * {rggx0} * {prod_all};")
            for r in range(len(xs)):
                prod_except_r = mul_terms([
                    rx1[q] for q in range(len(xs)) if q != r
                ])
                ap(f"            {rdx1[r]} += scalar_t({c}) * {rggx0} * {rgo} * {prod_except_r};")
            for t in range(len(xs)):
                prod_except_t = mul_terms([rx1[q] for q in range(len(xs)) if q != t])
                ap(f"            {rdgo} += scalar_t({c}) * {rgg[t]} * {rx0} * {prod_except_t};")
                ap(f"            {rdx0} += scalar_t({c}) * {rgg[t]} * {rgo} * {prod_except_t};")
                for r in range(len(xs)):
                    if r == t:
                        continue
                    prod_except_tr = mul_terms([
                        rx1[q] for q in range(len(xs)) if q != t and q != r
                    ])
                    ap(f"            {rdx1[r]} += scalar_t({c}) * {rgg[t]} * {rgo} * {rx0} * {prod_except_tr};")
        elif inst.op == "store_acc":
            ref, reg = inst.args
            kind, idx = parse_ref(str(ref))
            if kind == "dx0":
                # Multiple b values may map to the same original x0 row.
                # LARS reduces all path contributions within one b/block; only
                # the final cross-block gather-backward reduction is atomic.
                ap(f"            atomicAdd(&{output_lhs(kind, idx)}, {reg});")
            else:
                ap(f"            {output_lhs(kind, idx)} = {reg};")
        elif inst.op == "release":
            pass
        else:
            raise ValueError(f"Unsupported STC double backward instruction op: {inst.op}")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")
    ap("static inline bool stc_db_mul3_fits_i32(int64_t a, int64_t b, int64_t c) {")
    ap("    if (a < 0 || b < 0 || c < 0) return false;")
    ap("    if (a == 0 || b == 0 || c == 0) return true;")
    ap("    constexpr int64_t L = 2147483647LL;")
    ap("    return a <= L / b && a * b <= L / c;")
    ap("}")
    ap("")
    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, const int64_t* i0,")
    ap("    const scalar_t* grad_grad_x0, const scalar_t* grad_grad_x1,")
    ap("    scalar_t* d_grad_out, scalar_t* d_x1, scalar_t* d_x0,")
    ap("    int B, int B0, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        grad_out, x1, x0, i0, grad_grad_x0, grad_grad_x1, d_grad_out, d_x1, d_x0, B, B0, X1, X0, V, U);")
    ap("}")
    ap("")
    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, const int64_t* i0,")
    ap("    const scalar_t* grad_grad_x0, const scalar_t* grad_grad_x1,")
    ap("    scalar_t* d_grad_out, scalar_t* d_x1, scalar_t* d_x0,")
    ap("    int B, int B0, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap("    bool use_i32 = stc_db_mul3_fits_i32(B, X1, U) && stc_db_mul3_fits_i32(B0, X0, U) && stc_db_mul3_fits_i32(B, V, U);")
    ap("    if (use_i32) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(grad_out,x1,x0,i0,grad_grad_x0,grad_grad_x1,d_grad_out,d_x1,d_x0,B,B0,X1,X0,V,U,stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(grad_out,x1,x0,i0,grad_grad_x0,grad_grad_x1,d_grad_out,d_x1,d_x0,B,B0,X1,X0,V,U,stream);")
    ap("    }")
    ap("}")
    return "\n".join(lines)


def emit_stc_double_bwd_launcher(bundle_name: str, *, u_dim: Optional[int] = None,
                                 x0_dim: Optional[int] = None,
                                 x1_dim: Optional[int] = None,
                                 v_dim: Optional[int] = None) -> str:
    checks: List[str] = []
    if u_dim is not None: checks.append(f'    TORCH_CHECK(U == {int(u_dim)}, "U mismatch for generated STC double backward kernel");')
    if x0_dim is not None: checks.append(f'    TORCH_CHECK(X0 == {int(x0_dim)}, "X0 mismatch for generated STC double backward kernel");')
    if x1_dim is not None: checks.append(f'    TORCH_CHECK(X1 == {int(x1_dim)}, "X1 mismatch for generated STC double backward kernel");')
    if v_dim is not None: checks.append(f'    TORCH_CHECK(V == {int(v_dim)}, "V mismatch for generated STC double backward kernel");')
    check_text = "\n".join(checks)
    return f'''

std::vector<torch::Tensor> launcher_{bundle_name}(
    torch::Tensor grad_out,
    torch::Tensor x1,
    torch::Tensor x0,
    torch::Tensor i0,
    torch::Tensor grad_grad_x0,
    torch::Tensor grad_grad_x1,
    int64_t V64)
{{
    TORCH_CHECK(grad_out.is_cuda() && x1.is_cuda() && x0.is_cuda() && i0.is_cuda() && grad_grad_x0.is_cuda() && grad_grad_x1.is_cuda(), "all tensors must be CUDA/HIP");
    TORCH_CHECK(grad_out.is_contiguous() && x1.is_contiguous() && x0.is_contiguous() && i0.is_contiguous() && grad_grad_x0.is_contiguous() && grad_grad_x1.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(x1.dim() == 3 && x0.dim() == 3 && grad_out.dim() == 3 && grad_grad_x0.dim() == 3 && grad_grad_x1.dim() == 3, "floating tensors must be 3D");
    TORCH_CHECK(i0.dim() == 1, "i0 must be 1D");
    TORCH_CHECK(i0.scalar_type() == torch::kInt64, "i0 must be int64");
    TORCH_CHECK(grad_out.scalar_type() == x1.scalar_type() && x1.scalar_type() == x0.scalar_type() && x1.scalar_type() == grad_grad_x0.scalar_type() && x1.scalar_type() == grad_grad_x1.scalar_type(), "dtype mismatch");
    TORCH_CHECK(grad_grad_x0.sizes() == x0.sizes(), "grad_grad_x0 shape must match x0");
    TORCH_CHECK(grad_grad_x1.sizes() == x1.sizes(), "grad_grad_x1 shape must match x1");
    int B = (int)x1.size(0);
    int B0 = (int)x0.size(0);
    int X1 = (int)x1.size(1);
    int X0 = (int)x0.size(1);
    int U = (int)x1.size(2);
    int V = (int)V64;
    TORCH_CHECK((int)x0.size(2) == U, "x0 U mismatch");
    TORCH_CHECK((int)i0.numel() == B, "i0 length must equal x1 batch size B");
    TORCH_CHECK((int)grad_out.size(0) == B && (int)grad_out.size(1) == V && (int)grad_out.size(2) == U, "grad_out shape mismatch");
{check_text}
    auto d_grad_out = torch::zeros_like(grad_out);
    auto d_x1 = torch::zeros_like(x1);
    auto d_x0 = torch::zeros_like(x0);
    GPU_Guard device_guard(x1.device());
    gpuStream_t stream = getCurrentGPUStream(x1.device().index());
    AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "{bundle_name}", [&] {{
        launch_{bundle_name}<scalar_t>(
            (const scalar_t*)grad_out.data_ptr<scalar_t>(),
            (const scalar_t*)x1.data_ptr<scalar_t>(),
            (const scalar_t*)x0.data_ptr<scalar_t>(),
            (const int64_t*)i0.data_ptr<int64_t>(),
            (const scalar_t*)grad_grad_x0.data_ptr<scalar_t>(),
            (const scalar_t*)grad_grad_x1.data_ptr<scalar_t>(),
            (scalar_t*)d_grad_out.data_ptr<scalar_t>(),
            (scalar_t*)d_x1.data_ptr<scalar_t>(),
            (scalar_t*)d_x0.data_ptr<scalar_t>(),
            B, B0, X1, X0, V, U, stream);
    }});
    GPU_KERNEL_LAUNCH_CHECK();
    return {{d_grad_out, d_x1, d_x0}};
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} STC double backward LARS jit impl");
}}
'''


def generate_code_stc_double_bwd_with_scheduler(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    num_out_segments: int,
    u_dim: int,
    out_path: str = "generated_stc_double_bwd_lars.cu",
    kernel_name: str = "stc_double_bwd_lars",
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    return_schedule: bool = False,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    block_size: int = 32,
) -> Any:
    """Generate fully-unrolled LARS STC double backward CUDA/HIP source."""
    paths = make_stc_paths_from_padded_lists(
        idx_lists, coeff_list, path_lens=path_lens, pad_value=pad_value
    )
    scheduler = LARSSTCDoubleBwdScheduler(
        paths,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name="stc_double_bwd_lars",
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    schedule_result = scheduler.schedule()
    _dump_and_print_schedule_label_stats(
        operation="stc_double_bwd",
        candidate_tag="stc_double_bwd_lars",
        path_order=schedule_result.path_order,
        labels_by_path=_stats_labels_stc_double_bwd(scheduler.paths),
    )
    _check_stc_path_bounds_for_bwd(
        paths, x0_dim=x0_dim, x1_dim=x1_dim, v_dim=int(num_out_segments)
    )
    idx_norm = _normalize_stc_padded_paths(
        idx_lists, coeff_list=coeff_list, path_lens=path_lens
    )
    lens = (
        infer_stc_path_lens_from_padded(idx_norm, coeff_list, pad_value=pad_value)
        if path_lens is None else
        path_lens.detach().cpu().to(torch.int64).reshape(-1)
    )
    base_kernel_name = (
        f"{kernel_name}_u{int(u_dim)}_path{len(paths)}_maxlen{int(lens.max().item())}"
    )
    code = emit_stc_double_bwd_kernel_from_lars_schedule(
        schedule_result,
        kernel_name=base_kernel_name,
        u_dim=int(u_dim), x0_dim=x0_dim, x1_dim=x1_dim,
        v_dim=int(num_out_segments), block_size=int(block_size),
    )
    code += "\n" + emit_stc_double_bwd_launcher(
        base_kernel_name, u_dim=int(u_dim), x0_dim=x0_dim,
        x1_dim=x1_dim, v_dim=int(num_out_segments),
    )
    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")
    if return_schedule:
        return {
            "name": "stc_double_bwd_lars",
            "code": code,
            "kernel_name": base_kernel_name,
            "schedule": schedule_result,
            "num_paths": len(paths),
            "max_live": schedule_result.max_live,
            "profile": schedule_result.profile,
        }
    return code


# =============================================================================
# Baseline full-unrolled codegen: no LARS and no resident accumulator reuse
# =============================================================================

def _make_baseline_wxy_paths_from_uniform1d_lists(
    i_list: List[int],
    j_list: List[int],
    k_list: List[int],
    v_list: List[int],
    coeff_list: List[float],
    *,
    path_semantics: str,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Convert external Uniform1D path indices to baseline-native tuples:
        (w_index, x_index, y_index, out_index, coeff)

    path_semantics="wxy": external (i,j,k) means w[i], x[j], y[k].
    path_semantics="xyw": external (i,j,k) means x[i], y[j], w[k].

    This baseline intentionally does not perform path scheduling, operand reuse,
    emits one straight-line block per cg path, in the original path order.
    """
    if path_semantics not in ("wxy", "xyw"):
        raise ValueError("path_semantics must be 'wxy' or 'xyw'")
    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

    paths: List[Tuple[int, int, int, int, float]] = []
    for i, j, k, v, c in zip(i_list, j_list, k_list, v_list, coeff_list):
        i = int(i); j = int(j); k = int(k); v = int(v); c = float(c)
        if path_semantics == "wxy":
            paths.append((i, j, k, v, c))
        else:
            paths.append((k, i, j, v, c))
    return paths


def _resolve_baseline_unroll_size(
    path_count: int,
    unroll_size: Optional[int],
) -> int:
    """Resolve the source-level manual path-unroll width.

    ``None`` keeps every path in one source-generated block.  A positive
    value smaller than ``path_count`` partitions the path list into static
    source-generated blocks of at most ``unroll_size`` paths.  Every path index
    and coefficient is embedded as a compile-time constant; no runtime path
    metadata lookup and no compiler unroll directive are used.
    """
    path_count = int(path_count)
    if path_count <= 0:
        raise ValueError("baseline codegen requires at least one path")
    if unroll_size is None:
        return path_count
    resolved = int(unroll_size)
    if resolved <= 0:
        raise ValueError("unroll_size must be a positive integer or None")
    return min(resolved, path_count)


def _baseline_path_write_targets(
    path: Tuple[int, int, int, int, float],
    *,
    direction: str,
    need_grad_w: bool,
) -> Tuple[Tuple[str, int], ...]:
    wi, xj, yk, ov, _coeff = path
    if direction == "fwd":
        return (("out", int(ov)),)
    if direction == "bwd":
        targets: List[Tuple[str, int]] = []
        if need_grad_w:
            targets.append(("grad_w", int(wi)))
        targets.extend([
            ("grad_x", int(xj)),
            ("grad_y", int(yk)),
        ])
        return tuple(targets)
    raise ValueError(f"Unsupported baseline direction: {direction!r}")


def _baseline_unroll_block_parallelism(
    block_paths: Sequence[Tuple[int, int, int, int, float]],
    *,
    direction: str,
    need_grad_w: bool,
    block_id: int,
    path_start: int,
) -> Dict[str, Any]:
    """Estimate source-level path ILP from write-dependency chains.

    Paths are assigned to the earliest dependency level that is legal with
    respect to previous writes to the same destination.  Paths on the same
    level have no write-after-write conflict and can be issued independently.

    For ten forward paths that all update the same ``out[v]``, the levels are
    1..10, ``parallel_paths`` is 1, and effective parallelism is 1/10.
    """
    last_level_by_target: Dict[Tuple[str, int], int] = {}
    level_counts: Counter[int] = Counter()
    target_counts: Counter[Tuple[str, int]] = Counter()
    path_levels: List[int] = []

    for path in block_paths:
        targets = _baseline_path_write_targets(
            path,
            direction=direction,
            need_grad_w=need_grad_w,
        )
        level = 1 + max(
            (last_level_by_target.get(target, 0) for target in targets),
            default=0,
        )
        path_levels.append(int(level))
        level_counts[int(level)] += 1
        for target in targets:
            last_level_by_target[target] = int(level)
            target_counts[target] += 1

    path_count = len(block_paths)
    parallel_paths = int(max(level_counts.values(), default=0))
    dependency_depth = int(max(path_levels, default=0))
    effective = (
        float(parallel_paths) / float(path_count)
        if path_count > 0 else 0.0
    )

    target_kind_counts: Dict[str, int] = Counter()
    for kind, _idx in target_counts:
        target_kind_counts[str(kind)] += 1

    return {
        "block_id": int(block_id),
        "path_start": int(path_start),
        "path_end": int(path_start + path_count),
        "path_count": int(path_count),
        "parallel_paths": int(parallel_paths),
        "dependency_depth": int(dependency_depth),
        "effective_parallelism": float(effective),
        "level_counts": {
            str(level): int(count)
            for level, count in sorted(level_counts.items())
        },
        "distinct_write_targets": int(len(target_counts)),
        "distinct_write_targets_by_kind": dict(target_kind_counts),
        "max_paths_per_write_target": int(max(target_counts.values(), default=0)),
    }


def analyze_baseline_unroll_parallelism(
    paths_wxy: Sequence[Tuple[int, int, int, int, float]],
    *,
    unroll_size: Optional[int],
    direction: str,
    need_grad_w: bool = True,
) -> Dict[str, Any]:
    """Return per-block and aggregate manual-unroll path parallelism stats."""
    path_count = len(paths_wxy)
    resolved_unroll = _resolve_baseline_unroll_size(path_count, unroll_size)
    blocks: List[Dict[str, Any]] = []
    for block_id, start in enumerate(range(0, path_count, resolved_unroll)):
        block_paths = paths_wxy[start: start + resolved_unroll]
        blocks.append(_baseline_unroll_block_parallelism(
            block_paths,
            direction=direction,
            need_grad_w=need_grad_w,
            block_id=block_id,
            path_start=start,
        ))

    parallel_sum = sum(int(block["parallel_paths"]) for block in blocks)
    block_count = len(blocks)
    weighted_effective = (
        float(parallel_sum) / float(path_count)
        if path_count > 0 else 0.0
    )
    return {
        "direction": str(direction),
        "num_paths": int(path_count),
        "requested_unroll_size": (
            None if unroll_size is None else int(unroll_size)
        ),
        "resolved_unroll_size": int(resolved_unroll),
        "num_unroll_blocks": int(block_count),
        "fully_unrolled": bool(resolved_unroll >= path_count),
        "parallel_paths_sum": int(parallel_sum),
        "average_parallel_paths": (
            float(parallel_sum) / float(block_count)
            if block_count > 0 else 0.0
        ),
        "min_parallel_paths": int(min(
            (int(block["parallel_paths"]) for block in blocks),
            default=0,
        )),
        "max_parallel_paths": int(max(
            (int(block["parallel_paths"]) for block in blocks),
            default=0,
        )),
        "effective_parallelism": float(weighted_effective),
        "average_dependency_depth": (
            sum(float(block["dependency_depth"]) for block in blocks)
            / float(block_count)
            if block_count > 0 else 0.0
        ),
        "blocks": blocks,
    }


def _print_baseline_unroll_parallelism_stats(stats: Dict[str, Any]) -> None:
    direction = str(stats["direction"])
    print(
        f"[BaselineUnroll][{direction}] "
        f"paths={stats['num_paths']} "
        f"unroll={stats['resolved_unroll_size']} "
        f"blocks={stats['num_unroll_blocks']} "
        f"avg_parallel_paths={stats['average_parallel_paths']:.3f} "
        f"effective_parallelism={stats['effective_parallelism']:.6f}",
        flush=True,
    )

    blocks = list(stats.get("blocks", []))
    if len(blocks) <= 16:
        visible = blocks
        omitted = 0
    else:
        visible = blocks[:8] + blocks[-8:]
        omitted = len(blocks) - len(visible)

    for pos, block in enumerate(visible):
        if omitted and pos == 8:
            print(
                f"[BaselineUnroll][{direction}] ... omitted {omitted} blocks ...",
                flush=True,
            )
        print(
            f"[BaselineUnroll][{direction}] "
            f"block={block['block_id']} "
            f"paths=[{block['path_start']},{block['path_end']}) "
            f"count={block['path_count']} "
            f"parallel={block['parallel_paths']} "
            f"depth={block['dependency_depth']} "
            f"effective={block['effective_parallelism']:.6f}",
            flush=True,
        )


def _dump_baseline_unroll_parallelism_stats(
    stats: Dict[str, Any],
    out_path: Optional[str],
) -> None:
    if not out_path:
        return
    target = Path(out_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def emit_fused_fwd_kernel_baseline_unrolled(
    paths_wxy: List[Tuple[int, int, int, int, float]],
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
    unroll_size: Optional[int] = None,
) -> str:
    """
    Emit a forward baseline kernel with tunable source-level path unrolling.

    ``unroll_size=None`` places all paths in one static source-generated
    block.  A smaller positive value partitions paths into static blocks of at
    most ``unroll_size`` paths.  Path indices and coefficients are emitted as
    literals, and each block performs operand loads, arithmetic, output
    accumulation, and delayed writeback through block-local scalar registers.
    No runtime metadata lookup or compiler unroll directive is emitted.
    """
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    path_count = len(paths_wxy)
    resolved_unroll = _resolve_baseline_unroll_size(path_count, unroll_size)
    mode_scalar_y = mode == "u,u,,u"
    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    scalar_t* __restrict__ out,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
        ap("")
    ap("    const int e_orig = e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    ap("    const int x_row = " + ("src_idx[e_orig];" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src_idx[e_orig];" if use_y_src else "e_orig;"))
    ap("    const int out_row = " + ("dst_idx[e_orig];" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;")
    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;")
    if mode_scalar_y:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base = (index_t)y_row * (index_t)Ky * (index_t)U;")
    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t out_base = (index_t)out_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t out_base = (index_t)out_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")

    # Each path block is emitted statically by Python.  ``unroll_size``
    # controls only the lexical register scope; there is no runtime path loop
    # and no device metadata table.
    for block_id, block_start in enumerate(
        range(0, path_count, resolved_unroll)
    ):
        block_paths = paths_wxy[
            block_start: block_start + resolved_unroll
        ]
        block_end = block_start + len(block_paths)
        out_targets = list(dict.fromkeys(
            int(path[3]) for path in block_paths
        ))
        out_target_slot = {
            int(ov): int(slot) for slot, ov in enumerate(out_targets)
        }

        ap(
            f"            // Manual source block#{block_id}: "
            f"paths [{block_start},{block_end}); constant metadata; "
            f"register-only intermediates."
        )
        ap("            {")

        # One register accumulator per distinct output touched by this block.
        for target_slot, _ov in enumerate(out_targets):
            ap(
                f"                scalar_t out_acc_b{block_id}_t{target_slot} "
                f"= scalar_t(0);"
            )
        if out_targets:
            ap("")

        # Phase 1: load every path operand into a uniquely named scalar local.
        # All path metadata is a source literal here.
        for local_slot, (wi, xj, yk, _ov, _coeff) in enumerate(block_paths):
            pid = block_start + local_slot
            w_expr = _lars_label_index_expr(
                "w", int(wi),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                x_dim=ix_dim,
                y_dim=ky_dim,
                w_dim=iw_dim,
                v_dim=v_dim,
            )
            x_expr = _lars_label_index_expr(
                "x", int(xj),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                x_dim=ix_dim,
                y_dim=ky_dim,
                w_dim=iw_dim,
                v_dim=v_dim,
            )
            y_expr = _lars_label_index_expr(
                "y", int(yk),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                x_dim=ix_dim,
                y_dim=ky_dim,
                w_dim=iw_dim,
                v_dim=v_dim,
            )
            ap(
                f"                // path#{pid}: "
                f"w[{int(wi)}], x[{int(xj)}], y[{int(yk)}]"
            )
            ap(
                f"                const scalar_t wv_b{block_id}_s{local_slot} "
                f"= w[{w_expr}];"
            )
            ap(
                f"                const scalar_t xv_b{block_id}_s{local_slot} "
                f"= x[{x_expr}];"
            )
            ap(
                f"                const scalar_t yv_b{block_id}_s{local_slot} "
                f"= y[{y_expr}];"
            )
        if block_paths:
            ap("")

        # Phase 2: compute one independent path delta in a scalar register.
        for local_slot, (_wi, _xj, _yk, _ov, coeff) in enumerate(block_paths):
            c = _fmt_lars_float(float(coeff))
            ap(
                f"                const scalar_t delta_b{block_id}_s{local_slot} "
                f"= scalar_t({c}) * "
                f"(wv_b{block_id}_s{local_slot} * "
                f"xv_b{block_id}_s{local_slot}) * "
                f"yv_b{block_id}_s{local_slot};"
            )
        if block_paths:
            ap("")

        # Phase 3: resolve same-output dependencies only in register state.
        for local_slot, (_wi, _xj, _yk, ov, _coeff) in enumerate(block_paths):
            target_slot = out_target_slot[int(ov)]
            ap(
                f"                out_acc_b{block_id}_t{target_slot} += "
                f"delta_b{block_id}_s{local_slot};"
            )
        if block_paths:
            ap("")

        # Phase 4: one global write per distinct output in this block.
        for target_slot, ov in enumerate(out_targets):
            out_expr = _lars_label_index_expr(
                "o", int(ov),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                x_dim=ix_dim,
                y_dim=ky_dim,
                w_dim=iw_dim,
                v_dim=v_dim,
            )
            value = f"out_acc_b{block_id}_t{target_slot}"
            if use_scatter:
                ap(f"                atomicAdd(&out[{out_expr}], {value});")
            else:
                ap(f"                out[{out_expr}] += {value};")

        ap("            }")
    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32_baseline_fwd(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32_baseline_fwd(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32_baseline_fwd(a, b)) return false;")
    ap("    return mul_fits_int32_baseline_fwd(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_baseline_fwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32_baseline_fwd((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32_baseline_fwd(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32_baseline_fwd(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32_baseline_fwd(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t out_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool out_ok = mul3_fits_int32_baseline_fwd(out_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && out_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        w, x, y, out,")
    ap("        src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_baseline_fwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                            kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    ap("            w, x, y, out, src_idx, dst_idx,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    ap("            w, x, y, out, src_idx, dst_idx,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    ap("        w, x, y, out, src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def emit_fused_bwd_kernel_baseline_unrolled(
    paths_wxy: List[Tuple[int, int, int, int, float]],
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
    unroll_size: Optional[int] = None,
) -> str:
    """
    Emit a backward baseline kernel with tunable source-level path unrolling.

    ``unroll_size=None`` places all paths in one static source-generated
    block.  A smaller positive value partitions paths into static blocks of at
    most ``unroll_size`` paths.  Path indices and coefficients are emitted as
    literals.  Each block loads all operands, computes all path gradients,
    accumulates conflicting gradient targets, and performs delayed global
    writeback through block-local scalar registers.  No runtime metadata lookup
    or compiler unroll directive is emitted.
    """
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    path_count = len(paths_wxy)
    resolved_unroll = _resolve_baseline_unroll_size(path_count, unroll_size)
    mode_scalar_y = mode == "u,u,,u"
    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")

    ap("template <typename scalar_t>")
    ap("__device__ __forceinline__ scalar_t warp_sum_xor_baseline_bwd(scalar_t v) {")
    ap("    for (int offset = 16; offset > 0; offset >>= 1) {")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("        v += __shfl_xor(v, offset);")
    ap("#else")
    ap("        v += __shfl_xor_sync(0xffffffff, v, offset);")
    ap("#endif")
    ap("    }")
    ap("    return v;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    if need_grad_w:
        ap("    scalar_t* __restrict__ grad_w,")
    ap("    scalar_t* __restrict__ grad_x,")
    ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
    ap("    const int e_orig = e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e_orig];")
    else:
        ap("    const int src = e_orig;")
    if use_scatter:
        ap("    const int dst = dst_idx[e_orig];")
    else:
        ap("    const int dst = e_orig;")
    ap("")
    ap("    const int x_row = " + ("src;" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src;" if use_y_src else "e_orig;"))
    ap("    const int go_row = " + ("dst;" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base  = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
        if need_grad_w:
            ap(f"    const index_t gw_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base  = (index_t)w_row * (index_t)Iw * (index_t)U;")
        if need_grad_w:
            ap("    const index_t gw_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base  = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
        ap(f"    const index_t gx_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base  = (index_t)x_row * (index_t)Ix * (index_t)U;")
        ap("    const index_t gx_base = (index_t)x_row * (index_t)Ix * (index_t)U;")

    if mode_scalar_y:
        ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky;")
        ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base  = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
            ap(f"    const index_t gy_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky * (index_t)U;")
            ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky * (index_t)U;")

    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t go_base = (index_t)go_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t go_base = (index_t)go_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")

    # Static source-generated path blocks.  All path metadata is embedded
    # as literals, and every block keeps path intermediates and same-target
    # accumulations in scalar locals until the final writeback phase.
    for block_id, block_start in enumerate(
        range(0, path_count, resolved_unroll)
    ):
        block_paths = paths_wxy[
            block_start: block_start + resolved_unroll
        ]
        block_end = block_start + len(block_paths)
        gw_targets = (
            list(dict.fromkeys(int(path[0]) for path in block_paths))
            if need_grad_w else []
        )
        gx_targets = list(dict.fromkeys(
            int(path[1]) for path in block_paths
        ))
        gy_targets = list(dict.fromkeys(
            int(path[2]) for path in block_paths
        ))
        gw_target_slot = {
            int(idx): int(slot) for slot, idx in enumerate(gw_targets)
        }
        gx_target_slot = {
            int(idx): int(slot) for slot, idx in enumerate(gx_targets)
        }
        gy_target_slot = {
            int(idx): int(slot) for slot, idx in enumerate(gy_targets)
        }

        ap(
            f"            // Manual source block#{block_id}: "
            f"paths [{block_start},{block_end}); constant metadata; "
            f"register-only intermediates."
        )
        ap("            {")

        # Block-local gradient accumulators.  They merge all write conflicts
        # before any global store or atomic operation is issued.
        for target_slot, _wi in enumerate(gw_targets):
            ap(
                f"                scalar_t gw_acc_b{block_id}_t{target_slot} "
                f"= scalar_t(0);"
            )
        for target_slot, _xj in enumerate(gx_targets):
            ap(
                f"                scalar_t gx_acc_b{block_id}_t{target_slot} "
                f"= scalar_t(0);"
            )
        for target_slot, _yk in enumerate(gy_targets):
            ap(
                f"                scalar_t gy_acc_b{block_id}_t{target_slot} "
                f"= scalar_t(0);"
            )
        if gw_targets or gx_targets or gy_targets:
            ap("")

        # Phase 1: load all path operands into registers with constant offsets.
        for local_slot, (wi, xj, yk, ov, _coeff) in enumerate(block_paths):
            pid = block_start + local_slot
            w_expr = _bwd_label_index_expr(
                "w", int(wi),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            x_expr = _bwd_label_index_expr(
                "x", int(xj),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            y_expr = _bwd_label_index_expr(
                "y", int(yk),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            go_expr = _bwd_label_index_expr(
                "go", int(ov),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            ap(
                f"                // path#{pid}: w[{int(wi)}], "
                f"x[{int(xj)}], y[{int(yk)}], grad_out[{int(ov)}]"
            )
            ap(
                f"                const scalar_t wv_b{block_id}_s{local_slot} "
                f"= w[{w_expr}];"
            )
            ap(
                f"                const scalar_t xv_b{block_id}_s{local_slot} "
                f"= x[{x_expr}];"
            )
            ap(
                f"                const scalar_t yv_b{block_id}_s{local_slot} "
                f"= y[{y_expr}];"
            )
            ap(
                f"                const scalar_t gov_b{block_id}_s{local_slot} "
                f"= grad_out[{go_expr}];"
            )
        if block_paths:
            ap("")

        # Phase 2: compute all path contributions in independent registers.
        for local_slot, (_wi, _xj, _yk, _ov, coeff) in enumerate(block_paths):
            c = _fmt_lars_float(float(coeff))
            if need_grad_w:
                ap(
                    f"                const scalar_t dgw_b{block_id}_s{local_slot} "
                    f"= scalar_t({c}) * gov_b{block_id}_s{local_slot} * "
                    f"xv_b{block_id}_s{local_slot} * "
                    f"yv_b{block_id}_s{local_slot};"
                )
            ap(
                f"                const scalar_t wg_b{block_id}_s{local_slot} "
                f"= wv_b{block_id}_s{local_slot} * "
                f"gov_b{block_id}_s{local_slot};"
            )
            ap(
                f"                const scalar_t dgx_b{block_id}_s{local_slot} "
                f"= scalar_t({c}) * wg_b{block_id}_s{local_slot} * "
                f"yv_b{block_id}_s{local_slot};"
            )
            ap(
                f"                const scalar_t dgy_b{block_id}_s{local_slot} "
                f"= scalar_t({c}) * wg_b{block_id}_s{local_slot} * "
                f"xv_b{block_id}_s{local_slot};"
            )
        if block_paths:
            ap("")

        # Phase 3: merge all conflicting gradient destinations in registers.
        for local_slot, (wi, xj, yk, _ov, _coeff) in enumerate(block_paths):
            if need_grad_w:
                target_slot = gw_target_slot[int(wi)]
                ap(
                    f"                gw_acc_b{block_id}_t{target_slot} += "
                    f"dgw_b{block_id}_s{local_slot};"
                )
            target_slot = gx_target_slot[int(xj)]
            ap(
                f"                gx_acc_b{block_id}_t{target_slot} += "
                f"dgx_b{block_id}_s{local_slot};"
            )
            target_slot = gy_target_slot[int(yk)]
            ap(
                f"                gy_acc_b{block_id}_t{target_slot} += "
                f"dgy_b{block_id}_s{local_slot};"
            )
        if block_paths:
            ap("")

        # Phase 4: delayed global writeback, once per distinct target/block.
        if need_grad_w:
            for target_slot, wi in enumerate(gw_targets):
                gw_expr = _bwd_label_index_expr(
                    "gw", int(wi),
                    mode_scalar_y=mode_scalar_y,
                    u_dim=u_dim,
                    iw_dim=iw_dim,
                    ix_dim=ix_dim,
                    ky_dim=ky_dim,
                    v_dim=v_dim,
                )
                ap(
                    f"                grad_w[{gw_expr}] += "
                    f"gw_acc_b{block_id}_t{target_slot};"
                )

        for target_slot, xj in enumerate(gx_targets):
            gx_expr = _bwd_label_index_expr(
                "gx", int(xj),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            ap(
                f"                atomicAdd(&grad_x[{gx_expr}], "
                f"gx_acc_b{block_id}_t{target_slot});"
            )

        for target_slot, yk in enumerate(gy_targets):
            gy_expr = _bwd_label_index_expr(
                "gy", int(yk),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            value = f"gy_acc_b{block_id}_t{target_slot}"
            if mode_scalar_y:
                reduced = f"gy_sum_b{block_id}_t{target_slot}"
                ap(
                    f"                const scalar_t {reduced} = "
                    f"warp_sum_xor_baseline_bwd({value});"
                )
                ap("                if (lane == 0) {")
                ap(
                    f"                    atomicAdd(&grad_y[{gy_expr}], "
                    f"{reduced});"
                )
                ap("                }")
            else:
                ap(
                    f"                atomicAdd(&grad_y[{gy_expr}], "
                    f"{value});"
                )

        ap("            }")
    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32_baseline_bwd(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32_baseline_bwd(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32_baseline_bwd(a, b)) return false;")
    ap("    return mul_fits_int32_baseline_bwd(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_baseline_bwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32_baseline_bwd((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32_baseline_bwd(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32_baseline_bwd(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32_baseline_bwd(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool go_ok = mul3_fits_int32_baseline_bwd(go_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && go_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_baseline_bwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                            kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def generate_code_uniform1d_fwd_baseline_unrolled(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    out_path: str = "generated_uniform1d_fwd_baseline_unrolled.cu",
    kernel_name: str = "uniform1d_fwd_baseline_unrolled",
    path_semantics: str = "wxy",
    unroll_size: Optional[int] = None,
    parallelism_stats_print: bool = True,
    parallelism_stats_path: Optional[str] = None,
    return_metadata: bool = False,
):
    """
    Generate a forward baseline with tunable manual path unrolling.

    ``unroll_size=None`` emits one static all-path register block.  Otherwise
    paths are partitioned into static blocks of at most ``unroll_size`` paths.
    All path indices and coefficients are literals, and all block-local path
    intermediates and same-target accumulations remain in scalar registers until
    delayed writeback.
    """
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    paths_wxy = _make_baseline_wxy_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )
    parallelism_stats = analyze_baseline_unroll_parallelism(
        paths_wxy,
        unroll_size=unroll_size,
        direction="fwd",
        need_grad_w=False,
    )
    resolved_unroll = int(parallelism_stats["resolved_unroll_size"])
    if parallelism_stats_print:
        _print_baseline_unroll_parallelism_stats(parallelism_stats)
    _dump_baseline_unroll_parallelism_stats(
        parallelism_stats, parallelism_stats_path
    )

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"
    if kernel_name and kernel_name != "uniform1d_fwd_baseline_unrolled":
        base_kernel_name = kernel_name
    else:
        base_kernel_name = f"uniform1d_baseline_unrolled_u{u_dim}_path{P}_unroll{resolved_unroll}_{mode_str}_{layout_tag}_fwd"

    code = emit_fused_fwd_kernel_baseline_unrolled(
        paths_wxy,
        kernel_name=base_kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        block_size=32,
        unroll_size=resolved_unroll,
    )

    code = code + "\n" + emit_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")

    if return_metadata:
        return {
            "name": "baseline_unrolled_fwd",
            "code": code,
            "kernel_name": base_kernel_name,
            "num_paths": int(P),
            "parallelism": parallelism_stats,
            "config": {
                "mode": mode,
                "path_semantics": path_semantics,
                "use_x_src": use_x_src,
                "use_y_src": use_y_src,
                "use_scatter": use_scatter,
                "manual_register_management": True,
                "manual_data_reuse": False,
                "manual_output_accumulation": True,
                "constant_path_metadata": True,
                "runtime_path_metadata": False,
                "manual_unroll": True,
                "unroll_size": int(resolved_unroll),
                "num_unroll_blocks": int(parallelism_stats["num_unroll_blocks"]),
                "fully_unrolled_paths": bool(parallelism_stats["fully_unrolled"]),
            },
        }
    return code


def generate_code_uniform1d_bwd_baseline_unrolled(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    out_path: str = "generated_uniform1d_bwd_baseline_unrolled.cu",
    kernel_name: str = "uniform1d_bwd_baseline_unrolled",
    path_semantics: str = "wxy",
    unroll_size: Optional[int] = None,
    parallelism_stats_print: bool = True,
    parallelism_stats_path: Optional[str] = None,
    return_metadata: bool = False,
):
    """
    Generate a backward baseline with tunable manual path unrolling.

    ``unroll_size=None`` emits one static all-path register block.  Otherwise
    paths are partitioned into static blocks of at most ``unroll_size`` paths.
    All path indices and coefficients are literals, and all block-local path
    intermediates and same-target accumulations remain in scalar registers until
    delayed writeback.
    """
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    paths_wxy = _make_baseline_wxy_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )
    parallelism_stats = analyze_baseline_unroll_parallelism(
        paths_wxy,
        unroll_size=unroll_size,
        direction="bwd",
        need_grad_w=need_grad_w,
    )
    resolved_unroll = int(parallelism_stats["resolved_unroll_size"])
    if parallelism_stats_print:
        _print_baseline_unroll_parallelism_stats(parallelism_stats)
    _dump_baseline_unroll_parallelism_stats(
        parallelism_stats, parallelism_stats_path
    )

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"
    grad_tag = "full" if need_grad_w else "nogradw"
    if kernel_name and kernel_name != "uniform1d_bwd_baseline_unrolled":
        base_kernel_name = kernel_name
    else:
        base_kernel_name = f"uniform1d_baseline_unrolled_u{u_dim}_path{P}_unroll{resolved_unroll}_{mode_str}_{layout_tag}_{grad_tag}_bwd"

    code = emit_fused_bwd_kernel_baseline_unrolled(
        paths_wxy,
        kernel_name=base_kernel_name,
        mode=mode,
        need_grad_w=need_grad_w,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        block_size=32,
        unroll_size=resolved_unroll,
    )

    code = code + "\n" + emit_lars_bwd_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        need_grad_w=need_grad_w,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")

    if return_metadata:
        return {
            "name": "baseline_unrolled_bwd",
            "code": code,
            "kernel_name": base_kernel_name,
            "num_paths": int(P),
            "parallelism": parallelism_stats,
            "config": {
                "mode": mode,
                "path_semantics": path_semantics,
                "need_grad_w": need_grad_w,
                "use_x_src": use_x_src,
                "use_y_src": use_y_src,
                "use_scatter": use_scatter,
                "manual_register_management": True,
                "manual_data_reuse": False,
                "block_local_gradient_accumulators": True,
                "resident_gradient_accumulators": False,
                "constant_path_metadata": True,
                "runtime_path_metadata": False,
                "manual_unroll": True,
                "unroll_size": int(resolved_unroll),
                "num_unroll_blocks": int(parallelism_stats["num_unroll_blocks"]),
                "fully_unrolled_paths": bool(parallelism_stats["fully_unrolled"]),
            },
        }
    return code
