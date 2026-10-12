"""FastEq scheduling: baseline codegen."""
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

from .cuda_common import (
    _bwd_label_index_expr,
    _cuda_scalar_warp_sum_preamble,
    _emit_backend_gradient_atomic,
    _fmt_lars_float,
    _lars_label_index_expr,
    _to_float_list,
    _to_int_list,
)
from .uniform1d_backward import (
    emit_lars_bwd_launcher,
)
from .uniform1d_forward import (
    emit_launcher,
)


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

    ap(_cuda_scalar_warp_sum_preamble())

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
            _emit_backend_gradient_atomic(ap, f"&grad_y[{gy_expr}]", value, scalar_channel=mode_scalar_y)

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
