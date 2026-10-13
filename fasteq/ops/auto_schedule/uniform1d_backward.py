"""FastEq scheduling: uniform1d backward."""
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
    _max_lars_reg_count_any,
    _parse_bwd_lars_label_ref,
    _sanitize_cuda_comment,
    _split_grad_acc_name,
    _to_float_list,
    _to_int_list,
)
from .metadata import (
    _make_lars_bwd_paths_from_uniform1d_lists,
)
from .models import (
    BWD_OUTPUT_KINDS,
    BWD_SPLIT_KINDS,
    BwdCodegenContext,
    LARSPlacementConfig,
    ScheduleResult,
)
from .placement import (
    _resolve_lars_placement_config,
    apply_lars_lifetime_placement,
    dump_lars_variable_stats,
)
from .schedule_stats import (
    _dump_and_print_schedule_label_stats,
    _stats_labels_uniform1d_bwd_fused,
    _stats_labels_uniform1d_bwd_split,
)
from .schedulers import (
    LARSUniform1DBwdScheduler,
    LARSUniform1DBwdSplitScheduler,
)


def _dense_gradient_allocation(schedule_result: Optional[ScheduleResult], kind: str,
                               tensor: str, default_statement: str) -> str:
    """Skip memset only for complete final stores, with runtime shape guards."""
    if os.environ.get("FASTEQ_DENSE_BWD_SKIP_ZERO", "1").strip().lower() in ("0", "false", "off", "no"):
        return default_statement
    if schedule_result is None:
        return default_statement
    summary = schedule_result.placement_summary
    count = summary.get("fully_written_segment_counts", {}).get("grad_" + kind[1])
    if not summary.get("exclusive_outputs") or count is None:
        return default_statement
    if kind == "gw":
        condition = f"(WB != 1 || B == 1) && w.size(1) == {int(count)}"
    else:
        condition = f"{tensor}.size(0) == B && {tensor}.size(1) == {int(count)}"
    return (f"    auto grad_{kind[1]} = ({condition}) ? torch::empty_like({tensor})"
            f" : torch::zeros_like({tensor});\n")


def _emit_gradient_writeback(ap, kind: str, index_expr: str, value_expr: str, *,
                             exclusive: bool, scalar_y: bool) -> None:
    """Emit one final gradient write; scalar-y and broadcast weights still sum."""
    pointer = f"&grad_{kind[1]}[{index_expr}]"
    if kind == "gw":
        if exclusive:
            ap("            if (WB == 1 && B > 1) {")
            ap(f"                atomicAdd({pointer}, {value_expr});")
            ap("            } else {")
            ap(f"                grad_w[{index_expr}] = {value_expr};")
            ap("            }")
        else:
            ap(f"            grad_w[{index_expr}] = {value_expr};")
    elif kind == "gx" or (kind == "gy" and not scalar_y):
        if exclusive:
            ap(f"            grad_{kind[1]}[{index_expr}] = {value_expr};")
        else:
            ap(f"            atomicAdd({pointer}, {value_expr});")
    elif kind == "gy":
        _emit_backend_gradient_atomic(ap, pointer, value_expr, scalar_channel=True)
    else:
        raise ValueError(f"Bad backward gradient kind: {kind}")


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
    exclusive = bool(schedule_result.placement_summary.get("exclusive_outputs")) and not (
        use_x_src or use_y_src or use_scatter
    )

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
            _emit_gradient_writeback(ap, "gw", expr, str(value_expr), exclusive=exclusive, scalar_y=mode_scalar_y)
        elif kind == "gx":
            _emit_gradient_writeback(ap, "gx", expr, str(value_expr), exclusive=exclusive, scalar_y=mode_scalar_y)
        elif kind == "gy":
            _emit_gradient_writeback(ap, "gy", expr, value_expr, exclusive=exclusive, scalar_y=mode_scalar_y)
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
    ap(_cuda_scalar_warp_sum_preamble())
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
                _emit_gradient_writeback(ap, "gw", expr, str(reg), exclusive=exclusive, scalar_y=mode_scalar_y)
            elif kind == "gx":
                _emit_gradient_writeback(ap, "gx", expr, str(reg), exclusive=exclusive, scalar_y=mode_scalar_y)
            elif kind == "gy":
                _emit_gradient_writeback(ap, "gy", expr, str(reg), exclusive=exclusive, scalar_y=mode_scalar_y)
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
        _emit_gradient_writeback(ap, "gy", expr, f"gy_acc_k_{gy_idx}", exclusive=exclusive, scalar_y=mode_scalar_y)

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
    schedule_result: Optional[ScheduleResult] = None,
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
    allocation_schedule = schedule_result if not (use_x_src or use_y_src or use_scatter) else None
    if need_grad_w:
        grad_w_alloc = _dense_gradient_allocation(allocation_schedule, "gw", "w", grad_w_alloc)
    grad_x_alloc = _dense_gradient_allocation(allocation_schedule, "gx", "x", "    auto grad_x = torch::zeros_like(x);\n")
    if mode == "u,u,u,u":
        gy_alloc = _dense_gradient_allocation(allocation_schedule, "gy", "y", gy_alloc).rstrip("\n")
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
{grad_w_alloc}{grad_x_alloc}{gy_alloc}

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

// CUDA scalar gradients use warp sums; HIP uses per-lane atomics.
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
''' + _cuda_scalar_warp_sum_preamble()


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
    exclusive = bool(schedule_result.placement_summary.get("exclusive_outputs")) and not (
        use_x_src or use_y_src or use_scatter
    )

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
            _emit_gradient_writeback(ap, "gw", expr, str(value_expr), exclusive=exclusive, scalar_y=mode_scalar_y)
        elif grad_kind == "gx":
            _emit_gradient_writeback(ap, "gx", expr, str(value_expr), exclusive=exclusive, scalar_y=mode_scalar_y)
        else:
            _emit_gradient_writeback(ap, "gy", expr, value_expr, exclusive=exclusive, scalar_y=mode_scalar_y)

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
            _emit_gradient_writeback(ap, "gw", expr, str(acc_expr), exclusive=exclusive, scalar_y=mode_scalar_y)
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
            _emit_gradient_writeback(ap, "gx", expr, str(acc_expr), exclusive=exclusive, scalar_y=mode_scalar_y)
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
            _emit_gradient_writeback(ap, "gy", expr, acc_expr, exclusive=exclusive, scalar_y=mode_scalar_y)
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
    gradient_schedules: Optional[Dict[str, ScheduleResult]] = None,
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
    allocation_schedules = gradient_schedules or {}
    if use_x_src or use_y_src or use_scatter:
        allocation_schedules = {}
    if need_grad_w:
        grad_w_alloc = _dense_gradient_allocation(allocation_schedules.get("gw"), "gw", "w", grad_w_alloc)
    if need_grad_y and mode == "u,u,u,u":
        gy_alloc = _dense_gradient_allocation(allocation_schedules.get("gy"), "gy", "y", gy_alloc).rstrip("\n")
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
    if need_grad_x:
        grad_x_alloc = _dense_gradient_allocation(allocation_schedules.get("gx"), "gx", "x", grad_x_alloc)
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
    need_grad_x: bool = True,
    need_grad_y: bool = True,
) -> bool:
    """Resolve strategy while respecting the current fused gx/gy capability.

    New flags default to True so existing callers keep their previous behavior.
    Invalid strategy names remain errors, including partial-gradient requests.
    """
    force_split = not bool(need_grad_x) or not bool(need_grad_y)
    if isinstance(split_backward, str):
        split_mode = split_backward.lower()
        if split_mode in ("auto",):
            return force_split or int(path_count) > int(split_path_threshold)
        if split_mode in ("split", "true", "always", "on", "yes"):
            return True
        if split_mode in ("fused", "false", "never", "off", "no"):
            return force_split
        raise ValueError(
            "split_backward must be bool or one of "
            "'auto', 'split'/'always'/'on', 'fused'/'never'/'off'"
        )
    return force_split or bool(split_backward)


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
                exclusive_outputs=not (ctx.input_indices or ctx.output_indices),
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
                gradient_schedules=schedules,
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
        # Also protect direct callers that bypass the unified dispatcher.
        return _generate_code_uniform1d_bwd_split_from_context(
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
            block_size=block_size,
            placement_config=placement_config,
        )
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
        exclusive_outputs=not (ctx.input_indices or ctx.output_indices),
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
        schedule_result=schedule_result,
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
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    split_backward: Union[bool, str] = "fused",
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
        need_grad_x=need_grad_x,
        need_grad_y=need_grad_y,
    )

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
