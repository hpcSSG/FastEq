"""FastEq scheduling: cuda common."""
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

from .models import (
    Label,
    ScheduleResult,
)


def _to_int_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().tolist()]


def _to_float_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]


def _lars_reg_id(reg: str) -> int:
    if not isinstance(reg, str) or not reg.startswith("r"):
        raise ValueError(f"Bad LARS register name: {reg!r}")
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


def _cuda_scalar_warp_sum_preamble() -> str:
    """CUDA-only reduction for full or contiguous partial 32-lane tiles."""
    return r"""
#if !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)
#ifndef FASTEQ_CUDA_SCALAR_WARP_SUM_DEFINED
#define FASTEQ_CUDA_SCALAR_WARP_SUM_DEFINED
template <typename scalar_t>
__device__ __forceinline__ scalar_t fasteq_cuda_warp_sum(scalar_t value) {
    const unsigned mask = __activemask();
    const int lane = (int)threadIdx.x & 31;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const scalar_t other = __shfl_down_sync(mask, value, offset, 32);
        const int source_lane = lane + offset;
        if (source_lane < 32 && ((mask >> source_lane) & 1u)) {
            value += other;
        }
    }
    return value;
}
#endif
#endif
"""


def _emit_backend_gradient_atomic(ap, pointer: str, value: str, *, scalar_channel: bool) -> None:
    """Scalar channels reduce on CUDA; HIP and vector channels use per-lane atomics."""
    if not scalar_channel:
        ap(f"            atomicAdd({pointer}, {value});")
        return
    ap("            {")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap(f"                atomicAdd({pointer}, {value});")
    ap("#else")
    ap(f"                const scalar_t warp_sum = fasteq_cuda_warp_sum({value});")
    ap("                if (((int)threadIdx.x & 31) == 0) {")
    ap(f"                    atomicAdd({pointer}, warp_sum);")
    ap("                }")
    ap("#endif")
    ap("            }")


def _split_grad_acc_name(grad_kind: str, idx: int) -> str:
    if grad_kind == "gw":
        return f"gw_acc_i_{int(idx)}"
    if grad_kind == "gx":
        return f"gx_acc_j_{int(idx)}"
    if grad_kind == "gy":
        return f"gy_acc_k_{int(idx)}"
    raise ValueError(f"Bad grad_kind: {grad_kind}")
