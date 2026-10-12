"""FastEq scheduling: placement."""
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
    Inst,
    LARSPlacementConfig,
    ScheduleResult,
)


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
