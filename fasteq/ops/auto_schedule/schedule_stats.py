"""FastEq scheduling: schedule stats."""
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
    STCDoubleBwdPath,
    U1DDoubleBwdPath,
)


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
