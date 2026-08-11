#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCU runner + CSV exporter + parser:
- Filter only selected kernels via regex
- Export CSV from .ncu-rep
- Handle BOTH CSV layouts:
  1) WIDE: metrics are columns (e.g., gpu__time_duration.sum is a column)
  2) LONG: columns include Metric Name/Metric Unit/Metric Value (your current CSV)
- For SAME Kernel Name (multiple launches due to different input sizes), report:
  MAX time (converted to ms) + util at that max-time instance.

Usage:
  python ncu_max_time_by_kernelname.py --out cueq_mace_large -- python3 mace_batch.py large float64

Optional:
  --target-processes application-only|all
  --extra-ncu "--set full --clock-control none"
  --tmpdir ~/tmp_ncu
  --export-mode csv|raw   (csv: usually WIDE, raw: often LONG; but exporter may still vary by ncu version)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =========================
# Kernels you care about
# =========================
MPTP_FWD = [
    #"segmented_polynomial_algo6_4",
    "segmented_polynomial_algo4_2"
]
MPTP_BWD = [
    #"segmented_polynomial_bwd_split1_of_3_algo1_1",
    #"segmented_polynomial_bwd_split2_of_3_algo1_1",
    #"segmented_polynomial_bwd_split3_of_3_algo1_4",
    "segmented_polynomial_bwd_algo1_1"
]
STC_FWD = [
    "segmented_polynomial_algo1_1",
    #"segmented_polynomial_split1_of_2_algo1_4",
    #"segmented_polynomial_split2_of_2_algo1_4",
]
STC_BWD = [
    #"segmented_polynomial_bwd_algo1_1", 
    "segmented_polynomial_bwd_algo1_3"
]

ALL_KERNEL_SUBSTRS: List[str] = MPTP_FWD + MPTP_BWD + STC_FWD + STC_BWD


# =========================
# Metrics (names)
# =========================
TIME_METRIC = "gpu__time_duration.sum"
COMPUTE_UTIL_METRIC = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
MEM_UTIL_METRIC = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
DEFAULT_METRICS = [TIME_METRIC, COMPUTE_UTIL_METRIC, MEM_UTIL_METRIC]


@dataclass
class KernelMax:
    max_time_ms: float = 0.0
    compute_util_at_max: Optional[float] = None
    mem_util_at_max: Optional[float] = None
    # debug info for the slowest instance
    max_id: Optional[str] = None
    max_grid: Optional[str] = None
    max_block: Optional[str] = None
    max_stream: Optional[str] = None
    max_context: Optional[str] = None
    max_process_id: Optional[str] = None


def ensure_tmpdir(tmpdir: str) -> None:
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir


def run_checked(cmd: List[str]) -> None:
    print("+ " + " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def build_kernel_regex(substrings: List[str]) -> str:
    escaped = [re.escape(s) for s in substrings]
    return r".*(" + "|".join(escaped) + r").*"


def export_csv(rep_path: str, csv_path: str, export_mode: str) -> None:
    """
    export_mode:
      - "csv": ncu --import rep --csv
      - "raw": ncu --import rep --csv --page raw  (often LONG format)
    """
    cmd = ["ncu", "--import", rep_path, "--csv"]
    if export_mode == "raw":
        cmd += ["--page", "raw"]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        p = subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.PIPE, text=True)
    if p.stderr.strip():
        print("[NCU STDERR]", p.stderr.strip())


def _parse_float(s: str) -> float:
    s = (s or "").strip()
    if not s or s.upper() == "N/A":
        return float("nan")
    return float(s.replace(",", ""))


def _unit_to_ms_scale(unit: str) -> float:
    """
    Convert unit string -> multiplier to convert that unit into milliseconds.
    """
    u = (unit or "").strip().lower()
    if u in ("ms", "msec", "millisecond", "milliseconds"):
        return 1.0
    if u in ("us", "µs", "usec", "microsecond", "microseconds"):
        return 1e-3
    if u in ("ns", "nsec", "nanosecond", "nanoseconds"):
        return 1e-6
    if u in ("s", "sec", "second", "seconds"):
        return 1e3
    # Unknown -> assume ms, but warn
    print(f"[WARN] Unknown time unit {unit!r}, assuming ms.")
    return 1.0


def _detect_csv_layout(csv_path: str) -> Tuple[str, List[str]]:
    """
    Return ("wide"|"long", fieldnames)
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f)
        fieldnames = dr.fieldnames or []

    if "Metric Name" in fieldnames and "Metric Value" in fieldnames:
        return "long", fieldnames
    if TIME_METRIC in fieldnames and "Kernel Name" in fieldnames:
        return "wide", fieldnames
    # Fallback: if Kernel Name exists and Metric Name exists, treat as long
    if "Kernel Name" in fieldnames and "Metric Name" in fieldnames:
        return "long", fieldnames
    return "unknown", fieldnames


def parse_wide_csv_max(csv_path: str) -> Dict[str, KernelMax]:
    """
    WIDE CSV: metrics are columns. Usually has a unit row after header.
    We read the unit row for TIME_METRIC to convert to ms, then pick max per Kernel Name.
    """
    out: Dict[str, KernelMax] = {}

    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f)
        fieldnames = dr.fieldnames or []

        required = ["Kernel Name", TIME_METRIC, COMPUTE_UTIL_METRIC, MEM_UTIL_METRIC]
        missing = [c for c in required if c not in fieldnames]
        if missing:
            raise RuntimeError(f"WIDE CSV missing columns: {missing}. Have: {fieldnames[:30]} ...")

        unit_row = next(dr, None)
        time_scale = 1.0
        first_data_row = None

        if unit_row is not None:
            k0 = (unit_row.get("Kernel Name") or "").strip()
            u0 = (unit_row.get(TIME_METRIC) or "").strip()
            # unit row heuristic
            is_unit_row = (k0 == "") and (u0 != "") and (not u0.replace(".", "", 1).isdigit())
            if is_unit_row:
                time_scale = _unit_to_ms_scale(u0)
            else:
                first_data_row = unit_row

        def handle_row(row: Dict[str, str]) -> None:
            kname = (row.get("Kernel Name") or "").strip()
            if not kname:
                return

            t_raw = _parse_float(row.get(TIME_METRIC) or "")
            if t_raw != t_raw:
                return
            t_ms = t_raw * time_scale

            cu = _parse_float(row.get(COMPUTE_UTIL_METRIC) or "")
            mu = _parse_float(row.get(MEM_UTIL_METRIC) or "")
            cu_val = None if (cu != cu) else cu
            mu_val = None if (mu != mu) else mu

            km = out.setdefault(kname, KernelMax())
            if t_ms > km.max_time_ms:
                km.max_time_ms = t_ms
                km.compute_util_at_max = cu_val
                km.mem_util_at_max = mu_val
                km.max_id = (row.get("ID") or "").strip() or None
                km.max_grid = (row.get("Grid Size") or "").strip() or None
                km.max_block = (row.get("Block Size") or "").strip() or None
                km.max_stream = (row.get("Stream") or "").strip() or None
                km.max_context = (row.get("Context") or "").strip() or None
                km.max_process_id = (row.get("Process ID") or "").strip() or None

        if first_data_row is not None:
            handle_row(first_data_row)

        for row in dr:
            handle_row(row)

    print(f"[INFO] Parsed WIDE CSV. TIME unit scale_to_ms={time_scale}.")
    return out


def parse_long_csv_max(csv_path: str) -> Dict[str, KernelMax]:
    """
    LONG CSV: has columns:
      Kernel Name, Metric Name, Metric Unit, Metric Value, plus ID/Process ID/Grid/Block/Stream/Context ...
    Multiple rows per launch instance (one per metric).
    We:
      1) Group rows by an "instance key" (Kernel Name + ID + Context + Stream + Grid + Block + Process ID)
      2) For each instance, collect TIME_METRIC / COMPUTE_UTIL_METRIC / MEM_UTIL_METRIC
      3) For each Kernel Name, select the instance with MAX time (converted to ms using Metric Unit)
    """
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f)
        fieldnames = dr.fieldnames or []

        for c in ["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]:
            if c not in fieldnames:
                raise RuntimeError(f"LONG CSV missing required column {c!r}. Have: {fieldnames[:30]} ...")

        # choose stable columns if present
        def get(row: Dict[str, str], col: str) -> str:
            return (row.get(col) or "").strip()

        # Collect per-instance metrics
        # instance key: (Kernel Name, ID, Process ID, Context, Stream, Grid Size, Block Size)
        per_inst: Dict[Tuple[str, str, str, str, str, str, str], Dict[str, Tuple[float, str]]] = {}

        for row in dr:
            kname = get(row, "Kernel Name")
            mname = get(row, "Metric Name")
            if not kname or not mname:
                continue
            if mname not in (TIME_METRIC, COMPUTE_UTIL_METRIC, MEM_UTIL_METRIC):
                continue

            inst_key = (
                kname,
                get(row, "ID"),
                get(row, "Process ID"),
                get(row, "Context"),
                get(row, "Stream"),
                get(row, "Grid Size"),
                get(row, "Block Size"),
            )

            val = _parse_float(get(row, "Metric Value"))
            unit = get(row, "Metric Unit")
            if val != val:
                continue

            d = per_inst.setdefault(inst_key, {})
            d[mname] = (val, unit)

        out: Dict[str, KernelMax] = {}

        for inst_key, md in per_inst.items():
            kname, _id, pid, ctx, stream, grid, block = inst_key

            if TIME_METRIC not in md:
                continue

            time_val, time_unit = md[TIME_METRIC]
            time_scale = _unit_to_ms_scale(time_unit)
            t_ms = time_val * time_scale

            cu_val: Optional[float] = None
            mu_val: Optional[float] = None
            if COMPUTE_UTIL_METRIC in md:
                cu_val = md[COMPUTE_UTIL_METRIC][0]  # unit is %
            if MEM_UTIL_METRIC in md:
                mu_val = md[MEM_UTIL_METRIC][0]

            km = out.setdefault(kname, KernelMax())
            if t_ms > km.max_time_ms:
                km.max_time_ms = t_ms
                km.compute_util_at_max = cu_val
                km.mem_util_at_max = mu_val
                km.max_id = _id or None
                km.max_process_id = pid or None
                km.max_context = ctx or None
                km.max_stream = stream or None
                km.max_grid = grid or None
                km.max_block = block or None

    print("[INFO] Parsed LONG CSV (Metric Name/Value rows).")
    return out


def fmt_pct(x: Optional[float]) -> str:
    return "  N/A " if x is None else f"{x:6.2f}%"


def classify(kernel_name: str) -> Optional[Tuple[str, str]]:
    for s in MPTP_FWD:
        if s in kernel_name:
            return ("MPTP", "FWD")
    for s in MPTP_BWD:
        if s in kernel_name:
            return ("MPTP", "BWD")
    for s in STC_FWD:
        if s in kernel_name:
            return ("STC", "FWD")
    for s in STC_BWD:
        if s in kernel_name:
            return ("STC", "BWD")
    return None


def print_max_summary(maxmap: Dict[str, KernelMax]) -> None:
    rows: List[Tuple[str, str, float, Optional[float], Optional[float], str, KernelMax]] = []
    for kname, km in maxmap.items():
        cls = classify(kname)
        if cls is None:
            continue
        grp, direction = cls
        rows.append((grp, direction, km.max_time_ms, km.compute_util_at_max, km.mem_util_at_max, kname, km))

    if not rows:
        print("[WARN] No matched kernels found in parsed results.")
        return

    rows.sort(key=lambda x: (x[0], x[1], -x[2]))

    print("\n=== Per-kernel MAX time by Kernel Name ===")
    print(f"{'Group':<5} {'Dir':<3} {'MaxTime(ms)':>12} {'Comp@Max':>10} {'Mem@Max':>10}  Kernel Name  [slowest instance]")
    print("-" * 170)
    for grp, direction, tms, cu, mu, kname, km in rows:
        extra = []
        if km.max_id: extra.append(f"ID={km.max_id}")
        if km.max_process_id: extra.append(f"PID={km.max_process_id}")
        if km.max_context: extra.append(f"Ctx={km.max_context}")
        if km.max_stream: extra.append(f"Stream={km.max_stream}")
        if km.max_grid: extra.append(f"Grid={km.max_grid}")
        if km.max_block: extra.append(f"Block={km.max_block}")
        extra_s = (" [" + ", ".join(extra) + "]") if extra else ""
        print(f"{grp:<5} {direction:<3} {tms:12.3f} {fmt_pct(cu):>10} {fmt_pct(mu):>10}  {kname}{extra_s}")

    # Optional: group max (max of max)
    group_max: Dict[Tuple[str, str], float] = {}
    for grp, direction, tms, *_ in rows:
        group_max[(grp, direction)] = max(group_max.get((grp, direction), 0.0), tms)

    print("\n=== Group MAX (max of max) ===")
    print(f"{'Group':<5} {'Dir':<3} {'MaxTime(ms)':>12}")
    print("-" * 30)
    for key in [("MPTP", "FWD"), ("MPTP", "BWD"), ("STC", "FWD"), ("STC", "BWD")]:
        if key in group_max:
            print(f"{key[0]:<5} {key[1]:<3} {group_max[key]:12.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output prefix, writes .ncu-rep and .csv")
    ap.add_argument("--target-processes", default="application-only", choices=["application-only", "all"])
    ap.add_argument("--extra-ncu", default="", help='extra ncu args, e.g. \'--set full --clock-control none\'')
    ap.add_argument("--tmpdir", default=os.path.expanduser("~/tmp_ncu"))
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="comma-separated metrics")
    ap.add_argument("--export-mode", default="csv", choices=["csv", "raw"], help="ncu --import export mode")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="workload command after --")
    args = ap.parse_args()

    if not args.cmd or args.cmd[0] != "--":
        print("ERROR: workload must be specified after '--'")
        print("Example: python script.py --out run1 -- python3 mace_batch.py large float64")
        sys.exit(2)

    ensure_tmpdir(args.tmpdir)

    out_prefix = args.out
    rep_path = out_prefix + ".ncu-rep"
    csv_path = out_prefix + ".csv"

    kernel_regex = build_kernel_regex(ALL_KERNEL_SUBSTRS)
    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metrics_arg = ",".join(metrics_list)

    workload_cmd = args.cmd[1:]
    extra_ncu = shlex.split(args.extra_ncu) if args.extra_ncu.strip() else []

    ncu_cmd = [
        "ncu",
        "-f",
        "--target-processes", args.target_processes,
        "--kernel-name-base", "demangled",
        "--kernel-name", f"regex:{kernel_regex}",
        "--metrics", metrics_arg,
        "-o", out_prefix,
    ] + extra_ncu + ["--"] + workload_cmd

    print("\n[1/3] Running ncu...")
    run_checked(ncu_cmd)

    if not os.path.exists(rep_path):
        raise RuntimeError(f"NCU report not found: {rep_path}")

    print("\n[2/3] Exporting CSV...")
    export_csv(rep_path, csv_path, export_mode=args.export_mode)

    layout, fields = _detect_csv_layout(csv_path)
    print(f"[INFO] Detected CSV layout: {layout}. First columns: {fields[:15]}")

    print("\n[3/3] Parsing + computing MAX time per Kernel Name...")
    if layout == "wide":
        maxmap = parse_wide_csv_max(csv_path)
    elif layout == "long":
        maxmap = parse_long_csv_max(csv_path)
    else:
        raise RuntimeError(f"Unknown CSV layout. Columns: {fields}")

    print_max_summary(maxmap)

    print("\nSaved:")
    print("  -", rep_path)
    print("  -", csv_path)
    print("\nDone.")


if __name__ == "__main__":
    main()