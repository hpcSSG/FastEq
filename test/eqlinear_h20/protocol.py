"""One numerical gate, timer, journal and independent coverage audit for all adapters."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import time

from common import Evidence


class Accuracy:
    @staticmethod
    def evaluate(output, reference, torch_error, precision, repeat_equal, different_buffer):
        import torch

        value, gold = output.detach().double().reshape(-1), reference.detach().double().reshape(-1)
        if value.shape != gold.shape:
            raise ValueError("Output shape disagrees with the reference")
        finite = bool(torch.isfinite(value).all().item())
        reference_finite = bool(torch.isfinite(gold).all().item())
        baseline_valid = math.isfinite(torch_error) and torch_error >= 0
        error = value - gold
        norm = gold.norm().item()
        error_norm = error.norm().item()
        exact_equal = torch.equal(value, gold)
        zero_reference = bool(torch.count_nonzero(gold).item() == 0)
        relative = error_norm / norm if norm else (0.0 if exact_equal else math.inf)
        threshold = 1e-12 if precision == "fp64" else (2e-5 if precision in ("tf32", "tf32x3") else
                                                     1.05 * torch_error if baseline_valid else None)
        if threshold is not None and not math.isfinite(threshold):
            threshold = None
        # Zero gold is governed by exact zero, never by Inf <= Inf or an epsilon.
        numerical = exact_equal if norm == 0 else math.isfinite(relative) and threshold is not None and relative <= threshold
        passed = finite and reference_finite and numerical and threshold is not None and repeat_equal and different_buffer
        maximum, mean = error.abs().max().item(), error.abs().mean().item()
        reason = "PASS" if passed else ("NONFINITE_REFERENCE" if not reference_finite else
                 "INVALID_TORCH_BASELINE" if threshold is None else
                 "ZERO_REFERENCE_NONZERO_OUTPUT" if zero_reference and not exact_equal else
                 "REFERENCE_NORM_UNDERFLOW" if norm == 0 and not exact_equal else
                 "NUMERICAL_OR_REPEAT_OR_BUFFER_GATE_FAILED")
        return {"rel_fro": relative if math.isfinite(relative) else None,
                "zero_reference": zero_reference, "max_abs": maximum if math.isfinite(maximum) else None,
                "mean_abs": mean if math.isfinite(mean) else None, "finite": finite,
                "reference_finite": reference_finite, "torch_baseline_valid": baseline_valid,
                "repeat_equal": bool(repeat_equal), "different_buffer": bool(different_buffer),
                "torch_rel_fro": torch_error if math.isfinite(torch_error) else None, "threshold": threshold, "accuracy_tier": precision,
                "passed": passed, "reason": reason}


class Statistics:
    @staticmethod
    def summarize(values):
        if not values or any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("Statistics require nonempty finite nonnegative timings")
        median = statistics.median(values)
        return {"count": len(values), "median": median, "min": min(values), "max": max(values),
                "spread": (max(values) - min(values)) / median if median else (0.0 if max(values) == 0 else None)}


class Timer:
    """Every call drains the selected device; events materialized before measurement."""

    def __init__(self, calls_per_group):
        import torch

        if calls_per_group < 1:
            raise ValueError("calls_per_group must be positive and pilot-locked")
        self.events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                       for _ in range(calls_per_group)]
        for pair in self.events:
            for event in pair:
                event.record()
        self.events[-1][-1].synchronize()

    def sample(self, function):
        import torch

        torch.cuda.synchronize()
        wall, host = [], []
        group_start = time.perf_counter_ns()
        for start, end in self.events:
            call_start = time.perf_counter_ns()
            start.record()
            host_start = time.perf_counter_ns()
            output = function()
            host_end = time.perf_counter_ns()
            end.record()
            torch.cuda.synchronize()
            wall.append((time.perf_counter_ns() - call_start) / 1e6)
            host.append((host_end - host_start) / 1e6)
            del output
        elapsed = (time.perf_counter_ns() - group_start) / 1e6
        event = [start.elapsed_time(end) for start, end in self.events]
        return {"wall": elapsed / len(wall), "event": statistics.mean(event), "host": statistics.mean(host),
                "per_call_wall": wall, "per_call_event": event, "per_call_host": host,
                "calls_per_group": len(wall), "value_unit": "ms",
                "sync_method": "device_sync_before_group;device_sync_after_every_call"}

    @staticmethod
    def warmup(function, limit=30, window=5, relative_tolerance=0.05):
        import torch

        times = []
        for _ in range(limit):
            start = time.perf_counter_ns()
            output = function()
            torch.cuda.synchronize()
            times.append((time.perf_counter_ns() - start) / 1e6)
            del output
            if len(times) >= 2 * window:
                old, new = statistics.median(times[-2 * window:-window]), statistics.median(times[-window:])
                if old > 0 and abs(new / old - 1) <= relative_tolerance:
                    return {"calls": len(times), "wall_ms": times, "stabilized": True}
        return {"calls": len(times), "wall_ms": times, "stabilized": False}


class Journal:
    FIELDS = ("run_id", "attempt_id", "cycle_id", "round", "variant_id", "source_hash", "binary_hash", "GPU_UUID",
              "protocol_hash", "config", "B", "U", "V", "i_dims", "P", "I", "J", "precision", "input_hash",
              "reference_hash", "operation", "call_boundary", "cache_mode", "weight_state", "samples", "group",
              "calls_per_group", "order", "wall", "event", "host", "value_unit", "status", "accuracy_pass",
              "per_call_wall", "per_call_event", "per_call_host", "sync_method")

    def __init__(self, path, identity, resume=False):
        self.path = Path(path)
        metadata = self.path.with_suffix(".identity.json")
        self.identity_hash = Evidence.fingerprint(identity)
        if self.path.exists() or metadata.exists():
            if not resume or not self.path.exists() or not metadata.exists():
                raise ValueError("Existing or partial journal requires intact explicit resume")
            if json.loads(metadata.read_text()) != {"identity": identity, "identity_hash": self.identity_hash}:
                raise ValueError("Mixed source/binary/input/reference/GPU/protocol fingerprint")
            self.read(self.path)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            Evidence.write(metadata, {"identity": identity, "identity_hash": self.identity_hash})
            with self.path.open("x", newline="") as stream:
                csv.DictWriter(stream, fieldnames=self.FIELDS).writeheader()
        self.stream = self.path.open("a", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=self.FIELDS)

    def emit(self, row):
        if set(row) != set(self.FIELDS):
            raise ValueError(f"Raw schema mismatch: missing={set(self.FIELDS)-set(row)}, extra={set(row)-set(self.FIELDS)}")
        self.writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                              for key, value in row.items()})
        self.stream.flush()

    def close(self):
        self.stream.close()

    @classmethod
    def read(cls, path):
        with Path(path).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != cls.FIELDS:
                raise ValueError("Raw header mismatch")
            rows = list(reader)
        if any(set(row) != set(cls.FIELDS) or any(v is None or v == "" for v in row.values()) for row in rows):
            raise ValueError("Truncated/partial raw record")
        return rows


class CoverageAuditor:
    """A cycle is complete only by its own expected matrix; partial cycles never mix."""
    KEYS = ("variant_id", "config", "B", "precision", "operation", "call_boundary", "cache_mode")

    @classmethod
    def audit(cls, rows, expected, rounds=3, groups=20):
        expected = {tuple(str(case[key]) for key in cls.KEYS) for case in expected}
        cycles = sorted({int(row["cycle_id"]) for row in rows})
        reports = []
        for cycle in cycles:
            found, duplicate, invalid = set(), [], []
            for row in rows:
                if int(row["cycle_id"]) != cycle:
                    continue
                case = tuple(str(row[key]) for key in cls.KEYS)
                key = (case, int(row["round"]), int(row["group"]))
                if key in found:
                    duplicate.append(key)
                found.add(key)
                if case not in expected or row["status"] != "OK" or row["accuracy_pass"] not in (True, "True", "true"):
                    invalid.append(key)
                calls = int(row["calls_per_group"])
                for field in ("per_call_wall", "per_call_event", "per_call_host"):
                    values = json.loads(row[field]) if isinstance(row[field], str) else row[field]
                    if len(values) != calls or any(not math.isfinite(value) or value < 0 for value in values):
                        invalid.append(key)
            wanted = {(case, round_index, group_index) for case in expected for round_index in range(rounds) for group_index in range(groups)}
            reports.append({"cycle_id": cycle, "complete": found == wanted and not duplicate and not invalid,
                            "missing_groups": len(wanted - found), "extra_groups": len(found - wanted),
                            "duplicate_groups": len(duplicate), "invalid_groups": len(invalid)})
        # The latest attempted cycle must stand on its own; older complete cycles stay historical.
        return {"complete": bool(reports and reports[-1]["complete"]), "cycles": reports,
                "selected_cycle": reports[-1]["cycle_id"] if reports and reports[-1]["complete"] else None}


class TraceSummary:
    @staticmethod
    def kernels(trace):
        return [event for event in trace.get("traceEvents", []) if event.get("ph") == "X" and event.get("cat") == "kernel"]
