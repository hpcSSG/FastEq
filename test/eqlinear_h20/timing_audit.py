"""Independent raw timing integrity, round coverage and untrimmed stability tables."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re

from cases import CONFIGS, points
from common import Evidence
from performance import MeasurementIdentity, PerformancePlan, SamplingProtocol
from protocol import Journal, Statistics
from qualification import AccuracyContracts, RecordedAccuracy


class TimingStatistics:
    """Untrimmed round summaries and variation of all three round medians."""
    KEY = ("variant_id", "config", "B", "precision", "operation", "call_boundary")

    @classmethod
    def summarize(cls, rows):
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row[key] for key in cls.KEY + ("round",))].append(row)
        rounds = []
        for key, values in sorted(grouped.items()):
            result = dict(zip(cls.KEY + ("round",), key))
            result["qualified"] = all(row["accuracy_pass"] for row in values)
            for metric in ("wall", "event", "host"):
                result.update({metric + "_" + name: value for name, value in
                               Statistics.summarize([row[metric] for row in values]).items()})
            result["unstable_spread"] = result["wall_spread"] is None or result["wall_spread"] > .10
            result["warmup_stabilized"] = all(row.get("warmup_stabilized") is True for row in values)
            rounds.append(result)
        by_call = defaultdict(list)
        for row in rounds:
            by_call[tuple(row[key] for key in cls.KEY)].append(row)
        across = []
        for key, values in sorted(by_call.items()):
            result = dict(zip(cls.KEY, key))
            observed = sorted(row["round"] for row in values)
            complete = observed == [0, 1, 2] and all(row["wall_count"] == 20 for row in values)
            medians = {str(row["round"]): row["wall_median"] for row in values}
            stats = Statistics.summarize(list(medians.values()))
            result.update(rounds_complete=complete, round_medians_ms=medians,
                          wall_median_of_round_medians_ms=stats["median"],
                          cross_round_variation=stats["spread"] if complete else None,
                          unstable_cross_round=complete and (stats["spread"] is None or stats["spread"] > .05),
                          unstable_spread=any(row["unstable_spread"] for row in values),
                          warmup_stabilized=all(row["warmup_stabilized"] for row in values),
                          qualified=all(row["qualified"] for row in values))
            result["requires_complete_scan_repeat"] = result["unstable_spread"] or result["unstable_cross_round"]
            across.append(result)
        return rounds, across


class ScanRepetition:
    @staticmethod
    def status(complete, unstable, prior_complete_scan, external_reproduction_required=False):
        return {"requires_complete_scan_repeat": complete and (external_reproduction_required or unstable and not prior_complete_scan),
                "stability_limited": complete and unstable,
                "noise_repeat_limit_reached": complete and unstable and prior_complete_scan}


class TimingAudit:
    def __init__(self, args):
        self.args = args
        self.plan = PerformancePlan.load(args.plan.resolve())
        self.directory = Path(self.plan["directory"])
        self.checked = set()

    def check_file(self, path, digest):
        key = str(path), digest
        if key not in self.checked:
            if not Path(path).is_file() or Evidence.sha256(Path(path)) != digest:
                raise ValueError("Missing/changed evidence file: " + str(path))
            self.checked.add(key)

    @staticmethod
    def decode(row):
        value = dict(row)
        for key in ("cycle_id", "round", "B", "U", "V", "P", "I", "J", "group", "samples", "calls_per_group", "order"):
            value[key] = int(value[key])
        for key in ("wall", "event", "host"):
            value[key] = float(value[key])
        for key in ("per_call_wall", "per_call_event", "per_call_host", "i_dims"):
            value[key] = json.loads(value[key])
        if value["accuracy_pass"] not in ("True", "False"):
            raise ValueError("Invalid accuracy flag")
        value["accuracy_pass"] = value["accuracy_pass"] == "True"
        return value

    @staticmethod
    def quality_pass(value, precision):
        return RecordedAccuracy.passes(value, precision)

    @staticmethod
    def member_errors(quality, binding, operation, workload=None):
        declared = binding.get("quality_member_identities")
        components = quality.get("components", {})
        observed = any("members" in value for value in components.values())
        rotating = workload is not None and workload.get("kind") == "rotating_inputs"
        if declared is None:
            return ["missing ordered workload member declaration"] if observed or rotating else []
        kinds = set(MeasurementIdentity.result_kinds(operation))
        if not isinstance(declared, dict) or set(declared) != kinds or set(components) != kinds:
            return ["quality member component declaration differs"]
        errors = []
        for kind, identities in declared.items():
            if (not isinstance(identities, list) or not identities
                    or any(not isinstance(value, dict) for value in identities)
                    or [value.get("index") for value in identities] != list(range(len(identities)))):
                errors.append("ordered quality member identities are missing or duplicated")
                continue
            component = components[kind]
            members = component.get("members", {})
            if (component.get("member_count") != len(identities)
                    or set(members) != {str(index) for index in range(len(identities))}):
                errors.append("quality member set differs from binding")
                continue
            for index, identity in enumerate(identities):
                if any(not identity.get(key) for key in ("label", "input_hash", "reference_hash")):
                    errors.append("quality member identity is incomplete")
                if {key: members[str(index)].get(key) for key in ("label", "index", "input_hash", "reference_hash")} != identity:
                    errors.append("quality member identity differs from binding")
            if rotating:
                if [value["input_hash"] for value in identities] != workload.get("ordered_input_hashes"):
                    errors.append("quality inputs differ from frozen workload")
                if [value["reference_hash"] for value in identities] != [value.get(kind) for value in workload.get("ordered_reference_hashes", [])]:
                    errors.append("quality references differ from frozen workload")
        return errors

    def point(self, point):
        directory = self.directory / "points" / point.key.replace("/", "_")
        pilot_path = directory / "anchor_pilot.json"
        if not pilot_path.exists():
            return {"point": point.key, "complete": False, "reason": "missing locked pilot"}, []
        pilot = json.loads(pilot_path.read_text())
        all_rows, claims, errors = [], [], []
        expected_calls = set(self.plan["expected_calls"][self.args.family])
        invariants = None
        for path in sorted(directory.glob("*/raw.csv")):
            metadata_path = path.with_suffix(".identity.json")
            setup_path, receipt_path = path.parent / "setup.json", path.parent / "receipt.json"
            if not metadata_path.exists() or not setup_path.exists():
                errors.append(str(path) + ":missing metadata")
                continue
            metadata, setup = json.loads(metadata_path.read_text()), json.loads(setup_path.read_text())
            identity = metadata["identity"]
            if identity.get("family") != self.args.family:
                continue
            if "frame" in identity and (identity["frame"]["mode"] != "measure" or identity["frame"]["cycle"] != self.args.cycle):
                continue
            rows = [self.decode(row) for row in Journal.read(path)]
            if rows and any(row["cycle_id"] != self.args.cycle for row in rows):
                continue
            if identity != setup["identity"] or Evidence.fingerprint(identity) != metadata["identity_hash"]:
                errors.append(str(path) + ":mixed identity")
                continue
            if not receipt_path.exists():
                errors.append(str(path) + ":no terminal receipt")
                continue
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "COMPLETE" or receipt["identity_hash"] != metadata["identity_hash"]:
                errors.append(str(path) + ":invalid terminal receipt")
            self.check_file(path, receipt["raw_sha256"])
            source_expected = {key: identity[key] for key in pilot["identity"]}
            if source_expected != pilot["identity"] or identity["producer"] != self.plan["producer"]:
                errors.append(str(path) + ":pilot/source mismatch")
            frozen_identity = {key: identity[key] for key in ("GPU_UUID", "protocol_hash", "input_hash", "reference_hashes", "bindings", "affinity")}
            if "workload" in identity:
                frozen_identity["workload"] = identity["workload"]
                workload = identity["workload"]
                self.check_file(workload["manifest"], workload["manifest_sha256"])
            if invariants is not None and invariants != frozen_identity:
                errors.append(str(path) + ":round source/input/reference/GPU/CPU drift")
            invariants = frozen_identity
            calls = set(setup["quality"])
            if calls != expected_calls:
                errors.append(str(path) + ":call set drift")
            for call_key, quality in setup["quality"].items():
                variant, operation, _ = call_key.split("/")
                if variant in identity["bindings"]:
                    errors.extend(self.member_errors(quality, identity["bindings"][variant], operation, identity.get("workload")))
            warmup = setup.get("warmup", {})
            if set(warmup) != expected_calls:
                errors.append(str(path) + ":missing warmup records")
            reference = identity["reference_receipt"]
            self.check_file(reference["path"], reference["sha256"])
            self.check_file(reference["data_path"], reference["data_sha256"])
            for binding in identity["bindings"].values():
                for name, record in binding.get("binary_files", {}).items():
                    self.check_file(name, record["sha256"])
                for binary in binding.get("binaries", []):
                    self.check_file(binary.get("installed_path", binary.get("path")), binary["sha256"])
            for row in rows:
                MeasurementIdentity.check_row(row, identity)
                u, dims = CONFIGS[point.config]
                geometry = dict(config=point.config, B=point.batch, precision=point.precision,
                    U=u, V=u, i_dims=dims, P=len(dims), I=sum(dims),
                    J=sum(value for index, value in enumerate(dims) if index == 0 or value != dims[index-1]))
                if any(row[key] != value for key, value in geometry.items()):
                    errors.append("raw geometry differs from canonical point")
                if row["cache_mode"] != self.plan["protocol"]["cache_mode"] or row["weight_state"] != self.plan["protocol"]["weight_state"]:
                    errors.append("cache/weight state differs from declared protocol")
                call_key = "/".join((row["variant_id"], row["operation"], row["call_boundary"]))
                quality = setup["quality"].get(call_key)
                if quality is None:
                    errors.append("unknown call")
                    continue
                local_passed = self.quality_pass(quality, point.precision)
                upstream = AccuracyContracts.resolve(identity["bindings"][row["variant_id"]], row["variant_id"], point.key,
                    row["operation"], identity["input_hash"], identity["reference_hashes"])
                if upstream is not None and (quality.get("upstream_contract") != upstream or quality.get("local_passed") != local_passed):
                    errors.append("upstream full-operation qualification or local metric claim changed")
                qualified = local_passed and (upstream is None or upstream["passed"])
                expected_tier = ("fp64" if point.precision == "fp64" else "fp32") if row["variant_id"].startswith("torch_ieee_") else point.precision
                expected_tier = (self.plan.get("factory_scope") or {}).get("accuracy_tiers_by_variant", {}).get(row["variant_id"], {}).get(point.precision, expected_tier)
                components = quality.get("components", {MeasurementIdentity.result_kinds(row["operation"])[0]: quality})
                if set(components) != set(MeasurementIdentity.result_kinds(row["operation"])):
                    errors.append("gradient component set differs from operation")
                if any(value.get("accuracy_tier") not in ({"tf32", "tf32x3"} if expected_tier == "tf32" else {expected_tier}) for value in components.values()):
                    errors.append("numeric tier was changed")
                if quality.get("passed") != qualified or row["accuracy_pass"] != qualified:
                    errors.append("accuracy flag disagrees with independently applied numeric gate")
                if row["status"] != ("OK" if qualified else "RAW_DIAGNOSTIC"):
                    errors.append("diagnostic status mismatch")
                row["warmup_stabilized"] = warmup.get(call_key, {}).get("stabilized")
                if row["round"] != receipt["round"] or row["cycle_id"] != receipt["cycle"]:
                    errors.append("round/cycle claim mismatch")
                if row["value_unit"] != "ms" or row["sync_method"] != SamplingProtocol.VALUE["sync"]:
                    errors.append("unit/synchronization mismatch")
                if row["calls_per_group"] != pilot["anchors"][row["operation"] + "/" + row["call_boundary"]]["calls_per_group"]:
                    errors.append("group count changed after pilot")
                for field in ("per_call_wall", "per_call_event", "per_call_host"):
                    values = row[field]
                    if len(values) != row["calls_per_group"] or any(not math.isfinite(x) or x < 0 for x in values):
                        errors.append("missing/nonfinite per-call samples")
                if abs(sum(row["per_call_event"]) / row["calls_per_group"] - row["event"]) > 1e-9:
                    errors.append("event mean not derived from raw calls")
                if abs(sum(row["per_call_host"]) / row["calls_per_group"] - row["host"]) > 1e-9:
                    errors.append("host mean not derived from raw calls")
                if row["wall"] + 1e-9 < sum(row["per_call_wall"]) / row["calls_per_group"]:
                    errors.append("completed group wall excludes call windows")
            ordered_groups = defaultdict(list)
            for row in rows:
                ordered_groups[(row["operation"], row["call_boundary"], row["group"], row["round"])].append(row)
            for (operation, boundary, group, round_index), values in ordered_groups.items():
                candidates = [key.split("/")[0] for key in expected_calls if key.split("/")[1:] == [operation, boundary]]
                order = sorted(candidates, reverse=bool((group + round_index) % 2))
                if [row["variant_id"] for row in values] != order or [row["order"] for row in values] != list(range(len(order))):
                    errors.append("balanced candidate order differs from protocol")
            all_rows.extend(rows)
            claims.extend(("/".join((row["variant_id"], row["operation"], row["call_boundary"])), row["round"], row["group"]) for row in rows)
        wanted = {(key, round_index, group) for key in expected_calls or () for round_index in range(3) for group in range(20)}
        found = set(claims)
        complete = bool(expected_calls) and len(claims) == len(found) and found == wanted and not errors
        return {"point": point.key, "complete": complete, "expected_groups": len(wanted), "actual_groups": len(claims),
                "missing_groups": len(wanted-found), "extra_groups": len(found-wanted), "duplicate_groups": len(claims)-len(found),
                "errors": errors, "identity": invariants}, all_rows

    def run(self):
        expected_points = set(self.plan.get("selected_points", {}).get(self.args.family, [point.key for point in points()]))
        selected = [point for point in points() if point.key in expected_points and (self.args.point is None or point.key == self.args.point)]
        if not selected:
            raise ValueError("No canonical points selected")
        reports, rows = [], []
        for point in selected:
            report, values = self.point(point)
            reports.append(report)
            rows.extend(values)
        summaries, across_rounds = TimingStatistics.summarize(rows)
        phase_errors, phase_cycles = [], []
        for path in (self.directory / "phases").glob("cycle*_round*.json"):
            match = re.fullmatch(r"cycle(\d+)_round\d+", path.stem)
            value = json.loads(path.read_text())
            if match and self.args.family in value.get("families", []):
                phase_cycles.append(int(match.group(1)))
        latest = max(phase_cycles, default=self.args.cycle)
        full_scope = len(selected) == len(expected_points)
        if full_scope:
            previous_end = None
            for round_index in range(3):
                path = self.directory / "phases" / f"cycle{self.args.cycle}_round{round_index}.json"
                if not path.exists():
                    phase_errors.append(f"whole-matrix round{round_index} ledger missing")
                    continue
                phase = json.loads(path.read_text())
                if phase.get("status") != "COMPLETE" or phase.get("plan_hash") != self.plan["plan_hash"]:
                    phase_errors.append(f"whole-matrix round{round_index} not complete or changed plan")
                if phase.get("driver_sha256") != self.plan["producer"]["performance_driver.py"]:
                    phase_errors.append("whole-matrix driver source differs from frozen producer")
                if set(phase.get("families", [])) != set(self.plan["include_families"]) or len(phase.get("families", [])) != len(self.plan["include_families"]):
                    phase_errors.append("whole-matrix family cohort changed")
                if previous_end and phase.get("start_utc", "") < previous_end:
                    phase_errors.append("next round started before previous full scan ended")
                previous_end = phase.get("end_utc", "")
                chunks = [task for task in phase.get("tasks", []) if task["family"] == self.args.family]
                wanted_configs = {key.split("/")[0] for key in expected_points}
                if phase.get("task_schema") == 2:
                    observed_points = [key for task in chunks for key in task.get("point_keys", [])]
                    if set(observed_points) != expected_points or len(observed_points) != len(expected_points):
                        phase_errors.append("whole-matrix point chunks overlap or omit declared points")
                    planned = [task for task in phase.get("planned_tasks", []) if task["family"] == self.args.family]
                    fields = ("config", "family", "round", "batch", "precision", "point_keys")
                    if [{key: task.get(key) for key in fields} for task in chunks] != [{key: task.get(key) for key in fields} for task in planned]:
                        phase_errors.append("executed point chunks differ from frozen phase schedule")
                elif {task["config"] for task in chunks} != wanted_configs or len(chunks) != len(wanted_configs):
                    phase_errors.append("whole-matrix round config set missing/duplicate")
                if any(task.get("terminal") != "PASS" or not task.get("complete_scope") for task in chunks):
                    phase_errors.append("whole-matrix round contains nonterminal/partial chunk")
            if latest != self.args.cycle:
                phase_errors.append("newer attempted cycle exists; older data is historical only")
        scope_complete = full_scope and all(report["complete"] for report in reports) and not phase_errors
        complete = len(selected) == 75 and scope_complete
        adjudication_path = self.directory / "comparison_adjudications.json"
        adjudications = []
        if adjudication_path.exists():
            for value in json.loads(adjudication_path.read_text()):
                if value["cycle"] == self.args.cycle and self.args.family in value["families"]:
                    for filename, digest in value.get("evidence_files", {}).items():
                        self.check_file(filename, digest)
                    adjudications.append(value)
        reproduction_required = any(value.get("complete_scan_repeat_required") for value in adjudications)
        prior_complete_scan = self.args.cycle > 1
        if prior_complete_scan:
            for round_index in range(3):
                path = self.directory / "phases" / f"cycle{self.args.cycle-1}_round{round_index}.json"
                value = json.loads(path.read_text()) if path.exists() else {}
                if (value.get("status") != "COMPLETE" or value.get("plan_hash") != self.plan["plan_hash"]
                        or self.args.family not in value.get("families", [])):
                    prior_complete_scan = False
        repetition = ScanRepetition.status(scope_complete, any(row["requires_complete_scan_repeat"] for row in across_rounds),
                                          prior_complete_scan, reproduction_required)
        report = {"full75x3_complete": complete,
                  "selected_scope_complete": all(row["complete"] for row in reports),
                  "scope_point_count": len(selected), "family": self.args.family, "cycle": self.args.cycle,
                  "plan_path": str(self.args.plan.resolve()), "plan_hash": self.plan["plan_hash"],
                  "protocol_hash": self.plan["protocol_hash"],
                  "group_rows": len(rows), "points": reports, "summaries": summaries,
                  "unstable_cells": sum(row["unstable_spread"] for row in summaries), "all_samples_preserved": True,
                  "across_rounds": across_rounds,
                  "cross_round_unstable_cells": sum(row["unstable_cross_round"] for row in across_rounds),
                  "warmup_unstable_cells": sum(not row["warmup_stabilized"] for row in summaries),
                  "comparison_qualified": scope_complete and not adjudications,
                  "comparison_adjudications": adjudications,
                  "comparison_adjudication_source": {"path": str(adjudication_path), "sha256": Evidence.sha256(adjudication_path)} if adjudication_path.exists() else None,
                  **repetition, "prior_complete_scan_available": prior_complete_scan,
                  "stability_rule": {"spread": "(max-min)/median of all20 groups; >10% flags instability",
                                     "cross_round": "(max-min)/median of all3 round medians; >5% flags instability",
                                     "repeat": "repeat complete affected family scan once; keep both cycles even if still unstable"},
                  "full_factory_scope_complete": scope_complete, "phase_errors": phase_errors, "latest_attempted_cycle": latest,
                  "auditor_sha256": Evidence.sha256(Path(__file__))}
        Evidence.write(self.args.output, report)
        print(json.dumps({key: report[key] for key in ("full75x3_complete", "selected_scope_complete", "scope_point_count", "group_rows", "unstable_cells")}))
        return 0 if (report["selected_scope_complete"] if self.args.point is not None else report["full_factory_scope_complete"]) else 1

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--family", required=True)
        parser.add_argument("--cycle", type=int, required=True)
        parser.add_argument("--point", help="Explicitly audit a subset; can never claim full75 completeness")
        parser.add_argument("--output", type=Path, required=True)
        return cls(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(TimingAudit.main())
