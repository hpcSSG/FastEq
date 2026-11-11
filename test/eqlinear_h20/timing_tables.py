"""Derive round tables and matched-workload sums from independently audited raw data."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from common import Evidence
from performance import PerformancePlan
from protocol import Statistics


class TimingTables:
    KEY = ("variant_id", "precision", "operation", "call_boundary")

    @staticmethod
    def boundary_eligible(audit, row):
        return not any(all(row.get(key) in values for key, values in entry["selectors"].items())
                       for entry in audit.get("boundary_adjudications", []))

    @staticmethod
    def ranking_eligible(audit):
        return (audit["full_factory_scope_complete"] and audit.get("comparison_qualified", True)
                and audit.get("latest_cycle_verified", True) and not audit["requires_complete_scan_repeat"]
                and not audit.get("stability_limited", False))

    @classmethod
    def point_ranking_eligible(cls, audit, row):
        if cls.ranking_eligible(audit):
            return True
        if (not audit["full_factory_scope_complete"] or not audit.get("comparison_qualified", True)
                or not audit.get("latest_cycle_verified", True) or audit["requires_complete_scan_repeat"]
                or not audit.get("noise_repeat_limit_reached", False)):
            return False
        fields = ("variant_id", "config", "B", "precision", "operation", "call_boundary")
        matches = [value for value in audit.get("across_rounds", [])
                   if all(value[key] == row[key] for key in fields)]
        return (len(matches) == 1 and matches[0]["rounds_complete"] and matches[0]["qualified"]
                and not matches[0]["unstable_spread"] and not matches[0]["unstable_cross_round"])

    @staticmethod
    def write_csv(path, rows):
        fields = sorted({key for row in rows for key in row})
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})

    @staticmethod
    def sum_ratio(baseline, candidate):
        if not baseline or len(baseline) != len(candidate):
            raise ValueError("Matched baseline and candidate points are required")
        if any(not math.isfinite(value) or value <= 0 for value in baseline + candidate):
            raise ValueError("Speedup requires finite positive completed wall latencies")
        return sum(baseline) / sum(candidate)

    @classmethod
    def aggregate(cls, audit, plan):
        grouped = defaultdict(list)
        for row in audit["summaries"]:
            grouped[tuple(row[key] for key in cls.KEY) + (row["round"],)].append(row)
        expected = defaultdict(set)
        for key in plan["selected_points"][audit["family"]]:
            config, batch, precision = key.split("/")
            expected[precision].add((config, int(batch)))
        sums = []
        for key, values in sorted(grouped.items()):
            variant, precision, operation, boundary, round_index = key
            for small in (False, True):
                wanted = {point for point in expected[precision] if not small or point[1] in (512, 1024)}
                chosen = [row for row in values if not small or row["B"] in (512, 1024)]
                if not wanted:
                    continue
                observed = [(row["config"], row["B"]) for row in chosen]
                complete = set(observed) == wanted and len(observed) == len(wanted) and all(row["wall_count"] == 20 for row in chosen)
                sums.append(dict(zip(cls.KEY + ("round",), key),
                    shape_scope=("small10" if small else "all25") if len(wanted) == (10 if small else 25) else "declared_scope",
                    small_batch_subset=small, expected_points=len(wanted), observed_points=len(chosen),
                    round_scope_complete=complete, all_three_rounds_audited=bool(audit["full_factory_scope_complete"]),
                    accuracy_qualified=complete and all(row["qualified"] for row in chosen),
                    ranking_eligible=complete and all(row["qualified"] and cls.point_ranking_eligible(audit, row) and cls.boundary_eligible(audit, row) for row in chosen),
                    sum_of_point_wall_medians_ms=sum(row["wall_median"] for row in chosen),
                    unstable_spread=any(row["unstable_spread"] for row in chosen),
                    warmup_stabilized=all(row["warmup_stabilized"] for row in chosen)))
        return sums

    @classmethod
    def paired_speedups(cls, audit):
        rows = audit["summaries"]
        anchors = {}
        for row in rows:
            if row["variant_id"] == "torch_ieee_full" and row["call_boundary"] == "public_warm":
                key = (row["config"], row["B"], row["precision"], row["operation"], row["round"])
                if key in anchors:
                    raise ValueError("Duplicate same-process Torch anchor")
                anchors[key] = row
        pairs = []
        for row in rows:
            if row["variant_id"].startswith("torch_ieee_") or row["call_boundary"] not in ("public_warm", "forward_allocating", "public_allocating", "adapted_public_e2e"):
                continue
            anchor = anchors.get((row["config"], row["B"], row["precision"], row["operation"], row["round"]))
            if anchor is None:
                continue
            qualified = row["qualified"] and anchor["qualified"]
            pairs.append({key: row[key] for key in ("variant_id", "config", "B", "precision", "operation", "call_boundary", "round")} |
                {"baseline": "torch_ieee_full/public_warm", "baseline_wall_median_ms": anchor["wall_median"],
                 "candidate_wall_median_ms": row["wall_median"], "accuracy_qualified": qualified,
                 "diagnostic_speed_ratio": cls.sum_ratio([anchor["wall_median"]], [row["wall_median"]]),
                 "boundary_comparable": cls.boundary_eligible(audit, row) and cls.boundary_eligible(audit, anchor),
                 "ranking_eligible": qualified and cls.point_ranking_eligible(audit, row) and cls.point_ranking_eligible(audit, anchor) and cls.boundary_eligible(audit, row) and cls.boundary_eligible(audit, anchor),
                 "other_points_stability_limited": audit.get("stability_limited", False),
                 "unstable_spread": row["unstable_spread"] or anchor["unstable_spread"]})
        return pairs

    @classmethod
    def paired_sums(cls, pairs, audit, plan):
        grouped = defaultdict(list)
        for row in pairs:
            grouped[tuple(row[key] for key in cls.KEY) + (row["round"],)].append(row)
        results = []
        for key, values in sorted(grouped.items()):
            wanted = {tuple(point.split("/")[:2]) for point in plan["selected_points"][audit["family"]]
                      if point.split("/")[2] == key[1]}
            for small in (False, True):
                selected = [row for row in values if not small or row["B"] in (512, 1024)]
                expected = {point for point in wanted if not small or int(point[1]) in (512, 1024)}
                if not selected:
                    continue
                observed = [(row["config"], str(row["B"])) for row in selected]
                complete = len(observed) == len(expected) and set(observed) == expected
                qualified = complete and all(row["accuracy_qualified"] for row in selected)
                results.append(dict(zip(cls.KEY + ("round",), key), small_batch_subset=small,
                    expected_points=len(expected), observed_points=len(selected), complete=complete,
                    accuracy_qualified=qualified, ranking_eligible=qualified and all(row["ranking_eligible"] for row in selected),
                    sum_baseline_ms=sum(row["baseline_wall_median_ms"] for row in selected),
                    sum_candidate_ms=sum(row["candidate_wall_median_ms"] for row in selected),
                    diagnostic_sum_speed_ratio=cls.sum_ratio([row["baseline_wall_median_ms"] for row in selected],
                                                           [row["candidate_wall_median_ms"] for row in selected])))
        return results

    @classmethod
    def run(cls, args):
        args.output.mkdir(parents=True, exist_ok=False)
        all_rounds, all_points, all_sums, all_pairs, all_pair_sums, sources = [], [], [], [], [], []
        for path in args.audit:
            audit = json.loads(path.read_text())
            plan = PerformancePlan.load(Path(audit["plan_path"]))
            if audit["plan_hash"] != plan["plan_hash"] or audit["protocol_hash"] != plan["protocol_hash"]:
                raise ValueError("Audit and frozen plan differ")
            attempted = [int(path.stem.split("_", 1)[0].removeprefix("cycle"))
                for path in (Path(plan["directory"]) / "phases").glob("cycle*_round*.json")
                if audit["family"] in json.loads(path.read_text()).get("families", [])]
            actual_latest = max(attempted, default=audit["cycle"])
            audit["latest_cycle_verified"] = actual_latest == audit["cycle"]
            boundary_path = Path(plan["directory"]) / "boundary_adjudications.json"
            if boundary_path.exists():
                audit["boundary_adjudications"] = json.loads(boundary_path.read_text())
                for entry in audit["boundary_adjudications"]:
                    for filename, digest in entry["evidence_files"].items():
                        if Evidence.sha256(Path(filename)) != digest:
                            raise ValueError("Changed boundary adjudication evidence")
            source = {"path": str(path.resolve()), "sha256": Evidence.sha256(path), "family": audit["family"],
                      "cycle": audit["cycle"], "full_factory_scope_complete": audit["full_factory_scope_complete"],
                      "requires_complete_scan_repeat": audit["requires_complete_scan_repeat"],
                      "latest_cycle_verified": audit["latest_cycle_verified"], "actual_latest_attempted_cycle": actual_latest}
            if boundary_path.exists():
                source["boundary_adjudication"] = {"path": str(boundary_path), "sha256": Evidence.sha256(boundary_path)}
            sources.append(source)
            pairs = cls.paired_speedups(audit)
            for destination, rows in ((all_rounds, audit["summaries"]), (all_points, audit["across_rounds"]),
                (all_sums, cls.aggregate(audit, plan)), (all_pairs, pairs), (all_pair_sums, cls.paired_sums(pairs, audit, plan))):
                destination.extend({**row, "family": audit["family"], "cycle": audit["cycle"],
                                    "audit_sha256": source["sha256"], "plan_hash": plan["plan_hash"]} for row in rows)
        artifacts = {}
        for name, rows in (("rounds", all_rounds), ("points", all_points), ("sums", all_sums),
                           ("paired_speedups", all_pairs), ("paired_sums", all_pair_sums)):
            path = args.output / (name + ".csv")
            cls.write_csv(path, rows)
            artifacts[name] = {"path": str(path), "sha256": Evidence.sha256(path), "rows": len(rows)}
        Evidence.write(args.output / "index.json", {"sources": sources, "artifacts": artifacts,
            "generator_sha256": Evidence.sha256(Path(__file__)),
            "rules": ["Completed wall is primary; every point contributes its untrimmed20-group round median",
                      "Aggregate ratio is sum(baseline point medians)/sum(candidate point medians), never mean speedup",
                      "Only complete public_warm/forward_allocating/public_allocating/adapted_public_e2e calls compare to same-process same-operation Torch full path; allocating API includes its own per-call constructor/preparation",
                      "Native layout/prepared/component boundaries are separate and have no public speedup claim",
                      "Different precisions/fixture hashes stay separate; no cross-input accuracy Pareto claim",
                      "Partial/failed/unstable scans remain diagnostic and cannot qualify for rankings"]})
        return 0

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--audit", type=Path, action="append", required=True)
        parser.add_argument("--output", type=Path, required=True)
        return cls.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(TimingTables.main())
