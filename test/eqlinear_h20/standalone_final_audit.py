"""Aggregate the independent public timing auditor over a complete frozen cohort."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import sys


class FinalCohortAudit:
    @classmethod
    def run(cls, args):
        from common import Evidence
        from producer_closure import ProducerClosure

        root = args.plan.resolve().parent
        output = args.output.resolve()
        if output.exists() or not output.is_relative_to(root):
            raise ValueError("Final cohort audit needs a fresh output inside the frozen plan")
        audit_root = output.with_suffix(".evidence")
        producer = audit_root / "producer"
        producer.mkdir(parents=True, exist_ok=False)
        files = ProducerClosure(Path(__file__).parent, producer).freeze(("standalone_final_audit.py", "timing_audit.py"))
        sys.path.insert(0, str(producer))
        audit_module = importlib.import_module("timing_audit")
        from performance import PerformancePlan

        plan = PerformancePlan.load(args.plan.resolve())
        families = plan["include_families"]
        expected = {(family, key, round_index) for family in families
                    for key in plan["selected_points"][family] for round_index in range(3)}
        observed, affinity_errors, selected_setups = [], [], []
        for path in sorted(root.glob("points/*/*/setup.json")):
            value = json.loads(path.read_text())
            identity, frame = value["identity"], value["identity"]["frame"]
            if identity.get("plan_hash") != plan["plan_hash"]:
                raise ValueError("An actual setup belongs to a different plan")
            if identity.get("affinity") != {"allowed": [args.cpu], "selected": args.cpu}:
                affinity_errors.append(str(path))
            if frame["mode"] != "measure" or frame["cycle"] != args.cycle:
                continue
            point = identity["point"]
            key = "/".join(map(str, (point["config"], point["batch"], point["precision"])))
            observed.append((identity["family"], key, frame["round"]))
            selected_setups.append({"path": str(path), "sha256": Evidence.sha256(path)})
        counts = Counter(observed)
        missing = sorted(expected - counts.keys())
        extra = sorted(counts.keys() - expected)
        duplicate = sorted(key for key, count in counts.items() if count != 1)
        reports, complete = {}, not (affinity_errors or missing or extra or duplicate)
        for family in families:
            target = audit_root / (family + ".json")
            audit_args = argparse.Namespace(plan=args.plan, family=family, cycle=args.cycle, point=None, output=target)
            code = audit_module.TimingAudit(audit_args).run()
            value = json.loads(target.read_text())
            expected_rows = len(plan["selected_points"][family]) * len(plan["expected_calls"][family]) * 3 * 20
            valid = code == 0 and value["full_factory_scope_complete"] is True and value["group_rows"] == expected_rows
            complete = complete and valid
            reports[family] = {"path": str(target), "sha256": Evidence.sha256(target), "complete": valid,
                "expected_rows": expected_rows, "observed_rows": value["group_rows"],
                "requires_complete_scan_repeat": value["requires_complete_scan_repeat"],
                "noise_repeat_limit_reached": value["noise_repeat_limit_reached"],
                "unstable_cells": value["unstable_cells"], "cross_round_unstable_cells": value["cross_round_unstable_cells"]}
        if files != {name: Evidence.sha256(producer / name) for name in files}:
            raise ValueError("Independent auditor changed while reading the final cohort")
        repeat = complete and any(value["requires_complete_scan_repeat"] for value in reports.values())
        if args.cycle == 2 and repeat:
            raise ValueError("The allowed single complete noise repeat cannot schedule a third cycle")
        Evidence.write(output, {"classification": "INDEPENDENT_COMPLETE_COHORT_TIMING_AUDIT", "complete": complete,
            "plan": str(args.plan.resolve()), "plan_hash": plan["plan_hash"], "cycle": args.cycle,
            "families": len(families), "successful_families": sum(row["complete"] for row in reports.values()),
            "expected_setup_count": len(expected), "observed_setup_count": len(observed),
            "required_affinity": {"allowed": [args.cpu], "selected": args.cpu}, "affinity_errors": affinity_errors,
            "missing_setups": missing, "unexpected_setups": extra, "duplicate_setups": duplicate,
            "expected_raw_rows": sum(row["expected_rows"] for row in reports.values()),
            "observed_raw_rows": sum(row["observed_rows"] for row in reports.values()),
            "requires_complete_cohort_repeat": repeat,
            "noise_repeat_limit_reached": complete and any(row["noise_repeat_limit_reached"] for row in reports.values()),
            "reports": reports, "setup_evidence": selected_setups, "independent_auditor_producer": files,
            "source_root": str(producer), "original_raw_modified": False,
            "qualification_and_statistics": "Unmodified public TimingAudit/RecordedAccuracy/AccuracyContracts; no new numeric or timing gate"})
        return 0 if complete else 1


class AuditCLI:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--cycle", type=int, choices=(1, 2), required=True)
        parser.add_argument("--cpu", type=int, required=True)
        parser.add_argument("--output", type=Path, required=True)
        return FinalCohortAudit.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(AuditCLI.main())
