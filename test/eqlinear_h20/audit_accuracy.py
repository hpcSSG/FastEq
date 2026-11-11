"""Independently audit the current FastEq 75-point, five-path accuracy cycle."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from cases import points
from common import Evidence
from session import Session


class AccuracyAudit:
    REQUIRED = {("forward", "public_warm"), ("forward", "run_with_activation_prep"),
                ("forward_autograd", "public_warm"), ("grad_x", "public_warm"), ("grad_x", "run_with_activation_prep")}

    def __init__(self, args):
        self.args, self.session = args, Session(args.session)

    def expected_identity(self):
        manifest = json.loads((self.session.root / "manifests/current_build.json").read_text())
        return {"source_hash": manifest["source"]["source_hash"],
                "binary_hash": Evidence.fingerprint({row["installed_path"]: row["sha256"] for row in manifest["binaries"]})}

    def attempt_name(self, point, cycle):
        return f"current_accuracy_{point.config}_{cycle}_{point.batch}_{point.precision}"

    def producer_unchanged(self, result):
        return result.get("producer_changed_during_run") is False

    def selected_points(self):
        return points()

    def run(self):
        root, cycle = self.session.root, self.args.cycle
        expected_identity = self.expected_identity()
        selected = self.selected_points()
        failures, missing, counts, producers = [], [], Counter(), set()
        for point in selected:
            attempt = self.attempt_name(point, cycle)
            result_path = root / "accuracy" / attempt / "result.json"
            if not result_path.exists():
                missing.append(point.key)
                continue
            result = json.loads(result_path.read_text())
            rows = result.get("boundaries", [])
            reasons = []
            if result.get("status") != "PASS" or not self.producer_unchanged(result):
                reasons.append("nonpassing, nonterminal or producer drift")
            if result.get("point") != point.columns():
                reasons.append("case or seed identity")
            identity = result.get("identity", {})
            if any(identity.get(key) != value for key, value in expected_identity.items()):
                reasons.append("source/binary identity")
            producer = result.get("producer", {})
            producers.add(Evidence.fingerprint(producer))
            for name, digest in producer.items():
                path = result_path.parent / "producer" / name
                if not path.is_file() or Evidence.sha256(path) != digest:
                    reasons.append("producer snapshot changed or missing: " + name)
            fixture_path = root / "fixtures" / (point.key.replace("/", "_") + ".pt")
            fixture_metadata = fixture_path.with_suffix(".json")
            if not fixture_path.is_file() or not fixture_metadata.is_file():
                reasons.append("missing canonical tensor bytes")
            else:
                fixture = json.loads(fixture_metadata.read_text())
                if fixture.get("point") != point.columns() or fixture.get("file_sha256") != Evidence.sha256(fixture_path):
                    reasons.append("fixture file identity")
                if any(result.get("tensor_hashes", {}).get(name + "_sha256") != digest
                       for name, digest in fixture["tensor_hashes"].items()):
                    reasons.append("result did not bind canonical tensor hashes")
                if result.get("layouts") != fixture.get("layouts"):
                    reasons.append("fixture layout identity")
            keys = [(row.get("operation"), row.get("call_boundary")) for row in rows]
            if set(keys) != self.REQUIRED or len(keys) != len(self.REQUIRED):
                reasons.append("missing or duplicate operation/boundary")
            for row in rows:
                limit = 1e-12 if point.precision == "fp64" else (2e-5 if point.precision == "tf32" else 1.05 * row["torch_rel_fro"])
                if row.get("threshold") != limit or row.get("rel_fro") is None or row["rel_fro"] > limit:
                    reasons.append("numerical threshold independently failed")
                if not all(row.get(key) is True for key in ("finite", "repeat_equal", "different_buffer", "passed")):
                    reasons.append("finite/repeat/buffer/gate")
                if not result.get("reference_hashes", {}).get("forward" if row["operation"] == "forward_autograd" else row["operation"]):
                    reasons.append("reference hash missing")
                frozen = row.get("frozen_p14", {})
                counts["frozen_" + frozen.get("status", "MISSING")] += 1
            if reasons:
                failures.append({"point": point.key, "reasons": sorted(set(reasons))})
            else:
                counts["points_verified"] += 1
                counts["paths_verified"] += len(rows)
        if len(producers) > 1:
            failures.append({"reason": "different producer epochs in one claimed cycle", "count": len(producers)})
        expected_points, expected_paths = len(selected), len(selected) * len(self.REQUIRED)
        complete = not missing and not failures and counts["points_verified"] == expected_points and counts["paths_verified"] == expected_paths
        report = {"classification": "UNIFIED_BASE_ACCURACY_ONLY_NOT_PERFORMANCE", "cycle": cycle,
                  "base_accuracy_complete": complete, "expected_points": expected_points, "expected_paths": expected_paths,
                  "counts": dict(counts), "missing_points": missing, "failures": failures,
                  "frozen_h100_p14_additional_gate_verified": complete and counts["frozen_PASS"] == 125,
                  "note": "Unmatched H100 input fingerprints remain unverified; no historical values are re-signed",
                  "auditor_sha256": Evidence.sha256(Path(__file__))}
        Evidence.write(self.args.output, report)
        print(json.dumps({"complete": complete, "counts": dict(counts), "missing": len(missing), "failures": len(failures)}))
        return 0 if complete else 1

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--cycle", required=True)
        parser.add_argument("--output", type=Path, required=True)
        return cls(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(AccuracyAudit.main())
