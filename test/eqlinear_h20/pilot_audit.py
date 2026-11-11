"""Read back locked pilots independently before any formal matrix scan."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from cases import points
from common import Evidence
from performance import PilotAnchors, SamplingProtocol
from timing_audit import TimingAudit


class PilotAudit(TimingAudit):
    def point(self, point):
        path = self.directory / "points" / point.key.replace("/", "_") / "anchor_pilot.json"
        if not path.exists():
            return {"point": point.key, "valid": False, "errors": ["missing pilot"]}
        pilot = json.loads(path.read_text())
        source = Path(pilot["source_attempt"]) / "setup.json"
        setup = json.loads(source.read_text())
        identity = setup["identity"]
        errors = []
        if pilot["plan_hash"] != self.plan["plan_hash"] or identity["plan_hash"] != self.plan["plan_hash"]:
            errors.append("plan changed")
        if identity["producer"] != self.plan["producer"] or pilot["identity"] != {key: identity[key] for key in pilot["identity"]}:
            errors.append("producer/input/reference/GPU/CPU identity changed")
        if identity["point"] != point.columns():
            errors.append("point geometry changed")
        expected_math = {"precision": "highest", "allow_tf32": False, "cudnn_allow_tf32": False, "NVIDIA_TF32_OVERRIDE": "0"}
        if setup["math_actual"] != expected_math:
            errors.append("Torch math settings changed")
        reference = identity["reference_receipt"]
        self.check_file(reference["path"], reference["sha256"])
        self.check_file(reference["data_path"], reference["data_sha256"])
        declared = PilotAnchors.declared(self.plan, point.key)
        if not declared <= set(pilot["anchors"]):
            errors.append("missing operation/boundary group lock")
        quality = {**setup.get("quality", {}), **setup.get("pilot_quality", {})}
        for key, anchor in pilot["anchors"].items():
            samples = anchor["samples"]
            if len(samples) != SamplingProtocol.VALUE["pilot_groups"]:
                errors.append(key + ":pilot sample count")
            values = [sample["wall"] for sample in samples]
            if any(not math.isfinite(value) or value <= 0 for value in values):
                errors.append(key + ":invalid wall timing")
                continue
            median = statistics.median(values)
            expected_count = 1 if key.split("/")[1] in SamplingProtocol.VALUE["one_call_components"] else SamplingProtocol.count(median)
            if median != anchor["median_wall_ms"] or expected_count != anchor["calls_per_group"]:
                errors.append(key + ":median/group count not derived from raw pilot")
            for sample in samples:
                if sample["calls_per_group"] != 1 or sample["value_unit"] != "ms" or sample["sync_method"] != SamplingProtocol.VALUE["sync"]:
                    errors.append(key + ":pilot sampling protocol changed")
                for metric in ("wall", "event", "host"):
                    raw = sample["per_call_" + metric]
                    if len(raw) != 1 or any(not math.isfinite(value) or value < 0 for value in raw):
                        errors.append(key + ":missing/nonfinite raw pilot call")
                    elif (sample[metric] + 1e-9 < raw[0] if metric == "wall" else abs(sample[metric] - raw[0]) > 1e-9):
                        errors.append(key + ":pilot metric not derived from raw call")
            operation = key.split("/")[0]
            boundary = "prepared_gemm" if anchor["variant"] == "torch_ieee_mm" else "public_warm"
            source_key = "/".join((anchor["variant"], operation, boundary))
            value = quality.get(source_key)
            if value is None or not self.quality_pass(value, "fp64" if point.precision == "fp64" else "fp32") or not anchor["accuracy_pass"]:
                errors.append(key + ":Torch anchor failed independent numeric gate")
        return {"point": point.key, "valid": not errors, "errors": errors, "path": str(path), "sha256": Evidence.sha256(path),
                "source_setup": str(source), "source_setup_sha256": Evidence.sha256(source),
                "input_hash": identity["input_hash"], "reference_hashes": identity["reference_hashes"],
                "GPU_UUID": identity["GPU_UUID"], "affinity": identity["affinity"],
                "groups": {key: anchor["calls_per_group"] for key, anchor in pilot["anchors"].items()},
                "quality_cells": len(setup["quality"]), "quality_passed": sum(value["passed"] for value in setup["quality"].values()),
                "warmup_capped": [key for key, value in setup["warmup"].items() if not value["stabilized"]]}

    def run(self):
        wanted = {key for values in self.plan["selected_points"].values() for key in values}
        reports = [self.point(point) for point in points() if point.key in wanted]
        phase_path = self.directory / "phases/pilot.json"
        phase = json.loads(phase_path.read_text()) if phase_path.exists() else {}
        phase_valid = phase.get("status") == "COMPLETE" and phase.get("plan_hash") == self.plan["plan_hash"]
        report = {"valid": len(reports) == len(wanted) and all(value["valid"] for value in reports) and phase_valid,
            "point_count": len(reports), "expected_point_count": len(wanted), "phase_complete": phase_valid,
            "plan_hash": self.plan["plan_hash"], "protocol_hash": self.plan["protocol_hash"], "points": reports,
            "group_count_distribution": dict(Counter(str(count) for value in reports for count in value.get("groups", {}).values())),
            "auditor_sha256": Evidence.sha256(Path(__file__))}
        Evidence.write(self.args.output, report)
        print(json.dumps({key: report[key] for key in ("valid", "point_count", "phase_complete", "group_count_distribution")}))
        return 0 if report["valid"] else 1

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        return cls(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(PilotAudit.main())
