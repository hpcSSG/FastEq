"""Audit a limited formal-sampling equivalence across frozen producer epochs."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from common import Evidence
from performance import PerformancePlan


class ProtocolCompatibility:
    CHECKS = (("protocol.py", "Timer", None), ("protocol.py", "Accuracy", None),
              ("cases.py", "References", None), ("cases.py", "FixtureStore", None),
              ("reference_store.py", "ReferenceStore", None),
              ("performance.py", "MeasurementIdentity", None),
              ("performance.py", "SamplingProtocol", None),
              ("performance.py", "PerformanceWorker", "pin_cpu"))

    @staticmethod
    def node(directory, filename, class_name, method=None):
        tree = ast.parse((directory / "producer" / filename).read_text())
        value = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        if method:
            value = next(node for node in value.body if isinstance(node, ast.FunctionDef) and node.name == method)
        return value

    @staticmethod
    def dump(node):
        return ast.dump(node, include_attributes=False)

    @classmethod
    def formal_tail(cls, directory):
        function = cls.node(directory, "performance.py", "PerformanceWorker", "point")
        block = next(node for node in function.body if isinstance(node, ast.Try))
        starts = [index for index, node in enumerate(block.body) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "pilot" for target in node.targets)
                  and isinstance(node.value, ast.Call)]
        if len(starts) != 1:
            raise ValueError("Cannot uniquely identify formal sampling tail")
        return ast.Module(body=block.body[starts[0]:], type_ignores=[])

    @classmethod
    def run(cls, args):
        left, right = PerformancePlan.load(args.left), PerformancePlan.load(args.right)
        checks = []
        for filename, class_name, method in cls.CHECKS:
            a = cls.node(args.left.parent, filename, class_name, method)
            b = cls.node(args.right.parent, filename, class_name, method)
            checks.append({"source": filename, "scope": class_name + ("." + method if method else ""),
                           "identical_ast": cls.dump(a) == cls.dump(b)})
        checks.append({"source": "performance.py", "scope": "PerformanceWorker.point formal group/order/emit/receipt tail",
                       "identical_ast": cls.dump(cls.formal_tail(args.left.parent)) == cls.dump(cls.formal_tail(args.right.parent))})
        report = {"left": {"path": str(args.left.resolve()), "plan_hash": left["plan_hash"], "protocol_hash": left["protocol_hash"],
                            "producer": left["producer"]},
                  "right": {"path": str(args.right.resolve()), "plan_hash": right["plan_hash"], "protocol_hash": right["protocol_hash"],
                             "producer": right["producer"]},
                  "sampling_rules_equal": left["protocol"] == right["protocol"], "checks": checks,
                  "formal_measurement_path_equivalent": left["protocol"] == right["protocol"] and all(row["identical_ast"] for row in checks),
                  "same_protocol_hash": left["protocol_hash"] == right["protocol_hash"],
                  "automatic_raw_merge_permitted": False, "automatic_cross_epoch_ranking_permitted": False,
                  "limitations": ["This proves only the listed code/rule equivalences; producer epochs and raw identities stay distinct",
                      "Factory loading, quality-helper refactoring and validation tensor lifetime are outside this equivalence claim",
                      "Cross-epoch comparison still requires actual fixture/reference/binary/GPU/CPU identity and matched Torch drift/noise checks",
                      "An observation of equivalent sampling code never authorizes resuming or re-signing data under a different fingerprint"],
                  "auditor_sha256": Evidence.sha256(Path(__file__))}
        if args.output.exists():
            raise FileExistsError(args.output)
        Evidence.write(args.output, report)
        print(json.dumps({key: report[key] for key in ("formal_measurement_path_equivalent", "same_protocol_hash", "automatic_raw_merge_permitted")}))
        return 0 if report["formal_measurement_path_equivalent"] else 1

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--left", type=Path, required=True)
        parser.add_argument("--right", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        return cls.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(ProtocolCompatibility.main())
