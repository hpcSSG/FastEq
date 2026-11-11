"""Frozen complete handwritten scope using the common fixture, probe and gate."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import subprocess

from cases import CONFIGS, points
from common import Evidence
from handwritten_adapter import HandwrittenProbe
from producer_closure import ProducerClosure
from qualification import RecordedAccuracy
from session import Session


class HandwrittenPlan:
    @staticmethod
    def specification(source, session):
        kind = source["kind"]
        if kind == "tf32_v7":
            return {"precision": "tf32", "operations": ["forward", "grad_x"], "boundary": "raw_allocating",
                    "factory": "handwritten_adapter.HandwrittenProbe", "variant_id": "handwritten_tf32_v7", "adapter_path_counts": [4, 10, 17]}
        if kind == "original_cuda_fp64_4463b10":
            return {"precision": "fp64", "operations": ["forward", "grad_x"], "boundary": "raw_allocating",
                    "factory": "original_cuda_probe.OriginalCudaFp64Probe", "variant_id": kind, "adapter_path_counts": [4]}
        policy = session.root / "coverage/handwritten_fp64_source02/fp64_contracts.json"
        contract = next(row for row in json.loads(policy.read_text())["contracts"] if row["kind"] == kind)
        if contract["math_allowed"] is not True or contract["source_sha256"] not in source["original_files"].values():
            raise ValueError("Unknown or changed FP64 source contract")
        return {"precision": "fp64", "operations": ["forward"], "boundary": "adapted_public_e2e",
                "factory": "handwritten_fp64.HandwrittenFp64Probe", "variant_id": "handwritten_" + kind,
                "adapter_path_counts": [4, 10, 17] if kind == "fp64_gmm_p1_v2" else contract["allowed_path_counts"],
                "source_contract": {"path": str(policy), "sha256": Evidence.sha256(policy)}}

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.value = json.loads(self.path.read_text())
        self.directory = self.path.parent
        self.session = Session(Path(self.value["session"]))
        if Evidence.fingerprint({key: value for key, value in self.value.items() if key != "plan_hash"}) != self.value["plan_hash"]:
            raise ValueError("Handwritten plan changed")
        self.verify()

    def verify(self):
        for name, digest in self.value["producer"].items():
            if Evidence.sha256(self.directory / "producer" / name) != digest:
                raise ValueError("Handwritten frozen producer changed")
        for filename, digest in self.value["source_evidence"].items():
            if Evidence.sha256(Path(filename)) != digest:
                raise ValueError("Handwritten source/build/DSO changed")

    @classmethod
    def create(cls, args):
        session = Session(args.session)
        source_path, build_path = args.build / "source.json", args.build / "build.json"
        source, build = json.loads(source_path.read_text()), json.loads(build_path.read_text())
        if build["status"] != "PASS":
            raise ValueError("A successfully rebuilt handwritten source is required")
        specification = cls.specification(source, session)
        directory = args.directory.resolve()
        if not directory.is_relative_to(session.root):
            raise ValueError("Matrix must stay in its session")
        producer = directory / "producer"
        producer.mkdir(parents=True, exist_ok=False)
        files = ProducerClosure(Path(__file__).parent, producer).freeze(["handwritten_matrix.py", specification["factory"].split(".")[0] + ".py"])
        evidence = {str(path.resolve()): Evidence.sha256(path) for path in (source_path, build_path, Path(build["binary"]["path"]))}
        evidence.update({str(Path(source["archive"]) / name): digest for name, digest in source["original_files"].items()})
        evidence.update({str(args.build / "source" / name): digest for name, digest in source["staged_files"].items()})
        for record in (source.get("math_policy"), source.get("historical_source_manifest"), specification.get("source_contract")):
            if record:
                evidence[record["path"]] = record["sha256"]
        gpu = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).strip()
        candidates = [point for point in points() if point.precision == specification["precision"]]
        selected = [point for point in candidates if len(CONFIGS[point.config][1]) in specification["adapter_path_counts"]]
        value = {"session": str(session.root), **specification, "kind": source["kind"], "build": str(args.build.resolve()),
                 "source_hash": source["source_hash"], "binary_hash": build["binary"]["sha256"],
                 "gpu": {"index": args.gpu, "uuid": gpu.removeprefix("GPU-").lower()},
                 "source_evidence": evidence, "producer": files, "jobs": [point.columns() for point in selected],
                 "source_limit_exclusions_pending_minimal_native_probe": [point.columns() for point in candidates if point not in selected],
                 "leaf_deadline_seconds": 90 if specification["precision"] == "tf32" else 120,
                 "deadline_basis": "TF32 same-DSO real probe7.03s; FP64 is fresh already-compiled code with bounded larger tensor/reference IO, no unbounded retry",
                 "classification": "HANDWRITTEN_COMPLETE_ACCURACY_NOT_PERFORMANCE"}
        value["plan_hash"] = Evidence.fingerprint(value)
        Evidence.write(directory / "plan.json", value)
        return cls(directory / "plan.json")


class HandwrittenMatrix:
    def __init__(self, plan):
        self.plan = plan
        self.reference_errors = {}

    @staticmethod
    def worker_argv(plan, point, output):
        return [plan.session.paths["FASTEQ_PY"], "-B", str(plan.directory / "producer/handwritten_matrix.py"),
                "worker", "--plan", str(plan.path), "--point", point["config"] + "/" + str(point["batch"]), "--output", str(output)]

    def run(self):
        plan, session = self.plan, self.plan.session
        receipt = plan.directory / "receipt.json"
        if receipt.exists():
            raise FileExistsError("Inspect existing terminal/PID; do not blindly restart a matrix")
        value = {"status": "RUNNING", "plan_hash": plan.value["plan_hash"], "attempts": []}
        Evidence.write(receipt, value)
        for point in plan.value["jobs"]:
            plan.verify()
            key = "_".join(str(point[name]) for name in ("config", "batch", "precision"))
            name = plan.directory.name + "_" + key
            gpu = session.idle_gpu(name, plan.value["gpu"]["index"])
            if gpu.removeprefix("GPU-").lower() != plan.value["gpu"]["uuid"]:
                raise ValueError("Physical GPU identity changed")
            output = plan.directory / "jobs" / key
            terminal = session.runner().run(name, self.worker_argv(plan, point, output),
                plan.value["leaf_deadline_seconds"], Path(session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": gpu})
            result_path = output / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            acceptable = terminal["status"] in ("PASS", "FAIL") and result.get("status") in ("PASS", "ACCURACY_FAIL") and result.get("source_unchanged") is True
            value["attempts"].append({"point": point, "name": name, "terminal": terminal["status"], "result": str(result_path),
                                      "result_sha256": Evidence.sha256(result_path) if result_path.exists() else None, "numerical_status": result.get("status")})
            Evidence.write(receipt, value)
            if not acceptable:
                value["status"] = "INCOMPLETE_DIAGNOSIS_REQUIRED"
                Evidence.write(receipt, value)
                return 1
        plan.verify()
        value["status"] = "COMPLETE"
        Evidence.write(receipt, value)
        return 0

    def audit(self, output):
        plan = self.plan
        receipt_path = plan.directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        expected = plan.value["jobs"]
        errors, counts = [], Counter()
        if receipt.get("status") != "COMPLETE" or receipt.get("plan_hash") != plan.value["plan_hash"] or [row["point"] for row in receipt.get("attempts", [])] != expected:
            errors.append("matrix incomplete, reordered, duplicated, or changed plan")
        for attempt in receipt.get("attempts", []):
            path = Path(attempt["result"])
            result = json.loads(path.read_text())
            terminal = json.loads((plan.session.root / "logs" / (attempt["name"] + ".json")).read_text())
            reasons = []
            if Evidence.sha256(path) != attempt["result_sha256"] or result.get("point") != attempt["point"]:
                reasons.append("result bytes or point differs")
            if terminal.get("status") not in ("PASS", "FAIL") or result.get("status") not in ("PASS", "ACCURACY_FAIL") or result.get("source_unchanged") is not True:
                reasons.append("nonterminal or source drift")
            if terminal.get("argv") != self.worker_argv(plan, attempt["point"], path.parent):
                reasons.append("actual producer command differs")
            if result.get("plan_hash") != plan.value["plan_hash"] or result.get("producer_drift") is not False:
                reasons.append("plan or producer drift")
            if result.get("variant") != plan.value["variant_id"]:
                reasons.append("actual tested variant differs")
            if result.get("GPU_UUID", "").removeprefix("GPU-").lower() != plan.value["gpu"]["uuid"]:
                reasons.append("actual GPU differs")
            identity = result.get("identity", {})
            build = json.loads((Path(plan.value["build"]) / "build.json").read_text())
            source = json.loads((Path(plan.value["build"]) / "source.json").read_text())
            if any(identity.get(key) != plan.value[key] for key in ("source_hash", "binary_hash")):
                reasons.append("actual source/DSO differs")
            if (identity.get("actual_mapped_dso") != build["binary"]["path"]
                    or identity.get("binary_files") != {build["binary"]["path"]: {"sha256": build["binary"]["sha256"]}}
                    or identity.get("operator_namespace") != source["namespace"]):
                reasons.append("actual mapped backend or namespace differs")
            fixture = plan.session.root / "fixtures" / ("_".join(str(attempt["point"][name]) for name in ("config", "batch", "precision")) + ".json")
            metadata = json.loads(fixture.read_text())
            if Evidence.sha256(fixture.with_suffix(".pt")) != metadata["file_sha256"] or result["layouts"] != metadata["layouts"]:
                reasons.append("canonical fixture file or layout differs")
            if any(result["tensor_hashes"].get(name + "_sha256") != digest for name, digest in metadata["tensor_hashes"].items()):
                reasons.append("canonical tensor content differs")
            reference = result.get("reference_store", {})
            if not reference or Evidence.sha256(Path(reference["path"])) != reference["sha256"] or Evidence.sha256(Path(reference["data_path"])) != reference["data_sha256"]:
                reasons.append("reference bytes differ or missing")
            elif (reference["identity"]["input_hash"] != result["input_hash"] or reference["value_hashes"]["fp64"] != result["reference_hashes"]):
                reasons.append("actual reference/input mapping differs")
            else:
                stored = json.loads(Path(reference["path"]).read_text())
                if (stored["identity"] != reference["identity"] or stored["value_hashes"] != reference["value_hashes"]
                        or stored["file_sha256"] != reference["data_sha256"]
                        or stored["identity"]["point"] != attempt["point"]
                        or stored["identity"]["tensor_hashes"] != result["tensor_hashes"]):
                    reasons.append("actual cached reference metadata differs")
                key = reference["data_path"], reference["data_sha256"]
                if key not in self.reference_errors:
                    import torch
                    self.reference_errors[key] = torch.load(reference["data_path"], map_location="cpu", weights_only=True, mmap=True)["torch_errors"]
                if self.reference_errors[key] != result["torch_errors"]:
                    reasons.append("actual cached Torch comparator differs")
            rows = result.get("rows", [])
            keys = [(row["operation"], row["result"], row["boundary"]) for row in rows]
            wanted = {(operation, operation, plan.value["boundary"]) for operation in plan.value["operations"]}
            if set(keys) != wanted or len(keys) != len(wanted):
                reasons.append("missing/duplicate operation/boundary")
            if result.get("producer_files_start") != result.get("producer_files_end"):
                reasons.append("producer changed during probe")
            for filename, digest in result.get("producer_files_start", {}).items():
                if (plan.value["producer"].get(Path(filename).name) != digest or Evidence.sha256(Path(filename)) != digest
                        or Evidence.sha256(path.parent / "producer_sources" / Path(filename).name) != digest):
                    reasons.append("producer snapshot differs")
            for row in rows:
                passed = RecordedAccuracy.passes(row, attempt["point"]["precision"])
                if row.get("passed") is not passed or row.get("torch_rel_fro") != result["torch_errors"][row["result"]]:
                    reasons.append("independent gate or Torch comparator differs")
                for failure in row.get("failed_outputs", {}).values():
                    if Evidence.sha256(Path(failure["path"])) != failure["sha256"]:
                        reasons.append("failure tensor changed")
                if not passed and set(row.get("failed_outputs", {})) != {"first", "repeat"}:
                    reasons.append("actual failure tensors missing")
                counts["cells"] += 1
                counts["qualified_cells"] += int(passed)
            if reasons:
                errors.append({"point": attempt["point"], "reasons": sorted(set(reasons))})
            else:
                counts["verified_points"] += 1
                counts["qualified_points"] += int(all(row["passed"] for row in rows))
        complete = not errors and counts["verified_points"] == len(expected)
        if output.exists():
            raise FileExistsError(output)
        Evidence.write(output, {"complete": complete, "errors": errors, "counts": dict(counts),
            "selected_scope_complete": complete,
            "source_limit_exclusions_pending_minimal_native_probe": plan.value.get("source_limit_exclusions_pending_minimal_native_probe", []),
            "plan": {"path": str(plan.path), "sha256": Evidence.sha256(plan.path)},
            "receipt": {"path": str(receipt_path), "sha256": Evidence.sha256(receipt_path)}, "auditor_sha256": Evidence.sha256(Path(__file__))})
        return 0 if complete else 1

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        sub = parser.add_subparsers(dest="action", required=True)
        create = sub.add_parser("create")
        create.add_argument("--session", type=Path, required=True)
        create.add_argument("--build", type=Path, required=True)
        create.add_argument("--directory", type=Path, required=True)
        create.add_argument("--gpu", type=int, required=True)
        for action in ("run", "audit", "worker"):
            child = sub.add_parser(action)
            child.add_argument("--plan", type=Path, required=True)
            if action in ("audit", "worker"):
                child.add_argument("--output", type=Path, required=True)
            if action == "worker":
                child.add_argument("--point", required=True)
        args = parser.parse_args()
        if args.action == "create":
            print(HandwrittenPlan.create(args).path)
            return 0
        plan = HandwrittenPlan(args.plan)
        if args.action == "worker":
            probe_args = argparse.Namespace(session=plan.session.root, build=Path(plan.value["build"]), precision=plan.value.get("precision", "tf32"),
                point=args.point, output=args.output, operations=",".join(plan.value["operations"]),
                impl=7 if plan.value.get("precision", "tf32") == "tf32" else 0,
                bimpl=8 if plan.value.get("precision", "tf32") == "tf32" else 0, trace=False)
            module_name, class_name = plan.value.get("factory", "handwritten_adapter.HandwrittenProbe").rsplit(".", 1)
            probe = getattr(importlib.import_module(module_name), class_name)(probe_args)
            code = probe.run()
            plan.verify()
            probe.result.update(plan_hash=plan.value["plan_hash"], source_unchanged=True)
            probe.flush()
            return code
        return cls(plan).run() if args.action == "run" else cls(plan).audit(args.output)


if __name__ == "__main__":
    raise SystemExit(HandwrittenMatrix.main())
