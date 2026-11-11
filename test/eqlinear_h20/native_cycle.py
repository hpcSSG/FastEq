"""Frozen current-native selector accuracy plan using the shared chunk coordinator."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from audit_accuracy import AccuracyAudit
from cases import CONFIGS, points
from common import Evidence
from native_bridge import NativeAdapter, NativeBridge, NativeProbe
from producer_closure import ProducerClosure
from reference_store import ReferenceStore
from session import Session
from standalone_matrix import SharedPointBatch, MatrixAudit


class NativePlan:
    FILES = ("native_cycle.py", "native_bridge.py", "native_bridge.cpp.in", "current_matrix.py", "standalone_probe.py",
             "standalone.py", "reference_store.py", "cases.py", "common.py", "protocol.py", "session.py",
             "audit_accuracy.py", "standalone_matrix.py", "current_probe.py", "current.py")

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.value = json.loads(self.path.read_text())
        self.root = self.path.parent
        self.session = Session(Path(self.value["session"]))
        if self.value["plan_hash"] != Evidence.fingerprint({k: v for k, v in self.value.items() if k != "plan_hash"}):
            raise ValueError("Native selector plan fingerprint differs")
        batched = self.value.get("execution_model") == "same_build_point_batch"
        if self.value["jobs"] != self.jobs(self.value["cycle"], batched):
            raise ValueError("Missing, duplicate or excluded native selector/point")
        if batched and self.value["batch_groups"] != self.batch_groups(self.value):
            raise ValueError("Native batch mapping omits or changes a selector/point")
        self.verify()

    @staticmethod
    def jobs(cycle, batched=False):
        result = []
        for precision, config in NativeBridge.CONFIG.items():
            for impl in config[4]:
                for point in points():
                    if point.precision == precision:
                        job = {"precision": precision, "impl": impl, "point": point.columns(),
                                       "operation": "forward", "boundaries": list(NativeAdapter.BOUNDARIES),
                                       "attempt": f"native_{precision}_impl{impl}_accuracy_{point.config}_{cycle}_{point.batch}_{precision}"}
                        if batched:
                            job.update(variant_id=f"native_{precision}_impl{impl}",
                                       job_id=f"native_{precision}_impl{impl}_{point.config}_{point.batch}")
                        result.append(job)
        return result

    @staticmethod
    def batch_groups(value):
        groups = {}
        for job in value["jobs"]:
            key = (job["precision"], job["point"]["config"], job["point"]["batch"])
            group = groups.setdefault(key, {"batch_id": "native_batch_" + "_".join(map(str, key)) + "_" + value["cycle"],
                "precision": job["precision"], "point": job["point"], "job_ids": [],
                "deadline_seconds": 120 if job["precision"] == "fp64" else 60})
            group["job_ids"].append(job["job_id"])
        return list(groups.values())

    @staticmethod
    def uuid(value):
        return ReferenceStore.device_uuid(value)

    def job(self, job_id):
        return next(job for job in self.value["jobs"] if job["job_id"] == job_id)

    def verify_producer(self, root):
        NativeBridge.verify_files({str(Path(root) / name): sha for name, sha in self.value["producer"].items()})

    def verify_sources(self, session):
        if session.root != self.session.root:
            raise ValueError("Batch session substitution")
        self.verify()

    def verify(self):
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") not in (None, "0"):
            raise ValueError("TF32 comparator override is forbidden before reference or native work")
        NativeBridge.verify_files({str(self.root / "producer" / name): sha for name, sha in self.value["producer"].items()})
        NativeBridge.verify_files({str(self.session.root / "config/scope.yaml"): self.value["scope_sha256"]})
        for precision, source in self.value["sources"].items():
            bridge = NativeBridge(self.session, precision, self.value["build_id"])
            if Evidence.sha256(bridge.manifest_path) != source["manifest_sha256"]:
                raise ValueError("Native bridge manifest differs from frozen cycle")
            bridge.verify()

    @classmethod
    def create(cls, args):
        if not re.fullmatch(r"[A-Za-z0-9_]+", args.cycle):
            raise ValueError("Use a simple, fresh cycle name")
        session = Session(args.session)
        sources = {}
        for precision in NativeBridge.CONFIG:
            bridge = NativeBridge(session, precision, args.build_id)
            receipt = bridge.verify()
            sources[precision] = {"manifest": str(bridge.manifest_path), "manifest_sha256": Evidence.sha256(bridge.manifest_path),
                                  "binary": str(bridge.library), "binary_sha256": receipt["binary_sha256"],
                                  "native_objects_sha256": Evidence.fingerprint(receipt["contract"]["native_files"]),
                                  "source_commit": receipt["contract"]["source_commit"]}
        root = session.root / "accuracy/native_selector" / args.cycle
        (root / "producer").mkdir(parents=True, exist_ok=False)
        producer = ProducerClosure(Path(__file__).parent, root / "producer").freeze(cls.FILES)
        batched = getattr(args, "execution_model", "point_workers") == "same_build_point_batch"
        value = {"classification": "CURRENT_NATIVE_SELECTOR_ACCURACY_NOT_FORMAL_PERFORMANCE",
                 "session": str(session.root), "cycle": args.cycle, "build_id": args.build_id,
                 "producer": producer,
                 "sources": sources, "scope_sha256": Evidence.sha256(session.root / "config/scope.yaml"),
                 "jobs": cls.jobs(args.cycle, batched), "expected_points": 475, "expected_paths": 1425,
                 "allowed_impls": {precision: list(config[4]) for precision, config in NativeBridge.CONFIG.items()},
                 "excluded_tf32_impls": [1, 2], "boundaries": list(NativeAdapter.BOUNDARIES),
                 "runtime_controls": {}, "math": ReferenceStore.MATH,
                 "queued_safety_verified": False, "H100_frozen_additional_gate": "UNVERIFIED_INPUT_HASH_MISMATCH"}
        if batched:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                                             text=True, timeout=10)
            gpu = next(line.split(",")[1].strip() for line in output.splitlines() if int(line.split(",")[0]) == args.gpu)
            value.update(execution_model="same_build_point_batch", reference_root=str(session.root / "reference_store"),
                         gpu={"physical_index": args.gpu, "uuid": cls.uuid(gpu), "selection_uuid": gpu})
            value["batch_groups"] = cls.batch_groups(value)
        value["plan_hash"] = Evidence.fingerprint(value)
        Evidence.write(root / "plan.json", value)
        print(json.dumps({"plan": str(root / "plan.json"), "points": 475, "paths": 1425}))
        return cls(root / "plan.json")


class NativeBatch(SharedPointBatch):
    PLAN_CLASS = NativePlan
    CLASSIFICATION = "CURRENT_NATIVE_SELECTOR_ACCURACY_NOT_TIMING"

    def source_build(self, precision):
        self.bridge = NativeBridge(self.session, precision, self.plan.value["build_id"])
        self.verified_receipt = json.loads(self.bridge.manifest_path.read_text())
        return super().source_build(precision)

    def create_worker(self, args):
        job = self.plan.job(args.job_id)
        point = job["point"]
        probe = argparse.Namespace(session=self.session.root, output=args.output, precision=job["precision"],
            impl=job["impl"], bimpl=0, point=f"{point['config']}/{point['batch']}", operations="forward",
            boundaries=",".join(job["boundaries"]), trace=False, build_id=self.plan.value["build_id"])
        worker = NativeProbe(probe, shared=self)
        for name, sha in self.plan.value["producer"].items():
            path = self.plan.root / "producer" / name
            worker.producer_files[str(path)] = sha
            shutil.copy2(path, worker.output / "producer_sources" / name)
        worker.result.update(plan_hash=self.plan.value["plan_hash"], job_id=job["job_id"], cycle=self.plan.value["cycle"],
            shared_batch={"batch_id": self.group["batch_id"], "receipt": str(self.receipt),
                          "variant_index": self.group["job_ids"].index(job["job_id"])})
        worker.result["producer_files_start"] = dict(worker.producer_files)
        return worker


class NativeAudit(AccuracyAudit):
    """Extend the common audit identity model; share standalone row-gate derivation."""
    CELL_COUNT = 3
    REPORT_CLASSIFICATION = "CURRENT_NATIVE_SELECTOR_ACCURACY_NOT_FORMAL_PERFORMANCE"

    def scope_complete(self, jobs):
        return len(jobs) == 475

    def auditor_file(self):
        return Path(__file__)

    def __init__(self, plan):
        self.plan, self.session = plan, plan.session
        self.file_hashes, self.references = {}, {}

    def hash(self, path):
        path = Path(path)
        if str(path) not in self.file_hashes:
            self.file_hashes[str(path)] = Evidence.sha256(path)
        return self.file_hashes[str(path)]

    def canonical_binding(self, job, result):
        import torch

        root, reasons = self.session.root, []
        point_key = job["point"]["config"] + "_" + str(job["point"]["batch"]) + "_" + job["precision"]
        fixture_path = root / "fixtures" / (point_key + ".pt")
        fixture = json.loads(fixture_path.with_suffix(".json").read_text())
        if (fixture["file_sha256"] != self.hash(fixture_path) or fixture["point"] != job["point"] or
                fixture["layouts"] != result.get("layouts") or
                any(result.get("tensor_hashes", {}).get(name + "_sha256") != sha for name, sha in fixture["tensor_hashes"].items())):
            reasons.append("canonical tensor contents/layouts differ")
        reference = result.get("reference_store", {})
        path = Path(reference["path"])
        metadata = json.loads(path.read_text())
        ref_identity = metadata["identity"]
        if (self.hash(path) != reference["sha256"] or self.hash(path.parent / "references.pt") != metadata["file_sha256"] or
                metadata["value_hashes"]["fp64"] != result.get("reference_hashes") or
                ref_identity.get("input_hash") != result.get("input_hash") or ref_identity.get("point") != job["point"] or
                ref_identity.get("tensor_hashes") != result.get("tensor_hashes") or ref_identity.get("layouts") != result.get("layouts") or
                ref_identity.get("math") != ReferenceStore.MATH or
                ref_identity.get("reference_producer_sha256") != self.plan.value["producer"]["cases.py"] or
                ref_identity.get("store_producer_sha256") != self.plan.value["producer"]["reference_store.py"] or
                ReferenceStore.device_uuid(ref_identity["hardware"]["uuid"]) != ReferenceStore.device_uuid(result.get("GPU_UUID"))):
            reasons.append("canonical gold/reference producer or hardware identity differs")
        if str(path) not in self.references:
            self.references[str(path)] = torch.load(path.parent / "references.pt", map_location="cpu", weights_only=True, mmap=True)["torch_errors"]
        return reasons, self.references[str(path)]

    def record(self, job):
        from standalone_matrix import MatrixAudit
        import torch

        root = self.session.root
        result_path = root / "accuracy" / job["attempt"] / "result.json"
        group = next((group for group in self.plan.value.get("batch_groups", []) if job["job_id"] in group["job_ids"]), None)
        terminal_path = root / "logs" / ((group["batch_id"] if group else job["attempt"]) + ".json")
        if not result_path.exists() or not terminal_path.exists():
            return {"verified": False, "reasons": ["missing result or process terminal"]}
        result, terminal = json.loads(result_path.read_text()), json.loads(terminal_path.read_text())
        reasons = []
        expected_status = "PASS" if all(row.get("passed") for row in result.get("rows", [])) else "ACCURACY_FAIL"
        if (result.get("status") != expected_status or terminal.get("returncode") != (0 if group or expected_status == "PASS" else 1)
                or terminal.get("status") != ("PASS" if group or expected_status == "PASS" else "FAIL")):
            reasons.append("nonterminal, process error or result/gate disagreement")
        if group:
            path = self.plan.root / "batches" / group["batch_id"] / "receipt.json"
            batch = json.loads(path.read_text())
            completed = next((row for row in batch.get("completed", []) if row.get("job_id") == job["job_id"]), {})
            argv = [self.session.paths["FASTEQ_PY"], "-B", str(self.plan.root / "producer/native_cycle.py"),
                    "batch-worker", "--plan", str(self.plan.path), "--batch-id", group["batch_id"]]
            if (not MatrixAudit.check_batch(group, batch) or batch.get("plan_hash") != self.plan.value["plan_hash"] or
                    terminal.get("argv") != argv or completed.get("result_sha256") != self.hash(result_path) or
                    completed.get("status") != result.get("status") or completed.get("result") != str(result_path) or
                    result.get("plan_hash") != self.plan.value["plan_hash"] or result.get("job_id") != job["job_id"] or
                    result.get("shared_batch") != {"batch_id": group["batch_id"], "receipt": str(path),
                                                  "variant_index": group["job_ids"].index(job["job_id"])} or
                    batch.get("source_build") != self.plan.value["sources"][job["precision"]] or
                    batch.get("gpu_uuid") != self.plan.value["gpu"]["uuid"] or
                    ReferenceStore.device_uuid(result.get("GPU_UUID")) != self.plan.value["gpu"]["uuid"] or
                    any(batch.get(key) != result.get(key) for key in ("input_hash", "reference_hashes", "tensor_hashes", "layouts", "reference_store"))):
                reasons.append("batch process, frozen source, shared reference, case or result mapping differs")
        if result.get("producer_drift") is not False or result.get("source_unchanged") is not True:
            reasons.append("producer or source changed during probe")
        if result.get("point") != job["point"] or result.get("impl") != job["impl"]:
            reasons.append("point, seed or impl differs")
        producer_start, producer_end = result.get("producer_files_start", {}), result.get("producer_files_end", {})
        if not producer_start or producer_start != producer_end:
            reasons.append("probe producer epochs differ")
        for filename, sha in producer_start.items():
            name = Path(filename).name
            if self.plan.value["producer"].get(name) != sha or self.hash(result_path.parent / "producer_sources" / name) != sha:
                reasons.append("producer snapshot differs: " + name)
        source, identity = self.plan.value["sources"][job["precision"]], result.get("identity", {})
        for field in ("binary", "binary_sha256", "manifest", "manifest_sha256", "native_objects_sha256", "source_commit"):
            if identity.get(field) != source[field]:
                reasons.append("source/ABI/DSO identity differs: " + field)
        if (identity.get("family") != "current_native_selector" or identity.get("impl") != job["impl"] or
                identity.get("precision") != job["precision"] or identity.get("mapped_libraries") != [source["binary"]] or
                identity.get("production_DSO_loaded") is not False or identity.get("runtime_controls") != {}):
            reasons.append("native operation selector, actual mapping or control differs")
        if job["precision"] == "tf32" and identity.get("TF32x1_cpp_entry_rejections_verified") is not True:
            reasons.append("C++ x1 guard lacks proof")
        canonical_reasons, torch_errors = self.canonical_binding(job, result)
        reasons += canonical_reasons
        reasons += MatrixAudit.check_rows(job, result.get("rows", []), torch_errors)
        for row in result.get("rows", []):
            if not row["passed"]:
                if set(row.get("failed_outputs", {})) != {"first", "repeat"}:
                    reasons.append("missing failed output tensors")
                for artifact in row.get("failed_outputs", {}).values():
                    if self.hash(artifact["path"]) != artifact["sha256"]:
                        reasons.append("failed output content differs")
        return {"verified": not reasons, "reasons": sorted(set(reasons)), "accuracy_pass": expected_status == "PASS",
                "result": str(result_path), "result_sha256": self.hash(result_path), "paths": len(result.get("rows", []))}

    def run(self, output, precision=None, impl=None, config=None):
        self.plan.verify()
        jobs = [job for job in self.plan.value["jobs"] if (precision is None or job["precision"] == precision)
                and (impl is None or job["impl"] == impl) and (config is None or job["point"]["config"] == config)]
        if not jobs:
            raise ValueError("Native audit selection is empty")
        records = {}
        for job in jobs:
            try:
                records[job["attempt"]] = self.record(job)
            except (OSError, KeyError, ValueError, TypeError) as error:
                records[job["attempt"]] = {"verified": False, "reasons": [type(error).__name__ + ": " + str(error)]}
        counts = Counter(verified_points=sum(row["verified"] for row in records.values()),
                         verified_paths=sum(row.get("paths", 0) for row in records.values() if row["verified"]),
                         accuracy_failed_points=sum(row.get("accuracy_pass") is False for row in records.values() if row["verified"]))
        complete = counts["verified_points"] == len(jobs) and counts["verified_paths"] == len(jobs) * self.CELL_COUNT
        report = {"classification": self.REPORT_CLASSIFICATION,
                  "plan": str(self.plan.path), "plan_hash": self.plan.value["plan_hash"],
                  "complete": complete, "accuracy_pass": complete and counts["accuracy_failed_points"] == 0,
                  "full_registry_complete": complete and self.scope_complete(jobs), "counts": dict(counts),
                  "expected_points": len(jobs), "expected_paths": len(jobs) * self.CELL_COUNT, "records": records,
                  "auditor_sha256": Evidence.sha256(self.auditor_file()), "H100_frozen_additional_gate": "UNVERIFIED_INPUT_HASH_MISMATCH"}
        Evidence.write(output, report)
        print(json.dumps({key: report[key] for key in ("complete", "accuracy_pass", "expected_points", "counts")}))
        return report


class NativeController:
    def __init__(self, plan):
        self.plan, self.session = plan, plan.session

    def run(self, args):
        plan, session = self.plan, self.session
        audit = NativeAudit(plan)
        selected = [job for job in plan.value["jobs"] if (args.precision is None or job["precision"] == args.precision)
                    and (args.impl is None or job["impl"] == args.impl) and (args.config is None or job["point"]["config"] == args.config)]
        chunks = list(dict.fromkeys((job["precision"], job["impl"], job["point"]["config"]) for job in selected))
        if not chunks:
            raise ValueError("Empty native controller selection")
        records = []
        for precision, impl, config in chunks:
            name = f"native_{precision}_impl{impl}_accuracy_{config}_{plan.value['cycle']}"
            path = session.root / "accuracy" / (name + ".json")
            jobs = [job for job in selected if (job["precision"], job["impl"], job["point"]["config"]) == (precision, impl, config)]
            if path.exists():
                chunk = json.loads(path.read_text())
                if not chunk.get("complete") or not all(audit.record(job)["verified"] for job in jobs):
                    raise ValueError("Existing nonverified chunk must be diagnosed, not restarted: " + name)
                records.append({"chunk": name, "status": "ALREADY_VERIFIED"})
                continue
            plan.verify()
            argv = [session.paths["FASTEQ_PY"], "-B", str(plan.root / "producer/native_bridge.py"), "matrix",
                    "--session", str(session.root), "--precision", precision, "--impl", str(impl),
                    "--build-id", plan.value["build_id"], "--config", config, "--gpu", str(args.gpu),
                    "--attempt", plan.value["cycle"]]
            terminal = session.runner().run(name + "_coordinator", argv, 300, Path(session.paths["FASTEQ_ROOT"]),
                                            {"CUDA_VISIBLE_DEVICES": ""})
            plan.verify()
            chunk = json.loads(path.read_text()) if path.exists() else {}
            verified = terminal["status"] == "PASS" and chunk.get("complete") and all(audit.record(job)["verified"] for job in jobs)
            records.append({"chunk": name, "process_status": terminal["status"], "verified": bool(verified),
                            "seconds": terminal["elapsed_seconds"]})
            Evidence.write(plan.root / ("controller_" + precision + "_" + str(impl) + "_" + config + ".json"), records[-1])
            if not verified:
                raise RuntimeError("Native chunk requires diagnosis: " + name)
        Evidence.write(plan.root / "latest_controller.json", records)
        return 0


class NativeBatchController(NativeController):
    def run(self, args):
        if args.impl is not None:
            raise ValueError("A batch must retain every allowed selector for its DSO/point")
        plan, session = self.plan, self.session
        audit = NativeAudit(plan)
        groups = [group for group in plan.value["batch_groups"] if
                  (args.precision is None or group["precision"] == args.precision) and
                  (args.config is None or group["point"]["config"] == args.config)]
        completed = []
        for group in groups:
            jobs = [plan.job(job_id) for job_id in group["job_ids"]]
            path = plan.root / "batches" / group["batch_id"] / "receipt.json"
            if path.exists():
                if not all(audit.record(job)["verified"] for job in jobs):
                    raise ValueError("Existing partial batch requires diagnosis; never restart it")
                continue
            if len(completed) == args.max_groups:
                break
            plan.verify()
            uuid = session.idle_gpu(group["batch_id"], plan.value["gpu"]["physical_index"])
            argv = [session.paths["FASTEQ_PY"], "-B", str(plan.root / "producer/native_cycle.py"),
                    "batch-worker", "--plan", str(plan.path), "--batch-id", group["batch_id"]]
            terminal = session.runner().run(group["batch_id"], argv, group["deadline_seconds"],
                Path(session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": uuid})
            plan.verify()
            valid = terminal["status"] == "PASS" and all(audit.record(job)["verified"] for job in jobs)
            record = {"batch_id": group["batch_id"], "verified": bool(valid), "seconds": terminal["elapsed_seconds"]}
            completed.append(record)
            Evidence.write(plan.root / "batches" / group["batch_id"] / "audit.json", record)
            if not valid:
                raise RuntimeError("Batch process or evidence requires diagnosis")
        Evidence.write(plan.root / "latest_batch_controller.json", completed)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "audit", "batch-worker"))
    parser.add_argument("--session", type=Path)
    parser.add_argument("--cycle")
    parser.add_argument("--build-id", default="bridge01")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--precision", choices=tuple(NativeBridge.CONFIG))
    parser.add_argument("--impl", type=int)
    parser.add_argument("--config", choices=tuple(CONFIGS))
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-model", choices=("point_workers", "same_build_point_batch"), default="point_workers")
    parser.add_argument("--batch-id")
    parser.add_argument("--max-groups", type=int, default=75)
    args = parser.parse_args()
    if args.action == "plan":
        NativePlan.create(args)
        return 0
    plan = NativePlan(args.plan)
    if args.action == "batch-worker":
        return NativeBatch(args).run()
    if args.action == "run":
        return (NativeBatchController if plan.value.get("execution_model") == "same_build_point_batch" else NativeController)(plan).run(args)
    if not args.output or not args.output.resolve().is_relative_to(plan.session.root):
        parser.error("Audit output must stay under the configured session")
    report = NativeAudit(plan).run(args.output, args.precision, args.impl, args.config)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
