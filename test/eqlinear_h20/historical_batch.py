"""Historical wrapper probes sharing only identical actual DSO/ABI/control groups."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

from cases import points
from common import Evidence
from historical_matrix import HistoricalIdentity
from historical_probe import HistoricalProbe
from native_cycle import NativeAudit
from producer_closure import ProducerClosure
from reference_store import ReferenceStore
from session import Session
from standalone_matrix import MatrixAudit, SharedPointBatch


class HistoryPlan:
    @staticmethod
    def uuid(value):
        return ReferenceStore.device_uuid(value)

    @staticmethod
    def jobs(value):
        result = []
        for variant, source in value["sources"].items():
            for point in points():
                if point.precision != source["precision"] or (value["stage"] == "minimal" and
                        (point.config != "SMALL_P1" or point.batch != 512)):
                    continue
                result.append({"variant_id": variant, "job_id": variant + "_" + point.key.replace("/", "_"),
                    "precision": point.precision, "point": point.columns(), "operation": "historical_five_paths",
                    "attempt": f"{variant}_accuracy_{point.config}_{value['cycle']}_{point.batch}_{point.precision}"})
        return result

    @staticmethod
    def groups(value):
        groups = {}
        for job in value["jobs"]:
            source = value["sources"][job["variant_id"]]
            key = source["dso_group"] + "_" + job["point"]["config"] + "_" + str(job["point"]["batch"])
            group = groups.setdefault(key, {"batch_id": "history_batch_" + key + "_" + value["cycle"],
                "precision": job["precision"], "point": job["point"], "job_ids": [],
                "deadline_seconds": 240 if job["precision"] == "fp64" else 180,
                "dso_group": source["dso_group"]})
            group["job_ids"].append(job["job_id"])
        return list(groups.values())

    def __init__(self, path):
        self.path, self.root = Path(path).resolve(), Path(path).resolve().parent
        self.value = json.loads(self.path.read_text())
        self.session = Session(Path(self.value["session"]))
        if self.value["plan_hash"] != Evidence.fingerprint({k: v for k, v in self.value.items() if k != "plan_hash"}):
            raise ValueError("Historical batch plan changed")
        if self.value["jobs"] != self.jobs(self.value) or self.value["batch_groups"] != self.groups(self.value):
            raise ValueError("Historical wrapper/point/batch scope changed")
        self.verify()

    def job(self, job_id):
        return next(row for row in self.value["jobs"] if row["job_id"] == job_id)

    def verify_producer(self, root):
        for name, sha in self.value["producer"].items():
            if Evidence.sha256(Path(root) / name) != sha:
                raise ValueError("Historical frozen producer changed")

    def verify_sources(self, session):
        if session.root != self.session.root or os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") not in (None, "0"):
            raise ValueError("Session substitution or forbidden TF32 comparator override")
        for variant, source in self.value["sources"].items():
            if HistoricalIdentity.observe(session, variant) != source["identity"]:
                raise ValueError("Historical source/DSO/ABI contract changed")
            if Evidence.sha256(Path(source["wrapper"])) != source["wrapper_sha256"]:
                raise ValueError("Historical private wrapper changed")

    def verify(self):
        self.verify_producer(self.root / "producer")
        self.verify_sources(self.session)

    @classmethod
    def create(cls, args):
        session = Session(args.session)
        parent = session.root / "coverage/historical_execution_plan05.json"
        requirements = json.loads(parent.read_text())["wrapper_tests"]
        selected = [row for row in requirements if row["known_default_accuracy_coverage"] == "UNTESTED"
                    and (not args.group_prefix or row["dispatcher_key"].startswith(args.group_prefix))
                    and (not getattr(args, "exclude_group_prefix", None) or
                         not row["dispatcher_key"].startswith(args.exclude_group_prefix))
                    and (not args.precision or row["precision"] == args.precision)]
        if not selected:
            raise ValueError("No historical wrapper selected")
        sources = {}
        for row in selected:
            variant = row["variant_id"]
            identity = HistoricalIdentity.observe(session, variant)
            manifest = json.loads(Path(identity["build_manifest_path"]).read_text())
            if manifest["runtime_contract_sha256"] != row["manifest_sha256"]:
                raise ValueError("Source contract does not match the reviewed history scope")
            wrapper = Path(manifest["harness"]) / "generated" / manifest["wrapper_name"]
            binaries = {value["path"]: value["sha256"] for value in manifest["binaries"]}
            controls = manifest.get("runtime_controls", {})
            group = Evidence.fingerprint({"binaries": binaries, "ABI": manifest["torch_abi"],
                                          "precision": row["precision"], "runtime_controls": controls})[:16]
            sources[variant] = {"identity": identity, "precision": row["precision"], "binaries": binaries,
                "runtime_controls": controls, "wrapper": str(wrapper), "wrapper_sha256": Evidence.sha256(wrapper),
                "source": manifest["source"], "dso_group": group, "tf32_policy": manifest["tf32_policy"]}
        output = session.root / "accuracy/historical_batch" / args.cycle
        (output / "producer").mkdir(parents=True, exist_ok=False)
        producer = ProducerClosure(Path(__file__).parent, output / "producer").freeze(("historical_batch.py",))
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                                      text=True, timeout=10)
        gpu = next(line.split(",")[1].strip() for line in gpu.splitlines() if int(line.split(",")[0]) == args.gpu)
        value = {"classification": "HISTORICAL_WRAPPER_ACCURACY_NOT_PERFORMANCE", "cycle": args.cycle,
            "session": str(session.root), "stage": args.stage, "execution_model": args.execution_model,
            "parent": {"path": str(parent), "sha256": Evidence.sha256(parent)}, "sources": sources, "producer": producer,
            "gpu": {"physical_index": args.gpu, "uuid": cls.uuid(gpu)},
            "reference_root": str(session.root / "reference_store"), "queued_safety_verified": False}
        value["jobs"] = cls.jobs(value)
        value["batch_groups"] = cls.groups(value)
        value["plan_hash"] = Evidence.fingerprint(value)
        Evidence.write(output / "plan.json", value)
        print(json.dumps({"plan": str(output / "plan.json"), "wrappers": len(sources), "jobs": len(value["jobs"]),
                          "cells": len(value["jobs"]) * 5, "groups": len(value["batch_groups"])}))
        return cls(output / "plan.json")


class HistoryWorker(HistoricalProbe):
    def __init__(self, plan, job, output, shared=None):
        self.plan, self.job, self.shared = plan, job, shared
        self.store = ReferenceStore(plan.value["reference_root"])
        source = plan.value["sources"][job["variant_id"]]
        os.environ["PYTHONPATH"] = source["source"]
        point = job["point"]
        args = argparse.Namespace(session=plan.session.root, variant=job["variant_id"], output=output,
            point=f"{point['config']}/{point['batch']}/{job['precision']}", reference_store=True, trace=False)
        kwargs = {"case_factory": shared.case_factory} if shared else {}
        super().__init__(args, reference_factory=self.reference, **kwargs)
        self.result.update(plan_hash=plan.value["plan_hash"], job_id=job["job_id"], cycle=plan.value["cycle"])
        if shared:
            self.result["shared_batch"] = {"batch_id": shared.group["batch_id"], "receipt": str(shared.receipt),
                "variant_index": shared.group["job_ids"].index(job["job_id"])}

    def reference(self, case):
        return self.shared.reference if self.shared else self.store.get(case)

    def producer_files(self):
        return tuple(self.plan.value["producer"])

    def run(self):
        result = super().run()
        if not self.shared:
            self.plan.verify()
        self.result["source_unchanged"] = True
        self.flush()
        return result


class HistoryBatch(SharedPointBatch):
    PLAN_CLASS = HistoryPlan
    CLASSIFICATION = "HISTORICAL_WRAPPER_ACCURACY_NOT_PERFORMANCE"

    def source_build(self, precision):
        return {self.plan.job(key)["variant_id"]: self.plan.value["sources"][self.plan.job(key)["variant_id"]]
                for key in self.group["job_ids"]}

    def case_factory(self, point, device, fixture_root):
        # Lease fresh descriptor metadata while sharing the exact immutable GPU tensor bytes.
        case = copy.copy(super().case_factory(point, device, fixture_root))
        case.descriptor = copy.deepcopy(case.descriptor)
        case.dims, case.cg = list(case.dims), list(case.cg)
        return case

    def create_worker(self, args):
        return HistoryWorker(self.plan, self.plan.job(args.job_id), args.output, shared=self)


class HistoryAudit(NativeAudit):
    CELL_COUNT = 5
    REPORT_CLASSIFICATION = "HISTORICAL_WRAPPER_ACCURACY_NOT_PERFORMANCE"
    REQUIRED = {"forward": ["public_warm", "run_with_activation_prep"], "forward_autograd": ["public_warm"],
                "grad_x": ["public_warm", "run_with_activation_prep"]}

    def scope_complete(self, jobs):
        return self.plan.value["stage"] == "full" and len(self.plan.value["sources"]) == 63 and len(jobs) == 1575

    def auditor_file(self):
        return Path(__file__)

    def record(self, job):
        output = self.session.root / "accuracy" / job["attempt"]
        result_path = output / "result.json"
        batched = self.plan.value["execution_model"] == "same_build_point_batch"
        group = next(group for group in self.plan.value["batch_groups"] if job["job_id"] in group["job_ids"])
        name = group["batch_id"] if batched else job["attempt"]
        terminal_path = self.session.root / "logs" / (name + ".json")
        if not result_path.exists() or not terminal_path.exists():
            return {"verified": False, "reasons": ["missing result/process terminal"]}
        result, terminal = json.loads(result_path.read_text()), json.loads(terminal_path.read_text())
        rows, reasons = result.get("boundaries", []), []
        passed = all(row.get("passed") for row in rows)
        if (result.get("status") != ("PASS" if passed else "ACCURACY_FAIL") or
                terminal.get("returncode") != (0 if batched or passed else 1) or
                result.get("point") != job["point"] or result.get("variant") != job["variant_id"] or
                result.get("job_id") != job["job_id"] or result.get("plan_hash") != self.plan.value["plan_hash"] or
                result.get("producer_unchanged") is not True or result.get("source_unchanged") is not True):
            reasons.append("process, producer or case/contract identity differs")
        for name, sha in self.plan.value["producer"].items():
            if result.get("producer", {}).get(name) != sha or self.hash(output / "producer" / name) != sha:
                reasons.append("historical producer snapshot differs")
        source, identity = self.plan.value["sources"][job["variant_id"]], result.get("identity", {})
        expected = source["identity"]
        for field in ("source_hash", "binary_hash", "source_commit", "namespace", "source_manifest_sha256"):
            if identity.get(field) != expected[field]:
                reasons.append("historical source/ABI/DSO identity differs: " + field)
        expected_maps = sorted(name for name in source["binaries"] if "eqlinear" in name)
        if (identity.get("mapped_libraries") != expected_maps or identity.get("module_file") != source["wrapper"] or
                identity.get("adapted_wrapper_sha256") != source["wrapper_sha256"] or
                identity.get("runtime_controls") != source["runtime_controls"] or
                identity.get("isolated_current_package_not_imported") is not True):
            reasons.append("actual DSO/module/control or current-package isolation differs")
        if job["precision"] == "tf32" and identity.get("plan_info", {}).get("forward", [None, None])[1] not in source["tf32_policy"]["allowed_forward_impls"]:
            reasons.append("forward plan is not this source ref's permitted TF32x3 enum")
        if batched:
            path = self.plan.root / "batches" / group["batch_id"] / "receipt.json"
            batch = json.loads(path.read_text())
            completed = next((row for row in batch.get("completed", []) if row.get("job_id") == job["job_id"]), {})
            if (not MatrixAudit.check_batch(group, batch) or completed.get("result_sha256") != self.hash(result_path) or
                    result.get("shared_batch") != {"batch_id": group["batch_id"], "receipt": str(path),
                                                   "variant_index": group["job_ids"].index(job["job_id"])} or
                    batch.get("plan_hash") != self.plan.value["plan_hash"] or
                    batch.get("gpu_uuid") != self.plan.value["gpu"]["uuid"] or
                    self.plan.uuid(result.get("GPU_UUID")) != self.plan.value["gpu"]["uuid"] or
                    any(batch.get(key) != result.get(key) for key in ("input_hash", "reference_hashes", "tensor_hashes", "layouts", "reference_store"))):
                reasons.append("batch/shared input/reference/result mapping differs")
        canonical, errors = self.canonical_binding(job, result)
        reasons += canonical
        errors = {**errors, "forward_autograd": errors["forward"]}
        for operation, boundaries in self.REQUIRED.items():
            converted = [{**row, "result": operation, "boundary": row["call_boundary"]}
                         for row in rows if row["operation"] == operation]
            reasons += MatrixAudit.check_rows({"precision": job["precision"], "operation": operation, "boundaries": boundaries}, converted, errors)
        if len(rows) != 5:
            reasons.append("missing or duplicate historical five-path cells")
        for row in rows:
            if not row["passed"]:
                if set(row.get("failed_outputs", {})) != {"first", "repeat"}:
                    reasons.append("missing numerical failure tensors")
                for artifact in row.get("failed_outputs", {}).values():
                    if self.hash(artifact["path"]) != artifact["sha256"]:
                        reasons.append("numerical failure tensor changed")
        return {"verified": not reasons, "reasons": sorted(set(reasons)), "accuracy_pass": passed, "paths": len(rows),
                "result": str(result_path), "result_sha256": self.hash(result_path)}


class HistoryController:
    def __init__(self, plan):
        self.plan, self.session = plan, plan.session

    def run(self):
        plan, session = self.plan, self.session
        batched = plan.value["execution_model"] == "same_build_point_batch"
        selected = plan.value["batch_groups"] if batched else plan.value["jobs"]
        audit = HistoryAudit(plan)
        for item in selected:
            jobs = [plan.job(key) for key in item["job_ids"]] if batched else [item]
            if all(audit.record(job)["verified"] for job in jobs):
                continue
            if any((session.root / "accuracy" / job["attempt"]).exists() for job in jobs):
                raise RuntimeError("Partial/error historical attempts must be diagnosed, not restarted")
            plan.verify()
            name = item["batch_id"] if batched else item["attempt"]
            uuid = session.idle_gpu(name, plan.value["gpu"]["physical_index"])
            argv = [session.paths["FASTEQ_PY"], "-B", str(plan.root / "producer/historical_batch.py"),
                    "batch-worker" if batched else "worker", "--plan", str(plan.path),
                    "--batch-id" if batched else "--job-id", item["batch_id"] if batched else item["job_id"]]
            deadline = item["deadline_seconds"] if batched else (120 if item["precision"] == "fp64" else 60)
            terminal = session.runner().run(name, argv, deadline, Path(session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": uuid})
            plan.verify()
            if terminal["status"] not in ("PASS", "FAIL") or not all(audit.record(job)["verified"] for job in jobs):
                raise RuntimeError("Historical worker or audit requires diagnosis")
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "worker", "batch-worker", "run", "audit"))
    parser.add_argument("--session", type=Path)
    parser.add_argument("--cycle")
    parser.add_argument("--group-prefix")
    parser.add_argument("--exclude-group-prefix")
    parser.add_argument("--precision", choices=("tf32", "fp32", "fp64"))
    parser.add_argument("--stage", choices=("minimal", "full"), default="minimal")
    parser.add_argument("--execution-model", choices=("point_workers", "same_build_point_batch"), default="point_workers")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--batch-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "plan":
        HistoryPlan.create(args)
        return 0
    plan = HistoryPlan(args.plan)
    if args.action == "worker":
        job = plan.job(args.job_id)
        return HistoryWorker(plan, job, plan.session.root / "accuracy" / job["attempt"]).run()
    if args.action == "batch-worker":
        return HistoryBatch(args).run()
    if args.action == "run":
        return HistoryController(plan).run()
    report = HistoryAudit(plan).run(args.output)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
