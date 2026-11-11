"""Plan, run bounded chunks, and independently audit the full two-version Luna accuracy matrix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

from cases import CONFIGS, Case, Point, points
from common import Evidence
from luna import LunaAdapter
from luna_probe import LunaProbe
from reference_store import ReferenceStore
from session import Session


class SourceTree:
    @staticmethod
    def manifest(root):
        root = Path(root).resolve()
        files = {str(path.relative_to(root)): Evidence.sha256(path)
                 for path in sorted((root / "src").rglob("*"))
                 if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".so")}
        files["pyproject.toml"] = Evidence.sha256(root / "pyproject.toml")
        return {"root": str(root), "files": files, "source_hash": Evidence.fingerprint(files)}


class RuntimeIdentity:
    FLAGS = ("CC", "CXX", "CUDA_HOME", "PYTORCH_NVCC", "NVCC_CCBIN", "NVCC_PREPEND_FLAGS",
             "NVCC_APPEND_FLAGS", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH",
             "CUBLAS_WORKSPACE_CONFIG", "TORCH_CUDA_ARCH_LIST")

    @classmethod
    def read(cls, session):
        import torch

        environment = {**os.environ, **session.environment()}
        compilers = {}
        for name in (environment.get("CXX", "c++"), "nvcc"):
            executable = shutil.which(name, path=environment["PATH"])
            if executable is None:
                raise FileNotFoundError(name)
            compilers[name] = {"path": str(Path(executable).resolve()),
                               "version": subprocess.check_output([executable, "--version"], text=True, timeout=10)}
        return {"python": sys.executable, "prefix": sys.prefix, "torch": torch.__version__,
                "torch_cuda": torch.version.cuda, "torch_origin": torch.__file__,
                "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
                "compilers": compilers, "flags": {name: environment.get(name) for name in cls.FLAGS}}


class DeadlinePolicy:
    """Lock an accuracy/JIT deadline from preserved per-operation pilot progress."""

    def __init__(self, session):
        self.pilots = {}
        paths = list((session.root / "accuracy").glob("luna*/result.json"))
        paths += list((session.root / "accuracy/luna_matrix").glob("*/jobs/*/attempt[0-9][0-9][0-9][0-9]/raw/result.json"))
        for path in sorted(paths):
            value = json.loads(path.read_text())
            if (value.get("status") not in ("PASS", "ACCURACY_FAIL")
                    or value.get("producer_unchanged") is not True or "ncu_range" in value
                    or value.get("diagnostic_reproduction") is True):
                continue
            version = value["variant"].removeprefix("luna_")
            precision, config = value["point"]["precision"], value["point"]["config"]
            for row in value["operation_boundaries"]:
                if row["boundary"] != "native_public":
                    continue
                key = (version, precision, row["operation"])
                seconds = row["setup_or_JIT_seconds"]
                if key not in self.pilots or seconds > self.pilots[key]["seconds"]:
                    self.pilots[key] = {"seconds": seconds, "config": config,
                                        "path": str(path), "sha256": Evidence.sha256(path)}

    def for_job(self, version, precision, config, operation):
        pilot = self.pilots.get((version, precision, operation))
        baseline = max(45.0, pilot["seconds"] if pilot else 0.0)
        factor = 1.0
        if pilot and pilot["config"] != config and operation in ("grad_w", "grad_x_w"):
            pu, pd = CONFIGS[pilot["config"]]
            cu, cd = CONFIGS[config]
            factor = max(1.0, min(4.25, len(cd) / len(pd) * cu / pu))
        seconds = min(300, max(120, math.ceil(baseline * factor * 1.8 + 30)))
        return {"seconds": seconds, "pilot": pilot, "geometry_factor": factor,
                "rule": "ceil(max(45s,preserved per-operation JIT/setup pilot)*geometry_factor*1.8+30s), bounded 120..300s",
                "timeout_is_not_unsupported": True}


class MatrixPlan:
    CLASSIFICATION = "FULL_25_POINT_LUNA_ACCURACY_NOT_TIMING"
    FILES = ("cases.py", "common.py", "session.py", "protocol.py", "luna.py", "luna_probe.py",
             "reference_store.py", "luna_matrix.py")
    VERSIONS = ("0.1.0", "0.2.0")
    PRECISIONS = ("fp32", "fp64")
    OPERATIONS = ("forward", "grad_x", "grad_w", "grad_x_w")
    BOUNDARIES = LunaAdapter.BOUNDARIES

    @staticmethod
    def device_uuid(value):
        return ReferenceStore.device_uuid(value)

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.value = json.loads(self.path.read_text())
        if Evidence.fingerprint({key: value for key, value in self.value.items() if key != "plan_hash"}) != self.value["plan_hash"]:
            raise ValueError("Matrix plan fingerprint differs")
        self.verify_scope(self.value["jobs"])
        if set(self.value["sources"]) != set(self.VERSIONS) or self.value["reference_math"] != ReferenceStore.MATH:
            raise ValueError("Matrix versions or reference mathematics differ")
        self.verify_producer(self.root / "producer")

    @classmethod
    def verify_scope(cls, jobs):
        expected = {(version, point.config, point.batch, point.precision, operation): point.columns()
                    for version in cls.VERSIONS for point in points() if point.precision in cls.PRECISIONS
                    for operation in cls.OPERATIONS}
        found = [(job["version"], job["point"]["config"], job["point"]["batch"], job["point"]["precision"], job["operation"])
                 for job in jobs]
        if set(found) != set(expected) or len(found) != len(expected) or len({job["job_id"] for job in jobs}) != len(expected):
            raise ValueError("Full two-version 25-point/precision/operation matrix is missing or duplicated")
        if any(job["point"] != expected[key] or job["boundaries"] != list(cls.BOUNDARIES)
               for key, job in zip(found, jobs)):
            raise ValueError("Matrix point seed or boundary scope differs")

    @classmethod
    def cells(cls, job):
        results = LunaProbe.result_kinds(job["operation"])
        return {(job["operation"], result, boundary) for result in results for boundary in cls.BOUNDARIES}

    @classmethod
    def jobs(cls, policy):
        jobs = []
        # Batch-major per shape keeps each source's operation-specific JIT warm.
        for version in cls.VERSIONS:
            for point in [point for point in points() if point.precision in cls.PRECISIONS]:
                for operation in cls.OPERATIONS:
                    key = f"luna{version.replace('.', '')}_{point.config}_b{point.batch}_{point.precision}_{operation}"
                    jobs.append({"job_id": key, "version": version, "point": point.columns(), "operation": operation,
                                 "boundaries": list(cls.BOUNDARIES),
                                 "deadline": policy.for_job(version, point.precision, point.config, operation)})
        return jobs

    @classmethod
    def extra_fields(cls, args, session, value):
        return {}

    @classmethod
    def create(cls, args):
        session = Session(args.session)
        if re.fullmatch(r"[A-Za-z0-9_.-]+", args.cycle) is None:
            raise ValueError("Cycle must be a simple fresh directory name")
        root = session.root / "accuracy/luna_matrix" / args.cycle
        root.mkdir(parents=True, exist_ok=False)
        producer = root / "producer"
        producer.mkdir()
        for name in cls.FILES:
            shutil.copy2(Path(__file__).parent / name, producer / name)
        sources = {"0.1.0": SourceTree.manifest(args.luna010_root),
                   "0.2.0": SourceTree.manifest(Path(session.paths["LUNAEQ_ROOT"]))}
        gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                                           text=True, timeout=10).splitlines()
        gpu = next(row.split(",")[1].strip() for row in gpu_rows if int(row.split(",")[0]) == args.gpu)
        value = {"schema": 1, "classification": cls.CLASSIFICATION, "cycle_id": args.cycle,
                 "session": str(session.root), "created_utc": datetime.now(timezone.utc).isoformat(),
                 "producer": {name: Evidence.sha256(producer / name) for name in cls.FILES},
                 "sources": sources, "runtime": RuntimeIdentity.read(session),
                 "gpu": {"physical_index": args.gpu, "uuid": cls.device_uuid(gpu), "selection_uuid": gpu,
                         "uuid_encoding": "UUID hex; NVML GPU- prefix removed", "planning_does_not_reserve_gpu": True},
                 "reference_store": str(session.root / "references"), "reference_math": ReferenceStore.MATH,
                 "jobs": cls.jobs(DeadlinePolicy(session)),
                 "resume": "same cycle/source/producer/runtime/GPU/input/reference only; complete latest attempts; no minimal-probe data imported"}
        extra = cls.extra_fields(args, session, value)
        if set(extra) & set(value):
            raise ValueError("Supplement metadata cannot replace the canonical plan identity")
        value.update(extra)
        value["plan_hash"] = Evidence.fingerprint(value)
        Evidence.write(root / "plan.json", value)
        print(json.dumps({"plan": str(root / "plan.json"), "jobs": len(value["jobs"]),
                          "result_cells": sum(len(cls.cells(job)) for job in value["jobs"]), "gpu_work_started": False}))
        return cls(root / "plan.json")

    def verify_producer(self, producer):
        if {name: Evidence.sha256(producer / name) for name in self.FILES} != self.value["producer"]:
            raise ValueError("Frozen producer epoch differs")

    def verify_runtime_sources(self, session):
        if RuntimeIdentity.read(session) != self.value["runtime"]:
            raise ValueError("Runtime/compiler/ABI/flags differ from the locked matrix")
        for source in self.value["sources"].values():
            if SourceTree.manifest(source["root"]) != source:
                raise ValueError("Source tree differs from the locked matrix")

    def job(self, job_id):
        return next(job for job in self.value["jobs"] if job["job_id"] == job_id)


class MatrixWorker(LunaProbe):
    PLAN_TYPE = MatrixPlan
    ADAPTER_TYPE = LunaAdapter

    def __init__(self, args):
        self.plan = self.PLAN_TYPE(args.plan)
        self.job = self.plan.job(args.job_id)
        self.store = ReferenceStore(self.plan.value["reference_store"])
        self.reference_receipt = None
        source = self.plan.value["sources"][self.job["version"]]
        self.start_source = SourceTree.manifest(source["root"])
        if self.start_source != source:
            raise ValueError("Worker source differs before CUDA work")
        point = self.job["point"]
        probe = argparse.Namespace(session=Path(self.plan.value["session"]), output=args.output,
                                   source_root=Path(source["root"]), version=self.job["version"],
                                   point=f"{point['config']}/{point['batch']}", precision=point["precision"],
                                   operations=self.job["operation"],
                                   boundaries="native_public" if args.reproduction else ",".join(self.plan.BOUNDARIES),
                                   trace=False, ncu=False)
        super().__init__(probe, reference_factory=self.reference, adapter_factory=self.ADAPTER_TYPE)
        for name in ("reference_store.py", "luna_matrix.py"):
            path = Path(__file__).with_name(name)
            self.producer_files[str(path)] = Evidence.sha256(path)
            shutil.copy2(path, self.output / "producer" / name)
        self.result.update(cycle_id=self.plan.value["cycle_id"], plan_hash=self.plan.value["plan_hash"],
                           job_id=self.job["job_id"], diagnostic_reproduction=args.reproduction,
                           source_manifest_hash=source["source_hash"],
                           reference_factory={"name": "reference_store.ReferenceStore.get",
                                              "source_sha256": Evidence.sha256(Path(__file__).with_name("reference_store.py"))})

    def reference(self, case):
        value = self.store.get(case)
        self.reference_receipt = value.store_receipt
        self.result["reference_store"] = value.store_receipt
        self.flush()
        return value

    def observe_result(self, row, output):
        if row["accuracy_pass"]:
            return
        import torch

        directory = self.output / "failed_outputs"
        directory.mkdir(exist_ok=True)
        path = directory / (row["result"] + "_" + row["output_hash"] + ".pt")
        if not path.exists():
            temporary = path.with_suffix(".partial")
            torch.save(output.detach().cpu(), temporary)
            temporary.replace(path)
        row["diagnostic_output"] = {"path": str(path), "file_sha256": Evidence.sha256(path),
                                    "tensor_sha256": row["output_hash"]}

    def run(self):
        import torch

        self.flush()
        try:
            raw_uuid = str(torch.cuda.get_device_properties().uuid)
            normalized_uuid = MatrixPlan.device_uuid(raw_uuid)
            self.result.update(gpu_uuid_raw=raw_uuid, gpu_uuid_normalized=normalized_uuid)
            if normalized_uuid != self.plan.value["gpu"]["uuid"]:
                raise ValueError("Worker GPU UUID differs from plan")
            self.plan.verify_producer(Path(__file__).parent)
            return super().run()
        except BaseException as error:
            if self.result.get("status") == "RUNNING":
                self.result.update(status="ERROR", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                self.flush()
            raise
        finally:
            unchanged = SourceTree.manifest(self.start_source["root"]) == self.start_source
            self.result["source_unchanged"] = unchanged
            if not unchanged:
                self.result["status"] = "INVALID_SOURCE_CHANGED_DURING_RUN"
            self.flush()


class MatrixAudit:
    PLAN_TYPE = MatrixPlan

    @classmethod
    def check_rows(cls, job, rows):
        reasons = []
        keys = [(row.get("operation"), row.get("result"), row.get("boundary")) for row in rows]
        if set(keys) != cls.PLAN_TYPE.cells(job) or len(keys) != len(cls.PLAN_TYPE.cells(job)):
            reasons.append("missing/duplicate operation-result-boundary")
        for row in rows:
            precision = job["point"]["precision"]
            comparator = row["torch_rel_fro"]
            comparator_valid = comparator is not None and math.isfinite(comparator) and comparator >= 0
            threshold = 1e-12 if precision == "fp64" else 1.05 * comparator if comparator_valid else None
            relative = row["rel_fro"]
            numerical = (relative is not None and math.isfinite(relative) and relative >= 0
                         and threshold is not None and relative <= threshold)
            if row.get("zero_reference") is True:
                numerical = numerical and relative == 0 and row.get("max_abs") == 0
            passed = numerical and all(row.get(field) is True for field in
                                       ("finite", "reference_finite", "repeat_equal", "different_buffer"))
            if row.get("threshold") != threshold or row.get("accuracy_pass") is not passed or row.get("passed") is not passed:
                reasons.append("independent numerical gate mismatch")
            if row.get("torch_baseline_valid") is not comparator_valid:
                reasons.append("invalid strict-FP32 comparator flag")
            if row.get("accuracy_tier") != precision:
                reasons.append("numerical tier differs")
        return reasons

    @staticmethod
    def coverage(expected, records):
        expected = set(expected)
        return {"complete": set(records) == expected and all(value["verified"] for value in records.values()),
                "expected": len(expected), "verified": sum(value["verified"] for value in records.values()),
                "missing": sorted(expected - records.keys()), "extra": sorted(records.keys() - expected)}

    def __init__(self, plan):
        self.plan = plan
        self.checked_reference_files = {}

    def latest(self, job):
        directory = self.plan.root / "jobs" / job["job_id"]
        attempts = sorted(directory.glob("attempt[0-9][0-9][0-9][0-9]")) if directory.exists() else []
        return attempts[-1] if attempts else None

    def verify(self, job, directory):
        try:
            return self.verify_raw(job, directory)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return {"verified": False, "reason": ["invalid evidence: " + type(error).__name__ + ": " + str(error)],
                    "attempt": str(directory)}

    def verify_raw(self, job, directory):
        reasons = []
        receipt_path, result_path = directory / "terminal.json", directory / "raw/result.json"
        if not receipt_path.is_file() or not result_path.is_file():
            return {"verified": False, "reason": ["missing terminal receipt or result"], "attempt": str(directory)}
        receipt = json.loads(receipt_path.read_text())
        result = json.loads(result_path.read_text())
        if (receipt["status"] not in ("PASS", "FAIL") or result.get("status") not in ("PASS", "ACCURACY_FAIL")
                or result.get("diagnostic_reproduction") is not False):
            reasons.append("nonterminal/incomplete/error/reproduction attempt")
        for field, expected in (("cycle_id", self.plan.value["cycle_id"]), ("plan_hash", self.plan.value["plan_hash"]),
                                ("job_id", job["job_id"]), ("point", job["point"]),
                                ("gpu_uuid_normalized", self.plan.value["gpu"]["uuid"]),
                                ("source_manifest_hash", self.plan.value["sources"][job["version"]]["source_hash"])):
            if result.get(field) != expected:
                reasons.append("identity differs: " + field)
        if result.get("producer_unchanged") is not True or result.get("source_unchanged") is not True:
            reasons.append("source/producer epoch changed")
        try:
            self.plan.verify_producer(directory / "producer")
            self.plan.verify_producer(directory / "raw/producer")
        except (OSError, ValueError):
            reasons.append("producer snapshot missing or mixed")
        identity = result.get("identity", {})
        if identity.get("backend") != "cuda_fused" or not identity.get("jit_libraries"):
            reasons.append("fused JIT backend not proved")
        source = self.plan.value["sources"][job["version"]]
        expected_python = {name: digest for name, digest in source["files"].items() if name.endswith(".py")}
        if (identity.get("python_files") != expected_python or identity.get("version") != job["version"]
                or identity.get("package_origin") != str(Path(source["root"]) / "src/__init__.py")
                or identity.get("adapter_sha256") != self.plan.value["producer"]["luna.py"]):
            reasons.append("actual imported Luna/adapter identity differs")
        if not set(identity.get("jit_libraries", {})) <= set(identity.get("mapped_jit_libraries", [])):
            reasons.append("exact JIT libraries were not mapped")
        for path, digest in identity.get("jit_libraries", {}).items():
            if not Path(path).is_file() or Evidence.sha256(Path(path)) != digest:
                reasons.append("DSO bytes differ")
        reference = result.get("reference_store", {})
        if not reference.get("path") or not Path(reference.get("path", "")).is_file():
            reasons.append("canonical reference receipt missing")
        else:
            metadata_path = Path(reference["path"])
            metadata = json.loads(metadata_path.read_text())
            if Evidence.sha256(metadata_path) != reference.get("sha256") or metadata.get("identity", {}).get("input_hash") != result.get("input_hash"):
                reasons.append("reference receipt/input identity differs")
            if metadata.get("identity", {}).get("hardware", {}).get("uuid") != result.get("gpu_uuid_normalized"):
                reasons.append("reference GPU differs")
            if (MatrixPlan.device_uuid(reference.get("observed_gpu_uuid_raw")) != result.get("gpu_uuid_normalized")
                    or MatrixPlan.device_uuid(result.get("gpu_uuid_raw")) != result.get("gpu_uuid_normalized")):
                reasons.append("raw/normalized GPU identity differs")
            if result.get("reference_hashes") != metadata.get("value_hashes", {}).get("fp64"):
                reasons.append("actual gold hashes differ")
            reference_identity = metadata.get("identity", {})
            if (reference_identity.get("reference_producer_sha256") != self.plan.value["producer"]["cases.py"]
                    or reference_identity.get("store_producer_sha256") != self.plan.value["producer"]["reference_store.py"]
                    or reference_identity.get("math") != self.plan.value["reference_math"]
                    or reference_identity.get("point") != job["point"]
                    or reference_identity.get("tensor_hashes") != result.get("tensor_hashes")
                    or reference_identity.get("layouts") != result.get("layouts")
                    or metadata.get("key") != Evidence.fingerprint(reference_identity)):
                reasons.append("reference source/math/tensor/layout identity differs")
            data_path = metadata_path.parent / "references.pt"
            expected_data = metadata.get("file_sha256")
            cache_key = str(data_path), expected_data
            if cache_key not in self.checked_reference_files:
                self.checked_reference_files[cache_key] = data_path.is_file() and Evidence.sha256(data_path) == expected_data
            if not self.checked_reference_files[cache_key]:
                reasons.append("canonical reference file bytes differ")
        reasons += self.check_rows(job, result.get("operation_boundaries", []))
        return {"verified": not reasons, "reason": sorted(set(reasons)), "attempt": str(directory),
                "accuracy_pass": all(row["accuracy_pass"] for row in result.get("operation_boundaries", [])),
                "numerical_rows": len(result.get("operation_boundaries", [])),
                "passed_rows": sum(row["accuracy_pass"] for row in result.get("operation_boundaries", [])),
                "input_hash": result.get("input_hash"), "reference_hashes": result.get("reference_hashes"),
                "result_sha256": Evidence.sha256(result_path)}

    def run(self):
        records = {}
        shared = {}
        for job in self.plan.value["jobs"]:
            attempt = self.latest(job)
            if attempt is None:
                continue
            record = self.verify(job, attempt)
            records[job["job_id"]] = record
            if record["verified"]:
                point = job["point"]
                key = f"{point['config']}/{point['batch']}/{point['precision']}"
                shared.setdefault(key, set()).add(Evidence.fingerprint({"input": record["input_hash"], "reference": record["reference_hashes"]}))
        mismatched = sorted(key for key, values in shared.items() if len(values) != 1)
        report = {"classification": self.plan.value["classification"], "cycle_id": self.plan.value["cycle_id"],
                  "plan_hash": self.plan.value["plan_hash"], **self.coverage((job["job_id"] for job in self.plan.value["jobs"]), records),
                  "jobs": records, "shared_input_reference_mismatches": mismatched,
                  "numerical_rows": sum(record.get("numerical_rows", 0) for record in records.values()),
                  "passed_rows": sum(record.get("passed_rows", 0) for record in records.values()),
                  "auditor_sha256": Evidence.sha256(Path(__file__)),
                  "note": "Numerical FAIL is tested and excluded from qualified rankings; ERROR/timeout/missing is incomplete, never unsupported."}
        report["complete"] = report["complete"] and not mismatched
        Evidence.write(self.plan.root / "audit.json", report)
        return report


class MatrixController:
    PLAN_TYPE = MatrixPlan
    AUDIT_TYPE = MatrixAudit
    ENTRYPOINT = "luna_matrix.py"
    LOG_SCOPE = ""

    def __init__(self, args):
        self.args, self.plan = args, self.PLAN_TYPE(args.plan)
        self.session = Session(Path(self.plan.value["session"]))

    @staticmethod
    def cache_operation(job):
        return job["operation"]

    def log_name(self, job, attempt):
        fields = [self.plan.value["cycle_id"]]
        if self.LOG_SCOPE:
            fields.append(self.LOG_SCOPE)
        return "_".join((*fields, job["job_id"], attempt.name))

    def selected(self):
        args = self.args
        return [job for job in self.plan.value["jobs"]
                if (not args.version or job["version"] == args.version)
                and (not args.precision or job["point"]["precision"] == args.precision)
                and (not args.config or job["point"]["config"] == args.config)
                and (args.batch is None or job["point"]["batch"] == args.batch)
                and (not args.operation or job["operation"] == args.operation)]

    def launch(self, job, attempt, reproduction=False):
        producer = attempt / "producer"
        shutil.copytree(self.plan.root / "producer", producer)
        self.plan.verify_producer(producer)
        name = self.log_name(job, attempt)
        if reproduction:
            name += "_reproduction"
        gpu = self.session.idle_gpu(name, self.plan.value["gpu"]["physical_index"])
        if MatrixPlan.device_uuid(gpu) != self.plan.value["gpu"]["uuid"]:
            raise ValueError("Selected GPU differs from locked matrix")
        source = self.plan.value["sources"][job["version"]]
        cache = self.session.root / "cache/luna_matrix" / source["source_hash"] / self.plan.value["producer"]["luna.py"] / job["point"]["precision"] / self.cache_operation(job)
        argv = [self.session.paths["LUNAEQ_PY"], "-B", str(producer / self.ENTRYPOINT), "worker",
                "--plan", str(self.plan.path), "--job-id", job["job_id"], "--output", str(attempt / "raw")]
        if reproduction:
            argv.append("--reproduction")
        # The inner GNU timeout survives a lost coordinator; every worker has
        # a deadline from startup, in addition to CommandRunner's own watchdog.
        deadline = job["deadline"]["seconds"]
        argv = ["timeout", "--signal=INT", "--kill-after=5s", str(deadline) + "s", *argv]
        receipt = self.session.runner().run(name, argv, deadline + 10,
                                            Path(self.session.paths["FASTEQ_ROOT"]),
                                            {"CUDA_VISIBLE_DEVICES": gpu, "LUNAEQ_CACHE_DIR": str(cache),
                                             "LUNATP_CUTLASS_INCLUDE": str(Path(source["root"]) / "src/third_party/cutlass/include")})
        Evidence.write(attempt / "terminal.json", receipt)
        return receipt

    def run(self):
        args = self.args
        self.plan.verify_producer(Path(__file__).parent)
        self.plan.verify_runtime_sources(self.session)
        start = time.monotonic()
        audit = self.AUDIT_TYPE(self.plan)
        control = {"status": "RUNNING", "pid": os.getpid(), "start_utc": datetime.now(timezone.utc).isoformat(),
                   "plan_hash": self.plan.value["plan_hash"], "jobs": [], "max_jobs": args.max_jobs,
                   "batch_deadline_seconds": args.batch_deadline}
        with (self.plan.root / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            running = self.plan.root / "controller.running.json"
            if running.exists():
                previous = json.loads(running.read_text())
                if Path(f"/proc/{previous['pid']}").exists():
                    raise RuntimeError("A prior controller PID is still live; inspect it before resuming")
                running.rename(self.plan.root / f"controller.abandoned.{time.time_ns()}.json")
            Evidence.write(running, control)
            try:
                for job in self.selected():
                    previous = audit.latest(job)
                    if previous is not None:
                        if audit.verify(job, previous)["verified"]:
                            continue
                        if not args.retry_incomplete:
                            control.update(status="INCOMPLETE_REVIEW_REQUIRED", stopped_at=job["job_id"])
                            break
                    if len(control["jobs"]) >= args.max_jobs:
                        control["status"] = "BOUNDED_CHUNK_FINISHED"
                        break
                    if time.monotonic() - start + job["deadline"]["seconds"] + 15 > args.batch_deadline:
                        control["status"] = "BATCH_BUDGET_FINISHED"
                        break
                    directory = self.plan.root / "jobs" / job["job_id"]
                    number = len(list(directory.glob("attempt[0-9][0-9][0-9][0-9]"))) + 1
                    attempt = directory / f"attempt{number:04d}"
                    attempt.mkdir(parents=True)
                    receipt = self.launch(job, attempt)
                    record = audit.verify(job, attempt)
                    control["jobs"].append({"job_id": job["job_id"], "attempt": str(attempt),
                                            "process_status": receipt["status"], **record})
                    Evidence.write(running, control)
                    if not record["verified"]:
                        # One exact case/operation public boundary is the minimal fresh-process reproduction.
                        if receipt["status"] == "FAIL":
                            reproduction = attempt / "reproduction"
                            reproduction.mkdir()
                            self.launch(job, reproduction, reproduction=True)
                        control.update(status="INCOMPLETE_DIAGNOSIS_REQUIRED", stopped_at=job["job_id"])
                        break
                else:
                    control["status"] = "SELECTED_CHUNK_FINISHED"
            except BaseException as error:
                control.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "INCOMPLETE_ERROR",
                               error_type=type(error).__name__, error=str(error))
                raise
            finally:
                control["elapsed_seconds"] = time.monotonic() - start
                Evidence.write(self.plan.root / f"controller.{time.time_ns()}.json", control)
                running.unlink(missing_ok=True)
                report = audit.run()
                print(json.dumps({"controller_status": control["status"], "chunk_jobs": len(control["jobs"]),
                                  "matrix_complete": report["complete"], "verified": report["verified"],
                                  "missing": len(report["missing"])}), flush=True)
        return 0 if control["status"] in ("SELECTED_CHUNK_FINISHED", "BOUNDED_CHUNK_FINISHED", "BATCH_BUDGET_FINISHED") else 1


class CpuSelfcheck:
    @classmethod
    def run(cls, output):
        import torch
        from types import SimpleNamespace

        output.mkdir(parents=True, exist_ok=False)
        policy = type("SyntheticPolicy", (), {"for_job": lambda self, *args: {"seconds": 120}})()
        jobs = MatrixPlan.jobs(policy)
        assert len(jobs) == 400 and len({job["job_id"] for job in jobs}) == 400
        assert sum(len(MatrixPlan.cells(job)) for job in jobs) == 1500
        assert MatrixPlan.device_uuid("GPU-a383162c-b587-39dd-b5d0-c666f4e011ab") == MatrixPlan.device_uuid("a383162c-b587-39dd-b5d0-c666f4e011ab")
        assert MatrixPlan.device_uuid("GPU-ABC") == MatrixPlan.device_uuid("abc")
        assert MatrixPlan.device_uuid("GPU-A") != MatrixPlan.device_uuid("GPU-B")
        MatrixPlan.verify_scope(jobs)
        try:
            MatrixPlan.verify_scope(jobs[:-1])
        except ValueError:
            pass
        else:
            raise AssertionError("Narrowed full matrix accepted")
        assert not MatrixAudit.coverage((job["job_id"] for job in jobs), {jobs[0]["job_id"]: {"verified": True}})["complete"]
        rows = []
        for operation, result, boundary in MatrixPlan.cells(jobs[0]):
            rows.append({"operation": operation, "result": result, "boundary": boundary, "rel_fro": 1e-7,
                         "torch_rel_fro": 1e-7, "threshold": 1.05e-7, "accuracy_pass": True, "passed": True,
                         "finite": True, "reference_finite": True, "torch_baseline_valid": True,
                         "zero_reference": False, "max_abs": 1e-7,
                         "repeat_equal": True, "different_buffer": True, "accuracy_tier": "fp32"})
        assert MatrixAudit.check_rows(jobs[0], rows) == []
        assert MatrixAudit.check_rows(jobs[0], rows + rows[:1])
        rows[0]["threshold"] *= 2
        assert MatrixAudit.check_rows(jobs[0], rows)
        # Mixed batch sizes leave both index-width entries in a shared Luna
        # config. Prepared dispatch must select the exact key, never first().
        forward = lambda *args: None
        backward = lambda *args: None
        keys = {(1, (True, False, False), ("weight", "grad_out"), "90", bits, False, torch.float32): launch
                for bits, launch in ((32, forward), (64, backward))}
        adapter = LunaAdapter.__new__(LunaAdapter)
        adapter.source = SimpleNamespace(kernels=SimpleNamespace(TASK_ALIGNMENT=32, get_cuda_arch=lambda _: "90",
                                        run_kernel=lambda *args, **kwargs: None,
                                        select_program=lambda *args, **kwargs: {"inputs": ("weight", "grad_out"), "weight_terms": 1}))
        adapter.config = SimpleNamespace(index_batch_span=1_000_000, index_workspace_span=1,
                                         index_bits=32, meta={"bias_numel": 0}, kernels=keys)
        adapter.batch, adapter.dtype, adapter.device = 3000, torch.float32, "cpu"
        adapter.x_native = adapter.w_native = adapter.dy_native = object()
        adapter._prepared, adapter._prepared_keys = {}, {}
        adapter._prepare_launch("grad_x")
        assert adapter._prepared["grad_x"][0] is backward
        rows[0].update(threshold=None, torch_rel_fro=None, torch_baseline_valid=False, zero_reference=True,
                       rel_fro=None, max_abs=1.0)
        assert MatrixAudit.check_rows(jobs[0], rows)
        case = Case(Point("SMALL_P1", 2, "fp32", 0, 0), device="cpu")
        store = ReferenceStore(output / "reference_cache")
        first, second = store.get(case), store.get(case)
        assert first.hashes == second.hashes and first.store_receipt["created"] and not second.store_receipt["created"]
        assert all(torch.equal(first.values[format_name][kind], second.values[format_name][kind])
                   for format_name in ReferenceStore.FORMATS for kind in ReferenceStore.RESULTS)
        identity = ReferenceStore.identity(case)
        altered = {**identity, "math": {**identity["math"], "matmul_allow_tf32": True}}
        assert Evidence.fingerprint(identity) != Evidence.fingerprint(altered)
        altered = {**identity, "hardware": {"type": "cuda", "uuid": "different-device"}}
        assert Evidence.fingerprint(identity) != Evidence.fingerprint(altered)
        altered = {**identity, "reference_producer_sha256": "different-source"}
        assert Evidence.fingerprint(identity) != Evidence.fingerprint(altered)
        metadata = Path(first.store_receipt["path"])
        payload = Path(first.store_receipt["data_path"])
        with payload.open("ab") as stream:
            stream.write(b"intentional CPU-only integrity counterexample")
        try:
            store.get(case)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt reference cache accepted")
        value_store = ReferenceStore(output / "value_hash_cache")
        cached = value_store.get(case)
        value_path = Path(cached.store_receipt["data_path"])
        metadata_path = Path(cached.store_receipt["path"])
        value = torch.load(value_path, map_location="cpu", weights_only=True)
        value["values"]["fp64"]["forward"][0, 0] += 1
        torch.save(value, value_path)
        metadata = json.loads(metadata_path.read_text())
        metadata["file_sha256"] = Evidence.sha256(value_path)
        Evidence.write(metadata_path, metadata)
        try:
            value_store.get(case)
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong tensor values accepted after file-SHA rewrite")
        partial_store = ReferenceStore(output / "partial_cache")
        partial = partial_store.root / Evidence.fingerprint(identity)
        partial.mkdir()
        (partial / "references.pt").write_bytes(b"incomplete CPU fixture")
        try:
            partial_store.get(case)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Partial reference committed without receipt")
        Evidence.write(output / "result.json", {"classification": "CPU_SYNTHETIC_SELFTEST_NOT_H20_MEASUREMENTS",
                                               "passed": True, "full_plan_jobs": len(jobs), "full_plan_cells": 1500,
                                               "checks": ["full_scope", "NVML/Torch_UUID_normalization", "mixed_index32_index64_exact_kernel_key",
                                                          "narrowed_plan_rejected", "partial_not_complete", "duplicate_cells", "gate_tampering",
                                                          "canonical_reference_roundtrip", "math/GPU/source_identity_split",
                                                          "corrupt_file_rejected", "wrong_value_hash_rejected", "partial_cache_rejected"]})
        print(json.dumps({"passed": True, "jobs": len(jobs), "result_cells": 1500}))
        return 0


class Cli:
    PLAN_TYPE = MatrixPlan
    CONTROLLER_TYPE = MatrixController
    WORKER_TYPE = MatrixWorker
    AUDIT_TYPE = MatrixAudit
    SELF_CHECK_TYPE = CpuSelfcheck

    @classmethod
    def configure_plan(cls, parser):
        pass

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        plan = commands.add_parser("plan", help="CPU discovery and immutable complete matrix plan; no GPU work")
        plan.add_argument("--session", type=Path, required=True)
        plan.add_argument("--cycle", required=True)
        plan.add_argument("--luna010-root", type=Path, required=True)
        plan.add_argument("--gpu", type=int, required=True)
        cls.configure_plan(plan)
        run = commands.add_parser("run", help="Run only a bounded filtered chunk of one full cycle")
        run.add_argument("--plan", type=Path, required=True)
        run.add_argument("--max-jobs", type=int, default=4)
        run.add_argument("--batch-deadline", type=float, default=1200)
        run.add_argument("--version", choices=cls.PLAN_TYPE.VERSIONS)
        run.add_argument("--precision", choices=cls.PLAN_TYPE.PRECISIONS)
        run.add_argument("--config", choices=tuple(CONFIGS))
        run.add_argument("--batch", type=int)
        run.add_argument("--operation", choices=cls.PLAN_TYPE.OPERATIONS)
        run.add_argument("--retry-incomplete", action="store_true", help="Explicitly create a new attempt after reviewing preserved failure")
        worker = commands.add_parser("worker", help=argparse.SUPPRESS)
        worker.add_argument("--plan", type=Path, required=True)
        worker.add_argument("--job-id", required=True)
        worker.add_argument("--output", type=Path, required=True)
        worker.add_argument("--reproduction", action="store_true")
        audit = commands.add_parser("audit", help="Independent complete-set audit; never merges other cycles")
        audit.add_argument("--plan", type=Path, required=True)
        check = commands.add_parser("selfcheck", help="CPU-only scheduler/reference-cache counterexamples")
        check.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        if args.command == "plan":
            cls.PLAN_TYPE.create(args)
            return 0
        if args.command == "run":
            if args.max_jobs < 1 or args.batch_deadline <= 0:
                parser.error("Positive bounded job count/deadline required")
            return cls.CONTROLLER_TYPE(args).run()
        if args.command == "worker":
            return cls.WORKER_TYPE(args).run()
        if args.command == "audit":
            report = cls.AUDIT_TYPE(cls.PLAN_TYPE(args.plan)).run()
            print(json.dumps({"complete": report["complete"], "verified": report["verified"], "missing": len(report["missing"])}))
            return 0 if report["complete"] else 1
        return cls.SELF_CHECK_TYPE.run(args.output)


if __name__ == "__main__":
    raise SystemExit(Cli.main())
