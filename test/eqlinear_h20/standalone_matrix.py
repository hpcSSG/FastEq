"""Frozen standalone registry matrix; reuse the common chunk and probe loops."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

from cases import Case, CONFIGS, points
from common import Evidence
from current_matrix import AccuracyChunk
from current_probe import CurrentProbe
from reference_store import ReferenceStore
from session import Session
from standalone import StandaloneAdapter, StandaloneSource
from standalone_probe import StandaloneProbe


class VariantSpecs:
    FAMILIES = {"standalone_tf32": "tf32", "standalone_fp32": "fp32", "standalone_fp64": "fp64"}
    OPERATIONS = {"forward": "forward", "dx": "grad_x", "dw": "grad_w", "dx+dw": "grad_x_w"}
    EXPECTED = {"tf32": {0, 3, 4, 5, 6, 7, 8}, "fp32": set(range(4)), "fp64": set(range(8))}

    @classmethod
    def from_registry(cls, registry, additional=None):
        result = []
        for row in registry["variants"]:
            if row["family"] not in cls.FAMILIES or row["scope_status"] != "IN_SCOPE":
                continue
            precision = cls.FAMILIES[row["family"]]
            operation = cls.OPERATIONS[row["operation"]]
            impl, bimpl = 0, 1
            if operation == "forward":
                impl = int(row["impl_id"])
                if impl not in cls.EXPECTED[precision]:
                    raise ValueError("Unregistered or excluded forward selector")
            else:
                bimpl = {"grad_x": 1, "grad_w": 2, "grad_x_w": 3}[operation]
            result.append({"variant_id": row["variant_id"], "precision": precision, "operation": operation,
                           "impl": impl, "bimpl": bimpl,
                           "boundaries": list(StandaloneAdapter.BOUNDARIES) if operation == "forward" else ["run_with_activation_prep"],
                           "registry_row": row})
        for precision, expected in cls.EXPECTED.items():
            selected = [item for item in result if item["precision"] == precision]
            if {item["impl"] for item in selected if item["operation"] == "forward"} != expected:
                raise ValueError("Forward registry scope incomplete")
            if {item["operation"] for item in selected if item["operation"] != "forward"} != {"grad_x", "grad_w", "grad_x_w"}:
                raise ValueError("Backward registry scope incomplete")
        if len(result) != 28 or len({item["variant_id"] for item in result}) != 28:
            raise ValueError("Additional or duplicate registry contracts need explicit adapter support")
        if additional is not None:
            original = {row["variant_id"]: row for row in result}
            for row in additional["variants"]:
                if row["base_variant_id"] not in original or row["scope_status"] != "IN_SCOPE":
                    raise ValueError("Named policy does not refer to an in-scope base contract")
                variant = copy.deepcopy(original[row["base_variant_id"]])
                runtime, api = row["runtime_controls"], row["api_controls"]
                if any(re.fullmatch(r"EQLIN_[A-Z0-9_]+", name) is None or not isinstance(value, str) for name, value in runtime.items()):
                    raise ValueError("Unknown runtime control encoding")
                if api and (variant["precision"] != "fp64" or variant["operation"] not in ("grad_w", "grad_x_w")):
                    raise ValueError("Named wgrad API policy applies only to FP64 weight/both gradients")
                variant.update(variant_id=row["variant_id"], registry_row=row, runtime_controls=runtime,
                               api_controls=api, activation_layout="pbu" if runtime.get("EQLIN_A_LAYOUT") == "pbu" else "biu")
                if "probe_point" in row:
                    variant["probe_point"] = row["probe_point"]
                if row.get("build_profile"):
                    if re.fullmatch(r"[A-Za-z0-9_.-]+", row["build_profile"]) is None:
                        raise ValueError("Unsafe compiler profile identifier")
                    variant["build_profile"] = row["build_profile"]
                result.append(variant)
            if len(result) != len({row["variant_id"] for row in result}):
                raise ValueError("Named policies duplicate a registry contract")
        return result


class StandalonePlan:
    FILES = ("standalone_matrix.py", "standalone_probe.py", "standalone.py", "current_matrix.py", "current.py", "current_probe.py", "cases.py",
             "common.py", "protocol.py", "session.py", "reference_store.py")

    @staticmethod
    def uuid(value):
        return str(value).removeprefix("GPU-").lower()

    @staticmethod
    def runtime():
        import torch
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") not in (None, "0"):
            raise ValueError("Forced TF32 cuBLAS override is forbidden by this scope")
        return {"python": sys.executable, "prefix": sys.prefix, "torch": torch.__version__,
                "torch_origin": torch.__file__, "torch_cuda": torch.version.cuda,
                "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
                "math": ReferenceStore.MATH, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "math_environment": {key: os.environ.get(key) for key in
                                     ("CUBLAS_WORKSPACE_CONFIG", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")},
                "runtime_controls": "all inherited EQLIN_* removed before DSO load"}

    @staticmethod
    def cells(job):
        kinds = ("grad_x", "grad_w") if job["operation"] == "grad_x_w" else (job["operation"],)
        return {(job["operation"], kind, boundary) for kind in kinds for boundary in job["boundaries"]}

    @staticmethod
    def batch_groups(value):
        groups = {}
        for job in value["jobs"]:
            point = job["point"]
            key = (job["precision"], point["config"], point["batch"])
            if job.get("build_profile"):
                key += (job["build_profile"],)
            group = groups.setdefault(key, {"batch_id": "standalone_batch_" + "_".join(map(str, key)) + "_" + value["cycle"],
                "precision": job["precision"], "point": point, "job_ids": [],
                "deadline_seconds": 180 if job["precision"] == "fp64" else 120})
            group["job_ids"].append(job["job_id"])
            if job.get("build_profile"):
                group["build_profile"] = job["build_profile"]
        if value.get("variant_scope") == "named" and value["stage"] == "full":
            for group in groups.values():
                group["deadline_seconds"] = max(group["deadline_seconds"], 10 * len(group["job_ids"]))
        return list(groups.values())

    @classmethod
    def expected_jobs(cls, variants, stage):
        jobs = []
        for variant in variants:
            for point in points():
                if point.precision != variant["precision"]:
                    continue
                probe = variant.get("probe_point", {"config": "SMALL_P1", "batch": 512})
                if stage != "full" and (point.config != probe["config"] or point.batch != probe["batch"] or (stage == "minimal" and variant["operation"] != "forward")):
                    continue
                jobs.append({key: value for key, value in variant.items() if key != "registry_row"} | {
                    "job_id": variant["variant_id"] + "_" + point.config + "_" + str(point.batch),
                    "point": point.columns(), "boundaries": list(variant["boundaries"]),
                    "deadline_seconds": 120 if point.precision == "fp64" else 60,
                    "diagnostic_reproduction": False})
        return jobs

    @classmethod
    def verify_scope(cls, value):
        expected = cls.expected_jobs(value["variants"], value["stage"])
        if value["jobs"] != expected or len({row["job_id"] for row in value["jobs"]}) != len(expected):
            raise ValueError("Missing/duplicate/changed matrix selector, point, seed, operation or boundary")
        if value.get("execution_model") == "same_build_point_batch" and value.get("batch_groups") != cls.batch_groups(value):
            raise ValueError("Batch process-to-job mapping changed")

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.value = json.loads(self.path.read_text())
        if Evidence.fingerprint({key: value for key, value in self.value.items() if key != "plan_hash"}) != self.value["plan_hash"]:
            raise ValueError("Matrix plan fingerprint changed")
        registry = Path(self.value["registry"]["path"])
        if Evidence.sha256(registry) != self.value["registry"]["sha256"]:
            raise ValueError("Locked registry changed")
        additional = None
        if self.value.get("additional_registry"):
            metadata = self.value["additional_registry"]
            path = Path(metadata["path"])
            if Evidence.sha256(path) != metadata["sha256"]:
                raise ValueError("Additional named policy registry changed")
            additional = json.loads(path.read_text())
        variants = VariantSpecs.from_registry(json.loads(registry.read_text()), additional)
        if self.value.get("variant_scope") == "named":
            variants = [row for row in variants if "base_variant_id" in row["registry_row"]]
        if variants != self.value["variants"]:
            raise ValueError("Matrix variants do not match the actual registry")
        if Evidence.sha256(Path(self.value["session"]) / "config/scope.yaml") != self.value["scope_sha256"]:
            raise ValueError("User scope changed")
        self.verify_scope(self.value)
        self.verify_producer(self.root / "producer")

    def verify_producer(self, producer):
        if {name: Evidence.sha256(producer / name) for name in self.FILES} != self.value["producer"]:
            raise ValueError("Standalone producer differs from frozen plan")

    def verify_sources(self, session):
        if self.runtime() != self.value["runtime"]:
            raise ValueError("Standalone runtime/ABI/reference mathematics changed")
        sources = [(precision, source) for precision, source in self.value["sources"].items()]
        sources += [(source["precision"], source) for source in self.value.get("build_profiles", {}).values()]
        for precision, source in sources:
            current = StandaloneSource(session, precision)
            if current.stage() != source["source_manifest"] or Evidence.sha256(Path(source["binary"])) != source["binary_hash"]:
                raise ValueError("Standalone source/header/DSO identity changed")
            for field in ("build_manifest", "cmake_cache", "ninja"):
                row = source[field]
                if Evidence.sha256(Path(row["path"])) != row["sha256"]:
                    raise ValueError("Standalone build provenance changed")

    def source_record(self, selection):
        return self.value["build_profiles"][selection["build_profile"]] if selection.get("build_profile") else self.value["sources"][selection["precision"]]

    def job(self, job_id):
        return next(row for row in self.value["jobs"] if row["job_id"] == job_id)

    @classmethod
    def create(cls, args):
        session = Session(args.session)
        if re.fullmatch(r"[A-Za-z0-9_.-]+", args.cycle) is None:
            raise ValueError("Use a simple fresh cycle id")
        root = session.root / "accuracy/standalone_matrix" / args.cycle
        root.mkdir(parents=True, exist_ok=False)
        producer = root / "producer"
        producer.mkdir()
        for name in cls.FILES:
            shutil.copy2(Path(__file__).with_name(name), producer / name)
        registry_path = root / "registry.json"
        shutil.copy2(session.root / "coverage/variants.json", registry_path)
        registry = json.loads(registry_path.read_text())
        additional_path = root / "policies.json" if args.additional_registry else None
        if additional_path:
            shutil.copy2(args.additional_registry, additional_path)
        additional = json.loads(additional_path.read_text()) if additional_path else None
        variants = VariantSpecs.from_registry(registry, additional)
        if args.variant_scope == "named":
            if additional is None:
                raise ValueError("Named-only scope requires its actual additional registry")
            variants = [row for row in variants if "base_variant_id" in row["registry_row"]]
        sources = {}
        for precision in StandaloneSource.CONFIG:
            source = StandaloneSource(session, precision)
            build_path = session.root / "manifests" / ("standalone_" + precision + "_01_build.json")
            build = json.loads(build_path.read_text())
            if build["binary_hash"] != Evidence.sha256(source.library):
                raise ValueError("Fresh build manifest does not match the DSO")
            sources[precision] = {"source_manifest": source.stage(), "binary": str(source.library),
                "binary_hash": build["binary_hash"],
                "build_manifest": {"path": str(build_path), "sha256": Evidence.sha256(build_path)},
                "cmake_cache": {"path": str(source.build / "CMakeCache.txt"), "sha256": Evidence.sha256(source.build / "CMakeCache.txt")},
                "ninja": {"path": str(source.build / "build.ninja"), "sha256": Evidence.sha256(source.build / "build.ninja")}}
        profiles = {}
        profile_rows = {row["registry_row"]["build_profile"]: row["registry_row"] for row in variants if row.get("build_profile")}
        for name, row in profile_rows.items():
            original = Path(row["profile_record"]["path"])
            if Evidence.sha256(original) != row["profile_record"]["sha256"]:
                raise ValueError("Documented profile build record changed")
            record = json.loads(original.read_text())
            if record["status"] != "BUILD_PASS_REQUIRES_H20_PROBE":
                raise ValueError("A profile without a fresh successful build is not executable")
            target = root / "profiles" / (name + ".json")
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(original, target)
            profiles[name] = {key: record[key] for key in ("source_manifest", "binary", "binary_hash", "cmake_cache", "ninja", "macro_definitions")}
            profiles[name].update(precision=record["source_recipe"]["precision"],
                build_manifest={"path": str(target), "sha256": Evidence.sha256(target)})
        import subprocess
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True, timeout=10)
        gpu = next(line.split(",")[1].strip() for line in output.splitlines() if int(line.split(",")[0]) == args.gpu)
        value = {"schema": 1, "classification": "STANDALONE_ACCURACY_NOT_TIMING", "cycle": args.cycle,
            "session": str(session.root), "stage": args.stage, "created_utc": datetime.now(timezone.utc).isoformat(),
            "registry": {"path": str(registry_path), "sha256": Evidence.sha256(registry_path)},
            "scope_sha256": Evidence.sha256(session.root / "config/scope.yaml"),
            "variants": variants, "sources": sources, "runtime": cls.runtime(),
            "producer": {name: Evidence.sha256(producer / name) for name in cls.FILES},
            "gpu": {"physical_index": args.gpu, "selection_uuid": gpu, "uuid": cls.uuid(gpu), "plan_is_not_gpu_reservation": True},
            "jobs": cls.expected_jobs(variants, args.stage), "reference_root": str(session.root / "references"),
            "additional_existing_controls": cls.control_audit(session),
            "all_existing_parameter_strategies_covered": False,
            "note": "Default-control enum registry matrix only; control audit preserves additional existing policies for scope mapping. No claim of a parameter Cartesian-product sweep."}
        if profiles:
            value["build_profiles"] = profiles
        if args.variant_scope == "named":
            value["variant_scope"] = "named"
        if args.execution_model == "same_build_point_batch":
            value["execution_model"] = args.execution_model
            value["batch_groups"] = cls.batch_groups(value)
        value["plan_hash"] = Evidence.fingerprint(value)
        if args.additional_registry:
            value["additional_registry"] = {"path": str(additional_path), "sha256": Evidence.sha256(additional_path),
                                            "origin": str(args.additional_registry.resolve())}
            value["plan_hash"] = Evidence.fingerprint({key: item for key, item in value.items() if key != "plan_hash"})
        Evidence.write(root / "plan.json", value)
        print(json.dumps({"plan": str(root / "plan.json"), "jobs": len(value["jobs"]),
                          "numerical_cells": sum(len(cls.cells(row)) for row in value["jobs"]), "GPU_work_started": False}))
        return cls(root / "plan.json")

    @staticmethod
    def control_audit(session):
        rows = []
        for precision in StandaloneSource.CONFIG:
            source = StandaloneSource(session, precision)
            for path in sorted((source.original / "csrc").rglob("*")):
                if not path.is_file() or path.suffix not in (".cpp", ".cu", ".h", ".hpp"):
                    continue
                for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    for name in re.findall(r'getenv\s*\(\s*"(EQLIN_[A-Z0-9_]+)"', line):
                        rows.append({"precision": precision, "name": name, "source": str(path), "line": line_number,
                                     "text": line.strip(), "status": "DISCOVERED_NOT_EXPANDED_IN_DEFAULT_ENUM_MATRIX",
                                     "role": "diagnostic_only" if name in ("EQLIN_DEBUG", "EQLIN_WGRAD_PREP_ONLY") else "existing_runtime_policy"})
        return rows


class MatrixWorker(StandaloneProbe):
    def __init__(self, args, shared=None):
        self.shared = shared
        self.plan = shared.plan if shared else StandalonePlan(args.plan)
        self.job = self.plan.job(args.job_id)
        session = Session(Path(self.plan.value["session"]))
        if shared is None:
            self.plan.verify_sources(session)
        self.cleared_controls = {name: os.environ.pop(name) for name in list(os.environ) if name.startswith("EQLIN_")}
        os.environ.update(self.job.get("runtime_controls", {}))
        import torch
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        actual_uuid = self.plan.uuid(torch.cuda.get_device_properties().uuid)
        if actual_uuid != self.plan.value["gpu"]["uuid"]:
            raise ValueError("Worker GPU differs from locked matrix")
        self.store = ReferenceStore(Path(self.plan.value["reference_root"]))
        point = self.job["point"]
        boundaries = [args.diagnostic_boundary] if args.diagnostic_boundary else self.job["boundaries"]
        probe = argparse.Namespace(session=session.root, output=args.output, precision=self.job["precision"],
            impl=self.job["impl"], bimpl=self.job["bimpl"], point=f"{point['config']}/{point['batch']}",
            operations=self.job["operation"], boundaries=",".join(boundaries), trace=self.plan.value["stage"] == "minimal")
        super().__init__(probe, reference_factory=self.reference, adapter_factory=self.make_adapter,
                         case_factory=shared.case_factory if shared else Case)
        for name in self.plan.FILES:
            path = Path(__file__).with_name(name)
            self.producer_files[str(path)] = Evidence.sha256(path)
            shutil.copy2(path, self.output / "producer_sources" / name)
        self.result.update(plan_hash=self.plan.value["plan_hash"], cycle=self.plan.value["cycle"], job_id=self.job["job_id"],
            variant_id=self.job["variant_id"], diagnostic_reproduction=bool(args.diagnostic_boundary),
            gpu_uuid=actual_uuid, cleared_inherited_controls=self.cleared_controls,
            intended_runtime_controls=self.job.get("runtime_controls", {}), selected_boundaries=boundaries,
            actual_math_flags={"float32_matmul_precision": torch.get_float32_matmul_precision(),
                               "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32})
        if shared:
            self.result["shared_batch"] = {"batch_id": shared.group["batch_id"], "receipt": str(shared.receipt),
                "variant_index": shared.group["job_ids"].index(self.job["job_id"])}

    def make_adapter(self, case, source):
        return StandaloneAdapter(case, self.plan.source_record(self.job)["binary"], self.job["precision"], self.job["impl"], self.job["bimpl"],
            source.manifest_path, api_controls=self.job.get("api_controls"), activation_layout=self.job.get("activation_layout", "biu"))

    def reference(self, case):
        self.case = case
        value = self.shared.reference if self.shared else self.store.get(case)
        self.result["reference_store"] = value.store_receipt
        self.flush()
        return value

    def observe_result(self, row, output, repeated):
        if self.case.point.precision == "tf32" and row["result"] in ("forward", "grad_x"):
            helper = object.__new__(CurrentProbe)
            helper.session = self.session
            boundary = "public_warm" if row["boundary"] == "forward_allocating" else "run_with_activation_prep"
            row["frozen_p14"] = helper.frozen(self.case, row["result"], boundary, row["rel_fro"])
            if row["frozen_p14"]["status"] == "FAIL":
                row["base_accuracy_pass"] = row["passed"]
                row["passed"] = False
                row["reason"] = "FROZEN_P14_ADDITIONAL_GATE_FAILED"
        else:
            row["frozen_p14"] = {"status": "NOT_APPLICABLE" if self.case.point.precision != "tf32" else "UNVERIFIED_NO_HISTORICAL_GRAD_W"}
        if row["passed"]:
            return
        import torch
        root = self.output / "failed_outputs"
        root.mkdir(exist_ok=True)
        artifacts = {}
        for name, value in (("first", output), ("repeat", repeated)):
            target = root / (row["operation"] + "_" + row["result"] + "_" + name + "_" + self.adapter.case.tensor_hash(value) + ".pt")
            if not target.exists():
                temporary = target.with_suffix(".partial")
                torch.save(value.detach().cpu(), temporary)
                temporary.replace(target)
            artifacts[name] = {"path": str(target), "sha256": Evidence.sha256(target), "tensor_hash": self.adapter.case.tensor_hash(value)}
        row["failed_outputs"] = artifacts

    def run(self):
        result = super().run()
        if self.shared is None:
            self.plan.verify_sources(self.session)
        self.result["source_unchanged"] = True
        Evidence.write(self.output / "result.json", self.result)
        return result


class SharedPointBatch:
    """One process, one immutable DSO, one actual fixture/reference allocation."""

    PLAN_CLASS = StandalonePlan
    CLASSIFICATION = "STANDALONE_ACCURACY_NOT_TIMING"

    def __init__(self, args):
        self.args, self.plan = args, self.PLAN_CLASS(args.plan)
        self.session = Session(Path(self.plan.value["session"]))
        if self.plan.value.get("execution_model") != "same_build_point_batch":
            raise ValueError("Batch worker requires a batch-bound plan")
        self.group = next(row for row in self.plan.value["batch_groups"] if row["batch_id"] == args.batch_id)
        self.receipt = self.plan.root / "batches" / self.group["batch_id"] / "receipt.json"
        self.receipt.parent.mkdir(parents=True, exist_ok=False)
        self.case = self.reference = None

    def create_worker(self, args):
        raise NotImplementedError("A backend supplies its probe without duplicating fixture/reference/gate loops")

    def source_build(self, precision):
        return self.plan.source_record(self.group) if hasattr(self.plan, "source_record") else self.plan.value["sources"][precision]

    def case_factory(self, point, device, fixture_root):
        if point.columns() != self.group["point"] or device != "cuda" or Path(fixture_root) != self.session.root / "fixtures":
            raise ValueError("Batch attempted to substitute a different fixture")
        return self.case

    @staticmethod
    def tensors(case, reference):
        return {"input": {name + "_sha256": case.tensor_hash(getattr(case, name)) for name in ("x", "w", "grad")},
                "reference": ReferenceStore.tensor_hashes(reference.values)}

    def run(self):
        import torch
        value = {"status": "RUNNING", "plan_hash": self.plan.value["plan_hash"], "batch_id": self.group["batch_id"],
                 "group": self.group, "completed": [], "classification": self.CLASSIFICATION}
        Evidence.write(self.receipt, value)
        try:
            self.plan.verify_producer(Path(__file__).parent)
            self.plan.verify_sources(self.session)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            uuid = self.plan.uuid(torch.cuda.get_device_properties().uuid)
            if uuid != self.plan.value["gpu"]["uuid"]:
                raise ValueError("Batch GPU differs from locked plan")
            point = next(point for point in points() if point.columns() == self.group["point"])
            self.case = Case(point, "cuda", self.session.root / "fixtures")
            self.reference = ReferenceStore(self.plan.value["reference_root"]).get(self.case)
            initial = self.tensors(self.case, self.reference)
            value.update(gpu_uuid=uuid, input_hash=self.case.input_hash, reference_hashes=self.reference.hashes,
                         tensor_hashes=self.case.hashes, layouts=self.case.layouts(), initial_tensor_hashes=initial,
                         reference_store=self.reference.store_receipt, fixture_allocations=1, reference_store_loads=1,
                         source_build=self.source_build(point.precision))
            Evidence.write(self.receipt, value)
            for job_id in self.group["job_ids"]:
                job = self.plan.job(job_id)
                name = f"{job['variant_id']}_accuracy_{point.config}_{self.plan.value['cycle']}_{point.batch}_{point.precision}"
                args = argparse.Namespace(plan=self.plan.path, job_id=job_id,
                    output=self.session.root / "accuracy" / name, diagnostic_boundary=None)
                print(json.dumps({"batch_id": self.group["batch_id"], "job_id": job_id, "status": "START"}), flush=True)
                worker = self.create_worker(args)
                code = worker.run()
                if worker.result["status"] not in ("PASS", "ACCURACY_FAIL"):
                    raise RuntimeError("Variant did not terminate as a numerical result")
                value["completed"].append({"job_id": job_id, "result": str(worker.output / "result.json"),
                    "result_sha256": Evidence.sha256(worker.output / "result.json"), "status": worker.result["status"],
                    "numerical_returncode": code})
                Evidence.write(self.receipt, value)
                del worker
            final = self.tensors(self.case, self.reference)
            if final != initial:
                raise ValueError("A variant mutated the shared canonical fixture/reference")
            self.plan.verify_sources(self.session)
            self.plan.verify_producer(Path(__file__).parent)
            value.update(status="COMPLETE", final_tensor_hashes=final, source_unchanged=True,
                         shared_input_reference_unchanged=True)
            Evidence.write(self.receipt, value)
            return 0
        except Exception as error:
            import traceback
            value.update(status="PROCESS_ERROR_REQUIRES_FRESH_DIAGNOSIS", error=str(error), traceback=traceback.format_exc())
            Evidence.write(self.receipt, value)
            raise


class BatchWorker(SharedPointBatch):
    def create_worker(self, args):
        return MatrixWorker(args, shared=self)


class StandaloneChunk(AccuracyChunk):
    FILES = StandalonePlan.FILES

    def __init__(self, args):
        super().__init__(args)
        self.plan = StandalonePlan(args.plan)
        self.variant = next(item for item in self.plan.value["variants"] if item["variant_id"] == args.variant)

    def family_name(self):
        return self.args.variant

    def selected_points(self):
        keys = {job["point"]["config"] + "/" + str(job["point"]["batch"]) + "/" + job["point"]["precision"]
                for job in self.plan.value["jobs"] if job["variant_id"] == self.args.variant}
        return [point for point in points() if point.config == self.args.config and point.key in keys]

    def probe_argv(self, producer, point, output):
        self.plan.verify_producer(producer)
        job = next(job for job in self.plan.value["jobs"] if job["variant_id"] == self.args.variant and job["point"] == point.columns())
        return ["timeout", "--signal=INT", "--kill-after=5s", str(job["deadline_seconds"]) + "s",
                self.session.paths["FASTEQ_PY"], "-B", str(producer / "standalone_matrix.py"), "worker",
                "--plan", str(self.plan.path), "--job-id", job["job_id"], "--output", str(output)]

    def accept_terminal(self, result, terminal):
        return (terminal["status"] in ("PASS", "FAIL") and result.get("status") in ("PASS", "ACCURACY_FAIL")
                and result.get("producer_drift") is False and result.get("source_unchanged") is True)


class MatrixAudit:
    def __init__(self, plan):
        self.plan = plan
        self.reference_checks = {}
        self.batch_checks = {}

    @staticmethod
    def check_batch(group, batch):
        completed = batch.get("completed", [])
        return (batch.get("status") == "COMPLETE" and batch.get("group") == group and
                [row.get("job_id") for row in completed] == group["job_ids"] and
                batch.get("source_unchanged") is True and batch.get("shared_input_reference_unchanged") is True and
                batch.get("initial_tensor_hashes") == batch.get("final_tensor_hashes") and
                batch.get("fixture_allocations") == 1 and batch.get("reference_store_loads") == 1)

    def batch_record(self, group, result_path, result, terminal):
        path = self.plan.root / "batches" / group["batch_id"] / "receipt.json"
        if path not in self.batch_checks:
            self.batch_checks[path] = json.loads(path.read_text())
        batch = self.batch_checks[path]
        metadata = result.get("shared_batch", {})
        completed = next((row for row in batch.get("completed", []) if row.get("job_id") == result.get("job_id")), {})
        entrypoint = self.plan.value.get("batch_entrypoint", [self.plan.value["runtime"]["python"], "-B",
            str(self.plan.root / "producer/standalone_matrix.py"), "batch-worker"])
        wanted_argv = entrypoint + ["--plan", str(self.plan.path), "--batch-id", group["batch_id"]]
        valid = (self.check_batch(group, batch) and batch.get("plan_hash") == self.plan.value["plan_hash"] and
                 batch.get("gpu_uuid") == self.plan.value["gpu"]["uuid"] and terminal.get("status") == "PASS" and
                 terminal.get("argv") == wanted_argv and
                 metadata == {"batch_id": group["batch_id"], "receipt": str(path),
                              "variant_index": group["job_ids"].index(result["job_id"])} and
                 completed.get("result") == str(result_path) and completed.get("result_sha256") == Evidence.sha256(result_path) and
                 completed.get("status") == result.get("status") and
                 completed.get("numerical_returncode") == (0 if result.get("status") == "PASS" else 1) and
                 batch.get("source_build") == self.plan.source_record(group) and
                 batch.get("initial_tensor_hashes", {}).get("input") ==
                    {key: result.get("tensor_hashes", {}).get(key) for key in ("x_sha256", "w_sha256", "grad_sha256")} and
                 batch.get("initial_tensor_hashes", {}).get("reference", {}).get("fp64") == result.get("reference_hashes") and
                 batch.get("reference_store") == result.get("reference_store") and
                 all(batch.get(key) == result.get(key) for key in ("input_hash", "reference_hashes", "tensor_hashes", "layouts")))
        verification = self.plan.value.get("verification")
        if verification:
            observed = json.loads((self.plan.root / "numpy" / (group["batch_id"] + ".json")).read_text())
            valid = valid and (observed.get("status") == "COMPLETE" and observed.get("pid") == terminal.get("pid") and
                observed.get("plan_hash") == self.plan.value["plan_hash"] and observed.get("batch_id") == group["batch_id"] and
                observed.get("numpy_before") == observed.get("numpy_after") == verification["expected_numpy"] and
                observed.get("original_producer") == self.plan.value["producer"] and
                observed.get("batch_receipt_sha256") == Evidence.sha256(path) and
                observed.get("wrapper_sha256") == verification["wrapper"]["sha256"] == Evidence.sha256(Path(verification["wrapper"]["path"])))
        return [] if valid else ["batch process/point/variant/shared reference mapping or verification runtime differs"]

    @staticmethod
    def check_rows(job, rows, torch_errors=None):
        keys = [(row.get("operation"), row.get("result"), row.get("boundary")) for row in rows]
        reasons = []
        if set(keys) != StandalonePlan.cells(job) or len(keys) != len(set(keys)):
            reasons.append("missing or duplicate operation/result/boundary cell")
        for row in rows:
            if torch_errors is not None and row.get("torch_rel_fro") != torch_errors[row["result"]]:
                reasons.append("row comparator differs from actual cached Torch reference")
            limit = 1e-12 if job["precision"] == "fp64" else (2e-5 if job["precision"] == "tf32" else
                    1.05 * row["torch_rel_fro"] if row.get("torch_rel_fro") is not None and math.isfinite(row["torch_rel_fro"]) else None)
            numeric = row.get("rel_fro") is not None and math.isfinite(row["rel_fro"]) and limit is not None and row["rel_fro"] <= limit
            derived = numeric and all(row.get(name) is True for name in ("finite", "reference_finite", "repeat_equal", "different_buffer"))
            derived = derived and row.get("frozen_p14", {}).get("status") != "FAIL"
            if row.get("threshold") != limit or row.get("passed") is not derived:
                reasons.append("threshold or numerical gate differs")
        return reasons

    def record(self, job):
        value = self.plan.value
        name = f"{job['variant_id']}_accuracy_{job['point']['config']}_{value['cycle']}"
        attempt = name + "_" + str(job["point"]["batch"]) + "_" + job["precision"]
        session_root = Path(value["session"])
        output = session_root / "accuracy" / attempt
        group = next((row for row in value.get("batch_groups", []) if job["job_id"] in row["job_ids"]), None)
        result_path = output / "result.json"
        terminal_path = session_root / "logs" / ((group["batch_id"] if group else attempt) + ".json")
        if not result_path.exists() or not terminal_path.exists():
            return {"verified": False, "reasons": ["missing result or terminal receipt"]}
        result, terminal = json.loads(result_path.read_text()), json.loads(terminal_path.read_text())
        reasons = []
        if group:
            reasons += self.batch_record(group, result_path, result, terminal)
        if terminal.get("status") not in ("PASS", "FAIL") or result.get("status") not in ("PASS", "ACCURACY_FAIL"):
            reasons.append("nonterminal, process error, timeout or invalid producer")
        for field, expected in (("job_id", job["job_id"]), ("variant_id", job["variant_id"]), ("point", job["point"]),
                                ("cycle", value["cycle"]), ("plan_hash", value["plan_hash"]),
                                ("gpu_uuid", value["gpu"]["uuid"]), ("diagnostic_reproduction", False)):
            if result.get(field) != expected:
                reasons.append("identity differs: " + field)
        if result.get("producer_drift") is not False or result.get("source_unchanged") is not True:
            reasons.append("producer/source drift")
        try:
            self.plan.verify_producer(output / "producer_sources")
        except (OSError, ValueError):
            reasons.append("producer snapshot differs")
        identity = result.get("identity", {})
        source = self.plan.source_record(job)
        if (identity.get("binary_hash") != source["binary_hash"] or identity.get("binary") != source["binary"] or
                source["binary"] not in identity.get("mapped_libraries", []) or
                identity.get("runtime_controls") != job.get("runtime_controls", {})):
            reasons.append("actual DSO or runtime controls differ")
        if identity.get("impl") != job["impl"] or identity.get("bimpl") != job["bimpl"]:
            reasons.append("actual impl/bimpl differs")
        if (identity.get("api_controls", {}) != job.get("api_controls", {}) or
                identity.get("activation_layout", "biu") != job.get("activation_layout", "biu")):
            reasons.append("actual API/layout policy differs")
        if identity.get("source_hash") != source["source_manifest"]["source_hash"] or identity.get("headers_hash") != source["source_manifest"]["headers_hash"]:
            reasons.append("source/header manifest differs")
        for field in ("torch", "torch_cuda", "cxx11_abi"):
            if result.get(field) != value["runtime"][field]:
                reasons.append("actual runtime differs: " + field)
        if StandalonePlan.uuid(result.get("GPU_UUID")) != value["gpu"]["uuid"]:
            reasons.append("actual CUDA UUID differs")
        reference = result.get("reference_store", {})
        if not reference.get("path"):
            reasons.append("missing canonical reference receipt")
        else:
            path = Path(reference["path"])
            metadata = json.loads(path.read_text())
            ref_identity = metadata["identity"]
            if (Evidence.sha256(path) != reference["sha256"] or metadata["value_hashes"]["fp64"] != result.get("reference_hashes") or
                    ref_identity.get("input_hash") != result.get("input_hash") or ref_identity.get("point") != job["point"] or
                    ref_identity.get("tensor_hashes") != result.get("tensor_hashes") or ref_identity.get("layouts") != result.get("layouts") or
                    ref_identity.get("math") != ReferenceStore.MATH or self.plan.uuid(ref_identity.get("hardware", {}).get("uuid")) != value["gpu"]["uuid"] or
                    ref_identity.get("reference_producer_sha256") != value["producer"]["cases.py"] or
                    ref_identity.get("store_producer_sha256") != value["producer"]["reference_store.py"]):
                reasons.append("canonical input/reference identity differs")
            key = str(path), metadata["file_sha256"]
            if key not in self.reference_checks:
                import torch
                if Evidence.sha256(path.parent / "references.pt") != metadata["file_sha256"]:
                    reasons.append("canonical reference bytes differ")
                else:
                    self.reference_checks[key] = torch.load(path.parent / "references.pt", map_location="cpu", weights_only=True, mmap=True)["torch_errors"]
        reasons += self.check_rows(job, result.get("rows", []), self.reference_checks.get(key) if reference.get("path") else None)
        for row in result.get("rows", []):
            if not row["passed"]:
                if set(row.get("failed_outputs", {})) != {"first", "repeat"}:
                    reasons.append("missing actual numerical failure tensors")
                for failure in row.get("failed_outputs", {}).values():
                    if not Path(failure["path"]).is_file() or Evidence.sha256(Path(failure["path"])) != failure["sha256"]:
                        reasons.append("failure tensor file identity differs")
        return {"verified": not reasons, "reasons": sorted(set(reasons)), "result": str(result_path),
                "result_sha256": Evidence.sha256(result_path), "numerical_rows": len(result.get("rows", [])),
                "passed_rows": sum(row["passed"] for row in result.get("rows", [])),
                "accuracy_pass": result.get("status") == "PASS", "input_hash": result.get("input_hash"),
                "reference_hashes": result.get("reference_hashes"), "layouts": result.get("layouts")}

    def run(self, output, variant=None, operation=None, point_keys=None):
        if output.exists():
            raise FileExistsError("Independent audit evidence must use a fresh output path")
        auditor_root = output.with_suffix(".producer")
        auditor_root.mkdir(parents=True, exist_ok=False)
        for name in StandalonePlan.FILES:
            shutil.copy2(Path(__file__).with_name(name), auditor_root / name)
        auditor_sources = {name: Evidence.sha256(auditor_root / name) for name in StandalonePlan.FILES}
        records, shared = {}, {}
        jobs = MatrixController.selected_jobs(self.plan.value["jobs"],
            argparse.Namespace(precision=None, variant=variant, config=None, operation=operation))
        if point_keys is not None:
            available = {"/".join(map(str, (job["point"]["config"], job["point"]["batch"], job["precision"]))) for job in jobs}
            if len(point_keys) != len(set(point_keys)) or not set(point_keys) <= available:
                raise ValueError("Point subset contains a duplicate or unplanned point")
            jobs = [job for job in jobs if "/".join(map(str, (job["point"]["config"], job["point"]["batch"], job["precision"]))) in point_keys]
        if not jobs:
            raise ValueError("Audit selector has no planned jobs")
        for job in jobs:
            try:
                records[job["job_id"]] = self.record(job)
            except (OSError, ValueError, KeyError, TypeError) as error:
                records[job["job_id"]] = {"verified": False, "reasons": [type(error).__name__ + ": " + str(error)]}
            record = records[job["job_id"]]
            if record["verified"]:
                point_key = (job["point"]["config"], job["point"]["batch"], job["point"]["precision"])
                shared.setdefault(point_key, set()).add(Evidence.fingerprint({key: record[key] for key in ("input_hash", "reference_hashes", "layouts")}))
        mismatches = [list(key) for key, identities in shared.items() if len(identities) > 1]
        summary = {}
        for job in jobs:
            row = records[job["job_id"]]
            aggregate = summary.setdefault(job["variant_id"], Counter())
            aggregate["expected_jobs"] += 1
            aggregate["expected_cells"] += len(self.plan.cells(job))
            if row["verified"]:
                aggregate["verified_jobs"] += 1
                aggregate["verified_cells"] += row["numerical_rows"]
                aggregate["passed_cells"] += row["passed_rows"]
                aggregate["qualified_jobs"] += int(row["accuracy_pass"])
        source_consistent = auditor_sources == {name: Evidence.sha256(Path(__file__).with_name(name)) for name in StandalonePlan.FILES}
        report = {"classification": "STANDALONE_ACCURACY_NOT_TIMING", "plan_hash": self.plan.value["plan_hash"],
                  "stage": self.plan.value["stage"], "expected_jobs": len(records),
                  "expected_cells": sum(len(self.plan.cells(job)) for job in jobs),
                  "verified_jobs": sum(row["verified"] for row in records.values()),
                  "complete": all(row["verified"] for row in records.values()) and not mismatches and source_consistent,
                  "qualified_jobs": sum(row.get("accuracy_pass", False) for row in records.values() if row["verified"]),
                  "records": records, "shared_input_reference_mismatches": mismatches,
                  "variant_selector": variant, "operation_selector": operation,
                  "point_selector": point_keys,
                  "full_plan_complete": variant is None and operation is None and point_keys is None and all(row["verified"] for row in records.values()) and not mismatches and source_consistent,
                  "variant_summary": {key: dict(value) for key, value in summary.items()},
                  "passed_cells": sum(row.get("passed_rows", 0) for row in records.values() if row["verified"]),
                  "auditor_producer": {"directory": str(auditor_root), "files": auditor_sources, "source_consistent": source_consistent},
                  "auditor_sha256": Evidence.sha256(Path(__file__)),
                  "additional_policy_strategies_complete": False}
        Evidence.write(output, report)
        return report


class MatrixController:
    def __init__(self, args):
        self.args, self.plan = args, StandalonePlan(args.plan)
        self.session = Session(Path(self.plan.value["session"]))

    @staticmethod
    def selected_jobs(jobs, args):
        operation = getattr(args, "operation", None)
        variants = set(args.variant.split(",")) if args.variant else None
        if variants and not variants <= {job["variant_id"] for job in jobs}:
            raise ValueError("Variant selector contains an unplanned contract")
        return [job for job in jobs if (not args.precision or job["precision"] == args.precision)
                and (variants is None or job["variant_id"] in variants)
                and (not args.config or job["point"]["config"] == args.config)
                and (not operation or job["operation"] == operation or
                     (operation == "backward" and job["operation"] != "forward"))]

    def run(self):
        import time
        args, plan = self.args, self.plan
        if re.fullmatch(r"[A-Za-z0-9_.-]+", args.label) is None or args.max_groups < 1 or args.batch_deadline <= 0:
            raise ValueError("Use a simple label and positive bounded job limits")
        plan.verify_producer(Path(__file__).parent)
        plan.verify_sources(self.session)
        jobs = self.selected_jobs(plan.value["jobs"], args)
        groups = list(dict.fromkeys((job["variant_id"], job["point"]["config"]) for job in jobs))
        if not groups:
            raise ValueError("Selection has no planned jobs")
        start = time.monotonic()
        audit = MatrixAudit(plan)
        selected = []
        for variant, config in groups:
            if len(selected) >= args.max_groups or time.monotonic() - start > args.batch_deadline:
                break
            group_jobs = [job for job in jobs if job["variant_id"] == variant and job["point"]["config"] == config]
            existing = [audit.record(job) for job in group_jobs]
            if all(record["verified"] for record in existing):
                continue
            if any(record.get("result") for record in existing):
                raise RuntimeError("Partial/invalid chunk preserved; inspect and use a fresh cycle, never mix attempts")
            chunk = argparse.Namespace(plan=plan.path, session=self.session.root, config=config,
                variant=variant, gpu=plan.value["gpu"]["physical_index"], attempt=plan.value["cycle"])
            code = StandaloneChunk(chunk).run()
            records = [audit.record(job) for job in group_jobs]
            selected.append({"variant_id": variant, "config": config, "returncode": code,
                             "verified_jobs": sum(record["verified"] for record in records), "expected_jobs": len(records)})
            Evidence.write(plan.root / ("controller_" + args.label + ".json"), {
                "classification": "STANDALONE_ACCURACY_NOT_TIMING", "plan_hash": plan.value["plan_hash"],
                "status": "RUNNING" if code == 0 else "INCOMPLETE_DIAGNOSIS_REQUIRED", "chunks": selected})
            if code != 0 or not all(record["verified"] for record in records):
                return 1
        Evidence.write(plan.root / ("controller_" + args.label + ".json"), {
            "classification": "STANDALONE_ACCURACY_NOT_TIMING", "plan_hash": plan.value["plan_hash"],
            "status": "SELECTED_CHUNKS_COMPLETE", "chunks": selected,
            "full_plan_complete_not_implied": True})
        return 0


class BatchController(MatrixController):
    def run(self):
        import time
        args, plan = self.args, self.plan
        if args.variant or args.operation:
            raise ValueError("A point batch must retain every planned variant of its DSO/point")
        if re.fullmatch(r"[A-Za-z0-9_.-]+", args.label) is None or args.max_groups < 1 or args.batch_deadline <= 0:
            raise ValueError("Use a simple label and positive bounded job limits")
        plan.verify_producer(Path(__file__).parent)
        plan.verify_sources(self.session)
        groups = [row for row in plan.value["batch_groups"] if (not args.precision or row["precision"] == args.precision)
                  and (not args.config or row["point"]["config"] == args.config)]
        if not groups:
            raise ValueError("Batch selection has no planned groups")
        start, completed = time.monotonic(), []
        audit = MatrixAudit(plan)
        for group in groups:
            if len(completed) >= args.max_groups or time.monotonic() - start + group["deadline_seconds"] > args.batch_deadline:
                break
            jobs = [plan.job(job_id) for job_id in group["job_ids"]]
            existing = [audit.record(job) for job in jobs]
            if all(row["verified"] for row in existing):
                continue
            if any(row.get("result") for row in existing) or (plan.root / "batches" / group["batch_id"]).exists():
                raise RuntimeError("Partial batch preserved; diagnose in a fresh process/cycle")
            gpu = self.session.idle_gpu(group["batch_id"], plan.value["gpu"]["physical_index"])
            if plan.uuid(gpu) != plan.value["gpu"]["uuid"]:
                raise ValueError("Idle selection differs from planned GPU")
            command = [plan.value["runtime"]["python"], "-B", str(plan.root / "producer/standalone_matrix.py"),
                       "batch-worker", "--plan", str(plan.path), "--batch-id", group["batch_id"]]
            terminal = self.session.runner().run(group["batch_id"], command, group["deadline_seconds"],
                Path(self.session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": gpu})
            records = [audit.record(job) for job in jobs]
            valid = terminal["status"] == "PASS" and all(row["verified"] for row in records)
            completed.append({"batch_id": group["batch_id"], "verified": valid,
                              "jobs": len(jobs), "qualified_jobs": sum(row.get("accuracy_pass", False) for row in records)})
            Evidence.write(plan.root / ("controller_" + args.label + ".json"), {
                "plan_hash": plan.value["plan_hash"], "status": "RUNNING" if valid else "INCOMPLETE_DIAGNOSIS_REQUIRED",
                "execution_model": "same_build_point_batch", "batches": completed, "full_plan_complete_not_implied": True})
            if not valid:
                return 1
        Evidence.write(plan.root / ("controller_" + args.label + ".json"), {
            "plan_hash": plan.value["plan_hash"], "status": "SELECTED_BATCHES_COMPLETE", "batches": completed,
            "execution_model": "same_build_point_batch", "full_plan_complete_not_implied": True})
        return 0


class BatchEquivalence:
    """Compare actual new batch outputs with separately audited old processes."""

    @staticmethod
    def run(args):
        plan = StandalonePlan(args.plan)
        if plan.value.get("execution_model") != "same_build_point_batch":
            raise ValueError("Equivalence target must be a fresh batch cycle")
        if not args.output.resolve().is_relative_to(Path(plan.value["session"])):
            raise ValueError("Comparison evidence must stay in session")
        audit = MatrixAudit(plan)
        baselines = [(StandalonePlan(path), MatrixAudit(StandalonePlan(path))) for path in args.baseline_plan]
        jobs = [job for job in plan.value["jobs"] if job["precision"] == args.precision and
                job["point"]["config"] == args.config and job["point"]["batch"] == args.batch]
        fields = ("output_hash", "rel_fro", "max_abs", "mean_abs", "finite", "reference_finite", "repeat_equal",
                  "different_buffer", "threshold", "torch_rel_fro", "passed")
        rows = []
        for job in jobs:
            current = audit.record(job)
            before = None
            for old_plan, old_audit in baselines:
                old_job = next((item for item in old_plan.value["jobs"] if item["variant_id"] == job["variant_id"] and item["point"] == job["point"]), None)
                if old_job:
                    candidate = old_audit.record(old_job)
                    if candidate["verified"]:
                        before = (old_plan, candidate)
                        break
            reasons = []
            if not current["verified"] or before is None:
                reasons.append("missing independently verified current or baseline evidence")
            else:
                old_plan, previous = before
                after = json.loads(Path(current["result"]).read_text())
                old = json.loads(Path(previous["result"]).read_text())
                for name in ("input_hash", "reference_hashes", "layouts", "torch_errors", "actual_math_flags"):
                    if after.get(name) != old.get(name):
                        reasons.append(name + " differs")
                precision = job["precision"]
                for name in ("binary_hash", "source_manifest"):
                    if plan.value["sources"][precision][name] != old_plan.value["sources"][precision][name]:
                        reasons.append("DSO/source differs: " + name)
                if plan.value["gpu"]["uuid"] != old_plan.value["gpu"]["uuid"]:
                    reasons.append("GPU differs")
                normalize = lambda value: {(row["operation"], row["result"], row["boundary"]): {key: row[key] for key in fields} for row in value["rows"]}
                if normalize(after) != normalize(old):
                    reasons.append("actual output bytes, numerical values, gates or buffer/repeat evidence differ")
            rows.append({"job_id": job["job_id"], "equivalent": not reasons, "reasons": reasons,
                         "current": current, "baseline": before[1] if before else None,
                         "baseline_plan_hash": before[0].value["plan_hash"] if before else None})
        complete = bool(rows) and all(row["equivalent"] for row in rows)
        Evidence.write(args.output, {"classification": "BATCH_CORRECTNESS_EQUIVALENCE_NOT_TIMING", "passed": complete,
            "plan_hash": plan.value["plan_hash"], "expected_jobs": len(jobs), "records": rows,
            "source_sha256": Evidence.sha256(Path(__file__)), "old_evidence_resigned": False})
        return 0 if complete else 1


class Selfcheck:
    @staticmethod
    def run(args):
        session = Session(args.session)
        if not args.output.resolve().is_relative_to(session.root):
            raise ValueError("CPU selfcheck output must stay in the configured session")
        registry = json.loads((session.root / "coverage/variants.json").read_text())
        variants = VariantSpecs.from_registry(registry)
        checks = []
        for stage, jobs_count, cells_count in (("minimal", 19, 57), ("capability", 28, 69), ("full", 700, 1725)):
            value = {"variants": variants, "stage": stage, "jobs": StandalonePlan.expected_jobs(variants, stage)}
            StandalonePlan.verify_scope(value)
            assert len(value["jobs"]) == jobs_count and sum(len(StandalonePlan.cells(job)) for job in value["jobs"]) == cells_count
            for corruption in ("missing", "duplicate", "seed", "boundary"):
                changed = copy.deepcopy(value)
                if corruption == "missing":
                    changed["jobs"].pop()
                elif corruption == "duplicate":
                    changed["jobs"].append(copy.deepcopy(changed["jobs"][0]))
                elif corruption == "seed":
                    changed["jobs"][0]["point"]["seed"] += 1
                else:
                    changed["jobs"][0]["boundaries"] = ["public_warm"]
                try:
                    StandalonePlan.verify_scope(changed)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Corrupt scope accepted: " + corruption)
            checks.append(stage + " scope/count and four corruption rejections")
        assert all(not (spec["precision"] == "tf32" and spec["operation"] == "forward" and spec["impl"] in (1, 2)) for spec in variants)
        assert sum(spec["operation"] != "forward" for spec in variants) == 9
        checks.append("excluded TF32x1 and independent nine backward contracts")
        job = StandalonePlan.expected_jobs(variants, "minimal")[0]
        rows = [{"operation": operation, "result": result, "boundary": boundary, "torch_rel_fro": 1e-7,
                 "threshold": 2e-5, "rel_fro": 1e-7, "finite": True, "reference_finite": True,
                 "repeat_equal": True, "different_buffer": True, "passed": True}
                for operation, result, boundary in StandalonePlan.cells(job)]
        assert not MatrixAudit.check_rows(job, rows)
        changed = copy.deepcopy(rows)
        changed[0]["threshold"] = 1.
        assert MatrixAudit.check_rows(job, changed)
        changed = copy.deepcopy(rows)
        changed[0]["passed"] = False
        assert MatrixAudit.check_rows(job, changed)
        changed = copy.deepcopy(rows)
        changed[0]["reference_finite"] = False
        assert MatrixAudit.check_rows(job, changed)
        assert MatrixAudit.check_rows(job, rows + [copy.deepcopy(rows[0])])
        assert MatrixAudit.check_rows(job, rows, {"forward": 2e-7})
        changed = copy.deepcopy(rows)
        changed[0]["rel_fro"] = 1e-3
        changed[0]["passed"] = False
        assert not MatrixAudit.check_rows(job, changed)
        checks.append("numerical FAIL is valid evidence; comparator/gate/duplicate tampering rejected")
        assert StandalonePlan.uuid("GPU-ABC123") == StandalonePlan.uuid("abc123")
        checks.append("NVML/Torch UUID encodings normalized")
        fake_plan = type("Plan", (), {"value": {"jobs": StandalonePlan.expected_jobs(variants, "full")}})()
        chunk = object.__new__(StandaloneChunk)
        chunk.plan = fake_plan
        chunk.args = argparse.Namespace(variant="standalone_fp32_impl2", config="LARGE_P3")
        selected = chunk.selected_points()
        assert len(selected) == 5 and all(point.precision == "fp32" for point in selected)
        checks.append("shared AccuracyChunk selector yields five points of the variant precision")
        probe = object.__new__(StandaloneProbe)
        probe.args = argparse.Namespace(boundaries=",".join(StandaloneAdapter.BOUNDARIES), precision="tf32")
        probe.boundary_factory = None
        assert probe.boundaries("forward") == list(StandaloneAdapter.BOUNDARIES)
        assert probe.boundaries("grad_x") == ["run_with_activation_prep"]
        probe.boundary_factory = lambda operation: ["raw_allocating"]
        assert probe.boundaries("grad_x") == ["raw_allocating"]
        case_marker, source_marker = object(), object()
        probe.source_factory = lambda case: source_marker if case is case_marker else None
        assert probe.make_source(case_marker) is source_marker
        probe.boundary_factory = lambda operation: ["raw_allocating", "raw_allocating"]
        try:
            probe.boundaries("grad_x")
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate raw boundaries accepted")
        checks.append("source/boundary hooks preserve raw allocating dx labels and reject duplicate boundaries")
        selected_bwd = MatrixController.selected_jobs(StandalonePlan.expected_jobs(variants, "capability"),
            argparse.Namespace(precision=None, variant=None, config=None, operation="backward"))
        assert len(selected_bwd) == 9 and sum(len(StandalonePlan.cells(row)) for row in selected_bwd) == 12
        checks.append("backward capability selector retains all nine contracts and twelve numeric cells")
        selected_two = MatrixController.selected_jobs(StandalonePlan.expected_jobs(variants, "capability"),
            argparse.Namespace(precision=None, variant="standalone_fp32_b1,standalone_tf32_impl0", config=None, operation=None))
        assert len(selected_two) == 2
        try:
            MatrixController.selected_jobs(StandalonePlan.expected_jobs(variants, "capability"),
                argparse.Namespace(precision=None, variant="standalone_fp32_b1,unknown", config=None, operation=None))
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown member silently ignored in a finite variant subset")
        checks.append("explicit finite subset preserves both contracts and rejects any unknown member")
        batched = {"variants": variants, "stage": "full", "cycle": "cpu_only",
                   "execution_model": "same_build_point_batch", "jobs": StandalonePlan.expected_jobs(variants, "full")}
        batched["batch_groups"] = StandalonePlan.batch_groups(batched)
        StandalonePlan.verify_scope(batched)
        assert len(batched["batch_groups"]) == 75
        assert Counter(row["precision"] for row in batched["batch_groups"]) == {"tf32": 25, "fp32": 25, "fp64": 25}
        assert sum(len(row["job_ids"]) for row in batched["batch_groups"]) == 700
        changed = copy.deepcopy(batched)
        changed["batch_groups"][0]["job_ids"].pop()
        try:
            StandalonePlan.verify_scope(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("Batch omitted an expected variant")
        group = batched["batch_groups"][0]
        receipt = {"status": "COMPLETE", "group": group, "completed": [{"job_id": key} for key in group["job_ids"]],
                   "source_unchanged": True, "shared_input_reference_unchanged": True,
                   "initial_tensor_hashes": {"x": "same"}, "final_tensor_hashes": {"x": "same"},
                   "fixture_allocations": 1, "reference_store_loads": 1}
        assert MatrixAudit.check_batch(group, receipt)
        for field, bad in (("status", "PROCESS_ERROR"), ("reference_store_loads", 2),
                           ("final_tensor_hashes", {"x": "mutated"}), ("completed", receipt["completed"][:-1])):
            assert not MatrixAudit.check_batch(group, {**receipt, field: bad})
        checks.append("75 independent DSO/point batches retain all700 jobs; partial/mutated/shared-load receipts rejected")
        profiled = copy.deepcopy(StandalonePlan.expected_jobs(variants, "capability")[-1])
        profiled["build_profile"] = "profile_a"
        other = {**profiled, "job_id": profiled["job_id"] + "_other", "build_profile": "profile_b"}
        groups = StandalonePlan.batch_groups({"jobs": [profiled, other], "cycle": "cpu_profiles"})
        assert len(groups) == 2 and len({row["batch_id"] for row in groups}) == 2
        checks.append("different compiler DSOs remain separate processes at the same precision/point")
        import torch
        from unittest.mock import patch
        x = torch.arange(16, dtype=torch.float64).reshape(2, 4, 2)
        packed = StandaloneAdapter.pack_activation(x, [1, 3])
        expected = torch.tensor([0, 1, 8, 9, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15], dtype=torch.float64).reshape_as(x)
        assert torch.equal(packed, expected) and not torch.equal(packed, x)
        checks.append("PBU pack is a real path/batch permutation, not a reshape alias")
        class ContractOps:
            received = None
            def wgrad_create(self, *values):
                self.received = values
                return 42
        adapter = object.__new__(StandaloneAdapter)
        case = type("SyntheticCase", (), {"x3": x, "w3": torch.zeros(2, 2, 2), "dims": [1, 3], "u": 2, "cg": [.01, .011]})()
        adapter.case, adapter.ops = case, ContractOps()
        adapter.precision, adapter.activation_layout = "fp64", "biu"
        adapter.handles, adapter.outputs, adapter.prepared, adapter.setup_seconds = {}, {}, set(), {}
        adapter.resolved_api_controls = {"wgrad_tile": 14, "wgrad_split_k": 4, "wgrad_kernel": 0, "wgrad_transpose_mode": 7}
        with patch("torch.cuda.synchronize"):
            adapter.prepare("grad_w", "run_with_activation_prep")
        assert adapter.ops.received[-4:] == (14, 4, 0, 7)
        checks.append("FP64 named API controls preserve existing tile/split/kernel/transpose argument order")
        choices = ExistingChoices(session).build()
        additional = {"variants": choices.rows}
        extended = VariantSpecs.from_registry(registry, additional)
        extended_jobs = StandalonePlan.expected_jobs(extended, "full")
        extended_cells = sum(len(StandalonePlan.cells(row)) for row in extended_jobs)
        assert len(choices.rows) == 157 and len(extended) == 185 and len(extended_jobs) == 4625
        named = [row for row in extended if "base_variant_id" in row["registry_row"]]
        named_jobs = StandalonePlan.expected_jobs(named, "full")
        assert len(named_jobs) == 3925 and sum(len(StandalonePlan.cells(row)) for row in named_jobs) == 8325
        named_groups = StandalonePlan.batch_groups({"jobs": named_jobs, "cycle": "named_cpu_only", "stage": "full", "variant_scope": "named"})
        assert len(named_groups) == 75 and all(row["deadline_seconds"] >= 10 * len(row["job_ids"]) for row in named_groups)
        capability_jobs = StandalonePlan.expected_jobs(extended, "capability")
        assert len(capability_jobs) == 185
        assert all(row["point"]["config"].startswith("LARGE") for row in capability_jobs
                   if "EQLIN_FP32_CFG" in row.get("runtime_controls", {}) and row["variant_id"] != "standalone_fp32_impl3__cfg_wide")
        assert not any(row["base_variant_id"] == "standalone_fp32_b1" and "EQLIN_BWD_CHAIN" in row["runtime_controls"] for row in choices.rows)
        assert any(row["runtime_controls"].get("EQLIN_BWD_CHAIN") for row in choices.aliases)
        assert all(row["status"] != "UNCLASSIFIED_REQUIRES_REVIEW" for row in choices.runtime_inventory())
        assert len(choices.historical_builds()) == 2
        checks.append("157 finite named policies, all getenv controls classified, two preserved compiler tuples")
        checks.append("named-only complete epoch retains157 policies/3925jobs/8325cells in75 point processes")
        checks.append("named capability points exercise shape-dependent policies; ignored dgrad chain deduplicated")
        Evidence.write(args.output, {"classification": "CPU_SYNTHETIC_NOT_H20_PERFORMANCE", "passed": True,
            "checks": checks, "full_default_registry_jobs": 700, "full_default_registry_cells": 1725,
            "extended_existing_named_policy_jobs": len(extended_jobs), "extended_existing_named_policy_cells": extended_cells,
            "tool_sha256": Evidence.sha256(Path(__file__)), "GPU_work_started": False})
        print(json.dumps({"passed": True, "checks": len(checks), "jobs": 700, "cells": 1725}))
        return 0


class ExistingChoices:
    """Finite named policies already present in this source; no Cartesian sweep."""

    def __init__(self, session):
        self.session = session
        self.registry = json.loads((session.root / "coverage/variants.json").read_text())
        self.base = {row["variant_id"]: row for row in self.registry["variants"]}
        self.rows, self.aliases = [], []

    def evidence(self, precision, relative, needle):
        path = StandaloneSource(self.session, precision).original / relative
        lines = path.read_text().splitlines()
        line = next(index for index, text in enumerate(lines, 1) if needle in text)
        return {"path": str(path), "line": line, "text": lines[line - 1].strip(), "sha256": Evidence.sha256(path)}

    def add(self, base_id, suffix, runtime, api, evidence, effective, adapter_required=False):
        base = copy.deepcopy(self.base[base_id])
        base.update(variant_id=base_id + "__" + suffix, base_variant_id=base_id,
                    status="UNTESTED", runtime_status="UNTESTED", backend_verified=False,
                    runtime_controls=runtime, api_controls=api, named_choice_evidence=evidence,
                    effective_when=effective, special_adapter_required=adapter_required, adapter_extension_required=False,
                    strategy_kind="existing_single_named_policy", scope_status="IN_SCOPE",
                    reason="Finite named source policy; not covered by the default-control enum matrix")
        base["probe_point"] = self.probe_point(base)
        base["semantic_contract_hash"] = Evidence.fingerprint({key: base[key] for key in
            ("base_variant_id", "runtime_controls", "api_controls", "effective_when", "probe_point")})
        self.rows.append(base)

    @staticmethod
    def probe_point(row):
        """Pick a source-active shape before expanding the policy to all 25 points."""
        controls, api = row["runtime_controls"], row["api_controls"]
        config, batch = "SMALL_P1", 512
        if row["family"] == "standalone_fp32":
            if "EQLIN_FP32_CFG" in controls or "EQLIN_FP32_STAGES" in controls or "EQLIN_FP32_MAINLOOP" in controls:
                config, batch = ("LARGE_P3" if row["operation"] != "forward" or row["base_variant_id"].endswith("impl3") else "LARGE_P1"), 2048
            if row["base_variant_id"] == "standalone_fp32_impl3" and (
                    controls.get("EQLIN_FP32_CFG") == "wide" or controls.get("EQLIN_FP32_STAGES") == "3"):
                config = "MEDIUM_P2"
            if row["operation"] != "forward" and controls.get("EQLIN_FP32_STAGES") == "3":
                config, batch = "SMALL_P1", 512
            if "EQLIN_PATH_SPLIT" in controls or "EQLIN_FP32_FUSED" in controls:
                config = "LARGE_P3"
            if "EQLIN_BWD_CHAIN" in controls:
                config, batch = "LARGE_P3", (512 if controls["EQLIN_BWD_CHAIN"] == "18" else 5888)
        elif row["family"] == "standalone_fp64":
            if "EQLIN_WGRAD_ROW_ORDER" in controls or "wgrad_transpose_mode" in api:
                config = "MEDIUM_P2"
            if controls.get("EQLIN_WGRAD_ROW_ORDER") == "descending_k" or api.get("wgrad_transpose_mode") == 2:
                config = "LARGE_P3"
            if controls.get("EQLIN_TMA_N") == "32":
                config = "MEDIUM_P1"
            if controls.get("EQLIN_2X_TILE") == "m128n32s3":
                config = "MEDIUM_P1"
        elif row["family"] == "standalone_tf32":
            if set(controls) & {"EQLIN_N_POLICY", "EQLIN_RASTER", "EQLIN_CLUSTER_M", "EQLIN_STREAMK", "EQLIN_SPLITK", "EQLIN_WGRAD_CHAIN"}:
                config = "LARGE_P3"
            if "EQLIN_FOLD_K" in controls:
                config, batch = "LARGE_P3", 5888
            if controls.get("EQLIN_WGRAD_CHAIN") == "512":
                config = "SMALL_P1"
        return {"config": config, "batch": batch}

    def build(self):
        p = "fp32"
        tile = self.evidence(p, "csrc/grouped_gemm_fp32.cu", 'getenv("EQLIN_FP32_TILE_POLICY")')
        for value in (1, 2, 3):
            self.add("standalone_fp32_impl0", "tile_policy" + str(value), {"EQLIN_FP32_TILE_POLICY": str(value)}, {}, [tile], "stock SIMT impl0")
        cfg = self.evidence(p, "csrc/simt_gemm.cu", 'strcmp(cfg_from_env(), "wide")')
        loop = self.evidence(p, "csrc/simt_gemm.cu", 'getenv("EQLIN_FP32_MAINLOOP")')
        stages = self.evidence(p, "csrc/simt_gemm.cu", "(s == 2 || s == 3 || s == 4)")
        for impl in (1, 2, 3):
            base = "standalone_fp32_impl" + str(impl)
            for value in ("narrow", "wide"):
                if value == "narrow" and impl in (1, 2):
                    self.aliases.append({"base_variant_id": base, "runtime_controls": {"EQLIN_FP32_CFG": "narrow"},
                        "equivalence": "plain N>128 defaults to Cfg224 narrow; N<=128 ignores cfg; auto stock branch ignores cfg",
                        "boundaries": ["run_with_activation_prep", "prepared_gemm"], "evidence": [cfg]})
                    continue
                self.add(base, "cfg_" + value, {"EQLIN_FP32_CFG": value}, {}, [cfg],
                         "custom/fused path; narrow/wide ignored in plain N<=128 or stock auto branches")
            self.add(base, "mainloop_cpasync", {"EQLIN_FP32_MAINLOOP": "cpasync"}, {}, [loop],
                     "custom/fused path; default already cpasync for N<=128 and M<1024")
            for value in (2, 3, 4):
                if value == 4 and impl in (1, 2):
                    self.aliases.append({"base_variant_id": base, "runtime_controls": {"EQLIN_FP32_STAGES": "4"},
                        "equivalence": "plain custom default stages4; stock branch ignores this control",
                        "boundaries": ["run_with_activation_prep", "prepared_gemm"], "evidence": [stages]})
                    continue
                self.add(base, "stages" + str(value), {"EQLIN_FP32_STAGES": str(value)}, {}, [stages],
                         "custom/fused path; explicit finite instantiated stage2/3/4")
        split = self.evidence(p, "csrc/runner.cu", 'getenv("EQLIN_PATH_SPLIT")')
        view = self.evidence(p, "csrc/runner.cu", 'getenv("EQLIN_A_VIEW")')
        for impl in range(4):
            base = "standalone_fp32_impl" + str(impl)
            self.add(base, "k_concat", {"EQLIN_PATH_SPLIT": "0"}, {}, [split], "multi-path configs; single-path aliases baseline")
            self.add(base, "packed_a", {"EQLIN_A_VIEW": "0"}, {}, [view], "forces A preparation instead of direct view")
        fused = self.evidence(p, "csrc/runner.cu", 'getenv("EQLIN_FP32_FUSED")')
        self.add("standalone_fp32_impl3", "split_reduce", {"EQLIN_FP32_FUSED": "0"}, {}, [fused], "multi-path; disables fused CTA accumulation")
        bcfg = self.evidence(p, "csrc/simt_gemm.cu", 'strcmp(cfg_from_env(), "hybrid")')
        bstages = self.evidence(p, "csrc/bwd.cu", "(s == 2 || s == 3 || s == 4)")
        for bimpl in (1, 2, 3):
            base = "standalone_fp32_b" + str(bimpl)
            for value in ("wide", "halfn", "hybrid"):
                self.add(base, "cfg_" + value, {"EQLIN_FP32_CFG": value}, {}, [bcfg],
                         "N>128; hybrid selects wide/stage4 only at B>=2048")
            for value in (2, 3, 4):
                self.add(base, "stages" + str(value), {"EQLIN_FP32_STAGES": str(value)}, {}, [bstages],
                         "BWD stage override; chain tiles clamped to pipeline depth")
            chain = self.evidence(p, "M8_REPORT.md", "3/6/12/18/24/32")
            for value in (3, 6, 12, 18, 24, 32):
                if bimpl == 1:
                    self.aliases.append({"base_variant_id": base, "runtime_controls": {"EQLIN_BWD_CHAIN": str(value)},
                        "equivalence": "dgrad-only b1 does not instantiate WgradPlan; chain control is not consumed",
                        "evidence": [chain, self.evidence(p, "csrc/bwd.cu", "chain_tiles_from_env")]})
                    continue
                self.add(base, "chain" + str(value), {"EQLIN_BWD_CHAIN": str(value)}, {}, [chain],
                         "finite historical M8 ladder; weight-gradient component only")
        p = "fp64"
        tile_source = StandaloneSource(self.session, p).original / "csrc/bwd.cu"
        tile_rows = []
        for line, text in enumerate(tile_source.read_text().splitlines(), 1):
            match = re.search(r"case (\d+): return Gemm2xConfig::(\w+)", text)
            if match:
                tile_rows.append((int(match.group(1)), match.group(2), {"path": str(tile_source), "line": line, "text": text.strip(), "sha256": Evidence.sha256(tile_source)}))
        kernel = self.evidence(p, "csrc/bwd.cu", "if (kernel == 0)")
        modes = self.evidence(p, "csrc/bwd.cu", "transpose_mode == 3 ?")
        for base in ("standalone_fp64_wgrad", "standalone_fp64_bwd"):
            for value, name, item in tile_rows:
                if value:
                    self.add(base, "tile" + str(value) + "_" + name, {}, {"wgrad_tile": value}, [item],
                             "wgrad kernel0 only; source rejects tile!=0 for direct/SM90 kernels", True)
            for value in (1, 2, 3):
                self.add(base, "kernel" + str(value), {}, {"wgrad_kernel": value}, [kernel],
                         "tile0 required; kernel1=CuTe k8, kernel2=CuTe k16, kernel3=SM90 TMA", True)
            for value in (0, 1, 2, 4, 5, 6, 7):
                self.add(base, "transpose" + str(value), {}, {"wgrad_transpose_mode": value}, [modes],
                         "replaces auto mode3 (P10/U128=>2, other shapes=>0)", True)
        raster = self.evidence(p, "csrc/sm90_fp64_grouped_gemm.cu", 'getenv("EQLIN_SM90_RASTER")')
        swizzle = self.evidence(p, "csrc/sm90_fp64_grouped_gemm.cu", 'v != 1 && v != 2 && v != 4 && v != 8')
        self.add("standalone_fp64_impl1", "raster_alongm", {"EQLIN_SM90_RASTER": "alongm"}, {}, [raster], "forced SM90 path; default alongn")
        for value in (2, 4, 8):
            self.add("standalone_fp64_impl1", "swizzle" + str(value), {"EQLIN_SM90_SWIZZLE": str(value)}, {}, [swizzle], "forced SM90 path; default swizzle1")
        names = self.evidence(p, "csrc/runner.cu", "Gemm2xConfig gemm2x_from_name")
        for name in ("base", "m128n32s3", "m128n32s4", "m64n32s4", "m64n64s3", "ins8", "m32n128s3", "m32n128s4"):
            self.add("standalone_fp64_impl3", "tile_" + name, {"EQLIN_2X_TILE": name}, {}, [names],
                     "impl3 only; effective 2.x branch; multi-path SM90 ignores this tile")
        tma = self.evidence(p, "csrc/runner.cu", 'getenv("EQLIN_TMA_N")')
        for value in (32, 64):
            self.add("standalone_fp64_impl6", "tma_n" + str(value), {"EQLIN_TMA_N": str(value)}, {}, [tma],
                     "forced TMA implementation6; impl7 aliases this control only where its auto rule chooses6")
        pack = self.evidence(p, "csrc/runner.cu", 'getenv("EQLIN_A_PACK")')
        self.add("standalone_fp64_impl1", "packed_a", {"EQLIN_A_PACK": "1"}, {}, [pack],
                 "forced SM90 implementation1; automatic entrypoints share this exact branch when SM90 is selected")
        view64 = self.evidence(p, "csrc/runner.cu", 'getenv("EQLIN_A_VIEW")')
        self.add("standalone_fp64_impl0", "packed_a", {"EQLIN_A_VIEW": "0"}, {}, [view64],
                 "2.x single-path A-view branch; automatic entrypoints share this preparation branch")
        order = self.evidence(p, "csrc/bwd.cu", 'std::getenv("EQLIN_WGRAD_ROW_ORDER")')
        for base in ("standalone_fp64_wgrad", "standalone_fp64_bwd"):
            for value in ("original", "ascending_k", "descending_k"):
                self.add(base, "row_order_" + value, {"EQLIN_WGRAD_ROW_ORDER": value}, {}, [order],
                         "wgrad row metadata; default sorts descending only at P10/U128/V128")
        p = "tf32"
        for suffix, name, value, source, needle, condition, needs_adapter in (
            ("packed_a", "EQLIN_A_VIEW", "0", "csrc/runner.cu", 'getenv("EQLIN_A_VIEW")', "one path per output row; otherwise already packed", False),
            ("legacy_n_policy", "EQLIN_N_POLICY", "0", "csrc/sm90_grouped_gemm.cu", 'getenv("EQLIN_N_POLICY")', "in-kernel/register split modes", False),
            ("raster_n", "EQLIN_RASTER", "1", "csrc/sm90_grouped_gemm.cu", 'getenv("EQLIN_RASTER")', "SM90 scheduler, default alongM", False),
            ("cluster_m2", "EQLIN_CLUSTER_M", "2", "csrc/sm90_grouped_gemm.cu", "case 2:", "only compiled cluster extents1/2", False),
            ("streamk", "EQLIN_STREAMK", "1", "csrc/sm90_grouped_gemm.cu", 'getenv("EQLIN_STREAMK")', "existing stream-K group scheduler", False),
            ("layout_pbu", "EQLIN_A_LAYOUT", "pbu", "csrc/runner.cu", 'strcmp(l, "pbu")', "one-path A-view only; actual per-call PBU packing required", True)):
            self.add("standalone_tf32_impl5", suffix, {name: value}, {}, [self.evidence(p, source, needle)], condition, needs_adapter)
        fold = self.evidence(p, "P1_NUMERICS_REPORT.md", "for FK in 2 4 8 16 32")
        for impl in (7, 8):
            for value in (2, 4, 8, 16, 32):
                self.add("standalone_tf32_impl" + str(impl), "fold" + str(value), {"EQLIN_FOLD_K": str(value)}, {}, [fold],
                         "finite P1 numerical ladder; only staged TF32x3 impl7/8 consume fold control")
        splitk = self.evidence(p, "P1_NUMERICS_REPORT.md", "for S in 1 2 4")
        for value in (2, 4):
            self.add("standalone_tf32_impl5", "splitk" + str(value), {"EQLIN_SPLITK": str(value)}, {}, [splitk],
                     "finite P1 split-K ladder with required FP64 reduction; S1 aliases disabled baseline")
        raster_sweep = self.evidence(p, "tests/sweep_raster.sh", '"m_sw1:0:1"')
        for raster_value, prefix in ((0, "m"), (1, "n")):
            for swizzle_value in (1, 2, 4, 8):
                if raster_value == 0 and swizzle_value == 1:
                    continue
                self.add("standalone_tf32_impl3", "raster_" + prefix + "_sw" + str(swizzle_value),
                         {"EQLIN_RASTER": str(raster_value), "EQLIN_SWIZZLE": str(swizzle_value)}, {}, [raster_sweep],
                         "exact finite existing M3 source sweep tuple; default m/sw1 is baseline")
        chain = self.evidence(p, "B3_REPORT.md", "c512 / c384 / c256 / c128")
        for bimpl in (2, 3):
            for value in (128, 256, 384, 512):
                self.add("standalone_tf32_b" + str(bimpl), "wgrad_chain" + str(value), {"EQLIN_WGRAD_CHAIN": str(value)}, {}, [chain],
                         "finite historical B3 chain ladder; weight-gradient component only")
        self.aliases.append({"base_variant_id": "standalone_fp32_impl1/2/3", "runtime_controls": {"EQLIN_FP32_MAINLOOP": "tma"},
            "equivalence": "every non-cpasync string is same conditional default; do not register arbitrary strings as implementations", "evidence": [loop]})
        ids = [row["variant_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate named policy registration")
        return self

    def runtime_inventory(self):
        controls = {}
        for precision in StandaloneSource.CONFIG:
            source = StandaloneSource(self.session, precision)
            for path in sorted((source.original / "csrc").rglob("*")):
                if not path.is_file():
                    continue
                for line, text in enumerate(path.read_text().splitlines(), 1):
                    for name in re.findall(r'(?:std::)?getenv\("(EQLIN_[A-Z0-9_]+)"\)', text):
                        controls.setdefault((precision, name), []).append({"path": str(path), "line": line,
                            "text": text.strip(), "sha256": Evidence.sha256(path)})
        numeric = {"EQLIN_WGRAD_CHAIN", "EQLIN_BWD_CHAIN", "EQLIN_FOLD_K", "EQLIN_SPLITK",
                   "EQLIN_SPLITK_ABLATION", "EQLIN_SWIZZLE", "EQLIN_PREP_X"}
        result = []
        for (precision, name), evidence in sorted(controls.items()):
            registered = [row["variant_id"] for row in self.rows if row["family"] == "standalone_" + precision and name in row["runtime_controls"]]
            status = "FINITE_NAMED_POLICIES_REGISTERED" if registered else (
                "DIAGNOSTIC_NOT_COMPLETE_OPERATION" if name in ("EQLIN_WGRAD_PREP_ONLY", "EQLIN_SPLITK_ABLATION") else
                "NUMERIC_ACTUAL_HISTORICAL_TUPLES_PENDING" if name in numeric else
                "DIAGNOSTIC_LOGGING_ONLY" if name == "EQLIN_DEBUG" else "UNCLASSIFIED_REQUIRES_REVIEW")
            result.append({"precision": precision, "control": name, "status": status,
                           "registered_variants": registered, "source_evidence": evidence})
        if any(row["status"] == "UNCLASSIFIED_REQUIRES_REVIEW" for row in result):
            raise ValueError("A source runtime control was silently omitted from the scope inventory")
        return result

    def historical_builds(self):
        history = Path(self.session.paths["OPT_ROOT"]).parents[1] / "h100/standalone"
        result = {}
        for precision, (relative, _, _) in StandaloneSource.CONFIG.items():
            for path in sorted((history / relative).rglob("*")):
                if not path.is_file() or path.suffix not in (".log", ".json", ".meta", ".md"):
                    continue
                for line, text in enumerate(path.read_text().splitlines(), 1):
                    if "nvcc " not in text or " -c " not in text:
                        continue
                    definitions = dict(re.findall(r"-D((?:F64M2_|F64_|EQLIN_)[A-Z0-9_]+)=([^\s]+)", text))
                    if not definitions:
                        continue
                    key = Evidence.fingerprint({"precision": precision, "definitions": definitions})
                    row = result.setdefault(key, {"tuple_id": "historical_" + precision + "_" + key[:12],
                        "precision": precision, "definitions": definitions, "status": "PRESERVED_COMPILE_COMMAND_NOT_H20_TESTED",
                        "producer_source_at_execution": "not proven by this compile log; current snapshot hash is bound separately",
                        "evidence": []})
                    row["evidence"].append({"path": str(path), "line": line, "text": text,
                                            "sha256": Evidence.sha256(path)})
        return list(result.values())

    def write(self, output):
        output.mkdir(parents=True, exist_ok=False)
        build_controls = []
        for precision in StandaloneSource.CONFIG:
            source = StandaloneSource(self.session, precision)
            for path in sorted(source.original.rglob("CMakeLists.txt")):
                for line, text in enumerate(path.read_text().splitlines(), 1):
                    if "CACHE STRING" in text and re.search(r"\b(?:EQLIN_|F64_|F64M2_)", text):
                        build_controls.append({"precision": precision, "path": str(path), "line": line, "text": text.strip(),
                            "status": "BUILD_KNOB_DISCOVERED_ACTUAL_HISTORICAL_TUPLES_PENDING",
                            "note": "Arbitrary numeric compiler values are not infinite extra versions"})
        value = {"classification": "EXISTING_NAMED_POLICIES_NOT_H20_RESULTS", "scope_status": "IN_SCOPE_REQUIRES_TESTING",
            "variants": self.rows, "kernel_aliases_or_ignored_parameters": self.aliases,
            "build_control_inventory": build_controls, "preserved_compile_tuples": self.historical_builds(),
            "runtime_control_inventory": self.runtime_inventory(), "all_existing_parameter_strategies_covered": False,
            "default_registry_matrix": {"variants": 28, "jobs": 700, "cells": 1725},
            "policy_registry_rows": len(self.rows), "source_sha256": Evidence.sha256(Path(__file__)),
            "coverage_remaining": "H20 results are tracked separately. Finite documented chain/fold/split/raster recipes are registered above; unresolved build recipes and numeric preparation launch caps remain in separate inventories. Arbitrary numeric values do not create infinite additional versions."}
        Evidence.write(output / "variants.json", value)
        fields = ("variant_id", "base_variant_id", "family", "operation", "runtime_controls", "api_controls", "effective_when", "probe_point",
                  "adapter_extension_required", "status", "semantic_contract_hash", "named_choice_evidence")
        with (output / "coverage_remaining.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (dict, list)) else row[key] for key in fields})
        print(json.dumps({"named_existing_variants": len(self.rows), "build_knobs_pending_tuple_resolution": len(build_controls), "GPU_work_started": False}))


class HistoricalTupleInventory:
    """Documented finite recipes; replay current preserved source, never invent old builds."""

    @staticmethod
    def write(session, output):
        if not output.resolve().is_relative_to(session.root):
            raise ValueError("Historical tuple inventory must stay in session")
        output.mkdir(parents=True, exist_ok=False)
        choices = ExistingChoices(session)
        source = StandaloneSource(session, "fp64")
        manifest = source.stage()
        recipes = (
            ("historic_grouped_split4", {"wgrad_split_k": 4}, "FP64_B11_REPORT.md", "不替换 B5/B9 grouped split4"),
            ("historic_b11_split4_kernel1", {"wgrad_split_k": 4, "wgrad_kernel": 1}, "FP64_B11_REPORT.md", "--wgrad-split-k 4 --wgrad-kernel 1"),
            ("historic_b12_split4_kernel2", {"wgrad_split_k": 4, "wgrad_kernel": 2}, "FP64_B12_REPORT.md", "--wgrad-split-k 4 --wgrad-kernel 2"),
            ("historic_b15_split4_transpose1", {"wgrad_split_k": 4, "wgrad_kernel": 0, "wgrad_transpose_mode": 1}, "FP64_B15_REPORT.md", "--wgrad-split-k 4 --wgrad-kernel 0 --wgrad-transpose-mode 1"))
        row_order = choices.evidence("fp64", "FP64_B33_REPORT.md", "stable-sorts rows")
        for suffix, controls, report, needle in recipes:
            evidence = choices.evidence("fp64", report, needle)
            for base in ("standalone_fp64_wgrad", "standalone_fp64_bwd"):
                choices.add(base, suffix, {}, controls, [evidence],
                    "reported finite split4 recipe; replay uses current preserved source and current unspecified defaults", True)
                row = choices.rows[-1]
                row["probe_point"] = {"config": "LARGE_P3", "batch": 512}
                row.update(strategy_kind="documented_historical_parameter_recipe", reported_api_controls=controls,
                    current_resolved_api_controls={"wgrad_tile": 0, "wgrad_split_k": 1,
                        "wgrad_kernel": 0, "wgrad_transpose_mode": 3, **controls},
                    historical_unrecorded_defaults=sorted({"wgrad_tile", "wgrad_split_k", "wgrad_kernel", "wgrad_transpose_mode"} - set(controls)),
                    reconstruction_status="CURRENT_PRESERVED_SOURCE_REPLAY_ONLY_ORIGINAL_PRODUCER_UNPROVEN",
                    current_source_manifest={"path": str(source.manifest_path), "sha256": Evidence.sha256(source.manifest_path),
                        "source_hash": manifest["source_hash"], "headers_hash": manifest["headers_hash"]},
                    reconstruction_limit="Reports omit the full original build/source revision and some defaults; current auto transpose and B33 row-order defaults may differ. These rows do not claim exact historical-binary reproduction.",
                    later_default_change_evidence=[row_order])
                row["semantic_contract_hash"] = Evidence.fingerprint({key: row[key] for key in
                    ("base_variant_id", "runtime_controls", "api_controls", "current_resolved_api_controls", "probe_point")})
        value = {"classification": "FINITE_HISTORICAL_RECIPES_NOT_H20_RESULTS", "variants": choices.rows,
            "additional_contracts": len(choices.rows), "full25_jobs": len(choices.rows) * 25,
            "full25_cells": 300, "original_157_scope_rewritten": False, "H100_rerun": False,
            "all_existing_parameter_strategies_covered": False, "source_sha256": Evidence.sha256(Path(__file__))}
        Evidence.write(output / "variants.json", value)
        fields = ("variant_id", "base_variant_id", "api_controls", "current_resolved_api_controls", "probe_point",
                  "reconstruction_status", "historical_unrecorded_defaults", "named_choice_evidence", "reconstruction_limit")
        with (output / "coverage_remaining.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            for row in choices.rows:
                writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (dict, list)) else row[key] for key in fields})
        print(json.dumps({"additional_contracts": len(choices.rows), "full25_jobs": 200, "GPU_work_started": False}))


class MatrixCLI:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        sub = parser.add_subparsers(dest="command", required=True)
        plan = sub.add_parser("plan")
        plan.add_argument("--session", type=Path, required=True)
        plan.add_argument("--cycle", required=True)
        plan.add_argument("--gpu", type=int, required=True)
        plan.add_argument("--stage", choices=("minimal", "capability", "full"), required=True)
        plan.add_argument("--additional-registry", type=Path)
        plan.add_argument("--variant-scope", choices=("all", "named"), default="all")
        plan.add_argument("--execution-model", choices=("point_workers", "same_build_point_batch"), default="point_workers")
        chunk = sub.add_parser("chunk")
        chunk.add_argument("--plan", type=Path, required=True)
        chunk.add_argument("--variant", required=True)
        chunk.add_argument("--config", choices=tuple(CONFIGS), required=True)
        worker = sub.add_parser("worker")
        worker.add_argument("--plan", type=Path, required=True)
        worker.add_argument("--job-id", required=True)
        worker.add_argument("--output", type=Path, required=True)
        worker.add_argument("--diagnostic-boundary", choices=StandaloneAdapter.BOUNDARIES)
        batch_worker = sub.add_parser("batch-worker")
        batch_worker.add_argument("--plan", type=Path, required=True)
        batch_worker.add_argument("--batch-id", required=True)
        compare = sub.add_parser("compare-batch")
        compare.add_argument("--plan", type=Path, required=True)
        compare.add_argument("--baseline-plan", type=Path, action="append", required=True)
        compare.add_argument("--precision", choices=tuple(StandaloneSource.CONFIG), required=True)
        compare.add_argument("--config", choices=tuple(CONFIGS), required=True)
        compare.add_argument("--batch", type=int, required=True)
        compare.add_argument("--output", type=Path, required=True)
        audit = sub.add_parser("audit")
        audit.add_argument("--plan", type=Path, required=True)
        audit.add_argument("--output", type=Path, required=True)
        audit.add_argument("--variant", help="one variant id, or a comma-separated explicit finite subset")
        audit.add_argument("--operation", choices=("forward", "grad_x", "grad_w", "grad_x_w", "backward"))
        audit.add_argument("--verification-scope", action="store_true", help="audit only the explicit independent verification point subset")
        run = sub.add_parser("run")
        run.add_argument("--plan", type=Path, required=True)
        run.add_argument("--precision", choices=tuple(StandaloneSource.CONFIG))
        run.add_argument("--variant", help="one variant id, or a comma-separated explicit finite subset")
        run.add_argument("--operation", choices=("forward", "grad_x", "grad_w", "grad_x_w", "backward"))
        run.add_argument("--config", choices=tuple(CONFIGS))
        run.add_argument("--max-groups", type=int, required=True)
        run.add_argument("--batch-deadline", type=float, required=True)
        run.add_argument("--label", required=True)
        check = sub.add_parser("selfcheck")
        check.add_argument("--session", type=Path, required=True)
        check.add_argument("--output", type=Path, required=True)
        choices = sub.add_parser("existing-choices")
        choices.add_argument("--session", type=Path, required=True)
        choices.add_argument("--output", type=Path, required=True)
        historical_tuples = sub.add_parser("historical-tuples")
        historical_tuples.add_argument("--session", type=Path, required=True)
        historical_tuples.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        if args.command == "selfcheck":
            return Selfcheck.run(args)
        if args.command == "existing-choices":
            session = Session(args.session)
            if not args.output.resolve().is_relative_to(session.root):
                raise ValueError("Existing policy registry must stay in session")
            ExistingChoices(session).build().write(args.output)
            return 0
        if args.command == "historical-tuples":
            HistoricalTupleInventory.write(Session(args.session), args.output)
            return 0
        if args.command == "plan":
            StandalonePlan.create(args)
            return 0
        if args.command == "worker":
            return MatrixWorker(args).run()
        if args.command == "batch-worker":
            return BatchWorker(args).run()
        if args.command == "compare-batch":
            return BatchEquivalence.run(args)
        if args.command == "run":
            plan = StandalonePlan(args.plan)
            controller = (BatchController if plan.value.get("execution_model") == "same_build_point_batch" else MatrixController)(args)
            with (controller.plan.root / "controller.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return controller.run()
        locked = StandalonePlan(args.plan)
        if args.command == "audit":
            if not args.output.resolve().is_relative_to(Path(locked.value["session"])):
                raise ValueError("Audit output must stay inside its session")
            point_keys = locked.value["verification"]["point_keys"] if args.verification_scope else None
            return 0 if MatrixAudit(locked).run(args.output, args.variant, args.operation, point_keys)["complete"] else 1
        args.session = Path(locked.value["session"])
        args.gpu = locked.value["gpu"]["physical_index"]
        args.attempt = locked.value["cycle"]
        locked.verify_producer(Path(__file__).parent)
        locked.verify_sources(Session(args.session))
        return StandaloneChunk(args).run()


if __name__ == "__main__":
    raise SystemExit(MatrixCLI.main())
