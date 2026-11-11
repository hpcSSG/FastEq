"""Luna public forward-with-autograd and its separate complete accuracy supplement."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from common import Evidence
from luna import LunaAdapter
from luna_matrix import Cli, MatrixAudit, MatrixController, MatrixPlan, MatrixWorker


class LunaAutogradAdapter(LunaAdapter):
    PUBLIC_BOUNDARIES = ("native_public", "adapted_public_e2e")
    AUTOGRAD_CONTROLS = {"x_requires_grad": True, "weight_requires_grad": False,
                         "x_leaf_prepared_outside_window": True}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.autograd_x = {"native_public": self.x_native.detach().requires_grad_(True),
                           "adapted_public_e2e": self.case.x.detach().requires_grad_(True)}
        if self.w_native.requires_grad:
            raise ValueError("Public forward/autograd requires constant weights")

    def prepare(self, operation, boundary):
        if operation == "forward_autograd":
            return self.call(operation, boundary)
        return super().prepare(operation, boundary)

    def call(self, operation, boundary):
        if operation != "forward_autograd":
            return super().call(operation, boundary)
        import torch

        if boundary not in self.PUBLIC_BOUNDARIES:
            raise ValueError("Raw prepared launcher has no declared public autograd boundary")
        with torch.enable_grad():
            x = self.autograd_x[boundary]
            return self.native_forward(x) if boundary == "native_public" else self.adapted_forward(x)

    def common_output(self, operation, boundary, value):
        return super().common_output("forward" if operation == "forward_autograd" else operation, boundary, value)

    def graph_evidence(self, boundary, value):
        leaf = self.autograd_x[boundary]
        function = value.grad_fn
        return {"x_requires_grad": leaf.requires_grad, "x_is_leaf": leaf.is_leaf,
                "weight_requires_grad": self.w_native.requires_grad, "output_requires_grad": value.requires_grad,
                "grad_fn": None if function is None else type(function).__module__ + "." + type(function).__name__,
                "leaf_data_ptr": leaf.data_ptr(), "leaf_preparation": "outside_call"}

    @staticmethod
    def valid_graph(value):
        return (all(value.get(key) is True for key in ("x_requires_grad", "x_is_leaf", "output_requires_grad"))
                and value.get("weight_requires_grad") is False and bool(value.get("grad_fn"))
                and value.get("leaf_preparation") == "outside_call" and value.get("leaf_data_ptr", 0) > 0)

    def identity(self):
        value = super().identity()
        value.update(autograd_adapter_sha256=Evidence.sha256(Path(__file__)), api_controls=self.AUTOGRAD_CONTROLS)
        return value


class AutogradPlan(MatrixPlan):
    CLASSIFICATION = "FULL_25_POINT_LUNA_FORWARD_AUTOGRAD_SUPPLEMENT_NOT_TIMING"
    FILES = (*MatrixPlan.FILES, "luna_autograd.py", "luna_autograd_capability.py")
    OPERATIONS = ("forward_autograd",)
    BOUNDARIES = LunaAutogradAdapter.PUBLIC_BOUNDARIES

    @classmethod
    def jobs(cls, policy):
        class ForwardPolicy:
            def for_job(self, version, precision, config, operation):
                value = dict(policy.for_job(version, precision, config, "forward"))
                value["public_autograd_uses_same_forward_JIT"] = True
                return value
        return super().jobs(ForwardPolicy())

    @classmethod
    def extra_fields(cls, args, session, value):
        base = MatrixPlan(args.base_accuracy_plan)
        audit_path = base.root / "audit.json"
        audit = json.loads(audit_path.read_text())
        if not audit["complete"] or audit["expected"] != 400 or audit["verified"] != 400:
            raise ValueError("Supplement requires an independently complete four-operation matrix")
        if base.value["sources"] != value["sources"] or base.value["producer"]["luna.py"] != value["producer"]["luna.py"]:
            raise ValueError("Public-autograd supplement changed the production source or base adapter")
        return {"base_accuracy": {"plan": str(base.path), "plan_hash": base.value["plan_hash"],
                                  "audit": str(audit_path), "audit_sha256": Evidence.sha256(audit_path)},
                "public_autograd_controls": LunaAutogradAdapter.AUTOGRAD_CONTROLS,
                "prepared_autograd": {"status": "PENDING_SOURCE_AND_MINIMAL_CAPABILITY_PROBES",
                                      "not_in_public_accuracy_cell_count": True},
                "minimum_capability_points": ["SMALL_P1/512/fp32", "SMALL_P1/512/fp64"]}


class AutogradWorker(MatrixWorker):
    PLAN_TYPE = AutogradPlan
    ADAPTER_TYPE = LunaAutogradAdapter

    def __init__(self, args):
        super().__init__(args)
        self.graph_pairs = {}
        for name in ("luna_autograd.py", "luna_autograd_capability.py"):
            source = Path(__file__).with_name(name)
            self.producer_files[str(source)] = Evidence.sha256(source)
            shutil.copy2(source, self.output / "producer" / name)

    def observe_pair(self, operation, boundary, first, second):
        pair = [self.adapter.graph_evidence(boundary, value) for value in (first, second)]
        if not all(self.adapter.valid_graph(value) for value in pair):
            raise ValueError("Public forward/autograd did not build an input-connected output graph")
        if pair[0]["leaf_data_ptr"] != pair[1]["leaf_data_ptr"]:
            raise ValueError("Forward/autograd recreated its prebuilt input leaf inside calls")
        self.graph_pairs[boundary] = pair

    def observe_result(self, row, output):
        row["autograd_evidence"] = self.graph_pairs[row["boundary"]]
        super().observe_result(row, output)


class AutogradAudit(MatrixAudit):
    PLAN_TYPE = AutogradPlan

    @classmethod
    def check_rows(cls, job, rows):
        reasons = super().check_rows(job, rows)
        for row in rows:
            pair = row.get("autograd_evidence", [])
            if (len(pair) != 2 or not all(LunaAutogradAdapter.valid_graph(value) for value in pair)
                    or pair[0]["leaf_data_ptr"] != pair[1]["leaf_data_ptr"]):
                reasons.append("public autograd graph/input/weight/window contract differs")
        return reasons

    def verify_raw(self, job, directory):
        record = super().verify_raw(job, directory)
        result = json.loads((directory / "raw/result.json").read_text())
        identity = result.get("identity", {})
        if (identity.get("autograd_adapter_sha256") != self.plan.value["producer"]["luna_autograd.py"]
                or identity.get("api_controls") != LunaAutogradAdapter.AUTOGRAD_CONTROLS):
            record["reason"].append("autograd adapter/leaf/constant-weight controls differ")
            record["verified"] = False
        return record

    def capability_evidence(self):
        path = self.plan.root / "capabilities/audit.json"
        if not path.is_file():
            return {"complete": False, "reason": "missing four minimum capability probes"}
        value = json.loads(path.read_text())
        expected = {job["job_id"] for job in self.plan.value["jobs"]
                    if job["point"]["config"] == "SMALL_P1" and job["point"]["batch"] == 512}
        observed = [row["job_id"] for row in value.get("jobs", [])]
        if (value.get("status") != "PASS" or value.get("plan_hash") != self.plan.value["plan_hash"]
                or set(observed) != expected or len(observed) != 4):
            return {"complete": False, "reason": "minimum capability scope incomplete/duplicated"}
        for record in value["jobs"]:
            raw_path = Path(record["raw"])
            if not record["verified"] or Evidence.sha256(raw_path) != record["sha256"]:
                return {"complete": False, "reason": "capability raw bytes changed"}
            raw = json.loads(raw_path.read_text())
            job = self.plan.job(record["job_id"])
            if (raw.get("status") != "PASS" or raw.get("diagnostic_reproduction") is not False
                    or raw.get("producer") != self.plan.value["producer"]
                    or raw.get("plan_hash") != self.plan.value["plan_hash"]
                    or raw.get("source_unchanged") is not True or raw.get("producer_unchanged") is not True
                    or raw.get("GPU_UUID") != self.plan.value["gpu"]["uuid"] or raw.get("point") != job["point"]
                    or raw.get("source_manifest_hash") != self.plan.value["sources"][job["version"]]["source_hash"]):
                return {"complete": False, "reason": "capability source/input/producer/GPU identity differs"}
            self.plan.verify_producer(raw_path.parent / "producer")
            diagnostics = [{**row, "operation": "grad_x", "autograd_evidence": row["graphs"]}
                           for row in raw["gradient_diagnostic_rows"]]
            if (self.check_rows({**job, "operation": "grad_x"}, diagnostics)
                    or not all(row.get("dy_nonzero") is True for row in diagnostics)):
                return {"complete": False, "reason": "gradient graph/numeric diagnostic gate differs"}
            old = raw["c03_dgrad_agreement"]
            if not old["exact_match"] or Evidence.sha256(Path(old["evidence"])) != old["sha256"]:
                return {"complete": False, "reason": "complete c03 real-dy dgrad agreement missing"}
            for identity in (raw["training_identity"], raw["raw_prepared_identity"]):
                if (identity.get("backend") != "cuda_fused" or identity.get("version") != job["version"]
                        or not identity.get("jit_libraries") or not set(identity["jit_libraries"]) <= set(identity["mapped_jit_libraries"])):
                    return {"complete": False, "reason": "capability actual mapped fused backend not proved"}
                if any(Evidence.sha256(Path(filename)) != digest for filename, digest in identity["jit_libraries"].items()):
                    return {"complete": False, "reason": "capability DSO changed"}
            prepared = raw["prepared_autograd"]
            if (prepared.get("classification") != "N/A_RAW_LAUNCH_HAS_NO_AUTOGRAD_GRAPH"
                    or prepared.get("input_requires_grad") is not True or prepared.get("weight_requires_grad") is not False
                    or prepared.get("output_grad_fn") is not None or not prepared["source_witness"]["compiled_binding_sources"]):
                return {"complete": False, "reason": "prepared autograd N/A lacks real source/input/graph evidence"}
            for source in prepared["source_witness"]["compiled_binding_sources"]:
                if Evidence.sha256(Path(source["path"])) != source["sha256"]:
                    return {"complete": False, "reason": "compiled prepared binding source changed"}
                for path_key, digest_key in (("request", "request_sha256"), ("DSO_load_receipt", "DSO_load_receipt_sha256")):
                    if path_key in source and Evidence.sha256(Path(source[path_key])) != source[digest_key]:
                        return {"complete": False, "reason": "prepared DSO-to-source receipt changed"}
            reference = raw["reference_store"]
            metadata_path = Path(reference["path"])
            metadata = json.loads(metadata_path.read_text())
            identity = metadata["identity"]
            if (Evidence.sha256(metadata_path) != reference["sha256"]
                    or Evidence.sha256(metadata_path.parent / "references.pt") != metadata["file_sha256"]
                    or metadata["value_hashes"]["fp64"] != raw["reference_hashes"]
                    or identity["input_hash"] != raw["input_hash"] or identity["tensor_hashes"] != raw["tensor_hashes"]
                    or identity["layouts"] != raw["layouts"] or identity["point"] != job["point"]
                    or identity["hardware"]["uuid"] != raw["GPU_UUID"]
                    or identity["math"] != self.plan.value["reference_math"]
                    or identity["reference_producer_sha256"] != self.plan.value["producer"]["cases.py"]
                    or identity["store_producer_sha256"] != self.plan.value["producer"]["reference_store.py"]):
                return {"complete": False, "reason": "capability canonical reference evidence differs"}
        return {"complete": True, "probes": 4, "prepared_autograd": "N/A_RAW_LAUNCH_HAS_NO_AUTOGRAD_GRAPH",
                "audit": str(path), "sha256": Evidence.sha256(path),
                "note": "Strict FP32 dgrad failures stay recorded; connectivity does not promote failed training accuracy."}

    def run(self):
        report = super().run()
        report["public_accuracy_complete"] = report["complete"]
        try:
            capability = self.capability_evidence()
        except (OSError, ValueError, KeyError, TypeError) as error:
            capability = {"complete": False, "reason": type(error).__name__ + ": " + str(error)}
        report["capability_evidence"] = capability
        report["complete"] = report["public_accuracy_complete"] and capability["complete"]
        Evidence.write(self.plan.root / "audit.json", report)
        return report


class AutogradController(MatrixController):
    PLAN_TYPE = AutogradPlan
    AUDIT_TYPE = AutogradAudit
    ENTRYPOINT = "luna_autograd.py"

    @staticmethod
    def cache_operation(job):
        return "forward"


class AutogradSelfcheck:
    @classmethod
    def run(cls, output):
        from copy import deepcopy
        from types import SimpleNamespace
        from luna_autograd_capability import CapabilityController

        class Policy:
            def for_job(self, *args):
                return {"seconds": 120, "basis": "CPU_SYNTHETIC_ONLY"}
        jobs = AutogradPlan.jobs(Policy())
        AutogradPlan.verify_scope(jobs)
        assert len(jobs) == 100 and sum(len(AutogradPlan.cells(job)) for job in jobs) == 200
        try:
            AutogradPlan.verify_scope(jobs[:-1])
        except ValueError:
            pass
        else:
            raise AssertionError("Partial autograd supplement accepted")
        evidence = {"x_requires_grad": True, "x_is_leaf": True, "weight_requires_grad": False,
                    "output_requires_grad": True, "grad_fn": "CPU_SYNTHETIC", "leaf_data_ptr": 1,
                    "leaf_preparation": "outside_call"}
        rows = [{"operation": "forward_autograd", "result": "forward", "boundary": boundary,
                 "rel_fro": 1e-7, "torch_rel_fro": 1e-6, "threshold": 1.05e-6,
                 "finite": True, "reference_finite": True, "repeat_equal": True, "different_buffer": True,
                 "torch_baseline_valid": True, "passed": True, "accuracy_pass": True, "accuracy_tier": "fp32",
                 "autograd_evidence": [dict(evidence), dict(evidence)]} for boundary in AutogradPlan.BOUNDARIES]
        assert not AutogradAudit.check_rows(jobs[0], rows)
        for key, value in (("grad_fn", None), ("weight_requires_grad", True), ("x_is_leaf", False),
                           ("output_requires_grad", False), ("leaf_preparation", "inside_call"), ("leaf_data_ptr", 2)):
            changed = deepcopy(rows)
            changed[0]["autograd_evidence"][1][key] = value
            assert AutogradAudit.check_rows(jobs[0], changed)
        changed = deepcopy(rows)
        changed.append({**changed[0], "boundary": "prepared_native_launch_with_alloc"})
        assert AutogradAudit.check_rows(jobs[0], changed)
        regular, capability = AutogradController.__new__(AutogradController), CapabilityController.__new__(CapabilityController)
        regular.plan = capability.plan = SimpleNamespace(value={"cycle_id": "CPU_SYNTHETIC"})
        assert regular.log_name(jobs[0], Path("attempt0001")) != capability.log_name(jobs[0], Path("attempt0001"))
        output.mkdir(parents=True, exist_ok=False)
        Evidence.write(output / "result.json", {"status": "PASS", "classification": "CPU_SYNTHETIC_NOT_GPU",
                      "jobs": 100, "cells": 200, "checks": ["complete_scope", "partial_rejected",
                      "no_graph_rejected", "nonleaf_rejected", "variable_weight_rejected", "inside_window_leaf_rejected",
                      "recreated_leaf_rejected", "prepared_autograd_label_rejected", "capability_accuracy_log_namespace"]})
        print(json.dumps({"status": "PASS", "jobs": 100, "cells": 200}))
        return 0


class AutogradCli(Cli):
    PLAN_TYPE = AutogradPlan
    CONTROLLER_TYPE = AutogradController
    WORKER_TYPE = AutogradWorker
    AUDIT_TYPE = AutogradAudit
    SELF_CHECK_TYPE = AutogradSelfcheck

    @classmethod
    def configure_plan(cls, parser):
        parser.add_argument("--base-accuracy-plan", type=Path, required=True)


if __name__ == "__main__":
    raise SystemExit(AutogradCli.main())
