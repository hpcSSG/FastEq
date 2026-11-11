"""Bind already validated Luna public/prepared calls to the shared performance core."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from cases import points
from common import Evidence
from luna import LunaAdapter
from luna_autograd import LunaAutogradAdapter
from luna_autograd import AutogradAudit, AutogradPlan
from luna_matrix import MatrixAudit, MatrixPlan, SourceTree
from performance_bindings import BoundCall, TorchCalls, TorchIdentity
from qualification import AccuracyContracts


class LunaScope:
    VERSIONS = {"luna010": "0.1.0", "luna020": "0.2.0"}
    OPERATIONS = (*LunaAdapter.OPERATIONS, "forward_autograd")
    FAMILIES = tuple(prefix + "_" + operation for prefix in ("luna010", "luna020")
                     for operation in (*LunaAdapter.OPERATIONS, "forward_autograd"))

    @classmethod
    def parse(cls, family):
        prefix, separator, operation = family.partition("_")
        if not separator or prefix not in cls.VERSIONS or operation not in cls.OPERATIONS:
            raise ValueError("Unknown Luna version/operation family: " + family)
        return cls.VERSIONS[prefix], operation

    @staticmethod
    def boundaries(operation):
        return LunaAutogradAdapter.PUBLIC_BOUNDARIES if operation == "forward_autograd" else LunaAdapter.BOUNDARIES

    @staticmethod
    def variant(version):
        return "luna_" + version

    @classmethod
    def entry_key(cls, version, point_key, operation):
        return "/".join((cls.variant(version), point_key, operation))


class LunaContracts:
    NAME = "luna_accuracy_contract.json"

    @classmethod
    def create(cls, base_path, auto_path, output):
        plans = ((MatrixPlan(base_path), MatrixAudit, 400), (AutogradPlan(auto_path), AutogradAudit, 100))
        entries, epochs = {}, []
        for plan, audit_type, expected in plans:
            path = plan.root / "audit.json"
            report = json.loads(path.read_text())
            if (not report["complete"] or report["expected"] != expected or report["verified"] != expected
                    or report["missing"] or report["extra"] or report["shared_input_reference_mismatches"]):
                raise ValueError("Every accuracy epoch must be independently complete")
            reader = audit_type(plan)
            for job in plan.value["jobs"]:
                record = report["jobs"][job["job_id"]]
                attempt = Path(record["attempt"])
                fresh = reader.verify(job, attempt)
                raw_path = attempt / "raw/result.json"
                if not fresh["verified"] or Evidence.sha256(raw_path) != record["result_sha256"]:
                    raise ValueError("Latest independently audited accuracy bytes or binding changed")
                raw = json.loads(raw_path.read_text())
                identity = AccuracyContracts.identity(raw)
                point_key = "/".join((job["point"]["config"], str(job["point"]["batch"]), job["point"]["precision"]))
                key = LunaScope.entry_key(job["version"], point_key, job["operation"])
                if key in entries:
                    raise ValueError("Accuracy epochs overlap or duplicate an operation")
                entries[key] = {"variant_id": LunaScope.variant(job["version"]), "point_key": point_key,
                    "operation": job["operation"], "input_hash": raw["input_hash"], "reference_hashes": raw["reference_hashes"],
                    "production_source_hash": identity["source_hash"], "binary_hash": identity["binary_hash"],
                    "runtime_controls": identity.get("runtime_controls", {}), "api_controls": identity.get("api_controls", {}),
                    "required_boundaries": list(job["boundaries"]), "passed": fresh["accuracy_pass"],
                    "evidence": {"path": str(raw_path), "sha256": record["result_sha256"]},
                    "accuracy_epoch": plan.value["cycle_id"]}
            epochs.append({"plan": str(plan.path), "plan_hash": plan.value["plan_hash"], "expected": expected,
                           "audit": str(path), "audit_sha256": Evidence.sha256(path)})
        base, auto = plans[0][0], plans[1][0]
        if base.value["sources"] != auto.value["sources"] or base.value["producer"]["luna.py"] != auto.value["producer"]["luna.py"]:
            raise ValueError("Accuracy epochs changed the production source/base adapter")
        expected = {LunaScope.entry_key(version, point.key, operation) for version in LunaScope.VERSIONS.values()
                    for point in points() if point.precision in ("fp32", "fp64") for operation in LunaScope.OPERATIONS}
        if set(entries) != expected or len(entries) != 500:
            raise ValueError("Complete five-operation/two-version scope differs")
        value = {"schema": 1, "complete": True, "entries": entries, "epochs": epochs,
                 "sources": base.value["sources"], "base_adapter_sha256": base.value["producer"]["luna.py"],
                 "public_autograd_adapter_sha256": auto.value["producer"]["luna_autograd.py"],
                 "prepared_autograd": json.loads((auto.root / "audit.json").read_text())["capability_evidence"],
                 "note": "Two independently complete non-overlapping epochs: four original operations plus public forward/autograd."}
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if json.loads(output.read_text()) != value:
                raise FileExistsError("Existing source-bound contract is immutable")
        else:
            Evidence.write(output, value)
        print(json.dumps({"contract": str(output), "entries": 500, "sha256": Evidence.sha256(output)}))
        return value

    @classmethod
    def read(cls, session):
        path = session.root / "manifests" / cls.NAME
        value = json.loads(path.read_text())
        if not value.get("complete") or len(value.get("entries", {})) != 500:
            raise ValueError("Complete existing-adapter accuracy contract is required")
        return path, Evidence.sha256(path), value


class LunaFamily:
    FILES = ("luna_performance.py", "luna.py", "luna_probe.py", "luna_matrix.py",
             "luna_autograd.py", "luna_autograd_capability.py")
    PRECISION_CHUNKS = True

    @classmethod
    def configure_scope(cls, session):
        path, digest, value = LunaContracts.read(session)
        return {"evidence_files": {str(path): digest}, "accuracy_contract": str(path),
                "families": list(LunaScope.FAMILIES), "points_per_family": 50, "call_points": 2300,
                "base_adapter_sha256": value["base_adapter_sha256"],
                "prepared_autograd": value["prepared_autograd"],
                "required_affinity": {"allowed": [196], "selected": 196}}

    @classmethod
    def verify_scope(cls, expected, session):
        if cls.configure_scope(session) != expected:
            raise ValueError("Frozen Luna scope/accuracy evidence changed")

    @classmethod
    def expected_calls(cls, family):
        version, operation = LunaScope.parse(family)
        result = ["/".join((LunaScope.variant(version), operation, boundary)) for boundary in LunaScope.boundaries(operation)]
        result.append("/".join(("torch_ieee_full", operation, "public_warm")))
        if operation != "forward_autograd":
            result.append("/".join(("torch_ieee_mm", operation, "prepared_gemm")))
        return sorted(result)

    @staticmethod
    def accept_point(family, point):
        LunaScope.parse(family)
        return point.precision in ("fp32", "fp64")

    def __init__(self, case, session, family):
        import torch

        self.case, self.session = case, session
        if os.sched_getaffinity(0) != {196}:
            raise ValueError("Formal Luna process must inherit the reserved CPU196 affinity")
        self.version, self.operation = LunaScope.parse(family)
        if not self.accept_point(family, case.point):
            raise ValueError("Luna accepts only the existing FP32/FP64 scope")
        path, digest, contract = LunaContracts.read(session)
        source = contract["sources"][self.version]
        self.source_before = SourceTree.manifest(source["root"])
        if self.source_before != source or Evidence.sha256(Path(__file__).with_name("luna.py")) != contract["base_adapter_sha256"]:
            raise ValueError("Source/base adapter differs from complete tested accuracy")
        cache_operation = "forward" if self.operation == "forward_autograd" else self.operation
        cache = session.root / "cache/luna_matrix" / source["source_hash"] / contract["base_adapter_sha256"] / case.point.precision / cache_operation
        self.previous_environment = {key: os.environ.get(key) for key in ("LUNAEQ_CACHE_DIR", "LUNATP_CUTLASS_INCLUDE")}
        os.environ.update(LUNAEQ_CACHE_DIR=str(cache), LUNATP_CUTLASS_INCLUDE=str(Path(source["root"]) / "src/third_party/cutlass/include"))
        adapter_type = LunaAutogradAdapter if self.operation == "forward_autograd" else LunaAdapter
        self.adapter = adapter_type(case, source["root"], self.version,
            session.root / "manifests/luna_performance_jit" / self.version / self.operation)
        self.setup_metadata = {"cache": str(cache), "prepare_seconds": {}, "autograd_checks": {}}
        for boundary in LunaScope.boundaries(self.operation):
            start = time.perf_counter()
            self.adapter.prepare(self.operation, boundary)
            torch.cuda.synchronize()
            self.setup_metadata["prepare_seconds"][boundary] = time.perf_counter() - start
        binding = json.loads(json.dumps(self.adapter.identity(), allow_nan=False))
        binding.update(production_source_hash=source["source_hash"], binary_hash=Evidence.fingerprint(binding["jit_libraries"]),
                       runtime_controls={}, api_controls={}, accuracy_tier=case.point.precision)
        adapter_hashes = {name: Evidence.sha256(Path(__file__).with_name(name)) for name in ("luna.py", "luna_autograd.py", "luna_performance.py")}
        binding["source_hash"] = Evidence.fingerprint({"production": source["source_hash"], "adapters": adapter_hashes})
        binding["sampler_adapter_hashes"] = adapter_hashes
        if self.operation == "forward_autograd":
            if adapter_hashes["luna_autograd.py"] != contract["public_autograd_adapter_sha256"]:
                raise ValueError("Autograd adapter differs from complete tested accuracy")
            binding["public_autograd_controls"] = LunaAutogradAdapter.AUTOGRAD_CONTROLS
        key = LunaScope.entry_key(self.version, case.point.key, self.operation)
        entry = contract["entries"][key]
        if binding["binary_hash"] != entry["binary_hash"] or case.input_hash != entry["input_hash"]:
            raise ValueError("Actual cached DSO set/input differs from complete accuracy")
        binding["accuracy_contract"] = {"path": str(path), "sha256": digest, "entry_key": key}
        self.calls = [BoundCall(LunaScope.variant(self.version), self.operation, boundary,
            lambda bound=boundary: self.adapter.call(self.operation, bound),
            lambda bound=boundary: self.checked_call(bound),
            lambda value, bound=boundary: self.adapter.common_output(self.operation, bound, value),
            binding, self.description(boundary)) for boundary in LunaScope.boundaries(self.operation)]
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = TorchCalls(case)
        self.torch.prepare((self.operation,) if self.operation.startswith("grad_") else ())
        self.torch.full("forward")
        torch.cuda.synchronize()
        self.torch_binding = TorchIdentity.capture(session)
        self.torch_binding.update(accuracy_tier=case.point.precision,
            source_hash=Evidence.fingerprint({name: Evidence.sha256(Path(__file__).with_name(name))
                                            for name in ("performance_bindings.py", "cases.py")}))
        self.calls.extend(self.torch_calls(self.operation))
        if sorted(call.key for call in self.calls) != self.expected_calls(family):
            raise ValueError("Existing-adapter call set differs from declared scope")

    def checked_call(self, boundary):
        value = self.adapter.call(self.operation, boundary)
        if self.operation == "forward_autograd":
            graph = self.adapter.graph_evidence(boundary, value)
            if not self.adapter.valid_graph(graph):
                raise ValueError("Public autograd call lost its prebuilt input-connected graph")
            self.setup_metadata["autograd_checks"].setdefault(boundary, []).append(graph)
        return value

    def description(self, boundary):
        if boundary == "adapted_public_e2e":
            return "Original layouts; activation packing/output unpacking and backward layout/weight chain are inside the public call; weight values constant between samples."
        if boundary == "native_public":
            return "Native packed inputs/weights/dy; original-layout inversions and weight-gradient rescaling are outside this public native window."
        return "Retained fused raw binding on prepared native inputs; binding output/workspace allocations included; original-layout inversions and weight-gradient rescaling outside."

    def torch_calls(self, operation):
        calls = [BoundCall("torch_ieee_full", operation, "public_warm", lambda op=operation: self.torch.full(op),
                          lambda op=operation: self.torch.full(op), lambda value: value, self.torch_binding,
                          "Original layouts, full operation; forward/autograd input leaf prepared outside the window.")]
        if operation != "forward_autograd":
            calls.append(BoundCall("torch_ieee_mm", operation, "prepared_gemm", lambda op=operation: self.torch.mm(op),
                lambda op=operation: self.torch.mm(op, fresh=True), lambda value, op=operation: self.torch.normalize_mm(op, value),
                self.torch_binding, "Pregathered per-path matrix products; cg/reduction/layout normalization outside the window.",
                measurement_work="matrix_products_only"))
        return calls

    def pilot_calls(self):
        self.torch.prepare(("grad_x", "grad_w", "grad_x_w"))
        return [call for operation in LunaScope.OPERATIONS for call in self.torch_calls(operation)]

    def close(self):
        try:
            self.adapter.close()
            self.torch.graphs.clear()
            self.torch.products.clear()
            if SourceTree.manifest(self.source_before["root"]) != self.source_before:
                raise ValueError("Luna production source changed during measurement")
        finally:
            for key, value in self.previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class Cli:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        contract = commands.add_parser("contract")
        contract.add_argument("--base-plan", type=Path, required=True)
        contract.add_argument("--auto-plan", type=Path, required=True)
        contract.add_argument("--output", type=Path, required=True)
        commands.add_parser("selfcheck")
        args = parser.parse_args()
        if args.command == "contract":
            LunaContracts.create(args.base_plan, args.auto_plan, args.output)
            return 0
        accepted = [point for point in points() if point.precision in ("fp32", "fp64")]
        assert len(LunaScope.FAMILIES) == 10 and len(accepted) == 50
        assert sum(len(LunaFamily.expected_calls(family)) * len(accepted) for family in LunaScope.FAMILIES) == 2300
        for family in LunaScope.FAMILIES:
            version, operation = LunaScope.parse(family)
            keys = LunaFamily.expected_calls(family)
            assert len(keys) == len(set(keys)) == (3 if operation == "forward_autograd" else 5)
            assert all(LunaFamily.accept_point(family, point) == (point.precision in ("fp32", "fp64")) for point in points())
        print(json.dumps({"status": "PASS", "families": list(LunaScope.FAMILIES), "points_per_family": 50,
                          "call_points": 2300, "precision_chunks": True}))
        return 0


if __name__ == "__main__":
    raise SystemExit(Cli.main())
