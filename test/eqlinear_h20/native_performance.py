"""Source-bound native-selector factories for the shared H20 sampler."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from common import Evidence
from native_bridge import NativeAdapter, NativeBridge
from native_cycle import NativePlan
from performance_bindings import BoundCall, TorchCalls, TorchIdentity
from session import Session
from standalone_performance import StandalonePerformanceFamily
from standalone_qualification import StandaloneAccuracyContracts


class NativeAccuracyContracts:
    @classmethod
    def create(cls, plan_path, audit_path, output):
        from qualification import AccuracyContracts

        plan = NativePlan(plan_path)
        audit = json.loads(audit_path.read_text())
        if (output.exists() or not output.resolve().is_relative_to(plan.session.root) or
                audit.get("complete") is not True or audit.get("full_registry_complete") is not True or
                audit.get("plan_hash") != plan.value["plan_hash"] or len(plan.value["jobs"]) != 475 or
                set(audit["records"]) != {job["attempt"] for job in plan.value["jobs"]}):
            raise ValueError("Only a complete, disjoint475-point native audit can provide contracts")
        entries = {}
        for job in plan.value["jobs"]:
            record = audit["records"][job["attempt"]]
            path = Path(record["result"])
            if record.get("verified") is not True or Evidence.sha256(path) != record["result_sha256"]:
                raise ValueError("Original native evidence differs from the complete independent audit")
            result = json.loads(path.read_text())
            identity = AccuracyContracts.identity(result)
            variant = f"native_{job['precision']}_impl{job['impl']}"
            point = "/".join(map(str, (job["point"]["config"], job["point"]["batch"], job["precision"])))
            key = StandaloneAccuracyContracts.entry_key(variant, point, "forward")
            entries[key] = {"variant_id": variant, "point_key": point, "operation": "forward",
                "input_hash": result["input_hash"], "reference_hashes": result["reference_hashes"],
                "production_source_hash": identity["source_hash"], "binary_hash": identity["binary_hash"],
                "runtime_controls": identity.get("runtime_controls", {}), "api_controls": {},
                "required_boundaries": list(NativeAdapter.BOUNDARIES), "passed": record["accuracy_pass"],
                "reason": "FULL_CONTRACT_PASS" if record["accuracy_pass"] else "UPSTREAM_FULL_OPERATION_ACCURACY_FAIL",
                "evidence": {"path": str(path), "sha256": record["result_sha256"]},
                "upstream_gpu_uuid": result["GPU_UUID"], "upstream_plan_hash": plan.value["plan_hash"]}
        output.parent.mkdir(parents=True, exist_ok=True)
        producer = output.with_suffix(".producer.py")
        if producer.exists():
            raise FileExistsError(producer)
        shutil.copy2(Path(__file__), producer)
        Evidence.write(output, {"schema": 1, "complete": True, "entries": entries, "variants": 19,
            "expected_entries": 475, "qualified_entries": sum(row["passed"] for row in entries.values()),
            "original_result_files_modified": False, "partial_epoch_mixing": False,
            "evidence_files": {str(path.resolve()): Evidence.sha256(path) for path in (plan_path, audit_path, producer)},
            "qualification_scope": "Base numeric/repeat/buffer contract; unmatched H100 gate remains unverified"})
        return 0


class NativePerformanceScope:
    @staticmethod
    def pointer(session):
        return session.root / "config/native_performance_scope.json"

    @classmethod
    def create(cls, session, contracts_path, cycle, build_id="bridge01"):
        import re

        if re.fullmatch(r"[A-Za-z0-9_.-]+", cycle) is None:
            raise ValueError("Use a fresh simple native scope identity")
        contracts = json.loads(contracts_path.read_text())
        if contracts.get("schema") != 1 or contracts.get("complete") is not True or len(contracts["entries"]) != 475:
            raise ValueError("Native performance requires the complete upstream475-point qualification manifest")
        root = session.root / "performance/native_scope" / cycle
        root.mkdir(parents=True, exist_ok=False)
        files = {**contracts.get("evidence_files", {}), str(contracts_path.resolve()): Evidence.sha256(contracts_path)}
        sources, families = {}, {}
        for precision, config in NativeBridge.CONFIG.items():
            bridge = NativeBridge(session, precision, build_id)
            receipt = bridge.verify()
            target = root / (precision + "_build.json")
            shutil.copy2(bridge.manifest_path, target)
            contract = receipt["contract"]
            for name in ("source_files", "native_files", "dependency_files"):
                files.update(contract[name])
            files.update({str(target): Evidence.sha256(target), str(bridge.manifest_path): Evidence.sha256(bridge.manifest_path),
                str(bridge.library): receipt["binary_sha256"], str(bridge.inventory_path): contract["inventory_sha256"],
                contract["current_build"]: contract["current_build_sha256"],
                str(Path(contract["build_graph"]) / "build.ninja"): contract["build_ninja_sha256"],
                str(Path(contract["build_graph"]) / "CMakeCache.txt"): contract["cmake_cache_sha256"],
                receipt["generated_source"]: receipt["generated_source_sha256"]})
            for stage in ("elf", "exports", "cuda_arch"):
                item = receipt[stage + "_receipt"]["stdout"]
                files[item["path"]] = item["sha256"]
            sources[precision] = {"manifest": str(bridge.manifest_path), "manifest_sha256": Evidence.sha256(bridge.manifest_path),
                "binary": str(bridge.library), "binary_hash": receipt["binary_sha256"],
                "production_source_hash": Evidence.fingerprint({"source_commit": contract["source_commit"],
                    "native_objects_sha256": Evidence.fingerprint(contract["native_files"])})}
            families["native_" + precision] = {"precision": precision, "runtime_controls": {}, "variants": [
                {"variant_id": f"native_{precision}_impl{impl}", "precision": precision, "impl": impl,
                 "operation": "forward", "boundaries": list(NativeAdapter.BOUNDARIES)} for impl in config[4]]}
        value = {"schema": 1, "classification": "NATIVE_SELECTOR_PERFORMANCE_SCOPE_NOT_RESULTS", "build_id": build_id,
            "families": families, "sources": sources, "variants": 19,
            "accuracy_contracts": {"path": str(contracts_path.resolve()), "sha256": Evidence.sha256(contracts_path)},
            "evidence_files": files, "constructor_boundary": "constructor_and_prepare_weight",
            "prepare_weight_separate_api": False, "prepare_act_separate_api": True,
            "native_forward_autograd": "test-only low-level bridge is not a production public autograd wrapper"}
        value["scope_hash"] = Evidence.fingerprint(value)
        path = root / "scope.json"
        Evidence.write(path, value)
        Evidence.write(cls.pointer(session), {"path": str(path), "sha256": Evidence.sha256(path)})
        return path

    @classmethod
    def verify(cls, metadata):
        from cases import points

        scope = metadata["scope"]
        if scope["scope_hash"] != Evidence.fingerprint({key: value for key, value in scope.items() if key != "scope_hash"}):
            raise ValueError("Native finite scope changed")
        NativeBridge.verify_files(metadata["evidence_files"])
        if json.loads(Path(metadata["scope_path"]).read_text()) != scope:
            raise ValueError("Embedded native scope differs from its immutable file")
        expected = set()
        for precision, config in NativeBridge.CONFIG.items():
            group = scope["families"]["native_" + precision]
            if [row["impl"] for row in group["variants"]] != list(config[4]) or group["runtime_controls"]:
                raise ValueError("A native selector was omitted, duplicated or supplied undeclared controls")
            for row in group["variants"]:
                if row != {"variant_id": f"native_{precision}_impl{row['impl']}", "precision": precision, "impl": row["impl"],
                           "operation": "forward", "boundaries": list(NativeAdapter.BOUNDARIES)}:
                    raise ValueError("A native operation or boundary differs from its source-backed registry")
                expected.update(StandaloneAccuracyContracts.entry_key(row["variant_id"], point.key, "forward")
                    for point in points() if point.precision == precision)
        link = scope["accuracy_contracts"]
        if set(json.loads(Path(link["path"]).read_text())["entries"]) != expected or len(scope["families"]) != 3:
            raise ValueError("Native upstream accuracy scope is incomplete or contains unrelated variants")
        return scope


class NativeColdCalls:
    def __init__(self, adapter):
        self.adapter, self.pending = adapter, []

    def create(self):
        a, case = self.adapter, self.adapter.case
        self.pending.append(a.ops.create(case.x3, case.w3, case.dims, case.cg, a.impl))

    def cleanup(self):
        while self.pending:
            self.adapter.ops.destroy(self.pending.pop())

    def warm(self):
        self.create()
        self.cleanup()

    def check(self):
        import torch

        a, case = self.adapter, self.adapter.case
        handle = a.ops.create(case.x3, case.w3, case.dims, case.cg, a.impl)
        try:
            a.ops.prepare_act(handle, case.x3)
            output = StandalonePerformanceFamily.fresh_output(a, "forward")
            a.ops.launch(handle, output)
            torch.cuda.synchronize()
            return output.reshape_as(case.grad)
        finally:
            a.ops.destroy(handle)


class NativePerformanceFamily(StandalonePerformanceFamily):
    FILES = ("native_performance.py", "native_bridge.py", "native_bridge.cpp.in", "native_cycle.py",
             "standalone_performance.py", "standalone_qualification.py")
    SCOPE = None

    @classmethod
    def configure_scope(cls, session):
        pointer = json.loads(NativePerformanceScope.pointer(session).read_text())
        path = Path(pointer["path"])
        if Evidence.sha256(path) != pointer["sha256"]:
            raise ValueError("Native scope pointer changed")
        scope = json.loads(path.read_text())
        metadata = {"scope": scope, "scope_path": str(path), "evidence_files": {**scope["evidence_files"], str(path): pointer["sha256"]}}
        cls.SCOPE = NativePerformanceScope.verify(metadata)
        return metadata

    @classmethod
    def verify_scope(cls, metadata, session):
        cls.SCOPE = NativePerformanceScope.verify(metadata)

    @staticmethod
    def components(variant):
        return ["cold_create", "prepare_act"]

    @staticmethod
    def invoke(adapter, boundary, output=None):
        import torch
        with torch.no_grad():
            return adapter.call("forward", boundary, output)

    def __init__(self, case, session, family):
        import torch

        self.case, self.session, self.family = case, session, family
        self.group = self.specification(family)
        if not self.accept_point(family, case.point):
            raise ValueError("Native family consumes a different fixture precision")
        self.previous_controls = {key: value for key, value in os.environ.items() if key.startswith(("EQLIN_", "FASTEQ_EQLINEAR_"))}
        for key in self.previous_controls:
            os.environ.pop(key)
        self.adapters, self.outputs, self.calls, self.cold, self.setup_metadata = [], {}, [], [], {}
        bridge = NativeBridge(session, case.point.precision, self.SCOPE["build_id"])
        receipt = bridge.verify()
        source = self.SCOPE["sources"][case.point.precision]
        if Evidence.sha256(bridge.manifest_path) != source["manifest_sha256"] or receipt["binary_sha256"] != source["binary_hash"]:
            raise ValueError("Native actual bridge differs from the frozen performance source")
        self.torch = TorchCalls(case)
        self.torch.prepare(())
        self.torch.full("forward")
        self.torch_identity = TorchIdentity.capture(session)
        self.torch_identity.update(accuracy_tier="fp64" if case.point.precision == "fp64" else "fp32",
            source_hash=Evidence.fingerprint({name: Evidence.sha256(Path(__file__).with_name(name))
                for name in ("cases.py", "performance_bindings.py")}))
        for variant in self.group["variants"]:
            adapter = NativeAdapter(case, bridge, variant["impl"], receipt)
            self.adapters.append(adapter)
            self.outputs[variant["variant_id"]] = adapter.new_output("forward")
            output = self.outputs[variant["variant_id"]]
            for boundary in variant["boundaries"]:
                adapter.prepare("forward", boundary)
                self.calls.append(BoundCall(variant["variant_id"], "forward", boundary,
                    lambda a=adapter, b=boundary, o=output: self.invoke(a, b, None if b == "forward_allocating" else o),
                    lambda a=adapter, b=boundary: self.invoke(a, b, None if b == "forward_allocating" else self.fresh_output(a, "forward")),
                    lambda value: value, {}, "original native bridge boundaries; low-level timed calls reuse preallocated output"))
            cold = NativeColdCalls(adapter)
            self.cold.append(cold)
            self.calls.append(BoundCall(variant["variant_id"], "forward", "cold_create", cold.create, cold.check,
                lambda value: value, {}, "existing create includes Runner construction and weight preparation; destruction outside timing",
                after_sample=cold.cleanup, warm_function=cold.warm, measurement_work="constructor_and_prepare_weight"))
            self.calls.append(BoundCall(variant["variant_id"], "forward", "prepare_act",
                lambda a=adapter: a.ops.prepare_act(a.handle, case.x3),
                lambda a=adapter: self.check_preparation(a), lambda value: value, {},
                "existing activation preparation; subsequent launch validates the component outside its timing window", measurement_work="prepare_act"))
            identity = adapter.identity()
            identity.update(source_hash=source["production_source_hash"], production_source_hash=source["production_source_hash"],
                binary_hash=source["binary_hash"], api_controls={}, execution_grad_mode="no_grad", accuracy_tier=case.point.precision,
                performance_scope_hash=self.SCOPE["scope_hash"], variant_specification=variant,
                accuracy_contract={**self.SCOPE["accuracy_contracts"],
                    "entry_key": StandaloneAccuracyContracts.entry_key(variant["variant_id"], case.point.key, "forward")},
                sampler_adapter_hashes={name: Evidence.sha256(Path(__file__).with_name(name)) for name in self.FILES})
            identity["source_hash"] = Evidence.fingerprint({"production": identity["production_source_hash"],
                "adapters": identity["sampler_adapter_hashes"], "variant": variant, "scope": self.SCOPE["scope_hash"]})
            for call in self.calls:
                if call.variant == variant["variant_id"]:
                    call.binding = identity
        self.calls.extend(self.torch_calls(("forward",)))
        torch.cuda.synchronize()
        if sorted(call.key for call in self.calls) != self.expected_calls(family):
            raise ValueError("Actual native call set differs from the immutable source-backed declaration")

    def check_preparation(self, adapter):
        adapter.ops.prepare_act(adapter.handle, self.case.x3)
        return self.invoke(adapter, "prepared_gemm", self.fresh_output(adapter, "forward"))

    def pilot_calls(self):
        return self.torch_calls(("forward",))

    def close(self):
        import torch
        torch.cuda.synchronize()
        for cold in self.cold:
            cold.cleanup()
        for adapter in self.adapters:
            adapter.close()
        self.adapters.clear()
        self.outputs.clear()
        self.calls.clear()
        self.cold.clear()
        self.torch.products.clear()
        self.torch.graphs.clear()
        for key in list(os.environ):
            if key.startswith(("EQLIN_", "FASTEQ_EQLINEAR_")):
                os.environ.pop(key)
        os.environ.update(self.previous_controls)


class NativePerformanceCLI:
    @classmethod
    def selfcheck(cls, session, output):
        from cases import points
        from unittest.mock import patch
        import torch

        if output is None or output.exists() or not output.resolve().is_relative_to(session.root):
            raise ValueError("Native CPU selfcheck needs a fresh session output")
        metadata = NativePerformanceFamily.configure_scope(session)
        scope = metadata["scope"]
        variants = [row for group in scope["families"].values() for row in group["variants"]]
        assert len(variants) == 19 and len({row["variant_id"] for row in variants}) == 19
        for family, group in scope["families"].items():
            assert sum(NativePerformanceFamily.accept_point(family, point) for point in points()) == 25
            calls = NativePerformanceFamily.expected_calls(family)
            assert len(calls) == len(group["variants"]) * 5 + 2 and len(calls) == len(set(calls))
        for impl in (1, 2):
            try:
                NativeBridge.check_impl("tf32", impl)
            except ValueError:
                pass
            else:
                raise AssertionError("Excluded TF32x1 selector was admitted")
        class SyntheticOps:
            def __init__(self):
                self.events = []
            def create(self, *args):
                self.events.append("constructor_and_prepare_weight")
                return 42
            def prepare_act(self, *args):
                self.events.append("prepare_act")
            def launch(self, handle, output):
                self.events.append("launch")
                output.fill_(1)
            def destroy(self, *args):
                self.events.append("destroy")
        class SyntheticAdapter:
            impl, handle = 0, 42
            case = type("Case", (), {"x3": torch.ones(2), "w3": torch.ones(2), "grad": torch.ones(2), "dims": [1], "cg": [.01]})()
            def __init__(self):
                self.ops = SyntheticOps()
            def new_output(self, operation):
                return torch.empty(2, dtype=torch.float64)
            def call(self, operation, boundary, output=None):
                assert not torch.is_grad_enabled()
                return torch.ones(2) if boundary == "forward_allocating" else output
        adapter = SyntheticAdapter()
        cold = NativeColdCalls(adapter)
        cold.create()
        assert adapter.ops.events == ["constructor_and_prepare_weight"]
        cold.cleanup()
        assert adapter.ops.events == ["constructor_and_prepare_weight", "destroy"] and not cold.pending
        adapter.ops.events.clear()
        with patch("torch.cuda.synchronize"):
            assert torch.equal(cold.check(), torch.ones(2, dtype=torch.float64))
        assert adapter.ops.events == ["constructor_and_prepare_weight", "prepare_act", "launch", "destroy"]
        output_buffer = torch.empty(2)
        with torch.enable_grad():
            assert NativePerformanceFamily.invoke(adapter, "prepared_gemm", output_buffer).data_ptr() == output_buffer.data_ptr()
            first = NativePerformanceFamily.invoke(adapter, "forward_allocating")
            second = NativePerformanceFamily.invoke(adapter, "forward_allocating")
            assert first.data_ptr() != second.data_ptr() and torch.is_grad_enabled()
        Evidence.write(output, {"classification": "CPU_NATIVE_FACTORY_SELF_CHECK_NOT_GPU_PERFORMANCE", "passed": True,
            "variants": len(variants), "families": len(scope["families"]), "native_call_cells_per_round": 19 * 5 * 25,
            "checks": ["exact19selectors and25points each", "TF32x1 exclusion", "all three forward boundaries and two real components",
                       "cold times only existing combined create; launch and destroy outside window", "low-level output reuses caller storage",
                       "public allocating output remains independent", "explicit no_grad restores caller mode"],
            "source_sha256": Evidence.sha256(Path(__file__)), "GPU_work_started": False})
        return 0

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--accuracy-plan", type=Path)
        parser.add_argument("--accuracy-audit", type=Path)
        parser.add_argument("--create-contracts", type=Path)
        parser.add_argument("--contracts", type=Path)
        parser.add_argument("--cycle")
        parser.add_argument("--performance-plan", type=Path)
        parser.add_argument("--selfcheck", action="store_true")
        parser.add_argument("--output", type=Path)
        args = parser.parse_args()
        if args.create_contracts:
            return NativeAccuracyContracts.create(args.accuracy_plan, args.accuracy_audit, args.create_contracts)
        session = Session(args.session)
        if args.selfcheck:
            return cls.selfcheck(session, args.output)
        if args.performance_plan:
            from performance import PerformancePlan
            NativePerformanceFamily.configure_scope(session)
            PerformancePlan.create(session, args.performance_plan, "native_performance.NativePerformanceFamily", list(NativePerformanceFamily.SCOPE["families"]))
            return 0
        if args.contracts is None or args.cycle is None:
            parser.error("Scope creation needs --contracts and --cycle")
        NativePerformanceScope.create(session, args.contracts, args.cycle)
        return 0


if __name__ == "__main__":
    raise SystemExit(NativePerformanceCLI.main())
