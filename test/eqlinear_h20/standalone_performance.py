"""Standalone factories for the common H20 sampler; no independent timer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from common import Evidence
from performance_bindings import BoundCall, TorchCalls, TorchIdentity
from session import Session
from standalone import StandaloneAdapter, StandaloneSource
from standalone_matrix import VariantSpecs
from standalone_qualification import StandaloneAccuracyContracts


class StandalonePerformanceScope:
    """Immutable registry snapshots and finite environment groups, bound into plans."""

    @staticmethod
    def pointer(session):
        return session.root / "config/standalone_performance_scope.json"

    @staticmethod
    def groups(variants, max_variants=None):
        groups = {}
        for variant in variants:
            controls = variant.get("runtime_controls", {})
            precision, digest = variant["precision"], Evidence.fingerprint(controls)
            family = "standalone_" + precision + "_" + (digest[:12] if controls else "default")
            profile = variant.get("build_profile")
            if profile:
                family += "_profile_" + profile
            group = groups.setdefault(family, {"precision": precision, "runtime_controls": controls, "variants": [],
                **({"build_profile": profile} if profile else {})})
            group["variants"].append({name: value for name, value in variant.items() if name != "registry_row"})
        if max_variants is None:
            return groups
        if max_variants < 1:
            raise ValueError("Family size must be positive")
        partitioned = {}
        for family, group in groups.items():
            chunks = [group["variants"][offset:offset + max_variants]
                      for offset in range(0, len(group["variants"]), max_variants)]
            for index, chunk in enumerate(chunks):
                name = family if len(chunks) == 1 else family + "_" + str(index)
                partitioned[name] = {**group, "variants": chunk}
        return partitioned

    @classmethod
    def create(cls, session, policy_registry, cycle, accuracy_contracts=None, max_variants=16, supporting_evidence=()):
        import re

        if re.fullmatch(r"[A-Za-z0-9_.-]+", cycle) is None:
            raise ValueError("Use a simple fresh scope cycle")
        root = session.root / "performance/standalone_scope" / cycle
        root.mkdir(parents=True, exist_ok=False)
        files = {}
        for name, source in (("registry.json", session.root / "coverage/variants.json"), ("policies.json", policy_registry)):
            target = root / name
            shutil.copy2(source, target)
            files[str(target)] = Evidence.sha256(target)
        registry, policies = (json.loads((root / name).read_text()) for name in ("registry.json", "policies.json"))
        variants = VariantSpecs.from_registry(registry, policies)
        for variant in variants:
            for record in variant.get("registry_row", {}).get("named_choice_evidence", []):
                if record.get("sha256"):
                    path = Path(record["path"])
                    if Evidence.sha256(path) != record["sha256"]:
                        raise ValueError("Named strategy source/report evidence changed")
                    files[str(path.resolve())] = record["sha256"]
        groups = cls.groups(variants, max_variants)
        sources = {}
        for precision in StandaloneSource.CONFIG:
            source = StandaloneSource(session, precision)
            manifest = source.stage()
            build_path = session.root / "manifests" / ("standalone_" + precision + "_01_build.json")
            build = json.loads(build_path.read_text())
            if Evidence.sha256(source.library) != build["binary_hash"]:
                raise ValueError("Standalone DSO differs from preserved build receipt")
            sources[precision] = {"source_hash": manifest["source_hash"], "headers_hash": manifest["headers_hash"],
                "binary": str(source.library), "binary_hash": build["binary_hash"], "namespace": source.namespace,
                "source_manifest": str(source.manifest_path)}
            for path in (source.manifest_path, source.library, build_path, source.build / "CMakeCache.txt", source.build / "build.ninja"):
                files[str(path)] = Evidence.sha256(path)
        profiles = {}
        for variant in variants:
            if not variant.get("build_profile"):
                continue
            name, link = variant["build_profile"], variant["registry_row"]["profile_record"]
            original = Path(link["path"])
            if Evidence.sha256(original) != link["sha256"]:
                raise ValueError("Documented compiler-profile evidence changed")
            record = json.loads(original.read_text())
            if record["status"] != "BUILD_PASS_REQUIRES_H20_PROBE" or Evidence.sha256(Path(record["binary"])) != record["binary_hash"]:
                raise ValueError("Compiler profile lacks its successful source-bound build")
            target = root / "profiles" / (name + ".json")
            if not target.exists():
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(original, target)
            sources_record = sources[variant["precision"]]
            if record["source_manifest"]["source_hash"] != sources_record["source_hash"] or record["source_manifest"]["headers_hash"] != sources_record["headers_hash"]:
                raise ValueError("Profile and baseline production source identities disagree")
            profiles[name] = {**sources_record, "binary": record["binary"], "binary_hash": record["binary_hash"],
                "macro_definitions": record["macro_definitions"], "build_profile": name,
                "build_record": {"path": str(target), "sha256": Evidence.sha256(target)}}
            for path in (target, Path(record["binary"]), Path(record["cmake_cache"]["path"]), Path(record["ninja"]["path"])):
                files[str(path)] = Evidence.sha256(path)
        value = {"classification": "STANDALONE_FINITE_PERFORMANCE_SCOPE_NOT_RESULTS", "families": groups,
            "registry": str(root / "registry.json"), "policies": str(root / "policies.json"),
            "sources": sources, "variants": len(variants), "named_policies": len(policies["variants"]),
            "max_variants_per_family": max_variants,
            "evidence_files": files, "sampling": "performance.PerformanceWorker; protocol.Timer; protocol.Journal",
            "grouping": "one precision and identical runtime environment per family; different DSO builds remain separate",
            "coverage_remaining": "rotating-input diagnostics, instruction profiling and original-producer-unproven archival replay remain separate scope rows"}
        if profiles:
            value["build_profiles"] = profiles
        if accuracy_contracts is not None:
            contracts = json.loads(accuracy_contracts.read_text())
            if contracts.get("schema") != 1 or contracts.get("complete") is not True:
                raise ValueError("Formal scope requires complete upstream accuracy contracts")
            value["accuracy_contracts"] = {"path": str(accuracy_contracts.resolve()), "sha256": Evidence.sha256(accuracy_contracts)}
            files.update(contracts.get("evidence_files", {}))
            files[str(accuracy_contracts.resolve())] = Evidence.sha256(accuracy_contracts)
        value["supporting_evidence"] = []
        for path in supporting_evidence:
            document = json.loads(path.read_text())
            if not (document.get("passed") is True or document.get("complete") is True):
                raise ValueError("Supporting verification is not complete")
            record = {"path": str(path.resolve()), "sha256": Evidence.sha256(path)}
            value["supporting_evidence"].append(record)
            files[record["path"]] = record["sha256"]
        value["scope_hash"] = Evidence.fingerprint(value)
        target = root / "scope.json"
        Evidence.write(target, value)
        pointer = cls.pointer(session)
        Evidence.write(pointer, {"path": str(target), "sha256": Evidence.sha256(target)})
        return target, value

    @staticmethod
    def verify(metadata):
        scope = metadata["scope"]
        if Evidence.fingerprint({key: value for key, value in scope.items() if key != "scope_hash"}) != scope["scope_hash"]:
            raise ValueError("Standalone performance scope contents changed")
        for path, digest in metadata["evidence_files"].items():
            if Evidence.sha256(Path(path)) != digest:
                raise ValueError("Immutable standalone scope evidence changed: " + path)
        path = Path(metadata["scope_path"])
        if str(path) not in metadata["evidence_files"] or json.loads(path.read_text()) != scope:
            raise ValueError("Embedded scope differs from actual immutable scope document")
        variants = VariantSpecs.from_registry(json.loads(Path(scope["registry"]).read_text()), json.loads(Path(scope["policies"]).read_text()))
        if StandalonePerformanceScope.groups(variants, scope.get("max_variants_per_family")) != scope["families"]:
            raise ValueError("Performance families omit or alter actual registered variants")
        if scope.get("accuracy_contracts"):
            from cases import points
            link = scope["accuracy_contracts"]
            contracts = json.loads(Path(link["path"]).read_text())
            expected = {StandaloneAccuracyContracts.entry_key(row["variant_id"], point.key, row["operation"])
                        for group in scope["families"].values() for row in group["variants"]
                        for point in points() if point.precision == group["precision"]}
            if contracts.get("schema") != 1 or contracts.get("complete") is not True or set(contracts["entries"]) != expected:
                raise ValueError("Upstream qualification manifest does not cover exactly this finite full25 scope")
        return scope


class StandaloneColdCalls:
    """Time only the existing constructor, with destruction outside the sample."""

    def __init__(self, adapter, operation):
        self.adapter, self.operation, self.pending = adapter, operation, []
        self.activation = adapter.activation if operation == "forward" else None

    def create(self):
        destroy, handle = self.adapter.create_handle(self.operation, self.activation)
        self.pending.append((destroy, handle))
        return handle

    def cleanup(self):
        import torch
        torch.cuda.synchronize()
        while self.pending:
            destroy, handle = self.pending.pop()
            getattr(self.adapter.ops, destroy)(handle)

    def warm(self):
        handle = self.create()
        self.cleanup()
        return handle

    def check(self):
        import torch
        handle = self.create()
        try:
            output = StandalonePerformanceFamily.fresh_output(self.adapter, self.operation)
            if self.operation == "forward":
                self.adapter.ops.prepare_weight(handle, self.adapter.case.w3, self.adapter.case.cg)
                self.adapter.ops.prepare_act(handle, self.activation)
                self.adapter.ops.launch(handle, output)
            else:
                self.adapter.run_backward_handle(self.operation, handle, output)
            torch.cuda.synchronize()
            return output
        finally:
            self.cleanup()


class StandalonePreparationCalls:
    """Existing forward preparation APIs; validate via a separate full launch."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.handle = adapter.handles["forward"][1]

    def weight(self):
        return self.adapter.ops.prepare_weight(self.handle, self.adapter.case.w3, self.adapter.case.cg)

    def activation(self):
        self.adapter.activation = self.adapter.activation_input()
        return self.adapter.ops.prepare_act(self.handle, self.adapter.activation)

    def pack(self):
        return self.adapter.activation_input()

    def check(self, component):
        if component == "prepare_weight":
            self.weight()
        else:
            self.activation()
        output = StandalonePerformanceFamily.fresh_output(self.adapter, "forward")
        self.adapter.ops.launch(self.handle, output)
        return output


class StandalonePerformanceFamily:
    """All contracts sharing one immutable DSO and runtime environment."""

    FILES = ("standalone_performance.py", "standalone.py", "standalone_matrix.py", "standalone_probe.py")
    SCOPE = None

    @classmethod
    def configure_scope(cls, session):
        pointer = json.loads(StandalonePerformanceScope.pointer(session).read_text())
        path = Path(pointer["path"])
        if Evidence.sha256(path) != pointer["sha256"]:
            raise ValueError("Standalone scope pointer differs from immutable scope bytes")
        scope = json.loads(path.read_text())
        metadata = {"scope": scope, "scope_path": str(path), "evidence_files": {**scope["evidence_files"], str(path): pointer["sha256"]}}
        cls.SCOPE = StandalonePerformanceScope.verify(metadata)
        return metadata

    @classmethod
    def verify_scope(cls, metadata, session):
        cls.SCOPE = StandalonePerformanceScope.verify(metadata)
        if any(not Path(path).is_relative_to(Path("/work-ict")) for path in metadata["evidence_files"]):
            raise ValueError("Standalone scope references evidence outside the authorized workspace")

    @classmethod
    def specification(cls, family):
        if cls.SCOPE is None:
            raise RuntimeError("Sampler must configure or verify the immutable factory_scope first")
        return cls.SCOPE["families"][family]

    @classmethod
    def accept_point(cls, family, point):
        return point.precision == cls.specification(family)["precision"]

    @staticmethod
    def operations(group):
        return sorted({variant["operation"] for variant in group["variants"]})

    @staticmethod
    def components(variant):
        result = ["cold_create"]
        if variant["operation"] == "forward":
            result += ["prepare_weight", "prepare_act"]
            if variant.get("activation_layout") == "pbu":
                result += ["layout_pack"]
        return result

    @staticmethod
    def fresh_output(adapter, operation):
        """Validation only: detect partial writes without changing any timed call."""
        value = adapter.new_output(operation)
        for tensor in value if isinstance(value, tuple) else (value,):
            tensor.fill_(float("nan"))
        return value

    @classmethod
    def native_call(cls, adapter, operation, boundary, fresh=False):
        import torch

        if operation == "forward":
            with torch.no_grad():
                return adapter.call(operation, boundary, cls.fresh_output(adapter, operation) if fresh else None)
        return adapter.call(operation, boundary, cls.fresh_output(adapter, operation) if fresh else None)

    @classmethod
    def expected_calls(cls, family):
        group = cls.specification(family)
        calls = ["/".join((variant["variant_id"], variant["operation"], boundary))
                 for variant in group["variants"] for boundary in variant["boundaries"] + cls.components(variant)]
        for operation in cls.operations(group):
            calls.extend((f"torch_ieee_full/{operation}/public_warm", f"torch_ieee_mm/{operation}/prepared_gemm"))
        return sorted(calls)

    def __init__(self, case, session, family):
        import torch

        self.case, self.session, self.family = case, session, family
        self.group = self.specification(family)
        if not self.accept_point(family, case.point):
            raise ValueError("Standalone family precision differs from canonical point")
        self.previous_controls = {key: value for key, value in os.environ.items() if key.startswith("EQLIN_")}
        for key in list(self.previous_controls):
            os.environ.pop(key)
        os.environ.update(self.group["runtime_controls"])
        self.adapters, self.calls, self.cold, self.setup_metadata = [], [], [], {}
        self.memory_receipt = session.root / "performance/standalone_memory" / (str(os.getpid()) + "_" + family + "_" + case.point.key.replace("/", "_") + ".json")
        self.setup_metadata["memory_before_prepare"] = dict(zip(("free_bytes", "total_bytes"), torch.cuda.mem_get_info()))
        torch.cuda.reset_peak_memory_stats()
        self.torch = TorchCalls(case)
        operations = self.operations(self.group)
        self.torch.prepare(tuple(operation for operation in ("grad_x", "grad_w", "grad_x_w") if operation in operations))
        self.torch.full("forward")  # ensure the actual cuBLAS binaries are mapped before hashing
        self.torch_identity = TorchIdentity.capture(session)
        self.torch_identity.update(accuracy_tier="fp64" if case.point.precision == "fp64" else "fp32",
            source_hash=Evidence.fingerprint({"cases.py": Evidence.sha256(Path(__file__).with_name("cases.py")),
                                             "performance_bindings.py": Evidence.sha256(Path(__file__).with_name("performance_bindings.py"))}))
        source = StandaloneSource(session, case.point.precision)
        expected = (self.SCOPE["build_profiles"][self.group["build_profile"]] if self.group.get("build_profile")
                    else self.SCOPE["sources"][case.point.precision])
        library = Path(expected["binary"])
        manifest = source.stage()
        if (manifest["source_hash"] != expected["source_hash"] or manifest["headers_hash"] != expected["headers_hash"] or
                Evidence.sha256(library) != expected["binary_hash"]):
            raise ValueError("Standalone actual source/DSO differs from immutable performance scope")
        for variant in self.group["variants"]:
            adapter = StandaloneAdapter(case, library, case.point.precision, variant["impl"], variant["bimpl"],
                source.manifest_path, api_controls=variant.get("api_controls"), activation_layout=variant.get("activation_layout", "biu"))
            self.adapters.append(adapter)
            operation = variant["operation"]
            for boundary in variant["boundaries"]:
                adapter.prepare(operation, boundary)
                self.calls.append(BoundCall(variant["variant_id"], operation, boundary,
                    lambda adapter=adapter, operation=operation, boundary=boundary: self.native_call(adapter, operation, boundary),
                    lambda adapter=adapter, operation=operation, boundary=boundary: self.native_call(adapter, operation, boundary, fresh=True),
                    lambda value: value, {},
                    "original allocating API includes its per-call construction/preparation; run includes activation preparation; prepared_gemm launches the already prepared handle"))
            cold = StandaloneColdCalls(adapter, operation)
            self.cold.append(cold)
            self.calls.append(BoundCall(variant["variant_id"], operation, "cold_create", cold.create, cold.check,
                lambda value: value, {}, "original native constructor; validation launch and destruction outside timing",
                after_sample=cold.cleanup, warm_function=cold.warm,
                measurement_work="constructor_and_dgrad_weight_prepare" if operation in ("grad_x", "grad_x_w") else "native_plan_constructor"))
            if operation == "forward":
                preparation = StandalonePreparationCalls(adapter)
                for component in self.components(variant)[1:]:
                    function = {"prepare_weight": preparation.weight, "prepare_act": preparation.activation, "layout_pack": preparation.pack}[component]
                    self.calls.append(BoundCall(variant["variant_id"], operation, component, function,
                        lambda component=component, preparation=preparation: preparation.check(component),
                        lambda value: value, {}, "original preparation API or required adapter layout pack; full check outside timing",
                        measurement_work="adapter_layout_pack_plus_native_prepare" if component == "prepare_act" and adapter.pack_pbu else component))
            identity = adapter.identity()
            self.setup_metadata[variant["variant_id"]] = {"setup_seconds": identity.pop("setup_seconds", {})}
            identity.update(production_source_hash=identity["source_hash"], accuracy_tier=case.point.precision,
                execution_grad_mode="no_grad" if operation == "forward" else "manual_derivative_api",
                performance_scope_hash=self.SCOPE["scope_hash"], variant_specification=variant,
                validation_output_initialization="NaN for caller-owned outputs; allocating API owns its internal output",
                sampler_adapter_hashes={name: Evidence.sha256(Path(__file__).with_name(name)) for name in self.FILES})
            if self.group.get("build_profile"):
                identity.update(build_profile=self.group["build_profile"], macro_definitions=expected["macro_definitions"])
            if self.SCOPE.get("accuracy_contracts"):
                identity["accuracy_contract"] = {**self.SCOPE["accuracy_contracts"],
                    "entry_key": StandaloneAccuracyContracts.entry_key(variant["variant_id"], case.point.key, operation)}
            identity["source_hash"] = Evidence.fingerprint({"production": identity["production_source_hash"],
                "adapters": identity["sampler_adapter_hashes"], "variant": variant, "scope": self.SCOPE["scope_hash"]})
            for call in self.calls:
                if call.variant == variant["variant_id"]:
                    call.binding = identity
        self.calls.extend(self.torch_calls(operations))
        torch.cuda.synchronize()
        self.setup_metadata["memory_after_prepare"] = dict(zip(("free_bytes", "total_bytes"), torch.cuda.mem_get_info()))
        self.setup_metadata["memory_receipt"] = str(self.memory_receipt)
        if sorted(call.key for call in self.calls) != self.expected_calls(family):
            raise ValueError("Actual standalone call set differs from the immutable declared scope")

    def torch_calls(self, operations):
        calls = []
        for operation in operations:
            calls.append(BoundCall("torch_ieee_full", operation, "public_warm",
                lambda operation=operation: self.torch.full(operation), lambda operation=operation: self.torch.full(operation),
                lambda value: value, self.torch_identity, "eager complete operation on the canonical layouts"))
            calls.append(BoundCall("torch_ieee_mm", operation, "prepared_gemm",
                lambda operation=operation: self.torch.mm(operation), lambda operation=operation: self.torch.mm(operation, fresh=True),
                lambda values, operation=operation: self.torch.normalize_mm(operation, values),
                self.torch_identity, "pre-gathered per-path products; preparation and reduction outside timing",
                measurement_work="matrix_products_only"))
        return calls

    def pilot_calls(self):
        operations = ("forward", "grad_x", "grad_w", "grad_x_w")
        self.torch.prepare(("grad_x", "grad_w", "grad_x_w"))
        return self.torch_calls(operations)

    def close(self):
        import torch
        torch.cuda.synchronize()
        Evidence.write(self.memory_receipt, {"classification": "MEMORY_OBSERVATION_NOT_TIMING_IDENTITY",
            "point": self.case.point.columns(), "family": self.family,
            "GPU_UUID": str(torch.cuda.get_device_properties().uuid),
            "before_prepare": self.setup_metadata["memory_before_prepare"],
            "after_prepare": self.setup_metadata["memory_after_prepare"],
            "before_close": dict(zip(("free_bytes", "total_bytes"), torch.cuda.mem_get_info())),
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
            "native_allocations_are_not_in_torch_peak_counters": True})
        for cold in self.cold:
            cold.cleanup()
        for adapter in self.adapters:
            adapter.close()
        self.adapters.clear()
        self.calls.clear()
        self.cold.clear()
        self.torch.graphs.clear()
        self.torch.products.clear()
        for key in list(os.environ):
            if key.startswith("EQLIN_"):
                os.environ.pop(key)
        os.environ.update(self.previous_controls)


class ScopeCLI:
    @staticmethod
    def identity_snapshot(args):
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        import torch
        from cases import Case, points
        from performance import PerformanceWorker
        from reference_store import ReferenceStore

        session = Session(args.session)
        if args.output is None or not args.output.resolve().is_relative_to(session.root) or args.output.exists():
            raise ValueError("Identity snapshot needs a fresh session output")
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        affinity = PerformanceWorker.pin_cpu()
        metadata = StandalonePerformanceFamily.configure_scope(session)
        point = next(point for point in points() if point.key == args.point + "/" + args.precision)
        case = Case(point, fixture_root=session.root / "fixtures")
        references = ReferenceStore(session.root / "references").get(case)
        default_family = "standalone_" + args.precision + "_default"
        if default_family not in metadata["scope"]["families"]:
            default_family += "_0"
        family = StandalonePerformanceFamily(case, session, args.family or default_family)
        try:
            bindings = json.loads(json.dumps({call.key: call.binding for call in family.calls}))
            if any("setup_seconds" in value for value in bindings.values()):
                raise ValueError("Observed construction duration leaked into binding identity")
            Evidence.write(args.output, {"classification": "FRESH_FACTORY_IDENTITY_NOT_PERFORMANCE", "bindings": bindings,
                "setup_metadata": family.setup_metadata, "GPU_UUID": str(torch.cuda.get_device_properties().uuid),
                "input_hash": case.input_hash, "reference_hashes": references.hashes, "point": point.columns(),
                "affinity": affinity, "scope_hash": metadata["scope"]["scope_hash"],
                "source_sha256": Evidence.sha256(Path(__file__))})
        finally:
            family.close()
        return 0

    @staticmethod
    def compare_identities(session, paths, output):
        if output is None or not output.resolve().is_relative_to(session.root) or output.exists():
            raise ValueError("Identity comparison needs a fresh session output")
        values = [json.loads(path.read_text()) for path in paths]
        fields = ("bindings", "GPU_UUID", "input_hash", "reference_hashes", "point", "affinity", "scope_hash", "source_sha256")
        mismatches = [key for key in fields if values[0][key] != values[1][key]]
        Evidence.write(output, {"classification": "FRESH_FACTORY_IDENTITY_NOT_PERFORMANCE", "passed": not mismatches,
            "mismatches": mismatches, "setup_metadata_excluded_from_identity": True,
            "snapshots": [{"path": str(path), "sha256": Evidence.sha256(path)} for path in paths]})
        return 0 if not mismatches else 1

    @staticmethod
    def selfcheck(session, output):
        import copy
        from cases import points

        if output is None or not output.resolve().is_relative_to(session.root):
            raise ValueError("CPU metadata selfcheck output must stay in the session")
        metadata = StandalonePerformanceFamily.configure_scope(session)
        scope = metadata["scope"]
        variants = [variant for group in scope["families"].values() for variant in group["variants"]]
        assert len(variants) == 28 + scope["named_policies"] and len({row["variant_id"] for row in variants}) == len(variants)
        assert all(not (row["precision"] == "tf32" and row["impl"] in (1, 2)) for row in variants)
        for family, group in scope["families"].items():
            assert all(row.get("runtime_controls", {}) == group["runtime_controls"] for row in group["variants"])
            assert all(row.get("build_profile") == group.get("build_profile") for row in group["variants"])
            assert sum(StandalonePerformanceFamily.accept_point(family, point) for point in points()) == 25
            calls = StandalonePerformanceFamily.expected_calls(family)
            assert len(calls) == len(set(calls))
            native = [key for key in calls if not key.startswith("torch_")]
            assert len(native) == sum(len(row["boundaries"] + StandalonePerformanceFamily.components(row)) for row in group["variants"])
        changed = copy.deepcopy(metadata)
        first = next(iter(changed["scope"]["families"].values()))
        first["variants"].pop()
        changed["scope"]["scope_hash"] = Evidence.fingerprint({key: value for key, value in changed["scope"].items() if key != "scope_hash"})
        try:
            StandalonePerformanceScope.verify(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("Self-consistent scope omitted an actual registered variant")
        changed = copy.deepcopy(metadata)
        changed["evidence_files"][scope["registry"]] = "0" * 64
        try:
            StandalonePerformanceScope.verify(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed immutable registry bytes accepted")
        from unittest.mock import patch
        class SyntheticOps:
            def __init__(self):
                self.events = []
            def prepare_weight(self, *args):
                self.events.append("weight")
            def prepare_act(self, *args):
                self.events.append("activation")
            def launch(self, *args):
                self.events.append("launch")
            def destroy(self, *args):
                self.events.append("destroy")
        class SyntheticAdapter:
            activation = object()
            case = type("Case", (), {"w3": object(), "cg": [0.01]})()
            def __init__(self):
                self.ops = SyntheticOps()
            def create_handle(self, *args):
                self.ops.events.append("create")
                return "destroy", 42
            def new_output(self, *args):
                import torch
                return torch.empty(2, dtype=torch.float64)
        adapter = SyntheticAdapter()
        cold = StandaloneColdCalls(adapter, "forward")
        with patch("torch.cuda.synchronize"):
            cold.create()
            assert adapter.ops.events == ["create"] and len(cold.pending) == 1
            cold.cleanup()
            assert adapter.ops.events == ["create", "destroy"] and not cold.pending
            adapter.ops.events.clear()
            cold.check()
            assert adapter.ops.events == ["create", "weight", "activation", "launch", "destroy"] and not cold.pending
        synthetic = [{"variant_id": "default", "precision": "fp64"},
                     {"variant_id": "profile_a", "precision": "fp64", "build_profile": "a"},
                     {"variant_id": "profile_b", "precision": "fp64", "build_profile": "b"}]
        assert len(StandalonePerformanceScope.groups(synthetic)) == 3
        import torch
        class GradObservation:
            def call(self, *args):
                return torch.is_grad_enabled()
        with torch.enable_grad():
            assert StandalonePerformanceFamily.native_call(GradObservation(), "forward", "prepared_gemm") is False
            assert torch.is_grad_enabled()
            assert StandalonePerformanceFamily.native_call(GradObservation(), "grad_x", "run_with_activation_prep") is True
        Evidence.write(output, {"classification": "CPU_SCOPE_METADATA_ONLY_NOT_TIMING", "passed": True,
            "families": len(scope["families"]), "variants": len(variants), "named_policies": scope["named_policies"],
            "native_calls_per_full_precision_grid": sum(len(row["boundaries"] + StandalonePerformanceFamily.components(row)) for row in variants) * 25,
            "checks": ["all actual registry contracts retained once", "one fixed env per precision family",
                       "each family selects all25 matching canonical points", "unique declared native and Torch calls",
                       "scope omission and immutable registry SHA corruption rejected",
                       "different compiler-profile DSOs split into independent process families",
                       "forward explicitly disables grad and restores caller grad mode",
                       "cold function performs constructor only; validation and destroy remain outside its window"],
            "cold_and_preparation_components_complete": False, "source_sha256": Evidence.sha256(Path(__file__)),
            "GPU_work_started": False})
        return 0

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--policy-registry", type=Path)
        parser.add_argument("--cycle")
        parser.add_argument("--selfcheck", action="store_true")
        parser.add_argument("--output", type=Path)
        parser.add_argument("--accuracy-contracts", type=Path)
        parser.add_argument("--max-variants-per-family", type=int, default=16)
        parser.add_argument("--supporting-evidence", type=Path, action="append", default=[])
        parser.add_argument("--identity-snapshot", action="store_true")
        parser.add_argument("--compare-identities", type=Path, nargs=2)
        parser.add_argument("--precision", choices=("tf32", "fp32", "fp64"), default="fp32")
        parser.add_argument("--point", default="SMALL_P1/512")
        parser.add_argument("--family")
        parser.add_argument("--performance-plan", type=Path)
        parser.add_argument("--plan-families", help="explicit comma-separated families; omitted selects the complete scope")
        args = parser.parse_args()
        if args.performance_plan:
            from performance import PerformancePlan
            session = Session(args.session)
            metadata = StandalonePerformanceFamily.configure_scope(session)
            families = args.plan_families.split(",") if args.plan_families else list(metadata["scope"]["families"])
            value = PerformancePlan.create(session, args.performance_plan, "standalone_performance.StandalonePerformanceFamily", families)
            print(json.dumps({"plan_hash": value["plan_hash"], "families": families}))
            return 0
        if args.identity_snapshot:
            return cls.identity_snapshot(args)
        if args.compare_identities:
            return cls.compare_identities(Session(args.session), args.compare_identities, args.output)
        if args.selfcheck:
            return cls.selfcheck(Session(args.session), args.output)
        if args.policy_registry is None or args.cycle is None:
            parser.error("scope creation requires --policy-registry and --cycle")
        target, scope = StandalonePerformanceScope.create(Session(args.session), args.policy_registry, args.cycle,
            args.accuracy_contracts, args.max_variants_per_family, args.supporting_evidence)
        print(json.dumps({"scope": str(target), "families": list(scope["families"]), "variants": scope["variants"],
                          "GPU_work_started": False}))
        return 0


if __name__ == "__main__":
    raise SystemExit(ScopeCLI.main())
