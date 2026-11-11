"""Per-wrapper, precision-locked historical factories for the shared sampler."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from cases import points
from common import Evidence
from historical_batch import HistoryPlan, HistoryAudit
from performance_bindings import CurrentTorchFamily
from qualification import AccuracyContracts, RecordedAccuracy
from session import Session


class HistoricalAccuracyContracts:
    @staticmethod
    def entry_key(variant, point, operation):
        return "/".join(("historical_" + variant, point, operation))

    @classmethod
    def link(cls, manifest, variant, point):
        return {**manifest, "entry_keys_by_operation": {operation: cls.entry_key(variant, point, operation)
                                                        for operation in HistoryAudit.REQUIRED}}

    @classmethod
    def operation_entries(cls, job, record, plan_hash):
        path = Path(record["result"])
        if not record.get("verified") or Evidence.sha256(path) != record["result_sha256"]:
            raise ValueError("Audited historical accuracy evidence changed")
        result = json.loads(path.read_text())
        point = "/".join(map(str, (job["point"]["config"], job["point"]["batch"], job["precision"])))
        identity, entries = result["identity"], {}
        for operation, boundaries in HistoryAudit.REQUIRED.items():
            rows = [row for row in result["boundaries"] if row["operation"] == operation]
            if sorted(row["call_boundary"] for row in rows) != sorted(boundaries):
                raise ValueError("Historical operation boundary coverage changed")
            passes = [RecordedAccuracy.passes(row, job["precision"]) for row in rows]
            if any(row.get("passed") is not passed for row, passed in zip(rows, passes)):
                raise ValueError("Historical recorded gates disagree with shared independent qualification")
            key = cls.entry_key(job["variant_id"], point, operation)
            entries[key] = {"variant_id": "historical_" + job["variant_id"], "tested_variant_id": job["variant_id"],
                "point_key": point, "operation": operation, "input_hash": result["input_hash"],
                "reference_hashes": result["reference_hashes"], "production_source_hash": identity["source_hash"],
                "binary_hash": identity["binary_hash"], "runtime_controls": identity.get("runtime_controls", {}),
                "api_controls": {}, "passed": all(passes), "required_boundaries": boundaries,
                "evidence": {"path": str(path), "sha256": record["result_sha256"]},
                "reason": "FULL_OPERATION_PASS" if all(passes) else "FULL_OPERATION_ACCURACY_FAIL",
                "upstream_plan_hash": plan_hash, "upstream_gpu_uuid": result["GPU_UUID"]}
        return entries

    @classmethod
    def create(cls, plan, audit_path, output):
        audit = json.loads(audit_path.read_text())
        jobs = plan.value["jobs"]
        if (plan.value["stage"] != "full" or len(plan.value["sources"]) != 63 or len(jobs) != 1575 or
                not audit.get("complete") or not audit.get("full_registry_complete") or
                audit.get("classification") != HistoryAudit.REPORT_CLASSIFICATION or
                audit.get("auditor_sha256") != plan.value["producer"]["historical_batch.py"] or
                audit.get("expected_points") != 1575 or audit.get("expected_paths") != 7875 or
                audit.get("plan_hash") != plan.value["plan_hash"] or
                set(audit["records"]) != {job["attempt"] for job in jobs}):
            raise ValueError("Historical performance requires the independently audited full63/1575/7875 epoch")
        entries = {}
        for job in jobs:
            record = audit["records"][job["attempt"]]
            for key, entry in cls.operation_entries(job, record, plan.value["plan_hash"]).items():
                if key in entries:
                    raise ValueError("Duplicate historical upstream entry")
                entries[key] = entry
        if len(entries) != 4725:
            raise ValueError("Historical qualification omits or adds source/point/operation contracts")
        value = {"schema": 1, "complete": True, "entries": entries, "variants": 63, "expected_entries": 4725,
            "qualified_entries": sum(row["passed"] for row in entries.values()),
            "evidence_files": {str(plan.path): Evidence.sha256(plan.path), str(audit_path): Evidence.sha256(audit_path)},
            "scope": "Exact complete synchronized63-wrapper epoch; queued safety and unmatched H100 frozen P14 hash gate stay unverified",
            "partial_epoch_mixing": False}
        if output.exists():
            raise FileExistsError(output)
        Evidence.write(output, value)
        # Use the same qualification resolver as the actual sampler.
        digest = Evidence.sha256(output)
        for key, entry in entries.items():
            identity = json.loads(Path(entry["evidence"]["path"]).read_text())["identity"]
            binding = {**identity, "production_source_hash": entry["production_source_hash"],
                       "accuracy_contract": {"path": str(output), "sha256": digest, "entry_key": key}}
            AccuracyContracts.resolve(binding, entry["variant_id"], entry["point_key"], entry["operation"],
                                      entry["input_hash"], entry["reference_hashes"])
        return value


class HistoricalPerformanceScope:
    @staticmethod
    def pointer(session):
        return session.root / "config/historical_performance_scope.json"

    @classmethod
    def create(cls, session, source_plan, cycle, audit_path=None):
        if Path(cycle).name != cycle:
            raise ValueError("Use a simple fresh scope cycle")
        plan = HistoryPlan(source_plan)
        if plan.session.root != session.root or len(plan.value["sources"]) != 63 or plan.value["stage"] != "full":
            raise ValueError("Expected the exact full63 historical source plan")
        root = session.root / "performance/historical_scope" / cycle
        root.mkdir(parents=True, exist_ok=False)
        files = {str(plan.path): Evidence.sha256(plan.path)}
        families = {}
        for variant, source in plan.value["sources"].items():
            identity = source["identity"]
            families[variant] = {"variant": variant, "precision": source["precision"], "source": source,
                "queued_safety_verified": False,
                "prepare_components": {"weight": "included in original forward_create/dgrad_create", "activation": "included in original forward_run/dgrad_run", "separate_prepare_entry": False}}
            files[identity["build_manifest_path"]] = identity["build_manifest_sha256"]
            files[source["wrapper"]] = source["wrapper_sha256"]
            files.update(source["binaries"])
        value = {"classification": "HISTORICAL_PERFORMANCE_SCOPE_NOT_RESULTS", "session": str(session.root), "families": families,
            "source_plan": {"path": str(plan.path), "sha256": Evidence.sha256(plan.path), "plan_hash": plan.value["plan_hash"]},
            "variants": 63, "selector_points": 1575, "original_accuracy_paths": 7875,
            "qualified": False, "accuracy_contracts": None, "evidence_files": files,
            "isolation": "One wrapper/precision per worker; all different wrapper identities retained; Torch anchor in same process",
            "sampling": "performance.PerformanceWorker; protocol.Timer; protocol.Journal; no independent clock",
            "math": "IEEE Torch anchors; historical per-ref TF32x3 plan guard; no TF32x1,FP16,BF16,TileLang,CUDA Graph",
            "input_leaves": "HistoricalAdapter.x_grad and TorchCalls.forward_x created during setup, outside every timed call",
            "synchronization": "device synchronize per completed call; original P0 queued safety not asserted",
            "component_scope": "Existing registered Torch API only; cold_create includes weight preparation, run includes activation preparation; no manufactured split entrypoints"}
        value["qualification_link_scheme"] = "entry_keys_by_operation; one identical source binding per historical variant, operation-specific independently audited entry"
        for label, path in (("final_user_scope", session.root / "coverage/final_scope_user_resume01/decision.json"),
                            ("measurement_cpu_isolation", session.root / "manifests/final_cpu_affinity_adjudication01.json")):
            if path.exists():
                files[str(path)] = Evidence.sha256(path)
                value[label] = {"path": str(path), "sha256": files[str(path)]}
        if audit_path is not None:
            audit_path = audit_path.resolve()
            contracts_path = root / "accuracy_contracts.json"
            contracts = HistoricalAccuracyContracts.create(plan, audit_path, contracts_path)
            value.update(qualified=True, accuracy_contracts={"path": str(contracts_path), "sha256": Evidence.sha256(contracts_path)},
                         qualified_operation_points=contracts["qualified_entries"])
            files.update(contracts["evidence_files"])
            files[str(contracts_path)] = Evidence.sha256(contracts_path)
        value["scope_hash"] = Evidence.fingerprint(value)
        path = root / "scope.json"
        Evidence.write(path, value)
        if value["qualified"]:
            cls.verify({"scope": value, "scope_path": str(path), "evidence_files": {**files, str(path): Evidence.sha256(path)}})
            Evidence.write(cls.pointer(session), {"path": str(path), "sha256": Evidence.sha256(path)})
        return path, value

    @staticmethod
    def verify(metadata, require_qualified=True):
        scope = metadata["scope"]
        if Evidence.fingerprint({k: v for k, v in scope.items() if k != "scope_hash"}) != scope["scope_hash"]:
            raise ValueError("Historical performance scope identity changed")
        if require_qualified and (scope.get("qualified") is not True or not scope.get("accuracy_contracts")):
            raise ValueError("Historical accuracy qualification is pending; GPU performance forbidden")
        if len(scope["families"]) != 63 or scope["variants"] != 63:
            raise ValueError("Historical wrapper scope differs from full63")
        if any(metadata["evidence_files"].get(path) != digest for path, digest in scope["evidence_files"].items()):
            raise ValueError("Historical external evidence bindings omitted")
        for path, digest in metadata["evidence_files"].items():
            if not Path(path).is_relative_to(Path("/work-ict")) or Evidence.sha256(Path(path)) != digest:
                raise ValueError("Historical source/binary/scope evidence changed: " + path)
        path = Path(metadata["scope_path"])
        if str(path) not in metadata["evidence_files"] or json.loads(path.read_text()) != scope:
            raise ValueError("Historical embedded scope differs from immutable scope bytes")
        source_plan = json.loads(Path(scope["source_plan"]["path"]).read_text())
        if (source_plan["session"] != scope["session"] or source_plan["plan_hash"] != scope["source_plan"]["plan_hash"] or
                set(source_plan["sources"]) != set(scope["families"]) or
                any(group["source"] != source_plan["sources"][family] or group["variant"] != family or
                    group["precision"] != source_plan["sources"][family]["precision"]
                    for family, group in scope["families"].items())):
            raise ValueError("Historical performance families differ from the frozen accuracy source/precision plan")
        if scope["qualified"]:
            link = scope["accuracy_contracts"]
            contracts = json.loads(Path(link["path"]).read_text())
            expected = {HistoricalAccuracyContracts.entry_key(family, point.key, operation)
                for family, specification in scope["families"].items() for point in points()
                if point.precision == specification["precision"] for operation in HistoryAudit.REQUIRED}
            if contracts.get("complete") is not True or set(contracts["entries"]) != expected:
                raise ValueError("Historical qualification is missing source/point/operation contracts")
        return scope


class HistoricalPerformanceFamily(CurrentTorchFamily):
    FILES = CurrentTorchFamily.FILES + ("historical_performance.py", "historical.py", "historical_plan.py", "historical_native.py", "registry.py")
    PRECISION_CHUNKS = True
    SCOPE = None

    @classmethod
    def configure_scope(cls, session):
        pointer = json.loads(HistoricalPerformanceScope.pointer(session).read_text())
        path = Path(pointer["path"])
        if Evidence.sha256(path) != pointer["sha256"]:
            raise ValueError("Historical scope pointer changed")
        scope = json.loads(path.read_text())
        metadata = {"scope": scope, "scope_path": str(path), "evidence_files": {**scope["evidence_files"], str(path): pointer["sha256"]}}
        cls.SCOPE = HistoricalPerformanceScope.verify(metadata)
        return metadata

    @classmethod
    def verify_scope(cls, metadata, session):
        if metadata["scope"]["session"] != str(session.root):
            raise ValueError("Historical scope belongs to a different session")
        cls.SCOPE = HistoricalPerformanceScope.verify(metadata)

    @classmethod
    def specification(cls, family):
        if cls.SCOPE is None:
            raise ValueError("Configure or verify the immutable historical scope first")
        return cls.SCOPE["families"][family]

    @classmethod
    def accept_point(cls, family, point):
        return point.precision == cls.specification(family)["precision"]

    @classmethod
    def expected_calls(cls, family):
        cls.specification(family)
        return CurrentTorchFamily.expected_calls(family)

    def __init__(self, case, session, family):
        if not self.accept_point(family, case.point) or self.SCOPE.get("qualified") is not True:
            raise ValueError("Wrong historical precision or incomplete upstream qualification")
        super().__init__(case, session, family)
        source = self.specification(family)["source"]
        observed = self.current.identity()
        if any(observed[key] != source["identity"][key] for key in ("source_hash", "binary_hash", "source_commit", "namespace", "source_manifest_sha256")):
            raise ValueError("Actual historical source/DSO differs from fixed performance scope")
        if observed["module_file"] != source["wrapper"] or observed["adapted_wrapper_sha256"] != source["wrapper_sha256"]:
            raise ValueError("Actual historical host wrapper differs")
        for call in self.calls:
            if call.variant != "historical_" + family:
                continue
            binding = dict(call.binding)
            adapters = {name: Evidence.sha256(Path(__file__).with_name(name)) for name in self.FILES}
            binding.update(performance_scope_hash=self.SCOPE["scope_hash"], sampler_adapter_hashes=adapters,
                accuracy_contract=HistoricalAccuracyContracts.link(self.SCOPE["accuracy_contracts"], family, case.point.key),
                synchronized_completed_call=True, queued_safety_verified=False,
                precision_contract=source["precision"], input_leaf_preparation_outside_timing=True)
            binding["source_hash"] = Evidence.fingerprint({"production": binding["production_source_hash"], "adapters": adapters,
                                                         "scope": self.SCOPE["scope_hash"]})
            call.binding = binding


class CpuCheck:
    @classmethod
    def run(cls, path, output):
        scope = json.loads(path.read_text())
        metadata = {"scope": scope, "scope_path": str(path), "evidence_files": {**scope["evidence_files"], str(path): Evidence.sha256(path)}}
        HistoricalPerformanceScope.verify(metadata, require_qualified=False)
        HistoricalPerformanceFamily.SCOPE = scope
        selected, rejected, call_sets, upstream_keys = 0, 0, set(), set()
        for family in scope["families"]:
            observed = [point for point in points() if HistoricalPerformanceFamily.accept_point(family, point)]
            if len(observed) != 25:
                raise ValueError("Wrapper precision leaked across canonical points")
            selected += len(observed)
            for point in observed:
                link = HistoricalAccuracyContracts.link({}, family, point.key)
                if set(link["entry_keys_by_operation"]) != set(HistoryAudit.REQUIRED):
                    raise ValueError("Missing operation-specific upstream qualification key")
                upstream_keys.update(link["entry_keys_by_operation"].values())
            rejected += sum(not HistoricalPerformanceFamily.accept_point(family, point) for point in points())
            calls = HistoricalPerformanceFamily.expected_calls(family)
            if len(calls) != 12 or len(set(calls)) != 12:
                raise ValueError("Historical/Torch call boundary scope changed")
            call_sets.add(tuple(calls))
        if selected != 1575 or rejected != 3150 or len(call_sets) != 63 or len(upstream_keys) != 4725:
            raise ValueError("Missing/duplicated wrapper contracts")
        if not scope["qualified"]:
            try:
                HistoricalPerformanceScope.verify(metadata)
            except ValueError as error:
                if "qualification is pending" not in str(error):
                    raise
            else:
                raise ValueError("Unqualified source scope accepted for GPU performance")
        altered = copy.deepcopy(metadata)
        altered["scope"]["variants"] = 62
        try:
            HistoricalPerformanceScope.verify(altered, require_qualified=False)
        except ValueError:
            pass
        else:
            raise ValueError("Altered wrapper scope accepted")
        Evidence.write(output, {"status": "CPU_SCOPE_AND_PRECISION_GUARDS_PASS", "families": 63,
            "selected_points": selected, "wrong_precision_points_rejected": rejected,
            "calls_per_family_point": 12, "source_scope_sha256": Evidence.sha256(path),
            "operation_specific_upstream_keys": len(upstream_keys), "per_variant_binding_stable_across_operations": True,
            "partial_accuracy_cannot_qualify": not scope["qualified"], "GPU_work_started": False})


class OperationLinkCheck:
    @staticmethod
    def run(source_plan, output):
        plan = HistoryPlan(source_plan)
        job = plan.value["jobs"][0]
        record = HistoryAudit(plan).record(job)
        entries = HistoricalAccuracyContracts.operation_entries(job, record, plan.value["plan_hash"])
        fixture = output.with_name(output.stem + "_three_entry_fixture.json")
        if fixture.exists() or output.exists():
            raise FileExistsError(output)
        Evidence.write(fixture, {"classification": "CPU_SINGLE_POINT_THREE_ENTRY_LINK_FIXTURE_NOT_GLOBAL_QUALIFICATION",
            "schema": 1, "complete": True, "entries": entries, "global_full63_qualified": False,
            "point_scope": "Exactly one already terminal wrapper point; never accepted as a full63 performance qualification"})
        result = json.loads(Path(record["result"]).read_text())
        first = next(iter(entries.values()))
        binding = {**result["identity"], "production_source_hash": first["production_source_hash"],
            "accuracy_contract": HistoricalAccuracyContracts.link({"path": str(fixture), "sha256": Evidence.sha256(fixture)}, job["variant_id"], first["point_key"])}
        verified = {}
        for operation in HistoryAudit.REQUIRED:
            verified[operation] = AccuracyContracts.resolve(binding, first["variant_id"], first["point_key"], operation,
                                                            first["input_hash"], first["reference_hashes"])
        bad = copy.deepcopy(binding)
        bad["accuracy_contract"]["entry_keys_by_operation"]["grad_x"] = bad["accuracy_contract"]["entry_keys_by_operation"]["forward"]
        try:
            AccuracyContracts.resolve(bad, first["variant_id"], first["point_key"], "grad_x", first["input_hash"], first["reference_hashes"])
        except ValueError:
            pass
        else:
            raise ValueError("Wrong operation upstream link accepted")
        Evidence.write(output, {"status": "CPU_SHARED_BINDING_THREE_REAL_OPERATION_LINKS_PASS", "verified": verified,
            "wrong_grad_x_to_forward_link_rejected": True, "one_identical_binding_all_three_operations": True,
            "shared_qualification_sha256": Evidence.sha256(Path(__file__).with_name("qualification.py")),
            "global_full63_qualified": False, "GPU_work_started": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("draft", "qualify", "selfcheck", "linkcheck"))
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--cycle")
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    session = Session(args.session)
    if args.action == "linkcheck":
        OperationLinkCheck.run(args.source_plan, args.output)
    elif args.action == "selfcheck":
        CpuCheck.run(args.scope, args.output)
    else:
        if args.action == "qualify" and args.audit is None:
            parser.error("qualify requires the completed independent --audit")
        path, value = HistoricalPerformanceScope.create(session, args.source_plan, args.cycle, args.audit if args.action == "qualify" else None)
        print(json.dumps({"scope": str(path), "qualified": value["qualified"], "families": value["variants"]}))


if __name__ == "__main__":
    main()
