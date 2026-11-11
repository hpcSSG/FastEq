"""Immutable upstream qualification rows derived only from complete audited scopes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from cases import points
from common import Evidence
from standalone_matrix import StandalonePlan


class StandaloneAccuracyContracts:
    @staticmethod
    def union_policies(paths, output):
        if output.exists():
            raise FileExistsError(output)
        rows, sources = [], []
        for path in paths:
            value = json.loads(path.read_text())
            rows += value["variants"]
            sources.append({"path": str(path.resolve()), "sha256": Evidence.sha256(path)})
        if len(rows) != len({row["variant_id"] for row in rows}):
            raise ValueError("Finite policy scopes overlap")
        Evidence.write(output, {"classification": "DISJOINT_FINITE_POLICY_UNION_NOT_RESULTS", "variants": rows,
            "sources": sources, "named_policy_count": len(rows), "old_scope_counts_modified": False,
            "all_existing_parameter_strategies_covered": False})
        return len(rows)

    @staticmethod
    def entry_key(variant, point_key, operation):
        return "/".join((variant, point_key, operation))

    @staticmethod
    def validate(path, output):
        from qualification import AccuracyContracts

        value = json.loads(path.read_text())
        digest = Evidence.sha256(path)
        checked, passed = 0, 0
        for key, entry in value["entries"].items():
            source = json.loads(Path(entry["evidence"]["path"]).read_text())
            binding = {**AccuracyContracts.identity(source), "production_source_hash": entry["production_source_hash"]}
            binding["accuracy_contract"] = {"path": str(path.resolve()), "sha256": digest, "entry_key": key}
            result = AccuracyContracts.resolve(binding, entry["variant_id"], entry["point_key"], entry["operation"],
                                               entry["input_hash"], entry["reference_hashes"])
            if result["passed"] is not entry["passed"]:
                raise ValueError("Shared independent qualification disagrees with original complete audit")
            checked += 1
            passed += int(result["passed"])
        if output.exists():
            raise FileExistsError(output)
        Evidence.write(output, {"classification": "CPU_INDEPENDENT_QUALIFICATION_READBACK", "passed": True,
            "manifest": {"path": str(path.resolve()), "sha256": digest}, "checked_entries": checked,
            "qualified_entries": passed, "shared_qualification_sha256": Evidence.sha256(Path(__file__).with_name("qualification.py")),
            "GPU_work_started": False})
        return checked, passed

    @classmethod
    def create(cls, plans, audits, output):
        if not plans or len(plans) != len(audits) or output.exists():
            raise ValueError("Use complete paired plan/audit inputs and a fresh output")
        entries, provenance, files, coverage = {}, [], {}, {}
        session_root = None
        for plan_path, audit_path in zip(plans, audits):
            plan = StandalonePlan(plan_path)
            audit = json.loads(audit_path.read_text())
            session_root = session_root or Path(plan.value["session"])
            if Path(plan.value["session"]) != session_root or not output.resolve().is_relative_to(session_root):
                raise ValueError("Qualification inputs/output must belong to one session")
            if (plan.value["stage"] != "full" or not audit.get("complete") or not audit.get("full_plan_complete") or
                    audit.get("plan_hash") != plan.value["plan_hash"] or
                    set(audit["records"]) != {job["job_id"] for job in plan.value["jobs"]}):
                raise ValueError("Partial/invalid epochs cannot supply qualification contracts")
            files[str(plan_path.resolve())], files[str(audit_path.resolve())] = Evidence.sha256(plan_path), Evidence.sha256(audit_path)
            provenance.append({"plan": str(plan_path.resolve()), "plan_hash": plan.value["plan_hash"],
                               "audit": str(audit_path.resolve()), "scope": plan.value.get("variant_scope", "default")})
            for job in plan.value["jobs"]:
                record = audit["records"][job["job_id"]]
                path = Path(record["result"])
                if not record.get("verified") or Evidence.sha256(path) != record["result_sha256"]:
                    raise ValueError("Audited original result bytes changed")
                result = json.loads(path.read_text())
                point_key = "/".join(map(str, (job["point"]["config"], job["point"]["batch"], job["precision"])))
                key = cls.entry_key(job["variant_id"], point_key, job["operation"])
                if key in entries:
                    raise ValueError("Duplicate contract would mix epochs: " + key)
                identity = result["identity"]
                passed = record["accuracy_pass"]
                reasons = sorted({row["reason"] for row in result["rows"] if not row["passed"]})
                entries[key] = {"variant_id": job["variant_id"], "point_key": point_key, "operation": job["operation"],
                    "input_hash": result["input_hash"], "reference_hashes": result["reference_hashes"],
                    "production_source_hash": identity["source_hash"], "binary_hash": identity["binary_hash"],
                    "runtime_controls": identity.get("runtime_controls", {}), "api_controls": identity.get("api_controls", {}),
                    "passed": passed, "evidence": {"path": str(path), "sha256": record["result_sha256"]},
                    "reason": "FULL_CONTRACT_PASS" if passed else ";".join(reasons),
                    "required_boundaries": job["boundaries"], "upstream_gpu_uuid": result["gpu_uuid"],
                    "upstream_plan_hash": plan.value["plan_hash"], "upstream_audit": str(audit_path.resolve()),
                    "activation_layout": identity.get("activation_layout", "biu")}
                coverage.setdefault((job["variant_id"], job["precision"]), set()).add(point_key)
        for (_, precision), observed in coverage.items():
            if observed != {point.key for point in points() if point.precision == precision}:
                raise ValueError("A variant is missing canonical full25 accuracy points")
        output.parent.mkdir(parents=True, exist_ok=True)
        producer = output.with_suffix(".producer.py")
        if producer.exists():
            raise FileExistsError(producer)
        shutil.copy2(Path(__file__), producer)
        files[str(producer)] = Evidence.sha256(producer)
        value = {"schema": 1, "complete": True, "entries": entries, "evidence_files": files,
            "provenance": provenance, "expected_entries": len(entries), "variants": len(coverage),
            "qualified_entries": sum(row["passed"] for row in entries.values()),
            "qualification_scope": "Base numeric/repeat/buffer contract; unmatched frozen H100 P14 bound remains unverified",
            "original_result_files_modified": False, "partial_epoch_mixing": False,
            "scope_union": "disjoint complete per-variant full25 epochs"}
        Evidence.write(output, value)
        return value


class QualificationCLI:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, action="append")
        parser.add_argument("--audit", type=Path, action="append")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--validate-manifest", type=Path)
        parser.add_argument("--union-policies", type=Path, nargs="+")
        args = parser.parse_args()
        if args.union_policies:
            count = StandaloneAccuracyContracts.union_policies(args.union_policies, args.output)
            print(json.dumps({"named_policies": count, "GPU_work_started": False}))
            return 0
        if args.validate_manifest:
            checked, passed = StandaloneAccuracyContracts.validate(args.validate_manifest, args.output)
            print(json.dumps({"checked": checked, "qualified": passed, "GPU_work_started": False}))
            return 0
        if not args.plan or not args.audit:
            parser.error("--plan and --audit are required when creating a manifest")
        value = StandaloneAccuracyContracts.create(args.plan, args.audit, args.output)
        print(json.dumps({"complete": value["complete"], "entries": len(value["entries"]),
                          "qualified": value["qualified_entries"], "GPU_work_started": False}))
        return 0


if __name__ == "__main__":
    raise SystemExit(QualificationCLI.main())
