"""Isolated builds of finite, report-bound standalone compiler profiles."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil

from common import Evidence
from session import Session
from standalone import StandaloneSource


class StandaloneProfiles:
    CMAKE_NAMES = {"EQLIN_TMA_HINT": "F64M2_TMA_CACHE_HINT"}

    @classmethod
    def audit_scope(cls, session, registry, output):
        if output.exists() or not output.resolve().is_relative_to(session.root):
            raise ValueError("Source-scope audit needs a fresh session output")
        value = json.loads(registry.read_text())
        source = StandaloneSource(session, "fp64")
        manifest = source.stage()
        profiles = {row["build_profile"]: row["profile_record"] for row in value["variants"]}
        macros, builds = set(), {}
        for name, link in profiles.items():
            path = Path(link["path"])
            if Evidence.sha256(path) != link["sha256"]:
                raise ValueError("Compiler-profile source evidence changed")
            record = json.loads(path.read_text())
            macros.update(record["macro_definitions"])
            builds[name] = {"record": link, "macro_definitions": record["macro_definitions"],
                "binary": record["binary"], "binary_hash": record["binary_hash"]}
        usages = {macro: [] for macro in sorted(macros)}
        files = {}
        for path in sorted((source.original / "csrc").rglob("*")):
            if not path.is_file() or path.suffix not in (".cu", ".cpp", ".cuh", ".h", ".hpp"):
                continue
            for line, text in enumerate(path.read_text().splitlines(), 1):
                for macro in macros:
                    if re.search(r"\b" + re.escape(macro) + r"\b", text):
                        usages[macro].append({"path": str(path), "line": line, "text": text.strip()})
                        files[str(path)] = Evidence.sha256(path)
        if any(not rows for rows in usages.values()):
            raise ValueError("A report-bound macro has no preserved production-source use")
        registered = {int(row["impl_id"]) for row in value["variants"] if row["operation"] == "forward"}
        if registered != {1, 2, 3, 5, 7}:
            raise ValueError("Compiler-profile scope omits a source-active existing forward route")
        for name in ("runner.cu", "eqlinear.h", "bwd.cu", "sm90_fp64_grouped_gemm.cu"):
            path = source.original / "csrc" / name
            files[str(path)] = Evidence.sha256(path)
        producer = output.with_suffix(".producer.py")
        if producer.exists():
            raise FileExistsError(producer)
        producer.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), producer)
        Evidence.write(output, {"classification": "BOUNDED_COMPILER_PROFILE_SOURCE_SCOPE_NOT_RUNTIME_RESULTS", "complete": True,
            "registry": {"path": str(registry), "sha256": Evidence.sha256(registry)}, "builds": builds,
            "source_hash": manifest["source_hash"], "source_files": files, "macro_uses": usages,
            "forward_routes_registered": sorted(registered), "source_inactive_forward_routes": [0, 4, 6],
            "routing_evidence": "runner.cu resolves impl1/2/3/5/7 to Sm90Fp64Plan on source-selected shapes; impl0 uses GroupedGemmPlan, impl4 DirectGemmPlan, impl6 the separate TMA direct kernel",
            "backward_routes": "bwd.cu kernel3 constructs Sm90Fp64Plan; report-bound wgrad_kernel3/split4 tuple and paired backward are registered",
            "point_policy": "Every registered route is tested on all25 canonical points, retaining unchanged automatic fallbacks",
            "unmeasured_cross_products": "No report supports pairing each compiler profile with every independent runtime policy; these arbitrary combinations are not introduced",
            "archive_provenance": "CURRENT_PRESERVED_SOURCE_REPLAY_ONLY_ORIGINAL_PRODUCER_UNPROVEN",
            "producer": {"path": str(producer), "sha256": Evidence.sha256(producer)}, "GPU_work_started": False})
        return 0

    @classmethod
    def register(cls, session, inventory, output, source_active_forward=False):
        from standalone_matrix import ExistingChoices

        if not output.resolve().is_relative_to(session.root) or output.exists():
            raise ValueError("Profile registry needs a fresh session path")
        with inventory.open(newline="") as stream:
            profiles = list(csv.DictReader(stream))
        choices = ExistingChoices(session)
        for profile in profiles:
            name = profile["profile_id"]
            path = session.root / "build_profiles" / name / "profile.json"
            record = json.loads(path.read_text())
            if record["status"] != "BUILD_PASS_REQUIRES_H20_PROBE":
                raise ValueError("Fresh profile build did not pass: " + name)
            report = json.loads(profile["report_evidence"])
            bases = ["standalone_fp64_wgrad", "standalone_fp64_bwd"]
            if source_active_forward:
                bases += ["standalone_fp64_impl" + str(impl) for impl in (1, 2, 3, 5, 7)]
            for base in bases:
                forward = "_impl" in base
                evidence = [report]
                if forward:
                    evidence += [choices.evidence("fp64", "csrc/runner.cu", needle) for needle in
                        ("auto const auto3_choice", "impl == F64Impl::kAuto ||", "effective == F64Impl::kSm90Dmma")]
                    evidence.append(choices.evidence("fp64", "csrc/sm90_fp64_grouped_gemm.cu", "using TileShape"))
                choices.add(base, "profile_" + name, {}, {} if forward else {"wgrad_kernel": 3, "wgrad_split_k": 4}, evidence,
                    "compiler profile affects Sm90Fp64Plan; automatic forward fallbacks retain current source routing" if forward else
                    "documented compiler profile, SM90 packed wgrad kernel3/split4; current unspecified defaults", not forward)
                row = choices.rows[-1]
                row.update(build_profile=name, profile_record={"path": str(path), "sha256": Evidence.sha256(path)},
                    strategy_kind="source_active_forward_of_documented_compiler_profile" if forward else "documented_compiler_profile_current_source_replay",
                    probe_point={"config": "LARGE_P3", "batch": 512},
                    reconstruction_status="CURRENT_PRESERVED_SOURCE_REPLAY_ONLY_ORIGINAL_PRODUCER_UNPROVEN")
                row["semantic_contract_hash"] = Evidence.fingerprint({key: row[key] for key in
                    ("base_variant_id", "runtime_controls", "api_controls", "build_profile", "profile_record", "probe_point")})
        cells = sum(3 if "_impl" in row["base_variant_id"] else 2 if row["base_variant_id"].endswith("_bwd") else 1 for row in choices.rows) * 25
        Evidence.write(output, {"classification": "FRESH_BUILT_PROFILE_CONTRACTS_NOT_H20_RESULTS", "variants": choices.rows,
            "profiles": len(profiles), "contracts": len(choices.rows), "full25_jobs": len(choices.rows) * 25,
            "full25_cells": cells, "source_active_forward_registered": source_active_forward,
            "excluded_compile_parameter_cross_products": "No compiler-profile by runtime-policy Cartesian product; only report-bound backward tuple and source-active default forward routes",
            "production_defaults_changed": False,
            "all_existing_parameter_strategies_covered": False})
        return 0

    @classmethod
    def build(cls, session, inventory, selected):
        with inventory.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows = [row for row in rows if not selected or row["profile_id"] == selected]
        if not rows:
            raise ValueError("No documented compiler profile selected")
        results = []
        for row in rows:
            name = row["profile_id"]
            if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
                raise ValueError("Invalid compiler profile id")
            root = session.root / "build_profiles" / name
            root.mkdir(parents=True, exist_ok=False)
            shutil.copy2(inventory, root / "inventory.csv")
            macro_definitions = json.loads(row["resolved_current_source_definitions"])
            options = {cls.CMAKE_NAMES.get(key, key): value for key, value in macro_definitions.items()}
            source = StandaloneSource(session, row["precision"], root / "build")
            record = {"classification": "CURRENT_SOURCE_HISTORICAL_PROFILE_REPLAY_NOT_ORIGINAL_BINARY", "profile_id": name,
                "source_recipe": row, "inventory": {"path": str(root / "inventory.csv"), "sha256": Evidence.sha256(root / "inventory.csv")},
                "macro_definitions": macro_definitions, "cmake_options": options,
                "source_manifest": source.stage(), "status": "BUILDING", "production_defaults_changed": False}
            Evidence.write(root / "profile.json", record)
            terminal = source.compile("profile_" + name + "_01", options)
            record.update(status="BUILD_PASS_REQUIRES_H20_PROBE" if terminal["status"] == "PASS" else "COMPILE_FAIL",
                          build_terminal=terminal)
            if terminal["status"] == "PASS":
                ninja = source.build / "build.ninja"
                actual = ninja.read_text()
                if any("-D" + key + "=" + str(value) not in actual for key, value in macro_definitions.items()):
                    raise ValueError("Actual compiler definitions differ from the report-bound profile")
                record.update(binary=str(source.library), binary_hash=Evidence.sha256(source.library),
                    cmake_cache={"path": str(source.build / "CMakeCache.txt"), "sha256": Evidence.sha256(source.build / "CMakeCache.txt")},
                    ninja={"path": str(ninja), "sha256": Evidence.sha256(ninja)})
            Evidence.write(root / "profile.json", record)
            results.append({"profile_id": name, "status": record["status"], "record": str(root / "profile.json")})
        Evidence.write(session.root / "manifests" / ("standalone_profiles_" + (selected or "all") + "_01.json"),
                       {"profiles": results, "GPU_work_started": False})
        return 0 if all(row["status"] == "BUILD_PASS_REQUIRES_H20_PROBE" for row in results) else 1


class ProfileCLI:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--inventory", type=Path, required=True)
        parser.add_argument("--profile")
        parser.add_argument("--register", action="store_true")
        parser.add_argument("--source-active-forward", action="store_true")
        parser.add_argument("--audit-scope", type=Path, help="Read back the registered finite source-active route scope")
        parser.add_argument("--output", type=Path)
        args = parser.parse_args()
        if args.audit_scope:
            if args.output is None:
                parser.error("--audit-scope requires --output")
            return StandaloneProfiles.audit_scope(Session(args.session), args.audit_scope, args.output)
        if args.register:
            if args.output is None:
                parser.error("--register requires --output")
            return StandaloneProfiles.register(Session(args.session), args.inventory, args.output, args.source_active_forward)
        return StandaloneProfiles.build(Session(args.session), args.inventory, args.profile)


if __name__ == "__main__":
    raise SystemExit(ProfileCLI.main())
