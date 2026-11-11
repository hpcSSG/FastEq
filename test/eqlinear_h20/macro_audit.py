"""Separate preserved failed compiler commands from reported finite FP64 profiles."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

from common import Evidence
from registry import Serialization
from session import Session


class MacroEvidence:
    def __init__(self, session):
        self.session = session
        self.source = Path(session.paths["OPT_ROOT"]) / "gmm_src/cutlass_impl"

    @staticmethod
    def line(path, text):
        lines = Path(path).read_text().splitlines()
        found = next((i + 1, line.strip()) for i, line in enumerate(lines) if text in line)
        return {"path": str(path), "sha256": Evidence.sha256(Path(path)), "line": found[0], "text": found[1]}

    def run(self, output):
        output.mkdir(parents=True, exist_ok=False)
        old = self.session.root / "coverage/standalone_existing_choices07/variants.json"
        inventory = json.loads(old.read_text())["preserved_compile_tuples"]
        source = self.source / "csrc/sm90_fp64_grouped_gemm.cu"
        constraints = self.source / "csrc/collective/sm90_mma_fp64_array_tma_warpspecialized.hpp"
        defaults = {name: value for name, value in re.findall(r"#define\s+(F64M2_[A-Z0-9_]+)\s+(\d+)", source.read_text())}
        required = {"F64M2_TB_M", "F64M2_TB_N", "F64M2_TB_K", "F64M2_STAGES", "F64M2_EPI_STAGES", "F64M2_SMEM_ATOM", "F64M2_PINGPONG"}
        if set(defaults) != required:
            raise ValueError("FP64 profile defaults need source re-review")
        defaults["EQLIN_TMA_HINT"] = "0"
        failed = []
        for row in inventory:
            entry = copy.deepcopy(row)
            path = Path(entry["evidence"][0]["path"])
            if Evidence.sha256(path) != entry["evidence"][0]["sha256"]:
                raise ValueError("Preserved failed compiler log changed")
            stage = entry["definitions"]["F64M2_STAGES"]
            report = self.source / ("FP64_B28_REPORT.md" if stage == "1" else "FP64_B29_REPORT.md")
            needle = "stage1/epilogue1 build failed" if stage == "1" else "first 64x64 cooperative-schedule build"
            entry.update(status="PRESERVED_FAILED_BUILD_NOT_A_SUCCESSFUL_PROFILE", execution_performed=False,
                         report_failure_evidence=self.line(report, needle),
                         constraint="STAGES_GE_2" if stage == "1" else "COOPERATIVE_M_GE_128_AND_256_MMA_THREADS",
                         H20_build_attempted=False, counts_as_tested_implementation=False)
            failed.append(entry)
        recipes = (("B28_stage2_epi1", "FP64_B28_REPORT.md", "B28 used TMA stage2/epilogue1",
                    {"F64M2_STAGES": "2", "F64M2_EPI_STAGES": "1"}),
                   ("B29_pingpong64", "FP64_B29_REPORT.md", "tested build used 64x64x16",
                    {"F64M2_TB_M": "64", "F64M2_TB_N": "64", "F64M2_TB_K": "16", "F64M2_STAGES": "2",
                     "F64M2_EPI_STAGES": "1", "F64M2_PINGPONG": "1"}),
                   ("B30_TMA_hint", "FP64_B30_REPORT.md", "B30 exposes the existing TMA cache policy",
                    {"F64M2_TB_M": "128", "F64M2_TB_N": "64", "F64M2_STAGES": "4", "F64M2_EPI_STAGES": "2",
                     "F64M2_PINGPONG": "0", "EQLIN_TMA_HINT": "1"}))
        profiles = []
        for name, report, needle, explicit in recipes:
            resolved = {**defaults, **explicit}
            m, n, k, stages, ping = [int(resolved[key]) for key in ("F64M2_TB_M", "F64M2_TB_N", "F64M2_TB_K", "F64M2_STAGES", "F64M2_PINGPONG")]
            if stages < 2 or (m // 32) * (n // 32) * (k // 16) * 32 != (128 if ping else 256):
                raise ValueError("Reported profile conflicts with preserved source constraints")
            profiles.append({"profile_id": name, "precision": "fp64", "report_explicit_or_directly_implied_definitions": explicit,
                "resolved_current_source_definitions": resolved,
                "definitions_filled_from_current_source": {key: value for key, value in defaults.items() if key not in explicit},
                "report_evidence": self.line(self.source / report, needle), "source": str(source),
                "source_sha256": Evidence.sha256(source), "status": "REPORTED_PROFILE_PENDING_FRESH_H20_BUILD_PROBE",
                "reconstructability": "CURRENT_PRESERVED_SOURCE_REPLAY_ONLY_ORIGINAL_PRODUCER_UNPROVEN",
                "original_full_compiler_command_proven": False, "original_H100_binary_reproduced": False,
                "runtime_recipe": {"wgrad_kernel": 3, "wgrad_split_k": 4,
                    "unspecified_API_defaults": "resolve and declare current preserved defaults separately"},
                "source_constraints_satisfied": True, "H20_build_attempted": False})
        result = {"classification": "FINITE_PROFILE_DEPENDENCIES_NOT_H20_RESULTS", "failed_builds": failed, "reported_profiles": profiles,
            "source_constraints": [self.line(constraints, 'static_assert(DispatchPolicy::Stages >= 2'),
                                   self.line(source, 'static_assert(size(TiledMma{}) == (F64M2_PINGPONG ? 128 : 256)')],
            "parents": {"choices07": {"path": str(old), "sha256": Evidence.sha256(old)}},
            "unchanged_scopes": {"default_plus_named": 185, "named": 157, "historical_runtime_recipes": 8},
            "H100_rerun": False, "defaults_or_CUDA_math_modified": False, "all_existing_parameter_strategies_covered": False,
            "auditor_sha256": Evidence.sha256(Path(__file__))}
        Evidence.write(output / "audit.json", result)
        Serialization.csv(output / "failed_builds.csv", failed, sorted(set().union(*(row.keys() for row in failed))))
        Serialization.csv(output / "reported_profiles.csv", profiles, sorted(set().union(*(row.keys() for row in profiles))))
        print(json.dumps({"failed_builds": len(failed), "reported_profiles_pending_H20": len(profiles)}))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    MacroEvidence(Session(args.session)).run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
