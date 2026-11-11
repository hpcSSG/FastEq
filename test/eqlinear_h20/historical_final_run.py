"""Finish the existing historical factory with one shared sampler and bounded replay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from common import Evidence
from performance import PerformancePlan
from historical_performance import HistoricalPerformanceFamily
from producer_closure import ProducerClosure
from session import Session


class HistoricalFinalRun:
    CPU, GPU = 198, 7

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "campaign.json"
        self.value = json.loads(self.path.read_text())
        self.session = Session(Path(self.value["session"]))
        self.tools = self.root / "tools"
        self.verify()

    @classmethod
    def prepare(cls, session, root):
        root.mkdir(parents=True, exist_ok=False)
        tools = root / "tools"
        tools.mkdir()
        seeds = set(PerformancePlan.FILES) | set(HistoricalPerformanceFamily.FILES) | {
            "historical_final_run.py", "performance_driver.py", "pilot_audit.py", "timing_audit.py"}
        sources = ProducerClosure(Path(__file__).parent, tools).freeze(sorted(seeds))
        original = session.root / "accuracy/historical_batch/historyfull63_01/plan.json"
        files = {str(original): Evidence.sha256(original)}
        for path in (session.root / "coverage/final_scope_user_resume01/decision.json",
                     session.root / "manifests/final_cpu_affinity_adjudication01.json"):
            files[str(path)] = Evidence.sha256(path)
        value = {"classification": "EXISTING63_FACTORY_FINAL_PERFORMANCE_EXECUTION_NOT_NEW_ADAPTER",
            "session": str(session.root), "source_plan": str(original), "source_tools": sources, "evidence_files": files,
            "accuracy_audit": str(original.parent / "full_audit.json"), "cpu": cls.CPU, "gpu": cls.GPU,
            "sampler_directory": str(session.root / "timing/historical_final63_cpu198_c01"),
            "factory_scope_cycle": "historical_final63_cpu198_01", "noise_repeat_limit": 1,
            "repeat_policy": "Only a complete affected-family full25 scan, all3 rounds, once; never individual point retiming",
            "initial_qualification": "Wait for complete independently audited same original accuracy math epoch; no partial promotion",
            "new_adapters_versions_parameters": False}
        value["campaign_hash"] = Evidence.fingerprint(value)
        Evidence.write(root / "campaign.json", value)
        return value

    def verify(self):
        if Evidence.fingerprint({k: v for k, v in self.value.items() if k != "campaign_hash"}) != self.value["campaign_hash"]:
            raise ValueError("Historical execution campaign changed")
        for name, digest in self.value["source_tools"].items():
            if Evidence.sha256(self.tools / name) != digest:
                raise ValueError("Frozen execution/sampler/auditor tools changed")
        for path, digest in self.value["evidence_files"].items():
            if Evidence.sha256(Path(path)) != digest:
                raise ValueError("Final scope/source/CPU evidence changed")
        if self.value["cpu"] != 198 or self.value["gpu"] != 7 or self.value["noise_repeat_limit"] != 1:
            raise ValueError("Unapproved execution resources or replay limit")

    def stage(self, name, argv, deadline):
        self.verify()
        Evidence.write(self.root / "status.json", {"status": "RUNNING", "stage": name,
            "utc": datetime.now(timezone.utc).isoformat(), "cpu": self.CPU, "gpu": self.GPU})
        result = self.session.runner().run("historical_final63_" + name, ["taskset", "-c", str(self.CPU), *argv], deadline,
            Path(self.session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": ""})
        if result["status"] != "PASS":
            Evidence.write(self.root / "status.json", {"status": "STOPPED_REQUIRES_DIAGNOSIS", "stage": name, "receipt": result})
            raise RuntimeError("Historical final stage stopped; never restart nonterminal/error processes blindly: " + name)
        self.verify()
        return result

    def wait_accuracy(self):
        target = Path(self.value["accuracy_audit"])
        start = time.monotonic()
        while not target.exists():
            terminal = self.session.root / "logs/history_full63_user_resume_controller01.json"
            if terminal.exists() and json.loads(terminal.read_text())["status"] != "PASS":
                raise RuntimeError("Accuracy recovery stopped; no qualification or performance started")
            audit_terminal = self.session.root / "logs/history_full63_user_resume_final_audit01.json"
            if audit_terminal.exists() and json.loads(audit_terminal.read_text())["status"] != "PASS":
                raise RuntimeError("Accuracy audit failed; no qualification or performance started")
            if time.monotonic() - start > 15000:
                raise TimeoutError("Full accuracy still incomplete; no performance started")
            time.sleep(5)
        audit = json.loads(target.read_text())
        if not audit.get("complete") or not audit.get("full_registry_complete") or audit.get("expected_points") != 1575 or audit.get("expected_paths") != 7875:
            raise ValueError("Partial accuracy audit cannot qualify final historical performance")
        Evidence.write(self.root / "upstream_accuracy.json", {"path": str(target), "sha256": Evidence.sha256(target),
            "complete": True, "recovery": audit.get("recovery"), "counts": audit["counts"]})

    def assert_affinity(self, sampler, cycle=None):
        observed = 0
        for path in sampler.glob("points/*/*/setup.json"):
            identity = json.loads(path.read_text())["identity"]
            if cycle is not None and identity["frame"]["cycle"] != cycle:
                continue
            if identity["affinity"] != {"allowed": [self.CPU], "selected": self.CPU}:
                raise ValueError("Historical performance worker did not retain the isolated CPU198 affinity")
            observed += 1
        if observed == 0:
            raise ValueError("No historical performance setup affinity was observed")
        return observed

    def audit_families(self, sampler, families, cycle):
        reports = {}
        for family in families:
            output = sampler / "audits" / ("cycle" + str(cycle)) / (family + ".json")
            self.stage("audit_c" + str(cycle) + "_" + family,
                [self.session.paths["FASTEQ_PY"], "-B", str(self.tools / "timing_audit.py"), "--plan", str(sampler / "plan.json"),
                 "--family", family, "--cycle", str(cycle), "--output", str(output)], 300)
            report = json.loads(output.read_text())
            if not report.get("full_factory_scope_complete") or report.get("scope_point_count") != 25:
                raise ValueError("Historical precision-locked family timing audit is incomplete")
            reports[family] = {"path": str(output), "sha256": Evidence.sha256(output),
                "requires_complete_scan_repeat": report["requires_complete_scan_repeat"],
                "stability_limited": report["stability_limited"], "cycle": cycle}
        return reports

    def run(self):
        if os.sched_getaffinity(0) != {self.CPU}:
            raise ValueError("Launch the entire historical final controller with taskset -c198")
        self.wait_accuracy()
        py = self.session.paths["FASTEQ_PY"]
        source = json.loads(Path(self.value["source_plan"]).read_text())
        families = list(source["sources"])
        sampler = Path(self.value["sampler_directory"])
        self.stage("qualification01", [py, "-B", str(self.tools / "historical_performance.py"), "qualify",
            "--session", str(self.session.root), "--source-plan", self.value["source_plan"], "--audit", self.value["accuracy_audit"],
            "--cycle", self.value["factory_scope_cycle"]], 600)
        self.stage("create_plan01", [py, "-B", str(self.tools / "performance.py"), "create", "--session", str(self.session.root),
            "--directory", str(sampler), "--factory", "historical_performance.HistoricalPerformanceFamily", "--families", ",".join(families)], 120)
        driver = sampler / "producer/performance_driver.py"
        base = [py, "-B", str(driver), "--plan", str(sampler / "plan.json"), "--families", ",".join(families), "--gpu", str(self.GPU), "--chunk-deadline", "600"]
        self.stage("full75_pilot01", base + ["--phase", "pilot"], 10800)
        self.assert_affinity(sampler)
        self.stage("pilot_independent_audit01", [py, "-B", str(self.tools / "pilot_audit.py"), "--plan", str(sampler / "plan.json"),
            "--output", str(sampler / "pilot_audit.json")], 600)
        for round_index in range(3):
            self.stage("cycle1_round" + str(round_index), base + ["--phase", "round", "--cycle", "1", "--round", str(round_index)], 21600)
            self.assert_affinity(sampler, 1)
        reports = self.audit_families(sampler, families, 1)
        repeat = [family for family in families if reports[family]["requires_complete_scan_repeat"]]
        if repeat:
            replay = [py, "-B", str(driver), "--plan", str(sampler / "plan.json"), "--families", ",".join(repeat), "--gpu", str(self.GPU), "--chunk-deadline", "600"]
            for round_index in range(3):
                self.stage("cycle2_round" + str(round_index), replay + ["--phase", "round", "--cycle", "2", "--round", str(round_index)], 21600)
                self.assert_affinity(sampler, 2)
            reports.update(self.audit_families(sampler, repeat, 2))
        value = {"classification": "HISTORICAL63_EXISTING_ADAPTER_FINAL_COMPARISON_EVIDENCE", "complete": True,
            "families": 63, "points_per_family": 25, "formal_rounds_per_scan": 3, "sampler_plan": str(sampler / "plan.json"),
            "upstream_accuracy": str(self.root / "upstream_accuracy.json"), "final_family_audits": reports,
            "noise_repeat_families": repeat, "noise_repeats_per_family_max": 1, "raw_deleted_or_trimmed": False,
            "cpu": self.CPU, "gpu": self.GPU, "final_user_scope": self.value["evidence_files"],
            "H100_frozen_additional_gate": "UNVERIFIED_INPUT_HASH_MISMATCH"}
        Evidence.write(self.root / "final_manifest.json", value)
        Evidence.write(self.root / "status.json", {"status": "COMPLETE", "manifest": str(self.root / "final_manifest.json")})
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--session", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        HistoricalFinalRun.prepare(Session(args.session), args.root)
        return 0
    return HistoricalFinalRun(args.root).run()


if __name__ == "__main__":
    raise SystemExit(main())
