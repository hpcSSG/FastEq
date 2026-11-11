"""Executable remaining-history dependency scope, preserving every distinct wrapper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import Evidence
from historical_plan import HistoricalContracts
from producer_closure import ProducerClosure
from session import Session


class HistoricalScope:
    def __init__(self, session, history_repo):
        self.session = session
        self.history_repo = history_repo
        self.path = session.root / "coverage/historical_execution_plan05.json"
        self.value = json.loads(self.path.read_text())
        self.contracts = HistoricalContracts(session, history_repo)

    def create(self, output):
        if output.exists() or not output.resolve().is_relative_to(self.session.root):
            raise ValueError("Use a fresh historical scope directory under the configured session")
        cached, _ = self.contracts.existing()
        required = [row for row in self.value["wrapper_tests"] if row["known_default_accuracy_coverage"] == "UNTESTED"]
        if len(required) != 63 or len({row["variant_id"] for row in required}) != 63:
            raise ValueError("Remaining history must preserve the reviewed63 source/precision wrapper contracts")
        missing = [row for key, row in self.value["native_groups"].items() if key not in cached]
        producer = output / "producer"
        producer.mkdir(parents=True)
        sources = ProducerClosure(Path(__file__).parent, producer).freeze(
            ("historical_scope.py", "historical.py", "historical_native.py", "historical_plan.py",
             "historical_matrix.py", "historical_probe.py", "registry.py", "reference_store.py", "standalone_matrix.py"))
        python = self.session.paths["FASTEQ_PY"]

        def command(action, variant):
            result = [python, "-B", str(producer / "historical.py"), action, "--session", str(self.session.root),
                      "--variant", variant, "--attempt", "historyscope01"]
            if action == "prepare":
                result += ["--history-repo", str(self.history_repo)]
            return result

        native_jobs = [{"key": row["key"], "precision": row["precision"], "variant_id": row["representative_contract"],
                       "commands": {action: command(action, row["representative_contract"])
                                    for action in ("prepare", "configure", "build", "verify")},
                       "compile_deadline_seconds": 600, "jobs": 2}
                      for row in missing]
        wrappers = [{"variant_id": row["variant_id"], "labels": row["labels"], "precision": row["precision"],
                     "runtime_controls": row["runtime_controls"], "native_keys": row["native_keys"],
                     "dispatcher_key": row["dispatcher_key"],
                     "commands": {action: command(action, row["variant_id"])
                                  for action in ("prepare", "configure", "build", "verify")},
                     "accuracy_points": 25, "accuracy_paths": 125,
                     "accuracy_status": "PENDING_SOURCE_ABI_FREEZE_AND_FULL25_CYCLE"}
                    for row in required]
        value = {"classification": "EXECUTABLE_REMAINING_HISTORY_SCOPE_NOT_RESULTS", "session": str(self.session.root),
                 "parent": {"path": str(self.path), "sha256": Evidence.sha256(self.path)},
                 "producer": sources, "native_build_jobs": native_jobs, "wrapper_contracts": wrappers,
                 "summary": {"missing_native_groups": len(native_jobs), "remaining_wrappers": 63,
                             "accuracy_points": 1575, "accuracy_paths": 7875},
                 "native_reuse_rule": "Exact source, headers, build flags, ABI and mappedDSO; distinct host wrappers remain distinct tests",
                 "grouping_policy": "DifferentDSO sets stay in independent processes; identicalDSO wrapper batches require isolated equivalence proof first",
                 "queued_safety_verified": False, "H100_frozen_additional_gate": "UNVERIFIED_INPUT_HASH_MISMATCH"}
        value["scope_hash"] = Evidence.fingerprint(value)
        Evidence.write(output / "plan.json", value)
        return value


class HistoricalBuildStep:
    """Resume at the next declared build stage; never restart a live/failed process."""

    def __init__(self, path, variant, label):
        self.path, self.variant, self.label = path, variant, label
        self.plan = json.loads(path.read_text())
        self.session = Session(Path(self.plan["session"]))
        if Evidence.fingerprint({key: value for key, value in self.plan.items() if key != "scope_hash"}) != self.plan["scope_hash"]:
            raise ValueError("Historical remaining-scope fingerprint changed")
        for name, sha in self.plan["producer"].items():
            if Evidence.sha256(path.parent / "producer" / name) != sha:
                raise ValueError("Historical frozen producer changed")
        self.job = next(row for row in self.plan["native_build_jobs"] + self.plan["wrapper_contracts"]
                        if row["variant_id"] == variant)

    @staticmethod
    def configured(binding):
        if not binding or not (Path(binding["build"]) / "compile_commands.json").is_file():
            return False
        expected = {precision for precision, value in binding["native_bindings"].items() if value["reuse"]}
        verified = {row["precision"] for row in binding.get("native_compile_flag_reuse_guards", []) if row.get("passed") is True}
        return expected == verified

    def run(self):
        record = self.session.root / "manifests" / ("historical_" + self.variant + "_build.json")
        receipt = self.path.parent / ("build_" + self.variant + "_" + self.label + ".json")
        if receipt.exists():
            raise FileExistsError("Preserve old build attempts")
        state = {"variant": self.variant, "classification": "HISTORICAL_SOURCE_BUILD_NOT_ACCURACY",
                 "scope_hash": self.plan["scope_hash"], "status": "RUNNING", "steps": []}
        Evidence.write(receipt, state)
        for action, deadline in (("prepare", 90), ("configure", 120), ("build", 600), ("verify", 90)):
            binding = json.loads(record.read_text()) if record.exists() else None
            built = binding is not None and binding.get("status") == "BUILT_SOURCE_ELF_ARCH_VERIFIED"
            skip = ((action == "prepare" and binding is not None) or
                    (action == "configure" and self.configured(binding)) or
                    (action == "build" and built))
            if skip:
                state["steps"].append({"action": action, "status": "ALREADY_PRESENT_SOURCE_GUARDED_AT_VERIFY"})
                continue
            command = list(self.job["commands"][action])
            command[command.index("--attempt") + 1] = self.label
            terminal = self.session.runner().run("historyscope_" + self.variant + "_" + self.label + "_" + action,
                command, deadline, Path(self.session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": ""})
            state["steps"].append({"action": action, "status": terminal["status"], "seconds": terminal["elapsed_seconds"]})
            Evidence.write(receipt, state)
            if terminal["status"] != "PASS":
                state["status"] = "BUILD_STAGE_REQUIRES_DIAGNOSIS"
                Evidence.write(receipt, state)
                return 1
        binding = json.loads(record.read_text())
        if binding.get("status") != "BUILT_SOURCE_ELF_ARCH_VERIFIED":
            raise ValueError("Build did not produce a source/ELF/architecture verified receipt")
        state.update(status="BUILT_SOURCE_ELF_ARCH_VERIFIED", build_manifest=str(record), build_manifest_sha256=Evidence.sha256(record))
        Evidence.write(receipt, state)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "build-one"), nargs="?", default="plan")
    parser.add_argument("--session", type=Path)
    parser.add_argument("--history-repo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--label", default="scope01")
    args = parser.parse_args()
    if args.action == "build-one":
        return HistoricalBuildStep(args.plan, args.variant, args.label).run()
    if not all((args.session, args.history_repo, args.output)):
        parser.error("plan requires --session, --history-repo and --output")
    value = HistoricalScope(Session(args.session), args.history_repo).create(args.output)
    print(json.dumps(value["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
