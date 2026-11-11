"""Historical accuracy config chunks and full-cycle audit using shared CPU orchestration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from audit_accuracy import AccuracyAudit
from cases import CONFIGS, points
from common import Evidence
from current_matrix import AccuracyChunk
from session import Session


class HistoricalIdentity:
    @staticmethod
    def observe(session, variant):
        import torch

        path = session.root / "manifests" / ("historical_" + variant + "_build.json")
        manifest = json.loads(path.read_text())
        if manifest["status"] != "BUILT_SOURCE_ELF_ARCH_VERIFIED":
            raise ValueError("Historical source/ELF/architecture verification required")
        binaries = {row["path"]: Evidence.sha256(Path(row["path"])) for row in manifest["binaries"]}
        if binaries != {row["path"]: row["sha256"] for row in manifest["binaries"]}:
            raise RuntimeError("Historical binary changed after build")
        source = {name: Evidence.sha256(Path(manifest["source"]) / name)
                  for name in manifest["source_manifest"]["source_sha256"]}
        if source != manifest["source_manifest"]["source_sha256"]:
            raise RuntimeError("Historical mathematical sources changed after build")
        abi = f"{torch.__version__} {torch.version.cuda} {torch._C._GLIBCXX_USE_CXX11_ABI}"
        if abi != manifest["torch_abi"]:
            raise RuntimeError("Historical runtime ABI differs from build")
        return {"source_hash": Evidence.fingerprint(source), "binary_hash": Evidence.fingerprint(binaries),
                "source_commit": manifest["commit"], "namespace": ("eqlinear_" + manifest.get("backend_namespace", variant)
                                                                   if manifest.get("backend_namespace", variant) else "eqlinear"),
                "source_manifest_sha256": manifest["source_manifest_sha256"], "torch_abi": abi,
                "build_manifest_path": str(path), "build_manifest_sha256": Evidence.sha256(path)}


class HistoricalChunk(AccuracyChunk):
    FILES = ("historical_matrix.py", "historical_probe.py", "historical.py", "historical_plan.py", "historical_native.py",
             "reference_store.py", "registry.py", "current_matrix.py",
             "current_probe.py", "current.py", "audit_accuracy.py", "cases.py", "common.py", "protocol.py", "session.py")

    def family_name(self):
        return "historical_" + self.args.variant

    def selected_points(self):
        selected = super().selected_points()
        if self.args.variant in ("p0", "p14"):
            return selected
        manifest = json.loads((self.session.root / "manifests" / ("historical_" + self.args.variant + "_build.json")).read_text())
        return [point for point in selected if point.precision == manifest["precision"]]

    def probe_argv(self, producer, point, output):
        command = [self.session.paths["FASTEQ_PY"], "-B", str(producer / "historical_probe.py"),
                "--session", str(self.session.root), "--variant", self.args.variant, "--output", str(output), "--point", point.key]
        if self.args.variant not in ("p0", "p14"):
            command.append("--reference-store")
        return command

    def accept_terminal(self, result, terminal):
        valid_producer = result.get("producer_unchanged") is True
        if terminal["status"] == "PASS":
            return valid_producer and result.get("status") == "PASS"
        return (valid_producer and terminal["status"] == "FAIL" and terminal.get("returncode") == 1
                and result.get("status") == "ACCURACY_FAIL")

    def run(self):
        identity = HistoricalIdentity.observe(self.session, self.args.variant)
        name = f"{self.family_name()}_accuracy_{self.args.config}_{self.args.attempt}"
        path = self.session.root / "accuracy" / (name + ".source_freeze.json")
        if path.exists():
            raise FileExistsError("Use a fresh cycle; never overwrite a source freeze")
        Evidence.write(path, {"status": "RUNNING", "identity": identity, "source_and_abi_frozen_in_parent": True})
        previous_pythonpath = os.environ.get("PYTHONPATH")
        source_prefix = json.loads(Path(identity["build_manifest_path"]).read_text())["source"]
        os.environ["PYTHONPATH"] = source_prefix + (os.pathsep + previous_pythonpath if previous_pythonpath else "")
        try:
            code = super().run()
            after = HistoricalIdentity.observe(self.session, self.args.variant)
            if after != identity:
                raise RuntimeError("Historical source/binary/ABI drifted during chunk")
            Evidence.write(path, {"status": "VERIFIED_UNCHANGED", "identity": identity,
                                  "source_and_abi_frozen_in_parent": True, "child_fingerprints_verified_per_point": True})
            return code
        except BaseException as error:
            Evidence.write(path, {"status": "FAILED_OR_INTERRUPTED", "identity": identity,
                                  "error_type": type(error).__name__, "error": str(error)})
            raise
        finally:
            if previous_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pythonpath


class HistoricalAudit(AccuracyAudit):
    def selected_points(self):
        selected = super().selected_points()
        if self.args.variant in ("p0", "p14"):
            return selected
        path = self.session.root / "manifests" / ("historical_" + self.args.variant + "_build.json")
        precision = json.loads(path.read_text())["precision"]
        return [point for point in selected if point.precision == precision]

    def expected_identity(self):
        self.frozen = HistoricalIdentity.observe(self.session, self.args.variant)
        return {key: self.frozen[key] for key in ("source_hash", "binary_hash", "source_commit", "namespace", "source_manifest_sha256")}

    def attempt_name(self, point, cycle):
        return f"historical_{self.args.variant}_accuracy_{point.config}_{cycle}_{point.batch}_{point.precision}"

    def producer_unchanged(self, result):
        return result.get("producer_unchanged") is True and result.get("identity", {}).get("isolated_current_package_not_imported") is True

    def run(self):
        # The base auditor checks each actual fixture SHA, dtype/stride, reference binding,
        # exact operation set, independent numerical thresholds and one producer epoch.
        code = super().run()
        report = json.loads(self.args.output.read_text())
        guards = []
        selected = self.selected_points()
        for config in CONFIGS:
            path = self.session.root / "accuracy" / f"historical_{self.args.variant}_accuracy_{config}_{self.args.cycle}.source_freeze.json"
            if not path.is_file():
                guards.append({"config": config, "status": "MISSING"})
                continue
            value = json.loads(path.read_text())
            coordinator_path = self.session.root / "accuracy" / f"historical_{self.args.variant}_accuracy_{config}_{self.args.cycle}.json"
            coordinator = json.loads(coordinator_path.read_text()) if coordinator_path.is_file() else {}
            attempts = coordinator.get("attempts", [])
            expected_count = sum(point.config == config for point in selected)
            terminal_valid = (coordinator.get("status") == "PASS" and coordinator.get("complete") is True
                              and len(attempts) == expected_count and all(row.get("process_status") == row.get("accuracy_status") == "PASS" for row in attempts))
            for attempt in attempts:
                log = self.session.root / "logs" / (attempt["attempt"] + ".json")
                terminal = json.loads(log.read_text()) if log.is_file() else {}
                terminal_valid = terminal_valid and terminal.get("status") == "PASS" and terminal.get("returncode") == 0
            guards.append({"config": config, "path": str(path), "sha256": Evidence.sha256(path),
                           "status": value.get("status"), "identity_matches": value.get("identity") == self.frozen,
                           "coordinator_receipt": str(coordinator_path), "expected_processes": expected_count,
                           "all_expected_processes_terminal_pass": terminal_valid})
        valid = all(row.get("status") == "VERIFIED_UNCHANGED" and row.get("identity_matches")
                    and row.get("all_expected_processes_terminal_pass") for row in guards)
        reference_bindings = []
        for point in selected:
            current_path = self.session.root / "accuracy" / f"current_accuracy_{point.config}_{self.args.reference_cycle}_{point.batch}_{point.precision}" / "result.json"
            historical_path = self.session.root / "accuracy" / self.attempt_name(point, self.args.cycle) / "result.json"
            if not current_path.is_file() or not historical_path.is_file():
                reference_bindings.append({"point": point.key, "matches": False, "reason": "missing current/historical result"})
                continue
            current, historical = json.loads(current_path.read_text()), json.loads(historical_path.read_text())
            mismatches = [name for name in ("point", "input_hash", "tensor_hashes", "reference_hashes", "layouts")
                          if current.get(name) != historical.get(name)]
            reference_bindings.append({"point": point.key, "matches": not mismatches, "mismatches": mismatches,
                                       "current_path": str(current_path), "current_sha256": Evidence.sha256(current_path),
                                       "historical_path": str(historical_path), "historical_sha256": Evidence.sha256(historical_path)})
        references_valid = all(row["matches"] for row in reference_bindings)
        report.update(variant=self.args.variant, source_guards=guards, source_guards_verified=valid,
                      historical_auditor_sha256=Evidence.sha256(Path(__file__)),
                      reference_cycle=self.args.reference_cycle, reference_bindings=reference_bindings,
                      all_reference_bindings_match_current=references_valid,
                      reference_binding_points=len(selected),
                      all75_reference_bindings_match_current=references_valid and len(selected) == 75,
                      queued_safety_verified=False, synchronous_accuracy_only=True)
        if not valid:
            report["base_accuracy_complete"] = False
            report["failures"].append({"reason": "missing or changed parent source/binary/ABI guards"})
        if not references_valid:
            report["base_accuracy_complete"] = False
            report["failures"].append({"reason": "historical/current canonical tensor/reference bindings differ"})
        Evidence.write(self.args.output, report)
        return 0 if code == 0 and valid and references_valid else 1


class HistoricalCycle:
    """Bound config chunks using one already frozen producer epoch."""

    def __init__(self, args):
        self.args, self.session = args, Session(args.session)

    def run(self):
        args, session = self.args, self.session
        producer = args.producer_root.resolve()
        hashes = json.loads((producer / "SHA256.json").read_text())
        if any(Evidence.sha256(producer / name) != digest for name, digest in hashes.items()):
            raise RuntimeError("Frozen cycle producer changed")
        configs = args.configs.split(",")
        if len(set(configs)) != len(configs) or any(name not in CONFIGS for name in configs):
            raise ValueError("Use distinct canonical config names")
        receipt = session.root / "accuracy" / f"historical_{args.variant}_{args.cycle}_controller.json"
        if receipt.exists():
            raise FileExistsError("Use a fresh controller/cycle; do not overwrite attempts")
        state = {"status": "RUNNING", "cycle": args.cycle, "variant": args.variant, "configs": configs,
                 "producer": hashes, "producer_root": str(producer), "chunks": [], "formal_performance": False}
        Evidence.write(receipt, state)
        for config in configs:
            name = f"historical_{args.variant}_{config}_{args.cycle}_chunk"
            argv = [session.paths["FASTEQ_PY"], "-B", str(producer / "historical_matrix.py"), "chunk",
                    "--session", str(session.root), "--variant", args.variant, "--config", config,
                    "--gpu", str(args.gpu), "--attempt", args.cycle]
            terminal = session.runner().run(name, argv, 300, session.root, {"CUDA_VISIBLE_DEVICES": ""})
            state["chunks"].append({"config": config, "process_status": terminal["status"], "receipt": str(session.root / "logs" / (name + ".json"))})
            Evidence.write(receipt, state)
            if terminal["status"] != "PASS":
                state["status"] = "INCOMPLETE_DIAGNOSIS_REQUIRED"
                Evidence.write(receipt, state)
                return 1
        output = session.root / "accuracy" / f"historical_{args.variant}_{args.cycle}_audit.json"
        name = f"historical_{args.variant}_{args.cycle}_full_audit"
        terminal = session.runner().run(name, [session.paths["FASTEQ_PY"], "-B", str(producer / "historical_matrix.py"), "audit",
            "--session", str(session.root), "--variant", args.variant, "--cycle", args.cycle,
            "--reference-cycle", args.reference_cycle, "--output", str(output)], 90)
        state.update(status="PASS" if terminal["status"] == "PASS" else "AUDIT_FAILED", audit=str(output))
        Evidence.write(receipt, state)
        return 0 if terminal["status"] == "PASS" else 1


class HistoricalMatrix:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        sub = parser.add_subparsers(dest="action", required=True)
        chunk = sub.add_parser("chunk")
        audit = sub.add_parser("audit")
        cycle = sub.add_parser("cycle")
        for target in (chunk, audit, cycle):
            target.add_argument("--session", type=Path, required=True)
            target.add_argument("--variant", required=True, help="p0/p14 full75, or an exact source/precision contract full25")
        chunk.add_argument("--config", choices=tuple(CONFIGS), required=True)
        chunk.add_argument("--gpu", type=int, required=True)
        chunk.add_argument("--attempt", required=True, help="Cycle id shared across complete five config chunks")
        audit.add_argument("--cycle", required=True)
        audit.add_argument("--reference-cycle", default="cycle02", help="Current full cycle whose actual input and reference hashes must match")
        audit.add_argument("--output", type=Path, required=True)
        cycle.add_argument("--cycle", required=True)
        cycle.add_argument("--gpu", type=int, required=True)
        cycle.add_argument("--producer-root", type=Path, required=True, help="Immutable producer from a successful pilot chunk")
        cycle.add_argument("--configs", default=",".join(CONFIGS), help="Remaining distinct configs; full audit still requires all five configs in this cycle")
        cycle.add_argument("--reference-cycle", default="cycle02")
        args = parser.parse_args()
        if args.action == "chunk":
            return HistoricalChunk(args).run()
        return HistoricalAudit(args).run() if args.action == "audit" else HistoricalCycle(args).run()


if __name__ == "__main__":
    raise SystemExit(HistoricalMatrix.main())
