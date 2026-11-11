"""Explicit same-math-epoch recovery; original interrupted evidence stays in place.

Only the output/log names of a user-interrupted batch are remapped. All GPU case,
reference, adapter, numerical gate and loop code are imported from the original
frozen producer, which is verified before and after every child process.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys


class Recovery:
    def __init__(self, plan_path, ledger_path):
        self.path, self.ledger_path = Path(plan_path).resolve(), Path(ledger_path).resolve()
        original = self.path.parent / "producer"
        sys.path.insert(0, str(original))
        from common import Evidence
        from historical_batch import HistoryPlan, HistoryAudit, HistoryBatch
        self.Evidence, self.Audit, self.Batch = Evidence, HistoryAudit, HistoryBatch
        self.base = HistoryPlan(self.path)
        self.root, self.session = self.base.root, self.base.session
        self.ledger = json.loads(self.ledger_path.read_text())
        self.ledger_hash = Evidence.sha256(self.ledger_path)
        self.value = copy.deepcopy(self.base.value)
        self.job_map = self.ledger["jobs"]
        self.group_map = self.ledger["batches"]
        for job in self.value["jobs"]:
            if job["job_id"] in self.job_map:
                job["attempt"] = self.job_map[job["job_id"]]["recovery_attempt"]
        for group in self.value["batch_groups"]:
            if group["batch_id"] in self.group_map:
                group["batch_id"] = self.group_map[group["batch_id"]]["recovery_batch_id"]
        self.verify()

    @classmethod
    def prepare(cls, plan_path, output):
        plan_path = plan_path.resolve()
        sys.path.insert(0, str(plan_path.parent / "producer"))
        from common import Evidence
        from historical_batch import HistoryPlan
        plan = HistoryPlan(plan_path)
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        executor = output / "historical_recovery.py"
        shutil.copy2(Path(__file__), executor)
        jobs, batches, preserved = {}, {}, {}
        for group in plan.value["batch_groups"]:
            terminal = plan.session.root / "logs" / (group["batch_id"] + ".json")
            if not terminal.exists() or json.loads(terminal.read_text())["status"] == "PASS":
                continue
            process = json.loads(terminal.read_text())
            if process["status"] != "INTERRUPTED":
                raise ValueError("Only explicit user interruption is recoverable; other process errors require diagnosis")
            batches[group["batch_id"]] = {"recovery_batch_id": group["batch_id"] + "__user_resume01",
                                        "original_terminal": str(terminal), "original_status": process["status"]}
            for job_id in group["job_ids"]:
                job = plan.job(job_id)
                raw = plan.session.root / "accuracy" / job["attempt"]
                if (raw / "result.json").exists():
                    raise ValueError("Interrupted group already has result rows; assess them explicitly before recovery")
                jobs[job_id] = {"original_attempt": job["attempt"], "recovery_attempt": job["attempt"] + "__user_resume01",
                                "old_result_existed": False}
                if raw.exists():
                    preserved.update({str(path): Evidence.sha256(path) for path in raw.rglob("*") if path.is_file()})
            batch_dir = plan.root / "batches" / group["batch_id"]
            preserved.update({str(path): Evidence.sha256(path) for path in batch_dir.rglob("*") if path.is_file()})
            for suffix in ("json", "stdout", "stderr"):
                path = plan.session.root / "logs" / (group["batch_id"] + "." + suffix)
                if path.exists():
                    preserved[str(path)] = Evidence.sha256(path)
        if len(batches) != 1 or len(jobs) != 1:
            raise ValueError("Expected exactly the independently inspected single interrupted batch/job")
        value = {"classification": "EXPLICIT_USER_RESUME_SAME_FROZEN_MATH_EPOCH", "utc": datetime.now(timezone.utc).isoformat(),
            "original_plan": {"path": str(plan_path), "sha256": Evidence.sha256(plan_path), "plan_hash": plan.value["plan_hash"]},
            "original_math_producer": plan.value["producer"], "executor": {"path": str(executor), "sha256": Evidence.sha256(executor)},
            "jobs": jobs, "batches": batches, "preserved_interrupted_files": preserved,
            "scope_fixture_reference_source_DSO_precision_controls_changed": False, "old_raw_deleted_or_overwritten": False,
            "new_output_names_only": True, "original_cycle_retained": plan.value["cycle"]}
        Evidence.write(output / "ledger.json", value)
        return output / "ledger.json"

    def job(self, job_id):
        return next(job for job in self.value["jobs"] if job["job_id"] == job_id)

    def verify(self):
        self.base.verify()
        E = self.Evidence
        if E.sha256(self.ledger_path) != self.ledger_hash or E.sha256(Path(__file__)) != self.ledger["executor"]["sha256"]:
            raise ValueError("Recovery ledger/executor drift")
        if E.sha256(self.path) != self.ledger["original_plan"]["sha256"] or self.base.value["producer"] != self.ledger["original_math_producer"]:
            raise ValueError("Recovery changed original scope/math epoch")
        expected = copy.deepcopy(self.base.value)
        for job in expected["jobs"]:
            if job["job_id"] in self.job_map:
                mapping = self.job_map[job["job_id"]]
                if job["attempt"] != mapping["original_attempt"]:
                    raise ValueError("Wrong original recovery job")
                job["attempt"] = mapping["recovery_attempt"]
        for group in expected["batch_groups"]:
            if group["batch_id"] in self.group_map:
                group["batch_id"] = self.group_map[group["batch_id"]]["recovery_batch_id"]
        if expected != self.value:
            raise ValueError("Recovery changed more than declared output/log names")
        for path, digest in self.ledger["preserved_interrupted_files"].items():
            if E.sha256(Path(path)) != digest:
                raise ValueError("Interrupted original evidence changed")

    def verify_producer(self, root):
        self.base.verify_producer(root)

    def verify_sources(self, session):
        self.base.verify_sources(session)

    def uuid(self, value):
        return self.base.uuid(value)

    def batch(self, batch_id):
        recovery = self
        class PlanFactory:
            def __new__(cls, path):
                if Path(path).resolve() != recovery.path:
                    raise ValueError("Recovery plan substitution")
                return recovery
        class Batch(self.Batch):
            PLAN_CLASS = PlanFactory
            def create_worker(self, args):
                args.output = recovery.session.root / "accuracy" / recovery.job(args.job_id)["attempt"]
                return super().create_worker(args)
        code = Batch(argparse.Namespace(plan=self.path, batch_id=batch_id)).run()
        group = next(group for group in self.value["batch_groups"] if group["batch_id"] == batch_id)
        path = self.root / "batches" / batch_id / "recovery_executor.json"
        self.Evidence.write(path, {"ledger": str(self.ledger_path), "ledger_sha256": self.ledger_hash,
            "executor": self.ledger["executor"], "original_math_producer": self.value["producer"], "group": group})
        return code

    def run(self):
        audit = self.Audit(self)
        completed = []
        for group in self.value["batch_groups"]:
            jobs = [self.job(key) for key in group["job_ids"]]
            if all(audit.record(job)["verified"] for job in jobs):
                completed.append({"batch_id": group["batch_id"], "status": "EXISTING_VERIFIED_NO_RERUN"})
                continue
            if any((self.session.root / "accuracy" / job["attempt"]).exists() for job in jobs):
                raise RuntimeError("Unexpected partial/error output requires diagnosis; no blind restart")
            self.verify()
            gpu = self.session.idle_gpu(group["batch_id"], self.value["gpu"]["physical_index"])
            recovered = group["batch_id"] in {row["recovery_batch_id"] for row in self.group_map.values()}
            if recovered:
                argv = [self.session.paths["FASTEQ_PY"], "-B", str(Path(__file__)), "batch", "--plan", str(self.path), "--ledger", str(self.ledger_path), "--batch-id", group["batch_id"]]
            else:
                argv = [self.session.paths["FASTEQ_PY"], "-B", str(self.root / "producer/historical_batch.py"), "batch-worker", "--plan", str(self.path), "--batch-id", group["batch_id"]]
            terminal = self.session.runner().run(group["batch_id"], argv, group["deadline_seconds"],
                Path(self.session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": gpu})
            self.verify()
            verified = all(audit.record(job)["verified"] for job in jobs)
            completed.append({"batch_id": group["batch_id"], "status": terminal["status"], "verified": verified})
            self.Evidence.write(self.ledger_path.parent / "progress.json", {"completed_batches": completed, "original_plan_hash": self.value["plan_hash"]})
            if terminal["status"] not in ("PASS", "FAIL") or not verified:
                raise RuntimeError("Historical process/source/audit error; stop and diagnose")
        return 0

    def audit(self, output):
        raw = output.with_name(output.stem + "_recovery_native_report.json")
        if output.exists() or raw.exists():
            raise FileExistsError(output)
        report = self.Audit(self).run(raw)
        canonical = {}
        for original in self.base.value["jobs"]:
            key = self.job(original["job_id"])["attempt"]
            canonical[original["attempt"]] = report["records"][key]
        report["records"] = canonical
        report["recovery"] = {"ledger": str(self.ledger_path), "ledger_sha256": self.ledger_hash, "executor": self.ledger["executor"],
            "same_original_frozen_math_producer": True, "interrupted_old_raw_preserved": True, "canonical_job_keys_retained": True,
            "underlying_independent_audit": {"path": str(raw), "sha256": self.Evidence.sha256(raw)}}
        self.Evidence.write(output, report)
        return 0 if report["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "run", "batch", "audit"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-id")
    args = parser.parse_args()
    if args.action == "prepare":
        print(Recovery.prepare(args.plan, args.output))
        return 0
    recovery = Recovery(args.plan, args.ledger)
    if args.action == "preflight":
        audit = recovery.Audit(recovery)
        valid = sum(audit.record(job)["verified"] for job in recovery.value["jobs"])
        print(json.dumps({"verified_points": valid, "pending_points": len(recovery.value["jobs"]) - valid,
                          "remapped_interrupted_jobs": len(recovery.job_map), "original_math_epoch_unchanged": True}))
        return 0
    if args.action == "batch":
        return recovery.batch(args.batch_id)
    if args.action == "audit":
        return recovery.audit(args.output)
    return recovery.run()


if __name__ == "__main__":
    raise SystemExit(main())
