"""Run exact frozen public-driver phases without changing their sampling protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys


class FrozenPerformanceSequence:
    @classmethod
    def run(cls, args):
        root = args.plan.resolve().parent
        sys.path.insert(0, str(root / "producer"))
        from cases import CONFIGS
        from common import Evidence
        from performance import PerformancePlan
        from performance_driver import PerformanceDriver
        from session import Session

        plan = PerformancePlan.load(args.plan.resolve())
        session = Session(Path(plan["session"]))
        phases = args.phases.split(",")
        if (re.fullmatch(r"[A-Za-z0-9_.-]+", args.label) is None or len(phases) != len(set(phases)) or
                set(phases) - {"pilot", "0", "1", "2"} or set(args.split_family) - set(plan["include_families"])):
            raise ValueError("Select unique existing phases/families and a fresh simple sequence label")
        output = root / "sequences" / (args.label + ".json")
        producer = output.with_suffix(".producer.py")
        if output.exists() or producer.exists():
            raise FileExistsError("An existing sequence requires explicit diagnosis, not blind restart")
        producer.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), producer)
        source_sha = Evidence.sha256(Path(__file__))
        result = {"classification": "FROZEN_PUBLIC_DRIVER_SEQUENCE_NOT_A_NEW_SAMPLER", "status": "RUNNING",
            "plan_hash": plan["plan_hash"], "cycle": args.cycle, "gpu_index": args.gpu,
            "families": plan["include_families"], "split_families": args.split_family,
            "child_deadline_seconds": 300, "requested_phases": phases, "completed": [],
            "source": {"path": str(producer), "sha256": source_sha}}
        Evidence.write(output, result)
        for phase in phases:
            tasks, covered = [], set()
            for index, config in enumerate(CONFIGS):
                families = (plan["include_families"] if phase == "pilot" else
                            PerformanceDriver.order(int(phase), index, plan["include_families"]))
                for family in families:
                    for task in PerformanceDriver.chunks(plan, config, family, None if phase == "pilot" else int(phase),
                                                        split_families=args.split_family):
                        if phase != "pilot" or task.points(plan) - covered:
                            tasks.append(task)
                            covered.update(task.points(plan))
            name = args.label + "_" + ("pilot" if phase == "pilot" else "round" + phase)
            command = [session.paths["FASTEQ_PY"], "-B", str(root / "producer/performance_driver.py"),
                "--plan", str(args.plan.resolve()), "--phase", "pilot" if phase == "pilot" else "round",
                "--cycle", str(args.cycle), "--families", ",".join(plan["include_families"]),
                "--gpu", str(args.gpu), "--chunk-deadline", "300"]
            if phase != "pilot":
                command += ["--round", phase]
            for family in args.split_family:
                command += ["--split-family", family]
            # This only bounds the controller. Every actual GPU leaf retains its300s limit.
            deadline = max(600, len(tasks) * 330)
            terminal = session.runner().run(name, command, deadline, Path(session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": ""})
            path = root / "phases" / ("pilot.json" if phase == "pilot" else f"cycle{args.cycle}_round{phase}.json")
            value = json.loads(path.read_text()) if path.exists() else {}
            valid = (terminal["status"] == "PASS" and value.get("status") == "COMPLETE" and
                     len(value.get("tasks", [])) == len(tasks) and Evidence.sha256(Path(__file__)) == source_sha)
            result["completed"].append({"phase": phase, "terminal": terminal["status"], "complete": valid,
                "tasks": len(tasks), "elapsed_seconds": terminal["elapsed_seconds"],
                "phase_ledger": str(path), "phase_ledger_sha256": Evidence.sha256(path) if path.exists() else None})
            if not valid:
                result["status"] = "INCOMPLETE_DIAGNOSIS_REQUIRED"
                Evidence.write(output, result)
                return 1
            Evidence.write(output, result)
        result["status"] = "COMPLETE"
        Evidence.write(output, result)
        return 0


class SequenceCLI:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--gpu", type=int, required=True)
        parser.add_argument("--phases", default="pilot,0,1,2")
        parser.add_argument("--cycle", type=int, choices=(1, 2), default=1)
        parser.add_argument("--split-family", action="append", default=[])
        parser.add_argument("--label", required=True)
        return FrozenPerformanceSequence.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(SequenceCLI.main())
