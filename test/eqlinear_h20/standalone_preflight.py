"""Read back a frozen standalone plan and its exact public-driver coverage on CPU."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import shutil
import sys


class StandaloneFormalPreflight:
    @classmethod
    def run(cls, args):
        sys.path.insert(0, str(args.plan.resolve().parent / "producer"))
        from cases import CONFIGS, points
        from common import Evidence
        from performance import PerformancePlan
        from performance_driver import PerformanceDriver
        from session import Session

        plan = PerformancePlan.load(args.plan.resolve())
        session = Session(Path(plan["session"]))
        if args.output.exists() or not args.output.resolve().is_relative_to(session.root):
            raise ValueError("Preflight evidence needs a fresh session output")
        module, name = plan["factory"].rsplit(".", 1)
        factory = getattr(importlib.import_module(module), name)
        factory.verify_scope(plan["factory_scope"], session)
        families = plan["include_families"]
        if set(args.split_family) - set(families) or len(args.split_family) != len(set(args.split_family)):
            raise ValueError("Split-family controls must select unique declared families")
        expected = {(family, point) for family in families for point in plan["selected_points"][family]}
        for family in families:
            selected = [point.key for point in points() if factory.accept_point(family, point)]
            if selected != plan["selected_points"][family] or factory.expected_calls(family) != plan["expected_calls"][family]:
                raise ValueError("Frozen factory disagrees with declared points/calls")
        rounds = []
        for round_index in range(3):
            tasks = [task for index, config in enumerate(CONFIGS)
                     for family in PerformanceDriver.order(round_index, index, families)
                     for task in PerformanceDriver.chunks(plan, config, family, round_index,
                         split_families=args.split_family)]
            observed = [(task.family, key) for task in tasks for key in task.points(plan)]
            if set(observed) != expected or len(observed) != len(expected):
                raise ValueError("The formal round omits or duplicates a family/point")
            if any(task.family in args.split_family and len(task.points(plan)) != 1 for task in tasks):
                raise ValueError("A designated heavy family was not split into single-point leaves")
            rounds.append({"round": round_index, "tasks": len(tasks), "family_points": len(observed),
                "expected_calls": sum(len(plan["expected_calls"][family]) for family, _ in observed)})
        covered, pilot = set(), []
        for config in CONFIGS:
            for family in families:
                for task in PerformanceDriver.chunks(plan, config, family, None, split_families=args.split_family):
                    selected = task.points(plan)
                    if selected - covered:
                        pilot.append(task)
                        covered.update(selected)
        if covered != {point for _, point in expected}:
            raise ValueError("The pilot omits a formal point")
        producer = args.output.with_suffix(".producer.py")
        if producer.exists():
            raise FileExistsError(producer)
        producer.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), producer)
        Evidence.write(args.output, {"classification": "CPU_FROZEN_DRIVER_COVERAGE_NOT_GPU_PERFORMANCE", "passed": True,
            "plan": {"path": str(args.plan.resolve()), "sha256": Evidence.sha256(args.plan)},
            "factory_scope_hash": plan["factory_scope"]["scope"]["scope_hash"],
            "families": len(families), "split_families": args.split_family,
            "pilot_tasks": len(pilot), "pilot_points": len(covered), "rounds": rounds,
            "frozen_producer_files": len(plan["producer"]), "GPU_work_started": False,
            "source": {"path": str(producer), "sha256": Evidence.sha256(producer)}})
        return 0


class PreflightCLI:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--split-family", action="append", default=[])
        parser.add_argument("--output", type=Path, required=True)
        return StandaloneFormalPreflight.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(PreflightCLI.main())
