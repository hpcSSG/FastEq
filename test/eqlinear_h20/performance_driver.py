"""Bounded config chunks; finish a whole75-point scan before the next round."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from cases import BATCHES, CONFIGS, PRECISIONS
from common import Evidence
from performance import PerformancePlan
from session import Session


@dataclass(frozen=True)
class MeasurementChunk:
    config: str
    family: str
    round: int | None
    batch: int | None = None
    precision: str | None = None

    def points(self, plan):
        return {key for key in plan["selected_points"][self.family]
                if key.split("/")[0] == self.config
                and (self.batch is None or int(key.split("/")[1]) == self.batch)
                and (self.precision is None or key.split("/")[2] == self.precision)}

    def suffix(self):
        return ("_" + self.precision if self.precision else "") + ("_b" + str(self.batch) if self.batch else "")


class PerformanceDriver:
    def __init__(self, args):
        self.args = args
        self.plan = PerformancePlan.load(args.plan.resolve())
        self.root = Path(self.plan["directory"])
        self.session = Session(Path(self.plan["session"]))

    @staticmethod
    def order(round_index, config_index, families):
        if len(families) > 3:
            offset = config_index % len(families)
            base = families[offset:] + families[:offset]
            if round_index == 1:
                return list(reversed(base))
            if round_index == 2:
                middle = len(base) // 2
                return base[middle:] + base[:middle]
            return base
        offset = (round_index + config_index) % len(families)
        return families[offset:] + families[:offset]

    @staticmethod
    def chunks(plan, config, family, round_index, per_batch=False, split_families=(), split_precisions=()):
        full = MeasurementChunk(config, family, round_index)
        keys = full.points(plan)
        selected_precisions = {key.split("/")[2] for key in keys}
        separate = (plan.get("factory_precision_chunks", False) or per_batch or family in split_families
                    or bool(selected_precisions & set(split_precisions)))
        chunks = []
        for precision in (PRECISIONS if separate else (None,)):
            split = per_batch or family in split_families or precision in split_precisions
            for batch in (BATCHES if split else (None,)):
                task = MeasurementChunk(config, family, round_index, batch, precision)
                if task.points(plan):
                    chunks.append(task)
        seen = [key for task in chunks for key in task.points(plan)]
        if set(seen) != keys or len(seen) != len(keys):
            raise ValueError("Point chunks overlap or omit declared scope")
        return chunks

    def run(self):
        args = self.args
        families = args.families.split(",")
        if set(families) - set(self.plan["include_families"]) or len(set(families)) != len(families):
            raise ValueError("Unknown/duplicate family")
        if args.phase == "pilot":
            tasks, covered = [], set()
            for config in CONFIGS:
                for family in families:
                    for task in self.chunks(self.plan, config, family, None, args.per_batch, args.split_family, args.split_precision):
                        selected = task.points(self.plan)
                        if selected - covered:
                            tasks.append(task)
                            covered.update(selected)
        else:
            if args.round > 0:
                previous = self.root / "phases" / f"cycle{args.cycle}_round{args.round-1}.json"
                if not previous.exists() or json.loads(previous.read_text()).get("status") != "COMPLETE":
                    raise ValueError("Previous whole-matrix round is incomplete; do not mix point-local rounds")
            # Group sizes for all75 points must be locked before the first formal sample.
            pilot = self.root / "phases/pilot.json"
            if not pilot.exists() or json.loads(pilot.read_text()).get("status") != "COMPLETE":
                raise ValueError("Full75 pilot is not complete")
            tasks = [task for index, config in enumerate(CONFIGS) for family in self.order(args.round, index, families)
                     for task in self.chunks(self.plan, config, family, args.round, args.per_batch, args.split_family, args.split_precision)]
        label = "pilot" if args.phase == "pilot" else f"cycle{args.cycle}_round{args.round}"
        path = self.root / "phases" / (label + ".json")
        if path.exists():
            raise FileExistsError("Use explicit recovery audit for an existing phase, not blind restart")
        result = {"status": "RUNNING", "phase": label, "families": families, "tasks": [],
                  "driver_sha256": Evidence.sha256(Path(__file__)), "plan_hash": self.plan["plan_hash"],
                  "planned_tasks": [asdict(task) | {"point_keys": sorted(task.points(self.plan))} for task in tasks],
                  "task_schema": 2, "start_utc": datetime.now(timezone.utc).isoformat(),
                  "child_deadline_seconds": args.chunk_deadline, "full_matrix_round_scope": True}
        result["family_order_rule"] = ("Latin rotation for up to3 families; for larger cohorts reverse the complete first-round order in round1 and rotate half in round2; rotate config start")
        Evidence.write(path, result)
        try:
            for task in tasks:
                config, family, round_index = task.config, task.family, task.round
                name = f"perf_{self.root.name}_{label}_{family}_{config}" + task.suffix()
                gpu = self.session.idle_gpu(name, args.gpu)
                argv = [self.session.paths["FASTEQ_PY"], "-B", str(self.root / "producer/performance.py"), "worker",
                        "--plan", str(args.plan.resolve()), "--mode", "pilot" if args.phase == "pilot" else "measure",
                        "--family", family, "--config", config, "--cycle", str(args.cycle), "--attempt", name]
                if round_index is not None:
                    argv += ["--round", str(round_index)]
                if task.batch is not None:
                    argv += ["--batch", str(task.batch)]
                if task.precision is not None:
                    argv += ["--precision", task.precision]
                terminal = self.session.runner().run(name, argv, args.chunk_deadline,
                    Path(self.session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": gpu})
                chunk_path = self.root / "chunks" / (name + ".json")
                chunk = json.loads(chunk_path.read_text()) if chunk_path.exists() else {}
                expected = task.points(self.plan)
                observed = [value["point"] for value in chunk.get("points", [])]
                complete = terminal["status"] == "PASS" and chunk.get("status") == "PASS" and set(observed) == expected and len(observed) == len(expected)
                result["tasks"].append({"config": config, "family": family, "round": round_index,
                    "batch": task.batch, "precision": task.precision, "point_keys": sorted(expected),
                    "terminal": terminal["status"], "complete_scope": complete, "point_count": len(observed),
                    "chunk": str(chunk_path), "GPU_UUID": gpu})
                Evidence.write(path, result)
                if not complete:
                    result["status"] = "INCOMPLETE_DIAGNOSIS_REQUIRED"
                    Evidence.write(path, result)
                    return 1
        except BaseException as error:
            result.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "ERROR", error=str(error))
            Evidence.write(path, result)
            raise
        result.update(status="COMPLETE", end_utc=datetime.now(timezone.utc).isoformat())
        Evidence.write(path, result)
        return 0

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--phase", choices=("pilot", "round"), required=True)
        parser.add_argument("--round", type=int, choices=(0, 1, 2), default=0)
        parser.add_argument("--cycle", type=int, default=1)
        parser.add_argument("--families", default="current,p0,p14")
        parser.add_argument("--gpu", type=int, required=True)
        parser.add_argument("--chunk-deadline", type=float, default=300.)
        parser.add_argument("--per-batch", "--per-point", dest="per_batch", action="store_true",
                            help="Use a separate worker for each batch and precision")
        parser.add_argument("--split-family", action="append", default=[], help="Split only named heavy families into per-point workers")
        parser.add_argument("--split-precision", choices=PRECISIONS, action="append", default=[],
                            help="Split this precision into per-point workers")
        return cls(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(PerformanceDriver.main())
