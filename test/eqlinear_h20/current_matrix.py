"""Run a bounded current-FastEq accuracy config chunk, one canonical point per process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from cases import CONFIGS, points
from common import Evidence
from session import Session


class AccuracyChunk:
    FILES = ("current_matrix.py", "current_probe.py", "current.py", "cases.py", "common.py", "protocol.py", "session.py")

    def __init__(self, args):
        self.args, self.session = args, Session(args.session)

    def family_name(self):
        return "current"

    def probe_argv(self, producer, point, output):
        return [self.session.paths["FASTEQ_PY"], "-B", str(producer / "current_probe.py"), "--session", str(self.session.root),
                "--output", str(output), "--point", point.key]

    def accept_terminal(self, result, terminal):
        return terminal["status"] == "PASS"

    def selected_points(self):
        return [point for point in points() if point.config == self.args.config]

    def run(self):
        args, session = self.args, self.session
        selected_points = self.selected_points()
        if not selected_points:
            raise ValueError("An accuracy chunk requires at least one canonical point")
        name = f"{self.family_name()}_accuracy_{args.config}_{args.attempt}"
        producer = session.root / "manifests/producers" / name
        producer.mkdir(parents=True, exist_ok=False)
        for filename in self.FILES:
            shutil.copy2(Path(__file__).parent / filename, producer / filename)
        hashes = {p.name: Evidence.sha256(p) for p in producer.iterdir()}
        Evidence.write(producer / "SHA256.json", hashes)
        status = {"classification": "UNIFIED_ACCURACY_NOT_FORMAL_PERFORMANCE", "status": "RUNNING",
                  "config": args.config, "producer": hashes, "attempts": [], "complete": False}
        receipt = session.root / "accuracy" / (name + ".json")
        Evidence.write(receipt, status)
        try:
            for point in selected_points:
                attempt = f"{name}_{point.batch}_{point.precision}"
                gpu = session.idle_gpu(attempt, args.gpu)
                output = session.root / "accuracy" / attempt
                argv = self.probe_argv(producer, point, output)
                deadline = 120 if point.precision == "fp64" else 60
                terminal = session.runner().run(attempt, argv, deadline, Path(session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": gpu})
                result_path = output / "result.json"
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                status["attempts"].append({"point": point.key, "attempt": attempt, "process_status": terminal["status"],
                                           "accuracy_status": result.get("status", "MISSING"), "result": str(result_path), "GPU_UUID": gpu})
                Evidence.write(receipt, status)
                if not self.accept_terminal(result, terminal):
                    # A failing process is terminal. Keep evidence and stop this scan for diagnosis.
                    status["status"] = "INCOMPLETE_DIAGNOSIS_REQUIRED"
                    Evidence.write(receipt, status)
                    return 1
        except BaseException as error:
            status.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "INCOMPLETE_RESOURCE_OR_DRIVER_ERROR", error_type=type(error).__name__, error=str(error))
            Evidence.write(receipt, status)
            raise
        accuracy_failures = sum(row["accuracy_status"] == "ACCURACY_FAIL" for row in status["attempts"])
        status.update(status="COMPLETE_WITH_ACCURACY_FAILURES" if accuracy_failures else "PASS",
                      complete=len(status["attempts"]) == len(selected_points), accuracy_failures=accuracy_failures)
        Evidence.write(receipt, status)
        return 0 if status["complete"] else 1

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--config", choices=tuple(CONFIGS), required=True)
        parser.add_argument("--gpu", type=int, required=True)
        parser.add_argument("--attempt", required=True, help="Fresh chunk id; partial chunks remain incomplete")
        return cls(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(AccuracyChunk.main())
