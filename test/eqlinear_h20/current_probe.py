"""Current fresh-build accuracy, backend and optional host/NCU representative probe."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

from cases import Case, References, points
from common import Evidence
from current import CurrentAdapter
from protocol import Accuracy, TraceSummary
from session import Session


class CurrentProbe:
    def __init__(self, args):
        self.args, self.session = args, Session(args.session)

    def run(self):
        import torch

        args, session = self.args, self.session
        output = args.output.resolve()
        if not output.is_relative_to(session.root):
            raise ValueError("Output must stay inside configured session")
        output.mkdir(parents=True, exist_ok=False)
        producer_files = [Path(__file__).parent / name for name in
                          ("current_probe.py", "current.py", "cases.py", "common.py", "protocol.py", "session.py")]
        producer = {p.name: Evidence.sha256(p) for p in producer_files}
        (output / "producer").mkdir()
        for path in producer_files:
            shutil.copy2(path, output / "producer" / path.name)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result = {"status": "RUNNING", "GPU_UUID": str(torch.cuda.get_device_properties().uuid),
                  "device_properties": repr(torch.cuda.get_device_properties()),
                  "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "producer": producer,
                  "boundaries": [], "performance_measured": False}
        Evidence.write(output / "result.json", result)
        for point in points():
            if point.key != args.point:
                continue
            case = Case(point, fixture_root=session.root / "fixtures")
            reference = References(case)
            result.update(point=point.columns(), input_hash=case.input_hash, tensor_hashes=case.hashes,
                          reference_hashes=reference.hashes, layouts=case.layouts())
            adapter = CurrentAdapter(case, session)
            try:
                for operation in ("forward", "forward_autograd", "grad_x"):
                    for boundary in ("public_warm", "run_with_activation_prep"):
                        if operation == "forward_autograd" and boundary != "public_warm":
                            continue
                        adapter.prepare(operation, boundary)
                        first = adapter.call(operation, boundary, fresh_output=True)
                        torch.cuda.synchronize()
                        second = adapter.call(operation, boundary, fresh_output=True)
                        torch.cuda.synchronize()
                        kind = "forward" if operation == "forward_autograd" else operation
                        row = Accuracy.evaluate(first, reference.values["fp64"][kind], reference.torch_errors[kind],
                                                point.precision, torch.equal(first, second), first.data_ptr() != second.data_ptr())
                        row.update(operation=operation, call_boundary=boundary, output_hash=case.tensor_hash(first))
                        row["frozen_p14"] = self.frozen(case, kind, boundary, row["rel_fro"])
                        if row["frozen_p14"]["status"] == "FAIL":
                            row["passed"] = False
                        result["boundaries"].append(row)
                        result["identity"] = adapter.identity()
                        Evidence.write(output / "result.json", result)
                        print(json.dumps(row), flush=True)
                if args.trace:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                        adapter.call("forward", "public_warm")
                        torch.cuda.synchronize()
                    trace_path = output / "public_forward.trace.json"
                    profile.export_chrome_trace(str(trace_path))
                    kernels = TraceSummary.kernels(json.loads(trace_path.read_text()))
                    result["trace"] = {"path": str(trace_path), "sha256": Evidence.sha256(trace_path),
                                       "kernels": [{"name": row["name"], "duration_us": row["dur"]} for row in kernels]}
                if args.ncu:
                    # Pre-existing plan/input/reference/allocation and warmup outside NCU window.
                    for _ in range(3):
                        adapter.call("forward", "run_with_activation_prep")
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    adapter.call("forward", "run_with_activation_prep")
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                result["identity"] = adapter.identity()
            finally:
                adapter.close()
            result["status"] = "PASS" if all(row["passed"] for row in result["boundaries"]) else "ACCURACY_FAIL"
            current = {p.name: Evidence.sha256(p) for p in Path(__file__).parent.glob("*.py") if p.name in producer}
            result["producer_changed_during_run"] = current != producer
            if current != producer:
                result["status"] = "INVALID_PRODUCER_DRIFT"
            Evidence.write(output / "result.json", result)
            return 0 if result["status"] == "PASS" else 1
        raise ValueError("Point must match canonical config/B/precision")

    def frozen(self, case, operation, boundary, error):
        if case.point.precision != "tf32":
            return {"status": "NOT_APPLICABLE"}
        path = Path(self.session.paths["HANDOFF_ROOT"]) / "h100/fasteq_fix/frozen_p14_accuracy/frozen_accuracy.csv"
        name = ("public_" if boundary == "public_warm" else "run_") + ("forward" if operation == "forward" else "grad_x")
        with path.open() as stream:
            row = next(row for row in csv.DictReader(stream) if row["config"] == case.point.config and
                       int(row["batch"]) == case.point.batch and row["precision"] == "tf32" and row["operation"] == name)
        mismatches = {key: {"historical": row[key], "current": value} for key, value in case.hashes.items() if row[key] != value}
        if mismatches:
            return {"status": "UNVERIFIED_INPUT_HASH_MISMATCH", "mismatches": mismatches, "source_sha256": Evidence.sha256(path)}
        limit = 1.1 * float(row["rel_fro"])
        return {"status": "PASS" if error is not None and error <= limit else "FAIL", "threshold": limit,
                "historical_error": float(row["rel_fro"]), "source_sha256": Evidence.sha256(path)}

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--point", required=True, help="config/B/precision, e.g. SMALL_P1/512/tf32")
        parser.add_argument("--trace", action="store_true")
        parser.add_argument("--ncu", action="store_true")
        return cls(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(CurrentProbe.main())
