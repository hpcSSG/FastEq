"""Profile the exact BoundCall used by the common performance harness."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from cases import Case, points
from common import Evidence
from performance import CallValidation, PerformancePlan, PerformanceWorker
from protocol import TraceSummary
from reference_store import ReferenceStore


class ProfileHarness:
    def __init__(self, args):
        self.args = args
        self.worker = PerformanceWorker(args)
        self.session, self.plan = self.worker.session, self.worker.plan

    def run(self):
        args = self.args
        gpu, affinity = self.worker.initialize_device()
        import torch

        selected = [point for point in points() if point.key == args.point and
                    point.key in self.plan["selected_points"][args.family]]
        if len(selected) != 1 or args.call not in self.plan["expected_calls"][args.family]:
            raise ValueError("Point/call is outside frozen factory scope")
        point = selected[0]
        output = args.output.resolve()
        if not output.is_relative_to(self.session.root):
            raise ValueError("Profile output must be on the configured work disk")
        output.mkdir(parents=True, exist_ok=False)
        case, references, workload, cache_mode = self.worker.prepare_case(point)
        family = self.worker.factory_type(case, self.session, family=args.family)
        result = {"status": "RUNNING", "profiled": True, "mode": args.mode,
            "point": point.columns(), "family": args.family, "call": args.call,
            "GPU_UUID": gpu, "affinity": affinity, "input_hash": case.input_hash,
            "reference_hashes": references.hashes, "reference_receipt": references.store_receipt,
            "plan_hash": self.plan["plan_hash"], "protocol_hash": self.plan["protocol_hash"],
            "producer": self.plan["producer"], "profile_calls": args.calls,
            "cache_mode": cache_mode, "weight_state": "constant",
            "interpretation": "Instrumented evidence; never substitutes for unprofiled completed wall latency"}
        if workload is not None:
            result["workload"] = workload
        try:
            call = next(value for value in family.calls if value.key == args.call)
            quality, warmup = CallValidation.run(call, case, references, output)
            result.update(binding=call.binding, quality=quality, warmup=warmup, measurement_work=call.measurement_work,
                          description=call.description)
            Evidence.write(output / "result.json", result)
            torch.cuda.synchronize()
            if call.before_sample is not None:
                call.before_sample(0, args.calls, 0)
                result["sample_reset"] = {"group": 0, "calls": args.calls, "round": 0,
                                          "interpretation": "Same before-sample hook as unprofiled sampling; reset outside the capture"}
            if args.mode == "torch":
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True, profile_memory=True) as profiler:
                    for _ in range(args.calls):
                        with torch.profiler.record_function(call.key):
                            value = call.function()
                            torch.cuda.synchronize()
                        del value
                        if call.after_sample is not None:
                            call.after_sample()
                trace = output / "trace.json"
                profiler.export_chrome_trace(str(trace))
                events = json.loads(trace.read_text())
                kernels = TraceSummary.kernels(events)
                result["trace"] = {"path": str(trace), "sha256": Evidence.sha256(trace),
                    "kernel_filter": "cat=kernel AND ph=X", "kernels": kernels,
                    "kernel_count": len(kernels), "kernel_names": dict(Counter(value["name"] for value in kernels)),
                    "profiled_kernel_duration_us": sum(value["dur"] for value in kernels),
                    "host_spans": [{key: event.get(key) for key in ("name", "cat", "ts", "dur", "tid")}
                                   for event in events.get("traceEvents", [])
                                   if event.get("ph") == "X" and event.get("cat") in ("cpu_op", "user_annotation")],
                    "host_span_interpretation": "Inclusive overlapping spans, not additive component costs"}
            else:
                cudart = torch.cuda.cudart()
                if int(cudart.cudaProfilerStart()) != 0:
                    raise RuntimeError("cudaProfilerStart failed")
                try:
                    for _ in range(args.calls):
                        value = call.function()
                        torch.cuda.synchronize()
                        del value
                finally:
                    if int(cudart.cudaProfilerStop()) != 0:
                        raise RuntimeError("cudaProfilerStop failed")
                if call.after_sample is not None:
                    call.after_sample()
                result["ncu_range"] = {"calls": args.calls, "start": "after all plan/JIT/allocation/quality/warmup",
                                       "stop": "after completion of full target BoundCall pipeline"}
            PerformancePlan.load(args.plan.resolve())
            result["status"] = "PASS" if quality["passed"] else "RAW_DIAGNOSTIC_ACCURACY_FAIL"
        except BaseException as error:
            result.update(status="ERROR", error=str(error))
            raise
        finally:
            family.close()
            Evidence.write(output / "result.json", result)
        return 0

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--family", required=True)
        parser.add_argument("--point", required=True, help="config/batch/precision canonical key")
        parser.add_argument("--call", required=True, help="variant/operation/boundary from frozen plan")
        parser.add_argument("--mode", choices=("torch", "ncu"), required=True)
        parser.add_argument("--calls", type=int, choices=(1, 2, 3), default=1)
        parser.add_argument("--output", type=Path, required=True)
        return cls(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(ProfileHarness.main())
