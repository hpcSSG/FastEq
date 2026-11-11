"""Run one version/point Luna JIT and accuracy probe with shared canonical inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from cases import Case, References, points
from common import Evidence
from luna import LunaAdapter
from protocol import Accuracy, Timer
from session import Session


class LunaProbe:
    def __init__(self, args, reference_factory=References, adapter_factory=LunaAdapter):
        self.args = args
        self.session = Session(args.session)
        self.output = args.output.resolve()
        if not self.output.is_relative_to(self.session.root):
            raise ValueError("Probe output must be inside its configured session")
        if self.output.exists():
            raise FileExistsError(self.output)
        self.output.mkdir(parents=True)
        self.result = {"status": "RUNNING", "variant": "luna_" + args.version,
                       "operation_boundaries": [], "gpu_performance_measured": False,
                       "note": "Probe setup/JIT elapsed seconds are not benchmark latency"}
        producer = Path(__file__).resolve().parent
        self.producer_files = {str(producer / name): Evidence.sha256(producer / name)
                               for name in ("cases.py", "common.py", "session.py", "protocol.py", "luna.py", "luna_probe.py")}
        snapshot = self.output / "producer"
        snapshot.mkdir()
        for filename in self.producer_files:
            shutil.copy2(filename, snapshot / Path(filename).name)
        self.result["producer_files"] = self.producer_files
        self.result["producer_snapshot"] = str(snapshot)
        self.adapter = None
        self.reference_factory = reference_factory
        self.adapter_factory = adapter_factory

    def observe_result(self, row, output):
        """Subclass hook for optional diagnostic tensor persistence, outside timing."""

    def observe_pair(self, operation, boundary, first, second):
        """Validate raw API properties outside measured function calls."""

    @staticmethod
    def result_kinds(operation):
        return ("grad_x", "grad_w") if operation == "grad_x_w" else ("forward" if operation == "forward_autograd" else operation,)

    def flush(self):
        Evidence.write(self.output / "result.json", self.result)

    @staticmethod
    def values(value):
        return value if isinstance(value, tuple) else (value,)

    def run(self):
        import torch

        self.flush()
        try:
            point = next(point for point in points()
                         if f"{point.config}/{point.batch}" == self.args.point
                         and point.precision == self.args.precision)
            self.result.update(point=point.columns(),
                               gpu_uuid=str(torch.cuda.get_device_properties().uuid),
                               gpu_name=torch.cuda.get_device_name(),
                               logical_device=torch.cuda.current_device(),
                               CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                               torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                               cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                               cpu_affinity=sorted(os.sched_getaffinity(0)),
                               environment={key: os.environ.get(key) for key in self.session.environment()})
            self.flush()
            case = Case(point, device="cuda", fixture_root=self.session.root / "fixtures")
            reference = self.reference_factory(case)
            self.result.update(input_hash=case.input_hash, tensor_hashes=case.hashes,
                               reference_hashes=reference.hashes,
                               torch_errors=reference.torch_errors, layouts=case.layouts())
            self.adapter = self.adapter_factory(case, self.args.source_root, self.args.version,
                                               self.session.root / "manifests/luna_jit" / self.args.version)
            self.flush()
            for operation in self.args.operations.split(","):
                for boundary in self.args.boundaries.split(","):
                    self.result["preparing"] = {"operation": operation, "boundary": boundary}
                    self.flush()
                    start = time.perf_counter()
                    self.adapter.prepare(operation, boundary)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    first_raw = self.adapter.call(operation, boundary)
                    first = self.adapter.common_output(operation, boundary, first_raw)
                    second_raw = self.adapter.call(operation, boundary)
                    second = self.adapter.common_output(operation, boundary, second_raw)
                    torch.cuda.synchronize()
                    self.observe_pair(operation, boundary, first_raw, second_raw)
                    kinds = self.result_kinds(operation)
                    for kind, value, repeated, raw, repeated_raw in zip(
                            kinds, self.values(first), self.values(second),
                            self.values(first_raw), self.values(second_raw)):
                        repeat_equal = bool(torch.equal(value, repeated))
                        different_buffer = raw.data_ptr() != repeated_raw.data_ptr()
                        metrics = Accuracy.evaluate(value, reference.values["fp64"][kind],
                                                    reference.torch_errors[kind], point.precision,
                                                    repeat_equal, different_buffer)
                        row = {"operation": operation, "result": kind, "boundary": boundary,
                               **metrics, "accuracy_pass": metrics["passed"],
                               "output_hash": case.tensor_hash(value),
                               "setup_or_JIT_seconds": elapsed,
                               "backend": self.adapter.config.last_execution_backend}
                        self.result["operation_boundaries"].append(row)
                        self.observe_result(row, value)
                        print(json.dumps(row), flush=True)
                    self.result["identity"] = self.adapter.identity()
                    self.flush()
            if self.args.trace:
                self.adapter.prepare("forward", "prepared_native_launch_with_alloc")
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as profile:
                    self.adapter.call("forward", "prepared_native_launch_with_alloc")
                    torch.cuda.synchronize()
                trace = self.output / "prepared_forward.trace.json"
                profile.export_chrome_trace(str(trace))
                self.result["trace"] = {"path": str(trace), "sha256": Evidence.sha256(trace),
                                        "boundary": "prepared_native_launch_with_alloc",
                                        "calls": 1, "classification": "raw unaggregated trace"}
            if self.args.ncu:
                self.adapter.prepare("forward", "prepared_native_launch_with_alloc")
                target = lambda: self.adapter.call("forward", "prepared_native_launch_with_alloc")
                self.result["ncu_warmup"] = Timer.warmup(target)
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
                try:
                    target()
                    torch.cuda.synchronize()
                finally:
                    torch.cuda.profiler.stop()
                self.result["ncu_range"] = {"boundary": "prepared_native_launch_with_alloc", "calls": 1,
                                             "compile_allocate_reference_accuracy_warmup_outside_range": True}
            self.result.pop("preparing", None)
            self.result["identity"] = self.adapter.identity()
            native = self.adapter.config.last_execution_backend in ("cuda_fused", "cuda_native")
            passed = all(row["accuracy_pass"] for row in self.result["operation_boundaries"])
            self.result["status"] = "PASS" if native and passed else "ACCURACY_FAIL" if native else "BACKEND_FAIL"
            self.result["producer_unchanged"] = all(Evidence.sha256(Path(filename)) == digest
                                                     for filename, digest in self.producer_files.items())
            if not self.result["producer_unchanged"]:
                self.result["status"] = "INVALID_PRODUCER_CHANGED_DURING_RUN"
            self.flush()
            return 0 if self.result["status"] == "PASS" else 1
        except BaseException as error:
            self.result.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "ERROR",
                               error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            if self.adapter is not None:
                self.result["identity"] = self.adapter.identity()
            self.flush()
            raise
        finally:
            if self.adapter is not None:
                self.adapter.close()

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--source-root", type=Path, required=True)
        parser.add_argument("--version", choices=("0.1.0", "0.2.0"), required=True)
        parser.add_argument("--point", required=True, help="e.g. SMALL_P1/512")
        parser.add_argument("--precision", choices=("fp32", "fp64"), required=True)
        parser.add_argument("--operations", default=",".join(LunaAdapter.OPERATIONS))
        parser.add_argument("--boundaries", default=",".join(LunaAdapter.BOUNDARIES))
        parser.add_argument("--output", type=Path, required=True)
        profiler = parser.add_mutually_exclusive_group()
        profiler.add_argument("--trace", action="store_true")
        profiler.add_argument("--ncu", action="store_true", help="Bracket one warmed prepared forward with cudaProfilerStart/Stop")
        return cls(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(LunaProbe.main())
