"""One standalone variant/point probe, sharing canonical fixtures and numerical gates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

from cases import Case, References, points
from common import Evidence
from protocol import Accuracy, TraceSummary
from session import Session
from standalone import StandaloneAdapter, StandaloneSource


class StandaloneProbe:
    def __init__(self, args, reference_factory=References, adapter_factory=None, case_factory=Case,
                 source_factory=None, boundary_factory=None):
        self.args, self.session = args, Session(args.session)
        self.reference_factory = reference_factory
        self.adapter_factory = adapter_factory
        self.case_factory = case_factory
        self.source_factory = source_factory
        self.boundary_factory = boundary_factory
        self.output = args.output.resolve()
        if not self.output.is_relative_to(self.session.root):
            raise ValueError("Probe evidence must stay inside configured session")
        self.output.mkdir(parents=True, exist_ok=False)
        self.result = {"status": "RUNNING", "precision": args.precision, "impl": args.impl, "bimpl": args.bimpl,
                       "rows": [], "performance_measured": False, "trace": None}
        self.producer_files = {str(Path(__file__)): Evidence.sha256(Path(__file__))}
        for name in ("standalone.py", "cases.py", "protocol.py", "common.py", "session.py"):
            path = Path(__file__).with_name(name)
            self.producer_files[str(path)] = Evidence.sha256(path)
        self.result["producer_files_start"] = self.producer_files
        producer_root = self.output / "producer_sources"
        producer_root.mkdir()
        for source in self.producer_files:
            (producer_root / Path(source).name).write_bytes(Path(source).read_bytes())
        self.adapter = None

    def flush(self):
        Evidence.write(self.output / "result.json", self.result)

    def observe_result(self, row, output, repeated):
        """Optional diagnostic sink; every adapter uses the same gate loop."""

    def make_source(self, case):
        return self.source_factory(case) if self.source_factory else StandaloneSource(self.session, self.args.precision)

    def boundaries(self, operation):
        result = (self.boundary_factory(operation) if self.boundary_factory else
                  self.args.boundaries.split(",") if operation == "forward" else ["run_with_activation_prep"])
        if not isinstance(result, (list, tuple)) or not result or len(result) != len(set(result)):
            raise ValueError("A probe needs a nonempty unique declared boundary list")
        return list(result)

    def run(self):
        import torch

        self.flush()
        try:
            point = next(point for point in points() if point.key == self.args.point + "/" + self.args.precision)
            self.result.update(point=point.columns(), GPU_UUID=str(torch.cuda.get_device_properties().uuid),
                               gpu_name=torch.cuda.get_device_name(), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                               torch=torch.__version__, torch_cuda=torch.version.cuda, cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI)
            case = self.case_factory(point, "cuda", self.session.root / "fixtures")
            reference = self.reference_factory(case)
            self.result.update(input_hash=case.input_hash, tensor_hashes=case.hashes, reference_hashes=reference.hashes,
                               layouts=case.layouts(), torch_errors=reference.torch_errors)
            source = self.make_source(case)
            self.adapter = (self.adapter_factory(case, source) if self.adapter_factory else
                            StandaloneAdapter(case, source.library, self.args.precision, self.args.impl, self.args.bimpl,
                                              source.manifest_path))
            for operation in self.args.operations.split(","):
                boundaries = self.boundaries(operation)
                for boundary in boundaries:
                    self.result["preparing"] = {"operation": operation, "boundary": boundary}
                    self.flush()
                    self.adapter.prepare(operation, boundary)
                    first = self.adapter.call(operation, boundary, self.adapter.new_output(operation))
                    torch.cuda.synchronize()
                    second = self.adapter.call(operation, boundary, self.adapter.new_output(operation))
                    torch.cuda.synchronize()
                    kinds = ("grad_x", "grad_w") if operation == "grad_x_w" else (operation,)
                    firsts, seconds = (first, second) if operation == "grad_x_w" else ((first,), (second,))
                    for kind, value, repeated in zip(kinds, firsts, seconds):
                        accuracy = Accuracy.evaluate(value, reference.values["fp64"][kind], reference.torch_errors[kind],
                                                     point.precision, torch.equal(value, repeated), value.data_ptr() != repeated.data_ptr())
                        row = {"operation": operation, "result": kind, "boundary": boundary, **accuracy,
                               "output_hash": case.tensor_hash(value)}
                        self.result["rows"].append(row)
                        self.observe_result(row, value, repeated)
                        print(json.dumps(row), flush=True)
                    self.result["identity"] = self.adapter.identity()
                    self.flush()
            if self.args.trace:
                operation = self.args.operations.split(",")[0] if self.boundary_factory else "forward"
                boundary = self.boundaries(operation)[0] if self.boundary_factory else "prepared_gemm"
                self.adapter.prepare(operation, boundary)
                self.adapter.call(operation, boundary)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as profile:
                    self.adapter.call(operation, boundary)
                    torch.cuda.synchronize()
                trace_path = self.output / ("prepared_forward.trace.json" if not self.boundary_factory else operation + "_" + boundary + ".trace.json")
                profile.export_chrome_trace(str(trace_path))
                kernels = TraceSummary.kernels(json.loads(trace_path.read_text()))
                self.result["trace"] = {"path": str(trace_path), "sha256": Evidence.sha256(trace_path),
                                        "kernels": [{"name": row.get("name"), "dur_us": row.get("dur")} for row in kernels],
                                        "calls": 1, "operation": operation, "boundary": boundary}
            self.result.pop("preparing", None)
            self.result["producer_files_end"] = {path: Evidence.sha256(Path(path)) for path in self.producer_files}
            drift = self.result["producer_files_end"] != self.producer_files
            self.result["producer_drift"] = drift
            passed = all(row["passed"] for row in self.result["rows"])
            self.result["status"] = "INVALID_PRODUCER_DRIFT" if drift else "PASS" if passed else "ACCURACY_FAIL"
            self.flush()
            return 0 if self.result["status"] == "PASS" else 1
        except Exception as error:
            self.result.update(status="PROBE_ERROR", error=str(error), traceback=traceback.format_exc())
            self.flush()
            raise
        finally:
            if self.adapter is not None:
                self.adapter.close()

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--precision", choices=tuple(StandaloneSource.CONFIG), required=True)
        parser.add_argument("--point", default="SMALL_P1/512")
        parser.add_argument("--impl", type=int, default=0)
        parser.add_argument("--bimpl", type=int, default=1)
        parser.add_argument("--operations", default="forward,grad_x")
        parser.add_argument("--boundaries", default="forward_allocating,run_with_activation_prep,prepared_gemm")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--trace", action="store_true")
        return cls(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(StandaloneProbe.main())
