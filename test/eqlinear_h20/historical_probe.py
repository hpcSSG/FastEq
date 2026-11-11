"""Synchronized historical public/run forward and dx probe on shared exact fixtures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import traceback

from cases import Case, References, points
from common import Evidence
from current_probe import CurrentProbe
from historical import HistoricalAdapter
from protocol import Accuracy, TraceSummary
from session import Session


class HistoricalProbe(CurrentProbe):
    def __init__(self, args, reference_factory=References, case_factory=Case):
        self.args, self.session = args, Session(args.session)
        self.reference_factory, self.case_factory = reference_factory, case_factory
        self.output = args.output.resolve()
        if not self.output.is_relative_to(self.session.root):
            raise ValueError("Historical probe output must stay inside configured session")
        self.output.mkdir(parents=True, exist_ok=False)
        self.result = {"status": "RUNNING", "variant": args.variant, "boundaries": [], "performance_measured": False,
                       "sync_protocol": "device synchronize after each call", "queued_safety_verified": False}
        self.adapter = None

    def flush(self):
        Evidence.write(self.output / "result.json", self.result)

    def observe_result(self, row, output, repeated):
        if row["passed"]:
            return
        import torch
        root = self.output / "failed_outputs"
        root.mkdir(exist_ok=True)
        row["failed_outputs"] = {}
        for name, value in (("first", output), ("repeat", repeated)):
            target = root / (row["operation"] + "_" + row["call_boundary"] + "_" + name + ".pt")
            torch.save(value.detach().cpu(), target)
            row["failed_outputs"][name] = {"path": str(target), "sha256": Evidence.sha256(target),
                                           "tensor_hash": self.adapter.case.tensor_hash(value)}

    def producer_files(self):
        files = ("historical_probe.py", "historical.py", "historical_plan.py", "historical_native.py", "registry.py", "current.py", "current_probe.py", "cases.py", "common.py", "protocol.py", "session.py")
        files += tuple(name for name in ("current_matrix.py", "historical_matrix.py") if (Path(__file__).parent / name).is_file())
        files += tuple(name for name in ("reference_store.py",) if getattr(self.args, "reference_store", False))
        return files

    def run(self) -> int:
        import torch

        self.flush()
        files = self.producer_files()
        producer = {name: Evidence.sha256(Path(__file__).parent / name) for name in files}
        snapshots = self.output / "producer"
        snapshots.mkdir()
        for name in files:
            shutil.copy2(Path(__file__).parent / name, snapshots / name)
        try:
            point = next(point for point in points() if point.key == self.args.point)
            if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") not in (None, "0"):
                raise ValueError("TF32 comparator override is forbidden before reference or historical work")
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.result.update(point=point.columns(), GPU_UUID=str(torch.cuda.get_device_properties().uuid),
                               gpu_name=torch.cuda.get_device_name(), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                               PYTHONPATH=os.environ.get("PYTHONPATH"),
                               torch=torch.__version__, torch_cuda=torch.version.cuda, cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                               producer=producer, explicit_sync_not_queued=True)
            case = self.case_factory(point, "cuda", self.session.root / "fixtures")
            reference = self.reference_factory(case)
            self.result.update(input_hash=case.input_hash, tensor_hashes=case.hashes, reference_hashes=reference.hashes,
                               layouts=case.layouts(), torch_errors=reference.torch_errors)
            if hasattr(reference, "store_receipt"):
                self.result["reference_store"] = reference.store_receipt
            self.adapter = HistoricalAdapter(case, self.session, self.args.variant)
            torch.cuda.synchronize()
            for operation in ("forward", "forward_autograd", "grad_x"):
                for boundary in ("public_warm", "run_with_activation_prep"):
                    if operation == "forward_autograd" and boundary != "public_warm":
                        continue
                    self.result["preparing"] = {"operation": operation, "boundary": boundary}
                    self.flush()
                    self.adapter.prepare(operation, boundary)
                    first = self.adapter.call(operation, boundary, fresh_output=True)
                    torch.cuda.synchronize()
                    second = self.adapter.call(operation, boundary, fresh_output=True)
                    torch.cuda.synchronize()
                    kind = "forward" if operation == "forward_autograd" else operation
                    accuracy = Accuracy.evaluate(first, reference.values["fp64"][kind], reference.torch_errors[kind],
                                                 point.precision, torch.equal(first, second), first.data_ptr() != second.data_ptr())
                    row = {"operation": operation, "call_boundary": boundary, **accuracy,
                           "output_hash": case.tensor_hash(first)}
                    row["frozen_p14"] = self.frozen(case, kind, boundary, row["rel_fro"])
                    if row["frozen_p14"]["status"] == "FAIL":
                        row["passed"] = False
                    self.observe_result(row, first, second)
                    self.result["boundaries"].append(row)
                    self.result["identity"] = self.adapter.identity()
                    self.flush()
                    print(json.dumps(row), flush=True)
            if self.args.trace:
                for _ in range(3):
                    self.adapter.call("forward", "public_warm")
                    torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                    self.adapter.call("forward", "public_warm")
                    torch.cuda.synchronize()
                path = self.output / "public_forward.trace.json"
                profile.export_chrome_trace(str(path))
                kernels = TraceSummary.kernels(json.loads(path.read_text()))
                self.result["trace"] = {"path": str(path), "sha256": Evidence.sha256(path),
                                        "kernels": [{"name": row["name"], "duration_us": row["dur"]} for row in kernels]}
            self.result.pop("preparing", None)
            self.result["producer_unchanged"] = all(Evidence.sha256(Path(__file__).parent / name) == digest for name, digest in producer.items())
            self.result["status"] = "INVALID_PRODUCER_DRIFT" if not self.result["producer_unchanged"] else (
                "PASS" if all(row["passed"] for row in self.result["boundaries"]) else "ACCURACY_FAIL")
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
    def main(cls) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--variant", required=True, help="p0, p14, or a prepared source/precision contract")
        parser.add_argument("--point", required=True, help="config/B/precision canonical point")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--trace", action="store_true")
        parser.add_argument("--reference-store", action="store_true", help="Use the verified canonical shared References cache")
        args = parser.parse_args()
        if args.reference_store:
            from reference_store import ReferenceStore
            return cls(args, reference_factory=ReferenceStore(args.session / "reference_store").get).run()
        return cls(args).run()


if __name__ == "__main__":
    raise SystemExit(HistoricalProbe.main())
