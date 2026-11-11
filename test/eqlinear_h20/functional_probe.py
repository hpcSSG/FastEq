"""Queued pointer updates and enqueue/destroy lifetime against synchronized controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from cases import Case, References, points
from common import Evidence
from current import CurrentAdapter
from performance import PerformanceWorker
from producer_closure import ProducerClosure
from protocol import Accuracy
from reference_store import ReferenceStore
from session import Session


class DerivedCase:
    def __init__(self, base, x3, grad3):
        self.base, self.x3, self.grad3 = base, x3, grad3

    def __getattr__(self, key):
        return getattr(self.base, key)


class FunctionalProbe:
    def __init__(self, args):
        self.args, self.session = args, Session(args.session)

    @staticmethod
    def producer():
        path = Path(__file__).parent / "producer.json"
        if not path.is_file():
            raise RuntimeError("Execute only a frozen functional producer")
        values = json.loads(path.read_text())
        if any(Evidence.sha256(path.parent / name) != digest for name, digest in values.items()):
            raise ValueError("Frozen functional producer changed")
        return {"path": str(path), "sha256": Evidence.sha256(path), "files": values}

    def run(self):
        args = self.args
        producer = self.producer()
        gpu, affinity = PerformanceWorker.initialize_device()
        import torch
        import numpy

        point = next(value for value in points() if value.key == args.point)
        case = Case(point, fixture_root=self.session.root / "fixtures")
        reference = ReferenceStore(self.session.root / "references").get(case)
        second = DerivedCase(case, case.x3 + .125, case.grad3 + .125)
        second_reference = References(second)
        output = args.output.resolve()
        if not output.is_relative_to(self.session.root):
            raise ValueError("All functional evidence must be on the work disk")
        output.mkdir(parents=True, exist_ok=False)
        if args.family == "current":
            adapter = CurrentAdapter(case, self.session)
        else:
            from historical import HistoricalAdapter
            adapter = HistoricalAdapter(case, self.session, args.family)
        result = {"status": "RUNNING", "classification": "FUNCTIONAL_DIAGNOSTIC_NOT_PERFORMANCE",
            "point": point.columns(), "family": args.family, "GPU_UUID": gpu, "affinity": affinity,
            "canonical_input_hash": case.input_hash, "canonical_reference_hashes": reference.hashes,
            "derived_inputs": {"x": case.tensor_hash(second.x3), "grad": case.tensor_hash(second.grad3)},
            "derived_reference_hashes": second_reference.hashes,
            "delay_cycles": args.delay_cycles, "sync_policy": "synchronize only setup/control and after two queued calls; no between-call synchronize",
            "producer_sha256": Evidence.sha256(Path(__file__)), "scenarios": []}
        result.update(producer=producer, numpy={"version": numpy.__version__, "origin": numpy.__file__,
                                               "source_sha256": Evidence.sha256(Path(numpy.__file__))})
        try:
            for operation in ("forward", "grad_x"):
                adapter.prepare(operation, "run_with_activation_prep")
            result["identity"] = adapter.identity()
            for operation in ("forward", "grad_x"):
                for stream_mode in ("default", "nondefault"):
                    for destroy in (False, True):
                        row = self.scenario(adapter, case, second, reference, second_reference, operation, stream_mode, destroy, output)
                        result["scenarios"].append(row)
                        Evidence.write(output / "result.json", result)
            if self.producer() != producer:
                raise ValueError("Functional producer changed during execution")
            result["status"] = "PASS" if all(row["passed"] for row in result["scenarios"]) else "FUNCTIONAL_FAIL"
        except BaseException as error:
            result.update(status="ERROR", error=str(error))
            raise
        finally:
            adapter.close()
            Evidence.write(output / "result.json", result)
        # Numerical/queued failures are complete diagnostic evidence, not launch failures.
        return 0

    def scenario(self, adapter, first, second, first_ref, second_ref, operation, stream_mode, destroy, directory):
        import torch

        handle = adapter.handle(operation)
        run = adapter.namespace.forward_run if operation == "forward" else adapter.namespace.dgrad_run
        release = adapter.namespace.forward_destroy if operation == "forward" else adapter.namespace.dgrad_destroy
        sources = [value.x3 if operation == "forward" else value.grad3 for value in (first, second)]
        shape = first.grad3 if operation == "forward" else first.x3
        controls, control_quality = [], []
        for source, reference in zip(sources, (first_ref, second_ref)):
            values = [torch.empty_like(shape) for _ in range(2)]
            for value in values:
                run(handle, source, value)
                torch.cuda.synchronize()
            control_quality.append(Accuracy.evaluate(values[0].reshape(first.point.batch, -1), reference.values["fp64"][operation],
                reference.torch_errors[operation], first.point.precision, torch.equal(*values), values[0].data_ptr() != values[1].data_ptr()))
            controls.append(values[0])
        outputs = [torch.full_like(shape, float("nan")) for _ in range(2)]
        torch.cuda.synchronize()
        stream = torch.cuda.default_stream() if stream_mode == "default" else torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(self.args.delay_cycles)
            for source, value in zip(sources, outputs):
                run(handle, source, value)
        destroy_ms = None
        if destroy:
            started = time.perf_counter_ns()
            release(adapter.handles.pop(operation))
            destroy_ms = (time.perf_counter_ns() - started) / 1e6
        torch.cuda.synchronize()
        quality = [Accuracy.evaluate(value.reshape(first.point.batch, -1), reference.values["fp64"][operation],
            reference.torch_errors[operation], first.point.precision, torch.equal(value, control), value.data_ptr() != control.data_ptr())
            for value, control, reference in zip(outputs, controls, (first_ref, second_ref))]
        name = f"{operation}_{stream_mode}_destroy{int(destroy)}"
        diagnostic = directory / (name + ".pt")
        torch.save({"queued": [value.cpu() for value in outputs], "synchronized_control": [value.cpu() for value in controls]}, diagnostic)
        return {"scenario": name, "operation": operation, "stream": stream_mode, "destroy_before_explicit_sync": destroy,
            "destroy_host_ms_auxiliary": destroy_ms, "control_quality": control_quality, "queued_quality": quality,
            "repeat_equal_definition": "queued output equals own synchronized control on identical actual input",
            "different_output_addresses": outputs[0].data_ptr() != outputs[1].data_ptr(),
            "input_hashes": [first.tensor_hash(value) for value in sources],
            "output_hashes": [first.tensor_hash(value) for value in outputs],
            "diagnostic": {"path": str(diagnostic), "sha256": Evidence.sha256(diagnostic)},
            "passed": all(value["passed"] for value in control_quality + quality)}

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--point", required=True)
        parser.add_argument("--family", required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--delay-cycles", type=int, default=5_000_000)
        parser.add_argument("--freeze", type=Path, help="CPU only: freeze the complete producer dependency closure and stop")
        args = parser.parse_args()
        if args.freeze is not None:
            args.freeze.mkdir(parents=True, exist_ok=False)
            sources = ProducerClosure(Path(__file__).parent, args.freeze).freeze(["functional_probe.py"])
            Evidence.write(args.freeze / "producer.json", sources)
            return 0
        return cls(args).run()


if __name__ == "__main__":
    raise SystemExit(FunctionalProbe.main())
