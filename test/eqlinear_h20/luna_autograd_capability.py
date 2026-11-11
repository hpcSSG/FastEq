"""Separate real-dy training and raw prepared-launch autograd capability evidence."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import traceback

from cases import Case, points
from common import Evidence
from luna_autograd import AutogradController, AutogradPlan, LunaAutogradAdapter
from luna_matrix import MatrixPlan, SourceTree
from protocol import Accuracy
from reference_store import ReferenceStore


class CapabilityProbe:
    def __init__(self, args):
        self.args, self.plan = args, AutogradPlan(args.plan)
        self.job = self.plan.job(args.job_id)
        self.output = args.output.resolve()
        if not self.output.is_relative_to(Path(self.plan.value["session"])):
            raise ValueError("Capability output must stay in its session")
        if self.output.exists():
            raise FileExistsError(self.output)
        self.output.mkdir(parents=True)
        self.plan.verify_producer(Path(__file__).parent)
        shutil.copytree(Path(__file__).parent, self.output / "producer")
        self.result = {"status": "RUNNING", "classification": "MINIMUM_CAPABILITY_NOT_FULL_ACCURACY_OR_TIMING",
                       "plan_hash": self.plan.value["plan_hash"], "job_id": self.job["job_id"],
                       "producer": self.plan.value["producer"], "diagnostic_reproduction": args.reproduction,
                       "gradient_diagnostic_rows": []}
        self.adapter = None

    def source_witness(self, launch, identity):
        module = launch.__module__.split(".")[0]
        libraries = [Path(path) for path in identity["jit_libraries"] if Path(path).stem == module]
        if len(libraries) != 1:
            raise ValueError("Prepared function did not resolve to one actual mapped DSO")
        library, witnesses = libraries[0], []
        manifest_root = Path(identity["jit_manifest_root"])
        for receipt in manifest_root.glob("*/loads/*.json"):
            load = json.loads(receipt.read_text())
            if load.get("path") != str(library) or load.get("sha256") != identity["jit_libraries"][str(library)]:
                continue
            request = receipt.parent.parent / "request.json"
            if Evidence.sha256(request) != load["request_sha256"]:
                raise ValueError("Actual DSO-to-source request link changed")
            value = json.loads(request.read_text())
            for label, suffix in (("cpp_sources", "cpp"), ("cuda_sources", "cu")):
                text = value["arguments"].get(label)
                if text:
                    source = request.parent / ("generated." + suffix)
                    if source.read_text() != text:
                        raise ValueError("Generated binding/source request bytes differ")
                    witnesses.append({"path": str(source), "sha256": Evidence.sha256(source),
                                      "request": str(request), "request_sha256": load["request_sha256"],
                                      "DSO_load_receipt": str(receipt), "DSO_load_receipt_sha256": Evidence.sha256(receipt)})
        for receipt in (manifest_root / "builds").glob("*/arguments.json"):
            value = json.loads(receipt.read_text())
            if Path(value["arguments"]["build_directory"]).resolve() != library.parent.resolve():
                continue
            for name, digest in value["source_hashes"].items():
                source = receipt.parent / name
                if source.suffix == ".cpp":
                    if Evidence.sha256(source) != digest:
                        raise ValueError("Recorded compiled binding source changed")
                    witnesses.append({"path": str(source), "sha256": digest})
        if not witnesses:
            raise ValueError("Prepared C++ binding source witness was not preserved")
        return {"module": launch.__module__, "function": launch.__name__, "library": str(library),
                "library_sha256": identity["jit_libraries"][str(library)], "compiled_binding_sources": witnesses}

    def run(self):
        import torch

        source = self.plan.value["sources"][self.job["version"]]
        before = SourceTree.manifest(source["root"])
        try:
            point = next(point for point in points() if point.columns() == self.job["point"])
            if point.config != "SMALL_P1" or point.batch != 512:
                raise ValueError("Capability scope is the declared smallest canonical point")
            uuid = ReferenceStore.device_uuid(torch.cuda.get_device_properties().uuid)
            if uuid != self.plan.value["gpu"]["uuid"]:
                raise ValueError("Capability CUDA device differs from the plan")
            case = Case(point, fixture_root=Path(self.plan.value["session"]) / "fixtures")
            references = ReferenceStore(Path(self.plan.value["reference_store"])).get(case)
            self.result.update(point=point.columns(), GPU_UUID=uuid, input_hash=case.input_hash,
                               tensor_hashes=case.hashes, layouts=case.layouts(),
                               reference_hashes=references.hashes, reference_store=references.store_receipt,
                               source_manifest_hash=source["source_hash"], environment={key: os.environ.get(key)
                                   for key in ("CUDA_VISIBLE_DEVICES", "LUNAEQ_CACHE_DIR", "TMPDIR", "XDG_CACHE_HOME")})
            self.adapter = LunaAutogradAdapter(case, source["root"], self.job["version"],
                Path(self.plan.value["session"]) / "manifests/luna_jit" / self.job["version"])
            boundaries = ("native_public",) if self.args.reproduction else self.adapter.PUBLIC_BOUNDARIES
            for boundary in boundaries:
                raw_gradients, gradients, graphs = [], [], []
                for repeat in range(2):
                    output = self.adapter.call("forward_autograd", boundary)
                    graph = self.adapter.graph_evidence(boundary, output)
                    if not self.adapter.valid_graph(graph):
                        raise ValueError("Public forward did not construct its declared autograd graph")
                    dy = self.adapter.dy_native if boundary == "native_public" else case.grad
                    if not torch.count_nonzero(dy).item():
                        raise ValueError("Training capability requires the real nonzero canonical dy")
                    gradient = torch.autograd.grad(output, self.adapter.autograd_x[boundary], dy)[0]
                    raw_gradients.append(gradient)
                    gradients.append(self.adapter.common_output("grad_x", boundary, gradient))
                    graphs.append(graph)
                torch.cuda.synchronize()
                metrics = Accuracy.evaluate(gradients[0], references.values["fp64"]["grad_x"],
                    references.torch_errors["grad_x"], point.precision, torch.equal(*gradients),
                    raw_gradients[0].data_ptr() != raw_gradients[1].data_ptr())
                self.result["gradient_diagnostic_rows"].append({"boundary": boundary, "result": "grad_x",
                    **metrics, "accuracy_pass": metrics["passed"], "graphs": graphs,
                    "output_hash": case.tensor_hash(gradients[0]), "dy_hash": case.tensor_hash(dy), "dy_nonzero": True})
            # Load the same complete dgrad binding context as the original c03 probe.
            self.adapter.prepare("grad_x", "prepared_native_launch_with_alloc")
            torch.cuda.synchronize()
            self.result["training_identity"] = self.adapter.identity()
            base = MatrixPlan(Path(self.plan.value["base_accuracy"]["plan"]))
            base_audit_path = Path(self.plan.value["base_accuracy"]["audit"])
            if (base.value["plan_hash"] != self.plan.value["base_accuracy"]["plan_hash"]
                    or Evidence.sha256(base_audit_path) != self.plan.value["base_accuracy"]["audit_sha256"]):
                raise ValueError("Pinned complete c03 plan/audit changed")
            base_audit = json.loads(base_audit_path.read_text())
            identifier = self.job["job_id"].removesuffix("forward_autograd") + "grad_x"
            record = base_audit["jobs"][identifier]
            original = Path(record["attempt"]) / "raw/result.json"
            if not record["verified"] or Evidence.sha256(original) != record["result_sha256"]:
                raise ValueError("Original complete dgrad evidence changed")
            old = json.loads(original.read_text())
            differences = [key for key in ("input_hash", "reference_hashes", "source_manifest_hash")
                           if old[key] != self.result[key]]
            if old["identity"]["jit_libraries"] != self.result["training_identity"]["jit_libraries"]:
                differences.append("dgrad binding DSO map")
            for row in self.result["gradient_diagnostic_rows"]:
                original_row = next(value for value in old["operation_boundaries"] if value["boundary"] == row["boundary"])
                differences.extend(row["boundary"] + "." + key for key in ("output_hash", "rel_fro", "threshold",
                    "torch_rel_fro", "max_abs", "mean_abs", "accuracy_pass", "finite", "repeat_equal", "different_buffer")
                    if row[key] != original_row[key])
            self.result["c03_dgrad_agreement"] = {"exact_match": not differences, "differences": differences,
                                                 "evidence": str(original), "sha256": record["result_sha256"]}
            self.adapter.prepare("forward", "prepared_native_launch_with_alloc")
            launch, arguments, order = self.adapter._prepared["forward"]
            if sum(value is self.adapter.x_native for value in arguments) != 1:
                raise ValueError("Prepared input leaf substitution is ambiguous")
            arguments = tuple(self.adapter.autograd_x["native_public"] if value is self.adapter.x_native else value
                              for value in arguments)
            original_prepared = self.adapter._prepared["forward"]
            self.adapter._prepared["forward"] = launch, arguments, order
            try:
                with torch.enable_grad():
                    raw = self.adapter.call("forward", "prepared_native_launch_with_alloc")
            finally:
                self.adapter._prepared["forward"] = original_prepared
            torch.cuda.synchronize()
            identity = self.adapter.identity()
            self.result["raw_prepared_identity"] = identity
            self.result["prepared_autograd"] = {"classification": "N/A_RAW_LAUNCH_HAS_NO_AUTOGRAD_GRAPH",
                "input_requires_grad": self.adapter.autograd_x["native_public"].requires_grad,
                "weight_requires_grad": self.adapter.w_native.requires_grad,
                "output_requires_grad": raw.requires_grad, "output_grad_fn": None if raw.grad_fn is None else type(raw.grad_fn).__name__,
                "source_witness": self.source_witness(launch, identity)}
            capability = raw.grad_fn is None and not differences and all(row["finite"] and row["repeat_equal"]
                and row["different_buffer"] for row in self.result["gradient_diagnostic_rows"])
            self.result.update(status="PASS" if capability else "CAPABILITY_MISMATCH",
                training_gradient_numeric_pass=all(row["accuracy_pass"] for row in self.result["gradient_diagnostic_rows"]),
                qualification_note="Finite real-dy connectivity and exact c03 dgrad agreement; strict FP32 gradient failures remain explicit.")
        except BaseException as error:
            self.result.update(status="ERROR", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            self.result["source_unchanged"] = before == source == SourceTree.manifest(source["root"])
            self.plan.verify_producer(Path(__file__).parent)
            self.result["producer_unchanged"] = True
            if not self.result["source_unchanged"]:
                self.result["status"] = "ERROR_SOURCE_CHANGED"
            Evidence.write(self.output / "result.json", self.result)
            if self.adapter is not None:
                self.adapter.close()
        return 0 if self.result["status"] == "PASS" else 1


class CapabilityController(AutogradController):
    ENTRYPOINT = "luna_autograd_capability.py"
    LOG_SCOPE = "capability"

    @staticmethod
    def cache_operation(job):
        return "grad_x"

    def run(self):
        self.plan.verify_runtime_sources(self.session)
        root = self.plan.root / "capabilities"
        root.mkdir(exist_ok=False)
        expected = [job for job in self.plan.value["jobs"] if job["point"]["config"] == "SMALL_P1" and job["point"]["batch"] == 512]
        result = {"status": "RUNNING", "plan_hash": self.plan.value["plan_hash"], "expected": [job["job_id"] for job in expected], "jobs": []}
        Evidence.write(root / "audit.json", result)
        with (self.plan.root / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for job in expected:
                attempt = root / job["job_id"] / "attempt0001"
                attempt.mkdir(parents=True)
                terminal = self.launch(job, attempt)
                path = attempt / "raw/result.json"
                raw = json.loads(path.read_text()) if path.exists() else {}
                valid = terminal["status"] == "PASS" and raw.get("status") == "PASS" and raw.get("source_unchanged") and raw.get("producer_unchanged")
                result["jobs"].append({"job_id": job["job_id"], "verified": bool(valid), "raw": str(path),
                                       "sha256": Evidence.sha256(path) if path.exists() else None})
                Evidence.write(root / "audit.json", result)
                if not valid:
                    reproduction = attempt / "reproduction"
                    reproduction.mkdir()
                    self.launch(job, reproduction, reproduction=True)
                    break
        result["status"] = "PASS" if len(result["jobs"]) == 4 and all(row["verified"] for row in result["jobs"]) else "INCOMPLETE_DIAGNOSIS_REQUIRED"
        self.plan.verify_runtime_sources(self.session)
        Evidence.write(root / "audit.json", result)
        print(json.dumps(result))
        return 0 if result["status"] == "PASS" else 1


class CapabilityCli:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        run = commands.add_parser("run")
        run.add_argument("--plan", type=Path, required=True)
        worker = commands.add_parser("worker")
        worker.add_argument("--plan", type=Path, required=True)
        worker.add_argument("--job-id", required=True)
        worker.add_argument("--output", type=Path, required=True)
        worker.add_argument("--reproduction", action="store_true")
        args = parser.parse_args()
        return CapabilityController(args).run() if args.command == "run" else CapabilityProbe(args).run()


if __name__ == "__main__":
    raise SystemExit(CapabilityCli.main())
