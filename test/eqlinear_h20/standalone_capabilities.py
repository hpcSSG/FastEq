"""Probe the original standalone public autograd API without inventing a wrapper."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback
import warnings

from cases import Case, points
from common import Evidence
from protocol import Accuracy
from reference_store import ReferenceStore
from session import Session
from standalone import StandaloneAdapter, StandaloneSource


class PublicAutogradCapability:
    FILES = ("standalone_capabilities.py", "standalone.py", "cases.py", "common.py", "protocol.py", "reference_store.py", "session.py")

    @classmethod
    def run(cls, args):
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        import torch

        session = Session(args.session)
        if not args.output.resolve().is_relative_to(session.root) or args.output.exists():
            raise ValueError("Capability evidence needs a fresh session directory")
        args.output.mkdir(parents=True)
        producer = {name: Evidence.sha256(Path(__file__).with_name(name)) for name in cls.FILES}
        producer_root = args.output / "producer"
        producer_root.mkdir()
        for name in cls.FILES:
            producer_root.joinpath(name).write_bytes(Path(__file__).with_name(name).read_bytes())
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(1)
        actual_gpu = str(torch.cuda.get_device_properties().uuid)
        selected_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if actual_gpu.lower().removeprefix("gpu-") != selected_gpu.lower().removeprefix("gpu-"):
            raise ValueError("Capability process did not use its twice-idle selected physical GPU")
        point = next(point for point in points() if point.key == "SMALL_P1/512/" + args.precision)
        case = Case(point, fixture_root=session.root / "fixtures")
        reference = ReferenceStore(session.root / "references").get(case)
        source = StandaloneSource(session, args.precision)
        manifest = source.stage()
        adapter = StandaloneAdapter(case, source.library, args.precision, 0, 1, source.manifest_path)
        value = {"classification": "PUBLIC_API_CAPABILITY_NOT_PERFORMANCE", "status": "RUNNING", "point": point.columns(),
            "input_hash": case.input_hash, "reference_hashes": reference.hashes, "reference_store": reference.store_receipt,
            "GPU_UUID": str(torch.cuda.get_device_properties().uuid), "w_requires_grad": False,
            "source_manifest": manifest, "source_sha256": Evidence.sha256(Path(__file__)), "manual_dgrad_called": False}
        value["producer"] = producer
        try:
            with torch.no_grad():
                first = adapter.ops.forward(case.x3, case.w3.detach(), case.dims, case.cg, 0)
                second = adapter.ops.forward(case.x3, case.w3.detach(), case.dims, case.cg, 0)
                torch.cuda.synchronize()
            value["no_grad_forward"] = Accuracy.evaluate(first, reference.values["fp64"]["forward"],
                reference.torch_errors["forward"], args.precision, torch.equal(first, second), first.data_ptr() != second.data_ptr())
            value["no_grad_forward_output_hash"] = case.tensor_hash(first)
            if not value["no_grad_forward"]["passed"]:
                raise ValueError("Public no_grad baseline did not pass; cannot infer autograd support")
            gradients = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for repeat in range(2):
                    x = case.x3.detach().requires_grad_(True)
                    with torch.enable_grad():
                        output = adapter.ops.forward(x, case.w3.detach(), case.dims, case.cg, 0)
                        torch.cuda.synchronize()
                        observation = {"x_requires_grad": x.requires_grad, "w_requires_grad": case.w3.requires_grad,
                            "output_requires_grad": output.requires_grad, "grad_fn": type(output.grad_fn).__name__ if output.grad_fn else None,
                            "actual_dy_hash": case.tensor_hash(case.grad3), "actual_dy_nonzero": bool(torch.count_nonzero(case.grad3).item())}
                        try:
                            gradient = torch.autograd.grad(output, x, grad_outputs=case.grad3, allow_unused=True)[0]
                            torch.cuda.synchronize()
                            observation["x_gradient_is_none"] = gradient is None
                        except RuntimeError as error:
                            message = str(error)
                            if not any(text in message.lower() for text in ("not implemented", "not have been used", "does not require grad", "does not have a grad_fn")):
                                raise
                            gradient = None
                            observation.update(x_gradient_is_none=True, autograd_error=message)
                    value.setdefault("public_autograd_observations", []).append(observation)
                    gradients.append(gradient)
                value["warnings"] = [str(item.message) for item in caught]
            if all(gradient is not None for gradient in gradients):
                value["autograd_grad_x"] = Accuracy.evaluate(gradients[0], reference.values["fp64"]["grad_x"],
                    reference.torch_errors["grad_x"], args.precision, torch.equal(*gradients), gradients[0].data_ptr() != gradients[1].data_ptr())
                value["status"] = "PUBLIC_AUTOGRAD_PASS" if value["autograd_grad_x"]["passed"] else "PUBLIC_AUTOGRAD_NUMERICAL_FAIL"
            else:
                value["status"] = "PUBLIC_AUTOGRAD_UNSUPPORTED_ACTUAL_X_GRADIENT_NONE"
            binding = source.original / "csrc/torch_bind.cpp"
            text = binding.read_text()
            if producer != {name: Evidence.sha256(Path(__file__).with_name(name)) for name in cls.FILES}:
                raise ValueError("Capability producer changed while the probe was running")
            value.update(identity=adapter.identity(), binding_source={"path": str(binding), "sha256": Evidence.sha256(binding),
                "forward_registration": [line.strip() for line in text.splitlines() if "m.def(\"forward(" in line or "m.impl(\"forward\"" in line],
                "explicit_autograd_registration_present": "Autograd" in text or "register_autograd" in text},
                complete=True, no_production_autograd_wrapper_created=True)
            value["actual_dispatch_registration"] = [line for line in torch._C._dispatch_dump_table(adapter.namespace + "::forward").splitlines()
                                                     if "CUDA:" in line or "Autograd" in line]
            value["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "NVIDIA_TF32_OVERRIDE": os.environ["NVIDIA_TF32_OVERRIDE"]}
            Evidence.write(args.output / "result.json", value)
            return 0
        except Exception as error:
            value.update(status="CAPABILITY_PROBE_ERROR", error=str(error), traceback=traceback.format_exc())
            Evidence.write(args.output / "result.json", value)
            raise
        finally:
            adapter.close()


class CapabilityAudit:
    @classmethod
    def run(cls, root, session, terminals):
        from qualification import RecordedAccuracy

        records = {}
        for precision in StandaloneSource.CONFIG:
            path = root / precision / "result.json"
            value = json.loads(path.read_text())
            if terminals[precision]["status"] != "PASS" or value.get("complete") is not True:
                raise ValueError("A capability process did not complete safely")
            producer = root / precision / "producer"
            if value["producer"] != {name: Evidence.sha256(producer / name) for name in PublicAutogradCapability.FILES}:
                raise ValueError("Saved capability producer differs from its result")
            identity = value["identity"]
            source = StandaloneSource(session, precision)
            binding = value["binding_source"]
            if (Evidence.sha256(Path(identity["binary"])) != identity["binary_hash"] or
                    value["source_manifest"] != source.stage() or identity["source_hash"] != value["source_manifest"]["source_hash"] or
                    Evidence.sha256(Path(binding["path"])) != binding["sha256"]):
                raise ValueError("Actual public operator source/DSO changed")
            if not RecordedAccuracy.passes(value["no_grad_forward"], precision):
                raise ValueError("Minimal no_grad forward failed independent recorded-metric audit")
            observations = value["public_autograd_observations"]
            if (len(observations) != 2 or any(row["x_requires_grad"] is not True or row["w_requires_grad"] is not False or
                    row["actual_dy_nonzero"] is not True for row in observations) or
                    len({row["actual_dy_hash"] for row in observations}) != 1 or value["manual_dgrad_called"] is not False):
                raise ValueError("The public autograd test omitted its real nonzero-dy x-gradient obligation")
            status = value["status"]
            if status == "PUBLIC_AUTOGRAD_UNSUPPORTED_ACTUAL_X_GRADIENT_NONE":
                if not all(row["x_gradient_is_none"] is True for row in observations) or binding["explicit_autograd_registration_present"]:
                    raise ValueError("Autograd limitation lacks matching original binding and minimal-probe evidence")
            elif status in ("PUBLIC_AUTOGRAD_PASS", "PUBLIC_AUTOGRAD_NUMERICAL_FAIL"):
                if any(row["x_gradient_is_none"] for row in observations) or (
                        RecordedAccuracy.passes(value["autograd_grad_x"], precision) != (status == "PUBLIC_AUTOGRAD_PASS")):
                    raise ValueError("Autograd qualification disagrees with actual gradient evidence")
            else:
                raise ValueError("An unsafe/unclassified result cannot establish API capability")
            records[precision] = {"status": status, "no_grad_forward_passed": True,
                "result": {"path": str(path), "sha256": Evidence.sha256(path)},
                "terminal": {"path": str(session.root / "logs" / ("standalone_public_autograd_" + precision + "_01.json")),
                    "returncode": terminals[precision]["returncode"]}}
        Evidence.write(root / "audit.json", {"classification": "ORIGINAL_PUBLIC_AUTOGRAD_CAPABILITY_NOT_PERFORMANCE",
            "complete": True, "independent_readback_passed": True, "records": records,
            "scope": "original per-precision TORCH_LIBRARY forward registration; minimal impl0 at SMALL_P1/512",
            "manual_derivative_api_is_not_public_autograd": True, "source_sha256": Evidence.sha256(Path(__file__)),
            "qualification_sha256": Evidence.sha256(Path(__file__).with_name("qualification.py"))})
        return 0


class CapabilityCLI:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--precision", choices=("tf32", "fp32", "fp64"))
        parser.add_argument("--scan", action="store_true")
        parser.add_argument("--gpu", type=int, default=5)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        if args.scan:
            session = Session(args.session)
            if not args.output.resolve().is_relative_to(session.root) or args.output.exists():
                raise ValueError("A capability scan needs a fresh session output")
            args.output.mkdir(parents=True)
            terminals = {}
            for precision in StandaloneSource.CONFIG:
                name = "standalone_public_autograd_" + precision + "_01"
                gpu = session.idle_gpu(name, args.gpu)
                terminals[precision] = session.runner().run(name, [session.paths["FASTEQ_PY"], "-B", str(Path(__file__)),
                    "--session", str(session.root), "--precision", precision, "--output", str(args.output / precision)],
                    90, Path(session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": gpu})
                if terminals[precision]["status"] != "PASS":
                    return 1
            return CapabilityAudit.run(args.output, session, terminals)
        if args.precision is None:
            parser.error("Use --precision for one isolated operator, or --scan for three isolated processes")
        return PublicAutogradCapability.run(args)


if __name__ == "__main__":
    raise SystemExit(CapabilityCLI.main())
