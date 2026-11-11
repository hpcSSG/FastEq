"""Archived handwritten raw APIs with source, ABI and mapped-DSO verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import Evidence
from reference_store import ReferenceStore
from session import Session
from standalone_probe import StandaloneProbe


class HandwrittenLibrary:
    def __init__(self, session, build):
        import torch

        self.session, self.root = session, Path(build).resolve()
        if not self.root.is_relative_to(session.root / "build/handwritten"):
            raise ValueError("Handwritten build must belong to this session")
        self.manifest_path = self.root / "source.json"
        self.build_path = self.root / "build.json"
        self.source = json.loads(self.manifest_path.read_text())
        self.build = json.loads(self.build_path.read_text())
        self.receipts = {str(path): Evidence.sha256(path) for path in (self.manifest_path, self.build_path)}
        if self.build["status"] != "PASS" or self.build["source_manifest_sha256"] != self.receipts[str(self.manifest_path)]:
            raise ValueError("A successful fresh source build is required")
        if (self.build["torch"], self.build["torch_cuda"], self.build["cxx11_abi"]) != (torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI):
            raise ValueError("Handwritten Torch/CUDA/C++ ABI differs")
        self.library = Path(self.build["binary"]["path"]).resolve()
        if not self.library.is_relative_to(self.root / "binary"):
            raise ValueError("DSO escaped its fresh build")
        self.verify()
        torch.ops.load_library(str(self.library))
        self.ops = getattr(torch.ops, self.source["namespace"])
        if self.library not in self.mapped():
            raise ValueError("Requested handwritten DSO is not actually mapped")

    @staticmethod
    def mapped():
        return {Path(line.split()[-1]).resolve() for line in Path("/proc/self/maps").read_text().splitlines()
                if line.split() and line.split()[-1].startswith("/") and ".so" in line.split()[-1]}

    def verify(self):
        for filename, digest in self.receipts.items():
            if Evidence.sha256(Path(filename)) != digest:
                raise ValueError("Changed handwritten build receipt")
        if Evidence.sha256(self.library) != self.build["binary"]["sha256"]:
            raise ValueError("Handwritten DSO changed")
        for name, digest in self.source["original_files"].items():
            if Evidence.sha256(Path(self.source["archive"]) / name) != digest:
                raise ValueError("Original handwritten source changed")
        for name, digest in self.source["staged_files"].items():
            if Evidence.sha256(self.root / "source" / name) != digest:
                raise ValueError("Staged handwritten source changed")
        policy = self.source.get("math_policy")
        if policy and Evidence.sha256(Path(policy["path"])) != policy["sha256"]:
            raise ValueError("Handwritten arithmetic policy changed")

    def identity(self):
        self.verify()
        if self.library not in self.mapped():
            raise ValueError("Handwritten DSO is no longer mapped")
        return {"kind": self.source["kind"], "source_hash": self.source["source_hash"],
                "production_source_hash": self.source["source_hash"],
                "source_manifest": {"path": str(self.manifest_path), "sha256": self.receipts[str(self.manifest_path)]},
                "build_manifest": {"path": str(self.build_path), "sha256": self.receipts[str(self.build_path)]},
                "binary_hash": self.build["binary"]["sha256"],
                "binary_files": {str(self.library): {"sha256": self.build["binary"]["sha256"]}},
                "operator_namespace": self.source["namespace"], "actual_mapped_dso": str(self.library),
                "architecture": self.build["architectures"], "torch": self.build["torch"],
                "torch_cuda": self.build["torch_cuda"], "cxx11_abi": self.build["cxx11_abi"],
                "math_policy": self.source.get("math_policy")}


class HandwrittenAdapter:
    BOUNDARY = "raw_allocating"

    def __init__(self, case, source):
        if source.source["kind"] != "tf32_v7" or case.point.precision != "tf32":
            raise ValueError("This adapter currently implements only reviewed TF32-v7 forward/backward-v8")
        if case.j != 16 or len(case.dims) not in (4, 10, 17):
            raise ValueError("Raw TF32-v7 source fixes J=16 and P=4/10/17")
        self.case, self.source, self.prepared = case, source, set()
        self.operations = {"forward": source.ops.forward, "grad_x": source.ops.backward}
        self.schemas = {name: str(op._schemas) for name, op in self.operations.items()}

    def prepare(self, operation, boundary):
        if operation not in self.operations or boundary != self.BOUNDARY:
            raise ValueError("Raw handwritten API does not expose this operation/boundary")
        self.prepared.add((operation, boundary))

    def new_output(self, operation):
        if operation not in self.operations:
            raise ValueError(operation)
        return None

    def call(self, operation, boundary, output=None):
        if (operation, boundary) not in self.prepared or output is not None:
            raise ValueError("Prepare the raw allocating API; caller-owned output is not supported")
        case = self.case
        value = self.operations[operation](case.x3 if operation == "forward" else case.grad3, case.w3, case.dims, case.cg)
        return value.reshape(case.point.batch, -1)

    def identity(self):
        return {**self.source.identity(), "variant_id": "handwritten_tf32_v7", "api_schemas": self.schemas,
                "prepared": sorted([list(value) for value in self.prepared]),
                "call_boundary": self.BOUNDARY, "layout": "canonical original BIU/PUV; no pack or coefficient fold",
                "adapter_sha256": Evidence.sha256(Path(__file__))}

    def close(self):
        self.source.verify()
        if any(self.case.tensor_hash(getattr(self.case, name)) != self.case.hashes[name + "_sha256"]
               for name in ("x", "w", "grad")):
            raise ValueError("Handwritten backend mutated canonical input tensors")


class HandwrittenProbe(StandaloneProbe):
    def __init__(self, args):
        session = Session(args.session)
        self.reference_store = ReferenceStore(session.root / "references")
        super().__init__(args, reference_factory=self.reference,
                         source_factory=lambda case: HandwrittenLibrary(session, args.build),
                         adapter_factory=HandwrittenAdapter, boundary_factory=lambda operation: [HandwrittenAdapter.BOUNDARY])
        self.result.update(variant="handwritten_tf32_v7", original_input_layout=True)
        path = Path(__file__)
        self.producer_files[str(path)] = Evidence.sha256(path)
        (self.output / "producer_sources" / path.name).write_bytes(path.read_bytes())

    def reference(self, case):
        reference = self.reference_store.get(case)
        self.result["reference_store"] = reference.store_receipt
        return reference

    def observe_result(self, row, output, repeated):
        if row["passed"]:
            return
        import torch

        row["failed_outputs"] = {}
        for name, tensor in (("first", output), ("repeat", repeated)):
            path = self.output / (row["operation"] + "_" + row["boundary"] + "_" + name + ".pt")
            torch.save(tensor.detach().cpu(), path)
            row["failed_outputs"][name] = {"path": str(path), "sha256": Evidence.sha256(path)}

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--build", type=Path, required=True)
        parser.add_argument("--precision", choices=("tf32",), default="tf32")
        parser.add_argument("--point", default="SMALL_P1/512")
        parser.add_argument("--operations", default="forward,grad_x")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--trace", action="store_true")
        args = parser.parse_args()
        args.impl, args.bimpl = 7, 8
        return cls(args).run()


if __name__ == "__main__":
    raise SystemExit(HandwrittenProbe.main())
