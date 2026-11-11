"""Original CUDA entrypoints; hard float64 guards precede every operator call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import Evidence
from handwritten_adapter import HandwrittenAdapter, HandwrittenLibrary, HandwrittenProbe
from reference_store import ReferenceStore
from session import Session
from standalone_probe import StandaloneProbe


class OriginalCudaFp64Adapter(HandwrittenAdapter):
    @staticmethod
    def require_double(case):
        import torch

        if case.point.precision != "fp64" or any(value.dtype != torch.float64 for value in (case.x, case.w, case.grad)):
            raise ValueError("Original CUDA float32 branches are TF32x1 and must never execute")

    def __init__(self, case, source):
        self.require_double(case)
        if source.source["kind"] != "original_cuda_fp64_4463b10" or source.source["runtime_policy"]["allowed_dtype"] != "float64":
            raise ValueError("Expected original source build with explicit FP64-only policy")
        if len(case.dims) != 4 or case.j != 16 or case.u % 32 or case.point.batch % 32:
            raise ValueError("Original double helper supports the P4/J16 tile contract only")
        self.case, self.source, self.prepared = case, source, set()
        self.operations = {"forward": source.ops.forward, "grad_x": source.ops.backward}
        self.schemas = {name: str(op._schemas) for name, op in self.operations.items()}

    def call(self, operation, boundary, output=None):
        self.require_double(self.case)
        return super().call(operation, boundary, output)

    def identity(self):
        return {**super().identity(), "variant_id": "original_cuda_fp64_4463b10",
                "adapter_sha256": Evidence.sha256(Path(__file__)),
                "runtime_policy": "Double tensors checked at construction and before every call; compiled f32 branches never entered",
                "grad_x_work": "Original registered backward includes W transpose/contiguous and calls original double forward helper"}


class OriginalCudaFp64Probe(HandwrittenProbe):
    def __init__(self, args):
        session = Session(args.session)
        self.reference_store = ReferenceStore(session.root / "references")
        StandaloneProbe.__init__(self, args, reference_factory=self.reference,
            source_factory=lambda case: HandwrittenLibrary(session, args.build), adapter_factory=OriginalCudaFp64Adapter,
            boundary_factory=lambda operation: [OriginalCudaFp64Adapter.BOUNDARY])
        self.result.update(variant="original_cuda_fp64_4463b10", original_input_layout=True)
        self.result.pop("impl")
        self.result.pop("bimpl")
        for name in ("original_cuda_probe.py", "handwritten_adapter.py", "reference_store.py"):
            path = Path(__file__).with_name(name)
            self.producer_files[str(path)] = Evidence.sha256(path)
            (self.output / "producer_sources" / path.name).write_bytes(path.read_bytes())

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--build", type=Path, required=True)
        parser.add_argument("--point", default="SMALL_P1/512")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--trace", action="store_true")
        args = parser.parse_args()
        args.precision, args.impl, args.bimpl, args.operations = "fp64", 0, 0, "forward,grad_x"
        return cls(args).run()


if __name__ == "__main__":
    raise SystemExit(OriginalCudaFp64Probe.main())
