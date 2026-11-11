"""Explicit full-operation adaptation of existing FP64 handwritten forward APIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import Evidence
from handwritten_adapter import HandwrittenAdapter, HandwrittenLibrary, HandwrittenProbe
from performance_bindings import TorchCalls
from reference_store import ReferenceStore
from session import Session
from standalone_probe import StandaloneProbe


class HandwrittenFp64Adapter(HandwrittenAdapter):
    BOUNDARY = "adapted_public_e2e"

    def __init__(self, case, source):
        import torch

        self.case, self.source, self.prepared = case, source, set()
        policy = source.session.root / "coverage/handwritten_fp64_source02/fp64_contracts.json"
        self.policy = {"path": str(policy), "sha256": Evidence.sha256(policy)}
        self.contract = next(row for row in json.loads(policy.read_text())["contracts"] if row["kind"] == source.source["kind"])
        if case.dtype != torch.float64 or self.contract["math_allowed"] is not True:
            raise ValueError("FP64 handwritten adapter requires reviewed double-precision math")
        if self.contract["source_sha256"] not in source.source["original_files"].values():
            raise ValueError("FP64 source policy does not bind the compiled source")
        self.single_path = source.source["kind"] == "fp64_gmm_p1_v2"
        if not self.single_path and len(case.dims) not in self.contract["allowed_path_counts"]:
            raise ValueError("Original FP64 API path-count restriction; do not launch this canonical shape")
        if case.u % 16 or case.u % 32 or case.point.batch % 32:
            raise ValueError("Conservative original unmasked K/M/N tile requirements differ")
        if any(not tensor.is_contiguous() or tensor.dtype != torch.float64 or tensor.device != case.x.device
               for tensor in (case.x, case.w, case.grad)):
            raise ValueError("Original FP64 API requires same-device contiguous double inputs")
        self.grouped = self.contract["output_semantics"] == "ADJACENT_EQUAL_I_DIM_GROUP_REDUCTION"
        self.operations = {"forward": source.ops.forward}
        self.schemas = {name: str(op._schemas) for name, op in self.operations.items()}
        self.post = TorchCalls(case)

    def call(self, operation, boundary, output=None):
        if (operation, boundary) not in self.prepared or output is not None:
            raise ValueError("Prepare the original allocating forward and its declared adapter")
        import torch

        case = self.case
        with torch.no_grad():
            if self.grouped:
                return self.operations[operation](case.x3, case.w3, case.dims, case.cg).reshape(case.point.batch, -1)
            if self.single_path:
                values, offset = [], 0
                for index, dim in enumerate(case.dims):
                    # A P1 kernel is invoked once per original path. Input slicing
                    # and its required contiguous copy are inside the adapter window.
                    values.append(self.operations[operation](case.x3[:, offset:offset+dim].contiguous(), case.w3[index], 1.0))
                    offset += dim
            else:
                raw = self.operations[operation](case.x3, case.w3, case.dims, 1.0).reshape(case.point.batch, sum(case.dims), case.u)
                values = list(raw.split(case.dims, dim=1))
            # Reuse the existing product adapter's post-CG/run-length reducer.
            # Canonical W and Gold are unchanged; no coefficient is folded into W.
            return self.post.normalize_mm("forward", values)

    def identity(self):
        if Evidence.sha256(Path(self.policy["path"])) != self.policy["sha256"]:
            raise ValueError("FP64 source contract policy changed")
        return {**super().identity(), "variant_id": "handwritten_" + self.source.source["kind"],
                "source_contract": self.policy, "native_source_path_counts": self.contract["allowed_path_counts"],
                "native_kernel_calls_per_adapter_call": len(self.case.dims) if self.single_path else 1,
                "native_coefficient_argument": "canonical cg vector" if self.grouped else "scalar1; canonical post-CG in common product normalizer",
                "layout": "original canonical inputs; P1 slices/copies, native allocations, post-CG and run-length reduction included in adapted_public_e2e",
                "native_grouped_reduction": self.grouped,
                "adapter_sha256": Evidence.sha256(Path(__file__))}


class HandwrittenFp64Probe(HandwrittenProbe):
    def __init__(self, args):
        session = Session(args.session)
        self.reference_store = ReferenceStore(session.root / "references")
        StandaloneProbe.__init__(self, args, reference_factory=self.reference,
            source_factory=lambda case: HandwrittenLibrary(session, args.build), adapter_factory=HandwrittenFp64Adapter,
            boundary_factory=lambda operation: [HandwrittenFp64Adapter.BOUNDARY])
        self.result.update(variant="handwritten_" + json.loads((args.build / "source.json").read_text())["kind"],
                           original_input_layout=True)
        self.result.pop("impl")
        self.result.pop("bimpl")
        for name in ("handwritten_fp64.py", "handwritten_adapter.py", "performance_bindings.py", "reference_store.py"):
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
        args.precision, args.impl, args.bimpl, args.operations = "fp64", 0, 0, "forward"
        return cls(args).run()


if __name__ == "__main__":
    raise SystemExit(HandwrittenFp64Probe.main())
