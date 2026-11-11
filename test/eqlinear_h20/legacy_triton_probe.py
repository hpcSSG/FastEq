"""Original Triton entries with explicit math and actually loaded cubin evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from common import Evidence
from handwritten_adapter import HandwrittenAdapter, HandwrittenProbe
from legacy_triton import LegacyTritonSource
from reference_store import ReferenceStore
from session import Session
from standalone_probe import StandaloneProbe


class LegacyTritonLibrary:
    def __init__(self, session, source_root):
        self.session, self.source = session, LegacyTritonSource(source_root)
        self.saved = {}
        import triton
        import triton.compiler.compiler
        import triton.runtime.jit

        self.producers = {str(Path(module.__file__)): Evidence.sha256(Path(module.__file__))
                          for module in (triton, triton.compiler.compiler, triton.runtime.jit)}

    def verify(self):
        self.source.verify()
        for filename, digest in (self.saved | self.producers).items():
            if Evidence.sha256(Path(filename)) != digest:
                raise ValueError("Loaded legacy Triton kernel evidence changed")

    def save(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(data).hexdigest()
        if path.exists():
            if Evidence.sha256(path) != digest:
                raise ValueError("Changed cached Triton binary evidence")
        else:
            with path.open("xb") as stream:
                stream.write(data)
        self.saved[str(path)] = digest
        return {"path": str(path), "sha256": digest}

    def identity(self):
        import triton
        import triton.compiler.compiler
        import triton.runtime.jit

        kernels = {}
        for name, function in vars(self.source.module).items():
            if not name.startswith("_so3_linear_") or not hasattr(function, "device_caches"):
                continue
            for device, cache in function.device_caches.items():
                for kernel in cache[0].values():
                    if kernel.module is None or kernel.function is None:
                        continue
                    digest = hashlib.sha256(kernel.kernel).hexdigest()
                    root = self.session.root / "build/legacy_triton/loaded_kernels" / digest
                    metadata = json.loads(json.dumps(kernel.metadata._asdict(), default=str))
                    cubin = self.save(root / "kernel.cubin", kernel.kernel)
                    files = {"cubin": cubin}
                    for format_name in ("ptx", "llir", "ttgir", "ttir"):
                        if format_name in kernel.asm:
                            files[format_name] = self.save(root / ("kernel." + format_name), kernel.asm[format_name].encode())
                    kernels[kernel.hash] = {"entry": name, "name": kernel.name, "device": device,
                        "compiled_hash": kernel.hash, "cuda_module_loaded": True, "cuda_function_loaded": True,
                        "metadata": metadata, "files": files, "registers": kernel.n_regs, "spills": kernel.n_spills}
        if self.source.gpu_execution_requested and not kernels:
            raise ValueError("Original Triton wrapper ran without an actually loaded expected CUDA kernel")
        self.verify()
        return {"backend": "original_triton_cuda", "source": self.source.identity(),
                "source_hash": self.source.digest, "production_source_hash": self.source.digest,
                "binary_hash": Evidence.fingerprint({key: value["files"]["cubin"]["sha256"] for key, value in kernels.items()}),
                "loaded_kernels": kernels, "binary_files": {value["files"]["cubin"]["path"]: {"sha256": value["files"]["cubin"]["sha256"]}
                                                             for value in kernels.values()},
                "triton_version": triton.__version__, "triton_producers": self.producers,
                "loaded_kernel_evidence": "In-memory CompiledKernel.kernel bytes passed to CUDA loader; initialized module/function required; not inferred from cache directory presence"}


class LegacyTritonAdapter(HandwrittenAdapter):
    BOUNDARY = "raw_allocating"

    def __init__(self, case, source, entry):
        self.case, self.source, self.entry, self.prepared = case, source, entry, set()
        self.precision = {"tf32": "tf32x3", "fp32": "ieee"}[case.point.precision]
        self.controls = source.source.controls(entry, self.precision)
        self.operation = "grad_x" if entry == "backward_x" else "forward"
        self.operations = {self.operation: entry}
        self.layout = source.source.layout(case.u, case.dims, case.cg)
        self.setup_metadata = {}

    def prepare(self, operation, boundary):
        if (operation, boundary) in self.prepared:
            return
        super().prepare(operation, boundary)
        import torch

        torch.cuda.synchronize()
        start = time.perf_counter()
        self.call(operation, boundary)
        torch.cuda.synchronize()
        self.setup_metadata[operation + "/" + boundary] = {"first_call_prepare_seconds": time.perf_counter() - start,
            "includes": "Original JIT/cache lookup, first allocating execution and original auto tuning if selected; not a pure compilation duration"}

    def call(self, operation, boundary, output=None):
        if (operation, boundary) not in self.prepared or output is not None:
            raise ValueError("Prepare the original allocating Triton entry before sampling")
        import torch

        with torch.no_grad():
            return self.source.source.call(self.entry, self.precision, self.case.w,
                                           self.case.grad if operation == "grad_x" else self.case.x, self.layout)

    def identity(self):
        return {**self.source.identity(), "variant_id": f"legacy_triton_{self.entry}_{self.case.point.precision}",
                "entry": self.entry, "math_controls": self.controls, "call_boundary": self.BOUNDARY,
                "layout_sizes": self.layout.sizes, "adapter_sha256": Evidence.sha256(Path(__file__)),
                "actual_auto_cache": repr(self.source.source.module.get_best_impl_cache()) if self.entry == "public_auto" else None}


class LegacyTritonProbe(HandwrittenProbe):
    def __init__(self, args):
        session = Session(args.session)
        self.reference_store = ReferenceStore(session.root / "references")
        StandaloneProbe.__init__(self, args, reference_factory=self.reference,
            source_factory=lambda case: LegacyTritonLibrary(session, args.source_root),
            adapter_factory=lambda case, source: LegacyTritonAdapter(case, source, args.entry),
            boundary_factory=lambda operation: [LegacyTritonAdapter.BOUNDARY])
        self.result["variant"] = f"legacy_triton_{args.entry}_{args.precision}"
        for name in ("legacy_triton_probe.py", "legacy_triton.py", "handwritten_adapter.py", "reference_store.py"):
            path = Path(__file__).with_name(name)
            self.producer_files[str(path)] = Evidence.sha256(path)
            (self.output / "producer_sources" / path.name).write_bytes(path.read_bytes())

    def run(self):
        result = super().run()
        self.result["setup_metadata"] = self.adapter.setup_metadata
        self.result["source_unchanged"] = True
        self.flush()
        return result

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--source-root", type=Path, required=True)
        parser.add_argument("--entry", choices=tuple(LegacyTritonSource.ENTRIES), required=True)
        parser.add_argument("--precision", choices=("fp32", "tf32"), required=True)
        parser.add_argument("--point", default="SMALL_P1/512")
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--trace", action="store_true")
        args = parser.parse_args()
        LegacyTritonSource.controls(args.entry, "ieee" if args.precision == "fp32" else "tf32x3")
        args.operations, args.impl, args.bimpl = "grad_x" if args.entry == "backward_x" else "forward", 0, 0
        return cls(args).run()


if __name__ == "__main__":
    raise SystemExit(LegacyTritonProbe.main())
