"""Private loading of preserved legacy Triton entrypoints with explicit math guards."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from common import Evidence
from cases import Case, CONFIGS


class LegacyTritonSource:
    REF = "4463b10c1fa290f76bd2cbb8213a6e9377a34e78"
    PATH = "fasteq/triton/so3_linear.py"
    ENTRIES = {"serial_cta": "so3_linear_serial_cta", "path_parallel": "so3_linear_path_parallel",
               "group_parallel": "so3_linear_group_parallel", "backward_x": "so3_linear_backward_x",
               "public_auto": "so3_linear_auto"}

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "source_manifest.json"
        self.manifest_hash = Evidence.sha256(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.path = self.root / self.PATH
        self.digest = self.manifest["files"][self.PATH]
        self.module_name = "_eqlinear_h20_legacy_triton_" + self.digest[:20]
        self.verify()
        if self.module_name in sys.modules:
            raise ValueError("Legacy Triton module already loaded; use one isolated worker per source contract")
        spec = importlib.util.spec_from_file_location(self.module_name, self.path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[self.module_name]
            raise
        self.module = module
        self.gpu_execution_requested = False
        if Path(module.__file__).resolve() != self.path:
            raise ValueError("Wrong actual legacy module path")
        for entry in self.ENTRIES.values():
            if not callable(getattr(module, entry, None)):
                raise ValueError("Preserved entry missing: " + entry)
        self.verify()

    def verify(self):
        if self.manifest["ref"] != self.REF or Evidence.sha256(self.manifest_path) != self.manifest_hash:
            raise ValueError("Legacy source manifest drift")
        if Evidence.sha256(self.path) != self.digest:
            raise ValueError("Legacy Triton source drift")

    @classmethod
    def controls(cls, entry, precision):
        if entry not in cls.ENTRIES:
            raise ValueError("Unknown original Triton entry")
        if precision not in ("ieee", "tf32x3"):
            raise ValueError("Only explicit IEEE or TF32x3 permitted; TF32x1 and non-f32 legacy input contracts are forbidden")
        if entry == "public_auto":
            if precision != "ieee":
                raise ValueError("Original bool auto selector cannot choose TF32x3; never use tf32=True")
            return {"tf32": False}
        return {"input_precision": precision}

    def layout(self, u, dims, coefficients):
        self.verify()
        output = tuple(d for i, d in enumerate(dims) if i == 0 or d != dims[i - 1])
        return self.module.uniform_layout(u, u, tuple(dims), output, coefficients=tuple(coefficients))

    def call(self, entry, precision, w, x, layout):
        controls = self.controls(entry, precision)
        import torch
        if x.dtype != torch.float32 or w.dtype != torch.float32:
            raise ValueError("Original legacy Triton API accepts float32 only within this user scope")
        self.verify()
        self.gpu_execution_requested = True
        return getattr(self.module, self.ENTRIES[entry])(w, x, layout, **controls)

    def identity(self):
        self.verify()
        return {"module": self.module_name, "actual_file": str(Path(self.module.__file__).resolve()),
                "source_sha256": self.digest, "manifest": str(self.manifest_path),
                "manifest_sha256": self.manifest_hash, "entries": self.ENTRIES,
                "precision_policy": {"explicit": ["ieee", "tf32x3"], "public_auto": {"tf32": False}},
                "gpu_execution_requested": self.gpu_execution_requested}


class LegacyTritonCpuCheck:
    @staticmethod
    def run(source):
        records = []
        for name, (u, dims) in CONFIGS.items():
            case = Case.__new__(Case)
            case.u, case.dims, case.cg = u, dims, [.01 + .001 * p for p in range(len(dims))]
            descriptor = case.make_descriptor()
            layout = source.layout(u, dims, case.cg)
            if tuple(tuple(operand.segments) for operand in descriptor.operands) != (layout.weight_segments, layout.input_segments, layout.output_segments):
                raise ValueError("Legacy metadata segments differ from the actual shared canonical descriptor")
            if tuple(tuple(path.indices) for path in descriptor.paths) != layout.paths or tuple(float(path.coefficients) for path in descriptor.paths) != layout.coefficients:
                raise ValueError("Legacy path mapping/coefficients differ from the actual canonical descriptor")
            if layout.sizes != tuple(operand.size for operand in descriptor.operands):
                raise ValueError("Legacy flattened shapes differ from the actual canonical descriptor")
            records.append({"config": name, "sizes": layout.sizes, "paths": len(layout.paths),
                            "canonical_descriptor_segments_paths_nonuniform_coefficients_exact": True})
        rejected = []
        for entry, precision in [("serial_cta", "tf32"), ("public_auto", "tf32x3"), ("public_auto", "tf32"), ("serial_cta", "fp64")]:
            try:
                source.controls(entry, precision)
            except ValueError:
                rejected.append([entry, precision])
            else:
                raise ValueError("Forbidden precision control accepted")
        return {"status": "CPU_IMPORT_AND_CANONICAL_DESCRIPTOR_AND_CONTROL_GUARDS_PASS_GPU_UNTESTED",
                "identity": source.identity(), "canonical_configs": records,
                "rejected_controls": rejected, "gpu_execution": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = LegacyTritonSource(args.source_root)
    Evidence.write(args.output, LegacyTritonCpuCheck.run(source))


if __name__ == "__main__":
    main()
