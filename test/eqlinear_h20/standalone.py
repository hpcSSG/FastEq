"""Isolated source builds and adapters for the archived TF32x3/FP32/FP64 APIs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from common import Evidence
from session import Session


class StandaloneSource:
    CONFIG = {
        "tf32": ("gmm_tf32_src/cutlass_impl", "gmm_cutlass", "libgmm_cutlass.so"),
        "fp32": ("gmm_tf32_src/cutlass_fp32_impl", "gmm_fp32", "libgmm_fp32.so"),
        "fp64": ("gmm_src/cutlass_impl", "gmm_cutlass_fp64", "libgmm_cutlass_fp64.so"),
    }

    def __init__(self, session, precision, build_root=None):
        self.session, self.precision = session, precision
        relative, self.namespace, self.library_name = self.CONFIG[precision]
        self.original = Path(session.paths["OPT_ROOT"]) / relative
        self.root = Path(session.paths["ARTIFACT_ROOT"]).parent / "eqlinear_h20_sources" / ("standalone_" + precision)
        self.build = Path(build_root).resolve() if build_root is not None else self.root / "build"
        if build_root is not None and not self.build.is_relative_to(session.root):
            raise ValueError("An isolated build must stay in its session")
        self.library = self.build / "bin" / self.library_name
        self.headers = Path(session.paths["OPT_ROOT"]) / "3rdparty/cutlass"
        self.manifest_path = session.root / "manifests" / ("standalone_" + precision + "_source.json")

    @staticmethod
    def files(root):
        return {str(path.relative_to(root)): Evidence.sha256(path) for path in sorted(root.rglob("*"))
                if path.is_file() and "build" not in path.relative_to(root).parts and "__pycache__" not in path.parts}

    def stage(self):
        original = self.files(self.original)
        if not self.root.exists():
            shutil.copytree(self.original, self.root)
        staged = self.files(self.root)
        if staged != original:
            raise RuntimeError("Existing standalone source copy differs from the immutable snapshot")
        header_files = self.files(self.headers)
        manifest = {"precision": self.precision, "original": str(self.original), "source_root": str(self.root),
                    "files": original, "source_hash": Evidence.fingerprint(original), "headers_root": str(self.headers),
                    "headers_hash": Evidence.fingerprint(header_files), "header_files": header_files,
                    "build_only_patch": "none", "mathematical_sources_modified": False}
        if self.manifest_path.exists():
            if json.loads(self.manifest_path.read_text()) != manifest:
                raise RuntimeError("Source manifest changed; use a new build identity")
        else:
            Evidence.write(self.manifest_path, manifest)
        return manifest

    def compile(self, attempt, definitions=None):
        manifest = self.stage()
        runner = self.session.runner()
        python = self.session.paths["FASTEQ_PY"]
        prefix = subprocess.check_output([python, "-c", "import torch;print(torch.utils.cmake_prefix_path)"],
                                         text=True, timeout=20).strip()
        configure = ["cmake", "-S", str(self.root), "-B", str(self.build), "-G", "Ninja",
                     "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_POLICY_DEFAULT_CMP0146=OLD",
                     "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", "-DCMAKE_PREFIX_PATH=" + prefix,
                     "-DCUTLASS_ROOT=" + str(self.headers), "-DPython3_EXECUTABLE=" + python,
                     "-DPYTHON_EXECUTABLE=" + python, "-DCMAKE_CUDA_ARCHITECTURES=90a-real"]
        definitions = dict(definitions or {})
        if definitions:
            import re
            if any(re.fullmatch(r"(?:F64M2_|F64_|EQLIN_)[A-Z0-9_]+", key) is None or
                   re.fullmatch(r"[0-9]+", str(value)) is None for key, value in definitions.items()):
                raise ValueError("Only existing numeric CMake compile controls are accepted")
            configure += ["-D" + key + "=" + str(value) for key, value in sorted(definitions.items())]
        environment = {"CUDA_VISIBLE_DEVICES": "", "TORCH_CUDA_ARCH_LIST": "9.0a"}
        result = runner.run(f"standalone_{self.precision}_{attempt}_configure", configure, 90, self.root, environment)
        if result["status"] != "PASS":
            return result
        result = runner.run(f"standalone_{self.precision}_{attempt}_build",
                            ["cmake", "--build", str(self.build), "--target", self.namespace, "--parallel", "2"],
                            600, self.root, environment)
        if result["status"] == "PASS":
            if not self.library.is_file():
                raise FileNotFoundError(self.library)
            record = {"source_manifest": str(self.manifest_path), "source_hash": manifest["source_hash"],
                      "headers_hash": manifest["headers_hash"], "configure_argv": configure,
                      "binary": str(self.library), "binary_hash": Evidence.sha256(self.library),
                      "namespace": self.namespace, "build_receipt": result,
                      "compile_definitions": definitions, "build_root": str(self.build),
                      "cmake_cache_sha256": Evidence.sha256(self.build / "CMakeCache.txt"),
                      "ninja_sha256": Evidence.sha256(self.build / "build.ninja"),
                      "tool_sha256": Evidence.sha256(Path(__file__))}
            Evidence.write(self.session.root / "manifests" / f"standalone_{self.precision}_{attempt}_build.json", record)
            runner.run(f"standalone_{self.precision}_{attempt}_elf", ["readelf", "-d", str(self.library)], 20)
            runner.run(f"standalone_{self.precision}_{attempt}_cuda_arch", ["cuobjdump", "--list-elf", str(self.library)], 20)
        return result


class StandaloneAdapter:
    """Same canonical layouts; never converts a call boundary into a different label."""
    BOUNDARIES = ("forward_allocating", "run_with_activation_prep", "prepared_gemm")

    def __init__(self, case, library, precision, impl=0, bimpl=1, source_manifest=None,
                 api_controls=None, activation_layout="biu"):
        import torch

        if precision == "tf32" and impl in (1, 2):
            raise ValueError("TF32x1 execution is explicitly excluded by the user")
        if precision != case.point.precision:
            raise ValueError("Standalone precision must consume the matching canonical fixture")
        limits = {"tf32": 9, "fp32": 4, "fp64": 8}
        if not 0 <= impl < limits[precision]:
            raise ValueError("Unknown standalone forward implementation id")
        known = {"wgrad_tile", "wgrad_split_k", "wgrad_kernel", "wgrad_transpose_mode"}
        if api_controls and (precision != "fp64" or set(api_controls) - known or
                             any(type(value) is not int for value in api_controls.values())):
            raise ValueError("Only existing FP64 integer wgrad API controls are accepted")
        if activation_layout not in ("biu", "pbu") or (activation_layout == "pbu" and (precision != "tf32" or impl not in (4, 5, 6, 7, 8))):
            raise ValueError("PBU activation layout requires an existing TF32 in-kernel/register-split path")
        self.case, self.precision, self.impl, self.bimpl = case, precision, impl, bimpl
        self.api_controls = dict(api_controls or {})
        self.resolved_api_controls = {"wgrad_tile": 0, "wgrad_split_k": 1, "wgrad_kernel": 0, "wgrad_transpose_mode": 3,
                                      **self.api_controls}
        self.activation_layout = activation_layout
        self.pack_pbu = activation_layout == "pbu" and case.j == sum(case.dims)
        self.activation = self.last_public_activation = None
        self.library = Path(library).resolve()
        self.source_manifest = Path(source_manifest).resolve() if source_manifest else None
        self.loaded_binary_hash = Evidence.sha256(self.library)
        self.adapter_hash = Evidence.sha256(Path(__file__))
        names = {value[2] for value in StandaloneSource.CONFIG.values()}
        mapped = {Path(line.split()[-1]).resolve() for line in Path("/proc/self/maps").read_text().splitlines()
                  if line.split() and Path(line.split()[-1]).name in names}
        if mapped - {self.library}:
            raise RuntimeError("Run standalone families in separate subprocesses; C++ symbol isolation is not proven")
        torch.ops.load_library(str(self.library))
        self.namespace = StandaloneSource.CONFIG[precision][1]
        self.ops = getattr(torch.ops, self.namespace)
        self.handles = {}
        self.outputs = {}
        self.prepared = set()
        self.setup_seconds = {}

    def _allocate(self, operation):
        import torch

        if operation == "forward":
            return torch.empty_like(self.case.grad3)
        if operation == "grad_x":
            return torch.empty_like(self.case.x3)
        if operation == "grad_w":
            return torch.empty_like(self.case.w3)
        if operation == "grad_x_w":
            return torch.empty_like(self.case.x3), torch.empty_like(self.case.w3)
        raise ValueError(operation)

    @staticmethod
    def pack_activation(x, dims):
        """Real BIU -> path/batch/component/channel packing, never a bare reshape."""
        import torch

        if x.ndim != 3 or x.shape[1] != sum(dims):
            raise ValueError("Activation geometry differs from path dimensions")
        pieces, offset = [], 0
        for dim in dims:
            pieces.append(x[:, offset:offset + dim, :].contiguous().reshape(-1))
            offset += dim
        return torch.cat(pieces).reshape_as(x)

    def activation_input(self):
        return self.pack_activation(self.case.x3, self.case.dims) if self.pack_pbu else self.case.x3

    def validate_operation(self, operation):
        if operation not in ("forward", "grad_x", "grad_w", "grad_x_w"):
            raise ValueError(operation)
        if operation != "forward" and self.activation_layout != "biu":
            raise ValueError("PBU is a forward input-layout contract, not a backward layout alias")
        if operation != "forward" and self.precision in ("tf32", "fp32"):
            expected = {"grad_x": 1, "grad_w": 2, "grad_x_w": 3}[operation]
            if self.bimpl != expected:
                raise ValueError(f"{self.precision} {operation} requires the matching standalone bimpl={expected}")

    def create_handle(self, operation, activation=None):
        """Existing native constructor only; caller owns preparation and destruction."""
        self.validate_operation(operation)
        case, ops = self.case, self.ops
        if operation == "forward":
            if activation is None:
                raise ValueError("Forward constructor requires an explicitly owned activation tensor")
            return "destroy", ops.create(activation, case.w3, case.dims, self.impl)
        if self.precision == "tf32":
            if operation == "grad_x":
                return "dgrad_destroy", ops.dgrad_create(case.grad3, case.w3, case.dims, case.cg, self.bimpl)
            if operation == "grad_w":
                return "wgrad_destroy", ops.wgrad_create(case.x3, case.grad3, case.dims, case.cg, self.bimpl, 0)
            return "backward_destroy", ops.backward_create(case.x3, case.w3, case.grad3, case.dims, case.cg, self.bimpl, 0)
        if self.precision == "fp32":
            return "backward_destroy", ops.backward_create(case.x3, case.w3, case.dims, case.cg, self.bimpl)
        if operation == "grad_x":
            return "dgrad_destroy", ops.dgrad_create(case.x3, case.w3, case.dims, case.cg)
        c = self.resolved_api_controls
        if operation == "grad_w":
            return "wgrad_destroy", ops.wgrad_create(case.x3, case.dims, case.u, case.cg,
                c["wgrad_tile"], c["wgrad_split_k"], c["wgrad_kernel"], c["wgrad_transpose_mode"])
        return "bwd_destroy", ops.bwd_create(case.x3, case.w3, case.dims, case.cg,
            c["wgrad_tile"], c["wgrad_split_k"], c["wgrad_kernel"], c["wgrad_transpose_mode"])

    def prepare(self, operation, boundary):
        import torch

        if (operation, boundary) in self.prepared:
            return
        self.validate_operation(operation)
        start = time.perf_counter()
        case, ops = self.case, self.ops
        if operation == "forward":
            if boundary not in self.BOUNDARIES:
                raise ValueError(boundary)
            if operation not in self.handles and boundary != "forward_allocating":
                self.activation = self.activation_input()
                self.handles[operation] = self.create_handle(operation, self.activation)
                handle = self.handles[operation][1]
                ops.prepare_weight(handle, case.w3, case.cg)
                ops.prepare_act(handle, self.activation)
        elif boundary != "run_with_activation_prep":
            raise ValueError("Backward API exposes a complete run, not a separate pure-GEMM launch")
        elif operation not in self.handles:
            self.handles[operation] = self.create_handle(operation)
        self.outputs[operation] = self._allocate(operation)
        if self.precision == "fp32" and operation in ("grad_x", "grad_w"):
            self.outputs[operation + "_unused"] = torch.empty_like(case.w3 if operation == "grad_x" else case.x3)
        torch.cuda.synchronize()
        self.setup_seconds[f"{operation}/{boundary}"] = time.perf_counter() - start
        self.prepared.add((operation, boundary))

    def call(self, operation, boundary, output=None):
        if (operation, boundary) not in self.prepared:
            raise RuntimeError("prepare must happen outside the measurement window")
        case, ops = self.case, self.ops
        output = self.outputs[operation] if output is None else output
        if operation == "forward":
            if boundary == "forward_allocating":
                self.last_public_activation = self.activation_input()
                return ops.forward(self.last_public_activation, case.w3, case.dims, case.cg, self.impl)
            handle = self.handles[operation][1]
            if boundary == "run_with_activation_prep":
                self.activation = self.activation_input()
                ops.run_act(handle, self.activation, output)
            elif boundary == "prepared_gemm":
                ops.launch(handle, output)
            return output
        return self.run_backward_handle(operation, self.handles[operation][1], output)

    def run_backward_handle(self, operation, handle, output):
        """Run an explicitly owned backward handle, using the original entrypoint."""
        if operation == "forward":
            raise ValueError("Forward has separate run/launch entrypoints")
        case, ops = self.case, self.ops
        if self.precision == "fp32":
            dx, dw = output if operation == "grad_x_w" else (
                (output, self.outputs[operation + "_unused"]) if operation == "grad_x"
                else (self.outputs[operation + "_unused"], output))
            ops.backward_run(handle, case.x3, case.grad3, dx, dw)
        elif operation == "grad_x":
            ops.dgrad_run(handle, case.grad3, output)
        elif operation == "grad_w":
            ops.wgrad_run(handle, case.x3, case.grad3, output)
        elif self.precision == "tf32":
            ops.backward_run(handle, case.x3, case.grad3, *output)
        else:
            ops.bwd_run(handle, case.x3, case.grad3, *output)
        return output

    def new_output(self, operation):
        return self._allocate(operation)

    @staticmethod
    def common_output(operation, boundary, value):
        return value

    def identity(self):
        maps = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                       if ".so" in line and ("gmm" in line or "eqlinear" in line or "_cuda" in line)})
        if (str(self.library) not in maps or Evidence.sha256(self.library) != self.loaded_binary_hash
                or Evidence.sha256(Path(__file__)) != self.adapter_hash):
            raise RuntimeError("Standalone DSO mapping, file identity or adapter source changed")
        result = {"family": "standalone_" + self.precision, "impl": self.impl, "bimpl": self.bimpl,
                  "namespace": self.namespace, "binary": str(self.library), "binary_hash": self.loaded_binary_hash,
                  "mapped_libraries": maps, "setup_seconds": self.setup_seconds,
                  "adapter_sha256": self.adapter_hash,
                  "api_controls": self.api_controls, "wgrad_resolved_api_controls": self.resolved_api_controls,
                  "activation_layout": self.activation_layout, "pbu_pack_effective": self.pack_pbu,
                  "activation_content_hashes": {name: self.case.tensor_hash(value) for name, value in
                      (("bound", self.activation), ("allocating", self.last_public_activation)) if value is not None},
                  "runtime_controls": {key: value for key, value in os.environ.items() if key.startswith("EQLIN_")}}
        if self.source_manifest:
            manifest = json.loads(self.source_manifest.read_text())
            result.update(source_manifest=str(self.source_manifest), source_hash=manifest["source_hash"],
                          headers_hash=manifest["headers_hash"])
        return result

    def close(self):
        import torch

        torch.cuda.synchronize()
        for destroy, handle in self.handles.values():
            getattr(self.ops, destroy)(handle)
        self.handles.clear()


class StandaloneBuild:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--precision", choices=tuple(StandaloneSource.CONFIG), required=True)
        parser.add_argument("--attempt", default="01")
        args = parser.parse_args()
        result = StandaloneSource(Session(args.session), args.precision).compile(args.attempt)
        return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(StandaloneBuild.main())
