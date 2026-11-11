"""Test-only selectors linked to exact, locally built current H20 native objects."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from common import Evidence
from session import Session
from standalone_probe import StandaloneProbe
from reference_store import ReferenceStore


class NativeBridge:
    """Own the source/object/ABI contract; never load the production dispatcher."""
    CONFIG = {"tf32": ("eqlin", "Impl", "float", "at::kFloat", (0, 3, 4, 5, 6, 7, 8)),
              "fp32": ("eqlin_fp32", "Impl", "float", "at::kFloat", tuple(range(4))),
              "fp64": ("eqlin", "F64Impl", "double", "at::kDouble", tuple(range(8)))}

    def __init__(self, session, precision, build_id="bridge01"):
        if precision not in self.CONFIG or not re.fullmatch(r"[a-zA-Z0-9_]+", build_id):
            raise ValueError("Unknown precision or invalid fresh bridge identity")
        self.session, self.precision, self.build_id = session, precision, build_id
        self.inventory_path = session.root / "manifests/native_selector_object_inventory02.json"
        self.inventory = json.loads(self.inventory_path.read_text())
        self.group = self.inventory["groups"][precision]
        self.root = session.root / "build/native_bridge" / build_id / precision
        self.namespace = "h20_native_" + build_id + "_" + precision
        self.library = self.root / ("lib" + self.namespace + ".so")
        self.manifest_path = session.root / "manifests" / (self.namespace + ".json")

    @classmethod
    def check_impl(cls, precision, impl):
        if type(impl) is not int or impl not in cls.CONFIG[precision][4]:
            raise ValueError("Excluded or unknown native selector: " + precision + "/" + str(impl))

    @staticmethod
    def verify_files(files):
        for filename, expected in files.items():
            if Evidence.sha256(Path(filename)) != expected:
                raise ValueError("Native source/object/ABI drift: " + filename)

    def object_contract(self):
        inv, group = self.inventory, self.group
        current_path = Path(inv["current_build"])
        if Evidence.sha256(current_path) != inv["current_build_sha256"]:
            raise ValueError("Current fresh-build receipt differs from object inventory")
        current = json.loads(current_path.read_text())
        source_root = Path(self.session.paths["FASTEQ_ROOT"])
        source_files = {str(source_root / name): sha for name, sha in current["source"]["files"].items()}
        self.verify_files(source_files)
        if (current["source"]["commit"] != inv["source_commit"] or
                tuple(group["allowed_impls"]) != self.CONFIG[self.precision][4] or
                group["enum_class"] != self.CONFIG[self.precision][1]):
            raise ValueError("Object source, allowed enum or scalar contract differs")
        self.verify_files({group["header"]: group["header_sha256"]})
        graph = Path(inv["build_graph"])
        dependencies, commands, native_files = {}, {}, {}
        for row in group["objects"]:
            obj, src = Path(row["object"]), Path(row["source"])
            if not obj.is_relative_to(graph) or not obj.is_relative_to(self.session.root):
                raise ValueError("Object is outside the local fresh H20 build")
            native_files.update({str(obj): row["object_sha256"], str(src): row["source_sha256"]})
            target = str(obj.relative_to(graph))
            command = subprocess.check_output(["ninja", "-C", str(graph), "-t", "commands", target],
                                              text=True, timeout=20).strip().splitlines()[-1]
            if command != row["actual_compile_command"]:
                raise ValueError("Ninja compile command differs: " + target)
            if src.suffix == ".cu" and not ("compute_90a" in command and "sm_90a" in command):
                raise ValueError("Native CUDA object is not explicitly sm90a")
            commands[target] = command
            deps = subprocess.check_output(["ninja", "-C", str(graph), "-t", "deps", target],
                                           text=True, timeout=20)
            if "(VALID)" not in deps.splitlines()[0]:
                raise ValueError("Missing/invalid Ninja source dependency receipt: " + target)
            paths = [Path(line.strip()).resolve() for line in deps.splitlines()[1:] if line.strip()]
            if src.resolve() not in paths:
                raise ValueError("Compile dependency closure omits its source")
            for dep in paths:
                if dep.stat().st_mtime_ns > obj.stat().st_mtime_ns:
                    raise ValueError("Compile dependency newer than frozen object: " + str(dep))
                if str(dep) not in dependencies:
                    dependencies[str(dep)] = Evidence.sha256(dep)
        self.verify_files(native_files)
        return {"inventory": str(self.inventory_path), "inventory_sha256": Evidence.sha256(self.inventory_path),
                "current_build": str(current_path), "current_build_sha256": inv["current_build_sha256"],
                "source_commit": inv["source_commit"], "source_files": source_files,
                "native_files": native_files, "dependency_files": dependencies,
                "native_compile_commands": commands, "native_objects": group["objects"],
                "enum": group["enum"], "allowed_impls": group["allowed_impls"],
                "prohibited_impls": group["prohibited_impls"], "header": group["header"],
                "header_sha256": group["header_sha256"], "build_graph": str(graph),
                "build_ninja_sha256": Evidence.sha256(graph / "build.ninja"),
                "cmake_cache_sha256": Evidence.sha256(graph / "CMakeCache.txt")}

    def build(self):
        import torch
        from torch.utils.cpp_extension import include_paths

        if self.manifest_path.exists() or self.root.exists():
            raise FileExistsError("Bridge builds require a fresh build-id; old evidence is immutable")
        contract = self.object_contract()
        abi = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
               "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI), "torch_origin": torch.__file__}
        if any(abi[key] != self.inventory[key] for key in ("torch", "torch_cuda", "cxx11_abi")):
            raise ValueError("Torch/CUDA/CXX ABI differs from local object inventory")
        self.root.mkdir(parents=True)
        template = Path(__file__).with_name("native_bridge.cpp.in")
        backend, enum, scalar, dtype, allowed = self.CONFIG[self.precision]
        text = template.read_text()
        for token, value in {"BACKEND": backend, "ENUM": enum, "SCALAR": scalar, "DTYPE": dtype,
                             "ALLOWED": " || ".join("impl == " + str(i) for i in allowed),
                             "TORCH_NAMESPACE": self.namespace}.items():
            text = text.replace("@" + token + "@", value)
        if "@" in text:
            raise ValueError("Unresolved binding template token")
        source, obj = self.root / "binding.cpp", self.root / "binding.o"
        source.write_text(text)
        version_script = self.root / "local_symbols.map"
        version_script.write_text("{ local: *; };\n")
        torch_lib = Path(torch.__file__).parent / "lib"
        cuda_root = Path("/usr/local/cuda")
        compile_argv = ["/usr/bin/c++", "-std=c++17", "-O2", "-fPIC", "-fvisibility=hidden",
                        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(abi["cxx11_abi"])),
                        *["-I" + p for p in include_paths()], "-I" + str(cuda_root / "include"),
                        "-I" + str(Path(contract["header"]).parent), "-MMD", "-MF", str(self.root / "binding.d"),
                        "-c", str(source), "-o", str(obj)]
        link_argv = ["/usr/bin/c++", "-shared", "-Wl,-z,defs", "-Wl,-Bsymbolic",
                     "-Wl,-soname," + self.library.name, "-Wl,--version-script=" + str(version_script),
                     str(obj), *[row["object"] for row in self.group["objects"]],
                     "-L" + str(torch_lib), "-L" + str(cuda_root / "lib64"),
                     "-Wl,-rpath," + str(torch_lib), "-Wl,-rpath," + str(cuda_root / "lib64"),
                     "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda", "-lcudart", "-lcuda",
                     "-o", str(self.library)]
        receipt = {"classification": "TEST_ONLY_BINDING_TO_CURRENT_H20_OBJECTS", "status": "BUILDING",
                   "precision": self.precision, "namespace": self.namespace, "binary": str(self.library),
                   "native_mathematics_modified": False, "CUDA_recompiled": False,
                   "ABI": abi, "contract": contract, "compile_argv": compile_argv, "link_argv": link_argv,
                   "bridge_producers": {str(Path(__file__)): Evidence.sha256(Path(__file__)),
                                        str(template): Evidence.sha256(template)},
                   "generated_source": str(source), "generated_source_sha256": Evidence.sha256(source),
                   "version_script": str(version_script), "version_script_sha256": Evidence.sha256(version_script)}
        Evidence.write(self.manifest_path, receipt)
        runner = self.session.runner()
        prefix = self.namespace
        for stage, argv in (("compile", compile_argv), ("link", link_argv),
                            ("elf", ["readelf", "-d", str(self.library)]),
                            ("exports", ["nm", "-D", "--defined-only", str(self.library)]),
                            ("cuda_arch", ["cuobjdump", "--list-elf", str(self.library)])):
            terminal = runner.run(prefix + "_" + stage, argv, 120 if stage in ("compile", "link") else 30,
                                  self.root, {"CUDA_VISIBLE_DEVICES": ""})
            receipt[stage + "_receipt"] = terminal
            if terminal["status"] != "PASS":
                receipt["status"] = "BUILD_ERROR"
                Evidence.write(self.manifest_path, receipt)
                return 1
        self.verify_files(contract["native_files"])
        self.verify_files(contract["dependency_files"])
        receipt.update(status="PASS", binary_sha256=Evidence.sha256(self.library), binding_object_sha256=Evidence.sha256(obj))
        Evidence.write(self.manifest_path, receipt)
        return 0

    def verify(self):
        import torch

        receipt = json.loads(self.manifest_path.read_text())
        if receipt["status"] != "PASS" or receipt["namespace"] != self.namespace:
            raise ValueError("No completed matching test-only bridge build")
        if (receipt["ABI"]["torch"] != torch.__version__ or receipt["ABI"]["torch_cuda"] != torch.version.cuda or
                receipt["ABI"]["cxx11_abi"] != bool(torch._C._GLIBCXX_USE_CXX11_ABI)):
            raise ValueError("Runtime Torch/CUDA/CXX ABI differs")
        contract = receipt["contract"]
        for group in (contract["source_files"], contract["native_files"], contract["dependency_files"]):
            self.verify_files(group)
        self.verify_files({str(self.inventory_path): contract["inventory_sha256"], str(self.library): receipt["binary_sha256"],
                           contract["current_build"]: contract["current_build_sha256"],
                           str(Path(contract["build_graph"]) / "build.ninja"): contract["build_ninja_sha256"],
                           str(Path(contract["build_graph"]) / "CMakeCache.txt"): contract["cmake_cache_sha256"],
                           receipt["generated_source"]: receipt["generated_source_sha256"]})
        for stage in ("elf", "exports", "cuda_arch"):
            self.verify_files({receipt[stage + "_receipt"]["stdout"]["path"]:
                               receipt[stage + "_receipt"]["stdout"]["sha256"]})
        elf = Path(receipt["elf_receipt"]["stdout"]["path"]).read_text()
        if ("Library soname: [" + self.library.name + "]" not in elf or
                any("fasteq" in name or "gmm" in name for name in re.findall(r"Shared library: \[([^]]+)\]", elf))):
            raise ValueError("Bridge SONAME or dependencies violate process isolation")
        arch = Path(receipt["cuda_arch_receipt"]["stdout"]["path"]).read_text()
        if set(re.findall(r"\.sm_([^.]+)\.cubin", arch)) != {"90a"}:
            raise ValueError("Linked native code is not exclusively sm90a")
        if Path(receipt["exports_receipt"]["stdout"]["path"]).read_text().strip():
            raise ValueError("Native CABI exports were not localized")
        return receipt


class NativeAdapter:
    BOUNDARIES = ("forward_allocating", "run_with_activation_prep", "prepared_gemm")

    def __init__(self, case, bridge, impl, verified_receipt=None):
        import torch

        NativeBridge.check_impl(bridge.precision, impl)
        if case.point.precision != bridge.precision:
            raise ValueError("Native precision differs from canonical fixture")
        self.case, self.bridge, self.impl = case, bridge, impl
        self.receipt = bridge.verify() if verified_receipt is None else verified_receipt
        if self.receipt["binary"] != str(bridge.library) or self.receipt["status"] != "PASS":
            raise ValueError("Shared source guard does not belong to this native bridge")
        self.manifest_hash = Evidence.sha256(bridge.manifest_path)
        self.assert_isolation()
        torch.ops.load_library(str(bridge.library))
        self.ops = getattr(torch.ops, bridge.namespace)
        self.ops.check_impl(impl)
        # Test the independent C++ x1 guard without constructing a Runner.
        if bridge.precision == "tf32":
            for excluded in (1, 2):
                try:
                    self.ops.check_impl(excluded)
                except RuntimeError as error:
                    if "Excluded or unknown native implementation" not in str(error):
                        raise
                else:
                    raise ValueError("C++ TF32x1 entry guard failed")
        self.handle, self.prepared = None, set()
        self.identity()

    def assert_isolation(self):
        maps = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                       if ".so" in line and ("eqlinear" in line or "gmm" in line or "h20_native" in line)})
        if any(path != str(self.bridge.library) for path in maps):
            raise ValueError("Another eqlinear/standalone/native DSO is already mapped: " + str(maps))
        if any(name == "fasteq" or name.startswith("fasteq.") for name in sys.modules):
            raise ValueError("Production FastEq Python module imported in isolated bridge process")
        return maps

    def prepare(self, operation, boundary):
        import torch

        if operation != "forward" or boundary not in self.BOUNDARIES:
            raise ValueError("Current native selector bridge exposes forward only")
        if (operation, boundary) in self.prepared:
            return
        if boundary != "forward_allocating":
            if self.handle is None:
                self.handle = self.ops.create(self.case.x3, self.case.w3, self.case.dims, self.case.cg, self.impl)
            self.ops.prepare_act(self.handle, self.case.x3)
        # Drain all setup before the first independent repeated call.
        torch.cuda.synchronize()
        self.prepared.add((operation, boundary))

    def new_output(self, operation):
        import torch

        if operation != "forward":
            raise ValueError(operation)
        return torch.empty_like(self.case.grad3)

    def call(self, operation, boundary, output=None):
        if (operation, boundary) not in self.prepared:
            raise ValueError("Prepare native call outside measurement window")
        if boundary == "forward_allocating":
            value = self.ops.forward(self.case.x3, self.case.w3, self.case.dims, self.case.cg, self.impl)
        else:
            value = self.new_output(operation) if output is None else output
            if boundary == "run_with_activation_prep":
                self.ops.run_act(self.handle, self.case.x3, value)
            else:
                self.ops.launch(self.handle, value)
        return value.reshape_as(self.case.grad)

    @staticmethod
    def common_output(operation, boundary, value):
        return value

    def identity(self):
        maps = self.assert_isolation()
        if self.handle is not None and str(self.bridge.library) not in maps:
            raise ValueError("Expected native bridge DSO is absent from actual maps")
        if (Evidence.sha256(self.bridge.library) != self.receipt["binary_sha256"] or
                Evidence.sha256(self.bridge.manifest_path) != self.manifest_hash):
            raise ValueError("Native runtime file/receipt drift")
        return {"family": "current_native_selector", "precision": self.bridge.precision, "impl": self.impl,
                "impl_name": self.bridge.group["enum"][str(self.impl)], "namespace": self.bridge.namespace,
                "binary": str(self.bridge.library), "binary_sha256": self.receipt["binary_sha256"],
                "manifest": str(self.bridge.manifest_path), "manifest_sha256": self.manifest_hash,
                "mapped_libraries": maps, "native_objects_sha256": Evidence.fingerprint(self.receipt["contract"]["native_files"]),
                "source_commit": self.receipt["contract"]["source_commit"], "CUDA_architecture": "sm90a",
                "production_DSO_loaded": False, "native_math_modified": False,
                "TF32x1_cpp_entry_rejections_verified": self.bridge.precision == "tf32",
                "runtime_controls": {key: value for key, value in os.environ.items() if key.startswith(("EQLIN_", "FASTEQ_EQLINEAR_"))}}

    def close(self):
        if self.handle is not None:
            self.ops.destroy(self.handle)
            self.handle = None


class NativeProbe(StandaloneProbe):
    """Inject isolated native adapters into the one shared numerical gate loop."""

    def __init__(self, args, shared=None):
        import torch
        from cases import Case

        NativeBridge.check_impl(args.precision, args.impl)
        if any(key.startswith(("EQLIN_", "FASTEQ_EQLINEAR_")) for key in os.environ):
            raise ValueError("Unspecified native runtime control inherited")
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
            raise ValueError("TF32 comparator override is forbidden")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        session = Session(args.session)
        self.shared = shared
        self.bridge = (shared.bridge if shared else NativeBridge(session, args.precision, args.build_id))
        self.refs = ReferenceStore(session.root / "reference_store")
        super().__init__(args, reference_factory=self.references, adapter_factory=self.make_adapter,
                         case_factory=shared.case_factory if shared else Case)
        for name in ("native_bridge.py", "native_bridge.cpp.in", "reference_store.py", "standalone_probe.py", "current_matrix.py"):
            path = Path(__file__).with_name(name)
            self.producer_files[str(path)] = Evidence.sha256(path)
            (self.output / "producer_sources" / name).write_bytes(path.read_bytes())
        self.result["producer_files_start"] = dict(self.producer_files)

    def make_adapter(self, case, source):
        return NativeAdapter(case, self.bridge, self.args.impl,
                             self.shared.verified_receipt if self.shared else None)

    def references(self, case):
        reference = self.shared.reference if self.shared else self.refs.get(case)
        self.result["reference_store"] = reference.store_receipt
        return reference

    def observe_result(self, row, output, repeated):
        if row["passed"]:
            return
        import torch
        root = self.output / "failed_outputs"
        root.mkdir(exist_ok=True)
        row["failed_outputs"] = {}
        for name, value in (("first", output), ("repeat", repeated)):
            target = root / (row["boundary"] + "_" + name + ".pt")
            torch.save(value.detach().cpu(), target)
            row["failed_outputs"][name] = {"path": str(target), "sha256": Evidence.sha256(target),
                                           "tensor_hash": self.adapter.case.tensor_hash(value)}

    def run(self):
        result = super().run()
        if self.shared is None:
            self.bridge.verify()
        self.result["source_unchanged"] = True
        self.flush()
        return result


def probe(args):
    return NativeProbe(args).run()


def matrix(args):
    from cases import points
    from current_matrix import AccuracyChunk

    NativeBridge.check_impl(args.precision, args.impl)
    bridge = NativeBridge(Session(args.session), args.precision, args.build_id)
    bridge.verify()

    class Chunk(AccuracyChunk):
        FILES = ("native_bridge.py", "native_bridge.cpp.in", "standalone_probe.py", "standalone.py",
                 "current_matrix.py", "reference_store.py", "cases.py", "common.py", "protocol.py", "session.py")

        def family_name(self):
            return f"native_{args.precision}_impl{args.impl}"

        def selected_points(self):
            return [point for point in points() if point.config == args.config and point.precision == args.precision]

        def probe_argv(self, producer, point, output):
            return [self.session.paths["FASTEQ_PY"], "-B", str(producer / "native_bridge.py"), "probe",
                    "--session", str(self.session.root), "--precision", args.precision, "--impl", str(args.impl),
                    "--point", point.key.rsplit("/", 1)[0], "--output", str(output), "--build-id", args.build_id]

        def accept_terminal(self, result, terminal):
            return (terminal["status"] == "PASS" and result.get("status") == "PASS" or
                    terminal.get("returncode") == 1 and result.get("status") == "ACCURACY_FAIL"
                    and result.get("producer_drift") is False)

    return Chunk(args).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "probe", "matrix"))
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--precision", choices=tuple(NativeBridge.CONFIG), required=True)
    parser.add_argument("--build-id", default="bridge01")
    parser.add_argument("--impl", type=int, default=0)
    parser.add_argument("--point", default="SMALL_P1/512")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--attempt")
    args = parser.parse_args()
    args.operations, args.boundaries, args.bimpl = "forward", ",".join(NativeAdapter.BOUNDARIES), 0
    if args.action == "probe":
        return probe(args)
    if args.action == "matrix":
        if not args.config or not args.attempt:
            parser.error("matrix requires --config and --attempt")
        return matrix(args)
    bridge = NativeBridge(Session(args.session), args.precision, args.build_id)
    if args.action == "build":
        return bridge.build()
    bridge.verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
