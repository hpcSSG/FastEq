"""Isolated source-preserving builds of archived handwritten CUDA implementations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from common import Evidence
from producer_closure import ProducerClosure
from session import Session


class HandwrittenSource:
    FP64 = ("gmm_p1_v2", "gmm_v0", "gmm_v1", "gmm_v2", "gmm_v2_multi_path", "gmm_v3", "gmm_v4",
            "gmm_v4_multi_path", "gmm_v5_multi_path", "gmm_v6_producer_consumer")
    REGISTRATION = re.compile(r"\bTORCH_LIBRARY\s*\(\s*([A-Za-z_]\w*)\s*,")

    def __init__(self, session, kind, name):
        if Path(name).name != name:
            raise ValueError("Use a simple build attempt name")
        self.session, self.kind = session, kind
        self.archive = Path(session.paths["OPT_ROOT"])
        self.root = session.root / "build/handwritten" / name
        if kind == "tf32_v7":
            folder = self.archive / "gmm_tf32_src/cuda_impl"
            self.sources = [folder / path for path in ("torch_bind_v4.cpp", "kernels/gmm_tf32_v7.cu",
                                                       "kernels/gmm_bwd_tf32_v8.cu", "kernels/host_cuda_helper.cu")]
        elif kind.startswith("fp64_") and kind.removeprefix("fp64_") in self.FP64:
            self.sources = [self.archive / "gmm_src/cuda_impl" / (kind.removeprefix("fp64_") + ".cu")]
        else:
            raise ValueError("Not a reviewed/permitted handwritten build selection")

    def stage(self):
        self.root.mkdir(parents=True, exist_ok=False)
        target = self.root / "source"
        target.mkdir()
        pending, copied, unresolved = list(self.sources), {}, []
        while pending:
            path = pending.pop().resolve()
            if not path.is_relative_to(self.archive.resolve()):
                raise ValueError("Quoted dependency escaped archived sources")
            key = str(path.relative_to(self.archive.resolve()))
            if key in copied:
                continue
            output = target / key
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            copied[key] = Evidence.sha256(output)
            for include in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', output.read_text(), re.M):
                dependency = (path.parent / include).resolve()
                if dependency.is_file():
                    pending.append(dependency)
                else:
                    unresolved.append({"source": key, "quoted_include": include,
                                       "status": "EXTERNAL_INCLUDE_SEARCH_OR_MISSING; compiler decides, never hardware unsupported"})
        source_hash = Evidence.fingerprint(copied)
        namespace = "h20_hw_" + self.kind + "_" + source_hash[:12]
        patches, registrations = [], 0
        for key in copied:
            path = target / key
            before = path.read_text()
            after, count = self.REGISTRATION.subn("TORCH_LIBRARY(" + namespace + ",", before)
            registrations += count
            if count:
                path.write_text(after)
                if self.REGISTRATION.sub("TORCH_LIBRARY(NAMESPACE,", before) != self.REGISTRATION.sub("TORCH_LIBRARY(NAMESPACE,", after):
                    raise RuntimeError("A change beyond registration namespace occurred")
                patches.append({"file": key, "count": count, "reason": "Unique test-only TORCH_LIBRARY namespace; operator math preserved",
                                "original_sha256": copied[key], "staged_sha256": Evidence.sha256(path)})
        if registrations != 1:
            raise ValueError("Expected one existing Torch operator registration; source-only binding needs a separate reviewed adapter")
        manifest = {"kind": self.kind, "namespace": namespace, "source_hash": source_hash,
            "archive": str(self.archive), "original_files": copied, "registration_patches": patches,
            "staged_files": {key: Evidence.sha256(target / key) for key in copied},
            "sources": [str(target / path.relative_to(self.archive)) for path in self.sources],
            "unresolved_quoted_includes": unresolved,
            "scope": "Fresh H20 source build; no claim of matching an unknown historical producer/build configuration"}
        if self.kind == "tf32_v7":
            policy = self.session.root / "coverage/handwritten_source_math_audit02/audit.json"
            reviewed = json.loads(policy.read_text())
            required = {"gmm_tf32_v7.cu", "gmm_bwd_tf32_v8.cu"}
            accepted = {Path(row["source"]).name for row in reviewed["results"]
                        if row["classification"] == "TF32_THREE_PRODUCT" and Evidence.sha256(Path(row["source"])) == row["source_sha256"]}
            if not required <= accepted:
                raise ValueError("TF32 three-product source review missing or changed")
            manifest["math_policy"] = {"path": str(policy), "sha256": Evidence.sha256(policy), "required": sorted(required)}
        Evidence.write(self.root / "source.json", manifest)
        return manifest

    def build(self):
        import torch
        from torch.utils.cpp_extension import load

        manifest = self.stage()
        output = self.root / "binary"
        output.mkdir()
        cxx = ["-O3", "-fvisibility=hidden"]
        cuda = ["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a", "-Xcompiler=-fvisibility=hidden"]
        linker = ["-lcuda", "-Wl,-soname," + manifest["namespace"] + ".so"]
        result = {"status": "BUILDING", "source_manifest": str(self.root / "source.json"),
            "source_manifest_sha256": Evidence.sha256(self.root / "source.json"), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
            "flags": {"cxx": cxx, "cuda": cuda, "linker": linker}, "CUDA_VISIBLE_DEVICES": "CPU_BUILD_NO_KERNEL_EXECUTION"}
        Evidence.write(self.root / "build.json", result)
        try:
            load(name=manifest["namespace"], sources=manifest["sources"], extra_cflags=cxx, extra_cuda_cflags=cuda,
                 extra_ldflags=linker, build_directory=str(output), with_cuda=True, is_python_module=False, verbose=True)
            binary = output / (manifest["namespace"] + ".so")
            if not binary.is_file():
                raise RuntimeError("Expected freshly linked DSO missing")
            for key, digest in manifest["staged_files"].items():
                if Evidence.sha256(self.root / "source" / key) != digest:
                    raise ValueError("Staged source changed during build")
            elf = subprocess.check_output(["cuobjdump", "--list-elf", str(binary)], text=True, timeout=30)
            (self.root / "elf.txt").write_text(elf)
            architectures = sorted(set(re.findall(r"sm_\d+a?", elf)))
            if architectures != ["sm_90a"]:
                raise ValueError("Fresh handwritten cubins are not exclusively sm_90a: " + str(architectures))
            dynamic = subprocess.check_output(["readelf", "-d", str(binary)], text=True, timeout=30)
            (self.root / "dynamic.txt").write_text(dynamic)
            if manifest["namespace"] + ".so" not in dynamic:
                raise ValueError("Unique handwritten SONAME missing")
            result.update(status="PASS", binary={"path": str(binary), "sha256": Evidence.sha256(binary)},
                          architectures=architectures, operator_namespace=manifest["namespace"],
                          runtime_backend_verified=False, note="Build/registration only; GPU capability and numerical probes still required")
        except BaseException as error:
            result.update(status="BUILD_FAILED", error=str(error))
            raise
        finally:
            Evidence.write(self.root / "build.json", result)
        return 0

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--kind", choices=("tf32_v7",) + tuple("fp64_" + value for value in cls.FP64), required=True)
        parser.add_argument("--name", required=True)
        parser.add_argument("--freeze", type=Path, help="Freeze this CPU build producer and stop")
        args = parser.parse_args()
        if args.freeze:
            args.freeze.mkdir(parents=True, exist_ok=False)
            Evidence.write(args.freeze / "producer.json", ProducerClosure(Path(__file__).parent, args.freeze).freeze(["handwritten.py"]))
            return 0
        return cls(Session(args.session), args.kind, args.name).build()


if __name__ == "__main__":
    raise SystemExit(HandwrittenSource.main())
