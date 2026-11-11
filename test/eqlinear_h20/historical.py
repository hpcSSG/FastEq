"""Prepare and build isolated historical eqlinear DSOs using the audited P0 harness.

Only namespace/SONAME/build plumbing is adapted. Historical numerical source is
checked against Git before and after building; the current editable install stays intact.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from common import Evidence
from current import CurrentAdapter
from registry import GitHistory
from session import Session


class HistoricalBuild:
    REFS = {"p0": "229a1ce76099a57cef6e81b5e27a076531782dfe", "p14": "d171efe"}
    TOOLS = ("CMakeLists.txt.in", "source_audit.py", "load_p0.py", "verify_elf.py",
             "exports_tf32.map", "exports_fp32.map", "exports_fp64.map")

    def __init__(self, session: Session, variant: str):
        self.session, self.variant = session, variant
        self.source = session.root / "sources" / ("historical_" + variant)
        self.root = session.root / "build" / ("historical_" + variant)
        self.harness, self.build = self.root / "harness", self.root / "compiled"
        self.record = session.root / "manifests" / ("historical_" + variant + "_build.json")
        self.python = session.paths["FASTEQ_PY"]
        if self.record.exists():
            observed = json.loads(self.record.read_text())
            if observed.get("binding_format") == "historical_contract_v1":
                self.source, self.harness, self.build = (Path(observed[name]) for name in ("source", "harness", "build"))

    def prepare(self, history_repo: Path) -> None:
        if self.record.exists() or self.harness.exists():
            raise FileExistsError("Historical preparation already exists; use configure/build/verify to continue")
        history = GitHistory(history_repo, Path(self.session.paths["HANDOFF_ROOT"]) / "sources/fasteq_history.bundle")
        commit = history.resolve(self.REFS[self.variant])
        result = self.session.runner().run("historical_" + self.variant + "_worktree", [
            "git", "--git-dir=" + str(history.repository), "worktree", "add", "--detach", str(self.source), commit], 60)
        self.require(result)
        cutlass = self.source / "3rdparty/cutlass"
        if cutlass.is_dir() and not any(cutlass.iterdir()):
            cutlass.rmdir()
        if cutlass.exists():
            raise FileExistsError("Unexpected existing historical CUTLASS contents")
        cutlass.symlink_to(Path(self.session.paths["FASTEQ_ROOT"]) / "3rdparty/cutlass", target_is_directory=True)
        self.harness.mkdir(parents=True)
        template = Path(self.session.paths["HANDOFF_ROOT"]) / "scripts/fasteq/test/eqlinear_acceptance/p0_harness"
        changes, templates = [], {}
        for name in self.TOOLS:
            original = (template / name).read_text()
            adapted = original.replace("/app/work/fasteq_venv/bin/python", self.python)
            if self.variant != "p0":
                adapted = adapted.replace("229a1ce76099a57cef6e81b5e27a076531782dfe", commit)
                adapted = adapted.replace("229a1ce", commit[:7]).replace("P0", self.variant.upper()).replace("p0", self.variant)
            destination = "CMakeLists.txt" if name == "CMakeLists.txt.in" else name.replace("p0", self.variant)
            (self.harness / destination).write_text(adapted)
            templates[name] = {"original_path": str(template / name), "original_sha256": Evidence.sha256(template / name),
                               "adapted_path": str(self.harness / destination), "adapted_sha256": Evidence.sha256(self.harness / destination)}
            changes.extend(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True),
                                               fromfile="handoff/" + name, tofile="harness/" + destination))
        (self.harness / "harness_adaptations.diff").write_text("".join(changes))
        # P14 already has its own tracked recipe; P0 alone needs a build-only recipe.
        recipe_root = self.source if self.variant != "p0" else Path(self.session.paths["FASTEQ_ROOT"])
        recipe = "fasteq/cuda/src/eqlinear/CMakeLists.txt"
        before = (self.source / recipe).read_text() if (self.source / recipe).exists() else ""
        after = (recipe_root / recipe).read_text()
        (self.harness / "temporary_cmake.diff").write_text("".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile="original/" + recipe, tofile="isolated_build/" + recipe)))
        result = self.session.runner().run("historical_" + self.variant + "_source_prepare", [
            self.python, "-B", str(self.harness / "source_audit.py"), "prepare", "--" + self.variant + "-root", str(self.source),
            "--prep-root", str(recipe_root), "--output-root", str(self.harness)], 60)
        # Generalized source_audit retains historical CLI names, since only data changed.
        self.require(result)
        source_manifest = json.loads((self.harness / "source_manifest.json").read_text())
        Evidence.write(self.record, {"variant": self.variant, "commit": commit, "status": "PREPARED_NOT_BUILT",
            "source": str(self.source), "harness": str(self.harness), "build": str(self.build),
            "templates": templates, "source_manifest": source_manifest,
            "source_manifest_sha256": Evidence.sha256(self.harness / "source_manifest.json"),
            "harness_patch_sha256": Evidence.sha256(self.harness / "harness_adaptations.diff"),
            "cmake_patch_sha256": Evidence.sha256(self.harness / "temporary_cmake.diff"),
            "editable_install_changed": False, "mathematical_sources_changed": False})

    def configure(self, attempt: str) -> None:
        if not self.record.exists():
            raise FileNotFoundError("Run prepare first")
        result = self.session.runner().run("historical_" + self.variant + "_configure_" + attempt, [
            "cmake", "-S", str(self.harness), "-B", str(self.build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
            "-DPython3_EXECUTABLE=" + self.python, "-DCMAKE_CUDA_ARCHITECTURES=90a-real",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], 90)
        self.require(result)

    def compile(self, attempt: str, jobs: int) -> None:
        result = self.session.runner().run("historical_" + self.variant + "_compile_" + attempt,
            ["cmake", "--build", str(self.build), "--parallel", str(jobs)], 600)
        self.require(result)

    def verify(self, attempt: str) -> dict:
        for name, argv in (("source", [self.python, "-B", str(self.harness / "source_audit.py"), "verify-source"]),
                           ("elf", [self.python, "-B", str(self.harness / "verify_elf.py"), "--artifacts", str(self.build / "artifacts")])):
            result = self.session.runner().run("historical_" + self.variant + "_verify_" + name + "_" + attempt, argv, 60)
            self.require(result)
        commands = json.loads((self.build / "compile_commands.json").read_text())
        cuda = [row for row in commands if row["file"].endswith(".cu")]
        if not cuda or any("compute_90a" not in row["command"] or "sm_90a" not in row["command"] for row in cuda):
            raise RuntimeError("Historical CUDA compile commands do not all preserve sm_90a")
        record = json.loads(self.record.read_text())
        record.update(status="BUILT_SOURCE_ELF_ARCH_VERIFIED", binaries=[
            {"path": str(path.resolve()), "sha256": Evidence.sha256(path), "bytes": path.stat().st_size}
            for path in sorted((self.build / "artifacts").glob("*.so"))],
            compile_commands_sha256=Evidence.sha256(self.build / "compile_commands.json"),
            cuda_compile_commands=cuda,
            torch_abi=subprocess.check_output([self.python, "-B", "-c", "import torch; print(torch.__version__,torch.version.cuda,torch._C._GLIBCXX_USE_CXX11_ABI)"], text=True, timeout=30).strip(),
            source_after_sha256=Evidence.sha256(self.harness / "source_manifest.json"))
        Evidence.write(self.record, record)
        return record

    def probe(self, attempt: str, preferred: int | None, point: str) -> dict:
        # Archive exact producer bytes before invoking a runtime process.
        tag = "historical_" + self.variant + "_" + point.replace("/", "_") + "_" + attempt
        producer = self.session.root / "manifests/producers" / tag
        producer.mkdir(parents=True, exist_ok=False)
        for name in ("historical_probe.py", "historical.py", "historical_plan.py", "historical_native.py", "registry.py", "current.py", "current_probe.py", "cases.py", "common.py", "protocol.py", "session.py"):
            shutil.copy2(Path(__file__).parent / name, producer / name)
        Evidence.write(producer / "SHA256.json", {p.name: Evidence.sha256(p) for p in producer.iterdir()})
        uuid = self.session.idle_gpu(tag, preferred)
        output = self.session.root / "accuracy" / tag
        return self.session.runner().run(tag, [self.python, "-B", str(producer / "historical_probe.py"),
            "--session", str(self.session.root), "--variant", self.variant, "--point", point, "--output", str(output)],
            120, cwd=self.session.root, environment={"CUDA_VISIBLE_DEVICES": uuid, "PYTHONPATH": str(self.source)})

    @staticmethod
    def require(result: dict) -> None:
        if result["status"] != "PASS":
            raise RuntimeError("Historical subprocess failed: " + str(result))

    @classmethod
    def main(cls) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("action", choices=("plan", "stage", "selector-objects", "prepare", "configure", "build", "verify", "all", "probe"))
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--variant", help="p0, p14, or an exact historical_required_runtimes variant_id")
        parser.add_argument("--output", type=Path, help="Fresh dependency plan JSON for action=plan")
        parser.add_argument("--history-repo", type=Path)
        parser.add_argument("--attempt", default="01")
        parser.add_argument("--jobs", type=int, choices=(1, 2, 4), default=2)
        parser.add_argument("--gpu", type=int)
        parser.add_argument("--point", default="SMALL_P1/512/tf32")
        args = parser.parse_args()
        if args.action == "selector-objects":
            if args.output is None:
                parser.error("selector-objects requires --output")
            from historical_plan import NativeSelectorObjects
            result = NativeSelectorObjects.capture(Session(args.session), args.output)
            print(json.dumps({"allowed_impl_count": result["allowed_impl_count"], "bridge_binding_implemented": False}))
            return 0
        if args.action in ("plan", "stage"):
            if args.history_repo is None or (args.action == "plan" and args.output is None):
                parser.error("plan/stage require --history-repo; plan also requires --output")
            from historical_plan import HistoricalExecutionPlan, HistoricalSourceStage
            session = Session(args.session)
            if args.action == "plan":
                print(json.dumps(HistoricalExecutionPlan(session, args.history_repo).run(args.output)["summary"]))
            else:
                if args.variant is None:
                    parser.error("stage requires --variant")
                print(json.dumps(HistoricalSourceStage(session, args.history_repo).stage(args.variant)))
            return 0
        if args.variant is None:
            parser.error("build/probe actions require --variant")
        if args.variant in cls.REFS:
            build_class = cls
        else:
            from historical_native import ContractHistoricalBuild
            build_class = ContractHistoricalBuild
        build = build_class(Session(args.session), args.variant)
        if args.action in ("prepare", "all"):
            if args.history_repo is None:
                parser.error("--history-repo required for prepare/all")
            build.prepare(args.history_repo)
        if args.action in ("configure", "all"):
            build.configure(args.attempt)
        if args.action in ("build", "all"):
            build.compile(args.attempt, args.jobs)
        if args.action in ("verify", "all"):
            build.verify(args.attempt)
        if args.action == "probe":
            return 0 if build.probe(args.attempt, args.gpu, args.point)["status"] == "PASS" else 1
        return 0


class HistoricalAdapter(CurrentAdapter):
    """Reuse the shared handle/public call implementation with an audited old module."""

    def __init__(self, case, session: Session, variant: str):
        import torch

        if any(name == "fasteq" or name.startswith("fasteq.") for name in sys.modules):
            raise RuntimeError("Historical probes must not import the current editable package")
        self.case, self.session, self.variant = case, session, variant
        build = HistoricalBuild(session, variant)
        self.manifest = json.loads(build.record.read_text())
        if self.manifest["status"] != "BUILT_SOURCE_ELF_ARCH_VERIFIED":
            raise RuntimeError("Historical source/ELF/architecture verification is incomplete")
        if self.manifest.get("precision", case.point.precision) != case.point.precision:
            raise RuntimeError("A source/precision contract cannot run a different input precision")
        # All inherited tuning knobs are removed in this dedicated runtime process.
        # Only the exact existing candidate controls declared by the plan are applied.
        for name in list(os.environ):
            if name.startswith(("EQLIN_", "FASTEQ_EQLINEAR_")):
                os.environ.pop(name)
        for name, value in self.manifest.get("runtime_controls", {}).items():
            os.environ[name] = str(value)
        self.audit_path = build.harness / "source_manifest.json"
        if Evidence.sha256(self.audit_path) != self.manifest["source_manifest_sha256"]:
            raise RuntimeError("Historical source manifest changed since build")
        audit = json.loads(self.audit_path.read_text())
        recipe = build.source / "fasteq/cuda/src/eqlinear/CMakeLists.txt"
        if Evidence.sha256(recipe) != audit["temporary_prep_cmake_sha256"]:
            raise RuntimeError("Historical build-only recipe changed since build")
        runtime_abi = f"{torch.__version__} {torch.version.cuda} {torch._C._GLIBCXX_USE_CXX11_ABI}"
        if runtime_abi != self.manifest["torch_abi"]:
            raise RuntimeError("Historical Torch/CUDA/C++ ABI differs from the compiled manifest")
        for name, digest in audit["source_sha256"].items():
            if Evidence.sha256(build.source / name) != digest:
                raise RuntimeError("Historical mathematical source changed since build: " + name)
        wrapper_name = self.manifest.get("wrapper_name", "eqlinear_" + variant + ".py")
        self.wrapper = build.harness / "generated" / wrapper_name
        if Evidence.sha256(self.wrapper) != audit["generated_sha256"][wrapper_name]:
            raise RuntimeError("Historical generated wrapper fingerprint mismatch")
        dispatcher = Path(self.manifest.get("dispatcher_path", str(build.build / "artifacts" / ("libeqlinear_" + variant + "_dispatch.so"))))
        for binary in self.manifest["binaries"]:
            if Evidence.sha256(Path(binary["path"])) != binary["sha256"]:
                raise RuntimeError("Historical DSO fingerprint changed")
        torch.ops.load_library(str(dispatcher))
        mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if ".so" in line}
        self.mapped = sorted(path for path in mapped if "eqlinear" in path)
        for binary in self.manifest["binaries"]:
            if binary["path"] not in mapped:
                raise RuntimeError("Expected historical binary is not mapped: " + binary["path"])
        forbidden = str(Path(session.paths["FASTEQ_ROOT"]) / "fasteq/cuda")
        permitted = {row["path"] for row in self.manifest["binaries"]}
        if any(path.startswith(forbidden) and path not in permitted for path in mapped):
            raise RuntimeError("Current editable DSOs leaked into a historical process")
        module_name = "_h20_historical_" + variant + "_" + self.manifest["commit"][:12]
        spec = importlib.util.spec_from_file_location(module_name, self.wrapper)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = self.module
        spec.loader.exec_module(self.module)
        suffix = self.manifest.get("backend_namespace", variant)
        self.namespace_name = "eqlinear_" + suffix if suffix else "eqlinear"
        self.namespace = getattr(torch.ops, self.namespace_name)
        self.options = {"compute_precision": case.point.precision} if case.point.precision != "fp64" else {}
        self.handles = {}
        self.outputs = {"forward": torch.empty_like(case.grad3), "grad_x": torch.empty_like(case.x3)}
        if case.point.precision == "tf32":
            from historical_plan import Tf32MathPolicy
            names = ("eqlinear.h", "tf32_api.cpp", "bwd.cu")
            prefix = "fasteq/cuda/src/eqlinear/tf32/"
            policy = Tf32MathPolicy.inspect({prefix + name: (build.source / prefix / name).read_bytes() for name in names})
            if not policy["default_forward_source_verified_x3"] or not policy["dgrad_regsplit_source_verified"]:
                raise RuntimeError("This ref's forward/backward math requires audit before any execution")
            info = self.namespace.forward_plan_info(self.handle("forward"))
            if info[1] not in policy["allowed_forward_impls"]:
                raise RuntimeError("Historical default is not in this ref's verified TF32x3 set; refusing execution")
            self.tf32_policy = policy
        self.x_grad = case.x.detach().requires_grad_(True)
        with torch.enable_grad():
            self.graph = self.module.fast_equi_linear(case.descriptor, case.w, self.x_grad, **self.options)
        # P0 queued pinned-host pointer updates are unsafe. Drain setup before any
        # separately measured call; this is a synchronized protocol, not a repair.
        torch.cuda.synchronize()

    def prepare(self, operation, boundary) -> None:
        import torch

        super().prepare(operation, boundary)
        torch.cuda.synchronize()

    def identity(self) -> dict:
        return {"variant_id": "historical_fasteq_" + self.variant, "source_commit": self.manifest["commit"],
                "source_hash": Evidence.fingerprint(self.manifest["source_manifest"]["source_sha256"]),
                "binary_hash": Evidence.fingerprint({row["path"]: row["sha256"] for row in self.manifest["binaries"]}),
                "binaries": self.manifest["binaries"], "mapped_libraries": self.mapped,
                "module_file": str(Path(self.module.__file__).resolve()), "adapted_wrapper_sha256": Evidence.sha256(self.wrapper),
                "source_manifest_sha256": Evidence.sha256(self.audit_path),
                "namespace": self.namespace_name, "math_sources_unchanged": True,
                "runtime_contract_sha256": self.manifest.get("runtime_contract_sha256"),
                "native_reuse_evidence": self.manifest.get("native_bindings", {}),
                "runtime_controls": self.manifest.get("runtime_controls", {}),
                "tf32_source_math_policy": getattr(self, "tf32_policy", None),
                "plan_info_fields": ["precision", "forward_or_backward_impl", "a_view", "j_rows"],
                "isolated_current_package_not_imported": not any(name == "fasteq" or name.startswith("fasteq.") for name in sys.modules),
                "plan_info": {operation: list((self.namespace.forward_plan_info if operation == "forward" else
                                               self.namespace.dgrad_plan_info)(handle)) for operation, handle in self.handles.items()}}

    def close(self) -> None:
        super().close()
        self.module.eqlinear_clear_plan_cache()


if __name__ == "__main__":
    raise SystemExit(HistoricalBuild.main())
