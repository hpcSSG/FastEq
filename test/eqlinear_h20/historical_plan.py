"""Data-driven historical source contracts, native cache groups and portable staging."""
from __future__ import annotations

from collections import Counter
import difflib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile

from common import Evidence
from registry import GitHistory, PatchOverlay, Serialization, SourceState


class Tf32MathPolicy:
    @classmethod
    def inspect(cls, files):
        prefix = "fasteq/cuda/src/eqlinear/tf32/"
        header, api = files[prefix + "eqlinear.h"], files[prefix + "tf32_api.cpp"]
        match = re.search(r"enum class Impl\s*:\s*int\s*\{(.*?)\};", header.decode(), re.S)
        if not match:
            raise ValueError("TF32 enum syntax requires explicit source audit")
        enum_body = re.sub(r"/\*.*?\*/|//[^\n]*", "", match[1], flags=re.S)
        entries = re.findall(r"(k\w+)\s*=\s*(\d+)", enum_body)
        mapping = {int(value): name for name, value in entries}
        if len(mapping) != len(entries):
            raise ValueError("Aliased TF32 enum integers require an explicit mathematical audit")
        allowed = sorted(value for value, name in mapping.items() if "3xTF32" in name)
        forbidden = sorted(value for value, name in mapping.items() if "1xTF32" in name)
        unknown = sorted(set(mapping) - set(allowed) - set(forbidden))
        selection = re.search(r"select_forward_impl\s*\([^)]*\)\s*\{", api.decode())
        body = ""
        if selection:
            depth, end = 1, selection.end()
            while depth and end < len(api.decode()):
                character = api.decode()[end]
                depth += (character == "{") - (character == "}")
                end += 1
            body = api.decode()[selection.end():end - 1]
        selected = sorted(set(re.findall(r"eqlin::Impl::(\w+)", body)))
        safe_names = {mapping[value] for value in allowed}
        static_safe = bool(selected) and not unknown and set(selected) <= safe_names
        backward = files[prefix + "bwd.cu"].decode()
        start = backward.find("struct DgradRunner::Impl_")
        end = backward.find("DgradRunner::DgradRunner", start)
        dgrad = backward[start:end] if start >= 0 and end > start else ""
        backward_safe = all(token in dgrad for token in ("Sm90Operands::kFp32RegSplit", "d_b_big", "d_b_res", "launch_pack_w_dgrad"))
        return {"header_sha256": Serialization.sha(header), "api_sha256": Serialization.sha(api),
                "enum": {str(key): value for key, value in mapping.items()}, "allowed_forward_impls": allowed,
                "prohibited_forward_impls": forbidden, "unknown_forward_impls": unknown,
                "selector_names": selected, "selector_body_sha256": Serialization.sha(body.encode()),
                "default_forward_source_verified_x3": static_safe,
                "dgrad_regsplit_source_verified": backward_safe,
                "dgrad_source_sha256": Serialization.sha(files[prefix + "bwd.cu"]),
                "runtime_rule": "Before any forward, verify this ref's header hash and require actual forward_plan_info impl in allowed_forward_impls; backward numbering is separate"}


class HistoricalContracts:
    PRECISIONS = ("tf32", "fp32", "fp64")

    def __init__(self, session, history_repo):
        self.session = session
        coverage = session.root / "coverage"
        self.requirements = json.loads((coverage / "historical_required_runtimes.json").read_text())
        self.contracts = json.loads((coverage / "historical_sources.json").read_text())
        self.by_id = {row["variant_id"]: row for row in self.requirements}
        self.history = GitHistory(history_repo, Path(session.paths["HANDOFF_ROOT"]) / "sources/fasteq_history.bundle")
        self.states = {}
        self.overlays = {}
        self.recipe_path = "fasteq/cuda/src/eqlinear/CMakeLists.txt"
        self.default_recipe = (Path(session.paths["FASTEQ_ROOT"]) / self.recipe_path).read_bytes()
        import torch
        self.toolchain = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                          "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                          "nvcc": subprocess.check_output(["nvcc", "--version"], text=True, timeout=10),
                          "gxx": subprocess.check_output(["g++", "--version"], text=True, timeout=10).splitlines()[0]}

    def overlay(self, requirement):
        overlay = requirement["reconstruction"]
        if not overlay:
            return None
        key = overlay["member_sha256"]
        if key not in self.overlays:
            with tarfile.open(Path(self.session.paths["HANDOFF_ROOT"]) / overlay["archive"]) as stream:
                member = stream.getmember(overlay["member"])
                if not member.isfile() or member.size > (8 << 20):
                    raise ValueError("Unsafe historical source overlay")
                payload = stream.extractfile(member).read()
            if Serialization.sha(payload) != key:
                raise ValueError("Historical archive overlay SHA differs")
            self.overlays[key] = payload
        return self.overlays[key]

    def state(self, requirement):
        key = Serialization.digest({"commit": requirement["source_commit"], "reconstruction": requirement["reconstruction"]})
        if key not in self.states:
            state = self.history.snapshot(requirement["source_commit"])
            overlay = requirement["reconstruction"]
            if overlay:
                payload = self.overlay(requirement)
                if overlay["member"].endswith(".py"):
                    state.files["fasteq/ops/eqlinear.py"] = payload
                else:
                    changed = PatchOverlay.apply(state.files, payload)
                    if changed != overlay["modified_runtime_paths"]:
                        raise ValueError("Historical overlay changed unexpected runtime paths")
                state.provenance = overlay
            self.states[key] = state
        state = self.states[key]
        manifest = state.manifest(requirement["precision"], requirement["runtime_controls"])
        if manifest["identity_sha256"] != requirement["manifest_sha256"]:
            raise ValueError("Reconstructed source differs from frozen registry contract: " + requirement["variant_id"])
        return state

    def build_inputs(self, state):
        files = {name: Serialization.sha(state.files[name]) if name in state.files else "MISSING"
                 for name in SourceState.BUILD_FILES}
        files[self.recipe_path] = Serialization.sha(state.files.get(self.recipe_path, self.default_recipe))
        return {"source_build_files": files, "gitlinks": state.gitlinks, "toolchain": self.toolchain,
                "cmake_build_type": "Release", "architecture": "90a-real", "CMAKE_ARGS_extra": [],
                "compiler_environment": {name: os.environ.get(name, "") for name in
                    ("CXXFLAGS", "CPPFLAGS", "CUDAFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")}}

    def native_key(self, state, precision):
        prefixes = ("fasteq/cuda/src/eqlinear/common/", f"fasteq/cuda/src/eqlinear/{precision}/")
        hashes = {name: Serialization.sha(value) for name, value in state.files.items() if name.startswith(prefixes)}
        return Serialization.digest({"source": hashes, "build": self.build_inputs(state), "precision": precision}), hashes

    def dispatcher_key(self, state, native_keys):
        prefixes = ("fasteq/cuda/src/eqlinear/common/", "fasteq/cuda/src/eqlinear/dispatch/")
        return Serialization.digest({"source": {name: Serialization.sha(value) for name, value in state.files.items()
                                               if name.startswith(prefixes)},
                                     "build": self.build_inputs(state), "native_keys": native_keys})

    def existing(self):
        natives, dispatchers = {}, {}
        for tag in ("p0", "p14", "current"):
            path = self.session.root / "manifests" / ("current_build.json" if tag == "current" else "historical_" + tag + "_build.json")
            if not path.is_file():
                continue
            record = json.loads(path.read_text())
            current = tag == "current"
            if current:
                record = {**record, "status": "BUILT_SOURCE_ELF_ARCH_VERIFIED", "commit": record["source"]["commit"],
                          "source_manifest": {"source_sha256": record["source"]["files"]},
                          "binaries": [{"path": row["installed_path"], "sha256": row["sha256"], "build_path": row["build_path"]}
                                       for row in record["binaries"]]}
            if record["status"] != "BUILT_SOURCE_ELF_ARCH_VERIFIED":
                continue
            state = self.history.snapshot(record["commit"])
            actual = record["source_manifest"]["source_sha256"]
            for name, value in state.files.items():
                if name.startswith("fasteq/cuda/src/eqlinear/") and not name.endswith("CMakeLists.txt") and name not in actual:
                    raise ValueError("Cached native build manifest omitted a source input: " + name)
                if name in actual and Serialization.sha(value) != actual[name]:
                    raise ValueError("Existing build/source Git record differs")
            keys = {}
            for precision in self.PRECISIONS:
                key, source_hashes = self.native_key(state, precision)
                keys[precision] = key
                binary = next(row for row in record["binaries"] if row["path"].endswith("_" + precision + ".so"))
                if Evidence.sha256(Path(binary["path"])) != binary["sha256"]:
                    raise ValueError("Cached historical DSO changed")
                command_path = Path(record.get("build", "")) / "compile_commands.json" if not current else None
                command_rows = json.loads(command_path.read_text()) if command_path and command_path.is_file() else record.get("cuda_compile_commands", [])
                commands = [row["command"] for row in command_rows if f"/{precision}/" in row["file"]]
                if current:
                    directory = Path(binary["build_path"]).parent
                    while directory != self.session.root and not (directory / "build.ninja").is_file():
                        directory = directory.parent
                    if not (directory / "build.ninja").is_file():
                        raise ValueError("Current fresh build flags cannot be recovered from its Ninja graph")
                    commands = subprocess.check_output(["ninja", "-C", str(directory), "-t", "commands", "fasteq_eqlinear_" + precision],
                                                       text=True, timeout=30).splitlines()
                cuda_commands = [command for command in commands if "nvcc" in command]
                if not cuda_commands or any("compute_90a" not in command or "sm_90a" not in command for command in cuda_commands):
                    raise ValueError("Cached native actual architecture flags are unproven")
                natives.setdefault(key, {"precision": precision, "key": key, "source_hashes": source_hashes,
                    "source_root": str(Path(self.session.paths["FASTEQ_ROOT"])) if current else record["source"],
                    "cutlass_root": str(Path(self.session.paths["FASTEQ_ROOT"]) / "3rdparty/cutlass") if current else str(Path(record["source"]) / "3rdparty/cutlass"),
                    "source_commit": record["commit"], "source_build_inputs": self.build_inputs(state),
                    "binary": binary, "api_symbol": "fasteq_eqlinear_" + (tag + "_" if not current else "") + precision + "_api",
                    "build_manifest": str(path), "build_manifest_sha256": Evidence.sha256(path),
                    "compile_commands_sha256": Serialization.digest(commands), "actual_compile_commands": commands,
                    "status": "BUILT_AUDITED_EXACT_SOURCE_INPUTS",
                    "namespace_only_build_adaptations": True})
            dispatcher = next(row for row in record["binaries"] if ("dispatch.so" in row["path"] if not current else Path(row["path"]).name.startswith("_cuda")))
            dispatchers[self.dispatcher_key(state, keys)] = {"dispatcher": dispatcher, "backend_namespace": tag if not current else "",
                "native_keys": keys, "build_manifest": str(path), "build_manifest_sha256": Evidence.sha256(path)}
        for path in sorted((self.session.root / "manifests").glob("historical_*_build.json")):
            record = json.loads(path.read_text())
            if record.get("binding_format") != "historical_contract_v1" or record.get("status") != "BUILT_SOURCE_ELF_ARCH_VERIFIED":
                continue
            for precision, value in record["native_bindings"].items():
                if Evidence.sha256(Path(value["binary"]["path"])) != value["binary"]["sha256"]:
                    raise ValueError("Generalized cached DSO changed")
                observed = dict(value)
                command_path = Path(record["build"]) / "compile_commands.json"
                if not value["reuse"] and command_path.is_file():
                    observed.update(actual_compile_commands=[row["command"] for row in json.loads(command_path.read_text()) if f"/{precision}/" in row["file"]],
                                    source_root=record["source"], cutlass_root=record["source_manifest"]["cutlass_root"])
                natives.setdefault(value["key"], {**observed, "status": "BUILT_AUDITED_EXACT_SOURCE_INPUTS",
                    "build_manifest": str(path), "build_manifest_sha256": Evidence.sha256(path)})
            dispatcher = next(row for row in record["binaries"] if row["path"] == record["dispatcher_path"])
            dispatchers.setdefault(record["dispatcher_key"], {"dispatcher": dispatcher,
                "backend_namespace": record["backend_namespace"],
                "native_keys": {precision: value["key"] for precision, value in record["native_bindings"].items()},
                "build_manifest": str(path), "build_manifest_sha256": Evidence.sha256(path)})
        return natives, dispatchers


class PortableHeaders:
    @classmethod
    def stage(cls, session, expected_gitlink):
        original = Path(session.paths["FASTEQ_ROOT"]) / "3rdparty/cutlass"
        commit = subprocess.check_output(["git", "-C", str(original), "rev-parse", "HEAD"], text=True, timeout=10).strip()
        dirty = subprocess.check_output(["git", "-C", str(original), "status", "--porcelain", "--untracked-files=no"], text=True, timeout=10)
        if commit != expected_gitlink or dirty:
            raise ValueError("CUTLASS Git pin/dirty state disagrees with source contract")
        root = session.root / "sources/dependencies" / ("cutlass_" + commit)
        manifest = root / "headers_manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text())
            if any(not (root / name).is_file() or Evidence.sha256(root / name) != digest for name, digest in data["files"].items()):
                raise ValueError("Portable CUTLASS closure changed")
            return root, Evidence.sha256(manifest)
        root.mkdir(parents=True, exist_ok=False)
        files = {}
        for subtree in ("include", "tools/util/include"):
            for source in sorted((original / subtree).rglob("*")):
                if source.is_symlink():
                    raise ValueError("Explicitly audit CUTLASS symlink before portable staging: " + str(source))
                if not source.is_file():
                    continue
                name = source.relative_to(original).as_posix()
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                files[name] = Evidence.sha256(target)
        for name in ("LICENSE.txt", "LICENSE", "NOTICE"):
            if (original / name).is_file():
                shutil.copy2(original / name, root / name)
                files[name] = Evidence.sha256(root / name)
        Evidence.write(manifest, {"commit": commit, "original": str(original), "files": files,
                                  "regular_files_only": True, "portable_include_closure": ["include", "tools/util/include"]})
        return root, Evidence.sha256(manifest)


class HistoricalExecutionPlan:
    def __init__(self, session, history_repo):
        self.session = session
        self.contracts = HistoricalContracts(session, history_repo)

    def run(self, output):
        if output.exists():
            raise FileExistsError("Preserve old plans; choose a fresh output")
        cached, dispatchers = self.contracts.existing()
        native_groups, dispatcher_groups, wrappers = {}, {}, []
        for requirement in self.contracts.requirements:
            state = self.contracts.state(requirement)
            keys = {}
            for precision in self.contracts.PRECISIONS:
                key, hashes = self.contracts.native_key(state, precision)
                keys[precision] = key
                native_groups.setdefault(key, {"precision": precision, "key": key, "source_hashes": hashes,
                    "build_inputs": self.contracts.build_inputs(state), "source_commit": state.commit,
                    "reconstruction": state.provenance, "representative_contract": requirement["variant_id"],
                    "cached": cached.get(key), "status": "BUILT_AUDITED" if key in cached else "NEEDS_FRESH_BUILD"})
            dispatcher_key = self.contracts.dispatcher_key(state, keys)
            dispatcher_groups.setdefault(dispatcher_key, {"key": dispatcher_key, "native_keys": keys,
                "representative_contract": requirement["variant_id"], "source_commit": state.commit,
                "reconstruction": state.provenance, "cached": dispatchers.get(dispatcher_key),
                "status": "BUILT_AUDITED" if dispatcher_key in dispatchers else "NEEDS_DISPATCHER_BUILD"})
            policy = Tf32MathPolicy.inspect(state.files)
            wrapper_hashes = {name: Serialization.sha(state.files[name]) for name in SourceState.WRAPPERS if name in state.files}
            covered = any(label in ("P0", "P14", "current") for label in requirement["labels"])
            command = [self.session.paths["FASTEQ_PY"], "-B", str(Path(__file__).with_name("historical.py")), "prepare",
                       "--session", str(self.session.root), "--variant", requirement["variant_id"],
                       "--history-repo", str(self.contracts.history.repository)]
            actions = {action + "_argv": [self.session.paths["FASTEQ_PY"], "-B", str(Path(__file__).with_name("historical.py")), action,
                        "--session", str(self.session.root), "--variant", requirement["variant_id"]]
                       for action in ("configure", "build", "verify", "probe")}
            actions["probe_argv"] += ["--point", "SMALL_P1/512/" + requirement["precision"]]
            wrappers.append({**requirement, "wrapper_hashes": wrapper_hashes, "native_keys": keys,
                "dispatcher_key": dispatcher_key, "tf32_policy": policy,
                "known_default_accuracy_coverage": "25_POINTS_FIVE_PATHS_VERIFIED_VIA_FULL75_BUNDLE" if covered else "UNTESTED",
                "performance_status": "PENDING_THREE_ROUNDS", "prepare_argv": command, **actions,
                "scope": "public/default+run boundaries only; never substitutes for native per-impl selector coverage"})
        registry = json.loads((self.session.root / "coverage/variants.json").read_text())
        selectors = [{"variant_id": row["variant_id"], "impl_id": row["impl_id"], "precision": row["variant_id"].split("_")[2],
                      "scope_status": row["scope_status"], "status": "EXCLUDED_USER_TF32X1" if row["scope_status"] == "EXCLUDED_BY_USER" else "PENDING_TEST_ONLY_SELECTOR_BRIDGE",
                      "dependency": "Existing native enum is not selectable through production C ABI; add isolated test-only selector API, preserve default routes",
                      "required_points": 25, "default_route_is_not_full_impl_coverage": True}
                     for row in registry["variants"] if row["family"] == "current_fasteq_native"]
        result = {"schema_version": 1, "classification": "EXECUTION_DEPENDENCY_PLAN_NOT_TEST_RESULTS",
            "source_registry": {name: Evidence.sha256(self.session.root / "coverage" / name) for name in
                ("historical_required_runtimes.json", "historical_sources.json")},
            "toolchain": self.contracts.toolchain, "history_repository": str(self.contracts.history.repository),
            "native_groups": native_groups, "dispatcher_groups": dispatcher_groups,
            "wrapper_tests": wrappers, "nondefault_selector_dependencies": selectors,
            "summary": {"source_precision_contracts": len(wrappers), "distinct_native_groups": len(native_groups),
                "cached_native_groups": sum(row["cached"] is not None for row in native_groups.values()),
                "native_groups_requiring_fresh_build": sum(row["cached"] is None for row in native_groups.values()),
                "distinct_dispatcher_groups": len(dispatcher_groups), "wrappers_pending_accuracy": sum(row["known_default_accuracy_coverage"] == "UNTESTED" for row in wrappers),
                "selector_states": dict(Counter(row["status"] for row in selectors))},
            "tool_sha256": Evidence.sha256(Path(__file__))}
        selector_inventory = self.session.root / "manifests/native_selector_object_inventory02.json"
        result["test_only_selector_link_strategy"] = {
            "strategy": "Link exact current H20 native/common PIC objects plus a new isolated per-precision Torch CPU binding; localize original C ABI exports; never co-load production DSOs",
            "inventory": str(selector_inventory) if selector_inventory.is_file() else None,
            "inventory_sha256": Evidence.sha256(selector_inventory) if selector_inventory.is_file() else None,
            "dependency_status": "EXISTING_OBJECTS_AUDITED_CPU_BINDING_IMPLEMENTATION_PENDING" if selector_inventory.is_file() else "OBJECT_INVENTORY_PENDING",
            "CUDA_recompile_expected": False, "runtime_coverage_claimed": False,
            "execution_guards": ["per-ref enum/header SHA", "prohibit this ref's TF32x1 ids before create/forward", "object/source/flags/ABI SHA", "actual mapped test DSO, no production DSO"]}
        Evidence.write(output, result)
        return result


class HistoricalSourceStage:
    def __init__(self, session, history_repo):
        self.session, self.contracts = session, HistoricalContracts(session, history_repo)

    def stage(self, variant):
        requirement = self.contracts.by_id[variant]
        state = self.contracts.state(requirement)
        key = Serialization.digest({"files": {name: Serialization.sha(value) for name, value in state.files.items()}, "gitlinks": state.gitlinks})
        root = self.session.root / "sources/historical_contracts" / key
        path = root / "stage_manifest.json"
        if path.exists():
            data = json.loads(path.read_text())
            if any(Evidence.sha256(root / name) != digest for name, digest in data["files"].items()):
                raise ValueError("Historical portable source stage changed")
            self.overlay_provenance(root, requirement)
            return data
        root.mkdir(parents=True, exist_ok=False)
        files = {}
        for name, payload in state.files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            files[name] = Evidence.sha256(target)
        recipe = self.contracts.recipe_path
        if recipe not in state.files:
            (root / recipe).write_bytes(self.contracts.default_recipe)
            (root / "build_only_cmake.diff").write_text("".join(difflib.unified_diff([], self.contracts.default_recipe.decode().splitlines(True),
                                                      fromfile="original/MISSING", tofile=recipe)))
        headers, headers_sha = PortableHeaders.stage(self.session, state.gitlinks["3rdparty/cutlass"])
        data = {"stage_key": key, "root": str(root), "source_commit": state.commit, "source_reconstruction": state.provenance,
                "files": files, "mathematical_source_policy": "Exact Git ref plus frozen authorized archived overlay; only missing CMake added separately",
                "temporary_cmake_sha256": Evidence.sha256(root / recipe), "gitlinks": state.gitlinks,
                "cutlass_root": str(headers), "cutlass_manifest_sha256": headers_sha, "regular_files_only": True,
                "tf32_policy": Tf32MathPolicy.inspect(state.files), "contract": requirement["manifest_sha256"]}
        Evidence.write(path, data)
        self.overlay_provenance(root, requirement)
        return data

    def overlay_provenance(self, root, requirement):
        payload = self.contracts.overlay(requirement)
        if payload is None:
            return
        overlay = requirement["reconstruction"]
        directory = root / "source_provenance"
        directory.mkdir(exist_ok=True)
        path = directory / Path(overlay["member"]).name
        if path.exists() and Evidence.sha256(path) != overlay["member_sha256"]:
            raise ValueError("Existing authorized overlay provenance differs")
        if not path.exists():
            path.write_bytes(payload)
        receipt = path.with_suffix(path.suffix + ".receipt.json")
        if not receipt.exists():
            Evidence.write(receipt, {"overlay": overlay, "copied_sha256": Evidence.sha256(path),
                                      "mathematical_stage_manifest_sha256": Evidence.sha256(root / "stage_manifest.json"),
                                      "original_handoff_not_modified": True})


class NativeSelectorObjects:
    """Audit current H20 PIC objects for a future CPU-only, test-only selector binding."""

    @classmethod
    def capture(cls, session, output):
        if output.exists():
            raise FileExistsError("Preserve object inventory epochs")
        manifest_path = session.root / "manifests/current_build.json"
        manifest = json.loads(manifest_path.read_text())
        source_root = Path(session.paths["FASTEQ_ROOT"])
        probe = Path(manifest["binaries"][0]["build_path"]).parent
        while probe != session.root and not (probe / "build.ninja").is_file():
            probe = probe.parent
        if not (probe / "build.ninja").is_file():
            raise ValueError("Current build Ninja graph is missing")
        groups = {}
        for precision in HistoricalContracts.PRECISIONS:
            commands = subprocess.check_output(["ninja", "-C", str(probe), "-t", "commands", "fasteq_eqlinear_" + precision], text=True, timeout=30).splitlines()
            objects = []
            for command in commands:
                tokens = shlex.split(command)
                if "-c" not in tokens or "-o" not in tokens:
                    continue
                source = Path(tokens[tokens.index("-c") + 1])
                relative = source.relative_to(source_root).as_posix()
                if not relative.startswith("fasteq/cuda/src/eqlinear/"):
                    raise ValueError("Unrelated source entered the native object link set")
                if relative not in manifest["source"]["files"] or Evidence.sha256(source) != manifest["source"]["files"][relative]:
                    raise ValueError("Current native object source differs from its build manifest")
                target = Path(tokens[tokens.index("-o") + 1])
                target = target if target.is_absolute() else probe / target
                if not target.is_file() or target.is_symlink():
                    raise ValueError("Native object must be a regular existing H20 artifact")
                if source.suffix == ".cu" and ("compute_90a" not in command or "sm_90a" not in command):
                    raise ValueError("Native object architecture was downgraded")
                if "-fPIC" not in command:
                    raise ValueError("Native object is not PIC")
                objects.append({"source": str(source), "source_sha256": Evidence.sha256(source), "object": str(target),
                                "object_sha256": Evidence.sha256(target), "object_bytes": target.stat().st_size,
                                "actual_compile_command": command})
            header = source_root / "fasteq/cuda/src/eqlinear" / precision / "eqlinear.h"
            enum_type = "F64Impl" if precision == "fp64" else "Impl"
            enum = re.search(r"enum class " + enum_type + r"\s*:\s*int\s*\{(.*?)\};", header.read_text(), re.S)
            if not enum:
                raise ValueError("Current enum requires explicit parsing")
            enum_body = re.sub(r"/\*.*?\*/|//[^\n]*", "", enum[1], flags=re.S)
            entries = re.findall(r"(k\w+)\s*=\s*(\d+)", enum_body)
            mapping = {int(value): name for name, value in entries}
            if len(mapping) != len(entries):
                raise ValueError("Ambiguous current native enum aliases")
            allowed = [key for key, name in mapping.items() if precision != "tf32" or "3xTF32" in name]
            groups[precision] = {"objects": objects, "header": str(header), "header_sha256": Evidence.sha256(header),
                "enum_class": enum_type, "enum": mapping, "allowed_impls": allowed, "prohibited_impls": sorted(set(mapping) - set(allowed)),
                "isolated_dso_per_precision": True, "status": "OBJECTS_AUDITED_TEST_ONLY_BINDING_PENDING",
                "link_policy": "Link exact audited .o + one new Torch CPU binding, localize existing C ABI exports; never co-load production DSOs; verify mappings and source enums before execution"}
        import torch
        result = {"classification": "CPU_OBJECT_INVENTORY_NOT_IMPL_RUNTIME_COVERAGE", "current_build": str(manifest_path),
                  "current_build_sha256": Evidence.sha256(manifest_path), "source_commit": manifest["source"]["commit"],
                  "build_graph": str(probe), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI), "groups": groups,
                  "allowed_impl_count": sum(len(group["allowed_impls"]) for group in groups.values()),
                  "CUDA_recompilation_needed_for_existing_impls": False, "bridge_binding_implemented": False,
                  "tool_sha256": Evidence.sha256(Path(__file__))}
        Evidence.write(output, result)
        return result
