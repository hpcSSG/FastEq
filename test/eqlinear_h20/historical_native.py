"""Build/reuse native groups for a frozen historical source contract, without installation."""
from __future__ import annotations

import difflib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from common import Evidence
from historical import HistoricalBuild
from historical_plan import HistoricalContracts, HistoricalSourceStage


class ContractSourceAudit:
    def __init__(self, harness):
        self.harness = harness

    def run(self):
        data = json.loads((self.harness / "source_manifest.json").read_text())
        root = Path(data["worktree"])
        for name, digest in data["source_sha256"].items():
            if Evidence.sha256(root / name) != digest:
                raise ValueError("Historical contract source changed: " + name)
        recipe = root / "fasteq/cuda/src/eqlinear/CMakeLists.txt"
        if Evidence.sha256(recipe) != data["temporary_prep_cmake_sha256"]:
            raise ValueError("Historical build-only CMake changed")
        for name, digest in data["generated_sha256"].items():
            if Evidence.sha256(self.harness / "generated" / name) != digest:
                raise ValueError("Historical namespace-only generation changed")
        header_root = Path(data["cutlass_root"])
        header_manifest = header_root / "headers_manifest.json"
        if Evidence.sha256(header_manifest) != data["cutlass_manifest_sha256"]:
            raise ValueError("Portable CUTLASS manifest changed")
        for name, digest in json.loads(header_manifest.read_text())["files"].items():
            if Evidence.sha256(header_root / name) != digest:
                raise ValueError("Portable CUTLASS source changed: " + name)
        print("PASS: exact authorized historical source, namespace adapters and portable headers")


class ContractElfAudit:
    def __init__(self, harness):
        self.harness = harness

    @staticmethod
    def permitted_exports(rows, api_symbol, current_cache=False):
        strong = {name for kind, name in rows if kind != "W"}
        weak = {name for kind, name in rows if kind == "W"}
        return (strong == {api_symbol} and
                (not weak or current_cache and all(re.match(r"_ZNK?St", name) for name in weak)))

    def run(self):
        data = json.loads((self.harness / "binding.json").read_text())
        audits = {}
        for precision, binding in data["native_bindings"].items():
            library = Path(binding["binary"]["path"])
            dynamic = subprocess.check_output(["readelf", "-d", str(library)], text=True, timeout=15)
            symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(library)], text=True, timeout=15)
            exported = [(line.split()[-2], line.split()[-1]) for line in symbols.splitlines() if line.strip()]
            current_cache = False
            if binding.get("reuse") and binding.get("build_manifest"):
                manifest_path = Path(binding["build_manifest"])
                if manifest_path.name == "current_build.json":
                    manifest = json.loads(manifest_path.read_text())
                    current_cache = (Evidence.sha256(manifest_path) == binding["build_manifest_sha256"] and
                        any(row["installed_path"] == str(library) and row["sha256"] == Evidence.sha256(library)
                            for row in manifest["binaries"]))
            if not self.permitted_exports(exported, binding["api_symbol"], current_cache) or f"[{library.name}]" not in dynamic:
                raise ValueError("Historical backend SONAME/API isolation failed: " + precision)
            audits[precision] = {"path": str(library), "sha256": Evidence.sha256(library), "exports": exported,
                "current_cache_sha_verified": current_cache,
                "policy": "one strong API; exact-current cache may retain weak std templates only"}
        dispatcher = Path(data["dispatcher_path"])
        dynamic = subprocess.check_output(["readelf", "-d", str(dispatcher)], text=True, timeout=15)
        undefined = subprocess.check_output(["nm", "-D", "--undefined-only", str(dispatcher)], text=True, timeout=15)
        for binding in data["native_bindings"].values():
            if f"[{Path(binding['binary']['path']).name}]" not in dynamic or binding["api_symbol"] not in undefined:
                raise ValueError("Historical dispatcher did not bind the exact audited native group")
        print("PASS: exact native SONAMEs and C API references")
        return audits


class ContractHistoricalBuild(HistoricalBuild):
    def prepare(self, history_repo):
        if self.record.exists() or self.harness.exists():
            raise FileExistsError("Historical contract binding already exists")
        contracts = HistoricalContracts(self.session, history_repo)
        requirement = contracts.by_id[self.variant]
        state = contracts.state(requirement)
        stage = HistoricalSourceStage(self.session, history_repo).stage(self.variant)
        self.source = Path(stage["root"])
        self.harness.mkdir(parents=True)
        cached, dispatchers = contracts.existing()
        keys = {precision: contracts.native_key(state, precision)[0] for precision in contracts.PRECISIONS}
        dispatch_key = contracts.dispatcher_key(state, keys)
        reused_dispatcher = dispatchers.get(dispatch_key)
        tag = "hc_" + dispatch_key[:12]
        namespace = reused_dispatcher["backend_namespace"] if reused_dispatcher else tag
        generated = self.harness / "generated"
        generated.mkdir()
        wrapper_name = "eqlinear_" + self.variant + ".py"
        source_name = "fasteq/ops/eqlinear.py"
        original_wrapper = (self.source / source_name).read_text()
        namespace_token = "eqlinear_" + namespace if namespace else "eqlinear"
        wrapper = original_wrapper.replace("torch.ops.eqlinear.", "torch.ops." + namespace_token + ".")
        (generated / wrapper_name).write_text(wrapper)
        dispatcher_name = "eqlinear_dispatch_" + tag + ".cpp"
        original_dispatch = (self.source / "fasteq/cuda/src/eqlinear/dispatch/eqlinear_dispatch.cpp").read_text()
        dispatcher = original_dispatch.replace("TORCH_LIBRARY(eqlinear, m)", "TORCH_LIBRARY(" + namespace_token + ", m)")
        if dispatcher == original_dispatch and namespace:
            raise ValueError("Historical dispatcher namespace token is not recognized")
        (generated / dispatcher_name).write_text(dispatcher)
        (self.harness / "namespace_adaptations.diff").write_text("".join(difflib.unified_diff(
            original_wrapper.splitlines(True), wrapper.splitlines(True), fromfile="historical/" + source_name, tofile="generated/" + wrapper_name)) +
            "".join(difflib.unified_diff(original_dispatch.splitlines(True), dispatcher.splitlines(True),
                                        fromfile="historical/dispatch.cpp", tofile="generated/" + dispatcher_name)))
        native = {}
        for precision, key in keys.items():
            if key in cached:
                if cached[key]["source_hashes"] != contracts.native_key(state, precision)[1] or cached[key]["source_build_inputs"] != contracts.build_inputs(state):
                    raise ValueError("Exact native source/build input comparison failed")
                native[precision] = {**cached[key], "reuse": True}
            else:
                native[precision] = {"key": key, "precision": precision,
                    "source_hashes": contracts.native_key(state, precision)[1],
                    "source_build_inputs": contracts.build_inputs(state), "reuse": False,
                    "api_symbol": "fasteq_eqlinear_" + tag + "_" + precision + "_api",
                    "binary": {"path": str(self.build / "artifacts" / ("libfasteq_eqlinear_" + tag + "_" + precision + ".so"))}}
            (self.harness / ("exports_" + precision + ".map")).write_text("{ global: " + native[precision]["api_symbol"] + "; local: *; };\n")
        dispatcher_path = (reused_dispatcher["dispatcher"]["path"] if reused_dispatcher else
                           str(self.build / "artifacts" / ("libeqlinear_" + tag + "_dispatch.so")))
        binding = {"binding_format": "historical_contract_v1", "variant": self.variant, "commit": state.commit,
            "precision": requirement["precision"], "runtime_contract_sha256": requirement["manifest_sha256"],
            "historical_labels": requirement["labels"], "source_reconstruction": requirement["reconstruction"],
            "runtime_controls": requirement["runtime_controls"], "source": str(self.source), "harness": str(self.harness),
            "build": str(self.build), "backend_namespace": namespace, "wrapper_name": wrapper_name,
            "dispatcher_source_name": dispatcher_name, "dispatcher_key": dispatch_key, "dispatcher_path": dispatcher_path,
            "native_bindings": native, "reused_dispatcher": reused_dispatcher,
            "tf32_policy": stage["tf32_policy"], "status": "PREPARED_NOT_BUILT",
            "build_mode": "REUSED_EXACT_DISPATCHER" if reused_dispatcher else "COMPOSE_EXACT_NATIVE_GROUPS",
            "fresh_native_precisions": [precision for precision, value in native.items() if not value["reuse"]],
            "editable_install_changed": False, "new_mathematical_edits": False}
        audit = {"commit": state.commit, "worktree": str(self.source), "prep_root": str(self.source),
                 "source_sha256": stage["files"], "generated_sha256": {name: Evidence.sha256(generated / name)
                     for name in (wrapper_name, dispatcher_name)},
                 "temporary_prep_cmake_sha256": stage["temporary_cmake_sha256"], "cutlass_root": stage["cutlass_root"],
                 "cutlass_manifest_sha256": stage["cutlass_manifest_sha256"], "authorized_overlay": state.provenance}
        Evidence.write(self.harness / "source_manifest.json", audit)
        binding["source_manifest"] = audit
        binding["source_manifest_sha256"] = Evidence.sha256(self.harness / "source_manifest.json")
        producer = self.harness / "producer"
        producer.mkdir()
        for name in ("historical.py", "historical_native.py", "historical_plan.py", "registry.py", "current.py", "common.py", "session.py"):
            shutil.copy2(Path(__file__).parent / name, producer / name)
        for filename, audit_class in (("source_audit.py", "ContractSourceAudit"), ("verify_elf.py", "ContractElfAudit")):
            (self.harness / filename).write_text("from pathlib import Path\nimport sys\n" +
                "sys.path.insert(0,str(Path(__file__).parent/'producer'))\nfrom historical_native import " + audit_class + "\n" +
                audit_class + "(Path(__file__).parent).run()\n")
        (self.harness / "CMakeLists.txt").write_text(self.cmake(binding, stage))
        Evidence.write(self.harness / "binding.json", binding)
        Evidence.write(self.record, binding)
        ContractSourceAudit(self.harness).run()

    def cmake(self, binding, stage):
        def quote(path):
            return str(path).replace("\\", "/").replace('"', '\\"')
        lines = ["cmake_minimum_required(VERSION 3.19)", "project(historical_contract LANGUAGES CXX CUDA)",
            "set(CMAKE_CXX_STANDARD 17)", "set(CMAKE_CUDA_STANDARD 17)", "set(CMAKE_POSITION_INDEPENDENT_CODE ON)",
            'set(CMAKE_CUDA_ARCHITECTURES "90a-real")', 'set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts")',
            f'set(CUTLASS_ROOT "{quote(stage["cutlass_root"])}")',
            "find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)", "find_package(CUDAToolkit REQUIRED)",
            'execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/source_audit.py" RESULT_VARIABLE audit_result)',
            'if(NOT audit_result EQUAL 0)\n message(FATAL_ERROR "Frozen historical source audit failed")\nendif()',
            f'add_subdirectory("{quote(self.source / "fasteq/cuda/src/eqlinear")}" "${{CMAKE_BINARY_DIR}}/backends")',
            'execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import os,torch;print(os.path.join(torch.utils.cmake_prefix_path,\'Torch\'))" OUTPUT_VARIABLE Torch_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)',
            "find_package(Torch REQUIRED)", 'set(CMAKE_CUDA_ARCHITECTURES "90a-real")',
            "if(NOT TARGET CUDA::nvToolsExt)\n add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)\nendif()",
            'separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")',
            'execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" OUTPUT_VARIABLE torch_abi OUTPUT_STRIP_TRAILING_WHITESPACE)']
        definitions, libraries, rpaths = [], [], {"$ORIGIN"}
        for precision, value in binding["native_bindings"].items():
            target = "fasteq_eqlinear_" + precision
            definitions.append("fasteq_eqlinear_" + precision + "_api=" + value["api_symbol"])
            if value["reuse"]:
                lines.append(f"set_target_properties({target} PROPERTIES EXCLUDE_FROM_ALL TRUE)")
                libraries.append('"' + quote(value["binary"]["path"]) + '"')
                rpaths.add(str(Path(value["binary"]["path"]).parent))
            else:
                output = Path(value["binary"]["path"]).stem.removeprefix("lib")
                lines += [f'set_target_properties({target} PROPERTIES OUTPUT_NAME "{output}")',
                    f'target_compile_definitions({target} PRIVATE "{definitions[-1]}")',
                    f'target_link_options({target} PRIVATE "LINKER:--version-script=${{CMAKE_CURRENT_SOURCE_DIR}}/exports_{precision}.map")']
                libraries.append(target)
        target = Path(binding["dispatcher_path"]).stem.removeprefix("lib")
        lines += [f'add_library({target} SHARED "${{CMAKE_CURRENT_SOURCE_DIR}}/generated/{binding["dispatcher_source_name"]}" "{quote(self.source / "fasteq/cuda/src/eqlinear/common/eqlinear_group_table.cpp")}")',
            f'target_include_directories({target} PRIVATE ${{Python3_INCLUDE_DIRS}} ${{TORCH_INCLUDE_DIRS}} ${{CUDAToolkit_INCLUDE_DIRS}} "{quote(self.source / "fasteq/cuda/src/eqlinear/common")}")',
            f'target_compile_options({target} PRIVATE ${{torch_flags}})',
            f'target_compile_definitions({target} PRIVATE "_GLIBCXX_USE_CXX11_ABI=${{torch_abi}}" ' + " ".join('"' + item + '"' for item in definitions) + ")",
            f'target_link_libraries({target} PRIVATE Python3::Module ${{TORCH_LIBRARIES}} ' + " ".join(libraries) + ")",
            f'set_target_properties({target} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON BUILD_RPATH "' + ";".join(sorted(rpaths)) + ';${TORCH_INSTALL_PREFIX}/lib")']
        return "\n".join(lines) + "\n"

    def configure(self, attempt):
        super().configure(attempt)
        record = json.loads(self.record.read_text())
        rows = json.loads((self.build / "compile_commands.json").read_text())
        stage = json.loads((self.source / "stage_manifest.json").read_text())
        guards = []
        for precision, value in record["native_bindings"].items():
            if not value["reuse"]:
                continue
            expected = [row["command"] for row in rows if f"/{precision}/" in row["file"]]
            old = self.normalized_commands(value["actual_compile_commands"], value["source_root"], value["cutlass_root"], precision)
            new = self.normalized_commands(expected, str(self.source), stage["cutlass_root"], precision)
            if old != new:
                mismatch = self.harness / ("flags_mismatch_" + precision + "_" + attempt + ".json")
                if mismatch.exists():
                    raise FileExistsError("Preserve the previous flag mismatch evidence")
                Evidence.write(mismatch, {"cached": old, "expected": new})
                raise ValueError("Cached native actual compile flags differ: " + precision)
            guards.append({"precision": precision, "normalized_per_file_flags_sha256": Evidence.fingerprint(new), "passed": True})
        record["native_compile_flag_reuse_guards"] = guards
        Evidence.write(self.record, record)
        Evidence.write(self.harness / "binding.json", record)

    @staticmethod
    def normalized_commands(commands, source_root, cutlass_root, precision):
        result = {}
        for command in commands:
            tokens = shlex.split(command)
            if "-c" not in tokens:
                continue
            source = tokens[tokens.index("-c") + 1]
            if f"/{precision}/" not in source:
                continue
            flags, index = [], 0
            while index < len(tokens):
                token = tokens[index]
                if token in ("-o", "-MT", "-MF", "-c"):
                    index += 2
                    continue
                if token == "-MD" or re.match(r"-Dfasteq_eqlinear_(tf32|fp32|fp64)_api=", token):
                    index += 1
                    continue
                # CMake may spell the exact pinned include tree through ../ or a symlink.
                # Resolve filesystem includes before mapping the already audited source roots;
                # preserve option order, all macros and every arithmetic/compiler flag.
                if token.startswith("-I") and len(token) > 2:
                    include = Path(token[2:])
                    if not include.is_absolute():
                        raise ValueError("Relative include path needs an explicit compile working directory")
                    token = "-I" + str(include.resolve())
                token = token.replace(str(Path(cutlass_root).resolve()), "<CUTLASS>").replace(str(Path(source_root).resolve()), "<SOURCE>")
                if index == 0 and Path(token).is_file():
                    token = str(Path(token).resolve())
                flags.append(token)
                index += 1
            result[str(Path(source).relative_to(source_root))] = flags
        return result

    def compile(self, attempt, jobs):
        if json.loads(self.record.read_text())["reused_dispatcher"]:
            print("REUSE: exact existing native groups; no recompilation", flush=True)
            return
        super().compile(attempt, jobs)

    def verify(self, attempt):
        record = json.loads(self.record.read_text())
        ContractSourceAudit(self.harness).run()
        symbol_audit = ContractElfAudit(self.harness).run()
        commands = json.loads((self.build / "compile_commands.json").read_text()) if not record["reused_dispatcher"] else []
        cuda = [row for row in commands if row["file"].endswith(".cu")]
        if any("compute_90a" not in row["command"] or "sm_90a" not in row["command"] for row in cuda):
            raise ValueError("Historical native compile architecture downgraded")
        binaries = []
        for value in record["native_bindings"].values():
            path = Path(value["binary"]["path"])
            value["binary"].update(sha256=Evidence.sha256(path), bytes=path.stat().st_size)
            if not value["reuse"]:
                value["actual_compile_commands"] = [row["command"] for row in commands if f"/{value['precision']}/" in row["file"]]
                value["compile_commands_sha256"] = Evidence.fingerprint(value["actual_compile_commands"])
                value["source_root"] = str(self.source)
                value["cutlass_root"] = record["source_manifest"]["cutlass_root"]
            binaries.append(value["binary"])
        dispatcher = Path(record["dispatcher_path"])
        binaries.append({"path": str(dispatcher), "sha256": Evidence.sha256(dispatcher), "bytes": dispatcher.stat().st_size})
        record.update(status="BUILT_SOURCE_ELF_ARCH_VERIFIED", binaries=binaries,
            symbol_audit=symbol_audit, elf_auditor_source=str(Path(__file__)), elf_auditor_sha256=Evidence.sha256(Path(__file__)),
            source_after_sha256=Evidence.sha256(self.harness / "source_manifest.json"), cuda_compile_commands=cuda,
            torch_abi=subprocess.check_output([self.python, "-B", "-c", "import torch; print(torch.__version__,torch.version.cuda,torch._C._GLIBCXX_USE_CXX11_ABI)"], text=True, timeout=30).strip(),
            source_equivalence_policy="Every reused precision has identical native/common source hashes, recipe/build inputs and ABI; every mapped DSO is separately SHA-checked")
        Evidence.write(self.record, record)
        Evidence.write(self.harness / "binding.json", record)
        return record
