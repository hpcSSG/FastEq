"""Freeze scope and conservatively map historical source states, without running a GPU.

Historical equivalence requires byte-identical relevant wrapper, native and build inputs,
the same dtype, explicit runtime controls, and the same set of measured call boundaries.
The mapping declares required tests, never claims those tests have already run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile


class Serialization:
    @staticmethod
    def sha(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    @classmethod
    def digest(cls, value: object) -> str:
        return cls.sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())

    @staticmethod
    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    @staticmethod
    def csv(path: Path, rows: list[dict], columns: list[str]) -> None:
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True, ensure_ascii=False)
                                 if isinstance(value, (dict, list, tuple)) else value
                                 for key, value in row.items()})


class GitHistory:
    """Read bundle objects in an isolated bare repository; never touches project refs."""

    def __init__(self, repository: Path, bundle: Path):
        self.repository, self.bundle = repository.resolve(), bundle.resolve()
        if not self.repository.exists():
            self.repository.parent.mkdir(parents=True, exist_ok=True)
            self.run(["git", "clone", "--bare", str(self.bundle), str(self.repository)], 120)
        if self.git("rev-parse", "--is-bare-repository").strip() != b"true":
            raise ValueError("History analysis requires an isolated bare repository")
        self.blobs: dict[str, bytes] = {}

    @staticmethod
    def run(argv: list[str], timeout: int = 30) -> bytes:
        return subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).stdout

    def git(self, *args: str) -> bytes:
        return self.run(["git", "--git-dir=" + str(self.repository), *args])

    def resolve(self, ref: str) -> str:
        return self.git("rev-parse", ref + "^{commit}").decode().strip()

    def snapshot(self, ref: str) -> "SourceState":
        commit = self.resolve(ref)
        files, gitlinks = {}, {}
        for record in self.git("ls-tree", "-r", "-z", commit).split(b"\0"):
            if not record:
                continue
            header, name = record.split(b"\t", 1)
            mode, kind, oid = header.decode().split()
            name = name.decode()
            if kind == "commit":
                gitlinks[name] = oid
            elif kind == "blob" and SourceState.relevant(name):
                if oid not in self.blobs:
                    self.blobs[oid] = self.git("cat-file", "blob", oid)
                files[name] = self.blobs[oid]
        return SourceState(commit, files, gitlinks)


@dataclass
class SourceState:
    commit: str
    files: dict[str, bytes]
    gitlinks: dict[str, str]
    provenance: dict = field(default_factory=dict)

    WRAPPERS = ("fasteq/__init__.py", "fasteq/ops/__init__.py", "fasteq/ops/so3_linear.py",
                "fasteq/ops/eqlinear.py", "fasteq/cuda/__init__.py")
    BUILD_FILES = ("setup.py", "pyproject.toml", ".gitmodules", "fasteq/cuda/CMakeLists.txt",
                   "fasteq/cuda/src/eqlinear/CMakeLists.txt")
    BOUNDARIES = ("public_warm_no_grad", "public_warm_autograd", "public_dx",
                  "run_with_activation_prep_forward", "run_with_activation_prep_dx",
                  "prepared_gemm_forward", "prepared_gemm_dx", "cold_create", "prepare_weight",
                  "prepare_act", "flat_3d_fallback")

    @classmethod
    def relevant(cls, path: str) -> bool:
        return (path in cls.WRAPPERS + cls.BUILD_FILES or
                path.startswith("fasteq/cuda/src/eqlinear/") or
                path.startswith("fasteq/cuda/src/equi_linear") or
                path == "fasteq/triton/so3_linear.py")

    def selected(self, precision: str) -> dict[str, bytes]:
        prefixes = ("fasteq/cuda/src/eqlinear/common/", "fasteq/cuda/src/eqlinear/dispatch/",
                    f"fasteq/cuda/src/eqlinear/{precision}/")
        return {path: value for path, value in self.files.items()
                if path in self.WRAPPERS + self.BUILD_FILES or path.startswith(prefixes)}

    def manifest(self, precision: str, controls: dict | None = None) -> dict:
        selected = self.selected(precision)
        hashes = {name: Serialization.sha(value) for name, value in sorted(selected.items())}
        # Explicitly preserve an absent build entry instead of silently fixing P0.
        missing = [name for name in self.BUILD_FILES if name not in selected]
        math_files = {name: value for name, value in hashes.items()
                      if name.startswith("fasteq/cuda/src/eqlinear/") and not name.endswith("CMakeLists.txt")}
        wrapper_files = {name: hashes.get(name, "MISSING") for name in self.WRAPPERS}
        build_files = {name: hashes.get(name, "MISSING") for name in self.BUILD_FILES}
        contract = {"source_files": hashes, "submodule_gitlinks": self.gitlinks,
                    "precision": precision, "input_dtype": "float64" if precision == "fp64" else "float32",
                    "runtime_controls": controls or {}, "call_boundaries": self.BOUNDARIES,
                    "build_options": "source-defined defaults; actual compiler/ABI/DSO still unverified"}
        return {"identity_sha256": Serialization.digest(contract), "contract": contract,
                "native_sha256": Serialization.digest(math_files),
                "wrapper_sha256": Serialization.digest(wrapper_files),
                "build_sha256": Serialization.digest({"files": build_files, "gitlinks": self.gitlinks}),
                "missing_build_files": missing, "source_commit": self.commit,
                "reconstruction": self.provenance}


class PatchOverlay:
    """Apply selected archived text hunks in memory with exact old-context checks."""

    HEADER = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    @classmethod
    def apply(cls, files: dict[str, bytes], patch: bytes) -> list[str]:
        changed = []
        for section in re.split(rb"(?=^diff --git )", patch, flags=re.MULTILINE):
            if not section.startswith(b"diff --git "):
                continue
            lines = section.decode().splitlines(keepends=True)
            target = lines[0].rstrip().split(" b/", 1)[1]
            if not SourceState.relevant(target):
                continue
            if target not in files:
                raise ValueError("Archived patch adds/deletes a selected path; explicit reconstruction needed: " + target)
            old = files[target].decode().splitlines(keepends=True)
            output, cursor, index = [], 0, 1
            while index < len(lines):
                header = cls.HEADER.match(lines[index])
                if not header:
                    index += 1
                    continue
                old_start = int(header[1]) - 1
                old_count = int(header[2] or "1")
                new_count = int(header[4] or "1")
                if old_start < cursor:
                    raise ValueError("Overlapping archived patch hunks: " + target)
                output.extend(old[cursor:old_start])
                cursor = old_start
                removed = added = 0
                index += 1
                while index < len(lines) and not cls.HEADER.match(lines[index]):
                    line = lines[index]
                    if line.startswith("\\ No newline"):
                        raise ValueError("Non-newline patch requires explicit handling")
                    if line.startswith((" ", "-", "+")):
                        marker, content = line[0], line[1:]
                        if marker in " -":
                            if cursor >= len(old) or old[cursor] != content:
                                raise ValueError(f"Archived patch context mismatch in {target}:{cursor + 1}")
                            cursor += 1
                            removed += 1
                        if marker in " +":
                            output.append(content)
                            added += 1
                    elif line.strip():
                        raise ValueError("Unexpected archived patch body")
                    index += 1
                if (removed, added) != (old_count, new_count):
                    raise ValueError("Archived patch hunk count mismatch: " + target)
            output.extend(old[cursor:])
            files[target] = "".join(output).encode()
            changed.append(target)
        return changed


class ScopePolicy:
    """Explicit user exclusions remain rows; unknown math is held for source audit."""

    @staticmethod
    def apply(row: dict) -> dict:
        value = copy.deepcopy(row)
        value["discovery_status"] = value.pop("status", "DISCOVERED_UNTESTED")
        value["discovery_reason"] = value.pop("reason", "")
        value.update(scope_status="IN_SCOPE", runtime_status="UNTESTED", status="UNTESTED",
                     reason="User-confirmed scope; build, capability and backend need H20 evidence")
        identifier, family = value["variant_id"], value["family"]
        math = str(value.get("compute_format", "")).lower()
        excluded = None
        if "fp16" in family or "bf16" in family:
            excluded = "User excludes FP16/BF16, including FP32-input multi-component emulations"
        elif "tilelang" in family:
            excluded = "User excludes TileLang"
        elif ("1pass" in math or identifier in ("standalone_tf32_impl1", "standalone_tf32_impl2",
              "current_native_tf32_impl1", "current_native_tf32_impl2", "handwritten_tf32_gmm_tf32_v6")):
            excluded = "User prohibits TF32x1 execution"
        elif identifier == "torch_allow_tf32":
            excluded = "User prohibits TF32x1; allow_tf32 does not ensure a permitted multi-pass algorithm"
        if excluded:
            value.update(scope_status="EXCLUDED_BY_USER", status="EXCLUDED_BY_USER", reason=excluded)
        elif value["discovery_status"] in ("SOURCE_STUB", "NOT_IMPLEMENTED", "NOT_EQUI_LINEAR", "SOURCE_DIFFERENT_OPERATOR"):
            value.update(status=value["discovery_status"], reason=value["discovery_reason"])
        elif "unverified" in math or math == "unknown":
            value.update(status="PENDING_MATH_AUDIT",
                         reason="Do not execute before verifying that arithmetic is permitted by user scope")
        # API/source declarations never establish a runtime capability result.
        value["backend_verified"] = False
        return value


class HistoryMapping:
    CURRENT = "54b151fbf81aad0baa35d7b538f98c5c9b18e4b6"
    PREP = "26355126f3ccae1deb271bab4662a2080f728f59"
    P0 = "229a1ce76099a57cef6e81b5e27a076531782dfe"
    CANDIDATES = {
        1: "7a6a8f5", 2: "52b61b7", 3: "7c02b5f", 4: "1934511", 5: "71c56d5",
        6: "ff4697f", 7: "d526b2e", 8: "a474bac", 9: "d47908a", 10: "0344d2c",
        11: "10a9072", 12: "e88dc0f", 13: "c2e96ac", 14: "d171efe", 15: "b650b5d",
        16: "6d40681", 17: "8f1f967", 23: "28fb263", 24: "c3a9a4b", 25: "31189c6",
        26: "5fbdb46", 27: "afb5285", 28: "061aa6b", 29: "39df410", 30: PREP,
    }
    PATCHES = {19: "p19_runtime_patch.diff", 21: "p21_combined_python_patch.diff", 22: "p22_candidate.diff"}

    def __init__(self, history: GitHistory, handoff: Path):
        self.history, self.handoff = history, handoff
        self.rows, self.manifests, self.required = [], {}, {}

    def archive_blob(self, name: str) -> bytes:
        archive = self.handoff / "h100/fasteq_fix/archive/p1_p22_evidence.tar.gz"
        member = "doc/eqlinear_prep_evidence/" + name
        with tarfile.open(archive) as stream:
            info = stream.getmember(member)
            if not info.isfile() or info.size > (8 << 20):
                raise ValueError("Unsafe or oversized source archive member")
            return stream.extractfile(info).read()

    def candidate(self, label: int) -> SourceState:
        if label in self.CANDIDATES:
            return self.history.snapshot(self.CANDIDATES[label])
        base = self.history.snapshot(self.CANDIDATES[14])
        if label == 20:
            name = "p20_eqlinear_snapshot.py"
            payload = self.archive_blob(name)
            base.files["fasteq/ops/eqlinear.py"] = payload
            modified = ["fasteq/ops/eqlinear.py"]
        else:
            name = self.PATCHES[label]
            payload = self.archive_blob(name)
            modified = PatchOverlay.apply(base.files, payload)
        base.provenance = {"archive": "h100/fasteq_fix/archive/p1_p22_evidence.tar.gz",
                           "member": "doc/eqlinear_prep_evidence/" + name,
                           "member_sha256": Serialization.sha(payload), "base_commit": base.commit,
                           "modified_runtime_paths": modified, "context_verification": "EXACT"}
        return base

    def add(self, label: str, state: SourceState, note: str = "", controls: dict | None = None) -> None:
        for precision in ("tf32", "fp32", "fp64"):
            applicable_controls = {name: value for name, value in (controls or {}).items()
                                   if name != "FASTEQ_EQLINEAR_REUSE_SM90_PARAMS" or precision == "tf32"}
            manifest = state.manifest(precision, applicable_controls)
            digest = manifest["identity_sha256"]
            current = self.history.snapshot(self.CURRENT).manifest(precision)["identity_sha256"]
            variant = ("current_fasteq_" + precision if digest == current
                       else "historical_runtime_" + precision + "_" + digest[:12])
            self.manifests.setdefault(digest, manifest)
            self.required.setdefault(variant, {"variant_id": variant, "labels": [], "precision": precision,
                                               "manifest_sha256": digest, "source_commit": state.commit,
                                               "reconstruction": state.provenance,
                                               "runtime_controls": applicable_controls,
                                               "runtime_status": "UNTESTED", "requires_h20_rebuild": True})
            self.required[variant]["labels"].append(label)
            self.rows.append({"historical_label": label, "precision": precision, "source_commit": state.commit,
                              "reconstruction": state.provenance, "runtime_controls": applicable_controls,
                              "source_contract_sha256": digest, "native_sha256": manifest["native_sha256"],
                              "wrapper_sha256": manifest["wrapper_sha256"], "build_sha256": manifest["build_sha256"],
                              "tested_variant": variant, "mapping_status": "EXACT_SOURCE_CONTRACT_MATCH",
                              "runtime_status": "UNTESTED", "note": note,
                              "call_boundaries": list(SourceState.BOUNDARIES)})

    def run(self) -> None:
        for ref, member, path in (
            ("d171efe", "p14_recheck_p14_eqlinear_snapshot.py", "fasteq/ops/eqlinear.py"),
            ("8f1f967", "p18_p17_eqlinear_snapshot.py", "fasteq/ops/eqlinear.py"),
            ("8f1f967", "p18_p17_eqlinear_dispatch_snapshot.cpp", "fasteq/cuda/src/eqlinear/dispatch/eqlinear_dispatch.cpp"),
        ):
            if self.history.snapshot(ref).files[path] != self.archive_blob(member):
                raise ValueError("Historical ref disagrees with archived producer source: " + member)
        self.add("P0", self.history.snapshot(self.P0), "Missing eqlinear CMake must be patched only in isolated build; manifest preserves missing file")
        for label in range(1, 31):
            if label == 18:
                self.add("P18/P14_control", self.candidate(14), "P18 changes timing protocol, not a new production runtime")
                self.add("P18/P17_candidate", self.candidate(17), "P18 re-evaluates P17 with grouped timing")
            else:
                controls = {"FASTEQ_EQLINEAR_CACHE_OUTPUT_POINTERS": "1"} if label == 19 else (
                    {"FASTEQ_EQLINEAR_REUSE_SM90_PARAMS": "1"} if label == 22 else {})
                self.add("P" + str(label), self.candidate(label), controls=controls)
        self.add("current", self.history.snapshot(self.CURRENT))


class Registry:
    COLUMNS = ["variant_id", "family", "repo_commit", "source_hash", "impl_id", "actual_kernel",
               "input_dtype", "compute_format", "accumulator_dtype", "passes", "operation",
               "supported_shapes", "call_boundaries", "backend_verified", "scope_status",
               "discovery_status", "runtime_status", "status", "reason", "source_path", "evidence"]

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.handoff, self.output = args.handoff_root.resolve(), args.out_dir.resolve()
        if self.output.is_relative_to(self.handoff):
            raise ValueError("Registry output must be outside immutable handoff")
        self.inventory = json.loads(args.inventory.read_text())
        self.history = GitHistory(args.history_repo, self.handoff / "sources/fasteq_history.bundle")

    def original_variants(self) -> list[dict]:
        state = self.history.snapshot("4463b10c1fa290f76bd2cbb8213a6e9377a34e78")
        rows = []
        source_hash = Serialization.digest({name: Serialization.sha(value) for name, value in state.files.items()})
        for precision in ("fp32", "tf32x3", "fp64"):
            for implementation in ("public_auto", "serial_cta", "path_parallel", "group_parallel", "backward_x"):
                if precision == "tf32x3" and implementation == "public_auto":
                    continue  # Old public auto exposes ieee or prohibited tf32x1, no tf32x3 auto knob.
                operation = "dx" if implementation == "backward_x" else "forward"
                rows.append({"variant_id": f"historical_triton_{precision}_{implementation}",
                             "family": "historical_triton", "repo_commit": state.commit,
                             "source_hash": source_hash, "impl_id": implementation,
                             "actual_kernel": "UNVERIFIED_REQUIRES_H20_PROFILE",
                             "input_dtype": "float64" if precision == "fp64" else "float32",
                             "compute_format": {"fp32": "IEEE_FP32", "tf32x3": "TF32_3product", "fp64": "FP64"}[precision],
                             "accumulator_dtype": "float64" if precision == "fp64" else "float32",
                             "passes": "SOURCE_DECLARED_RUNTIME_UNVERIFIED", "operation": operation,
                             "supported_shapes": "UNVERIFIED_REQUIRES_PROBE", "call_boundaries": ["public_warm" if implementation == "public_auto" else "prepared_native_launch_with_alloc"],
                             "backend_verified": False, "status": "DISCOVERED_UNTESTED",
                             "reason": "Original Triton wrapper and native raw CUDA are distinct source entries",
                             "source_path": "git:" + state.commit + ":fasteq/triton/so3_linear.py",
                             "evidence": [{"path": "fasteq/triton/so3_linear.py", "lines": [419, 473, 529, 583, 685]}]})
        for precision in ("tf32x1", "fp64"):
            rows.append({"variant_id": "historical_original_cuda_" + precision, "family": "historical_original_cuda",
                         "repo_commit": state.commit, "source_hash": source_hash, "impl_id": "raw_cuda",
                         "actual_kernel": "UNVERIFIED_REQUIRES_H20_PROFILE",
                         "input_dtype": "float64" if precision == "fp64" else "float32",
                         "compute_format": "FP64_DMMA" if precision == "fp64" else "TF32_1pass",
                         "accumulator_dtype": "float64" if precision == "fp64" else "float32",
                         "passes": 1, "operation": "forward;dx", "supported_shapes": "FP64 source limits P=4; runtime probe required",
                         "call_boundaries": ["raw_3d_allocating"], "backend_verified": False,
                         "status": "DISCOVERED_UNTESTED", "reason": "Separate equi_linear TORCH_LIBRARY; isolate from current runtime",
                         "source_path": "git:" + state.commit + ":fasteq/cuda/src/equi_linear.cu", "evidence": []})
        return rows

    def run(self) -> dict:
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError("Registry output directory must be fresh")
        self.output.mkdir(parents=True, exist_ok=True)
        rows = [ScopePolicy.apply(row) for row in self.inventory["variants"]]
        mapping = HistoryMapping(self.history, self.handoff)
        mapping.run()
        rows.extend(ScopePolicy.apply(row) for row in self.original_variants())
        for requirement in mapping.required.values():
            if requirement["variant_id"].startswith("current_fasteq_"):
                continue
            precision = requirement["precision"]
            manifest = mapping.manifests[requirement["manifest_sha256"]]
            rows.append(ScopePolicy.apply({
                "variant_id": requirement["variant_id"], "family": "historical_fasteq_runtime",
                "repo_commit": requirement["source_commit"], "source_hash": requirement["manifest_sha256"],
                "impl_id": "source_defined_default", "actual_kernel": "UNVERIFIED_REQUIRES_H20_PROFILE",
                "input_dtype": "float64" if precision == "fp64" else "float32",
                "compute_format": {"tf32": "TF32_3product", "fp32": "FP32_FFMA", "fp64": "FP64_DMMA"}[precision],
                "accumulator_dtype": "float64" if precision == "fp64" else "float32",
                "passes": "SOURCE_DECLARED_RUNTIME_UNVERIFIED", "operation": "forward;dx",
                "supported_shapes": "UNVERIFIED_REQUIRES_PROBE", "call_boundaries": list(SourceState.BOUNDARIES),
                "backend_verified": False, "status": "DISCOVERED_UNTESTED",
                "reason": "Distinct wrapper/native/build/dtype contract, including archived candidate changes",
                "source_path": "git:" + requirement["source_commit"], "evidence": [
                    {"historical_labels": requirement["labels"], "manifest": "historical_sources.json#" + requirement["manifest_sha256"],
                     "reconstruction": manifest["reconstruction"], "runtime_controls": requirement["runtime_controls"]}]}))
        # Preserve coarse discovery rows as pointers rather than counting them as a tested runtime.
        aliases = {"historical_fasteq_p0": "P0", "historical_fasteq_p14": "P14",
                   "historical_fasteq_p27": "P27", "historical_fasteq_prep_head": "P30"}
        for row in rows:
            if row["variant_id"] in aliases:
                label = aliases[row["variant_id"]]
                targets = [r["tested_variant"] for r in mapping.rows if r["historical_label"] == label]
                row.update(status="DISCOVERY_ALIAS", reason="See historical_mapping.csv; required runtime states: " + ";".join(targets))
            elif row["variant_id"] == "historical_fasteq_original":
                row.update(status="DISCOVERY_ALIAS", reason="Split into historical_triton_* and historical_original_cuda_* entries")
            if row["variant_id"] == "lunaeq_current_fp64":
                row["compute_format"] = "FP64_SM90_MMA_SOURCE_SELECTED_FOR_U_GE_32"
        identifiers = [row["variant_id"] for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate variant IDs")
        labels = {row["historical_label"].split("/")[0] for row in mapping.rows}
        if not {"P" + str(i) for i in range(1, 31)} <= labels:
            raise ValueError("Incomplete historical label mapping")
        summary = {"registry_rows": len(rows), "scope_counts": dict(Counter(r["scope_status"] for r in rows)),
                   "status_counts": dict(Counter(r["status"] for r in rows)), "history_mapping_rows": len(mapping.rows),
                   "historical_required_source_dtype_contracts": len(mapping.required), "h20_tests_run": False,
                   "note": "Exact source equivalence declares coverage obligations only; runtime, ABI, DSO, accuracy and performance remain unverified."}
        Serialization.write(self.output / "variants.json", {"schema_version": 2, "scope_confirmed": True,
                            "inventory_sha256": Serialization.sha(self.args.inventory.read_bytes()),
                            "excluded": ["TF32x1", "FP16", "BF16", "TileLang", "CUDA_Graph"], "summary": summary,
                            "variants": rows, "family_discovery": self.inventory["families"], "warnings": self.inventory["warnings"]})
        Serialization.csv(self.output / "variants.csv", rows, self.COLUMNS)
        Serialization.csv(self.output / "historical_mapping.csv", mapping.rows, list(mapping.rows[0]))
        Serialization.write(self.output / "historical_sources.json", mapping.manifests)
        Serialization.write(self.output / "historical_required_runtimes.json", list(mapping.required.values()))
        Serialization.write(self.output / "registry_summary.json", summary)
        Serialization.write(self.output / "registry_provenance.json", {
            "tool_path": str(Path(__file__).resolve()), "tool_sha256": Serialization.sha(Path(__file__).read_bytes()),
            "input_inventory": str(self.args.inventory.resolve()), "input_sha256": Serialization.sha(self.args.inventory.read_bytes()),
            "history_repository": str(self.history.repository), "bundle_sha256": Serialization.sha(self.history.bundle.read_bytes()),
            "history_fingerprint_rule": "Byte-identical selected Python wrappers, dispatch/common/precision native, source-defined build files, gitlinks, dtype, applicable controls and boundary set; no runtime evidence inferred",
            "scope_file": str(self.args.session / "config/scope.yaml") if self.args.session else None,
            "scope_sha256": Serialization.sha((self.args.session / "config/scope.yaml").read_bytes()) if self.args.session else None})
        return summary

    @classmethod
    def main(cls) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, help="Read confirmed paths/scope and default coverage destination from this session")
        parser.add_argument("--inventory", type=Path)
        parser.add_argument("--handoff-root", type=Path)
        parser.add_argument("--history-repo", type=Path)
        parser.add_argument("--out-dir", type=Path)
        parser.add_argument("--selfcheck", action="store_true", help="Run pure-CPU contract and patch safety counterexamples")
        parser.add_argument("--revision", action="store_true", help="Write a new evidence-linked registry revision; preserve the original registry")
        parser.add_argument("--choices", type=Path)
        parser.add_argument("--historical-plan", type=Path)
        parser.add_argument("--source-math-audit", type=Path, help="Append independently checked source arithmetic adjudications")
        parser.add_argument("--historical-tuples", type=Path, help="Append finite documented current-source replay recipes")
        parser.add_argument("--macro-audit", type=Path, help="Separate preserved failed compiler attempts and reported profile recipes")
        parser.add_argument("--append-from", type=Path, help="Append immutable source/runtime evidence to an existing revision")
        parser.add_argument("--profile-contracts", type=Path)
        parser.add_argument("--legacy-entry-audit", type=Path)
        parser.add_argument("--public-autograd-audit", type=Path)
        parser.add_argument("--handwritten-build-audit", type=Path)
        args = parser.parse_args()
        if args.selfcheck:
            print(json.dumps(RegistrySelfcheck.run(), indent=2))
            return
        if args.append_from:
            if not args.session or not args.out_dir:
                parser.error("--append-from requires --session and a fresh --out-dir")
            print(json.dumps(RegistryEvidenceRevision(args).run(), indent=2))
            return
        if args.revision:
            if not args.session:
                parser.error("--revision requires --session")
            print(json.dumps(RegistryRevision(args).run(), indent=2))
            return
        if args.session:
            paths = json.loads((args.session / "config/paths.yaml").read_text())
            scope = json.loads((args.session / "config/scope.yaml").read_text())
            if scope["status"] != "USER_CONFIRMED" or not {"tf32x1", "fp16", "bf16", "tilelang", "cuda_graph"} <= set(scope["exclude"]):
                parser.error("Registry policy requires the currently confirmed user exclusions")
            args.handoff_root = args.handoff_root or Path(paths["HANDOFF_ROOT"])
            args.out_dir = args.out_dir or args.session / "coverage"
        if not all((args.inventory, args.handoff_root, args.history_repo, args.out_dir)):
            parser.error("Provide --inventory, --history-repo and either --session or --handoff-root/--out-dir")
        print(json.dumps(cls(args).run(), indent=2))


class RegistrySelfcheck:
    @staticmethod
    def run() -> dict:
        files = {"fasteq/ops/eqlinear.py": b"original wrapper\n", "setup.py": b"build\n"}
        state = SourceState("synthetic", files, {"3rdparty/cutlass": "synthetic"})
        identity = state.manifest("fp32")["identity_sha256"]
        checks = {}
        state.files["fasteq/ops/eqlinear.py"] = b"different wrapper\n"
        checks["wrapper_change_prevents_dedup"] = state.manifest("fp32")["identity_sha256"] != identity
        state.files["fasteq/ops/eqlinear.py"] = b"original wrapper\n"
        checks["dtype_change_prevents_dedup"] = state.manifest("fp64")["identity_sha256"] != identity
        checks["control_change_prevents_dedup"] = state.manifest("fp32", {"flag": "1"})["identity_sha256"] != identity
        state.files["setup.py"] = b"different build\n"
        checks["build_change_prevents_dedup"] = state.manifest("fp32")["identity_sha256"] != identity
        state.files["setup.py"] = b"build\n"
        checks["identical_contract_reuses_identity"] = state.manifest("fp32")["identity_sha256"] == identity
        patch = b"diff --git a/fasteq/ops/eqlinear.py b/fasteq/ops/eqlinear.py\n--- a/fasteq/ops/eqlinear.py\n+++ b/fasteq/ops/eqlinear.py\n@@ -1 +1 @@\n-original wrapper\n+candidate wrapper\n"
        PatchOverlay.apply(state.files, patch)
        checks["archived_candidate_applied"] = state.files["fasteq/ops/eqlinear.py"] == b"candidate wrapper\n"
        try:
            PatchOverlay.apply(state.files, patch)
            checks["mismatched_patch_rejected"] = False
        except ValueError:
            checks["mismatched_patch_rejected"] = True
        template = {"variant_id": "synthetic", "family": "standalone_fp16", "compute_format": "emulation", "status": "DISCOVERED"}
        checks["fp32_input_fp16_emulation_excluded"] = ScopePolicy.apply(template)["scope_status"] == "EXCLUDED_BY_USER"
        template.update(family="current_fasteq_native", variant_id="current_native_tf32_impl2")
        checks["native_x1_excluded"] = ScopePolicy.apply(template)["scope_status"] == "EXCLUDED_BY_USER"
        if not all(checks.values()):
            raise AssertionError(checks)
        return {"synthetic_cpu_only": True, "h20_tests_run": False, "checks": checks, "passed": True}


class RegistryRevision:
    """Append finite source contracts and evidence links without re-signing old rows."""

    def __init__(self, args):
        self.args, self.root = args, args.session.resolve()
        self.coverage = self.root / "coverage"

    @staticmethod
    def reference(path):
        path = Path(path)
        return {"path": str(path), "sha256": Serialization.sha(path.read_bytes())}

    def run(self):
        base_path = self.coverage / "variants.json"
        choices_path = self.args.choices or self.coverage / "standalone_existing_choices07/variants.json"
        history_path = self.args.historical_plan or self.coverage / "historical_execution_plan05.json"
        base, choices, history = [json.loads(path.read_text()) for path in (base_path, choices_path, history_path)]
        original_hash = self.reference(base_path)["sha256"]
        rows = copy.deepcopy(base["variants"])
        identifiers = {row["variant_id"] for row in rows}
        if len(identifiers) != len(rows) or len(rows) != 194:
            raise ValueError("Expected immutable original 194-row registry")
        additional = choices["variants"]
        if len(additional) != 157 or choices["default_registry_matrix"]["variants"] != 28:
            raise ValueError("Finite policy registry is not the reviewed choices07 scope")
        relations = []
        row_origins = {row["variant_id"]: str(base_path) for row in rows}
        for row in additional:
            if row["base_variant_id"] not in identifiers or row["variant_id"] in identifiers:
                raise ValueError("A named policy is missing its original base, or duplicates a contract")
            if ScopePolicy.apply(copy.deepcopy(row))["scope_status"] != "IN_SCOPE":
                raise ValueError("Named policy includes an excluded precision or selector")
            rows.append(copy.deepcopy(row))
            identifiers.add(row["variant_id"])
            row_origins[row["variant_id"]] = str(choices_path)
            relations.append({"variant_id": row["variant_id"], "relationship": "FINITE_EXISTING_POLICY_OF",
                              "target": row["base_variant_id"], "evidence": str(choices_path),
                              "semantic_contract_hash": row["semantic_contract_hash"]})
        tuple_registry = None
        if self.args.historical_tuples:
            tuple_registry = json.loads(self.args.historical_tuples.read_text())
            if tuple_registry["additional_contracts"] != 8 or tuple_registry["original_157_scope_rewritten"] or tuple_registry["H100_rerun"]:
                raise ValueError("Unexpected historical recipe scope or provenance")
            for row in tuple_registry["variants"]:
                if row["variant_id"] in identifiers or row["base_variant_id"] not in identifiers or row["scope_status"] != "IN_SCOPE":
                    raise ValueError("Historical recipe is duplicate, unbased or excluded")
                for evidence in row["named_choice_evidence"]:
                    path = Path(evidence["path"])
                    if Serialization.sha(path.read_bytes()) != evidence["sha256"] or path.read_text().splitlines()[evidence["line"] - 1].strip() != evidence["text"].strip():
                        raise ValueError("Historical recipe report evidence differs")
                rows.append(copy.deepcopy(row))
                identifiers.add(row["variant_id"])
                row_origins[row["variant_id"]] = str(self.args.historical_tuples)
                relations.append({"variant_id": row["variant_id"], "relationship": "DOCUMENTED_CURRENT_SOURCE_REPLAY_RECIPE_OF",
                    "target": row["base_variant_id"], "evidence": str(self.args.historical_tuples),
                    "semantic_contract_hash": row["semantic_contract_hash"]})
        math_amendment = None
        if self.args.source_math_audit:
            math_amendment = SourceMathAmendment(self.root, self.args.source_math_audit)
            math_amendment.apply(rows, relations)
        minimum_path = self.coverage / "native_bridge_minimum_summary01.json"
        minimum = json.loads(minimum_path.read_text())
        minimum_rows = {f"current_native_{row['precision']}_impl{row['impl']}": row for row in minimum["rows"]}
        history_rows = {row["variant_id"]: row for row in history["wrapper_tests"]}
        native_plan = self.root / "accuracy/native_selector/nativebatchcycle02/plan.json"
        full_path = native_plan.parent / "full_audit.json"
        full = json.loads(full_path.read_text()) if full_path.exists() else None
        if full and (not full.get("complete") or not full.get("full_registry_complete") or
                     full.get("expected_points") != 475 or full.get("expected_paths") != 1425 or
                     full.get("plan_hash") != json.loads(native_plan.read_text())["plan_hash"]):
            raise ValueError("Existing native full audit is incomplete or belongs to a different cycle")
        native_jobs = json.loads(native_plan.read_text())["jobs"] if full else []
        plans = []
        for path in sorted((self.root / "accuracy").glob("standalone_matrix/*/plan.json")):
            plan = json.loads(path.read_text())
            plans.append((path, {row["variant_id"] for row in plan["variants"]}))
        for row in rows:
            ident = row["variant_id"]
            row["static_observation"] = {key: row.get(key) for key in
                ("discovery_status", "discovery_reason", "source_hash", "scope_status")}
            observations = []
            if ident in history_rows:
                contract = history_rows[ident]
                row["execution_dependencies"] = {key: contract[key] for key in
                    ("native_keys", "dispatcher_key", "runtime_controls", "labels", "precision")}
                relations.append({"variant_id": ident, "relationship": "EXACT_HISTORY_WRAPPER_CONTRACT",
                                  "target": contract["manifest_sha256"], "evidence": str(history_path)})
                observations.append({"classification": "DEPENDENCY_PLAN_NOT_RUNTIME_RESULT", "evidence": str(history_path)})
            if ident in minimum_rows:
                record = minimum_rows[ident]
                observations.append({"classification": "MINIMUM_PROBE_ONLY", "status": record["status"],
                    "point": "SMALL_P1/512", "cells": record["cells"], "evidence": record["result"],
                    "sha256": record["result_sha256"], "kernel_trace": record["trace"],
                    "full_matrix_plan": str(self.root / "accuracy/native_selector/nativebatchcycle02/plan.json"),
                    "full_matrix_claimed_complete": False})
                if full:
                    matched = [job for job in native_jobs if job["precision"] == record["precision"] and job["impl"] == record["impl"]]
                    records = [{"point": job["point"], **full["records"][job["attempt"]]} for job in matched]
                    if len(records) != 25 or not all(item["verified"] for item in records):
                        raise ValueError("A native selector lacks its full25-point evidence")
                    observations.append({"classification": "FULL_UNIFIED_ACCURACY_AUDIT", "evidence": str(full_path),
                        "sha256": self.reference(full_path)["sha256"], "points": 25, "cells": 75,
                        "accuracy_pass_points": sum(item["accuracy_pass"] for item in records), "records": records,
                        "performance_measured": False})
                relations.append({"variant_id": ident, "relationship": "TEST_ONLY_CURRENT_OBJECT_BRIDGE",
                                  "target": record["actual_mapped_DSO"][0], "evidence": str(minimum_path)})
            for path, variants in plans:
                if ident in variants:
                    observations.append({"classification": "FROZEN_MATRIX_PLAN_NOT_RUNTIME_RESULT", "evidence": str(path)})
            if row["scope_status"] != "IN_SCOPE" and observations:
                raise ValueError("Excluded row unexpectedly has a native runtime execution link")
            row["runtime_observations"] = observations
            row["source_registry"] = row_origins[ident]
        output = (self.args.out_dir or self.coverage / "registry_revision02").resolve()
        if not output.is_relative_to(self.root):
            raise ValueError("Registry revisions must stay under the configured session")
        output.mkdir(parents=True, exist_ok=False)
        summary = {"registry_rows": len(rows), "scope_counts": dict(Counter(row["scope_status"] for row in rows)),
                   "original_rows_preserved": 194, "additional_finite_policy_rows": 157,
                   "standalone_default_plus_named_in_scope": 185, "historical_wrapper_contracts": len(history_rows),
                   "native_selectors_in_scope_with_minimum_evidence": len(minimum_rows),
                   "native_full_accuracy_complete": bool(full),
                   "native_full_accuracy_all_pass": full["accuracy_pass"] if full else False,
                   "all_existing_parameter_strategies_covered": False}
        if math_amendment:
            summary.update(source_math_reviewed_variants=13, source_math_new_user_exclusions=math_amendment.new_exclusions)
        if tuple_registry:
            summary.update(additional_historical_recipe_rows=8, standalone_total_finite_in_scope=193,
                           historical_recipe_full25_jobs=200, historical_recipe_full25_cells=300)
        value = {"schema_version": 3, "revision": output.name, "summary": summary,
                 "parents": {"original": self.reference(base_path), "finite_choices": self.reference(choices_path),
                             "historical_plan": self.reference(history_path), "native_minimum": self.reference(minimum_path)},
                 "variants": rows, "relations": relations,
                 "unresolved_dependencies": {"compile_knobs": choices["build_control_inventory"],
                     "named_source_aliases": choices["kernel_aliases_or_ignored_parameters"],
                     "preserved_compile_tuples": choices["preserved_compile_tuples"],
                     "history_native_groups": history["native_groups"], "history_dispatcher_groups": history["dispatcher_groups"],
                     "note": "Source contracts, dependency plans and actual runtime evidence remain separate; minimum probes never imply complete accuracy or performance."},
                 "H100_frozen_additional_gate": "UNVERIFIED_INPUT_HASH_MISMATCH",
                 "tool_sha256": Serialization.sha(Path(__file__).read_bytes())}
        if full:
            value["parents"]["native_full"] = self.reference(full_path)
        if math_amendment:
            value["parents"]["source_math"] = self.reference(self.args.source_math_audit)
            value["parents"]["independent_source_math_readback"] = self.reference(math_amendment.proof_path)
        if tuple_registry:
            value["parents"]["historical_finite_recipes"] = self.reference(self.args.historical_tuples)
        if self.args.macro_audit:
            macros = json.loads(self.args.macro_audit.read_text())
            if len(macros["failed_builds"]) != 2 or len(macros["reported_profiles"]) != 3:
                raise ValueError("Unexpected finite macro profile adjudication")
            value["parents"]["macro_profile_adjudication"] = self.reference(self.args.macro_audit)
            value["unresolved_dependencies"]["preserved_failed_builds"] = macros["failed_builds"]
            value["unresolved_dependencies"]["reported_current_source_profile_replays"] = macros["reported_profiles"]
            value["unresolved_dependencies"]["preserved_compile_tuples_note"] = "Original two nvcc logs are failed builds, not successful profiles; see separate adjudications and reported replay recipes."
            summary.update(preserved_failed_compile_commands=2, reported_macro_profiles_pending_H20=3)
        Serialization.write(output / "variants.json", value)
        Serialization.csv(output / "variants.csv", rows, sorted(set().union(*(row.keys() for row in rows))))
        Serialization.csv(output / "relationships.csv", relations, sorted(set().union(*(row.keys() for row in relations))))
        Serialization.write(output / "summary.json", summary)
        if self.reference(base_path)["sha256"] != original_hash:
            raise ValueError("Original registry changed during revision")
        return {"output": str(output), **summary, "original_registry_sha256_unchanged": original_hash}


class SourceMathAmendment:
    """Apply the user's arithmetic exclusion while retaining the original source row."""

    def __init__(self, root, path):
        self.path = path
        self.audit = json.loads(path.read_text())
        self.proof_path = root / "coverage/handwritten_source_math_independent01.json"
        proof = json.loads(self.proof_path.read_text())
        if proof["status"] != "PASS" or proof["audit_sha256"] != Serialization.sha(path.read_bytes()):
            raise ValueError("Source arithmetic amendment lacks matching independent readback")
        self.new_exclusions = 0

    def apply(self, rows, relations):
        by_id = {row["variant_id"]: row for row in rows}
        if len(self.audit["results"]) != 13:
            raise ValueError("Unexpected handwritten arithmetic scope")
        for result in self.audit["results"]:
            row = by_id[result["variant_id"]]
            source, snapshot = Path(result["source"]), Path(result["source_snapshot"])
            if (Serialization.sha(source.read_bytes()) != result["source_sha256"] or
                    Serialization.sha(snapshot.read_bytes()) != row["source_hash"] or result["execution_performed"]):
                raise ValueError("Math amendment source identity or no-execution claim differs")
            row["prior_scope_status"] = row["scope_status"]
            row["source_math_adjudication"] = {"classification": result["classification"],
                "products_per_unpadded_k_tile": result["products_per_unpadded_k_tile"],
                "explicit_tf32_conversion_active": result["explicit_tf32_conversion_active"],
                "action": result["action"], "source_evidence": result["source_evidence"],
                "audit": str(self.path), "runtime_backend_verified": False}
            classification = result["classification"]
            if classification == "TF32_SINGLE_PRODUCT":
                if result["products_per_unpadded_k_tile"] != 1:
                    raise ValueError("Inconsistent single-product source adjudication")
                self.new_exclusions += row["scope_status"] == "IN_SCOPE"
                row.update(scope_status="EXCLUDED_BY_USER", runtime_status="NOT_EXECUTED_USER_EXCLUDED",
                           discovery_status="SOURCE_MATH_EXCLUDED_USER_TF32_SINGLE_PRODUCT",
                           status="EXCLUDED_BY_USER", reason="User prohibits TF32x1; exact source/helper adjudication: " + result["reason"],
                           compute_format="TF32_SINGLE_PRODUCT_SOURCE_INPUT_ENCODING_RUNTIME_UNVERIFIED")
            elif classification == "TF32_THREE_PRODUCT":
                if result["products_per_unpadded_k_tile"] != 3:
                    raise ValueError("Inconsistent three-product source adjudication")
                row.update(discovery_status="SOURCE_MATH_PERMITTED_BUILD_RUNTIME_PENDING", status="PENDING_BUILD_OR_PROBE",
                           compute_format="TF32_THREE_PRODUCT_SOURCE_RUNTIME_UNVERIFIED")
            elif classification == "SOURCE_GEMM_BODY_COMMENTED_OUT":
                if result["products_per_unpadded_k_tile"] != 0:
                    raise ValueError("Commented-out body has an active product")
                row.update(discovery_status="SOURCE_GEMM_BODY_COMMENTED_OUT", status="PENDING_BUILD_OR_PROBE",
                           reason=result["reason"] + " Build/runtime capability remains pending.")
            else:
                raise ValueError("Unknown source math adjudication")
            relations.append({"variant_id": row["variant_id"], "relationship": "SOURCE_MATH_SCOPE_ADJUDICATION",
                              "target": classification, "evidence": str(self.path)})


class RegistryEvidenceRevision:
    """Append facts to a prior revision while retaining its original evidence epochs."""

    def __init__(self, args):
        self.args = args

    @staticmethod
    def link(path):
        return {"path": str(path.resolve()), "sha256": Serialization.sha(path.read_bytes())}

    def run(self):
        args = self.args
        parent_link = self.link(args.append_from)
        value = copy.deepcopy(json.loads(args.append_from.read_text()))
        rows, relations = value["variants"], value["relations"]
        by_id = {row["variant_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("Duplicate parent registry rows")
        additional, added_links = 0, dict(value.get("append_evidence", {}))
        if args.profile_contracts:
            profiles = json.loads(args.profile_contracts.read_text())
            if profiles["contracts"] != 21 or profiles["profiles"] != 3 or profiles["full25_jobs"] != 525 or profiles["full25_cells"] != 1350:
                raise ValueError("Unexpected reviewed source-active macro recipe scope")
            for original in profiles["variants"]:
                row = copy.deepcopy(original)
                ident, base = row["variant_id"], row["base_variant_id"]
                if ident in by_id or base not in by_id or ScopePolicy.apply(copy.deepcopy(row))["scope_status"] != "IN_SCOPE":
                    raise ValueError("Duplicate, unbased or excluded profile contract")
                link = row["profile_record"]
                if self.link(Path(link["path"])) != link:
                    raise ValueError("Fresh compiler-profile record changed")
                profile = json.loads(Path(link["path"]).read_text())
                if profile["status"] != "BUILD_PASS_REQUIRES_H20_PROBE" or Serialization.sha(Path(profile["binary"]).read_bytes()) != profile["binary_hash"]:
                    raise ValueError("Profile lacks a source-bound fresh DSO")
                row.update(source_registry=str(args.profile_contracts),
                    historical_reproduction="CURRENT_PRESERVED_SOURCE_REPLAY_ONLY_ORIGINAL_PRODUCER_UNPROVEN",
                    runtime_observations=[{"classification": "FRESH_SM90A_BUILD_ONLY_NUMERICAL_PROBE_SEPARATE", "evidence": link,
                                           "binary_sha256": profile["binary_hash"], "performance_measured": False}])
                rows.append(row)
                by_id[ident] = row
                relations.append({"variant_id": ident, "relationship": "SOURCE_ACTIVE_PROFILE_RECIPE_OF", "target": base,
                                  "evidence": str(args.profile_contracts), "build_profile": row["build_profile"]})
                additional += 1
            added_links["source_active_macro_contracts"] = self.link(args.profile_contracts)
        if args.legacy_entry_audit:
            audit = json.loads(args.legacy_entry_audit.read_text())
            if audit.get("gpu_execution") is not False or audit.get("no_tf32x1_executed") is not True:
                raise ValueError("Legacy source-only audit has an unexpected execution claim")
            for contract in audit["contracts"]:
                path = Path(contract["source"])
                if Serialization.sha(path.read_bytes()) != contract["source_sha256"]:
                    raise ValueError("Legacy source adjudication changed")
                ident = contract["id"]
                if ident.startswith("historical_triton_"):
                    if ident.endswith("single_tf32"):
                        continue  # Forbidden control remains explicit in source evidence, never execute it.
                    precision = "fp64" if ident.endswith("fp64") else "tf32x3" if "ieee_and_tf32x3" in ident else "fp32"
                    selected = [row for key, row in by_id.items() if key.startswith("historical_triton_" + precision + "_")]
                    if "ieee_and_tf32x3" in ident:
                        selected += [row for key, row in by_id.items() if key.startswith("historical_triton_fp32_") and not key.endswith("public_auto")]
                    if ident.endswith("auto_f32"):
                        selected = [by_id["historical_triton_fp32_public_auto"]]
                else:
                    selected = [by_id[ident]] if ident in by_id else []
                for row in selected:
                    row.setdefault("source_entry_observations", []).append({"classification": "SOURCE_ONLY_NOT_GPU_CAPABILITY_VERDICT",
                        "contract": contract, "evidence": self.link(args.legacy_entry_audit)})
            added_links["legacy_source_entry_adjudication"] = self.link(args.legacy_entry_audit)
        if args.public_autograd_audit:
            audit = json.loads(args.public_autograd_audit.read_text())
            if audit.get("complete") is not True or audit.get("independent_readback_passed") is not True:
                raise ValueError("Public autograd minimal probe lacks independent verification")
            for precision, record in audit["records"].items():
                path = Path(record["result"]["path"])
                if self.link(path) != record["result"]:
                    raise ValueError("Original public autograd probe changed")
                row = by_id["standalone_" + precision + "_impl0"]
                row.setdefault("runtime_observations", []).append({"classification": "MINIMUM_PUBLIC_AUTOGRAD_CAPABILITY_ONLY",
                    "evidence": self.link(args.public_autograd_audit), "record": record,
                    "scope": "Original forward impl0 at SMALL_P1/512 only; no_grad forward passed; nonzero dy public x-gradient chain absent in two actual autograd.grad calls",
                    "dispatch_note": "Catch-all math registration is present, including AutogradCUDA math dispatch; observed missing grad_fn is decisive. Manual dx and native test bridge are separate APIs."})
            added_links["original_public_autograd_minimum"] = self.link(args.public_autograd_audit)
        if args.handwritten_build_audit:
            audit = json.loads(args.handwritten_build_audit.read_text())
            if audit.get("complete") is not True or audit.get("gpu_execution") is not False:
                raise ValueError("Handwritten build evidence is partial or claims GPU execution")
            for record in audit["records"]:
                source_path = Path(record["source_manifest"])
                if self.link(source_path)["sha256"] != record["source_manifest_sha256"]:
                    raise ValueError("Handwritten staged source receipt changed")
                source = json.loads(source_path.read_text())
                kind = source["kind"]
                ident = "historical_original_cuda_fp64" if kind == "original_cuda_fp64_4463b10" else "handwritten_" + kind
                if ident not in by_id or record["status"] not in ("PASS", "BUILD_FAILED"):
                    raise ValueError("Unknown handwritten source build contract")
                if record["status"] == "PASS" and Serialization.sha(Path(record["binary"]["path"]).read_bytes()) != record["binary"]["sha256"]:
                    raise ValueError("Handwritten fresh DSO drift")
                by_id[ident].setdefault("build_observations", []).append({"classification": "CPU_BUILD_READBACK_ONLY_RUNTIME_PENDING",
                    "evidence": self.link(args.handwritten_build_audit), "record": record,
                    "original_producer_match": "UNPROVEN", "backend_verified": False})
            added_links["handwritten_fp64_independent_build_readback"] = self.link(args.handwritten_build_audit)
        output = args.out_dir.resolve()
        if not output.is_relative_to(args.session.resolve()):
            raise ValueError("Append-only registry evidence must stay in the configured session")
        output.mkdir(parents=True, exist_ok=False)
        value.update(revision=output.name, append_parent=parent_link, append_evidence=added_links,
                     append_tool_sha256=Serialization.sha(Path(__file__).read_bytes()))
        prior_profiles = value["summary"].get("additional_source_active_profile_contracts", 0)
        prior_standalone = value["summary"].get("standalone_total_finite_in_scope", 193)
        value["summary"].update(registry_rows=len(rows), scope_counts=dict(Counter(row["scope_status"] for row in rows)),
            additional_source_active_profile_contracts=prior_profiles + additional, standalone_total_finite_in_scope=prior_standalone + additional,
            source_active_profile_full25_jobs=525 if additional else value["summary"].get("source_active_profile_full25_jobs", 0),
            source_active_profile_full25_cells=1350 if additional else value["summary"].get("source_active_profile_full25_cells", 0),
            previous_revision_bytes_unchanged=True)
        Serialization.write(output / "variants.json", value)
        Serialization.csv(output / "variants.csv", rows, sorted(set().union(*(row.keys() for row in rows))))
        Serialization.csv(output / "relationships.csv", relations, sorted(set().union(*(row.keys() for row in relations))))
        Serialization.write(output / "summary.json", value["summary"])
        if self.link(args.append_from) != parent_link:
            raise ValueError("Parent registry bytes changed")
        return {"output": str(output), **value["summary"]}


if __name__ == "__main__":
    Registry.main()
