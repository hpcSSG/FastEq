"""Create a configured H20 session or run a bounded command inside its environment."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from common import CommandRunner, Evidence


class Session:
    DIRECTORIES = ("config", "env", "manifests", "coverage", "fixtures", "accuracy", "raw", "timing",
                   "profiles/torch", "profiles/nsys", "profiles/ncu", "roofline", "logs", "tables", "reports", "build", "tmp")
    CACHES = {"CUDA_CACHE_PATH": "cuda", "TORCH_EXTENSIONS_DIR": "torch_extensions", "TRITON_CACHE_DIR": "triton",
              "LUNAEQ_CACHE_DIR": "lunaeq", "UV_CACHE_DIR": "uv", "PIP_CACHE_DIR": "pip", "XDG_CACHE_HOME": "xdg",
              "MPLCONFIGDIR": "matplotlib"}

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.paths = json.loads((self.root / "config/paths.yaml").read_text())

    @classmethod
    def create(cls, args: argparse.Namespace) -> Session:
        root = args.root.resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("New sessions must have fresh directories")
        root.mkdir(parents=True, exist_ok=True)
        for directory in cls.DIRECTORIES:
            (root / directory).mkdir(parents=True, exist_ok=True)
        for path in (args.fasteq_root, args.luna_root, args.handoff_root):
            if not path.is_dir():
                raise FileNotFoundError(path)
        if not args.python.is_file():
            raise FileNotFoundError(args.python)
        paths = {"FASTEQ_ROOT": str(args.fasteq_root.resolve()), "LUNAEQ_ROOT": str(args.luna_root.resolve()),
                 "OPT_ROOT": str(args.handoff_root.resolve() / "source_snapshots/standalone"),
                 "FASTEQ_PY": str(args.python.absolute()), "LUNAEQ_PY": str(args.python.absolute()),
                 "ARTIFACT_ROOT": str(root.parent), "RUN_ROOT": str(root), "HANDOFF_ROOT": str(args.handoff_root.resolve()),
                 "H100_EVIDENCE_ROOT": str(args.handoff_root.resolve() / "h100"),
                 "SOURCE_REFS": {name: subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
                                 for name, repo in (("fasteq", args.fasteq_root), ("luna_020", args.luna_root))}}
        paths["SOURCE_REFS"]["luna_010"] = "5a24ffeb74ecf74de95211c72acfaa1f342126c3 (handoff source snapshot)"
        Evidence.write(root / "config/paths.yaml", paths)
        Evidence.write(root / "config/scope.yaml", {
            "status": "USER_CONFIRMED", "authorization": "User answers in this thread, 2026-11-07",
            "include": ["current_fasteq", "historical_P1_P30_semantic_dedup", "all_existing_standalone_in_scope",
                        "original_triton", "original_cuda_fp64", "luna_010", "luna_020", "torch_ieee_fp32", "torch_fp64"],
            "exclude": ["tf32x1", "fp16", "bf16", "tilelang", "cuda_graph"],
            "exclusions_override_goal_default_diagnostic_timing": True,
            "precisions": ["tf32x3", "fp32", "fp64"], "rounds": 3, "groups_per_point_min": 20,
            "operations": ["forward_no_grad", "forward_autograd", "grad_x", "grad_w_if_supported", "grad_xw_if_supported"],
            "boundaries": ["public_warm", "run_with_activation_prep", "prepared_gemm_if_api_supports", "cold_create", "prepare_weight", "prepare_act", "JIT_cold", "layout_components"],
            "cache_modes": ["warm_same_buffer", "rotating_inputs_representatives_above_2x_L2"],
            "gpu_policy": "Any GPU twice confirmed idle; do not change clocks/MIG/MPS or other users' tasks",
            "h100_access": "historical_evidence_only_no_rerun", "system_python_mutation": "FORBIDDEN",
            "production_defaults_changes": "FORBIDDEN", "merge_push": "FORBIDDEN", "new_algorithm_optimization": "OUT_OF_SCOPE"})
        session = cls(root)
        environment = session.environment()
        (root / "config/environment.sh").write_text("\n".join(f"export {key}={shlex.quote(value)}" for key, value in environment.items()) + "\n")
        Evidence.write(root / "manifests/handoff.json", {"manifest_sha256": Evidence.sha256(args.handoff_root / "manifests/files.json"),
                                                        "goal_sha256": Evidence.sha256(args.handoff_root / "GOAL_EQLINEAR_H20.md")})
        return session

    def environment(self) -> dict[str, str]:
        environment = {key: str(self.root / "cache" / suffix) for key, suffix in self.CACHES.items()}
        environment.update(TMPDIR=str(self.root / "tmp"), TMP=str(self.root / "tmp"), TEMP=str(self.root / "tmp"))
        for value in environment.values():
            Path(value).mkdir(parents=True, exist_ok=True)
        environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", UV_LINK_MODE="copy",
                           PYTHONPYCACHEPREFIX=str(self.root / "cache/pycache"), MAX_JOBS="2", CMAKE_BUILD_PARALLEL_LEVEL="2",
                           LUNAEQ_BUILD_TIMEOUT="240", TORCH_CUDA_ARCH_LIST="9.0a",
                           PATH=str(Path(self.paths["FASTEQ_PY"]).parent) + os.pathsep + os.environ["PATH"])
        return environment

    def runner(self) -> CommandRunner:
        return CommandRunner(self.root / "logs", self.environment())

    def idle_gpu(self, name: str, preferred: int | None = None) -> str:
        samples = []
        previous_idle = set()
        policy_path = self.root / "config/gpu_idle_baselines.json"
        policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
        limits = {uuid: min(256, max(16, int(item["max_memory_mib"]))) for uuid, item in policy.items()}
        for index in range(6):
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,nounits"], text=True, timeout=10)
            processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,nounits"], text=True, timeout=10)
            data = {"gpus": list(csv.DictReader(io.StringIO(output), skipinitialspace=True)), "processes": processes,
                    "idle_memory_limits_mib": limits, "default_idle_memory_limit_mib": 16}
            samples.append(data)
            busy = {r["gpu_uuid"] for r in csv.DictReader(io.StringIO(processes), skipinitialspace=True)}
            idle = {row["uuid"] for row in data["gpus"]
                    if int(row["memory.used [MiB]"]) <= limits.get(row["uuid"], 16) and int(row["utilization.gpu [%]"]) == 0 and row["uuid"] not in busy
                    and (preferred is None or int(row["index"]) == preferred)}
            candidates = previous_idle & idle
            if candidates:
                Evidence.write(self.root / "env" / f"{name}_idle_samples.json", samples)
                return sorted(candidates)[0]
            previous_idle = idle
            if index < 5:
                time.sleep(1.5)
        Evidence.write(self.root / "env" / f"{name}_idle_samples.json", samples)
        raise RuntimeError("No twice-consecutively-idle GPU in requested set after six bounded samples")

    @classmethod
    def main(cls) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        sub = parser.add_subparsers(dest="command", required=True)
        init = sub.add_parser("init")
        init.add_argument("--root", type=Path, required=True)
        for flag in ("fasteq-root", "luna-root", "handoff-root", "python"):
            init.add_argument("--" + flag, type=Path, required=True)
        run = sub.add_parser("run")
        run.add_argument("--root", type=Path, required=True)
        run.add_argument("--name", required=True)
        run.add_argument("--deadline", type=float, required=True)
        run.add_argument("--cwd", type=Path)
        run.add_argument("--gpu", type=int, help="Require this physical GPU twice idle and select by UUID")
        run.add_argument("argv", nargs=argparse.REMAINDER)
        args = parser.parse_args()
        if args.command == "init":
            print(cls.create(args).root)
            return 0
        session = cls(args.root)
        argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        if not argv:
            parser.error("run requires command after --")
        environment = {"CUDA_VISIBLE_DEVICES": ""}
        if args.gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = session.idle_gpu(args.name, args.gpu)
        result = session.runner().run(args.name, argv, args.deadline, args.cwd, environment)
        return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(Session.main())
