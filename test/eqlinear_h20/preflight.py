"""Read-only H20 discovery and CPU handoff checks; never builds or launches kernels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


from common import Evidence, CommandRunner


class Preflight:
    PACKAGES = ("torch", "numpy", "setuptools", "wheel", "packaging", "cmake", "ninja", "pytest",
                "cuequivariance", "cuequivariance_torch", "triton", "e3nn", "fasteq", "LunaEquivariance")
    ENV_PROGRAM = """
import importlib.metadata as m, importlib.util, json, sys
result = {'executable':sys.executable, 'version':sys.version, 'packages':{}}
for name in sys.argv[1:]:
    try:
        dist=m.distribution(name)
        result['packages'][name]={'version':dist.version, 'root':str(dist.locate_file('')), 'direct_url':dist.read_text('direct_url.json')}
    except m.PackageNotFoundError: result['packages'][name]={'status':'MISSING'}
import torch
result['torch']={'version':torch.__version__, 'cuda':torch.version.cuda, 'cxx11_abi':torch._C._GLIBCXX_USE_CXX11_ABI, 'file':torch.__file__}
for name in ['fasteq', 'LunaEquivariance']:
    spec=importlib.util.find_spec(name)
    result[name]={'origin':spec.origin, 'paths':list(spec.submodule_search_locations or [])} if spec else None
print(json.dumps(result, indent=2))
"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output = args.output.resolve()
        self.handoff = args.handoff_root.resolve()
        if self.output.is_relative_to(self.handoff):
            raise ValueError("Output must be outside the immutable handoff")
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.iterdir()):
            raise FileExistsError("Use a fresh preflight output directory")
        self.runner = CommandRunner(self.output / "logs")
        self.results = {}

    def run(self) -> int:
        args = self.args
        jobs = [
            ("handoff_verify", [str(args.fast_python), str(self.handoff / "tools/handoff.py"), "verify", "--root", str(self.handoff)], 60),
            ("handoff_catalog", [str(args.fast_python), str(self.handoff / "tools/handoff.py"), "catalog", "--root", str(self.handoff)], 30),
            ("h100_recheck", [str(args.fast_python), str(self.handoff / "tools/recheck_h100.py"), "--root", str(self.handoff), "--out-dir", str(self.output / "h100_recheck")], 90),
            ("fast_environment", [str(args.fast_python), "-c", self.ENV_PROGRAM, *self.PACKAGES], 30),
            ("luna_environment", [str(args.luna_python), "-c", self.ENV_PROGRAM, *self.PACKAGES], 30),
            ("nvcc_version", ["nvcc", "--version"], 10),
            ("ncu_version", ["ncu", "--version"], 10),
            ("ncu_sections", ["ncu", "--list-sections"], 20),
            ("ncu_metrics", ["ncu", "--query-metrics"], 30),
            ("nsys_version", ["nsys", "--version"], 10),
            ("gpu_topology", ["nvidia-smi", "topo", "-m"], 10),
            ("gpu_details", ["nvidia-smi", "-q", "-x"], 10),
            ("cpu", ["lscpu"], 10),
            ("processes", ["ps", "-eo", "pid,ppid,lstart,etime,args"], 10),
        ]
        for name, argv, deadline in jobs:
            self.results[name] = self.runner.run(name, argv, deadline, environment={"CUDA_VISIBLE_DEVICES": ""} if name in ("fast_environment", "luna_environment", "handoff_verify", "handoff_catalog", "h100_recheck") else {})
        for index in range(2):
            self.results[f"gpu_sample_{index}"] = self.runner.run(f"gpu_sample_{index}", [
                "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,mig.mode.current",
                "--format=csv,nounits"], 10)
            self.runner.run(f"gpu_processes_{index}", ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], 10)
            if index == 0:
                time.sleep(1.5)
        for name, root in (("fasteq", args.fasteq_root), ("luna", args.luna_root)):
            for suffix, command in (("head", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain=v1"]),
                                    ("patch", ["diff", "HEAD", "--binary"]), ("submodules", ["submodule", "status"])):
                self.runner.run(f"{name}_{suffix}", ["git", "-C", str(root), *command], 20)
        self.runner.run("bundle_refs", ["git", "bundle", "list-heads", str(self.handoff / "sources/fasteq_history.bundle")], 20)
        self.runner.run("tracked_eqlinear_cmake", ["git", "-C", str(args.fasteq_root), "ls-files", "fasteq/cuda/src/eqlinear/CMakeLists.txt"], 10)
        native = [{"path": str(p), "bytes": p.stat().st_size, "sha256": Evidence.sha256(p),
                   "status": "PRESENT_BUILD_PROVENANCE_UNVERIFIED"}
                  for p in sorted((args.fasteq_root / "fasteq/cuda").glob("*.so"))]
        stat = os.statvfs(self.output)
        Evidence.write(self.output / "env/inventory.json", {
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "disk_available_bytes": stat.f_bavail * stat.f_frsize,
            "native_binaries": native, "gpu_work_started": False,
            "handoff_manifest_sha256": Evidence.sha256(self.handoff / "manifests/files.json"),
            "goal_sha256": Evidence.sha256(self.handoff / "GOAL_EQLINEAR_H20.md"),
            "tool_sha256": Evidence.sha256(Path(__file__))})
        Evidence.write(self.output / "summary.json", {
            "stage": "H20-00", "status": "INCOMPLETE_CONFIGURATION_PENDING",
            "checks": {name: row["status"] for name, row in self.results.items()},
            "note": "CPU recheck success is data integrity only; inspect strict_task_g_passed separately. No H20 performance measured."})
        return 0 if self.results["handoff_verify"]["status"] == self.results["h100_recheck"]["status"] == "PASS" else 1

    @classmethod
    def main(cls) -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--fasteq-root", type=Path, required=True)
        parser.add_argument("--luna-root", type=Path, required=True)
        parser.add_argument("--handoff-root", type=Path, required=True)
        parser.add_argument("--fast-python", type=Path, required=True)
        parser.add_argument("--luna-python", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True, help="Fresh small local discovery directory; not a confirmed benchmark ARTIFACT_ROOT")
        return cls(parser.parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(Preflight.main())
