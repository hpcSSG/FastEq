"""Capture source identity and install only freshly built current FastEq artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sysconfig

from common import Evidence
from session import Session


class BuildEvidence:
    def __init__(self, session):
        self.session = session
        self.source = Path(session.paths["FASTEQ_ROOT"])

    def source_manifest(self):
        tracked = subprocess.check_output(["git", "-C", str(self.source), "ls-files", "-z"], timeout=20).decode().split("\0")
        files = {name: Evidence.sha256(self.source / name) for name in tracked if name and
                 ((name.startswith("fasteq/cuda/") or name.startswith("fasteq/ops/")) or
                  name in ("setup.py", "pyproject.toml", ".gitmodules", "fasteq/__init__.py")) and (self.source / name).is_file()}
        return {"commit": subprocess.check_output(["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True, timeout=10).strip(),
                "files": files, "source_hash": Evidence.fingerprint(files),
                "submodules": subprocess.check_output(["git", "-C", str(self.source), "submodule", "status"], text=True, timeout=20)}

    def install(self, directory, receipt):
        status = json.loads(receipt.read_text())
        if status["status"] != "PASS" or status["returncode"] != 0:
            raise ValueError("Build must have a terminal successful receipt")
        native = self.source / "fasteq/cuda"
        pairs = [(directory / "_cuda.so", native / ("_cuda" + sysconfig.get_config_var("EXT_SUFFIX")))]
        pairs += [(directory / ("libfasteq_eqlinear_" + kind + ".so"), native / ("libfasteq_eqlinear_" + kind + ".so"))
                  for kind in ("tf32", "fp32", "fp64")]
        if any(not source.is_file() for source, _ in pairs):
            raise FileNotFoundError("All four fresh artifacts are required")
        backup = self.session.root / "manifests/preexisting_native"
        backup.mkdir(exist_ok=False)
        files = []
        for source, target in pairs:
            row = {"build_path": str(source), "installed_path": str(target), "sha256": Evidence.sha256(source)}
            if target.exists():
                previous = backup / target.name
                shutil.copy2(target, previous)
                row["previous"] = {"path": str(previous), "sha256": Evidence.sha256(previous)}
            temporary = target.with_suffix(".h20_rebuild_partial")
            shutil.copy2(source, temporary)
            temporary.replace(target)
            if Evidence.sha256(target) != row["sha256"]:
                raise RuntimeError("Installed binary differs from fresh artifact")
            files.append(row)
        manifest = {"source": self.source_manifest(), "build_receipt": str(receipt),
                    "build_receipt_sha256": Evidence.sha256(receipt), "binaries": files,
                    "system_python_mutated": False, "note": "Direct CMake build; existing editable metadata preserved"}
        Evidence.write(self.session.root / "manifests/current_build.json", manifest)
        return manifest

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--install-from", type=Path)
        parser.add_argument("--build-receipt", type=Path)
        args = parser.parse_args()
        build = cls(Session(args.session))
        if args.install_from:
            if args.build_receipt is None:
                parser.error("--install-from requires --build-receipt")
            build.install(args.install_from, args.build_receipt)
        else:
            Evidence.write(build.session.root / "manifests/current_source_observation.json", build.source_manifest())
        return 0


if __name__ == "__main__":
    raise SystemExit(BuildEvidence.main())
