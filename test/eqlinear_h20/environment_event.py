"""Immutable dependency-change window and affected process receipt inventory."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from importlib.metadata import version
import json
from pathlib import Path
import re

from common import Evidence
from session import Session


class EnvironmentEvent:
    @classmethod
    def witness(cls, args):
        import importlib
        import sys
        modules = {}
        for name in ("numpy", "numpy.typing", "numpy._core._multiarray_umath", "scipy", "cuequivariance", "torch", "torch._C"):
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            modules[name] = {"origin": str(path), "sha256": Evidence.sha256(path),
                             "version": str(getattr(module, "__version__", "not_declared"))}
        if args.output.exists():
            raise FileExistsError(args.output)
        Evidence.write(args.output, {"classification": "DEPENDENCY_WITNESS_NOT_GPU_MEASUREMENT",
            "label": args.witness_label, "utc": datetime.now().astimezone().isoformat(), "python": sys.executable,
            "modules": modules, "active_distribution_versions": {name: version(name) for name in
                ("numpy", "scipy", "torch", "cuequivariance", "matplotlib", "pillow")},
            "tool_sha256": Evidence.sha256(Path(__file__))})
        return 0

    @classmethod
    def run(cls, args):
        session = Session(args.session)
        install_path = session.root / "logs/plotting_dependency_install01.json"
        restore_path = session.root / "logs/plotting_dependency_restore_numpy01.json"
        install, restore = json.loads(install_path.read_text()), json.loads(restore_path.read_text())
        start = datetime.fromisoformat(install["start_utc"])
        end = datetime.fromisoformat(restore["start_utc"]) + timedelta(seconds=restore["elapsed_seconds"])
        rows = []
        for path in sorted((session.root / "logs").glob("*.json")):
            if path.name.endswith(".running.json"):
                continue
            record = json.loads(path.read_text())
            if "start_utc" not in record or not any(session.paths["FASTEQ_PY"] in item for item in record.get("argv", [])):
                continue
            when = datetime.fromisoformat(record["start_utc"])
            if not start-timedelta(seconds=2) <= when <= end+timedelta(seconds=2):
                continue
            stderr = Path(record["stderr"]["path"])
            text = stderr.read_text()
            failed_import = record["status"] != "PASS" and "numpy" in text and ("ImportError" in text or "ModuleNotFoundError" in text)
            rows.append({"name": path.stem, "receipt": str(path), "receipt_sha256": Evidence.sha256(path),
                "start_utc": record["start_utc"], "status": record["status"], "argv": record["argv"],
                "stderr": {"path": str(stderr), "sha256": Evidence.sha256(stderr)},
                "numpy_version_in_process": "NOT_RECORDED; do not infer solely from start time",
                "numpy_import_error": failed_import,
                "adjudication": "PRESERVE_FAILED_ATTEMPT_AND_RETRY_FRESH" if failed_import else "KEEP_RAW_PENDING_ISOLATED_OVERLAY_VERIFICATION"})
        import numpy
        import numpy.typing
        import scipy.linalg
        import cuequivariance
        restored = {"version": numpy.__version__, "origin": str(Path(numpy.__file__).resolve()),
                    "sha256": Evidence.sha256(Path(numpy.__file__)),
                    "typing_origin": str(Path(numpy.typing.__file__).resolve()),
                    "typing_sha256": Evidence.sha256(Path(numpy.typing.__file__))}
        if restored["version"] != "2.3.2" or not restored["origin"].startswith("/opt/conda/"):
            raise ValueError("Expected original inherited NumPy was not restored")
        additions = re.findall(r"^ \+ (.+)$", Path(install["stderr"]["path"]).read_text(), re.M)
        result = {"classification": "DEPENDENCY_OVERLAY_EVENT_NOT_RAW_REWRITE", "start_utc": start.isoformat(), "end_utc": end.isoformat(),
            "selection_margin_seconds": 2, "cause": "uv resolver added venv-local NumPy2.4.6 while installing plotting dependencies into the authorized inherited-system-site-packages venv",
            "correction": "Removed only the newly installed venv-local NumPy; original inherited2.3.2 files and all system packages were untouched",
            "installer_receipt": {"path": str(install_path), "sha256": Evidence.sha256(install_path)},
            "restore_receipt": {"path": str(restore_path), "sha256": Evidence.sha256(restore_path)},
            "installer_additions": additions, "restored_numpy": restored,
            "plotting_versions": {name: version(name) for name in ("matplotlib", "pillow", "fonttools")},
            "candidates": rows, "candidate_count": len(rows), "import_error_count": sum(row["numpy_import_error"] for row in rows),
            "status": "RESTORED_ISOLATED_RECHECKS_PENDING",
            "verification_plan": {"luna": "Four successful FP64 jobs rechecked separately after LARGE_P1 driver ends; the failed pre-kernel import already has a fresh successful attempt",
                "standalone": "Four FP32 LARGE_P1 named batches rechecked separately after current full epoch ends",
                "timing": "c04 cycle1/P0/MEDIUM_P2/round1 retains all raw; next full3-round cycle runs under restored dependencies and replaces cycle1 as final cohort",
                "input_reference": "All actual canonical tensor/reference hashes and native binary guards continue to apply; no old fingerprint is re-signed"},
            "limitation": "PASS exit alone does not reconstruct the NumPy origin of a process started in this window; retain this uncertainty until linked rechecks exist",
            "tool_sha256": Evidence.sha256(Path(__file__))}
        if args.output.exists():
            raise FileExistsError(args.output)
        Evidence.write(args.output, result)
        print(json.dumps({key: result[key] for key in ("candidate_count", "import_error_count", "status")}))
        return 0

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--witness-label", help="Capture restored dependency origins/bytes without creating another environment event")
        args = parser.parse_args()
        return cls.witness(args) if args.witness_label else cls.run(args)


if __name__ == "__main__":
    raise SystemExit(EnvironmentEvent.main())
