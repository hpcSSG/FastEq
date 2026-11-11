"""Shared evidence IO and bounded process runner for all H20 tools."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


class Evidence:
    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    @staticmethod
    def fingerprint(value: object) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class CommandRunner:
    """No shell; persist startup and terminal receipts, bounded own process group."""

    def __init__(self, root: Path, environment: dict | None = None):
        self.root, self.environment = root, environment or {}
        self.root.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, argv: list[str], deadline: float = 30,
            cwd: Path | None = None, environment: dict | None = None) -> dict:
        if Path(name).name != name or deadline <= 0:
            raise ValueError("Use a simple log name and positive deadline")
        base = self.root / name
        final, running = Path(str(base) + ".json"), Path(str(base) + ".running.json")
        if final.exists() or running.exists():
            raise FileExistsError(base)
        overrides = {"PYTHONDONTWRITEBYTECODE": "1", **self.environment, **(environment or {})}
        record = {"argv": argv, "cwd": str(cwd or Path.cwd()), "deadline_seconds": deadline,
                  "start_utc": datetime.now(timezone.utc).isoformat(), "env_overrides": overrides}
        start = time.monotonic()
        print(f"{name}: START deadline={deadline}s", flush=True)
        with Path(str(base) + ".stdout").open("w") as stdout, Path(str(base) + ".stderr").open("w") as stderr:
            try:
                process = subprocess.Popen(argv, cwd=cwd, env=dict(os.environ, **overrides),
                                           stdout=stdout, stderr=stderr, start_new_session=True)
            except OSError as error:
                record.update(status="START_FAILED", error=str(error), returncode=None)
            else:
                record.update(pid=process.pid, pgid=process.pid, status="RUNNING")
                Evidence.write(running, record)
                try:
                    process.wait(timeout=deadline)
                    record["status"] = "PASS" if process.returncode == 0 else "FAIL"
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    record["status"] = "TIMEOUT" if isinstance(error, subprocess.TimeoutExpired) else "INTERRUPTED"
                    self.signal_group(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    self.signal_group(process.pid, signal.SIGKILL)
                    process.wait()
                record["returncode"] = process.returncode
        record["elapsed_seconds"] = time.monotonic() - start
        for suffix in ("stdout", "stderr"):
            path = Path(str(base) + "." + suffix)
            record[suffix] = {"path": str(path), "sha256": Evidence.sha256(path)}
        Evidence.write(final, record)
        running.unlink(missing_ok=True)
        print(f"{name}: {record['status']} ({record['elapsed_seconds']:.2f}s)", flush=True)
        return record

    @staticmethod
    def signal_group(pid: int, value: int) -> None:
        try:
            os.killpg(pid, value)
        except ProcessLookupError:
            pass
