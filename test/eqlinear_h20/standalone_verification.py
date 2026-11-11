"""Replay selected original frozen batches with observed NumPy provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


class NumpyRuntime:
    @staticmethod
    def capture():
        import numpy as np
        from numpy._core import _multiarray_umath

        def digest(path):
            value = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    value.update(block)
            return value.hexdigest()

        files = (np.__file__, _multiarray_umath.__file__)
        return {"version": np.__version__, "origin": np.__file__, "core": _multiarray_umath.__file__,
                "files": {path: digest(path) for path in files}, "python": sys.executable, "prefix": sys.prefix}


class VerificationScope:
    POINTS = ["LARGE_P1/" + str(batch) + "/fp32" for batch in (1024, 2048, 2944, 5888)]

    @classmethod
    def prepare(cls, args):
        from common import Evidence
        from standalone_matrix import StandalonePlan

        original = StandalonePlan(args.source_plan)
        audit = json.loads(args.source_audit.read_text())
        if not audit.get("full_plan_complete") or audit.get("plan_hash") != original.value["plan_hash"]:
            raise ValueError("Original full epoch must be independently complete before verification")
        if re.fullmatch(r"[A-Za-z0-9_.-]+", args.cycle) is None:
            raise ValueError("Use a simple fresh verification cycle")
        root = Path(original.value["session"]) / "accuracy/standalone_matrix" / args.cycle
        root.mkdir(parents=True, exist_ok=False)
        shutil.copytree(original.root / "producer", root / "producer")
        value = json.loads(json.dumps(original.value))
        for field, name in (("registry", "registry.json"), ("additional_registry", "policies.json")):
            if value.get(field):
                shutil.copy2(value[field]["path"], root / name)
                value[field]["path"] = str(root / name)
        old_cycle = value["cycle"]
        value.update(cycle=args.cycle, created_utc=datetime.now(timezone.utc).isoformat())
        for group in value["batch_groups"]:
            group["batch_id"] = group["batch_id"][:-len(old_cycle)] + args.cycle
        wrapper = root / "verification_worker.py"
        shutil.copy2(Path(__file__), wrapper)
        event = root / "numpy_overlay_event.json"
        shutil.copy2(args.event_manifest, event)
        value["batch_entrypoint"] = [value["runtime"]["python"], "-B", str(wrapper), "worker"]
        value["verification"] = {"kind": "NUMPY_OVERLAY_REPLAY", "point_keys": cls.POINTS,
            "source_plan": {"path": str(original.path), "sha256": Evidence.sha256(original.path), "plan_hash": original.value["plan_hash"]},
            "source_audit": {"path": str(args.source_audit.resolve()), "sha256": Evidence.sha256(args.source_audit)},
            "event_manifest": {"path": str(event), "sha256": Evidence.sha256(event), "origin": str(args.event_manifest.resolve())},
            "wrapper": {"path": str(wrapper), "sha256": Evidence.sha256(wrapper)}, "expected_numpy": NumpyRuntime.capture(),
            "original_producer_unchanged": True, "old_results_resigned_or_merged": False}
        value["plan_hash"] = Evidence.fingerprint({key: item for key, item in value.items() if key != "plan_hash"})
        Evidence.write(root / "plan.json", value)
        StandalonePlan(root / "plan.json")
        print(json.dumps({"plan": str(root / "plan.json"), "point_keys": cls.POINTS, "GPU_work_started": False}))

    @staticmethod
    def worker(args):
        # No project modules were imported above. Resolve every numerical import from the original byte-identical snapshot.
        value = json.loads(args.plan.read_text())
        sys.path.insert(0, str(args.plan.parent / "producer"))
        from common import Evidence
        from standalone_matrix import BatchWorker

        worker = BatchWorker(args)
        import torch  # the original BatchWorker.run also begins by importing Torch
        before = NumpyRuntime.capture()
        target = args.plan.parent / "numpy" / (args.batch_id + ".json")
        result = {"status": "RUNNING", "pid": os.getpid(), "batch_id": args.batch_id,
            "plan_hash": value["plan_hash"], "numpy_before": before,
            "wrapper_sha256": Evidence.sha256(Path(__file__)), "original_producer": value["producer"]}
        Evidence.write(target, result)
        try:
            if before != value["verification"]["expected_numpy"]:
                raise ValueError("NumPy version/origin/files differ from the restored verification runtime")
            code = worker.run()
            after = NumpyRuntime.capture()
            if after != before:
                raise ValueError("NumPy runtime changed during the verification process")
            result.update(status="COMPLETE", numpy_after=after,
                batch_receipt_sha256=Evidence.sha256(worker.receipt), torch=torch.__version__, torch_origin=torch.__file__)
            Evidence.write(target, result)
            return code
        except Exception as error:
            import traceback
            result.update(status="VERIFICATION_ERROR", error=str(error), traceback=traceback.format_exc())
            Evidence.write(target, result)
            raise

    @classmethod
    def run(cls, args):
        from common import Evidence
        from session import Session
        from standalone_matrix import StandalonePlan

        plan = StandalonePlan(args.plan)
        spec = plan.value["verification"]
        for name in ("source_plan", "source_audit", "event_manifest", "wrapper"):
            if Evidence.sha256(Path(spec[name]["path"])) != spec[name]["sha256"]:
                raise ValueError("Verification evidence changed: " + name)
        session = Session(Path(plan.value["session"]))
        selected = [row for row in plan.value["batch_groups"] if "/".join(map(str, (row["point"]["config"], row["point"]["batch"], row["precision"]))) in spec["point_keys"]]
        if len(selected) != 4:
            raise ValueError("Verification must retain exactly the four declared batches")
        receipts = []
        for group in selected:
            gpu = session.idle_gpu(group["batch_id"], plan.value["gpu"]["physical_index"])
            if plan.uuid(gpu) != plan.value["gpu"]["uuid"]:
                raise ValueError("Verification GPU differs from the original epoch")
            command = plan.value["batch_entrypoint"] + ["--plan", str(plan.path), "--batch-id", group["batch_id"]]
            terminal = session.runner().run(group["batch_id"], command, group["deadline_seconds"],
                Path(session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": gpu})
            receipts.append(terminal)
            if terminal["status"] != "PASS":
                return 1
        Evidence.write(plan.root / "verification_controller.json", {"status": "COMPLETE", "batches": receipts,
            "plan_hash": plan.value["plan_hash"], "original_epoch_counts_unchanged": True})
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source-plan", type=Path, required=True)
    prepare.add_argument("--source-audit", type=Path, required=True)
    prepare.add_argument("--event-manifest", type=Path, required=True)
    prepare.add_argument("--cycle", required=True)
    for command in ("worker", "run"):
        item = sub.add_parser(command)
        item.add_argument("--plan", type=Path, required=True)
        if command == "worker":
            item.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        VerificationScope.prepare(args)
        return 0
    return VerificationScope.worker(args) if args.command == "worker" else VerificationScope.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
