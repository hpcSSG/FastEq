"""Bounded Torch/NCU collection around a frozen performance BoundCall."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from common import Evidence
from hardware import CounterEvidence
from performance import MeasurementIdentity, PerformancePlan
from session import Session


class ProfileCollection:
    METRICS = ("gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum",
        "launch__registers_per_thread", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed", "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sector_hit_rate.pct", "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum", "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
        "sm__inst_executed_pipe_tensor.sum", "sm__inst_executed_pipe_tensor_op_dmma.sum",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio")

    @classmethod
    def run(cls, args):
        plan = PerformancePlan.load(args.plan.resolve())
        session = Session(Path(plan["session"]))
        if Path(args.name).name != args.name:
            raise ValueError("Use a simple profile name")
        output = session.root / "profiles" / args.mode / args.name
        if output.exists():
            raise FileExistsError(output)
        report = output.with_name(output.name + "_capture")
        gpu = session.idle_gpu(args.name, args.gpu)
        argv = [session.paths["FASTEQ_PY"], "-B", str(args.plan.parent / "producer/profile_h20.py"),
                "--plan", str(args.plan.resolve()), "--family", args.family, "--point", args.point,
                "--call", args.call, "--mode", args.mode, "--output", str(output)]
        if args.mode == "ncu":
            argv = ["ncu", "--profile-from-start", "off", "--clock-control", "none", "--cache-control", "none",
                    "--replay-mode", "kernel", "--section", "LaunchStats", "--section", "Occupancy",
                    "--metrics", ",".join(cls.METRICS), "--export", str(report), *argv]
        terminal = session.runner().run(args.name, argv, args.deadline, Path(session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": gpu})
        receipt = {"status": terminal["status"], "terminal": str(session.root / "logs" / (args.name + ".json")),
                   "plan_hash": plan["plan_hash"], "selected_GPU_UUID": gpu, "profile_output": str(output),
                   "collector_sha256": Evidence.sha256(Path(__file__))}
        result_path = output / "result.json"
        if terminal["status"] == "PASS":
            result = json.loads(result_path.read_text())
            if result["GPU_UUID"] != MeasurementIdentity.gpu_uuid(gpu) or result["plan_hash"] != plan["plan_hash"]:
                raise ValueError("Profile GPU/producer identity changed")
            receipt["result"] = {"path": str(result_path), "sha256": Evidence.sha256(result_path), "status": result["status"]}
        if args.mode == "ncu" and terminal["status"] == "PASS":
            binary = Path(str(report) + ".ncu-rep")
            if not binary.is_file():
                receipt.update(status="NO_KERNEL_REPORT_DIAGNOSIS_REQUIRED", reason="NCU completed but exported no kernel report")
            else:
                export = session.runner().run(args.name + "_raw_import", ["ncu", "--import", str(binary), "--page", "raw", "--csv"], 30)
                receipt["raw_import_terminal"] = export["status"]
                if export["status"] != "PASS":
                    receipt["status"] = "RAW_EXPORT_FAILED"
                else:
                    raw = output / "raw.csv"
                    shutil.copyfile(export["stdout"]["path"], raw)
                    derived = CounterEvidence.parse(raw)
                    derived["classification"] = "PROFILED_OPERATOR_COUNTERS_NOT_UNPROFILED_MAIN_LATENCY"
                    with raw.open(newline="") as stream:
                        records = list(csv.reader(stream))
                    derived["all_metric_rows"] = [{"kernel": row[records[0].index("Kernel Name")],
                        "metrics": {name: {"raw_value": value, "raw_unit": unit}
                                    for name, value, unit in zip(records[0], row, records[1])}}
                        for row in records[2:]]
                    Evidence.write(output / "counters.json", derived)
                    receipt.update(report={"path": str(binary), "sha256": Evidence.sha256(binary)},
                        raw={"path": str(raw), "sha256": Evidence.sha256(raw)},
                        counters={"path": str(output / "counters.json"), "sha256": Evidence.sha256(output / "counters.json")},
                        metrics=cls.METRICS, sections=["LaunchStats", "Occupancy"], replay_mode="kernel", cache_control="none", clock_control="none")
        Evidence.write(output.with_name(output.name + ".collection.json"), receipt)
        print(json.dumps({key: receipt[key] for key in ("status", "profile_output")}))
        return 0 if receipt["status"] == "PASS" else 1

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--family", required=True)
        parser.add_argument("--point", required=True)
        parser.add_argument("--call", required=True)
        parser.add_argument("--mode", choices=("torch", "ncu"), required=True)
        parser.add_argument("--name", required=True)
        parser.add_argument("--gpu", type=int, required=True)
        parser.add_argument("--deadline", type=float, default=120.)
        return cls.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(ProfileCollection.main())
