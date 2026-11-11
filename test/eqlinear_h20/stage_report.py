"""Generate an explicitly incomplete H20 stage report from current receipts and probes."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from common import Evidence
from protocol import TraceSummary
from session import Session


class StageReport:
    def __init__(self, root, name):
        self.session, self.name = Session(root), name
        if Path(name).name != name:
            raise ValueError("Use a simple report name")

    @staticmethod
    def read(path):
        return json.loads(path.read_text())

    @staticmethod
    def csv(path, rows, fields):
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def snapshot_evidence(self):
        root = self.session.root
        patterns = ("coverage/registry_revision*/summary.json", "accuracy/current_accuracy_cycle02_audit_complete01.json",
            "accuracy/historical_*_histcycle01_audit.json", "accuracy/luna_matrix/luna_accuracy_full_c03/audit.json",
            "accuracy/standalone_matrix/standalone_batch_full_c01/*audit*.json",
            "accuracy/native_selector/nativebatchcycle02/*audit*.json", "coverage/native_fp64_impl1_diagnosis01.json",
            "timing/hot_protocol_c04/pilot_independent_readback02.json", "timing/hot_protocol_c04/phases/*.json",
            "coverage/handwritten_source_math_audit02/audit.json", "profiles/*/*.collection.json", "roofline/hardware.json")
        patterns += ("manifests/numpy_overlay_event01.json", "coverage/functional_queue_summary01.json",
                     "roofline/figures/*/data.json", "profiles/sass/*/result.json",
                     "manifests/luna_matrix_c03_completion.json", "tables/luna_accuracy_full_c03/summary.json",
                     "accuracy/standalone_matrix/standalone_named_full_c01/*audit*.json",
                     "accuracy/standalone_matrix/standalone_histtuples_full_c01/*audit*.json",
                     "manifests/torch_autograd_boundary_annotation01.json",
                     "timing/hot_protocol_c04/boundary_adjudications.json",
                     "coverage/handwritten_fp64_source02/fp64_contracts.json",
                     "coverage/handwritten_fp64_source02/legacy/audit.json")
        directory = root / "reports" / (self.name + ".evidence")
        directory.mkdir(exist_ok=False)
        records = []
        fields = ("status", "classification", "complete", "base_accuracy_complete", "full75x3_complete", "valid",
            "expected", "verified", "expected_points", "expected_paths", "counts", "numerical_rows", "passed_rows",
            "shared_input_reference_mismatches", "registry_rows", "scope_counts", "point_count", "expected_point_count",
            "phase_complete", "phase", "families", "planned_tasks", "all_existing_parameter_strategies_covered",
            "candidate_count", "import_error_count", "restored_numpy", "start_utc", "end_utc")
        for path in sorted({path for pattern in patterns for path in root.glob(pattern)}):
            data = path.read_bytes()
            value = json.loads(data)
            digest = hashlib.sha256(data).hexdigest()
            frozen = directory / (digest[:16] + "_" + path.name)
            frozen.write_bytes(data)
            summary = {key: value[key] for key in fields if key in value} if isinstance(value, dict) else {"entries": len(value)}
            if path.name == "functional_queue_summary01.json":
                summary = {key: value[key] for key in ("scope", "results")}
            if isinstance(value, dict) and "tasks" in value:
                summary["terminal_complete_chunks"] = sum(item.get("complete_scope") is True for item in value["tasks"])
            records.append({"source": str(path), "snapshot": str(frozen), "sha256": digest, "summary": summary})
        Evidence.write(directory / "index.json", records)
        return records

    def run(self):
        root = self.session.root
        rows, kernels, excluded = [], [], []
        drifted = {self.read(path).get("attempt_id", path.stem.removesuffix("_producer_drift"))
                   for path in (root / "manifests").glob("*_producer_drift.json")}
        for path in sorted((root / "accuracy").glob("*/result.json")):
            result = self.read(path)
            classification = result.get("classification", "").upper()
            if "CPU" in classification or "SYNTHETIC" in classification:
                excluded.append({"path": str(path), "reason": "CPU synthetic selfcheck is not a GPU probe",
                                 "sha256": Evidence.sha256(path)})
                continue
            if "status" not in result:
                raise ValueError(f"Unknown result schema; refusing to treat as GPU evidence: {path}")
            attempt = path.parent.name
            valid = attempt not in drifted and result["status"] in ("PASS", "ACCURACY_FAIL")
            if result.get("producer_unchanged") is False or result.get("producer_drift") or result.get("producer_changed_during_run"):
                valid = False
            point = result.get("point", {})
            identity = result.get("identity", {})
            variant = result.get("variant", result.get("variant_id"))
            if variant is None:
                if identity.get("family") == "current_native_selector":
                    variant = "native_" + identity["precision"] + "_impl" + str(identity["impl"])
                else:
                    variant = ("standalone_" + identity["family"] + "_impl" + str(result["impl"])) if "impl" in result else "current_fasteq"
            for item in result.get("operation_boundaries", result.get("boundaries", result.get("rows", []))):
                rows.append({"attempt": attempt, "status": result["status"], "valid_attempt": valid,
                    "variant": variant,
                    "config": point.get("config"), "B": point.get("batch"), "precision": point.get("precision"),
                    "operation": item.get("operation"), "result": item.get("result", item.get("operation")),
                    "boundary": item.get("boundary", item.get("call_boundary")),
                    "rel_fro": item.get("rel_fro"), "threshold": item.get("threshold"),
                    "passed": item.get("accuracy_pass", item.get("passed")), "evidence": str(path),
                    "frozen_p14_status": item.get("frozen_p14", {}).get("status", "NOT_CHECKED")})
            trace = result.get("trace") or {}
            trace_path = Path(trace.get("path", "/nonexistent"))
            if trace_path.is_file():
                for kernel in TraceSummary.kernels(self.read(trace_path)):
                    kernels.append({"attempt": attempt, "valid_attempt": valid, "kernel": kernel["name"],
                                    "duration_us_profiled": kernel["dur"], "trace": str(trace_path),
                                    "trace_sha256": Evidence.sha256(trace_path)})
        fields = ("attempt", "status", "valid_attempt", "variant", "config", "B", "precision", "operation", "result",
                  "boundary", "rel_fro", "threshold", "passed", "frozen_p14_status", "evidence")
        accuracy_path = root / "tables" / (self.name + "_probe_accuracy.csv")
        kernel_path = root / "tables" / (self.name + "_probe_kernels.csv")
        self.csv(accuracy_path, rows, fields)
        self.csv(kernel_path, kernels, ("attempt", "valid_attempt", "kernel", "duration_us_profiled", "trace", "trace_sha256"))
        logs = {path.stem: self.read(path) for path in (root / "logs").glob("*.json")
                if not path.name.endswith(".running.json") and "status" in self.read(path)}
        compare_path = root / "accuracy/current_compare75.csv"
        compare = list(csv.DictReader(compare_path.open())) if compare_path.exists() else []
        revisions = sorted((root / "coverage").glob("registry_revision*/variants.csv"),
                           key=lambda path: int(path.parent.name.removeprefix("registry_revision")))
        registry_path = revisions[-1] if revisions else root / "coverage/variants.csv"
        registry = list(csv.DictReader(registry_path.open())) if registry_path.exists() else []
        snapshots = self.snapshot_evidence()
        valid_rows = [row for row in rows if row["valid_attempt"] and row["status"] in ("PASS", "ACCURACY_FAIL")]
        grouped = Counter((row["variant"], row["config"], row["B"], row["precision"], row["passed"]) for row in valid_rows)
        report = [f"# H20 阶段记录：{self.name}", "", "**INCOMPLETE：目标保持进行中。此报告不是完整性能报告或版本排名。**", "",
                  f"生成时间：{datetime.now(timezone.utc).isoformat()}。所有表由脚本读取原始证据生成。", "",
                  "## 已确认范围", "",
                  "当前 FastEq、Luna 0.1.0/0.2.0、交接包既有实现与 P1–P30 语义去重版本；排除 TF32x1、FP16/BF16、TileLang、CUDA Graph。",
                  "唯一环境为用户指定的 uv 环境；所有新构建、缓存、临时输出和数据在工作盘。H100 仅消费历史资料。", "",
                  f"registry 已登记 {len(registry)} 行；静态登记不等于运行能力已验证。配置见 `../config/paths.yaml`、`../config/scope.yaml`。", "",
                  f"本次使用的清单：`{registry_path.relative_to(root)}`。各完整矩阵与正在运行phase的只读快照位于 `{self.name}.evidence/index.json`，包含当时字节及SHA；后续audit更新不改变此报告。", "",
                  "## 当前 FastEq 回归", "", "| 检查 | 终态 | 原始日志 |", "|---|---|---|"]
        for name in ("current_build", "current_configure_hopper", "current_build_hopper", "current_pytest", "current_compare75", "current_smoke_other_ops", "cpu_selfcheck", "cpu_selfcheck_02", "ncu_metrics_visible"):
            receipt = logs.get(name)
            if receipt:
                report.append(f"| {name} | {receipt['status']} | `../logs/{name}.json` |")
        if compare:
            report += ["", f"compare 读取到 {len(compare)} 行，其中 supported=True 有 {sum(row['supported']=='True' for row in compare)} 行。",
                       "该旧 compare 使用独立 seed、简化 descriptor 和 event 计时，仅作为迁移回归证据，不进入统一三轮主性能表。"]
        report += ["", "## 完整矩阵与当前阶段快照", "", "下表按独立audit/phase记录摘录；RUNNING或complete=false始终表示未完成，accuracy不是性能。", ""]
        for snapshot in snapshots:
            report += [f"- `{Path(snapshot['source']).relative_to(root)}`：`{json.dumps(snapshot['summary'], ensure_ascii=False)}`。"]
        report += ["", "首轮构建在隐藏 GPU 时触发 Torch 多架构自动探测，包含 sm_50，其他算子 double atomic 编译失败；失败记录完整保留。",
                   "随后显式设置 TORCH_CUDA_ARCH_LIST=9.0a，在新的构建目录重建。生产数学和默认策略未修改。", "",
                   "## 统一 fixture 的代表点探测", "", "| 版本 | config | B | 精度 | gate | 结果行数 |", "|---|---|---|---|---|---|"]
        for (variant, config, batch, precision, passed), count in sorted(grouped.items(), key=str):
            report.append(f"| {variant} | {config} | {batch} | {precision} | {'PASS' if passed else 'FAIL'} | {count} |")
        report += ["", "行数按操作、结果分量及调用边界计，不代表独立 shape 数；失败与被判 producer 漂移的 attempts 均保留在 CSV。",
                   "Luna FP32 的 gate 使用同点 Torch IEEE FP32 误差的1.05倍；超限记录不进入合格候选。setup/JIT时间不当作kernel延迟。", "",
                   "TF32 的 H100 冻结 P14 检查仅在所有张量和 descriptor hash 一致时适用；当前不匹配的点保持 UNVERIFIED_INPUT_HASH_MISMATCH，未更改旧 hash 或放宽限制。", "",
                   f"逐行精度：`../tables/{accuracy_path.name}`。真实 kernel（仅 cat=kernel 且 ph=X）：`../tables/{kernel_path.name}`。",
                   "profiled duration受插桩扰动，不能替代未插桩wall主表；热缓存逻辑字节不等于DRAM实测流量。", "",
                   "## 尚未完成的原目标", "",
                   "- 全部独立历史 runtime/impl 的 H20 构建、功能矩阵和统一25点accuracy。",
                   "- 两版 Luna 已完成原400项/1500数值检查；公开autograd补充、统一三轮性能和代表profile仍需完成。",
                   "- 统一pilot锁定、三轮每点至少20组的wall原始样本、独立覆盖与稳定性复核。",
                   "- public/run/prepared/cold/JIT/布局及梯度工作量分表，跨进程Torch锚点。",
                   "- hot与超过实际L2工作集的ring实验，每类机制的完整host/NCU代表覆盖。",
                   "- 硬件屋顶查证和实测校准、分精度分层Roofline、Pareto、H100可比性和最终建议。",
                   "- 最终报告、完整CHECKLIST、无残留任务审计、SHA封包及独立archive读回。", "",
                   "## 恢复入口", "",
                   "先检查 `../logs/*.running.json` 对应PID/进程启动信息及原tool session是否仍存活；观察超时不能当作终止，不重复启动仍存活作业。",
                   "每次新GPU批次用session.py的两次idle采样；构建失败、精度失败、超时、producer漂移都使用新attempt并保留旧输出。",
                   "工具CLI均在 `test/eqlinear_h20/`。原交接包和H100 receipt保持只读；没有merge/push。", ""]
        report_path = root / "reports" / (self.name + ".md")
        with report_path.open("x") as stream:
            stream.write("\n".join(report))
        Evidence.write(root / "reports" / (self.name + ".receipt.json"), {
            "status": "INCOMPLETE", "tool_sha256": Evidence.sha256(Path(__file__)),
            "report_sha256": Evidence.sha256(report_path), "tables": {str(p): Evidence.sha256(p) for p in (accuracy_path, kernel_path)},
            "snapshot_index": {"path": str(root / "reports" / (self.name + ".evidence/index.json")),
                               "sha256": Evidence.sha256(root / "reports" / (self.name + ".evidence/index.json"))},
            "valid_probe_rows": len(valid_rows), "invalid_or_nonterminal_probe_rows": len(rows) - len(valid_rows)})
        Evidence.write(root / "reports" / (self.name + ".excluded_cpu.json"), excluded)
        print(report_path)
        return 0

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--name", required=True)
        args = parser.parse_args()
        return cls(args.session, args.name).run()


if __name__ == "__main__":
    raise SystemExit(StageReport.main())
