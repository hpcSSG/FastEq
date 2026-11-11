"""Small forward/dx tables from completed public TimingAudit JSON; no benchmarks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

from common import Evidence
from timing_tables import TimingTables


class ForwardScope:
    CALLS = {
        "fasteq_fp32": (("fasteq_default", "public_warm"), ("fasteq_strict_fp32", "public_warm")),
        "luna_fp32": (("lunaeq_0.2.0", "adapted_public_e2e"),),
        "fasteq_fp64": (("fasteq_default", "public_warm"),),
        "luna_fp64": (("lunaeq_0.2.0", "adapted_public_e2e"),),
    }
    LABELS = {"fasteq_default": "FastEq 默认", "fasteq_strict_fp32": "FastEq strict FP32", "lunaeq_0.2.0": "Luna 0.2"}
    ANCHOR = "torch_ieee_full"
    MAIN_ROWS = (("fasteq_fp32", ANCHOR, "Torch FP32"),
                 ("fasteq_fp32", "fasteq_default", "FastEq 默认（TF32x3）"),
                 ("fasteq_fp32", "fasteq_strict_fp32", "FastEq strict FP32"),
                 ("luna_fp32", "lunaeq_0.2.0", "Luna FP32"),
                 ("fasteq_fp64", ANCHOR, "Torch FP64"),
                 ("fasteq_fp64", "fasteq_default", "FastEq FP64"),
                 ("luna_fp64", "lunaeq_0.2.0", "Luna FP64"))

    @staticmethod
    def key(row):
        return row["variant_id"], row["config"], int(row["B"]), row["precision"], row["operation"], row["call_boundary"]

    @staticmethod
    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0


@dataclass
class AuditView:
    value: dict
    metrics_complete: bool = False

    def __post_init__(self):
        self.points = {row["point"]: row for row in self.value.get("points", [])}
        if len(self.points) != len(self.value.get("points", [])):
            raise ValueError("Duplicate audited point")
        self.cells = {}
        for row in self.value.get("across_rounds", []):
            key = ForwardScope.key(row)
            if key in self.cells:
                raise ValueError("Duplicate audited three-round cell")
            self.cells[key] = row
        self.rounds = {}
        for row in self.value.get("summaries", []):
            key = (*ForwardScope.key(row), int(row["round"]))
            if key in self.rounds:
                raise ValueError("Duplicate audited round summary")
            self.rounds[key] = row

    @property
    def complete(self):
        return (self.metrics_complete and self.value.get("full_factory_scope_complete") is True
                and self.value.get("selected_scope_complete") is True
                and self.value.get("scope_point_count") == 25 and len(self.points) == 25
                and all(row.get("complete") is True for row in self.points.values())
                and not self.value.get("phase_errors")
                and self.value.get("latest_attempted_cycle", self.value["cycle"]) == self.value["cycle"])


class CoreReport:
    def __init__(self, plan, audits):
        self.plan, self.audits = plan, {family: AuditView(value) for family, value in audits.items()}
        if set(plan["include_families"]) != set(ForwardScope.CALLS):
            raise ValueError("Report accepts only the existing four current forward families")
        operations = {key.split("/")[1] for calls in plan["expected_calls"].values() for key in calls}
        if len(operations) != 1 or not operations <= {"forward", "grad_x"}:
            raise ValueError("Each report requires one existing forward or dx operation")
        self.operation = next(iter(operations))
        if set(self.audits) - set(ForwardScope.CALLS):
            raise ValueError("Unrequested family in audit inputs")
        for family, audit in self.audits.items():
            value = audit.value
            if value["family"] != family or value["plan_hash"] != plan["plan_hash"] or value["protocol_hash"] != plan["protocol_hash"]:
                raise ValueError("Audit family/plan/protocol identity differs")
            expected = {f"{variant}/{self.operation}/{boundary}" for variant, boundary in ForwardScope.CALLS[family]}
            expected.add(f"torch_ieee_full/{self.operation}/public_warm")
            if set(plan["expected_calls"][family]) != expected or len(plan["selected_points"][family]) != 25:
                raise ValueError("Declared core forward call/point set differs")
            if set(audit.points) - set(plan["selected_points"][family]):
                raise ValueError("Audit contains points outside its declared family")
            expected_cells = set()
            for point in plan["selected_points"][family]:
                config, batch, precision = point.split("/")
                for call in expected:
                    variant, operation, boundary = call.split("/")
                    expected_cells.add((variant, config, int(batch), precision, operation, boundary))
            audit.metrics_complete = (set(audit.cells) == expected_cells
                and set(audit.rounds) == {(*key, index) for key in expected_cells for index in range(3)}
                and all(value.get("wall_count") == 20 for value in audit.rounds.values())
                and all(value.get("rounds_complete") is True for value in audit.cells.values()))
            for key in audit.cells:
                if key[4] != self.operation or f"{key[0]}/{self.operation}/{key[5]}" not in expected:
                    raise ValueError("Audit contains an unrequested operation/boundary")
        self.identity_errors = self.cross_family_errors()
        self.complete = len(self.audits) == 4 and all(audit.complete for audit in self.audits.values())
        self.comparable = self.complete and not self.identity_errors

    def cross_family_errors(self):
        errors = []
        if len({audit.value["cycle"] for audit in self.audits.values()}) > 1:
            errors.append("all_families:cycle")
        devices = set()
        for audit in self.audits.values():
            for point, value in audit.points.items():
                identity = value.get("identity") or {}
                if identity.get("GPU_UUID"):
                    devices.add(identity["GPU_UUID"])
                if value.get("complete") and any(not identity.get(key) for key in
                        ("GPU_UUID", "input_hash", "reference_hashes", "protocol_hash", "affinity")):
                    errors.append(point + ":missing audited process identity")
        if len(devices) > 1:
            errors.append("all_points:GPU_UUID")
        for precision in ("fp32", "fp64"):
            left, right = self.audits.get("fasteq_" + precision), self.audits.get("luna_" + precision)
            if left is None or right is None:
                continue
            for point in set(left.points) & set(right.points):
                a, b = left.points[point].get("identity"), right.points[point].get("identity")
                if a is None or b is None:
                    continue
                for field in ("GPU_UUID", "input_hash", "reference_hashes", "protocol_hash", "affinity"):
                    if a.get(field) != b.get(field):
                        errors.append(point + ":" + field)
        return errors

    @staticmethod
    def cell_complete(audit, key, value):
        if not value or value.get("rounds_complete") is not True:
            return False
        if set(value.get("round_medians_ms", {})) != {"0", "1", "2"}:
            return False
        return all(audit.rounds.get((*key, round_index), {}).get("wall_count") == 20 for round_index in range(3))

    def point_rows(self):
        result = []
        for family, calls in ForwardScope.CALLS.items():
            audit = self.audits.get(family)
            for point in self.plan["selected_points"][family]:
                config, batch, precision = point.split("/")
                common = (config, int(batch), precision, self.operation)
                anchor_key = (ForwardScope.ANCHOR, *common, "public_warm")
                anchor = audit.cells.get(anchor_key, {}) if audit else {}
                point_report = audit.points.get(point, {}) if audit else {}
                identity = point_report.get("identity") or {}
                for variant, boundary in calls:
                    key = (variant, *common, boundary)
                    candidate = audit.cells.get(key, {}) if audit else {}
                    completed = bool(audit and self.cell_complete(audit, key, candidate)
                                     and self.cell_complete(audit, anchor_key, anchor) and point_report.get("complete"))
                    candidate_ms = candidate.get("wall_median_of_round_medians_ms")
                    anchor_ms = anchor.get("wall_median_of_round_medians_ms")
                    available = ForwardScope.positive(candidate_ms) and ForwardScope.positive(anchor_ms)
                    numerical = completed and candidate.get("qualified") is True and anchor.get("qualified") is True
                    spread = bool(candidate.get("unstable_spread") or anchor.get("unstable_spread"))
                    cross = bool(candidate.get("unstable_cross_round") or anchor.get("unstable_cross_round"))
                    warmed = candidate.get("warmup_stabilized") is True and anchor.get("warmup_stabilized") is True
                    eligible = bool(self.comparable and completed and numerical and available
                        and TimingTables.point_ranking_eligible(audit.value, candidate)
                        and TimingTables.point_ranking_eligible(audit.value, anchor)
                        and TimingTables.boundary_eligible(audit.value, candidate)
                        and TimingTables.boundary_eligible(audit.value, anchor))
                    flags = []
                    for flag, condition in (("PARTIAL_NOT_RANKED", not self.complete or not completed),
                        ("IDENTITY_MISMATCH", bool(self.identity_errors)), ("CANDIDATE_ACCURACY_FAIL", candidate.get("qualified") is False),
                        ("TORCH_ACCURACY_FAIL", anchor.get("qualified") is False), ("WITHIN_ROUND_NOISE", spread),
                        ("CROSS_ROUND_NOISE", cross), ("WARMUP_NOT_STABILIZED", not warmed)):
                        if condition:
                            flags.append(flag)
                    if audit:
                        for name in ("requires_complete_scan_repeat", "stability_limited", "noise_repeat_limit_reached"):
                            if audit.value.get(name):
                                flags.append(name.upper())
                        if not audit.value.get("comparison_qualified", True):
                            flags.append("COMPARISON_ADJUDICATION")
                    ratio = anchor_ms / candidate_ms if self.comparable and completed and available else None
                    binding = identity.get("bindings", {})
                    row = {"family": family, "precision": precision, "config": config, "B": int(batch),
                        "variant_id": variant, "candidate": ForwardScope.LABELS[variant], "operation": self.operation,
                        "cycle": audit.value["cycle"] if audit else None, "protocol_hash": self.plan["protocol_hash"],
                        "small10": int(batch) in (512, 1024),
                        "accuracy_tier": self.plan.get("factory_scope", {}).get("accuracy_tiers_by_variant", {}).get(variant, {}).get(precision, precision),
                        "call_boundary": boundary, "baseline": "Torch eager reference", "baseline_boundary": "public_warm",
                        "three_rounds_complete": completed, "candidate_wall_ms": candidate_ms, "torch_wall_ms": anchor_ms,
                        "diagnostic_torch_over_candidate": ratio, "accuracy_qualified": numerical,
                        "candidate_accuracy_pass": candidate.get("qualified"), "torch_accuracy_pass": anchor.get("qualified"),
                        "unstable_spread": spread, "unstable_cross_round": cross, "warmup_stabilized": warmed,
                        "ranking_eligible": eligible, "GPU_UUID": identity.get("GPU_UUID"), "input_hash": identity.get("input_hash"),
                        "reference_hashes": identity.get("reference_hashes"), "affinity": identity.get("affinity"),
                        "candidate_source_hash": binding.get(variant, {}).get("source_hash"),
                        "candidate_binary_hash": binding.get(variant, {}).get("binary_hash"),
                        "torch_source_hash": binding.get(ForwardScope.ANCHOR, {}).get("source_hash"),
                        "torch_binary_hash": binding.get(ForwardScope.ANCHOR, {}).get("binary_hash"),
                        "flags": flags, "comparison_adjudications": audit.value.get("comparison_adjudications", []) if audit else [],
                        "phase_errors": audit.value.get("phase_errors", []) if audit else ["missing audit"],
                        "candidate_audit_flags": {key: value for key, value in candidate.items() if type(value) is bool},
                        "torch_audit_flags": {key: value for key, value in anchor.items() if type(value) is bool},
                        "family_audit_flags": {key: value for key, value in audit.value.items() if type(value) is bool} if audit else {},
                        "candidate_cross_round_variation": candidate.get("cross_round_variation"),
                        "torch_cross_round_variation": anchor.get("cross_round_variation"),
                        "point_errors": point_report.get("errors", []), "point_reason": point_report.get("reason"),
                        "missing_groups": point_report.get("missing_groups"), "duplicate_groups": point_report.get("duplicate_groups")}
                    for round_index in range(3):
                        c = audit.rounds.get((*key, round_index), {}) if audit else {}
                        a = audit.rounds.get((*anchor_key, round_index), {}) if audit else {}
                        row.update({f"candidate_round{round_index}_ms": c.get("wall_median"),
                                    f"torch_round{round_index}_ms": a.get("wall_median"),
                                    f"candidate_round{round_index}_groups": c.get("wall_count"),
                                    f"torch_round{round_index}_groups": a.get("wall_count"),
                                    f"candidate_round{round_index}_spread": c.get("wall_spread"),
                                    f"torch_round{round_index}_spread": a.get("wall_spread"),
                                    f"candidate_round{round_index}_flags": {key: value for key, value in c.items() if type(value) is bool},
                                    f"torch_round{round_index}_flags": {key: value for key, value in a.items() if type(value) is bool}})
                    result.append(row)
        return result

    def sum_rows(self, rows):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["family"], row["variant_id"]].append(row)
        result = []
        for (family, variant), values in grouped.items():
            for small in (False, True):
                chosen = [row for row in values if not small or row["B"] in (512, 1024)]
                count = 10 if small else 25
                complete = (self.comparable and len(chosen) == count
                            and all(row["three_rounds_complete"] and ForwardScope.positive(row["candidate_wall_ms"])
                                    and ForwardScope.positive(row["torch_wall_ms"]) for row in chosen))
                baseline = [row["torch_wall_ms"] for row in chosen]
                candidate = [row["candidate_wall_ms"] for row in chosen]
                value = {"family": family, "precision": family.rsplit("_", 1)[1], "variant_id": variant,
                    "candidate": values[0]["candidate"], "operation": self.operation,
                    "normalization": values[0].get("normalization", "same_process_same_precision_torch"),
                    "shape_scope": "small10" if small else "all25",
                    "expected_points": count, "available_points": sum(row["three_rounds_complete"] for row in chosen),
                    "complete": complete, "sum_torch_ms": sum(baseline) if complete else None,
                    "sum_candidate_ms": sum(candidate) if complete else None,
                    "diagnostic_sum_torch_over_candidate": TimingTables.sum_ratio(baseline, candidate) if complete else None,
                    "qualified_points": sum(row["accuracy_qualified"] for row in chosen),
                    "noise_points": sum(row["unstable_spread"] or row["unstable_cross_round"] for row in chosen),
                    "warmup_unstable_points": sum(not row["warmup_stabilized"] for row in chosen),
                    "ranking_eligible": complete and all(row["ranking_eligible"] for row in chosen),
                    "flags": sorted({flag for row in chosen for flag in row["flags"]})}
                for round_index in range(3):
                    value[f"round{round_index}_diagnostic_sum_ratio"] = TimingTables.sum_ratio(
                        [row[f"torch_round{round_index}_ms"] for row in chosen],
                        [row[f"candidate_round{round_index}_ms"] for row in chosen]) if complete else None
                result.append(value)
        return result

    def view(self, row, torch_anchor=False):
        variant = ForwardScope.ANCHOR if torch_anchor else row["variant_id"]
        boundary = "public_warm" if torch_anchor else row["call_boundary"]
        prefix = "torch" if torch_anchor else "candidate"
        audit = self.audits.get(row["family"])
        key = (variant, row["config"], row["B"], row["precision"], self.operation, boundary)
        cell = audit.cells.get(key, {}) if audit else {}
        complete = bool(audit and self.cell_complete(audit, key, cell)
                        and audit.points.get(f"{row['config']}/{row['B']}/{row['precision']}", {}).get("complete"))
        rankable = bool(self.comparable and complete and cell.get("qualified") is True
            and TimingTables.point_ranking_eligible(audit.value, cell) and TimingTables.boundary_eligible(audit.value, cell))
        return {"family": row["family"], "variant": variant, "boundary": boundary,
                "wall": row[prefix + "_wall_ms"], "qualified": cell.get("qualified"), "complete": complete,
                "rankable": rankable, "flags": {key: value for key, value in cell.items() if type(value) is bool},
                "source_hash": row[prefix + "_source_hash"], "binary_hash": row[prefix + "_binary_hash"],
                "rounds": {index: row[f"{prefix}_round{index}_ms"] for index in range(3)},
                "round_flags": {index: row[f"{prefix}_round{index}_flags"] for index in range(3)},
                "round_groups": {index: row[f"{prefix}_round{index}_groups"] for index in range(3)},
                "round_spread": {index: row[f"{prefix}_round{index}_spread"] for index in range(3)},
                "cross_variation": row[prefix + "_cross_round_variation"]}

    def normalized_rows(self, paired):
        index = {(row["family"], row["variant_id"], row["config"], row["B"]): row for row in paired}
        result = []
        for family, variant, label in ForwardScope.MAIN_ROWS:
            for point in self.plan["selected_points"][family]:
                config, batch, precision = point.split("/")
                source_variant = "fasteq_default" if variant == ForwardScope.ANCHOR else variant
                source = index[family, source_variant, config, int(batch)]
                base = index["fasteq_fp32", "fasteq_default", config, int(batch)]
                candidate = self.view(source, variant == ForwardScope.ANCHOR)
                baseline = self.view(base, True)
                complete = candidate["complete"] and baseline["complete"]
                positive = ForwardScope.positive(candidate["wall"]) and ForwardScope.positive(baseline["wall"])
                accuracy = complete and candidate["qualified"] is True and baseline["qualified"] is True
                spread = bool(candidate["flags"].get("unstable_spread") or baseline["flags"].get("unstable_spread"))
                cross = bool(candidate["flags"].get("unstable_cross_round") or baseline["flags"].get("unstable_cross_round"))
                warm = candidate["flags"].get("warmup_stabilized") is True and baseline["flags"].get("warmup_stabilized") is True
                flags = []
                for name, condition in (("PARTIAL_NOT_RANKED", not self.comparable or not complete),
                    ("CANDIDATE_ACCURACY_FAIL", candidate["qualified"] is False),
                    ("TORCH_FP32_ACCURACY_FAIL", baseline["qualified"] is False), ("WITHIN_ROUND_NOISE", spread),
                    ("CROSS_ROUND_NOISE", cross), ("WARMUP_NOT_STABILIZED", not warm)):
                    if condition:flags.append(name)
                family_flags = source["family_audit_flags"]
                base_flags = base["family_audit_flags"]
                for name in ("requires_complete_scan_repeat", "stability_limited", "noise_repeat_limit_reached"):
                    if family_flags.get(name) or base_flags.get(name):flags.append(name.upper())
                if family_flags.get("comparison_qualified") is False or base_flags.get("comparison_qualified") is False:
                    flags.append("COMPARISON_ADJUDICATION")
                row = {**source, "variant_id": variant, "candidate": label, "call_boundary": candidate["boundary"],
                    "normalization": "unified_fasteq_family_torch_fp32", "baseline": "Torch FP32",
                    "baseline_family": "fasteq_fp32", "baseline_precision": "fp32", "baseline_boundary": "public_warm",
                    "candidate_wall_ms": candidate["wall"], "torch_wall_ms": baseline["wall"],
                    "diagnostic_torch_over_candidate": baseline["wall"] / candidate["wall"] if self.comparable and complete and positive else None,
                    "three_rounds_complete": complete, "accuracy_qualified": accuracy,
                    "candidate_accuracy_pass": candidate["qualified"], "torch_accuracy_pass": baseline["qualified"],
                    "unstable_spread": spread, "unstable_cross_round": cross, "warmup_stabilized": warm,
                    "ranking_eligible": complete and positive and candidate["rankable"] and baseline["rankable"],
                    "candidate_audit_flags": candidate["flags"], "torch_audit_flags": baseline["flags"],
                    "candidate_source_hash": candidate["source_hash"], "candidate_binary_hash": candidate["binary_hash"],
                    "torch_source_hash": baseline["source_hash"], "torch_binary_hash": baseline["binary_hash"],
                    "candidate_cross_round_variation": candidate["cross_variation"], "torch_cross_round_variation": baseline["cross_variation"],
                    "torch_fp32_input_hash": base["input_hash"], "torch_fp32_reference_hashes": base["reference_hashes"],
                    "input_hash_equal_to_torch_fp32": source["input_hash"] == base["input_hash"],
                    "shape_only_cross_precision": precision != "fp32", "normalizer_family_audit_flags": base_flags, "flags": flags}
                for round_index in range(3):
                    for prefix, value in (("candidate", candidate), ("torch", baseline)):
                        row.update({f"{prefix}_round{round_index}_ms": value["rounds"][round_index],
                                    f"{prefix}_round{round_index}_groups": value["round_groups"][round_index],
                                    f"{prefix}_round{round_index}_spread": value["round_spread"][round_index],
                                    f"{prefix}_round{round_index}_flags": value["round_flags"][round_index]})
                result.append(row)
        return result

    def anchor_differences(self, paired):
        index = {(row["family"], row["config"], row["B"]): row for row in paired if row["variant_id"] != "fasteq_strict_fp32"}
        rows, sums = [], []
        for precision in ("fp32", "fp64"):
            chosen = []
            for point in self.plan["selected_points"]["fasteq_" + precision]:
                config, batch, _ = point.split("/")
                left, right = index["fasteq_" + precision, config, int(batch)], index["luna_" + precision, config, int(batch)]
                a, b = self.view(left, True), self.view(right, True)
                valid = self.comparable and a["complete"] and b["complete"] and ForwardScope.positive(a["wall"]) and ForwardScope.positive(b["wall"])
                row = {"operation": self.operation, "precision": precision, "config": config, "B": int(batch),
                    "fasteq_family_torch_ms": a["wall"], "luna_family_torch_ms": b["wall"],
                    "luna_anchor_over_fasteq_anchor": b["wall"] / a["wall"] if valid else None,
                    "relative_difference": (b["wall"] - a["wall"]) / a["wall"] if valid else None,
                    "same_input_hash": left["input_hash"] == right["input_hash"], "same_GPU": left["GPU_UUID"] == right["GPU_UUID"],
                    "fasteq_torch_flags": a["flags"], "luna_torch_flags": b["flags"], "complete": valid}
                rows.append(row);chosen.append(row)
            for small in (False, True):
                selected = [row for row in chosen if not small or row["B"] in (512, 1024)]
                complete = all(row["complete"] for row in selected)
                a = [row["fasteq_family_torch_ms"] for row in selected]
                b = [row["luna_family_torch_ms"] for row in selected]
                sums.append({"operation": self.operation, "precision": precision, "shape_scope": "small10" if small else "all25",
                    "points": len(selected), "complete": complete, "sum_fasteq_family_torch_ms": sum(a) if complete else None,
                    "sum_luna_family_torch_ms": sum(b) if complete else None,
                    "luna_anchor_sum_over_fasteq_anchor_sum": TimingTables.sum_ratio(b, a) if complete else None})
        return rows, sums

    def markdown(self, sums, resources=None):
        status = "完整三轮" if self.comparable else "partial：不计算速度比、不排名"
        if self.identity_errors:
            status = "身份不一致：不计算速度比、不排名"
        operation_label = "forward" if self.operation == "forward" else "dx（grad_x）"
        lines = [f"当前 eqlinear {operation_label} 对比：{status}。", "",
            "采用每点三轮 wall 中位数的中位数。统一 Torch FP32=1.00×；SUM 比值 = ΣTorch FP32 耗时 / Σ对应实现耗时；大于 1 表示候选总耗时更低。",
            "小10点为五个配置各取 B=512、1024。全部失败与噪声点保留在诊断 SUM 中。", "",
            "| 输入 | 候选 | 25点 SUM 比值 | 小10点 SUM 比值 | 数值合格（25 / 10） | 噪声点（25 / 10） | 标记 |",
            "|---|---|---:|---:|---|---|---|"]
        if resources:
            gpu = ",".join(str(value) for value in resources["GPU_indices"]) or "UUID见CSV"
            cpu = ",".join(str(value) for value in resources["CPU_selected"]) or "未完整"
            lines[1:1] = [f"设备：GPU {gpu}；CPU {cpu}。"]
        lookup = {(row["family"], row["variant_id"], row["shape_scope"]): row for row in sums}
        fmt = lambda value: "—" if value is None else f"{value:.2f}×"
        for family, variant, label in ForwardScope.MAIN_ROWS:
            a, b = lookup[family, variant, "all25"], lookup[family, variant, "small10"]
            flags = []
            if not a["complete"]:
                flags.append("partial/身份检查未通过")
            if a["qualified_points"] != 25:
                flags.append("含精度失败")
            if a["noise_points"]:
                flags.append("含噪声")
            if a["warmup_unstable_points"]:
                flags.append("含预热未稳")
            if not a["ranking_eligible"] and a["complete"]:
                flags.append("诊断比值")
            lines.append(f"| {a['precision'].upper()} | {label} | {fmt(a['diagnostic_sum_torch_over_candidate'])} | "
                f"{fmt(b['diagnostic_sum_torch_over_candidate'])} | {a['qualified_points']}/25；{b['qualified_points']}/10 | "
                f"{a['noise_points']}；{b['noise_points']} | {'；'.join(flags) or '审计可比'} |")
        boundary = ("Luna forward 调用内包含 activation pack/output unpack，恒定权重转换在计时外。" if self.operation == "forward"
                    else "dx 计时真实 autograd.grad；前向建图和固定权重预处理在窗口外，Luna 反向布局链在调用内。")
        lines.extend(["", "主表统一使用 fasteq_fp32 family 的 Torch FP32。相同输入精度内核对实际输入及上游梯度；跨 FP32/FP64 仅按同 shape 耗时归一化，输入字节/seed 不同，不作跨精度准确性比较。",
            "同进程同精度配对指标及两套 Torch anchor 差异保留在辅助 CSV。Torch FP64 行使用 fasteq_fp64 family 的 Torch。",
            "Torch 使用现有 TorchCalls.full eager 参考路径，本结果仅覆盖该实现。" + boundary,
            "失败或噪声标记对应的比值仅作诊断；逐点延迟、三轮数据及全部标记见 points.csv。", ""])
        return "\n".join(lines)


class ReportWriter:
    @staticmethod
    def resources(plan, rows):
        uuids = sorted({row["GPU_UUID"] for row in rows if row["GPU_UUID"]})
        cpus = sorted({row["affinity"]["selected"] for row in rows if row.get("affinity") and "selected" in row["affinity"]})
        result = {"GPU_UUIDs": uuids, "CPU_selected": cpus, "GPU_indices": [], "mapping_source": None}
        wanted = {str(value).lower().removeprefix("gpu-") for value in uuids}
        pattern = "perf_" + Path(plan["directory"]).name + "_*_idle_samples.json"
        for path in sorted((Path(plan["session"]) / "env").glob(pattern)):
            content = path.read_bytes()
            samples = json.loads(content)
            mapping = {str(gpu["uuid"]).lower().removeprefix("gpu-"): int(gpu["index"])
                       for sample in samples for gpu in sample.get("gpus", [])}
            if wanted and wanted <= mapping.keys():
                result.update(GPU_indices=sorted({mapping[uuid] for uuid in wanted}),
                              mapping_source={"path": str(path), "sha256": hashlib.sha256(content).hexdigest()})
                break
        return result

    @staticmethod
    def csv(path, rows):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                                 for key, value in row.items()})

    @classmethod
    def run(cls, args):
        plan_bytes = args.plan.read_bytes()
        plan = json.loads(plan_bytes)
        if Evidence.fingerprint({key: value for key, value in plan.items() if key != "plan_hash"}) != plan["plan_hash"]:
            raise ValueError("Plan fingerprint differs")
        audits, snapshots = {}, {"plan.json": plan_bytes}
        inputs = {str(args.plan.resolve()): hashlib.sha256(plan_bytes).hexdigest()}
        for path in args.audit:
            content = path.read_bytes()
            value = json.loads(content)
            if value["family"] in audits:
                raise ValueError("Supply one explicit audited cycle per family")
            audits[value["family"]] = value
            inputs[str(path.resolve())] = hashlib.sha256(content).hexdigest()
            snapshots[value["family"] + ".audit.json"] = content
        report = CoreReport(plan, audits)
        paired = report.point_rows()
        paired_sums = report.sum_rows(paired)
        rows = report.normalized_rows(paired)
        sums = report.sum_rows(rows)
        anchor_rows, anchor_sums = report.anchor_differences(paired)
        resources = cls.resources(plan, rows)
        output = args.output.resolve()
        if not output.is_relative_to(Path(plan["session"]).resolve()):
            raise ValueError("Report output must remain in the configured session")
        output.mkdir(parents=True, exist_ok=False)
        (output / "inputs").mkdir()
        for name, content in snapshots.items():
            (output / "inputs" / name).write_bytes(content)
        (output / "reporter.py").write_bytes(Path(__file__).read_bytes())
        csv_outputs = {"points.csv": rows, "sum_speedups.csv": sums,
                       "same_process_points.csv": paired, "same_process_sum_speedups.csv": paired_sums,
                       "torch_anchor_differences.csv": anchor_rows, "torch_anchor_sum_differences.csv": anchor_sums}
        for name, values in csv_outputs.items():
            cls.csv(output / name, values)
        (output / "results_zh.md").write_text(report.markdown(sums, resources))
        metadata = {"status": "COMPLETE_AUDITS" if report.comparable else "PARTIAL_OR_IDENTITY_MISMATCH_NOT_RANKED",
                    "complete": report.complete, "identity_errors": report.identity_errors,
                    "inputs": inputs, "reporter_sha256": Evidence.sha256(Path(__file__)),
                    "protocol_hash": plan["protocol_hash"], "operation": report.operation,
                    "point_rows": len(rows), "sum_rows": len(sums),
                    "auxiliary_row_counts": {name: len(values) for name, values in csv_outputs.items()
                                             if name not in {"points.csv", "sum_speedups.csv"}},
                    "CPU_affinity": sorted(os.sched_getaffinity(0)),
                    "measurement_resources": resources,
                    "statistic": "sum of point medians-of-three-round-medians; unified Torch FP32 sum / candidate sum",
                    "normalization": {"baseline_family": "fasteq_fp32", "baseline_variant": ForwardScope.ANCHOR,
                                      "torch_fp64_row_family": "fasteq_fp64",
                                      "cross_precision": "matching shapes only; input bytes and seeds may differ",
                                      "auxiliary": "matched same-process same-precision Torch anchors"},
                    "failure_noise_policy": "Keep every latency and point; raw diagnostic ratios do not promote failed/noisy cases",
                    "input_snapshots": {name: Evidence.sha256(output / "inputs" / name) for name in snapshots},
                    "outputs": {name: Evidence.sha256(output / name) for name in (*csv_outputs, "results_zh.md")}}
        Evidence.write(output / "manifest.json", metadata)
        print(json.dumps({"output": str(output), "status": metadata["status"], "point_rows": len(rows)}))
        return 0 if report.comparable else 2


class Cli:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--audit", type=Path, action="append", required=True)
        parser.add_argument("--output", type=Path, required=True)
        return ReportWriter.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(Cli.main())
