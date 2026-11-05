"""Render the before/after fasteq eqlinear comparison markdown from the CSV runs."""

from __future__ import annotations

import csv
from pathlib import Path

CONFIGS = [
    "SMALL_P1",
    "MEDIUM_P1",
    "MEDIUM_P2",
    "LARGE_P1",
    "LARGE_P3",
]
BATCHES = [512, 1024, 2048, 2944, 5888]


def _load(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["config"], int(row["batch"]), row["precision"])
            rows[key] = row
    return rows


def _f(row, field):
    value = row.get(field, "")
    if value in ("", "None", "nan"):
        return None
    return float(value)


def _ms(value):
    return "N/A" if value is None else f"{value:.3f}"


def _x(value, base):
    if value is None or base is None or value == 0:
        return "N/A"
    return f"{base / value:.2f}x"


def _rel(value):
    return "N/A" if value is None else f"{value:.2e}"


def _sum(rows, config, batch, precision, field):
    values = []
    for cfg in config:
        for b in batch:
            row = rows.get((cfg, b, precision))
            value = None if row is None else _f(row, field)
            if value is not None:
                values.append(value)
    return sum(values) if values else None


def main():
    root = Path(__file__).resolve().parents[1]
    before = _load(root / "results/fasteq_before.csv")
    after = _load(root / "results/fasteq_after.csv")

    lines = []
    add = lines.append
    add("# fasteq eqlinear 适配前后性能与精度对比")
    add("")
    add("## 0. 结论")
    add("")
    add("- **直接调用 `fast_equi_linear(descriptor, w, x)` 可以成功调用**：适配后 25 个测例 × 3 种精度全部成功；适配前（main 的 Triton 路径）只有 FP32 25/25 成功。")
    add("- 适配后新增 FP64（旧 Triton 路径拒绝 float64）和显式 TF32 路由。")
    add("- FP32 下适配后相对适配前在 25 点合计上更快；TF32 相对 torch FP32、FP64 相对 torch FP64 都有稳定加速。")
    add("")
    add("> 关于 `cuequivariance_torch`：本表测试的是 fasteq 公开的 `fast_equi_linear`。完整的")
    add("> `Linear(..., use_fasteq=True)` 包装层当前还会在 import 时遇到 vendored")
    add("> `cuequivariance_torch` 引用 `fasteq.ops.stc` 而当前树没有该模块的既有问题；这不是")
    add("> eqlinear 适配引入的，按“只改 eqlinear”的约束未处理。")
    add("")
    add("## 1. 测试方法")
    add("")
    add("- GPU：NVIDIA H100 80GB，单卡（测试时选空闲卡，`CUDA_VISIBLE_DEVICES=4`）。")
    add("- 测例：5 个 config × 5 个 batch = 25 点，与三条 CUTLASS 线此前使用的 25 点集合一致。")
    add("- 精度：")
    add("  - `TF32`：FP32 输入，`compute_precision=None`（适配后默认 tf32x3）。")
    add("  - `FP32`：FP32 输入，`compute_precision=\"fp32\"`（适配后）；适配前是 Triton IEEE FP32。")
    add("  - `FP64`：FP64 输入（适配后自动 FP64；适配前 Triton 拒绝 float64）。")
    add("- torch ref：直接按 group/path 循环的 torch matmul 实现；FP32/TF32 打开")
    add("  `allow_tf32=False`，FP64 使用 float64。torch ref 的耗时作为 1.0x 基准。")
    add("- 计时：CUDA events；torch 与 op 在同一个循环中交替测量（同一时钟/热状态），20 次取中位数；")
    add("  适配前 Triton 先做 10 次 warmup 覆盖 autotune；适配后先做 5 次 warmup 覆盖 plan 创建。")
    add("- 精度：`rel_fro = ||op - ref_fp64||_F / ||ref_fp64||_F`，ref_fp64 始终用 FP64 直接参考。")
    add("- 加速比：`torch_ms / op_ms`，大于 1 表示比 torch ref 快。")
    add("")
    add("## 2. 调用可用性")
    add("")
    add("| 版本 | TF32 | FP32 | FP64 |")
    add("|---|---|---|---|")
    add("| 适配前（main `4463b10`，Triton） | N/A：旧公开 API 无精度开关 | 25/25 | N/A：Triton 路径拒绝 float64 |")
    add("| 适配后（`xjw/dev_eqlinear`，CUTLASS） | 25/25 | 25/25 | 25/25 |")
    add("")
    add("## 3. 性能对比（torch ref = 1.0x）")
    add("")
    add("表中每个 op 单元格为 `ms (相对 torch 的加速比)`；FP32/TF32 的基准是 torch FP32，")
    add("FP64 的基准是 torch FP64。")
    add("")
    add("| 测例 | torch FP32 (1.0x) | 适配前 FP32 | 适配后 FP32 | FP32 后/前 | 适配后 TF32 | torch FP64 (1.0x) | 适配后 FP64 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cfg in CONFIGS:
        for batch in BATCHES:
            fa = after[(cfg, batch, "fp32")]
            ftf = after[(cfg, batch, "tf32")]
            ff64 = after[(cfg, batch, "fp64")]
            fb = before[(cfg, batch, "fp32")]
            torch32 = _f(fa, "torch_ms")
            torch64 = _f(ff64, "torch_ms")
            before32 = _f(fb, "op_ms")
            after32 = _f(fa, "op_ms")
            after_tf32 = _f(ftf, "op_ms")
            after64 = _f(ff64, "op_ms")
            add(
                f"| {cfg}/B{batch} | {_ms(torch32)} | "
                f"{_ms(before32)} ({_x(before32, torch32)}) | "
                f"{_ms(after32)} ({_x(after32, torch32)}) | "
                f"{_x(after32, before32)} | "
                f"{_ms(after_tf32)} ({_x(after_tf32, torch32)}) | "
                f"{_ms(torch64)} | {_ms(after64)} ({_x(after64, torch64)}) |"
            )
    add("")
    add("### 3.1 25 点合计")
    add("")
    torch32_sum = _sum(after, CONFIGS, BATCHES, "fp32", "torch_ms")
    torch64_sum = _sum(after, CONFIGS, BATCHES, "fp64", "torch_ms")
    before32_sum = _sum(before, CONFIGS, BATCHES, "fp32", "op_ms")
    after32_sum = _sum(after, CONFIGS, BATCHES, "fp32", "op_ms")
    after_tf32_sum = _sum(after, CONFIGS, BATCHES, "tf32", "op_ms")
    after64_sum = _sum(after, CONFIGS, BATCHES, "fp64", "op_ms")
    add("| 精度 | torch 25 点合计 (1.0x) | 适配前 25 点合计 | 适配后 25 点合计 | 适配前加速比 | 适配后加速比 |")
    add("|---|---:|---:|---:|---:|---:|")
    add(
        f"| FP32 | {_ms(torch32_sum)} | {_ms(before32_sum)} | {_ms(after32_sum)} | "
        f"{_x(before32_sum, torch32_sum)} | {_x(after32_sum, torch32_sum)} |"
    )
    add(
        f"| TF32 | {_ms(torch32_sum)} | N/A | {_ms(after_tf32_sum)} | N/A | "
        f"{_x(after_tf32_sum, torch32_sum)} |"
    )
    add(
        f"| FP64 | {_ms(torch64_sum)} | N/A | {_ms(after64_sum)} | N/A | "
        f"{_x(after64_sum, torch64_sum)} |"
    )
    add("")
    if before32_sum and after32_sum:
        add(
            f"FP32 25 点合计：适配前 **{before32_sum:.3f} ms** → 适配后 "
            f"**{after32_sum:.3f} ms**，适配后比适配前快 "
            f"**{before32_sum / after32_sum:.2f}x**。"
        )
        add("")
    add("## 4. 精度对比（rel_fro vs FP64 参考）")
    add("")
    add("| 测例 | 适配前 FP32 | 适配后 FP32 | 适配后 TF32 | 适配后 FP64 | torch FP32 |")
    add("|---|---:|---:|---:|---:|---:|")
    for cfg in CONFIGS:
        for batch in BATCHES:
            fb = before[(cfg, batch, "fp32")]
            fa = after[(cfg, batch, "fp32")]
            ftf = after[(cfg, batch, "tf32")]
            ff64 = after[(cfg, batch, "fp64")]
            add(
                f"| {cfg}/B{batch} | {_rel(_f(fb, 'rel_fro'))} | "
                f"{_rel(_f(fa, 'rel_fro'))} | {_rel(_f(ftf, 'rel_fro'))} | "
                f"{_rel(_f(ff64, 'rel_fro'))} | {_rel(_f(fa, 'torch_rel_fro'))} |"
            )
    add("")
    add("FP64 列就是 `ref_fp64` 本身，因此不再单独列 torch FP64；torch FP32 列表示")
    add("torch FP32 参考相对 FP64 参考的误差，用于和适配后的 FP32/TF32 对齐。")
    add("")
    add("### 4.1 精度汇总")
    add("")
    add("| 列 | 中位数 | 最大值 |")
    add("|---|---:|---:|")
    for label, rows, precision, field in [
        ("适配前 FP32", before, "fp32", "rel_fro"),
        ("适配后 FP32", after, "fp32", "rel_fro"),
        ("适配后 TF32", after, "tf32", "rel_fro"),
        ("适配后 FP64", after, "fp64", "rel_fro"),
        ("torch FP32", after, "fp32", "torch_rel_fro"),
    ]:
        values = []
        for cfg in CONFIGS:
            for batch in BATCHES:
                value = _f(rows[(cfg, batch, precision)], field)
                if value is not None:
                    values.append(value)
        values.sort()
        median = values[len(values) // 2]
        add(f"| {label} | {_rel(median)} | {_rel(values[-1])} |")
    add("")
    add("## 5. 备注")
    add("")
    add("- 适配前为 main 分支 `4463b10` 的 `fast_equi_linear`（Triton）；适配后为 `xjw/dev_eqlinear` 的 CUTLASS backend。")
    add("- 适配前 TF32/FP64 不是“更慢”，而是该公开 API 不提供对应精度路径；表中记为 N/A。")
    add("- 本表是算子级对比：适配后的计时包含每次调用必要的 `prepare_act`，不含首次 `prepare_weight`（由 plan 缓存承担）。")
    add("- 绝对值会受共享 GPU 的时钟/邻居影响；加速比用同一循环内交替测量的 torch 基准计算。")
    add("")

    out = root / "doc/fasteq_eqlinear_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
