#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据截图中的 MACE-OFF Small/Medium/Large 前向算子耗时数据，
生成 3 张 Fasteq vs Cueq 分组柱状图。

输出文件：
  - mace_off_small_forward.png
  - mace_off_medium_forward.png
  - mace_off_large_forward.png
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams


# ===== 1. 字体设置：尽量自动寻找中文字体 =====
def setup_chinese_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun",          # Windows 常见
        "PingFang SC", "Heiti SC", "STHeiti",          # macOS 常见
        "Noto Sans CJK SC", "Noto Sans CJK JP",
        "WenQuanYi Zen Hei", "Source Han Sans SC",     # Linux 常见
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.sans-serif"] = [name]
            break
    rcParams["axes.unicode_minus"] = False


# ===== 2. 数据 =====
operators = ["MPTP 0", "FCTP 0", "STC 0", "FCTP 1", "MPTP 1", "STC 1"]
precisions = ["FP64", "FP32"]

DATA = {
    "Small": {
        "MPTP 0": {"FP64": {"fasteq": 0.36, "cueq": 0.51},  "FP32": {"fasteq": 0.25, "cueq": 0.50}},
        "FCTP 0": {"FP64": {"fasteq": 0.38, "cueq": 0.87},  "FP32": {"fasteq": 0.24, "cueq": 0.74}},
        "STC 0":  {"FP64": {"fasteq": 0.27, "cueq": 0.99},  "FP32": {"fasteq": 0.34, "cueq": 1.00}},
        "FCTP 1": {"FP64": {"fasteq": 0.23, "cueq": 0.23},  "FP32": {"fasteq": 0.23, "cueq": 0.23}},
        "MPTP 1": {"FP64": {"fasteq": 0.35, "cueq": 0.50},  "FP32": {"fasteq": 0.23, "cueq": 0.48}},
        "STC 1":  {"FP64": {"fasteq": 0.27, "cueq": 0.96},  "FP32": {"fasteq": 0.34, "cueq": 0.98}},
    },
    "Medium": {
        "MPTP 0": {"FP64": {"fasteq": 0.54, "cueq": 0.68}, "FP32": {"fasteq": 0.28, "cueq": 0.51}},
        "FCTP 0": {"FP64": {"fasteq": 0.45, "cueq": 1.09}, "FP32": {"fasteq": 0.23, "cueq": 0.87}},
        "STC 0":  {"FP64": {"fasteq": 0.31, "cueq": 3.16}, "FP32": {"fasteq": 0.29, "cueq": 3.15}},
        "FCTP 1": {"FP64": {"fasteq": 0.24, "cueq": 0.23}, "FP32": {"fasteq": 0.16, "cueq": 0.23}},
        "MPTP 1": {"FP64": {"fasteq": 1.17, "cueq": 1.33}, "FP32": {"fasteq": 0.52, "cueq": 0.69}},
        "STC 1":  {"FP64": {"fasteq": 0.28, "cueq": 1.01}, "FP32": {"fasteq": 0.27, "cueq": 1.01}},
    },
    "Large": {
        "MPTP 0": {"FP64": {"fasteq": 0.87, "cueq": 1.00}, "FP32": {"fasteq": 0.41, "cueq": 0.55}},
        "FCTP 0": {"FP64": {"fasteq": 1.24, "cueq": 2.45}, "FP32": {"fasteq": 0.51, "cueq": 1.82}},
        "STC 0":  {"FP64": {"fasteq": 0.61, "cueq": 7.56}, "FP32": {"fasteq": 0.43, "cueq": 7.61}},
        "FCTP 1": {"FP64": {"fasteq": 0.22, "cueq": 0.23}, "FP32": {"fasteq": 0.25, "cueq": 0.26}},
        "MPTP 1": {"FP64": {"fasteq": 3.40, "cueq": 3.56}, "FP32": {"fasteq": 1.36, "cueq": 1.56}},
        "STC 1":  {"FP64": {"fasteq": 0.31, "cueq": 0.98}, "FP32": {"fasteq": 0.28, "cueq": 1.01}},
    },
}


# ===== 3. 画图函数 =====
def fmt_time(v):
    """柱子数值显示格式：保留原图风格，小数位不强制一致。"""
    if abs(v - 0.232) < 1e-12:
        return "0.232"
    return f"{v:.2f}"


def draw_one_chart(model_size, data, out_dir="."):
    # 展开为 12 个 x 位置：6 个算子 × 2 个精度
    group_centers = []
    pair_x = []
    pair_labels = []

    x = 0.0
    pair_gap = 0.90      # 同一算子内 FP64/FP32 间距
    op_gap = 1.35        # 不同算子之间间距

    for op in operators:
        start_x = x
        for p in precisions:
            pair_x.append(x)
            pair_labels.append(p)
            x += pair_gap
        end_x = x - pair_gap
        group_centers.append((start_x + end_x) / 2.0)
        x += op_gap

    pair_x = np.array(pair_x)

    fasteq_values, cueq_values = [], []
    speedups = []
    for op in operators:
        for p in precisions:
            f = data[op][p]["fasteq"]
            c = data[op][p]["cueq"]
            fasteq_values.append(f)
            cueq_values.append(c)
            speedups.append(c / f)

    fasteq_values = np.array(fasteq_values)
    cueq_values = np.array(cueq_values)
    speedups = np.array(speedups)

    # y 轴范围
    max_y = float(max(fasteq_values.max(), cueq_values.max()))
    y_top = max_y * 1.22
    if model_size == "Small":
        y_top = 1.20
    elif model_size == "Medium":
        y_top = 3.50
    elif model_size == "Large":
        y_top = 9.00

    fig, ax = plt.subplots(figsize=(14, 9), dpi=180)

    bar_w = 0.22
    red = "#E41A1C"
    blue = "#2166E8"

    bars_f = ax.bar(pair_x - bar_w / 2, fasteq_values, width=bar_w, color=red, label="Fasteq")
    bars_c = ax.bar(pair_x + bar_w / 2, cueq_values, width=bar_w, color=blue, label="Cueq")

    # 柱子数值标签
    value_offset = y_top * 0.018
    for rect, val in zip(bars_f, fasteq_values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + value_offset,
            fmt_time(float(val)),
            ha="center",
            va="bottom",
            fontsize=11,
            color=red,
        )
    for rect, val in zip(bars_c, cueq_values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + value_offset,
            fmt_time(float(val)),
            ha="center",
            va="bottom",
            fontsize=11,
            color=blue,
        )

    # 加速比标签：Cueq / Fasteq
    for px, f, c, s in zip(pair_x, fasteq_values, cueq_values, speedups):
        y = max(f, c) + y_top * 0.07
        ax.text(
            px,
            y,
            f"{s:.2f}×",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black",
        )
        # 小括号/横线效果
        bracket_y = y - y_top * 0.015
        left = px - bar_w * 0.9
        right = px + bar_w * 0.9
        tick = y_top * 0.012
        ax.plot([left, right], [bracket_y, bracket_y], color="black", linewidth=0.8)
        ax.plot([left, left], [bracket_y, bracket_y - tick], color="black", linewidth=0.8)
        ax.plot([right, right], [bracket_y, bracket_y - tick], color="black", linewidth=0.8)

    # 坐标轴与标签
    ax.set_title(f"MACE-OFF {model_size}", fontsize=22, fontweight="bold", pad=28)
    ax.set_ylabel("Latency (ms)", fontsize=14, fontweight="bold")
    #ax.set_xlabel("算子类型", fontsize=14, fontweight="bold", labelpad=36)

    ax.set_xticks(pair_x)
    ax.set_xticklabels(pair_labels, fontsize=12, fontweight="bold")

    # 主算子标签放在第二行
    for center, op in zip(group_centers, operators):
        ax.text(
            center,
            -0.075 * y_top,
            op,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            clip_on=False,
        )

    # 分隔线
    # 算子内 FP64/FP32 中间分隔
    for i in range(0, len(pair_x), 2):
        mid = (pair_x[i] + pair_x[i + 1]) / 2.0
        ax.axvline(mid, ymin=0, ymax=0.16, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

    # 算子组边界
    for i in range(1, len(operators)):
        boundary = (pair_x[2 * i - 1] + pair_x[2 * i]) / 2.0
        ax.axvline(boundary, ymin=-0.10, ymax=0.16, color="black", linewidth=0.6, alpha=0.55, clip_on=False)

    ax.set_ylim(0, y_top)
    ax.set_xlim(pair_x[0] - 0.6, pair_x[-1] + 0.6)

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=True,
        fontsize=13,
        columnspacing=2.5,
    )

    # 简化边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.20)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"mace_off_{model_size.lower()}_forward.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return filename


def main():
    setup_chinese_font()
    out_dir = Path("mace_off_forward_charts")
    out_dir.mkdir(exist_ok=True)

    for size in ["Small", "Medium", "Large"]:
        path = draw_one_chart(size, DATA[size], out_dir=out_dir)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

