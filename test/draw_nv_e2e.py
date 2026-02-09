import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) 数据
# -----------------------------
models = ["MACE-OFF Small", "MACE-OFF Medium", "MACE-OFF Large"]
series_names = ["e3nn", "cuEq", "fastEq"]

# ================ H100 ==============
# shape = [num_models, num_series]，单位：ms（越小越好）
values_fp64_fwd = np.array([
    [10.16, 8.2,  5.3],
    [33.85, 13.7, 9.4],
    [120.66, 25.2, 20.0],
    #[37.0, 29.0],
], dtype=float)

# FP32 子图数据（示例：请替换为你的真实测量值）
values_fp32_fwd = np.array([
    [8.0, 8.3,  5.0],
    [26.89, 12.5,  8.6],
    [97.34, 25.27, 21.5],
    #[21.5, 16.0],
], dtype=float)

# 新增：后向（示例数据，请替换成真实测量值）
values_fp64_bwd = np.array([
    [19.35, 9.9, 8.0],
    [60.70, 14.3, 13.7],
    [217.23, 23, 28],
    #[1, 1],
], dtype=float)

values_fp32_bwd = np.array([
    [14.59, 9.7,  7.5],
    [41.46, 14.0, 9.7],
    [153.95, 21, 20],
    #[1, 1],
], dtype=float)

values_fp64_total = np.array([
    [29.51, 18.19, 13.4],
    [93.92, 28.18, 24.24],
    [337.89, 48.58, 47.44],
    #[1, 1],
], dtype=float)

values_fp32_total = np.array([
    [22.59, 18.1,  10.0],
    [68.35, 26.6, 18.15],
    [251.29, 43.6, 41.01],
    #[1, 1],
], dtype=float)

# -----------------------------
# 2) 画图参数
# -----------------------------
n_groups = len(models)
n_series = len(series_names)
x = np.arange(n_groups)

total_group_width = 0.80
bar_w = total_group_width / n_series

cmap = plt.get_cmap("Blues")
colors = [
    (0.45, 0.45, 0.45, 1.0),  # e3nn：中性灰
    cmap(0.55),               # cuEq：蓝
    (0.70, 0.20, 0.22, 1.0),  # fastEq：深红
]

def draw_grouped(ax, values, title_text):
    """grouped bar + (cuEq vs e3nn) + (fastEq vs e3nn) speedup 标注"""
    # bars
    for j in range(n_series):
        offset = (j - (n_series - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            values[:, j],
            width=bar_w * 0.95,
            color=colors[j],
            label=series_names[j],
            zorder=3,
        )

    # indices
    idx_e3nn   = series_names.index("e3nn")
    idx_openeq = series_names.index("cuEq")
    idx_fasteq = series_names.index("fastEq")

    # speedups (越大越好)：e3nn / target
    sp_openeq = values[:, idx_e3nn] / values[:, idx_openeq]
    sp_fasteq = values[:, idx_e3nn] / values[:, idx_fasteq]

    # offsets: 标注放到对应柱子上方
    openeq_offset = (idx_openeq - (n_series - 1) / 2) * bar_w
    fasteq_offset = (idx_fasteq - (n_series - 1) / 2) * bar_w

    y_max = float(np.max(values))
    pad = 0.03 * y_max

    # 为避免两行文字挤在一起：cuEq 标在更高的位置（pad2）
    pad2 = 0.08 * y_max

    for i in range(n_groups):
        # cuEq vs e3nn
        ax.text(
            x[i] + openeq_offset,
            values[i, idx_openeq] + pad2,
            f"×{sp_openeq[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            zorder=5,
        )
        # fastEq vs e3nn
        ax.text(
            x[i] + fasteq_offset,
            values[i, idx_fasteq] + pad,
            f"×{sp_fasteq[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            zorder=5,
        )

    # axes style
    ax.grid(axis="y", linewidth=1, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(models,  ha="right")

    # y 上界多留点空间给两套标注
    ax.set_ylim(0, y_max * 1.35)

    ax.text(
        0.02, 0.95,
        title_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13
    )
    ax.set_ylabel("Latency (ms)")


# -----------------------------
# 3) 四个子图：FP64/FP32 的 FWD/BWD
# -----------------------------
fig, axes = plt.subplots(3, 2, figsize=(15.5, 7.6), sharex=True)

draw_grouped(axes[0, 0], values_fp64_fwd, "H100, FP64, Forward, atoms: 2208")
draw_grouped(axes[0, 1], values_fp32_fwd, "H100, FP32, Forward, atoms: 2208")
draw_grouped(axes[1, 0], values_fp64_bwd, "H100, FP64, Backward, atoms: 2208")
draw_grouped(axes[1, 1], values_fp32_bwd, "H100, FP32, Backward, atoms: 2208")
draw_grouped(axes[2, 0], values_fp64_total, "H100, FP64, Total, atoms: 2208")
draw_grouped(axes[2, 1], values_fp32_total, "H100, FP32, Total, atoms: 2208")

# 上面一行不显示 x label（更干净）
for ax in axes[0, :]:
    ax.tick_params(labelbottom=False)

# 只放一次图例（顶部居中）
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    ncol=3, fontsize=11, frameon=False,
    loc="upper center", bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

out_path = "./nv_e2e_evaluation_fp64_fp32_fwd_bwd_speedup.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
