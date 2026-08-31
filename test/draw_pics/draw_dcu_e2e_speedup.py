import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) 数据
# -----------------------------
models = ["MACE-OFF Small", "MACE-OFF Medium", "MACE-OFF Large"]
series_names = ["e3nn", "openEq", "fastEq"]

# ================ BW100 ==============
values_fp64_fwd = np.array([
    [61,  28,  14.8],
    [130, 75,  25.4],
    [792, 223, 79],
], dtype=float)

values_fp32_fwd = np.array([
    [24,  19,  11.6],
    [70,  45,  19.0],
    [226, 132, 45],
], dtype=float)

values_fp64_bwd = np.array([
    [58,  29,  23],
    [320, 78,  41],
    [950, 286, 152],
], dtype=float)

values_fp32_bwd = np.array([
    [41,  17,  14],
    [115, 39,  23],
    [388, 165, 62],
], dtype=float)

values_fp64_total = np.array([
    [119.79,  58.05,  37.9],
    [451.56, 154.39,  66.72],
    [1743.68, 510.06, 231.73],
], dtype=float)

values_fp32_total = np.array([
    [66.2,  36.61,  26.08],
    [185.15, 85.68,  42.39],
    [615.21, 297.58, 107.63],
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
    """
    纵轴画 speedup（相对 e3nn）：e3nn=1
    柱顶显示真实耗时（ms）
    """
    idx_e3nn = series_names.index("e3nn")

    # speedup_values: shape [n_groups, n_series]
    # speedup = e3nn_time / target_time
    speedup_values = values[:, idx_e3nn][:, None] / values
    speedup_values[:, idx_e3nn] = 1.0  # 强制 e3nn 恒为 1

    # 绘柱（用 speedup 值）
    for j in range(n_series):
        offset = (j - (n_series - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            speedup_values[:, j],
            width=bar_w * 0.95,
            color=colors[j],
            label=series_names[j],
            zorder=3,
        )

        # 柱顶标注真实耗时（ms）
        for i, b in enumerate(bars):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.02 * float(np.max(speedup_values)),
                f"{values[i, j]:.2f}ms",
                ha="center",
                va="bottom",
                fontsize=9,
                zorder=5,
            )

    # axes style
    ax.grid(axis="y", linewidth=1, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(models, ha="right")

    # y 上界给标注留空间
    y_max = float(np.max(speedup_values))
    ax.set_ylim(0, y_max * 1.25)

    ax.text(
        0.02, 0.95,
        title_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13
    )
    ax.set_ylabel("Speedup vs e3nn (×)")



# -----------------------------
# 3) 四个子图：FP64/FP32 的 FWD/BWD
# -----------------------------
fig, axes = plt.subplots(3, 2, figsize=(15.5, 7.6), sharex=True)

draw_grouped(axes[0, 0], values_fp64_fwd, "BW200, FP64, Forward, atoms: 2208")
draw_grouped(axes[0, 1], values_fp32_fwd, "BW200, FP32, Forward, atoms: 2208")
draw_grouped(axes[1, 0], values_fp64_bwd, "BW200, FP64, Backward, atoms: 2208")
draw_grouped(axes[1, 1], values_fp32_bwd, "BW200, FP32, Backward, atoms: 2208")
draw_grouped(axes[2, 0], values_fp64_total, "BW200, FP64, Total, atoms: 2208")
draw_grouped(axes[2, 1], values_fp32_total, "BW200, FP32, Total, atoms: 2208")

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

out_path = "./dcu_e2e_evaluation_fp64_fp32_fwd_bwd_speedup.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
