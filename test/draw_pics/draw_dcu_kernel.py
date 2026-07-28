import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) 数据
# -----------------------------
models = ["CWTP Fused", "FCTP", "STC"]
series_names = ["e3nn", "openEq", "fastEq"]

# ================ BW100 ==============

# mace-OFF medium
values_fp64_fwd = np.array([
    [40.09,  9,  6],
    [2.35, 2.35,  1],
    [72.34, 41, 3],
], dtype=float)

values_fp32_fwd = np.array([
    [30,  7,  4],
    [3.37, 1.37,  0.7],
    [32, 21, 2],
], dtype=float)

# mace-OFF small
values_fp64_bwd = np.array([
    [4.1,  2.6,  1.4],
    [1.5, 1.5,  0.9],
    [24, 8, 0.6],
], dtype=float)

values_fp32_bwd = np.array([
    [3.8,  2.1,  1.3],
    [0.9, 0.9,  0.6],
    [7.1, 7.1, 0.6],
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
    cmap(0.55),               # openEq：蓝
    (0.70, 0.20, 0.22, 1.0),  # fastEq：深红
]

def draw_grouped(ax, values, title_text):
    """grouped bar + (openEq vs e3nn) + (fastEq vs e3nn) speedup 标注"""
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
    idx_openeq = series_names.index("openEq")
    idx_fasteq = series_names.index("fastEq")

    # speedups (越大越好)：e3nn / target
    sp_openeq = values[:, idx_e3nn] / values[:, idx_openeq]
    sp_fasteq = values[:, idx_e3nn] / values[:, idx_fasteq]

    # offsets: 标注放到对应柱子上方
    openeq_offset = (idx_openeq - (n_series - 1) / 2) * bar_w
    fasteq_offset = (idx_fasteq - (n_series - 1) / 2) * bar_w

    y_max = float(np.max(values))
    pad = 0.03 * y_max

    # 为避免两行文字挤在一起：openEq 标在更高的位置（pad2）
    pad2 = 0.08 * y_max

    for i in range(n_groups):
        # openEq vs e3nn
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
    ax.set_xticklabels(models, ha="right")

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
fig, axes = plt.subplots(2, 2, figsize=(15.5, 7.6), sharex=True)

draw_grouped(axes[0, 0], values_fp64_fwd, "BW200, FP64, MACE-OFF Medium Forward, atoms: 2208")
draw_grouped(axes[0, 1], values_fp32_fwd, "BW200, FP32, MACE-OFF Medium Forward, atoms: 2208")
draw_grouped(axes[1, 0], values_fp64_bwd, "BW200, FP64, MACE-OFF Small Forward, atoms: 2208")
draw_grouped(axes[1, 1], values_fp32_bwd, "BW200, FP32, MACE-OFF Small Forward, atoms: 2208")

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

out_path = "./dcu_kernel_evaluation_fp64_fp32_fwd_bwd_speedup.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
