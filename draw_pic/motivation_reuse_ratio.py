import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ============================================================
# 1. Data
# ============================================================
models = [
    "NequIP-L",
    "Allegro-L",
    "MACE-OFF-M",
    "SevenNet0",
]

# Reusable occurrences / total occurrences
node_features = np.array([91.30, 93.48, 88.37, 92.92])
edge_features = np.array([95.33, 95.47, 95.35, 97.17])
weights = np.array([96.60, 95.47, 81.40, 97.94])
accumulators = np.array([63.69, 95.47, 53.49, 68.73])


# ============================================================
# 2. Global plotting configuration
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 3. Create figure
# ============================================================
fig, ax = plt.subplots(figsize=(8.4, 5.3))

x = np.arange(len(models))
bar_width = 0.19

colors = {
    "node": "#2F7EB8",
    "edge": "#69B0CF",
    "weight": "#4EAD38",
    "acc": "#FA7818",
}


# ============================================================
# 4. Draw grouped bars
# ============================================================
bars_node = ax.bar(
    x - 1.5 * bar_width,
    node_features,
    width=bar_width,
    color=colors["node"],
    edgecolor="white",
    linewidth=0.7,
    label="Node Features",
    zorder=3,
)

bars_edge = ax.bar(
    x - 0.5 * bar_width,
    edge_features,
    width=bar_width,
    color=colors["edge"],
    edgecolor="white",
    linewidth=0.7,
    label="Edge Features",
    zorder=3,
)

bars_weight = ax.bar(
    x + 0.5 * bar_width,
    weights,
    width=bar_width,
    color=colors["weight"],
    edgecolor="white",
    linewidth=0.7,
    label="Weights",
    zorder=3,
)

bars_acc = ax.bar(
    x + 1.5 * bar_width,
    accumulators,
    width=bar_width,
    color=colors["acc"],
    edgecolor="white",
    linewidth=0.7,
    label="Accumulators",
    zorder=3,
)


# ============================================================
# 5. Add value labels
# ============================================================
def add_value_labels(bars):
    for bar in bars:
        value = bar.get_height()

        ax.annotate(
            f"{value:.1f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=8,
            clip_on=False,
            zorder=5,
        )


add_value_labels(bars_node)
add_value_labels(bars_edge)
add_value_labels(bars_weight)
add_value_labels(bars_acc)


# ============================================================
# 6. Configure axes
# ============================================================
ax.set_ylabel(
    "Reusable Occurrences / Total Occurrences",
    fontsize=10,
    fontweight="normal",
    labelpad=6,
)

ax.set_xticks(x)
ax.set_xticklabels(models)

# Keep a zero baseline to avoid exaggerating percentage differences.
# Extra space above 100% is reserved for value labels.
ax.set_ylim(0, 106)

ax.set_yticks(np.arange(0, 101, 20))
ax.yaxis.set_major_formatter(
    PercentFormatter(xmax=100, decimals=0)
)

ax.tick_params(
    axis="both",
    direction="out",
    width=1.0,
    length=4,
)


# ============================================================
# 7. Grid and borders
# ============================================================
ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    alpha=0.35,
    zorder=0,
)

ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(1.1)


# ============================================================
# 8. Title and legend
# ============================================================

# Use a figure-level legend so it does not overlap with value labels.
handles, labels = ax.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.88),
    ncol=4,
    frameon=False,
    columnspacing=1.7,
    handlelength=2.1,
    handletextpad=0.6,
)

# Reserve space at the top for title and legend.
fig.subplots_adjust(
    left=0.13,
    right=0.98,
    bottom=0.14,
    top=0.79,
)


# ============================================================
# 9. Save and display
# ============================================================

plt.savefig(
    "cross_path_reuse_opportunity.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()