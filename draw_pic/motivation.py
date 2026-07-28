import numpy as np
import matplotlib.pyplot as plt
from textwrap import fill

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 20,
    "axes.labelsize": 16,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# Data
# ============================================================

# (a) Left subplot: stacked bar chart
models_left = ["MF-S", "MF-M", "MF-L", "MP-M", "7Net"]
cwtp = np.array([18.0, 22.4, 63.0, 26.4, 70.7])
stc = np.array([32.5, 42.9, 21.4, 18.9, 0.0])
fctp = np.array([19.1, 13.9, 6.8, 8.8, 0.0])
eq_linear = np.array([14.5, 13.7, 5.3, 34.6, 16.9])

# (b) Right subplot: grouped bar chart
models_right = ["NequIP", "Allegro", "MACE", "7Net"]
node = np.array([91.3, 93.5, 88.4, 92.9])
edge = np.array([95.3, 95.5, 95.3, 97.2])
weight = np.array([96.6, 95.5, 81.4, 97.9])
acc = np.array([63.7, 95.5, 53.5, 68.7])

# ============================================================
# Colors
# ============================================================
# Try to match the screenshot style:
color_blue = "#a9a7ee"      # lavender blue
color_red = "#eca3a3"       # soft pink/red
color_beige = "#dccdb8"     # beige
color_gray = "#8a8a8a"      # gray

# Use same color logic in both subplots
col_cwtp = color_blue
col_stc = color_red
col_fctp = color_beige
col_eq = color_gray

col_node = color_blue
col_edge = color_red
col_weight = color_beige
col_acc = color_gray

# ============================================================
# Create figure and subplots
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10.2, 6.8))
ax1, ax2 = axes

# ============================================================
# (a) Stacked bar chart
# ============================================================
x1 = np.arange(len(models_left))
w1 = 0.42

ax1.bar(x1, cwtp, width=w1, color=col_cwtp, label="CWTP")
ax1.bar(x1, stc, width=w1, bottom=cwtp, color=col_stc, label="STC")
ax1.bar(x1, fctp, width=w1, bottom=cwtp + stc, color=col_fctp, label="FCTP")
ax1.bar(x1, eq_linear, width=w1, bottom=cwtp + stc + fctp, color=col_eq, label="Eq.-Linear")

ax1.set_ylim(0, 100)
ax1.set_yticks(np.arange(0, 101, 20))
ax1.set_ylabel("Time (%)")
ax1.set_xticks(x1)
ax1.set_xticklabels(models_left)

# Ticks on four sides, pointing inward
ax1.tick_params(axis="both", direction="in", top=True, right=True, length=7)

# Legend inside the plot
leg1 = ax1.legend(
    loc="center",
    bbox_to_anchor=(0.7, 0.2),
    ncol=2,
    frameon=True,
    framealpha=1.0,
    borderpad=0.4,
    handlelength=0.8,
    columnspacing=0.8,
)
leg1.get_frame().set_edgecolor("black")
leg1.get_frame().set_linewidth(0.8)

# ============================================================
# (b) Grouped bar chart
# ============================================================
x2 = np.arange(len(models_right))
w2 = 0.16

ax2.bar(x2 - 1.5 * w2, node, width=w2, color=col_node, label="Node")
ax2.bar(x2 - 0.5 * w2, edge, width=w2, color=col_edge, label="Edge")
ax2.bar(x2 + 0.5 * w2, weight, width=w2, color=col_weight, label="Weight")
ax2.bar(x2 + 1.5 * w2, acc, width=w2, color=col_acc, label="Accumulator")

ax2.set_ylim(0, 100)
ax2.set_yticks(np.arange(0, 101, 20))
ax2.set_yticklabels([f"{i}%" for i in range(0, 101, 20)])
ax2.set_ylabel("Reuse (%)")
ax2.set_xticks(x2)
ax2.set_xticklabels(models_right)

ax2.tick_params(axis="both", direction="in", top=True, right=True, length=7)

# Light horizontal grid
ax2.grid(axis="y", linestyle=(0, (4, 4)), linewidth=1.0, alpha=0.35)
ax2.set_axisbelow(True)

leg2 = ax2.legend(
    loc="center",
    bbox_to_anchor=(0.7, 0.2),
    ncol=2,
    frameon=True,
    framealpha=1.0,
    borderpad=0.4,
    handlelength=0.8,
    columnspacing=0.8,
)
leg2.get_frame().set_edgecolor("black")
leg2.get_frame().set_linewidth(0.8)

# ============================================================
# Border thickness
# ============================================================
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

# ============================================================
# Subfigure labels: (a), (b)
# ============================================================
ax1.text(0.5, -0.15, "(a)", transform=ax1.transAxes,
         ha="center", va="center", fontsize=20)
ax2.text(0.5, -0.15, "(b)", transform=ax2.transAxes,
         ha="center", va="center", fontsize=20)

# ============================================================
# Caption
# ============================================================
# Leave space for caption
fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.48, wspace=0.30)

fig.text(
    0.05, 0.08,
    "",
    ha="left",
    va="bottom",
    fontsize=18,
)

# ============================================================
# Save and show
# ============================================================
plt.savefig("two_subplots_operator_reuse.png", dpi=300, bbox_inches="tight")
plt.show()