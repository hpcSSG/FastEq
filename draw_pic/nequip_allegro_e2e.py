import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter


# ============================================================
# Data
# ============================================================

models = ["NequIP-S", "NequIP-M", "NequIP-L", "Allegro"]

# atoms = 1K
fasteq_1k = np.array([
    115.361,
    59.02,
    33.91,
    5.79,
])

cueq_1k = np.array([
    107.89,
    21.38,
    4.85,
    5.67,
])

# atoms = 4K
fasteq_4k = np.array([
    108.913,
    38.19,
    11.98,
    1.41,
])

cueq_4k = np.array([
    91.78,
    5.83,
    1.16,
    1.44,
])

# FastTP speedup over cuEq
speedup_1k = fasteq_1k / cueq_1k
speedup_4k = fasteq_4k / cueq_4k


# ============================================================
# AAAI half-column style
# ============================================================

FIGURE_WIDTH = 3.35
FIGURE_HEIGHT = 2.35

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",

    "font.size": 7.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3,

    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "hatch.linewidth": 0.4,
    "axes.unicode_minus": False,
})

FASTEQ_COLOR = "#C43C39"
CUEQ_COLOR = "#8C8C8C"
EDGE_COLOR = "#202020"
BAR_EDGE_WIDTH = 0.45
BAR_WIDTH = 0.13
HATCH_1K = ""
HATCH_4K = "///"


# ============================================================
# Helpers
# ============================================================

def add_speedup_labels(ax, bars, speedups, fontsize=5.2):
    for bar, s in zip(bars, speedups):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.08,
            f"{s:.2f}×",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
            zorder=5,
        )

def log_tick_formatter(value, _):
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:g}"


# ============================================================
# Plot
# ============================================================

x = np.arange(len(models))
offsets = np.array([
    -1.5 * BAR_WIDTH,
    -0.5 * BAR_WIDTH,
     0.5 * BAR_WIDTH,
     1.5 * BAR_WIDTH,
])

fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

bars_fasteq_1k = ax.bar(
    x + offsets[0], fasteq_1k,
    width=BAR_WIDTH,
    color=FASTEQ_COLOR,
    edgecolor=EDGE_COLOR,
    linewidth=BAR_EDGE_WIDTH,
    hatch=HATCH_1K,
    zorder=3,
)

bars_cueq_1k = ax.bar(
    x + offsets[1], cueq_1k,
    width=BAR_WIDTH,
    color=CUEQ_COLOR,
    edgecolor=EDGE_COLOR,
    linewidth=BAR_EDGE_WIDTH,
    hatch=HATCH_1K,
    zorder=3,
)

bars_fasteq_4k = ax.bar(
    x + offsets[2], fasteq_4k,
    width=BAR_WIDTH,
    color=FASTEQ_COLOR,
    edgecolor=EDGE_COLOR,
    linewidth=BAR_EDGE_WIDTH,
    hatch=HATCH_4K,
    zorder=3,
)

bars_cueq_4k = ax.bar(
    x + offsets[3], cueq_4k,
    width=BAR_WIDTH,
    color=CUEQ_COLOR,
    edgecolor=EDGE_COLOR,
    linewidth=BAR_EDGE_WIDTH,
    hatch=HATCH_4K,
    zorder=3,
)


# ============================================================
# Axes
# ============================================================

ax.set_ylabel("Throughput (timesteps/s)", labelpad=2)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_xlim(-0.5, len(models) - 0.5)

ax.set_yscale("log")
ax.set_ylim(0.9, 180)

yticks = [1, 2, 5, 10, 20, 50, 100]
ax.yaxis.set_major_locator(FixedLocator(yticks))
ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

ax.grid(
    axis="y",
    which="major",
    linestyle=(0, (3, 2)),
    linewidth=0.45,
    color="0.80",
    alpha=0.8,
    zorder=0,
)

ax.tick_params(axis="both", which="major", direction="out", pad=1.5)
ax.tick_params(axis="y", which="minor", length=0)

for spine in ax.spines.values():
    spine.set_linewidth(0.7)


# ============================================================
# Legend
# ============================================================

legend_handles = [
    Patch(facecolor=FASTEQ_COLOR, edgecolor=EDGE_COLOR,
          linewidth=BAR_EDGE_WIDTH, label="FastTP, 1K"),
    Patch(facecolor=CUEQ_COLOR, edgecolor=EDGE_COLOR,
          linewidth=BAR_EDGE_WIDTH, label="cuEq, 1K"),
    Patch(facecolor=FASTEQ_COLOR, edgecolor=EDGE_COLOR,
          linewidth=BAR_EDGE_WIDTH, hatch=HATCH_4K, label="FastTP, 4K"),
    Patch(facecolor=CUEQ_COLOR, edgecolor=EDGE_COLOR,
          linewidth=BAR_EDGE_WIDTH, hatch=HATCH_4K, label="cuEq, 4K"),
]

leg = ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.7, 0.78),
    ncol=2,
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    facecolor="white",
    edgecolor="0.75",
    borderpad=0.22,
    labelspacing=0.25,
    handlelength=1.2,
    handletextpad=0.35,
    columnspacing=0.8,
)
leg.get_frame().set_linewidth(0.45)


# ============================================================
# Labels: only keep FastTP speedups
# ============================================================

add_speedup_labels(ax, bars_fasteq_1k, speedup_1k)
add_speedup_labels(ax, bars_fasteq_4k, speedup_4k)


# ============================================================
# Layout and save
# ============================================================

fig.subplots_adjust(
    left=0.16,
    right=0.995,
    bottom=0.22,
    top=0.78,
)


fig.savefig(
    "combined_nequip_allegro_fp32_halfcol.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.015,
)

plt.show()