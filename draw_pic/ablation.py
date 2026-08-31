import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ============================================================
# Figure configuration
# ============================================================

FIGURE_WIDTH = 6.0
FIGURE_HEIGHT = 3

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Data
# ============================================================

stages = [
    "Unroll",
    "+ Path Scheduling",
    "+ Candidate Generation",
]

# (a) MACE-OFF Large, forward
mace_labels = [
    "CWTP\n(path=215)",
    "STC\n(path=1,225)",
]

mace_data = np.array([
    [2.16, 1.68, 1.33],
    [1.73, 1.13, 0.39],
])

# (b) SevenNet-omini, backward
sevennet_labels = [
    "CWTP-0\n(path=1,554)",
    "CWTP-1\n(path=1,490)",
]

sevennet_data = np.array([
    [15.45, 12.57, 8.18],
    [14.11, 12.00, 8.145],
])


COLORS = [
    "#1f77b4",  # Unroll
    "#ff7f0e",  # Path scheduling
    "#d62728",  # Candidate generation
]


# ============================================================
# Helper functions
# ============================================================

def annotate_pairwise_speedup(
    ax,
    x_baseline,
    x_optimized,
    baseline,
    optimized,
    ylim,
):
    """
    Annotate the speedup between two optimization stages.

    Speedup = baseline latency / optimized latency
    """
    speedup = baseline / optimized
    x_arrow = (x_baseline + x_optimized) / 2.0

    # Horizontal dashed guide at baseline latency.
    ax.plot(
        [x_baseline, x_arrow],
        [baseline, baseline],
        linestyle="--",
        linewidth=0.65,
        color="0.35",
        zorder=5,
    )

    # Horizontal dashed guide at optimized latency.
    ax.plot(
        [x_optimized, x_arrow],
        [optimized, optimized],
        linestyle="--",
        linewidth=0.65,
        color="0.35",
        zorder=5,
    )

    # Straight arrow from baseline to optimized stage.
    # Matplotlib arrows point from xytext to xy.
    ax.annotate(
        "",
        xy=(x_arrow, optimized),
        xytext=(x_arrow, baseline),
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 1.0,
            "color": "black",
            "shrinkA": 1.5,
            "shrinkB": 1.5,
        },
        zorder=6,
    )

    ax.text(
        x_arrow,
        baseline + ylim[1] * 0.008,
        f"{speedup:.2f}$\\times$",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        zorder=7,
    )


def draw_subplot(
    ax,
    data,
    category_labels,
    subtitle,
    ylim,
    major_tick,
):
    num_categories = len(category_labels)
    num_stages = data.shape[1]

    x = np.arange(num_categories)
    bar_width = 0.22

    offsets = (
        np.arange(num_stages)
        - (num_stages - 1) / 2
    ) * bar_width

    # --------------------------------------------------------
    # Draw bars
    # --------------------------------------------------------

    for stage_idx in range(num_stages):
        ax.bar(
            x + offsets[stage_idx],
            data[:, stage_idx],
            width=bar_width,
            label=stages[stage_idx],
            color=COLORS[stage_idx],
            edgecolor="none",
            zorder=3,
        )

    # --------------------------------------------------------
    # Add pairwise speedups
    # --------------------------------------------------------

    for category_idx in range(num_categories):
        bar_x = x[category_idx] + offsets

        # Unroll -> Path Scheduling
        annotate_pairwise_speedup(
            ax=ax,
            x_baseline=bar_x[0],
            x_optimized=bar_x[1],
            baseline=data[category_idx, 0],
            optimized=data[category_idx, 1],
            ylim=ylim,
        )

        # Path Scheduling -> Candidate Generation
        annotate_pairwise_speedup(
            ax=ax,
            x_baseline=bar_x[1],
            x_optimized=bar_x[2],
            baseline=data[category_idx, 1],
            optimized=data[category_idx, 2],
            ylim=ylim,
        )

    # --------------------------------------------------------
    # Axis styling
    # --------------------------------------------------------

    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, fontsize=10,)

    ax.set_xlim(
        -0.50,
        num_categories - 1 + 0.50,
    )
    ax.set_ylim(*ylim)

    ax.yaxis.set_major_locator(
        MultipleLocator(major_tick)
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.55,
        alpha=0.35,
        zorder=0,
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        pad=2,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # --------------------------------------------------------
    # Independent legend inside each subplot
    # --------------------------------------------------------

    # Compact horizontal legend inside each subplot.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="0.75",
        borderpad=0.22,
        columnspacing=0.75,
        handlelength=1.15,
        handletextpad=0.30,
        fontsize=7,
    )

    # --------------------------------------------------------
    # Subfigure title below the x-axis labels
    # --------------------------------------------------------

    ax.text(
        0.5,
        -0.2,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
)


draw_subplot(
    ax=axes[0],
    data=mace_data,
    category_labels=mace_labels,
    subtitle="(a) MACE-OFF Large Forward",
    ylim=(0, 2.85),
    major_tick=0.5,
)


draw_subplot(
    ax=axes[1],
    data=sevennet_data,
    category_labels=sevennet_labels,
    subtitle="(b) SevenNet-omini Backward",
    ylim=(0, 19.5),
    major_tick=4,
)


axes[0].set_ylabel(
    "Latency (ms)",
    labelpad=3,
    fontsize=10,
)


# ============================================================
# Layout and output
# ============================================================

fig.subplots_adjust(
    left=0.085,
    right=0.99,
    top=0.97,
    bottom=0.29,
    wspace=0.08,
)

fig.savefig(
    "mace_sevennet_optimization_breakdown.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
)

fig.savefig(
    "mace_sevennet_optimization_breakdown.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()