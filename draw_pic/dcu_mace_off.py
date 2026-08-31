from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parent

# AAAI double-column full-width figure.
FIGURE_WIDTH = 7.0
FIGURE_HEIGHT = 2.35

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Data from the two provided tables
# OOM entries are excluded because their runtimes are unavailable.
# ============================================================

DATA_FP32 = {
    "fasteq_hip": {
        "x": np.array([
            192, 648, 1536, 3000, 5184, 12288,
            24000, 41472, 65856, 98304, 139968, 192000,
        ]),
        "y": np.array([
            23.246480, 24.790000, 29.470878, 36.697601,
            50.825201, 94.705761, 170.584160, 284.195038,
            444.323120, 654.497437, 922.522888, 1249.117065,
        ]),
    },
    "openeq": {
        "x": np.array([
            192, 648, 1536, 3000, 5184,
            12288, 24000, 41472, 65856,
        ]),
        "y": np.array([
            20.192000, 22.542400, 39.734720, 70.369678,
            116.225521, 265.448166, 510.516403,
            876.105042, 1382.719238,
        ]),
    },
    "e3nn": {
        "x": np.array([
            192, 648, 1536, 3000, 5184,
            12288, 24000, 41472,
        ]),
        "y": np.array([
            24.565280, 28.675839, 58.184160, 105.447357,
            176.274719, 407.218323, 787.243408, 1377.042419,
        ]),
    },
}


DATA_FP64 = {
    "fasteq_hip": {
        "x": np.array([
            192, 648, 1536, 3000, 5184,
            12288, 24000, 41472, 65856, 98304,
        ]),
        "y": np.array([
            24.906467, 29.382784, 40.647097, 62.832283,
            78.460438, 165.079903, 310.963333, 523.899780,
            821.719757, 1215.626831,
        ]),
    },
    "openeq": {
        "x": np.array([
            192, 648, 1536, 3000, 5184, 12288, 24000,
        ]),
        "y": np.array([
            20.153190, 41.160141, 86.132195, 160.292229,
            253.541039, 592.546234, 1143.966675,
        ]),
    },
    "e3nn": {
        "x": np.array([
            192, 648, 1536, 3000, 5184, 12288, 24000,
        ]),
        "y": np.array([
            22.386868, 48.895569, 106.275776, 198.788429,
            319.666901, 747.351654, 1438.671265,
        ]),
    },
}


LABELS = {
    "fasteq_hip": "FastEq",
    "openeq": "OpenEq",
    "e3nn": "e3nn",
}

COLORS = {
    "fasteq_hip": "#d62728",
    "openeq": "#ff7f0e",
    "e3nn": "#1f77b4",
}

MARKERS = {
    "fasteq_hip": "o",
    "openeq": "^",
    "e3nn": "D",
}


def format_k(value, position):
    """Format large atom counts as 25K, 50K, etc."""
    if value == 0:
        return "0"

    if abs(value) >= 1000:
        scaled = value / 1000.0
        if float(scaled).is_integer():
            return f"{int(scaled)}K"
        return f"{scaled:g}K"

    return f"{int(value)}"


def plot_panel(ax, data, title, xlim, xticks):
    """Draw one precision panel."""
    # Draw FastEq last so that it stays visible in overlapping regions.
    plot_order = ("e3nn", "openeq", "fasteq_hip")

    for backend in plot_order:
        series = data[backend]

        ax.plot(
            series["x"],
            series["y"],
            label=LABELS[backend],
            color=COLORS[backend],
            marker=MARKERS[backend],
            linewidth=1.5,
            markersize=4.0,
            markeredgewidth=0.45,
            markeredgecolor=COLORS[backend],
            clip_on=True,
            zorder=3 if backend == "fasteq_hip" else 2,
        )

    ax.set_title(title, pad=3)
    ax.set_xlabel("Number of Simulated Atoms", labelpad=2)

    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1600)
    ax.set_xticks(xticks)
    ax.yaxis.set_major_locator(MultipleLocator(400))
    ax.xaxis.set_major_formatter(FuncFormatter(format_k))

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.55,
        alpha=0.35,
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        pad=1.8,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.legend(
        loc="lower right",
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="0.75",
        borderpad=0.22,
        labelspacing=0.22,
        handlelength=1.45,
        handletextpad=0.35,
    )


def main():
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        sharey=True,
    )

    plot_panel(
        axes[0],
        DATA_FP32,
        "(a) FP32",
        xlim=(0, 200000),
        xticks=[0, 50000, 100000, 150000, 200000],
    )

    plot_panel(
        axes[1],
        DATA_FP64,
        "(b) FP64",
        xlim=(0, 105000),
        xticks=[0, 25000, 50000, 75000, 100000],
    )

    axes[0].set_ylabel("Inference Time (ms)", labelpad=2)

    # The right panel shares the y-axis; remove redundant tick labels.
    axes[1].tick_params(labelleft=False)

    fig.subplots_adjust(
        left=0.085,
        right=0.992,
        top=0.88,
        bottom=0.22,
        wspace=0.16,
    )

    for suffix in ("png", "pdf"):
        fig.savefig(
            ROOT / f"mace_off_small_fp32_fp64.{suffix}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )

    plt.close(fig)


if __name__ == "__main__":
    main()