import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Data
# ============================================================

# (a) MACE-OFF large, path=215, forward
mace_unroll = np.array([1, 4, 8, 16, 32, 64, 128, 215])
mace_runtime = np.array([10.65, 5.27, 4.31, 3.83, 3.53, 3.79, 3.35, 3.35])
mace_registers = np.array([40, 40, 40, 56, 64, 80, 96, 154])

# (b) SevenNet, path=1554, backward
sevennet_unroll = np.array([1, 4, 8, 16, 32, 64, 128, 215])
sevennet_runtime = np.array(
    [45.00, 30.97, 25.43, 20.14, 16.69, 14.61, 16.08, 16.94]
)

sevennet_registers = np.array(
    [220, 220, 220, 232, 254, 254, 255 + 26, 255 + 66]
)

# Equal spacing for different unroll factors
x_mace = np.arange(len(mace_unroll))
x_sevennet = np.arange(len(sevennet_unroll))

# ============================================================
# Plot configuration
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

runtime_color = "#2878B5"
register_color = "#C82423"

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))


# ============================================================
# Helper function
# ============================================================
def draw_dual_axis_plot(
    ax,
    x,
    unroll,
    runtime,
    registers,
    panel_label,
    runtime_ylim,
    register_ylim,
    add_reg_limit=False,
    show_runtime_label=False,
    show_register_label=False,
):
    ax_reg = ax.twinx()

    runtime_line = ax.plot(
        x,
        runtime,
        color=runtime_color,
        marker="o",
        markersize=5,
        linewidth=1.8,
        label="Runtime (ms)",
        zorder=3,
    )

    register_line = ax_reg.plot(
        x,
        registers,
        color=register_color,
        marker="s",
        markersize=5,
        linewidth=1.8,
        linestyle="--",
        label="Registers/thread",
        zorder=3,
    )

    # Register upper bound
    if add_reg_limit:
        reg_limit = ax_reg.axhline(
            y=255,
            color=register_color,
            linestyle=":",
            linewidth=1.5,
            label="Register upper bound",
            zorder=2,
        )
        legend_lines = runtime_line + register_line + [reg_limit]
    else:
        legend_lines = runtime_line + register_line

    ax.set_xlabel("Unroll factor")

    # Only retain the outermost y-axis labels
    if show_runtime_label:
        ax.set_ylabel(
            "Runtime (ms)",
            color=runtime_color,
            labelpad=1,          # reduce distance from y-axis
        )
    else:
        ax.set_ylabel("")

    if show_register_label:
        ax_reg.set_ylabel(
            "Registers/thread",
            color=register_color,
            labelpad=4,
        )
    else:
        ax_reg.set_ylabel("")

    ax.set_xticks(x)
    ax.set_xticklabels(unroll)

    ax.set_ylim(*runtime_ylim)
    ax_reg.set_ylim(*register_ylim)

    ax.tick_params(axis="y", colors=runtime_color)
    ax_reg.tick_params(axis="y", colors=register_color)

    ax.spines["left"].set_color(runtime_color)
    ax_reg.spines["right"].set_color(register_color)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
    )
    ax.set_axisbelow(True)

    labels = [line.get_label() for line in legend_lines]
    ax.legend(
        legend_lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=1,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        handlelength=2.0,
        columnspacing=1.2,
    )

    ax.text(
        0.5,
        -0.27,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    return ax_reg


# ============================================================
# (a) MACE
# Only show the left Runtime (ms) label
# ============================================================
draw_dual_axis_plot(
    ax=axes[0],
    x=x_mace,
    unroll=mace_unroll,
    runtime=mace_runtime,
    registers=mace_registers,
    panel_label="(a) MACE-OFF large, path=215, fwd",
    runtime_ylim=(2.5, 11.5),
    register_ylim=(20, 170),
    add_reg_limit=False,
    show_runtime_label=True,
    show_register_label=False,
)

# ============================================================
# (b) SevenNet
# Only show the right Registers/thread label
# ============================================================
draw_dual_axis_plot(
    ax=axes[1],
    x=x_sevennet,
    unroll=sevennet_unroll,
    runtime=sevennet_runtime,
    registers=sevennet_registers,
    panel_label="(b) SevenNet, path=1554, bwd",
    runtime_ylim=(10, 48),
    register_ylim=(200, 330),
    add_reg_limit=True,
    show_runtime_label=False,
    show_register_label=True,
)

# ============================================================
# Layout and output
# ============================================================
fig.subplots_adjust(
    left=0.075,
    right=0.925,
    bottom=0.27,
    top=0.86,
    wspace=0.23,
)

plt.savefig(
    "unroll_runtime_registers.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()