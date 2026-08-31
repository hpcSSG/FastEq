import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
models = [
    "MACE-OFF small",
    "MACE-OFF medium",
    "MACE-OFF large",
    "MACE-MP-0 medium",
    "SevenNet-0",
]

data = {
    "cwtp":      np.array([18.0, 22.4, 63.0, 26.4, 70.7]),
    "stc":       np.array([32.5, 42.9, 21.4, 18.9,  0.0]),
    "fctp":      np.array([19.1, 13.9,  6.8,  8.8,  0.0]),
    "eq-linear": np.array([14.5, 13.7,  5.3, 34.6, 16.9]),
}

# ----------------------------------------------------------------------
# Colors
# cwtp and stc: two different reds
# fctp and eq-linear: grays
# ----------------------------------------------------------------------
colors = {
    "cwtp": "#B22222",      # firebrick red
    "stc": "#1F4E79",       # dark blue
    "fctp": "#9E9E9E",      # medium gray
    "eq-linear": "#D0D0D0", # light gray
}

# Plot order: bottom -> top
plot_order = ["cwtp", "stc", "fctp", "eq-linear"]

# ----------------------------------------------------------------------
# Plot configuration
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

fig, ax = plt.subplots(figsize=(16, 8.2))

x = np.arange(len(models))
width = 0.50
bottom = np.zeros(len(models))

# ----------------------------------------------------------------------
# Draw stacked bars
# ----------------------------------------------------------------------
for operator in plot_order:
    values = data[operator]

    ax.bar(
        x,
        values,
        width=width,
        bottom=bottom,
        label=operator,
        color=colors[operator],
        edgecolor="none",
    )

    # percentage labels
    for i, value in enumerate(values):
        if value > 0:
            # white text for darker bars, black for lighter bars
            text_color = "white" if operator in {"cwtp", "stc", "fctp"} else "black"
            ax.text(
                x[i],
                bottom[i] + value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=14,
                color=text_color,
            )

    bottom += values

# ----------------------------------------------------------------------
# Axes and annotations
# ----------------------------------------------------------------------
""" ax.set_title(
    "MACE-OFF/SevenNet-0 operator time breakdown with cuEquivariance on Nvidia H100",
    pad=12,
) """
ax.set_ylabel("Share of total inference time (%)")
#ax.set_xlabel("Model size (MACE-OFF)")

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))

ax.grid(False)
ax.tick_params(axis="both", direction="out", length=5, width=1)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.9)

fig.tight_layout()

# Save figures
plt.savefig("operator_time_breakdown_red_gray.pdf", bbox_inches="tight")
plt.savefig("operator_time_breakdown_red_gray.png", dpi=300, bbox_inches="tight")

plt.show()