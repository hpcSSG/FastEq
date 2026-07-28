import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


# ============================================================
# Data: MACE-OFF Small FP32
# ============================================================

mace_atoms = np.array([
    192,
    648,
    1536,
    3000,
    5184,
    12288,
    24000,
    41472,
    65856,
    98304,
    117912,
    139968,
    192000,
    222264,
])

mace_fasteq = np.array([
    16.759276599623263,
    25.424715201370418,
    38.03455980378203,
    63.60123080085032,
    93.86208321084268,
    226.5598708007019,
    459.3549905985128,
    825.2432755951304,
    1288.6043348000385,
    1971.1644539958797,
    2921.4739045943134,
    3986.0751414031256,
    4639.946709008655,
    np.nan,
])

mace_cueq = np.array([
    21.56334000173956,
    28.879656398203224,
    42.181493394309655,
    68.14464880153537,
    96.21388100204058,
    230.0554321904201,
    457.1950417943299,
    846.4174863998778,
    1273.5645960085094,
    1954.6546879981179,
    2914.8211087915115,
    3962.83895879169,
    np.nan,
    np.nan,
])

mace_openeq = np.array([
    21.361155342310667,
    30.296109477058053,
    52.4792717769742,
    95.21473618224263,
    148.34089623764157,
    363.41949701309204,
    855.8876396156847,
    1321.7705802060664,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
])

mace_e3nn = np.array([
    16.93520499393344,
    25.948624999728054,
    47.96108300215565,
    86.1810031987261,
    136.2629100040067,
    371.63265320123173,
    668.3164326066617,
    1173.7415871932171,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
])



mace_large_atoms = np.array([
    192,
    648,
    1536,
    3000,
    5184,
    12288,
    24000,
])

mace_fasteq = np.array([
    16.759276599623263,
    25.424715201370418,
    38.03455980378203,
    63.60123080085032,
    93.86208321084268,
    226.5598708007019,
    459.3549905985128,
    825.2432755951304,
    1288.6043348000385,
    1971.1644539958797,
    2921.4739045943134,
    3986.0751414031256,
    4639.946709008655,
    np.nan,
])

mace_cueq = np.array([
    21.56334000173956,
    28.879656398203224,
    42.181493394309655,
    68.14464880153537,
    96.21388100204058,
    230.0554321904201,
    457.1950417943299,
    846.4174863998778,
    1273.5645960085094,
    1954.6546879981179,
    2914.8211087915115,
    3962.83895879169,
    np.nan,
    np.nan,
])

mace_openeq = np.array([
    21.361155342310667,
    30.296109477058053,
    52.4792717769742,
    95.21473618224263,
    148.34089623764157,
    363.41949701309204,
    855.8876396156847,
    1321.7705802060664,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
])

mace_e3nn = np.array([
    16.93520499393344,
    25.948624999728054,
    47.96108300215565,
    86.1810031987261,
    136.2629100040067,
    371.63265320123173,
    668.3164326066617,
    1173.7415871932171,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
])

# ============================================================
# Data: MACE-OFF Large FP32
# ============================================================

mace_large_atoms = np.array([
    192,
    648,
    1536,
    3000,
    5184,
    12288,
    24000,
    41472,
])

mace_large_fasteq = np.array([
    15.362164783477784,
    17.924116895758301,
    25.107608032226562,
    38.2617998112316895,
    58.48506889343262,
    127.06586532592773,
    239.9001579284668,
    np.nan,
])

mace_large_cueq = np.array([
    29.05015363693237,
    29.796476936334033,
    33.19008159637451,
    44.674035263061526,
    65.6384994506836,
    133.17401657104492,
    245.06804275512695,
    np.nan,
])

mace_large_e3nn = np.array([
    37.41431369781494,
    110.1701000213623,
    252.7553909301758,
    488.8339309692383,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
])

# ============================================================
# Data: SevenNet-Omni FP32
# ============================================================

sevennet_atoms = np.array([
    500,
    864,
    1372,
    2048,
    2916,
    4000,
    6912,
    10976,
    16384,
])

sevennet_fasteq = np.array([
    36.245376197621226,
    49.89969999878667,
    62.82718460424803,
    93.96224840311334,
    118.42087719705887,
    150.863706599921,
    257.50058800331317,
    403.4879732003901,
    622.2975098062307,
])

sevennet_cueq = np.array([
    49.00882180663757,
    54.282122605945915,
    70.29029499972239,
    92.56120499921963,
    116.62601379794069,
    144.6919449896086,
    256.36935679940507,
    387.45688180206344,
    np.nan,
])

sevennet_flashtp = np.array([
    44.65960260713473,
    70.25024560280144,
    100.12488900101744,
    150.11146140750498,
    203.85834680637345,
    266.8345075973775,
    453.8010181975551,
    728.5447387956083,
    np.nan,
])

sevennet_e3nn = np.array([
    252.18089900445193,
    421.8464214063715,
    654.9719100003131,
    984.15811159648,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
])


# ============================================================
# AAAI single-column style
# ============================================================

FIGURE_WIDTH = 3.45
FIGURE_HEIGHT = 2.15

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",

    "font.size": 7.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.6,

    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.0,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,

    # Embed editable TrueType fonts in PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


COLORS = {
    "FastTP": "#d62728",
    "cuEq": "#7f7f7f",
    "OpenEq": "#ff7f0e",
    "FlashTP": "#ff7f0e",
    "e3nn": "#1f77b4",
}

MARKERS = {
    "FastTP": "o",
    "cuEq": "s",
    "OpenEq": "^",
    "FlashTP": "^",
    "e3nn": "D",
}


# ============================================================
# Helper functions
# ============================================================

def format_k(value, position):
    """Format atom counts using K."""
    if value == 0:
        return "0"

    if abs(value) >= 1000:
        scaled = value / 1000.0

        if float(scaled).is_integer():
            return f"{int(scaled)}K"

        return f"{scaled:g}K"

    return f"{int(value)}"


def draw_curve(ax, x, y, label):
    ax.plot(
        x,
        y,
        label=label,
        color=COLORS[label],
        marker=MARKERS[label],
        linewidth=1.5,
        markersize=4.0,
        markeredgewidth=0.45,
        markeredgecolor=COLORS[label],
        clip_on=True,
    )


def configure_axes(ax):
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


def configure_legend(ax):
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles,
        labels,
        loc="upper right",
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="0.75",
        borderpad=0.22,
        labelspacing=0.22,
        handlelength=1.45,
        handletextpad=0.35,
        columnspacing=0.70,
    )


# ============================================================
# Figure 1: MACE-OFF Small FP32
# ============================================================

fig_mace, ax_mace = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
)

draw_curve(
    ax_mace,
    mace_atoms,
    mace_fasteq,
    "FastTP",
)
draw_curve(
    ax_mace,
    mace_atoms,
    mace_cueq,
    "cuEq",
)
draw_curve(
    ax_mace,
    mace_atoms,
    mace_openeq,
    "OpenEq",
)
draw_curve(
    ax_mace,
    mace_atoms,
    mace_e3nn,
    "e3nn",
)

ax_mace.set_title(
    "MACE-OFF Small FP32",
    pad=3,
)
ax_mace.set_xlabel(
    "Number of Simulated Atoms",
    labelpad=2,
)
ax_mace.set_ylabel(
    "Inference Time (ms)",
    labelpad=2,
)

ax_mace.set_xlim(0, 200000)
ax_mace.set_ylim(0, 5000)

ax_mace.set_xticks([
    0,
    50000,
    100000,
    150000,
    200000,
])

ax_mace.set_yticks([
    0,
    1000,
    2000,
    3000,
    4000,
    5000,
])

ax_mace.xaxis.set_major_formatter(
    FuncFormatter(format_k)
)

configure_axes(ax_mace)
configure_legend(ax_mace)

fig_mace.subplots_adjust(
    left=0.18,
    right=0.985,
    top=0.88,
    bottom=0.22,
)

fig_mace.savefig(
    "mace_off_small_fp32.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
)


# ============================================================
# Figure 2: MACE-OFF Large FP32
# ============================================================

fig_mace_large, ax_mace_large = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
)

draw_curve(
    ax_mace_large,
    mace_large_atoms,
    mace_large_fasteq,
    "FastTP",
)

draw_curve(
    ax_mace_large,
    mace_large_atoms,
    mace_large_cueq,
    "cuEq",
)

draw_curve(
    ax_mace_large,
    mace_large_atoms,
    mace_large_e3nn,
    "e3nn",
)

ax_mace_large.set_title(
    "MACE-OFF Large FP32",
    pad=3,
)

ax_mace_large.set_xlabel(
    "Number of Simulated Atoms",
    labelpad=2,
)

ax_mace_large.set_ylabel(
    "Inference Time (ms)",
    labelpad=2,
)

ax_mace_large.set_xlim(0, 25000)
ax_mace_large.set_ylim(0, 550)

ax_mace_large.set_xticks([
    0,
    5000,
    10000,
    15000,
    20000,
    25000,
])

ax_mace_large.yaxis.set_major_locator(
    MultipleLocator(100)
)

ax_mace_large.xaxis.set_major_formatter(
    FuncFormatter(format_k)
)

configure_axes(ax_mace_large)
configure_legend(ax_mace_large)

fig_mace_large.subplots_adjust(
    left=0.18,
    right=0.985,
    top=0.88,
    bottom=0.22,
)

fig_mace_large.savefig(
    "mace_off_large_fp32.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
)

# ============================================================
# Figure 3: SevenNet-Omni FP32
# ============================================================

fig_sevennet, ax_sevennet = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
)

draw_curve(
    ax_sevennet,
    sevennet_atoms,
    sevennet_fasteq,
    "FastTP",
)
draw_curve(
    ax_sevennet,
    sevennet_atoms,
    sevennet_cueq,
    "cuEq",
)
draw_curve(
    ax_sevennet,
    sevennet_atoms,
    sevennet_flashtp,
    "FlashTP",
)
draw_curve(
    ax_sevennet,
    sevennet_atoms,
    sevennet_e3nn,
    "e3nn",
)

ax_sevennet.set_title(
    "SevenNet-Omni FP32",
    pad=3,
)
ax_sevennet.set_xlabel(
    "Number of Simulated Atoms",
    labelpad=2,
)
ax_sevennet.set_ylabel(
    "Inference Time (ms)",
    labelpad=2,
)

ax_sevennet.set_xlim(0, 17000)
ax_sevennet.set_ylim(0, 1100)

ax_sevennet.set_xticks([
    0,
    4000,
    8000,
    12000,
    16000,
])

ax_sevennet.yaxis.set_major_locator(
    MultipleLocator(200)
)

ax_sevennet.xaxis.set_major_formatter(
    FuncFormatter(format_k)
)

configure_axes(ax_sevennet)
configure_legend(ax_sevennet)

fig_sevennet.subplots_adjust(
    left=0.18,
    right=0.985,
    top=0.88,
    bottom=0.22,
)

fig_sevennet.savefig(
    "sevennet_omni_fp32.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
)


# Show both independent figures.
plt.show()