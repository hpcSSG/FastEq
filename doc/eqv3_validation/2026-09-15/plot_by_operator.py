"""Render separate operator plots from archived timings; no GPU work."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
GROUPS = {
    "layernorm": (("norm", "LayerNorm"), ("separable", "SeparableLayerNorm")),
    "graph_softmax": (("softmax", "GraphSoftmax (cached CSR)"),),
    "attention_alpha": (("alpha", "AttentionAlpha"),),
    "equivariant_gate": (("gate", "e3nn Gate"),),
    "equivariant_dropout": (("dropout", "EquivariantDropout"),),
}
NOTES = {
    "layernorm": "Public calls include native Torch reductions in backward.",
    "graph_softmax": "Cached topology. A separate broadcast-rescale gradient check fails intermittently.",
    "attention_alpha": "Red x: parameter-gradient accuracy failed. Backward recomputes Torch expressions.",
    "equivariant_gate": "Reference: e3nn.nn.Gate. EQv3 GateActivation is a separate, failing API check.",
    "equivariant_dropout": "Active dropout: p=0.3, training=True, including no_grad forward.",
}


def main():
    with (ROOT / "paired.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = ROOT / "operator_plots"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    for name, operators in GROUPS.items():
        fig, axes = plt.subplots(1, len(operators), figsize=(6.2 * len(operators), 4.3),
                                 squeeze=False)
        for ax, (operator, title) in zip(axes.flat, operators):
            for mode, label, color in (
                ("fwd", "Forward", "#3066a5"),
                ("fwd_bwd", "Forward + backward", "#cc7129"),
            ):
                selected = sorted((row for row in rows if row["operator"] == operator
                                   and row["mode"] == mode), key=lambda row: int(row["atoms"]))
                ax.plot([int(row["atoms"]) for row in selected],
                        [float(row["gpu_speedup"]) for row in selected],
                        color=color, label=label, marker=".", linewidth=1.4)
                failed = [row for row in selected if row["correctness"] != "PASS"]
                if failed:
                    ax.scatter([int(row["atoms"]) for row in failed],
                               [float(row["gpu_speedup"]) for row in failed],
                               color="red", marker="x", s=45, zorder=5)
            ax.axhline(1, color="black", linestyle="--", linewidth=.7)
            ax.set_xscale("log", base=2)
            ax.set_title(title)
            ax.set_xlabel("Atoms N")
            ax.set_ylabel("Torch time / FastEq time")
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle("H100, FP32 — archived standalone measurements", fontsize=12)
        fig.text(.03, .025, NOTES[name], fontsize=8)
        fig.tight_layout(rect=(0, .07, 1, .95))
        fig.savefig(output / f"{name}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
