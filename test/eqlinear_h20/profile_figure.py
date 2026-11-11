"""Standalone SVG/PNG from measured kernel durations and DRAM/L2 byte counters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import Evidence


class ProfileFigure:
    @classmethod
    def run(cls, args):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy

        traffic = json.loads(args.traffic.read_text())
        collection_path = Path(traffic["collection"]["path"])
        if Evidence.sha256(collection_path) != traffic["collection"]["sha256"]:
            raise ValueError("Traffic derivation points to changed profile evidence")
        collection = json.loads(collection_path.read_text())
        counters_path = Path(collection["counters"]["path"])
        if Evidence.sha256(counters_path) != collection["counters"]["sha256"]:
            raise ValueError("Counter data changed")
        counters = json.loads(counters_path.read_text())
        args.output.mkdir(parents=True, exist_ok=False)
        durations = [kernel["metrics"].get("gpu__time_duration.sum", {}).get("si") for kernel in counters["kernels"]]
        if any(value is None or value["unit"] != "s" for value in durations):
            raise ValueError("Missing measured kernel duration")
        names = [kernel["kernel"] for kernel in counters["kernels"]]
        latency_us = [value["value"] * 1e6 for value in durations]
        byte_keys = ("dram_read_bytes", "dram_write_bytes", "l2_bytes")
        if any(traffic[key] is None for key in byte_keys):
            raise ValueError("Missing memory-layer counters; do not replace with zero")
        byte_values = [traffic[key] for key in byte_keys]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        bars = axes[0].bar(range(len(latency_us)), latency_us, color="#4477aa")
        axes[0].set_xticks(range(len(names)), ["Kernel " + str(index) for index in range(len(names))])
        axes[0].set_ylabel("Instrumented kernel duration (us)")
        axes[0].set_title("Observed kernel pipeline")
        axes[0].bar_label(bars, fmt="%.3g", padding=3)
        bars = axes[1].bar(("DRAM read", "DRAM write", "L2 total"), [value / 2**20 for value in byte_values],
                           color=("#228833", "#66ccee", "#cc6677"))
        axes[1].set_ylabel("Measured traffic (MiB)")
        axes[1].set_title("Warm invocation counters")
        axes[1].bar_label(bars, labels=["0 counted" if value == 0 else f"{value/2**20:.3g}" for value in byte_values], padding=3)
        for axis in axes:
            axis.set_ylim(top=max(axis.get_ylim()[1] * 1.18, .1))
            axis.spines[["top", "right"]].set_visible(False)
        point = traffic["point"]
        fig.suptitle(f"{point['config']} / B={point['batch']} / {point['precision']} / {traffic['call']}", fontsize=10)
        fig.text(.5, .035, "NCU replay evidence; durations are not completed wall latency. Zero DRAM writes do not mean no output stores.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .08, 1, .93))
        files = {}
        for extension in ("svg", "png"):
            path = args.output / ("profile." + extension)
            fig.savefig(path, dpi=180, metadata={"Description": "Derived from real NCU counters; see data.json for provenance"})
            files[extension] = {"path": str(path), "sha256": Evidence.sha256(path)}
        plt.close(fig)
        Evidence.write(args.output / "data.json", {"kernel_names": names, "profiled_kernel_duration_us": latency_us,
            "byte_keys": byte_keys, "measured_bytes": byte_values, "traffic": {"path": str(args.traffic), "sha256": Evidence.sha256(args.traffic)},
            "source": traffic["collection"], "files": files, "tool_sha256": Evidence.sha256(Path(__file__)),
            "matplotlib_version": matplotlib.__version__, "numpy_version": numpy.__version__, "numpy_origin": numpy.__file__,
            "scope": "Single real instrumented invocation; incomplete mechanism coverage, no version ranking or physical roof claim"})
        return 0

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--traffic", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        return cls.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(ProfileFigure.main())
