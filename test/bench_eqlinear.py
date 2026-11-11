"""Kernel-only benchmark for the CUTLASS eqlinear backends.

Usage:
  python test/bench_eqlinear.py --out results/eqlinear_25pt.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics

import torch

import fasteq  # noqa: F401  (loads _cuda.so and registers torch.ops.eqlinear)

CONFIGS = {
    "SMALL_P1": (96, [1, 3, 5, 7]),
    "MEDIUM_P1": (128, [1, 3, 5, 7]),
    "MEDIUM_P2": (128, [1, 1, 3, 3, 3, 5, 5, 5, 7, 7]),
    "LARGE_P1": (224, [1, 3, 5, 7]),
    "LARGE_P3": (224, [1, 1, 1, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 7]),
}
BATCHES = [512, 1024, 2048, 2944, 5888]
PRECISIONS = {0: ("tf32", torch.float32), 1: ("fp32", torch.float32), 2: ("fp64", torch.float64)}


def _median_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/eqlinear_25pt.csv")
    args = parser.parse_args()

    rows = []
    for config, (u, i_dims) in CONFIGS.items():
        p = len(i_dims)
        cg = [0.01 + 0.001 * i for i in range(p)]
        for batch in BATCHES:
            for precision, (name, dtype) in PRECISIONS.items():
                torch.manual_seed(0)
                x = torch.randn(batch, sum(i_dims), u, device="cuda", dtype=dtype)
                w = torch.randn(p, u, u, device="cuda", dtype=dtype)
                cg_d = torch.tensor(cg, dtype=torch.float64)
                j = torch.ops.eqlinear.out_rows(i_dims)
                out = torch.empty(batch, j, u, device="cuda", dtype=dtype)
                dx = torch.empty(batch, sum(i_dims), u, device="cuda", dtype=dtype)
                dy = torch.empty(batch, j, u, device="cuda", dtype=dtype)

                fh = torch.ops.eqlinear.forward_create(w, i_dims, cg, batch, u, p, u, precision)
                dh = torch.ops.eqlinear.dgrad_create(w, i_dims, cg, batch, u, p, u, precision)
                try:
                    fwd = _median_ms(lambda: torch.ops.eqlinear.forward_run(fh, x, out))
                    dgrad = _median_ms(lambda: torch.ops.eqlinear.dgrad_run(dh, dy, dx))
                    info = torch.ops.eqlinear.forward_plan_info(fh)
                finally:
                    torch.ops.eqlinear.forward_destroy(fh)
                    torch.ops.eqlinear.dgrad_destroy(dh)
                rows.append(
                    {
                        "config": config,
                        "batch": batch,
                        "precision": name,
                        "forward_impl": info[1],
                        "a_view": info[2],
                        "forward_ms": fwd,
                        "dgrad_ms": dgrad,
                    }
                )
                print(
                    f"{config:10s} B={batch:5d} {name:4s} fwd={fwd:9.4f} ms "
                    f"dgrad={dgrad:9.4f} ms impl={info[1]}",
                    flush=True,
                )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
