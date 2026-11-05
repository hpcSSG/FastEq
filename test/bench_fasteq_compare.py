"""Benchmark fasteq's eqlinear before/after the CUTLASS integration.

Run once from the old tree and once from the new tree:

  cd /app/work/fasteq_before && PYTHONPATH=. python test/bench_fasteq_compare.py --mode before --out before.csv
  cd /app/work/fasteq-pub    && python test/bench_fasteq_compare.py --mode after  --out after.csv

The torch reference is the direct per-path torch implementation.  FP32/TF32 use
torch.backends.cuda.matmul.allow_tf32=False; FP64 uses float64.
"""

from __future__ import annotations

import argparse
import csv
import statistics

import torch

import fasteq  # noqa: F401  (loads _cuda.so)
from fasteq.ops.so3_linear import fast_equi_linear  # both old and new trees expose this

CONFIGS = {
    "SMALL_P1": (96, [1, 3, 5, 7]),
    "MEDIUM_P1": (128, [1, 3, 5, 7]),
    "MEDIUM_P2": (128, [1, 1, 3, 3, 3, 5, 5, 5, 7, 7]),
    "LARGE_P1": (224, [1, 3, 5, 7]),
    "LARGE_P3": (224, [1, 1, 1, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 7]),
}
BATCHES = [512, 1024, 2048, 2944, 5888]
PRECISIONS = ("tf32", "fp32", "fp64")


class _Path:
    def __init__(self, indices, coefficient):
        self.indices = indices
        self.coefficients = coefficient


class _Descriptor:
    """Minimal stand-in for the cuEquivariance uv,iu,iv descriptor."""

    def __init__(self, i_dims, u, v, cg):
        p_count = len(i_dims)
        runs = []
        start = 0
        while start < p_count:
            i = i_dims[start]
            end = start + 1
            while end < p_count and i_dims[end] == i:
                end += 1
            runs.append((i, start, end - start))
            start = end

        self.paths = []
        for p in range(p_count):
            group = next(g for g, (_, p0, count) in enumerate(runs) if p0 <= p < p0 + count)
            self.paths.append(_Path((p, p, group), cg[p]))
        self.operands = [
            tuple((u, v) for _ in range(p_count)),
            tuple((i, u) for i in i_dims),
            tuple((i, v) for i, _, _ in runs),
        ]
        self._u = u
        self._v = v
        self._i = i_dims

    def get_dims(self, name):
        return [self._u] if name == "u" else [self._v]

    def get_dimensions_dict(self):
        return {"i": self._i}


def _layout(i_dims):
    runs = []
    start = 0
    while start < len(i_dims):
        i = i_dims[start]
        end = start + 1
        while end < len(i_dims) and i_dims[end] == i:
            end += 1
        runs.append((i, start, end - start))
        start = end
    prefix = [0]
    for i in i_dims[:-1]:
        prefix.append(prefix[-1] + i)
    return runs, prefix, sum(i for i, _, _ in runs)


def _reference(x3, w3, i_dims, cg, dtype):
    x = x3.to(dtype)
    w = w3.to(dtype)
    batch, _, u = x.shape
    v = w.shape[2]
    runs, prefix, j_rows = _layout(i_dims)
    out = torch.zeros(batch, j_rows, v, dtype=dtype, device=x.device)
    out_row = 0
    for i, p0, count in runs:
        for t in range(i):
            acc = None
            for p in range(p0, p0 + count):
                term = x[:, prefix[p] + t, :] @ w[p] * float(cg[p])
                acc = term if acc is None else acc + term
            out[:, out_row + t, :] = acc
        out_row += i
    return out


def _rel_fro(a, b):
    return (a - b).norm().item() / b.norm().clamp_min(1e-30).item()


def _median_ms(fn, warmup, iters):
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


def _median_pair_ms(op_fn, torch_fn, op_warmup, iters):
    """Interleave torch and op so both see the same clocks/thermal state."""
    for _ in range(op_warmup):
        torch_fn()
        op_fn()
    torch.cuda.synchronize()
    torch_samples = []
    op_samples = []
    for _ in range(iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        o0 = torch.cuda.Event(enable_timing=True)
        o1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        torch_fn()
        t1.record()
        o0.record()
        op_fn()
        o1.record()
        torch.cuda.synchronize()
        torch_samples.append(t0.elapsed_time(t1))
        op_samples.append(o0.elapsed_time(o1))
    return statistics.median(torch_samples), statistics.median(op_samples)


def _call(mode, desc, w, x, precision):
    if mode == "before":
        if precision != "fp32":
            raise RuntimeError("the pre-integration fast_equi_linear exposes only FP32 (IEEE Triton)")
        return fast_equi_linear(desc, w, x)
    if precision == "fp32":
        return fast_equi_linear(desc, w, x, compute_precision="fp32")
    return fast_equi_linear(desc, w, x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("before", "after", "torch"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    case_index = 0
    for config, (u, i_dims) in CONFIGS.items():
        p_count = len(i_dims)
        cg = [0.01 + 0.001 * p for p in range(p_count)]
        for batch in BATCHES:
            for precision in PRECISIONS:
                case_index += 1
                torch.manual_seed(1000 + case_index)
                dtype = torch.float64 if precision == "fp64" else torch.float32
                x = torch.randn(batch, sum(i_dims) * u, device="cuda", dtype=dtype)
                w = torch.randn(p_count * u * u, device="cuda", dtype=dtype)
                desc = _Descriptor(i_dims, u, u, cg)
                x3 = x.reshape(batch, sum(i_dims), u)
                w3 = w.reshape(p_count, u, u)

                ref64 = _reference(x3, w3, i_dims, cg, torch.float64)
                ref_work_dtype = torch.float64 if precision == "fp64" else torch.float32
                ref_work = _reference(x3, w3, i_dims, cg, ref_work_dtype)
                torch_fn = lambda: _reference(x3, w3, i_dims, cg, ref_work_dtype)

                supported = False
                error = "torch reference only" if args.mode == "torch" else ""
                op_ms = None
                rel = None
                out = None
                if args.mode != "torch":
                    try:
                        out = _call(args.mode, desc, w, x, precision)
                        supported = True
                    except Exception as exc:  # noqa: BLE001
                        error = str(exc).splitlines()[0][:160]

                if supported:
                    torch_ms, op_ms = _median_pair_ms(
                        lambda: _call(args.mode, desc, w, x, precision),
                        torch_fn,
                        op_warmup=10 if args.mode == "before" else 5,
                        iters=20,
                    )
                    rel = _rel_fro(out.reshape(batch, -1), ref64.reshape(batch, -1))
                else:
                    torch_ms = _median_ms(torch_fn, warmup=3, iters=20 if args.mode == "torch" else 10)

                speedup = None if op_ms is None else torch_ms / op_ms
                rows.append(
                    {
                        "mode": args.mode,
                        "config": config,
                        "batch": batch,
                        "precision": precision,
                        "supported": supported,
                        "op_ms": op_ms,
                        "torch_ms": torch_ms,
                        "speedup": speedup,
                        "rel_fro": rel,
                        "torch_rel_fro": _rel_fro(ref_work, ref64),
                        "error": error,
                    }
                )
                print(
                    f"{args.mode:6s} {config:10s} B={batch:5d} {precision:4s} "
                    f"supported={supported} op={op_ms} torch={torch_ms:.3f} "
                    f"speedup={speedup} rel={rel}",
                    flush=True,
                )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
