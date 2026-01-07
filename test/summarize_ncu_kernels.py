import argparse
from pathlib import Path
from io import StringIO
import pandas as pd

KERNEL_GROUPS = {
    "cwtp_fwd": ["tp_channel_wise_sparse_groupk_kernel"],
    "cwtp_bwd": ["tp_bwd_fused_kernel_sharedc"],

    "mptp_fwd": ["tp_channel_wise_sparse_groupk_fused_scatter_sender_major_kernel"],
    "mptp_bwd": ["tp17_bwd_fused_sender_major_densec_kernel"],

    "fctp_fwd": ["fused_fctp_kernel_fwd_multipath_tiledW"],
    "fctp_bwd": ["fused_fctp_kernel_bwd_grad_a_multipath_tiledU"],

    "stc_fwd": ["stc_fwd_kernel_notiled"],
    "stc_bwd": ["stc_bwd_kernel_v1", "stc_bwd_kernel_tiled"],

    "equi_linear": ["fused_gmm_kernel_v2"],
}

def read_ncu_raw_csv(path: str | Path) -> pd.DataFrame:
    """
    只从真正 CSV 表头开始读取，原样保留（允许引号字段内换行），不做逐行过滤。
    这能避免你遇到的 Kernel Name 读成 0.0（列错位）问题。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    idx = text.find('"ID","Process ID"')
    if idx < 0:
        raise RuntimeError(f"Cannot find CSV header in {path}")
    return pd.read_csv(StringIO(text[idx:]), engine="python")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="ncu raw csv files (run*_raw.csv)")
    ap.add_argument("-o", "--out_prefix", default="ncu_time")
    ap.add_argument("--topk", type=int, default=2, help="per group keep top-k (grid,block) buckets")
    args = ap.parse_args()

    dfs = []
    for p in args.csv:
        df = read_ncu_raw_csv(p)
        df["__srcfile"] = Path(p).name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print("Loaded rows:", len(df))

    # === 只使用 Kernel Name ===
    if "Kernel Name" not in df.columns:
        raise RuntimeError("Column 'Kernel Name' not found in CSV.")

    # 你贴的表头里也有这些列
    need_cols = ["Kernel Name", "launch__grid_size", "launch__block_size", "gpu__time_duration.sum", "__srcfile"]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}")

    out = df[need_cols].copy()
    out.rename(columns={
        "Kernel Name": "kernel_name",
        "launch__grid_size": "grid_size",
        "launch__block_size": "block_size",
        "gpu__time_duration.sum": "duration_raw",
    }, inplace=True)

    # kernel_name 清洗
    out["kernel_name"] = out["kernel_name"].astype("string").str.strip()
    out = out.dropna(subset=["kernel_name"])
    out = out[out["kernel_name"] != ""].copy()

    # duration：处理千分位逗号，如 '1,186,400.00'
    s = out["duration_raw"].astype(str).str.strip().str.replace(",", "", regex=False)
    out["duration_ns"] = pd.to_numeric(s, errors="coerce")
    out = out.dropna(subset=["duration_ns"])

    # 单位：你用 --print-units base 时是 ns
    out["duration_us"] = out["duration_ns"] / 1e3
    out["duration_ms"] = out["duration_ns"] / 1e6

    # 分组：只用 Kernel Name contains
    out["group"] = pd.NA
    for g, subs in KERNEL_GROUPS.items():
        mask = False
        for sub in subs:
            mask = mask | out["kernel_name"].str.contains(sub, regex=False, na=False)
        out.loc[mask, "group"] = g

    out = out.dropna(subset=["group"]).copy()
    print("Matched rows:", len(out))

    # (同名 kernel 可能两种规模) -> 按 grid/block 分桶
    out["shape_bucket"] = "grid=" + out["grid_size"].astype(str) + " block=" + out["block_size"].astype(str)

    g = out.groupby(["group", "shape_bucket"], dropna=False)
    stats = g["duration_ms"].agg(
        count="count",
        mean_ms="mean",
        p50_ms="median",
        p90_ms=lambda x: x.quantile(0.90),
        min_ms="min",
        max_ms="max",
    ).reset_index()

    # 每个 group 取 topk 个桶：按 count 多->少，再按 mean_ms 大->小
    stats = stats.sort_values(["group", "count", "mean_ms"], ascending=[True, False, False])
    stats["rank_in_group"] = stats.groupby("group").cumcount() + 1
    #topk = stats[stats["rank_in_group"] <= args.topk].copy()

    all_path = f"{args.out_prefix}_all.csv"
    topk_path = f"{args.out_prefix}_top{args.topk}.csv"
    stats.to_csv(all_path, index=False)
    #.to_csv(topk_path, index=False)

    print("\n=== Top buckets per group ===")
    print(topk.to_string(index=False))
    print(f"\nWrote: {all_path}")
    #print(f"Wrote: {topk_path}")

if __name__ == "__main__":
    main()
