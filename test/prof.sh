#!/usr/bin/env bash
set -euo pipefail

OUT_PREFIX="small-fp32"
shift || true

# fasteq
KREGEX='tp_channel_wise_sparse_groupk_kernel|tp_bwd_fused_kernel_sharedc|tp_channel_wise_sparse_groupk_fused_scatter_sender_major_kernel|tp17_bwd_fused_sender_major_densec_kernel|fused_fctp_kernel_fwd_multipath_tiledW|fused_fctp_kernel_bwd_grad_a_multipath_tiledU|stc_fwd_kernel_notiled|stc_bwd_kernel_v1|stc_bwd_kernel_tiled|fused_gmm_kernel_v2'

# cueq

ncu \
  --target-processes all \
  --replay-mode kernel \
  -k "regex:${KREGEX}" \
  --launch-skip 10 \
  --metrics gpu__time_duration.sum \
  --page raw --csv --print-units base --print-fp \
  --log-file ${OUT_PREFIX}__ncu_time_summary.csv \
  python3 mace_batch.py small float32

echo "Wrote:"
echo "  ${OUT_PREFIX}.ncu-rep"
echo "  ${OUT_PREFIX}_ncu_time_summary.csv"
