KNAME='fused_mp_warp_streaming_kernel_groupflush_scalar'   # 换成你的核名

ncu -k "${KNAME}" \
    --launch-skip 0 --launch-count 3 \
    --metrics \
lts__t_sectors_op_red.sum,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
smsp__warps_stalled_pipe_lsu_per_warp_active.pct,\
smsp__warps_stalled_selected_per_warp_active.pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
    python3 test_fused_message_passing.py

