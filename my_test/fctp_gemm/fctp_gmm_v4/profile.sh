# !/bin/bash

ncu -o ncu_prof -f -k regex:fused_gmm_kernel --devices 0 --set full python3 test_multi_path_mm.py
