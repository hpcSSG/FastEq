#export FASTEQ_CODEGEN_BACKEND=gluon
#run="python bench_luna_cueq_fasteq.py --layers 1 --atoms 16384   --backend fasteq --operators cwtp   --models mace_mp_medium  --index-modes both   --phases backward --output fasteq_triton_bwd_comparison.csv "
#export FASTEQ_STC_IMPLEMENTATION=baseline_unrolled
run="python3 mace_bench_new.py --mace-scale large"
ncu \
  --kernel-name-base function \
  --kernel-id '::regex:^(segmented_polynomial_split1_of_3_algo4_2|segmented_polynomial_split2_of_3_algo1_4|segmented_polynomial_split3_of_3_algo4_2|segmented_polynomial_bwd_algo1_1|stc_|uniform1d_|lunaeq_).*$:^[12]$' \
  --set full \
  -f \
  -o fasteq_gen_candidiate \
  ${run}
