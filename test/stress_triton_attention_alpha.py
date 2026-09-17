from pathlib import Path
exec(compile(Path(__file__).with_name('benchmark_triton_attention_alpha.py').read_text().split('\nif args.suite:')[0], 'validate.py', 'exec'))
for seed in [0,1,123]:
 run(4096*32,8,32,seed=seed)
for c,h in [(13,3),(64,8),(128,1)]:
 for act in ['silu','smooth_leaky_relu']:
  run(4096*32,h,c,act=act,seed=7)
for ln in [False,True]:
 run(16384*32,3,13,ln=ln,act='silu',seed=11)
run(4096*32,8,32,p=.25,seed=17)
run(4096*32,8,32,act='silu',seed=19)
save();bad=sum(c['bad'] for r in results['cases'] for c in r['checks'].values());print('TOTAL_BAD',bad,flush=True)
raise SystemExit(bool(bad))
