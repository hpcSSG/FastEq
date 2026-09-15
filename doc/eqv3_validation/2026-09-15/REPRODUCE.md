# Reproduce this validation on node70

Tested checkout: `/public-data/zhouxibo/zxb/FastEq-fork-layernorm`, commit
`3bc9a82d40ef646c3ce75f6f517e3f618fb12754`.
The archived performance records use operator files unchanged from that commit.
The current checkout includes scope cleanup; its separate correctness run is
identified by the source and test hashes in `regression.json`. Timings were not
rerun after the cleanup. The published harness now covers the retained list.

The environment is isolated in the original local artifact directory
`/public-data/zhouxibo/zxb/equiformer_v3_baseline/artifacts/eqv3_operators_20260915/venv`,
with system packages
from `/public-data/zhouxibo/miniconda3/envs/fasteq_v0.3.0` plus
`torch-geometric==2.6.1`. Runtime versions are in each result's `environment`.
The harness permits the builtin `slice` type when the installed e3nn 0.4.4
loads its constant data with PyTorch's weights-only loader.

Use an idle GPU. This run used physical GPU 5, UUID
`GPU-5701253b-7aa2-4bb7-3b69-15e0e7e72186`, on a shared H100 host; clocks were
not locked and the machine was not exclusively reserved.

```bash
source /public-data/zhouxibo/zxb/equiformer_v3_baseline/artifacts/eqv3_operators_20260915/venv/bin/activate
export CUDA_VISIBLE_DEVICES=5
export FASTEQ_BACKEND=cpu
export PYTHONPATH=/public-data/zhouxibo/zxb/FastEq-fork-layernorm
export OMP_NUM_THREADS=8
export PYTHONWARNINGS=ignore
export TRITON_CACHE_DIR=/public-data/zhouxibo/zxb/fasteq_unified_layernorm_20260915/cache
export FASTEQ_EQUIFORMER_V3_LAYER_NORM=/public-data/zhouxibo/zxb/fasteq_unified_layernorm_20260915/reference/layer_norm.py
export FASTEQ_EQUIFORMER_V2_LAYER_NORM=/public-data/zhouxibo/zxb/fasteq_unified_layernorm_20260915/reference/layer_norm_v2.py
```

`FASTEQ_BACKEND=cpu` bypasses unrelated package extension loading; every timed
operator still executes on the selected CUDA device.

From the tested checkout, use the activated environment's Python to run:

```bash
python -m pytest -q test/test_triton_equivariant_layer_norm.py \
  test/test_triton_equivariant_layer_norm_backward.py \
  test/test_triton_equivariant_layer_norm_precision.py \
  --junitxml=/path/to/new_results/layernorm.xml
```

For a new scaling run, copy `cases.py`, `run.py`, `extras.py`,
`diagnose_softmax.py`, `check_harness_edges.py`, `summarize.py`, `plot.py`, and `manifest.json` into a new
result directory. Record the new source hashes and environment in its manifest,
and write the new pytest log/XML there before aggregation. Run these scripts with the same Python interpreter:

```bash
python /path/to/new_results/run.py --smoke --gpu 5
python /path/to/new_results/extras.py
python /path/to/new_results/run.py --gpu 5 \
  --ops norm separable alpha gate dropout softmax
python /path/to/new_results/diagnose_softmax.py
python /path/to/new_results/diagnose_softmax.py --native --repeats 5
python /path/to/new_results/check_harness_edges.py
python /path/to/new_results/summarize.py
python /path/to/new_results/plot.py
```

The controller resumes existing point JSON files, including failures. Use a
new directory for an independent repetition; do not overwrite the original
measurements. Each backend/mode/shape runs in a fresh child process. Accuracy
comparison transfers one implementation's results to host RAM before running
the other, so large checks require substantial host RAM. The controller keeps
doubling N until every path hits OOM, an explicit limit, the checked signed
address range, or a public Torch fallback.

`cases.py` names the exact native reference locations. For another host, update
those locations and the checkout constants in the harness scripts; do not claim
the original source hash until verifying the copied files. For the same host
and source revision, no reference changes are needed.

The original small dropout checks inferred the shared mask from nonzero
outputs. Before the dropout scaling sweep, mask replay was changed to an
explicit compact mask generated using the same seed and RNG offsets, which
also handles exactly-zero inputs. The initial helper is retained as
`cases_initial_mask_replay.py`; the production dropout implementation and its
timing path were unchanged.
