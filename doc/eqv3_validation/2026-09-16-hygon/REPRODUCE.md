# Reproduce the Hygon validation

This run tests `xiaoxiaojiujiu/FastEq` commit
`d37eacaf21b5048159a1211794efbb58941485a9` on one allocated Hygon BW DCU.
It reuses the H100 harness's reference arithmetic, complete comparisons, timing
functions, shapes, warmups, sample count and stopping rules. The new harness
configures reference and checkout paths for another machine. The production
operators are unchanged; hashes are recorded in `manifest.json`.

The runtime is Torch 2.7.1 / HIP 6.3.26045 / HCU Triton 3.1.0, with
`torch-geometric==2.6.1` and `e3nn==0.4.4`. The latter was installed in an
isolated dependency directory to match the H100 Gate reference. Existing system
packages were not replaced. The five EQv3 source hashes match the independently
verified H100 references. LayerNorm also uses the pinned official V2 reference
from the earlier LayerNorm record.

## Environment

Use the site's supported HCU container in a scheduler allocation with exactly
one DCU. Set the following paths for the local deployment:

```bash
export FASTEQ_BACKEND=cpu
export PYTHONPATH=/path/to/FastEq:/path/to/dependencies
export FASTEQ_EQUIFORMER_V3_SOURCE_DIR=/path/to/original/experimental/models/equiformer_v3
export FASTEQ_REFERENCE_NORM_DIR=/path/to/reference
export FASTEQ_EQUIFORMER_V3_LAYER_NORM="$FASTEQ_REFERENCE_NORM_DIR/layer_norm.py"
export FASTEQ_EQUIFORMER_V2_LAYER_NORM="$FASTEQ_REFERENCE_NORM_DIR/layer_norm_v2.py"
export TRITON_CACHE_DIR=/path/to/writable/triton_cache
export OMP_NUM_THREADS=8
export PYTHONWARNINGS=ignore
export PYTHONDONTWRITEBYTECODE=1
```

`FASTEQ_BACKEND=cpu` bypasses unrelated compiled package extensions. The tested
Triton operators and their Torch references still run on the allocated HIP GPU
through PyTorch's `torch.cuda` API. Device 0 below is the one visible device
inside the allocation; do not select a physical device outside the allocation.

Use a new result directory under `doc/eqv3_validation/` for each independent
repetition. Copy the harness scripts and update its manifest; do not reuse the
published `points/` or `checks/` from another run. The controller resumes
existing point JSON files, including failures. Both source files in the
reference directory must be unmodified and match their recorded hashes.

## Run

From the checkout root, with `RUN` pointing to the fresh result directory:

```bash
RUN=doc/eqv3_validation/your-new-run
python "$RUN/run.py" --smoke --gpu 0
python -m pytest -q test/test_triton_equivariant_layer_norm.py \
  test/test_triton_equivariant_layer_norm_backward.py \
  test/test_triton_equivariant_layer_norm_precision.py \
  --junitxml="$RUN/layernorm.xml" > "$RUN/layernorm.log" 2>&1
python "$RUN/extras.py"
python "$RUN/check_harness_edges.py"
python "$RUN/diagnose_softmax.py" --native --repeats 5
python "$RUN/run.py" --gpu 0 --ops norm separable alpha gate dropout softmax
python "$RUN/summarize.py"
```

Run stages sequentially on the same allocated DCU. LayerNorm contributes two
variants (`norm`, `separable`), so the five operator families use six sweep
keys. Every backend/mode/shape has a fresh child process. The controller has no
GPU context. Large accuracy comparisons temporarily save results in host RAM;
reserve sufficient host memory as well as device memory.

The sweep starts at N=256 and doubles. Actual allocation failures are `OOM`;
`LIMIT`, `INDEX_GUARD` and `FALLBACK` are explicit prechecks inherited from the
H100 harness and do not allocate the next shape. Unexpected exceptions are
`ERROR`, not OOM. Failed accuracy points retain diagnostic timings and failure
metrics without changing tolerances. FastEq-only shapes above Torch's memory
ceiling cannot provide a matched Torch accuracy result.

Forward runs under `no_grad`; Dropout remains in training mode with p=0.3.
Forward + backward includes all requested input and parameter gradients but
no optimizer. The main GraphSoftmax curve reuses CSR; N<=65536 also records
per-call graph construction with synchronized wall time. Peak memory is
allocator-allocated memory, including live inputs/results and temporaries.

`plot_by_operator.py` reads this run's `paired.csv` and the archived H100 CSV
and creates five comparison figures. It performs no GPU work. Each panel shows
speedup against that device's native eager Torch, not a hardware speed ranking.

## Inspect the published archive without a GPU

`raw_results.json.gz` contains all 328 execution records under `points` and
all 91 full comparisons under `checks`, keyed by the original JSON basename.
The `check_file` column of `paired.csv` refers to those archived keys. Per-point
GPU and wall samples, source/environment information, full comparison metrics,
and exception tracebacks are retained. Unpacked duplicates are not committed.

The following commands regenerate the summary from the archive, check all
medians, tolerances, boundary sequences, source hashes and documentation links,
and regenerate the five PNG/SVG figures (the last step requires Matplotlib):

```bash
python doc/eqv3_validation/2026-09-16-hygon/summarize.py
python doc/eqv3_validation/2026-09-16-hygon/audit_results.py
python doc/eqv3_validation/2026-09-16-hygon/plot_by_operator.py
```

The archive is preserved when the unpacked point directories are absent.
Never use published results as a resume directory for a new GPU experiment.
`checksums.sha256` records the delivered files before any local regeneration.
