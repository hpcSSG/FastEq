# Reproduce current-source validation

Use the recorded CUDA or HCU/PyTorch environment on one available GPU/DCU.
The published tables describe their recorded runs; executing these commands
creates a new independent run and must not overwrite the retained evidence.
Reserve sufficient host RAM for the scaling harness, which saves one
implementation's outputs to host memory for full comparisons.

Every operator code change requires relevant accuracy and performance checks
on both Hygon and H100 using identical source hashes. If either platform is
unavailable, its validation remains pending. Unchanged sources may retain
their existing matched evidence; a documentation-only update does not imply
a new GPU run.

## References and environment

Run from the repository root. Supply unmodified reference files matching the
hashes in [the validation index](../eqv3_validation.md) and operator manifests.
The EQv3 source is commit `a7300c58df683dc99cb48027d5bfd4c887486c48` from
[malixian/equiformer_v3](https://github.com/malixian/equiformer_v3).
The EQv2 source is commit `d5ad4be729b56f74012ebb7f097f77c5b00a1004` from
[atomicarchitects/equiformer_v2](https://github.com/atomicarchitects/equiformer_v2).
Dependencies include pytest, torch-geometric 2.6.1, e3nn 0.4.4 for Gate, and
those required by the original source files.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export FASTEQ_BACKEND=cpu
export FASTEQ_EQUIFORMER_V3_SOURCE_DIR=/path/to/equiformer_v3/experimental/models/equiformer_v3
export FASTEQ_EQUIFORMER_V3_LAYER_NORM="$FASTEQ_EQUIFORMER_V3_SOURCE_DIR/layer_norm.py"
export FASTEQ_EQUIFORMER_V2_LAYER_NORM=/path/to/equiformer_v2/nets/equiformer_v2/layer_norm.py
export FASTEQ_REFERENCE_NORM_DIR="$FASTEQ_EQUIFORMER_V3_SOURCE_DIR"
export FASTEQ_ALPHA_LARGE_TESTS=1
export OMP_NUM_THREADS=8
export TRITON_CACHE_DIR=/path/to/writable/triton_cache
run_dir=$(mktemp -d /tmp/fasteq-validation.XXXXXX)
```

`FASTEQ_BACKEND=cpu` bypasses unrelated compiled FastEq package extensions;
these directly imported Triton operators still execute on the selected GPU/DCU.
Select a device according to the scheduler's allocation. Source variables and
the Alpha large-test flag are required for the published zero-skip suite counts.

## Regression tests

```bash
python -m pytest -q test/test_triton_graph_softmax_precision.py \
  test/test_triton_graph_edge_layout.py --junitxml="$run_dir/graph_softmax.xml"
python -m pytest -q test/test_triton_graph_softmax_index.py \
  --junitxml="$run_dir/graph_softmax_index.xml"
python -m pytest -q test/test_triton_attention_alpha.py \
  --junitxml="$run_dir/attention_alpha.xml"
python -m pytest -q test/test_triton_equivariant_layer_norm.py \
  test/test_triton_equivariant_layer_norm_backward.py \
  test/test_triton_equivariant_layer_norm_precision.py \
  --junitxml="$run_dir/layernorm.xml"
python test/stress_triton_attention_alpha.py --out "$run_dir/alpha_stress.json"
```

Expected retained suite counts are GraphSoftmax CSR 215, GraphSoftmax raw index
105, AttentionAlpha 67, and
LayerNorm 218 per device, plus 13 AttentionAlpha stress cases. Individual-suite
passes do not override the separately documented Separable scaling failure.
The retained Alpha evidence has 66 existing tests and one separately executed
padded-grid test per device; the command above runs all 67 together.
Saved historical failure tensors in `test/data/graph_softmax` are regression
inputs for the current implementation, not old implementation test results.

## Representative GraphSoftmax and AttentionAlpha benchmarks

```bash
for n in 4096 262144 1048576; do
  python test/benchmark_triton_graph_softmax.py --n "$n" \
    --out "$run_dir/graph_softmax_${n}.json"
done
python test/benchmark_triton_graph_softmax.py --n 4096 --topology random \
  --out "$run_dir/graph_softmax_random.json"
for n in 4096 32768 131072; do
  python test/benchmark_triton_attention_alpha_matched.py --n "$n" \
    --out "$run_dir/attention_alpha_${n}.json"
done
```

The matched Alpha benchmark stages reference tensors on the CPU and releases
GPU comparison temporaries before timing, so all three sizes were measured
successfully on both devices. The portable entry point preserves the measured
input generation, reference arithmetic, precision check and timing procedure;
it omits debug assembly dumps. The default runs Torch and the current source.
To reproduce the retained run's three-provider rotation, add the optional
control from the recorded source commit:

```bash
git show 7546658c05ff95a80985be28d45072caab1d94c2:fasteq/triton/fused_attention_alpha.py \
  > "$run_dir/alpha_control.py"
python test/benchmark_triton_attention_alpha_matched.py --n 4096 \
  --control-module "$run_dir/alpha_control.py" --out "$run_dir/alpha_rotated.json"
```

Only current-source checks and Torch/current timing arrays were selected for
publication from the measured three-provider JSON; their numeric values are
unchanged. The original comparison-file checksums are listed in the Alpha
summary. The older general Alpha harness remains the helper used by the
stress script, not the source of the current performance table. GraphSoftmax's
benchmark is byte-identical to its measured script.

## GraphSoftmax without topology reuse

```bash
for topology in random interleaved contiguous; do
  for n in 256 4096 32768 262144; do
    python test/benchmark_triton_graph_softmax_index.py --n "$n" \
      --topology "$topology" --repeat 20 \
      --out "$run_dir/graph_softmax_index_${topology}_${n}.json"
  done
done
```

This byte-identical measured script compares native EQv3 Torch, raw-index
fusion, original-order CSR rebuilt on every call, and sorted CSR rebuilt with
feature packing/output restoration on every call. Inputs are generated before
timing; topology remains fixed within each case but neither CSR path reuses a
mapping. The separate regression test checks in-place topology mutations.
Each of the 12 cases checks full output, logits and rescale gradients before
timing. Five warmups and 20 rotated samples record both synchronized wall and
GPU-event milliseconds. The raw-index documentation reports wall medians;
the earlier reusable-CSR tables report GPU-event medians.

## Gate, Dropout and LayerNorm scaling

```bash
export FASTEQ_VALIDATION_OUTPUT_DIR="$run_dir/scaling"
python test/eqv3_validation/run.py --gpu 0 --smoke --ops gate dropout norm separable
python test/eqv3_validation/run.py --gpu 0 --ops gate dropout norm separable
```

The harness preserves the measured reference arithmetic and sampling method;
its repository/output paths are portable, and the CLI is restricted to these
four sweep keys. It starts at N=256 and doubles, in separate processes per
implementation/mode/shape, until allocation failure, an indexing precheck or
runtime failure. Accuracy compares every output and requested gradient.
Only reuse an output directory to resume that same run: existing point files,
including failures, are not rerun. Never use `doc/validation` as the output.

For a precise accuracy reproduction at the known Hygon boundary:

```bash
CUDA_VISIBLE_DEVICES=0 python test/eqv3_validation/run.py --worker \
  --task check --op separable --n 262144 --mode fwd_bwd --backend fused \
  --output "$run_dir/separable_262144.json"
```

## Inspect retained evidence without a GPU

[manifest.json](manifest.json) lists current source/test hashes, record groups,
and checksums of every retained artifact. Each compressed sweep archive has
`points` and `checks` dictionaries keyed by the original JSON basename. CSV
`check_file` values refer to entries in that archive. Arrays of per-iteration
GPU/wall times, full-tensor error metrics, peak allocated memory and stopping
errors remain available. Other operators and mismatched-source records have
been filtered out; the retained entries themselves are unchanged.

```bash
python test/check_eqv3_validation_records.py
```

This checks source/artifact hashes, suite counts, current GraphSoftmax and
AttentionAlpha pass metrics, and retained sweep status counts. It does not run
GPU kernels or claim a fresh GPU validation. The unresolved current-source
Separable failure is expected evidence and is checked rather than suppressed.
