"""Audit the current-source documentation evidence without importing GPU libraries."""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "doc/validation"


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hashes(base, values):
    for name, expected in values.items():
        assert sha256(base / name) == expected, f"Hash mismatch: {base / name}"


def junit(path, count):
    suites = ET.parse(path).getroot().findall("testsuite")
    actual = {
        key: sum(int(s.attrib[key]) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    assert actual == dict(tests=count, failures=0, errors=0, skipped=0), (path, actual)


def audit_graph_softmax_index():
    folder = RECORDS / "graph_softmax_index"
    summary = read(folder / "summary.json")
    verify_hashes(ROOT, summary["source_sha256"])
    verify_hashes(folder, summary["files_sha256"])
    providers = {"torch", "index_fused", "csr_rebuild", "packed_rebuild"}
    components = {"output", "dx", "dr"}
    expected_cases = {(n, topology) for n in (256, 4096, 32768, 262144)
                      for topology in ("random", "interleaved", "contiguous")}
    for device in ("h100", "hygon"):
        info = summary["devices"][device]
        junit(folder / info["junit_file"], 105)
        assert info["junit"] == dict(tests=105, failures=0, errors=0, skipped=0)
        rows = info["performance"]
        assert len(rows) == 12
        assert {(r["N"], r["topology"]) for r in rows} == expected_cases
        assert {r["file"] for r in rows} == {
            str(p.relative_to(folder)) for p in (folder / device).glob("perf_*.json")}
        metrics = []
        for row in rows:
            path = folder / row["file"]
            result = read(path)
            verify_hashes(ROOT / "fasteq/triton", result["sources"])
            assert result["native_sha256"] == summary["reference"]["sha256"], path
            for key in ("host", "device", "target", "torch", "triton"):
                assert result[key] == info["machine"][key], (path, key)
            assert (result["N"], result["topology"]) == (row["N"], row["topology"])
            assert result["E"] == 32 * result["N"] and result["H"] == 8, path
            assert result["settings"] == dict(
                dtype="FP32", rescale="[E,1]", softcap=3.0, eps=1e-16,
                warmup=5, repeat=20, order="rotated", topology_reuse=False,
                statistic="median synchronized wall and GPU event ms"), path
            assert set(result["checks"]) == providers - {"torch"}, path
            for checks in result["checks"].values():
                assert set(checks) == components, path
                assert all(m["bad"] == 0 and 0 <= m["max_tolerance_ratio"] <= 1
                           and math.isfinite(m["max_abs"]) and m["max_abs"] >= 0
                           for m in checks.values()), path
            metrics.append(result["checks"]["index_fused"])
            for mode, field in (("fwd", "forward_ms"), ("fwd_bwd", "forward_backward_ms")):
                assert set(result["timings"][mode]) == providers, path
                for measurement in result["timings"][mode].values():
                    assert set(measurement) == {"wall_ms", "gpu_ms"}, path
                    for series in measurement.values():
                        assert len(series["samples"]) == 20, path
                        assert all(math.isfinite(v) and v > 0 for v in series["samples"]), path
                        assert statistics.median(series["samples"]) == series["median"], path
                assert row[field] == {
                    name: result["timings"][mode][name]["wall_ms"]["median"]
                    for name in providers}, path
        assert info["precision"] == {
            component: {
                "max_abs": max(m[component]["max_abs"] for m in metrics),
                "max_tolerance_ratio": max(m[component]["max_tolerance_ratio"] for m in metrics),
                "bad": sum(m[component]["bad"] for m in metrics)}
            for component in components}, device
        speedups = [r["forward_backward_ms"]["torch"] / r["forward_backward_ms"]["index_fused"]
                    for r in rows]
        assert info["forward_backward_speedup_vs_torch"] == [min(speedups), max(speedups)]
        faster = [r for r in rows if r["forward_backward_ms"]["csr_rebuild"]
                  < r["forward_backward_ms"]["index_fused"]]
        assert sorted(info["csr_rebuild_faster_cases"], key=lambda r: r["file"]) == sorted(
            faster, key=lambda r: r["file"]), device


def main():
    manifest = read(RECORDS / "manifest.json")
    verify_hashes(ROOT, manifest["runtime_sha256"])
    verify_hashes(ROOT, manifest["test_and_harness_sha256"])
    verify_hashes(RECORDS, manifest["artifacts_sha256"])
    audit_graph_softmax_index()

    graph = RECORDS / "graph_softmax"
    summary = read(graph / "summary.json")
    verify_hashes(ROOT, summary["source_sha256"])
    verify_hashes(graph, summary["files_sha256"])
    for device in ("h100", "hygon"):
        junit(graph / device / "pytest.xml", 215)
        for path in (graph / device).glob("perf_*.json"):
            result = read(path)
            verify_hashes(ROOT / "fasteq/triton", result["sources"])
            assert result["native_sha256"] == summary["reference"]["sha256"]
            for checks in result["checks"].values():
                assert all(m["bad"] == 0 and m["max_tolerance_ratio"] <= 1
                           for m in checks.values()), path

    alpha = RECORDS / "attention_alpha"
    summary = read(alpha / "summary.json")
    expected = summary["implementation"]["sha256"]
    assert sha256(ROOT / summary["implementation"]["path"]) == expected
    verify_hashes(alpha, summary["files_sha256"])
    for device in ("h100", "hygon"):
        junit(alpha / device / "pytest.xml", 66)
        junit(alpha / device / "pytest_grid.xml", 1)
        for path in (alpha / device).glob("*.json"):
            result = read(path)
            assert result["module_sha256"] == expected, path
            if path.name == "stress.json":
                assert len(result["cases"]) == 13, path
                checks = [case["checks"] for case in result["cases"]]
            else:
                checks = [result["checks"]["triton"]]
                assert result["N"] in (4096, 32768, 131072), path
                for mode in ("fwd", "fwd_bwd"):
                    assert set(result["timings"][mode]) == {"torch", "triton"}, path
                    for measurement in result["timings"][mode].values():
                        assert len(measurement["samples_ms"]) == 20, path
                        assert statistics.median(measurement["samples_ms"]) == measurement["median_ms"], path
            for metrics in checks:
                assert all(m["bad"] == 0 and m["max_ratio"] <= 1
                           for m in metrics.values()), path

    for device in ("h100", "hygon"):
        folder = RECORDS / "layernorm" / device
        summary = read(folder / "summary.json")
        source = summary["implementation"]
        assert sha256(ROOT / source["path"]) == source["sha256"]
        verify_hashes(ROOT, summary["test_module_sha256"])
        assert sha256(folder / "pytest.xml") == summary["pytest_sha256"]
        junit(folder / "pytest.xml", 218)

    for operator in ("gate", "dropout", "norm", "separable"):
        for folder in (RECORDS / operator).iterdir():
            source = read(folder / "manifest.json")
            implementation = source["implementation"]
            assert sha256(ROOT / implementation["path"]) == implementation["sha256"]
            verify_hashes(folder, source["files_sha256"])
            raw = json.loads(gzip.decompress((folder / "raw_results.json.gz").read_bytes()))
            assert all(r["operator"] == operator
                       for group in raw.values() for r in group.values())
            summary = read(folder / "correctness.json")
            assert len(raw["checks"]) == summary["checks"]
            assert Counter(r["status"] for r in raw["checks"].values()) == summary["statuses"]
            for row in csv.DictReader((folder / "paired.csv").open()):
                assert Path(row["check_file"]).name in raw["checks"], row
            for key, result in raw["checks"].items():
                if result["status"] == "FAIL":
                    assert operator == "separable" and folder.name == "hygon", key
                    assert result["atoms"] == 262144, key
                    assert result["checks"]["affine_weight"]["failures"] == 2, key

    print("Current-source hashes and retained artifacts verified.")
    print("H100 / Hygon: GraphSoftmax CSR 215 / 215; raw index 105 / 105 plus 12 / 12 matched cases;")
    print("AttentionAlpha 67 / 67 plus 13 / 13 stress;")
    print("LayerNorm 218 / 218. Current Separable scaling failure remains recorded.")


if __name__ == "__main__":
    main()
