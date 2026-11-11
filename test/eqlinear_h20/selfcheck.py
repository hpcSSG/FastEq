"""CPU counterexamples for evidence integrity, reference semantics and summaries."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cases import Case, Point, References, points
from common import Evidence
from protocol import Accuracy, CoverageAuditor, Journal, Statistics, TraceSummary
from session import Session


class IntegrityTests(unittest.TestCase):
    root = None

    def directory(self):
        return Path(tempfile.mkdtemp(dir=self.root))

    def test_fixture_roundtrip_corruption_and_descriptor_identity(self):
        root = self.directory()
        point = Point("SMALL_P1", 2, "fp32", 0, 0)
        first = Case(point, "cpu", root)
        second = Case(point, "cpu", root)
        self.assertEqual(first.input_hash, second.input_hash)
        self.assertIs(first.descriptor, second.descriptor)
        metadata_path = next(root.glob("*.json"))
        metadata = json.loads(metadata_path.read_text())
        metadata["tensor_hashes"]["x"] = "corrupt"
        Evidence.write(metadata_path, metadata)
        with self.assertRaisesRegex(RuntimeError, "tensor hash mismatch"):
            Case(point, "cpu", root)

    def test_partial_fixture_is_not_silently_replaced(self):
        root = self.directory()
        (root / "SMALL_P1_2_fp32.pt").write_bytes(b"interrupted")
        with self.assertRaisesRegex(RuntimeError, "Partial fixture"):
            Case(Point("SMALL_P1", 2, "fp32", 0, 0), "cpu", root)

    def test_reference_run_length_and_gradients(self):
        import torch

        # Equal non-adjacent dimensions remain separate output groups: J=1+3+1=5.
        # Binary-exact coefficients isolate indexing/autograd from reduction rounding.
        dims, cg = [1, 1, 3, 1], [.125, .25, .5, .0625]
        x = torch.arange(12., dtype=torch.float64).reshape(1, 6, 2).requires_grad_()
        w = torch.arange(24., dtype=torch.float64).reshape(4, 2, 3).requires_grad_()
        result = References.forward(x, w, dims, cg, torch.float64, False)
        expected = torch.cat([(x[:, 0:1] @ w[0]) * cg[0] + (x[:, 1:2] @ w[1]) * cg[1],
                              (x[:, 2:5] @ w[2]) * cg[2], (x[:, 5:6] @ w[3]) * cg[3]], dim=1)
        self.assertEqual(result.shape, (1, 5, 3))
        self.assertTrue(torch.equal(result, expected))
        gradients = torch.autograd.grad(result.sum(), (x, w), retain_graph=True)
        expected_gradients = torch.autograd.grad(expected.sum(), (x, w))
        for got, gold in zip(gradients, expected_gradients):
            self.assertTrue(torch.equal(got, gold))

    def test_zero_reference_strict_gate_and_nonfinite(self):
        import torch

        zero = torch.zeros(3)
        self.assertTrue(Accuracy.evaluate(zero, zero, 0., "fp32", True, True)["passed"])
        small = torch.ones(3) * 1e-30
        self.assertFalse(Accuracy.evaluate(small, zero, 0., "fp32", True, True)["passed"])
        self.assertFalse(Accuracy.evaluate(torch.full((3,), float("nan")), zero, 0., "fp32", True, True)["passed"])
        # A non-finite comparator must never turn infinite relative error into PASS.
        result = Accuracy.evaluate(torch.ones(3), zero, float("inf"), "fp32", True, True)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["threshold"])
        json.dumps(result, allow_nan=False)
        self.assertFalse(Accuracy.evaluate(zero, zero, float("nan"), "fp32", True, True)["passed"])
        self.assertFalse(Accuracy.evaluate(torch.full((3,), 1e-300, dtype=torch.float64), zero, 0., "fp64", True, True)["passed"])
        result = Accuracy.evaluate(zero, torch.full((3,), float("nan")), 0., "fp64", True, True)
        self.assertFalse(result["passed"])
        json.dumps(result, allow_nan=False)

    def test_journal_resume_mixed_identity_and_missing_raw(self):
        root = self.directory()
        path, identity = root / "raw.csv", {"GPU_UUID": "synthetic", "protocol_hash": "a", "binary_hash": "a"}
        Journal(path, identity).close()
        Journal(path, identity, resume=True).close()
        for field in identity:
            changed = dict(identity, **{field: "different"})
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                Journal(path, changed, resume=True)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "intact"):
            Journal(path, identity, resume=True)

    def test_truncated_raw_rejected(self):
        path = self.directory() / "raw.csv"
        Journal(path, {"synthetic": True}).close()
        with path.open("a") as stream:
            stream.write("interrupted,partial\n")
        with self.assertRaisesRegex(ValueError, "partial"):
            Journal(path, {"synthetic": True}, resume=True)

    def test_latest_partial_cycle_cannot_borrow_old_groups(self):
        expected = [{key: "synthetic" for key in CoverageAuditor.KEYS}]
        base = dict(expected[0], status="OK", accuracy_pass=True, calls_per_group=1,
                    per_call_wall=[1.], per_call_event=[1.], per_call_host=[1.])
        complete = [dict(base, cycle_id=0, round=r, group=g) for r in range(3) for g in range(20)]
        self.assertTrue(CoverageAuditor.audit(complete, expected)["complete"])
        partial = complete + [dict(base, cycle_id=1, round=0, group=0)]
        self.assertFalse(CoverageAuditor.audit(partial, expected)["complete"])
        duplicate = complete + [copy.deepcopy(complete[0])]
        self.assertFalse(CoverageAuditor.audit(duplicate, expected)["complete"])

    def test_annotation_does_not_double_count_kernel(self):
        trace = {"traceEvents": [{"cat": "kernel", "ph": "X", "dur": 7},
                                  {"cat": "gpu_user_annotation", "ph": "X", "dur": 9},
                                  {"cat": "cuda_runtime", "ph": "X", "dur": 2},
                                  {"cat": "kernel", "ph": "B", "dur": 7}]}
        self.assertEqual(sum(row["dur"] for row in TraceSummary.kernels(trace)), 7)

    def test_statistics_and_full_point_count(self):
        self.assertEqual(len(points()), 75)
        self.assertEqual(len({point.key for point in points()}), 75)
        self.assertEqual(points()[-1].seed, 1074)
        self.assertEqual(Statistics.summarize([1, 2, 3])["spread"], 1)
        with self.assertRaises(ValueError):
            Statistics.summarize([float("nan")])

    def test_gpu_idle_requires_consecutive_samples_after_telemetry_lag(self):
        session = object.__new__(Session)
        session.root = self.directory()
        (session.root / "env").mkdir()
        header = "index, uuid, memory.used [MiB], utilization.gpu [%]\n"
        processes = "gpu_uuid, pid, used_gpu_memory [MiB]\n"
        outputs = [value for util in (2, 0, 0) for value in
                   (header + f"4, GPU-synthetic, 4, {util}\n", processes)]
        with patch("session.subprocess.check_output", side_effect=outputs), patch("session.time.sleep"):
            self.assertEqual(session.idle_gpu("lag", preferred=4), "GPU-synthetic")
        self.assertEqual(len(json.loads((session.root / "env/lag_idle_samples.json").read_text())), 3)

    def test_gpu_util_zero_does_not_ignore_occupied_memory(self):
        session = object.__new__(Session)
        session.root = self.directory()
        (session.root / "env").mkdir()
        outputs = [value for _ in range(6) for value in
                   ("index, uuid, memory.used [MiB], utilization.gpu [%]\n4, GPU-synthetic, 50000, 0\n",
                    "gpu_uuid, pid, used_gpu_memory [MiB]\n")]
        with patch("session.subprocess.check_output", side_effect=outputs), patch("session.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "six bounded samples"):
                session.idle_gpu("busy", preferred=4)


class Selfcheck:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output", type=Path, required=True, help="Fresh CPU-only synthetic output under configured work disk")
        args = parser.parse_args()
        args.output.mkdir(parents=True, exist_ok=False)
        IntegrityTests.root = args.output
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntegrityTests))
        Evidence.write(args.output / "result.json", {"classification": "CPU_SYNTHETIC_NOT_H20_PERFORMANCE",
            "passed": result.wasSuccessful(), "tests": result.testsRun,
            "failures": [(str(test), detail) for test, detail in result.failures],
            "errors": [(str(test), detail) for test, detail in result.errors]})
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(Selfcheck.main())
