"""CPU counterexamples for performance grouping, binding and product-only arithmetic."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

from cases import Case, Point, References
from common import Evidence
from performance import MeasurementIdentity, PerformancePlan, PilotAnchors, SamplingProtocol
from performance_bindings import TorchCalls
from performance_driver import PerformanceDriver
from protocol import Journal
from qualification import AccuracyContracts, RecordedAccuracy
from roofline_model import EqlinearWorkload, ProfileTraffic
from timing_audit import ScanRepetition, TimingAudit, TimingStatistics
from timing_tables import TimingTables


class SamplerTests(unittest.TestCase):
    root = None

    def test_group_rule_is_anchor_only_and_bounded(self):
        self.assertEqual([SamplingProtocol.count(value) for value in (.01, 1.5, 3, 10)], [20, 10, 5, 1])
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SamplingProtocol.count(value)

    def test_pilot_locks_later_family_operations_without_changing_formal_calls(self):
        plan = {"include_families": ["forward", "backward"], "selected_points": {"forward": ["P"], "backward": ["P"]},
                "expected_calls": {"forward": ["v/forward/public_warm"], "backward":
                                   ["b/grad_x_w/prepared_kernel", "b/grad_x_w/prepare_weight"]}}
        declared = PilotAnchors.declared(plan, "P")
        anchors = {"forward/public_warm": {"calls_per_group": 10}, "grad_x_w/prepared_gemm": {"calls_per_group": 5}}
        completed = PilotAnchors.complete(anchors, declared)
        self.assertTrue(declared <= set(completed))
        self.assertEqual(completed["grad_x_w/prepared_kernel"]["calls_per_group"], 5)
        self.assertEqual(completed["grad_x_w/prepare_weight"]["calls_per_group"], 1)
        self.assertEqual(anchors["grad_x_w/prepared_gemm"]["calls_per_group"], 5)
        with self.assertRaisesRegex(ValueError, "No same-operation"):
            PilotAnchors.complete({"forward/public_warm": {}}, declared)

    def test_uuid_format_does_not_allow_a_different_device(self):
        self.assertEqual(MeasurementIdentity.gpu_uuid("GPU-ABcd"), MeasurementIdentity.gpu_uuid("abcd"))
        self.assertNotEqual(MeasurementIdentity.gpu_uuid("GPU-abcd"), MeasurementIdentity.gpu_uuid("GPU-abce"))

    def test_row_cannot_mix_source_binary_gpu_reference_or_protocol(self):
        identity = {"run_id": "CPU", "GPU_UUID": "a", "protocol_hash": "p", "input_hash": "i",
                    "bindings": {"v": {"source_hash": "s", "binary_hash": "b"}}, "reference_hashes": {"forward": "r"}}
        row = dict(run_id="CPU", GPU_UUID="a", protocol_hash="p", input_hash="i", source_hash="s", binary_hash="b",
                   variant_id="v", operation="forward", reference_hash="r", wall=1., event=1., host=1.)
        MeasurementIdentity.check_row(row, identity)
        for field in ("run_id", "GPU_UUID", "protocol_hash", "input_hash", "source_hash", "binary_hash", "reference_hash"):
            with self.assertRaises(ValueError):
                MeasurementIdentity.check_row(dict(row, **{field: "different"}), identity)
        for metric in ("wall", "event", "host"):
            with self.assertRaises(ValueError):
                MeasurementIdentity.check_row(dict(row, **{metric: float("nan")}), identity)

    def test_plan_producer_tampering_rejected(self):
        class SessionStub:
            pass
        session = SessionStub()
        session.root = Path(tempfile.mkdtemp(dir=self.root))
        directory = session.root / "plan"
        PerformancePlan.create(session, directory)
        PerformancePlan.load(directory / "plan.json")
        (directory / "producer/cases.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "producer changed"):
            PerformancePlan.load(directory / "plan.json")

    def test_torch_products_preserve_path_groups_and_dx_chain_rule(self):
        import torch

        case = Case(Point("MEDIUM_P2", 2, "fp64", 2, 0), "cpu")
        references = References(case)
        calls = TorchCalls(case)
        calls.prepare(("grad_x", "grad_w", "grad_x_w"))
        for operation in ("forward", "grad_x", "grad_w", "grad_x_w"):
            full = calls.full(operation)
            products = calls.mm(operation)
            normalized = calls.normalize_mm(operation, products)
            kinds = MeasurementIdentity.result_kinds(operation)
            for kind, result, packed in zip(kinds, full if isinstance(full, tuple) else (full,),
                                            normalized if isinstance(normalized, tuple) else (normalized,)):
                self.assertLessEqual(references.error(result, kind), 1e-12)
                self.assertLessEqual(references.error(packed, kind), 1e-12)
        # Separate coefficients for repeated paths must be applied to dy before MM.
        for index in range(3):
            expected = case.grad3[:, :1].reshape(-1, case.u) * case.cg[index] if index < 2 else case.grad3[:, 1:4].reshape(-1, case.u) * case.cg[index]
            self.assertTrue(torch.equal(calls.products["grad_x"][index][0], expected))

    def test_forward_autograd_uses_prepared_leaf_and_real_nonzero_dy(self):
        import torch

        case = Case(Point("MEDIUM_P2", 2, "fp64", 2, 0), "cpu")
        references, calls = References(case), TorchCalls(case)
        leaf = calls.forward_x
        self.assertTrue(leaf.is_leaf and leaf.requires_grad)
        self.assertFalse(case.x.requires_grad)
        outputs = [calls.full("forward_autograd") for _ in range(2)]
        self.assertIs(calls.forward_x, leaf)
        self.assertNotEqual(outputs[0].data_ptr(), outputs[1].data_ptr())
        for output in outputs:
            dx, = torch.autograd.grad(output, leaf, case.grad)
            self.assertLessEqual(references.error(dx, "grad_x"), 1e-12)

    def test_roofline_useful_work_requires_qualified_complete_boundary(self):
        profile = {"call": "v/forward/public_warm", "quality": {"passed": True}}
        self.assertTrue(ProfileTraffic.model_applicable(profile))
        self.assertFalse(ProfileTraffic.model_applicable({**profile, "quality": {"passed": False}}))
        self.assertFalse(ProfileTraffic.model_applicable({"call": profile["call"]}))
        self.assertFalse(ProfileTraffic.model_applicable({**profile, "call": "v/forward/prepare_weight"}))

    def test_ring_audit_rejects_deleted_member_and_changed_identity(self):
        identities = [{"label": f"ring{index:04d}", "index": index, "input_hash": str(index), "reference_hash": "gold" + str(index)} for index in range(2)]
        binding = {"quality_member_identities": {"forward": identities}}
        component = {"members": {str(index): dict(value) for index, value in enumerate(identities)}, "member_count": 2}
        quality = {"components": {"forward": component}}
        workload = {"kind": "rotating_inputs", "ordered_input_hashes": ["0", "1"],
                    "ordered_reference_hashes": [{"forward": "gold0"}, {"forward": "gold1"}]}
        self.assertEqual(TimingAudit.member_errors(quality, binding, "forward", workload), [])
        missing = copy.deepcopy(quality)
        del missing["components"]["forward"]["members"]["1"]
        self.assertTrue(TimingAudit.member_errors(missing, binding, "forward", workload))
        changed = copy.deepcopy(quality)
        changed["components"]["forward"]["members"]["1"]["reference_hash"] = "different"
        self.assertTrue(TimingAudit.member_errors(changed, binding, "forward", workload))
        self.assertTrue(TimingAudit.member_errors(quality, {}, "forward", workload))

    def test_handwritten_guard_rejects_changed_binary_source_and_policy(self):
        from handwritten_adapter import HandwrittenLibrary

        root = Path(tempfile.mkdtemp(dir=self.root))
        for folder in ("archive", "source"):
            (root / folder).mkdir()
            (root / folder / "kernel.cu").write_text("CPU_SYNTHETIC_SOURCE")
        binary, receipt, policy = (root / name for name in ("fake.so", "receipt.json", "policy.json"))
        for path in (binary, receipt, policy):
            path.write_text("CPU_SYNTHETIC_NOT_EXECUTABLE")
        guard = HandwrittenLibrary.__new__(HandwrittenLibrary)
        guard.root, guard.library = root, binary
        guard.receipts = {str(receipt): Evidence.sha256(receipt)}
        guard.source = {"archive": str(root / "archive"),
                        "original_files": {"kernel.cu": Evidence.sha256(root / "archive/kernel.cu")},
                        "staged_files": {"kernel.cu": Evidence.sha256(root / "source/kernel.cu")},
                        "math_policy": {"path": str(policy), "sha256": Evidence.sha256(policy)}}
        guard.build = {"binary": {"sha256": Evidence.sha256(binary)}}
        guard.verify()
        for path in (binary, receipt, policy, root / "archive/kernel.cu", root / "source/kernel.cu"):
            original = path.read_bytes()
            path.write_bytes(original + b"CHANGED")
            with self.assertRaises(ValueError):
                guard.verify()
            path.write_bytes(original)
            guard.verify()

    def test_native_accuracy_identity_uses_exact_mapped_objects_and_job(self):
        path = Path(tempfile.mkdtemp(dir=self.root)) / "CPU_native_manifest.json"
        manifest = {"status": "PASS", "binary": "CPU_DSO", "binary_sha256": "CPU_SHA", "namespace": "CPU_NAMESPACE",
                    "contract": {"source_commit": "CPU_COMMIT", "native_files": {"CPU_OBJECT": "CPU_OBJECT_SHA"}}}
        Evidence.write(path, manifest)
        identity = {"family": "current_native_selector", "precision": "fp64", "impl": 0,
                    "binary": "CPU_DSO", "binary_sha256": "CPU_SHA", "mapped_libraries": ["CPU_DSO"],
                    "native_math_modified": False, "production_DSO_loaded": False,
                    "namespace": "CPU_NAMESPACE", "manifest": str(path), "manifest_sha256": Evidence.sha256(path),
                    "source_commit": "CPU_COMMIT", "native_objects_sha256": Evidence.fingerprint(manifest["contract"]["native_files"])}
        source = {"identity": identity, "precision": "fp64", "impl": 0, "job_id": "native_fp64_impl0_CPU_1",
                  "point": {"config": "CPU", "batch": 1, "precision": "fp64"}}
        self.assertEqual(AccuracyContracts.identity(source)["binary_hash"], "CPU_SHA")
        self.assertEqual(AccuracyContracts.tested_variant(source), "native_fp64_impl0")
        for field, value in (("native_objects_sha256", "WRONG"), ("mapped_libraries", []), ("native_math_modified", True)):
            changed = copy.deepcopy(source)
            changed["identity"][field] = value
            with self.assertRaises(ValueError):
                AccuracyContracts.identity(changed)
        with self.assertRaises(ValueError):
            AccuracyContracts.tested_variant({**source, "job_id": "native_fp64_impl1_CPU_1"})

    def test_fp64_handwritten_adapter_keeps_post_cg_and_run_length_order(self):
        from types import SimpleNamespace
        from handwritten_fp64 import HandwrittenFp64Adapter
        import torch

        torch.set_num_threads(1)
        case = Case(Point("MEDIUM_P2", 32, "fp64", 2, 0), "cpu")
        references = References(case)
        root = Path(tempfile.mkdtemp(dir=self.root))
        policy = root / "coverage/handwritten_fp64_source02/fp64_contracts.json"
        kinds = ("fp64_gmm_p1_v2", "fp64_gmm_v4_multi_path")
        Evidence.write(policy, {"contracts": [{"kind": kind, "math_allowed": True, "source_sha256": "CPU_SHA",
            "allowed_path_counts": [1] if index == 0 else [4, 10, 17], "output_semantics": "PER_INPUT_PATH_OUTPUT"}
            for index, kind in enumerate(kinds)]})

        class Op:
            _schemas = {"": "CPU_SYNTHETIC_NOT_CUDA"}

            def __call__(self, x, w, *metadata):
                if w.ndim == 2:
                    return (x @ w) * metadata[0]
                dims, val = metadata
                values, offset = [], 0
                for index, dim in enumerate(dims):
                    values.append((x[:, offset:offset+dim] @ w[index]) * val)
                    offset += dim
                return torch.cat(values, dim=1)

        for kind in kinds:
            source = SimpleNamespace(session=SimpleNamespace(root=root), source={"kind": kind, "original_files": {"cpu": "CPU_SHA"}},
                                     ops=SimpleNamespace(forward=Op()), verify=lambda: None)
            adapter = HandwrittenFp64Adapter(case, source)
            adapter.prepare("forward", adapter.BOUNDARY)
            first, second = [adapter.call("forward", adapter.BOUNDARY) for _ in range(2)]
            self.assertLessEqual(references.error(first, "forward"), 1e-12)
            self.assertTrue(torch.equal(first, second))
            self.assertNotEqual(first.data_ptr(), second.data_ptr())
            adapter.close()

    def test_cross_process_family_order_balanced_across_rounds(self):
        from collections import Counter
        positions = Counter()
        families = ["current", "p0", "p14"]
        for round_index in range(3):
            for config_index in range(5):
                for position, family in enumerate(PerformanceDriver.order(round_index, config_index, families)):
                    positions[(family, position)] += 1
        self.assertEqual(set(positions.values()), {5})
        many = [str(index) for index in range(66)]
        for config_index in range(5):
            first = PerformanceDriver.order(0, config_index, many)
            self.assertEqual(PerformanceDriver.order(1, config_index, many), list(reversed(first)))
            for round_index in range(3):
                self.assertEqual(set(PerformanceDriver.order(round_index, config_index, many)), set(many))

    def test_point_chunks_preserve_scope_and_isolate_precision(self):
        plan = {"selected_points": {"F": [point.key for point in __import__("cases").points()]}}
        original = PerformanceDriver.chunks(plan, "SMALL_P1", "F", 0)
        self.assertEqual(len(original), 1)
        self.assertEqual(len(original[0].points(plan)), 15)
        plan["factory_precision_chunks"] = True
        typed = PerformanceDriver.chunks(plan, "SMALL_P1", "F", 0)
        self.assertEqual(len(typed), 3)
        self.assertEqual({len(chunk.points(plan)) for chunk in typed}, {5})
        split = PerformanceDriver.chunks(plan, "SMALL_P1", "F", 0, split_precisions=["fp64"])
        self.assertEqual(len(split), 7)
        full = PerformanceDriver.chunks(plan, "SMALL_P1", "F", 0, per_batch=True)
        keys = [key for task in full for key in task.points(plan)]
        self.assertEqual(len(keys), 15)
        self.assertEqual(len(set(keys)), 15)
        self.assertTrue(all(task.batch is not None and task.precision is not None for task in full))

    def test_cross_round_variation_keeps_slow_round_and_partial_status(self):
        rows = []
        for round_index, latency in enumerate((1., 1., 1.2)):
            for group in range(20):
                rows.append(dict(variant_id="CPU", config="CPU", B=1, precision="fp32", operation="forward",
                    call_boundary="public_warm", round=round_index, accuracy_pass=True, warmup_stabilized=True,
                    wall=latency, event=latency, host=latency))
        summaries, across = TimingStatistics.summarize(rows)
        self.assertEqual(len(summaries), 3)
        self.assertTrue(across[0]["rounds_complete"])
        self.assertTrue(across[0]["unstable_cross_round"])
        self.assertTrue(across[0]["requires_complete_scan_repeat"])
        self.assertAlmostEqual(across[0]["cross_round_variation"], .2)
        self.assertEqual(across[0]["round_medians_ms"]["2"], 1.2)
        _, partial = TimingStatistics.summarize(rows[:-20])
        self.assertFalse(partial[0]["rounds_complete"])
        self.assertIsNone(partial[0]["cross_round_variation"])

    def test_noise_repeat_is_bounded_and_cannot_qualify_an_unstable_second_scan(self):
        first = ScanRepetition.status(True, True, False)
        self.assertTrue(first["requires_complete_scan_repeat"])
        second = ScanRepetition.status(True, True, True)
        self.assertFalse(second["requires_complete_scan_repeat"])
        self.assertTrue(second["noise_repeat_limit_reached"])
        self.assertTrue(second["stability_limited"])
        self.assertFalse(TimingTables.ranking_eligible(dict(full_factory_scope_complete=True, **second)))
        self.assertTrue(ScanRepetition.status(True, False, True, True)["requires_complete_scan_repeat"])

    def test_stable_points_in_a_complete_repeat_do_not_borrow_unstable_points(self):
        row = dict(variant_id="CPU", config="CPU", B=512, precision="fp32", operation="forward", call_boundary="public_warm")
        point = {**row, "rounds_complete": True, "qualified": True, "unstable_spread": False, "unstable_cross_round": False}
        audit = {"full_factory_scope_complete": True, "comparison_qualified": True, "latest_cycle_verified": True,
                 **ScanRepetition.status(True, True, True), "across_rounds": [point]}
        self.assertFalse(TimingTables.ranking_eligible(audit))
        self.assertTrue(TimingTables.point_ranking_eligible(audit, row))
        point["unstable_cross_round"] = True
        self.assertFalse(TimingTables.point_ranking_eligible(audit, row))
        point["unstable_cross_round"] = False
        audit["requires_complete_scan_repeat"] = True
        self.assertFalse(TimingTables.point_ranking_eligible(audit, row))

    def test_speedup_sums_do_not_average_ratios_or_qualify_partial_scans(self):
        self.assertAlmostEqual(TimingTables.sum_ratio([1., 100.], [2., 50.]), 101. / 52.)
        self.assertNotEqual(TimingTables.sum_ratio([1., 100.], [2., 50.]), (.5 + 2.) / 2.)
        base = dict(config="CPU", B=512, precision="fp32", operation="forward", round=0,
                    call_boundary="public_warm", qualified=True, unstable_spread=False)
        audit = {"full_factory_scope_complete": False, "requires_complete_scan_repeat": False,
                 "summaries": [{**base, "variant_id": "torch_ieee_full", "wall_median": 2.},
                               {**base, "variant_id": "CPU_CANDIDATE", "wall_median": 1.}]}
        result = TimingTables.paired_speedups(audit)
        self.assertEqual(result[0]["diagnostic_speed_ratio"], 2.)
        self.assertFalse(result[0]["ranking_eligible"])
        audit["summaries"][1]["qualified"] = False
        audit["full_factory_scope_complete"] = True
        self.assertFalse(TimingTables.paired_speedups(audit)[0]["ranking_eligible"])
        audit["summaries"][1]["qualified"] = True
        audit["comparison_qualified"] = False
        self.assertFalse(TimingTables.paired_speedups(audit)[0]["ranking_eligible"])
        audit["comparison_qualified"] = True
        audit["summaries"][1]["call_boundary"] = "public_allocating"
        self.assertTrue(TimingTables.paired_speedups(audit)[0]["ranking_eligible"])
        audit["boundary_adjudications"] = [{"selectors": {"operation": ["forward"], "variant_id": ["torch_ieee_full"]}}]
        self.assertFalse(TimingTables.paired_speedups(audit)[0]["ranking_eligible"])
        self.assertFalse(TimingTables.paired_speedups(audit)[0]["boundary_comparable"])

    def test_workload_model_keeps_run_length_groups_and_unknown_counters(self):
        workload = EqlinearWorkload(2, 3, 5, (1, 3, 1), 4)
        forward = workload.operation("forward")
        self.assertEqual(forward["j"], 5)
        self.assertEqual(forward["logical_run_length_reduction_add_flops"], 0)
        self.assertEqual(forward["useful_matrix_product_flops"], 2*2*5*3*5)
        combined = workload.operation("grad_x_w")
        self.assertEqual(combined["useful_matrix_product_flops"], 2*forward["useful_matrix_product_flops"])
        self.assertEqual(combined["logical_cg_multiply_flops"], forward["logical_cg_multiply_flops"])
        self.assertIsNone(forward["executed_flops"])
        zero = {"metrics": {"m": {"si": {"value": 0., "unit": "byte"}}}}
        self.assertEqual(ProfileTraffic.total([zero], "m", "byte"), 0.)
        self.assertIsNone(ProfileTraffic.total([zero, {"metrics": {}}], "m", "byte"))

    def test_failed_upstream_contract_cannot_be_upgraded_by_a_local_pass(self):
        directory = Path(tempfile.mkdtemp(dir=self.root))
        evidence = directory / "CPU_SYNTHETIC_ACCURACY.json"
        metric = dict(operation="forward", result="forward", boundary="CPU_BOUNDARY", accuracy_tier="fp32",
            torch_rel_fro=.1, threshold=.10500000000000001, rel_fro=.2, passed=False,
            finite=True, reference_finite=True, repeat_equal=True, different_buffer=True)
        source = dict(status="ACCURACY_FAIL", variant_id="CPU", input_hash="I", reference_hashes={"forward": "R"},
            selected_boundaries=["CPU_BOUNDARY"], identity=dict(source_hash="S", binary_hash="B", runtime_controls={}, api_controls={}), rows=[metric])
        Evidence.write(evidence, source)
        entry = dict(variant_id="CPU", point_key="P", operation="forward", input_hash="I", reference_hashes={"forward": "R"},
            production_source_hash="S", binary_hash="B", runtime_controls={}, api_controls={}, passed=False,
            evidence={"path": str(evidence), "sha256": Evidence.sha256(evidence)})
        path = directory / "CPU_SYNTHETIC_CONTRACTS.json"
        Evidence.write(path, {"schema": 1, "complete": True, "entries": {"K": entry}})
        binding = dict(source_hash="ADAPTER", production_source_hash="S", binary_hash="B", runtime_controls={}, api_controls={},
                       accuracy_contract={"path": str(path), "sha256": Evidence.sha256(path), "entry_key": "K"})
        verdict = AccuracyContracts.resolve(binding, "CPU", "P", "forward", "I", {"forward": "R"})
        self.assertFalse(verdict["passed"])
        shared = copy.deepcopy(binding)
        shared["accuracy_contract"] = {key: value for key, value in shared["accuracy_contract"].items() if key != "entry_key"}
        shared["accuracy_contract"]["entry_keys_by_operation"] = {"forward": "K", "grad_x": "K"}
        self.assertFalse(AccuracyContracts.resolve(shared, "CPU", "P", "forward", "I", {"forward": "R"})["passed"])
        with self.assertRaisesRegex(ValueError, "differ"):
            AccuracyContracts.resolve(shared, "CPU", "P", "grad_x", "I", {"forward": "R"})
        self.assertTrue(RecordedAccuracy.passes({**metric, "rel_fro": 0.}, "fp32"))
        with self.assertRaisesRegex(ValueError, "differ"):
            AccuracyContracts.resolve(binding, "CPU", "P", "forward", "OTHER_INPUT", {"forward": "R"})
        entry["passed"] = True
        Evidence.write(path, {"schema": 1, "complete": True, "entries": {"K": entry}})
        binding["accuracy_contract"]["sha256"] = Evidence.sha256(path)
        with self.assertRaisesRegex(ValueError, "independently"):
            AccuracyContracts.resolve(binding, "CPU", "P", "forward", "I", {"forward": "R"})
        self.assertFalse(RecordedAccuracy.passes({**metric, "accuracy_tier": "fp64", "threshold": 1e-12,
                                                 "rel_fro": 0., "torch_rel_fro": None}, "fp64"))

    def test_original_cuda_rejects_float32_before_operator_dispatch(self):
        from original_cuda_probe import OriginalCudaFp64Adapter
        import torch

        case = Case(Point("SMALL_P1", 2, "fp64", 0, 0), "cpu")
        OriginalCudaFp64Adapter.require_double(case)
        case.x = case.x.float()
        with self.assertRaisesRegex(ValueError, "TF32x1"):
            OriginalCudaFp64Adapter.require_double(case)

    def test_luna_upstream_identity_requires_actual_fused_mapped_library(self):
        source = {"source_manifest_hash": "CPU_SOURCE", "identity": {"python_source_hash": "CPU_PYTHON",
            "jit_libraries": {"CPU_BINARY": "CPU_SHA"}, "mapped_jit_libraries": ["CPU_BINARY"], "backend": "cuda_fused"}}
        identity = AccuracyContracts.identity(source)
        self.assertEqual(identity["source_hash"], "CPU_SOURCE")
        self.assertEqual(identity["binary_hash"], Evidence.fingerprint({"CPU_BINARY": "CPU_SHA"}))
        source["identity"]["mapped_jit_libraries"] = []
        with self.assertRaisesRegex(ValueError, "mapped fused"):
            AccuracyContracts.identity(source)

    def test_raw_auditor_rejects_partial_cycles_and_separates_diagnostics(self):
        class SessionStub:
            pass
        session = SessionStub()
        session.root = Path(tempfile.mkdtemp(dir=self.root))
        directory = session.root / "plan"
        plan = PerformancePlan.create(session, directory)
        point = Point("SMALL_P1", 512, "fp32", 0, 0)
        point_dir = directory / "points/SMALL_P1_512_fp32"
        point_dir.mkdir(parents=True)
        data = session.root / "CPU_SYNTHETIC_BINARY_AND_REFERENCE"
        data.write_bytes(b"CPU counterexample, not GPU evidence")
        digest = Evidence.sha256(data)
        bindings = {key.split("/")[0]: {"source_hash": "CPU_SOURCE", "binary_hash": digest,
                    "binary_files": {str(data): {"sha256": digest}}} for key in plan["expected_calls"]["current"]}
        identity = {"run_id": "CPU", "GPU_UUID": "CPU", "protocol_hash": plan["protocol_hash"], "input_hash": "CPU_INPUT",
                    "reference_hashes": {"forward": "F", "grad_x": "X"}, "bindings": bindings,
                    "family": "current", "producer": plan["producer"], "affinity": {"CPU": True},
                    "reference_receipt": {"path": str(data), "sha256": digest, "data_path": str(data), "data_sha256": digest}}
        anchors = {key.split("/", 1)[1]: {"calls_per_group": 1} for key in plan["expected_calls"]["current"]}
        Evidence.write(point_dir / "anchor_pilot.json", {"identity": {key: identity[key] for key in
            ("GPU_UUID", "protocol_hash", "input_hash", "reference_hashes", "affinity")}, "anchors": anchors})
        quality = {key: {"passed": True, "components": {"forward" if key.split("/")[1] == "forward_autograd" else key.split("/")[1]:
            {"passed": True, "accuracy_tier": "fp32", "torch_rel_fro": .1, "threshold": 1.05 * .1, "rel_fro": 0.,
             "finite": True, "reference_finite": True, "repeat_equal": True, "different_buffer": True}}}
                   for key in plan["expected_calls"]["current"]}
        for round_index in range(3):
            attempt = point_dir / f"CPU_r{round_index}"
            attempt.mkdir()
            frame = {**identity, "frame": {"cycle": 1, "round": round_index, "mode": "measure"}}
            Evidence.write(attempt / "setup.json", {"identity": frame, "quality": quality,
                "warmup": {key: {"stabilized": True} for key in quality}})
            path = attempt / "raw.csv"
            journal = Journal(path, frame)
            compare_groups = {}
            for key in plan["expected_calls"]["current"]:
                variant, operation, boundary = key.split("/")
                compare_groups.setdefault((operation, boundary), []).append(variant)
            for (operation, boundary), variants in compare_groups.items():
                for group in range(20):
                    for position, variant in enumerate(sorted(variants, reverse=bool((group + round_index) % 2))):
                        row = {name: "CPU" for name in Journal.FIELDS}
                        row.update(run_id="CPU", attempt_id=attempt.name, cycle_id=1, round=round_index, variant_id=variant,
                            source_hash="CPU_SOURCE", binary_hash=digest, GPU_UUID="CPU", protocol_hash=plan["protocol_hash"],
                            config=point.config, B=512, U=96, V=96, i_dims=[1,3,5,7], P=4, I=16, J=16, precision="fp32",
                            input_hash="CPU_INPUT", reference_hash="F" if operation in ("forward", "forward_autograd") else "X",
                            operation=operation, call_boundary=boundary, cache_mode="warm_same_buffer", weight_state="constant",
                            samples=20, group=group, calls_per_group=1, order=position, wall=1., event=.5, host=.3,
                            value_unit="ms", status="OK", accuracy_pass=True, per_call_wall=[.9], per_call_event=[.5],
                            per_call_host=[.3], sync_method=SamplingProtocol.VALUE["sync"])
                        journal.emit(row)
            journal.close()
            Evidence.write(attempt / "receipt.json", {"status": "COMPLETE", "identity_hash": Evidence.fingerprint(frame),
                "raw_sha256": Evidence.sha256(path), "round": round_index, "cycle": 1})
        args = argparse.Namespace(plan=directory / "plan.json", family="current", cycle=1, point=point.key,
                                  output=session.root / "CPU_audit.json")
        self.assertEqual(TimingAudit(args).run(), 0)
        result = json.loads(args.output.read_text())
        self.assertTrue(result["selected_scope_complete"])
        self.assertFalse(result["full75x3_complete"])
        # Missing a terminal third round must never borrow older/same-point evidence.
        (point_dir / "CPU_r2/receipt.json").unlink()
        args.output = session.root / "CPU_incomplete_audit.json"
        self.assertEqual(TimingAudit(args).run(), 1)


class PerformanceSelfcheck:
    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        args.output.mkdir(parents=True, exist_ok=False)
        SamplerTests.root = args.output
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SamplerTests))
        Evidence.write(args.output / "result.json", {"classification": "CPU_SYNTHETIC_NOT_H20_PERFORMANCE", "passed": result.wasSuccessful(),
            "tests": result.testsRun, "failures": [(str(test), detail) for test, detail in result.failures],
            "errors": [(str(test), detail) for test, detail in result.errors]})
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(PerformanceSelfcheck.main())
