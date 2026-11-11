"""CPU_SYNTHETIC_NOT_PERFORMANCE: storage, sequence and adversarial gate checks."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

from cases import Case, Point
from common import Evidence
from performance import CallValidation, SamplingProtocol
from performance_bindings import BoundCall, QualityMember, TorchCalls
from protocol import Accuracy
from qualification import RecordedAccuracy
from ring_bindings import RingCall
from ring_fixtures import RingDataset, RingSpecification
from ring_references import RingReferenceStore


class RingSelfTest:
    def __init__(self, directory):
        import torch

        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        torch.set_num_threads(1)
        self.checks = {}

    def require_rejection(self, label, function):
        try:
            function()
        except ValueError:
            self.checks[label] = True
        else:
            raise AssertionError("Counterexample was accepted: " + label)

    def storage_and_sequence(self):
        import torch

        canonical = Case(Point("SMALL_P1", 2, "fp32", 0, 0), device="cpu", fixture_root=self.directory / "canonical")
        specification = RingSpecification.for_case(canonical, l2_bytes=32)
        hardware = {"classification": "CPU_SYNTHETIC_NOT_H20_CAPACITY", "l2_bytes": 32}
        first = RingDataset.materialize(canonical, self.directory / "fixtures", specification, hardware, device="cpu")
        second = RingDataset.materialize(canonical, self.directory / "fixtures", specification, hardware, device="cpu")
        assert first.input_hash == second.input_hash and first.manifest_sha256 == second.manifest_sha256
        assert first.manifest["ordered_members"] == second.manifest["ordered_members"]
        assert len({member.hashes["x_sha256"] for member in first.members}) == specification.buffer_count
        assert all(member.w.data_ptr() == canonical.w.data_ptr() for member in first.members)
        assert first.members[0].hashes == canonical.hashes
        reference_store = RingReferenceStore(self.directory / "ring_references", self.directory / "member_references")
        references = reference_store.get(first)
        repeated = reference_store.get(second)
        assert references.hashes == repeated.hashes and not repeated.store_receipt["created"]
        assert references.store_receipt["data_sha256"] == repeated.store_receipt["data_sha256"]
        torch_calls = TorchCalls(first.case)
        original = BoundCall("cpu_synthetic", "forward", "public_warm", lambda: torch_calls.full("forward"),
            lambda: torch_calls.full("forward"), lambda value: value, {"accuracy_tier": "fp32"}, "CPU synthetic public path")
        wrapper = RingCall(original, first, references)
        first_raw, second_raw = wrapper.check(), wrapper.check()
        component = Accuracy.evaluate(first_raw["aggregate"], references.values["fp64"]["forward"],
            references.torch_errors["forward"], "fp32", torch.equal(first_raw["aggregate"], second_raw["aggregate"]), True)
        components = {"forward": component}
        CallValidation.add_members(wrapper.call, first_raw, second_raw, components, first.case)
        assert components["forward"]["passed"] and RecordedAccuracy.passes(components["forward"], "fp32")
        incomplete = copy.deepcopy(components["forward"])
        incomplete["members"].pop("1")
        assert not RecordedAccuracy.passes(incomplete, "fp32")
        wrapper.before_sample(3, 5, 0)
        observed = []
        for _ in range(5):
            observed.append(first.members[first.cursor].input_hash)
            wrapper.run()
        wrapper.before_sample(3, 5, 2)
        repeated_order = []
        for _ in range(5):
            repeated_order.append(first.members[first.cursor].input_hash)
            wrapper.run()
        assert observed == repeated_order
        assert len({value.data_ptr() for value in wrapper.outputs}) == specification.buffer_count
        original_hash = first.members[1].hashes["x_sha256"]
        first.members[1].x[0, 0] += 1
        self.require_rejection("mutated_actual_ring_bytes_rejected", first.verify)
        assert original_hash != Case.tensor_hash(first.members[1].x)
        self.checks.update(serialized_reload_exact=True, fixed_physical_weight=True, actual_inputs_distinct=True,
            common_references_aggregated=True, complete_ring_quality_pass=True, deterministic_candidate_sequence=True,
            retained_output_slots_distinct=True, recorded_missing_member_rejected=True)

    def adversarial_members(self):
        import torch

        gold = [torch.tensor([1.], dtype=torch.float64), torch.tensor([1e6], dtype=torch.float64)]
        actual = [torch.tensor([1.0001], dtype=torch.float64), gold[1].clone()]
        repeated = [value.clone() for value in actual]
        samples = [QualityMember(value, repeat, reference, 0., f"ring{index:04d}", index,
                                 f"synthetic_input_{index}", Case.tensor_hash(reference))
                   for index, (value, repeat, reference) in enumerate(zip(actual, repeated, gold))]
        call = BoundCall("cpu_synthetic", "forward", "public_warm", None, None, None,
            {"accuracy_tier": "tf32", "quality_member_identities": {"forward": [sample.identity() for sample in samples]}},
            "CPU synthetic counterexample", quality_members=lambda first, second: {"forward": samples})
        case = SimpleNamespace(point=SimpleNamespace(precision="tf32"))
        aggregate = Accuracy.evaluate(torch.cat(actual), torch.cat(gold), 0., "tf32", True, True)
        assert aggregate["passed"]
        components = {"forward": aggregate}
        CallValidation.add_members(call, {}, {}, components, case)
        assert not components["forward"]["passed"] and not RecordedAccuracy.passes(components["forward"], "tf32")
        self.checks["aggregate_frobenius_cannot_hide_member_failure"] = True
        original_samples = samples[:]
        samples.pop()
        self.require_rejection("missing_member_rejected", lambda: CallValidation.add_members(call, {}, {}, {"forward": {}}, case))
        samples[:] = original_samples
        samples[1] = copy.copy(samples[1])
        samples[1].index = 0
        self.require_rejection("duplicate_member_rejected", lambda: CallValidation.add_members(call, {}, {}, {"forward": {}}, case))
        samples[:] = original_samples
        samples[0] = copy.copy(samples[0])
        samples[0].gold = gold[0] + 1
        self.require_rejection("changed_member_gold_rejected", lambda: CallValidation.add_members(call, {}, {}, {"forward": {"passed": True}}, case))
        samples[:] = original_samples
        samples[0] = copy.copy(samples[0])
        samples[0].first = samples[0].second
        components = {"forward": {"passed": True}}
        CallValidation.add_members(call, {}, {}, components, case)
        assert not components["forward"]["member_output_addresses_unique"]
        assert not components["forward"]["members"]["0"]["different_buffer"]
        self.checks["actual_output_alias_rejected"] = True

    def protocols(self):
        SamplingProtocol.validate(SamplingProtocol.VALUE)
        SamplingProtocol.validate({**SamplingProtocol.VALUE, "cache_mode": "rotating_inputs"})
        self.require_rejection("queued_sync_protocol_rejected", lambda: SamplingProtocol.validate(
            {**SamplingProtocol.VALUE, "cache_mode": "rotating_inputs", "sync": "queued"}))
        self.checks["only_cache_mode_can_change"] = True

    def run(self):
        self.storage_and_sequence()
        self.adversarial_members()
        self.protocols()
        result = {"classification": "CPU_SYNTHETIC_NOT_PERFORMANCE", "status": "PASS", "checks": self.checks,
                  "gpu_used": False, "no_performance_conclusion": True}
        Evidence.write(self.directory / "result.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    print(RingSelfTest(args.directory).run())
