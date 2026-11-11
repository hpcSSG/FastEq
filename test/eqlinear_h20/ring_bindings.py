"""Representative rotating-input public-forward calls under the common sampler."""
from __future__ import annotations

import json
import os
from pathlib import Path

from cases import Case
from common import Evidence
from current import CurrentAdapter
from performance_bindings import BoundCall, QualityMember, TorchCalls, TorchIdentity
from ring_fixtures import RingDataset, RingSpecification
from ring_references import RingReferenceStore


class RingCall:
    """One shared cursor step and output retention policy for every candidate."""
    def __init__(self, call, dataset, references):
        if (call.operation, call.boundary, call.measurement_work) != ("forward", "public_warm", "full_operation"):
            raise ValueError("Ring public-forward wrapper cannot relabel prepared-layout/component work")
        self.original, self.dataset, self.references = call, dataset, references
        self.outputs = [None] * len(dataset.members)
        declaration = [self.member_identity(index) for index in range(len(dataset.members))]
        binding = {**call.binding, "quality_member_identities": {"forward": declaration},
                   "ring_call_contract": {"input_hash": dataset.input_hash,
                       "buffer_count": len(dataset.members), "public_original_layout": True,
                       "sequence": "start=(group*calls_per_group)%buffer_count; advance one modulo buffer_count per call; same start for every candidate and round",
                       "selection_cost": "shared Python cursor/index/retained-output assignment included in timed function",
                       "validation_sync": "device synchronization after every individual member public call",
                       "allocator_policy": dataset.specification.columns()["allocator_policy"]}}
        self.call = BoundCall(call.variant, call.operation, call.boundary, self.run, self.check,
            lambda raw: raw["aggregate"], binding,
            call.description + "; ordered rotating original-layout inputs, constant canonical weight, retained independent output slots",
            warm_function=self.run, measurement_work=call.measurement_work,
            before_sample=self.before_sample, quality_members=self.quality_members)

    def member_identity(self, index):
        return {"label": f"ring{index:04d}", "index": index,
                "input_hash": self.dataset.members[index].input_hash,
                "reference_hash": self.references.members[index].hashes["forward"]}

    def run(self):
        slot = self.dataset.cursor
        output = self.original.function()
        self.outputs[slot] = output
        self.dataset.advance()
        return output

    def before_sample(self, group, count, round_index):
        self.dataset.reset(group * count)

    def check(self):
        import torch

        self.dataset.reset()
        outputs = []
        for index in range(len(self.dataset.members)):
            output = self.original.check_function()
            if self.dataset.case.x.is_cuda:
                torch.cuda.synchronize()
            # Keep the real public output, rather than a clone used as a buffer proof.
            outputs.append(output)
            self.outputs[index] = output
            self.dataset.advance()
        return {"aggregate": torch.cat([self.original.normalize(output) for output in outputs]), "members": outputs}

    def quality_members(self, first_raw, second_raw):
        if len(first_raw["members"]) != len(self.dataset.members) or len(second_raw["members"]) != len(self.dataset.members):
            raise ValueError("A complete ordered ring is required for quality qualification")
        return {"forward": [QualityMember(self.original.normalize(first), self.original.normalize(second),
                    reference.values["fp64"]["forward"], reference.torch_errors["forward"], **self.member_identity(index))
                for index, (first, second, reference) in enumerate(zip(first_raw["members"], second_raw["members"], self.references.members))]}


class RotatingInputFamily:
    """Current/P0/P14 plus IEEE Torch, retaining the full hot-buffer plan elsewhere."""
    CACHE_MODE = "rotating_inputs"
    FILES = ("ring_bindings.py", "ring_fixtures.py", "ring_references.py", "qualification.py")

    @staticmethod
    def accept_point(family, point):
        return family in ("current", "p0", "p14") and (point.config, point.batch) in (("SMALL_P1", 512), ("LARGE_P3", 5888))

    @staticmethod
    def expected_calls(family):
        if family not in ("current", "p0", "p14"):
            raise ValueError("This first ring factory supports current/P0/P14; other original-layout adapters may reuse RingCall")
        native = "current_fasteq" if family == "current" else "historical_" + family
        return sorted([f"{native}/forward/public_warm", "torch_ieee_full/forward/public_warm"])

    @staticmethod
    def hardware_snapshot(session):
        return session.root / "config" / "rotating_input_hardware.json"

    @classmethod
    def configure_scope(cls, session):
        snapshot = cls.hardware_snapshot(session)
        if not snapshot.exists():
            source = session.root / "roofline" / "hardware.json"
            Evidence.write(snapshot, {"source_path": str(source), "source_sha256_at_selection": Evidence.sha256(source),
                                      "hardware": json.loads(source.read_text())})
        value = json.loads(snapshot.read_text())
        attrs = value["hardware"]["h20"]["observed_device_attributes"]
        raw_l2_bytes = attrs["device__attribute_l2_cache_size"]
        if not isinstance(raw_l2_bytes, (int, str)) or not str(raw_l2_bytes).isdigit():
            raise ValueError("Measured positive H20 L2 attribute is required")
        l2_bytes = int(raw_l2_bytes)
        if l2_bytes <= 0:
            raise ValueError("Measured positive H20 L2 attribute is required")
        return {"kind": "supplementary_representative_rotating_inputs", "l2_bytes": l2_bytes,
                "target_input_read_working_set_l2_multiple": 3, "minimum_buffers": 2,
                "points": [f"{config}/{batch}/{precision}" for config, batch in (("SMALL_P1", 512), ("LARGE_P3", 5888))
                           for precision in ("tf32", "fp32", "fp64")],
                "operations": ["forward/public_warm"], "canonical75_unchanged": True,
                "evidence_files": {str(snapshot): Evidence.sha256(snapshot)},
                "traffic_interpretation": "Resident unique input bytes determine ring size; logical bytes are never labeled physical HBM/L2 traffic"}

    @classmethod
    def verify_scope(cls, scope, session):
        if cls.configure_scope(session) != scope:
            raise ValueError("Frozen rotating-input scope differs")

    @staticmethod
    def allocator_state():
        import torch

        state = {"PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                 "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF")}
        if torch.cuda.is_initialized():
            state.update(backend=torch.cuda.memory.get_allocator_backend(),
                         allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
        return state

    @classmethod
    def prepare_workload(cls, session, point):
        scope = cls.configure_scope(session)
        canonical = Case(point, fixture_root=session.root / "fixtures")
        specification = RingSpecification.for_case(canonical, scope["l2_bytes"])
        snapshot = cls.hardware_snapshot(session)
        hardware = {"snapshot": str(snapshot), "sha256": Evidence.sha256(snapshot),
                    "attribute": "h20.observed_device_attributes.device__attribute_l2_cache_size", "value": scope["l2_bytes"]}
        dataset = RingDataset.materialize(canonical, session.root / "fixtures" / "rotating_inputs", specification, hardware)
        references = RingReferenceStore(session.root / "references" / "rotating_inputs", session.root / "references").get(dataset)
        identity = {"kind": "rotating_inputs", "input_hash": dataset.input_hash,
                    "manifest": str(dataset.manifest_path), "manifest_sha256": dataset.manifest_sha256,
                    "ordered_input_hashes": [member.input_hash for member in dataset.members],
                    "ordered_reference_hashes": [member.hashes for member in references.members],
                    "fixed_w_sha256": canonical.hashes["w_sha256"], "specification": specification.columns(),
                    "reference_receipt": RingReferenceStore.stable_receipt(references.store_receipt),
                    "traffic_interpretation": scope["traffic_interpretation"]}
        dataset.references, dataset.workload_identity = references, identity
        dataset.workload_setup_metadata = {"allocator_state_after_workload_preparation": cls.allocator_state()}
        return {"case": dataset.case, "references": references, "identity": identity, "cache_mode": cls.CACHE_MODE}

    def __init__(self, case, session, family="current"):
        import torch

        self.case, self.dataset = case, case.dataset
        self.expected_calls(family)
        if family == "current":
            self.native, native_name = CurrentAdapter(case, session), "current_fasteq"
        else:
            from historical import HistoricalAdapter
            self.native, native_name = HistoricalAdapter(case, session, family), "historical_" + family
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = TorchCalls(case)
        self.native.prepare("forward", "public_warm")
        torch.cuda.synchronize()
        native_binding = self.native.identity()
        native_binding["production_source_hash"] = native_binding["source_hash"]
        adapters = ("ring_bindings.py", "ring_fixtures.py", "ring_references.py", "performance_bindings.py", "current.py")
        if family != "current":
            adapters += ("historical.py", "registry.py")
        native_binding["sampler_adapter_hashes"] = {name: Evidence.sha256(Path(__file__).with_name(name)) for name in adapters}
        native_binding["source_hash"] = Evidence.fingerprint({"production": native_binding["production_source_hash"],
                                                             "adapters": native_binding["sampler_adapter_hashes"]})
        native_binding["accuracy_tier"] = case.point.precision
        torch_binding = TorchIdentity.capture(session)
        torch_binding["source_hash"] = Evidence.fingerprint({name: Evidence.sha256(Path(__file__).with_name(name))
            for name in ("ring_bindings.py", "ring_fixtures.py", "ring_references.py", "performance_bindings.py", "cases.py")})
        torch_binding["accuracy_tier"] = "fp64" if case.point.precision == "fp64" else "fp32"
        originals = [BoundCall(native_name, "forward", "public_warm",
                        lambda: self.native.call("forward", "public_warm"),
                        lambda: self.native.call("forward", "public_warm", fresh_output=True), lambda value: value,
                        native_binding, "original public API, output allocation included"),
                     BoundCall("torch_ieee_full", "forward", "public_warm", lambda: self.torch.full("forward"),
                        lambda: self.torch.full("forward"), lambda value: value, torch_binding,
                        "IEEE eager full path on original layouts, including views/mm/cg/reduction/output allocation")]
        self.wrappers = [RingCall(call, self.dataset, self.dataset.references) for call in originals]
        self.calls = [wrapper.call for wrapper in self.wrappers]
        if sorted(call.key for call in self.calls) != self.expected_calls(family):
            raise ValueError("Rotating-input call scope differs")
        self.setup_metadata = {**self.dataset.workload_setup_metadata, "ring_manifest": str(self.dataset.manifest_path),
                               "retained_output_slots_per_candidate": len(self.dataset.members),
                               "all_member_preparation": "Both complete-ring common quality passes run and synchronize every input before adaptive warmup and pilot/timing"}
        self.setup_metadata["allocator_state_after_family_preparation"] = self.allocator_state()

    def close(self):
        self.dataset.verify()
        self.native.close()
        for wrapper in self.wrappers:
            wrapper.outputs.clear()
