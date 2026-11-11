"""Shared serialized rotating inputs with fixed canonical weights and explicit seeds."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from cases import Case
from common import Evidence


@dataclass(frozen=True)
class RingSpecification:
    l2_bytes: int
    target_multiple: int
    input_bytes_per_buffer: int
    output_bytes_per_buffer: int
    buffer_count: int
    minimum_buffers: int = 2
    seed_stride: int = 1_000_003

    @classmethod
    def for_case(cls, case, l2_bytes, target_multiple=3):
        if l2_bytes <= 0 or target_multiple < 2:
            raise ValueError("Ring sizing requires a positive measured L2 and >=2x target")
        input_bytes = case.x.numel() * case.x.element_size()
        output_bytes = case.grad.numel() * case.grad.element_size()
        count = max(2, math.ceil(target_multiple*l2_bytes/input_bytes))
        return cls(int(l2_bytes), int(target_multiple), input_bytes, output_bytes, count)

    def seed(self, canonical_seed, index):
        if not 0 <= index < self.buffer_count:
            raise IndexError("Ring member is outside the declared buffer count")
        return canonical_seed + self.seed_stride*index

    def columns(self):
        return {**asdict(self), "resident_input_bytes": self.buffer_count*self.input_bytes_per_buffer,
            "logical_retained_output_bytes": self.buffer_count*self.output_bytes_per_buffer,
            "resident_input_l2_multiple": self.buffer_count*self.input_bytes_per_buffer/self.l2_bytes,
            "sizing_rule": "max(2, ceil(target_multiple*measured_L2/input_bytes)); actual input set is the lower-bound working set",
            "granularity_note": "A single large input can exceed the target; preserve at least2 distinct buffers and report the actual larger multiple",
            "allocator_policy": {"api_allocated_outputs": True, "retained_output_slots_per_candidate": self.buffer_count,
                "replacement": "retain each completed output until the same ring slot is replaced; new allocation exists before old slot release",
                "shared_wrapper_cost": "Python cursor/index/retention inside every timed candidate call"}}


class RingMemberCase(Case):
    """Reuse Case/FixtureStore, changing only an explicitly described input generator."""
    def __init__(self, canonical, index, specification, directory, device):
        self.canonical, self.ring_index, self.specification = canonical, index, specification
        self.generator_spec = {"schema": 1, "kind": "canonical_copy" if index == 0 else "cpu_generator_randn",
            "ring_index": index, "actual_seed": None if index == 0 else specification.seed(canonical.point.seed, index),
            "nominal_point_seed": canonical.point.seed, "seed_stride": specification.seed_stride,
            "actual_rng_device": "none_canonical_serialized_copy" if index == 0 else "cpu",
            "outer_generator_device_interpretation": "FixtureStore generator_device is the Case.draw destination; actual x RNG device is specified here",
            "canonical_input_hash": canonical.input_hash, "fixed_w_sha256": canonical.hashes["w_sha256"],
            "fixed_auxiliary_grad_sha256": canonical.hashes["grad_sha256"],
            "draw_order": "x only; reuse serialized canonical w and auxiliary grad; canonical member0 reuses x too"}
        super().__init__(canonical.point, device=device, fixture_root=directory)
        metadata_path = Path(directory) / (self.point.key.replace("/", "_") + ".json")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("generator") != self.generator_spec:
            raise ValueError("Ring generator metadata is absent/different; no seed-only comparability claim")
        if self.hashes["w_sha256"] != canonical.hashes["w_sha256"] or self.hashes["grad_sha256"] != canonical.hashes["grad_sha256"]:
            raise ValueError("Ring changed canonical weights or auxiliary gradient")
        # All candidates and all members share the same physical constant-weight tensor.
        self.w, self.w3 = canonical.w, canonical.w3
        self.grad, self.grad3 = canonical.grad, canonical.grad3
        self.fixture_receipt = {"metadata_path": str(metadata_path), "metadata_sha256": Evidence.sha256(metadata_path),
            "data_path": str(metadata_path.with_suffix(".pt")), "data_sha256": metadata["file_sha256"]}

    def draw(self, device):
        import torch

        if self.ring_index == 0:
            self.x = self.canonical.x.to(device)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.generator_spec["actual_seed"])
            self.x = torch.randn(self.canonical.x.shape, generator=generator, dtype=self.dtype, device="cpu").to(device)
        self.w = self.canonical.w.to(device)
        self.grad = self.canonical.grad.to(device)


class RingDataset:
    def __init__(self, canonical, members, specification, manifest_path):
        self.canonical, self.members, self.specification = canonical, members, specification
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.manifest_sha256 = Evidence.sha256(self.manifest_path)
        self.input_hash = self.manifest["input_hash"]
        self.cursor = 0
        self.case = RingCaseView(self)

    @classmethod
    def materialize(cls, canonical, root, specification, hardware_evidence, device=None):
        import torch

        device = canonical.x.device if device is None else device
        root = Path(root)
        definition = {"point": canonical.point.columns(), "canonical_input_hash": canonical.input_hash,
            "specification": specification.columns(), "hardware_evidence": hardware_evidence,
            "generator_torch_version": torch.__version__}
        directory = root / Evidence.fingerprint(definition)
        directory.mkdir(parents=True, exist_ok=True)
        members = [RingMemberCase(canonical, index, specification, directory / f"member{index:04d}", device)
                   for index in range(specification.buffer_count)]
        if len({case.hashes["x_sha256"] for case in members}) != specification.buffer_count:
            raise ValueError("Ring inputs are not actually distinct")
        if len({case.x.data_ptr() for case in members}) != specification.buffer_count:
            raise ValueError("Ring inputs alias the same storage")
        manifest = {"classification": "ROTATING_INPUT_FIXTURE_NOT_TIMING", "definition": definition,
            "ordered_members": [{"index": index, "input_hash": case.input_hash, "tensor_hashes": case.hashes,
                                 "layouts": case.layouts(), "generator": case.generator_spec, "files": case.fixture_receipt}
                                for index, case in enumerate(members)],
            "weight_state": "constant", "fixed_w_sha256": canonical.hashes["w_sha256"],
            "input_hash": Evidence.fingerprint({"definition": definition, "ordered_member_hashes": [case.input_hash for case in members]})}
        path = directory / "ring.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError("Existing ring manifest differs; preserve it and use another epoch")
        else:
            Evidence.write(path, manifest)
        return cls(canonical, members, specification, path)

    def reset(self, index=0):
        self.cursor = int(index) % len(self.members)

    def advance(self):
        self.cursor = (self.cursor + 1) % len(self.members)

    def verify(self):
        if Evidence.sha256(self.manifest_path) != self.manifest_sha256:
            raise ValueError("Ring manifest changed")
        if len(self.members) != self.specification.buffer_count:
            raise ValueError("Ring member count changed")
        for member, declared in zip(self.members, self.manifest["ordered_members"]):
            actual = {name + "_sha256": Case.tensor_hash(getattr(member, name)) for name in ("x", "w", "grad")}
            if any(actual[key] != declared["tensor_hashes"][key] for key in actual):
                raise ValueError("Ring tensor content changed during the experiment")
        if len({member.w.data_ptr() for member in self.members}) != 1:
            raise ValueError("Ring weights no longer share one physical tensor")


class RingCaseView:
    """The public adapters see one selected input and a constant descriptor/weight."""
    def __init__(self, dataset):
        self.dataset = dataset

    @property
    def x(self):
        return self.dataset.members[self.dataset.cursor].x

    @property
    def x3(self):
        return self.dataset.members[self.dataset.cursor].x3

    @property
    def input_hash(self):
        return self.dataset.input_hash

    @property
    def hashes(self):
        return {"ring_sha256": self.dataset.input_hash, "w_sha256": self.dataset.canonical.hashes["w_sha256"],
                "descriptor_sha256": self.dataset.canonical.hashes["descriptor_sha256"]}

    def __getattr__(self, key):
        return getattr(self.dataset.canonical, key)
