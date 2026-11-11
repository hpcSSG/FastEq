"""Concatenate verified shared References; never introduce reference arithmetic."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from cases import Case, References
from common import Evidence
from reference_store import ReferenceStore, StoredReferences


class RingReferences(References):
    """Forward-only References interface plus the individual member baselines."""

    def __init__(self, case, members, payload, receipt):
        self.case, self.members = case, members
        self.values = payload["values"]
        self.norms = payload["norms"]
        self.torch_errors = payload["torch_errors"]
        self.hashes = payload["hashes"]
        self.store_receipt = receipt


class RingReferenceStore:
    FORMATS = ReferenceStore.FORMATS

    def __init__(self, root, canonical_reference_root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.member_store = ReferenceStore(canonical_reference_root)

    @staticmethod
    def stable_receipt(receipt):
        return {name: receipt[name] for name in ("key", "path", "sha256", "data_path", "data_sha256", "identity", "value_hashes")}

    @classmethod
    def hashes(cls, values):
        return {format_name: {"forward": Case.tensor_hash(values[format_name]["forward"])}
                for format_name in cls.FORMATS}

    def member(self, case):
        # ReferenceStore owns both arithmetic and immutable serialization. Drop its
        # backward tensors immediately; this supplementary workload is forward-only.
        original = self.member_store.get(case)
        payload = {"values": {name: {"forward": original.values[name]["forward"]} for name in self.FORMATS},
                   "norms": {"forward": original.norms["forward"]},
                   "torch_errors": {"forward": original.torch_errors["forward"]},
                   "hashes": {"forward": original.hashes["forward"]}}
        return StoredReferences(case, payload, original.store_receipt)

    def get(self, dataset):
        import torch

        dataset.verify()
        members = [self.member(member) for member in dataset.members]
        first_identity = members[0].store_receipt["identity"]
        identity = {"schema": 1, "kind": "ordered_forward_reference_concatenation",
                    "input_hash": dataset.input_hash, "ring_manifest": str(dataset.manifest_path),
                    "ring_manifest_sha256": dataset.manifest_sha256,
                    "ordered_member_receipts": [self.stable_receipt(reference.store_receipt) for reference in members],
                    "producer_sha256": Evidence.sha256(Path(__file__)),
                    "math": first_identity["math"], "hardware": first_identity["hardware"],
                    "torch": first_identity["torch"], "torch_cuda": first_identity["torch_cuda"],
                    "point": dataset.case.point.columns(),
                    "aggregation": "torch.cat of stored forward outputs in serialized ring order; no additional contractions"}
        key = Evidence.fingerprint(identity)
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        data_path, metadata_path = directory / "references.pt", directory / "receipt.json"
        expected_hashes = self.hashes({name: {"forward": torch.cat([member.values[name]["forward"] for member in members])}
                                      for name in self.FORMATS})
        with (directory / "references.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if data_path.exists() != metadata_path.exists():
                raise RuntimeError("Partial ring reference cache; preserve and inspect: " + str(directory))
            if data_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("identity") != identity or metadata.get("key") != key:
                    raise ValueError("Ring reference identity differs")
                if Evidence.sha256(data_path) != metadata["file_sha256"]:
                    raise ValueError("Ring reference file changed")
                payload = torch.load(data_path, map_location="cpu", weights_only=True)
                created = False
            else:
                if any(directory.glob("*.partial*")):
                    raise RuntimeError("Interrupted ring reference cache is preserved")
                values = {name: {"forward": torch.cat([member.values[name]["forward"].detach().cpu() for member in members])}
                          for name in self.FORMATS}
                norm = values["fp64"]["forward"].norm().item()
                # Reuse References.error for aggregate metadata as well as the
                # unmodified common Accuracy gate used later by CallValidation.
                temporary_reference = RingReferences(dataset.case, members,
                    {"values": values, "norms": {"forward": norm}, "torch_errors": {}, "hashes": {}}, {})
                payload = {"values": values, "norms": {"forward": norm},
                           "torch_errors": {"forward": temporary_reference.error(values["torch"]["forward"], "forward")},
                           "hashes": expected_hashes["fp64"]}
                metadata = {"key": key, "identity": identity, "value_hashes": expected_hashes,
                            "producer": "RingReferenceStore concatenates original ReferenceStore forward tensors; no alternate Gold"}
                temporary = directory / f"references.pt.partial.{os.getpid()}"
                torch.save(payload, temporary)
                temporary.replace(data_path)
                metadata["file_sha256"] = Evidence.sha256(data_path)
                Evidence.write(metadata_path, metadata)
                created = True
            if (self.hashes(payload["values"]) != expected_hashes or metadata["value_hashes"] != expected_hashes
                    or payload["hashes"] != expected_hashes["fp64"]):
                raise ValueError("Ring reference concatenation hashes differ from stored members")
            values = {name: {"forward": payload["values"][name]["forward"].to(dataset.case.x.device)} for name in self.FORMATS}
            if self.hashes(values) != expected_hashes:
                raise ValueError("Ring reference device transfer changed content")
            receipt = {"key": key, "path": str(metadata_path), "sha256": Evidence.sha256(metadata_path),
                       "data_path": str(data_path), "data_sha256": metadata["file_sha256"], "created": created,
                       "identity": identity, "value_hashes": expected_hashes,
                       "observed_gpu_uuid_raw": members[0].store_receipt.get("observed_gpu_uuid_raw"),
                       "gpu_uuid_normalized": first_identity["hardware"].get("uuid")}
            return RingReferences(dataset.case, members, {**payload, "values": values}, receipt)
