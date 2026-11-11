"""Content-verified canonical References cache shared by all H20 adapters."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path

from cases import Case, References
from common import Evidence


class StoredReferences(References):
    """The References interface with already computed, verified tensor values."""

    def __init__(self, case, payload, receipt):
        self.case = case
        self.values = payload["values"]
        self.norms = payload["norms"]
        self.torch_errors = payload["torch_errors"]
        self.hashes = payload["hashes"]
        self.store_receipt = receipt


class ReferenceStore:
    """Only cases.References performs mathematics; this class owns storage."""

    FORMATS = ("fp64", "torch")
    RESULTS = ("forward", "grad_x", "grad_w")
    MATH = {"float32_matmul_precision": "highest", "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False, "arithmetic": "ordered x@w, then cg, then run-length reduction"}

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def device_uuid(value):
        return str(value).lower().removeprefix("gpu-")

    @classmethod
    def identity(cls, case):
        import torch

        device = case.x.device
        hardware = {"type": device.type}
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            hardware.update(uuid=cls.device_uuid(props.uuid), architecture=[props.major, props.minor],
                            name=props.name, sm_count=props.multi_processor_count)
        return {"schema": 1, "input_hash": case.input_hash, "tensor_hashes": case.hashes,
                "layouts": case.layouts(), "point": case.point.columns(),
                "semantics": {"dims": case.dims, "cg": case.cg, "U": case.u,
                              "V": case.w3.shape[-1], "J": case.j, "subscripts": "uv,iu,iv"},
                "reference_producer_sha256": Evidence.sha256(Path(__file__).with_name("cases.py")),
                "store_producer_sha256": Evidence.sha256(Path(__file__)),
                "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "torch_origin": torch.__file__, "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
                "math": cls.MATH, "hardware": hardware,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}

    @staticmethod
    def tensor_hashes(values):
        return {format_name: {result: Case.tensor_hash(values[format_name][result])
                              for result in ReferenceStore.RESULTS}
                for format_name in ReferenceStore.FORMATS}

    @classmethod
    def verify_values(cls, case, payload, metadata):
        import torch

        if set(payload) != {"values", "norms", "torch_errors", "hashes"}:
            raise ValueError("Reference payload schema differs")
        if set(payload["values"]) != set(cls.FORMATS):
            raise ValueError("Reference formats differ")
        shapes = {"forward": tuple(case.grad.shape), "grad_x": tuple(case.x.shape), "grad_w": tuple(case.w.shape)}
        for format_name in cls.FORMATS:
            if set(payload["values"][format_name]) != set(cls.RESULTS):
                raise ValueError("Reference result set differs")
            dtype = torch.float64 if format_name == "fp64" else case.dtype
            for result in cls.RESULTS:
                value = payload["values"][format_name][result]
                if tuple(value.shape) != shapes[result] or value.dtype != dtype:
                    raise ValueError("Reference shape/dtype differs")
        hashes = cls.tensor_hashes(payload["values"])
        if hashes != metadata["value_hashes"] or payload["hashes"] != hashes["fp64"]:
            raise ValueError("Reference tensor content hash differs")
        for field in ("norms", "torch_errors", "hashes"):
            if set(payload[field]) != set(cls.RESULTS):
                raise ValueError("Reference scalar/hash result set differs")
        for field in ("norms", "torch_errors"):
            if any(not math.isfinite(value) or value < 0 for value in payload[field].values()):
                raise ValueError("Nonfinite or negative reference metadata")

    def get(self, case):
        import torch

        identity = self.identity(case)
        observed_gpu_uuid_raw = str(torch.cuda.get_device_properties(case.x.device).uuid) if case.x.is_cuda else None
        key = Evidence.fingerprint(identity)
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        data_path, metadata_path = directory / "references.pt", directory / "receipt.json"
        with (directory / "references.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if data_path.exists() != metadata_path.exists():
                raise RuntimeError("Partial reference cache; preserve and inspect: " + str(directory))
            if data_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("identity") != identity or metadata.get("key") != key:
                    raise ValueError("Reference cache mixed input/source/GPU/math identity")
                if Evidence.sha256(data_path) != metadata["file_sha256"]:
                    raise ValueError("Reference file content hash differs")
                payload = torch.load(data_path, map_location="cpu", weights_only=True)
                self.verify_values(case, payload, metadata)
                created = False
            else:
                if any(directory.glob("*.partial*")):
                    raise RuntimeError("Interrupted reference cache is preserved: " + str(directory))
                reference = References(case)
                payload = {"values": {format_name: {result: value.detach().cpu()
                                                    for result, value in reference.values[format_name].items()}
                                      for format_name in self.FORMATS},
                           "norms": reference.norms, "torch_errors": reference.torch_errors, "hashes": reference.hashes}
                metadata = {"identity": identity, "key": key, "value_hashes": self.tensor_hashes(payload["values"]),
                            "observed_gpu_uuid_raw": observed_gpu_uuid_raw,
                            "producer": "cases.References; no alternate reference arithmetic"}
                self.verify_values(case, payload, metadata)
                for result in self.RESULTS:
                    if (payload["values"]["torch"][result].dtype == torch.float64
                            and metadata["value_hashes"]["torch"][result] == metadata["value_hashes"]["fp64"][result]):
                        payload["values"]["torch"][result] = payload["values"]["fp64"][result]
                temporary = directory / f"references.pt.partial.{os.getpid()}"
                torch.save(payload, temporary)
                temporary.replace(data_path)
                metadata["file_sha256"] = Evidence.sha256(data_path)
                Evidence.write(metadata_path, metadata)
                created = True
            values = {format_name: {result: value.to(case.x.device)
                                    for result, value in payload["values"][format_name].items()}
                      for format_name in self.FORMATS}
            for result in self.RESULTS:
                if (values["torch"][result].dtype == torch.float64
                        and metadata["value_hashes"]["torch"][result] == metadata["value_hashes"]["fp64"][result]):
                    values["torch"][result] = values["fp64"][result]
            restored = {**payload, "values": values}
            # Verify transferred bytes too; all versions consume one actual gold.
            if self.tensor_hashes(values) != metadata["value_hashes"]:
                raise ValueError("Reference device transfer changed tensor bytes")
            receipt = {"key": key, "path": str(metadata_path), "sha256": Evidence.sha256(metadata_path),
                       "data_path": str(data_path), "data_sha256": metadata["file_sha256"],
                       "created": created, "identity": identity, "value_hashes": metadata["value_hashes"],
                       "observed_gpu_uuid_raw": observed_gpu_uuid_raw,
                       "gpu_uuid_normalized": identity["hardware"].get("uuid")}
            return StoredReferences(case, restored, receipt)
