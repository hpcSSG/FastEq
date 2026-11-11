"""Fixed real-descriptor workload, exact input hashes, and independent Torch references."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
import fcntl

from common import Evidence

CONFIGS = {
    "SMALL_P1": (96, [1, 3, 5, 7]),
    "MEDIUM_P1": (128, [1, 3, 5, 7]),
    "MEDIUM_P2": (128, [1, 1, 3, 3, 3, 5, 5, 5, 7, 7]),
    "LARGE_P1": (224, [1, 3, 5, 7]),
    "LARGE_P3": (224, [1, 1, 1, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 7, 7, 7, 7]),
}
BATCHES = (512, 1024, 2048, 2944, 5888)
PRECISIONS = ("tf32", "fp32", "fp64")
OPERATIONS = ("public_forward", "public_grad_x", "run_forward", "run_grad_x")


@dataclass(frozen=True)
class Point:
    config: str
    batch: int
    precision: str
    config_index: int
    batch_index: int

    @property
    def seed(self):
        return 1000 + self.config_index * 15 + self.batch_index * 3 + PRECISIONS.index(self.precision)

    @property
    def key(self):
        return f"{self.config}/{self.batch}/{self.precision}"

    def columns(self):
        return {"config": self.config, "batch": self.batch, "precision": self.precision, "seed": self.seed}


def points():
    return [Point(name, batch, precision, ci, bi)
            for ci, name in enumerate(CONFIGS)
            for bi, batch in enumerate(BATCHES)
            for precision in PRECISIONS]


class Case:
    def __init__(self, point, device="cuda", fixture_root=None):
        import torch

        self.point = point
        self.u, self.dims = CONFIGS[point.config]
        self.cg = [.01 + .001 * p for p in range(len(self.dims))]
        self.j = sum(value for index, value in enumerate(self.dims)
                     if index == 0 or value != self.dims[index - 1])
        self.precision_id = PRECISIONS.index(point.precision)
        self.dtype = torch.float64 if point.precision == "fp64" else torch.float32
        self.device = device
        if fixture_root is None:
            self.draw(device)
        else:
            FixtureStore(Path(fixture_root)).populate(self, device)
        self.x3 = self.x.reshape(point.batch, sum(self.dims), self.u)
        self.w3 = self.w.reshape(len(self.dims), self.u, self.u)
        self.grad3 = self.grad.reshape(point.batch, self.j, self.u)
        self.descriptor = self.make_descriptor()
        self.hashes = {name + "_sha256": self.tensor_hash(getattr(self, name))
                       for name in ("x", "w", "grad")}
        semantics = {"dims": self.dims, "cg": self.cg, "u": self.u, "j": self.j,
                     "subscripts": "uv,iu,iv"}
        self.hashes["descriptor_sha256"] = hashlib.sha256(
            json.dumps(semantics, sort_keys=True).encode()).hexdigest()
        self.input_hash = Evidence.fingerprint({"hashes": self.hashes, "layouts": self.layouts()})

    def draw(self, device):
        import torch

        torch.manual_seed(self.point.seed)
        self.x = torch.randn(self.point.batch, sum(self.dims) * self.u, device=device, dtype=self.dtype)
        self.w = torch.randn(len(self.dims) * self.u * self.u, device=device, dtype=self.dtype)
        self.grad = torch.randn(self.point.batch, self.j * self.u, device=device, dtype=self.dtype)

    def layouts(self):
        return {name: {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype)}
                for name in ("x", "w", "grad") for value in (getattr(self, name),)}

    def make_descriptor(self):
        import cuequivariance as cue
        import numpy as np

        key = (self.u, tuple(self.dims), tuple(self.cg))
        if key in DescriptorPool.values:
            return DescriptorPool.values[key]
        descriptor = cue.SegmentedTensorProduct.from_subscripts("uv,iu,iv")
        groups = []
        output_indices = []
        for dim in self.dims:
            descriptor.add_segment(0, (self.u, self.u))
            descriptor.add_segment(1, (dim, self.u))
            if not groups or groups[-1] != dim:
                descriptor.add_segment(2, (dim, self.u))
                groups.append(dim)
            output_indices.append(len(groups) - 1)
        for p, (coefficient, output_index) in enumerate(zip(self.cg, output_indices)):
            descriptor.add_path(p, p, output_index, c=np.asarray(coefficient, dtype=np.float64))
        # Keep actual descriptor identity stable across batches/rounds/variants.
        descriptor.get_dimensions_dict()
        DescriptorPool.values[key] = descriptor
        return descriptor

    @staticmethod
    def tensor_hash(tensor):
        import torch

        digest = hashlib.sha256()
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        for start in range(0, raw.numel(), 16 << 20):
            digest.update(raw[start:start + (16 << 20)].cpu().numpy())
        return digest.hexdigest()


class References:
    """Both forward and autograd use the historical x@w, then multiply cg arithmetic."""

    def __init__(self, case):
        import torch

        self.case = case
        self.values = {}
        with self.ieee_math():
            for name, dtype in (("fp64", torch.float64), ("torch", case.dtype)):
                with torch.enable_grad():
                    x = case.x3.detach().to(dtype).requires_grad_(True)
                    w = case.w3.detach().to(dtype).requires_grad_(True)
                    graph = self.forward(x, w, case.dims, case.cg, dtype, pre_cg=False)
                    grad_x, grad_w = torch.autograd.grad(graph, (x, w), case.grad3.to(dtype))
                self.values[name] = {"forward": graph.detach().reshape(case.point.batch, -1),
                                     "grad_x": grad_x.detach().reshape(case.point.batch, -1),
                                     "grad_w": grad_w.detach().reshape(-1)}
        self.norms = {kind: reference.norm().item() for kind, reference in self.values["fp64"].items()}
        self.torch_errors = {kind: self.error(value, kind) for kind, value in self.values["torch"].items()}
        self.hashes = {kind: case.tensor_hash(value) for kind, value in self.values["fp64"].items()}

    @staticmethod
    @contextmanager
    def ieee_math():
        import torch

        previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(previous[0])
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[1:]

    @staticmethod
    def forward(x, w, dims, cg, dtype, pre_cg):
        import torch

        x, w = x.to(dtype), w.to(dtype)
        groups = []
        input_offset = 0
        p = 0
        while p < len(dims):
            dim = dims[p]
            q = p + 1
            while q < len(dims) and dims[q] == dim:
                q += 1
            rows = []
            for i in range(dim):
                value = None
                for index in range(p, q):
                    weight = w[index] * float(cg[index]) if pre_cg else w[index]
                    term = x[:, input_offset + (index - p) * dim + i, :] @ weight
                    if not pre_cg:
                        term = term * float(cg[index])
                    value = term if value is None else value + term
                rows.append(value)
            groups.append(torch.stack(rows, dim=1))
            input_offset += (q - p) * dim
            p = q
        return torch.cat(groups, dim=1)

    def error(self, output, kind):
        norm = self.norms[kind]
        error = (output.double().reshape_as(self.values["fp64"][kind]) - self.values["fp64"][kind]).norm().item()
        return error / norm if norm else (0.0 if error == 0 else float("inf"))

    def limit(self, precision, kind):
        if precision == "fp64":
            return 1e-12
        return 2e-5 if precision == "tf32" else 1.05 * self.torch_errors[kind]


class DescriptorPool:
    values = {}


class FixtureStore:
    """Canonical tensor bytes, serialized once; reject corrupt or different inputs."""

    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def populate(self, case, device):
        import torch

        stem = case.point.key.replace("/", "_")
        tensor_path = self.root / (stem + ".pt")
        metadata_path = self.root / (stem + ".json")
        with (self.root / (stem + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if tensor_path.exists() != metadata_path.exists():
                raise RuntimeError(f"Partial fixture: {stem}; preserve and inspect before retry")
            if tensor_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata["file_sha256"] != Evidence.sha256(tensor_path) or metadata["point"] != case.point.columns():
                    raise RuntimeError(f"Fixture identity mismatch: {stem}")
                values = torch.load(tensor_path, map_location="cpu", weights_only=True)
                hashes = {name: case.tensor_hash(value) for name, value in values.items()}
                if hashes != metadata["tensor_hashes"]:
                    raise RuntimeError(f"Fixture tensor hash mismatch: {stem}")
                for name, value in values.items():
                    setattr(case, name, value.to(device))
            else:
                case.draw(device)
                values = {name: getattr(case, name).detach().cpu() for name in ("x", "w", "grad")}
                temporary = tensor_path.with_suffix(".partial")
                torch.save(values, temporary)
                temporary.replace(tensor_path)
                Evidence.write(metadata_path, {"schema": 1, "point": case.point.columns(),
                    "generator": getattr(case, "generator_spec", "torch.manual_seed; randn x,w,grad; generated on recorded device"),
                    "torch_version": torch.__version__, "generator_device": str(device),
                    "tensor_hashes": {name: case.tensor_hash(value) for name, value in values.items()},
                    "layouts": case.layouts(), "file_sha256": Evidence.sha256(tensor_path)})
