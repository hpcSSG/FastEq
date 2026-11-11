"""Call-boundary adapters for the shared H20 performance sampler."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from cases import References
from common import Evidence
from current import CurrentAdapter


@dataclass
class QualityMember:
    """Actual independent outputs and shared Gold for one declared workload member."""
    first: object
    second: object
    gold: object
    torch_error: float
    label: str
    index: int
    input_hash: str
    reference_hash: str

    def identity(self):
        return {"label": self.label, "index": self.index, "input_hash": self.input_hash,
                "reference_hash": self.reference_hash}


@dataclass
class BoundCall:
    variant: str
    operation: str
    boundary: str
    function: object
    check_function: object
    normalize: object
    binding: dict
    description: str
    after_sample: object = None
    warm_function: object = None
    measurement_work: str = "full_operation"
    before_sample: object = None
    quality_members: object = None

    @property
    def key(self):
        return "/".join((self.variant, self.operation, self.boundary))


class TorchIdentity:
    """Hash actually mapped Torch/CUDA/cuBLAS binaries; reuse only unchanged files."""
    @classmethod
    def capture(cls, session):
        import torch

        uuid = str(torch.cuda.get_device_properties().uuid).removeprefix("GPU-").lower()
        path = session.root / "manifests" / ("torch_performance_binaries_" + uuid + ".json")
        old = json.loads(path.read_text()) if path.exists() else {"files": {}}
        mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                         if ".so" in line and any(value in line for value in
                         ("libtorch_cuda", "libtorch_cpu", "libcublas", "libcuda.so", "libcudart", "/torch/_C."))})
        files = {}
        for filename in mapped:
            binary = Path(filename)
            stat = binary.stat()
            record = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            previous = old["files"].get(filename, {})
            record["sha256"] = previous["sha256"] if all(previous.get(key) == value for key, value in record.items()) else Evidence.sha256(binary)
            files[filename] = record
        identity = {"torch": torch.__version__, "cuda": torch.version.cuda, "abi": torch._C._GLIBCXX_USE_CXX11_ABI,
                    "files": files, "validation": "SHA256 initially; unchanged size/mtime on reuse; independent final SHA readback required"}
        if not files or not any("libcublas" in name for name in files):
            raise RuntimeError("Expected actual cuBLAS/Torch binaries are not mapped")
        Evidence.write(path, identity)
        return {"binary_hash": Evidence.fingerprint({name: record["sha256"] for name, record in files.items()}),
                "binary_manifest": str(path), "binary_files": files, "torch": identity["torch"], "cuda": identity["cuda"], "abi": identity["abi"]}


class ColdPlanCalls:
    """Keep destruction outside a one-call create window, with bounded owned handles."""
    def __init__(self, adapter, operation):
        self.adapter, self.operation, self.pending = adapter, operation, []

    def create(self):
        case = self.adapter.case
        function = self.adapter.namespace.forward_create if self.operation == "forward" else self.adapter.namespace.dgrad_create
        handle = function(case.w3, case.dims, case.cg, case.point.batch, case.u, len(case.dims), case.w3.shape[-1], case.precision_id)
        self.pending.append(handle)
        return handle

    def cleanup(self):
        import torch
        torch.cuda.synchronize()
        destroy = self.adapter.namespace.forward_destroy if self.operation == "forward" else self.adapter.namespace.dgrad_destroy
        while self.pending:
            destroy(self.pending.pop())

    def warm(self):
        handle = self.create()
        self.cleanup()
        return handle

    def check(self):
        import torch
        case = self.adapter.case
        handle = self.create()
        try:
            output = torch.empty_like(case.grad3 if self.operation == "forward" else case.x3)
            run = self.adapter.namespace.forward_run if self.operation == "forward" else self.adapter.namespace.dgrad_run
            run(handle, case.x3 if self.operation == "forward" else case.grad3, output)
            torch.cuda.synchronize()
            return output.reshape(case.point.batch, -1)
        finally:
            self.cleanup()


class TorchCalls:
    """Eager full path and explicitly separate pre-gathered product-only GEMMs."""
    def __init__(self, case):
        self.case, self.graphs, self.products = case, {}, {}
        self.forward_x = case.x.detach().requires_grad_(True)

    def full(self, operation):
        import torch

        case = self.case
        if operation == "forward":
            with torch.no_grad():
                return References.forward(case.x.reshape_as(case.x3), case.w.reshape_as(case.w3), case.dims, case.cg,
                                          case.dtype, False).reshape(case.point.batch, -1)
        if operation == "forward_autograd":
            with torch.enable_grad():
                return References.forward(self.forward_x.reshape_as(case.x3), case.w.reshape_as(case.w3), case.dims, case.cg,
                                          case.dtype, False).reshape(case.point.batch, -1)
        if operation not in self.graphs:
            raise RuntimeError("Backward graph must be prepared outside timing")
        graph, variables = self.graphs[operation]
        with torch.enable_grad():
            values = torch.autograd.grad(graph, variables, case.grad, retain_graph=True)
            return values if operation == "grad_x_w" else values[0]

    def prepare(self, operations=("grad_x",)):
        import torch

        case = self.case
        for operation in operations:
            with torch.enable_grad():
                x = case.x.detach().requires_grad_(operation in ("grad_x", "grad_x_w"))
                w = case.w.detach().requires_grad_(operation in ("grad_w", "grad_x_w"))
                graph = References.forward(x.reshape_as(case.x3), w.reshape_as(case.w3), case.dims, case.cg, case.dtype, False).reshape(case.point.batch, -1)
                variables = (x, w) if operation == "grad_x_w" else (x,) if operation == "grad_x" else (w,)
                self.graphs[operation] = graph, variables
        offset, group, previous = 0, -1, None
        forward, backward, weight_gradient = [], [], []
        output_offset = 0
        for index, dimension in enumerate(case.dims):
            if dimension != previous:
                group += 1
                if previous is not None:
                    output_offset += previous
            matrix = case.x3[:, offset:offset + dimension].reshape(-1, case.u).contiguous()
            weight = case.w3[index].contiguous()
            # Match autograd's chain rule rounding: scale dy before dgrad GEMM.
            dy = (case.grad3[:, output_offset:output_offset + dimension] * case.cg[index]).reshape(-1, case.u).contiguous()
            forward.append((matrix, weight, torch.empty((matrix.shape[0], case.u), dtype=case.dtype, device=case.x.device), group, index))
            backward.append((dy, weight.T.contiguous(), torch.empty_like(matrix), group, index))
            weight_gradient.append((matrix.T.contiguous(), dy, torch.empty_like(weight), group, index))
            offset += dimension
            previous = dimension
        self.products = {"forward": forward, "grad_x": backward, "grad_w": weight_gradient}
        if case.x.is_cuda:
            torch.cuda.synchronize()

    def mm(self, operation, fresh=False):
        import torch

        if operation == "grad_x_w":
            return self.mm("grad_x", fresh), self.mm("grad_w", fresh)
        result = []
        for a, b, output, _, _ in self.products[operation]:
            target = torch.empty_like(output) if fresh else output
            torch.mm(a, b, out=target)
            result.append(target)
        return result

    def normalize_mm(self, operation, values):
        import torch

        case = self.case
        if operation == "grad_x_w":
            return self.normalize_mm("grad_x", values[0]), self.normalize_mm("grad_w", values[1])
        if operation == "grad_w":
            return torch.stack(values).reshape(-1)
        if operation == "grad_x":
            return torch.cat([value.reshape(case.point.batch, dimension, case.u)
                              for index, (dimension, value) in enumerate(zip(case.dims, values))], dim=1).reshape(case.point.batch, -1)
        groups = []
        previous = None
        for index, (dimension, value) in enumerate(zip(case.dims, values)):
            term = value.reshape(case.point.batch, dimension, case.u) * case.cg[index]
            if dimension != previous:
                groups.append(term)
            else:
                groups[-1] = groups[-1] + term
            previous = dimension
        return torch.cat(groups, dim=1).reshape(case.point.batch, -1)


class CurrentTorchFamily:
    """One isolated current DSO set plus Torch; no standalone or historical mixed loads."""
    FILES = ("performance_bindings.py", "current.py")

    @staticmethod
    def expected_calls(family):
        native = "current_fasteq" if family == "current" else "historical_" + family
        result = ["/".join((native, operation, boundary)) for operation, boundary in
                  (("forward", "public_warm"), ("forward", "run_with_activation_prep"),
                   ("forward_autograd", "public_warm"), ("grad_x", "public_warm"), ("grad_x", "run_with_activation_prep"))]
        result += [f"torch_ieee_full/{operation}/public_warm" for operation in ("forward", "forward_autograd", "grad_x")]
        result += [f"torch_ieee_mm/{operation}/prepared_gemm" for operation in ("forward", "grad_x")]
        result += [f"{native}/{operation}/cold_create" for operation in ("forward", "grad_x")]
        return sorted(result)

    def __init__(self, case, session, family="current"):
        import torch

        self.case, self.session = case, session
        if family == "current":
            self.current = CurrentAdapter(case, session)
            native_name = "current_fasteq"
        else:
            from historical import HistoricalAdapter
            self.current = HistoricalAdapter(case, session, family)
            native_name = "historical_" + family
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = TorchCalls(case)
        self.torch.prepare()
        # Drain all setup, including the initial public autograd forward.
        torch.cuda.synchronize()
        torch_identity = TorchIdentity.capture(session)
        torch_identity["source_hash"] = Evidence.fingerprint({name: Evidence.sha256(Path(__file__).parent / name)
                                                             for name in ("performance_bindings.py", "cases.py")})
        torch_identity["accuracy_tier"] = "fp64" if case.point.precision == "fp64" else "fp32"
        self.calls = []
        self.cold = []
        identity = lambda value: value
        for operation in ("forward", "forward_autograd", "grad_x"):
            for boundary in ("public_warm", "run_with_activation_prep"):
                if operation == "forward_autograd" and boundary != "public_warm":
                    continue
                self.current.prepare(operation, boundary)
                self.calls.append(BoundCall(native_name, operation, boundary,
                    lambda op=operation, bound=boundary: self.current.call(op, bound),
                    lambda op=operation, bound=boundary: self.current.call(op, bound, fresh_output=True), identity,
                    {}, "public allocates independent output; low-level run uses preallocated output and includes activation preparation"))
            self.calls.append(BoundCall("torch_ieee_full", operation, "public_warm",
                lambda op=operation: self.torch.full(op), lambda op=operation: self.torch.full(op), identity,
                torch_identity, "eager full path on original layouts, including views/path mm/post-cg/reduction/output allocation"))
        for operation in ("forward", "grad_x"):
            self.calls.append(BoundCall("torch_ieee_mm", operation, "prepared_gemm",
                lambda op=operation: self.torch.mm(op), lambda op=operation: self.torch.mm(op, fresh=True),
                lambda value, op=operation: self.torch.normalize_mm(op, value), torch_identity,
                "product-only per-path GEMMs; gather/transpose/cg/reduction outside window; no end-to-end speedup against public",
                measurement_work="matrix_products_only"))
            cold = ColdPlanCalls(self.current, operation)
            self.cold.append(cold)
            self.calls.append(BoundCall(native_name, operation, "cold_create", cold.create, cold.check, identity, {},
                "fresh plan creation includes weight preparation; run-for-validation and destruction outside create timing",
                after_sample=cold.cleanup, warm_function=cold.warm, measurement_work="plan_create_and_weight_prepare"))
        current_identity = self.current.identity()
        current_identity["production_source_hash"] = current_identity["source_hash"]
        current_identity["sampler_adapter_hashes"] = {name: Evidence.sha256(Path(__file__).with_name(name))
            for name in (("performance_bindings.py", "current.py") if family == "current" else
                         ("performance_bindings.py", "current.py", "historical.py", "registry.py"))}
        current_identity["accuracy_tier"] = case.point.precision
        current_identity["source_hash"] = Evidence.fingerprint({"production": current_identity["production_source_hash"],
                                                              "adapters": current_identity["sampler_adapter_hashes"]})
        for call in self.calls:
            if call.variant == native_name:
                call.binding = current_identity
        if sorted(call.key for call in self.calls) != self.expected_calls(family):
            raise RuntimeError("Prepared call set differs from declared measurement scope")

    def close(self):
        for cold in self.cold:
            cold.cleanup()
        self.current.close()
        self.torch.graphs.clear()
