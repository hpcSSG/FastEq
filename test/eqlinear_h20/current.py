"""Current FastEq adapter; numerical reference and timer live in shared modules."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from common import Evidence


class CurrentAdapter:
    def __init__(self, case, session):
        import torch

        self.case, self.session = case, session
        self.module = importlib.import_module("fasteq.ops.eqlinear")
        self.cuda = importlib.import_module("fasteq.cuda")
        root = Path(session.paths["FASTEQ_ROOT"]).resolve()
        if not Path(self.module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Unexpected editable Python source")
        self.manifest = json.loads((session.root / "manifests/current_build.json").read_text())
        mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if ".so" in line}
        for row in self.manifest["binaries"]:
            if row["installed_path"] not in mapped or Evidence.sha256(Path(row["installed_path"])) != row["sha256"]:
                raise RuntimeError("Fresh binary fingerprint/mapping mismatch")
        self.namespace = torch.ops.eqlinear
        self.options = {"compute_precision": case.point.precision} if case.point.precision != "fp64" else {}
        self.handles = {}
        self.outputs = {"forward": torch.empty_like(case.grad3), "grad_x": torch.empty_like(case.x3)}
        self.x_grad = case.x.detach().requires_grad_(True)
        with torch.enable_grad():
            self.graph = self.module.fast_equi_linear(case.descriptor, case.w, self.x_grad, **self.options)

    def prepare(self, operation, boundary):
        if operation not in ("forward", "forward_autograd", "grad_x"):
            raise NotImplementedError("Current public API has grad_x only")
        if boundary == "run_with_activation_prep" and operation != "forward_autograd":
            self.handle(operation)
        elif boundary != "public_warm":
            raise NotImplementedError("Current Torch API has no separate prepare_act/launch entry point")

    def handle(self, operation):
        if operation not in self.handles:
            c = self.case
            create = self.namespace.forward_create if operation == "forward" else self.namespace.dgrad_create
            self.handles[operation] = create(c.w3, c.dims, c.cg, c.point.batch, c.u, len(c.dims), c.u, c.precision_id)
        return self.handles[operation]

    def call(self, operation, boundary, fresh_output=False):
        import torch

        c = self.case
        if boundary == "public_warm":
            if operation == "grad_x":
                with torch.enable_grad():
                    return torch.autograd.grad(self.graph, self.x_grad, c.grad, retain_graph=True)[0]
            if operation == "forward_autograd":
                with torch.enable_grad():
                    return self.module.fast_equi_linear(c.descriptor, c.w, self.x_grad, **self.options)
            with torch.no_grad():
                return self.module.fast_equi_linear(c.descriptor, c.w, c.x, **self.options)
        if boundary != "run_with_activation_prep":
            raise NotImplementedError(boundary)
        output = torch.empty_like(self.outputs[operation]) if fresh_output else self.outputs[operation]
        function = self.namespace.forward_run if operation == "forward" else self.namespace.dgrad_run
        function(self.handle(operation), c.x3 if operation == "forward" else c.grad3, output)
        return output.reshape(c.point.batch, -1)

    def identity(self):
        return {"variant_id": "current_fasteq", "source_hash": self.manifest["source"]["source_hash"],
                "binary_hash": Evidence.fingerprint({row["installed_path"]: row["sha256"] for row in self.manifest["binaries"]}),
                "binaries": self.manifest["binaries"],
                "plan_info": {operation: list((self.namespace.forward_plan_info if operation == "forward" else
                                               self.namespace.dgrad_plan_info)(handle))
                              for operation, handle in self.handles.items()}}

    def close(self):
        import torch

        torch.cuda.synchronize()
        for operation, handle in self.handles.items():
            (self.namespace.forward_destroy if operation == "forward" else self.namespace.dgrad_destroy)(handle)
        self.handles.clear()
        self.graph = None

