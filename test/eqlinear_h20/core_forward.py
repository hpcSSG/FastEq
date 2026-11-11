"""Current FastEq, IEEE Torch and installed Luna forward on identical inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path

from common import Evidence
from current import CurrentAdapter
from luna import LunaAdapter
from performance_bindings import BoundCall, TorchCalls, TorchIdentity


class CoreForwardFamily:
    OPERATION = "forward"
    PRECISION_CHUNKS = True
    FILES = ("core_forward.py", "luna.py")

    @classmethod
    def configure_scope(cls, session):
        cache_plan = session.root / "accuracy/luna_matrix/luna_accuracy_full_c03/plan.json"
        return {"operation": cls.OPERATION, "versions": ["current_fasteq", "torch_ieee", "installed_luna_0.2.0"],
                "accuracy_tiers_by_variant": {"fasteq_default": {"fp32": "tf32", "fp64": "fp64"}},
                "input_policy": "All implementations use the same serialized FP32/FP64 canonical inputs; TF32x3 is a FastEq compute mode on the FP32 input, not a different fixture",
                "call_policy": "Public allocating forward; Luna activation pack and output unpack included; fixed-weight CG/alpha mapping outside timing, matching constant-weight warm plans",
                "backward_policy": "For grad_x, prepare differentiable forward graph outside timing; measure only autograd.grad with common canonical nonzero grad_out",
                "evidence_files": {str(cache_plan): Evidence.sha256(cache_plan)},
                "excluded": ["history", "standalone_parameter_sweeps", "native_selector_sweeps", "grad_w", "grad_x_w", "autograd_forward", "cold_components", "extra_profiling"]}

    @classmethod
    def verify_scope(cls, scope, session):
        if scope != cls.configure_scope(session):
            raise ValueError("Current three-way forward scope changed")

    @staticmethod
    def accept_point(family, point):
        return point.precision == family.rsplit("_", 1)[1] and point.precision in ("fp32", "fp64")

    @classmethod
    def expected_calls(cls, family):
        operation = cls.OPERATION
        result = [f"torch_ieee_full/{operation}/public_warm"]
        if family.startswith("fasteq_"):
            result.append(f"fasteq_default/{operation}/public_warm")
            if family.endswith("fp32"):
                result.append(f"fasteq_strict_fp32/{operation}/public_warm")
        elif family.startswith("luna_"):
            result.append(f"lunaeq_0.2.0/{operation}/adapted_public_e2e")
        else:
            raise ValueError("Unknown current forward family")
        return sorted(result)

    def __init__(self, case, session, family):
        import torch

        self.case, self.current, self.luna = case, None, None
        self.graphs = {}
        self.torch = TorchCalls(case)
        if self.OPERATION == "grad_x":
            self.torch.prepare(("grad_x",))
        self.calls = []
        if family.startswith("fasteq_"):
            self.current = CurrentAdapter(case, session)
            binding = self.current.identity()
            binding["production_source_hash"] = binding["source_hash"]
            for variant, mode in (("fasteq_default", None), ("fasteq_strict_fp32", "fp32")):
                if variant == "fasteq_strict_fp32" and case.point.precision != "fp32":
                    continue
                actual = {**binding, "api_controls": {"compute_precision": mode},
                          "accuracy_tier": "tf32" if mode is None and case.point.precision == "fp32" else case.point.precision}
                if self.OPERATION == "grad_x":
                    with torch.enable_grad():
                        leaf = case.x.detach().requires_grad_(True)
                        graph = self.current.module.fast_equi_linear(case.descriptor, case.w, leaf,
                            **({} if mode is None else {"compute_precision": mode}))
                        self.graphs[mode] = graph, leaf
                function = (lambda selected=mode: self.fasteq_forward(selected)) if self.OPERATION == "forward" else (lambda selected=mode: self.fasteq_dx(selected))
                self.calls.append(BoundCall(variant, self.OPERATION, "public_warm", function, function, lambda value: value,
                    actual, "Current public API; default TF32x3/FP64 or explicit strict FP32; grad_x excludes graph construction"))
        else:
            cache_plan = json.loads((session.root / "accuracy/luna_matrix/luna_accuracy_full_c03/plan.json").read_text())
            if Evidence.sha256(Path(__file__).with_name("luna.py")) != cache_plan["producer"]["luna.py"]:
                raise ValueError("Existing validated Luna adapter changed")
            os.environ["LUNAEQ_CACHE_DIR"] = str(session.root / "cache/luna_matrix" / cache_plan["sources"]["0.2.0"]["source_hash"]
                / cache_plan["producer"]["luna.py"] / case.point.precision / self.OPERATION)
            os.environ["LUNATP_CUTLASS_INCLUDE"] = str(Path(session.paths["LUNAEQ_ROOT"]) / "src/third_party/cutlass/include")
            self.luna = LunaAdapter(case, session.paths["LUNAEQ_ROOT"], "0.2.0", session.root / "manifests/luna_jit/0.2.0")
            self.luna.prepare(self.OPERATION, "adapted_public_e2e")
            torch.cuda.synchronize()
            actual = self.luna.identity()
            if actual["backend"] != "cuda_fused" or not actual["jit_libraries"] or not set(actual["jit_libraries"]) <= set(actual["mapped_jit_libraries"]):
                raise ValueError("Luna forward must execute its actual mapped fused CUDA backend")
            actual.update(source_hash=Evidence.fingerprint({key: actual[key] for key in ("python_source_hash", "pyproject_sha256")}),
                          binary_hash=Evidence.fingerprint(actual["jit_libraries"]), accuracy_tier=case.point.precision)
            function = lambda: self.luna.call(self.OPERATION, "adapted_public_e2e")
            self.calls.append(BoundCall("lunaeq_0.2.0", self.OPERATION, "adapted_public_e2e", function, function,
                lambda value: value, actual, "Luna0.2 fused public operation on original layouts; grad_x excludes prepared forward graph and includes adapter backward"))
        self.torch.full(self.OPERATION)
        torch.cuda.synchronize()
        binding = TorchIdentity.capture(session)
        binding.update(source_hash=Evidence.sha256(Path(__file__)), accuracy_tier=case.point.precision)
        function = lambda: self.torch.full(self.OPERATION)
        self.calls.append(BoundCall("torch_ieee_full", self.OPERATION, "public_warm", function, function, lambda value: value,
            binding, "Existing IEEE eager Torch path; grad_x uses prepared forward graph and common original-layout grad_out"))
        if sorted(call.key for call in self.calls) != self.expected_calls(family):
            raise ValueError("Current three-way forward scope differs")

    def fasteq_forward(self, mode):
        import torch

        case = self.case
        with torch.no_grad():
            return self.current.module.fast_equi_linear(case.descriptor, case.w, case.x,
                **({} if mode is None else {"compute_precision": mode}))

    def fasteq_dx(self, mode):
        import torch

        graph, leaf = self.graphs[mode]
        with torch.enable_grad():
            return torch.autograd.grad(graph, leaf, self.case.grad, retain_graph=True)[0]

    def close(self):
        self.graphs.clear()
        if self.current is not None:
            self.current.close()
        if self.luna is not None:
            self.luna.close()


class CoreDxFamily(CoreForwardFamily):
    OPERATION = "grad_x"
