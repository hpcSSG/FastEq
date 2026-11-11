"""Useful-work model and measured DRAM/L2 traffic, with explicit unknown physical roofs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from cases import CONFIGS
from common import Evidence


@dataclass(frozen=True)
class EqlinearWorkload:
    batch: int
    u: int
    v: int
    dims: tuple
    element_bytes: int

    @property
    def j(self):
        return sum(value for index, value in enumerate(self.dims) if index == 0 or value != self.dims[index-1])

    def operation(self, operation):
        operation = "forward" if operation == "forward_autograd" else operation
        if operation not in ("forward", "grad_x", "grad_w", "grad_x_w"):
            raise ValueError("Unknown mathematical operation")
        b, i, u, v, p, s = self.batch, sum(self.dims), self.u, self.v, len(self.dims), self.element_bytes
        one_product = 2 * b * i * u * v
        tensors = {"x": b*i*u*s, "w": p*u*v*s, "dy": b*self.j*v*s,
                   "y": b*self.j*v*s, "dx": b*i*u*s, "dw": p*u*v*s}
        selected = {"forward": ("x", "w", "y"), "grad_x": ("w", "dy", "dx"),
                    "grad_w": ("x", "dy", "dw"), "grad_x_w": ("x", "w", "dy", "dx", "dw")}[operation]
        return {"operation": operation, "useful_matrix_product_flops": one_product * (2 if operation == "grad_x_w" else 1),
            "logical_cg_multiply_flops": b*i*v,
            "logical_run_length_reduction_add_flops": b*(i-self.j)*v if operation == "forward" else 0,
            "logical_tensor_bytes_once": {key: tensors[key] for key in selected},
            "logical_cold_minimum_bytes": sum(tensors[key] for key in selected),
            "j": self.j, "i": i, "p": p,
            "convention": "One multiply-add=2 FLOPs. DX/DW scale dy per path before products; combined gradients can share that scale.",
            "executed_flops": None,
            "executed_flops_status": "UNKNOWN_REQUIRES_VARIANT_PADDING_SPLITTING_AND_INSTRUCTION_MODEL; useful work is not multiplied by an assumed pass count",
            "logical_bytes_interpretation": "Optimistic cold logical tensor traffic, not observed DRAM/L2 traffic or guaranteed HBM writes"}


class ProfileTraffic:
    @staticmethod
    def model_applicable(profile):
        component = profile["call"].split("/")[2] in ("cold_create", "prepare_weight", "prepare_act", "layout_pack", "layout_unpack")
        return not component and profile.get("quality", {}).get("passed") is True

    @staticmethod
    def total(kernels, metric, unit):
        values = [kernel["metrics"].get(metric, {}).get("si") for kernel in kernels]
        if not values or any(value is None or value["unit"] != unit for value in values):
            return None
        return sum(value["value"] for value in values)

    @classmethod
    def run(cls, args):
        collection = json.loads(args.collection.read_text())
        if collection["status"] != "PASS" or "counters" not in collection:
            raise ValueError("A successful real NCU collection is required")
        for key in ("result", "raw", "report", "counters"):
            if Evidence.sha256(Path(collection[key]["path"])) != collection[key]["sha256"]:
                raise ValueError("Changed profile evidence: " + key)
        profile = json.loads(Path(collection["result"]["path"]).read_text())
        counters = json.loads(Path(collection["counters"]["path"]).read_text())
        u, dims = CONFIGS[profile["point"]["config"]]
        workload = EqlinearWorkload(profile["point"]["batch"], u, u, tuple(dims), 8 if profile["point"]["precision"] == "fp64" else 4)
        operation = profile["call"].split("/")[1]
        model = workload.operation(operation)
        kernels = counters["kernels"]
        duration = cls.total(kernels, "gpu__time_duration.sum", "s")
        read = cls.total(kernels, "dram__bytes_read.sum", "byte")
        write = cls.total(kernels, "dram__bytes_write.sum", "byte")
        l2 = cls.total(kernels, "lts__t_bytes.sum", "byte")
        dram = read + write if read is not None and write is not None else None
        calls = profile["profile_calls"]
        useful = model["useful_matrix_product_flops"] * calls
        applicable = cls.model_applicable(profile)
        streams = {row["metrics"].get("Stream", {}).get("raw_value") for row in counters["all_metric_rows"]}
        result = {"classification": "PROFILED_OPERATOR_TRAFFIC_AND_USEFUL_WORK_NOT_MAIN_WALL_OR_PHYSICAL_MFU",
            "point": profile["point"], "call": profile["call"], "GPU_UUID": profile["GPU_UUID"],
            "input_hash": profile["input_hash"], "reference_hashes": profile["reference_hashes"],
            "model": model, "profile_calls": calls, "model_applicable_to_measured_boundary": applicable,
            "accuracy_qualified": profile.get("quality", {}).get("passed") is True,
            "profiled_kernel_count": len(kernels), "observed_streams": sorted(str(value) for value in streams),
            "sum_profiled_kernel_seconds": duration, "dram_read_bytes": read, "dram_write_bytes": write,
            "dram_bytes": dram, "l2_bytes": l2,
            "useful_product_flops_per_measured_dram_byte": useful / dram if applicable and dram is not None and dram > 0 else None,
            "useful_product_flops_per_measured_l2_byte": useful / l2 if applicable and l2 is not None and l2 > 0 else None,
            "useful_product_flops_per_sum_profiled_kernel_second": useful / duration if applicable and duration and duration > 0 else None,
            "hbm_measured_bytes_per_sum_profiled_kernel_second": dram / duration if dram is not None and duration and duration > 0 else None,
            "l2_measured_bytes_per_sum_profiled_kernel_second": l2 / duration if l2 is not None and duration and duration > 0 else None,
            "physical_mfu": None, "physical_mfu_status": "UNKNOWN_PHYSICAL_COMPUTE_ROOF_AND_EXECUTED_OPERATION_MODEL",
            "warnings": ["NCU replay/cache state and sum of instrumented kernel durations do not give unprofiled completed wall time",
                         "Kernel duration sums do not establish pipeline critical-path latency if streams overlap",
                         "CG/reduction/preparation arithmetic is listed separately and excluded from the useful matrix-product numerator",
                         "Zero measured DRAM writes mean no writes counted in this captured interval, not absence of output stores"],
            "collection": {"path": str(args.collection), "sha256": Evidence.sha256(args.collection)},
            "tool_sha256": Evidence.sha256(Path(__file__))}
        if args.output.exists():
            raise FileExistsError(args.output)
        Evidence.write(args.output, result)
        print(json.dumps({key: result[key] for key in ("profiled_kernel_count", "sum_profiled_kernel_seconds", "dram_bytes", "l2_bytes", "physical_mfu")}))
        return 0

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--collection", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        return cls.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(ProfileTraffic.main())
