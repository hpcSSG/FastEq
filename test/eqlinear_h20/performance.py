"""Pilot-locked, per-call synchronized raw wall/event/host sampling for H20."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

from cases import CONFIGS, Case, points
from common import Evidence
from performance_bindings import CurrentTorchFamily
from protocol import Accuracy, Journal, Statistics, Timer
from reference_store import ReferenceStore
from producer_closure import ProducerClosure
from qualification import AccuracyContracts
from session import Session


class SamplingProtocol:
    """Only anchors determine group sizes, so future candidates keep the same count."""
    VALUE = {"version": 1, "primary": "completed_wall", "auxiliary": ["event", "host_enqueue"],
             "unit": "ms", "rounds": 3, "groups": 20, "cache_mode": "warm_same_buffer",
             "weight_state": "constant", "sync": "device_sync_before_group;device_sync_after_every_call",
             "group_rule": "largest of1,5,10,20 with anchor_median_ms*calls<=20ms; min1",
             "pilot_groups": 5, "target_group_ms": 20., "group_choices": [1, 5, 10, 20],
             "warmup_limit": 30, "warmup_windows": 5, "warmup_tolerance": .05,
             "one_call_components": ["cold_create", "prepare_weight", "prepare_act", "layout_pack", "layout_unpack"],
             "order": "alternate sorted/reversed candidates per group; round parity reverses first group",
             "round_scope": "driver scans full75points before next round; worker measures one round only",
             "math": {**ReferenceStore.MATH, "NVIDIA_TF32_OVERRIDE": "0"}, "source": "unprofiled", "no_trimming": True,
             "group_rule_pilot_basis": "Preliminary pilot c01 measured full-path anchors~1..2ms;20ms target allows5..20 calls for fast native comparisons. No formal results used to select this rule."}

    @classmethod
    def count(cls, median):
        if not math.isfinite(median) or median <= 0:
            raise ValueError("Pilot anchor must have a finite positive median")
        choices = [value for value in cls.VALUE["group_choices"] if value * median <= cls.VALUE["target_group_ms"]]
        return max(choices, default=1)

    @classmethod
    def validate(cls, value):
        cache_mode = value.get("cache_mode")
        if cache_mode not in ("warm_same_buffer", "rotating_inputs") or {**value, "cache_mode": cls.VALUE["cache_mode"]} != cls.VALUE:
            raise ValueError("Sampling protocol differs beyond its declared cache mode")


class PerformancePlan:
    FILES = ("performance.py", "performance_driver.py", "performance_bindings.py", "current.py", "historical.py", "registry.py",
             "cases.py", "common.py", "protocol.py", "reference_store.py", "session.py", "profile_h20.py")

    @classmethod
    def create(cls, session, directory, factory="performance_bindings.CurrentTorchFamily", families=None):
        directory = directory.resolve()
        if not directory.is_relative_to(session.root):
            raise ValueError("Performance plan must be inside its session")
        directory.mkdir(parents=True, exist_ok=False)
        producer = directory / "producer"
        producer.mkdir()
        module_name, class_name = factory.rsplit(".", 1)
        factory_type = getattr(importlib.import_module(module_name), class_name)
        factory_scope = factory_type.configure_scope(session) if hasattr(factory_type, "configure_scope") else None
        protocol = {**SamplingProtocol.VALUE, "cache_mode": getattr(factory_type, "CACHE_MODE", SamplingProtocol.VALUE["cache_mode"])}
        SamplingProtocol.validate(protocol)
        families = families or ["current", "p0", "p14"]
        file_names = sorted(set(cls.FILES) | set(getattr(factory_type, "FILES", ())) | {module_name + ".py"})
        sources = ProducerClosure(Path(__file__).parent, producer).freeze(file_names)
        value = {"session": str(session.root), "directory": str(directory), "producer": sources,
                 "protocol": protocol, "protocol_hash": Evidence.fingerprint({"protocol": protocol,
                    "core": {name: sources[name] for name in ("performance.py", "protocol.py", "cases.py", "reference_store.py")}}),
                 "fixtures": "canonical session fixtures; byte-verified at every load",
                 "factory": factory,
                 "factory_scope": factory_scope,
                 "factory_precision_chunks": bool(getattr(factory_type, "PRECISION_CHUNKS", False)),
                 "include_families": families,
                 "expected_calls": {family: factory_type.expected_calls(family) for family in families},
                 "selected_points": {family: [point.key for point in points()
                    if not hasattr(factory_type, "accept_point") or factory_type.accept_point(family, point)] for family in families},
                 "other_scope_families": "pending separate isolated factories under the same sampler, not excluded"}
        value["plan_hash"] = Evidence.fingerprint(value)
        Evidence.write(directory / "plan.json", value)
        return value

    @classmethod
    def load(cls, path):
        value = json.loads(path.read_text())
        if Evidence.fingerprint({key: data for key, data in value.items() if key != "plan_hash"}) != value["plan_hash"]:
            raise ValueError("Performance plan identity changed")
        SamplingProtocol.validate(value["protocol"])
        for name, digest in value["producer"].items():
            if Evidence.sha256(path.parent / "producer" / name) != digest:
                raise ValueError("Frozen performance producer changed: " + name)
        for filename, digest in (value.get("factory_scope") or {}).get("evidence_files", {}).items():
            if Evidence.sha256(Path(filename)) != digest:
                raise ValueError("Frozen factory scope changed: " + filename)
        return value


class MeasurementIdentity:
    @staticmethod
    def gpu_uuid(value):
        return str(value).removeprefix("GPU-").lower()

    @staticmethod
    def result_kinds(operation):
        return ("grad_x", "grad_w") if operation == "grad_x_w" else ("forward" if operation == "forward_autograd" else operation,)

    @staticmethod
    def output_tensors(value):
        if isinstance(value, (tuple, list)):
            return [tensor for item in value for tensor in MeasurementIdentity.output_tensors(item)]
        if isinstance(value, dict):
            return [tensor for item in value.values() for tensor in MeasurementIdentity.output_tensors(item)]
        if not hasattr(value, "data_ptr"):
            raise TypeError("Measured call did not return tensor outputs")
        return [value]

    @classmethod
    def reference_hash(cls, operation, hashes):
        kinds = cls.result_kinds(operation)
        return hashes[kinds[0]] if len(kinds) == 1 else Evidence.fingerprint({kind: hashes[kind] for kind in kinds})

    @staticmethod
    def check_row(row, identity):
        for key in ("run_id", "GPU_UUID", "protocol_hash", "input_hash"):
            if row[key] != identity[key]:
                raise ValueError("Raw identity mismatch: " + key)
        binding = identity["bindings"][row["variant_id"]]
        for key in ("source_hash", "binary_hash"):
            if row[key] != binding[key]:
                raise ValueError("Raw variant binding mismatch: " + key)
        if row["reference_hash"] != MeasurementIdentity.reference_hash(row["operation"], identity["reference_hashes"]):
            raise ValueError("Raw reference identity mismatch")
        if not all(math.isfinite(row[key]) and row[key] >= 0 for key in ("wall", "event", "host")):
            raise ValueError("Nonfinite/negative performance sample")


class CallValidation:
    """One quality and warmup path for formal calls and pilot-only anchors."""
    @staticmethod
    def run(call, case, references, directory):
        import torch

        first_raw = call.check_function()
        torch.cuda.synchronize()
        second_raw = call.check_function()
        torch.cuda.synchronize()
        first, second = call.normalize(first_raw), call.normalize(second_raw)
        raw_first = MeasurementIdentity.output_tensors(first_raw)
        raw_second = MeasurementIdentity.output_tensors(second_raw)
        kinds = MeasurementIdentity.result_kinds(call.operation)
        first_values = first if isinstance(first, tuple) else (first,)
        second_values = second if isinstance(second, tuple) else (second,)
        if len(first_values) != len(kinds) or len(second_values) != len(kinds) or len(raw_first) != len(raw_second):
            raise ValueError("Result components differ from declared operation")
        components = {kind: Accuracy.evaluate(value, references.values["fp64"][kind], references.torch_errors[kind],
            call.binding.get("accuracy_tier", case.point.precision), torch.equal(value, repeated),
            all(a.data_ptr() != b.data_ptr() for a, b in zip(raw_first, raw_second)))
            for kind, value, repeated in zip(kinds, first_values, second_values)}
        if call.quality_members is not None:
            CallValidation.add_members(call, first_raw, second_raw, components, case)
        quality = {"components": components, "passed": all(value["passed"] for value in components.values()),
            "description": call.description, "output_hashes": {kind: case.tensor_hash(value) for kind, value in zip(kinds, first_values)},
            "measurement_work": call.measurement_work}
        upstream = AccuracyContracts.resolve(call.binding, call.variant, case.point.key, call.operation, case.input_hash, references.hashes)
        if upstream is not None:
            quality.update(local_passed=quality["passed"], upstream_contract=upstream)
            quality["passed"] = quality["local_passed"] and upstream["passed"]
        if not quality["passed"]:
            diagnostic = directory / (call.key.replace("/", "_") + "_accuracy_failure.pt")
            torch.save({"first": {kind: value.detach().cpu() for kind, value in zip(kinds, first_values)},
                        "repeat": {kind: value.detach().cpu() for kind, value in zip(kinds, second_values)}}, diagnostic)
            quality["diagnostic"] = {"path": str(diagnostic), "sha256": Evidence.sha256(diagnostic)}
        del first_raw, second_raw, first, second
        return quality, Timer.warmup(call.warm_function or call.function)

    @staticmethod
    def add_members(call, first_raw, second_raw, components, case):
        """Only the common gate computes member metrics; hooks supply real samples."""
        import torch

        members = call.quality_members(first_raw, second_raw)
        declared = call.binding.get("quality_member_identities")
        if not isinstance(declared, dict) or set(members) != set(components) or set(declared) != set(components):
            raise ValueError("Quality member component declaration differs")
        for kind, samples in members.items():
            if not samples or [sample.identity() for sample in samples] != declared[kind]:
                raise ValueError("Quality member ordered input/reference identities differ or are incomplete")
            if [sample.index for sample in samples] != list(range(len(samples))):
                raise ValueError("Quality members are missing, duplicated, or out of order")
            tensors = [tensor for sample in samples for value in (sample.first, sample.second)
                       for tensor in MeasurementIdentity.output_tensors(value)]
            addresses_unique = len({tensor.data_ptr() for tensor in tensors}) == len(tensors)
            metrics = {}
            for sample in samples:
                if Case.tensor_hash(sample.gold) != sample.reference_hash:
                    raise ValueError("Quality member actual Gold hash differs")
                value = Accuracy.evaluate(sample.first, sample.gold, sample.torch_error,
                    call.binding.get("accuracy_tier", case.point.precision), torch.equal(sample.first, sample.second),
                    addresses_unique and sample.first.data_ptr() != sample.second.data_ptr())
                metrics[str(sample.index)] = {**value, **sample.identity()}
            components[kind]["members"] = metrics
            components[kind]["member_count"] = len(samples)
            components[kind]["member_output_addresses_unique"] = addresses_unique
            components[kind]["passed"] = components[kind]["passed"] and all(value["passed"] for value in metrics.values())


class PilotAnchors:
    @staticmethod
    def declared(plan, point_key):
        return {key.split("/", 1)[1] for family in plan["include_families"]
                if point_key in plan["selected_points"][family] for key in plan["expected_calls"][family]}

    @staticmethod
    def complete(anchors, declared):
        anchors = {key: dict(value) for key, value in anchors.items()}
        for key in sorted(declared):
            operation, boundary = key.split("/")
            if key not in anchors:
                source = next((operation + "/" + candidate for candidate in ("prepared_gemm", "public_warm")
                               if operation + "/" + candidate in anchors), None)
                if source is None:
                    raise ValueError("No same-operation Torch anchor declared for group sizing: " + key)
                anchors[key] = dict(anchors[source], group_sizing_only="not an end-to-end speedup baseline")
            if boundary in SamplingProtocol.VALUE["one_call_components"]:
                anchors[key].update(calls_per_group=1, one_call_component=True)
        return anchors


class PerformanceWorker:
    def __init__(self, args):
        self.args = args
        self.plan = PerformancePlan.load(args.plan.resolve())
        self.session = Session(Path(self.plan["session"]))
        self.directory = Path(self.plan["directory"])
        module, attribute = self.plan["factory"].rsplit(".", 1)
        self.factory_type = getattr(importlib.import_module(module), attribute)
        if self.plan.get("factory_scope") is not None:
            if not hasattr(self.factory_type, "verify_scope"):
                raise TypeError("Configured factory must verify its frozen scope")
            self.factory_type.verify_scope(self.plan["factory_scope"], self.session)

    @staticmethod
    def pin_cpu():
        """Pick from actual allowed CPUs; prefer selected GPU's NUMA node."""
        import subprocess

        allowed = sorted(os.sched_getaffinity(0))
        selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        candidate = allowed
        try:
            bus = subprocess.check_output(["nvidia-smi", "-i", selected, "--query-gpu=pci.bus_id", "--format=csv,noheader"], text=True, timeout=10).strip().lower()
            domain, device, function = bus.split(":")
            sysfs = Path("/sys/bus/pci/devices") / f"{int(domain,16):04x}:{device}:{function}"
            node = int((sysfs / "numa_node").read_text())
            local = set()
            for part in (Path("/sys/devices/system/node") / f"node{node}" / "cpulist").read_text().strip().split(","):
                bounds = [int(value) for value in part.split("-")]
                local.update(range(bounds[0], bounds[-1] + 1))
            candidate = sorted(local & set(allowed)) or allowed
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        os.sched_setaffinity(0, {candidate[0]})
        return {"allowed": allowed, "selected": candidate[0]}

    @classmethod
    def initialize_device(cls):
        affinity = cls.pin_cpu()
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        import torch

        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        physical = MeasurementIdentity.gpu_uuid(torch.cuda.get_device_properties().uuid)
        if physical != MeasurementIdentity.gpu_uuid(os.environ.get("CUDA_VISIBLE_DEVICES", "")):
            raise RuntimeError("Selected physical GPU differs from actual CUDA device")
        return physical, affinity

    def run(self):
        physical, affinity = self.initialize_device()
        selected = [point for point in points() if point.config == self.args.config
                    and point.key in self.plan["selected_points"][self.args.family]
                    and (self.args.precision is None or point.precision == self.args.precision)
                    and (self.args.batch is None or point.batch == self.args.batch)]
        if not selected:
            raise ValueError("No canonical points selected")
        summary = {"status": "RUNNING", "mode": self.args.mode, "family": self.args.family,
                   "round": self.args.round, "points": [], "affinity": affinity, "GPU_UUID": physical}
        summary_path = self.directory / "chunks" / (self.args.attempt + ".json")
        if summary_path.exists():
            raise FileExistsError(summary_path)
        Evidence.write(summary_path, summary)
        for point in selected:
            summary["points"].append(self.point(point, physical, affinity))
            Evidence.write(summary_path, summary)
        PerformancePlan.load(self.args.plan.resolve())
        summary["status"] = "PASS"
        Evidence.write(summary_path, summary)
        return 0

    def prepare_case(self, point):
        workload = self.factory_type.prepare_workload(self.session, point) if hasattr(self.factory_type, "prepare_workload") else None
        if workload is None:
            case = Case(point, fixture_root=self.session.root / "fixtures")
            references = ReferenceStore(self.session.root / "references").get(case)
            workload_identity = None
            cache_mode = "warm_same_buffer"
        else:
            case, references = workload["case"], workload["references"]
            workload_identity, cache_mode = workload["identity"], workload["cache_mode"]
        if cache_mode != self.plan["protocol"]["cache_mode"]:
            raise ValueError("Prepared workload cache mode differs from the frozen plan")
        return case, references, workload_identity, cache_mode

    def point(self, point, gpu, affinity):
        import torch

        case, references, workload_identity, cache_mode = self.prepare_case(point)
        family = self.factory_type(case, self.session, family=self.args.family)
        key = point.key.replace("/", "_")
        point_dir = self.directory / "points" / key
        point_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = point_dir / self.args.attempt
        attempt_dir.mkdir(exist_ok=False)
        quality, warmups = {}, {}
        try:
            for call in family.calls:
                quality[call.key], warmups[call.key] = CallValidation.run(call, case, references, attempt_dir)
            anchor_calls, pilot_quality, pilot_warmups = [], {}, {}
            if self.args.mode == "pilot":
                anchor_calls = list(family.pilot_calls()) if hasattr(family, "pilot_calls") else family.calls
                anchor_calls = [call for call in anchor_calls if call.variant in ("torch_ieee_full", "torch_ieee_mm")]
                if len({call.key for call in anchor_calls}) != len(anchor_calls):
                    raise ValueError("Duplicate pilot anchor")
                for call in anchor_calls:
                    if call.key in quality:
                        pilot_quality[call.key], pilot_warmups[call.key] = quality[call.key], warmups[call.key]
                    else:
                        pilot_quality[call.key], pilot_warmups[call.key] = CallValidation.run(call, case, references, attempt_dir)
            bindings = {call.variant: call.binding for call in family.calls}
            identity = {"run_id": self.session.root.name, "GPU_UUID": gpu, "protocol_hash": self.plan["protocol_hash"],
                        "input_hash": case.input_hash, "reference_hashes": references.hashes, "bindings": bindings,
                        "reference_receipt": references.store_receipt, "point": point.columns(), "affinity": affinity,
                        "plan_hash": self.plan["plan_hash"], "producer": self.plan["producer"], "family": self.args.family,
                        "frame": {"cycle": self.args.cycle, "round": self.args.round, "mode": self.args.mode}}
            identity["cache_mode"] = cache_mode
            if workload_identity is not None:
                identity["workload"] = workload_identity
            Evidence.write(attempt_dir / "setup.json", {"identity": identity, "quality": quality, "warmup": warmups,
                "pilot_quality": pilot_quality, "pilot_warmup": pilot_warmups,
                "family_setup_metadata": getattr(family, "setup_metadata", {}),
                "math_actual": {"precision": torch.get_float32_matmul_precision(), "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "NVIDIA_TF32_OVERRIDE": os.environ["NVIDIA_TF32_OVERRIDE"]}})
            pilot_path = point_dir / "anchor_pilot.json"
            if self.args.mode == "pilot":
                if pilot_path.exists():
                    locked = json.loads(pilot_path.read_text())
                    if locked["plan_hash"] != self.plan["plan_hash"] or locked["identity"] != {key: identity[key] for key in locked["identity"]}:
                        raise ValueError("Existing pilot changed identity")
                    if not PilotAnchors.declared(self.plan, point.key) <= set(locked["anchors"]):
                        raise ValueError("Existing pilot lacks a declared operation/boundary")
                    for anchor in locked["anchors"].values():
                        if anchor["calls_per_group"] != (1 if anchor.get("one_call_component") else SamplingProtocol.count(anchor["median_wall_ms"])):
                            raise ValueError("Existing pilot group count was changed")
                    return {"point": point.key, "status": "PILOT_REUSED_IDENTITY_VERIFIED", "pilot": str(pilot_path)}
                timer = Timer(1)
                anchors = {}
                for call in anchor_calls:
                    samples = []
                    for group in range(SamplingProtocol.VALUE["pilot_groups"]):
                        if call.before_sample is not None:
                            call.before_sample(group, 1, self.args.round)
                        samples.append(timer.sample(call.function))
                    median = statistics.median(row["wall"] for row in samples)
                    anchors[call.operation + "/" + call.boundary] = {"variant": call.variant, "median_wall_ms": median,
                        "calls_per_group": SamplingProtocol.count(median), "samples": samples,
                        "accuracy_pass": pilot_quality[call.key]["passed"]}
                anchors = PilotAnchors.complete(anchors, PilotAnchors.declared(self.plan, point.key))
                pilot = {"identity": {key: identity[key] for key in ("GPU_UUID", "protocol_hash", "input_hash", "reference_hashes", "affinity")},
                         "anchors": anchors, "source_attempt": str(attempt_dir), "rule": SamplingProtocol.VALUE["group_rule"],
                         "plan_hash": self.plan["plan_hash"]}
                Evidence.write(pilot_path, pilot)
                return {"point": point.key, "status": "PILOT_LOCKED", "pilot": str(pilot_path)}
            pilot = json.loads(pilot_path.read_text())
            if pilot["identity"] != {key: identity[key] for key in pilot["identity"]}:
                raise ValueError("Pilot GPU/protocol/input/reference identity changed")
            groups = defaultdict(list)
            for call in family.calls:
                groups[(call.operation, call.boundary)].append(call)
            raw_path = attempt_dir / "raw.csv"
            journal = Journal(raw_path, identity)
            try:
                for (operation, boundary), calls in groups.items():
                    count = pilot["anchors"][operation + "/" + boundary]["calls_per_group"]
                    timer = Timer(count)
                    for group in range(SamplingProtocol.VALUE["groups"]):
                        ordered = sorted(calls, key=lambda call: call.variant, reverse=bool((group + self.args.round) % 2))
                        for order, call in enumerate(ordered):
                            if call.before_sample is not None:
                                call.before_sample(group, count, self.args.round)
                            sample = timer.sample(call.function)
                            if call.after_sample is not None:
                                call.after_sample()
                            accuracy_pass = quality[call.key]["passed"]
                            row = {"run_id": identity["run_id"], "attempt_id": self.args.attempt, "cycle_id": self.args.cycle,
                                "round": self.args.round, "variant_id": call.variant, "source_hash": call.binding["source_hash"],
                                "binary_hash": call.binding["binary_hash"], "GPU_UUID": gpu, "protocol_hash": identity["protocol_hash"],
                                "config": point.config, "B": point.batch, "U": case.u, "V": case.w3.shape[-1], "i_dims": case.dims,
                                "P": len(case.dims), "I": sum(case.dims), "J": case.j, "precision": point.precision,
                                "input_hash": case.input_hash, "reference_hash": MeasurementIdentity.reference_hash(operation, references.hashes), "operation": operation,
                                "call_boundary": boundary, "cache_mode": cache_mode, "weight_state": "constant",
                                "samples": 20, "group": group, "calls_per_group": count, "order": order,
                                "status": "OK" if accuracy_pass else "RAW_DIAGNOSTIC", "accuracy_pass": accuracy_pass, **sample}
                            MeasurementIdentity.check_row(row, identity)
                            journal.emit(row)
            finally:
                journal.close()
            Evidence.write(attempt_dir / "receipt.json", {"status": "COMPLETE", "raw_sha256": Evidence.sha256(raw_path),
                "identity_hash": Evidence.fingerprint(identity), "expected_groups": len(family.calls) * 20,
                "round": self.args.round, "cycle": self.args.cycle, "no_trimming": True})
            return {"point": point.key, "status": "ROUND_COMPLETE", "raw": str(raw_path)}
        finally:
            family.close()

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        sub = parser.add_subparsers(dest="action", required=True)
        create = sub.add_parser("create")
        create.add_argument("--session", type=Path, required=True)
        create.add_argument("--directory", type=Path, required=True)
        create.add_argument("--factory", default="performance_bindings.CurrentTorchFamily")
        create.add_argument("--families", default="current,p0,p14")
        worker = sub.add_parser("worker")
        worker.add_argument("--plan", type=Path, required=True)
        worker.add_argument("--mode", choices=("pilot", "measure"), required=True)
        worker.add_argument("--family", required=True)
        worker.add_argument("--config", choices=tuple(CONFIGS), required=True)
        worker.add_argument("--precision", choices=("tf32", "fp32", "fp64"))
        worker.add_argument("--batch", type=int)
        worker.add_argument("--round", type=int, choices=(0, 1, 2), default=0)
        worker.add_argument("--cycle", type=int, default=1)
        worker.add_argument("--attempt", required=True)
        args = parser.parse_args()
        if args.action == "create":
            print(PerformancePlan.create(Session(args.session), args.directory, args.factory, args.families.split(","))["plan_hash"])
            return 0
        return cls(args).run()


if __name__ == "__main__":
    raise SystemExit(PerformanceWorker.main())
