"""Official hardware evidence and bounded, IEEE-only copy/GEMM calibration.

Calibration outputs are achieved payload/pipeline lower bounds, never inferred physical
peaks. TF32x1/FP16/BF16/Graph execution is deliberately absent.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import re
import urllib.request

from common import Evidence
from session import Session


class OfficialSources:
    URLS = {
        "h100_product": "https://www.nvidia.com/en-us/data-center/h100/",
        "h100_whitepaper": "https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf",
        "mig_guide": "https://docs.nvidia.com/datacenter/tesla/pdf/MIG_User_Guide.pdf",
        "line_card": "https://docs.nvidia.com/data-center-gpu/line-card.pdf",
    }

    @classmethod
    def capture(cls, output):
        output.mkdir(parents=True, exist_ok=False)
        evidence = []
        for name, url in cls.URLS.items():
            row = {"id": name, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "H20-eqlinear-hardware-evidence/1.0"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    content = response.read(24 * (1 << 20) + 1)
                    if len(content) > 24 * (1 << 20):
                        raise ValueError("Official response exceeds bounded 24MiB limit")
                    row.update(final_url=response.url, content_type=response.headers.get("Content-Type"))
                path = output / (name + (".pdf" if content.startswith(b"%PDF") else ".html"))
                path.write_bytes(content)
                row.update(status="CAPTURED", path=str(path), bytes=len(content), sha256=Evidence.sha256(path))
            except Exception as error:
                row.update(status="FETCH_FAILED", error_type=type(error).__name__, error=str(error))
            evidence.append(row)
        Evidence.write(output / "receipts.json", evidence)
        return evidence


class HardwareInventory:
    @staticmethod
    def unknown(reason):
        return {"value": None, "status": "UNKNOWN", "reason": reason}

    @classmethod
    def discover(cls, session, output):
        output.mkdir(parents=True, exist_ok=False)
        official = OfficialSources.capture(output / "official")
        sources = {row["id"]: row for row in official}
        historical_root = Path(session.paths["HANDOFF_ROOT"]) / "h100/fasteq_fix"
        meta_path = historical_root / "p29_acceptance_pooled/timing_metadata.json"
        trace_path = historical_root / "profile_results/p29_host_fp32/fp32_candidate_trace.json"
        metadata, trace = json.loads(meta_path.read_text()), json.loads(trace_path.read_text())
        history = {"environment": metadata["environment"], "device_properties": trace["deviceProperties"],
                   "sources": [{"path": str(path), "sha256": Evidence.sha256(path)} for path in (meta_path, trace_path)],
                   "hardware_identity": "P29 has name+UUID; trace has 132 SM and HBM3. Other historical datasets need their own identity checks.",
                   "sku_match": "CONSISTENT_WITH_H100_SXM5_80GB_132SM; board part number not recorded",
                   "measured_roofs": None, "fresh_h100_access": False}
        h20_peak = cls.unknown("No complete NVIDIA H20 precision/throughput specification located in the queried official sources; media/vendor constants not adopted")
        h20 = {"model_family": "H20", "official_identity": {"architecture": "Hopper GH100", "compute_capability": "9.0", "memory_as_published": "96GB", "source_id": "mig_guide", "pdf_page": 10},
               "official_dense_peak_flops_per_second": {name: dict(h20_peak) for name in ("tf32_tensor", "fp32_cuda_core", "fp64_cuda_core", "fp64_tensor_dmma")},
               "official_hbm_bandwidth_bytes_per_second": dict(h20_peak),
               "physical_l2_bandwidth_bytes_per_second": cls.unknown("No independent physical L2 bandwidth roof established"),
               "prohibited_precision_execution": ["tf32x1", "fp16", "bf16"], "measured_calibrations": []}
        raw = session.root / "profiles/ncu/luna020_sm512_fp64_ncu01.raw.csv"
        if raw.exists():
            with raw.open(newline="") as stream:
                records = list(csv.reader(stream))
            header, units, values = records[:3]
            mapping = dict(zip(header, values))
            keys = ("device__attribute_display_name", "device__attribute_l2_cache_size", "device__attribute_sm_count",
                    "device__attribute_clock_rate", "device__attribute_memory_clock_rate", "device__attribute_total_global_mem",
                    "device__attribute_global_memory_bus_width", "device__attribute_compute_capability_major", "device__attribute_compute_capability_minor")
            h20["observed_device_attributes"] = {key: mapping[key] for key in keys if key in mapping}
            h20["observed_attribute_source"] = {"path": str(raw), "sha256": Evidence.sha256(raw),
                                                  "note": "Device attributes only; no kernel throughput percentage used to infer a roof"}
        if sources["h100_whitepaper"]["status"] == "CAPTURED":
            history["official_sxm5_background_dense_peak_flops_per_second"] = {
                "tf32_tensor": 494.7e12, "fp32_cuda_core": 66.9e12, "fp64_cuda_core": 33.5e12, "fp64_tensor_dmma": 66.9e12}
            history["official_compute_source"] = {"source_id": "h100_whitepaper", "version": "1.04", "pdf_page": 20,
                                                    "source_page": "https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c",
                                                    "dense_sparse_separation": "TF32 494.7 dense; 989.4 sparse is not used"}
            history["official_l2_capacity_as_published"] = {"value": 50, "unit": "MB_as_published", "pdf_page": 40,
                                                            "exact_bytes": None}
        history["official_sxm5_background_hbm_bandwidth_bytes_per_second"] = {
            "value": 3.35e12 if sources["h100_product"]["status"] == "CAPTURED" else None,
            "source_id": "h100_product", "unit": "byte/s", "note": "Current official rounded 3.35TB/s; whitepaper 3352GB/s table retains an unfinished-memory-clock qualifier"}
        history["official_peak_applicability"] = "SPECIFICATION_BACKGROUND_CONDITIONAL_ON_SXM5_MATCH_AND_STATED_CLOCKS; not a calibrated historical MFU denominator"
        result = {"schema": 1, "status": "INCOMPLETE_HARDWARE_ROOFS", "generated_utc": datetime.now(timezone.utc).isoformat(),
                  "h20": h20, "h100": history, "official_sources": official,
                  "official_search_audit": {"queries": ["site:nvidia.com H20 FP64 TF32", "site:nvidia.com H20 96GB", "site:nvidia.com H100 Tensor Core GPU FP32 TF32 FP64"],
                     "h20_peak_spec_found": False, "finding": "MIG guide identifies H20, line card lacks H20 throughput; no claim that no official spec exists elsewhere"},
                  "tool_sha256": Evidence.sha256(Path(__file__))}
        Evidence.write(output / "hardware.json", result)
        return 0

    @classmethod
    def compile(cls, session, discovery, output):
        if output.exists():
            raise FileExistsError(output)
        producer = session.root / "manifests/producers" / ("hardware_compile_" + output.stem)
        producer.mkdir(parents=True, exist_ok=False)
        for name in HardwareCLI.FILES:
            shutil.copy2(Path(__file__).parent / name, producer / name)
        result = json.loads(discovery.read_text())
        result["derived_from"] = {"path": str(discovery), "sha256": Evidence.sha256(discovery)}
        result["generated_utc"] = datetime.now(timezone.utc).isoformat()
        result["tool_sha256"] = Evidence.sha256(Path(__file__))
        result["producer_snapshot"] = {str(path): Evidence.sha256(path) for path in producer.iterdir()}
        result["h20"]["measured_calibrations"] = []
        selected = {}
        for path in sorted((session.root / "roofline/calibrations").glob("*/result.json")):
            row = json.loads(path.read_text())
            if row.get("status") != "PASS" or "summary" not in row:
                continue
            receipt_path = session.root / "logs" / (path.parent.name + ".json")
            receipt = json.loads(receipt_path.read_text())
            if receipt["status"] != "PASS" or receipt["returncode"] != 0:
                raise ValueError("Successful calibration payload lacks matching terminal process receipt")
            rate = row["event_based_achieved_rate"]
            calibration = {
                "kind": row["kind"], "GPU_UUID": row["GPU_UUID"], "achieved_rate": rate,
                "event_summary_ms": row["summary"]["event"], "wall_summary_ms": row["summary"]["wall"],
                "spread_within_10pct": row["summary"]["event"]["spread"] <= .1,
                "workload": row["workload"], "trace": row.get("trace"),
                "input_hashes": row["input_hashes"], "start_utc": receipt["start_utc"],
                "terminal_receipt": {"path": str(receipt_path), "sha256": Evidence.sha256(receipt_path)},
                "source": {"path": str(path), "sha256": Evidence.sha256(path)}}
            result["h20"]["measured_calibrations"].append(calibration)
            if row["kind"] not in selected or calibration["start_utc"] > selected[row["kind"]]["start_utc"]:
                selected[row["kind"]] = calibration
        result["h20"]["calibration_selection"] = {"policy": "LATEST_COMPLETE_ATTEMPT_PER_KIND_NOT_BEST_RATE", "selected": selected}
        result["h20"]["profiled_calibration_counters"] = []
        for path in sorted((session.root / "profiles/ncu").glob("hardware_*.raw.csv")):
            profile = CounterEvidence.parse(path)
            report_path = Path(str(path).removesuffix(".raw.csv") + ".ncu-rep")
            for sass_receipt in (session.root / "logs").glob("hardware_*_sass.json"):
                receipt = json.loads(sass_receipt.read_text())
                argv = receipt.get("argv", [])
                if receipt.get("status") == "PASS" and "--import" in argv and argv[argv.index("--import") + 1] == str(report_path):
                    sass = SassEvidence.parse(Path(receipt["stdout"]["path"]))
                    profile["sass_evidence"] = sass
                    if sass["main_pipeline"] == "FP32_FFMA" and len(profile["kernels"]) == 1:
                        profile["kernels"][0]["observed_arithmetic_pipeline"] = "FP32_FFMA_CUDA_CORE_WITH_AUXILIARY_CONSTANT_SYNTHESIS"
            derived_path = path.with_suffix(".derived.json")
            if not derived_path.exists():
                Evidence.write(derived_path, profile)
            result["h20"]["profiled_calibration_counters"].append({"path": str(derived_path), "sha256": Evidence.sha256(derived_path)})
        h20_raw = session.root / "profiles/ncu/luna020_sm512_fp64_ncu01.raw.csv"
        if h20_raw.exists():
            with h20_raw.open(newline="") as stream:
                data = list(csv.reader(stream))
            mapping = dict(zip(data[0], data[2]))
            result["h20"]["observed_device_attributes"].update({key: mapping[key] for key in
                ("device__attribute_multiprocessor_count", "device__attribute_total_constant_memory", "device__attribute_total_memory") if key in mapping})
        result["h20"]["physical_l2_bandwidth_bytes_per_second"] = cls.unknown(
            "Resident pointwise calibration has host launch overhead; both full20-group scans exceed10% spread. All attempts retained, no further reruns. No established physical L2 roof")
        result["h20"]["tf32_tensor_calibration"] = cls.unknown(
            "TF32x1 execution prohibited. No separate TF32x3 physical pipeline calibration has been completed; no H100 peak substitution")
        Evidence.write(output, result)
        return 0


class CounterEvidence:
    FACTORS = {"ns": (1e-9, "s"), "us": (1e-6, "s"), "ms": (1e-3, "s"), "s": (1., "s"),
               "byte": (1., "byte"), "Kbyte": (1e3, "byte"), "Mbyte": (1e6, "byte"), "Gbyte": (1e9, "byte"),
               "inst": (1., "inst"), "Kinst": (1e3, "inst"), "Minst": (1e6, "inst"), "Ginst": (1e9, "inst")}

    @classmethod
    def parse(cls, path):
        with path.open(newline="") as stream:
            records = list(csv.reader(stream))
        if len(records) < 3 or "Kernel Name" not in records[0]:
            raise ValueError("Expected NCU wide CSV header, units, then kernel rows")
        header, units = records[:2]
        kernels = []
        for values in records[2:]:
            if len(values) != len(header):
                raise ValueError("Malformed NCU wide row")
            metrics = {}
            for name, value, unit in zip(header, values, units):
                if name.startswith(("gpu__time_duration.", "dram__bytes_", "lts__t_bytes.", "smsp__sass_thread_inst_executed_op_", "sm__inst_executed_pipe_tensor")):
                    converted = None
                    if unit in cls.FACTORS:
                        factor, si_unit = cls.FACTORS[unit]
                        converted = {"value": float(value.replace(",", "")) * factor, "unit": si_unit}
                    metrics[name] = {"raw_value": value, "raw_unit": unit, "si": converted}
            def number(name):
                row = metrics.get(name, {}).get("si")
                return row["value"] if row else None
            ffma = number("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
            dfma = number("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum")
            tensor = number("sm__inst_executed_pipe_tensor.sum")
            dmma = number("sm__inst_executed_pipe_tensor_op_dmma.sum")
            pipeline = "UNKNOWN"
            if ffma is not None and ffma > 0 and tensor == 0:
                pipeline = "FP32_FFMA_CUDA_CORE"
            if dfma is not None and dfma > 0 and tensor == 0:
                pipeline = "FP64_DFMA_CUDA_CORE"
            if dmma is not None and dmma > 0:
                pipeline = "FP64_DMMA_TENSOR_CORE"
            seconds = number("gpu__time_duration.sum")
            rates = {}
            if seconds and seconds > 0:
                read, write = number("dram__bytes_read.sum"), number("dram__bytes_write.sum")
                if read is not None and write is not None:
                    rates["hbm_measured_bytes_per_second_profiled"] = (read + write) / seconds
                l2 = number("lts__t_bytes.sum")
                if l2 is not None:
                    rates["l2_measured_bytes_per_second_profiled"] = l2 / seconds
            kernels.append({"kernel": values[header.index("Kernel Name")], "metrics": metrics,
                            "observed_arithmetic_pipeline": pipeline, "profiled_rates": rates})
        return {"classification": "PROFILED_CALIBRATION_COUNTERS_NOT_MAIN_TIMING_OR_PHYSICAL_PEAK", "kernels": kernels,
                "source": {"path": str(path), "sha256": Evidence.sha256(path)},
                "warning": "Replay may alter caches; profiled rates describe this collected invocation only"}

    @classmethod
    def selfcheck(cls, output):
        output.mkdir(parents=True, exist_ok=False)
        fixture = output / "synthetic_ncu.csv"
        with fixture.open("w", newline="") as stream:
            csv.writer(stream).writerows([
                ["Kernel Name", "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum", "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "sm__inst_executed_pipe_tensor.sum"],
                ["", "ns", "Kbyte", "byte", "Ginst", "inst"], ["synthetic", "1000", "2", "48", "0.5", "0"]])
        kernel = cls.parse(fixture)["kernels"][0]
        assert kernel["metrics"]["gpu__time_duration.sum"]["si"]["unit"] == "s"
        assert math.isclose(kernel["metrics"]["gpu__time_duration.sum"]["si"]["value"], 1e-6, rel_tol=1e-15)
        assert kernel["metrics"]["dram__bytes_read.sum"]["si"] == {"value": 2000., "unit": "byte"}
        assert kernel["metrics"]["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"]["si"] == {"value": 5e8, "unit": "inst"}
        assert math.isclose(kernel["profiled_rates"]["hbm_measured_bytes_per_second_profiled"], 2048e6, rel_tol=1e-15)
        assert kernel["observed_arithmetic_pipeline"] == "FP32_FFMA_CUDA_CORE"
        Evidence.write(output / "result.json", {"classification": "CPU_SYNTHETIC_NOT_HARDWARE_MEASUREMENT", "status": "PASS", "assertions": 5})
        return 0


class SassEvidence:
    @staticmethod
    def parse(path):
        counts, auxiliary = {}, []
        for line in path.read_text().splitlines():
            match = re.match(r"0x[0-9a-f]+\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)*)", line)
            if match:
                opcode = match[1]
                counts[opcode] = counts.get(opcode, 0) + 1
                if opcode == "HFMA2.MMA":
                    auxiliary.append(line)
        true_mma = {op: count for op, count in counts.items() if op.startswith(("HMMA", "HGMMA", "WGMMA", "DMMA"))}
        zero_constant_aux = all(re.search(r"HFMA2\.MMA\s+R\d+,\s*-?RZ,\s*RZ,", line) for line in auxiliary)
        main = "UNKNOWN"
        if counts.get("FFMA", 0) and not true_mma and zero_constant_aux:
            main = "FP32_FFMA"
        return {"source": {"path": str(path), "sha256": Evidence.sha256(path)}, "static_opcode_counts": counts,
                "true_matrix_mma_opcodes": true_mma, "auxiliary_hfma2_mma_lines": auxiliary,
                "auxiliary_operands_are_zero_registers_and_constants": zero_constant_aux,
                "main_pipeline": main,
                "note": "Static opcode occurrence counts are not dynamic FLOP counts; HFMA2.MMA zero-register constant synthesis is not evidence of TF32 matrix multiplication"}


class HardwareTables:
    @staticmethod
    def render(manifest, output):
        output.mkdir(parents=True, exist_ok=False)
        hardware = json.loads(manifest.read_text())
        rows = []
        for row in hardware["h20"]["measured_calibrations"]:
            rows.append({"kind": row["kind"], "source": row["source"]["path"], "GPU_UUID": row["GPU_UUID"],
                         "start_utc": row["start_utc"], "rate_value": row["achieved_rate"]["value"], "rate_unit": row["achieved_rate"]["unit"],
                         "event_median_ms": row["event_summary_ms"]["median"], "event_spread": row["event_summary_ms"]["spread"],
                         "stable_within10pct": row["spread_within_10pct"],
                         "selected_latest_complete": hardware["h20"]["calibration_selection"]["selected"][row["kind"]]["source"]["path"] == row["source"]["path"]})
        counters = []
        for reference in hardware["h20"]["profiled_calibration_counters"]:
            profile = json.loads(Path(reference["path"]).read_text())
            if Evidence.sha256(Path(reference["path"])) != reference["sha256"]:
                raise ValueError("Derived counter evidence changed")
            for kernel in profile["kernels"]:
                counters.append({"source": profile["source"]["path"], "kernel": kernel["kernel"],
                                 "pipeline": kernel["observed_arithmetic_pipeline"], **kernel["profiled_rates"]})
        for name, table in (("calibration_summary.csv", rows), ("calibration_counter_summary.csv", counters)):
            path = output / name
            columns = sorted({key for row in table for key in row})
            with path.open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(table)
        producer = output / "producer"
        producer.mkdir()
        for name in HardwareCLI.FILES:
            shutil.copy2(Path(__file__).parent / name, producer / name)
        Evidence.write(output / "receipt.json", {"hardware_manifest": {"path": str(manifest), "sha256": Evidence.sha256(manifest)},
            "producer": {str(path): Evidence.sha256(path) for path in producer.iterdir()},
            "outputs": {str(path): Evidence.sha256(path) for path in output.glob("*.csv")}})
        return 0


class Microbenchmark:
    KINDS = ("sm_copy_hbm", "sm_copy_l2", "gemm_fp32", "gemm_fp64")

    def __init__(self, session, kind, output, ncu=False, trace=False):
        self.session, self.kind, self.output, self.ncu, self.trace = session, kind, output, ncu, trace
        if not output.resolve().is_relative_to(session.root):
            raise ValueError("Calibration must stay in session")
        output.mkdir(parents=True, exist_ok=False)

    @staticmethod
    def library_identity():
        files = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if ".so" in line and any(name in line for name in ("libcublas", "libtorch_cuda", "libcudart"))})
        return {filename: Evidence.sha256(Path(filename)) for filename in files if Path(filename).is_file()}

    def run(self):
        import torch
        from cases import Case
        from protocol import Statistics, Timer, TraceSummary

        source_hashes = {name: Evidence.sha256(Path(__file__).parent / name) for name in
                         ("hardware.py", "cases.py", "common.py", "protocol.py", "session.py")}
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(7753)
        properties = torch.cuda.get_device_properties()
        result = {"status": "RUNNING", "classification": "CALIBRATION_ATTAINABLE_LOWER_BOUND_NOT_PHYSICAL_PEAK",
                  "kind": self.kind, "GPU_UUID": str(properties.uuid), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "device_properties": repr(properties), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI, "input_seed": 7753, "input_range": [-0.5, 0.5],
                  "math": {"float32_matmul_precision": torch.get_float32_matmul_precision(), "allow_tf32": torch.backends.cuda.matmul.allow_tf32},
                  "cpu_affinity": sorted(os.sched_getaffinity(0)), "producer": source_hashes}
        Evidence.write(self.output / "result.json", result)
        l2_bytes = properties.L2_cache_size
        if self.kind.startswith("sm_copy"):
            size = 256 << 20 if self.kind == "sm_copy_hbm" else 16 << 20
            source = torch.empty(size // 4, device="cuda", dtype=torch.float32).uniform_(-0.5, 0.5)
            target = torch.empty_like(source)
            function = lambda: torch.add(source, 0.0, out=target)
            function()
            torch.cuda.synchronize()
            if not torch.equal(source, target):
                raise RuntimeError("SM copy validation failed")
            workload = {"input_bytes": size, "output_bytes": size, "working_set_bytes": 2 * size,
                        "working_set_over_L2": 2 * size / l2_bytes, "L2_capacity_bytes": l2_bytes,
                        "logical_payload_bytes_per_call": 2 * size, "operator": "torch.add(source,0,out=target)",
                        "note": "One SM elementwise load/add-zero/store kernel; logical payload requires counters before any HBM/L2 roof interpretation"}
            inputs = {"source": Case.tensor_hash(source)}
            layouts = {"source": {"shape": list(source.shape), "stride": list(source.stride()), "dtype": str(source.dtype)}}
        else:
            dtype = torch.float32 if self.kind == "gemm_fp32" else torch.float64
            n = 4096
            left = torch.empty((n, n), device="cuda", dtype=dtype).uniform_(-0.5, 0.5)
            right = torch.empty_like(left).uniform_(-0.5, 0.5)
            target = torch.empty_like(left)
            function = lambda: torch.mm(left, right, out=target)
            function()
            torch.cuda.synchronize()
            gold = left[:8].double() @ right[:, :8].double()
            relative = ((target[:8, :8].double() - gold).norm() / gold.norm()).item()
            limit = 1e-5 if dtype == torch.float32 else 1e-12
            if not bool(torch.isfinite(target).all()) or relative > limit:
                raise RuntimeError(f"Calibration GEMM sample validation failed: {relative} > {limit}")
            workload = {"M": n, "N": n, "K": n, "flops_per_call": 2 * n ** 3,
                        "dtype": str(dtype), "operator": "torch.mm(left,right,out=target)",
                        "validation": {"sample": "output[:8,:8] against FP64 GEMM of matching input slices", "rel_fro": relative, "limit": limit},
                        "actual_pipeline": "UNVERIFIED_UNTIL_TRACE_NCU_OR_SASS", "peak_interpretation": "achievable cuBLAS path at stated size; never automatic CUDA-core/DMMA peak"}
            inputs = {"left": Case.tensor_hash(left), "right": Case.tensor_hash(right)}
            layouts = {name: {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype)}
                       for name, tensor in (("left", left), ("right", right))}
        result.update(workload=workload, input_hashes=inputs, layouts=layouts)
        # Pilot determines batching before the 20 groups; all event/host/wall samples persist.
        pilot_timer = Timer(1)
        pilot = [pilot_timer.sample(function) for _ in range(3)]
        calls = 10 if statistics.median(row["wall"] for row in pilot) < 0.2 else 1
        result.update(pilot=pilot, calls_per_group=calls, warmup=Timer.warmup(function, limit=20))
        Evidence.write(self.output / "result.json", result)
        if self.ncu:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            function()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            result["profile_window_calls"] = 1
            result["calibration_samples_measured"] = False
        else:
            timer = Timer(calls)
            result["raw_groups"] = []
            for group in range(20):
                result["raw_groups"].append({"group": group, **timer.sample(function)})
                Evidence.write(self.output / "result.json", result)
            result["summary"] = {metric: Statistics.summarize([row[metric] for row in result["raw_groups"]]) for metric in ("wall", "event", "host")}
            event_seconds = result["summary"]["event"]["median"] / 1e3
            work = workload.get("flops_per_call", workload.get("logical_payload_bytes_per_call"))
            result["event_based_achieved_rate"] = {"value": work / event_seconds,
                "unit": "flop/s" if "flops_per_call" in workload else "logical_payload_byte/s",
                "interpretation": "ATTAINABLE_UNDER_RECORDED_PROTOCOL; not physical roof; copy layer needs independently measured traffic"}
            if self.trace:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                    function()
                    torch.cuda.synchronize()
                trace_path = self.output / "kernel.trace.json"
                profile.export_chrome_trace(str(trace_path))
                result["trace"] = {"path": str(trace_path), "sha256": Evidence.sha256(trace_path),
                    "kernels": [{"name": row["name"], "duration_us_profiled": row["dur"]} for row in TraceSummary.kernels(json.loads(trace_path.read_text()))]}
        result["libraries"] = self.library_identity()
        result["producer_unchanged"] = all(Evidence.sha256(Path(__file__).parent / name) == digest for name, digest in source_hashes.items())
        result["status"] = "PASS" if result["producer_unchanged"] else "INVALID_PRODUCER_CHANGED_DURING_RUN"
        Evidence.write(self.output / "result.json", result)
        return 0 if result["status"] == "PASS" else 1


class HardwareCLI:
    FILES = ("hardware.py", "cases.py", "common.py", "protocol.py", "session.py")

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", choices=("discover", "compile", "tables", "selfcheck", "launch", "measure"))
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--output", type=Path)
        parser.add_argument("--kind", choices=Microbenchmark.KINDS)
        parser.add_argument("--gpu", type=int)
        parser.add_argument("--attempt")
        parser.add_argument("--ncu", action="store_true")
        parser.add_argument("--trace", action="store_true")
        parser.add_argument("--discovery", type=Path)
        parser.add_argument("--manifest", type=Path)
        args = parser.parse_args()
        session = Session(args.session)
        if args.command == "tables":
            if args.manifest is None or args.output is None or not args.output.resolve().is_relative_to(session.root):
                parser.error("tables requires --manifest and fresh --output within session")
            return HardwareTables.render(args.manifest, args.output)
        if args.command == "selfcheck":
            if args.output is None or not args.output.resolve().is_relative_to(session.root):
                parser.error("selfcheck requires fresh --output within session")
            return CounterEvidence.selfcheck(args.output)
        if args.command == "compile":
            if args.discovery is None or args.output is None or not args.output.resolve().is_relative_to(session.root):
                parser.error("compile requires --discovery and fresh --output within session")
            return HardwareInventory.compile(session, args.discovery, args.output)
        if args.command == "discover":
            if args.output is None or not args.output.resolve().is_relative_to(session.root):
                parser.error("discover requires fresh --output within session")
            return HardwareInventory.discover(session, args.output)
        if args.kind is None:
            parser.error("calibration requires --kind")
        if args.command == "measure":
            if args.output is None:
                parser.error("measure requires --output")
            return Microbenchmark(session, args.kind, args.output, args.ncu, args.trace).run()
        if args.gpu is None or not args.attempt or Path(args.attempt).name != args.attempt:
            parser.error("launch requires --gpu and simple fresh --attempt")
        name = "hardware_" + args.kind + "_" + args.attempt
        producer = session.root / "manifests/producers" / name
        producer.mkdir(parents=True, exist_ok=False)
        for filename in cls.FILES:
            shutil.copy2(Path(__file__).parent / filename, producer / filename)
        Evidence.write(producer / "SHA256.json", {name: Evidence.sha256(producer / name) for name in cls.FILES})
        try:
            gpu = session.idle_gpu(name, args.gpu)
        except RuntimeError as error:
            Evidence.write(session.root / "logs" / (name + "_launch.json"), {"status": "RESOURCE_UNAVAILABLE_NOT_STARTED", "error": str(error),
                "producer": str(producer), "idle_samples": str(session.root / "env" / (name + "_idle_samples.json"))})
            raise
        output = session.root / "roofline/calibrations" / name
        argv = [session.paths["FASTEQ_PY"], "-B", str(producer / "hardware.py"), "measure", "--session", str(session.root),
                "--kind", args.kind, "--output", str(output)]
        if args.trace:
            argv.append("--trace")
        if args.ncu:
            argv.append("--ncu")
            metrics = "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_tensor_op_dmma.sum"
            argv = ["ncu", "--profile-from-start", "off", "--cache-control", "none", "--clock-control", "none", "--metrics", metrics,
                    "--export", str(session.root / "profiles/ncu" / name), *argv]
        receipt = session.runner().run(name, argv, 120, Path(session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": gpu})
        if args.ncu and receipt["status"] == "PASS":
            base = session.root / "profiles/ncu" / name
            exported = session.runner().run(name + "_raw_import", ["ncu", "--import", str(base) + ".ncu-rep", "--page", "raw", "--csv"], 30)
            if exported["status"] == "PASS":
                shutil.copy2(Path(exported["stdout"]["path"]), Path(str(base) + ".raw.csv"))
                Evidence.write(Path(str(base) + ".receipt.json"), {"report": str(base) + ".ncu-rep",
                    "report_sha256": Evidence.sha256(Path(str(base) + ".ncu-rep")), "raw_csv": str(base) + ".raw.csv",
                    "raw_csv_sha256": Evidence.sha256(Path(str(base) + ".raw.csv")), "calibration_result": str(output / "result.json"),
                    "metrics": metrics, "cache_control": "none", "clock_control": "none"})
        return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(HardwareCLI.main())
