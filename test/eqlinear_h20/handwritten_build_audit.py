"""Independent CPU readback of source-preserving handwritten build campaigns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from common import Evidence


class HandwrittenBuildAudit:
    def __init__(self, session_root, campaign):
        self.session, self.campaign = Path(session_root), Path(campaign)

    def verify(self, name):
        root = self.session / "build/handwritten" / name
        manifest_path, source_path = root / "build.json", root / "source.json"
        build, source = json.loads(manifest_path.read_text()), json.loads(source_path.read_text())
        if build["source_manifest_sha256"] != Evidence.sha256(source_path):
            raise ValueError("Build source receipt drift")
        for key, digest in source["original_files"].items():
            if Evidence.sha256(Path(source["archive"]) / key) != digest:
                raise ValueError("Original preserved source drift")
        for key, digest in source["staged_files"].items():
            if Evidence.sha256(root / "source" / key) != digest:
                raise ValueError("Staged source drift")
        for patch in source["registration_patches"]:
            before = (Path(source["archive"]) / patch["file"]).read_text()
            after = (root / "source" / patch["file"]).read_text()
            pattern = r"\bTORCH_LIBRARY\s*\(\s*[A-Za-z_]\w*\s*,"
            if re.sub(pattern, "TORCH_LIBRARY(NAMESPACE,", before) != re.sub(pattern, "TORCH_LIBRARY(NAMESPACE,", after):
                raise ValueError("A patch changed more than the registration namespace")
        receipt_path = self.session / "logs" / ("handwritten_" + name + ".json")
        receipt = json.loads(receipt_path.read_text())
        for stream in ("stdout", "stderr"):
            if Evidence.sha256(Path(receipt[stream]["path"])) != receipt[stream]["sha256"]:
                raise ValueError("Build log drift")
        ninja = root / "binary/build.ninja"
        record = {"name": name, "status": build["status"], "source_manifest": str(source_path),
            "source_manifest_sha256": Evidence.sha256(source_path), "build_manifest": str(manifest_path),
            "build_manifest_sha256": Evidence.sha256(manifest_path), "process_receipt": str(receipt_path),
            "process_receipt_sha256": Evidence.sha256(receipt_path), "registration_namespace": source["namespace"],
            "source_patches_registration_only": True, "original_sources_unchanged": True,
            "torch": build["torch"], "torch_cuda": build["torch_cuda"], "cxx11_abi": build["cxx11_abi"],
            "flags": build["flags"], "GPU_runtime_status": "UNTESTED", "gpu_execution": False}
        if ninja.exists():
            record["ninja"] = {"path": str(ninja), "sha256": Evidence.sha256(ninja)}
        if build["status"] == "PASS":
            binary = Path(build["binary"]["path"])
            if receipt["status"] != "PASS" or Evidence.sha256(binary) != build["binary"]["sha256"]:
                raise ValueError("Successful source build lacks matching process/DSO receipt")
            elf = subprocess.check_output(["cuobjdump", "--list-elf", str(binary)], text=True, timeout=30)
            dynamic = subprocess.check_output(["readelf", "-d", str(binary)], text=True, timeout=30)
            if sorted(set(re.findall(r"sm_\d+a?", elf))) != ["sm_90a"] or source["namespace"] + ".so" not in dynamic:
                raise ValueError("Independent architecture/SONAME guard failed")
            if not ninja.exists() or "-gencode=arch=compute_90a,code=sm_90a" not in ninja.read_text():
                raise ValueError("Missing explicit actual sm90a compile command")
            record.update(binary=build["binary"], architectures=["sm_90a"], unique_soname=source["namespace"] + ".so")
        elif build["status"] != "BUILD_FAILED" or receipt["status"] != "FAIL":
            raise ValueError("Unrecognized/partial/timeout build state requires diagnosis")
        else:
            record.update(compiler_error=build.get("error"), classification="PRESERVED_SOURCE_BUILD_FAILURE; not hardware unsupported")
        return record

    def run(self, output, original=None):
        plan = json.loads((self.campaign / "plan.json").read_text())
        results = json.loads((self.campaign / "results.json").read_text())
        if results.get("complete") is not True or {row["kind"] for row in results["results"]} != set(plan["kinds"]):
            raise ValueError("Ten-file campaign not terminal complete")
        producer = Path(plan["producer"])
        if Evidence.sha256(producer / "producer.json") != plan["producer_manifest_sha256"]:
            raise ValueError("CPU build producer manifest drift")
        for name, digest in json.loads((producer / "producer.json").read_text()).items():
            if Evidence.sha256(producer / name) != digest:
                raise ValueError("CPU build producer source drift")
        headers_path = self.campaign / "cutlass_headers.json"
        if Evidence.sha256(headers_path) != plan["headers_manifest_sha256"]:
            raise ValueError("External include manifest drift")
        headers = json.loads(headers_path.read_text())
        for name, digest in headers["files"].items():
            if Evidence.sha256(Path(headers["root"]) / name) != digest:
                raise ValueError("External CUTLASS header drift")
        records = [self.verify(kind + "_build01") for kind in plan["kinds"]]
        if original:
            records.append(self.verify(original))
        if len({record["registration_namespace"] for record in records}) != len(records):
            raise ValueError("Historical handwritten namespaces collide")
        value = {"classification": "INDEPENDENT_CPU_SOURCE_BUILD_READBACK_NOT_NUMERICAL_GPU_COVERAGE",
            "complete": True, "records": records, "build_pass": sum(row["status"] == "PASS" for row in records),
            "preserved_source_build_failure": sum(row["status"] == "BUILD_FAILED" for row in records),
            "campaign": str(self.campaign), "campaign_results_sha256": Evidence.sha256(self.campaign / "results.json"),
            "external_headers": {"path": str(headers_path), "sha256": Evidence.sha256(headers_path), "verified_files": len(headers["files"])},
            "gpu_execution": False, "runtime_accuracy": "PENDING_SHARED_PROBE", "original_historical_binary_reproduction": "UNPROVEN"}
        if output.exists():
            raise FileExistsError(output)
        Evidence.write(output, value)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-build")
    parser.add_argument("--wait", type=int, default=0)
    args = parser.parse_args()
    start = time.monotonic()
    while args.wait:
        results = args.campaign / "results.json"
        complete = results.exists() and json.loads(results.read_text()).get("complete")
        if args.original_build:
            complete = complete and (args.session / "logs" / ("handwritten_" + args.original_build + ".json")).exists()
        if complete:
            break
        if time.monotonic() - start >= args.wait:
            raise TimeoutError("Build campaign still incomplete; no build was restarted")
        time.sleep(2)
    value = HandwrittenBuildAudit(args.session, args.campaign).run(args.output, args.original_build)
    print(json.dumps({key: value[key] for key in ("complete", "build_pass", "preserved_source_build_failure")}))


if __name__ == "__main__":
    main()
