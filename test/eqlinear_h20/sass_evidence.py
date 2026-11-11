"""Static instructions of an exact kernel actually present in a saved Torch trace."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shutil

from common import Evidence
from session import Session


class SassEvidence:
    @classmethod
    def run(cls, args):
        profile = json.loads(args.profile.read_text())
        if not profile.get("profiled") or profile.get("mode") != "torch":
            raise ValueError("A real Torch trace profile is required")
        trace = profile["trace"]
        if Evidence.sha256(Path(trace["path"])) != trace["sha256"]:
            raise ValueError("Trace changed")
        kernels = [key for key in trace["kernel_names"] if key.startswith("_Z")]
        symbol_evidence = None
        if not kernels and args.ncu_collection is not None:
            collection = json.loads(args.ncu_collection.read_text())
            for key in ("result", "report", "raw"):
                record = collection[key]
                if Evidence.sha256(Path(record["path"])) != record["sha256"]:
                    raise ValueError("NCU symbol evidence changed")
            ncu_profile = json.loads(Path(collection["result"]["path"]).read_text())
            if any(profile[key] != ncu_profile[key] for key in ("point", "call", "GPU_UUID", "input_hash", "reference_hashes", "profile_calls")):
                raise ValueError("Torch/NCU profile invocations are not the same case/call/device")
            if profile["binding"]["binary_hash"] != ncu_profile["binding"]["binary_hash"]:
                raise ValueError("Torch/NCU actual binaries differ")
            session = Session(args.session)
            exported = session.runner().run(args.name + "_mangled_symbols",
                ["ncu", "--import", collection["report"]["path"], "--page", "raw", "--csv", "--print-kernel-base", "mangled"], 30)
            if exported["status"] != "PASS":
                raise RuntimeError("NCU exact symbol export failed")
            with Path(exported["stdout"]["path"]).open(newline="") as stream:
                records = list(csv.reader(stream))
            column = records[0].index("Kernel Name")
            kernels = [row[column] for row in records[2:]]
            if not kernels or any(not value.startswith("_Z") for value in kernels) or len(kernels) != trace["kernel_count"]:
                raise ValueError("Exact NCU symbols are absent or pipeline launch counts differ")
            kernels = list(dict.fromkeys(kernels))
            symbol_evidence = {"collection": str(args.ncu_collection), "collection_sha256": Evidence.sha256(args.ncu_collection),
                               "exact_mangled_export": exported["stdout"]}
        if args.kernel is not None:
            if args.kernel not in trace["kernel_names"]:
                raise ValueError("Requested function was not actually observed in this trace")
            kernels = [args.kernel]
        if not kernels:
            raise ValueError("No exact mangled CUDA kernel symbol observed; do not invent a symbol mapping")
        binding = profile["binding"]
        known = {str(Path(key).resolve()): value["sha256"] for key, value in binding.get("binary_files", {}).items()}
        known.update({str(Path(value.get("installed_path", value.get("path"))).resolve()): value["sha256"]
                      for value in binding.get("binaries", [])})
        binary = args.binary.resolve()
        if str(binary) not in known or Evidence.sha256(binary) != known[str(binary)]:
            raise ValueError("Binary is not the exact file bound to this executed call")
        session = Session(args.session)
        directory = session.root / "profiles/sass" / args.name
        directory.mkdir(parents=True, exist_ok=False)
        results = []
        for index, kernel in enumerate(kernels):
            terminal = session.runner().run(args.name + "_sass" + str(index),
                ["cuobjdump", "--dump-sass", "--gpu-architecture", "sm_90a", "--function", kernel, str(binary)], args.deadline)
            if terminal["status"] != "PASS":
                raise RuntimeError("cuobjdump failed; original output retained")
            target = directory / ("kernel" + str(index) + ".sass.txt")
            shutil.copyfile(terminal["stdout"]["path"], target)
            text = target.read_text()
            found = ("Function : " + kernel) in text
            instructions = Counter(re.findall(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)", text))
            results.append({"kernel": kernel, "symbol_found_in_bound_binary": found,
                "sass": {"path": str(target), "sha256": Evidence.sha256(target)},
                "static_instruction_counts": dict(instructions),
                "arithmetic_mnemonics": {key: count for key, count in instructions.items()
                    if key.split(".")[0] in ("HMMA", "HGMMA", "DMMA", "FFMA", "DFMA", "FADD", "FMUL", "DADD", "DMUL")},
                "local_memory_mnemonics": {key: count for key, count in instructions.items() if key.split(".")[0] in ("LDL", "STL")}})
        unchanged = Evidence.sha256(binary) == known[str(binary)]
        result = {"status": "PASS" if unchanged and all(value["symbol_found_in_bound_binary"] for value in results) else "SYMBOL_MAPPING_UNVERIFIED",
            "profile": {"path": str(args.profile), "sha256": Evidence.sha256(args.profile)},
            "trace": {"path": trace["path"], "sha256": trace["sha256"]}, "GPU_UUID": profile["GPU_UUID"],
            "binary": {"path": str(binary), "sha256": known[str(binary)], "unchanged_after_dump": unchanged},
            "requested_architecture": "sm_90a", "kernels": results, "tool_sha256": Evidence.sha256(Path(__file__)),
            "ncu_exact_symbol_evidence": symbol_evidence,
            "interpretation": ["Static opcode counts are not executed instruction counts or algorithm product counts",
                "The exact kernel was launched in the linked trace; counters are needed to establish active arithmetic paths",
                "LDL/STL or local traffic alone do not prove compiler register spilling"]}
        Evidence.write(directory / "result.json", result)
        print(json.dumps({"status": result["status"], "functions": len(results), "result": str(directory / "result.json")}))
        return 0 if result["status"] == "PASS" else 1

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--profile", type=Path, required=True)
        parser.add_argument("--binary", type=Path, required=True)
        parser.add_argument("--kernel")
        parser.add_argument("--ncu-collection", type=Path, help="Use real NCU mangled symbols when Torch displays only demangled names")
        parser.add_argument("--name", required=True)
        parser.add_argument("--deadline", type=float, default=90.)
        return cls.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(SassEvidence.main())
