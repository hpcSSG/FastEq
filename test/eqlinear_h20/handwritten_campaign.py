"""Bounded CPU-only build campaign; failed sources stay unchanged and auditable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import Evidence
from handwritten import HandwrittenSource
from producer_closure import ProducerClosure
from session import Session


class HandwrittenCampaign:
    def __init__(self, session, root):
        self.session, self.root = session, root

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=False)
        producer = self.root / "producer"
        producer.mkdir()
        closure = ProducerClosure(Path(__file__).parent, producer).freeze(["handwritten_campaign.py", "handwritten.py"])
        Evidence.write(producer / "producer.json", closure)
        cutlass = Path(self.session.paths["OPT_ROOT"]) / "3rdparty/cutlass"
        includes = [cutlass / "include", cutlass / "tools/util/include"]
        headers = {str(p.relative_to(cutlass)): Evidence.sha256(p) for d in includes for p in sorted(d.rglob("*")) if p.is_file()}
        Evidence.write(self.root / "cutlass_headers.json", {"root": str(cutlass), "files": headers, "fingerprint": Evidence.fingerprint(headers)})
        plan = {"session": str(self.session.root), "producer": str(producer), "producer_manifest_sha256": Evidence.sha256(producer / "producer.json"),
                "include_roots": [str(p) for p in includes], "headers_manifest_sha256": Evidence.sha256(self.root / "cutlass_headers.json"),
                "kinds": ["fp64_" + n for n in HandwrittenSource.FP64], "deadline_per_build_seconds": 600,
                "scope": "FRESH_H20_CPU_BUILD_AND_TORCH_REGISTRATION_ONLY; no GPU kernels and no historical binary reproduction claim"}
        Evidence.write(self.root / "plan.json", plan)

    def run(self):
        plan = json.loads((self.root / "plan.json").read_text())
        producer = Path(plan["producer"])
        if Evidence.sha256(producer / "producer.json") != plan["producer_manifest_sha256"]:
            raise ValueError("Producer manifest drift")
        manifest = json.loads((producer / "producer.json").read_text())
        for name, record in manifest.items():
            digest = record["sha256"] if isinstance(record, dict) else record
            if Evidence.sha256(producer / name) != digest:
                raise ValueError("Frozen producer drift: " + name)
        headers_path = self.root / "cutlass_headers.json"
        if Evidence.sha256(headers_path) != plan["headers_manifest_sha256"]:
            raise ValueError("CUTLASS manifest drift")
        headers = json.loads(headers_path.read_text())
        for name, digest in headers["files"].items():
            if Evidence.sha256(Path(headers["root"]) / name) != digest:
                raise ValueError("CUTLASS header drift: " + name)
        env = {"CUDA_VISIBLE_DEVICES": "", "MAX_JOBS": "2", "CPATH": ":".join(plan["include_roots"])}
        results = []
        for kind in plan["kinds"]:
            name = kind + "_build01"
            receipt = self.session.runner().run("handwritten_" + name,
                [self.session.paths["FASTEQ_PY"], "-B", str(producer / "handwritten.py"), "--session", str(self.session.root), "--kind", kind, "--name", name],
                plan["deadline_per_build_seconds"], Path(self.session.paths["FASTEQ_ROOT"]), env)
            build = self.session.root / "build/handwritten" / name / "build.json"
            results.append({"kind": kind, "receipt": receipt, "build_manifest": str(build),
                            "build_manifest_sha256": Evidence.sha256(build) if build.exists() else None})
            Evidence.write(self.root / "results.json", {"results": results, "complete": len(results) == len(plan["kinds"]), "gpu_execution": False})
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    campaign = HandwrittenCampaign(Session(args.session), args.root)
    return campaign.prepare() if args.action == "prepare" else campaign.run()


if __name__ == "__main__":
    raise SystemExit(main())
