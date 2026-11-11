"""Rebuild the original EqLinear binding and native files without running kernels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from common import Evidence
from handwritten import HandwrittenSource
from producer_closure import ProducerClosure
from session import Session


class OriginalCudaSource(HandwrittenSource):
    def __init__(self, session, archive, name):
        if Path(name).name != name:
            raise ValueError("Use a simple isolated build name")
        self.session, self.kind = session, "original_cuda_fp64_4463b10"
        self.archive = Path(archive)
        manifest = json.loads((self.archive / "source_manifest.json").read_text())
        if manifest["ref"] != "4463b10c1fa290f76bd2cbb8213a6e9377a34e78":
            raise ValueError("Wrong original source ref")
        for path, digest in manifest["files"].items():
            if Evidence.sha256(self.archive / path) != digest:
                raise ValueError("Original staged history source drift: " + path)
        self.root = session.root / "build/handwritten" / name
        folder = self.archive / "fasteq/cuda/src"
        self.sources = [folder / "equi_linear.cu"] + [folder / "equi_linear_kernels" / path for path in
            ("equi_linear_fwd_f64.cu", "equi_linear_fwd_f32.cu", "equi_linear_bwd_f32.cu")]

    def stage(self):
        value = super().stage()
        value.update(historical_source_manifest={"path": str(self.archive / "source_manifest.json"),
                     "sha256": Evidence.sha256(self.archive / "source_manifest.json")},
                     runtime_policy={"allowed_dtype": "float64", "forbidden_dtype": "float32",
                         "f32_sources": "Compiled only to resolve the unchanged original binding; MUST NOT execute its TF32x1 branches",
                         "forward": "Original double native P4 only", "backward_x": "Original double W-transpose plus same f64 helper",
                         "required_adapter_guard": "Reject non-float64 before either original registered operator"})
        Evidence.write(self.root / "source.json", value)
        return value


class SequentialBuild:
    def __init__(self, session, campaign, producer, archive, name):
        self.session, self.campaign, self.producer, self.archive, self.name = session, campaign, producer, archive, name

    def run(self):
        start = time.monotonic()
        results = self.campaign / "results.json"
        while not results.is_file() or not json.loads(results.read_text()).get("complete"):
            if time.monotonic() - start > 6600:
                raise TimeoutError("Prior ten-file campaign not complete; original build not started")
            time.sleep(2)
        for path, digest in json.loads((self.producer / "producer.json").read_text()).items():
            if Evidence.sha256(self.producer / path) != digest:
                raise ValueError("Frozen original build producer drift")
        value = self.session.runner().run("handwritten_" + self.name,
            [self.session.paths["FASTEQ_PY"], "-B", str(self.producer / "legacy_cuda_build.py"), "build",
             "--session", str(self.session.root), "--archive", str(self.archive), "--name", self.name],
            600, Path(self.session.paths["FASTEQ_ROOT"]), {"CUDA_VISIBLE_DEVICES": "", "MAX_JOBS": "2"})
        Evidence.write(self.producer.parent / "result.json", {"receipt": value, "prior_campaign_results": str(results),
            "prior_campaign_results_sha256": Evidence.sha256(results), "gpu_execution": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "wait", "build"))
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--producer", type=Path)
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        args.producer.mkdir(parents=True, exist_ok=False)
        Evidence.write(args.producer / "producer.json", ProducerClosure(Path(__file__).parent, args.producer).freeze(["legacy_cuda_build.py"]))
    elif args.action == "wait":
        SequentialBuild(Session(args.session), args.campaign, args.producer, args.archive, args.name).run()
    else:
        return OriginalCudaSource(Session(args.session), args.archive, args.name).build()


if __name__ == "__main__":
    raise SystemExit(main())
