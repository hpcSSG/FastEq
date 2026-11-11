"""Append the user's final comparison scope without rewriting discovery evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from common import Evidence


class FinalScopeDecision:
    REMOVED = {"handwritten_tf32", "handwritten_fp64", "handwritten_legacy_tf32", "historical_original_cuda",
               "historical_triton", "standalone_triton"}
    INSTRUCTION = "这些（手写CUDA、Triton）不用接入了。基于当前已有适配，直接做最终的对比测试"

    @classmethod
    def write(cls, session, parent, output):
        if not output.resolve().is_relative_to(session.resolve()):
            raise ValueError("Final scope must stay in the configured session")
        digest = Evidence.sha256(parent)
        value = copy.deepcopy(json.loads(parent.read_text()))
        removed = []
        for row in value["variants"]:
            if row["family"] in cls.REMOVED:
                row["final_comparison_scope"] = "REMOVED_BY_USER_SCOPE_REDUCTION"
                row["final_comparison_reason"] = "User chose existing adapters only and removed handwritten CUDA/original CUDA handwritten entries and all legacy/standalone Triton from the final comparison; no hardware capability conclusion"
                removed.append({"variant_id": row["variant_id"], "family": row["family"], "prior_scope_status": row["scope_status"]})
            else:
                row["final_comparison_scope"] = "RETAIN_EXISTING_ADAPTER_SCOPE" if row["scope_status"] == "IN_SCOPE" else "PRIOR_USER_EXCLUSION_RETAINED"
        output.mkdir(parents=True, exist_ok=False)
        decision = {"classification": "IMMUTABLE_USER_FINAL_SCOPE_DECISION_NOT_CAPABILITY_RESULT", "utc": datetime.now(timezone.utc).isoformat(),
            "instruction": cls.INSTRUCTION, "authorization": "User explicitly resumed and reduced scope; instruction relayed by root",
            "parent_registry": {"path": str(parent), "sha256": digest}, "removed_families": sorted(cls.REMOVED),
            "removed_rows": removed, "counts": dict(Counter(row["final_comparison_scope"] for row in value["variants"])),
            "historical_existing_adapters_retained": {"wrappers": 63, "points": 1575, "accuracy_paths": 7875},
            "new_adapters_or_parameters_authorized": False, "old_evidence_deleted_or_resigned": False,
            "prior_scope_yaml_unchanged_for_frozen_epoch_compatibility": True}
        value["final_user_scope_decision"] = {"path": str(output / "decision.json"), "scope": "Final comparison only; original discovery and runtime facts retained"}
        Evidence.write(output / "decision.json", decision)
        Evidence.write(output / "registry.json", value)
        if Evidence.sha256(parent) != digest:
            raise ValueError("Parent registry changed")
        return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = FinalScopeDecision.write(args.session, args.parent, args.output)
    print(json.dumps(value["counts"]))


if __name__ == "__main__":
    main()
