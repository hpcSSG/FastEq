"""Independent recorded-metric gates and source-bound upstream accuracy contracts."""
from __future__ import annotations

import json
import math
from pathlib import Path

from common import Evidence


class RecordedAccuracy:
    """Recompute qualification from recorded metrics, independently of GPU evaluation."""
    @classmethod
    def passes(cls, value, precision):
        if "components" in value:
            return bool(value["components"]) and all(cls.passes(component, component.get("accuracy_tier", precision))
                                                     for component in value["components"].values())
        precision = value.get("accuracy_tier", precision)
        baseline, relative = value.get("torch_rel_fro"), value.get("rel_fro")
        valid_baseline = baseline is not None and math.isfinite(baseline) and baseline >= 0
        limit = 1e-12 if precision == "fp64" else 2e-5 if precision in ("tf32", "tf32x3") else 1.05*baseline if valid_baseline else None
        members = value.get("members")
        members_pass = True
        if members is not None:
            count = value.get("member_count")
            members_pass = (isinstance(members, dict) and type(count) is int and count > 0
                and set(members) == {str(index) for index in range(count)}
                and value.get("member_output_addresses_unique") is True
                and all(members[str(index)].get("index") == index for index in range(count)))
            if members_pass:
                members_pass = all(cls.passes(member, precision) for member in members.values())
        return (members_pass and valid_baseline and value.get("torch_baseline_valid", True) is True and value.get("threshold") == limit
            and limit is not None and relative is not None and math.isfinite(relative) and relative <= limit
            and all(value.get(key) is True for key in ("finite", "reference_finite", "repeat_equal", "different_buffer"))
            and (not value.get("zero_reference") or value.get("max_abs") == 0 and value.get("mean_abs") == 0))


class AccuracyContracts:
    """A local component pass cannot overrule a failed full-operation accuracy contract."""
    cache = {}

    @classmethod
    def read(cls, path, digest):
        path = Path(path)
        stat = path.stat()
        key = str(path.resolve()), digest
        identity = stat.st_size, stat.st_mtime_ns
        if key not in cls.cache or cls.cache[key][0] != identity:
            if Evidence.sha256(path) != digest:
                raise ValueError("Accuracy contract evidence changed: " + str(path))
            cls.cache[key] = identity, json.loads(path.read_text())
        return cls.cache[key][1]

    @staticmethod
    def rows(result):
        for key in ("rows", "operation_boundaries", "boundaries"):
            if isinstance(result.get(key), list):
                return result[key]
        raise ValueError("Unrecognized upstream accuracy result schema")

    @staticmethod
    def identity(source):
        identity = source["identity"]
        if "source_hash" in identity and "binary_hash" in identity:
            return dict(identity)
        if identity.get("family") == "current_native_selector":
            precision, impl = identity.get("precision"), identity.get("impl")
            allowed = {"tf32": {0, 3, 4, 5, 6, 7, 8}, "fp32": set(range(4)), "fp64": set(range(8))}
            if (precision not in allowed or type(impl) is not int or impl not in allowed[precision]
                    or identity.get("binary") not in identity.get("mapped_libraries", [])
                    or identity.get("native_math_modified") is not False
                    or identity.get("production_DSO_loaded") is not False):
                raise ValueError("Native selector identity is excluded, unmapped, or changed math")
            manifest = AccuracyContracts.read(identity["manifest"], identity["manifest_sha256"])
            if (manifest.get("status") != "PASS" or manifest.get("binary") != identity["binary"]
                    or manifest.get("binary_sha256") != identity.get("binary_sha256")
                    or manifest.get("namespace") != identity.get("namespace")
                    or manifest["contract"]["source_commit"] != identity.get("source_commit")
                    or Evidence.fingerprint(manifest["contract"]["native_files"]) != identity.get("native_objects_sha256")):
                raise ValueError("Native selector identity differs from exact fresh-build manifest")
            return {**identity, "source_hash": Evidence.fingerprint({key: identity[key] for key in ("source_commit", "native_objects_sha256")}),
                    "binary_hash": identity["binary_sha256"], "api_controls": {}}
        if (isinstance(source.get("source_manifest_hash"), str) and identity.get("python_source_hash")
                and isinstance(identity.get("jit_libraries"), dict) and identity["jit_libraries"]):
            if identity.get("backend") != "cuda_fused" or not set(identity["jit_libraries"]) <= set(identity.get("mapped_jit_libraries", [])):
                raise ValueError("Luna upstream evidence is not an actually mapped fused CUDA backend")
            return {**identity, "source_hash": source["source_manifest_hash"],
                    "binary_hash": Evidence.fingerprint(identity["jit_libraries"]), "runtime_controls": {}, "api_controls": {}}
        raise ValueError("Unrecognized upstream source/binary identity schema")

    @classmethod
    def tested_variant(cls, source):
        identity = source["identity"]
        if identity.get("family") == "current_native_selector":
            normalized = cls.identity(source)
            variant = f"native_{normalized['precision']}_impl{normalized['impl']}"
            point = source["point"]
            if (source.get("job_id") != variant + "_" + point["config"] + "_" + str(point["batch"])
                    or source.get("precision") != normalized["precision"] or point["precision"] != normalized["precision"]
                    or source.get("impl") != normalized["impl"]):
                raise ValueError("Native selector job/point/implementation mapping differs")
            return variant
        return source.get("variant_id", source.get("variant"))

    @classmethod
    def resolve(cls, binding, variant, point_key, operation, input_hash, reference_hashes):
        link = binding.get("accuracy_contract")
        if link is None:
            return None
        manifest = cls.read(link["path"], link["sha256"])
        if manifest.get("schema") != 1 or manifest.get("complete") is not True:
            raise ValueError("Upstream accuracy scope is incomplete or has an unknown schema")
        entry_key = link["entry_keys_by_operation"][operation] if "entry_keys_by_operation" in link else link["entry_key"]
        entry = manifest["entries"][entry_key]
        expected = {"variant_id": variant, "point_key": point_key, "operation": operation,
            "input_hash": input_hash, "reference_hashes": reference_hashes,
            "production_source_hash": binding.get("production_source_hash", binding["source_hash"]),
            "binary_hash": binding["binary_hash"]}
        specification = binding.get("variant_specification", {})
        expected.update(runtime_controls=binding.get("runtime_controls", specification.get("runtime_controls", {})),
                        api_controls=binding.get("api_controls", specification.get("api_controls", {})))
        if any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError("Upstream accuracy input/reference/source/DSO/controls differ from measured contract")
        evidence = entry["evidence"]
        source = cls.read(evidence["path"], evidence["sha256"])
        if (source.get("status") not in ("PASS", "ACCURACY_FAIL") or source.get("producer_drift")
                or source.get("source_unchanged") is False or source.get("producer_unchanged") is False):
            raise ValueError("Upstream accuracy evidence is nonterminal or its producer drifted")
        if cls.tested_variant(source) != entry.get("tested_variant_id", entry["variant_id"]):
            raise ValueError("Upstream tested variant differs")
        if source.get("input_hash") != input_hash or source.get("reference_hashes") != reference_hashes:
            raise ValueError("Upstream actual tensors/references differ")
        upstream = cls.identity(source)
        if upstream["source_hash"] != expected["production_source_hash"] or upstream["binary_hash"] != expected["binary_hash"]:
            raise ValueError("Upstream actual source/DSO differs")
        if upstream.get("runtime_controls", {}) != expected["runtime_controls"] or upstream.get("api_controls", {}) != expected["api_controls"]:
            raise ValueError("Upstream actual parameter controls differ")
        for key in ("impl", "bimpl", "activation_layout", "version", "instructions", "alpha", "weight_numel"):
            if key in upstream and binding.get(key) != upstream[key]:
                raise ValueError("Upstream actual implementation/layout/normalization differs: " + key)
        rows = [row for row in cls.rows(source) if row["operation"] == operation]
        if not rows:
            raise ValueError("Upstream accuracy operation was not tested")
        boundaries = entry.get("required_boundaries", source.get("selected_boundaries"))
        kinds = ("grad_x", "grad_w") if operation == "grad_x_w" else ("forward" if operation == "forward_autograd" else operation,)
        if not isinstance(boundaries, list) or not boundaries or len(set(boundaries)) != len(boundaries):
            raise ValueError("Upstream accuracy boundary scope is missing/duplicate")
        wanted = {(boundary, kind) for boundary in boundaries for kind in kinds}
        observed = [(row.get("boundary", row.get("call_boundary")), row.get("result", kinds[0])) for row in rows]
        if len(observed) != len(wanted) or set(observed) != wanted:
            raise ValueError("Upstream accuracy boundary/component set is incomplete or duplicated")
        passed = all(RecordedAccuracy.passes(row, row.get("accuracy_tier")) for row in rows)
        if entry.get("passed") is not passed or any(row.get("passed", row.get("accuracy_pass")) != RecordedAccuracy.passes(row, row.get("accuracy_tier")) for row in rows):
            raise ValueError("Upstream qualification differs from independently reapplied numeric/repeat/buffer gates")
        return {"passed": passed, "reason": "UPSTREAM_FULL_OPERATION_PASS" if passed else "UPSTREAM_FULL_OPERATION_ACCURACY_FAIL",
            "link": link, "evidence": evidence, "verified_rows": len(rows),
            "input_hash": input_hash, "reference_hashes": reference_hashes,
            "qualification_scope": "Base numeric/repeat/buffer contract; unmatched frozen H100 P14 gate is not promoted to verified"}
