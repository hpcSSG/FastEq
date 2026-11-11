"""Source-bound arithmetic adjudication before any handwritten TF32 execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from common import Evidence
from session import Session


class SourceMathAudit:
    SINGLE = {"gmm_tf32_v" + str(index) for index in range(7)} | {"gmm_tf32_v6_ibu", "gmm_bwd_tf32_v6", "gmm_bwd_tf32_v7"}
    THREE = {"gmm_tf32_v7", "gmm_bwd_tf32_v8"}

    @staticmethod
    def code(text):
        # Evidence extraction only; the adjudication also binds the exact complete source bytes.
        return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)

    @classmethod
    def run(cls, args):
        session = Session(args.session)
        registry = json.loads(args.registry.read_text())
        rows = registry["variants"] if isinstance(registry, dict) else registry
        args.output.mkdir(parents=True, exist_ok=False)
        selected = [row for row in rows if row["family"] in ("handwritten_tf32", "handwritten_legacy_tf32")]
        helper = Path(session.paths["OPT_ROOT"]) / "gmm_tf32_src/cuda_impl/kernels/ptx_inst.cuh"
        helper_text = helper.read_text()
        if 'template <int32_t ScaleD, int32_t ScaleA, int32_t ScaleB>' not in helper_text or 'cvt.rna.tf32.f32' not in helper_text:
            raise ValueError("Shared instruction helper differs from the manually reviewed source")
        helper_snapshot = args.output / "ptx_inst.cuh"
        helper_snapshot.write_bytes(helper.read_bytes())
        results = []
        for row in selected:
            path = Path(row["source_path"])
            digest = Evidence.sha256(path)
            if digest != row["source_hash"]:
                raise ValueError("Registry/source mismatch: " + row["variant_id"])
            text, stem = path.read_text(), path.stem
            code = cls.code(text)
            calls = list(re.finditer(r"\basm_wgmma_m64n\d+k8_tf32\s*<", code))
            legacy = row["family"] == "handwritten_legacy_tf32"
            if legacy:
                if calls or "// asm_wgmma_m64n96k8_tf32" not in text:
                    raise ValueError("Legacy incomplete-body adjudication needs re-review")
                classification, products, action = "SOURCE_GEMM_BODY_COMMENTED_OUT", 0, "KEEP_PENDING_BUILD_OR_PROBE; do not claim a valid TF32 implementation or hardware unsupported"
                reason = "Host launches fused_gmm_kernel_v0; its only matrix-multiply statement is commented out. This is distinct from an executed one-product TF32 kernel."
            elif stem in cls.SINGLE:
                if len(calls) != 1 or "a_res_reg" in code:
                    raise ValueError("Single-product adjudication needs re-review: " + stem)
                classification, products, action = "TF32_SINGLE_PRODUCT", 1, "EXCLUDED_BY_USER_NO_EXECUTION"
                reason = "Each K tile has one TF32 A/B WGMMA accumulation and no residual cross-products. Shared <1,1,1> parameters are ScaleD/ScaleA/ScaleB, not three products. Operand conversion may be absent/commented out; this does not add residual products or authorize execution."
            elif stem in cls.THREE:
                if len(calls) != 3 or "a_res_reg" not in code or "b_res_desc" not in code:
                    raise ValueError("Three-product adjudication needs re-review: " + stem)
                classification, products, action = "TF32_THREE_PRODUCT", 3, "PERMITTED_ARITHMETIC_STILL_NEEDS_BUILD_AND_RUNTIME"
                reason = "Explicit hi*hi, hi*residual and residual*hi accumulations occur inside each K tile; residual A registers and residual B descriptor are present."
            else:
                raise ValueError("Unreviewed handwritten source: " + str(path))
            snapshot = args.output / (row["variant_id"] + ".cu")
            snapshot.write_bytes(path.read_bytes())
            evidence = [{"line": index, "text": line.strip()} for index, line in enumerate(text.splitlines(), 1)
                if any(token in line for token in ("asm_wgmma_", "asm_cvt_tf32", "a_res_reg", "b_res_desc", "for (uint32_t _k", "for (uint32_t k_count", "<<<grid"))]
            results.append({"variant_id": row["variant_id"], "source": str(path), "source_sha256": digest,
                "source_snapshot": str(snapshot), "classification": classification, "products_per_unpadded_k_tile": products,
                "explicit_tf32_conversion_active": "asm_cvt_tf32_f32" in code,
                "action": action, "reason": reason, "source_evidence": evidence,
                "runtime_backend_verified": False, "execution_performed": False})
        result = {"classification": "CPU_SOURCE_ARITHMETIC_AUDIT_NOT_H20_RUNTIME_EVIDENCE", "results": results,
            "registry": {"path": str(args.registry), "sha256": Evidence.sha256(args.registry)},
            "helper": {"path": str(helper), "snapshot": str(helper_snapshot), "sha256": Evidence.sha256(helper)},
            "helper_interpretation": "cvt.rna.tf32.f32 rounds each operand; each asm helper emits one wgmma.mma_async.*.f32.tf32.tf32. ScaleD/ScaleA/ScaleB are scaling controls.",
            "scope_rule": "User prohibits TF32x1 execution; no numerical probe or timing of excluded arithmetic is authorized",
            "auditor_sha256": Evidence.sha256(Path(__file__))}
        Evidence.write(args.output / "audit.json", result)
        print(json.dumps({"reviewed": len(results), "single_product_excluded": sum(row["products_per_unpadded_k_tile"] == 1 for row in results),
                          "three_product_permitted": sum(row["products_per_unpadded_k_tile"] == 3 for row in results),
                          "incomplete_source": sum(row["products_per_unpadded_k_tile"] == 0 for row in results)}))
        return 0

    @classmethod
    def main(cls):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--session", type=Path, required=True)
        parser.add_argument("--registry", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        return cls.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(SourceMathAudit.main())
