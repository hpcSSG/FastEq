"""CPU-only source contracts for the preserved handwritten and legacy entries."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import subprocess

from common import Evidence
from handwritten import HandwrittenSource
from session import Session


class SourceContract:
    """Keep source observations separate from unverified runtime capabilities."""

    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text()
        self.lines = self.text.splitlines()
        self.active = re.sub(r"/\*.*?\*/|//[^\n]*", "", self.text, flags=re.S)

    def evidence(self, pattern):
        return [{"line": i, "text": line.strip()} for i, line in enumerate(self.lines, 1)
                if re.search(pattern, line) and not line.lstrip().startswith("//")]

    def identity(self):
        return {"source": str(self.path), "source_sha256": Evidence.sha256(self.path)}


class HandwrittenInventory:
    SINGLE = {"gmm_v0", "gmm_v1", "gmm_v2", "gmm_v3", "gmm_v4"}

    def __init__(self, session: Session):
        self.session = session
        self.archive = Path(session.paths["OPT_ROOT"])

    def fp64(self, name):
        source = SourceContract(self.archive / "gmm_src/cuda_impl" / (name + ".cu"))
        helper = source.path.parent / "ptx_inst.cuh"
        helper_source = SourceContract(helper)
        if "asm_mma_m16n8k4_f64_f64_f64_f64(" not in source.active or "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64" not in helper_source.active:
            raise ValueError("FP64 arithmetic source identity not established: " + name)
        grouped = name in {"gmm_v5_multi_path", "gmm_v6_producer_consumer"}
        p1 = name == "gmm_p1_v2"
        host = source.text[source.text.index("torch::Tensor"):]
        schemas = re.findall(r"torch::Tensor\s+\w+\s*\([^)]*\)", host)
        row = {"kind": "fp64_" + name, **source.identity(), "helper": helper_source.identity(),
               "source_schema": schemas[-1], "operator": "forward", "dtype": "float64",
               "math": "FP64_INPUT_FP64_MMA_FP64_ACCUMULATION", "math_allowed": True,
               "math_evidence": source.evidence(r"asm_mma_m16n8k4_f64|TORCH_CHECK.*kFloat64"),
               "helper_instruction_evidence": helper_source.evidence(r"mma\.sync.*f64"),
               "explicit_checks": source.evidence(r"TORCH_CHECK"),
               "control_evidence": source.evidence(r"case (4|10|17):|num_paths ==|out_num_paths ==|throw std::invalid_argument|constexpr uint32_t (M_WARPS|N_WARPS|W_TILE_M|W_TILE_N|TILE_K)|dim3 grid|m.def"),
               "allowed_path_counts": [1] if p1 else [4] if name in self.SINGLE else [4, 10, 17],
               "x_shape": "[B,total_i,U]", "weight_shape": "[U,V]" if p1 else "[P,U,V]",
               "coefficient_argument": "cg_vec[P]" if grouped else "scalar val",
               "output_semantics": "ADJACENT_EQUAL_I_DIM_GROUP_REDUCTION" if grouped else "PER_INPUT_PATH_OUTPUT",
               "output_groups": 4 if grouped else None,
               "output_shape": "[B,sum(group_i),V]" if grouped else "flat[B*total_i*V]" if name == "gmm_v3" else "[B,total_i,V]",
               "tile": {"M": 32, "N": 32, "K": 16},
               "required_adapter_guards": ["CUDA same-device contiguous double tensors", "w U matches x U", "positive dimensions and exact metadata lengths", "sum(i_dims)==x.size(1)", "U divisible by 16 (unmasked fixed K=16 loads)", "V divisible by 32"],
               "adapter_options": "For scalar-val entries, preserve native per-path output; canonical per-path cg may be folded into a private W copy with val=1 and explicitly recorded. Reduction/concatenation is an adapter operation, not original fused-kernel capability.",
               "source_defects": [], "runtime_status": "UNTESTED", "gpu_execution": False}
        if grouped:
            row["required_adapter_guards"].extend(["P in {4,10,17}; exactly four adjacent distinct-i groups", "cg_vec length equals P"])
        if name in {"gmm_p1_v2", "gmm_v0", "gmm_v1", "gmm_v2", "gmm_v3"}:
            row["required_adapter_guards"].append("B divisible by 32 (explicit source check)")
        if name == "gmm_v2_multi_path":
            row["required_adapter_guards"].append("B*total_i divisible by 32 (explicit source check)")
        if name in {"gmm_v0", "gmm_v1"}:
            row["required_adapter_guards"].append("i_dims[-1] >= every i_dim; grid uses final path size")
        if name == "gmm_v2":
            row["source_defects"] = source.evidence(r"cc TORCH_CHECK")
        return row

    def write(self, output):
        output.mkdir(parents=True, exist_ok=False)
        rows = [self.fp64(name) for name in HandwrittenSource.FP64]
        value = {"status": "SOURCE_REVIEWED_RUNTIME_UNTESTED", "contracts": rows,
                 "count": len(rows), "source_preservation": "No algorithm or original archive edits; builds only rename TORCH_LIBRARY registration", "gpu_execution": False}
        Evidence.write(output / "fp64_contracts.json", value)
        return value


class LegacySourceArchive:
    REF = "4463b10c1fa290f76bd2cbb8213a6e9377a34e78"
    GIT = Path("/work-ict/tmp/eqlinear_h20_sources/history.git")

    @classmethod
    def stage(cls, root):
        root.mkdir(parents=True, exist_ok=False)
        paths = subprocess.check_output(["git", "--git-dir=" + str(cls.GIT), "ls-tree", "-r", "--name-only", cls.REF], text=True, timeout=30).splitlines()
        selected = [path for path in paths if path.startswith(("fasteq/cuda/", "fasteq/triton/"))]
        result = {}
        for name in selected:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(["git", "--git-dir=" + str(cls.GIT), "show", cls.REF + ":" + name], timeout=30))
            result[name] = Evidence.sha256(path)
        Evidence.write(root / "source_manifest.json", {"git_dir": str(cls.GIT), "ref": cls.REF, "files": result, "mode": "READ_ONLY_LOCAL_HISTORY_NO_FETCH_NO_PRODUCTION_REFS_CHANGED"})
        return root


class LegacyEntryAudit:
    """Review active entrypoints without importing archived Triton GPU modules."""

    def __init__(self, session, root):
        self.session, self.root = session, root
        self.archive = Path(session.paths["OPT_ROOT"])

    def write(self, output):
        cuda = SourceContract(self.root / "fasteq/cuda/src/equi_linear.cu")
        f64 = SourceContract(self.root / "fasteq/cuda/src/equi_linear_kernels/equi_linear_fwd_f64.cu")
        helper = SourceContract(self.root / "fasteq/cuda/src/equi_linear_kernels/ptx_inst.cuh")
        triton = SourceContract(self.root / "fasteq/triton/so3_linear.py")
        stub = SourceContract(self.archive / "gmm_tf32_src/triton_impl/fused_eqlinear_kernel.py")
        fctp = SourceContract(self.archive / "gmm_src/triton_impl/fused_fctp_triton.py")
        fctp_ast = ast.parse(fctp.text)
        fctp_wrapper = next(n for n in fctp_ast.body if isinstance(n, ast.FunctionDef) and n.name == "fused_fctp")
        if any(isinstance(n, ast.Return) for n in ast.walk(fctp_wrapper)):
            raise ValueError("Preserved FCTP wrapper return contract changed")
        if "('ieee', 'tf32', 'tf32x3')" not in triton.text or "FP64 is not supported" not in triton.text:
            raise ValueError("Legacy Triton selector/dtype observations changed")
        rows = [
            {"id": "historical_original_cuda_fp64", **cuda.identity(), "native": f64.identity(), "helper": helper.identity(),
             "precision": "FP64_INPUT_FP64_MMA_FP64_ACCUMULATION", "math_allowed": True,
             "entrypoints": ["torch.ops.<isolated namespace>.forward(x,w,i_dims,cg_vec)", "torch.ops.<isolated namespace>.backward(dy,w,i_dims,cg_vec)"],
             "forward": "P=4 only; output [B,16,V]; V divisible by32; unmasked K16 loads require U divisible by16; original128-byte pointer alignment checks",
             "backward_x": "Explicit double branch transposes W contiguously and calls preserved f64 forward helper; P=4; source capability only, numerical safety untested",
             "native_output": "PER_PATH_OUTPUT; source f64 dispatcher does not implement P10/P17 reduction",
             "wrapper_evidence": cuda.evidence(r"TORCH_CHECK|torch::Tensor mutipath|Not implimented|w.transpose|fwd_f64_impl|m.def"),
             "native_evidence": f64.evidence(r"asm_mma.*f64|assert\(V|case 4:|case 10:|case 17:|Supported values are|out_total_i =|out_i_dims_vec =|dim3 grid"),
             "helper_evidence": helper.evidence(r"mma\.sync.*f64"),
             "runtime_status": "PENDING_ISOLATED_BUILD_AND_PROBE", "build_dependency": "Existing original CMake also includes unrelated FCTP/TF32 sources; use source-preserving isolated FP64-only test binding, guarded before entry, unique namespace/SONAME"},
            {"id": "historical_triton_f32_ieee_and_tf32x3", **triton.identity(), "math_allowed": True,
             "precision_contracts": [{"dtype": "float32", "input_precision": "ieee"}, {"dtype": "float32", "input_precision": "tf32x3"}],
             "entrypoints": ["so3_linear_serial_cta", "so3_linear_path_parallel", "so3_linear_group_parallel", "so3_linear_backward_x"],
             "math": "Each tl.dot receives exact explicit PRECISION; IEEE and Triton tf32x3 are distinct permitted contracts; verify generated TTIR/PTX and actual precision in future GPU probe",
             "precision_evidence": triton.evidence(r"tl.dot|input_precision not in|input_precision must be"),
             "dtype_evidence": triton.evidence(r"x.dtype not in|grad_out.dtype not in|FP64 is not supported"),
             "runtime_status": "PENDING_ISOLATED_IMPORT_AND_PROBE"},
            {"id": "historical_triton_auto_f32", **triton.identity(), "math_allowed": True,
             "controls": {"tf32": False}, "math": "public auto uses IEEE with tf32=False; explicit tf32x3 is unavailable through this bool auto selector",
             "evidence": triton.evidence(r"def so3_linear_auto|effective_tf32|precision =|input_precision=precision"), "runtime_status": "PENDING_ISOLATED_IMPORT_AND_PROBE"},
            {"id": "historical_triton_single_tf32", **triton.identity(), "math_allowed": False,
             "classification": "EXCLUDED_USER_TF32X1", "forbidden_controls": ["input_precision='tf32'", "so3_linear_auto(tf32=True) with float32"],
             "evidence": triton.evidence(r"effective_tf32|precision =|input_precision not in"), "runtime_status": "NOT_EXECUTED_USER_EXCLUSION"},
            {"id": "historical_triton_fp64", **triton.identity(), "math_allowed": None,
             "classification": "ORIGINAL_API_REJECTS_FP64", "evidence": triton.evidence(r"FP64 is not supported"),
             "runtime_status": "SOURCE_API_LIMITATION; not a claim of H20 hardware unsupported"},
            {"id": "standalone_triton_stub", **stub.identity(), "math_allowed": None,
             "classification": "SOURCE_STUB_NO_GEMM", "math": "No active tl.dot, kernel launch commented; wrapper returns ones. TF32 comment alone is not an executed TF32x1 algorithm.",
             "evidence": stub.evidence(r"pass$|torch.ones|return out|ACC_DTYPE"), "runtime_status": "PENDING_MINIMUM_PROBE; source facts only, no hardware capability verdict"},
            {"id": "standalone_triton_fctp", **fctp.identity(), "math_allowed": True,
             "classification": "FP64_FCTP_SEPARATE_OPERATION_API_DEFECT", "math": "float64 x,y,w and two tl.dot operations with float64 accumulators; no TF32x1 input contract",
             "api": "fused_fctp(x,y,w,B,U=96,W=96,V=10,TOTAL_I=16,TOTAL_J=16); three-input fully connected tensor product, not two-input EqLinear",
             "source_defects": "Wrapper allocates and launches but has no return; bmm2 accumulator uses tl.zeros(..., type=tl.float64), unverified compiler acceptance",
             "evidence": fctp.evidence(r"tl.dot|tl.zeros.*float64|assert.*dtype|def fused_fctp|out = torch.empty"),
             "runtime_status": "SOURCE_REVIEW_ONLY; keep separate operation obligation without repairing algorithm or claiming EqLinear numeric coverage"}
        ]
        output.mkdir(parents=True, exist_ok=False)
        manifest = self.root / "source_manifest.json"
        result = {"status": "SOURCE_AUDIT_COMPLETE_GPU_UNTESTED", "historical_sources": {"path": str(manifest), "sha256": Evidence.sha256(manifest)},
                  "contracts": rows, "gpu_execution": False, "no_tf32x1_executed": True}
        Evidence.write(output / "audit.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-stage", type=Path)
    parser.add_argument("--legacy-audit", type=Path)
    args = parser.parse_args()
    value = HandwrittenInventory(Session(args.session)).write(args.output)
    if args.legacy_stage:
        LegacySourceArchive.stage(args.legacy_stage)
    if args.legacy_audit:
        LegacyEntryAudit(Session(args.session), args.legacy_audit).write(args.output / "legacy")
    print("FP64_CONTRACTS", value["count"])


if __name__ == "__main__":
    main()
