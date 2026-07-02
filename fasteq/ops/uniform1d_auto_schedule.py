from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Union, Iterable, Set, Sequence
from time import perf_counter
from pathlib import Path
import torch




# =============================================================================
# STC LARS scheduler and CUDA code generation
# Moved from stc_uniform1d_jit.py so the STC runtime wrapper only keeps
# forward/backward dispatch and file-based JIT loading.
# =============================================================================

Label = Tuple[str, int]  # ('x0', i), ('x1', j), ('o', v)
STC_PAD_VALUE = -1



def _to_int_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]


def _to_float_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]


@dataclass
class Inst:
    op: str
    args: Tuple[Any, ...]
    comment: str = ""


@dataclass
class ScheduleResult:
    instructions: List[Inst]
    path_order: List[int]
    max_live: int
    spills: int
    reloads: int
    final_reg_map: Dict[Label, str]
    profile: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class STCPath:
    """One STC contraction path.

    Padded STC metadata uses baseline-compatible row semantics:

        len == 3: [x1_a,                 x0_d, out_v, pad, ...]
        len == 4: [x1_a, x1_b,          x0_d, out_v, pad, ...]
        len == 5: [x1_a, x1_b, x1_c,   x0_d, out_v, pad, ...]

    Only the first ``path_lens[p]`` entries are semantic; right-side padding is
    ignored.  The generated expression is therefore:

        out[v] += c * prod_i x1[x1_indices[i]] * x0[x0_index]

    Duplicate indices are intentionally preserved in ``product_labels`` so that
    x1[i] * x1[i] is emitted as a square/cube instead of being collapsed.
    """

    pid: int
    x1_indices: Tuple[int, ...]
    x0_index: int
    v: int
    c: float

    @property
    def product_labels(self) -> Tuple[Label, ...]:
        labels: List[Label] = [("x1", int(i)) for i in self.x1_indices]
        labels.append(("x0", int(self.x0_index)))
        return tuple(labels)

    @property
    def labels(self) -> Tuple[Label, ...]:
        # Remove duplicates only for liveness/register allocation.  The product
        # expression still uses product_labels, so x1[i] * x1[i] remains squared.
        seen: Set[Label] = set()
        out: List[Label] = []
        for lab in self.product_labels:
            if lab not in seen:
                seen.add(lab)
                out.append(lab)
        return tuple(out)


# STC now reuses LARSUniform1DScheduler(path_kind="stc").

def _lars_reg_id(reg: str) -> int:
    if not isinstance(reg, str) or not reg.startswith("r"):
        raise ValueError(f"Bad register name: {reg!r}")
    return int(reg[1:])


def _max_reg_count_any(schedule_result: ScheduleResult) -> int:
    max_id = -1

    def scan(arg: Any) -> None:
        nonlocal max_id
        if isinstance(arg, str) and arg.startswith("r"):
            max_id = max(max_id, _lars_reg_id(arg))
        elif isinstance(arg, (tuple, list)):
            for a in arg:
                scan(a)

    for inst in schedule_result.instructions:
        for arg in inst.args:
            scan(arg)
    return max_id + 1


def _parse_label_ref(ref: str) -> Label:
    if not isinstance(ref, str) or "[" not in ref or not ref.endswith("]"):
        raise ValueError(f"Bad label ref: {ref!r}")
    kind, rest = ref.split("[", 1)
    idx = int(rest[:-1])
    if kind == "out":
        kind = "o"
    if kind not in ("x0", "x1", "o"):
        raise ValueError(f"Bad STC label kind: {kind!r}")
    return kind, idx


def _fmt_float(x: float) -> str:
    return repr(float(x))


def _sanitize_cuda_comment(s: str) -> str:
    return str(s).replace("\n", " ").replace("\r", " ").replace("*/", "* /")


def _product_expr(regs: Sequence[str], coeff: float) -> str:
    if not regs:
        return f"scalar_t({_fmt_float(coeff)})"
    expr = str(regs[0])
    for r in regs[1:]:
        expr = f"({expr} * {r})"
    return f"scalar_t({_fmt_float(coeff)}) * {expr}"


def _index_expr(kind: str, idx: int, *, u_dim: Optional[int], x0_dim: Optional[int], x1_dim: Optional[int], v_dim: Optional[int]) -> str:
    if kind == "x0":
        if x0_dim is not None and idx >= x0_dim:
            raise ValueError(f"x0 index {idx} out of x0_dim={x0_dim}")
        if u_dim is not None:
            return f"x0_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"x0_base + (index_t){idx} * (index_t)U + (index_t)u"
    if kind == "x1":
        if x1_dim is not None and idx >= x1_dim:
            raise ValueError(f"x1 index {idx} out of x1_dim={x1_dim}")
        if u_dim is not None:
            return f"x1_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"x1_base + (index_t){idx} * (index_t)U + (index_t)u"
    if kind == "o":
        if v_dim is not None and idx >= v_dim:
            raise ValueError(f"out index {idx} out of v_dim={v_dim}")
        if u_dim is not None:
            return f"out_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"out_base + (index_t){idx} * (index_t)U + (index_t)u"
    raise ValueError(f"Bad kind: {kind}")


def emit_stc_fwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    u_dim: Optional[int] = None,
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    if block_size != 32:
        raise ValueError("this emitter assumes block_size=32")

    reg_count = _max_reg_count_any(schedule_result)
    resident_out_indices = sorted({int(inst.args[0]) for inst in schedule_result.instructions if inst.op == "mul_stc_resident"})
    direct_out_indices = sorted({int(inst.args[0]) for inst in schedule_result.instructions if inst.op == "mul_stc_direct"})

    lines: List[str] = []

    def ap(line: str = "") -> None:
        lines.append(line)

    def emit_out_write(idx: int, value_expr: str) -> None:
        expr = _index_expr("o", int(idx), u_dim=u_dim, x0_dim=x0_dim, x1_dim=x1_dim, v_dim=v_dim)
        ap(f"            out[{expr}] += {value_expr};")

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ x1,")
    ap("    const scalar_t* __restrict__ x0,")
    ap("    scalar_t* __restrict__ out,")
    ap("    int B, int X1, int X0, int V, int U)")
    ap("{")
    ap("    const int b = (int)blockIdx.x;")
    ap("    if (b >= B) return;")
    ap("    const int tid = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    ap("    const index_t x1_base = (index_t)b * (index_t)X1 * (index_t)U;")
    ap("    const index_t x0_base = (index_t)b * (index_t)X0 * (index_t)U;")
    ap("    const index_t out_base = (index_t)b * (index_t)V * (index_t)U;")
    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    for out_idx in resident_out_indices:
        ap(f"            scalar_t out_acc_v_{out_idx} = scalar_t(0);")
    if resident_out_indices:
        ap("")
    if direct_out_indices:
        ap(f"            // direct single-use output writeback enabled for {len(direct_out_indices)} out accumulator(s)")
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)
        if inst.op == "load":
            reg, ref = inst.args
            kind, idx = _parse_label_ref(ref)
            if kind == "o":
                raise ValueError("load must not target output labels")
            expr = _index_expr(kind, idx, u_dim=u_dim, x0_dim=x0_dim, x1_dim=x1_dim, v_dim=v_dim)
            base = "x0" if kind == "x0" else "x1"
            ap(f"            {reg} = {base}[{expr}];")
        elif inst.op == "mul_stc_resident":
            out_idx, regs, coeff = inst.args
            ap(f"            out_acc_v_{int(out_idx)} += {_product_expr(tuple(regs), float(coeff))};")
        elif inst.op == "mul_stc_direct":
            out_idx, regs, coeff = inst.args
            emit_out_write(int(out_idx), _product_expr(tuple(regs), float(coeff)))
        elif inst.op == "release":
            pass
        else:
            raise ValueError(f"Unsupported STC instruction op: {inst.op}")

    if resident_out_indices:
        ap("")
        ap("            // resident output accumulator writeback")
    for out_idx in resident_out_indices:
        emit_out_write(int(out_idx), f"out_acc_v_{out_idx}")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("static inline bool mul3_fits_int32(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32(a, b)) return false;")
    ap("    return mul_fits_int32(a * b, c);")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* x1, const scalar_t* x0, scalar_t* out,")
    ap("    int B, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(x1, x0, out, B, X1, X0, V, U);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* x1, const scalar_t* x0, scalar_t* out,")
    ap("    int B, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap("    bool use_i32 = mul3_fits_int32((int64_t)B, (int64_t)X1, (int64_t)U) &&")
    ap("                   mul3_fits_int32((int64_t)B, (int64_t)X0, (int64_t)U) &&")
    ap("                   mul3_fits_int32((int64_t)B, (int64_t)V,  (int64_t)U);")
    ap("    if (use_i32) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(x1, x0, out, B, X1, X0, V, U, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(x1, x0, out, B, X1, X0, V, U, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap(f"torch::Tensor launcher_{kernel_name}(torch::Tensor x1, torch::Tensor x0, int64_t V64) {{")
    ap("    TORCH_CHECK(x1.is_cuda() && x0.is_cuda(), \"x1/x0 must be CUDA/HIP\");")
    ap("    TORCH_CHECK(x1.is_contiguous() && x0.is_contiguous(), \"x1/x0 must be contiguous\");")
    ap("    TORCH_CHECK(x1.dim() == 3 && x0.dim() == 3, \"x1/x0 must be [B,S,U]\");")
    ap("    TORCH_CHECK(x1.scalar_type() == x0.scalar_type(), \"x1/x0 dtype mismatch\");")
    ap("    int B = (int)x1.size(0);")
    ap("    int X1 = (int)x1.size(1);")
    ap("    int U = (int)x1.size(2);")
    ap("    int X0 = (int)x0.size(1);")
    ap("    int V = (int)V64;")
    ap("    TORCH_CHECK((int)x0.size(0) == B, \"x0 batch mismatch\");")
    ap("    TORCH_CHECK((int)x0.size(2) == U, \"x0 U mismatch\");")
    ap("    TORCH_CHECK(V > 0, \"V must be > 0\");")
    ap("    TORCH_CHECK((U % 32) == 0, \"U must be a multiple of 32\");")
    ap("    auto out = torch::zeros({B, V, U}, x1.options());")
    ap("    GPU_Guard device_guard(x1.device());")
    ap("    gpuStream_t stream = getCurrentGPUStream(x1.device().index());")
    ap(f"    AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), \"{kernel_name}\", [&] {{")
    ap(f"        launch_{kernel_name}<scalar_t>((const scalar_t*)x1.data_ptr<scalar_t>(),")
    ap("            (const scalar_t*)x0.data_ptr<scalar_t>(),")
    ap("            (scalar_t*)out.data_ptr<scalar_t>(), B, X1, X0, V, U, stream);")
    ap("    });")
    ap("    GPU_KERNEL_LAUNCH_CHECK();")
    ap("    return out;")
    ap("}")
    ap("")
    ap("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    ap(f"    m.def(\"run\", &launcher_{kernel_name}, \"{kernel_name} STC forward jit impl\");")
    ap("}")

    return "\n".join(lines)


def _normalize_stc_padded_paths(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: Optional[torch.Tensor] = None,
    path_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return padded STC paths as a CPU int64 tensor with shape [max_len, P].

    Accepted inputs:
      * ``idx_lists_tensor`` style: [max_len, P]
      * ``paths_tensor`` style:    [P, max_len]
      * Python sequence of max_len tensors, each [P]

    Without ``path_lens``, the path dimension is inferred from
    ``coeff_list.numel()``.  This allows dropping ``path_lens_tensor`` from the
    public API as long as padding uses a sentinel value such as -1.
    """
    if path_lens is not None:
        P = int(path_lens.detach().cpu().reshape(-1).numel())
    elif coeff_list is not None:
        P = int(coeff_list.detach().cpu().reshape(-1).numel())
    else:
        raise ValueError("Either coeff_list or path_lens is required to infer the path dimension")

    if isinstance(idx_lists, torch.Tensor):
        idx = idx_lists.detach().cpu().to(torch.int64)
    else:
        idx = torch.stack([t.detach().cpu().to(torch.int64).reshape(-1) for t in idx_lists], dim=0)

    if idx.dim() != 2:
        raise ValueError(f"STC padded paths must be 2D, got shape={tuple(idx.shape)}")

    # Already [max_len, P], as in stc_meta['idx_lists_tensor'].
    if idx.shape[1] == P and idx.shape[0] != P:
        return idx.contiguous()

    # Row-major [P, max_len], as in stc_meta['paths'].
    if idx.shape[0] == P and idx.shape[1] != P:
        return idx.t().contiguous()

    # Ambiguous square case. Prefer the STC convention max_len <= 5 when possible.
    if idx.shape[0] == P and idx.shape[1] == P:
        if idx.shape[0] <= 8:
            raise ValueError(
                f"Ambiguous square STC padded path shape={tuple(idx.shape)}. "
                "Pass a non-square [P,max_len]/[max_len,P] tensor or keep path_lens."
            )
        return idx.t().contiguous()

    raise ValueError(
        f"Cannot infer STC path layout from shape={tuple(idx.shape)} and "
        f"P={P}; expected [max_len, P] or [P, max_len]"
    )


def infer_stc_path_lens_from_padded(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    pad_value: int = STC_PAD_VALUE,
) -> torch.Tensor:
    """Infer [P] int64 path lengths from sentinel-padded STC paths.

    Padding must be a suffix and must use ``pad_value``.  Valid indices are
    expected to be non-negative, so -1 is the recommended sentinel.
    """
    idx = _normalize_stc_padded_paths(idx_lists, coeff_list=coeff_list)
    valid = idx.ne(int(pad_value))
    lens = valid.to(torch.int64).sum(dim=0)
    max_len, P = idx.shape

    for p in range(P):
        L = int(lens[p].item())
        if L < 3 or L > max_len:
            raise ValueError(
                f"Bad inferred STC path length {L} for path {p}; "
                f"expected 3..{max_len}. Did you use pad_value={pad_value}?"
            )
        # Enforce suffix padding: valid entries must be exactly [:L].
        if not bool(valid[:L, p].all().item()) or bool(valid[L:, p].any().item()):
            vals = [int(v) for v in idx[:, p].tolist()]
            raise ValueError(
                f"Non-suffix padding in STC path {p}: {vals}. "
                f"Use valid entries first, then pad with {pad_value}."
            )
    return lens.contiguous()


def _infer_stc_path_lens_tensor_from_padded(
    padded_paths: torch.Tensor,
    coeff_list: torch.Tensor,
    *,
    pad_value: int = STC_PAD_VALUE,
) -> torch.Tensor:
    """Infer CUDA/CPU [P] int32 path lengths from sentinel-padded paths.

    This is used only to call the existing baseline backward kernel, whose API
    still expects path_lens_tensor.
    """
    if not isinstance(padded_paths, torch.Tensor) or padded_paths.dim() != 2:
        raise ValueError(f"padded_paths must be a 2D tensor, got {type(padded_paths)}")
    P = int(coeff_list.detach().reshape(-1).numel())

    if int(padded_paths.size(0)) == P and int(padded_paths.size(1)) != P:
        lens = padded_paths.ne(int(pad_value)).to(torch.int32).sum(dim=1)
    elif int(padded_paths.size(1)) == P and int(padded_paths.size(0)) != P:
        lens = padded_paths.ne(int(pad_value)).to(torch.int32).sum(dim=0)
    elif int(padded_paths.size(0)) == P and int(padded_paths.size(1)) == P:
        raise ValueError(
            f"Ambiguous square STC padded path shape={tuple(padded_paths.shape)}; "
            "cannot infer path dimension without path_lens_tensor."
        )
    else:
        raise ValueError(
            f"Cannot infer STC path layout from shape={tuple(padded_paths.shape)} and P={P}"
        )

    return lens.to(device=padded_paths.device, dtype=torch.int32).contiguous()


def make_stc_paths_from_padded_lists(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
) -> List[STCPath]:
    """Convert padded STC metadata into scheduler-native paths.

    If ``path_lens`` is omitted, lengths are inferred from sentinel padding.
    Recommended padded format is:

        len == 3: [x1_a,                 x0_d, out_v, -1, -1]
        len == 4: [x1_a, x1_b,          x0_d, out_v, -1]
        len == 5: [x1_a, x1_b, x1_c,   x0_d, out_v]

    Baseline-compatible semantics are:

        x1_indices = idx[0 : L-2, p]
        x0_index   = idx[L-2, p]
        v          = idx[L-1, p]
    """
    idx = _normalize_stc_padded_paths(idx_lists, coeff_list=coeff_list, path_lens=path_lens)
    if path_lens is None:
        lens = infer_stc_path_lens_from_padded(idx, coeff_list, pad_value=pad_value)
    else:
        lens = path_lens.detach().cpu().to(torch.int64).reshape(-1).contiguous()

    coeff = coeff_list.detach().cpu().reshape(-1)
    max_len, P = idx.shape

    if lens.numel() != P:
        raise ValueError(f"path_lens numel {lens.numel()} does not match P={P}")
    if coeff.numel() != P:
        raise ValueError(f"coeff_list numel {coeff.numel()} does not match P={P}")

    paths: List[STCPath] = []
    for p in range(P):
        L = int(lens[p].item())
        if L < 3 or L > max_len:
            raise ValueError(f"Bad STC path length {L} for path {p}; expected 3..{max_len}")

        vals = [int(idx[t, p].item()) for t in range(L)]
        x1_indices = tuple(vals[: L - 2])
        x0_index = vals[L - 2]
        v = vals[L - 1]

        paths.append(
            STCPath(
                pid=p,
                x1_indices=x1_indices,
                x0_index=x0_index,
                v=v,
                c=float(coeff[p].item()),
            )
        )
    return paths


def generate_code_stc_fwd_with_scheduler(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    num_out_segments: int,
    u_dim: int,
    out_path: str = "generated_stc_fwd_lars.cu",
    kernel_name: str = "stc_lars_fwd",
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    return_schedule: bool = False,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
) -> Any:
    paths = make_stc_paths_from_padded_lists(idx_lists, coeff_list, path_lens=path_lens, pad_value=pad_value)
    scheduler = LARSUniform1DScheduler(
        paths,
        reg_budget=0,
        path_kind="stc",
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name="stc_lars_all_inputs",
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    schedule_result = scheduler.schedule()

    idx_norm = _normalize_stc_padded_paths(idx_lists, coeff_list=coeff_list, path_lens=path_lens)
    if path_lens is None:
        inferred_lens = infer_stc_path_lens_from_padded(idx_norm, coeff_list, pad_value=pad_value)
    else:
        inferred_lens = path_lens.detach().cpu().to(torch.int64).reshape(-1)
    base_kernel_name = f"{kernel_name}_u{int(u_dim)}_path{len(paths)}_maxlen{int(inferred_lens.max().item())}"
    code = emit_stc_fwd_kernel_from_lars_schedule(
        schedule_result,
        kernel_name=base_kernel_name,
        u_dim=int(u_dim),
        x0_dim=x0_dim,
        x1_dim=x1_dim,
        v_dim=int(num_out_segments),
        block_size=32,
    )
    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")
    if return_schedule:
        return {
            "name": "stc_lars_all_inputs",
            "code": code,
            "kernel_name": base_kernel_name,
            "schedule": schedule_result,
            "num_paths": len(paths),
            "max_live": schedule_result.max_live,
        }
    return code



def _check_stc_path_bounds_for_bwd(
    paths: Sequence[STCPath],
    *,
    x0_dim: Optional[int],
    x1_dim: Optional[int],
    v_dim: Optional[int],
) -> None:
    for p in paths:
        if x0_dim is not None and int(p.x0_index) >= int(x0_dim):
            raise ValueError(f"x0 index {p.x0_index} out of x0_dim={x0_dim} in STC path {p.pid}")
        if v_dim is not None and int(p.v) >= int(v_dim):
            raise ValueError(f"out index {p.v} out of v_dim={v_dim} in STC path {p.pid}")
        if int(p.x0_index) < 0 or int(p.v) < 0:
            raise ValueError(f"negative x0/out index in STC path {p.pid}: x0={p.x0_index}, out={p.v}")
        for x1_idx in p.x1_indices:
            if int(x1_idx) < 0:
                raise ValueError(f"negative x1 index {x1_idx} in STC path {p.pid}")
            if x1_dim is not None and int(x1_idx) >= int(x1_dim):
                raise ValueError(f"x1 index {x1_idx} out of x1_dim={x1_dim} in STC path {p.pid}")


def _stc_shared_x1_expr(x1_idx: int) -> str:
    return f"x1_shared[(index_t){int(x1_idx)} * (index_t)TILE_U + (index_t)lj]"


def _stc_shared_grad_x1_expr(x1_idx: int) -> str:
    return f"grad_x1_shared[(index_t){int(x1_idx)} * (index_t)TILE_U + (index_t)lj]"


def _stc_bwd_grad_expr(base_expr: str, other_x1_indices: Sequence[int]) -> str:
    expr = base_expr
    for x1_idx in other_x1_indices:
        expr = f"({expr} * {_stc_shared_x1_expr(int(x1_idx))})"
    return expr


def emit_stc_bwd_kernel_from_paths(
    paths: Sequence[STCPath],
    *,
    kernel_name: str,
    path_order: Optional[Sequence[int]] = None,
    u_dim: Optional[int] = None,
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    tile_u: int = 32,
) -> str:
    """Emit STC backward CUDA code that computes only grad_x1.

    The generated kernel mirrors the baseline tiled STC backward structure:
    each block owns one ``(batch, U-tile)`` pair, stages all x1 segments for that
    tile in shared memory, accumulates grad_x1 in shared memory, and writes the
    final [B, X1, U] gradient once.  Path computation is statically unrolled.
    """
    if int(tile_u) <= 0:
        raise ValueError(f"tile_u must be positive, got {tile_u}")

    path_list = list(paths)
    if path_order is None:
        ordered_paths = path_list
    else:
        by_pid = {int(p.pid): p for p in path_list}
        ordered_paths = [by_pid[int(pid)] for pid in path_order]

    _check_stc_path_bounds_for_bwd(ordered_paths, x0_dim=x0_dim, x1_dim=x1_dim, v_dim=v_dim)

    lines: List[str] = []

    def ap(line: str = "") -> None:
        lines.append(line)

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")
    ap("template <typename scalar_t, typename index_t, int TILE_U>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ grad_out,")
    ap("    const scalar_t* __restrict__ x1,")
    ap("    const scalar_t* __restrict__ x0,")
    ap("    scalar_t* __restrict__ grad_x1,")
    ap("    int B, int X1, int X0, int V, int U)")
    ap("{")
    ap("    extern __shared__ __align__(sizeof(scalar_t)) unsigned char smem[];")
    ap("    scalar_t* x1_shared = reinterpret_cast<scalar_t*>(smem);")
    ap("    scalar_t* grad_x1_shared = x1_shared + (index_t)X1 * (index_t)TILE_U;")
    ap("")
    ap("    const int b = (int)blockIdx.x;")
    ap("    const int tile_id = (int)blockIdx.y;")
    ap("    const int lj = (int)threadIdx.x;")
    ap("    const int u = tile_id * TILE_U + lj;")
    ap("    if (b >= B) return;")
    ap("")
    ap("    for (int a = 0; a < X1; ++a) {")
    ap("        const index_t idx_global = ((index_t)b * (index_t)X1 + (index_t)a) * (index_t)U + (index_t)u;")
    ap("        const index_t idx_shared = (index_t)a * (index_t)TILE_U + (index_t)lj;")
    ap("        x1_shared[idx_shared] = x1[idx_global];")
    ap("        grad_x1_shared[idx_shared] = scalar_t(0);")
    ap("    }")
    ap("")
    ap("    __syncthreads();")
    ap("")

    for inst_id, p in enumerate(ordered_paths):
        x1s = tuple(int(v) for v in p.x1_indices)
        x0_idx = int(p.x0_index)
        out_v = int(p.v)
        coeff = float(p.c)
        base_name = f"base_{inst_id}"
        comment = _sanitize_cuda_comment(
            f"path#{p.pid}: grad_x1 for out[{out_v}] += coeff * grad_out * x0[{x0_idx}]"
        )
        ap(f"    // bwd inst {inst_id}: {comment}")
        ap(
            f"    const scalar_t {base_name} = "
            f"grad_out[((index_t)b * (index_t)V + (index_t){out_v}) * (index_t)U + (index_t)u] "
            f"* scalar_t({_fmt_float(coeff)}) "
            f"* x0[((index_t)b * (index_t)X0 + (index_t){x0_idx}) * (index_t)U + (index_t)u];"
        )
        for pos, target_x1 in enumerate(x1s):
            other = [idx for q, idx in enumerate(x1s) if q != pos]
            grad_expr = _stc_bwd_grad_expr(base_name, other)
            ap(f"    {_stc_shared_grad_x1_expr(target_x1)} += {grad_expr};")
        ap("")

    ap("    __syncthreads();")
    ap("")
    ap("    for (int a = 0; a < X1; ++a) {")
    ap("        const index_t idx_global = ((index_t)b * (index_t)X1 + (index_t)a) * (index_t)U + (index_t)u;")
    ap("        const index_t idx_shared = (index_t)a * (index_t)TILE_U + (index_t)lj;")
    ap("        grad_x1[idx_global] = grad_x1_shared[idx_shared];")
    ap("    }")
    ap("}")
    ap("")
    ap("static inline bool mul_fits_int32(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("static inline bool mul3_fits_int32(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32(a, b)) return false;")
    ap("    return mul_fits_int32(a * b, c);")
    ap("}")
    ap("")
    ap("template <typename scalar_t, typename index_t, int TILE_U>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, scalar_t* grad_x1,")
    ap("    int B, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap("    dim3 block(TILE_U);")
    ap("    dim3 grid(B, U / TILE_U);")
    ap("    size_t smem_bytes = (size_t)2 * (size_t)X1 * (size_t)TILE_U * sizeof(scalar_t);")
    ap(f"    {kernel_name}<scalar_t, index_t, TILE_U><<<grid, block, smem_bytes, stream>>>(")
    ap("        grad_out, x1, x0, grad_x1, B, X1, X0, V, U);")
    ap("}")
    ap("")
    ap("template <typename scalar_t, int TILE_U>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, scalar_t* grad_x1,")
    ap("    int B, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap("    bool use_i32 = mul3_fits_int32((int64_t)B, (int64_t)X1, (int64_t)U) &&")
    ap("                   mul3_fits_int32((int64_t)B, (int64_t)X0, (int64_t)U) &&")
    ap("                   mul3_fits_int32((int64_t)B, (int64_t)V,  (int64_t)U);")
    ap("    if (use_i32) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t, TILE_U>(grad_out, x1, x0, grad_x1, B, X1, X0, V, U, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t, TILE_U>(grad_out, x1, x0, grad_x1, B, X1, X0, V, U, stream);")
    ap("    }")
    ap("}")
    ap("")
    ap(f"torch::Tensor launcher_{kernel_name}(torch::Tensor grad_out, torch::Tensor x1, torch::Tensor x0, int64_t V64) {{")
    ap("    TORCH_CHECK(grad_out.is_cuda() && x1.is_cuda() && x0.is_cuda(), \"grad_out/x1/x0 must be CUDA/HIP\");")
    ap("    TORCH_CHECK(grad_out.is_contiguous() && x1.is_contiguous() && x0.is_contiguous(), \"grad_out/x1/x0 must be contiguous\");")
    ap("    TORCH_CHECK(x1.dim() == 3 && x0.dim() == 3, \"x1/x0 must be [B,S,U]\");")
    ap("    TORCH_CHECK(grad_out.scalar_type() == x1.scalar_type() && x1.scalar_type() == x0.scalar_type(), \"grad_out/x1/x0 dtype mismatch\");")
    ap("    int B = (int)x1.size(0);")
    ap("    int X1 = (int)x1.size(1);")
    ap("    int U = (int)x1.size(2);")
    ap("    int X0 = (int)x0.size(1);")
    ap("    int V = (int)V64;")
    ap("    TORCH_CHECK((int)x0.size(0) == B, \"x0 batch mismatch\");")
    ap("    TORCH_CHECK((int)x0.size(2) == U, \"x0 U mismatch\");")
    ap("    TORCH_CHECK(V > 0, \"V must be > 0\");")
    ap(f"    TORCH_CHECK((U % {int(tile_u)}) == 0, \"U must be a multiple of TILE_U\");")
    ap("    TORCH_CHECK(grad_out.numel() == (int64_t)B * (int64_t)V * (int64_t)U, \"grad_out numel mismatch; expected B*V*U\");")
    if u_dim is not None:
        ap(f"    TORCH_CHECK(U == {int(u_dim)}, \"U mismatch for generated STC backward kernel\");")
    if x0_dim is not None:
        ap(f"    TORCH_CHECK(X0 == {int(x0_dim)}, \"X0 mismatch for generated STC backward kernel\");")
    if x1_dim is not None:
        ap(f"    TORCH_CHECK(X1 == {int(x1_dim)}, \"X1 mismatch for generated STC backward kernel\");")
    if v_dim is not None:
        ap(f"    TORCH_CHECK(V == {int(v_dim)}, \"V mismatch for generated STC backward kernel\");")
    ap("    auto grad_x1 = torch::zeros_like(x1);")
    ap("    GPU_Guard device_guard(x1.device());")
    ap("    gpuStream_t stream = getCurrentGPUStream(x1.device().index());")
    ap(f"    AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), \"{kernel_name}\", [&] {{")
    ap(f"        launch_{kernel_name}<scalar_t, {int(tile_u)}>((const scalar_t*)grad_out.data_ptr<scalar_t>(),")
    ap("            (const scalar_t*)x1.data_ptr<scalar_t>(),")
    ap("            (const scalar_t*)x0.data_ptr<scalar_t>(),")
    ap("            (scalar_t*)grad_x1.data_ptr<scalar_t>(), B, X1, X0, V, U, stream);")
    ap("    });")
    ap("    GPU_KERNEL_LAUNCH_CHECK();")
    ap("    return grad_x1;")
    ap("}")
    ap("")
    ap("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    ap(f"    m.def(\"run\", &launcher_{kernel_name}, \"{kernel_name} STC backward x1-only jit impl\");")
    ap("}")
    return "\n".join(lines)


def generate_code_stc_bwd_with_scheduler(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    num_out_segments: int,
    u_dim: int,
    out_path: str = "generated_stc_bwd_lars.cu",
    kernel_name: str = "stc_lars_bwd",
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    return_schedule: bool = False,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    tile_u: int = 32,
) -> Any:
    """Generate STC backward code for grad_x1 only.

    This uses the same LARSUniform1DScheduler(path_kind="stc") to choose a
    static path order, but the emitted kernel follows the tiled STC backward
    computation: stage x1 and grad_x1 in shared memory, iterate paths, and write
    a single [B, X1, U] gradient tensor.
    """
    paths = make_stc_paths_from_padded_lists(idx_lists, coeff_list, path_lens=path_lens, pad_value=pad_value)
    scheduler = LARSUniform1DScheduler(
        paths,
        reg_budget=0,
        path_kind="stc",
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name="stc_lars_bwd_x1_only",
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    schedule_result = scheduler.schedule()

    idx_norm = _normalize_stc_padded_paths(idx_lists, coeff_list=coeff_list, path_lens=path_lens)
    if path_lens is None:
        inferred_lens = infer_stc_path_lens_from_padded(idx_norm, coeff_list, pad_value=pad_value)
    else:
        inferred_lens = path_lens.detach().cpu().to(torch.int64).reshape(-1)
    base_kernel_name = f"{kernel_name}_u{int(u_dim)}_path{len(paths)}_maxlen{int(inferred_lens.max().item())}"

    code = emit_stc_bwd_kernel_from_paths(
        paths,
        path_order=schedule_result.path_order,
        kernel_name=base_kernel_name,
        u_dim=int(u_dim),
        x0_dim=x0_dim,
        x1_dim=x1_dim,
        v_dim=int(num_out_segments),
        tile_u=int(tile_u),
    )
    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")
    if return_schedule:
        return {
            "name": "stc_lars_bwd_x1_only",
            "code": code,
            "kernel_name": base_kernel_name,
            "schedule": schedule_result,
            "num_paths": len(paths),
            "max_live": schedule_result.max_live,
        }
    return code


# =============================================================================
# LARS scheduler integration
# =============================================================================

Label = Tuple[str, int]  # ('x', i), ('y', j), ('w', k), ('o', v)

def _to_int_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().tolist()]


def _to_float_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]

@dataclass(frozen=True)
class U1DPath:
    pid: int
    i: int
    j: int
    k: int
    v: int
    c: float

    @property
    def labels(self) -> Tuple[Label, Label, Label]:
        # LARS schedules only input operands.  Output accumulators are emitted
        # as full-resident local variables and are not reload-managed candidates.
        return (("x", self.i), ("y", self.j), ("w", self.k))

    @property
    def product_labels(self) -> Tuple[Label, Label, Label]:
        # Uniform1D has no duplicate-input multiplicity beyond x/y/w.  This
        # property lets the common scheduler handle both fixed-arity Uniform1D
        # and variable-arity STC paths through the same emission path.
        return self.labels


@dataclass
class Inst:
    op: str
    args: Tuple[Any, ...]
    comment: str = ""

    def __str__(self) -> str:
        body = f"{self.op} " + ", ".join(map(str, self.args))
        return f"{body:<48} # {self.comment}" if self.comment else body


@dataclass
class ScheduleResult:
    instructions: List[Inst]
    path_order: List[int]
    max_live: int
    spills: int
    reloads: int
    final_reg_map: Dict[Label, str]
    profile: Dict[str, Any] = field(default_factory=dict)


class LARSUniform1DScheduler:
    """
    All-input-resident LARS-style scheduler for Uniform1D paths:

        out[v] += x[i] * y[j] * w[k] * c

    This scheduler no longer implements spill/victim selection.  It allocates
    one virtual register for every distinct input label, chooses labels by the
    LARS score, and fires paths as soon as their input labels are live.

    Notes:
      - reg_budget is kept only for API compatibility and is ignored.
      - ScheduleResult.spills is kept for compatibility and is always zero.
      - Output accumulators are handled by the emitter, not by the LARS label
        allocator.
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        reg_budget: int = 16,
        *,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        path_fallback_after: Optional[int] = None,
        prefer_path_fallback_when_full: bool = True,
        debug: bool = False,
        path_kind: str = "u1d",
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        self.path_kind = str(path_kind)
        if self.path_kind == "u1d":
            self.paths: List[Any] = [
                U1DPath(pid=p, i=i, j=j, k=k, v=v, c=c)
                for p, (i, j, k, v, c) in enumerate(paths)
            ]
        elif self.path_kind == "stc":
            # STC paths are already materialized as STCPath objects.  They expose
            # labels for live-set management and product_labels for multiplicity-
            # preserving emission, e.g. x1[i] * x1[i].
            self.paths = list(paths)
            for expected_pid, path in enumerate(self.paths):
                if int(path.pid) != expected_pid:
                    raise ValueError(
                        f"STC paths must be dense pid-ordered; "
                        f"got path.pid={path.pid} at position {expected_pid}"
                    )
        else:
            raise ValueError(f"Unsupported LARSUniform1DScheduler path_kind={self.path_kind!r}")

        # reg_budget is kept only for API compatibility.
        # Input labels are now unbounded: allocate one virtual register for every
        # distinct x/y/w label that appears in the schedule, so normal inputs are
        # never exceed the all-input-resident virtual register fileed because of a user-supplied budget.
        del reg_budget
        input_label_count = len({lab for path in self.paths for lab in path.labels})
        self.reg_budget = max(4, int(input_label_count))

        # path_fallback_after and prefer_path_fallback_when_full are legacy
        # keyword arguments.  They are intentionally ignored now that spill and
        # victim selection have been removed.
        del path_fallback_after, prefer_path_fallback_when_full

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
        self.debug = bool(debug)

        self.label_to_paths: Dict[Label, Set[int]] = defaultdict(set)
        for path in self.paths:
            for lab in path.labels:
                self.label_to_paths[lab].add(path.pid)

        self.path_label_sets: List[Set[Label]] = [set(p.labels) for p in self.paths]
        self.unscheduled: Set[int] = set(p.pid for p in self.paths)

        self.remaining_uses: Counter[Label] = Counter()
        for p in self.paths:
            for lab in p.labels:
                self.remaining_uses[lab] += 1

        # Forward output accumulator use counts.  out[v] with exactly one path
        # does not need a full-resident accumulator: emit one direct writeback
        # immediately after its computation instead of keeping out_acc_v alive
        # until the end of the kernel.
        self.output_uses: Counter[int] = Counter(int(p.v) for p in self.paths)

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = [f"r{r}" for r in range(self.reg_budget)]

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        # max_live is kept for backward compatibility. It records the maximum
        # number of live normal input labels in the logical schedule.
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}

        # Optional profile / progress reporting.  Kept disabled by default so
        # existing code generation remains silent unless requested.
        self.profile_enabled = bool(profile)
        self.profile_name = str(profile_name or self.__class__.__name__)
        self.profile_interval = int(profile_interval)
        self.profile_seconds = float(profile_seconds)
        self.profile_print = bool(profile_print)
        self._profile_t0 = perf_counter()
        self._profile_last_print_t = self._profile_t0
        self._profile_last_print_done = 0
        self._profile_loop_iter = 0
        self._profile_select_rounds = 0
        self._profile_fire_rounds = 0
        self._profile_load_rounds = 0
        self._profile_records: List[Dict[str, Any]] = []
        self._profile_last_event = ""
        self._profile_last_reason = ""

    # ------------------------------------------------------------------
    # Profile helpers
    # ------------------------------------------------------------------
    def _profile_total_live_regs(self) -> int:
        if hasattr(self, "_total_live_regs"):
            try:
                return int(self._total_live_regs())
            except Exception:
                pass
        return len(self.live)

    def _fmt_profile_eta(self, seconds: float) -> str:
        if seconds == float("inf"):
            return "inf"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"

    def _profile_candidate_count_safe(self) -> int:
        try:
            return len(self._candidate_labels())
        except Exception:
            return 0


    def _profile_snapshot(
        self,
        *,
        event: str,
        reason: str = "",
        fireable_n: Optional[int] = None,
        candidate_n: Optional[int] = None,
        select_ms: Optional[float] = None,
        lab: Optional[Label] = None,
    ) -> Dict[str, Any]:
        done = len(self.path_order)
        remain = len(self.unscheduled)
        total = len(self.paths)
        elapsed = max(perf_counter() - self._profile_t0, 1e-12)
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (remain / rate) if rate > 0 else float("inf")
        snap: Dict[str, Any] = {
            "name": self.profile_name,
            "event": event,
            "iter": int(self._profile_loop_iter),
            "done_paths": int(done),
            "remaining_paths": int(remain),
            "total_paths": int(total),
            "progress_pct": float(100.0 * done / max(1, total)),
            "live_labels": int(len(self.live)),
            "live_regs_total": int(self._profile_total_live_regs()),
            "free_regs": int(len(self.free_regs)),
            "reg_budget": int(self.reg_budget),
            "max_live": int(self.max_live),
            "max_live_labels": int(getattr(self, "max_live_labels", self.max_live)),
            "max_live_total": int(getattr(self, "max_live_total", self.max_live)),
            "fireable_paths": int(fireable_n if fireable_n is not None else 0),
            "candidate_labels": int(candidate_n if candidate_n is not None else 0),
            "select_ms": float(select_ms or 0.0),
            "spills": int(self.spills),
            "reloads": int(self.reloads),
            "instructions": int(len(self.instructions)),
            "select_rounds": int(self._profile_select_rounds),
            "fire_rounds": int(self._profile_fire_rounds),
            "load_rounds": int(self._profile_load_rounds),
            "elapsed_s": float(elapsed),
            "rate_paths_per_s": float(rate),
            "eta_s": float(eta),
            "reason": str(reason),
            "label": None if lab is None else self._label_name(lab),
        }
        return snap

    def _profile_emit(
        self,
        *,
        event: str,
        force: bool = False,
        reason: str = "",
        fireable_n: Optional[int] = None,
        candidate_n: Optional[int] = None,
        select_ms: Optional[float] = None,
        lab: Optional[Label] = None,
    ) -> None:
        if not self.profile_enabled:
            return

        now = perf_counter()
        done = len(self.path_order)
        enough_paths = (
            self.profile_interval > 0
            and done > 0
            and done != self._profile_last_print_done
            and done % self.profile_interval == 0
        )
        enough_time = (now - self._profile_last_print_t) >= self.profile_seconds
        if not (force or enough_paths or enough_time):
            return

        snap = self._profile_snapshot(
            event=event,
            reason=reason,
            fireable_n=fireable_n,
            candidate_n=candidate_n,
            select_ms=select_ms,
            lab=lab,
        )
        self._profile_records.append(snap)
        self._profile_last_event = event
        self._profile_last_reason = reason

        if self.profile_print:
            eta = self._fmt_profile_eta(snap["eta_s"])
            elapsed = self._fmt_profile_eta(snap["elapsed_s"])
            print(
                f"[LARS][profile] {snap['name']} {event:>8s} "
                f"iter={snap['iter']} "
                f"done={snap['done_paths']}/{snap['total_paths']} "
                f"remain={snap['remaining_paths']} "
                f"live={snap['live_regs_total']}/{snap['reg_budget']} "
                f"max=label/total "
                f"{snap['max_live_labels']}/{snap['max_live_total']} "
                f"free={snap['free_regs']} "
                f"fireable={snap['fireable_paths']} "
                f"candidates={snap['candidate_labels']} "
                f"label={snap['label']} "
                f"select={snap['select_ms']:.2f}ms "
                f"spills={snap['spills']} reloads={snap['reloads']} "
                f"insts={snap['instructions']} "
                f"rate={snap['rate_paths_per_s']:.1f} path/s "
                f"elapsed={elapsed} eta={eta} "
                f"reason={snap['reason']}",
                flush=True,
            )

        self._profile_last_print_t = now
        self._profile_last_print_done = done

    def _build_profile_summary(self) -> Dict[str, Any]:
        elapsed = max(perf_counter() - self._profile_t0, 1e-12)
        total = len(self.paths)
        done = len(self.path_order)
        remain = len(self.unscheduled)
        summary: Dict[str, Any] = {
            "name": self.profile_name,
            "total_paths": int(total),
            "done_paths": int(done),
            "remaining_paths": int(remain),
            "reg_budget": int(self.reg_budget),
            "max_live": int(self.max_live),
            "max_live_labels": int(getattr(self, "max_live_labels", self.max_live)),
            "max_live_total": int(getattr(self, "max_live_total", self.max_live)),
            "spills": int(self.spills),
            "reloads": int(self.reloads),
            "instructions": int(len(self.instructions)),
            "select_rounds": int(self._profile_select_rounds),
            "fire_rounds": int(self._profile_fire_rounds),
            "load_rounds": int(self._profile_load_rounds),
            "elapsed_s": float(elapsed),
            "rate_paths_per_s": float(done / elapsed if elapsed > 0 else 0.0),
            "records": list(self._profile_records),
        }
        return summary

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _path(self, pid: int) -> U1DPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        if kind == "o":
            return f"out[{idx}]"
        return f"{kind}[{idx}]"

    def _invalidate_score_cache(self) -> None:
        self._score_cache.clear()

    def _alloc_reg_no_spill(self, lab: Label) -> str:
        if not self.free_regs:
            raise RuntimeError("_alloc_reg_no_spill called with no free registers")

        reg = self.free_regs.pop(0)
        self.reg_of[lab] = reg
        self.live.add(lab)
        self.max_live_labels = max(self.max_live_labels, len(self.live))
        self.max_live_total = max(self.max_live_total, len(self.live))
        self.max_live = max(self.max_live, len(self.live))
        self._invalidate_score_cache()
        return reg

    def _emit_load_inst(self, lab: Label, reg: str, reason: str) -> None:
        if lab[0] == "o":
            self.instructions.append(
                Inst("load_acc", (reg, self._label_name(lab)), reason)
            )
        else:
            self.instructions.append(
                Inst("load", (reg, self._label_name(lab)), reason)
            )
        self.reloads += 1

    def _store_and_release(self, lab: Label, reason: str = "") -> None:
        if lab not in self.live:
            return

        reg = self.reg_of[lab]

        if lab[0] == "o":
            # If the output accumulator was never modified, a store_acc is not
            # needed. Emit a plain release comment instead. The emitter can
            # ignore release instructions.
            if lab in self.dirty_outputs:
                self.instructions.append(
                    Inst("store_acc", (self._label_name(lab), reg), reason)
                )
                self.dirty_outputs.discard(lab)
            else:
                self.instructions.append(
                    Inst("release", (reg, self._label_name(lab)), reason)
                )
        else:
            self.instructions.append(
                Inst("release", (reg, self._label_name(lab)), reason)
            )

        self.live.remove(lab)
        del self.reg_of[lab]
        self.free_regs.insert(0, reg)
        self._invalidate_score_cache()

    # ------------------------------------------------------------------
    # Fireability
    # ------------------------------------------------------------------
    def _is_fireable(self, pid: int) -> bool:
        return self.path_label_sets[pid] <= self.live

    def _fireable_paths(self) -> List[int]:
        return [pid for pid in self.unscheduled if self.path_label_sets[pid] <= self.live]

    def _fireable_count_after_load(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
        count = 0
        for pid in self.unscheduled:
            if self.path_label_sets[pid] <= tmp_live:
                count += 1
        return count

    def _best_fireable_release_after_load(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
        best = 0
        for pid in self.unscheduled:
            labels = self.path_label_sets[pid]
            if labels <= tmp_live:
                release_now = sum(1 for l in labels if self.remaining_uses[l] == 1)
                best = max(best, release_now)
        return best

    # ------------------------------------------------------------------
    # LARS score
    # ------------------------------------------------------------------
    def _release_potential_if_add(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
        score = 0

        for pid in self.unscheduled:
            labels = self.path_label_sets[pid]
            if labels <= tmp_live:
                for l in labels:
                    if self.remaining_uses[l] == 1:
                        score += 2 if l in self.live else 1

        return score

    def _fire_potential_if_add(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
        count = 0

        for pid in self.label_to_paths[lab] & self.unscheduled:
            if self.path_label_sets[pid] <= tmp_live:
                count += 1

        return count

    def _primary_affinity(self, lab: Label) -> int:
        score = 0
        for pid in self.label_to_paths[lab] & self.unscheduled:
            score += len(self.path_label_sets[pid] & self.live)
        return score

    def _secondary_affinity(self, lab: Label) -> int:
        if not self.enable_secondary_affinity:
            return 0

        neighbor_labels: Set[Label] = set()
        for pid in self.unscheduled:
            labels = self.path_label_sets[pid]
            if labels & self.live:
                neighbor_labels |= labels

        score = 0
        for pid in self.label_to_paths[lab] & self.unscheduled:
            labels = self.path_label_sets[pid]
            score += len((labels & neighbor_labels) - self.live - {lab})
        return score

    def _non_live_affinity_penalty(self, lab: Label) -> int:
        neighbors: Set[Label] = set()
        for pid in self.label_to_paths[lab] & self.unscheduled:
            neighbors |= self.path_label_sets[pid]
        return len(neighbors - self.live - {lab})

    def _priority(self, lab: Label) -> int:
        acc_bonus = 1 if lab[0] == "o" else 0
        return self.remaining_uses[lab] + acc_bonus

    def _label_score(self, lab: Label) -> Tuple[int, int, int, int, int, int]:
        cached = self._score_cache.get(lab)
        if cached is not None:
            return cached

        score = (
            self._release_potential_if_add(lab),
            self._fire_potential_if_add(lab),
            self._primary_affinity(lab),
            self._secondary_affinity(lab),
            -self._non_live_affinity_penalty(lab),
            self._priority(lab),
        )
        self._score_cache[lab] = score
        return score

    def _candidate_labels(self) -> List[Label]:
        candidates = {
            lab
            for pid in self.unscheduled
            for lab in self.path_label_sets[pid]
            if lab not in self.live
        }

        if not candidates:
            raise RuntimeError("No candidate label but no path is fireable.")

        # Optional cheap prefilter. This is useful for large path lists, because
        # full LARS score repeatedly scans unscheduled paths.
        if self.topk_candidates is not None and len(candidates) > self.topk_candidates:
            ranked = sorted(
                candidates,
                key=lambda lab: (
                    self.remaining_uses[lab],
                    len(self.label_to_paths[lab] & self.unscheduled),
                    1 if lab[0] == "o" else 0,
                    str(lab),
                ),
                reverse=True,
            )
            return ranked[: int(self.topk_candidates)]

        return list(candidates)

    # ------------------------------------------------------------------
    # Label selection / no-spill load
    # ------------------------------------------------------------------
    def _select_next_label_global(self) -> Tuple[Label, str]:
        candidates = self._candidate_labels()

        best_lab: Optional[Label] = None
        best_key = None

        for lab in candidates:
            lab_score = self._label_score(lab)
            fire_after = self._fireable_count_after_load(lab)
            release_after = self._best_fireable_release_after_load(lab)
            key = (
                fire_after,
                release_after,
                lab_score,
                str(lab),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_lab = lab

        if best_lab is None:
            raise RuntimeError("No candidate label but no path is fireable.")
        return best_lab, f"global key={best_key}"

    def _load_label_no_spill(self, lab: Label, reason: str = "") -> None:
        if lab in self.live:
            return
        if not self.free_regs:
            raise RuntimeError(
                "No free virtual register in no-spill scheduler. "
                "This should not happen because reg_budget is derived from "
                "the number of distinct input labels."
            )

        reg = self._alloc_reg_no_spill(lab)
        self._emit_load_inst(lab, reg, reason)

    # ------------------------------------------------------------------
    # Path firing
    # ------------------------------------------------------------------
    def _choose_fireable_path(self, fireable: List[int]) -> int:
        def key(pid: int):
            labels = self.path_label_sets[pid]
            p = self._path(pid)
            release_now = sum(1 for l in labels if self.remaining_uses[l] == 1)
            reuse_score = sum(self.remaining_uses[l] for l in labels)

            if getattr(self, "path_kind", "u1d") == "stc":
                # Prefer shorter products as a final tie-breaker; this keeps the
                # emitted STC expression compact when all other reuse signals tie.
                arity = len(getattr(p, "product_labels", p.labels))
                return (release_now, reuse_score, -arity, -pid)

            out_dirty = 1 if ("o", p.v) in self.dirty_outputs else 0
            return (release_now, out_dirty, reuse_score, -pid)

        return max(fireable, key=key)

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)

        if getattr(self, "path_kind", "u1d") == "stc":
            product_labels = tuple(getattr(p, "product_labels", p.labels))
            regs = tuple(self.reg_of[lab] for lab in product_labels)
            if self.output_uses[int(p.v)] <= 1:
                self.instructions.append(
                    Inst(
                        "mul_stc_direct",
                        (int(p.v), regs, float(p.c)),
                        f"path#{pid}: direct out[{p.v}] += c * product",
                    )
                )
            else:
                self.instructions.append(
                    Inst(
                        "mul_stc_resident",
                        (int(p.v), regs, float(p.c)),
                        f"path#{pid}: out[{p.v}] += c * product",
                    )
                )
        else:
            lx, ly, lw = p.labels

            rx = self.reg_of[lx]
            ry = self.reg_of[ly]
            rw = self.reg_of[lw]

            if self.output_uses[int(p.v)] <= 1:
                self.instructions.append(
                    Inst(
                        "fma_u1d_direct",
                        (p.v, rx, ry, rw, p.c),
                        (
                            f"path#{pid}: direct out[{p.v}] += "
                            f"x[{p.i}] * y[{p.j}] * w[{p.k}] * {p.c}"
                        ),
                    )
                )
            else:
                self.instructions.append(
                    Inst(
                        "fma_u1d_resident",
                        (p.v, rx, ry, rw, p.c),
                        f"path#{pid}: out[{p.v}] += x[{p.i}] * y[{p.j}] * w[{p.k}] * {p.c}",
                    )
                )

        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        for lab in p.labels:
            self.remaining_uses[lab] -= 1

        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")

    # ------------------------------------------------------------------
    # Public scheduling entry
    # ------------------------------------------------------------------
    def schedule(self) -> ScheduleResult:
        self._profile_emit(event="start", force=True, reason="schedule start")

        while self.unscheduled:
            self._profile_loop_iter += 1

            fireable = self._fireable_paths()
            fireable_n = len(fireable)
            if fireable:
                pid = self._choose_fireable_path(fireable)
                self._emit_path_compute(pid)
                self._profile_fire_rounds += 1
                self._profile_emit(
                    event="fire",
                    reason=f"path#{pid}",
                    fireable_n=fireable_n,
                    candidate_n=0,
                )
                continue

            select_t0 = perf_counter()
            candidate_n = self._profile_candidate_count_safe()
            lab, reason = self._select_next_label_global()
            select_ms = (perf_counter() - select_t0) * 1000.0
            self._profile_select_rounds += 1

            self._load_label_no_spill(lab, reason=reason)
            self._profile_load_rounds += 1
            self._profile_emit(
                event="load",
                reason=reason,
                fireable_n=fireable_n,
                candidate_n=candidate_n,
                select_ms=select_ms,
                lab=lab,
            )

        for lab in list(self.live):
            self._store_and_release(lab, reason="end of schedule")

        self._profile_emit(event="end", force=True, reason="schedule end")
        profile_summary = self._build_profile_summary()

        return ScheduleResult(
            instructions=self.instructions,
            path_order=self.path_order,
            max_live=self.max_live,
            spills=0,
            reloads=self.reloads,
            final_reg_map=dict(self.reg_of),
            profile=profile_summary,
        )


PairKey = Tuple[Label, Label]  # (w-label, x-label)




class ProgressLARSUniform1DScheduler(LARSUniform1DScheduler):
    """Progress-printing wrapper for the no-spill all-input-resident scheduler."""

    def __init__(
        self,
        *args,
        progress_interval: int = 1000,
        progress_seconds: float = 2.0,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.progress_interval = int(progress_interval)
        self.progress_seconds = float(progress_seconds)
        self.verbose = bool(verbose)

        self._total_paths = len(self.paths)
        self._loop_iter = 0
        self._select_rounds = 0
        self._fire_rounds = 0
        self._load_rounds = 0

        self._t0 = perf_counter()
        self._last_print_t = self._t0
        self._last_print_done = 0

        self._last_fireable_n = 0
        self._last_candidate_n = 0
        self._last_best_lab = None
        self._last_best_score = None
        self._last_select_ms = 0.0
        self._last_select_mode = ""
        self._last_select_reason = ""

    def _fmt_lab(self, lab) -> str:
        if lab is None:
            return "None"
        kind, idx = lab
        return f"{kind}[{idx}]"

    def _fmt_eta(self, seconds: float) -> str:
        if seconds == float("inf"):
            return "inf"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"

    def _candidate_count(self) -> int:
        try:
            return len(self._candidate_labels())
        except Exception:
            return 0

    def _print_progress(self, *, force: bool = False, event: str = "") -> None:
        if not self.verbose:
            return

        now = perf_counter()
        done = len(self.path_order)
        remain = len(self.unscheduled)

        enough_paths = (
            self.progress_interval > 0
            and done > 0
            and done != self._last_print_done
            and done % self.progress_interval == 0
        )
        enough_time = (now - self._last_print_t) >= self.progress_seconds
        if not (force or enough_paths or enough_time):
            return

        elapsed = max(now - self._t0, 1e-12)
        rate = done / elapsed
        eta = (remain / rate) if rate > 0 else float("inf")
        pct = 100.0 * done / max(1, self._total_paths)

        print(
            f"[LARS] {event:>8s} "
            f"iter={self._loop_iter} "
            f"done={done}/{self._total_paths} ({pct:.2f}%) "
            f"remain={remain} "
            f"live={len(self.live)}/{self.reg_budget} "
            f"fireable={self._last_fireable_n} "
            f"candidates={self._last_candidate_n} "
            f"mode={self._last_select_mode} "
            f"best={self._fmt_lab(self._last_best_lab)} "
            f"score={self._last_best_score} "
            f"select={self._last_select_ms:.2f}ms "
            f"reloads={self.reloads} "
            f"insts={len(self.instructions)} "
            f"rate={rate:.1f} path/s "
            f"elapsed={self._fmt_eta(elapsed)} eta={self._fmt_eta(eta)} "
            f"reason={self._last_select_reason}",
            flush=True,
        )

        self._last_print_t = now
        self._last_print_done = done

    def _select_label_with_progress(self):
        t0 = perf_counter()
        self._last_candidate_n = self._candidate_count()
        lab, reason = self._select_next_label_global()
        self._select_rounds += 1
        self._last_select_ms = (perf_counter() - t0) * 1000.0
        self._last_best_lab = lab
        self._last_select_mode = "global"
        self._last_select_reason = str(reason)
        try:
            self._last_best_score = self._label_score(lab)
        except Exception:
            self._last_best_score = None
        return lab, reason

    def schedule(self) -> ScheduleResult:
        self._print_progress(force=True, event="start")

        while self.unscheduled:
            self._loop_iter += 1
            fireable = self._fireable_paths()
            self._last_fireable_n = len(fireable)

            if fireable:
                pid = self._choose_fireable_path(fireable)
                self._emit_path_compute(pid)
                self._fire_rounds += 1
                self._last_select_mode = "fire"
                self._last_best_lab = None
                self._last_best_score = None
                self._last_select_reason = f"path#{pid}"
                self._print_progress(event="fire")
                continue

            lab, reason = self._select_label_with_progress()
            self._load_rounds += 1
            self._load_label_no_spill(
                lab,
                reason=(
                    f"LARS-score={self._last_best_score}, "
                    f"mode={self._last_select_mode}, "
                    f"select_reason={reason}"
                ),
            )
            self._print_progress(event="load")

        for lab in list(self.live):
            self._store_and_release(lab, reason="end of schedule")

        self._print_progress(force=True, event="end")

        if self.verbose:
            print(
                f"[LARS] summary "
                f"select_rounds={self._select_rounds} "
                f"fire_rounds={self._fire_rounds} "
                f"load_rounds={self._load_rounds} "
                f"max_live={self.max_live} "
                f"spills=0 "
                f"reloads={self.reloads}",
                flush=True,
            )

        return ScheduleResult(
            instructions=self.instructions,
            path_order=self.path_order,
            max_live=self.max_live,
            spills=0,
            reloads=self.reloads,
            final_reg_map=dict(self.reg_of),
            profile=self._build_profile_summary(),
        )


# =============================================================================
# CUDA codegen from LARSUniform1DScheduler ScheduleResult
# =============================================================================

def _lars_reg_id(reg: str) -> int:
    if not isinstance(reg, str) or not reg.startswith("r"):
        raise ValueError(f"Bad LARS register name: {reg!r}")
    return int(reg[1:])


def _parse_lars_label_ref(ref: str) -> Tuple[str, int]:
    """Parse refs emitted by LARSUniform1DScheduler: x[0], y[1], w[2], out[3]."""
    if not isinstance(ref, str) or "[" not in ref or not ref.endswith("]"):
        raise ValueError(f"Bad LARS label ref: {ref!r}")
    kind, rest = ref.split("[", 1)
    idx = int(rest[:-1])
    if kind == "out":
        kind = "o"
    if kind not in ("x", "y", "w", "o"):
        raise ValueError(f"Bad LARS label kind in ref: {ref!r}")
    return kind, idx


def _fmt_lars_float(x: float) -> str:
    # repr(float) is accepted by CUDA for normal finite constants.
    return repr(float(x))


def _sanitize_cuda_comment(s: str) -> str:
    return str(s).replace("\n", " ").replace("\r", " ").replace("*/", "* /")




def _lars_has_output_reload_after_store(schedule_result: ScheduleResult) -> bool:
    """
    Scatter mode cannot safely use out[] as an accumulator temporary buffer, because
    another block may atomically update the same out element between a temporary store
    and a later reload. Dense mode is safe because one block owns one output row.
    """
    stored = set()
    for inst in schedule_result.instructions:
        if inst.op == "store_acc":
            lab = _parse_lars_label_ref(inst.args[0])
            stored.add(lab)
        elif inst.op == "load_acc":
            lab = _parse_lars_label_ref(inst.args[1])
            if lab in stored:
                return True
    return False


def _lars_label_index_expr(
    kind: str,
    idx: int,
    *,
    mode_scalar_y: bool,
    u_dim: Optional[int],
    x_dim: Optional[int],
    y_dim: Optional[int],
    w_dim: Optional[int],
    v_dim: Optional[int],
) -> str:
    if kind == "x":
        if x_dim is not None and idx >= x_dim:
            raise ValueError(f"x index {idx} out of x_dim={x_dim}")
        if u_dim is not None:
            return f"x_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"x_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "w":
        if w_dim is not None and idx >= w_dim:
            raise ValueError(f"w index {idx} out of w_dim={w_dim}")
        if u_dim is not None:
            return f"w_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"w_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "y":
        if y_dim is not None and idx >= y_dim:
            raise ValueError(f"y index {idx} out of y_dim={y_dim}")
        if mode_scalar_y:
            return f"y_base + (index_t){idx}"
        if u_dim is not None:
            return f"y_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"y_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "o":
        if v_dim is not None and idx >= v_dim:
            raise ValueError(f"out index {idx} out of v_dim={v_dim}")
        if u_dim is not None:
            return f"out_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"out_base + (index_t){idx} * (index_t)U + (index_t)u"

    raise ValueError(f"Bad label kind: {kind}")



def emit_launcher(
    bundle_name: str,
    mode: str = "u,u,,u",
    *,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
    if mode == "u,u,,u":
        y_comment = "[B,Ky,1] or [S,Ky,1]"
        y_check_u = (
            'TORCH_CHECK((int)y.size(2) == 1, '
            '"y.size(2) must be 1 for mode u,u,,u");'
        )
    elif mode == "u,u,u,u":
        y_comment = "[B,Ky,U] or [S,Ky,U]"
        y_check_u = (
            'TORCH_CHECK((int)y.size(2) == U, '
            '"y.size(2) must equal U for mode u,u,u,u");'
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    param_lines = [
        "    torch::Tensor w",
        "    torch::Tensor x_all",
        "    torch::Tensor y",
    ]

    if use_x_src or use_y_src:
        param_lines.append("    torch::Tensor src_idx")

    if use_scatter:
        param_lines.append("    torch::Tensor dst_idx")

    param_lines.append("    int64_t V64")

    params = ",\n".join(param_lines)

    src_check = ""
    if use_x_src or use_y_src:
        src_check = r'''
    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''

    dst_check = ""
    if use_scatter:
        dst_check = r'''
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''

    if use_scatter:
        out_alloc = "    auto out = torch::zeros({S, V, U}, w.options());"
    else:
        out_alloc = "    auto out = torch::zeros({B, V, U}, w.options());"

    blist_logic = ""

    src_numel_check = ""
    if use_x_src or use_y_src:
        src_numel_check = (
            '    TORCH_CHECK(src_idx.numel() >= B, '
            '"src_idx numel must be >= B");'
        )

    dst_numel_check = ""
    if use_scatter:
        dst_numel_check = (
            '    TORCH_CHECK(dst_idx.numel() >= B, '
            '"dst_idx numel must be >= B");'
        )

    launch_src_arg = (
        "(const int32_t*)src_idx.data_ptr<int32_t>(),"
        if (use_x_src or use_y_src)
        else "nullptr,"
    )

    launch_dst_arg = (
        "(const int32_t*)dst_idx.data_ptr<int32_t>(),"
        if use_scatter
        else "nullptr,"
    )

    # ------------------------------------------------------------------
    # Select B source according to execution mode.
    #
    # scatter mode:
    #   B is number of destination/scatter entries.
    #
    # src-index mode:
    #   B is number of source-index entries.
    #
    # dense mode:
    #   B comes from w.size(0), where w is either [1,Iw,U] or [B,Iw,U].
    # ------------------------------------------------------------------
    if use_scatter:
        b_expr = "dst_idx.size(0)"
    elif use_x_src or use_y_src:
        b_expr = "src_idx.size(0)"
    else:
        b_expr = "w.size(0)"

    return rf'''

torch::Tensor launcher_{bundle_name}(
{params})
{{
    // Expected tensors:
    //   w      : [WB, Iw, U], WB can be 1 or B
    //   x_all  : [S, Ix, U] or [B, Ix, U]
    //   y      : {y_comment}
    // Optional:
    //   src_idx: [?] int32, enabled when x/y source indirection is used
    //   dst_idx: [?] int32, enabled when scatter is used

    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(),
                "w/x_all/y must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(),
                "w/x_all/y must be contiguous");
{src_check}{dst_check}

    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x_all.dim() == 3, "x_all must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");

    int B  = (int){b_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x_all.size(0);
    int Ix = (int)x_all.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");

    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(1) > 0, "Iw must be > 0");

    TORCH_CHECK((int)x_all.size(1) > 0, "Ix must be > 0");
    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");

    {y_check_u}

    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
{src_numel_check}
{dst_numel_check}

{blist_logic}
{out_alloc}

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{

        TORCH_CHECK(x_all.scalar_type() == w.scalar_type(),
                    "x_all dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(),
                    "y dtype must match w");

        launch_{bundle_name}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    }});

    GPU_KERNEL_LAUNCH_CHECK();

    return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} forward jit impl");
}}
'''



def _max_lars_reg_count_any(schedule_result: ScheduleResult) -> int:
    """
    Return max virtual register id + 1 by scanning string register arguments
    in the logical instruction stream.
    """
    max_id = -1
    for inst in schedule_result.instructions:
        for arg in inst.args:
            if isinstance(arg, str) and arg.startswith("r"):
                try:
                    max_id = max(max_id, _lars_reg_id(arg))
                except ValueError:
                    pass
    return max_id + 1






def emit_fused_fwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    w_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """
    Emit CUDA/HIP-compatible forward code directly from LARS instruction order.

    Supported logical ops:
      - load / release
      - fma_u1d_resident: accumulate into full-resident out_acc_v_<idx>
      - fma_u1d_direct: direct writeback for single-use out[v]
      - fma_u1d / load_acc / store_acc are kept only for old ScheduleResult
        compatibility.

    LARS-native path semantics:
        out[v] += x[i] * y[j] * w[k] * c
    """
    if block_size != 32:
        raise ValueError("this emitter currently assumes block_size=32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    reg_count = _max_lars_reg_count_any(schedule_result)

    resident_out_indices = sorted({
        int(inst.args[0])
        for inst in schedule_result.instructions
        if inst.op == "fma_u1d_resident"
    })
    direct_out_indices = sorted({
        int(inst.args[0])
        for inst in schedule_result.instructions
        if inst.op == "fma_u1d_direct"
    })

    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    def emit_out_write(idx: int, value_expr: str) -> None:
        expr = _lars_label_index_expr(
            "o", int(idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            w_dim=w_dim,
            v_dim=v_dim,
        )
        if use_scatter:
            ap(f"            atomicAdd(&out[{expr}], {value_expr});")
        else:
            ap(f"            out[{expr}] += {value_expr};")

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    scalar_t* __restrict__ out,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
        ap("")
    ap("    const int e_orig = e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    if use_x_src:
        ap("    const int x_row = src_idx[e_orig];")
    else:
        ap("    const int x_row = e_local;")
    if use_y_src:
        ap("    const int y_row = src_idx[e_orig];")
    else:
        ap("    const int y_row = e_orig;")
    if use_scatter:
        ap("    const int out_row = dst_idx[e_orig];")
    else:
        ap("    const int out_row = e_orig;")
    ap("")
    ap("    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;")
    ap("    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;")
    if mode_scalar_y:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky;")
    else:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky * (index_t)U;")
    ap("    const index_t out_base = (index_t)out_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    for out_idx in resident_out_indices:
        ap(f"            scalar_t out_acc_v_{out_idx} = scalar_t(0);")
    if resident_out_indices:
        ap("")
    if direct_out_indices:
        ap(f"            // direct single-use output writeback enabled for {len(direct_out_indices)} out accumulator(s)")
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op == "load":
            reg, ref = inst.args
            kind, idx = _parse_lars_label_ref(ref)
            if kind == "o":
                raise ValueError("load must not be used for output labels")
            expr = _lars_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                w_dim=w_dim,
                v_dim=v_dim,
            )
            ap(f"            {reg} = {kind}[{expr}];")

        elif inst.op == "load_acc":
            reg, ref = inst.args
            kind, _idx = _parse_lars_label_ref(ref)
            if kind != "o":
                raise ValueError("load_acc expects an output label")
            ap(f"            {reg} = scalar_t(0);")

        elif inst.op == "fma_u1d_resident":
            out_idx, rx, ry, rw, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            ap(f"            out_acc_v_{int(out_idx)} += scalar_t({c}) * ({rw} * {rx}) * {ry};")

        elif inst.op == "fma_u1d_direct":
            out_idx, rx, ry, rw, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            emit_out_write(
                int(out_idx),
                f"scalar_t({c}) * ({rw} * {rx}) * {ry}",
            )

        elif inst.op == "fma_u1d":
            ro, rx, ry, rw, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            ap(f"            {ro} += scalar_t({c}) * ({rw} * {rx}) * {ry};")

        elif inst.op == "store_acc":
            ref, reg = inst.args
            kind, idx = _parse_lars_label_ref(ref)
            if kind != "o":
                raise ValueError("store_acc expects an output label")
            emit_out_write(int(idx), str(reg))

        elif inst.op == "release":
            pass
        else:
            raise ValueError(f"Unsupported LARS instruction op: {inst.op}")

    if resident_out_indices:
        ap("")
        ap("            // resident output accumulator writeback")
    for out_idx in resident_out_indices:
        emit_out_write(int(out_idx), f"out_acc_v_{out_idx}")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32(a, b)) return false;")
    ap("    return mul_fits_int32(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_fwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t out_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool out_ok = mul3_fits_int32(out_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && out_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        w, x, y, out,")
    ap("        src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_fwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    ap("            w, x, y, out, src_idx, dst_idx,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    ap("            w, x, y, out, src_idx, dst_idx,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    ap("        w, x, y, out, src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def _make_lars_paths_from_uniform1d_lists(
    i_list: List[int],
    j_list: List[int],
    k_list: List[int],
    v_list: List[int],
    coeff_list: List[float],
    *,
    path_semantics: str,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Convert external Uniform1D path indices to LARS-native tuples.

    path_semantics="wxy" keeps compatibility with the previous emitter in this
    file, where external (i,j,k) means w[i], x[j], y[k]. Since LARS expects
    x[i], y[j], w[k], we feed it as (j,k,i,v,c).

    path_semantics="xyw" uses LARS-native semantics directly: x[i], y[j], w[k].
    """
    if path_semantics not in ("wxy", "xyw"):
        raise ValueError("path_semantics must be 'wxy' or 'xyw'")
    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

    paths: List[Tuple[int, int, int, int, float]] = []
    for i, j, k, v, c in zip(i_list, j_list, k_list, v_list, coeff_list):
        i = int(i); j = int(j); k = int(k); v = int(v); c = float(c)
        if path_semantics == "wxy":
            paths.append((j, k, i, v, c))  # LARS native: x[j], y[k], w[i]
        else:
            paths.append((i, j, k, v, c))  # LARS native: x[i], y[j], w[k]
    return paths










def generate_code_uniform1d_fwd_with_scheduler(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    mode: str = "u,u,,u",
    out_path: str = "generated_uniform1d_fwd_lars.cu",
    kernel_name: str = "stp_codegen_lars",
    scalar_t: str = "float",
    reg_budget: Union[int, Iterable[int]] = 16,
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    fold_reg_budgets_by_max_live: bool = True,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
):
    """
    Generate one LARS forward CUDA implementation.

    The user reg_budget and fold_reg_budgets_by_max_live arguments are kept for
    API compatibility, but this slim scheduler derives an unbounded virtual
    input-register file from all distinct x/y/w labels and emits a single
    single candidate.
    """
    del scalar_t, reg_budget, fold_reg_budgets_by_max_live

    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    lars_paths = _make_lars_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )

    cand_name = "lars_all_inputs"

    scheduler = LARSUniform1DScheduler(
        paths=lars_paths,
        reg_budget=0,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name=cand_name,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    schedule_result = scheduler.schedule()

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"

    if kernel_name and kernel_name != "stp_codegen_lars":
        base_kernel_name = f"{kernel_name}_{cand_name}"
    else:
        safe_cand = cand_name.replace("-", "_").replace(".", "_")
        base_kernel_name = (
            f"uniform1d_{safe_cand}_u{u_dim}_path{P}_"
            f"{mode_str}_{layout_tag}_fwd"
        )

    code = emit_fused_fwd_kernel_from_lars_schedule(
        schedule_result,
        kernel_name=base_kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        u_dim=u_dim,
        x_dim=None,
        y_dim=None,
        w_dim=None,
        v_dim=None,
        block_size=32,
    )

    code = code + "\n" + emit_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")

    if return_schedule:
        return {
            "name": cand_name,
            "code": code,
            "schedule": schedule_result,
            "config": {
                "name": cand_name,
                "reg_budget": 0,
                "auto_fallback_when_full": False,
            },
            "profile": schedule_result.profile,
        }

    return code

# =============================================================================
# LARS backward scheduler/codegen
# =============================================================================

BWD_OUTPUT_KINDS = {"gw", "gx", "gy"}

@dataclass(frozen=True)
class U1DBwdPath:
    """
    Backward path for forward expression:

        out[v] += w[i] * x[j] * y[k] * c

    Per path, backward contributes:
        grad_w[i] += c * grad_out[v] * x[j] * y[k]
        grad_x[j] += c * grad_out[v] * w[i] * y[k]
        grad_y[k] += c * grad_out[v] * w[i] * x[j]
    """
    pid: int
    i: int  # w index
    j: int  # x index
    k: int  # y index
    v: int  # grad_out/out index
    c: float
    need_grad_w: bool = True

    @property
    def labels(self) -> Tuple[Label, ...]:
        # LARS schedules only input operands.  Backward accumulators
        # (gw/gx/gy) are full-resident emitter variables, so they are never
        # reload-managed candidates.
        return (
            ("w", self.i),
            ("x", self.j),
            ("y", self.k),
            ("go", self.v),
        )


class LARSUniform1DBwdScheduler(LARSUniform1DScheduler):
    """
    LARS scheduler for fused Uniform1D backward.

    It reuses the forward LARS machinery, but a path now has four input labels
    (w, x, y, grad_out) and two or three output accumulators
    (grad_x, grad_y, and optionally grad_w).
    """

    output_kinds = BWD_OUTPUT_KINDS

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        reg_budget: int = 24,
        *,
        need_grad_w: bool = True,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        path_fallback_after: Optional[int] = None,
        prefer_path_fallback_when_full: bool = True,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        # Backward reuses the generic forward scheduler helpers such as
        # _choose_fireable_path().  Those helpers now branch on path_kind for
        # STC, so make the backward kind explicit and keep it on the normal
        # Uniform1D branch.
        self.path_kind = "u1d_bwd"
        self.need_grad_w = bool(need_grad_w)
        self.paths: List[U1DBwdPath] = [
            U1DBwdPath(pid=p, i=i, j=j, k=k, v=v, c=c, need_grad_w=self.need_grad_w)
            for p, (i, j, k, v, c) in enumerate(paths)
        ]

        # reg_budget is kept only for API compatibility.
        # Backward input labels (w/x/y/grad_out) are unbounded and never exceed the all-input-resident virtual register file
        # merely because of a user-supplied virtual budget.
        del reg_budget
        input_label_count = len({lab for path in self.paths for lab in path.labels})
        self.reg_budget = max(4, int(input_label_count))

        # Legacy fallback knobs are ignored in the no-spill scheduler.
        del path_fallback_after, prefer_path_fallback_when_full

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
        self.debug = bool(debug)

        self.label_to_paths: Dict[Label, Set[int]] = defaultdict(set)
        for path in self.paths:
            for lab in path.labels:
                self.label_to_paths[lab].add(path.pid)

        self.path_label_sets: List[Set[Label]] = [set(p.labels) for p in self.paths]
        self.unscheduled: Set[int] = set(p.pid for p in self.paths)

        self.remaining_uses: Counter[Label] = Counter()
        for p in self.paths:
            for lab in p.labels:
                self.remaining_uses[lab] += 1

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = [f"r{r}" for r in range(self.reg_budget)]

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}

        self.profile_enabled = bool(profile)
        self.profile_name = str(profile_name or self.__class__.__name__)
        self.profile_interval = int(profile_interval)
        self.profile_seconds = float(profile_seconds)
        self.profile_print = bool(profile_print)
        self._profile_t0 = perf_counter()
        self._profile_last_print_t = self._profile_t0
        self._profile_last_print_done = 0
        self._profile_loop_iter = 0
        self._profile_select_rounds = 0
        self._profile_fire_rounds = 0
        self._profile_load_rounds = 0
        self._profile_records: List[Dict[str, Any]] = []
        self._profile_last_event = ""
        self._profile_last_reason = ""

    def _path(self, pid: int) -> U1DBwdPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        if kind == "go":
            return f"grad_out[{idx}]"
        if kind == "gw":
            return f"grad_w[{idx}]"
        if kind == "gx":
            return f"grad_x[{idx}]"
        if kind == "gy":
            return f"grad_y[{idx}]"
        return f"{kind}[{idx}]"

    def _is_output_label(self, lab: Label) -> bool:
        return lab[0] in self.output_kinds

    def _emit_load_inst(self, lab: Label, reg: str, reason: str) -> None:
        if self._is_output_label(lab):
            self.instructions.append(
                Inst("load_acc", (reg, self._label_name(lab)), reason)
            )
        else:
            self.instructions.append(
                Inst("load", (reg, self._label_name(lab)), reason)
            )
        self.reloads += 1

    def _store_and_release(self, lab: Label, reason: str = "") -> None:
        if lab not in self.live:
            return

        reg = self.reg_of[lab]
        if self._is_output_label(lab):
            if lab in self.dirty_outputs:
                self.instructions.append(
                    Inst("store_acc", (self._label_name(lab), reg), reason)
                )
                self.dirty_outputs.discard(lab)
            else:
                self.instructions.append(
                    Inst("release", (reg, self._label_name(lab)), reason)
                )
        else:
            self.instructions.append(
                Inst("release", (reg, self._label_name(lab)), reason)
            )

        self.live.remove(lab)
        del self.reg_of[lab]
        self.free_regs.insert(0, reg)
        self._invalidate_score_cache()


    def _priority(self, lab: Label) -> int:
        acc_bonus = 1 if self._is_output_label(lab) else 0
        return self.remaining_uses[lab] + acc_bonus

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)

        lw = ("w", p.i)
        lx = ("x", p.j)
        ly = ("y", p.k)
        lgo = ("go", p.v)

        rw = self.reg_of[lw]
        rx = self.reg_of[lx]
        ry = self.reg_of[ly]
        rgo = self.reg_of[lgo]

        self.instructions.append(
            Inst(
                "bwd_fma_resident",
                (p.i, p.j, p.k, rw, rx, ry, rgo, p.c, self.need_grad_w),
                (
                    f"path#{pid}: "
                    f"gw[{p.i}] += go[{p.v}]*x[{p.j}]*y[{p.k}], "
                    f"gx[{p.j}] += w[{p.i}]*go[{p.v}]*y[{p.k}], "
                    f"gy[{p.k}] += w[{p.i}]*go[{p.v}]*x[{p.j}]"
                ),
            )
        )

        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        for lab in p.labels:
            self.remaining_uses[lab] -= 1

        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")


BwdPairKey = Tuple[Label, Label]  # (w-label, grad_out-label)




def _parse_bwd_lars_label_ref(ref: str) -> Tuple[str, int]:
    if not isinstance(ref, str) or "[" not in ref or not ref.endswith("]"):
        raise ValueError(f"Bad LARS backward label ref: {ref!r}")
    kind, rest = ref.split("[", 1)
    idx = int(rest[:-1])
    aliases = {
        "grad_out": "go",
        "grad_w": "gw",
        "grad_x": "gx",
        "grad_y": "gy",
    }
    kind = aliases.get(kind, kind)
    if kind not in ("w", "x", "y", "go", "gw", "gx", "gy"):
        raise ValueError(f"Bad backward label kind in ref: {ref!r}")
    return kind, idx


def _bwd_label_index_expr(
    kind: str,
    idx: int,
    *,
    mode_scalar_y: bool,
    u_dim: Optional[int],
    iw_dim: Optional[int],
    ix_dim: Optional[int],
    ky_dim: Optional[int],
    v_dim: Optional[int],
) -> str:
    if kind == "w":
        if iw_dim is not None and idx >= iw_dim:
            raise ValueError(f"w index {idx} out of iw_dim={iw_dim}")
        if u_dim is not None:
            return f"w_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"w_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "gw":
        if iw_dim is not None and idx >= iw_dim:
            raise ValueError(f"grad_w index {idx} out of iw_dim={iw_dim}")
        if u_dim is not None:
            return f"gw_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"gw_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "x":
        if ix_dim is not None and idx >= ix_dim:
            raise ValueError(f"x index {idx} out of ix_dim={ix_dim}")
        if u_dim is not None:
            return f"x_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"x_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "gx":
        if ix_dim is not None and idx >= ix_dim:
            raise ValueError(f"grad_x index {idx} out of ix_dim={ix_dim}")
        if u_dim is not None:
            return f"gx_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"gx_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "y":
        if ky_dim is not None and idx >= ky_dim:
            raise ValueError(f"y index {idx} out of ky_dim={ky_dim}")
        if mode_scalar_y:
            return f"y_base + (index_t){idx}"
        if u_dim is not None:
            return f"y_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"y_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "gy":
        if ky_dim is not None and idx >= ky_dim:
            raise ValueError(f"grad_y index {idx} out of ky_dim={ky_dim}")
        if mode_scalar_y:
            return f"gy_base + (index_t){idx}"
        if u_dim is not None:
            return f"gy_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"gy_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "go":
        if v_dim is not None and idx >= v_dim:
            raise ValueError(f"grad_out index {idx} out of v_dim={v_dim}")
        if u_dim is not None:
            return f"go_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"go_base + (index_t){idx} * (index_t)U + (index_t)u"

    raise ValueError(f"Bad backward label kind: {kind}")


def emit_fused_bwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
    acc_reg_budget: Optional[int] = None,
    smem_acc_volatile: bool = False,
) -> str:
    """
    Emit CUDA/HIP-compatible fused backward code from a backward LARS schedule.

    Supported instruction ops:
      - load/load_acc/release/store_acc
      - bwd_fma
    """
    if block_size != 32:
        raise ValueError("this emitter currently assumes block_size=32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    reg_count = _max_lars_reg_count_any(schedule_result)

    # ------------------------------------------------------------------
    # Mixed accumulator placement for backward gradients.
    #   acc_reg_budget=None: old behavior, keep every gw/gx/gy accumulator
    #                        as a scalar local variable/register.
    #   acc_reg_budget=K:    keep only the top-K hottest accumulators in
    #                        registers and demote the rest to per-thread
    #                        shared-memory accumulators.  Demoted gx/gy still
    #                        write back once at the end, preserving atomic
    #                        coalescing compared with path-level atomicAdd.
    # ------------------------------------------------------------------
    gw_use: Counter[int] = Counter()
    gx_use: Counter[int] = Counter()
    gy_use: Counter[int] = Counter()
    first_seen: Dict[Tuple[str, int], int] = {}

    def _note_acc(kind: str, idx: int, inst_id: int) -> None:
        first_seen.setdefault((kind, int(idx)), int(inst_id))

    for inst_id, inst in enumerate(schedule_result.instructions):
        if inst.op == "bwd_fma_resident":
            wi, xj, yk = map(int, inst.args[:3])
            if need_grad_w:
                gw_use[wi] += 1
                _note_acc("gw", wi, inst_id)
            gx_use[xj] += 1
            gy_use[yk] += 1
            _note_acc("gx", xj, inst_id)
            _note_acc("gy", yk, inst_id)

    all_accs: Set[Tuple[str, int]] = set()
    if need_grad_w:
        all_accs |= {("gw", int(i)) for i in gw_use}
    all_accs |= {("gx", int(j)) for j in gx_use}
    all_accs |= {("gy", int(k)) for k in gy_use}

    def _acc_use_count(acc: Tuple[str, int]) -> int:
        kind, idx = acc
        if kind == "gw":
            return int(gw_use.get(idx, 0))
        if kind == "gx":
            return int(gx_use.get(idx, 0))
        if kind == "gy":
            return int(gy_use.get(idx, 0))
        return 0

    def _acc_score(acc: Tuple[str, int]) -> Tuple[float, int, int, int, str, int]:
        # gx/gy have high value because register/shared accumulation reduces
        # global atomicAdd count.  gy is slightly favored in scalar-y mode
        # because writeback also needs a warp reduction.
        kind, idx = acc
        cnt = _acc_use_count(acc)
        if kind == "gy":
            weight = 5.0 if mode_scalar_y else 4.0
            kind_rank = 3
        elif kind == "gx":
            weight = 4.0
            kind_rank = 2
        else:  # gw uses non-atomic writeback, so its register value is lower.
            weight = 2.0
            kind_rank = 1
        return (float(cnt) * weight, cnt, -int(first_seen.get(acc, 10**9)), kind_rank, kind, -idx)

    # acc_reg_budget is intentionally ignored: every backward accumulator is
    # resident as a scalar local variable.  No low-frequency accumulator is
    # demoted to shared memory.
    del acc_reg_budget
    reg_accs: Set[Tuple[str, int]] = set(all_accs)

    smem_accs: Set[Tuple[str, int]] = set()

    resident_gw_indices: Set[int] = {idx for kind, idx in reg_accs if kind == "gw"}
    resident_gx_indices: Set[int] = {idx for kind, idx in reg_accs if kind == "gx"}
    resident_gy_indices: Set[int] = {idx for kind, idx in reg_accs if kind == "gy"}
    resident_gw_indices_sorted = sorted(resident_gw_indices)
    resident_gx_indices_sorted = sorted(resident_gx_indices)
    resident_gy_indices_sorted = sorted(resident_gy_indices)

    # Stable shared-memory layout: all demoted accumulators are laid out as
    # [acc_id][lane].  Each lane owns its own scalar slot for its current u.
    smem_accs_sorted: List[Tuple[str, int]] = sorted(
        smem_accs,
        key=lambda acc: ({"gw": 0, "gx": 1, "gy": 2}[acc[0]], acc[1]),
    )
    smem_acc_id: Dict[Tuple[str, int], int] = {
        acc: n for n, acc in enumerate(smem_accs_sorted)
    }

    def _is_reg_acc(kind: str, idx: int) -> bool:
        return (kind, int(idx)) in reg_accs

    def _is_smem_acc(kind: str, idx: int) -> bool:
        return (kind, int(idx)) in smem_acc_id

    def _reg_acc_name(kind: str, idx: int) -> str:
        if kind == "gw":
            return f"gw_acc_i_{int(idx)}"
        if kind == "gx":
            return f"gx_acc_j_{int(idx)}"
        if kind == "gy":
            return f"gy_acc_k_{int(idx)}"
        raise ValueError(f"Bad accumulator kind: {kind}")

    def _smem_acc_expr(kind: str, idx: int) -> str:
        sid = smem_acc_id[(kind, int(idx))]
        return f"smem_acc[{sid} * 32 + lane]"

    def _emit_bwd_acc_update(kind: str, idx: int, value_expr: str) -> None:
        if _is_reg_acc(kind, idx):
            ap(f"            {_reg_acc_name(kind, idx)} += {value_expr};")
        elif _is_smem_acc(kind, idx):
            ap(f"            {_smem_acc_expr(kind, idx)} += {value_expr};")
        else:
            # This path is reachable only if need_grad_w=False and kind==gw.
            pass

    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")
    ap("template <typename scalar_t>")
    ap("__device__ __forceinline__ scalar_t warp_sum_xor_lars_bwd(scalar_t v) {")
    ap("    for (int offset = 16; offset > 0; offset >>= 1) {")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("        v += __shfl_xor(v, offset);")
    ap("#else")
    ap("        v += __shfl_xor_sync(0xffffffff, v, offset);")
    ap("#endif")
    ap("    }")
    ap("    return v;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    if need_grad_w:
        ap("    scalar_t* __restrict__ grad_w,")
    ap("    scalar_t* __restrict__ grad_x,")
    ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    if smem_accs_sorted:
        ap("    extern __shared__ __align__(16) unsigned char smem_raw[];")
        if smem_acc_volatile:
            ap("    volatile scalar_t* smem_acc = reinterpret_cast<volatile scalar_t*>(smem_raw);")
        else:
            ap("    scalar_t* smem_acc = reinterpret_cast<scalar_t*>(smem_raw);")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
    ap("    const int e_orig = e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e_orig];")
    else:
        ap("    const int src = e_orig;")
    if use_scatter:
        ap("    const int dst = dst_idx[e_orig];")
    else:
        ap("    const int dst = e_orig;")
    ap("")
    ap("    const int x_row = " + ("src;" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src;" if use_y_src else "e_orig;"))
    ap("    const int go_row = " + ("dst;" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base  = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
        if need_grad_w:
            ap(f"    const index_t gw_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base  = (index_t)w_row * (index_t)Iw * (index_t)U;")
        if need_grad_w:
            ap("    const index_t gw_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base  = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
        ap(f"    const index_t gx_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base  = (index_t)x_row * (index_t)Ix * (index_t)U;")
        ap("    const index_t gx_base = (index_t)x_row * (index_t)Ix * (index_t)U;")

    if mode_scalar_y:
        ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky;")
        ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base  = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
            ap(f"    const index_t gy_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky * (index_t)U;")
            ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky * (index_t)U;")

    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t go_base = (index_t)go_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t go_base = (index_t)go_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for sid, (akind, aidx) in enumerate(smem_accs_sorted):
        ap(f"            smem_acc[{sid} * 32 + lane] = scalar_t(0);  // smem {akind}[{aidx}]")
    if smem_accs_sorted:
        ap("")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    for gw_idx in resident_gw_indices_sorted:
        ap(f"            scalar_t gw_acc_i_{gw_idx} = scalar_t(0);")
    for gx_idx in resident_gx_indices_sorted:
        ap(f"            scalar_t gx_acc_j_{gx_idx} = scalar_t(0);")
    for gy_idx in resident_gy_indices_sorted:
        ap(f"            scalar_t gy_acc_k_{gy_idx} = scalar_t(0);")
    if resident_gw_indices_sorted or resident_gx_indices_sorted or resident_gy_indices_sorted:
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op == "load":
            reg, ref = inst.args
            kind, idx = _parse_bwd_lars_label_ref(ref)
            if kind in BWD_OUTPUT_KINDS:
                raise ValueError("load must not be used for backward output labels")
            expr = _bwd_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            arr = "grad_out" if kind == "go" else kind
            ap(f"            {reg} = {arr}[{expr}];")

        elif inst.op == "load_acc":
            reg, ref = inst.args
            kind, _idx = _parse_bwd_lars_label_ref(ref)
            if kind not in BWD_OUTPUT_KINDS:
                raise ValueError("load_acc expects a backward output label")
            ap(f"            {reg} = scalar_t(0);")

        elif inst.op == "bwd_fma_resident":
            wi, xj, yk, rw, rx, ry, rgo, coeff, inst_need_grad_w = inst.args
            c = _fmt_lars_float(float(coeff))
            if bool(inst_need_grad_w):
                _emit_bwd_acc_update("gw", int(wi), f"scalar_t({c}) * {rgo} * {rx} * {ry}")
            _emit_bwd_acc_update("gx", int(xj), f"scalar_t({c}) * ({rw} * {rgo}) * {ry}")
            _emit_bwd_acc_update("gy", int(yk), f"scalar_t({c}) * ({rw} * {rgo}) * {rx}")

        elif inst.op == "bwd_fma":
            # Backward-compatible support for older schedules that kept
            # output accumulators inside the LARS register file.
            rgw, rgx, rgy, rw, rx, ry, rgo, coeff, inst_need_grad_w = inst.args
            c = _fmt_lars_float(float(coeff))
            if bool(inst_need_grad_w):
                ap(f"            {rgw} += scalar_t({c}) * {rgo} * {rx} * {ry};")
            ap(f"            {rgx} += scalar_t({c}) * ({rw} * {rgo}) * {ry};")
            ap(f"            {rgy} += scalar_t({c}) * ({rw} * {rgo}) * {rx};")

        elif inst.op == "store_acc":
            ref, reg = inst.args
            kind, idx = _parse_bwd_lars_label_ref(ref)
            expr = _bwd_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            if kind == "gw":
                if not need_grad_w:
                    raise ValueError("schedule stores grad_w but need_grad_w=False")
                # grad_w is owned by the current w_row in this generated
                # backward path, matching the baseline fused implementation:
                # use a plain store rather than atomicAdd.
                ap(f"            grad_w[{expr}] = {reg};")
            elif kind == "gx":
                ap(f"            atomicAdd(&grad_x[{expr}], {reg});")
            elif kind == "gy":
                if mode_scalar_y:
                    ap(f"            scalar_t gy_sum_{inst_id} = warp_sum_xor_lars_bwd({reg});")
                    ap("            if (lane == 0) {")
                    ap(f"                atomicAdd(&grad_y[{expr}], gy_sum_{inst_id});")
                    ap("            }")
                else:
                    ap(f"            atomicAdd(&grad_y[{expr}], {reg});")
            else:
                raise ValueError("store_acc expects a backward output label")

        elif inst.op == "release":
            pass
        else:
            raise ValueError(f"Unsupported backward LARS instruction op: {inst.op}")

    if (resident_gw_indices_sorted or resident_gx_indices_sorted or resident_gy_indices_sorted
            or smem_accs_sorted):
        ap("")
        ap("            // mixed register/shared-memory backward accumulator writeback")

    # Register-resident writeback.
    for gw_idx in resident_gw_indices_sorted:
        expr = _bwd_label_index_expr(
            "gw", int(gw_idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        ap(f"            grad_w[{expr}] = gw_acc_i_{gw_idx};")
    for gx_idx in resident_gx_indices_sorted:
        expr = _bwd_label_index_expr(
            "gx", int(gx_idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        ap(f"            atomicAdd(&grad_x[{expr}], gx_acc_j_{gx_idx});")
    for gy_idx in resident_gy_indices_sorted:
        expr = _bwd_label_index_expr(
            "gy", int(gy_idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        if mode_scalar_y:
            ap(f"            scalar_t gy_sum_resident_{gy_idx} = warp_sum_xor_lars_bwd(gy_acc_k_{gy_idx});")
            ap("            if (lane == 0) {")
            ap(f"                atomicAdd(&grad_y[{expr}], gy_sum_resident_{gy_idx});")
            ap("            }")
        else:
            ap(f"            atomicAdd(&grad_y[{expr}], gy_acc_k_{gy_idx});")

    # Shared-memory-resident writeback.
    for kind, idx in smem_accs_sorted:
        expr = _bwd_label_index_expr(
            kind, int(idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        smem_expr = _smem_acc_expr(kind, int(idx))
        if kind == "gw":
            if not need_grad_w:
                continue
            ap(f"            grad_w[{expr}] = {smem_expr};")
        elif kind == "gx":
            ap(f"            atomicAdd(&grad_x[{expr}], {smem_expr});")
        elif kind == "gy":
            if mode_scalar_y:
                ap(f"            scalar_t gy_sum_smem_{idx} = warp_sum_xor_lars_bwd({smem_expr});")
                ap("            if (lane == 0) {")
                ap(f"                atomicAdd(&grad_y[{expr}], gy_sum_smem_{idx});")
                ap("            }")
            else:
                ap(f"            atomicAdd(&grad_y[{expr}], {smem_expr});")
        else:
            raise ValueError(f"Bad smem accumulator kind: {kind}")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32_bwd(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32_bwd(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32_bwd(a, b)) return false;")
    ap("    return mul_fits_int32_bwd(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_bwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32_bwd((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32_bwd(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32_bwd(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32_bwd(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool go_ok = mul3_fits_int32_bwd(go_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && go_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    constexpr int kSmemAccCount = {len(smem_accs_sorted)};")
    ap("    size_t smem_bytes = (size_t)kSmemAccCount * 32u * sizeof(scalar_t);")
    ap("#if !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)")
    ap("    if (smem_bytes > 49152) {")
    ap(f"        cudaFuncSetAttribute({kernel_name}<scalar_t, index_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);")
    ap("    }")
    ap("#endif")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, smem_bytes, stream>>>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_bwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def emit_lars_bwd_launcher(
    bundle_name: str,
    mode: str,
    *,
    need_grad_w: bool,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
    if mode == "u,u,,u":
        y_comment = "[B,Ky,1] or [S,Ky,1]"
        gy_alloc = "    auto grad_y = torch::zeros({y.size(0), Ky, 1}, y.options());"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");'
    elif mode == "u,u,u,u":
        y_comment = "[B,Ky,U] or [S,Ky,U]"
        gy_alloc = "    auto grad_y = torch::zeros_like(y);"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == U, "y.size(2) must equal U for mode u,u,u,u");'
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    param_lines = [
        "    torch::Tensor w",
        "    torch::Tensor x",
        "    torch::Tensor y",
        "    torch::Tensor grad_out",
    ]
    if use_x_src or use_y_src:
        param_lines.append("    torch::Tensor src_idx")
    if use_scatter:
        param_lines.append("    torch::Tensor dst_idx")
    param_lines.append("    int64_t V64")
    params = ",\n".join(param_lines)

    src_check = ""
    if use_x_src or use_y_src:
        src_check = r'''
    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''
    dst_check = ""
    if use_scatter:
        dst_check = r'''
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''

    if use_x_src or use_y_src:
        b_expr = "src_idx.size(0)"
        src_numel_check = '    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");'
    elif use_scatter:
        b_expr = "dst_idx.size(0)"
        src_numel_check = ""
    else:
        b_expr = "grad_out.size(0)"
        src_numel_check = ""

    dst_numel_check = '    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");' if use_scatter else ""

    blist_logic = ""

    launch_src_arg = "(const int32_t*)src_idx.data_ptr<int32_t>()," if (use_x_src or use_y_src) else "nullptr,"
    launch_dst_arg = "(const int32_t*)dst_idx.data_ptr<int32_t>()," if use_scatter else "nullptr,"

    grad_w_alloc = "    auto grad_w = torch::zeros_like(w);\n" if need_grad_w else ""
    grad_w_launch_arg = "                (scalar_t*)grad_w.data_ptr<scalar_t>(),\n" if need_grad_w else ""
    ret_expr = "return {grad_w, grad_x, grad_y};" if need_grad_w else "return {grad_x, grad_y};"

    return rf'''

std::vector<torch::Tensor> launcher_{bundle_name}(
{params})
{{
    // Expected tensors:
    //   w        : [WB,Iw,U], WB can be 1 or B
    //   x        : [S,Ix,U] or [B,Ix,U]
    //   y        : {y_comment}
    //   grad_out : [B,V,U] or [S,V,U], matching forward output layout

    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x/y/grad_out must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");
{src_check}{dst_check}

    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x.dim() == 3, "x must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");

    int B  = (int){b_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x.size(0);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
    {y_check_u}
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
{src_numel_check}
{dst_numel_check}

{blist_logic}
{grad_w_alloc}    auto grad_x = torch::zeros_like(x);
{gy_alloc}

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{

        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");

        launch_{bundle_name}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
{grad_w_launch_arg}                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    }});

    GPU_KERNEL_LAUNCH_CHECK();

    {ret_expr}
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} backward LARS fused jit impl");
}}
'''


def _make_lars_bwd_paths_from_uniform1d_lists(
    i_list: List[int],
    j_list: List[int],
    k_list: List[int],
    v_list: List[int],
    coeff_list: List[float],
    *,
    path_semantics: str,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Convert external Uniform1D path indices to backward-LARS tuples.

    Backward LARS native path tuple is:
        (w_index, x_index, y_index, out_index, coeff)

    path_semantics="wxy":
        external (i,j,k) means w[i], x[j], y[k].
    path_semantics="xyw":
        external (i,j,k) means x[i], y[j], w[k].
    """
    if path_semantics not in ("wxy", "xyw"):
        raise ValueError("path_semantics must be 'wxy' or 'xyw'")
    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

    paths: List[Tuple[int, int, int, int, float]] = []
    for i, j, k, v, c in zip(i_list, j_list, k_list, v_list, coeff_list):
        i = int(i); j = int(j); k = int(k); v = int(v); c = float(c)
        if path_semantics == "wxy":
            paths.append((i, j, k, v, c))
        else:
            paths.append((k, i, j, v, c))
    return paths








# =============================================================================
# Split backward LARS codegen: grad_w / grad_x / grad_y in separate kernels
# =============================================================================

BWD_SPLIT_KINDS = {"gw", "gx", "gy"}


@dataclass(frozen=True)
class U1DBwdSplitPath:
    """
    Backward path used by split-gradient LARS scheduling.

    grad_kind="gw": grad_w[i] += c * grad_out[v] * x[j] * y[k]
    grad_kind="gx": grad_x[j] += c * w[i]        * grad_out[v] * y[k]
    grad_kind="gy": grad_y[k] += c * w[i]        * grad_out[v] * x[j]
    """
    pid: int
    i: int
    j: int
    k: int
    v: int
    c: float
    grad_kind: str

    @property
    def target_index(self) -> int:
        if self.grad_kind == "gw":
            return int(self.i)
        if self.grad_kind == "gx":
            return int(self.j)
        if self.grad_kind == "gy":
            return int(self.k)
        raise ValueError(f"Bad grad_kind: {self.grad_kind}")

    @property
    def labels(self) -> Tuple[Label, ...]:
        if self.grad_kind == "gw":
            return (("x", self.j), ("y", self.k), ("go", self.v))
        if self.grad_kind == "gx":
            return (("w", self.i), ("y", self.k), ("go", self.v))
        if self.grad_kind == "gy":
            return (("w", self.i), ("x", self.j), ("go", self.v))
        raise ValueError(f"Bad grad_kind: {self.grad_kind}")


class LARSUniform1DBwdSplitScheduler(LARSUniform1DScheduler):
    """
    Per-gradient backward scheduler.  It keeps the no-spill LARS label
    selection policy, but builds labels for a single gradient target only.
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        reg_budget: int = 24,
        *,
        grad_kind: str,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        path_fallback_after: Optional[int] = None,
        prefer_path_fallback_when_full: bool = True,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        if grad_kind not in BWD_SPLIT_KINDS:
            raise ValueError(f"grad_kind must be one of {sorted(BWD_SPLIT_KINDS)}, got {grad_kind!r}")
        self.grad_kind = str(grad_kind)
        self.paths: List[U1DBwdSplitPath] = [
            U1DBwdSplitPath(pid=p, i=i, j=j, k=k, v=v, c=c, grad_kind=self.grad_kind)
            for p, (i, j, k, v, c) in enumerate(paths)
        ]

        # reg_budget is kept only for API compatibility.
        # Split-backward input labels are unbounded as well.
        del reg_budget
        input_label_count = len({lab for path in self.paths for lab in path.labels})
        self.reg_budget = max(3, int(input_label_count))

        # Legacy fallback knobs are ignored in the no-spill scheduler.
        del path_fallback_after, prefer_path_fallback_when_full

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
        self.debug = bool(debug)

        self.label_to_paths: Dict[Label, Set[int]] = defaultdict(set)
        for path in self.paths:
            for lab in path.labels:
                self.label_to_paths[lab].add(path.pid)

        self.path_label_sets: List[Set[Label]] = [set(p.labels) for p in self.paths]
        self.unscheduled: Set[int] = set(p.pid for p in self.paths)

        self.remaining_uses: Counter[Label] = Counter()
        for p in self.paths:
            for lab in p.labels:
                self.remaining_uses[lab] += 1

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = [f"r{r}" for r in range(self.reg_budget)]

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}

        self.profile_enabled = bool(profile)
        self.profile_name = str(profile_name or f"{self.__class__.__name__}_{self.grad_kind}")
        self.profile_interval = int(profile_interval)
        self.profile_seconds = float(profile_seconds)
        self.profile_print = bool(profile_print)
        self._profile_t0 = perf_counter()
        self._profile_last_print_t = self._profile_t0
        self._profile_last_print_done = 0
        self._profile_loop_iter = 0
        self._profile_select_rounds = 0
        self._profile_fire_rounds = 0
        self._profile_load_rounds = 0
        self._profile_records: List[Dict[str, Any]] = []
        self._profile_last_event = ""
        self._profile_last_reason = ""

    def _path(self, pid: int) -> U1DBwdSplitPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        if kind == "go":
            return f"grad_out[{idx}]"
        return f"{kind}[{idx}]"

    def _priority(self, lab: Label) -> int:
        return self.remaining_uses[lab]

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)
        regs: Dict[str, str] = {}
        for kind, idx in p.labels:
            regs[kind] = self.reg_of[(kind, idx)]

        self.instructions.append(
            Inst(
                "bwd_split_fma_resident",
                (
                    self.grad_kind,
                    p.target_index,
                    regs.get("w", ""),
                    regs.get("x", ""),
                    regs.get("y", ""),
                    regs.get("go", ""),
                    p.c,
                ),
                (
                    f"path#{pid}: split {self.grad_kind}; "
                    f"w[{p.i}], x[{p.j}], y[{p.k}], go[{p.v}], c={p.c}"
                ),
            )
        )

        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        for lab in p.labels:
            self.remaining_uses[lab] -= 1

        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")


def _split_grad_acc_name(grad_kind: str, idx: int) -> str:
    if grad_kind == "gw":
        return f"gw_acc_i_{int(idx)}"
    if grad_kind == "gx":
        return f"gx_acc_j_{int(idx)}"
    if grad_kind == "gy":
        return f"gy_acc_k_{int(idx)}"
    raise ValueError(f"Bad grad_kind: {grad_kind}")


def emit_lars_bwd_split_preamble() -> str:
    return r'''
#include <stdint.h>
#include <torch/extension.h>
#include <vector>
#include <cstdint>

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #include <ATen/hip/HIPContext.h>
  #include <c10/hip/HIPGuard.h>
  using gpuStream_t = hipStream_t;
  #define getCurrentGPUStream at::hip::getCurrentHIPStream
  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()
#else
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <ATen/cuda/CUDAContext.h>
  using gpuStream_t = cudaStream_t;
  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream
  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

using GPU_Guard = c10::DeviceGuard;

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_sum_xor_lars_bwd_split(scalar_t v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
        v += __shfl_xor(v, offset);
#else
        v += __shfl_xor_sync(0xffffffff, v, offset);
#endif
    }
    return v;
}

static inline bool mul_fits_int32_bwd_split(int64_t a, int64_t b) {
    if (a < 0 || b < 0) return false;
    constexpr int64_t LIM = 2147483647LL;
    if (a == 0 || b == 0) return true;
    return a <= LIM / b;
}

static inline bool mul3_fits_int32_bwd_split(int64_t a, int64_t b, int64_t c) {
    if (!mul_fits_int32_bwd_split(a, b)) return false;
    return mul_fits_int32_bwd_split(a * b, c);
}

static inline bool should_use_int32_index_bwd_split(
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)
{
    bool w_ok = mul3_fits_int32_bwd_split((int64_t)WB, (int64_t)Iw, (int64_t)U);
    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;
    bool x_ok = mul3_fits_int32_bwd_split(x_dim0, (int64_t)Ix, (int64_t)U);
    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;
    bool y_ok = mode_scalar_y ? mul_fits_int32_bwd_split(y_dim0, (int64_t)Ky)
                              : mul3_fits_int32_bwd_split(y_dim0, (int64_t)Ky, (int64_t)U);
    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;
    bool go_ok = mul3_fits_int32_bwd_split(go_dim0, (int64_t)V, (int64_t)U);
    return w_ok && x_ok && y_ok && go_ok;
}
'''


def emit_lars_bwd_split_kernel_from_schedule(
    schedule_result: ScheduleResult,
    *,
    grad_kind: str,
    kernel_name: str,
    mode: str = "u,u,,u",
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
    acc_reg_budget: Optional[int] = None,
    smem_acc_volatile: bool = False,
) -> str:
    if grad_kind not in BWD_SPLIT_KINDS:
        raise ValueError(f"grad_kind must be one of {sorted(BWD_SPLIT_KINDS)}, got {grad_kind!r}")
    if block_size != 32:
        raise ValueError("this emitter currently assumes block_size=32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    reg_count = _max_lars_reg_count_any(schedule_result)

    target_use: Counter[int] = Counter()
    first_seen: Dict[int, int] = {}
    for inst_id, inst in enumerate(schedule_result.instructions):
        if inst.op == "bwd_split_fma_resident":
            ikind, target_idx = inst.args[0], int(inst.args[1])
            if ikind != grad_kind:
                raise ValueError(f"schedule contains {ikind}, but emitter grad_kind is {grad_kind}")
            target_use[target_idx] += 1
            first_seen.setdefault(target_idx, inst_id)

    all_targets: Set[int] = set(target_use)
    # acc_reg_budget is intentionally ignored: every split-backward accumulator
    # target is register-resident and no target is demoted to shared memory.
    del acc_reg_budget
    reg_targets: Set[int] = set(all_targets)

    smem_targets_sorted: List[int] = []
    smem_id: Dict[int, int] = {idx: n for n, idx in enumerate(smem_targets_sorted)}
    reg_targets_sorted = sorted(reg_targets)

    def _is_reg_target(idx: int) -> bool:
        return int(idx) in reg_targets

    def _target_acc_name(idx: int) -> str:
        return _split_grad_acc_name(grad_kind, int(idx))

    def _smem_expr(idx: int) -> str:
        return f"smem_acc[{smem_id[int(idx)]} * 32 + lane]"

    def _emit_acc_update(idx: int, value_expr: str) -> None:
        if _is_reg_target(idx):
            ap(f"            {_target_acc_name(idx)} += {value_expr};")
        else:
            ap(f"            {_smem_expr(idx)} += {value_expr};")

    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* __restrict__ grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* __restrict__ grad_x,")
    else:
        ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    if smem_targets_sorted:
        ap("    extern __shared__ __align__(16) unsigned char smem_raw[];")
        if smem_acc_volatile:
            ap("    volatile scalar_t* smem_acc = reinterpret_cast<volatile scalar_t*>(smem_raw);")
        else:
            ap("    scalar_t* smem_acc = reinterpret_cast<scalar_t*>(smem_raw);")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
    ap("    const int e_orig = e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e_orig];")
    else:
        ap("    const int src = e_orig;")
    if use_scatter:
        ap("    const int dst = dst_idx[e_orig];")
    else:
        ap("    const int dst = e_orig;")
    ap("")
    ap("    const int x_row = " + ("src;" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src;" if use_y_src else "e_orig;"))
    ap("    const int go_row = " + ("dst;" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base  = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
        if grad_kind == "gw":
            ap(f"    const index_t gw_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base  = (index_t)w_row * (index_t)Iw * (index_t)U;")
        if grad_kind == "gw":
            ap("    const index_t gw_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base  = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
        if grad_kind == "gx":
            ap(f"    const index_t gx_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base  = (index_t)x_row * (index_t)Ix * (index_t)U;")
        if grad_kind == "gx":
            ap("    const index_t gx_base = (index_t)x_row * (index_t)Ix * (index_t)U;")

    if mode_scalar_y:
        ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky;")
        if grad_kind == "gy":
            ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base  = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
            if grad_kind == "gy":
                ap(f"    const index_t gy_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky * (index_t)U;")
            if grad_kind == "gy":
                ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky * (index_t)U;")

    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t go_base = (index_t)go_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t go_base = (index_t)go_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for sid, idx in enumerate(smem_targets_sorted):
        ap(f"            smem_acc[{sid} * 32 + lane] = scalar_t(0);  // smem {grad_kind}[{idx}]")
    if smem_targets_sorted:
        ap("")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    for idx in reg_targets_sorted:
        ap(f"            scalar_t {_target_acc_name(idx)} = scalar_t(0);")
    if reg_targets_sorted:
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op == "load":
            reg, ref = inst.args
            kind, idx = _parse_bwd_lars_label_ref(ref)
            if kind in BWD_OUTPUT_KINDS:
                raise ValueError("load must not be used for backward output labels")
            expr = _bwd_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            arr = "grad_out" if kind == "go" else kind
            ap(f"            {reg} = {arr}[{expr}];")

        elif inst.op == "bwd_split_fma_resident":
            ikind, target_idx, rw, rx, ry, rgo, coeff = inst.args
            if ikind != grad_kind:
                raise ValueError(f"schedule contains {ikind}, but emitter grad_kind is {grad_kind}")
            c = _fmt_lars_float(float(coeff))
            target_idx = int(target_idx)
            if grad_kind == "gw":
                _emit_acc_update(target_idx, f"scalar_t({c}) * {rgo} * {rx} * {ry}")
            elif grad_kind == "gx":
                _emit_acc_update(target_idx, f"scalar_t({c}) * ({rw} * {rgo}) * {ry}")
            elif grad_kind == "gy":
                _emit_acc_update(target_idx, f"scalar_t({c}) * ({rw} * {rgo}) * {rx}")
            else:
                raise ValueError(f"Bad grad_kind: {grad_kind}")

        elif inst.op == "release":
            pass
        else:
            raise ValueError(f"Unsupported split backward LARS instruction op: {inst.op}")

    if reg_targets_sorted or smem_targets_sorted:
        ap("")
        ap("            // split backward accumulator writeback")

    def _emit_writeback(idx: int, acc_expr: str) -> None:
        if grad_kind == "gw":
            expr = _bwd_label_index_expr(
                "gw", int(idx),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            ap(f"            grad_w[{expr}] = {acc_expr};")
        elif grad_kind == "gx":
            expr = _bwd_label_index_expr(
                "gx", int(idx),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            ap(f"            atomicAdd(&grad_x[{expr}], {acc_expr});")
        elif grad_kind == "gy":
            expr = _bwd_label_index_expr(
                "gy", int(idx),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            if mode_scalar_y:
                safe_idx = str(int(idx)).replace("-", "m")
                ap(f"            scalar_t gy_sum_{safe_idx} = warp_sum_xor_lars_bwd_split({acc_expr});")
                ap("            if (lane == 0) {")
                ap(f"                atomicAdd(&grad_y[{expr}], gy_sum_{safe_idx});")
                ap("            }")
            else:
                ap(f"            atomicAdd(&grad_y[{expr}], {acc_expr});")
        else:
            raise ValueError(f"Bad grad_kind: {grad_kind}")

    for idx in reg_targets_sorted:
        _emit_writeback(int(idx), _target_acc_name(int(idx)))
    for idx in smem_targets_sorted:
        _emit_writeback(int(idx), _smem_expr(int(idx)))

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* grad_x,")
    else:
        ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    constexpr int kSmemAccCount = {len(smem_targets_sorted)};")
    ap("    size_t smem_bytes = (size_t)kSmemAccCount * 32u * sizeof(scalar_t);")
    ap("#if !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)")
    ap("    if (smem_bytes > 49152) {")
    ap(f"        cudaFuncSetAttribute({kernel_name}<scalar_t, index_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);")
    ap("    }")
    ap("#endif")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, smem_bytes, stream>>>(")
    if grad_kind == "gw":
        ap("        w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("        w, x, y, grad_out, grad_x,")
    else:
        ap("        w, x, y, grad_out, grad_y,")
    ap("        src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* grad_x,")
    else:
        ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_bwd_split(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                       kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    if grad_kind == "gw":
        ap("            w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("            w, x, y, grad_out, grad_x,")
    else:
        ap("            w, x, y, grad_out, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    if grad_kind == "gw":
        ap("            w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("            w, x, y, grad_out, grad_x,")
    else:
        ap("            w, x, y, grad_out, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* grad_x,")
    else:
        ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    if grad_kind == "gw":
        ap("        w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("        w, x, y, grad_out, grad_x,")
    else:
        ap("        w, x, y, grad_out, grad_y,")
    ap("        src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def emit_lars_bwd_split_launcher(
    bundle_name: str,
    mode: str,
    *,
    need_grad_w: bool,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
    gradw_kernel: Optional[str],
    gradx_kernel: str,
    grady_kernel: str,
) -> str:
    if mode == "u,u,,u":
        y_comment = "[B,Ky,1] or [S,Ky,1]"
        gy_alloc = "    auto grad_y = torch::zeros({y.size(0), Ky, 1}, y.options());"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");'
    elif mode == "u,u,u,u":
        y_comment = "[B,Ky,U] or [S,Ky,U]"
        gy_alloc = "    auto grad_y = torch::zeros_like(y);"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == U, "y.size(2) must equal U for mode u,u,u,u");'
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    param_lines = [
        "    torch::Tensor w",
        "    torch::Tensor x",
        "    torch::Tensor y",
        "    torch::Tensor grad_out",
    ]
    if use_x_src or use_y_src:
        param_lines.append("    torch::Tensor src_idx")
    if use_scatter:
        param_lines.append("    torch::Tensor dst_idx")
    param_lines.append("    int64_t V64")
    params = ",\n".join(param_lines)

    src_check = ""
    if use_x_src or use_y_src:
        src_check = r'''
    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''
    dst_check = ""
    if use_scatter:
        dst_check = r'''
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''

    if use_x_src or use_y_src:
        b_expr = "src_idx.size(0)"
        src_numel_check = '    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");'
    elif use_scatter:
        b_expr = "dst_idx.size(0)"
        src_numel_check = ""
    else:
        b_expr = "grad_out.size(0)"
        src_numel_check = ""

    dst_numel_check = '    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");' if use_scatter else ""

    blist_logic = ""

    launch_src_arg = "(const int32_t*)src_idx.data_ptr<int32_t>()," if (use_x_src or use_y_src) else "nullptr,"
    launch_dst_arg = "(const int32_t*)dst_idx.data_ptr<int32_t>()," if use_scatter else "nullptr,"

    grad_w_alloc = "    auto grad_w = torch::zeros_like(w);\n" if need_grad_w else ""
    grad_w_launch = ""
    if need_grad_w:
        assert gradw_kernel is not None
        grad_w_launch = f'''
        launch_{gradw_kernel}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                B, WB, Iw, Ix, Ky, V, U, S, stream);
'''
    ret_expr = "return {grad_w, grad_x, grad_y};" if need_grad_w else "return {grad_x, grad_y};"

    return rf'''

std::vector<torch::Tensor> launcher_{bundle_name}(
{params})
{{
    // Split backward: grad_w / grad_x / grad_y are computed by independent
    // LARS-scheduled kernels to reduce register pressure for large path lists.
    // Expected tensors:
    //   w        : [WB,Iw,U], WB can be 1 or B
    //   x        : [S,Ix,U] or [B,Ix,U]
    //   y        : {y_comment}
    //   grad_out : [B,V,U] or [S,V,U], matching forward output layout

    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x/y/grad_out must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");
{src_check}{dst_check}

    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x.dim() == 3, "x must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");

    int B  = (int){b_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x.size(0);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
    {y_check_u}
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
{src_numel_check}
{dst_numel_check}

{blist_logic}
{grad_w_alloc}    auto grad_x = torch::zeros_like(x);
{gy_alloc}

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{

        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");
{grad_w_launch}
        launch_{gradx_kernel}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                B, WB, Iw, Ix, Ky, V, U, S, stream);

        launch_{grady_kernel}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    }});

    GPU_KERNEL_LAUNCH_CHECK();

    {ret_expr}
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} backward LARS split jit impl");
}}
'''



@dataclass(frozen=True)
class BwdCodegenContext:
    """Pre-parsed Uniform1D backward codegen inputs shared by fused/split paths."""
    input_indices: Dict[int, Any]
    output_indices: Dict[int, Any]
    use_x_src: bool
    use_y_src: bool
    use_scatter: bool
    P: int
    i_cpu: List[int]
    j_cpu: List[int]
    k_cpu: List[int]
    v_cpu: List[int]
    c_cpu: List[float]
    bwd_paths: List[Tuple[int, int, int, int, float]]
    mode_str: str
    layout_tag: str


def _prepare_uniform1d_bwd_codegen_context(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]],
    output_indices: Optional[Dict[int, Any]],
    *,
    mode: str,
    path_semantics: str,
) -> BwdCodegenContext:
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = int(i_list.numel())
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    bwd_paths = _make_lars_bwd_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )

    return BwdCodegenContext(
        input_indices=input_indices,
        output_indices=output_indices,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        P=P,
        i_cpu=i_cpu,
        j_cpu=j_cpu,
        k_cpu=k_cpu,
        v_cpu=v_cpu,
        c_cpu=c_cpu,
        bwd_paths=bwd_paths,
        mode_str="uu_u" if mode == "u,u,,u" else "uuuu",
        layout_tag=f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}",
    )


def _resolve_split_backward(
    split_backward: Union[bool, str],
    *,
    path_count: int,
    split_path_threshold: int,
) -> bool:
    if isinstance(split_backward, str):
        split_mode = split_backward.lower()
        if split_mode in ("auto",):
            return int(path_count) > int(split_path_threshold)
        if split_mode in ("split", "true", "always", "on", "yes"):
            return True
        if split_mode in ("fused", "false", "never", "off", "no"):
            return False
        raise ValueError(
            "split_backward must be bool or one of "
            "'auto', 'split'/'always'/'on', 'fused'/'never'/'off'"
        )
    return bool(split_backward)


def _write_codegen_candidate(
    *,
    out_path: str,
    candidate_count: int,
    cand_name: str,
    code: str,
) -> None:
    if not out_path:
        return
    out_p = Path(out_path)
    if candidate_count > 1:
        stem = out_p.stem
        suffix = out_p.suffix or ".cu"
        out_p.with_name(f"{stem}_{cand_name}{suffix}").write_text(code, encoding="utf-8")
    else:
        out_p.write_text(code, encoding="utf-8")


def _finalize_codegen_candidates(candidates: List[Any], *, return_schedule: bool):
    if return_schedule:
        return candidates if len(candidates) != 1 else candidates[0]
    if len(candidates) == 1:
        return candidates[0][1]
    return candidates


def _bwd_base_kernel_name(
    *,
    kernel_name: str,
    cand_name: str,
    u_dim: int,
    path_count: int,
    mode_str: str,
    layout_tag: str,
    grad_tag: str,
) -> str:
    safe_cand = cand_name.replace("-", "_").replace(".", "_")
    if kernel_name and kernel_name != "uniform1d_bwd_lars":
        return f"{kernel_name}_{safe_cand}"
    return (
        f"uniform1d_{safe_cand}_u{u_dim}_path{path_count}_"
        f"{mode_str}_{layout_tag}_{grad_tag}_bwd"
    )


def _generate_code_uniform1d_bwd_split_from_context(
    ctx: BwdCodegenContext,
    *,
    u_dim: int,
    iw_dim: Optional[int],
    ix_dim: Optional[int],
    ky_dim: Optional[int],
    v_dim: Optional[int],
    mode: str,
    need_grad_w: bool,
    out_path: str,
    kernel_name: str,
    reg_budget: Union[int, Iterable[int]],
    acc_reg_budget: Optional[Union[int, Iterable[Optional[int]]]],
    smem_acc_volatile: bool,
    return_schedule: bool,
    fold_reg_budgets_by_max_live: bool,
    profile: bool,
    profile_interval: int,
    profile_seconds: float,
    profile_print: bool,
    enable_secondary_affinity: bool,
    topk_candidates: Optional[int],
):
    """Internal split-backward implementation.  The public entry is the unified wrapper."""
    # Disable reg_budget candidate search/folding and acc_reg_budget demotion.
    # One split candidate is emitted, with all input labels and all accumulators
    # resident.  The legacy arguments are retained for caller compatibility.
    del reg_budget, acc_reg_budget, fold_reg_budgets_by_max_live
    configs: List[Dict[str, Any]] = [{
        "name": "lars_bwd_split_all_inputs_accall",
        "reg_budget": 0,
        "acc_reg_budget": None,
        "smem_acc_volatile": False,
    }]
    max_effective_need = 0

    grad_tag = "split" if need_grad_w else "split_nogradw"
    candidates = []
    split_kinds = (["gw"] if need_grad_w else []) + ["gx", "gy"]

    for cfg_in in configs:
        cfg = dict(cfg_in)
        rb = int(cfg["reg_budget"])
        acc_rb = cfg.get("acc_reg_budget", None)
        acc_rb = None if acc_rb is None else int(acc_rb)
        cfg_smem_acc_volatile = bool(cfg.get("smem_acc_volatile", smem_acc_volatile))
        cand_name = str(cfg["name"])
        cfg["auto_fallback_when_full"] = False
        cfg["effective_live_need"] = int(max_effective_need)
        cfg["split_backward"] = True

        base_kernel_name = _bwd_base_kernel_name(
            kernel_name=kernel_name,
            cand_name=cand_name,
            u_dim=u_dim,
            path_count=ctx.P,
            mode_str=ctx.mode_str,
            layout_tag=ctx.layout_tag,
            grad_tag=grad_tag,
        )

        schedules: Dict[str, ScheduleResult] = {}
        profiles: Dict[str, Any] = {}
        code_parts: List[str] = [emit_lars_bwd_split_preamble()]
        kernel_names: Dict[str, str] = {}

        for gkind in split_kinds:
            scheduler = LARSUniform1DBwdSplitScheduler(
                paths=ctx.bwd_paths,
                reg_budget=rb,
                grad_kind=gkind,
                enable_secondary_affinity=bool(cfg.get("enable_secondary_affinity", enable_secondary_affinity)),
                topk_candidates=cfg.get("topk_candidates", topk_candidates),
                profile=bool(cfg.get("profile", profile)),
                profile_name=str(cfg.get("profile_name", f"{cand_name}_{gkind}")),
                profile_interval=int(cfg.get("profile_interval", profile_interval)),
                profile_seconds=float(cfg.get("profile_seconds", profile_seconds)),
                profile_print=bool(cfg.get("profile_print", profile_print)),
            )
            schedule_result = scheduler.schedule()
            schedules[gkind] = schedule_result
            profiles[gkind] = schedule_result.profile
            split_kernel_name = f"{base_kernel_name}_{gkind}"
            kernel_names[gkind] = split_kernel_name
            code_parts.append(
                emit_lars_bwd_split_kernel_from_schedule(
                    schedule_result,
                    grad_kind=gkind,
                    kernel_name=split_kernel_name,
                    mode=mode,
                    use_x_src=ctx.use_x_src,
                    use_y_src=ctx.use_y_src,
                    use_scatter=ctx.use_scatter,
                    u_dim=u_dim,
                    iw_dim=iw_dim,
                    ix_dim=ix_dim,
                    ky_dim=ky_dim,
                    v_dim=v_dim,
                    block_size=32,
                    acc_reg_budget=acc_rb,
                    smem_acc_volatile=cfg_smem_acc_volatile,
                )
            )

        code_parts.append(
            emit_lars_bwd_split_launcher(
                bundle_name=base_kernel_name,
                mode=mode,
                need_grad_w=need_grad_w,
                use_x_src=ctx.use_x_src,
                use_y_src=ctx.use_y_src,
                use_scatter=ctx.use_scatter,
                gradw_kernel=kernel_names.get("gw"),
                gradx_kernel=kernel_names["gx"],
                grady_kernel=kernel_names["gy"],
            )
        )
        code = "\n".join(code_parts)

        _write_codegen_candidate(
            out_path=out_path,
            candidate_count=len(configs),
            cand_name=cand_name,
            code=code,
        )

        if return_schedule:
            candidates.append({
                "name": cand_name,
                "code": code,
                "schedule": schedules,
                "config": cfg,
                "profiles": profiles,
                "budget_fold_effective_live_need": int(max_effective_need),
                "auto_fallback_when_full": False,
            })
        else:
            candidates.append((cand_name, code))

    return _finalize_codegen_candidates(candidates, return_schedule=return_schedule)



def _generate_code_uniform1d_bwd_fused_from_context(
    ctx: BwdCodegenContext,
    *,
    u_dim: int,
    iw_dim: Optional[int],
    ix_dim: Optional[int],
    ky_dim: Optional[int],
    v_dim: Optional[int],
    mode: str,
    need_grad_w: bool,
    out_path: str,
    kernel_name: str,
    reg_budget: Union[int, Iterable[int]],
    acc_reg_budget: Optional[Union[int, Iterable[Optional[int]]]],
    smem_acc_volatile: bool,
    return_schedule: bool,
    fold_reg_budgets_by_max_live: bool,
    profile: bool,
    profile_interval: int,
    profile_seconds: float,
    profile_print: bool,
    enable_secondary_affinity: bool,
    topk_candidates: Optional[int],
):
    """
    Generate one fused backward LARS candidate.

    The slim policy ignores reg_budget search and acc_reg_budget demotion:
    input labels are virtually unbounded and all gw/gx/gy accumulators stay
    register-resident in the emitter.
    """
    del reg_budget, acc_reg_budget, smem_acc_volatile, fold_reg_budgets_by_max_live

    grad_tag = "full" if need_grad_w else "nogradw"
    cand_name = "lars_bwd_all_inputs_accall"

    scheduler = LARSUniform1DBwdScheduler(
        paths=ctx.bwd_paths,
        reg_budget=0,
        need_grad_w=need_grad_w,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name=cand_name,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    schedule_result = scheduler.schedule()

    base_kernel_name = _bwd_base_kernel_name(
        kernel_name=kernel_name,
        cand_name=cand_name,
        u_dim=u_dim,
        path_count=ctx.P,
        mode_str=ctx.mode_str,
        layout_tag=ctx.layout_tag,
        grad_tag=grad_tag,
    )

    code = emit_fused_bwd_kernel_from_lars_schedule(
        schedule_result,
        kernel_name=base_kernel_name,
        mode=mode,
        need_grad_w=need_grad_w,
        use_x_src=ctx.use_x_src,
        use_y_src=ctx.use_y_src,
        use_scatter=ctx.use_scatter,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        block_size=32,
        acc_reg_budget=None,
        smem_acc_volatile=False,
    )

    code = code + "\n" + emit_lars_bwd_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        need_grad_w=need_grad_w,
        use_x_src=ctx.use_x_src,
        use_y_src=ctx.use_y_src,
        use_scatter=ctx.use_scatter,
    )

    _write_codegen_candidate(
        out_path=out_path,
        candidate_count=1,
        cand_name=cand_name,
        code=code,
    )

    if return_schedule:
        return {
            "name": cand_name,
            "code": code,
            "schedule": schedule_result,
            "config": {
                "name": cand_name,
                "reg_budget": 0,
                "acc_reg_budget": None,
                "auto_fallback_when_full": False,
                "need_grad_w": bool(need_grad_w),
                "split_backward": False,
            },
            "profile": schedule_result.profile,
        }

    return code


def generate_code_uniform1d_bwd_split_with_scheduler(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    out_path: str = "generated_uniform1d_bwd_lars_split.cu",
    kernel_name: str = "uniform1d_bwd_lars",
    reg_budget: Union[int, Iterable[int]] = 24,
    acc_reg_budget: Optional[Union[int, Iterable[Optional[int]]]] = None,
    smem_acc_volatile: bool = True,
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    fold_reg_budgets_by_max_live: bool = True,
    profile: bool = True,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = True,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
):
    """
    Backward-compatible wrapper.  New code should call
    generate_code_uniform1d_bwd_with_scheduler(..., split_backward=True).
    """
    return generate_code_uniform1d_bwd_with_scheduler(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
        input_indices=input_indices,
        output_indices=output_indices,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        mode=mode,
        need_grad_w=need_grad_w,
        out_path=out_path,
        kernel_name=kernel_name,
        # Keep legacy parameters in the public signature, but force the new
        # policy: input labels unbounded, accumulators all register-resident.
        reg_budget=reg_budget,
        acc_reg_budget=None,
        smem_acc_volatile=False,
        path_semantics=path_semantics,
        return_schedule=return_schedule,
        fold_reg_budgets_by_max_live=fold_reg_budgets_by_max_live,
        profile=profile,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        split_backward=True,
    )



def generate_code_uniform1d_bwd_with_scheduler(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    out_path: str = "generated_uniform1d_bwd_lars.cu",
    kernel_name: str = "uniform1d_bwd_lars",
    scalar_t: str = "float",
    reg_budget: Union[int, Iterable[int]] = 24,
    acc_reg_budget: Optional[Union[int, Iterable[Optional[int]]]] = None,
    smem_acc_volatile: bool = True,
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    fold_reg_budgets_by_max_live: bool = True,
    profile: bool = True,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = True,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    split_backward: Union[bool, str] = "auto",
    split_path_threshold: int = 256,
):
    """
    Unified LARS backward code generator.

    split_backward controls implementation strategy:
      - False / "fused": one fused LARS backward kernel.
      - True / "split": independent grad_w, grad_x and grad_y kernels.
      - "auto": split when path_count > split_path_threshold.
    """
    del scalar_t

    ctx = _prepare_uniform1d_bwd_codegen_context(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
        input_indices=input_indices,
        output_indices=output_indices,
        mode=mode,
        path_semantics=path_semantics,
    )

    use_split_backward = _resolve_split_backward(
        split_backward,
        path_count=ctx.P,
        split_path_threshold=split_path_threshold,
    )

    common_kwargs = dict(
        ctx=ctx,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        mode=mode,
        need_grad_w=need_grad_w,
        out_path=out_path,
        kernel_name=kernel_name,
        reg_budget=reg_budget,
        acc_reg_budget=None,
        smem_acc_volatile=False,
        return_schedule=return_schedule,
        fold_reg_budgets_by_max_live=fold_reg_budgets_by_max_live,
        profile=profile,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
    )

    if use_split_backward:
        return _generate_code_uniform1d_bwd_split_from_context(**common_kwargs)

    return _generate_code_uniform1d_bwd_fused_from_context(**common_kwargs)


# =============================================================================
# Baseline full-unrolled codegen: no LARS and no resident accumulator reuse
# =============================================================================

def _make_baseline_wxy_paths_from_uniform1d_lists(
    i_list: List[int],
    j_list: List[int],
    k_list: List[int],
    v_list: List[int],
    coeff_list: List[float],
    *,
    path_semantics: str,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Convert external Uniform1D path indices to baseline-native tuples:
        (w_index, x_index, y_index, out_index, coeff)

    path_semantics="wxy": external (i,j,k) means w[i], x[j], y[k].
    path_semantics="xyw": external (i,j,k) means x[i], y[j], w[k].

    This baseline intentionally does not perform path scheduling, operand reuse,
    register-budget management.  The Python code generator simply
    emits one straight-line block per cg path, in the original path order.
    """
    if path_semantics not in ("wxy", "xyw"):
        raise ValueError("path_semantics must be 'wxy' or 'xyw'")
    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

    paths: List[Tuple[int, int, int, int, float]] = []
    for i, j, k, v, c in zip(i_list, j_list, k_list, v_list, coeff_list):
        i = int(i); j = int(j); k = int(k); v = int(v); c = float(c)
        if path_semantics == "wxy":
            paths.append((i, j, k, v, c))
        else:
            paths.append((k, i, j, v, c))
    return paths


def emit_fused_fwd_kernel_baseline_unrolled(
    paths_wxy: List[Tuple[int, int, int, int, float]],
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """
    Emit a forward baseline kernel with every cg path fully unrolled.

    Baseline policy:
      - no LARS schedule_result;
      - no resident output accumulator grouping;
      - every path reloads w/x/y from global memory and immediately updates out.

    This is intended as a diagnostic lower-level baseline for comparing the
    benefit/cost of LARS scheduling and accumulator residency.
    """
    if block_size != 32:
        raise ValueError("this baseline emitter currently assumes block_size=32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    scalar_t* __restrict__ out,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
        ap("")
    ap("    const int e_orig = e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    ap("    const int x_row = " + ("src_idx[e_orig];" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src_idx[e_orig];" if use_y_src else "e_orig;"))
    ap("    const int out_row = " + ("dst_idx[e_orig];" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;")
    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;")
    if mode_scalar_y:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base = (index_t)y_row * (index_t)Ky * (index_t)U;")
    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t out_base = (index_t)out_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t out_base = (index_t)out_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")

    for pid, (wi, xj, yk, ov, coeff) in enumerate(paths_wxy):
        w_expr = _lars_label_index_expr(
            "w", int(wi),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=ix_dim,
            y_dim=ky_dim,
            w_dim=iw_dim,
            v_dim=v_dim,
        )
        x_expr = _lars_label_index_expr(
            "x", int(xj),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=ix_dim,
            y_dim=ky_dim,
            w_dim=iw_dim,
            v_dim=v_dim,
        )
        y_expr = _lars_label_index_expr(
            "y", int(yk),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=ix_dim,
            y_dim=ky_dim,
            w_dim=iw_dim,
            v_dim=v_dim,
        )
        out_expr = _lars_label_index_expr(
            "o", int(ov),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=ix_dim,
            y_dim=ky_dim,
            w_dim=iw_dim,
            v_dim=v_dim,
        )
        c = _fmt_lars_float(float(coeff))
        ap(f"            // baseline path#{pid}: out[{int(ov)}] += w[{int(wi)}] * x[{int(xj)}] * y[{int(yk)}] * {c}")
        ap("            {")
        ap(f"                const scalar_t wv = w[{w_expr}];")
        ap(f"                const scalar_t xv = x[{x_expr}];")
        ap(f"                const scalar_t yv = y[{y_expr}];")
        ap(f"                const scalar_t delta = scalar_t({c}) * (wv * xv) * yv;")
        if use_scatter:
            ap(f"                atomicAdd(&out[{out_expr}], delta);")
        else:
            ap(f"                out[{out_expr}] += delta;")
        ap("            }")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32_baseline_fwd(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32_baseline_fwd(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32_baseline_fwd(a, b)) return false;")
    ap("    return mul_fits_int32_baseline_fwd(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_baseline_fwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32_baseline_fwd((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32_baseline_fwd(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32_baseline_fwd(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32_baseline_fwd(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t out_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool out_ok = mul3_fits_int32_baseline_fwd(out_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && out_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        w, x, y, out,")
    ap("        src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_baseline_fwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                            kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    ap("            w, x, y, out, src_idx, dst_idx,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    ap("            w, x, y, out, src_idx, dst_idx,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    ap("        w, x, y, out, src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def emit_fused_bwd_kernel_baseline_unrolled(
    paths_wxy: List[Tuple[int, int, int, int, float]],
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """
    Emit a backward baseline kernel with every cg path fully unrolled.

    Baseline policy:
      - no LARS input-register scheduling;
      - no resident gw/gx/gy accumulators;
      - every path reloads w/x/y/grad_out and immediately writes gradients.

    grad_x/grad_y use atomicAdd, matching the fused backward semantics.  grad_w
    intentionally uses a plain += path update to stay consistent with the
    existing non-atomic grad_w assumption in this codebase.
    """
    if block_size != 32:
        raise ValueError("this baseline emitter currently assumes block_size=32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")
    ap("template <typename scalar_t>")
    ap("__device__ __forceinline__ scalar_t warp_sum_xor_baseline_bwd(scalar_t v) {")
    ap("    for (int offset = 16; offset > 0; offset >>= 1) {")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("        v += __shfl_xor(v, offset);")
    ap("#else")
    ap("        v += __shfl_xor_sync(0xffffffff, v, offset);")
    ap("#endif")
    ap("    }")
    ap("    return v;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    if need_grad_w:
        ap("    scalar_t* __restrict__ grad_w,")
    ap("    scalar_t* __restrict__ grad_x,")
    ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
    ap("    const int e_orig = e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e_orig];")
    else:
        ap("    const int src = e_orig;")
    if use_scatter:
        ap("    const int dst = dst_idx[e_orig];")
    else:
        ap("    const int dst = e_orig;")
    ap("")
    ap("    const int x_row = " + ("src;" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src;" if use_y_src else "e_orig;"))
    ap("    const int go_row = " + ("dst;" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base  = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
        if need_grad_w:
            ap(f"    const index_t gw_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base  = (index_t)w_row * (index_t)Iw * (index_t)U;")
        if need_grad_w:
            ap("    const index_t gw_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base  = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
        ap(f"    const index_t gx_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base  = (index_t)x_row * (index_t)Ix * (index_t)U;")
        ap("    const index_t gx_base = (index_t)x_row * (index_t)Ix * (index_t)U;")

    if mode_scalar_y:
        ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky;")
        ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base  = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
            ap(f"    const index_t gy_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky * (index_t)U;")
            ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky * (index_t)U;")

    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t go_base = (index_t)go_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t go_base = (index_t)go_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")

    for pid, (wi, xj, yk, ov, coeff) in enumerate(paths_wxy):
        w_expr = _bwd_label_index_expr(
            "w", int(wi),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        gw_expr = _bwd_label_index_expr(
            "gw", int(wi),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        x_expr = _bwd_label_index_expr(
            "x", int(xj),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        gx_expr = _bwd_label_index_expr(
            "gx", int(xj),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        y_expr = _bwd_label_index_expr(
            "y", int(yk),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        gy_expr = _bwd_label_index_expr(
            "gy", int(yk),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        go_expr = _bwd_label_index_expr(
            "go", int(ov),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        c = _fmt_lars_float(float(coeff))
        ap(f"            // baseline path#{pid}: immediate gradient writes for w[{int(wi)}], x[{int(xj)}], y[{int(yk)}], go[{int(ov)}]")
        ap("            {")
        ap(f"                const scalar_t wv = w[{w_expr}];")
        ap(f"                const scalar_t xv = x[{x_expr}];")
        ap(f"                const scalar_t yv = y[{y_expr}];")
        ap(f"                const scalar_t gov = grad_out[{go_expr}];")
        ap(f"                const scalar_t cg = scalar_t({c});")
        if need_grad_w:
            ap("                const scalar_t dgw = cg * gov * xv * yv;")
            ap(f"                grad_w[{gw_expr}] += dgw;")
        ap("                const scalar_t wg = wv * gov;")
        ap("                const scalar_t dgx = cg * wg * yv;")
        ap(f"                atomicAdd(&grad_x[{gx_expr}], dgx);")
        ap("                const scalar_t dgy = cg * wg * xv;")
        if mode_scalar_y:
            ap("                const scalar_t dgy_sum = warp_sum_xor_baseline_bwd(dgy);")
            ap("                if (lane == 0) {")
            ap(f"                    atomicAdd(&grad_y[{gy_expr}], dgy_sum);")
            ap("                }")
        else:
            ap(f"                atomicAdd(&grad_y[{gy_expr}], dgy);")
        ap("            }")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32_baseline_bwd(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32_baseline_bwd(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32_baseline_bwd(a, b)) return false;")
    ap("    return mul_fits_int32_baseline_bwd(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_baseline_bwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32_baseline_bwd((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32_baseline_bwd(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32_baseline_bwd(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32_baseline_bwd(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool go_ok = mul3_fits_int32_baseline_bwd(go_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && go_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_baseline_bwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                            kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def generate_code_uniform1d_fwd_baseline_unrolled(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    out_path: str = "generated_uniform1d_fwd_baseline_unrolled.cu",
    kernel_name: str = "uniform1d_fwd_baseline_unrolled",
    scalar_t: str = "float",
    path_semantics: str = "wxy",
    return_metadata: bool = False,
):
    """
    Generate a forward baseline CUDA implementation by fully unrolling cg paths.

    This entry deliberately bypasses LARS and all reuse-aware scheduling.  It is
    meant to answer: what happens if we just emit the cg path list as straight-
    line code?
    """
    del scalar_t

    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    paths_wxy = _make_baseline_wxy_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"
    if kernel_name and kernel_name != "uniform1d_fwd_baseline_unrolled":
        base_kernel_name = kernel_name
    else:
        base_kernel_name = f"uniform1d_baseline_unrolled_u{u_dim}_path{P}_{mode_str}_{layout_tag}_fwd"

    code = emit_fused_fwd_kernel_baseline_unrolled(
        paths_wxy,
        kernel_name=base_kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        block_size=32,
    )

    code = code + "\n" + emit_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")

    if return_metadata:
        return {
            "name": "baseline_unrolled_fwd",
            "code": code,
            "kernel_name": base_kernel_name,
            "num_paths": int(P),
            "config": {
                "mode": mode,
                "path_semantics": path_semantics,
                "use_x_src": use_x_src,
                "use_y_src": use_y_src,
                "use_scatter": use_scatter,
                "manual_register_management": False,
                "manual_data_reuse": False,
                "fully_unrolled_paths": True,
            },
        }
    return code


def generate_code_uniform1d_bwd_baseline_unrolled(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    out_path: str = "generated_uniform1d_bwd_baseline_unrolled.cu",
    kernel_name: str = "uniform1d_bwd_baseline_unrolled",
    scalar_t: str = "float",
    path_semantics: str = "wxy",
    return_metadata: bool = False,
):
    """
    Generate a backward baseline CUDA implementation by fully unrolling cg paths.

    This entry deliberately bypasses LARS and resident gradient
    accumulators.  Each path reloads w/x/y/grad_out and immediately writes
    grad_w/grad_x/grad_y.
    """
    del scalar_t

    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    paths_wxy = _make_baseline_wxy_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"
    grad_tag = "full" if need_grad_w else "nogradw"
    if kernel_name and kernel_name != "uniform1d_bwd_baseline_unrolled":
        base_kernel_name = kernel_name
    else:
        base_kernel_name = f"uniform1d_baseline_unrolled_u{u_dim}_path{P}_{mode_str}_{layout_tag}_{grad_tag}_bwd"

    code = emit_fused_bwd_kernel_baseline_unrolled(
        paths_wxy,
        kernel_name=base_kernel_name,
        mode=mode,
        need_grad_w=need_grad_w,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        block_size=32,
    )

    code = code + "\n" + emit_lars_bwd_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        need_grad_w=need_grad_w,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")

    if return_metadata:
        return {
            "name": "baseline_unrolled_bwd",
            "code": code,
            "kernel_name": base_kernel_name,
            "num_paths": int(P),
            "config": {
                "mode": mode,
                "path_semantics": path_semantics,
                "need_grad_w": need_grad_w,
                "use_x_src": use_x_src,
                "use_y_src": use_y_src,
                "use_scatter": use_scatter,
                "manual_register_management": False,
                "manual_data_reuse": False,
                "resident_gradient_accumulators": False,
                "fully_unrolled_paths": True,
            },
        }
    return code