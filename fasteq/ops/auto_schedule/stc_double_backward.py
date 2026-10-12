"""FastEq scheduling: stc double backward."""
from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Union, Iterable, Set, Sequence
from time import perf_counter
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import torch

from .cuda_common import (
    _fmt_lars_float,
    _max_lars_reg_count_any,
    _sanitize_cuda_comment,
)
from .metadata import (
    _check_stc_path_bounds_for_bwd,
    _normalize_stc_padded_paths,
    infer_stc_path_lens_from_padded,
    make_stc_paths_from_padded_lists,
)
from .models import (
    STC_PAD_VALUE,
    ScheduleResult,
)
from .schedule_stats import (
    _dump_and_print_schedule_label_stats,
    _stats_labels_stc_double_bwd,
)
from .schedulers import (
    LARSSTCDoubleBwdScheduler,
)


def emit_stc_double_bwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    u_dim: Optional[int] = None,
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """Emit fully-unrolled STC double backward from a LARS schedule."""
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")
    reg_count = _max_lars_reg_count_any(schedule_result)

    lines: List[str] = []
    def ap(line: str = "") -> None:
        lines.append(line)

    def parse_ref(ref: str) -> Tuple[str, int]:
        if "[" not in ref or not ref.endswith("]"):
            raise ValueError(f"Bad STC double backward ref: {ref!r}")
        kind, tail = ref.split("[", 1)
        idx = int(tail[:-1])
        aliases = {
            "grad_out": "go", "grad_grad_x0": "ggx0",
            "grad_grad_x1": "ggx1",
            "d_grad_out": "dgo", "d_x0": "dx0", "d_x1": "dx1",
        }
        return aliases.get(kind, kind), idx

    def base_index(dim_name: str, idx: int) -> str:
        return f"((index_t)b * (index_t){dim_name} + (index_t){int(idx)}) * (index_t)U + (index_t)u"

    def x0_base_index(idx: int) -> str:
        # x0 is the original ungathered tensor.  STC uses x0_g=x0[i0], so
        # double backward must read and reduce gradients through i0[b].
        return f"((index_t)x0_b * (index_t)X0 + (index_t){int(idx)}) * (index_t)U + (index_t)u"

    def input_expr(kind: str, idx: int) -> str:
        if kind == "go": return f"grad_out[{base_index('V', idx)}]"
        if kind == "x0": return f"x0[{x0_base_index(idx)}]"
        if kind == "x1": return f"x1[{base_index('X1', idx)}]"
        if kind == "ggx0": return f"grad_grad_x0[{x0_base_index(idx)}]"
        if kind == "ggx1": return f"grad_grad_x1[{base_index('X1', idx)}]"
        raise ValueError(f"Bad STC double backward input kind: {kind}")

    def output_lhs(kind: str, idx: int) -> str:
        if kind == "dgo": return f"d_grad_out[{base_index('V', idx)}]"
        if kind == "dx0": return f"d_x0[{x0_base_index(idx)}]"
        if kind == "dx1": return f"d_x1[{base_index('X1', idx)}]"
        raise ValueError(f"Bad STC double backward output kind: {kind}")

    def mul_terms(terms: Sequence[str]) -> str:
        if not terms:
            return "scalar_t(1)"
        expr = str(terms[0])
        for term in terms[1:]:
            expr = f"({expr} * {term})"
        return expr

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
    ap("    const scalar_t* __restrict__ grad_out,")
    ap("    const scalar_t* __restrict__ x1,")
    ap("    const scalar_t* __restrict__ x0,")
    ap("    const int64_t* __restrict__ i0,")
    ap("    const scalar_t* __restrict__ grad_grad_x0,")
    ap("    const scalar_t* __restrict__ grad_grad_x1,")
    ap("    scalar_t* __restrict__ d_grad_out,")
    ap("    scalar_t* __restrict__ d_x1,")
    ap("    scalar_t* __restrict__ d_x0,")
    ap("    int B, int B0, int X1, int X0, int V, int U)")
    ap("{")
    ap("    const int b = (int)blockIdx.x;")
    ap("    if (b >= B) return;")
    ap("    const int64_t x0_b64 = i0[b];")
    ap("    if (x0_b64 < 0 || x0_b64 >= (int64_t)B0) return;")
    ap("    const index_t x0_b = (index_t)x0_b64;")
    ap("    const int tid = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        ap(f"            // stc-double-bwd inst {inst_id}: {inst.op}" + (f" | {comment}" if comment else ""))
        if inst.op == "load":
            reg, ref = inst.args
            kind, idx = parse_ref(str(ref))
            ap(f"            {reg} = {input_expr(kind, idx)};")
        elif inst.op == "init_acc":
            reg, _ref = inst.args
            ap(f"            {reg} = scalar_t(0);")
        elif inst.op == "stc_double_bwd_path":
            x1_indices, x0_idx, ov, coeff, rgo, rx0, rggx0, rx1, rgg, rdgo, rdx0, rdx1 = inst.args
            xs = tuple(int(v) for v in x1_indices)
            rx1 = tuple(str(v) for v in rx1)
            rgg = tuple(str(v) for v in rgg)
            rdx1 = tuple(str(v) for v in rdx1)
            c = _fmt_lars_float(float(coeff))
            prod_all = mul_terms(rx1)
            ap(f"            {rdgo} += scalar_t({c}) * {rggx0} * {prod_all};")
            for r in range(len(xs)):
                prod_except_r = mul_terms([
                    rx1[q] for q in range(len(xs)) if q != r
                ])
                ap(f"            {rdx1[r]} += scalar_t({c}) * {rggx0} * {rgo} * {prod_except_r};")
            for t in range(len(xs)):
                prod_except_t = mul_terms([rx1[q] for q in range(len(xs)) if q != t])
                ap(f"            {rdgo} += scalar_t({c}) * {rgg[t]} * {rx0} * {prod_except_t};")
                ap(f"            {rdx0} += scalar_t({c}) * {rgg[t]} * {rgo} * {prod_except_t};")
                for r in range(len(xs)):
                    if r == t:
                        continue
                    prod_except_tr = mul_terms([
                        rx1[q] for q in range(len(xs)) if q != t and q != r
                    ])
                    ap(f"            {rdx1[r]} += scalar_t({c}) * {rgg[t]} * {rgo} * {rx0} * {prod_except_tr};")
        elif inst.op == "store_acc":
            ref, reg = inst.args
            kind, idx = parse_ref(str(ref))
            if kind == "dx0":
                # Multiple b values may map to the same original x0 row.
                # LARS reduces all path contributions within one b/block; only
                # the final cross-block gather-backward reduction is atomic.
                ap(f"            atomicAdd(&{output_lhs(kind, idx)}, {reg});")
            else:
                ap(f"            {output_lhs(kind, idx)} = {reg};")
        elif inst.op == "release":
            pass
        else:
            raise ValueError(f"Unsupported STC double backward instruction op: {inst.op}")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")
    ap("static inline bool stc_db_mul3_fits_i32(int64_t a, int64_t b, int64_t c) {")
    ap("    if (a < 0 || b < 0 || c < 0) return false;")
    ap("    if (a == 0 || b == 0 || c == 0) return true;")
    ap("    constexpr int64_t L = 2147483647LL;")
    ap("    return a <= L / b && a * b <= L / c;")
    ap("}")
    ap("")
    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, const int64_t* i0,")
    ap("    const scalar_t* grad_grad_x0, const scalar_t* grad_grad_x1,")
    ap("    scalar_t* d_grad_out, scalar_t* d_x1, scalar_t* d_x0,")
    ap("    int B, int B0, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        grad_out, x1, x0, i0, grad_grad_x0, grad_grad_x1, d_grad_out, d_x1, d_x0, B, B0, X1, X0, V, U);")
    ap("}")
    ap("")
    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, const int64_t* i0,")
    ap("    const scalar_t* grad_grad_x0, const scalar_t* grad_grad_x1,")
    ap("    scalar_t* d_grad_out, scalar_t* d_x1, scalar_t* d_x0,")
    ap("    int B, int B0, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap("    bool use_i32 = stc_db_mul3_fits_i32(B, X1, U) && stc_db_mul3_fits_i32(B0, X0, U) && stc_db_mul3_fits_i32(B, V, U);")
    ap("    if (use_i32) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(grad_out,x1,x0,i0,grad_grad_x0,grad_grad_x1,d_grad_out,d_x1,d_x0,B,B0,X1,X0,V,U,stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(grad_out,x1,x0,i0,grad_grad_x0,grad_grad_x1,d_grad_out,d_x1,d_x0,B,B0,X1,X0,V,U,stream);")
    ap("    }")
    ap("}")
    return "\n".join(lines)


def emit_stc_double_bwd_launcher(bundle_name: str, *, u_dim: Optional[int] = None,
                                 x0_dim: Optional[int] = None,
                                 x1_dim: Optional[int] = None,
                                 v_dim: Optional[int] = None) -> str:
    checks: List[str] = []
    if u_dim is not None: checks.append(f'    TORCH_CHECK(U == {int(u_dim)}, "U mismatch for generated STC double backward kernel");')
    if x0_dim is not None: checks.append(f'    TORCH_CHECK(X0 == {int(x0_dim)}, "X0 mismatch for generated STC double backward kernel");')
    if x1_dim is not None: checks.append(f'    TORCH_CHECK(X1 == {int(x1_dim)}, "X1 mismatch for generated STC double backward kernel");')
    if v_dim is not None: checks.append(f'    TORCH_CHECK(V == {int(v_dim)}, "V mismatch for generated STC double backward kernel");')
    check_text = "\n".join(checks)
    return f'''

std::vector<torch::Tensor> launcher_{bundle_name}(
    torch::Tensor grad_out,
    torch::Tensor x1,
    torch::Tensor x0,
    torch::Tensor i0,
    torch::Tensor grad_grad_x0,
    torch::Tensor grad_grad_x1,
    int64_t V64)
{{
    TORCH_CHECK(grad_out.is_cuda() && x1.is_cuda() && x0.is_cuda() && i0.is_cuda() && grad_grad_x0.is_cuda() && grad_grad_x1.is_cuda(), "all tensors must be CUDA/HIP");
    TORCH_CHECK(grad_out.is_contiguous() && x1.is_contiguous() && x0.is_contiguous() && i0.is_contiguous() && grad_grad_x0.is_contiguous() && grad_grad_x1.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(x1.dim() == 3 && x0.dim() == 3 && grad_out.dim() == 3 && grad_grad_x0.dim() == 3 && grad_grad_x1.dim() == 3, "floating tensors must be 3D");
    TORCH_CHECK(i0.dim() == 1, "i0 must be 1D");
    TORCH_CHECK(i0.scalar_type() == torch::kInt64, "i0 must be int64");
    TORCH_CHECK(grad_out.scalar_type() == x1.scalar_type() && x1.scalar_type() == x0.scalar_type() && x1.scalar_type() == grad_grad_x0.scalar_type() && x1.scalar_type() == grad_grad_x1.scalar_type(), "dtype mismatch");
    TORCH_CHECK(grad_grad_x0.sizes() == x0.sizes(), "grad_grad_x0 shape must match x0");
    TORCH_CHECK(grad_grad_x1.sizes() == x1.sizes(), "grad_grad_x1 shape must match x1");
    int B = (int)x1.size(0);
    int B0 = (int)x0.size(0);
    int X1 = (int)x1.size(1);
    int X0 = (int)x0.size(1);
    int U = (int)x1.size(2);
    int V = (int)V64;
    TORCH_CHECK((int)x0.size(2) == U, "x0 U mismatch");
    TORCH_CHECK((int)i0.numel() == B, "i0 length must equal x1 batch size B");
    TORCH_CHECK((int)grad_out.size(0) == B && (int)grad_out.size(1) == V && (int)grad_out.size(2) == U, "grad_out shape mismatch");
{check_text}
    auto d_grad_out = torch::zeros_like(grad_out);
    auto d_x1 = torch::zeros_like(x1);
    auto d_x0 = torch::zeros_like(x0);
    GPU_Guard device_guard(x1.device());
    gpuStream_t stream = getCurrentGPUStream(x1.device().index());
    AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "{bundle_name}", [&] {{
        launch_{bundle_name}<scalar_t>(
            (const scalar_t*)grad_out.data_ptr<scalar_t>(),
            (const scalar_t*)x1.data_ptr<scalar_t>(),
            (const scalar_t*)x0.data_ptr<scalar_t>(),
            (const int64_t*)i0.data_ptr<int64_t>(),
            (const scalar_t*)grad_grad_x0.data_ptr<scalar_t>(),
            (const scalar_t*)grad_grad_x1.data_ptr<scalar_t>(),
            (scalar_t*)d_grad_out.data_ptr<scalar_t>(),
            (scalar_t*)d_x1.data_ptr<scalar_t>(),
            (scalar_t*)d_x0.data_ptr<scalar_t>(),
            B, B0, X1, X0, V, U, stream);
    }});
    GPU_KERNEL_LAUNCH_CHECK();
    return {{d_grad_out, d_x1, d_x0}};
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} STC double backward LARS jit impl");
}}
'''


def generate_code_stc_double_bwd_with_scheduler(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    num_out_segments: int,
    u_dim: int,
    out_path: str = "generated_stc_double_bwd_lars.cu",
    kernel_name: str = "stc_double_bwd_lars",
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    return_schedule: bool = False,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    block_size: int = 32,
) -> Any:
    """Generate fully-unrolled LARS STC double backward CUDA/HIP source."""
    paths = make_stc_paths_from_padded_lists(
        idx_lists, coeff_list, path_lens=path_lens, pad_value=pad_value
    )
    scheduler = LARSSTCDoubleBwdScheduler(
        paths,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name="stc_double_bwd_lars",
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    schedule_result = scheduler.schedule()
    _dump_and_print_schedule_label_stats(
        operation="stc_double_bwd",
        candidate_tag="stc_double_bwd_lars",
        path_order=schedule_result.path_order,
        labels_by_path=_stats_labels_stc_double_bwd(scheduler.paths),
    )
    _check_stc_path_bounds_for_bwd(
        paths, x0_dim=x0_dim, x1_dim=x1_dim, v_dim=int(num_out_segments)
    )
    idx_norm = _normalize_stc_padded_paths(
        idx_lists, coeff_list=coeff_list, path_lens=path_lens
    )
    lens = (
        infer_stc_path_lens_from_padded(idx_norm, coeff_list, pad_value=pad_value)
        if path_lens is None else
        path_lens.detach().cpu().to(torch.int64).reshape(-1)
    )
    base_kernel_name = (
        f"{kernel_name}_u{int(u_dim)}_path{len(paths)}_maxlen{int(lens.max().item())}"
    )
    code = emit_stc_double_bwd_kernel_from_lars_schedule(
        schedule_result,
        kernel_name=base_kernel_name,
        u_dim=int(u_dim), x0_dim=x0_dim, x1_dim=x1_dim,
        v_dim=int(num_out_segments), block_size=int(block_size),
    )
    code += "\n" + emit_stc_double_bwd_launcher(
        base_kernel_name, u_dim=int(u_dim), x0_dim=x0_dim,
        x1_dim=x1_dim, v_dim=int(num_out_segments),
    )
    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")
    if return_schedule:
        return {
            "name": "stc_double_bwd_lars",
            "code": code,
            "kernel_name": base_kernel_name,
            "schedule": schedule_result,
            "num_paths": len(paths),
            "max_live": schedule_result.max_live,
            "profile": schedule_result.profile,
        }
    return code
