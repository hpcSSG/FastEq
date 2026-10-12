"""FastEq scheduling: uniform1d double backward."""
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
    _cuda_scalar_warp_sum_preamble,
    _emit_backend_gradient_atomic,
    _fmt_lars_float,
    _max_lars_reg_count_any,
    _sanitize_cuda_comment,
)
from .models import (
    ScheduleResult,
)
from .schedule_stats import (
    _dump_and_print_schedule_label_stats,
    _stats_labels_uniform1d_double_bwd,
)
from .schedulers import (
    LARSUniform1DDoubleBwdScheduler,
)
from .uniform1d_backward import (
    _prepare_uniform1d_bwd_codegen_context,
)


def emit_uniform1d_double_bwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    mode: str,
    need_grad_w: bool,
    need_grad_x: bool,
    need_grad_y: bool,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """Emit a fully-unrolled fused double-backward kernel from a LARS schedule."""
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")
    mode_scalar_y = mode == "u,u,,u"
    if not (need_grad_w or need_grad_x or need_grad_y):
        raise ValueError("at least one first-gradient output must be requested")
    reg_count = _max_lars_reg_count_any(schedule_result)

    lines: List[str] = []
    def ap(line: str = "") -> None:
        lines.append(line)

    def parse_ref(ref: str) -> Tuple[str, int]:
        if "[" not in ref or not ref.endswith("]"):
            raise ValueError(f"Bad double backward ref: {ref!r}")
        kind, tail = ref.split("[", 1)
        idx = int(tail[:-1])
        aliases = {
            "grad_out": "go",
            "grad_grad_w": "ggw",
            "grad_grad_x": "ggx",
            "grad_grad_y": "ggy",
        }
        return aliases.get(kind, kind), idx

    def input_expr(kind: str, idx: int) -> str:
        if kind == "w":
            return f"w[w_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "x":
            return f"x[x_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "y":
            if mode_scalar_y:
                return f"y[y_base + (index_t){idx}]"
            return f"y[y_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "go":
            return f"grad_out[go_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "ggw":
            return f"grad_grad_w[ggw_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "ggx":
            return f"grad_grad_x[ggx_base + (index_t){idx} * (index_t)U + (index_t)u]"
        if kind == "ggy":
            if mode_scalar_y:
                return f"grad_grad_y[ggy_base + (index_t){idx}]"
            return f"grad_grad_y[ggy_base + (index_t){idx} * (index_t)U + (index_t)u]"
        raise ValueError(f"Bad double backward input kind: {kind}")

    def output_index(kind: str, idx: int) -> str:
        if kind == "dgo":
            return f"dgo_base + (index_t){idx} * (index_t)U + (index_t)u"
        if kind == "dw":
            return f"dw_base + (index_t){idx} * (index_t)U + (index_t)u"
        if kind == "dx":
            return f"dx_base + (index_t){idx} * (index_t)U + (index_t)u"
        if kind == "dy":
            if mode_scalar_y:
                return f"dy_base + (index_t){idx}"
            return f"dy_base + (index_t){idx} * (index_t)U + (index_t)u"
        raise ValueError(kind)

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
    ap(_cuda_scalar_warp_sum_preamble())
    ap("")
    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    if need_grad_w:
        ap("    const scalar_t* __restrict__ grad_grad_w,")
    if need_grad_x:
        ap("    const scalar_t* __restrict__ grad_grad_x,")
    if need_grad_y:
        ap("    const scalar_t* __restrict__ grad_grad_y,")
    ap("    scalar_t* __restrict__ d_grad_out,")
    if need_grad_w:
        ap("    scalar_t* __restrict__ d_w,")
    if need_grad_x:
        ap("    scalar_t* __restrict__ d_x,")
    if need_grad_y:
        ap("    scalar_t* __restrict__ d_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U)")
    ap("{")
    ap("    const int e = (int)blockIdx.x;")
    ap("    if (e >= B) return;")
    ap("    const int tid = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e];")
    else:
        ap("    const int src = e;")
    if use_scatter:
        ap("    const int dst = dst_idx[e];")
    else:
        ap("    const int dst = e;")
    ap("    const int w_row = (WB == 1 ? 0 : e);")
    ap(f"    const int x_row = {'src' if use_x_src else 'e'};")
    ap(f"    const int y_row = {'src' if use_y_src else 'e'};")
    ap(f"    const int go_row = {'dst' if use_scatter else 'e'};")
    ap("    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;")
    ap("    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;")
    if mode_scalar_y:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky;")
    else:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky * (index_t)U;")
    ap("    const index_t go_base = (index_t)go_row * (index_t)V * (index_t)U;")
    if need_grad_w:
        ap("    const index_t ggw_base = w_base;")
    if need_grad_x:
        ap("    const index_t ggx_base = x_base;")
    if need_grad_y:
        ap("    const index_t ggy_base = y_base;")
    ap("    const index_t dgo_base = go_base;")
    if need_grad_w:
        ap("    const index_t dw_base = w_base;")
    if need_grad_x:
        ap("    const index_t dx_base = x_base;")
    if need_grad_y:
        ap("    const index_t dy_base = y_base;")
    ap("")
    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        ap(f"            // double-bwd inst {inst_id}: {inst.op}" + (f" | {comment}" if comment else ""))
        if inst.op == "load":
            reg, ref = inst.args
            kind, idx = parse_ref(str(ref))
            ap(f"            {reg} = {input_expr(kind, idx)};")
        elif inst.op == "double_bwd_fma_resident":
            wi, xj, yk, ov, rw, rx, ry, rgo, rggw, rggx, rggy, coeff, inst_need_gw, inst_need_gx, inst_need_gy = inst.args
            c = _fmt_lars_float(float(coeff))
            t_dgo: List[str] = []
            if bool(inst_need_gw):
                t_dgo.append(f"({rggw} * {rx} * {ry})")
            if bool(inst_need_gx):
                t_dgo.append(f"({rggx} * {rw} * {ry})")
            if bool(inst_need_gy):
                t_dgo.append(f"({rggy} * {rw} * {rx})")
            t_dw: List[str] = []
            if bool(inst_need_gx):
                t_dw.append(f"({rggx} * {rgo} * {ry})")
            if bool(inst_need_gy):
                t_dw.append(f"({rggy} * {rgo} * {rx})")
            t_dx: List[str] = []
            if bool(inst_need_gw):
                t_dx.append(f"({rggw} * {rgo} * {ry})")
            if bool(inst_need_gy):
                t_dx.append(f"({rggy} * {rgo} * {rw})")
            ap(f"            atomicAdd(&d_grad_out[{output_index('dgo', int(ov))}], scalar_t({c}) * ({' + '.join(t_dgo)}));")
            if bool(inst_need_gw) and t_dw:
                ap(f"            atomicAdd(&d_w[{output_index('dw', int(wi))}], scalar_t({c}) * ({' + '.join(t_dw)}));")
            if bool(inst_need_gx) and t_dx:
                ap(f"            atomicAdd(&d_x[{output_index('dx', int(xj))}], scalar_t({c}) * ({' + '.join(t_dx)}));")
            if bool(inst_need_gy):
                t_dy: List[str] = []
                if bool(inst_need_gw):
                    t_dy.append(f"({rggw} * {rgo} * {rx})")
                if bool(inst_need_gx):
                    t_dy.append(f"({rggx} * {rgo} * {rw})")
                if t_dy:
                    _emit_backend_gradient_atomic(ap, f"&d_y[{output_index('dy', int(yk))}]", f"scalar_t({c}) * ({' + '.join(t_dy)})", scalar_channel=mode_scalar_y)
        elif inst.op == "release":
            pass
        else:
            raise ValueError(f"Unsupported double backward instruction op: {inst.op}")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")
    grad_grad_names = []
    if need_grad_w:
        grad_grad_names.append("grad_grad_w")
    if need_grad_x:
        grad_grad_names.append("grad_grad_x")
    if need_grad_y:
        grad_grad_names.append("grad_grad_y")
    double_output_names = ["d_grad_out"]
    if need_grad_w:
        double_output_names.append("d_w")
    if need_grad_x:
        double_output_names.append("d_x")
    if need_grad_y:
        double_output_names.append("d_y")
    kernel_call_args = (["w", "x", "y", "grad_out"] + grad_grad_names +
                        double_output_names + ["src_idx", "dst_idx", "B", "WB",
                        "Iw", "Ix", "Ky", "V", "U"])

    def emit_double_launch_signature(template_line: str, function_line: str) -> None:
        ap(template_line)
        ap(function_line)
        ap("    const scalar_t* w, const scalar_t* x, const scalar_t* y, const scalar_t* grad_out,")
        for name in grad_grad_names:
            ap(f"    const scalar_t* {name},")
        for name in double_output_names:
            ap(f"    scalar_t* {name},")
        ap("    const int32_t* src_idx, const int32_t* dst_idx,")
        ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, gpuStream_t stream)")

    emit_double_launch_signature(
        "template <typename scalar_t, typename index_t>",
        f"void launch_{kernel_name}_typed(",
    )
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        " + ", ".join(kernel_call_args) + ");")
    ap("}")
    ap("")

    emit_double_launch_signature(
        "template <typename scalar_t>",
        f"void launch_{kernel_name}(",
    )
    ap("{")
    ap("    bool use_i32 = ((int64_t)WB * Iw * U <= 2147483647LL) &&")
    ap("                   ((int64_t)B * V * U <= 2147483647LL);")
    ap("    if (use_i32) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(" + ", ".join(kernel_call_args) + ", stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(" + ", ".join(kernel_call_args) + ", stream);")
    ap("    }")
    ap("}")
    return "\n".join(lines)


def emit_uniform1d_double_bwd_launcher(
    bundle_name: str,
    mode: str,
    *,
    need_grad_w: bool,
    need_grad_x: bool,
    need_grad_y: bool,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
    params = [
        "    torch::Tensor w",
        "    torch::Tensor x",
        "    torch::Tensor y",
        "    torch::Tensor grad_out",
    ]
    if need_grad_w:
        params.append("    torch::Tensor grad_grad_w")
    if need_grad_x:
        params.append("    torch::Tensor grad_grad_x")
    if need_grad_y:
        params.append("    torch::Tensor grad_grad_y")
    if use_x_src or use_y_src:
        params.append("    torch::Tensor src_idx")
    if use_scatter:
        params.append("    torch::Tensor dst_idx")
    params.append("    int64_t V64")
    params_s = ",\n".join(params)

    b_expr = "src_idx.size(0)" if (use_x_src or use_y_src) else ("dst_idx.size(0)" if use_scatter else "grad_out.size(0)")
    src_checks = ""
    if use_x_src or use_y_src:
        src_checks = '''
    TORCH_CHECK(src_idx.is_cuda() && src_idx.is_contiguous(), "src_idx must be contiguous CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''
    dst_checks = ""
    if use_scatter:
        dst_checks = '''
    TORCH_CHECK(dst_idx.is_cuda() && dst_idx.is_contiguous(), "dst_idx must be contiguous CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''
    ggw_checks = ""
    if need_grad_w:
        ggw_checks = '''
    TORCH_CHECK(grad_grad_w.is_cuda() && grad_grad_w.is_contiguous(), "grad_grad_w must be contiguous CUDA/HIP");
    TORCH_CHECK(grad_grad_w.sizes() == w.sizes(), "grad_grad_w shape must match w");
    TORCH_CHECK(grad_grad_w.scalar_type() == w.scalar_type(), "grad_grad_w dtype mismatch");
'''
    ggx_checks = ""
    if need_grad_x:
        ggx_checks = '''
    TORCH_CHECK(grad_grad_x.is_cuda() && grad_grad_x.is_contiguous(), "grad_grad_x must be contiguous CUDA/HIP");
    TORCH_CHECK(grad_grad_x.sizes() == x.sizes(), "grad_grad_x shape must match x");
    TORCH_CHECK(grad_grad_x.scalar_type() == w.scalar_type(), "grad_grad_x dtype mismatch");
'''
    y_shape_check = (
        '    TORCH_CHECK((int)y.size(2) == 1, "scalar-y mode expects y[...,1]");'
        if mode == "u,u,,u" else
        '    TORCH_CHECK((int)y.size(2) == U, "vector-y mode expects y[...,U]");'
    )
    ggy_checks = ""
    if need_grad_y:
        ggy_checks = (
            '    TORCH_CHECK(grad_grad_y.is_cuda() && grad_grad_y.is_contiguous(), "grad_grad_y must be contiguous CUDA/HIP");\n'
            '    TORCH_CHECK(grad_grad_y.sizes() == y.sizes(), "grad_grad_y shape must match y");\n'
            '    TORCH_CHECK(grad_grad_y.scalar_type() == w.scalar_type(), "grad_grad_y dtype mismatch");'
        )
    src_arg = "(const int32_t*)src_idx.data_ptr<int32_t>()" if (use_x_src or use_y_src) else "nullptr"
    dst_arg = "(const int32_t*)dst_idx.data_ptr<int32_t>()" if use_scatter else "nullptr"
    ggw_launch_decl = "        (const scalar_t*)grad_grad_w.data_ptr<scalar_t>(),\n" if need_grad_w else ""
    ggx_launch_decl = "            (const scalar_t*)grad_grad_x.data_ptr<scalar_t>(),\n" if need_grad_x else ""
    ggy_launch_decl = "            (const scalar_t*)grad_grad_y.data_ptr<scalar_t>(),\n" if need_grad_y else ""
    dw_alloc = "    auto d_w = torch::zeros_like(w);\n" if need_grad_w else ""
    dx_alloc = "    auto d_x = torch::zeros_like(x);\n" if need_grad_x else ""
    dy_alloc = "    auto d_y = torch::zeros_like(y);\n" if need_grad_y else ""
    dw_launch_decl = "            (scalar_t*)d_w.data_ptr<scalar_t>(),\n" if need_grad_w else ""
    dx_launch_decl = "            (scalar_t*)d_x.data_ptr<scalar_t>(),\n" if need_grad_x else ""
    dy_launch_decl = "            (scalar_t*)d_y.data_ptr<scalar_t>(),\n" if need_grad_y else ""
    double_result_names = ["d_grad_out"]
    if need_grad_w:
        double_result_names.append("d_w")
    if need_grad_x:
        double_result_names.append("d_x")
    if need_grad_y:
        double_result_names.append("d_y")
    ret_expr = "return {" + ", ".join(double_result_names) + "};"

    return f'''

torch::Tensor launcher_dummy_{bundle_name}();

std::vector<torch::Tensor> launcher_{bundle_name}(
{params_s})
{{
    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(), "w/x/y/grad_out must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(), "w/x/y/grad_out must be contiguous");
{src_checks}{dst_checks}{ggw_checks}{ggx_checks}{ggy_checks}
    TORCH_CHECK(w.dim() == 3 && x.dim() == 3 && y.dim() == 3 && grad_out.dim() == 3, "all primal tensors must be 3D");
    int B = (int){b_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U = (int)w.size(2);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V = (int)V64;
    TORCH_CHECK(B > 0 && V > 0, "B/V must be > 0");
    TORCH_CHECK(WB == 1 || WB == B, "w.size(0) must be 1 or B");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
{y_shape_check}
    TORCH_CHECK((int)grad_out.size(1) == V && (int)grad_out.size(2) == U, "grad_out shape mismatch");

    auto d_grad_out = torch::zeros_like(grad_out);
{dw_alloc}{dx_alloc}{dy_alloc}

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());
    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{
        launch_{bundle_name}<scalar_t>(
            (const scalar_t*)w.data_ptr<scalar_t>(),
            (const scalar_t*)x.data_ptr<scalar_t>(),
            (const scalar_t*)y.data_ptr<scalar_t>(),
            (const scalar_t*)grad_out.data_ptr<scalar_t>(),
{ggw_launch_decl}{ggx_launch_decl}{ggy_launch_decl}
            (scalar_t*)d_grad_out.data_ptr<scalar_t>(),
{dw_launch_decl}{dx_launch_decl}{dy_launch_decl}
            {src_arg}, {dst_arg}, B, WB, Iw, Ix, Ky, V, U, stream);
    }});
    GPU_KERNEL_LAUNCH_CHECK();
    {ret_expr}
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} Uniform1D double backward LARS jit impl");
}}
'''


def generate_code_uniform1d_double_bwd_with_scheduler(
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
    need_grad_x: bool = True,
    need_grad_y: bool = True,
    out_path: str = "generated_uniform1d_double_bwd_lars.cu",
    kernel_name: str = "uniform1d_double_bwd_lars",
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    block_size: int = 32,
) -> Any:
    """Generate the fused fully-unrolled LARS Uniform1D double backward."""
    ctx = _prepare_uniform1d_bwd_codegen_context(
        i_list=i_list, j_list=j_list, k_list=k_list, v_list=v_list,
        coeff_list=coeff_list, input_indices=input_indices,
        output_indices=output_indices, mode=mode,
        path_semantics=path_semantics,
    )
    cand_name = "lars_double_bwd_all_inputs"
    scheduler = LARSUniform1DDoubleBwdScheduler(
        paths=ctx.bwd_paths,
        need_grad_w=bool(need_grad_w),
        need_grad_x=bool(need_grad_x),
        need_grad_y=bool(need_grad_y),
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name=cand_name,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    schedule_result = scheduler.schedule()
    _dump_and_print_schedule_label_stats(
        operation="uniform1d_double_bwd",
        candidate_tag=cand_name,
        path_order=schedule_result.path_order,
        labels_by_path=_stats_labels_uniform1d_double_bwd(
            scheduler.paths,
            need_grad_w=bool(need_grad_w),
            need_grad_x=bool(need_grad_x),
            need_grad_y=bool(need_grad_y),
        ),
    )

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    grad_tag = f"gw{int(bool(need_grad_w))}_gx{int(bool(need_grad_x))}_gy{int(bool(need_grad_y))}"
    base_kernel_name = (
        f"{kernel_name}_{cand_name}_u{int(u_dim)}_path{ctx.P}_"
        f"{mode_str}_{ctx.layout_tag}_{grad_tag}"
    )
    code = emit_uniform1d_double_bwd_kernel_from_lars_schedule(
        schedule_result,
        kernel_name=base_kernel_name,
        mode=mode,
        need_grad_w=bool(need_grad_w),
        need_grad_x=bool(need_grad_x),
        need_grad_y=bool(need_grad_y),
        use_x_src=ctx.use_x_src,
        use_y_src=ctx.use_y_src,
        use_scatter=ctx.use_scatter,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        block_size=int(block_size),
    )
    code += "\n" + emit_uniform1d_double_bwd_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        need_grad_w=bool(need_grad_w),
        need_grad_x=bool(need_grad_x),
        need_grad_y=bool(need_grad_y),
        use_x_src=ctx.use_x_src,
        use_y_src=ctx.use_y_src,
        use_scatter=ctx.use_scatter,
    )
    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")
    if return_schedule:
        return {
            "name": cand_name,
            "code": code,
            "kernel_name": base_kernel_name,
            "schedule": schedule_result,
            "profile": schedule_result.profile,
        }
    return code
