"""FastEq scheduling: uniform1d forward."""
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
    _lars_label_index_expr,
    _max_lars_reg_count_any,
    _parse_lars_label_ref,
    _sanitize_cuda_comment,
    _to_float_list,
    _to_int_list,
)
from .metadata import (
    _make_lars_paths_from_uniform1d_lists,
)
from .models import (
    LARSPlacementConfig,
    ScheduleResult,
)
from .placement import (
    _resolve_lars_placement_config,
    apply_lars_lifetime_placement,
    dump_lars_variable_stats,
)
from .schedule_stats import (
    _dump_and_print_schedule_label_stats,
    _stats_labels_uniform1d_fwd,
)
from .schedulers import (
    LARSUniform1DScheduler,
)


def _dense_forward_writeback_enabled() -> bool:
    """Codegen-time switch; rebuild/restart to compare generated variants."""
    return os.environ.get("FASTEQ_DENSE_FWD_WRITEBACK", "1").strip().lower() not in (
        "0", "false", "off", "no"
    )


def emit_launcher(
    bundle_name: str,
    mode: str = "u,u,,u",
    *,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
    schedule_result: Optional[ScheduleResult] = None,
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

    if _dense_forward_writeback_enabled() and schedule_result is not None and not (use_x_src or use_y_src or use_scatter):
        summary = schedule_result.placement_summary
        count = summary.get("fully_written_segment_counts", {}).get("out")
        if summary.get("exclusive_outputs") and count is not None:
            out_alloc = (
                f"    auto out = (V == {int(count)}) ? torch::empty({{B, V, U}}, w.options())"
                " : torch::zeros({B, V, U}, w.options());"
            )

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
    #   B comes from x_all.size(0); w may be shared across all B rows.
    # ------------------------------------------------------------------
    if use_scatter:
        b_expr = "dst_idx.size(0)"
    elif use_x_src or use_y_src:
        b_expr = "src_idx.size(0)"
    else:
        b_expr = "x_all.size(0)"

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
    """Emit forward code from either a legacy or lifetime-placed schedule."""
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    reg_count = _max_lars_reg_count_any(schedule_result)
    shared_slots = int(getattr(schedule_result, "shared_slots", 0) or 0)
    exclusive = bool(schedule_result.placement_summary.get("exclusive_outputs")) and not (
        use_x_src or use_y_src or use_scatter
    )
    has_placed = any(
        inst.op == "fma_u1d_placed" for inst in schedule_result.instructions
    )

    resident_out_indices = [] if has_placed else sorted({
        int(inst.args[0])
        for inst in schedule_result.instructions
        if inst.op == "fma_u1d_resident"
    })
    direct_out_indices = [] if has_placed else sorted({
        int(inst.args[0])
        for inst in schedule_result.instructions
        if inst.op == "fma_u1d_direct"
    })

    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    def shared_expr(token: str) -> str:
        if not isinstance(token, str) or not token.startswith("s"):
            raise ValueError(f"Bad shared token: {token!r}")
        slot = int(token[1:])
        return f"lars_smem[(size_t){slot} * (size_t)blockDim.x + (size_t)tid]"

    def direct_input_expr(ref: str) -> str:
        kind, idx = _parse_lars_label_ref(ref)
        if kind == "o":
            raise ValueError("output cannot be used as a direct input")
        expr = _lars_label_index_expr(
            kind, idx,
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            w_dim=w_dim,
            v_dim=v_dim,
        )
        return f"{kind}[{expr}]"

    def operand_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        if token.startswith("d:"):
            return direct_input_expr(token[2:])
        raise ValueError(f"Bad forward operand token: {token!r}")

    def acc_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        raise ValueError(f"Bad forward accumulator token: {token!r}")

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
        elif exclusive:
            ap(f"            out[{expr}] = {value_expr};")
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
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    if shared_slots:
        ap("    extern __shared__ unsigned char lars_smem_raw[];")
        ap("    scalar_t* lars_smem = reinterpret_cast<scalar_t*>(lars_smem_raw);")
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
    ap("    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;")
    ap("    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;")
    if mode_scalar_y:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky;")
    else:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky * (index_t)U;")
    ap("    const index_t out_base = (index_t)out_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
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

        if inst.op in ("load", "load_shared"):
            token, ref = inst.args
            kind, idx = _parse_lars_label_ref(ref)
            if kind == "o":
                raise ValueError("input load must not target output labels")
            expr = _lars_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                w_dim=w_dim,
                v_dim=v_dim,
            )
            lhs = str(token) if inst.op == "load" else shared_expr(str(token))
            ap(f"            {lhs} = {kind}[{expr}];")

        elif inst.op == "init_acc":
            token, ref = inst.args
            kind, _idx = _parse_lars_label_ref(ref)
            if kind != "o":
                raise ValueError("forward init_acc expects out[]")
            ap(f"            {acc_expr(str(token))} = scalar_t(0);")

        elif inst.op == "fma_u1d_placed":
            out_idx, out_token, rx, ry, rw, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            value = (
                f"scalar_t({c}) * ({operand_expr(str(rw))} * "
                f"{operand_expr(str(rx))}) * {operand_expr(str(ry))}"
            )
            if str(out_token).startswith("d:"):
                emit_out_write(int(out_idx), value)
            else:
                ap(f"            {acc_expr(str(out_token))} += {value};")

        elif inst.op == "store_acc_placed":
            ref, token = inst.args
            kind, idx = _parse_lars_label_ref(ref)
            if kind != "o":
                raise ValueError("forward store_acc_placed expects out[]")
            emit_out_write(int(idx), acc_expr(str(token)))

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
            emit_out_write(int(out_idx), f"scalar_t({c}) * ({rw} * {rx}) * {ry}")

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

        elif inst.op in ("release", "release_shared"):
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
    ap(f"    size_t lars_shared_bytes = (size_t){shared_slots} * (size_t)block.x * sizeof(scalar_t);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, lars_shared_bytes, stream>>>(")
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
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    block_size: int = 32,
    placement_config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]] = None,
):
    """
    Generate one LARS forward CUDA implementation.

    API compatibility, but this slim scheduler derives an unbounded virtual
    input-register file from all distinct x/y/w labels and emits a single
    single candidate.
    """
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
        operation="uniform1d_fwd",
        candidate_tag=cand_name,
        path_order=schedule_result.path_order,
        labels_by_path=_stats_labels_uniform1d_fwd(scheduler.paths),
    )
    resolved_placement = _resolve_lars_placement_config(placement_config)
    schedule_result = apply_lars_lifetime_placement(
        schedule_result,
        schedule_kind="fwd",
        config=resolved_placement,
        exclusive_outputs=_dense_forward_writeback_enabled() and not (input_indices or output_indices),
    )
    dump_lars_variable_stats(
        schedule_result,
        print_stats=resolved_placement.stats_print,
        out_path=resolved_placement.stats_path,
    )

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
        block_size=int(block_size),
    )

    code = code + "\n" + emit_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        schedule_result=schedule_result,
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
                "auto_fallback_when_full": False,
            },
            "profile": schedule_result.profile,
        }

    return code
