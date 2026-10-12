"""FastEq scheduling: stc forward."""
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
    _index_expr,
    _max_reg_count_any,
    _parse_stc_placement_ref,
    _product_expr,
    _sanitize_cuda_comment,
)
from .metadata import (
    _normalize_stc_padded_paths,
    infer_stc_path_lens_from_padded,
    make_stc_paths_from_padded_lists,
)
from .models import (
    LARSPlacementConfig,
    STC_PAD_VALUE,
    ScheduleResult,
)
from .placement import (
    _resolve_lars_placement_config,
    apply_lars_lifetime_placement,
    dump_lars_variable_stats,
)
from .schedule_stats import (
    _dump_and_print_schedule_label_stats,
    _stats_labels_stc_fwd,
)
from .schedulers import (
    LARSUniform1DScheduler,
)


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
    """Emit STC forward code from either a legacy or lifetime-placed schedule."""
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")

    reg_count = _max_reg_count_any(schedule_result)
    shared_slots = int(getattr(schedule_result, "shared_slots", 0) or 0)
    has_placed = any(
        inst.op == "mul_stc_placed" for inst in schedule_result.instructions
    )

    resident_out_indices = [] if has_placed else sorted({
        int(inst.args[0])
        for inst in schedule_result.instructions
        if inst.op == "mul_stc_resident"
    })
    direct_out_indices = [] if has_placed else sorted({
        int(inst.args[0])
        for inst in schedule_result.instructions
        if inst.op == "mul_stc_direct"
    })

    lines: List[str] = []

    def ap(line: str = "") -> None:
        lines.append(line)

    def shared_expr(token: str) -> str:
        if not isinstance(token, str) or not token.startswith("s"):
            raise ValueError(f"Bad STC shared token: {token!r}")
        slot = int(token[1:])
        return (
            f"lars_smem[(size_t){slot} * (size_t)blockDim.x + "
            f"(size_t)tid]"
        )

    def direct_input_expr(ref: str) -> str:
        kind, idx = _parse_stc_placement_ref(ref)
        if kind not in ("x0", "x1"):
            raise ValueError(f"STC forward direct input expects x0/x1, got {ref!r}")
        expr = _index_expr(
            kind, idx,
            u_dim=u_dim,
            x0_dim=x0_dim,
            x1_dim=x1_dim,
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
        raise ValueError(f"Bad STC forward operand token: {token!r}")

    def acc_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        raise ValueError(f"Bad STC forward accumulator token: {token!r}")

    def emit_out_write(idx: int, value_expr: str) -> None:
        expr = _index_expr(
            "o", int(idx),
            u_dim=u_dim,
            x0_dim=x0_dim,
            x1_dim=x1_dim,
            v_dim=v_dim,
        )
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
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    if shared_slots:
        ap("    extern __shared__ unsigned char lars_smem_raw[];")
        ap("    scalar_t* lars_smem = reinterpret_cast<scalar_t*>(lars_smem_raw);")
    ap("    const index_t x1_base = (index_t)b * (index_t)X1 * (index_t)U;")
    ap("    const index_t x0_base = (index_t)b * (index_t)X0 * (index_t)U;")
    ap("    const index_t out_base = (index_t)b * (index_t)V * (index_t)U;")
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
        ap(
            f"            // direct single-use output writeback enabled for "
            f"{len(direct_out_indices)} out accumulator(s)"
        )
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op in ("load", "load_shared"):
            token, ref = inst.args
            kind, idx = _parse_stc_placement_ref(str(ref))
            if kind not in ("x0", "x1"):
                raise ValueError("STC forward input load must target x0/x1")
            expr = _index_expr(
                kind, idx,
                u_dim=u_dim,
                x0_dim=x0_dim,
                x1_dim=x1_dim,
                v_dim=v_dim,
            )
            lhs = str(token) if inst.op == "load" else shared_expr(str(token))
            ap(f"            {lhs} = {kind}[{expr}];")

        elif inst.op == "init_acc":
            token, ref = inst.args
            kind, _idx = _parse_stc_placement_ref(str(ref))
            if kind != "o":
                raise ValueError("STC forward init_acc expects out[]")
            ap(f"            {acc_expr(str(token))} = scalar_t(0);")

        elif inst.op == "mul_stc_placed":
            out_idx, out_token, operand_tokens, coeff = inst.args
            product = _product_expr(
                tuple(operand_expr(str(tok)) for tok in operand_tokens),
                float(coeff),
            )
            if str(out_token).startswith("d:"):
                emit_out_write(int(out_idx), product)
            else:
                ap(f"            {acc_expr(str(out_token))} += {product};")

        elif inst.op == "store_acc_placed":
            ref, token = inst.args
            kind, idx = _parse_stc_placement_ref(str(ref))
            if kind != "o":
                raise ValueError("STC forward store_acc_placed expects out[]")
            emit_out_write(int(idx), acc_expr(str(token)))

        elif inst.op == "mul_stc_resident":
            out_idx, regs, coeff = inst.args
            ap(
                f"            out_acc_v_{int(out_idx)} += "
                f"{_product_expr(tuple(regs), float(coeff))};"
            )

        elif inst.op == "mul_stc_direct":
            out_idx, regs, coeff = inst.args
            emit_out_write(
                int(out_idx), _product_expr(tuple(regs), float(coeff))
            )

        elif inst.op in ("release", "release_shared"):
            pass
        else:
            raise ValueError(f"Unsupported STC forward instruction op: {inst.op}")

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
    ap(
        f"    size_t lars_shared_bytes = (size_t){shared_slots} * "
        f"(size_t)block.x * sizeof(scalar_t);"
    )
    ap(
        f"    {kernel_name}<scalar_t, index_t><<<grid, block, "
        f"lars_shared_bytes, stream>>>(x1, x0, out, B, X1, X0, V, U);"
    )
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
    block_size: int = 32,
    placement_config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]] = None,
) -> Any:
    paths = make_stc_paths_from_padded_lists(
        idx_lists, coeff_list, path_lens=path_lens, pad_value=pad_value
    )
    scheduler = LARSUniform1DScheduler(
        paths,
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
    _dump_and_print_schedule_label_stats(
        operation="stc_fwd",
        candidate_tag="stc_lars_all_inputs",
        path_order=schedule_result.path_order,
        labels_by_path=_stats_labels_stc_fwd(paths),
    )
    resolved_placement = _resolve_lars_placement_config(placement_config)
    schedule_result = apply_lars_lifetime_placement(
        schedule_result,
        schedule_kind="stc_fwd",
        config=resolved_placement,
    )
    dump_lars_variable_stats(
        schedule_result,
        print_stats=resolved_placement.stats_print,
        out_path=resolved_placement.stats_path,
    )

    idx_norm = _normalize_stc_padded_paths(
        idx_lists, coeff_list=coeff_list, path_lens=path_lens
    )
    if path_lens is None:
        inferred_lens = infer_stc_path_lens_from_padded(
            idx_norm, coeff_list, pad_value=pad_value
        )
    else:
        inferred_lens = path_lens.detach().cpu().to(torch.int64).reshape(-1)
    base_kernel_name = (
        f"{kernel_name}_u{int(u_dim)}_path{len(paths)}_"
        f"maxlen{int(inferred_lens.max().item())}"
    )
    code = emit_stc_fwd_kernel_from_lars_schedule(
        schedule_result,
        kernel_name=base_kernel_name,
        u_dim=int(u_dim),
        x0_dim=x0_dim,
        x1_dim=x1_dim,
        v_dim=int(num_out_segments),
        block_size=int(block_size),
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
            "profile": schedule_result.profile,
        }
    return code
