"""FastEq scheduling: stc backward."""
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
    _fmt_float,
    _max_reg_count_any,
    _parse_stc_placement_ref,
    _sanitize_cuda_comment,
)
from .metadata import (
    _check_stc_path_bounds_for_bwd,
    _normalize_stc_padded_paths,
    infer_stc_path_lens_from_padded,
    make_stc_paths_from_padded_lists,
)
from .models import (
    Inst,
    LARSPlacementConfig,
    STCPath,
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
    _stats_labels_stc_bwd,
)
from .schedulers import (
    LARSUniform1DScheduler,
)


def _stc_x1_global_expr(x1_idx: int) -> str:
    return f"x1[((index_t)b * (index_t)X1 + (index_t){int(x1_idx)}) * (index_t)U + (index_t)u]"


def _stc_grad_x1_acc_name(x1_idx: int) -> str:
    return f"gx1_acc_a_{int(x1_idx)}"


def _stc_grad_x0_acc_name(x0_idx: int) -> str:
    return f"gx0_acc_d_{int(x0_idx)}"


def _stc_bwd_grad_expr(base_expr: str, other_x1_indices: Sequence[int]) -> str:
    expr = base_expr
    for x1_idx in other_x1_indices:
        expr = f"({expr} * {_stc_x1_global_expr(int(x1_idx))})"
    return expr


def _make_stc_bwd_logical_schedule(
    paths: Sequence[STCPath],
    base_schedule: ScheduleResult,
    *,
    need_grad_x0: bool = True,
) -> ScheduleResult:
    """Convert the STC LARS path order into backward placement records."""
    by_pid = {int(p.pid): p for p in paths}
    instructions: List[Inst] = []
    for pid in base_schedule.path_order:
        p = by_pid[int(pid)]
        instructions.append(
            Inst(
                "stc_bwd_path",
                (
                    tuple(int(v) for v in p.x1_indices),
                    int(p.x0_index),
                    int(p.v),
                    float(p.c),
                    bool(need_grad_x0),
                ),
                f"path#{p.pid}: STC grad_x1",
            )
        )
    return ScheduleResult(
        instructions=instructions,
        path_order=list(base_schedule.path_order),
        max_live=int(base_schedule.max_live),
        spills=int(base_schedule.spills),
        reloads=int(base_schedule.reloads),
        final_reg_map={},
        profile=dict(base_schedule.profile),
    )


def emit_stc_bwd_kernel_from_paths(
    paths: Sequence[STCPath],
    *,
    kernel_name: str,
    path_order: Optional[Sequence[int]] = None,
    schedule_result: Optional[ScheduleResult] = None,
    u_dim: Optional[int] = None,
    x0_dim: Optional[int] = None,
    x1_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    tile_u: int = 32,
    block_size: int = 32,
    need_grad_x0: bool = True,
) -> str:
    """Emit STC backward for x1 and, when requested, gathered x0.

    When ``schedule_result`` is supplied, it may contain ``stc_bwd_placed``
    instructions produced by :func:`apply_lars_lifetime_placement`.  Without a
    placed schedule, the function preserves the previous all-accumulator-local
    implementation.
    """
    if int(tile_u) != 32:
        raise ValueError(f"STC backward currently assumes tile_u=32, got {tile_u}")
    block_size = int(block_size)
    if block_size < 32 or block_size % 32 != 0:
        raise ValueError("block_size must be a positive multiple of 32")

    path_list = list(paths)
    by_pid = {int(p.pid): p for p in path_list}
    if schedule_result is None:
        order = list(path_order) if path_order is not None else [int(p.pid) for p in path_list]
        ordered_paths = [by_pid[int(pid)] for pid in order]
        legacy_insts = [
            Inst(
                "stc_bwd_path",
                (
                    tuple(int(v) for v in p.x1_indices),
                    int(p.x0_index),
                    int(p.v),
                    float(p.c),
                    bool(need_grad_x0),
                ),
                f"path#{p.pid}: STC grad_x1",
            )
            for p in ordered_paths
        ]
        schedule_result = ScheduleResult(
            instructions=legacy_insts,
            path_order=order,
            max_live=0,
            spills=0,
            reloads=0,
            final_reg_map={},
        )
    else:
        ordered_paths = [by_pid[int(pid)] for pid in schedule_result.path_order]

    _check_stc_path_bounds_for_bwd(
        ordered_paths, x0_dim=x0_dim, x1_dim=x1_dim, v_dim=v_dim
    )

    has_placed = any(
        inst.op == "stc_bwd_placed" for inst in schedule_result.instructions
    )
    reg_count = _max_reg_count_any(schedule_result)
    shared_slots = int(getattr(schedule_result, "shared_slots", 0) or 0)
    touched_x1_indices = sorted({
        int(idx) for p in ordered_paths for idx in p.x1_indices
    })
    touched_x0_indices = (
        sorted({int(p.x0_index) for p in ordered_paths})
        if need_grad_x0 else []
    )

    lines: List[str] = []

    def ap(line: str = "") -> None:
        lines.append(line)

    def shared_expr(token: str) -> str:
        if not isinstance(token, str) or not token.startswith("s"):
            raise ValueError(f"Bad STC backward shared token: {token!r}")
        slot = int(token[1:])
        return (
            f"lars_smem[(size_t){slot} * (size_t)blockDim.x + "
            f"(size_t)tid]"
        )

    def input_index_expr(kind: str, idx: int) -> str:
        if kind == "x1":
            return (
                f"((index_t)b * (index_t)X1 + (index_t){int(idx)}) * "
                f"(index_t)U + (index_t)u"
            )
        if kind == "x0":
            return (
                f"((index_t)b * (index_t)X0 + (index_t){int(idx)}) * "
                f"(index_t)U + (index_t)u"
            )
        if kind == "go":
            return (
                f"((index_t)b * (index_t)V + (index_t){int(idx)}) * "
                f"(index_t)U + (index_t)u"
            )
        raise ValueError(f"Bad STC backward input kind: {kind}")

    def direct_input_expr(ref: str) -> str:
        kind, idx = _parse_stc_placement_ref(ref)
        if kind not in ("x1", "x0", "go"):
            raise ValueError(f"Bad STC backward direct input ref: {ref!r}")
        arr = "grad_out" if kind == "go" else kind
        return f"{arr}[{input_index_expr(kind, idx)}]"

    def operand_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        if token.startswith("d:"):
            return direct_input_expr(token[2:])
        raise ValueError(f"Bad STC backward operand token: {token!r}")

    def acc_expr(token: str) -> str:
        token = str(token)
        if token.startswith("r"):
            return token
        if token.startswith("s"):
            return shared_expr(token)
        raise ValueError(f"Bad STC backward accumulator token: {token!r}")

    def grad_x1_index_expr(idx: int) -> str:
        return (
            f"((index_t)b * (index_t)X1 + (index_t){int(idx)}) * "
            f"(index_t)U + (index_t)u"
        )

    def grad_x0_index_expr(idx: int) -> str:
        return (
            f"((index_t)b * (index_t)X0 + (index_t){int(idx)}) * "
            f"(index_t)U + (index_t)u"
        )

    def emit_grad_x1_write(idx: int, value_expr: str) -> None:
        ap(f"            grad_x1[{grad_x1_index_expr(idx)}] = {value_expr};")

    def emit_grad_x0_write(idx: int, value_expr: str) -> None:
        ap(f"            grad_x0[{grad_x0_index_expr(idx)}] = {value_expr};")

    def product_from_tokens(base_expr: str, tokens: Sequence[str]) -> str:
        expr = str(base_expr)
        for token in tokens:
            expr = f"({expr} * {operand_expr(str(token))})"
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
    ap("    scalar_t* __restrict__ grad_x1,")
    ap("    scalar_t* __restrict__ grad_x0,")
    ap("    int B, int X1, int X0, int V, int U)")
    ap("{")
    ap("    const int b = (int)blockIdx.x;")
    ap("    if (b >= B) return;")
    ap("")
    ap("    const int tid = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    const int warp_id = tid >> 5;")
    ap("    const int warp_count = blockDim.x >> 5;")
    if shared_slots:
        ap("    extern __shared__ unsigned char lars_smem_raw[];")
        ap("    scalar_t* lars_smem = reinterpret_cast<scalar_t*>(lars_smem_raw);")
    ap("")
    ap("    for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    if not has_placed:
        for x1_idx in touched_x1_indices:
            ap(f"            scalar_t {_stc_grad_x1_acc_name(x1_idx)} = scalar_t(0);")
        for x0_idx in touched_x0_indices:
            ap(f"            scalar_t {_stc_grad_x0_acc_name(x0_idx)} = scalar_t(0);")
        if touched_x1_indices or touched_x0_indices:
            ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // bwd inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op in ("load", "load_shared"):
            token, ref = inst.args
            kind, idx = _parse_stc_placement_ref(str(ref))
            if kind not in ("x1", "x0", "go"):
                raise ValueError("STC backward input load expects x1/x0/grad_out")
            arr = "grad_out" if kind == "go" else kind
            lhs = str(token) if inst.op == "load" else shared_expr(str(token))
            ap(f"            {lhs} = {arr}[{input_index_expr(kind, idx)}];")

        elif inst.op == "init_acc":
            token, ref = inst.args
            kind, _idx = _parse_stc_placement_ref(str(ref))
            if kind not in ("gx1", "gx0"):
                raise ValueError("STC backward init_acc expects grad_x1[]/grad_x0[]")
            ap(f"            {acc_expr(str(token))} = scalar_t(0);")

        elif inst.op == "stc_bwd_placed":
            (
                x1_indices,
                target_tokens,
                x0_target_token,
                go_token,
                x0_token,
                x1_tokens,
                coeff,
            ) = inst.args
            x1_indices = tuple(int(v) for v in x1_indices)
            target_tokens = tuple(str(v) for v in target_tokens)
            x1_tokens = tuple(str(v) for v in x1_tokens)
            base_expr = (
                f"scalar_t({_fmt_float(float(coeff))}) * "
                f"{operand_expr(str(go_token))} * {operand_expr(str(x0_token))}"
            )
            for pos, target_idx in enumerate(x1_indices):
                other_tokens = [
                    x1_tokens[q]
                    for q in range(len(x1_indices))
                    if q != pos
                ]
                value = product_from_tokens(base_expr, other_tokens)
                target_token = target_tokens[pos]
                if target_token.startswith("d:"):
                    emit_grad_x1_write(int(target_idx), value)
                else:
                    ap(f"            {acc_expr(target_token)} += {value};")
            if need_grad_x0:
                gx0_value = product_from_tokens(
                    f"scalar_t({_fmt_float(float(coeff))}) * {operand_expr(str(go_token))}",
                    x1_tokens,
                )
                if str(x0_target_token).startswith("d:"):
                    emit_grad_x0_write(int(_parse_stc_placement_ref(str(x0_target_token)[2:])[1]), gx0_value)
                else:
                    ap(f"            {acc_expr(str(x0_target_token))} += {gx0_value};")

        elif inst.op == "store_acc_placed":
            ref, token = inst.args
            kind, idx = _parse_stc_placement_ref(str(ref))
            if kind == "gx1":
                emit_grad_x1_write(int(idx), acc_expr(str(token)))
            elif kind == "gx0":
                emit_grad_x0_write(int(idx), acc_expr(str(token)))
            else:
                raise ValueError("STC backward store_acc_placed expects grad_x1[]/grad_x0[]")

        elif inst.op == "stc_bwd_path":
            x1_indices, x0_idx, out_v, coeff, path_need_grad_x0 = inst.args
            x1_indices = tuple(int(v) for v in x1_indices)
            base_name = f"base_{inst_id}"
            ap(
                f"            const scalar_t {base_name} = "
                f"grad_out[{input_index_expr('go', int(out_v))}] * "
                f"scalar_t({_fmt_float(float(coeff))}) * "
                f"x0[{input_index_expr('x0', int(x0_idx))}];"
            )
            for pos, target_x1 in enumerate(x1_indices):
                other = [idx for q, idx in enumerate(x1_indices) if q != pos]
                grad_expr = _stc_bwd_grad_expr(base_name, other)
                ap(
                    f"            {_stc_grad_x1_acc_name(target_x1)} += "
                    f"{grad_expr};"
                )
            if bool(path_need_grad_x0):
                gx0_expr = _stc_bwd_grad_expr(
                    f"grad_out[{input_index_expr('go', int(out_v))}] * scalar_t({_fmt_float(float(coeff))})",
                    x1_indices,
                )
                ap(
                    f"            {_stc_grad_x0_acc_name(int(x0_idx))} += "
                    f"{gx0_expr};"
                )

        elif inst.op in ("release", "release_shared"):
            pass
        else:
            raise ValueError(f"Unsupported STC backward instruction op: {inst.op}")

    if not has_placed:
        for x1_idx in touched_x1_indices:
            emit_grad_x1_write(
                int(x1_idx), _stc_grad_x1_acc_name(int(x1_idx))
            )
        for x0_idx in touched_x0_indices:
            emit_grad_x0_write(
                int(x0_idx), _stc_grad_x0_acc_name(int(x0_idx))
            )

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
    ap("    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, scalar_t* grad_x1, scalar_t* grad_x0,")
    ap("    int B, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(
        f"    size_t lars_shared_bytes = (size_t){shared_slots} * "
        f"(size_t)block.x * sizeof(scalar_t);"
    )
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, lars_shared_bytes, stream>>>(")
    ap("        grad_out, x1, x0, grad_x1, grad_x0, B, X1, X0, V, U);")
    ap("}")
    ap("")
    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* grad_out, const scalar_t* x1, const scalar_t* x0, scalar_t* grad_x1, scalar_t* grad_x0,")
    ap("    int B, int X1, int X0, int V, int U, gpuStream_t stream)")
    ap("{")
    ap("    bool use_i32 = mul3_fits_int32((int64_t)B, (int64_t)X1, (int64_t)U) &&")
    ap("                   mul3_fits_int32((int64_t)B, (int64_t)X0, (int64_t)U) &&")
    ap("                   mul3_fits_int32((int64_t)B, (int64_t)V,  (int64_t)U);")
    ap("    if (use_i32) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(grad_out, x1, x0, grad_x1, grad_x0, B, X1, X0, V, U, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(grad_out, x1, x0, grad_x1, grad_x0, B, X1, X0, V, U, stream);")
    ap("    }")
    ap("}")
    ap("")
    ap(f"std::vector<torch::Tensor> launcher_{kernel_name}(torch::Tensor grad_out, torch::Tensor x1, torch::Tensor x0, int64_t V64) {{")
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
    if need_grad_x0:
        ap("    auto grad_x0 = torch::zeros_like(x0);")
    ap("    GPU_Guard device_guard(x1.device());")
    ap("    gpuStream_t stream = getCurrentGPUStream(x1.device().index());")
    ap(f"    AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), \"{kernel_name}\", [&] {{")
    ap(f"        launch_{kernel_name}<scalar_t>((const scalar_t*)grad_out.data_ptr<scalar_t>(),")
    ap("            (const scalar_t*)x1.data_ptr<scalar_t>(),")
    ap("            (const scalar_t*)x0.data_ptr<scalar_t>(),")
    ap("            (scalar_t*)grad_x1.data_ptr<scalar_t>(),")
    if need_grad_x0:
        ap("            (scalar_t*)grad_x0.data_ptr<scalar_t>(), B, X1, X0, V, U, stream);")
    else:
        ap("            nullptr, B, X1, X0, V, U, stream);")
    ap("    });")
    ap("    GPU_KERNEL_LAUNCH_CHECK();")
    if need_grad_x0:
        ap("    return {grad_x1, grad_x0};")
    else:
        ap("    return {grad_x1};")
    ap("}")
    ap("")
    ap("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
    ap(f"    m.def(\"run\", &launcher_{kernel_name}, \"{kernel_name} STC backward x1/x0 jit impl\");")
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
    block_size: int = 32,
    placement_config: Optional[Union[LARSPlacementConfig, Dict[str, Any]]] = None,
    need_grad_x0: bool = True,
) -> Any:
    """Generate STC backward for x1 and optionally gathered x0."""
    paths = make_stc_paths_from_padded_lists(
        idx_lists, coeff_list, path_lens=path_lens, pad_value=pad_value
    )
    scheduler = LARSUniform1DScheduler(
        paths,
        path_kind="stc",
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_name=("stc_lars_bwd_x1_x0" if need_grad_x0 else "stc_lars_bwd_x1_only"),
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    path_schedule = scheduler.schedule()
    _dump_and_print_schedule_label_stats(
        operation="stc_bwd",
        candidate_tag=("stc_lars_bwd_x1_x0" if need_grad_x0 else "stc_lars_bwd_x1_only"),
        path_order=path_schedule.path_order,
        labels_by_path=_stats_labels_stc_bwd(paths, need_grad_x0=need_grad_x0),
    )
    schedule_result = _make_stc_bwd_logical_schedule(
        paths, path_schedule, need_grad_x0=need_grad_x0
    )
    resolved_placement = _resolve_lars_placement_config(placement_config)
    schedule_result = apply_lars_lifetime_placement(
        schedule_result,
        schedule_kind="stc_bwd",
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

    code = emit_stc_bwd_kernel_from_paths(
        paths,
        schedule_result=schedule_result,
        kernel_name=base_kernel_name,
        u_dim=int(u_dim),
        x0_dim=x0_dim,
        x1_dim=x1_dim,
        v_dim=int(num_out_segments),
        tile_u=int(tile_u),
        block_size=int(block_size),
        need_grad_x0=bool(need_grad_x0),
    )
    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")
    if return_schedule:
        return {
            "name": ("stc_lars_bwd_x1_x0" if need_grad_x0 else "stc_lars_bwd_x1_only"),
            "code": code,
            "kernel_name": base_kernel_name,
            "schedule": schedule_result,
            "num_paths": len(paths),
            "max_live": schedule_result.max_live,
            "profile": schedule_result.profile,
        }
    return code
