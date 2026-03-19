from __future__ import annotations

from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
import math
from pathlib import Path
import struct
import numpy as np
import torch


def find_fasteq_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if p.name == "fasteq":
            return p
    raise RuntimeError("Cannot find fasteq project root from __file__")

fasteq_root = find_fasteq_root(Path(__file__).parent)
out_dir = fasteq_root / "cuda" / "src" / "uniform1d_codegen"
out_dir.mkdir(parents=True, exist_ok=True)


def stable_unique(xs: List[int]) -> List[int]:
    out = []
    seen = set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def chunk_list(xs: List[int], chunk_size: int) -> List[List[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def normalize_int_seq(xs) -> List[int]:
    if hasattr(xs, "detach") and hasattr(xs, "cpu") and hasattr(xs, "tolist"):
        return [int(x) for x in xs.detach().cpu().tolist()]
    out = []
    for x in xs:
        if hasattr(x, "item"):
            out.append(int(x.item()))
        else:
            out.append(int(x))
    return out


def normalize_float_seq(xs) -> List[float]:
    if hasattr(xs, "detach") and hasattr(xs, "cpu") and hasattr(xs, "tolist"):
        return [float(x) for x in xs.detach().cpu().tolist()]
    out = []
    for x in xs:
        if hasattr(x, "item"):
            out.append(float(x.item()))
        else:
            out.append(float(x))
    return out


def build_backward_schedule_from_lists(
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
    U_dim: int,
    *,
    candidate_block_sizes: Tuple[int, ...] = (32, 64, 128),
    enable_persistent_u: bool = True,
    prefer_warp_specialized: bool = True,
    reg_budget: int = 96,
    smem_budget_bytes: int = 4096,
    max_threads_per_block: int = 1024,
    scalar_nbytes: int = 8,   # FP64 default
    v_tile_size: Optional[int] = None,
    i_tile_size: Optional[int] = None,
    j_tile_size: Optional[int] = None,
    k_tile_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build backward schedule directly from static path lists.

    Backward:
      grad_w[b,i,u] += c * grad_out[dst,v,u] * x[src,j,u] * y[b,k]
      grad_x[src,j,u] += c * grad_out[dst,v,u] * w[b,i,u] * y[b,k]
      grad_y[b,k] += sum_u c * grad_out[dst,v,u] * w[b,i,u] * x[src,j,u]
    """
    i_list = normalize_int_seq(i_list)
    j_list = normalize_int_seq(j_list)
    k_list = normalize_int_seq(k_list)
    v_list = normalize_int_seq(v_list)
    coeff_list = normalize_float_seq(coeff_list)

    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff lists must have the same length")

    P = len(i_list)
    if P == 0:
        raise ValueError("Empty path lists")

    if U_dim <= 0:
        raise ValueError("U_dim must be > 0")

    # Build reverse contribution graphs
    gw = defaultdict(list)  # i -> [(v,j,k,c)]
    gx = defaultdict(list)  # j -> [(v,i,k,c)]
    gy = defaultdict(list)  # k -> [(v,i,j,c)]

    for i, j, k, v, c in zip(i_list, j_list, k_list, v_list, coeff_list):
        gw[i].append((v, j, k, c))
        gx[j].append((v, i, k, c))
        gy[k].append((v, i, j, c))

    uniq_i = stable_unique(i_list)
    uniq_j = stable_unique(j_list)
    uniq_k = stable_unique(k_list)
    uniq_v = stable_unique(v_list)

    # Reorder i/j/k
    ordered_i = sorted(uniq_i, key=lambda i: (-len(gw[i]), i))
    ordered_j = sorted(uniq_j, key=lambda j: (-len(gx[j]), j))
    ordered_k = sorted(uniq_k, key=lambda k: (-len(gy[k]), k))

    # Reorder v by first-seen locality
    first_meta = {}
    for i, j, k, v in zip(i_list, j_list, k_list, v_list):
        if v not in first_meta:
            first_meta[v] = (i, j, k)

    missing_vs = [v for v in uniq_v if v not in first_meta]
    if missing_vs:
        raise RuntimeError(f"Missing first_meta entries for v values: {missing_vs[:16]}")

    ordered_v = sorted(uniq_v, key=lambda v: first_meta[v] + (v,))

    num_i = len(ordered_i)
    num_j = len(ordered_j)
    num_k = len(ordered_k)
    num_v = len(ordered_v)

    avg_terms_per_i = sum(len(gw[i]) for i in ordered_i) / max(1, num_i)
    avg_terms_per_j = sum(len(gx[j]) for j in ordered_j) / max(1, num_j)
    avg_terms_per_k = sum(len(gy[k]) for k in ordered_k) / max(1, num_k)

    max_terms_per_i = max((len(gw[i]) for i in ordered_i), default=0)
    max_terms_per_j = max((len(gx[j]) for j in ordered_j), default=0)
    max_terms_per_k = max((len(gy[k]) for k in ordered_k), default=0)

    # Tile-size heuristics
    if v_tile_size is None:
        if num_v <= 4:
            v_tile_size = num_v
        elif num_v <= 16:
            v_tile_size = 4
        else:
            v_tile_size = 8 if prefer_warp_specialized else 4

    if i_tile_size is None:
        i_tile_size = min(2 if prefer_warp_specialized else 4, max(1, num_i))
    if j_tile_size is None:
        j_tile_size = min(2 if prefer_warp_specialized else 4, max(1, num_j))
    if k_tile_size is None:
        k_tile_size = min(2 if prefer_warp_specialized else 4, max(1, num_k))

    v_tile_size = max(1, min(v_tile_size, num_v))
    i_tile_size = max(1, min(i_tile_size, num_i))
    j_tile_size = max(1, min(j_tile_size, num_j))
    k_tile_size = max(1, min(k_tile_size, num_k))

    v_tiles = chunk_list(ordered_v, v_tile_size)
    i_tiles = chunk_list(ordered_i, i_tile_size)
    j_tiles = chunk_list(ordered_j, j_tile_size)
    k_tiles = chunk_list(ordered_k, k_tile_size)

    candidates: List[Dict[str, Any]] = []

    def estimate_base_regs(bs: int) -> int:
        reg_wi = min(i_tile_size, num_i)
        reg_xj = min(j_tile_size, num_j)
        reg_yk = min(k_tile_size, num_k)
        reg_go = min(v_tile_size, num_v)
        reg_gw = min(i_tile_size, num_i)
        reg_gx = min(j_tile_size, num_j)
        reg_gy = min(k_tile_size, num_k)
        base_temp = 18 if bs == 32 else 28
        est = reg_wi + reg_xj + reg_yk + reg_go + reg_gw + reg_gx + reg_gy + base_temp
        est += int(0.15 * avg_terms_per_i)
        est += int(0.15 * avg_terms_per_k)
        return est

    for bs in candidate_block_sizes:
        if bs <= 0 or bs > max_threads_per_block:
            continue
        if bs > U_dim:
            continue

        num_warps = math.ceil(bs / 32)
        u_tile = bs

        # --------------------------------
        # Strategy A: grid.y tiles U
        # --------------------------------
        if bs == 32:
            grad_y_mode = "warp_reduce_then_atomic"
            use_shared_memory = False
            estimated_smem_bytes = 0
        else:
            grad_y_mode = "warp_reduce_then_block_merge_then_atomic"
            use_shared_memory = True
            estimated_smem_bytes = num_k * num_warps * scalar_nbytes

        estimated_regs = estimate_base_regs(bs)
        num_u_tiles = math.ceil(U_dim / u_tile)
        estimated_redg_like_ops = num_k * num_u_tiles
        estimated_shfl_ops = 5 * min(k_tile_size, num_k)

        reg_penalty = max(0, estimated_regs - reg_budget) * 20.0
        smem_penalty = max(0, estimated_smem_bytes - smem_budget_bytes) * 0.02
        occupancy_penalty = 0.0
        if bs == 64:
            occupancy_penalty += 12.0
        elif bs >= 128:
            occupancy_penalty += 35.0
        strategy_bonus = -35.0 if (prefer_warp_specialized and bs == 32) else 0.0

        score = (
            estimated_regs
            + 0.003 * estimated_smem_bytes
            + 0.8 * estimated_redg_like_ops
            + 0.25 * estimated_shfl_ops
            + reg_penalty
            + smem_penalty
            + occupancy_penalty
            + strategy_bonus
        )

        candidates.append({
            "strategy": "warp_specialized_bwd" if bs == 32 else "multiwarp_bwd",
            "launch_style": "grid_y_tiled_u",
            "block_size": bs,
            "u_tile": u_tile,
            "num_warps": num_warps,
            "grid_x": "B",
            "grid_y": f"(U + {u_tile} - 1) / {u_tile}",
            "u_traversal": "grid_y",
            "u_loop_step": None,
            "full_u_covered_per_block": False,
            "grad_y_mode": grad_y_mode,
            "use_shared_memory": use_shared_memory,
            "estimated_regs": estimated_regs,
            "estimated_smem_bytes": estimated_smem_bytes,
            "estimated_redg_like_ops": estimated_redg_like_ops,
            "estimated_shfl_ops": estimated_shfl_ops,
            "score": score,
        })

        # --------------------------------
        # Strategy B: persistent U
        # --------------------------------
        if enable_persistent_u and bs == 32:
            # One warp/block, grid.y = 1, iterate u in-kernel by step=32
            estimated_regs_persistent = estimate_base_regs(32) + 6
            estimated_smem_bytes_persistent = 0
            estimated_redg_like_ops_persistent = num_k
            estimated_shfl_ops_persistent = 5 * min(k_tile_size, num_k)

            reg_penalty_p = max(0, estimated_regs_persistent - reg_budget) * 20.0
            smem_penalty_p = 0.0

            # Persistent-U gets a bonus because it avoids grid.y partials
            score_persistent = (
                estimated_regs_persistent
                + 0.003 * estimated_smem_bytes_persistent
                + 0.8 * estimated_redg_like_ops_persistent
                + 0.25 * estimated_shfl_ops_persistent
                + reg_penalty_p
                + smem_penalty_p
                - 55.0
            )

            candidates.append({
                "strategy": "warp_persistent_u",
                "launch_style": "persistent_u_inner_loop",
                "block_size": 32,
                "u_tile": 32,
                "num_warps": 1,
                "grid_x": "B",
                "grid_y": "1",
                "u_traversal": "inner_loop",
                "u_loop_step": 32,
                "full_u_covered_per_block": True,
                "grad_y_mode": "warp_reduce_then_atomic",
                "use_shared_memory": False,
                "estimated_regs": estimated_regs_persistent,
                "estimated_smem_bytes": estimated_smem_bytes_persistent,
                "estimated_redg_like_ops": estimated_redg_like_ops_persistent,
                "estimated_shfl_ops": estimated_shfl_ops_persistent,
                "score": score_persistent,
            })

    if not candidates:
        raise ValueError("No valid candidate schedule")

    best = min(candidates, key=lambda x: x["score"])

    schedule = {
        "strategy": best["strategy"],
        "launch_style": best["launch_style"],
        "block_size": best["block_size"],
        "u_tile": best["u_tile"],
        "num_warps": best["num_warps"],
        "U_dim": int(U_dim),
        "grid_x": best["grid_x"],
        "grid_y": best["grid_y"],
        "u_traversal": best["u_traversal"],
        "u_loop_step": best["u_loop_step"],
        "full_u_covered_per_block": best["full_u_covered_per_block"],
        "use_shared_memory": best["use_shared_memory"],
        "write_modes": {
            "grad_w": "direct_store",
            "grad_x": "atomic",
            "grad_y": best["grad_y_mode"],
        },
        "ordered_symbols": {
            "i": ordered_i,
            "j": ordered_j,
            "k": ordered_k,
            "v": ordered_v,
        },
        "tiling": {
            "v_tile_size": v_tile_size,
            "i_tile_size": i_tile_size,
            "j_tile_size": j_tile_size,
            "k_tile_size": k_tile_size,
            "v_tiles": v_tiles,
            "i_tiles": i_tiles,
            "j_tiles": j_tiles,
            "k_tiles": k_tiles,
        },
        "groups": {
            "grad_w": {
                i: [{"v": v, "j": j, "k": k, "c": c} for (v, j, k, c) in gw[i]]
                for i in ordered_i
            },
            "grad_x": {
                j: [{"v": v, "i": i, "k": k, "c": c} for (v, i, k, c) in gx[j]]
                for j in ordered_j
            },
            "grad_y": {
                k: [{"v": v, "i": i, "j": j, "c": c} for (v, i, j, c) in gy[k]]
                for k in ordered_k
            },
        },
        "stats": {
            "num_paths": P,
            "num_i": num_i,
            "num_j": num_j,
            "num_k": num_k,
            "num_v": num_v,
            "avg_terms_per_i": avg_terms_per_i,
            "avg_terms_per_j": avg_terms_per_j,
            "avg_terms_per_k": avg_terms_per_k,
            "max_terms_per_i": max_terms_per_i,
            "max_terms_per_j": max_terms_per_j,
            "max_terms_per_k": max_terms_per_k,
        },
        "cost_model": {
            "selected": best,
            "candidates": candidates,
            "reg_budget": reg_budget,
            "smem_budget_bytes": smem_budget_bytes,
        },
    }
    return schedule


def summarize_backward_schedule(schedule: Dict[str, Any]) -> str:
    s = []
    s.append("=== Backward Schedule Summary ===")
    s.append(f"strategy                 : {schedule['strategy']}")
    s.append(f"launch_style             : {schedule['launch_style']}")
    s.append(f"block_size               : {schedule['block_size']}")
    s.append(f"grid_x                   : {schedule['grid_x']}")
    s.append(f"grid_y                   : {schedule['grid_y']}")
    s.append(f"u_traversal              : {schedule['u_traversal']}")
    s.append(f"u_loop_step              : {schedule['u_loop_step']}")
    s.append(f"full_u_covered_per_block : {schedule['full_u_covered_per_block']}")
    s.append(f"grad_w mode              : {schedule['write_modes']['grad_w']}")
    s.append(f"grad_x mode              : {schedule['write_modes']['grad_x']}")
    s.append(f"grad_y mode              : {schedule['write_modes']['grad_y']}")
    s.append("")
    s.append("tiling:")
    s.append(f"  v_tile_size            : {schedule['tiling']['v_tile_size']}")
    s.append(f"  i_tile_size            : {schedule['tiling']['i_tile_size']}")
    s.append(f"  j_tile_size            : {schedule['tiling']['j_tile_size']}")
    s.append(f"  k_tile_size            : {schedule['tiling']['k_tile_size']}")
    s.append("")
    s.append("selected cost candidate:")
    for k, v in schedule["cost_model"]["selected"].items():
        s.append(f"  {k:24s}: {v}")
    return "\n".join(s)


def _fmt_coeff(cc: float, scalar_t: str) -> str:
    if abs(cc - 1.0) < 1e-12:
        return "1"
    if abs(cc + 1.0) < 1e-12:
        return "-1"
    if scalar_t == "double":
        return f"{cc:.17g}"
    return f"{cc:.9g}f"



def emit_combine_launcher(bundle_name: str) -> str:
    kernel_name = f"{bundle_name}"

    return rf'''

std::vector<torch::Tensor> launcher_{bundle_name}(
    torch::Tensor grad_out,     // [S,V,U]
    torch::Tensor w,            // [B,Iw,U]
    torch::Tensor x_all,        // [S,Ix,U]
    torch::Tensor y,            // [B,Ky,1]
    torch::Tensor src_idx,      // [B] int32
    torch::Tensor dst_idx,      // [B] int32
    torch::Tensor b_list,       // [B] int32
    int64_t V64)
{{

    TORCH_CHECK(w.scalar_type() == grad_out.scalar_type(),
                "w and grad_out must have the same dtype");
    TORCH_CHECK(x_all.scalar_type() == w.scalar_type(),
                "x_all dtype must match w");
    TORCH_CHECK(y.scalar_type() == w.scalar_type(),
                "y dtype must match w");

    TORCH_CHECK(w.dim() == 3, "w must be [B,Iw,U]");
    TORCH_CHECK(x_all.dim() == 3, "x_all must be [S,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be [B,Ky,1]");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be [S,V,U]");

    const int B  = (int)w.size(0);
    const int Iw = (int)w.size(1);
    const int U  = (int)w.size(2);

    const int S  = (int)x_all.size(0);
    const int Ix = (int)x_all.size(1);

    const int Ky = (int)y.size(1);
    const int V  = (int)V64;

    TORCH_CHECK((int)x_all.size(0) == S, "internal shape error for x_all");
    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");
    TORCH_CHECK((int)y.size(0) == B, "y B mismatch");
    TORCH_CHECK((int)y.size(2) == 1, "y must be [B,Ky,1]");
    TORCH_CHECK((int)grad_out.size(0) == S, "grad_out S mismatch");
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");

    TORCH_CHECK((int)src_idx.numel() == B, "src_idx must be [B]");
    TORCH_CHECK((int)dst_idx.numel() == B, "dst_idx must be [B]");
    TORCH_CHECK((int)b_list.numel() == B, "b_list must be [B]");

    TORCH_CHECK(U > 0, "U must be > 0");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
    TORCH_CHECK(B >= 0 && S >= 0 && Iw >= 0 && Ix >= 0 && Ky >= 0 && V >= 0,
                "invalid negative shape");

    c10::cuda::CUDAGuard device_guard(w.device());

    auto grad_w = torch::zeros_like(w);      // [B,Iw,U]
    auto grad_x = torch::zeros_like(x_all);  // [S,Ix,U]
    auto grad_y = torch::zeros_like(y);      // [B,Ky,1]

    cudaStream_t stream = at::cuda::getDefaultCUDAStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{

        launch_{kernel_name}<scalar_t>(
            (const scalar_t*)w.data_ptr<scalar_t>(),
            (const scalar_t*)x_all.data_ptr<scalar_t>(),
            (const scalar_t*)y.data_ptr<scalar_t>(),
            (const scalar_t*)grad_out.data_ptr<scalar_t>(),
            (scalar_t*)grad_w.data_ptr<scalar_t>(),
            (scalar_t*)grad_x.data_ptr<scalar_t>(),
            (scalar_t*)grad_y.data_ptr<scalar_t>(),
            (const int32_t*)src_idx.data_ptr<int32_t>(),
            (const int32_t*)dst_idx.data_ptr<int32_t>(),
            (const int32_t*)b_list.data_ptr<int32_t>(),
            B, Iw, Ix, Ky, V, U, stream);
    }});

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {{grad_w, grad_x, grad_y}};
}}

TORCH_LIBRARY({bundle_name}_codegen, m) {{
    m.def("run", &launcher_{bundle_name});
}}
'''



def emit_backward_cuda_from_schedule(
    schedule: Dict[str, Any],
    kernel_name: str = "generated_backward_kernel",
    scalar_t: str = "double",
) -> str:
    """
    Emit CUDA C++ from schedule built by build_backward_schedule_from_lists().

    Supported:
      - u_traversal = "grid_y"
      - u_traversal = "inner_loop"   # persistent-U
      - grad_w = direct_store
      - grad_x = atomic
      - grad_y = warp_reduce_then_atomic
      - grad_y = warp_reduce_then_block_merge_then_atomic
    """
    write_modes = schedule["write_modes"]
    launch_style = schedule["launch_style"]
    u_traversal = schedule["u_traversal"]

    if write_modes["grad_w"] != "direct_store":
        raise NotImplementedError("Only grad_w=direct_store is supported")
    if write_modes["grad_x"] != "atomic":
        raise NotImplementedError("Only grad_x=atomic is supported")
    if write_modes["grad_y"] not in (
        "warp_reduce_then_atomic",
        "warp_reduce_then_block_merge_then_atomic",
    ):
        raise NotImplementedError(f"Unsupported grad_y mode: {write_modes['grad_y']}")

    block_size = int(schedule["block_size"])
    u_tile = int(schedule["u_tile"])
    u_loop_step = schedule["u_loop_step"]
    U_dim = int(schedule["U_dim"])

    ordered_i = schedule["ordered_symbols"]["i"]
    ordered_j = schedule["ordered_symbols"]["j"]
    ordered_k = schedule["ordered_symbols"]["k"]

    v_tiles = schedule["tiling"]["v_tiles"]
    i_tiles = schedule["tiling"]["i_tiles"]
    j_tiles = schedule["tiling"]["j_tiles"]
    k_tiles = schedule["tiling"]["k_tiles"]

    grad_w_groups = schedule["groups"]["grad_w"]
    grad_x_groups = schedule["groups"]["grad_x"]
    grad_y_groups = schedule["groups"]["grad_y"]

    lines: List[str] = []
    ap = lines.append

    ap("#include <stdint.h>")
    ap("#include <cuda_runtime.h>")
    ap("#include <torch/extension.h>")
    ap("#include <ATen/cuda/CUDAContext.h>")
    ap("#include <c10/cuda/CUDAGuard.h>")
    ap("#include <vector>")
    ap('''#include "../cuda_utils.hpp"''')
    ap("")

    ap("template <typename scalar_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x_all,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    ap("    scalar_t* __restrict__ grad_w,")
    ap("    scalar_t* __restrict__ grad_x,")
    ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int Iw, int Ix, int Ky, int V, int U)")
    ap("{")
    ap("    int b_global = (int)blockIdx.x;")
    ap("    if (b_global >= B) return;")
    ap("")
    ap("    int b   = b_list ? b_list[b_global] : b_global;")
    ap("    int src = src_idx[b];")
    ap("    int dst = dst_idx[b];")
    ap("    int lane = ((int)threadIdx.x) & 31;")
    ap("    int warp = ((int)threadIdx.x) >> 5;")
    ap("    int64_t y_base = (int64_t)b * Ky;")
    ap("")

    # y(k) independent of u: preload once
    ap("    // preload y(k), independent of u")
    for kk in ordered_k:
        ap(f"    scalar_t yk_{kk} = y[y_base + {kk}];")
    ap("")

    # grad_y accumulators may span full U in inner_loop mode
    ap("    // init grad_y accumulators")
    for kk in ordered_k:
        ap(f"    scalar_t gy_acc_k_{kk} = scalar_t(0);")
    ap("")

    def emit_one_u_body(indent: str, u_expr: str):
        ap(f"{indent}int u = {u_expr};")
        ap(f"{indent}if (u < U) {{")

        # preload w/x for this u
        ap(f"{indent}    // preload w(i,u)")
        for ii in ordered_i:
            ap(f"{indent}    scalar_t wi_{ii} = w[((int64_t)b * Iw + {ii}) * (int64_t)U + u];")
        ap(f"{indent}")
        ap(f"{indent}    // preload x(j,u)")
        for jj in ordered_j:
            ap(f"{indent}    scalar_t xj_{jj} = x_all[((int64_t)src * Ix + {jj}) * (int64_t)U + u];")
        ap(f"{indent}")

        # grad_w / grad_x accumulators are per-u, so inside loop/body
        ap(f"{indent}    // init per-u accumulators")
        for ii in ordered_i:
            ap(f"{indent}    scalar_t gw_acc_i_{ii} = scalar_t(0);")
        for jj in ordered_j:
            ap(f"{indent}    scalar_t gx_acc_j_{jj} = scalar_t(0);")
        ap(f"{indent}")

        # v tiles
        for tile_id, vtile in enumerate(v_tiles):
            ap(f"{indent}    // ---- v tile {tile_id} ----")
            for vv in vtile:
                ap(f"{indent}    scalar_t go_v_{vv} = grad_out[((int64_t)dst * V + {vv}) * (int64_t)U + u];")
            ap(f"{indent}")

            # grad_w tile accumulation
            for i_tile_id, itile in enumerate(i_tiles):
                ap(f"{indent}    // grad_w i-tile {i_tile_id}")
                for ii in itile:
                    for entry in grad_w_groups[ii]:
                        vv = int(entry["v"])
                        if vv not in vtile:
                            continue
                        jj = int(entry["j"])
                        kk = int(entry["k"])
                        cc = float(entry["c"])
                        if abs(cc - 1.0) < 1e-12:
                            ap(f"{indent}    gw_acc_i_{ii} += xj_{jj} * yk_{kk} * go_v_{vv};")
                        elif abs(cc + 1.0) < 1e-12:
                            ap(f"{indent}    gw_acc_i_{ii} -= xj_{jj} * yk_{kk} * go_v_{vv};")
                        else:
                            cstr = _fmt_coeff(cc, scalar_t)
                            ap(f"{indent}    gw_acc_i_{ii} += scalar_t({cstr}) * xj_{jj} * yk_{kk} * go_v_{vv};")
                ap(f"{indent}")

            # grad_x tile accumulation
            for j_tile_id, jtile in enumerate(j_tiles):
                ap(f"{indent}    // grad_x j-tile {j_tile_id}")
                for jj in jtile:
                    for entry in grad_x_groups[jj]:
                        vv = int(entry["v"])
                        if vv not in vtile:
                            continue
                        ii = int(entry["i"])
                        kk = int(entry["k"])
                        cc = float(entry["c"])
                        if abs(cc - 1.0) < 1e-12:
                            ap(f"{indent}    gx_acc_j_{jj} += wi_{ii} * yk_{kk} * go_v_{vv};")
                        elif abs(cc + 1.0) < 1e-12:
                            ap(f"{indent}    gx_acc_j_{jj} -= wi_{ii} * yk_{kk} * go_v_{vv};")
                        else:
                            cstr = _fmt_coeff(cc, scalar_t)
                            ap(f"{indent}    gx_acc_j_{jj} += scalar_t({cstr}) * wi_{ii} * yk_{kk} * go_v_{vv};")
                ap(f"{indent}")

            # grad_y tile accumulation
            for k_tile_id, ktile in enumerate(k_tiles):
                ap(f"{indent}    // grad_y k-tile {k_tile_id}")
                for kk in ktile:
                    for entry in grad_y_groups[kk]:
                        vv = int(entry["v"])
                        if vv not in vtile:
                            continue
                        ii = int(entry["i"])
                        jj = int(entry["j"])
                        cc = float(entry["c"])
                        if abs(cc - 1.0) < 1e-12:
                            ap(f"{indent}    gy_acc_k_{kk} += wi_{ii} * xj_{jj} * go_v_{vv};")
                        elif abs(cc + 1.0) < 1e-12:
                            ap(f"{indent}    gy_acc_k_{kk} -= wi_{ii} * xj_{jj} * go_v_{vv};")
                        else:
                            cstr = _fmt_coeff(cc, scalar_t)
                            ap(f"{indent}    gy_acc_k_{kk} += scalar_t({cstr}) * wi_{ii} * xj_{jj} * go_v_{vv};")
                ap(f"{indent}")

        # write grad_w/grad_x for this u
        ap(f"{indent}    // direct-store grad_w for this u")
        for ii in ordered_i:
            ap(f"{indent}    grad_w[((int64_t)b * Iw + {ii}) * (int64_t)U + u] = gw_acc_i_{ii};")
        ap(f"{indent}")

        ap(f"{indent}    // atomic grad_x for this u")
        for jj in ordered_j:
            ap(f"{indent}    atomicAdd(&grad_x[((int64_t)src * Ix + {jj}) * (int64_t)U + u], gx_acc_j_{jj});")
        ap(f"{indent}}}")
        ap("")

    # U traversal
    if u_traversal == "grid_y":
        ap("    // u traversal: grid_y tiled-u")
        ap("    int ublk = (int)blockIdx.y;")
        emit_one_u_body("    ", f"ublk * {u_tile} + (int)threadIdx.x")

    elif u_traversal == "inner_loop":
        if u_loop_step is None:
            raise ValueError("schedule has u_traversal=inner_loop but u_loop_step is None")
        ap("    // u traversal: persistent inner loop over U")
        ap(f"    for (int u_base = 0; u_base < U; u_base += {int(u_loop_step)}) {{")
        emit_one_u_body("        ", "u_base + lane")
        ap("    }")
        ap("")
    else:
        raise NotImplementedError(f"Unsupported u_traversal: {u_traversal}")

    # Tail reduction for grad_y
    if write_modes["grad_y"] == "warp_reduce_then_atomic":
        ap("    // tail warp-reduce grad_y")
        for kk in ordered_k:
            ap(f"    scalar_t gy_sum_k_{kk} = warp_sum_xor(gy_acc_k_{kk});")
            ap(f"    if (lane == 0) atomicAdd(&grad_y[y_base + {kk}], gy_sum_k_{kk});")
        ap("")
    else:
        ap("    // tail warp-reduce + block-merge grad_y")
        ap("    extern __shared__ char smem_raw[];")
        ap("    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);")
        ap("    const int num_warps = (blockDim.x + 31) >> 5;")
        ap("")
        for k_idx, kk in enumerate(ordered_k):
            ap(f"    scalar_t gy_sum_k_{kk} = warp_sum_xor(gy_acc_k_{kk});")
            ap(f"    if (lane == 0) smem[{k_idx} * num_warps + warp] = gy_sum_k_{kk};")
        ap("    __syncthreads();")
        ap("    if (warp == 0) {")
        ap("        int red_lane = lane;")
        for k_idx, kk in enumerate(ordered_k):
            ap(f"        scalar_t block_sum_k_{kk} = scalar_t(0);")
            ap(f"        if (red_lane < num_warps) block_sum_k_{kk} = smem[{k_idx} * num_warps + red_lane];")
            ap(f"        block_sum_k_{kk} = warp_sum_xor(block_sum_k_{kk});")
            ap(f"        if (red_lane == 0) atomicAdd(&grad_y[y_base + {kk}], block_sum_k_{kk});")
        ap("    }")
        ap("")

    ap("}")
    ap("")

    # launcher
    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x_all,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int Iw, int Ix, int Ky, int V, int U,")
    ap("    cudaStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")

    if launch_style == "grid_y_tiled_u":
        ap(f"    dim3 grid(B, (U + {u_tile} - 1) / {u_tile});")
    elif launch_style == "persistent_u_inner_loop":
        ap("    dim3 grid(B, 1);")
    else:
        raise NotImplementedError(f"Unsupported launch_style: {launch_style}")

    if write_modes["grad_y"] == "warp_reduce_then_atomic":
        ap(f"    {kernel_name}<scalar_t><<<grid, block, 0, stream>>>(")
    else:
        smem_slots = len(ordered_k) * ((block_size + 31) // 32)
        ap(f"    size_t smem_bytes = sizeof(scalar_t) * {smem_slots};")
        ap(f"    {kernel_name}<scalar_t><<<grid, block, smem_bytes, stream>>>(")

    ap("        w, x_all, y, grad_out, grad_w, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list, B, Iw, Ix, Ky, V, U);")
    ap("}")
    ap("")
       
    #return '\n'.join(lines)
    code = '\n'.join(lines)
    code = code + "\n" + emit_combine_launcher(kernel_name)
    file_name = f"{kernel_name}.cu"
    #Path(f"../../fasteq/cuda/src/uniform1d_codegen/{file_name}").write_text(code, encoding="utf-8")
    (out_dir / file_name).write_text(code, encoding="utf-8")
    return code

# ================== GradX, GradW, GradY, Split Code Gen ==========================

@dataclass
class CGPath:
    i: int
    j: int
    k: int
    v: int
    c: float


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def group_by_key(paths: List[CGPath], key_fn):
    d = OrderedDict()
    for p in paths:
        key = key_fn(p)
        d.setdefault(key, []).append(p)
    return d


def chunk_keys_by_budget(
    groups: OrderedDict,
    *,
    base_regs: int,
    acc_cost: int,
    cache_cost_per_symbol: int,
    reg_budget: int,
    max_targets_per_chunk: int,
):
    """
    把 target groups 分块，使估算寄存器数不超过 reg_budget.
    估算：
      regs ~= base_regs
            + acc_cost * num_targets
            + cache_cost_per_symbol * (#uniq_j + #uniq_k + #uniq_v) 等
    """
    items = list(groups.items())
    chunks = []

    cur = []
    cur_targets = 0
    cur_syms = set()

    def estimate_regs(target_count, syms_count):
        return base_regs + acc_cost * target_count + cache_cost_per_symbol * syms_count

    for tgt, plist in items:
        local_syms = set()
        for p in plist:
            local_syms.add(("j", p.j))
            local_syms.add(("k", p.k))
            local_syms.add(("v", p.v))
            local_syms.add(("i", p.i))

        new_targets = cur_targets + 1
        new_syms = len(cur_syms | local_syms)

        too_many_targets = new_targets > max_targets_per_chunk
        too_many_regs = estimate_regs(new_targets, new_syms) > reg_budget

        if cur and (too_many_targets or too_many_regs):
            chunks.append(cur)
            cur = []
            cur_targets = 0
            cur_syms = set()

        cur.append((tgt, plist))
        cur_targets += 1
        cur_syms |= local_syms

    if cur:
        chunks.append(cur)

    return chunks


def fmt_coeff(x: float) -> str:
    s = f"{x:.17g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s


# ------------------------------------------------------------
# Code emitters
# ------------------------------------------------------------
def emit_preamble() -> str:
    return r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include "../cuda_utils.hpp"
'''


def emit_gradw_kernel(paths: List[CGPath], kernel_name: str, reg_budget: int = 64) -> str:
    """
    grad_w: 按 i 分组
    对每个 chunk 内若干个 i 做寄存器累加 acc_i
    组内展开 path, 最后各 acc_i 只写一次
    """
    groups = group_by_key(paths, lambda p: p.i)
    chunks = chunk_keys_by_budget(
        groups,
        base_regs=24,
        acc_cost=1,
        cache_cost_per_symbol=1,
        reg_budget=reg_budget,
        max_targets_per_chunk=4,
    )

    lines = []
    ap = lines.append

    ap('template <typename scalar_t>')
    ap(f'__launch_bounds__(32, 8) __global__ void {kernel_name}(')
    ap('    const scalar_t* __restrict__ grad_out,')
    ap('    const scalar_t* __restrict__ x_all,')
    ap('    const scalar_t* __restrict__ y,')
    ap('    scalar_t* __restrict__ grad_w,')
    ap('    const int32_t* __restrict__ src_idx,')
    ap('    const int32_t* __restrict__ dst_idx,')
    ap('    const int32_t* __restrict__ b_list,')
    ap('    int B, int Iw, int Ix, int Ky, int V)')
    ap('{')
    ap('    int lane = threadIdx.x;')
    ap('    int bidx = (int)blockIdx.x;')
    ap('    if (bidx >= B) return;')
    ap('')
    ap('    int b   = b_list ? b_list[bidx] : bidx;')
    ap('    int src = src_idx[b];')
    ap('    int dst = dst_idx[b];')
    ap('')
    ap('    int64_t x_base  = ((int64_t)src * Ix) * 32 + lane;')
    ap('    int64_t y_base  = (int64_t)b * Ky;')
    ap('    int64_t go_base = ((int64_t)dst * V) * 32 + lane;')
    ap('    int64_t gw_base = ((int64_t)b   * Iw) * 32 + lane;')
    ap('')

    for ci, chunk in enumerate(chunks):
        ap(f'    // ---- grad_w chunk {ci} ----')
        # declare acc
        for i, _plist in chunk:
            ap(f'    scalar_t acc_i_{i} = scalar_t(0);')
        ap('')

        # very small explicit caches at statement scope
        for i, plist in chunk:
            ap(f'    // target i = {i}')
            # 按 (k, v) 再分组，利于复用 y/go
            kv_groups = OrderedDict()
            for p in plist:
                kv_groups.setdefault((p.k, p.v), []).append(p)

            for (k, v), kv_plist in kv_groups.items():
                ap(f'    {{')
                ap(f'        scalar_t y_k_{k} = y[y_base + {k}];')
                ap(f'        scalar_t go_v_{v} = grad_out[go_base + ((int64_t){v} << 5)];')
                for p in kv_plist:
                    c = fmt_coeff(p.c)
                    ap(f'        scalar_t x_j_{p.j} = x_all[x_base + ((int64_t){p.j} << 5)];')
                    ap(f'        acc_i_{i} = fma((scalar_t)({c}) * x_j_{p.j}, y_k_{k} * go_v_{v}, acc_i_{i});')
                ap(f'    }}')
            ap('')

        # single write per i
        for i, _plist in chunk:
            ap(f'    grad_w[gw_base + ((int64_t){i} << 5)] = acc_i_{i};')
        ap('')

    ap('}')
    return "\n".join(lines)

def emit_gradx_kernel(paths: List[CGPath], kernel_name: str, reg_budget: int = 64) -> str:
    """
    grad_x: 按 j 分组
    """
    groups = group_by_key(paths, lambda p: p.j)
    chunks = chunk_keys_by_budget(
        groups,
        base_regs=24,
        acc_cost=1,
        cache_cost_per_symbol=1,
        reg_budget=reg_budget,
        max_targets_per_chunk=4,
    )

    lines = []
    ap = lines.append

    ap('template <typename scalar_t>')
    ap(f'__launch_bounds__(32, 8) __global__ void {kernel_name}(')
    ap('    const scalar_t* __restrict__ grad_out,')
    ap('    const scalar_t* __restrict__ w,')
    ap('    const scalar_t* __restrict__ y,')
    ap('    scalar_t* __restrict__ grad_x,')
    ap('    const int32_t* __restrict__ src_idx,')
    ap('    const int32_t* __restrict__ dst_idx,')
    ap('    const int32_t* __restrict__ b_list,')
    ap('    int B, int Iw, int Ix, int Ky, int V)')
    ap('{')
    ap('    int lane = threadIdx.x;')
    ap('    int bidx = (int)blockIdx.x;')
    ap('    if (bidx >= B) return;')
    ap('')
    ap('    int b   = b_list ? b_list[bidx] : bidx;')
    ap('    int src = src_idx[b];')
    ap('    int dst = dst_idx[b];')
    ap('')
    ap('    int64_t w_base  = ((int64_t)b   * Iw) * 32 + lane;')
    ap('    int64_t y_base  = (int64_t)b * Ky;')
    ap('    int64_t go_base = ((int64_t)dst * V) * 32 + lane;')
    ap('    int64_t gx_base = ((int64_t)src * Ix) * 32 + lane;')
    ap('')

    for ci, chunk in enumerate(chunks):
        ap(f'    // ---- grad_x chunk {ci} ----')
        for j, _plist in chunk:
            ap(f'    scalar_t acc_j_{j} = scalar_t(0);')
        ap('')

        for j, plist in chunk:
            ap(f'    // target j = {j}')
            iv_groups = OrderedDict()
            for p in plist:
                iv_groups.setdefault((p.i, p.k, p.v), []).append(p)

            for (i, k, v), iv_plist in iv_groups.items():
                ap('    {')
                ap(f'        scalar_t w_i_{i} = w[w_base + ((int64_t){i} << 5)];')
                ap(f'        scalar_t y_k_{k} = y[y_base + {k}];')
                ap(f'        scalar_t go_v_{v} = grad_out[go_base + ((int64_t){v} << 5)];')
                for p in iv_plist:
                    c = fmt_coeff(p.c)
                    ap(f'        acc_j_{j} = fma((scalar_t)({c}) * w_i_{i}, y_k_{k} * go_v_{v}, acc_j_{j});')
                ap('    }')
            ap('')

        for j, _plist in chunk:
            ap(f'    atomicAdd(&grad_x[gx_base + ((int64_t){j} << 5)], acc_j_{j});')
        ap('')

    ap('}')
    return "\n".join(lines)


def emit_grady_kernel(paths: List[CGPath], kernel_name: str, reg_budget: int = 64) -> str:
    """
    grad_y: 按 k 分组
    每个 k 只做一次 warp_sum 和一次 lane0 写回
    """
    groups = group_by_key(paths, lambda p: p.k)

    # grad_y 的局部 live 值稍多一点，chunk 更小
    chunks = chunk_keys_by_budget(
        groups,
        base_regs=26,
        acc_cost=1,   # local_k
        cache_cost_per_symbol=1,
        reg_budget=reg_budget,
        max_targets_per_chunk=4,
    )

    lines = []
    ap = lines.append

    ap('template <typename scalar_t>')
    ap(f'__launch_bounds__(32, 8) __global__ void {kernel_name}(')
    ap('    const scalar_t* __restrict__ grad_out,')
    ap('    const scalar_t* __restrict__ w,')
    ap('    const scalar_t* __restrict__ x_all,')
    ap('    scalar_t* __restrict__ grad_y,')
    ap('    const int32_t* __restrict__ src_idx,')
    ap('    const int32_t* __restrict__ dst_idx,')
    ap('    const int32_t* __restrict__ b_list,')
    ap('    int B, int Iw, int Ix, int Ky, int V)')
    ap('{')
    ap('    int lane = threadIdx.x;')
    ap('    int bidx = (int)blockIdx.x;')
    ap('    if (bidx >= B) return;')
    ap('')
    ap('    int b   = b_list ? b_list[bidx] : bidx;')
    ap('    int src = src_idx[b];')
    ap('    int dst = dst_idx[b];')
    ap('')
    ap('    int64_t w_base  = ((int64_t)b   * Iw) * 32 + lane;')
    ap('    int64_t x_base  = ((int64_t)src * Ix) * 32 + lane;')
    ap('    int64_t go_base = ((int64_t)dst * V) * 32 + lane;')
    ap('    int64_t gy_base = (int64_t)b * Ky;')
    ap('')

    for ci, chunk in enumerate(chunks):
        ap(f'    // ---- grad_y chunk {ci} ----')
        for k, plist in chunk:
            ap(f'    scalar_t local_k_{k} = scalar_t(0);')
            # 组内再按 (i,j,v) 组织
            ijv_groups = OrderedDict()
            for p in plist:
                ijv_groups.setdefault((p.i, p.j, p.v), []).append(p)

            for (i, j, v), sub in ijv_groups.items():
                ap('    {')
                ap(f'        scalar_t w_i_{i}  = w[w_base + ((int64_t){i} << 5)];')
                ap(f'        scalar_t x_j_{j}  = x_all[x_base + ((int64_t){j} << 5)];')
                ap(f'        scalar_t go_v_{v} = grad_out[go_base + ((int64_t){v} << 5)];')
                for p in sub:
                    c = fmt_coeff(p.c)
                    ap(f'        local_k_{k} = fma((scalar_t)({c}) * w_i_{i}, x_j_{j} * go_v_{v}, local_k_{k});')
                ap('    }')
            ap(f'    scalar_t sum_k_{k} = warp_sum(local_k_{k});')
            ap(f'    if (lane == 0) grad_y[gy_base + {k}] += sum_k_{k};')
            ap('')

    ap('}')
    return "\n".join(lines)


def build_gradx_segments(paths: List[CGPath], acc_slots: int = 8) -> List[Dict[str, Any]]:
    """
    grad_x:
      target = j
      term   = (slot, i, k, v, c)
    """
    by_j: "OrderedDict[int, List[CGPath]]" = OrderedDict()
    for p in sorted(paths, key=lambda p: (p.j, p.i, p.k, p.v)):
        by_j.setdefault(int(p.j), []).append(p)

    unique_j = list(by_j.keys())
    segments: List[Dict[str, Any]] = []

    for seg_start in range(0, len(unique_j), acc_slots):
        tgt_js = unique_j[seg_start: seg_start + acc_slots]
        slot_of_j = {j: s for s, j in enumerate(tgt_js)}

        raw_terms: List[Tuple[int, int, int, int, float]] = []
        for j in tgt_js:
            slot = slot_of_j[j]
            for p in by_j[j]:
                raw_terms.append((slot, int(p.i), int(p.k), int(p.v), float(p.c)))

        # 先按 (i,k,v)，再按 slot
        raw_terms.sort(key=lambda t: (t[1], t[2], t[3], t[0]))

        segments.append({
            "targets": tgt_js,
            "terms": raw_terms,
        })

    return segments


def emit_gradx_kernel_segmented_unrolled(
    paths: List[CGPath],
    kernel_name: str,
    acc_slots: int = 2,
    ikv_group_reuse_threshold: int = 2,
) -> str:
    """
    segmented-style fully-unrolled grad_x
    """
    segments = build_gradx_segments(paths, acc_slots=acc_slots)

    lines = []
    ap = lines.append

    ap('#include <stdint.h>')
    ap('#include <cuda_runtime.h>')
    ap('')

    ap('template <typename scalar_t>')
    ap(f'__global__ void {kernel_name}(')
    ap('    const scalar_t* __restrict__ grad_out,')
    ap('    const scalar_t* __restrict__ w,')
    ap('    const scalar_t* __restrict__ y,')
    ap('    scalar_t* __restrict__ grad_x,')
    ap('    const int32_t* __restrict__ src_idx,')
    ap('    const int32_t* __restrict__ dst_idx,')
    ap('    const int32_t* __restrict__ b_list,')
    ap('    int B, int Iw, int Ix, int Ky, int V)')
    ap('{')
    ap('    int lane = (int)threadIdx.x;')
    ap('    int bidx = (int)blockIdx.x;')
    ap('    if (bidx >= B) return;')
    ap('')
    ap('    int b   = b_list ? b_list[bidx] : bidx;')
    ap('    int src = src_idx[b];')
    ap('    int dst = dst_idx[b];')
    ap('')
    ap('    int64_t w_base  = ((int64_t)b   * Iw) * 32 + lane;')
    ap('    int64_t y_base  = (int64_t)b * Ky;')
    ap('    int64_t go_base = ((int64_t)dst * V) * 32 + lane;')
    ap('    int64_t gx_base = ((int64_t)src * Ix) * 32 + lane;')
    ap('')

    for seg_id, seg in enumerate(segments):
        targets: List[int] = seg["targets"]
        terms: List[Tuple[int, int, int, int, float]] = seg["terms"]

        ap(f'    // ============================================================')
        ap(f'    // grad_x segment {seg_id}: targets = {targets}')
        ap(f'    // ============================================================')
        ap('    {')

        for s in range(len(targets)):
            ap(f'        scalar_t acc{s} = scalar_t(0);')
        ap('')

        ikv_buckets: "OrderedDict[Tuple[int,int,int], List[Tuple[int,int,int,int,float]]]" = OrderedDict()
        for term in terms:
            slot, i, k, v, c = term
            ikv_buckets.setdefault((i, k, v), []).append(term)

        for (i, k, v), bucket in ikv_buckets.items():
            bucket.sort(key=lambda t: t[0])
            use_reuse = len(bucket) >= ikv_group_reuse_threshold

            if use_reuse:
                ap('        {')
                ap(f'            scalar_t wv  = w[w_base + ((int64_t){i} << 5)];')
                ap(f'            scalar_t yv  = y[y_base + {k}];')
                ap(f'            scalar_t gov = grad_out[go_base + ((int64_t){v} << 5)];')
                ap(f'            scalar_t t0  = wv * (yv * gov);')
                for (slot, _i, _k, _v, c) in bucket:
                    cstr = fmt_coeff(c)
                    ap(f'            acc{slot} = fma((scalar_t)({cstr}), t0, acc{slot});')
                ap('        }')
                ap('')
            else:
                for (slot, _i, _k, _v, c) in bucket:
                    cstr = fmt_coeff(c)
                    ap('        {')
                    ap(f'            scalar_t wv  = w[w_base + ((int64_t){i} << 5)];')
                    ap(f'            scalar_t yv  = y[y_base + {k}];')
                    ap(f'            scalar_t gov = grad_out[go_base + ((int64_t){v} << 5)];')
                    ap(f'            acc{slot} = fma((scalar_t)({cstr}), wv * (yv * gov), acc{slot});')
                    ap('        }')
                ap('')

        for s, j in enumerate(targets):
            ap(f'        atomicAdd(&grad_x[gx_base + ((int64_t){j} << 5)], acc{s});')

        ap('    }')
        ap('')

    ap('}')
    return "\n".join(lines)

def build_grady_segments(paths: List[CGPath], acc_slots: int = 8) -> List[Dict[str, Any]]:
    """
    grad_y:
      target = k
      term   = (slot, i, j, v, c)
    """
    by_k: "OrderedDict[int, List[CGPath]]" = OrderedDict()
    for p in sorted(paths, key=lambda p: (p.k, p.i, p.j, p.v)):
        by_k.setdefault(int(p.k), []).append(p)

    unique_k = list(by_k.keys())
    segments: List[Dict[str, Any]] = []

    for seg_start in range(0, len(unique_k), acc_slots):
        tgt_ks = unique_k[seg_start: seg_start + acc_slots]
        slot_of_k = {k: s for s, k in enumerate(tgt_ks)}

        raw_terms: List[Tuple[int, int, int, int, float]] = []
        for k in tgt_ks:
            slot = slot_of_k[k]
            for p in by_k[k]:
                raw_terms.append((slot, int(p.i), int(p.j), int(p.v), float(p.c)))

        # 先按 (i,j,v)，再按 slot
        raw_terms.sort(key=lambda t: (t[1], t[2], t[3], t[0]))

        segments.append({
            "targets": tgt_ks,
            "terms": raw_terms,
        })

    return segments


def emit_grady_kernel_segmented_unrolled(
    paths: List[CGPath],
    kernel_name: str,
    acc_slots: int = 2,
    ijv_group_reuse_threshold: int = 2,
) -> str:
    """
    segmented-style fully-unrolled grad_y
    """
    segments = build_grady_segments(paths, acc_slots=acc_slots)

    lines = []
    ap = lines.append

    ap('template <typename scalar_t>')
    ap(f'__global__ void {kernel_name}(')
    ap('    const scalar_t* __restrict__ grad_out,')
    ap('    const scalar_t* __restrict__ w,')
    ap('    const scalar_t* __restrict__ x_all,')
    ap('    scalar_t* __restrict__ grad_y,')
    ap('    const int32_t* __restrict__ src_idx,')
    ap('    const int32_t* __restrict__ dst_idx,')
    ap('    const int32_t* __restrict__ b_list,')
    ap('    int B, int Iw, int Ix, int Ky, int V)')
    ap('{')
    ap('    int lane = (int)threadIdx.x;')
    ap('    int bidx = (int)blockIdx.x;')
    ap('    if (bidx >= B) return;')
    ap('')
    ap('    int b   = b_list ? b_list[bidx] : bidx;')
    ap('    int src = src_idx[b];')
    ap('    int dst = dst_idx[b];')
    ap('')
    ap('    int64_t w_base  = ((int64_t)b   * Iw) * 32 + lane;')
    ap('    int64_t x_base  = ((int64_t)src * Ix) * 32 + lane;')
    ap('    int64_t go_base = ((int64_t)dst * V) * 32 + lane;')
    ap('    int64_t gy_base = (int64_t)b * Ky;')
    ap('')

    for seg_id, seg in enumerate(segments):
        targets: List[int] = seg["targets"]
        terms: List[Tuple[int, int, int, int, float]] = seg["terms"]

        ap(f'    // ============================================================')
        ap(f'    // grad_y segment {seg_id}: targets = {targets}')
        ap(f'    // ============================================================')
        ap('    {')

        for s in range(len(targets)):
            ap(f'        scalar_t acc{s} = scalar_t(0);')
        ap('')

        ijv_buckets: "OrderedDict[Tuple[int,int,int], List[Tuple[int,int,int,int,float]]]" = OrderedDict()
        for term in terms:
            slot, i, j, v, c = term
            ijv_buckets.setdefault((i, j, v), []).append(term)

        for (i, j, v), bucket in ijv_buckets.items():
            bucket.sort(key=lambda t: t[0])
            use_reuse = len(bucket) >= ijv_group_reuse_threshold

            if use_reuse:
                ap('        {')
                ap(f'            scalar_t wv  = w[w_base + ((int64_t){i} << 5)];')
                ap(f'            scalar_t xv  = x_all[x_base + ((int64_t){j} << 5)];')
                ap(f'            scalar_t gov = grad_out[go_base + ((int64_t){v} << 5)];')
                ap(f'            scalar_t t0  = xv * gov;')
                for (slot, _i, _j, _v, c) in bucket:
                    cstr = fmt_coeff(c)
                    ap(f'            acc{slot} = fma((scalar_t)({cstr}) * wv, t0, acc{slot});')
                ap('        }')
                ap('')
            else:
                for (slot, _i, _j, _v, c) in bucket:
                    cstr = fmt_coeff(c)
                    ap('        {')
                    ap(f'            scalar_t wv  = w[w_base + ((int64_t){i} << 5)];')
                    ap(f'            scalar_t xv  = x_all[x_base + ((int64_t){j} << 5)];')
                    ap(f'            scalar_t gov = grad_out[go_base + ((int64_t){v} << 5)];')
                    ap(f'            acc{slot} = fma((scalar_t)({cstr}) * wv, xv * gov, acc{slot});')
                    ap('        }')
                ap('')

        for s, k in enumerate(targets):
            ap(f'        scalar_t sum{s} = warp_sum(acc{s});')
            ap(f'        if (lane == 0) grad_y[gy_base + {k}] += sum{s};')

        ap('    }')
        ap('')

    ap('}')
    return "\n".join(lines)


def build_gradw_segments(paths: List[CGPath], acc_slots: int = 8) -> List[Dict[str, Any]]:
    """
    将全量 gradw paths 重排为 segmented-style 结构。

    返回:
      segments = [
        {
          "targets": [i0, i1, ...],              # 长度 <= acc_slots
          "terms": [
              (slot, j, k, v, c),
              ...
          ]
        },
        ...
      ]
    """

    # 1) 先按 i 分组
    by_i: "OrderedDict[int, List[CGPath]]" = OrderedDict()
    for p in sorted(paths, key=lambda p: (p.i, p.k, p.v, p.j)):
        by_i.setdefault(int(p.i), []).append(p)

    unique_i = list(by_i.keys())

    # 2) 每 acc_slots 个 i 形成一个 segment
    segments: List[Dict[str, Any]] = []
    for seg_start in range(0, len(unique_i), acc_slots):
        tgt_is = unique_i[seg_start: seg_start + acc_slots]
        slot_of_i = {i: s for s, i in enumerate(tgt_is)}

        # 收集该 segment 的全部 term
        raw_terms: List[Tuple[int, int, int, int, float]] = []
        for i in tgt_is:
            slot = slot_of_i[i]
            plist = by_i[i]
            for p in plist:
                raw_terms.append((slot, int(p.j), int(p.k), int(p.v), float(p.c)))

        # 3) 排序：先按 (k, v)，再按 slot，再按 j
        raw_terms.sort(key=lambda t: (t[2], t[3], t[0], t[1]))

        # 4) 再按 (k,v) 分小簇，便于 codegen 时做局部 y/go 复用
        kv_buckets: "OrderedDict[Tuple[int,int], List[Tuple[int,int,int,int,float]]]" = OrderedDict()
        for t in raw_terms:
            _, j, k, v, _ = t
            kv_buckets.setdefault((k, v), []).append(t)

        # 对小簇内部进一步按 (slot, j) 排
        terms: List[Tuple[int, int, int, int, float]] = []
        for (k, v), bucket in kv_buckets.items():
            bucket.sort(key=lambda t: (t[0], t[1]))
            terms.extend(bucket)

        segments.append({
            "targets": tgt_is,
            "terms": terms,
        })

    return segments


def emit_gradw_kernel_segmented_unrolled(
    paths: List[CGPath],
    kernel_name: str,
    acc_slots: int = 8,
    kv_group_reuse_threshold: int = 2,
) -> str:
    """
    segmented-style fully-unrolled grad_w kernel generator

    特点：
      - 覆盖全部 path
      - 每个 segment 最多 acc_slots 个 target i
      - segment 内 fully-unrolled
      - 对共享 (k,v) 的小簇做局部 y/go 复用
      - 固定少量 acc 槽位，减少寄存器数量
    """
    segments = build_gradw_segments(paths, acc_slots=acc_slots)

    lines: List[str] = []
    ap = lines.append

    ap('template <typename scalar_t>')
    ap(f'__global__ void {kernel_name}(')
    ap('    const scalar_t* __restrict__ grad_out,')
    ap('    const scalar_t* __restrict__ x_all,')
    ap('    const scalar_t* __restrict__ y,')
    ap('    scalar_t* __restrict__ grad_w,')
    ap('    const int32_t* __restrict__ src_idx,')
    ap('    const int32_t* __restrict__ dst_idx,')
    ap('    const int32_t* __restrict__ b_list,')
    ap('    int B, int Iw, int Ix, int Ky, int V)')
    ap('{')
    ap('    int lane = (int)threadIdx.x;')
    ap('    int bidx = (int)blockIdx.x;')
    ap('    if (bidx >= B) return;')
    ap('')
    ap('    int b   = b_list ? b_list[bidx] : bidx;')
    ap('    int src = src_idx[b];')
    ap('    int dst = dst_idx[b];')
    ap('')
    ap('    int64_t x_base  = ((int64_t)src * Ix) * 32 + lane;')
    ap('    int64_t y_base  = (int64_t)b * Ky;')
    ap('    int64_t go_base = ((int64_t)dst * V) * 32 + lane;')
    ap('    int64_t gw_base = ((int64_t)b   * Iw) * 32 + lane;')
    ap('')

    for seg_id, seg in enumerate(segments):
        targets: List[int] = seg["targets"]
        terms: List[Tuple[int, int, int, int, float]] = seg["terms"]

        ap(f'    // ============================================================')
        ap(f'    // segment {seg_id}: targets = {targets}')
        ap(f'    // ============================================================')
        ap('    {')

        # 固定少量 accumulator 槽位
        for s in range(len(targets)):
            ap(f'        scalar_t acc{s} = scalar_t(0);')
        ap('')

        # 按 (k,v) 再聚类，决定是否做局部 y/go 复用块
        kv_buckets: "OrderedDict[Tuple[int,int], List[Tuple[int,int,int,int,float]]]" = OrderedDict()
        for term in terms:
            slot, j, k, v, c = term
            kv_buckets.setdefault((k, v), []).append(term)

        for (k, v), bucket in kv_buckets.items():
            bucket.sort(key=lambda t: (t[0], t[1]))
            use_kv_reuse = len(bucket) >= kv_group_reuse_threshold

            if use_kv_reuse:
                ap('        {')
                ap(f'            scalar_t yv  = y[y_base + {k}];')
                ap(f'            scalar_t gov = grad_out[go_base + ((int64_t){v} << 5)];')
                ap(f'            scalar_t yg  = yv * gov;')
                for (slot, j, _k, _v, c) in bucket:
                    cstr = fmt_coeff(c)
                    ap('            {')
                    ap(f'                scalar_t xv = x_all[x_base + ((int64_t){j} << 5)];')
                    ap(f'                acc{slot} = fma((scalar_t)({cstr}), xv * yg, acc{slot});')
                    ap('            }')
                ap('        }')
                ap('')
            else:
                # 单条/很小簇，不额外拉长 y/go live range
                for (slot, j, _k, _v, c) in bucket:
                    cstr = fmt_coeff(c)
                    ap('        {')
                    ap(f'            scalar_t xv  = x_all[x_base + ((int64_t){j} << 5)];')
                    ap(f'            scalar_t yv  = y[y_base + {k}];')
                    ap(f'            scalar_t gov = grad_out[go_base + ((int64_t){v} << 5)];')
                    ap(f'            acc{slot} = fma((scalar_t)({cstr}), xv * (yv * gov), acc{slot});')
                    ap('        }')
                ap('')

        for s, i in enumerate(targets):
            ap(f'        grad_w[gw_base + ((int64_t){i} << 5)] = acc{s};')

        ap('    }')
        ap('')

    ap('}')
    return "\n".join(lines)




def emit_split_launcher(bundle_name: str,
                  gradw_kernel: str,
                  gradx_kernel: str,
                  grady_kernel: str,
                  ) -> str:
    return rf'''
std::vector<torch::Tensor> {bundle_name}(
    torch::Tensor grad_out,
    torch::Tensor w,
    torch::Tensor x_all,
    torch::Tensor y,
    torch::Tensor src_idx,
    torch::Tensor dst_idx,
    torch::Tensor b_list,
    int64_t Iw,
    int64_t Ix,
    int64_t Ky,
    int64_t V)
{{
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x_all.is_cuda(), "x_all must be CUDA");
    TORCH_CHECK(y.is_cuda(), "y must be CUDA");

    auto B = b_list.numel() > 0 ? (int)b_list.numel() : (int)w.size(0);

    auto grad_w = torch::zeros_like(w);
    auto grad_x = torch::zeros_like(x_all);
    auto grad_y = torch::zeros_like(y);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    dim3 block(32);
    dim3 grid(B);

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{
        {gradw_kernel}<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            x_all.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_w.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list.numel() ? b_list.data_ptr<int32_t>() : nullptr,
            B, (int)Iw, (int)Ix, (int)Ky, (int)V);

        {gradx_kernel}<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list.numel() ? b_list.data_ptr<int32_t>() : nullptr,
            B, (int)Iw, (int)Ix, (int)Ky, (int)V);

        {grady_kernel}<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            x_all.data_ptr<scalar_t>(),
            grad_y.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list.numel() ? b_list.data_ptr<int32_t>() : nullptr,
            B, (int)Iw, (int)Ix, (int)Ky, (int)V);
    }});

    return {{grad_w, grad_x, grad_y}};
}}

TORCH_LIBRARY({bundle_name}_codegen, m) {{
    m.def("run", &{bundle_name});
}}
'''


def generate_full_uniform1d_bwd_split_cuda(
    i_list: List[int],
    j_list: List[int],
    k_list: List[int],
    v_list: List[int],
    coeff_list: List[float],
    *,
    bundle_name: str = "stp_edge_parallel_bwd_codegen",
    reg_budget: int = 64,
) -> str:
    assert len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)
    paths = [CGPath(i, j, k, v, c) for i, j, k, v, c in zip(i_list, j_list, k_list, v_list, coeff_list)]

    gradw_kernel = bundle_name + "_gradw"
    gradx_kernel = bundle_name + "_gradx"
    grady_kernel = bundle_name + "_grady"

    parts = [
        emit_preamble(),
        emit_gradw_kernel_segmented_unrolled(paths, gradw_kernel),
        emit_gradx_kernel_segmented_unrolled(paths, gradx_kernel),
        emit_grady_kernel(paths, grady_kernel, reg_budget=reg_budget),
        emit_split_launcher(bundle_name, gradw_kernel, gradx_kernel, grady_kernel),
    ]
    #return "\n".join(parts)
    code = '\n'.join(parts)
    file_name = f"{bundle_name}.cu"
    #Path(f"../../fasteq/cuda/src/uniform1d_codegen/{file_name}").write_text(code, encoding="utf-8")
    (out_dir / file_name).write_text(code, encoding="utf-8")