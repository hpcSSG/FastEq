from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
import math
from pathlib import Path


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

from typing import Dict, Any, List


def _fmt_coeff(cc: float, scalar_t: str) -> str:
    if abs(cc - 1.0) < 1e-12:
        return "1"
    if abs(cc + 1.0) < 1e-12:
        return "-1"
    if scalar_t == "double":
        return f"{cc:.17g}"
    return f"{cc:.9g}f"


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
    ap("")

    ap("template <typename T>")
    ap("__device__ __forceinline__ T warp_sum_xor(T v) {")
    ap("    #pragma unroll")
    ap("    for (int mask = 16; mask > 0; mask >>= 1) {")
    ap("        v += __shfl_xor_sync(0xffffffff, v, mask);")
    ap("    }")
    ap("    return v;")
    ap("}")
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
    file_name = f"{kernel_name}.cu"
    Path(file_name).write_text(code, encoding="utf-8")
    return code