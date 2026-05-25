from __future__ import annotations

from collections import defaultdict, OrderedDict, Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
import math
from pathlib import Path
import struct
import numpy as np
import torch


# ============================================================
# Helpers
# ============================================================

def _to_int_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().tolist()]


def _to_float_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]

# =============================================================================
# Data structures
# =============================================================================
@dataclass(frozen=True)
class CGPath:
    i: int
    j: int
    k: int
    v: int
    c: float


# =============================================================================
# Generic helpers
# =============================================================================
def _stable_slot_map(vals) -> Dict[Any, int]:
    uniq = []
    seen = set()
    for x in vals:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    uniq = sorted(uniq)
    return {v: s for s, v in enumerate(uniq)}

def _count_pair_reuse(paths: List[CGPath]):
    cnt_wx = Counter()
    cnt_wy = Counter()
    cnt_xy = Counter()
    cnt_x = Counter()
    cnt_w = Counter()
    cnt_y = Counter()
    for p in paths:
        cnt_wx[(p.i, p.j)] += 1
        cnt_wy[(p.i, p.k)] += 1
        cnt_xy[(p.j, p.k)] += 1
        cnt_x[p.j] += 1
        cnt_w[p.i] += 1
        cnt_y[p.k] += 1
    return cnt_wx, cnt_wy, cnt_xy, cnt_x, cnt_w, cnt_y


def _phase_operand_cost(
    wi_vals: Set[int],
    x_vals: Set[int],
    y_vals: Set[int],
    pair_vals: Set[Tuple[int, int]],
    y_weight: float,
) -> float:
    return (
        float(len(wi_vals))
        + float(len(x_vals))
        + float(y_weight * len(y_vals))
        + float(len(pair_vals))
    )


def _extract_phase_sets_from_items(
    items: List[Tuple[int, CGPath]],
):
    w_set = {p.i for _, p in items}
    x_set = {p.j for _, p in items}
    y_set = {p.k for _, p in items}
    pair_wx_set = {(p.i, p.j) for _, p in items}
    return w_set, x_set, y_set, pair_wx_set


def _sort_items_for_wx_pair_reuse(items: List[Tuple[int, CGPath]]):
    items.sort(key=lambda it: (it[1].i, it[1].j, it[1].k, it[1].v, it[0]))

# =========================================================
# merge scoring
# =========================================================

def _merge_score_wx_seed(
    items_a: List[Tuple[int, CGPath]],
    items_b: List[Tuple[int, CGPath]],
    *,
    y_weight: float,
    r_rem: int,
    x_share_weight: float,
    w_share_weight: float,
    y_share_weight: float,
    pair_bonus_weight: float,
    overflow_penalty_weight: float,
) -> float:
    wa, xa, ya, pa = _extract_phase_sets_from_items(items_a)
    wb, xb, yb, pb = _extract_phase_sets_from_items(items_b)

    shared_x = len(xa & xb)
    shared_w = len(wa & wb)
    shared_y = len(ya & yb)

    # 合并前后 cost
    cost_a = _phase_operand_cost(wa, xa, ya, pa, y_weight)
    cost_b = _phase_operand_cost(wb, xb, yb, pb, y_weight)

    w_u = wa | wb
    x_u = xa | xb
    y_u = ya | yb
    p_u = pa | pb
    cost_u = _phase_operand_cost(w_u, x_u, y_u, p_u, y_weight)

    saved_cost = (cost_a + cost_b) - cost_u
    overflow = max(0.0, cost_u - float(r_rem))

    # 两个 wx seed phase 合并时，如果主 pair 数量没爆，默认给一点 bonus
    pair_bonus = 1.0 if len(p_u) <= r_rem else 0.0

    score = (
        x_share_weight * float(shared_x)
        + w_share_weight * float(shared_w)
        + y_share_weight * float(shared_y)
        + pair_bonus_weight * float(pair_bonus)
        + 1.5 * float(saved_cost)
        - overflow_penalty_weight * float(overflow)
    )
    return score


# =========================================================
# greedy merge on wx seeds
# =========================================================

def _build_wx_seed_clusters(
    paths: List[CGPath],
    cnt_wx: Counter,
) -> List[Dict[str, Any]]:
    buckets = defaultdict(list)
    for pid, p in enumerate(paths):
        buckets[(p.i, p.j)].append((pid, p))

    cluster_items = list(buckets.items())
    cluster_items.sort(key=lambda kv: (-cnt_wx[kv[0]], kv[0][0], kv[0][1]))

    seeds: List[Dict[str, Any]] = []
    for seed_id, ((i, j), items) in enumerate(cluster_items):
        _sort_items_for_wx_pair_reuse(items)
        seeds.append(
            {
                "seed_id": int(seed_id),
                "main_pair_kind": "wx",
                "seed_pair": (int(i), int(j)),
                "items": list(items),
            }
        )
    return seeds


def _greedy_merge_wx_seeds(
    seeds: List[Dict[str, Any]],
    *,
    y_weight: float,
    r_rem: int,
    phase_pair_max_slots: int,
    x_share_weight: float,
    w_share_weight: float,
    y_share_weight: float,
    pair_bonus_weight: float,
    overflow_penalty_weight: float,
    min_merge_score: float,
) -> List[Dict[str, Any]]:
    """
    目标：
      - 先保住 wx seed
      - 尽量把共享相同 x_j 的 seeds 合到同一个 phase
      - phase 内 pair 个数不超过 phase_pair_max_slots
      - 合并收益不足则停止
    """
    active = []
    for sd in seeds:
        active.append(
            {
                "main_pair_kind": "wx",
                "seed_pairs": [sd["seed_pair"]],
                "items": list(sd["items"]),
            }
        )

    while True:
        best_score = None
        best_pair = None

        n = len(active)
        for a in range(n):
            items_a = active[a]["items"]
            _, _, _, pa = _extract_phase_sets_from_items(items_a)

            for b in range(a + 1, n):
                items_b = active[b]["items"]
                _, _, _, pb = _extract_phase_sets_from_items(items_b)

                # 限制 phase-local wx pair cache 大小
                if len(pa | pb) > int(phase_pair_max_slots):
                    continue

                score = _merge_score_wx_seed(
                    items_a,
                    items_b,
                    y_weight=y_weight,
                    r_rem=r_rem,
                    x_share_weight=x_share_weight,
                    w_share_weight=w_share_weight,
                    y_share_weight=y_share_weight,
                    pair_bonus_weight=pair_bonus_weight,
                    overflow_penalty_weight=overflow_penalty_weight,
                )

                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (a, b)

        if best_pair is None:
            break
        if best_score is None or best_score < float(min_merge_score):
            break

        a, b = best_pair
        if a > b:
            a, b = b, a

        merged_items = list(active[a]["items"]) + list(active[b]["items"])
        _sort_items_for_wx_pair_reuse(merged_items)

        merged_seed_pairs = list(active[a]["seed_pairs"]) + list(active[b]["seed_pairs"])

        active[a] = {
            "main_pair_kind": "wx",
            "seed_pairs": merged_seed_pairs,
            "items": merged_items,
        }
        del active[b]

    return active


# =========================================================
# subphase split
# =========================================================

def _chunk_phase_items_by_budget_wx(
    items: List[Tuple[int, CGPath]],
    *,
    y_weight: float,
    r_rem: int,
) -> List[List[Tuple[int, CGPath]]]:
    chunks: List[List[Tuple[int, CGPath]]] = []
    cur: List[Tuple[int, CGPath]] = []

    cur_w = set()
    cur_x = set()
    cur_y = set()
    cur_p = set()

    def cost_of(w_set, x_set, y_set, p_set) -> float:
        return _phase_operand_cost(w_set, x_set, y_set, p_set, y_weight)

    for it in items:
        _, p = it
        nw = set(cur_w); nw.add(p.i)
        nx = set(cur_x); nx.add(p.j)
        ny = set(cur_y); ny.add(p.k)
        np = set(cur_p); np.add((p.i, p.j))

        if cur and cost_of(nw, nx, ny, np) > float(r_rem):
            chunks.append(cur)
            cur = [it]
            cur_w = {p.i}
            cur_x = {p.j}
            cur_y = {p.k}
            cur_p = {(p.i, p.j)}
        else:
            cur.append(it)
            cur_w = nw
            cur_x = nx
            cur_y = ny
            cur_p = np

    if cur:
        chunks.append(cur)

    return chunks


def _build_subphases_from_phase_items_wx(
    items: List[Tuple[int, CGPath]],
    *,
    out_acc_slot_map: Dict[int, int],
    wi_slot_map_phase: Dict[int, int],
    x_slot_map_phase: Dict[int, int],
    y_slot_map_phase: Dict[int, int],
    pair_slot_map_phase: Dict[Tuple[int, int], int],
    y_weight: float,
    r_rem: int,
) -> List[Dict[str, Any]]:
    ordered = list(items)
    _sort_items_for_wx_pair_reuse(ordered)

    chunks = _chunk_phase_items_by_budget_wx(
        ordered,
        y_weight=y_weight,
        r_rem=r_rem,
    )

    subphases: List[Dict[str, Any]] = []
    for sp_id, chunk in enumerate(chunks):
        sp_w = sorted({p.i for _, p in chunk})
        sp_x = sorted({p.j for _, p in chunk})
        sp_y = sorted({p.k for _, p in chunk})
        sp_p = sorted({(p.i, p.j) for _, p in chunk})

        reg_usage_inputs = _phase_operand_cost(set(sp_w), set(sp_x), set(sp_y), set(sp_p), y_weight)

        ops = []
        path_ids = []

        for pid, p in chunk:
            path_ids.append(pid)
            ops.append(
                {
                    "i": int(p.i),
                    "j": int(p.j),
                    "k": int(p.k),
                    "v": int(p.v),
                    "c": float(p.c),

                    "out_acc_slot": int(out_acc_slot_map[p.v]),
                    "wi_slot": int(wi_slot_map_phase[p.i]),
                    "x_slot": int(x_slot_map_phase[p.j]),
                    "y_slot": int(y_slot_map_phase[p.k]),
                    "pair_slot": int(pair_slot_map_phase[(p.i, p.j)]),
                }
            )

        subphases.append(
            {
                "subphase_id": int(sp_id),
                "wi_vals": list(sp_w),
                "x_vals": list(sp_x),
                "y_vals": list(sp_y),
                "pair_vals": list(sp_p),
                "path_ids": list(path_ids),
                "reg_usage_inputs": float(reg_usage_inputs),
                "reg_usage_inputs_rounded": int(round(reg_usage_inputs)),
                "ops": ops,
            }
        )

    return subphases


# =========================================================
# main scheduler: wx seed + shared-x merge
# =========================================================
# method1: operand-pair reuse maximization
def schedule_fwd_reuse_first_wx_seed_merge_x(
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
    reg_budget: int,
    *,
    mode: str = "u,u,,u",
    y_weight_scalar_mode: float = 0.5,
    y_weight_vector_mode: float = 1.0,

    phase_pair_max_slots: int = 8,

    # merge weights
    x_share_weight: float = 4.0,
    w_share_weight: float = 2.0,
    y_share_weight: float = 1.0,
    pair_bonus_weight: float = 0.5,
    overflow_penalty_weight: float = 3.0,
    min_merge_score: float = 0.25,

    sanity_check: bool = True,
) -> Dict[str, Any]:

    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

    if reg_budget <= 0:
        raise ValueError(f"reg_budget must be positive, got {reg_budget}")

    if phase_pair_max_slots <= 0:
        raise ValueError(f"phase_pair_max_slots must be > 0, got {phase_pair_max_slots}")

    paths = [
        CGPath(
            int(i_list[t]),
            int(j_list[t]),
            int(k_list[t]),
            int(v_list[t]),
            float(coeff_list[t]),
        )
        for t in range(len(i_list))
    ]

    uniq_i_all = sorted({p.i for p in paths})
    uniq_j_all = sorted({p.j for p in paths})
    uniq_k_all = sorted({p.k for p in paths})
    uniq_v_all = sorted({p.v for p in paths})

    out_acc_slot_map = _stable_slot_map(uniq_v_all)

    n_acc_out = len(uniq_v_all)
    n_acc_total = n_acc_out
    r_rem = reg_budget - n_acc_total
    if r_rem <= 0:
        print(f"reg_budget={reg_budget} is too small: need n_acc_out={n_acc_out}")
        r_rem = n_acc_out

    y_weight = float(y_weight_scalar_mode if mode == "u,u,,u" else y_weight_vector_mode)
    if y_weight <= 0:
        raise ValueError(f"y_weight must be positive, got {y_weight}")

    cnt_wx, cnt_wy, cnt_xy, cnt_x, cnt_w, cnt_y = _count_pair_reuse(paths)

    # 1) build wx seeds
    seeds = _build_wx_seed_clusters(paths, cnt_wx)

    # 2) merge seeds by shared-x/shared-w/shared-y
    merged_phases_raw = _greedy_merge_wx_seeds(
        seeds,
        y_weight=y_weight,
        r_rem=r_rem,
        phase_pair_max_slots=phase_pair_max_slots,
        x_share_weight=x_share_weight,
        w_share_weight=w_share_weight,
        y_share_weight=y_share_weight,
        pair_bonus_weight=pair_bonus_weight,
        overflow_penalty_weight=overflow_penalty_weight,
        min_merge_score=min_merge_score,
    )

    phases: List[Dict[str, Any]] = []
    covered: List[int] = []
    phase_id = 0

    for ph in merged_phases_raw:
        items = list(ph["items"])
        _sort_items_for_wx_pair_reuse(items)

        wi_vals_phase = sorted({p.i for _, p in items})
        x_vals_phase  = sorted({p.j for _, p in items})
        y_vals_phase  = sorted({p.k for _, p in items})
        pair_vals_phase = sorted({(p.i, p.j) for _, p in items})

        wi_slot_map_phase = _stable_slot_map(wi_vals_phase)
        x_slot_map_phase  = _stable_slot_map(x_vals_phase)
        y_slot_map_phase  = _stable_slot_map(y_vals_phase)
        pair_slot_map_phase = {pv: s for s, pv in enumerate(pair_vals_phase)}

        reg_usage_inputs_phase = _phase_operand_cost(
            set(wi_vals_phase),
            set(x_vals_phase),
            set(y_vals_phase),
            set(pair_vals_phase),
            y_weight,
        )

        subphases = _build_subphases_from_phase_items_wx(
            items,
            out_acc_slot_map=out_acc_slot_map,
            wi_slot_map_phase=wi_slot_map_phase,
            x_slot_map_phase=x_slot_map_phase,
            y_slot_map_phase=y_slot_map_phase,
            pair_slot_map_phase=pair_slot_map_phase,
            y_weight=y_weight,
            r_rem=r_rem,
        )

        phase_path_ids = []
        for sp in subphases:
            phase_path_ids.extend(sp["path_ids"])
        covered.extend(phase_path_ids)

        phases.append(
            {
                "phase_id": int(phase_id),
                "main_pair_kind": "wx",
                "seed_pairs": list(ph["seed_pairs"]),
                "seed_pair": list(ph["seed_pairs"])[0] if ph["seed_pairs"] else None,

                "pair_vals_phase": list(pair_vals_phase),
                "pair_slot_map_phase": dict(pair_slot_map_phase),

                "wi_vals_phase": list(wi_vals_phase),
                "x_vals_phase": list(x_vals_phase),
                "y_vals_phase": list(y_vals_phase),

                "wi_slot_map_phase": dict(wi_slot_map_phase),
                "x_slot_map_phase": dict(x_slot_map_phase),
                "y_slot_map_phase": dict(y_slot_map_phase),

                "path_ids": list(phase_path_ids),

                "reg_usage_inputs_phase": float(reg_usage_inputs_phase),
                "reg_usage_inputs_phase_rounded": int(round(reg_usage_inputs_phase)),

                "subphases": subphases,
            }
        )
        phase_id += 1

    if sanity_check:
        seen = sorted(covered)
        if seen != list(range(len(paths))):
            raise AssertionError("scheduler did not cover each path exactly once")

        for ph in phases:
            if len(ph["pair_vals_phase"]) > int(phase_pair_max_slots):
                raise AssertionError(
                    f"phase pair count exceeds phase_pair_max_slots: "
                    f"{len(ph['pair_vals_phase'])} > {phase_pair_max_slots}"
                )

            for sp in ph["subphases"]:
                if sp["reg_usage_inputs"] > float(r_rem) + 1e-12:
                    raise AssertionError(
                        f"subphase exceeds weighted reg budget: "
                        f"{sp['reg_usage_inputs']:.3f} > {r_rem}"
                    )

    schedule: Dict[str, Any] = {
        "kernel_mode": "fwd_fullacc_reuse_first_wx_seed_merge_x",
        "mode": str(mode),

        "reg_budget": int(reg_budget),
        "n_acc_out": int(n_acc_out),
        "n_acc_total": int(n_acc_total),
        "r_rem": int(r_rem),

        "y_weight": float(y_weight),
        "phase_pair_max_slots": int(phase_pair_max_slots),

        "merge_weights": {
            "x_share_weight": float(x_share_weight),
            "w_share_weight": float(w_share_weight),
            "y_share_weight": float(y_share_weight),
            "pair_bonus_weight": float(pair_bonus_weight),
            "overflow_penalty_weight": float(overflow_penalty_weight),
            "min_merge_score": float(min_merge_score),
        },

        "uniq_i_all": list(uniq_i_all),
        "uniq_j_all": list(uniq_j_all),
        "uniq_k_all": list(uniq_k_all),
        "uniq_v_all": list(uniq_v_all),

        "out_acc_slot_map": dict(out_acc_slot_map),

        "global_counts": {
            "cnt_x": dict(cnt_x),
            "cnt_w": dict(cnt_w),
            "cnt_y": dict(cnt_y),
            "cnt_wx": {str(k): int(v) for k, v in cnt_wx.items()},
            "cnt_wy": {str(k): int(v) for k, v in cnt_wy.items()},
            "cnt_xy": {str(k): int(v) for k, v in cnt_xy.items()},
        },

        "phases": phases,
    }
    return schedule


def emit_fused_fwd_kernel_from_reuse_first_schedule(
    schedule: Dict[str, Any],
    *,
    kernel_name: str,
    scalar_t: str = "float",
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

    if block_size != 32:
        raise ValueError("this emitter currently assumes block_size=32")

    uniq_v_all = list(schedule["uniq_v_all"])
    out_acc_slot_map = dict(schedule["out_acc_slot_map"])
    phases = list(schedule["phases"])

    max_wi_slots = 0
    max_x_slots = 0
    max_y_slots = 0
    max_pair_slots = 0

    for ph in phases:
        max_wi_slots = max(max_wi_slots, len(ph["wi_vals_phase"]))
        max_x_slots = max(max_x_slots, len(ph["x_vals_phase"]))
        max_y_slots = max(max_y_slots, len(ph["y_vals_phase"]))
        max_pair_slots = max(max_pair_slots, len(ph["pair_vals_phase"]))

    mode_scalar_y = mode == "u,u,,u"

    out_acc_by_slot = [None] * len(uniq_v_all)
    for v, s in out_acc_slot_map.items():
        out_acc_by_slot[s] = v

    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    def fmt_float(x: float) -> str:
        return repr(float(x))

    # ------------------------------------------------------------------
    # HIP / CUDA compatible header
    # ------------------------------------------------------------------
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
    ap("  using GPU_Guard = c10::cuda::CUDAGuard;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap('#include "cuda_utils.hpp"')
    ap("")

    # ------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------
    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    scalar_t* __restrict__ out,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
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

    ap("    const int e_orig = b_list ? b_list[e_local] : e_local;")
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
    ap("    const index_t w_base = (index_t)w_row  * (index_t)Iw * (index_t)U;")
    ap("    const index_t x_base = (index_t)x_row  * (index_t)Ix * (index_t)U;")

    if mode_scalar_y:
        ap("    const index_t y_base = (index_t)y_row  * (index_t)Ky;")
    else:
        ap("    const index_t y_base = (index_t)y_row  * (index_t)Ky * (index_t)U;")

    ap("    const index_t out_base = (index_t)out_row * (index_t)V  * (index_t)U;")
    ap("")

    ap("    // full-resident output accumulators across all phases")
    for slot_id in range(len(uniq_v_all)):
        ap(f"    scalar_t out_acc_v_{slot_id};")
    ap("")

    ap("    // phase-local operand / pair slots")
    for s in range(max_wi_slots):
        ap(f"    scalar_t wi_slot_{s};")
    for s in range(max_x_slots):
        ap(f"    scalar_t x_slot_{s};")
    for s in range(max_y_slots):
        ap(f"    scalar_t y_slot_{s};")
    for s in range(max_pair_slots):
        ap(f"    scalar_t pair_slot_{s};")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        int u = u_base + lane;")
    ap("        if (u < U) {")
    ap("")

    ap("            // reset output accumulators")
    for slot_id in range(len(uniq_v_all)):
        ap(f"            out_acc_v_{slot_id} = scalar_t(0);")
    ap("")

    # ------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------
    for ph in phases:
        phase_id = ph["phase_id"]
        main_pair_kind = ph["main_pair_kind"]

        wi_vals_phase = list(ph["wi_vals_phase"])
        x_vals_phase = list(ph["x_vals_phase"])
        y_vals_phase = list(ph["y_vals_phase"])
        pair_vals_phase = list(ph["pair_vals_phase"])

        wi_slot_map_phase = dict(ph["wi_slot_map_phase"])
        x_slot_map_phase = dict(ph["x_slot_map_phase"])
        y_slot_map_phase = dict(ph["y_slot_map_phase"])
        pair_slot_map_phase = dict(ph["pair_slot_map_phase"])

        ap(f"            // ===== phase {phase_id}: main_pair_kind={main_pair_kind} =====")
        ap("            {")

        # preload wi
        ap("                // phase-local wi preload")
        for i_val in wi_vals_phase:
            slot = wi_slot_map_phase[i_val]

            if iw_dim is not None and i_val >= iw_dim:
                raise ValueError(f"i={i_val} out of iw_dim={iw_dim}")

            if u_dim is not None:
                offset = int(i_val) * int(u_dim)
                ap(f"                wi_slot_{slot} = w[w_base + (index_t){offset} + (index_t)u];")
            else:
                ap(f"                wi_slot_{slot} = w[w_base + (index_t){i_val} * (index_t)U + (index_t)u];")
        ap("")

        # preload x
        ap("                // phase-local x preload")
        for j_val in x_vals_phase:
            slot = x_slot_map_phase[j_val]

            if ix_dim is not None and j_val >= ix_dim:
                raise ValueError(f"j={j_val} out of ix_dim={ix_dim}")

            if u_dim is not None:
                offset = int(j_val) * int(u_dim)
                ap(f"                x_slot_{slot} = x[x_base + (index_t){offset} + (index_t)u];")
            else:
                ap(f"                x_slot_{slot} = x[x_base + (index_t){j_val} * (index_t)U + (index_t)u];")
        ap("")

        # preload y
        ap("                // phase-local y preload")
        for k_val in y_vals_phase:
            slot = y_slot_map_phase[k_val]

            if ky_dim is not None and k_val >= ky_dim:
                raise ValueError(f"k={k_val} out of ky_dim={ky_dim}")

            if mode_scalar_y:
                ap(f"                y_slot_{slot} = y[y_base + (index_t){k_val}];")
            else:
                if u_dim is not None:
                    offset = int(k_val) * int(u_dim)
                    ap(f"                y_slot_{slot} = y[y_base + (index_t){offset} + (index_t)u];")
                else:
                    ap(f"                y_slot_{slot} = y[y_base + (index_t){k_val} * (index_t)U + (index_t)u];")
        ap("")

        # pair precompute
        ap(f"                // phase-local pair cache kind={main_pair_kind}")
        for pair_val in pair_vals_phase:
            pair_key = tuple(pair_val) if isinstance(pair_val, list) else pair_val
            pair_slot = pair_slot_map_phase[pair_key]

            if main_pair_kind == "wx":
                i_val, j_val = pair_val
                wi_slot = wi_slot_map_phase[i_val]
                x_slot = x_slot_map_phase[j_val]
                ap(f"                pair_slot_{pair_slot} = wi_slot_{wi_slot} * x_slot_{x_slot};")
            elif main_pair_kind == "wy":
                i_val, k_val = pair_val
                wi_slot = wi_slot_map_phase[i_val]
                y_slot = y_slot_map_phase[k_val]
                ap(f"                pair_slot_{pair_slot} = wi_slot_{wi_slot} * y_slot_{y_slot};")
            elif main_pair_kind == "xy":
                j_val, k_val = pair_val
                x_slot = x_slot_map_phase[j_val]
                y_slot = y_slot_map_phase[k_val]
                ap(f"                pair_slot_{pair_slot} = x_slot_{x_slot} * y_slot_{y_slot};")
            else:
                raise ValueError(f"bad main_pair_kind={main_pair_kind}")
        ap("")

        # subphases
        for sp in ph["subphases"]:
            subphase_id = sp["subphase_id"]
            ap(f"                // ---- subphase {subphase_id} ----")

            for op in sp["ops"]:
                out_acc_slot = int(op["out_acc_slot"])
                pair_slot = int(op["pair_slot"])
                wi_slot = int(op["wi_slot"])
                x_slot = int(op["x_slot"])
                y_slot = int(op["y_slot"])
                coeff = fmt_float(float(op["c"]))

                if main_pair_kind == "wx":
                    ap(
                        f"                out_acc_v_{out_acc_slot} += scalar_t({coeff}) * "
                        f"pair_slot_{pair_slot} * y_slot_{y_slot};"
                    )
                elif main_pair_kind == "wy":
                    ap(
                        f"                out_acc_v_{out_acc_slot} += scalar_t({coeff}) * "
                        f"pair_slot_{pair_slot} * x_slot_{x_slot};"
                    )
                elif main_pair_kind == "xy":
                    ap(
                        f"                out_acc_v_{out_acc_slot} += scalar_t({coeff}) * "
                        f"wi_slot_{wi_slot} * pair_slot_{pair_slot};"
                    )
                else:
                    raise ValueError(f"bad main_pair_kind={main_pair_kind}")
            ap("")

        ap("            }")
        ap("")

    # ------------------------------------------------------------------
    # write out
    # ------------------------------------------------------------------
    ap("            // write out")
    for slot_id, v_val in enumerate(out_acc_by_slot):
        if v_val is None:
            raise ValueError(f"missing out_acc slot {slot_id}")

        if v_dim is not None and v_val >= v_dim:
            raise ValueError(f"v={v_val} out of v_dim={v_dim}")

        if u_dim is not None:
            out_offset = int(v_val) * int(u_dim)
            idx_expr = f"out_base + (index_t){out_offset} + (index_t)u"
        else:
            idx_expr = f"out_base + (index_t){v_val} * (index_t)U + (index_t)u"

        if use_scatter:
            ap(f"            atomicAdd(&out[{idx_expr}], out_acc_v_{slot_id});")
        else:
            ap(f"            out[{idx_expr}] = out_acc_v_{slot_id};")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    # ------------------------------------------------------------------
    # index helper
    # ------------------------------------------------------------------
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
    ap("    bool y_ok = false;")
    ap("    if (mode_scalar_y) {")
    ap("        int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("        y_ok = mul_fits_int32(y_dim0, (int64_t)Ky);")
    ap("    } else {")
    ap("        int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("        y_ok = mul3_fits_int32(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    }")
    ap("    int64_t out_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool out_ok = mul3_fits_int32(out_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && out_ok;")
    ap("}")
    ap("")

    # ------------------------------------------------------------------
    # launchers
    # ------------------------------------------------------------------
    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        w, x, y, out,")
    ap("        src_idx, dst_idx, b_list,")
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
    ap("    const int32_t* b_list,")
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
    ap("            w, x, y, out, src_idx, dst_idx, b_list,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    ap("            w, x, y, out, src_idx, dst_idx, b_list,")
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
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    ap("        w, x, y, out, src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)



# Method3: output accumulator grouping may replace Method1

# ============================================================
# Data structures
# ============================================================

@dataclass
class PathOp:
    i: int
    j: int
    k: int
    v: int
    c: float

    wi_slot: int
    x_slot: int
    y_slot: int

    out_acc_slot: int


@dataclass
class PathPattern:
    pattern_id: int
    v: int

    wi_vals: Tuple[int, ...]
    x_vals: Tuple[int, ...]
    y_vals: Tuple[int, ...]

    wi_slot_map: Dict[int, int]
    x_slot_map: Dict[int, int]
    y_slot_map: Dict[int, int]

    path_ids: List[int]
    ops: List[PathOp]


@dataclass
class GroupPath:
    group_id: int
    group_type: str  # "xy" / "x" / "y" / "w" / "single"

    shared_wi_vals: Optional[List[int]]
    shared_x_vals: Optional[List[int]]
    shared_y_vals: Optional[List[int]]

    shared_wi_slot_map: Optional[Dict[int, int]]
    shared_x_slot_map: Optional[Dict[int, int]]
    shared_y_slot_map: Optional[Dict[int, int]]

    member_pattern_ids: List[int]
    members: List[PathPattern]


# ============================================================
# Utilities
# ============================================================


def _freeze_sorted(vals) -> Tuple[int, ...]:
    return tuple(sorted(set(int(x) for x in vals)))


def _bucket_by_key(items, key_fn):
    buckets: Dict[Any, List[Any]] = {}
    for x in items:
        k = key_fn(x)
        buckets.setdefault(k, []).append(x)
    return buckets


# ============================================================
# Step 1: Build base path patterns
#
# 基础规则：
#   pattern = same (v, i)
#
# 即：同一个输出 v 下，同一个 wi / i 的所有 raw path 构成一个基础 pattern。
# ============================================================

def build_base_path_patterns(
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
) -> Tuple[List[PathPattern], Dict[int, int], List[int]]:
    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

    raw_paths: List[Tuple[int, CGPath]] = [
        (
            pid,
            CGPath(
                int(i_list[pid]),
                int(j_list[pid]),
                int(k_list[pid]),
                int(v_list[pid]),
                float(coeff_list[pid]),
            ),
        )
        for pid in range(len(i_list))
    ]

    uniq_v_all = sorted({int(p.v) for _, p in raw_paths})
    out_acc_slot_map = _stable_slot_map(uniq_v_all)

    # 基础 pattern: group by (v, i)
    buckets = _bucket_by_key(raw_paths, key_fn=lambda it: (int(it[1].v), int(it[1].i)))

    patterns: List[PathPattern] = []
    pattern_id = 0

    for (vv, ii) in sorted(buckets.keys()):
        items = buckets[(vv, ii)]
        items = sorted(items, key=lambda it: (int(it[1].j), int(it[1].k), int(it[0])))

        wi_vals = _freeze_sorted([p.i for _, p in items])   # 对这个 builder 来说通常只有 1 个
        x_vals = _freeze_sorted([p.j for _, p in items])
        y_vals = _freeze_sorted([p.k for _, p in items])

        wi_slot_map = _stable_slot_map(list(wi_vals))
        x_slot_map = _stable_slot_map(list(x_vals))
        y_slot_map = _stable_slot_map(list(y_vals))

        ops: List[PathOp] = []
        path_ids: List[int] = []

        for pid, p in items:
            path_ids.append(int(pid))
            ops.append(
                PathOp(
                    i=int(p.i),
                    j=int(p.j),
                    k=int(p.k),
                    v=int(p.v),
                    c=float(p.c),
                    wi_slot=int(wi_slot_map[int(p.i)]),
                    x_slot=int(x_slot_map[int(p.j)]),
                    y_slot=int(y_slot_map[int(p.k)]),
                    out_acc_slot=int(out_acc_slot_map[int(p.v)]),
                )
            )

        patterns.append(
            PathPattern(
                pattern_id=int(pattern_id),
                v=int(vv),
                wi_vals=wi_vals,
                x_vals=x_vals,
                y_vals=y_vals,
                wi_slot_map=dict(wi_slot_map),
                x_slot_map=dict(x_slot_map),
                y_slot_map=dict(y_slot_map),
                path_ids=list(path_ids),
                ops=list(ops),
            )
        )
        pattern_id += 1

    return patterns, out_acc_slot_map, uniq_v_all


# ============================================================
# Step 2: Layered priority merge into GroupPath
#
# 优先级：
#   1) xy
#   2) x
#   3) y
#   4) w
#   5) single
# ============================================================

def build_group_paths_with_priority_merge(
    patterns: List[PathPattern],
) -> List[GroupPath]:
    remaining: Dict[int, PathPattern] = {p.pattern_id: p for p in patterns}
    groups: List[GroupPath] = []
    group_id = 0

    def emit_group(group_type: str, members: List[PathPattern]):
        nonlocal group_id
        if not members:
            return

        shared_wi_vals = None
        shared_x_vals = None
        shared_y_vals = None

        shared_wi_slot_map = None
        shared_x_slot_map = None
        shared_y_slot_map = None

        if group_type == "xy":
            shared_x_vals = list(members[0].x_vals)
            shared_y_vals = list(members[0].y_vals)
            shared_x_slot_map = dict(members[0].x_slot_map)
            shared_y_slot_map = dict(members[0].y_slot_map)

        elif group_type == "x":
            shared_x_vals = list(members[0].x_vals)
            shared_x_slot_map = dict(members[0].x_slot_map)

        elif group_type == "y":
            shared_y_vals = list(members[0].y_vals)
            shared_y_slot_map = dict(members[0].y_slot_map)

        elif group_type == "w":
            shared_wi_vals = list(members[0].wi_vals)
            shared_wi_slot_map = dict(members[0].wi_slot_map)

        groups.append(
            GroupPath(
                group_id=int(group_id),
                group_type=str(group_type),
                shared_wi_vals=shared_wi_vals,
                shared_x_vals=shared_x_vals,
                shared_y_vals=shared_y_vals,
                shared_wi_slot_map=shared_wi_slot_map,
                shared_x_slot_map=shared_x_slot_map,
                shared_y_slot_map=shared_y_slot_map,
                member_pattern_ids=[int(m.pattern_id) for m in members],
                members=list(members),
            )
        )
        group_id += 1

    # Round 1: XY
    round1 = list(remaining.values())
    xy_buckets = _bucket_by_key(round1, key_fn=lambda p: ("xy", p.x_vals, p.y_vals))
    for _, members in xy_buckets.items():
        if len(members) >= 2:
            emit_group("xy", members)
            for m in members:
                remaining.pop(m.pattern_id, None)

    # Round 2: X
    round2 = list(remaining.values())
    x_buckets = _bucket_by_key(round2, key_fn=lambda p: ("x", p.x_vals))
    for _, members in x_buckets.items():
        if len(members) >= 2:
            emit_group("x", members)
            for m in members:
                remaining.pop(m.pattern_id, None)

    # Round 3: Y
    round3 = list(remaining.values())
    y_buckets = _bucket_by_key(round3, key_fn=lambda p: ("y", p.y_vals))
    for _, members in y_buckets.items():
        if len(members) >= 2:
            emit_group("y", members)
            for m in members:
                remaining.pop(m.pattern_id, None)

    # Round 4: W
    round4 = list(remaining.values())
    w_buckets = _bucket_by_key(round4, key_fn=lambda p: ("w", p.wi_vals))
    for _, members in w_buckets.items():
        if len(members) >= 2:
            emit_group("w", members)
            for m in members:
                remaining.pop(m.pattern_id, None)

    # Round 5: single
    for p in remaining.values():
        emit_group("single", [p])

    return groups


# ============================================================
# Top-level schedule builder
# ============================================================

def schedule_fwd_path_group_priority_merge(
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
) -> Dict[str, Any]:
    patterns, out_acc_slot_map, uniq_v_all = build_base_path_patterns(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
    )

    groups = build_group_paths_with_priority_merge(patterns)

    return {
        "kernel_mode": "fwd_path_group_priority_merge",
        "merge_priority": ["xy", "x", "y", "w", "single"],
        "uniq_v_all": list(uniq_v_all),
        "out_acc_slot_map": dict(out_acc_slot_map),

        "path_patterns": [
            {
                "pattern_id": int(p.pattern_id),
                "v": int(p.v),

                "wi_vals": list(p.wi_vals),
                "x_vals": list(p.x_vals),
                "y_vals": list(p.y_vals),

                "wi_slot_map": dict(p.wi_slot_map),
                "x_slot_map": dict(p.x_slot_map),
                "y_slot_map": dict(p.y_slot_map),

                "path_ids": list(p.path_ids),
                "ops": [asdict(op) for op in p.ops],
            }
            for p in patterns
        ],

        "group_paths": [
            {
                "group_id": int(g.group_id),
                "group_type": str(g.group_type),

                "shared_wi_vals": g.shared_wi_vals,
                "shared_x_vals": g.shared_x_vals,
                "shared_y_vals": g.shared_y_vals,

                "shared_wi_slot_map": g.shared_wi_slot_map,
                "shared_x_slot_map": g.shared_x_slot_map,
                "shared_y_slot_map": g.shared_y_slot_map,

                "member_pattern_ids": list(g.member_pattern_ids),
                "members": [
                    {
                        "pattern_id": int(m.pattern_id),
                        "v": int(m.v),

                        "wi_vals": list(m.wi_vals),
                        "x_vals": list(m.x_vals),
                        "y_vals": list(m.y_vals),

                        "wi_slot_map": dict(m.wi_slot_map),
                        "x_slot_map": dict(m.x_slot_map),
                        "y_slot_map": dict(m.y_slot_map),

                        "path_ids": list(m.path_ids),
                        "ops": [asdict(op) for op in m.ops],
                    }
                    for m in g.members
                ],
            }
            for g in groups
        ],
    }


def emit_fused_fwd_kernel_from_group_schedule(
    schedule: Dict[str, Any],
    *,
    kernel_name: str,
    scalar_t: str,
    mode: str,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
    u_dim: int,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")
    if not isinstance(u_dim, int) or u_dim <= 0:
        raise ValueError(f"u_dim must be positive int, got {u_dim}")
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported, got {block_size}")

    uniq_v_all = [int(x) for x in schedule["uniq_v_all"]]
    out_acc_slot_map = {int(k): int(v) for k, v in schedule["out_acc_slot_map"].items()}
    group_paths = schedule["group_paths"]

    out_u_offsets = {vv: vv * u_dim for vv in uniq_v_all}

    w_row_stride_const = None if iw_dim is None else int(iw_dim) * u_dim
    x_row_stride_const = None if ix_dim is None else int(ix_dim) * u_dim
    y_row_stride_const = None if (mode == "u,u,,u" or ky_dim is None) else int(ky_dim) * u_dim
    out_row_stride_const = None if v_dim is None else int(v_dim) * u_dim

    def fmt_float(x: float) -> str:
        return repr(float(x))

    # --------------------------------------------------------
    # Find max slot counts across all groups / members
    # --------------------------------------------------------
    max_wi_slot = -1
    max_x_slot = -1
    max_y_slot = -1

    for g in group_paths:
        if g["shared_wi_slot_map"] is not None:
            for _, s in g["shared_wi_slot_map"].items():
                max_wi_slot = max(max_wi_slot, int(s))
        if g["shared_x_slot_map"] is not None:
            for _, s in g["shared_x_slot_map"].items():
                max_x_slot = max(max_x_slot, int(s))
        if g["shared_y_slot_map"] is not None:
            for _, s in g["shared_y_slot_map"].items():
                max_y_slot = max(max_y_slot, int(s))

        for m in g["members"]:
            for _, s in m["wi_slot_map"].items():
                max_wi_slot = max(max_wi_slot, int(s))
            for _, s in m["x_slot_map"].items():
                max_x_slot = max(max_x_slot, int(s))
            for _, s in m["y_slot_map"].items():
                max_y_slot = max(max_y_slot, int(s))

    lines: List[str] = []
    ap = lines.append

    def wi_expr(op: Dict[str, Any]) -> str:
        return f"wi_slot_{int(op['wi_slot'])}"

    def x_expr(op: Dict[str, Any]) -> str:
        return f"x_slot_{int(op['x_slot'])}"

    def y_expr(op: Dict[str, Any]) -> str:
        return f"y_slot_{int(op['y_slot'])}"

    def emit_preload_w(member: Dict[str, Any], indent: str):
        for i_val in member["wi_vals"]:
            slot = int(member["wi_slot_map"][i_val])
            ap(
                f"{indent}wi_slot_{slot} = "
                f"w[w_base + (index_t){int(i_val) * u_dim} + (index_t)u];"
            )

    def emit_preload_x_from_vals(x_vals, x_slot_map, indent: str):
        for j_val in x_vals:
            slot = int(x_slot_map[j_val])
            ap(
                f"{indent}x_slot_{slot} = "
                f"x[x_base + (index_t){int(j_val) * u_dim} + (index_t)u];"
            )

    def emit_preload_y_from_vals(y_vals, y_slot_map, indent: str):
        for k_val in y_vals:
            slot = int(y_slot_map[k_val])
            if mode == "u,u,,u":
                ap(f"{indent}y_slot_{slot} = y[y_base + (index_t){int(k_val)}];")
            else:
                ap(
                    f"{indent}y_slot_{slot} = "
                    f"y[y_base + (index_t){int(k_val) * u_dim} + (index_t)u];"
                )

    # --------------------------------------------------------
    # Headers
    # --------------------------------------------------------
    ap("#include <stdint.h>")
    ap("#include <cuda.h>")
    ap("#include <cuda_runtime.h>")
    ap("#include <torch/extension.h>")
    ap("#include <ATen/cuda/CUDAContext.h>")
    ap("#include <c10/cuda/CUDAGuard.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap('#include "cuda_utils.hpp"')
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    scalar_t* __restrict__ out,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    ap("")
    ap(f"    constexpr int U_CONST = {u_dim};")
    ap("    (void)U_CONST;")
    ap("    const int e_orig = b_list ? b_list[e_local] : e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")

    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e_orig];")
    if use_scatter:
        ap("    const int dst = dst_idx[e_orig];")
    ap("")

    if w_row_stride_const is not None:
        ap(f"    const index_t w_base = (index_t)w_row * (index_t){w_row_stride_const};")
    else:
        ap("    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if x_row_stride_const is not None:
        if use_x_src:
            ap(f"    const index_t x_base = (index_t)src * (index_t){x_row_stride_const};")
        else:
            ap(f"    const index_t x_base = (index_t)e_local * (index_t){x_row_stride_const};")
    else:
        if use_x_src:
            ap("    const index_t x_base = (index_t)src * (index_t)Ix * (index_t)U;")
        else:
            ap("    const index_t x_base = (index_t)e_local * (index_t)Ix * (index_t)U;")

    if mode == "u,u,,u":
        if use_y_src:
            ap("    const index_t y_base = (index_t)src * (index_t)Ky;")
        else:
            ap("    const index_t y_base = (index_t)e_orig * (index_t)Ky;")
    else:
        if y_row_stride_const is not None:
            if use_y_src:
                ap(f"    const index_t y_base = (index_t)src * (index_t){y_row_stride_const};")
            else:
                ap(f"    const index_t y_base = (index_t)e_orig * (index_t){y_row_stride_const};")
        else:
            if use_y_src:
                ap("    const index_t y_base = (index_t)src * (index_t)Ky * (index_t)U;")
            else:
                ap("    const index_t y_base = (index_t)e_orig * (index_t)Ky * (index_t)U;")

    if out_row_stride_const is not None:
        if use_scatter:
            ap(f"    const index_t out_base = (index_t)dst * (index_t){out_row_stride_const};")
        else:
            ap(f"    const index_t out_base = (index_t)e_orig * (index_t){out_row_stride_const};")
    else:
        if use_scatter:
            ap("    const index_t out_base = (index_t)dst * (index_t)V * (index_t)U;")
        else:
            ap("    const index_t out_base = (index_t)e_orig * (index_t)V * (index_t)U;")

    ap("")
    ap("    // full-resident output accumulators")
    for s in range(len(uniq_v_all)):
        ap(f"    scalar_t out_acc_v_{s};")
    ap("")

    ap("    // shared/local operand slots")
    for s in range(max_wi_slot + 1):
        ap(f"    scalar_t wi_slot_{s};")
    for s in range(max_x_slot + 1):
        ap(f"    scalar_t x_slot_{s};")
    for s in range(max_y_slot + 1):
        ap(f"    scalar_t y_slot_{s};")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        int u = u_base + lane;")
    ap("        if (u < U) {")
    ap("")

    ap("            // reset output accumulators")
    for s in range(len(uniq_v_all)):
        ap(f"            out_acc_v_{s} = scalar_t(0);")
    ap("")

    # --------------------------------------------------------
    # Emit group_path body
    # --------------------------------------------------------
    for g in group_paths:
        gid = int(g["group_id"])
        gtype = str(g["group_type"])

        ap(f"            // ===== group_path {gid}: type={gtype} =====")
        ap("            {")

        if gtype == "xy":
            if g["shared_x_vals"]:
                ap("                // shared x preload")
                emit_preload_x_from_vals(
                    g["shared_x_vals"],
                    g["shared_x_slot_map"],
                    "                ",
                )
            if g["shared_y_vals"]:
                ap("                // shared y preload")
                emit_preload_y_from_vals(
                    g["shared_y_vals"],
                    g["shared_y_slot_map"],
                    "                ",
                )

            for m in g["members"]:
                ap(f"                // member pattern {int(m['pattern_id'])}, v={int(m['v'])}")
                emit_preload_w(m, "                ")
                for op in m["ops"]:
                    wi = wi_expr(op)
                    xe = x_expr(op)
                    ye = y_expr(op)
                    c = fmt_float(op["c"])
                    out_slot = int(op["out_acc_slot"])
                    ap(
                        f"                out_acc_v_{out_slot} += scalar_t({c}) * {wi} * {xe} * {ye};"
                    )

        elif gtype == "x":
            if g["shared_x_vals"]:
                ap("                // shared x preload")
                emit_preload_x_from_vals(
                    g["shared_x_vals"],
                    g["shared_x_slot_map"],
                    "                ",
                )

            for m in g["members"]:
                ap(f"                // member pattern {int(m['pattern_id'])}, v={int(m['v'])}")
                emit_preload_w(m, "                ")
                emit_preload_y_from_vals(m["y_vals"], m["y_slot_map"], "                ")
                for op in m["ops"]:
                    wi = wi_expr(op)
                    xe = x_expr(op)
                    ye = y_expr(op)
                    c = fmt_float(op["c"])
                    out_slot = int(op["out_acc_slot"])
                    ap(
                        f"                out_acc_v_{out_slot} += scalar_t({c}) * {wi} * {xe} * {ye};"
                    )

        elif gtype == "y":
            if g["shared_y_vals"]:
                ap("                // shared y preload")
                emit_preload_y_from_vals(
                    g["shared_y_vals"],
                    g["shared_y_slot_map"],
                    "                ",
                )

            for m in g["members"]:
                ap(f"                // member pattern {int(m['pattern_id'])}, v={int(m['v'])}")
                emit_preload_w(m, "                ")
                emit_preload_x_from_vals(m["x_vals"], m["x_slot_map"], "                ")
                for op in m["ops"]:
                    wi = wi_expr(op)
                    xe = x_expr(op)
                    ye = y_expr(op)
                    c = fmt_float(op["c"])
                    out_slot = int(op["out_acc_slot"])
                    ap(
                        f"                out_acc_v_{out_slot} += scalar_t({c}) * {wi} * {xe} * {ye};"
                    )

        elif gtype == "w":
            if g["shared_wi_vals"]:
                ap("                // shared w preload")
                for i_val in g["shared_wi_vals"]:
                    slot = int(g["shared_wi_slot_map"][i_val])
                    ap(
                        f"                wi_slot_{slot} = "
                        f"w[w_base + (index_t){int(i_val) * u_dim} + (index_t)u];"
                    )

            for m in g["members"]:
                ap(f"                // member pattern {int(m['pattern_id'])}, v={int(m['v'])}")
                emit_preload_x_from_vals(m["x_vals"], m["x_slot_map"], "                ")
                emit_preload_y_from_vals(m["y_vals"], m["y_slot_map"], "                ")
                for op in m["ops"]:
                    wi = wi_expr(op)
                    xe = x_expr(op)
                    ye = y_expr(op)
                    c = fmt_float(op["c"])
                    out_slot = int(op["out_acc_slot"])
                    ap(
                        f"                out_acc_v_{out_slot} += scalar_t({c}) * {wi} * {xe} * {ye};"
                    )

        elif gtype == "single":
            for m in g["members"]:
                ap(f"                // single member pattern {int(m['pattern_id'])}, v={int(m['v'])}")
                emit_preload_w(m, "                ")
                emit_preload_x_from_vals(m["x_vals"], m["x_slot_map"], "                ")
                emit_preload_y_from_vals(m["y_vals"], m["y_slot_map"], "                ")
                for op in m["ops"]:
                    wi = wi_expr(op)
                    xe = x_expr(op)
                    ye = y_expr(op)
                    c = fmt_float(op["c"])
                    out_slot = int(op["out_acc_slot"])
                    ap(
                        f"                out_acc_v_{out_slot} += scalar_t({c}) * {wi} * {xe} * {ye};"
                    )

        else:
            raise ValueError(f"Unsupported group type: {gtype}")

        ap("            }")
        ap("")

    # --------------------------------------------------------
    # Final writeback
    # --------------------------------------------------------
    ap("            // final writeback")
    for s, vv in enumerate(uniq_v_all):
        if use_scatter:
            ap(
                f"            atomicAdd(&out[out_base + (index_t){out_u_offsets[vv]} + (index_t)u], "
                f"out_acc_v_{s});"
            )
        else:
            ap(
                f"            out[out_base + (index_t){out_u_offsets[vv]} + (index_t)u] = out_acc_v_{s};"
            )

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    # --------------------------------------------------------
    # Launcher
    # --------------------------------------------------------
    mode_scalar_y = "true" if mode == "u,u,,u" else "false"
    use_x_src_cpp = "true" if use_x_src else "false"
    use_y_src_cpp = "true" if use_y_src else "false"
    use_scatter_cpp = "true" if use_scatter else "false"

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
    ap("    bool y_ok = false;")
    ap("    if (mode_scalar_y) {")
    ap("        int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("        y_ok = mul_fits_int32(y_dim0, (int64_t)Ky);")
    ap("    } else {")
    ap("        int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("        y_ok = mul3_fits_int32(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    }")
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
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    cudaStream_t stream)")
    ap("{")
    ap("    dim3 block(32);")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        w, x, y, out,")
    ap("        src_idx, dst_idx, b_list,")
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
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    cudaStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {use_x_src_cpp};")
    ap(f"    constexpr bool kUseYSrc = {use_y_src_cpp};")
    ap(f"    constexpr bool kUseScatter = {use_scatter_cpp};")
    ap(f"    constexpr bool kModeScalarY = {mode_scalar_y};")
    ap("    if (should_use_int32_index_fwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    ap("            w, x, y, out, src_idx, dst_idx, b_list,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    ap("            w, x, y, out, src_idx, dst_idx, b_list,")
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
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    cudaStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    ap("        w, x, y, out, src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)



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
        param_lines.append("    torch::Tensor b_list")

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
    if use_scatter:
        blist_logic = r'''
    const int32_t* b_list_ptr = nullptr;
    if (b_list.defined() && b_list.numel() > 0) {
        TORCH_CHECK(b_list.is_cuda(), "b_list must be CUDA/HIP");
        TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list must be int32");
        TORCH_CHECK((int)b_list.numel() == B, "b_list must be [B]");
        b_list_ptr = (const int32_t*)b_list.data_ptr<int32_t>();
    }
'''
    else:
        blist_logic = "    const int32_t* b_list_ptr = nullptr;\n"

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
    //   b_list : [B] int32 optional, enabled when scatter is used

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
                b_list_ptr,
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    }});

    GPU_KERNEL_LAUNCH_CHECK();

    return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} forward jit impl");
}}
'''


def generate_code_uniform1d_fwd(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    mode: str = "u,u,,u",
    out_path: str = "generated_uniform1d_fwd.cu",
    kernel_name: str = "stp_codegen_two_warp_vgroup",
    scalar_t: str = "float",
):
    """
    input_indices:
        - key 1 exists: x uses src indexing
        - key 2 exists: y uses src indexing

    output_indices:
        - key 0 exists: scatter to dst_idx
        - else: no scatter, output shape is [B, V, U]
    """
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    kernel_name = f"uniform1d_u{u_dim}_path{P}_{mode_str}_{layout_tag}_fwd"

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)
    

    """ schedule = schedule_fwd_path_group_priority_merge(
        i_list=i_cpu,
        j_list=j_cpu,
        k_list=k_cpu,
        v_list=v_cpu,
        coeff_list=c_cpu,
    )

    code = emit_fused_fwd_kernel_from_group_schedule(
        schedule,
        kernel_name=kernel_name,
        scalar_t=scalar_t,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        u_dim=u_dim,
        iw_dim=None,
        ix_dim=None,
        ky_dim=None,
        v_dim=None,
        block_size=32,
    )"""



    schedule = schedule_fwd_reuse_first_wx_seed_merge_x(
        i_list=i_cpu,
        j_list=j_cpu,
        k_list=k_cpu,
        v_list=v_cpu,
        coeff_list=c_cpu,
        reg_budget=128,
        mode=mode, 
    )

    code = emit_fused_fwd_kernel_from_reuse_first_schedule(
        schedule,
        kernel_name=kernel_name,
        scalar_t=scalar_t,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        u_dim=u_dim,
        iw_dim=None,
        ix_dim=None,
        ky_dim=None,
        v_dim=None,
        block_size=32,
    )
    code = code + "\n" + emit_launcher(
        bundle_name=kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )
    return code