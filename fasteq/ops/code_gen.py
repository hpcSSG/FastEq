#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import torch


# ============================================================
# Helpers
# ============================================================

def _to_int_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().tolist()]


def _to_float_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]


def _ordered_unique(xs):
    out = []
    seen = set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _fmt_coeff(c: float, scalar_t: str = "float") -> str:
    # CUDA literal formatting
    if scalar_t == "float":
        return f"{c:.9g}f"
    return f"{c:.17g}"


# ============================================================
# Reorder logic
# ============================================================

def reorder_groups_for_reuse(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    wj: float = 4.0,
    wk: float = 1.0,
    penalty_vdist: float = 0.05,
):
    """
    先按 v 分组，再对 group 重排，使相邻 group 的 j/k 集合尽量重叠。
    最后每个 group 内按 (j, k) 排序。
    """
    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    device = i_list.device
    dtype_i = i_list.dtype
    dtype_c = coeff_list.dtype

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    # 按 v 分组
    groups: "OrderedDict[int, List[Tuple[int,int,int,int,float]]]" = OrderedDict()
    for ii, jj, kk, vv, cc in zip(i_cpu, j_cpu, k_cpu, v_cpu, c_cpu):
        groups.setdefault(vv, []).append((ii, jj, kk, vv, cc))

    group_keys = list(groups.keys())

    # 构建每组的 j/k 集合
    J: Dict[int, set] = {}
    K: Dict[int, set] = {}
    for vv, items in groups.items():
        J[vv] = set(x[1] for x in items)
        K[vv] = set(x[2] for x in items)

    def score(v1: int, v2: int) -> float:
        sj = len(J[v1] & J[v2])
        sk = len(K[v1] & K[v2])
        return wj * sj + wk * sk - penalty_vdist * abs(v1 - v2)

    # 贪心组排序
    start = max(group_keys, key=lambda vv: (len(groups[vv]), len(J[vv]), len(K[vv])))
    unvisited = set(group_keys)
    order = [start]
    unvisited.remove(start)

    cur = start
    while unvisited:
        nxt = max(
            unvisited,
            key=lambda vv: (score(cur, vv), len(groups[vv]), len(J[vv]), len(K[vv]))
        )
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    # 组内按 (j, k) 排序
    reordered = []
    for vv in order:
        items = groups[vv]
        items = sorted(items, key=lambda x: (x[1], x[2]))
        reordered.extend(items)

    i2 = torch.tensor([x[0] for x in reordered], device=device, dtype=dtype_i)
    j2 = torch.tensor([x[1] for x in reordered], device=device, dtype=dtype_i)
    k2 = torch.tensor([x[2] for x in reordered], device=device, dtype=dtype_i)
    v2 = torch.tensor([x[3] for x in reordered], device=device, dtype=dtype_i)
    c2 = torch.tensor([x[4] for x in reordered], device=device, dtype=dtype_c)

    return i2, j2, k2, v2, c2, order


# ============================================================
# Grouping for codegen
# ============================================================

def build_v_groups(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
):
    """
    输入已排序的 tensor，按 v 分组。
    """
    i_py = _to_int_list(i_list)
    j_py = _to_int_list(j_list)
    k_py = _to_int_list(k_list)
    v_py = _to_int_list(v_list)
    c_py = _to_float_list(coeff_list)

    groups = OrderedDict()
    for ii, jj, kk, vv, cc in zip(i_py, j_py, k_py, v_py, c_py):
        if vv not in groups:
            groups[vv] = {"i": ii, "terms": []}
        else:
            if groups[vv]["i"] != ii:
                raise ValueError(f"same v={vv} maps to different i: {groups[vv]['i']} vs {ii}")
        groups[vv]["terms"].append((jj, kk, cc))
    return groups


def split_groups_into_two_warps(groups: OrderedDict):
    """
    按 term 数量尽量均衡地把 v 组分给两个 warp。
    """
    items = [(vv, len(info["terms"])) for vv, info in groups.items()]

    warp0_vs = []
    warp1_vs = []
    load0 = 0
    load1 = 0

    # 大组优先分配
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    for vv, cost in items_sorted:
        if load0 <= load1:
            warp0_vs.append(vv)
            load0 += cost
        else:
            warp1_vs.append(vv)
            load1 += cost

    order = list(groups.keys())
    pos = {vv: idx for idx, vv in enumerate(order)}
    warp0_vs.sort(key=lambda vv: pos[vv])
    warp1_vs.sort(key=lambda vv: pos[vv])

    return warp0_vs, warp1_vs


def emit_two_warp_vgroup_forward_kernel(
    groups: OrderedDict,
    kernel_name: str = "stp_codegen_two_warp_vgroup",
    scalar_t: str = "float",
) -> str:
    """
    生成一个固定 2 warp 的 tiled-U forward kernel:
      - warp0 处理一部分 v
      - warp1 处理另一部分 v
      - blockIdx.x -> b
      - blockIdx.y -> u tile (32 channels)
    支持 U % 32 == 0
    """
    warp0_vs, warp1_vs = split_groups_into_two_warps(groups)

    lines: List[str] = []
    ap = lines.append

    ap("#include <stdint.h>")
    ap("#include <cuda_runtime.h>")
    ap("")
    ap("template <typename scalar_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")        # [B, Iw, U]
    ap("    const scalar_t* __restrict__ x_all,")    # [S, Ix, U]
    ap("    const scalar_t* __restrict__ y,")        # [B, Ky, 1] flatten as [B,Ky]
    ap("    scalar_t* __restrict__ out,")            # [S, V, U]
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int Iw, int Ix, int Ky, int V, int U)")
    ap("{")
    ap("    int b_global = (int)blockIdx.x;")
    ap("    int ublk     = (int)blockIdx.y;")
    ap("    if (b_global >= B) return;")
    ap("    int b = b_list ? b_list[b_global] : b_global;")
    ap("")
    ap("    int tid  = threadIdx.x;")
    ap("    int lane = tid & 31;")
    ap("    int warp = tid >> 5;")
    ap("    if (warp >= 2) return;")
    ap("")
    ap("    int u = (ublk << 5) + lane;")
    ap("    if (u >= U) return;")
    ap("")
    ap("    int src = src_idx[b];")
    ap("    int dst = dst_idx[b];")
    ap("")
    ap("    int64_t y_base = (int64_t)b * Ky;")
    ap("")

    def emit_warp_body(warp_id: int, warp_vs: List[int]):
        ap(f"    if (warp == {warp_id}) {{")
        if not warp_vs:
            ap("        return;")
            ap("    }")
            ap("")
            return

        uniq_i = []
        uniq_j = []
        uniq_k = []
        seen_i = set()
        seen_j = set()
        seen_k = set()

        for vv in warp_vs:
            ii = groups[vv]["i"]
            if ii not in seen_i:
                seen_i.add(ii)
                uniq_i.append(ii)

            for jj, kk, _ in groups[vv]["terms"]:
                if jj not in seen_j:
                    seen_j.add(jj)
                    uniq_j.append(jj)
                if kk not in seen_k:
                    seen_k.add(kk)
                    uniq_k.append(kk)

        ap("        // preload w(i, u)")
        for ii in uniq_i:
            ap(f"        scalar_t wi_{ii} = w[((int64_t)b * Iw + {ii}) * (int64_t)U + u];")
        ap("")

        ap("        // preload x(j, u)")
        for jj in uniq_j:
            ap(f"        scalar_t xj_{jj} = x_all[((int64_t)src * Ix + {jj}) * (int64_t)U + u];")
        ap("")

        ap("        // preload y(k)")
        for kk in uniq_k:
            ap(f"        scalar_t yk_{kk} = y[y_base + {kk}];")
        ap("")

        ap("        // per-v accumulation")
        for vv in warp_vs:
            ii = groups[vv]["i"]
            terms = groups[vv]["terms"]
            ap(f"        scalar_t sum_v_{vv} = scalar_t(0);")
            for jj, kk, cc in terms:
                if abs(cc - 1.0) < 1e-12:
                    ap(f"        sum_v_{vv} += wi_{ii} * xj_{jj} * yk_{kk};")
                elif abs(cc + 1.0) < 1e-12:
                    ap(f"        sum_v_{vv} -= wi_{ii} * xj_{jj} * yk_{kk};")
                else:
                    cstr = _fmt_coeff(cc, scalar_t)
                    ap(f"        sum_v_{vv} += scalar_t({cstr}) * wi_{ii} * xj_{jj} * yk_{kk};")
            ap("")

        ap("        // writeback")
        for vv in warp_vs:
            ap(f"        atomicAdd(&out[((int64_t)dst * V + {vv}) * (int64_t)U + u], sum_v_{vv});")
        ap("    }")
        ap("")

    emit_warp_body(0, warp0_vs)
    emit_warp_body(1, warp1_vs)

    ap("}")
    ap("")
    ap("// launcher helper")
    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x_all,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int Iw, int Ix, int Ky, int V, int U,")
    ap("    cudaStream_t stream)")
    ap("{")
    ap("    dim3 block(64);  // 2 warps")
    ap("    dim3 grid(B, (U + 31) / 32);")
    ap(f"    {kernel_name}<scalar_t><<<grid, block, 0, stream>>>(")
    ap("        w, x_all, y, out, src_idx, dst_idx, b_list,")
    ap("        B, Iw, Ix, Ky, V, U);")
    ap("}")

    return "\n".join(lines)


def generate_code_uniform1d_fwd(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    u_dim: int = 1,
    out_path: str = "generated_uniform1d_fwd.cu",
    kernel_name: str = "stp_codegen_two_warp_vgroup",
    scalar_t: str = "float",
    reorder_groups: bool = True,
):
    """
    reorder_groups:
    True  -> 先做 group-level 排序，提高 j/k 复用
    False -> 保持原始 v 顺序，仅组内按 (j,k) 排序
    """
    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    out_path = f"uniform1d_codegen_path{P}_u{u_dim}_fwd.cu"

    kernel_name = f"uniform1d_codegen_two_warp_vgroup_path{P}_u{u_dim}_fwd"

    if reorder_groups:
        i2, j2, k2, v2, c2, group_order = reorder_groups_for_reuse(
            i_list, j_list, k_list, v_list, coeff_list
        )
    else:
        # 只做 v 分组 + 组内 (j,k) 排序
        i_cpu = _to_int_list(i_list)
        j_cpu = _to_int_list(j_list)
        k_cpu = _to_int_list(k_list)
        v_cpu = _to_int_list(v_list)
        c_cpu = _to_float_list(coeff_list)

        groups = OrderedDict()
        for ii, jj, kk, vv, cc in zip(i_cpu, j_cpu, k_cpu, v_cpu, c_cpu):
            groups.setdefault(vv, []).append((ii, jj, kk, vv, cc))

        reordered = []
        group_order = list(groups.keys())
        for vv in group_order:
            items = sorted(groups[vv], key=lambda x: (x[1], x[2]))
            reordered.extend(items)

        device = i_list.device
        i2 = torch.tensor([x[0] for x in reordered], device=device, dtype=i_list.dtype)
        j2 = torch.tensor([x[1] for x in reordered], device=device, dtype=j_list.dtype)
        k2 = torch.tensor([x[2] for x in reordered], device=device, dtype=k_list.dtype)
        v2 = torch.tensor([x[3] for x in reordered], device=device, dtype=v_list.dtype)
        c2 = torch.tensor([x[4] for x in reordered], device=device, dtype=coeff_list.dtype)

    groups = build_v_groups(i2, j2, k2, v2, c2)
    code = emit_two_warp_vgroup_forward_kernel(groups, kernel_name=kernel_name, scalar_t=scalar_t)
    Path(out_path).write_text(code, encoding="utf-8")

    warp0_vs, warp1_vs = split_groups_into_two_warps(groups)
    stats = {
        "num_paths": int(P),
        "num_v_groups": int(len(groups)),
        "group_order": list(groups.keys()),
        "warp0_vs": warp0_vs,
        "warp1_vs": warp1_vs,
        "out_path": str(out_path),
    }
    return stats

def emit_two_warp_vgroup_backward_kernel(
    groups: OrderedDict,
    kernel_name: str = "stp_codegen_two_warp_vgroup_bwd",
    scalar_t: str = "float",
) -> str:
    warp0_vs, warp1_vs = split_groups_into_two_warps(groups)

    lines: List[str] = []
    ap = lines.append

    ap("#include <stdint.h>")
    ap("#include <cuda_runtime.h>")
    ap("")
    ap("template <typename T>")
    ap("__device__ __forceinline__ T warp_sum(T v) {")
    ap("    #pragma unroll")
    ap("    for (int off = 16; off > 0; off >>= 1) {")
    ap("        v += __shfl_down_sync(0xffffffff, v, off);")
    ap("    }")
    ap("    return v;")
    ap("}")
    ap("")
    ap("template <typename scalar_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")          # [B, Iw, U]
    ap("    const scalar_t* __restrict__ x_all,")      # [S, Ix, U]
    ap("    const scalar_t* __restrict__ y,")          # [B, Ky, 1]
    ap("    const scalar_t* __restrict__ grad_out,")   # [S, V, U]
    ap("    scalar_t* __restrict__ grad_w,")           # [B, Iw, U]
    ap("    scalar_t* __restrict__ grad_x,")           # [S, Ix, U]
    ap("    scalar_t* __restrict__ grad_y,")           # [B, Ky, 1]
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int Iw, int Ix, int Ky, int V, int U)")
    ap("{")
    ap("    int b_global = (int)blockIdx.x;")
    ap("    int ublk     = (int)blockIdx.y;")
    ap("    if (b_global >= B) return;")
    ap("")
    ap("    int b = b_list ? b_list[b_global] : b_global;")
    ap("    int tid  = threadIdx.x;")
    ap("    int lane = tid & 31;")
    ap("    int warp = tid >> 5;")
    ap("    if (warp >= 2) return;")
    ap("")
    ap("    int u = (ublk << 5) + lane;")
    ap("    if (u >= U) return;")
    ap("")
    ap("    int src = src_idx[b];")
    ap("    int dst = dst_idx[b];")
    ap("")
    ap("    // flattened row-major offsets with true U stride")
    ap("    int64_t y_base = (int64_t)b * Ky;")
    ap("")

    def emit_warp_body(warp_id: int, warp_vs: List[int]):
        ap(f"    if (warp == {warp_id}) {{")
        if not warp_vs:
            ap("        return;")
            ap("    }")
            ap("")
            return

        uniq_i, uniq_j, uniq_k = [], [], []
        seen_i, seen_j, seen_k = set(), set(), set()

        for vv in warp_vs:
            ii = groups[vv]["i"]
            if ii not in seen_i:
                seen_i.add(ii)
                uniq_i.append(ii)
            for jj, kk, _ in groups[vv]["terms"]:
                if jj not in seen_j:
                    seen_j.add(jj)
                    uniq_j.append(jj)
                if kk not in seen_k:
                    seen_k.add(kk)
                    uniq_k.append(kk)

        ap("        // preload w(i, u)")
        for ii in uniq_i:
            ap(f"        scalar_t wi_{ii} = w[((int64_t)b * Iw + {ii}) * (int64_t)U + u];")
        ap("")
        ap("        // preload x(j, u)")
        for jj in uniq_j:
            ap(f"        scalar_t xj_{jj} = x_all[((int64_t)src * Ix + {jj}) * (int64_t)U + u];")
        ap("")
        ap("        // preload y(k)")
        for kk in uniq_k:
            ap(f"        scalar_t yk_{kk} = y[y_base + {kk}];")
        ap("")
        ap("        // preload grad_out(v, u)")
        for vv in warp_vs:
            ap(f"        scalar_t go_v_{vv} = grad_out[((int64_t)dst * V + {vv}) * (int64_t)U + u];")
        ap("")

        ap("        // grad_w accumulate by unique i")
        for ii in uniq_i:
            ap(f"        scalar_t gw_acc_i_{ii} = scalar_t(0);")
            for vv in warp_vs:
                if groups[vv]["i"] != ii:
                    continue
                for jj, kk, cc in groups[vv]["terms"]:
                    if abs(cc - 1.0) < 1e-12:
                        ap(f"        gw_acc_i_{ii} += xj_{jj} * yk_{kk} * go_v_{vv};")
                    elif abs(cc + 1.0) < 1e-12:
                        ap(f"        gw_acc_i_{ii} -= xj_{jj} * yk_{kk} * go_v_{vv};")
                    else:
                        cstr = _fmt_coeff(cc, scalar_t)
                        ap(f"        gw_acc_i_{ii} += scalar_t({cstr}) * xj_{jj} * yk_{kk} * go_v_{vv};")
            ap(f"        atomicAdd(&grad_w[((int64_t)b * Iw + {ii}) * (int64_t)U + u], gw_acc_i_{ii});")
            ap("")

        ap("        // grad_x accumulate by unique j")
        j_terms = defaultdict(list)
        for vv in warp_vs:
            ii = groups[vv]["i"]
            for jj, kk, cc in groups[vv]["terms"]:
                j_terms[jj].append((vv, ii, kk, cc))

        for jj in uniq_j:
            ap(f"        scalar_t gx_acc_j_{jj} = scalar_t(0);")
            for vv, ii, kk, cc in j_terms[jj]:
                if abs(cc - 1.0) < 1e-12:
                    ap(f"        gx_acc_j_{jj} += wi_{ii} * yk_{kk} * go_v_{vv};")
                elif abs(cc + 1.0) < 1e-12:
                    ap(f"        gx_acc_j_{jj} -= wi_{ii} * yk_{kk} * go_v_{vv};")
                else:
                    cstr = _fmt_coeff(cc, scalar_t)
                    ap(f"        gx_acc_j_{jj} += scalar_t({cstr}) * wi_{ii} * yk_{kk} * go_v_{vv};")
            ap(f"        atomicAdd(&grad_x[((int64_t)src * Ix + {jj}) * (int64_t)U + u], gx_acc_j_{jj});")
            ap("")

        ap("        // grad_y accumulate by unique k, reduced across current 32-channel tile")
        k_terms = defaultdict(list)
        for vv in warp_vs:
            ii = groups[vv]["i"]
            for jj, kk, cc in groups[vv]["terms"]:
                k_terms[kk].append((vv, ii, jj, cc))

        for kk in uniq_k:
            ap(f"        scalar_t gy_lane_k_{kk} = scalar_t(0);")
            for vv, ii, jj, cc in k_terms[kk]:
                if abs(cc - 1.0) < 1e-12:
                    ap(f"        gy_lane_k_{kk} += wi_{ii} * xj_{jj} * go_v_{vv};")
                elif abs(cc + 1.0) < 1e-12:
                    ap(f"        gy_lane_k_{kk} -= wi_{ii} * xj_{jj} * go_v_{vv};")
                else:
                    cstr = _fmt_coeff(cc, scalar_t)
                    ap(f"        gy_lane_k_{kk} += scalar_t({cstr}) * wi_{ii} * xj_{jj} * go_v_{vv};")
            ap(f"        scalar_t gy_sum_k_{kk} = warp_sum(gy_lane_k_{kk});")
            ap(f"        if (lane == 0) atomicAdd(&grad_y[y_base + {kk}], gy_sum_k_{kk});")
            ap("")

        ap("    }")
        ap("")

    emit_warp_body(0, warp0_vs)
    emit_warp_body(1, warp1_vs)

    ap("}")
    ap("")
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
    ap("    dim3 block(64);")
    ap("    dim3 grid(B, (U + 31) / 32);")
    ap(f"    {kernel_name}<scalar_t><<<grid, block, 0, stream>>>(")
    ap("        w, x_all, y, grad_out, grad_w, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list, B, Iw, Ix, Ky, V, U);")
    ap("}")

    return '\n'.join(lines)


from collections import defaultdict
from typing import List, OrderedDict


def emit_blocku_vgroup_backward_kernel(
    groups: OrderedDict,
    kernel_name: str = "stp_codegen_blocku_vgroup_bwd",
    scalar_t: str = "float",
) -> str:
    all_vs = list(groups.keys())

    # 全局 unique i/j/k，按首次出现顺序保留
    uniq_i, uniq_j, uniq_k = [], [], []
    seen_i, seen_j, seen_k = set(), set(), set()

    for vv in all_vs:
        ii = groups[vv]["i"]
        if ii not in seen_i:
            seen_i.add(ii)
            uniq_i.append(ii)
        for jj, kk, _ in groups[vv]["terms"]:
            if jj not in seen_j:
                seen_j.add(jj)
                uniq_j.append(jj)
            if kk not in seen_k:
                seen_k.add(kk)
                uniq_k.append(kk)

    # 预先整理 j / k 的反向项
    j_terms = defaultdict(list)  # j -> [(v, i, k, c), ...]
    k_terms = defaultdict(list)  # k -> [(v, i, j, c), ...]
    i_terms = defaultdict(list)  # i -> [(v, j, k, c), ...]
    for vv in all_vs:
        ii = groups[vv]["i"]
        for jj, kk, cc in groups[vv]["terms"]:
            i_terms[ii].append((vv, jj, kk, cc))
            j_terms[jj].append((vv, ii, kk, cc))
            k_terms[kk].append((vv, ii, jj, cc))

    num_uniq_k = len(uniq_k)

    lines: List[str] = []
    ap = lines.append

    ap("#include <stdint.h>")
    ap("#include <cuda_runtime.h>")
    ap("")
    ap("template <typename T>")
    ap("__device__ __forceinline__ T warp_sum(T v) {")
    ap("    #pragma unroll")
    ap("    for (int off = 16; off > 0; off >>= 1) {")
    ap("        v += __shfl_down_sync(0xffffffff, v, off);")
    ap("    }")
    ap("    return v;")
    ap("}")
    ap("")
    ap("template <typename scalar_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")          # [B, Iw, U]
    ap("    const scalar_t* __restrict__ x_all,")      # [S, Ix, U]
    ap("    const scalar_t* __restrict__ y,")          # [B, Ky]
    ap("    const scalar_t* __restrict__ grad_out,")   # [S, V, U]
    ap("    scalar_t* __restrict__ grad_w,")           # [B, Iw, U]
    ap("    scalar_t* __restrict__ grad_x,")           # [S, Ix, U]
    ap("    scalar_t* __restrict__ grad_y,")           # [B, Ky]
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int Iw, int Ix, int Ky, int V, int U)")
    ap("{")
    ap("    int b_global = (int)blockIdx.x;")
    ap("    if (b_global >= B) return;")
    ap("")
    ap("    int b   = b_list ? b_list[b_global] : b_global;")
    ap("    int tid = (int)threadIdx.x;")
    ap("    if (tid >= U) return;")
    ap("")
    ap("    int lane    = tid & 31;")
    ap("    int warp_id = tid >> 5;")
    ap("    int num_warps = (U + 31) >> 5;")
    ap("    int u = tid;")
    ap("")
    ap("    int src = src_idx[b];")
    ap("    int dst = dst_idx[b];")
    ap("")
    ap("    int64_t y_base = (int64_t)b * Ky;")
    ap("")
    ap("    extern __shared__ char smem_raw[];")
    ap("    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);")
    ap("    // layout: [num_unique_k][num_warps]")
    ap("")

    ap("    // preload w(i, u)")
    for ii in uniq_i:
        ap(f"    scalar_t wi_{ii} = w[((int64_t)b * Iw + {ii}) * (int64_t)U + u];")
    ap("")

    ap("    // preload x(j, u)")
    for jj in uniq_j:
        ap(f"    scalar_t xj_{jj} = x_all[((int64_t)src * Ix + {jj}) * (int64_t)U + u];")
    ap("")

    ap("    // preload y(k)")
    for kk in uniq_k:
        ap(f"    scalar_t yk_{kk} = y[y_base + {kk}];")
    ap("")

    ap("    // preload grad_out(v, u)")
    for vv in all_vs:
        ap(f"    scalar_t go_v_{vv} = grad_out[((int64_t)dst * V + {vv}) * (int64_t)U + u];")
    ap("")

    # grad_w: direct store
    ap("    // grad_w accumulate by unique i (no atomic: one thread owns one (b, i, u))")
    for ii in uniq_i:
        ap(f"    scalar_t gw_acc_i_{ii} = scalar_t(0);")
        for vv, jj, kk, cc in i_terms[ii]:
            if abs(cc - 1.0) < 1e-12:
                ap(f"    gw_acc_i_{ii} += xj_{jj} * yk_{kk} * go_v_{vv};")
            elif abs(cc + 1.0) < 1e-12:
                ap(f"    gw_acc_i_{ii} -= xj_{jj} * yk_{kk} * go_v_{vv};")
            else:
                cstr = _fmt_coeff(cc, scalar_t)
                ap(f"    gw_acc_i_{ii} += scalar_t({cstr}) * xj_{jj} * yk_{kk} * go_v_{vv};")
        ap(f"    grad_w[((int64_t)b * Iw + {ii}) * (int64_t)U + u] = gw_acc_i_{ii};")
        ap("")
    
    # grad_x
    ap("    // grad_x accumulate by unique j")
    for jj in uniq_j:
        ap(f"    scalar_t gx_acc_j_{jj} = scalar_t(0);")
        for vv, ii, kk, cc in j_terms[jj]:
            if abs(cc - 1.0) < 1e-12:
                ap(f"    gx_acc_j_{jj} += wi_{ii} * yk_{kk} * go_v_{vv};")
            elif abs(cc + 1.0) < 1e-12:
                ap(f"    gx_acc_j_{jj} -= wi_{ii} * yk_{kk} * go_v_{vv};")
            else:
                cstr = _fmt_coeff(cc, scalar_t)
                ap(f"    gx_acc_j_{jj} += scalar_t({cstr}) * wi_{ii} * yk_{kk} * go_v_{vv};")
        ap(f"    atomicAdd(&grad_x[((int64_t)src * Ix + {jj}) * (int64_t)U + u], gx_acc_j_{jj});")
        ap("")
    
    # grad_y with block reduction
    ap("    // grad_y accumulate by unique k, reduced across full block (all U channels)")
    for kk_idx, kk in enumerate(uniq_k):
        ap(f"    scalar_t gy_lane_k_{kk} = scalar_t(0);")
        for vv, ii, jj, cc in k_terms[kk]:
            if abs(cc - 1.0) < 1e-12:
                ap(f"    gy_lane_k_{kk} += wi_{ii} * xj_{jj} * go_v_{vv};")
            elif abs(cc + 1.0) < 1e-12:
                ap(f"    gy_lane_k_{kk} -= wi_{ii} * xj_{jj} * go_v_{vv};")
            else:
                cstr = _fmt_coeff(cc, scalar_t)
                ap(f"    gy_lane_k_{kk} += scalar_t({cstr}) * wi_{ii} * xj_{jj} * go_v_{vv};")
        ap(f"    scalar_t gy_warp_k_{kk} = warp_sum(gy_lane_k_{kk});")
        ap(f"    if (lane == 0) smem[{kk_idx} * num_warps + warp_id] = gy_warp_k_{kk};")
        ap("")

    ap("    __syncthreads();")
    ap("")
    ap("    if (warp_id == 0) {")
    ap("        int red_lane = lane;")
    for kk_idx, kk in enumerate(uniq_k):
        ap(f"        scalar_t block_sum_k_{kk} = scalar_t(0);")
        ap(f"        if (red_lane < num_warps) block_sum_k_{kk} = smem[{kk_idx} * num_warps + red_lane];")
        ap(f"        block_sum_k_{kk} = warp_sum(block_sum_k_{kk});")
        ap(f"        if (red_lane == 0) atomicAdd(&grad_y[y_base + {kk}], block_sum_k_{kk});")
        ap("")
    ap("    }")
    ap("}")
    ap("")

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
    ap("    // 要求 U <= 1024")
    ap(f"    size_t smem_bytes = sizeof(scalar_t) * {num_uniq_k} * ((U + 31) / 32);")
    ap("    dim3 block(U);")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t><<<grid, block, smem_bytes, stream>>>(")
    ap("        w, x_all, y, grad_out, grad_w, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list, B, Iw, Ix, Ky, V, U);")
    ap("}")
    ap("")

    return '\n'.join(lines)

def generate_code_uniform1d_bwd(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    u_dim: int = 1,
    out_path: str = "generated_uniform1d_bwd.cu",
    kernel_name: str = "stp_codegen_blocku_vgroup_bwd",
    scalar_t: str = "float",
    reorder_groups: bool = True,
):
    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    out_path = f"uniform1d_codegen_path{P}_u{u_dim}_bwd.cu"

    kernel_name = f"uniform1d_codegen_blocku_vgroup_path{P}_u{u_dim}_bwd"

    if reorder_groups:
        i2, j2, k2, v2, c2, _ = reorder_groups_for_reuse(
            i_list, j_list, k_list, v_list, coeff_list
        )
    else:
        i2, j2, k2, v2, c2 = i_list, j_list, k_list, v_list, coeff_list

    groups = build_v_groups(i2, j2, k2, v2, c2)
    code = emit_blocku_vgroup_backward_kernel(
        groups=groups,
        kernel_name=kernel_name,
        scalar_t=scalar_t,
    )
    Path(out_path).write_text(code, encoding="utf-8")

    warp0_vs, warp1_vs = split_groups_into_two_warps(groups)
    return {
        "num_paths": int(P),
        "num_v_groups": int(len(groups)),
        "group_order": list(groups.keys()),
        "warp0_vs": warp0_vs,
        "warp1_vs": warp1_vs,
        "out_path": str(out_path),
    }