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


# ============================================================
# CUDA emitter
# ============================================================

def emit_two_warp_vgroup_forward_kernel(
    groups: OrderedDict,
    kernel_name: str = "stp_codegen_two_warp_vgroup",
    scalar_t: str = "float",
) -> str:
    """
    生成一个固定 2 warp 的 kernel:
      warp0 处理一部分 v
      warp1 处理另一部分 v
    按 v 粒度展开
    """
    warp0_vs, warp1_vs = split_groups_into_two_warps(groups)

    lines: List[str] = []
    ap = lines.append

    ap("#include <stdint.h>")
    ap("#include <cuda_runtime.h>")
    ap("")
    ap("template <typename scalar_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")        # [B, Iw, 32]
    ap("    const scalar_t* __restrict__ x_all,")    # [S, Ix, 32]
    ap("    const scalar_t* __restrict__ y,")        # [B, Ky]
    ap("    scalar_t* __restrict__ out,")            # [Dst, V, 32]
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int Iw, int Ix, int Ky, int V)")
    ap("{")
    ap("    int b_global = (int)blockIdx.x;")
    ap("    if (b_global >= B) return;")
    ap("    int b = b_list ? b_list[b_global] : b_global;")
    ap("")
    ap("    int tid  = threadIdx.x;")
    ap("    int lane = tid & 31;")
    ap("    int warp = tid >> 5;")
    ap("    if (warp >= 2) return;")
    ap("")
    ap("    int src = src_idx[b];")
    ap("    int dst = dst_idx[b];")
    ap("")
    ap("    int64_t w_base = (int64_t)b   * Iw * 32;")
    ap("    int64_t x_base = (int64_t)src * Ix * 32;")
    ap("    int64_t y_base = (int64_t)b   * Ky;")
    ap("    int64_t o_base = ((int64_t)dst * V) * 32 + lane;")
    ap("")

    def emit_warp_body(warp_id: int, warp_vs: List[int]):
        ap(f"    if (warp == {warp_id}) {{")
        if not warp_vs:
            ap("        return;")
            ap("    }")
            ap("")
            return

        # preload 该 warp 用到的 unique i/j/k
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

        ap("        // preload w(i)")
        for ii in uniq_i:
            ap(f"        scalar_t wi_{ii} = w[w_base + {ii}LL * 32 + lane];")
        ap("")

        ap("        // preload x(j)")
        for jj in uniq_j:
            ap(f"        scalar_t xj_{jj} = x_all[x_base + {jj}LL * 32 + lane];")
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
            ap(f"        atomicAdd(&out[o_base + ({vv}LL << 5)], sum_v_{vv});")
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
    ap("    int B, int Iw, int Ix, int Ky, int V,")
    ap("    cudaStream_t stream)")
    ap("{")
    ap("    dim3 block(64);  // 2 warps")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t><<<grid, block, 0, stream>>>(")
    ap("        w, x_all, y, out, src_idx, dst_idx, b_list,")
    ap("        B, Iw, Ix, Ky, V);")
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
    ap("    const scalar_t* __restrict__ w,")          # [B, Iw, 32]
    ap("    const scalar_t* __restrict__ x_all,")      # [S, Ix, 32]
    ap("    const scalar_t* __restrict__ y,")          # [B, Ky]
    ap("    const scalar_t* __restrict__ grad_out,")   # [Dst, V, 32]
    ap("    scalar_t* __restrict__ grad_w,")           # [B, Iw, 32]
    ap("    scalar_t* __restrict__ grad_x,")           # [S, Ix, 32]
    ap("    scalar_t* __restrict__ grad_y,")           # [B, Ky]
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int Iw, int Ix, int Ky, int V)")
    ap("{")
    ap("    int b_global = (int)blockIdx.x;")
    ap("    if (b_global >= B) return;")
    ap("    int b = b_list ? b_list[b_global] : b_global;")
    ap("")
    ap("    int tid  = threadIdx.x;")
    ap("    int lane = tid & 31;")
    ap("    int warp = tid >> 5;")
    ap("    if (warp >= 2) return;")
    ap("")
    ap("    int src = src_idx[b];")
    ap("    int dst = dst_idx[b];")
    ap("")
    ap("    int64_t w_base  = (int64_t)b   * Iw * 32;")
    ap("    int64_t x_base  = (int64_t)src * Ix * 32;")
    ap("    int64_t y_base  = (int64_t)b   * Ky;")
    ap("    int64_t go_base = ((int64_t)dst * V) * 32 + lane;")
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
            ii = groups[vv]['i']
            if ii not in seen_i:
                seen_i.add(ii)
                uniq_i.append(ii)
            for jj, kk, _ in groups[vv]['terms']:
                if jj not in seen_j:
                    seen_j.add(jj)
                    uniq_j.append(jj)
                if kk not in seen_k:
                    seen_k.add(kk)
                    uniq_k.append(kk)

        # preload forward inputs used by this warp
        ap("        // preload w(i)")
        for ii in uniq_i:
            ap(f"        scalar_t wi_{ii} = w[w_base + {ii}LL * 32 + lane];")
        ap("")
        ap("        // preload x(j)")
        for jj in uniq_j:
            ap(f"        scalar_t xj_{jj} = x_all[x_base + {jj}LL * 32 + lane];")
        ap("")
        ap("        // preload y(k)")
        for kk in uniq_k:
            ap(f"        scalar_t yk_{kk} = y[y_base + {kk}];")
        ap("")
        ap("        // preload grad_out(v)")
        for vv in warp_vs:
            ap(f"        scalar_t go_v_{vv} = grad_out[go_base + ({vv}LL << 5)];")
        ap("")

        # grad_w accumulation by unique i
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
            ap(f"        atomicAdd(&grad_w[w_base + {ii}LL * 32 + lane], gw_acc_i_{ii});")
            ap("")

        # grad_x accumulation by unique j
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
            ap(f"        atomicAdd(&grad_x[x_base + {jj}LL * 32 + lane], gx_acc_j_{jj});")
            ap("")

        # grad_y accumulation by unique k, with warp reduction
        ap("        // grad_y accumulate by unique k (warp-reduce across lane)")
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
    ap("    int B, int Iw, int Ix, int Ky, int V,")
    ap("    cudaStream_t stream)")
    ap("{")
    ap("    dim3 block(64);  // 2 warps")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t><<<grid, block, 0, stream>>>(")
    ap("        w, x_all, y, grad_out, grad_w, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list, B, Iw, Ix, Ky, V);")
    ap("}")
    return '\n'.join(lines)


def generate_code_uniform1d_bwd(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    u_dim: int = 1,
    out_path: str = "generated_uniform1d_bwd.cu",
    kernel_name: str = "stp_codegen_two_warp_vgroup_bwd",
    scalar_t: str = "float",
    reorder_groups: bool = True,
):
    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    out_path = f"uniform1d_codegen_path{P}_u{u_dim}_bwd.cu"

    kernel_name = f"uniform1d_codegen_two_warp_vgroup_path{P}_u{u_dim}_bwd"

    if reorder_groups:
        i2, j2, k2, v2, c2, _ = reorder_groups_for_reuse(
            i_list, j_list, k_list, v_list, coeff_list
        )
    else:
        i2, j2, k2, v2, c2 = i_list, j_list, k_list, v_list, coeff_list

    groups = build_v_groups(i2, j2, k2, v2, c2)
    code = emit_two_warp_vgroup_backward_kernel(
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