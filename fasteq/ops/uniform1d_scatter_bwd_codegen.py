from __future__ import annotations

from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
import math
from pathlib import Path
import struct
import numpy as np
import torch

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
#include "cuda_utils.hpp"
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
    ap(f'__global__ void {kernel_name}(')
    ap('    const scalar_t* __restrict__ grad_out,')
    ap('    const scalar_t* __restrict__ x,')
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
                    ap(f'        scalar_t x_j_{p.j} = x[x_base + ((int64_t){p.j} << 5)];')
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
        acc_cost=4,   # local_k
        cache_cost_per_symbol=1,
        reg_budget=reg_budget,
        max_targets_per_chunk=4,
    )

    lines = []
    ap = lines.append

    ap('template <typename scalar_t>')
    ap(f'__global__ void {kernel_name}(')
    ap('    const scalar_t* __restrict__ grad_out,')
    ap('    const scalar_t* __restrict__ w,')
    ap('    const scalar_t* __restrict__ x,')
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
                ap(f'        scalar_t x_j_{j}  = x[x_base + ((int64_t){j} << 5)];')
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
    acc_slots: int = 16,
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
    ap('    const scalar_t* __restrict__ x,')
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
                    ap(f'                scalar_t xv = x[x_base + ((int64_t){j} << 5)];')
                    ap(f'                acc{slot} = fma((scalar_t)({cstr}), xv * yg, acc{slot});')
                    ap('            }')
                ap('        }')
                ap('')
            else:
                # 单条/很小簇，不额外拉长 y/go live range
                for (slot, j, _k, _v, c) in bucket:
                    cstr = fmt_coeff(c)
                    ap('        {')
                    ap(f'            scalar_t xv  = x[x_base + ((int64_t){j} << 5)];')
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
std::vector<torch::Tensor> launcher_{bundle_name}(
    torch::Tensor w,
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor grad_out,
    torch::Tensor src_idx,
    torch::Tensor dst_idx,
    torch::Tensor b_list,
    int64_t V)
{{
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(y.is_cuda(), "y must be CUDA");

    auto B = b_list.numel() > 0 ? (int)b_list.numel() : (int)w.size(0);

    auto grad_w = torch::zeros_like(w);
    auto grad_x = torch::zeros_like(x);
    auto grad_y = torch::zeros_like(y);

    const int Iw = (int)w.size(1);
    const int Ix = (int)x.size(1);
    const int Ky = (int)y.size(1);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    dim3 block(32);
    dim3 grid(B);

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{
        {gradw_kernel}<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
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
            x.data_ptr<scalar_t>(),
            grad_y.data_ptr<scalar_t>(),
            src_idx.data_ptr<int32_t>(),
            dst_idx.data_ptr<int32_t>(),
            b_list.numel() ? b_list.data_ptr<int32_t>() : nullptr,
            B, (int)Iw, (int)Ix, (int)Ky, (int)V);
    }});

    return {{grad_w, grad_x, grad_y}};
}}

/*
TORCH_LIBRARY({bundle_name}_codegen, m) {{
    m.def("run", &&launcher_{bundle_name});
}}
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} backward jit impl");
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
    code = '\n'.join(parts)
    return code