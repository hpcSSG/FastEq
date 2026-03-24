from __future__ import annotations

from collections import defaultdict, OrderedDict
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

def _y_expr(kk: int, mode: str) -> str:
    if mode == "u,u,,u":
        return f"y[y_base + (int64_t){kk}]"
    elif mode == "u,u,u,u":
        return f"y[y_base + (int64_t){kk} * (int64_t)U + u]"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

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

def build_vi_groups(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
):
    """
    输入已排序的 tensor，按 (v, i) 分组。
    允许同一个 v 对应多个不同的 i。
    """
    i_py = _to_int_list(i_list)
    j_py = _to_int_list(j_list)
    k_py = _to_int_list(k_list)
    v_py = _to_int_list(v_list)
    c_py = _to_float_list(coeff_list)

    groups = OrderedDict()
    for ii, jj, kk, vv, cc in zip(i_py, j_py, k_py, v_py, c_py):
        key = (vv, ii)
        if key not in groups:
            groups[key] = {
                "v": vv,
                "i": ii,
                "terms": [],
            }
        groups[key]["terms"].append((jj, kk, cc))
    return groups

def split_groups_into_two_warps(groups: OrderedDict):
    """
    按 (v, i) group 的 term 数量尽量均衡地分给两个 warp。
    返回的是 group keys 列表，而不是 v 列表。
    """
    items = [(gk, len(info["terms"])) for gk, info in groups.items()]

    warp0_groups = []
    warp1_groups = []
    load0 = 0
    load1 = 0

    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    for gk, cost in items_sorted:
        if load0 <= load1:
            warp0_groups.append(gk)
            load0 += cost
        else:
            warp1_groups.append(gk)
            load1 += cost

    order = list(groups.keys())
    pos = {gk: idx for idx, gk in enumerate(order)}
    warp0_groups.sort(key=lambda gk: pos[gk])
    warp1_groups.sort(key=lambda gk: pos[gk])

    return warp0_groups, warp1_groups


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
        y_check_u = 'TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");'
    elif mode == "u,u,u,u":
        y_comment = "[B,Ky,U] or [S,Ky,U]"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == U, "y.size(2) must equal U for mode u,u,u,u");'
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    src_decl = '    torch::Tensor src_idx,    // [?] int32\n' if (use_x_src or use_y_src) else ""
    dst_decl = '    torch::Tensor dst_idx,    // [?] int32\n' if use_scatter else ""
    blist_decl = '    torch::Tensor b_list,     // [B] int32 optional\n' if use_scatter else ""

    src_check = ""
    if use_x_src or use_y_src:
        src_check = r'''
    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''

    dst_check = ""
    if use_scatter:
        dst_check = r'''
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''

    out_alloc = (
        '    auto out = torch::zeros({S, V, U}, w.options());'
        if use_scatter else
        '    auto out = torch::zeros({B, V, U}, w.options());'
    )

    blist_logic = ""
    if use_scatter:
        blist_logic = r'''
    const int32_t* b_list_ptr = nullptr;
    if (b_list.defined() && b_list.numel() > 0) {
        TORCH_CHECK(b_list.is_cuda(), "b_list must be CUDA");
        TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list must be int32");
        TORCH_CHECK((int)b_list.numel() == B, "b_list must be [B]");
        b_list_ptr = (const int32_t*)b_list.data_ptr<int32_t>();
    }
'''
    else:
        blist_logic = '    const int32_t* b_list_ptr = nullptr;\n'

    src_numel_check = ""
    if use_x_src or use_y_src:
        src_numel_check = '    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");'

    dst_numel_check = ""
    if use_scatter:
        dst_numel_check = '    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");'

    launch_src_arg = '(const int32_t*)src_idx.data_ptr<int32_t>(),' if (use_x_src or use_y_src) else 'nullptr,'
    launch_dst_arg = '(const int32_t*)dst_idx.data_ptr<int32_t>(),' if use_scatter else 'nullptr,'

    return rf'''

torch::Tensor launcher_{bundle_name}(
    torch::Tensor w,          // [B,Iw,U]
    torch::Tensor x_all,      // [S,Ix,U]
    torch::Tensor y,          // {y_comment}
{src_decl}{dst_decl}{blist_decl}    int64_t V64)
{{

    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(), "w/x_all/y must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(),
                "w/x_all/y must be contiguous");
{src_check}{dst_check}
    TORCH_CHECK(w.dim() == 3, "w must be [B,Iw,U]");
    TORCH_CHECK(x_all.dim() == 3, "x_all must be [S,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");

    int B  = (int)src_idx.size(0);
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x_all.size(0);
    int Ix = (int)x_all.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(w.dim() == 3, "w must be [WB, Iw, U]");
    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");

    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");
    {y_check_u}
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
{src_numel_check}
{dst_numel_check}

{blist_logic}

{out_alloc}

    c10::cuda::CUDAGuard device_guard(w.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{

        TORCH_CHECK(x_all.scalar_type() == w.scalar_type(), "x_all dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");

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

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}}
/*
TORCH_LIBRARY({bundle_name}_codegen, m) {{
    m.def("run", &launcher_{bundle_name});
}}
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} forward jit impl");
}}
'''



def emit_two_warp_vgroup_forward_kernel(
    groups: OrderedDict,
    kernel_name: str = "stp_codegen_two_warp_vgroup",
    scalar_t: str = "float",
    mode: str = "u,u,,u",
    *,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    warp0_vs, warp1_vs = split_groups_into_two_warps(groups)

    lines: List[str] = []
    ap = lines.append

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

    ap("template <typename scalar_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x_all,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    scalar_t* __restrict__ out,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")

    ap("    int e_local = (int)blockIdx.x;")
    ap("    int ublk    = (int)blockIdx.y;")
    ap("    if (e_local >= B) return;")

    ap("    int e_orig = b_list ? b_list[e_local] : e_local;")
    ap("    int w_row  = (WB == 1 ? 0 : e_local);")

    ap("    int tid  = (int)threadIdx.x;")
    ap("    int lane = tid & 31;")
    ap("    int warp = tid >> 5;")
    ap("    if (warp >= 2) return;")
    ap("")
    ap("    int u = (ublk << 5) + lane;")
    ap("    if (u >= U) return;")
    ap("")

    if use_x_src or use_y_src:
        ap("    int src = src_idx[e_orig];")
    if use_scatter:
        ap("    int dst = dst_idx[e_orig];")
    ap("")

    ap("    int64_t w_base = (int64_t)w_row * (int64_t)Iw * (int64_t)U;")
    

    if use_x_src:
        ap("    int64_t x_base = (int64_t)src * (int64_t)Ix * (int64_t)U;")
    else:
        ap("    int64_t x_base = (int64_t)e_local * (int64_t)Ix * (int64_t)U;")

    if mode == "u,u,u":
        if use_y_src:
            ap("    int64_t y_base = (int64_t)src * (int64_t)Ky;")
        else:
            ap("    int64_t y_base = (int64_t)e_local * (int64_t)Ky;")
    else:
        if use_y_src:
            ap("    int64_t y_base = (int64_t)src * (int64_t)Ky * (int64_t)U;")
        else:
            ap("    int64_t y_base = (int64_t)e_local * (int64_t)Ky * (int64_t)U;")
    ap("")

    def emit_warp_body(warp_id: int, warp_groups: List[tuple]):
        ap(f"    if (warp == {warp_id}) {{")
        if not warp_groups:
            ap("        return;")
            ap("    }")
            ap("")
            return

        uniq_i, uniq_j, uniq_k = [], [], []
        seen_i, seen_j, seen_k = set(), set(), set()

        for gk in warp_groups:
            info = groups[gk]
            ii = info["i"]
            if ii not in seen_i:
                seen_i.add(ii)
                uniq_i.append(ii)
            for jj, kk, _ in info["terms"]:
                if jj not in seen_j:
                    seen_j.add(jj)
                    uniq_j.append(jj)
                if kk not in seen_k:
                    seen_k.add(kk)
                    uniq_k.append(kk)

        ap("        // preload w(i)")
        for ii in uniq_i:
            ap(f"        scalar_t wi_{ii} = w[w_base + (int64_t){ii} * (int64_t)U + u];")
        ap("")

        ap("        // preload x(j)")
        for jj in uniq_j:
            ap(f"        scalar_t xj_{jj} = x_all[x_base + (int64_t){jj} * (int64_t)U + u];")
        ap("")

        ap("        // preload y(k)")
        for kk in uniq_k:
            ap(f"        scalar_t yk_{kk} = {_y_expr(kk, mode)};")
        ap("")

        ap("        // per-group accumulation")
        for idx, gk in enumerate(warp_groups):
            info = groups[gk]
            vv = info["v"]
            ii = info["i"]
            terms = info["terms"]

            ap(f"        scalar_t sum_g_{warp_id}_{idx} = scalar_t(0);")
            for jj, kk, cc in terms:
                if abs(cc - 1.0) < 1e-12:
                    ap(f"        sum_g_{warp_id}_{idx} += wi_{ii} * xj_{jj} * yk_{kk};")
                elif abs(cc + 1.0) < 1e-12:
                    ap(f"        sum_g_{warp_id}_{idx} -= wi_{ii} * xj_{jj} * yk_{kk};")
                else:
                    cstr = _fmt_coeff(cc, scalar_t)
                    ap(f"        sum_g_{warp_id}_{idx} += scalar_t({cstr}) * wi_{ii} * xj_{jj} * yk_{kk};")
            ap("")

        ap("        // writeback")
        for idx, gk in enumerate(warp_groups):
            vv = groups[gk]["v"]
            if use_scatter:
                ap(f"        atomicAdd(&out[((int64_t)dst * (int64_t)V + (int64_t){vv}) * (int64_t)U + u], sum_g_{warp_id}_{idx});")
            else:
                ap(f"        out[((int64_t)e_local * (int64_t)V + (int64_t){vv}) * (int64_t)U + u] += sum_g_{warp_id}_{idx};")
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
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    cudaStream_t stream)")
    ap("{")
    ap("    dim3 block(64);")
    ap("    dim3 grid(B, (U + 31) / 32);")
    ap(f"    {kernel_name}<scalar_t><<<grid, block, 0, stream>>>(")
    ap("        w, x_all, y, out, src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")

    return "\n".join(lines)


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
    reorder_groups: bool = True,
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

    mode_str = "u_u__u" if mode == "u,u,,u" else "u_u_u_u"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    kernel_name = f"uniform1d_combine_u{u_dim}_path{P}_{mode_str}_{layout_tag}_fwd"

    if reorder_groups:
        i2, j2, k2, v2, c2, group_order = reorder_groups_for_reuse(
            i_list, j_list, k_list, v_list, coeff_list
        )
    else:
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

    groups = build_vi_groups(i2, j2, k2, v2, c2)

    
    code = emit_two_warp_vgroup_forward_kernel(
        groups,
        kernel_name=kernel_name,
        scalar_t=scalar_t,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    code = code + "\n" + emit_launcher(
        kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    warp0_groups, warp1_groups = split_groups_into_two_warps(groups)
    stats = {
        "num_paths": int(P),
        "num_groups": int(len(groups)),
        "group_keys": list(groups.keys()),
        "warp0_groups": warp0_groups,
        "warp1_groups": warp1_groups,
        "out_path": str(out_path),
        "mode": mode,
        "use_x_src": use_x_src,
        "use_y_src": use_y_src,
        "use_scatter": use_scatter,
    }

    return code