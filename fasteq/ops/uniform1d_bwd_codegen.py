from __future__ import annotations

from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import math
import torch


# ============================================================
# Helpers
# ============================================================

def _to_int_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().tolist()]


def _to_float_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]


def _fmt_coeff(c: float, scalar_t: str = "float") -> str:
    c = float(c)

    if scalar_t in ("float", "at::Half", "half"):
        if c == float("inf"):
            return "INFINITY"
        if c == float("-inf"):
            return "-INFINITY"
        if c != c:
            return "NAN"
        s = f"{c:.9g}"
        if ("e" not in s) and ("E" not in s) and ("." not in s):
            s += ".0"
        return s + "f"

    elif scalar_t in ("double",):
        if c == float("inf"):
            return "INFINITY"
        if c == float("-inf"):
            return "-INFINITY"
        if c != c:
            return "NAN"
        s = f"{c:.17g}"
        if ("e" not in s) and ("E" not in s) and ("." not in s):
            s += ".0"
        return s

    else:
        s = f"{c:.9g}"
        if ("e" not in s) and ("E" not in s) and ("." not in s):
            s += ".0"
        return s


def _y_expr(kk: int, mode: str) -> str:
    if mode == "u,u,,u":
        return f"y[y_base + (int64_t){kk}]"
    elif mode == "u,u,u,u":
        return f"y[y_base + (int64_t){kk} * (int64_t)U + u]"
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _gradout_expr(vv: int) -> str:
    return f"grad_out[go_base + (int64_t){vv} * (int64_t)U + u]"


# ============================================================
# Reorder
# ============================================================

def reorder_paths_for_fused_bwd(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
):
    """
    一个折中的全局顺序，优先把 (j, k, v) 接近的 path 放近一些，
    让 fused backward 里 x/y/go 的局部复用更自然。
    """
    items = list(zip(
        _to_int_list(i_list),
        _to_int_list(j_list),
        _to_int_list(k_list),
        _to_int_list(v_list),
        _to_float_list(coeff_list),
    ))
    items.sort(key=lambda x: (x[1], x[2], x[3], x[0]))   # (j, k, v, i)

    dev = i_list.device
    return (
        torch.tensor([x[0] for x in items], device=dev, dtype=i_list.dtype),
        torch.tensor([x[1] for x in items], device=dev, dtype=j_list.dtype),
        torch.tensor([x[2] for x in items], device=dev, dtype=k_list.dtype),
        torch.tensor([x[3] for x in items], device=dev, dtype=v_list.dtype),
        torch.tensor([x[4] for x in items], device=dev, dtype=coeff_list.dtype),
    )


# ============================================================
# Group builders
# ============================================================

def build_gradw_groups(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
):
    """
    grad_w[..., i, u] += sum coeff * x[..., j, u] * y[..., k, *] * grad_out[..., v, u]
    """
    i_py = _to_int_list(i_list)
    j_py = _to_int_list(j_list)
    k_py = _to_int_list(k_list)
    v_py = _to_int_list(v_list)
    c_py = _to_float_list(coeff_list)

    groups = OrderedDict()
    for ii, jj, kk, vv, cc in zip(i_py, j_py, k_py, v_py, c_py):
        if ii not in groups:
            groups[ii] = {
                "i": ii,
                "terms": [],   # (j, k, v, coeff)
            }
        groups[ii]["terms"].append((jj, kk, vv, cc))
    return groups


def build_gradx_groups(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
):
    """
    grad_x[..., j, u] += sum coeff * w[..., i, u] * y[..., k, *] * grad_out[..., v, u]
    """
    i_py = _to_int_list(i_list)
    j_py = _to_int_list(j_list)
    k_py = _to_int_list(k_list)
    v_py = _to_int_list(v_list)
    c_py = _to_float_list(coeff_list)

    groups = OrderedDict()
    for ii, jj, kk, vv, cc in zip(i_py, j_py, k_py, v_py, c_py):
        if jj not in groups:
            groups[jj] = {
                "j": jj,
                "terms": [],   # (i, k, v, coeff)
            }
        groups[jj]["terms"].append((ii, kk, vv, cc))
    return groups


def build_grady_groups(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
):
    """
    grad_y[..., k, *] += sum coeff * w[..., i, u] * x[..., j, u] * grad_out[..., v, u]
    """
    i_py = _to_int_list(i_list)
    j_py = _to_int_list(j_list)
    k_py = _to_int_list(k_list)
    v_py = _to_int_list(v_list)
    c_py = _to_float_list(coeff_list)

    groups = OrderedDict()
    for ii, jj, kk, vv, cc in zip(i_py, j_py, k_py, v_py, c_py):
        if kk not in groups:
            groups[kk] = {
                "k": kk,
                "terms": [],   # (i, j, v, coeff)
            }
        groups[kk]["terms"].append((ii, jj, vv, cc))
    return groups


# ============================================================
# Emitters: grad_w / grad_x / grad_y inside one fused kernel
# ============================================================

def emit_gradw_body_fused(
    ap,
    groups: OrderedDict,
    mode: str,
    scalar_t: str,
):
    ap("    // ================= grad_w =================")
    ap("    for (int u = lane; u < U; u += 32) {")

    for ii, info in groups.items():
        terms = info["terms"]
        ap(f"        // grad_w: i = {ii}")
        ap("        {")
        ap("            scalar_t acc = scalar_t(0);")
        for jj, kk, vv, cc in terms:
            expr = (
                f"x_all[x_base + (int64_t){jj} * (int64_t)U + u] * "
                f"{_y_expr(kk, mode)} * "
                f"{_gradout_expr(vv)}"
            )
            if abs(cc - 1.0) < 1e-12:
                ap(f"            acc += {expr};")
            elif abs(cc + 1.0) < 1e-12:
                ap(f"            acc -= {expr};")
            else:
                cstr = _fmt_coeff(cc, scalar_t)
                ap(f"            acc += scalar_t({cstr}) * ({expr});")

        ap(f"            int64_t gw_idx = gw_base + (int64_t){ii} * (int64_t)U + u;")
        ap("            if (gradw_atomic) {")
        ap("                atomicAdd(&grad_w[gw_idx], acc);")
        ap("            } else {")
        ap("                grad_w[gw_idx] += acc;")
        ap("            }")
        ap("        }")
        ap("")

    ap("    }")
    ap("")


def emit_gradx_body_fused(
    ap,
    groups: OrderedDict,
    mode: str,
    scalar_t: str,
):
    ap("    // ================= grad_x =================")
    ap("    for (int u = lane; u < U; u += 32) {")

    for jj, info in groups.items():
        terms = info["terms"]
        ap(f"        // grad_x: j = {jj}")
        ap("        {")
        ap("            scalar_t acc = scalar_t(0);")
        for ii, kk, vv, cc in terms:
            expr = (
                f"w[w_base + (int64_t){ii} * (int64_t)U + u] * "
                f"{_y_expr(kk, mode)} * "
                f"{_gradout_expr(vv)}"
            )
            if abs(cc - 1.0) < 1e-12:
                ap(f"            acc += {expr};")
            elif abs(cc + 1.0) < 1e-12:
                ap(f"            acc -= {expr};")
            else:
                cstr = _fmt_coeff(cc, scalar_t)
                ap(f"            acc += scalar_t({cstr}) * ({expr});")

        ap(f"            int64_t gx_idx = gx_base + (int64_t){jj} * (int64_t)U + u;")
        ap("            if (gradx_atomic) {")
        ap("                atomicAdd(&grad_x[gx_idx], acc);")
        ap("            } else {")
        ap("                grad_x[gx_idx] += acc;")
        ap("            }")
        ap("        }")
        ap("")

    ap("    }")
    ap("")


def emit_grady_body_fused(
    ap,
    groups: OrderedDict,
    mode: str,
    scalar_t: str,
):
    ap("    // ================= grad_y =================")

    if mode == "u,u,u,u":
        ap("    for (int u = lane; u < U; u += 32) {")
        for kk, info in groups.items():
            terms = info["terms"]
            ap(f"        // grad_y vector: k = {kk}")
            ap("        {")
            ap("            scalar_t acc = scalar_t(0);")
            for ii, jj, vv, cc in terms:
                expr = (
                    f"w[w_base + (int64_t){ii} * (int64_t)U + u] * "
                    f"x_all[x_base + (int64_t){jj} * (int64_t)U + u] * "
                    f"{_gradout_expr(vv)}"
                )
                if abs(cc - 1.0) < 1e-12:
                    ap(f"            acc += {expr};")
                elif abs(cc + 1.0) < 1e-12:
                    ap(f"            acc -= {expr};")
                else:
                    cstr = _fmt_coeff(cc, scalar_t)
                    ap(f"            acc += scalar_t({cstr}) * ({expr});")

            ap(f"            int64_t gy_idx = gy_base + (int64_t){kk} * (int64_t)U + u;")
            ap("            if (grady_atomic) {")
            ap("                atomicAdd(&grad_y[gy_idx], acc);")
            ap("            } else {")
            ap("                grad_y[gy_idx] += acc;")
            ap("            }")
            ap("        }")
            ap("")

        ap("    }")
        ap("")

    elif mode == "u,u,,u":
        for kk, info in groups.items():
            terms = info["terms"]
            ap(f"    // grad_y scalar: k = {kk}")
            ap("    {")
            ap("        scalar_t acc = scalar_t(0);")
            ap("        for (int u = lane; u < U; u += 32) {")
            for ii, jj, vv, cc in terms:
                expr = (
                    f"w[w_base + (int64_t){ii} * (int64_t)U + u] * "
                    f"x_all[x_base + (int64_t){jj} * (int64_t)U + u] * "
                    f"{_gradout_expr(vv)}"
                )
                if abs(cc - 1.0) < 1e-12:
                    ap(f"            acc += {expr};")
                elif abs(cc + 1.0) < 1e-12:
                    ap(f"            acc -= {expr};")
                else:
                    cstr = _fmt_coeff(cc, scalar_t)
                    ap(f"            acc += scalar_t({cstr}) * ({expr});")
            ap("        }")
            ap("        for (int mask = 16; mask > 0; mask >>= 1) {")
            ap("            acc += __shfl_down_sync(0xffffffff, acc, mask);")
            ap("        }")
            ap("        if (lane == 0) {")
            ap(f"            int64_t gy_idx = gy_base + (int64_t){kk};")
            ap("            if (grady_atomic) {")
            ap("                atomicAdd(&grad_y[gy_idx], acc);")
            ap("            } else {")
            ap("                grad_y[gy_idx] += acc;")
            ap("            }")
            ap("        }")
            ap("    }")
            ap("")
    else:
        raise ValueError(f"Unsupported mode: {mode}")


# ============================================================
# Launcher
# ============================================================

def emit_fused_bwd_launcher(
    bundle_name: str,
    mode: str,
    *,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
    if mode == "u,u,,u":
        y_comment = "[B,Ky,1] or [S,Ky,1]"
        gy_alloc = '    auto grad_y = torch::zeros({y.size(0), Ky, 1}, y.options());'
        y_check_u = 'TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");'
    elif mode == "u,u,u,u":
        y_comment = "[B,Ky,U] or [S,Ky,U]"
        gy_alloc = '    auto grad_y = torch::zeros_like(y);'
        y_check_u = 'TORCH_CHECK((int)y.size(2) == U, "y.size(2) must equal U for mode u,u,u,u");'
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    src_decl = '    torch::Tensor src_idx,    // [?] int32\n' if (use_x_src or use_y_src) else ""
    dst_decl = '    torch::Tensor dst_idx,    // [?] int32\n' if use_scatter else ""
    blist_decl = '    torch::Tensor b_list,    // [B] int32 optional\n' if use_scatter else ""

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

    if use_x_src or use_y_src:
        B_expr = "(int)src_idx.size(0)"
        src_numel_check = '    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");'
    elif use_scatter:
        B_expr = "(int)dst_idx.size(0)"
        src_numel_check = ""
    else:
        B_expr = "(int)grad_out.size(0)"
        src_numel_check = ""

    dst_numel_check = '    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");' if use_scatter else ""

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

    launch_src_arg = '(const int32_t*)src_idx.data_ptr<int32_t>(),' if (use_x_src or use_y_src) else 'nullptr,'
    launch_dst_arg = '(const int32_t*)dst_idx.data_ptr<int32_t>(),' if use_scatter else 'nullptr,'

    return rf'''

std::vector<torch::Tensor> launcher_{bundle_name}(
    torch::Tensor w,           // [WB,Iw,U]
    torch::Tensor x_all,       // [S,Ix,U]
    torch::Tensor y,           // {y_comment}
    torch::Tensor grad_out,    // [B,V,U] or [S,V,U]
{src_decl}{dst_decl}{blist_decl}    int64_t V64)
{{
    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x_all/y/grad_out must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x_all/y/grad_out must be contiguous");
{src_check}{dst_check}
    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x_all.dim() == 3, "x_all must be [S,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");

    int B  = {B_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x_all.size(0);
    int Ix = (int)x_all.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");
    {y_check_u}
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
{src_numel_check}
{dst_numel_check}

{blist_logic}

    auto grad_w = torch::zeros_like(w);
    auto grad_x = torch::zeros_like(x_all);
{gy_alloc}

    c10::cuda::CUDAGuard device_guard(w.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{
        TORCH_CHECK(x_all.scalar_type() == w.scalar_type(), "x_all dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");

        launch_{bundle_name}<scalar_t>(
            (const scalar_t*)w.data_ptr<scalar_t>(),
            (const scalar_t*)x_all.data_ptr<scalar_t>(),
            (const scalar_t*)y.data_ptr<scalar_t>(),
            (const scalar_t*)grad_out.data_ptr<scalar_t>(),
            (scalar_t*)grad_w.data_ptr<scalar_t>(),
            (scalar_t*)grad_x.data_ptr<scalar_t>(),
            (scalar_t*)grad_y.data_ptr<scalar_t>(),
            {launch_src_arg}
            {launch_dst_arg}
            b_list_ptr,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    }});

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{grad_w, grad_x, grad_y}};
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} backward fused jit impl");
}}
'''


# ============================================================
# Main CUDA emitter
# ============================================================

def emit_fused_bwd_kernel(
    gradw_groups: OrderedDict,
    gradx_groups: OrderedDict,
    grady_groups: OrderedDict,
    u_dim: int,
    *,
    kernel_name: str,
    scalar_t: str,
    mode: str,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
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
    ap("    const scalar_t* __restrict__ grad_out,")
    ap("    scalar_t* __restrict__ grad_w,")
    ap("    scalar_t* __restrict__ grad_x,")
    ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    int e_orig = b_list ? b_list[e_local] : e_local;")
    ap("    int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    ap("    int tid  = (int)threadIdx.x;")
    ap("    int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    ap("")

    if use_x_src or use_y_src:
        ap("    int src = src_idx[e_orig];")
    if use_scatter:
        ap("    int dst = dst_idx[e_orig];")
    ap("")

    ap("    bool gradw_atomic = (WB == 1);")
    ap(f"    bool gradx_atomic = {'true' if use_x_src else 'false'};")
    ap(f"    bool grady_atomic = {'true' if use_y_src else 'false'};")
    ap("")

    ap("    int64_t w_base  = (int64_t)w_row * (int64_t)Iw * (int64_t)U;")
    ap("    int64_t gw_base = (int64_t)w_row * (int64_t)Iw * (int64_t)U;")

    if use_x_src:
        ap("    int64_t x_base  = (int64_t)src * (int64_t)Ix * (int64_t)U;")
        ap("    int64_t gx_base = (int64_t)src * (int64_t)Ix * (int64_t)U;")
    else:
        ap("    int64_t x_base  = (int64_t)e_local * (int64_t)Ix * (int64_t)U;")
        ap("    int64_t gx_base = (int64_t)e_local * (int64_t)Ix * (int64_t)U;")

    if mode == "u,u,,u":
        if use_y_src:
            ap("    int64_t y_base  = (int64_t)src * (int64_t)Ky;")
            ap("    int64_t gy_base = (int64_t)src * (int64_t)Ky;")
        else:
            ap("    int64_t y_base  = (int64_t)e_orig * (int64_t)Ky;")
            ap("    int64_t gy_base = (int64_t)e_orig * (int64_t)Ky;")
    elif mode == "u,u,u,u":
        if use_y_src:
            ap("    int64_t y_base  = (int64_t)src * (int64_t)Ky * (int64_t)U;")
            ap("    int64_t gy_base = (int64_t)src * (int64_t)Ky * (int64_t)U;")
        else:
            ap("    int64_t y_base  = (int64_t)e_orig * (int64_t)Ky * (int64_t)U;")
            ap("    int64_t gy_base = (int64_t)e_local * (int64_t)Ky * (int64_t)U;")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if use_scatter:
        ap("    int64_t go_base = (int64_t)dst * (int64_t)V * (int64_t)U;")
    else:
        ap("    int64_t go_base = (int64_t)e_local * (int64_t)V * (int64_t)U;")
    ap("")

    emit_gradw_body_fused(ap, gradw_groups, mode, scalar_t)
    emit_gradx_body_fused(ap, gradx_groups, mode, scalar_t)
    emit_grady_body_fused(ap, grady_groups, mode, scalar_t)

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
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    cudaStream_t stream)")
    ap("{")
    ap("    dim3 block(32);")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t><<<grid, block, 0, stream>>>(")
    ap("        w, x_all, y, grad_out, grad_w, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    return "\n".join(lines)



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


def emit_fused_bwd_kernel_from_schedule(
    schedule: Dict[str, Any],
    *,
    kernel_name: str,
    scalar_t: str,
    mode: str,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    write_modes = schedule["write_modes"]
    launch_style = schedule["launch_style"]
    u_traversal = schedule["u_traversal"]
    u_loop_step = schedule["u_loop_step"]
    block_size = int(schedule["block_size"])
    u_tile = int(schedule.get("u_tile", 32))

    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported here, got {block_size}")
    if launch_style not in ("persistent_u_inner_loop", "grid_y_tiled_u"):
        raise ValueError(f"Unsupported launch_style: {launch_style}")
    if u_traversal not in ("inner_loop", "grid_y"):
        raise ValueError(f"Unsupported u_traversal: {u_traversal}")
    if write_modes["grad_y"] not in ("warp_reduce_then_atomic", "warp_reduce_then_block_merge_then_atomic"):
        raise NotImplementedError(f"Unsupported grad_y mode: {write_modes['grad_y']}")

    ordered_i = [int(x) for x in schedule["ordered_symbols"]["i"]]
    ordered_j = [int(x) for x in schedule["ordered_symbols"]["j"]]
    ordered_k = [int(x) for x in schedule["ordered_symbols"]["k"]]

    v_tiles = schedule["tiling"]["v_tiles"]
    i_tiles = schedule["tiling"]["i_tiles"]
    j_tiles = schedule["tiling"]["j_tiles"]
    k_tiles = schedule["tiling"]["k_tiles"]

    grad_w_groups = schedule["groups"]["grad_w"]
    grad_x_groups = schedule["groups"]["grad_x"]
    grad_y_groups = schedule["groups"]["grad_y"]

    key_is_str_gw = len(grad_w_groups) > 0 and isinstance(next(iter(grad_w_groups.keys())), str)
    key_is_str_gx = len(grad_x_groups) > 0 and isinstance(next(iter(grad_x_groups.keys())), str)
    key_is_str_gy = len(grad_y_groups) > 0 and isinstance(next(iter(grad_y_groups.keys())), str)

    def _gw_entries(ii: int):
        return grad_w_groups[str(ii)] if key_is_str_gw else grad_w_groups[ii]

    def _gx_entries(jj: int):
        return grad_x_groups[str(jj)] if key_is_str_gx else grad_x_groups[jj]

    def _gy_entries(kk: int):
        return grad_y_groups[str(kk)] if key_is_str_gy else grad_y_groups[kk]

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
    ap("    const scalar_t* __restrict__ grad_out,")
    ap("    scalar_t* __restrict__ grad_w,")
    ap("    scalar_t* __restrict__ grad_x,")
    ap("    scalar_t* __restrict__ grad_y,")
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
    ap("    const int e_orig = b_list ? b_list[e_local] : e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")

    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e_orig];")
    if use_scatter:
        ap("    const int dst = dst_idx[e_orig];")

    ap("    const int64_t w_base  = (int64_t)w_row * (int64_t)Iw * (int64_t)U;")
    ap("    const int64_t gw_base = (int64_t)w_row * (int64_t)Iw * (int64_t)U;")

    if use_x_src:
        ap("    const int64_t x_base  = (int64_t)src * (int64_t)Ix * (int64_t)U;")
        ap("    const int64_t gx_base = (int64_t)src * (int64_t)Ix * (int64_t)U;")
    else:
        ap("    const int64_t x_base  = (int64_t)e_local * (int64_t)Ix * (int64_t)U;")
        ap("    const int64_t gx_base = (int64_t)e_local * (int64_t)Ix * (int64_t)U;")

    if mode == "u,u,,u":
        if use_y_src:
            ap("    const int64_t y_base  = (int64_t)src * (int64_t)Ky;")
            ap("    const int64_t gy_base = (int64_t)src * (int64_t)Ky;")
        else:
            ap("    const int64_t y_base  = (int64_t)e_orig * (int64_t)Ky;")
            ap("    const int64_t gy_base = (int64_t)e_orig * (int64_t)Ky;")
    else:
        if use_y_src:
            ap("    const int64_t y_base  = (int64_t)src * (int64_t)Ky * (int64_t)U;")
            ap("    const int64_t gy_base = (int64_t)src * (int64_t)Ky * (int64_t)U;")
        else:
            ap("    const int64_t y_base  = (int64_t)e_orig * (int64_t)Ky * (int64_t)U;")
            ap("    const int64_t gy_base = (int64_t)e_orig * (int64_t)Ky * (int64_t)U;")

    if use_scatter:
        ap("    const int64_t go_base = (int64_t)dst * (int64_t)V * (int64_t)U;")
    else:
        ap("    const int64_t go_base = (int64_t)e_local * (int64_t)V * (int64_t)U;")
    ap("")

    def emit_one_u_body(indent: str, u_expr: str):
        ap(f"{indent}{{")
        ap(f"{indent}    int u = {u_expr};")
        ap(f"{indent}    if (u < U) {{")

        # preload all wi/xj/yk needed by any later tile/use-site
        ap(f"{indent}        // preload all wi(i,u)")
        for ii in ordered_i:
            ap(f"{indent}        scalar_t wi_{ii} = w[w_base + (int64_t){ii} * (int64_t)U + u];")
        ap("")

        ap(f"{indent}        // preload all xj(j,u)")
        for jj in ordered_j:
            ap(f"{indent}        scalar_t xj_{jj} = x_all[x_base + (int64_t){jj} * (int64_t)U + u];")
        ap("")

        if mode == "u,u,,u":
            ap(f"{indent}        // preload all scalar yk(k)")
            for kk in ordered_k:
                ap(f"{indent}        scalar_t yk_{kk} = y[y_base + (int64_t){kk}];")
            ap("")

        # accumulators: declare exactly once per u
        ap(f"{indent}        // init accumulators")
        for ii in ordered_i:
            ap(f"{indent}        scalar_t gw_acc_i_{ii} = scalar_t(0);")
        for jj in ordered_j:
            ap(f"{indent}        scalar_t gx_acc_j_{jj} = scalar_t(0);")
        for kk in ordered_k:
            ap(f"{indent}        scalar_t gy_acc_k_{kk} = scalar_t(0);")
        ap("")

        # v-tiled accumulation
        for tile_id, vtile in enumerate(v_tiles):
            ap(f"{indent}        // ---- v tile {tile_id}: {vtile} ----")
            for vv in vtile:
                ap(f"{indent}        scalar_t go_v_{vv} = grad_out[go_base + (int64_t){vv} * (int64_t)U + u];")
            ap("")

            # grad_w
            for i_tile_id, itile in enumerate(i_tiles):
                ap(f"{indent}        // grad_w i-tile {i_tile_id}")
                for ii in itile:
                    for entry in _gw_entries(ii):
                        vv = int(entry["v"])
                        if vv not in vtile:
                            continue
                        jj = int(entry["j"])
                        kk = int(entry["k"])
                        cc = float(entry["c"])
                        yexpr = (
                            f"(y[y_base + (int64_t){kk} * (int64_t)U + u])"
                            if mode == "u,u,u,u"
                            else f"yk_{kk}"
                        )
                        if abs(cc - 1.0) < 1e-12:
                            ap(f"{indent}        gw_acc_i_{ii} += xj_{jj} * {yexpr} * go_v_{vv};")
                        elif abs(cc + 1.0) < 1e-12:
                            ap(f"{indent}        gw_acc_i_{ii} -= xj_{jj} * {yexpr} * go_v_{vv};")
                        else:
                            ap(f"{indent}        gw_acc_i_{ii} += scalar_t({cc}) * xj_{jj} * {yexpr} * go_v_{vv};")
                ap("")

            # grad_x
            for j_tile_id, jtile in enumerate(j_tiles):
                ap(f"{indent}        // grad_x j-tile {j_tile_id}")
                for jj in jtile:
                    for entry in _gx_entries(jj):
                        vv = int(entry["v"])
                        if vv not in vtile:
                            continue
                        ii = int(entry["i"])
                        kk = int(entry["k"])
                        cc = float(entry["c"])
                        yexpr = (
                            f"(y[y_base + (int64_t){kk} * (int64_t)U + u])"
                            if mode == "u,u,u,u"
                            else f"yk_{kk}"
                        )
                        if abs(cc - 1.0) < 1e-12:
                            ap(f"{indent}        gx_acc_j_{jj} += wi_{ii} * {yexpr} * go_v_{vv};")
                        elif abs(cc + 1.0) < 1e-12:
                            ap(f"{indent}        gx_acc_j_{jj} -= wi_{ii} * {yexpr} * go_v_{vv};")
                        else:
                            ap(f"{indent}        gx_acc_j_{jj} += scalar_t({cc}) * wi_{ii} * {yexpr} * go_v_{vv};")
                ap("")

            # grad_y: both modes accumulate across all vtiles into the same gy_acc_k_*
            for k_tile_id, ktile in enumerate(k_tiles):
                ap(f"{indent}        // grad_y k-tile {k_tile_id}")
                for kk in ktile:
                    for entry in _gy_entries(kk):
                        vv = int(entry["v"])
                        if vv not in vtile:
                            continue
                        ii = int(entry["i"])
                        jj = int(entry["j"])
                        cc = float(entry["c"])
                        if abs(cc - 1.0) < 1e-12:
                            ap(f"{indent}        gy_acc_k_{kk} += wi_{ii} * xj_{jj} * go_v_{vv};")
                        elif abs(cc + 1.0) < 1e-12:
                            ap(f"{indent}        gy_acc_k_{kk} -= wi_{ii} * xj_{jj} * go_v_{vv};")
                        else:
                            ap(f"{indent}        gy_acc_k_{kk} += scalar_t({cc}) * wi_{ii} * xj_{jj} * go_v_{vv};")
                ap("")

        # write grad_w / grad_x
        ap(f"{indent}        // write grad_w")
        for ii in ordered_i:
            if write_modes["grad_w"] == "direct_store":
                ap(f"{indent}        grad_w[gw_base + (int64_t){ii} * (int64_t)U + u] = gw_acc_i_{ii};")
            else:
                ap(f"{indent}        atomicAdd(&grad_w[gw_base + (int64_t){ii} * (int64_t)U + u], gw_acc_i_{ii});")
        ap("")

        ap(f"{indent}        // write grad_x")
        for jj in ordered_j:
            if write_modes["grad_x"] == "atomic":
                ap(f"{indent}        atomicAdd(&grad_x[gx_base + (int64_t){jj} * (int64_t)U + u], gx_acc_j_{jj});")
            else:
                ap(f"{indent}        grad_x[gx_base + (int64_t){jj} * (int64_t)U + u] += gx_acc_j_{jj};")
        ap("")

        # grad_y writeback
        if mode == "u,u,,u":
            ap(f"{indent}        // warp-reduce scalar grad_y over u lanes")
            for kk in ordered_k:
                ap(f"{indent}        scalar_t gy_sum_{kk} = warp_sum_xor(gy_acc_k_{kk});")
                ap(f"{indent}        if (lane == 0) {{")
                if use_y_src:
                    ap(f"{indent}            atomicAdd(&grad_y[gy_base + (int64_t){kk}], gy_sum_{kk});")
                else:
                    ap(f"{indent}            grad_y[gy_base + (int64_t){kk}] += gy_sum_{kk};")
                ap(f"{indent}        }}")
        else:
            ap(f"{indent}        // write vector grad_y[k,u]")
            for kk in ordered_k:
                if use_y_src:
                    ap(f"{indent}        atomicAdd(&grad_y[gy_base + (int64_t){kk} * (int64_t)U + u], gy_acc_k_{kk});")
                else:
                    ap(f"{indent}        grad_y[gy_base + (int64_t){kk} * (int64_t)U + u] += gy_acc_k_{kk};")

        ap(f"{indent}    }}")
        ap(f"{indent}}}")
        ap("")

    # U traversal
    if u_traversal == "grid_y":
        ap("    // u traversal: grid_y tiled-u")
        ap("    int ublk = (int)blockIdx.y;")
        emit_one_u_body("    ", f"ublk * {u_tile} + lane")
    else:
        if u_loop_step is None:
            raise ValueError("schedule has u_traversal=inner_loop but u_loop_step is None")
        ap("    // u traversal: persistent inner loop over U")
        ap(f"    for (int u_base = 0; u_base < U; u_base += {int(u_loop_step)}) {{")
        emit_one_u_body("        ", "u_base + lane")
        ap("    }")
        ap("")

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
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    cudaStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")

    if u_traversal == "inner_loop":
        ap("    dim3 grid((unsigned int)B, 1, 1);")
    else:
        ap(f"    dim3 grid((unsigned int)B, (unsigned int)((U + {u_tile} - 1) / {u_tile}), 1);")

    ap(f"    {kernel_name}<scalar_t><<<grid, block, 0, stream>>>(")
    ap("        w, x_all, y, grad_out, grad_w, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")
    return '\n'.join(lines)


# ============================================================
# Public API
# ============================================================

def generate_code_uniform1d_bwd_fused(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    mode: str = "u,u,,u",
    out_path: str = "generated_uniform1d_bwd_fused.cu",
    kernel_name: str = "uniform1d_bwd_fused",
    scalar_t: str = "float",
    reorder_groups: bool = True,
):
    """
    input_indices:
        - key 1 exists: x uses src indexing
        - key 2 exists: y uses src indexing

    output_indices:
        - key 0 exists: forward out scattered to dst_idx
        - else: forward out shape is [B, V, U]

    backward:
        grad_out layout follows forward out layout.
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

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"
    bundle_name = f"uniform1d_u{u_dim}_path{P}_{mode_str}_{layout_tag}_bwd_fused"

    if reorder_groups:
        i2, j2, k2, v2, c2 = reorder_paths_for_fused_bwd(
            i_list, j_list, k_list, v_list, coeff_list
        )
    else:
        i2, j2, k2, v2, c2 = i_list, j_list, k_list, v_list, coeff_list

    gradw_groups = build_gradw_groups(i2, j2, k2, v2, c2)
    gradx_groups = build_gradx_groups(i2, j2, k2, v2, c2)
    grady_groups = build_grady_groups(i2, j2, k2, v2, c2)

    """ code = emit_fused_bwd_kernel(
        gradw_groups,
        gradx_groups,
        grady_groups,
        kernel_name=bundle_name,
        scalar_t=scalar_t,
        mode=mode,
        u_dim=u_dim,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    ) """

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    coeff_cpu = _to_float_list(coeff_list)

    sched = build_backward_schedule_from_lists(
        i_list=i_cpu,
        j_list=j_cpu,
        k_list=k_cpu,
        v_list=v_cpu,
        coeff_list=coeff_cpu,
        U_dim=u_dim,
    )

    code = emit_fused_bwd_kernel_from_schedule(
        sched,
        kernel_name=bundle_name,
        scalar_t=scalar_t,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    code += "\n"
    code += emit_fused_bwd_launcher(
        bundle_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    return code