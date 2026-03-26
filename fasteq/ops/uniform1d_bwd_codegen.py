from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
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
    ap("    int w_row  = (WB == 1 ? 0 : e_local);")
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
            ap("    int64_t y_base  = (int64_t)e_local * (int64_t)Ky;")
            ap("    int64_t gy_base = (int64_t)e_local * (int64_t)Ky;")
    elif mode == "u,u,u,u":
        if use_y_src:
            ap("    int64_t y_base  = (int64_t)src * (int64_t)Ky * (int64_t)U;")
            ap("    int64_t gy_base = (int64_t)src * (int64_t)Ky * (int64_t)U;")
        else:
            ap("    int64_t y_base  = (int64_t)e_local * (int64_t)Ky * (int64_t)U;")
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

    mode_str = "u_u__u" if mode == "u,u,,u" else "u_u_u_u"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"
    bundle_name = f"uniform1d_combine_u{u_dim}_path{P}_{mode_str}_{layout_tag}_bwd_fused"

    if reorder_groups:
        i2, j2, k2, v2, c2 = reorder_paths_for_fused_bwd(
            i_list, j_list, k_list, v_list, coeff_list
        )
    else:
        i2, j2, k2, v2, c2 = i_list, j_list, k_list, v_list, coeff_list

    gradw_groups = build_gradw_groups(i2, j2, k2, v2, c2)
    gradx_groups = build_gradx_groups(i2, j2, k2, v2, c2)
    grady_groups = build_grady_groups(i2, j2, k2, v2, c2)

    code = emit_fused_bwd_kernel(
        gradw_groups,
        gradx_groups,
        grady_groups,
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