from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, OrderedDict, Counter
from typing import Any, Dict, List, Optional, Tuple, Set
from copy import deepcopy
from types import SimpleNamespace
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
                f"x[x_base + (int64_t){jj} * (int64_t)U + u] * "
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
                    f"x[x_base + (int64_t){jj} * (int64_t)U + u] * "
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
                    f"x[x_base + (int64_t){jj} * (int64_t)U + u] * "
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
    torch::Tensor x,       // [S,Ix,U]
    torch::Tensor y,           // {y_comment}
    torch::Tensor grad_out,    // [B,V,U] or [S,V,U]
{src_decl}{dst_decl}{blist_decl}    int64_t V64)
{{
    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x/y/grad_out must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");
{src_check}{dst_check}
    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x.dim() == 3, "x must be [S,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");

    int B  = {B_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x.size(0);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
    {y_check_u}
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
{src_numel_check}
{dst_numel_check}

{blist_logic}
    auto grad_x = torch::zeros_like(x);
    auto grad_w = torch::zeros_like(w);
{gy_alloc}

    c10::cuda::CUDAGuard device_guard(w.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{
        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");

        launch_{bundle_name}<scalar_t>(
            (const scalar_t*)w.data_ptr<scalar_t>(),
            (const scalar_t*)x.data_ptr<scalar_t>(),
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

def emit_fused_bwd_launcher_no_gradw(
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
    torch::Tensor x,       // [S,Ix,U]
    torch::Tensor y,           // {y_comment}
    torch::Tensor grad_out,    // [B,V,U] or [S,V,U]
{src_decl}{dst_decl}{blist_decl}    int64_t V64)
{{
    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x/y/grad_out must be CUDA");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");
{src_check}{dst_check}
    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x.dim() == 3, "x must be [S,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");

    int B  = {B_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x.size(0);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
    {y_check_u}
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
{src_numel_check}
{dst_numel_check}

{blist_logic}
    auto grad_x = torch::zeros_like(x);
{gy_alloc}

    c10::cuda::CUDAGuard device_guard(w.device());
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{
        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");

        launch_{bundle_name}<scalar_t>(
            (const scalar_t*)w.data_ptr<scalar_t>(),
            (const scalar_t*)x.data_ptr<scalar_t>(),
            (const scalar_t*)y.data_ptr<scalar_t>(),
            (const scalar_t*)grad_out.data_ptr<scalar_t>(),
            (scalar_t*)grad_x.data_ptr<scalar_t>(),
            (scalar_t*)grad_y.data_ptr<scalar_t>(),
            {launch_src_arg}
            {launch_dst_arg}
            b_list_ptr,
            B, WB, Iw, Ix, Ky, V, U, S, stream);
    }});

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {{grad_x, grad_y}};
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} backward fused jit impl (no grad_w)");
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
    ap("    const scalar_t* __restrict__ x,")
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
    ap("    const scalar_t* x,")
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
    ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
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
    ap("    const scalar_t* __restrict__ x,")
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
            ap(f"{indent}        scalar_t xj_{jj} = x[x_base + (int64_t){jj} * (int64_t)U + u];")
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
    ap("    const scalar_t* x,")
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
    ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")
    return '\n'.join(lines)

def emit_fused_bwd_kernel_from_schedule_no_gradw(
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
) -> str:
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")
    if not isinstance(u_dim, int) or u_dim <= 0:
        raise ValueError(f"u_dim must be positive int, got {u_dim}")

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
    j_tiles = schedule["tiling"]["j_tiles"]
    k_tiles = schedule["tiling"]["k_tiles"]

    grad_x_groups = schedule["groups"]["grad_x"]
    grad_y_groups = schedule["groups"]["grad_y"]

    key_is_str_gx = len(grad_x_groups) > 0 and isinstance(next(iter(grad_x_groups.keys())), str)
    key_is_str_gy = len(grad_y_groups) > 0 and isinstance(next(iter(grad_y_groups.keys())), str)

    def _gx_entries(jj: int):
        return grad_x_groups[str(jj)] if key_is_str_gx else grad_x_groups[jj]

    def _gy_entries(kk: int):
        return grad_y_groups[str(kk)] if key_is_str_gy else grad_y_groups[kk]

    # ---- compile-time constant offsets: segment offsets inside one row ----
    wi_offsets = {ii: ii * u_dim for ii in ordered_i}
    xj_offsets = {jj: jj * u_dim for jj in ordered_j}
    yk_u_offsets = {kk: kk * u_dim for kk in ordered_k}
    gy_u_offsets = {kk: kk * u_dim for kk in ordered_k}
    gx_u_offsets = {jj: jj * u_dim for jj in ordered_j}

    all_vs = []
    seen_v = set()
    for tile in v_tiles:
        for vv in tile:
            vv = int(vv)
            if vv not in seen_v:
                seen_v.add(vv)
                all_vs.append(vv)
    go_u_offsets = {vv: vv * u_dim for vv in all_vs}

    # ---- optional compile-time constant row strides ----
    w_row_stride_const = None if iw_dim is None else int(iw_dim) * u_dim
    x_row_stride_const = None if ix_dim is None else int(ix_dim) * u_dim
    y_row_stride_const = None if (mode == "u,u,,u" or ky_dim is None) else int(ky_dim) * u_dim
    go_row_stride_const = None if v_dim is None else int(v_dim) * u_dim

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
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
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

    # consistency check comment
    ap("    // Note: this generated kernel assumes compile-time specialized U.")
    ap("    // Caller should guarantee runtime U matches U_CONST.")
    ap("")

    # w_base
    if w_row_stride_const is not None:
        ap(f"    const int64_t w_base = (int64_t)w_row * {w_row_stride_const};")
    else:
        ap("    const int64_t w_base = (int64_t)w_row * (int64_t)Iw * (int64_t)U;")

    # x_base / gx_base
    if x_row_stride_const is not None:
        if use_x_src:
            ap(f"    const int64_t x_base  = (int64_t)src * {x_row_stride_const};")
            ap(f"    const int64_t gx_base = (int64_t)src * {x_row_stride_const};")
        else:
            ap(f"    const int64_t x_base  = (int64_t)e_local * {x_row_stride_const};")
            ap(f"    const int64_t gx_base = (int64_t)e_local * {x_row_stride_const};")
    else:
        if use_x_src:
            ap("    const int64_t x_base  = (int64_t)src * (int64_t)Ix * (int64_t)U;")
            ap("    const int64_t gx_base = (int64_t)src * (int64_t)Ix * (int64_t)U;")
        else:
            ap("    const int64_t x_base  = (int64_t)e_local * (int64_t)Ix * (int64_t)U;")
            ap("    const int64_t gx_base = (int64_t)e_local * (int64_t)Ix * (int64_t)U;")

    # y_base / gy_base
    if mode == "u,u,,u":
        if use_y_src:
            ap("    const int64_t y_base  = (int64_t)src * (int64_t)Ky;")
            ap("    const int64_t gy_base = (int64_t)src * (int64_t)Ky;")
        else:
            ap("    const int64_t y_base  = (int64_t)e_orig * (int64_t)Ky;")
            ap("    const int64_t gy_base = (int64_t)e_orig * (int64_t)Ky;")
    else:
        if y_row_stride_const is not None:
            if use_y_src:
                ap(f"    const int64_t y_base  = (int64_t)src * {y_row_stride_const};")
                ap(f"    const int64_t gy_base = (int64_t)src * {y_row_stride_const};")
            else:
                ap(f"    const int64_t y_base  = (int64_t)e_orig * {y_row_stride_const};")
                ap(f"    const int64_t gy_base = (int64_t)e_orig * {y_row_stride_const};")
        else:
            if use_y_src:
                ap("    const int64_t y_base  = (int64_t)src * (int64_t)Ky * (int64_t)U;")
                ap("    const int64_t gy_base = (int64_t)src * (int64_t)Ky * (int64_t)U;")
            else:
                ap("    const int64_t y_base  = (int64_t)e_orig * (int64_t)Ky * (int64_t)U;")
                ap("    const int64_t gy_base = (int64_t)e_orig * (int64_t)Ky * (int64_t)U;")

    # go_base
    if go_row_stride_const is not None:
        if use_scatter:
            ap(f"    const int64_t go_base = (int64_t)dst * {go_row_stride_const};")
        else:
            ap(f"    const int64_t go_base = (int64_t)e_local * {go_row_stride_const};")
    else:
        if use_scatter:
            ap("    const int64_t go_base = (int64_t)dst * (int64_t)V * (int64_t)U;")
        else:
            ap("    const int64_t go_base = (int64_t)e_local * (int64_t)V * (int64_t)U;")
    ap("")

    def emit_one_u_body(indent: str, u_expr: str):
        ap(f"{indent}{{")
        ap(f"{indent}    int u = {u_expr};")
        ap(f"{indent}    if (u < U) {{")

        ap(f"{indent}        // preload wi(i,u) for grad_y")
        for ii in ordered_i:
            wi_off = wi_offsets[ii]
            ap(f"{indent}        scalar_t wi_{ii} = w[w_base + {wi_off} + u];")
        ap("")

        ap(f"{indent}        // preload xj(j,u)")
        for jj in ordered_j:
            xj_off = xj_offsets[jj]
            ap(f"{indent}        scalar_t xj_{jj} = x[x_base + {xj_off} + u];")
        ap("")

        if mode == "u,u,,u":
            ap(f"{indent}        // preload scalar yk(k)")
            for kk in ordered_k:
                ap(f"{indent}        scalar_t yk_{kk} = y[y_base + {kk}];")
            ap("")

        ap(f"{indent}        // init accumulators")
        for jj in ordered_j:
            ap(f"{indent}        scalar_t gx_acc_j_{jj} = scalar_t(0);")
        for kk in ordered_k:
            ap(f"{indent}        scalar_t gy_acc_k_{kk} = scalar_t(0);")
        ap("")

        for tile_id, vtile in enumerate(v_tiles):
            ap(f"{indent}        // ---- v tile {tile_id}: {vtile} ----")
            for vv in vtile:
                go_off = go_u_offsets[int(vv)]
                ap(f"{indent}        scalar_t go_v_{vv} = grad_out[go_base + {go_off} + u];")
            ap("")

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

                        if mode == "u,u,u,u":
                            y_off = yk_u_offsets[kk]
                            yexpr = f"y[y_base + {y_off} + u]"
                        else:
                            yexpr = f"yk_{kk}"

                        if abs(cc - 1.0) < 1e-12:
                            ap(f"{indent}        gx_acc_j_{jj} += wi_{ii} * ({yexpr}) * go_v_{vv};")
                        elif abs(cc + 1.0) < 1e-12:
                            ap(f"{indent}        gx_acc_j_{jj} -= wi_{ii} * ({yexpr}) * go_v_{vv};")
                        else:
                            ap(f"{indent}        gx_acc_j_{jj} += scalar_t({cc}) * wi_{ii} * ({yexpr}) * go_v_{vv};")
                ap("")

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

        ap(f"{indent}        // write grad_x")
        for jj in ordered_j:
            gx_off = gx_u_offsets[jj]
            if write_modes["grad_x"] == "atomic":
                ap(f"{indent}        atomicAdd(&grad_x[gx_base + {gx_off} + u], gx_acc_j_{jj});")
            else:
                ap(f"{indent}        grad_x[gx_base + {gx_off} + u] += gx_acc_j_{jj};")
        ap("")

        if mode == "u,u,,u":
            ap(f"{indent}        // warp-reduce scalar grad_y over u lanes")
            for kk in ordered_k:
                ap(f"{indent}        scalar_t gy_sum_{kk} = warp_sum_xor(gy_acc_k_{kk});")
                ap(f"{indent}        if (lane == 0) {{")
                if use_y_src:
                    ap(f"{indent}            atomicAdd(&grad_y[gy_base + {kk}], gy_sum_{kk});")
                else:
                    ap(f"{indent}            grad_y[gy_base + {kk}] += gy_sum_{kk};")
                ap(f"{indent}        }}")
        else:
            ap(f"{indent}        // write vector grad_y[k,u]")
            for kk in ordered_k:
                gy_off = gy_u_offsets[kk]
                if use_y_src:
                    ap(f"{indent}        atomicAdd(&grad_y[gy_base + {gy_off} + u], gy_acc_k_{kk});")
                else:
                    ap(f"{indent}        grad_y[gy_base + {gy_off} + u] += gy_acc_k_{kk};")

        ap(f"{indent}    }}")
        ap(f"{indent}}}")
        ap("")

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
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
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
    ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    return '\n'.join(lines)


@dataclass(frozen=True)
class CGPath:
    i: int
    j: int
    k: int
    v: int
    c: float


@dataclass
class GroupPlanGXGY:
    group_id: int
    uniq_i: List[int]
    uniq_j: List[int]
    uniq_k: List[int]
    uniq_v: List[int]

    i_slot_map: Dict[int, int]

    # 这里只给“值得 preload”的 x/y 分配 slot
    x_slot_map: Dict[int, int]
    y_slot_map: Dict[int, int]

    v_slot_map: Dict[int, int]

    # 真正参与 preload 的 j/k
    uniq_j_preload: List[int]
    uniq_k_preload: List[int]

    # 原始展开 term（保留 sanity/debug）
    terms: List[Dict[str, Any]]

    # 聚合后的小组：wg = c * wi_slot * go_slot
    mul_groups: List[Dict[str, Any]]

    num_paths: int
    reg_usage_inputs: int  # = |uniq_i| + |uniq_j_preload| + |uniq_k_preload| + |uniq_v|


@dataclass
class SchedulerPlanGXGY:
    reg_budget: int
    n_acc_gx: int
    n_acc_gy: int
    n_acc_total: int
    r_rem: int
    uniq_i_all: List[int]
    uniq_j_all: List[int]
    uniq_k_all: List[int]
    uniq_v_all: List[int]
    gx_acc_slot_map: Dict[int, int]
    gy_acc_slot_map: Dict[int, int]
    iv_order: Tuple[str, str]
    jk_order: Tuple[str, str]
    groups: List[GroupPlanGXGY]


def _stable_slot_map(vals: List[int]) -> Dict[int, int]:
    return {x: s for s, x in enumerate(vals)}


def _choose_pair_order(n_first: int, n_second: int, first_name: str, second_name: str) -> Tuple[str, str]:
    if n_first <= n_second:
        return (first_name, second_name)
    return (second_name, first_name)


def _lookup_key(p: CGPath, name: str) -> int:
    if name == "i":
        return p.i
    if name == "j":
        return p.j
    if name == "k":
        return p.k
    if name == "v":
        return p.v
    raise ValueError(name)


def _path_sort_key_global(p: CGPath, iv_order: Tuple[str, str], jk_order: Tuple[str, str]):
    return (
        _lookup_key(p, iv_order[0]),
        _lookup_key(p, iv_order[1]),
        _lookup_key(p, jk_order[0]),
        _lookup_key(p, jk_order[1]),
    )


def _path_sort_key_in_group(p: CGPath, iv_order: Tuple[str, str], jk_order: Tuple[str, str]):
    return (
        _lookup_key(p, jk_order[0]),
        _lookup_key(p, jk_order[1]),
        _lookup_key(p, iv_order[0]),
        _lookup_key(p, iv_order[1]),
    )


def _build_xy_preload_slot_maps(paths: List[CGPath]):
    """
    只对组内复用 >= 2 的 j/k 分配 preload slot。
    """
    j_cnt = Counter(p.j for p in paths)
    k_cnt = Counter(p.k for p in paths)

    uniq_j_preload = sorted(j for j, cnt in j_cnt.items() if cnt >= 2)
    uniq_k_preload = sorted(k for k, cnt in k_cnt.items() if cnt >= 2)

    x_slot_map = _stable_slot_map(uniq_j_preload)
    y_slot_map = _stable_slot_map(uniq_k_preload)
    return uniq_j_preload, uniq_k_preload, x_slot_map, y_slot_map


def _estimate_group_input_regs_with_xy_reuse(paths: List[CGPath]) -> int:
    """
    group cut 预算估计：
      wi -> uniq_i
      go -> uniq_v
      x  -> 仅统计组内复用 >= 2 的 j
      y  -> 仅统计组内复用 >= 2 的 k
    """
    if not paths:
        return 0

    uniq_i = {p.i for p in paths}
    uniq_v = {p.v for p in paths}

    j_cnt = Counter(p.j for p in paths)
    k_cnt = Counter(p.k for p in paths)

    num_x_preload = sum(1 for _, cnt in j_cnt.items() if cnt >= 2)
    num_y_preload = sum(1 for _, cnt in k_cnt.items() if cnt >= 2)

    return len(uniq_i) + num_x_preload + num_y_preload + len(uniq_v)


def _build_mul_groups(
    paths: List[CGPath],
    i_slot_map: Dict[int, int],
    v_slot_map: Dict[int, int],
    x_slot_map: Dict[int, int],
    y_slot_map: Dict[int, int],
):
    """
    把 term 按 (coeff, wi_slot, go_slot) 聚合：
        wg = coeff * wi_slot * go_slot
    """
    buckets = defaultdict(list)
    for p in paths:
        wi_slot = i_slot_map[p.i]
        go_slot = v_slot_map[p.v]
        key = (float(p.c), wi_slot, go_slot)
        buckets[key].append(p)

    mul_groups: List[Dict[str, Any]] = []
    for (c, wi_slot, go_slot), plist in buckets.items():
        gx_terms = []
        gy_terms = []
        for p in plist:
            gx_terms.append(
                {
                    "j": int(p.j),
                    "k": int(p.k),
                    "use_y_slot": (p.k in y_slot_map),
                    "y_slot": int(y_slot_map[p.k]) if p.k in y_slot_map else -1,
                }
            )
            gy_terms.append(
                {
                    "k": int(p.k),
                    "j": int(p.j),
                    "use_x_slot": (p.j in x_slot_map),
                    "x_slot": int(x_slot_map[p.j]) if p.j in x_slot_map else -1,
                }
            )

        mul_groups.append(
            {
                "c": float(c),
                "wi_slot": int(wi_slot),
                "go_slot": int(go_slot),
                "gx_terms": gx_terms,
                "gy_terms": gy_terms,
            }
        )

    mul_groups.sort(key=lambda g: (g["wi_slot"], g["go_slot"], g["c"]))
    return mul_groups


def _make_group_plan(
    group_id: int,
    paths: List[CGPath],
    iv_order: Tuple[str, str],
    jk_order: Tuple[str, str],
) -> GroupPlanGXGY:
    paths = sorted(paths, key=lambda p: _path_sort_key_in_group(p, iv_order, jk_order))

    uniq_i = sorted({p.i for p in paths})
    uniq_j = sorted({p.j for p in paths})
    uniq_k = sorted({p.k for p in paths})
    uniq_v = sorted({p.v for p in paths})

    i_slot_map = _stable_slot_map(uniq_i)
    v_slot_map = _stable_slot_map(uniq_v)

    uniq_j_preload, uniq_k_preload, x_slot_map, y_slot_map = _build_xy_preload_slot_maps(paths)

    terms: List[Dict[str, Any]] = []
    for p in paths:
        terms.append(
            {
                "i": p.i,
                "j": p.j,
                "k": p.k,
                "v": p.v,
                "c": p.c,
                "wi_slot": i_slot_map[p.i],
                "x_slot": x_slot_map[p.j] if p.j in x_slot_map else -1,
                "y_slot": y_slot_map[p.k] if p.k in y_slot_map else -1,
                "go_slot": v_slot_map[p.v],
                "use_x_slot": p.j in x_slot_map,
                "use_y_slot": p.k in y_slot_map,
            }
        )

    mul_groups = _build_mul_groups(
        paths=paths,
        i_slot_map=i_slot_map,
        v_slot_map=v_slot_map,
        x_slot_map=x_slot_map,
        y_slot_map=y_slot_map,
    )

    return GroupPlanGXGY(
        group_id=group_id,
        uniq_i=uniq_i,
        uniq_j=uniq_j,
        uniq_k=uniq_k,
        uniq_v=uniq_v,
        i_slot_map=i_slot_map,
        x_slot_map=x_slot_map,
        y_slot_map=y_slot_map,
        v_slot_map=v_slot_map,
        uniq_j_preload=uniq_j_preload,
        uniq_k_preload=uniq_k_preload,
        terms=terms,
        mul_groups=mul_groups,
        num_paths=len(paths),
        reg_usage_inputs=len(uniq_i) + len(uniq_j_preload) + len(uniq_k_preload) + len(uniq_v),
    )


def _sanity_check_plan(plan: SchedulerPlanGXGY, globally_sorted_paths: List[CGPath]) -> None:
    flat_terms: List[Tuple[int, int, int, int, float]] = []
    for g in plan.groups:
        if g.reg_usage_inputs > plan.r_rem:
            raise AssertionError(
                f"group {g.group_id} exceeds r_rem: "
                f"{g.reg_usage_inputs} > {plan.r_rem}"
            )
        for t in g.terms:
            flat_terms.append((t["i"], t["j"], t["k"], t["v"], float(t["c"])))

    a = sorted(flat_terms)
    b = sorted((p.i, p.j, p.k, p.v, float(p.c)) for p in globally_sorted_paths)
    if a != b:
        raise AssertionError("flattened scheduled terms do not match original paths multiset")


# =============================================================================
# Phase + Subphase scheduler
#   - gw/gx/gy accumulators are full-resident across all phases/subphases
#   - outer loop: repeatedly choose pivot from remaining paths
#   - each phase owns a disjoint subset of paths (same pivot value)
#   - each phase is split into subphases
#   - each subphase uses a fixed operand working set under a reg-pool budget
#   - if the pool is full, use future-count eviction; if not worth replacing,
#     the missing operand is direct-loaded for that op only
#   - intentionally DOES NOT introduce extra temporaries like c*wi*go or c*go*x
# =============================================================================


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class SubphaseOp:
    i: int
    j: int
    k: int
    v: int
    c: float

    gw_acc_slot: int
    gx_acc_slot: int
    gy_acc_slot: int

    use_x_slot: bool
    x_slot: int
    use_y_slot: bool
    y_slot: int
    use_wi_slot: bool
    wi_slot: int
    use_go_slot: bool
    go_slot: int


@dataclass
class SubphasePlan:
    subphase_id: int

    x_vals: List[int]
    y_vals: List[int]
    wi_vals: List[int]
    go_vals: List[int]

    x_slot_map: Dict[int, int]
    y_slot_map: Dict[int, int]
    wi_slot_map: Dict[int, int]
    go_slot_map: Dict[int, int]

    path_ids: List[int]
    ops: List[SubphaseOp]
    reg_usage_inputs: int


@dataclass
class PhasePlan:
    phase_id: int
    pivot_kind: str   # 'x' | 'y' | 'wi' | 'go'
    pivot_index: int

    path_ids: List[int]
    subphases: List[SubphasePlan]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _operand_key(kind: str, idx: int) -> Tuple[str, int]:
    return (str(kind), int(idx))


def _operand_sort_key(kind: str, idx: int) -> Tuple[int, int]:
    pri = {"x": 0, "y": 1, "wi": 2, "go": 3}[kind]
    return (pri, int(idx))


def _path_matches_pivot(p, pivot_kind: str, pivot_index: int) -> bool:
    if pivot_kind == "x":
        return int(p.j) == int(pivot_index)
    if pivot_kind == "y":
        return int(p.k) == int(pivot_index)
    if pivot_kind == "wi":
        return int(p.i) == int(pivot_index)
    if pivot_kind == "go":
        return int(p.v) == int(pivot_index)
    raise ValueError(f"Unsupported pivot_kind={pivot_kind}")


def _remaining_operand_counts(paths: List[Tuple[int, Any]]) -> Dict[str, Dict[int, int]]:
    cx = defaultdict(int)
    cy = defaultdict(int)
    cw = defaultdict(int)
    cg = defaultdict(int)
    for _, p in paths:
        cx[int(p.j)] += 1
        cy[int(p.k)] += 1
        cw[int(p.i)] += 1
        cg[int(p.v)] += 1
    return {
        "x": dict(cx),
        "y": dict(cy),
        "wi": dict(cw),
        "go": dict(cg),
    }


def _choose_pivot_from_remaining(
    remaining_paths: List[Tuple[int, Any]],
) -> Tuple[str, int, Dict[str, Dict[int, int]]]:
    counts = _remaining_operand_counts(remaining_paths)

    best_kind = None
    best_idx = None
    best_score = -1
    best_addr = None

    for kind in ("x", "y", "wi", "go"):
        for idx, cnt in counts[kind].items():
            sc = int(cnt)
            ak = _operand_sort_key(kind, int(idx))
            if sc > best_score or (sc == best_score and (best_addr is None or ak < best_addr)):
                best_kind = kind
                best_idx = int(idx)
                best_score = sc
                best_addr = ak

    if best_kind is None:
        raise RuntimeError("No pivot found from remaining paths")
    return best_kind, best_idx, counts


def _op_operands(p) -> List[Tuple[str, int]]:
    return [
        ("x", int(p.j)),
        ("y", int(p.k)),
        ("wi", int(p.i)),
        ("go", int(p.v)),
    ]


def _future_counts_for_suffix(paths: List[Tuple[int, Any]], start: int) -> Dict[Tuple[str, int], int]:
    fut = defaultdict(int)
    for idx in range(start, len(paths)):
        _, p = paths[idx]
        fut[_operand_key("x", int(p.j))] += 1
        fut[_operand_key("y", int(p.k))] += 1
        fut[_operand_key("wi", int(p.i))] += 1
        fut[_operand_key("go", int(p.v))] += 1
    return dict(fut)


def _phase_path_priority(p, pivot_kind: str) -> Tuple:
    if pivot_kind == "x":
        return (int(p.j), int(p.k), int(p.i), int(p.v), float(p.c))
    if pivot_kind == "y":
        return (int(p.k), int(p.j), int(p.i), int(p.v), float(p.c))
    if pivot_kind == "wi":
        return (int(p.i), int(p.v), int(p.j), int(p.k), float(p.c))
    return (int(p.v), int(p.i), int(p.j), int(p.k), float(p.c))


# -----------------------------------------------------------------------------
# Subphase construction with future-count eviction
# -----------------------------------------------------------------------------


def _build_subphases_for_phase(
    candidate_paths: List[Tuple[int, Any]],
    *,
    pivot_kind: str,
    pivot_index: int,
    reg_budget_inputs: int,
    gw_acc_slot_map: Dict[int, int],
    gx_acc_slot_map: Dict[int, int],
    gy_acc_slot_map: Dict[int, int],
) -> List[SubphasePlan]:
    if reg_budget_inputs <= 0:
        raise ValueError("reg_budget_inputs must be positive")

    candidate_paths = list(candidate_paths)
    candidate_paths.sort(key=lambda it: _phase_path_priority(it[1], pivot_kind))

    pivot_operand = _operand_key(pivot_kind, pivot_index)

    subphases: List[SubphasePlan] = []
    sid = 0
    cursor = 0

    while cursor < len(candidate_paths):
        # Each subphase starts with the pivot protected in the pool.
        pool: Set[Tuple[str, int]] = {pivot_operand}
        protected: Set[Tuple[str, int]] = {pivot_operand}

        subphase_items: List[Tuple[int, Any, Dict[str, bool]]] = []
        used_paths: List[int] = []

        while cursor < len(candidate_paths):
            path_id, p = candidate_paths[cursor]
            fut = _future_counts_for_suffix(candidate_paths, cursor + 1)

            needed = _op_operands(p)
            present = {op for op in needed if op in pool}
            missing = [op for op in needed if op not in pool]

            if not missing:
                slot_usage = {key: True for key in needed}
                subphase_items.append((path_id, p, slot_usage))
                used_paths.append(int(path_id))
                cursor += 1
                continue

            # Try greedy insertion / replacement per missing operand.
            local_pool = set(pool)
            slot_usage = {key: (key in local_pool) for key in needed}
            can_continue_this_subphase = True

            for new_op in missing:
                if new_op in local_pool:
                    slot_usage[new_op] = True
                    continue

                if len(local_pool) < reg_budget_inputs:
                    local_pool.add(new_op)
                    slot_usage[new_op] = True
                    continue

                # pool full: replace only if new future count is better than the worst victim.
                victim = None
                victim_score = None
                for old_op in local_pool:
                    if old_op in protected:
                        continue
                    sc = int(fut.get(old_op, 0))
                    if victim is None or sc < victim_score or (sc == victim_score and old_op < victim):
                        victim = old_op
                        victim_score = sc

                new_score = int(fut.get(new_op, 0))
                if victim is None:
                    # Nothing replaceable. Use direct load.
                    slot_usage[new_op] = False
                    continue

                if new_score > int(victim_score):
                    # End current subphase here and restart from this path with a new pool.
                    can_continue_this_subphase = False
                    break
                else:
                    # Not worth replacing; keep as direct load for this op.
                    slot_usage[new_op] = False
                    continue

            if not can_continue_this_subphase:
                break

            # Commit this path and any worthwhile inserted operands.
            for opk, use_slot in slot_usage.items():
                if use_slot:
                    pool.add(opk)
            subphase_items.append((path_id, p, slot_usage))
            used_paths.append(int(path_id))
            cursor += 1

        # Safety: always make progress, even if the first path could not extend the pool.
        if not subphase_items:
            path_id, p = candidate_paths[cursor]
            needed = _op_operands(p)
            slot_usage = {key: (key == pivot_operand) for key in needed}
            subphase_items.append((path_id, p, slot_usage))
            used_paths.append(int(path_id))
            cursor += 1
            pool = {pivot_operand}

        # Build compact slot maps from the actual subphase pool.
        x_vals = sorted(idx for kind, idx in pool if kind == "x")
        y_vals = sorted(idx for kind, idx in pool if kind == "y")
        wi_vals = sorted(idx for kind, idx in pool if kind == "wi")
        go_vals = sorted(idx for kind, idx in pool if kind == "go")

        x_slot_map = _stable_slot_map(x_vals)
        y_slot_map = _stable_slot_map(y_vals)
        wi_slot_map = _stable_slot_map(wi_vals)
        go_slot_map = _stable_slot_map(go_vals)

        ops: List[SubphaseOp] = []
        for path_id, p, slot_usage in subphase_items:
            ux = bool(slot_usage.get(_operand_key("x", int(p.j)), False) and int(p.j) in x_slot_map)
            uy = bool(slot_usage.get(_operand_key("y", int(p.k)), False) and int(p.k) in y_slot_map)
            uw = bool(slot_usage.get(_operand_key("wi", int(p.i)), False) and int(p.i) in wi_slot_map)
            ug = bool(slot_usage.get(_operand_key("go", int(p.v)), False) and int(p.v) in go_slot_map)
            ops.append(
                SubphaseOp(
                    i=int(p.i),
                    j=int(p.j),
                    k=int(p.k),
                    v=int(p.v),
                    c=float(p.c),
                    gw_acc_slot=int(gw_acc_slot_map[int(p.i)]),
                    gx_acc_slot=int(gx_acc_slot_map[int(p.j)]),
                    gy_acc_slot=int(gy_acc_slot_map[int(p.k)]),
                    use_x_slot=ux,
                    x_slot=int(x_slot_map[int(p.j)]) if ux else -1,
                    use_y_slot=uy,
                    y_slot=int(y_slot_map[int(p.k)]) if uy else -1,
                    use_wi_slot=uw,
                    wi_slot=int(wi_slot_map[int(p.i)]) if uw else -1,
                    use_go_slot=ug,
                    go_slot=int(go_slot_map[int(p.v)]) if ug else -1,
                )
            )

        subphases.append(
            SubphasePlan(
                subphase_id=int(sid),
                x_vals=x_vals,
                y_vals=y_vals,
                wi_vals=wi_vals,
                go_vals=go_vals,
                x_slot_map=dict(x_slot_map),
                y_slot_map=dict(y_slot_map),
                wi_slot_map=dict(wi_slot_map),
                go_slot_map=dict(go_slot_map),
                path_ids=list(used_paths),
                ops=ops,
                reg_usage_inputs=len(x_vals) + len(y_vals) + len(wi_vals) + len(go_vals),
            )
        )
        sid += 1

    return subphases


# -----------------------------------------------------------------------------
# Public schedule builder
# -----------------------------------------------------------------------------


def schedule_gradw_gradx_grady_full_acc_subphase(
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
    reg_budget: int,
    *,
    dynamic_iv_order: bool = True,
    dynamic_jk_order: bool = True,
    sanity_check: bool = True,
) -> Dict[str, Any]:
    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

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

    gw_acc_slot_map = _stable_slot_map(uniq_i_all)
    gx_acc_slot_map = _stable_slot_map(uniq_j_all)
    gy_acc_slot_map = _stable_slot_map(uniq_k_all)

    n_acc_gw = len(uniq_i_all)
    n_acc_gx = len(uniq_j_all)
    n_acc_gy = len(uniq_k_all)
    n_acc_total = n_acc_gw + n_acc_gx + n_acc_gy

    r_rem = reg_budget - n_acc_total
    if r_rem <= 0:
        raise ValueError(
            f"reg_budget={reg_budget} is too small: need n_acc_total={n_acc_total}, "
            f"where n_acc_gw={n_acc_gw}, n_acc_gx={n_acc_gx}, n_acc_gy={n_acc_gy}"
        )

    if dynamic_iv_order:
        iv_order = _choose_pair_order(len(uniq_i_all), len(uniq_v_all), "i", "v")
    else:
        iv_order = ("i", "v")

    if dynamic_jk_order:
        jk_order = _choose_pair_order(len(uniq_j_all), len(uniq_k_all), "j", "k")
    else:
        jk_order = ("j", "k")

    enumerated_paths: List[Tuple[int, Any]] = list(enumerate(paths))
    enumerated_paths.sort(key=lambda it: _path_sort_key_global(it[1], iv_order, jk_order))

    phases: List[PhasePlan] = []
    remaining_paths = list(enumerated_paths)
    phase_id = 0

    while remaining_paths:
        pivot_kind, pivot_index, _ = _choose_pivot_from_remaining(remaining_paths)
        candidate_paths = [it for it in remaining_paths if _path_matches_pivot(it[1], pivot_kind, pivot_index)]
        if not candidate_paths:
            raise RuntimeError("Pivot chosen but no candidate paths found")

        subphases = _build_subphases_for_phase(
            candidate_paths,
            pivot_kind=pivot_kind,
            pivot_index=pivot_index,
            reg_budget_inputs=r_rem,
            gw_acc_slot_map=gw_acc_slot_map,
            gx_acc_slot_map=gx_acc_slot_map,
            gy_acc_slot_map=gy_acc_slot_map,
        )

        path_ids: List[int] = []
        for sp in subphases:
            path_ids.extend(sp.path_ids)

        phases.append(
            PhasePlan(
                phase_id=int(phase_id),
                pivot_kind=str(pivot_kind),
                pivot_index=int(pivot_index),
                path_ids=list(path_ids),
                subphases=subphases,
            )
        )

        done_set = set(path_ids)
        remaining_paths = [it for it in remaining_paths if int(it[0]) not in done_set]
        phase_id += 1

    if sanity_check:
        seen = []
        for ph in phases:
            seen.extend(ph.path_ids)
            for sp in ph.subphases:
                if sp.reg_usage_inputs > r_rem:
                    raise AssertionError(
                        f"subphase exceeds reg budget: {sp.reg_usage_inputs} > {r_rem}"
                    )
        if sorted(seen) != list(range(len(paths))):
            raise AssertionError("phase/subphase scheduler did not cover each path exactly once")

    schedule: Dict[str, Any] = {
        "kernel_mode": "gradw_gradx_grady_fullacc_phase_subphase_operand_pool",
        "reg_budget": int(reg_budget),
        "n_acc_gw": int(n_acc_gw),
        "n_acc_gx": int(n_acc_gx),
        "n_acc_gy": int(n_acc_gy),
        "n_acc_total": int(n_acc_total),
        "r_rem": int(r_rem),
        "uniq_i_all": list(uniq_i_all),
        "uniq_j_all": list(uniq_j_all),
        "uniq_k_all": list(uniq_k_all),
        "uniq_v_all": list(uniq_v_all),
        "gw_acc_slot_map": dict(gw_acc_slot_map),
        "gx_acc_slot_map": dict(gx_acc_slot_map),
        "gy_acc_slot_map": dict(gy_acc_slot_map),
        "iv_order": tuple(iv_order),
        "jk_order": tuple(jk_order),
        "phases": [],
    }

    for ph in phases:
        ph_dict = {
            "phase_id": int(ph.phase_id),
            "pivot_kind": str(ph.pivot_kind),
            "pivot_index": int(ph.pivot_index),
            "path_ids": list(ph.path_ids),
            "subphases": [],
        }
        for sp in ph.subphases:
            ph_dict["subphases"].append(
                {
                    "subphase_id": int(sp.subphase_id),
                    "x_vals": list(sp.x_vals),
                    "y_vals": list(sp.y_vals),
                    "wi_vals": list(sp.wi_vals),
                    "go_vals": list(sp.go_vals),
                    "x_slot_map": dict(sp.x_slot_map),
                    "y_slot_map": dict(sp.y_slot_map),
                    "wi_slot_map": dict(sp.wi_slot_map),
                    "go_slot_map": dict(sp.go_slot_map),
                    "path_ids": list(sp.path_ids),
                    "reg_usage_inputs": int(sp.reg_usage_inputs),
                    "ops": [
                        {
                            "i": op.i,
                            "j": op.j,
                            "k": op.k,
                            "v": op.v,
                            "c": float(op.c),
                            "gw_acc_slot": int(op.gw_acc_slot),
                            "gx_acc_slot": int(op.gx_acc_slot),
                            "gy_acc_slot": int(op.gy_acc_slot),
                            "use_x_slot": bool(op.use_x_slot),
                            "x_slot": int(op.x_slot),
                            "use_y_slot": bool(op.use_y_slot),
                            "y_slot": int(op.y_slot),
                            "use_wi_slot": bool(op.use_wi_slot),
                            "wi_slot": int(op.wi_slot),
                            "use_go_slot": bool(op.use_go_slot),
                            "go_slot": int(op.go_slot),
                        }
                        for op in sp.ops
                    ],
                }
            )
        schedule["phases"].append(ph_dict)

    return schedule


# -----------------------------------------------------------------------------
# Codegen
# -----------------------------------------------------------------------------
def emit_fused_bwd_kernel_from_subphase_schedule_with_gradw(
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
    write_grad_w_atomic: bool = False,
    write_grad_x_atomic: bool = True,
    write_grad_y_atomic: bool = True,
    enable_wg_subphase_cache: bool = True,
    wg_min_reuse: int = 2,
    wg_max_slots_per_subphase: int = 2,
    emit_dual_index_launcher: bool = True,
) -> str:
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")
    if not isinstance(u_dim, int) or u_dim <= 0:
        raise ValueError(f"u_dim must be positive int, got {u_dim}")
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported, got {block_size}")

    uniq_i_all = [int(x) for x in schedule["uniq_i_all"]]
    uniq_j_all = [int(x) for x in schedule["uniq_j_all"]]
    uniq_k_all = [int(x) for x in schedule["uniq_k_all"]]
    phases = schedule["phases"]

    gw_u_offsets = {ii: ii * u_dim for ii in uniq_i_all}
    gx_u_offsets = {jj: jj * u_dim for jj in uniq_j_all}
    gy_u_offsets = {kk: kk * u_dim for kk in uniq_k_all}

    w_row_stride_const = None if iw_dim is None else int(iw_dim) * u_dim
    x_row_stride_const = None if ix_dim is None else int(ix_dim) * u_dim
    y_row_stride_const = None if (mode == "u,u,,u" or ky_dim is None) else int(ky_dim) * u_dim
    go_row_stride_const = None if v_dim is None else int(v_dim) * u_dim

    max_wi_slot = -1
    max_go_slot = -1
    max_x_slot = -1
    max_y_slot = -1
    for ph in phases:
        for sp in ph["subphases"]:
            for _, s in sp["wi_slot_map"].items():
                max_wi_slot = max(max_wi_slot, int(s))
            for _, s in sp["go_slot_map"].items():
                max_go_slot = max(max_go_slot, int(s))
            for _, s in sp["x_slot_map"].items():
                max_x_slot = max(max_x_slot, int(s))
            for _, s in sp["y_slot_map"].items():
                max_y_slot = max(max_y_slot, int(s))

    lines: List[str] = []
    ap = lines.append

    def fmt_float(x: float) -> str:
        return repr(float(x))

    def x_expr(op: Dict[str, Any]) -> str:
        if op["use_x_slot"]:
            return f"x_slot_{int(op['x_slot'])}"
        return f"x[x_base + (index_t){int(op['j']) * u_dim} + (index_t)u]"

    def y_expr(op: Dict[str, Any]) -> str:
        if op["use_y_slot"]:
            return f"y_slot_{int(op['y_slot'])}"
        if mode == "u,u,,u":
            return f"y[y_base + (index_t){int(op['k'])}]"
        return f"y[y_base + (index_t){int(op['k']) * u_dim} + (index_t)u]"

    def wi_expr(op: Dict[str, Any]) -> str:
        if op["use_wi_slot"]:
            return f"wi_slot_{int(op['wi_slot'])}"
        return f"w[w_base + (index_t){int(op['i']) * u_dim} + (index_t)u]"

    def go_expr(op: Dict[str, Any]) -> str:
        if op["use_go_slot"]:
            return f"go_slot_{int(op['go_slot'])}"
        return f"grad_out[go_base + (index_t){int(op['v']) * u_dim} + (index_t)u]"

    def _op_wg_sort_key(op: Dict[str, Any]):
        use_wg_pair = bool(op["use_wi_slot"]) and bool(op["use_go_slot"])
        return (
            0 if use_wg_pair else 1,
            int(op["wi_slot"]) if bool(op["use_wi_slot"]) else -1,
            int(op["go_slot"]) if bool(op["use_go_slot"]) else -1,
            int(op["j"]),
            int(op["k"]),
            int(op["v"]),
            int(op["i"]),
        )

    def _build_wg_plan_for_sorted_ops(sp_ops: List[Dict[str, Any]]) -> Dict[Tuple[int, int], int]:
        if not enable_wg_subphase_cache:
            return {}
        pair_count: Dict[Tuple[int, int], int] = {}
        for op in sp_ops:
            if bool(op["use_wi_slot"]) and bool(op["use_go_slot"]):
                key = (int(op["wi_slot"]), int(op["go_slot"]))
                pair_count[key] = pair_count.get(key, 0) + 1

        cand = [(key, cnt) for key, cnt in pair_count.items() if cnt >= wg_min_reuse]
        if not cand:
            return {}

        cand.sort(key=lambda x: (-x[1], x[0][0], x[0][1]))
        cand = cand[:wg_max_slots_per_subphase]
        return {key: idx for idx, (key, _) in enumerate(cand)}

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
        ap(f"    const index_t w_base  = (index_t)w_row * (index_t){w_row_stride_const};")
        ap(f"    const index_t gw_base = (index_t)w_row * (index_t){w_row_stride_const};")
    else:
        ap("    const index_t w_base  = (index_t)w_row * (index_t)Iw * (index_t)U;")
        ap("    const index_t gw_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if x_row_stride_const is not None:
        if use_x_src:
            ap(f"    const index_t x_base  = (index_t)src * (index_t){x_row_stride_const};")
            ap(f"    const index_t gx_base = (index_t)src * (index_t){x_row_stride_const};")
        else:
            ap(f"    const index_t x_base  = (index_t)e_local * (index_t){x_row_stride_const};")
            ap(f"    const index_t gx_base = (index_t)e_local * (index_t){x_row_stride_const};")
    else:
        if use_x_src:
            ap("    const index_t x_base  = (index_t)src * (index_t)Ix * (index_t)U;")
            ap("    const index_t gx_base = (index_t)src * (index_t)Ix * (index_t)U;")
        else:
            ap("    const index_t x_base  = (index_t)e_local * (index_t)Ix * (index_t)U;")
            ap("    const index_t gx_base = (index_t)e_local * (index_t)Ix * (index_t)U;")

    if mode == "u,u,,u":
        ap("    const index_t y_base  = (index_t)e_orig * (index_t)Ky;")
        ap("    const index_t gy_base = (index_t)e_orig * (index_t)Ky;")
    else:
        if y_row_stride_const is not None:
            if use_y_src:
                ap(f"    const index_t y_base  = (index_t)src * (index_t){y_row_stride_const};")
                ap(f"    const index_t gy_base = (index_t)src * (index_t){y_row_stride_const};")
            else:
                ap(f"    const index_t y_base  = (index_t)e_orig * (index_t){y_row_stride_const};")
                ap(f"    const index_t gy_base = (index_t)e_orig * (index_t){y_row_stride_const};")
        else:
            if use_y_src:
                ap("    const index_t y_base  = (index_t)src * (index_t)Ky * (index_t)U;")
                ap("    const index_t gy_base = (index_t)src * (index_t)Ky * (index_t)U;")
            else:
                ap("    const index_t y_base  = (index_t)e_orig * (index_t)Ky * (index_t)U;")
                ap("    const index_t gy_base = (index_t)e_orig * (index_t)Ky * (index_t)U;")

    if go_row_stride_const is not None:
        if use_scatter:
            ap(f"    const index_t go_base = (index_t)dst * (index_t){go_row_stride_const};")
        else:
            ap(f"    const index_t go_base = (index_t)e_orig * (index_t){go_row_stride_const};")
    else:
        if use_scatter:
            ap("    const index_t go_base = (index_t)dst * (index_t)V * (index_t)U;")
        else:
            ap("    const index_t go_base = (index_t)e_orig * (index_t)V * (index_t)U;")

    ap("")
    ap("    // full-resident accumulators across all phases/subphases")
    for s in range(len(uniq_i_all)):
        ap(f"    scalar_t gw_acc_i_{s};")
    for s in range(len(uniq_j_all)):
        ap(f"    scalar_t gx_acc_j_{s};")
    for s in range(len(uniq_k_all)):
        ap(f"    scalar_t gy_acc_k_{s};")
    ap("")
    ap("    // subphase-local slot names (static upper bounds across all subphases)")
    for s in range(max_wi_slot + 1):
        ap(f"    scalar_t wi_slot_{s};")
    for s in range(max_go_slot + 1):
        ap(f"    scalar_t go_slot_{s};")
    for s in range(max_x_slot + 1):
        ap(f"    scalar_t x_slot_{s};")
    for s in range(max_y_slot + 1):
        ap(f"    scalar_t y_slot_{s};")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        int u = u_base + lane;")
    ap("        if (u < U) {")
    ap("")
    ap("            // reset full-resident accumulators")
    for s in range(len(uniq_i_all)):
        ap(f"            gw_acc_i_{s} = scalar_t(0);")
    for s in range(len(uniq_j_all)):
        ap(f"            gx_acc_j_{s} = scalar_t(0);")
    for s in range(len(uniq_k_all)):
        ap(f"            gy_acc_k_{s} = scalar_t(0);")
    ap("")

    for ph in phases:
        ap(f"            // ===== phase {int(ph['phase_id'])}: pivot={ph['pivot_kind']}[{int(ph['pivot_index'])}] =====")
        for sp in ph["subphases"]:
            sp_id = int(sp["subphase_id"])
            sp_ops = sorted(sp["ops"], key=_op_wg_sort_key)
            wg_plan = _build_wg_plan_for_sorted_ops(sp_ops)

            ap(f"            // ---- subphase {sp_id} ----")
            ap("            {")

            if sp["wi_vals"]:
                ap(f"                // preload wi slots for subphase {sp_id}")
                for i_val in sp["wi_vals"]:
                    slot = int(sp["wi_slot_map"][i_val])
                    ap(f"                wi_slot_{slot} = w[w_base + (index_t){int(i_val) * u_dim} + (index_t)u];")

            if sp["go_vals"]:
                ap(f"                // preload go slots for subphase {sp_id}")
                for v_val in sp["go_vals"]:
                    slot = int(sp["go_slot_map"][v_val])
                    ap(f"                go_slot_{slot} = grad_out[go_base + (index_t){int(v_val) * u_dim} + (index_t)u];")

            if sp["x_vals"]:
                ap(f"                // preload x slots for subphase {sp_id}")
                for j_val in sp["x_vals"]:
                    slot = int(sp["x_slot_map"][j_val])
                    ap(f"                x_slot_{slot} = x[x_base + (index_t){int(j_val) * u_dim} + (index_t)u];")

            if sp["y_vals"]:
                ap(f"                // preload y slots for subphase {sp_id}")
                for k_val in sp["y_vals"]:
                    slot = int(sp["y_slot_map"][k_val])
                    if mode == "u,u,,u":
                        ap(f"                y_slot_{slot} = y[y_base + (index_t){int(k_val)}];")
                    else:
                        ap(f"                y_slot_{slot} = y[y_base + (index_t){int(k_val) * u_dim} + (index_t)u];")

            if wg_plan:
                ap(f"                // wg cache for subphase {sp_id}: wg = wi * go")
                for (wi_slot, go_slot), wg_slot in sorted(wg_plan.items(), key=lambda kv: kv[1]):
                    ap(f"                {scalar_t} wg_slot_{wg_slot} = wi_slot_{wi_slot} * go_slot_{go_slot};")

            ap(f"                // accumulate ops for subphase {sp_id} (sorted by wg)")
            for op in sp_ops:
                xe = x_expr(op)
                ye = y_expr(op)
                go = go_expr(op)
                wi = wi_expr(op)
                c = fmt_float(op["c"])

                wg_expr = None
                if bool(op["use_wi_slot"]) and bool(op["use_go_slot"]):
                    key = (int(op["wi_slot"]), int(op["go_slot"]))
                    if key in wg_plan:
                        wg_expr = f"wg_slot_{wg_plan[key]}"

                if wg_expr is not None:
                    ap(f"                gx_acc_j_{int(op['gx_acc_slot'])} += scalar_t({c}) * {wg_expr} * {ye};")
                    ap(f"                gy_acc_k_{int(op['gy_acc_slot'])} += scalar_t({c}) * {wg_expr} * {xe};")
                else:
                    ap(f"                gx_acc_j_{int(op['gx_acc_slot'])} += scalar_t({c}) * {wi} * {go} * {ye};")
                    ap(f"                gy_acc_k_{int(op['gy_acc_slot'])} += scalar_t({c}) * {wi} * {go} * {xe};")

                ap(f"                gw_acc_i_{int(op['gw_acc_slot'])} += scalar_t({c}) * {go} * {xe} * {ye};")

            ap("            }")
            ap("")

    ap("            // final writeback after all phases/subphases")
    for s, ii in enumerate(uniq_i_all):
        if write_grad_w_atomic:
            ap(f"            atomicAdd(&grad_w[gw_base + (index_t){gw_u_offsets[ii]} + (index_t)u], gw_acc_i_{s});")
        else:
            ap(f"            grad_w[gw_base + (index_t){gw_u_offsets[ii]} + (index_t)u] = gw_acc_i_{s};")
    ap("")
    ap("            // grad_x writeback")
    for s, jj in enumerate(uniq_j_all):
        if write_grad_x_atomic:
            ap(f"            atomicAdd(&grad_x[gx_base + (index_t){gx_u_offsets[jj]} + (index_t)u], gx_acc_j_{s});")
        else:
            ap(f"            grad_x[gx_base + (index_t){gx_u_offsets[jj]} + (index_t)u] = gx_acc_j_{s};")
    ap("")
    ap("            // grad_y writeback")
    if mode == "u,u,,u":
        for s, kk in enumerate(uniq_k_all):
            ap(f"            scalar_t gy_sum_{s} = warp_sum_xor(gy_acc_k_{s});")
            ap("            if (lane == 0) {")
            if write_grad_y_atomic:
                ap(f"                atomicAdd(&grad_y[gy_base + (index_t){kk}], gy_sum_{s});")
            else:
                ap(f"                grad_y[gy_base + (index_t){kk}] = gy_sum_{s};")
            ap("            }")
    else:
        for s, kk in enumerate(uniq_k_all):
            if write_grad_y_atomic:
                ap(f"            atomicAdd(&grad_y[gy_base + (index_t){gy_u_offsets[kk]} + (index_t)u], gy_acc_k_{s});")
            else:
                ap(f"            grad_y[gy_base + (index_t){gy_u_offsets[kk]} + (index_t)u] = gy_acc_k_{s};")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    if emit_dual_index_launcher:
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

        ap("static inline bool should_use_int32_index(")
        ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
        ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
        ap("{")
        ap("    bool w_ok = mul3_fits_int32((int64_t)WB, (int64_t)Iw, (int64_t)U);")
        ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
        ap("    bool x_ok = mul3_fits_int32(x_dim0, (int64_t)Ix, (int64_t)U);")
        ap("    bool y_ok = false;")
        ap("    if (mode_scalar_y) {")
        ap("        y_ok = mul_fits_int32((int64_t)B, (int64_t)Ky);")
        ap("    } else {")
        ap("        int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
        ap("        y_ok = mul3_fits_int32(y_dim0, (int64_t)Ky, (int64_t)U);")
        ap("    }")
        ap("    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
        ap("    bool go_ok = mul3_fits_int32(go_dim0, (int64_t)V, (int64_t)U);")
        ap("    return w_ok && x_ok && y_ok && go_ok;")
        ap("}")
        ap("")

        ap("template <typename scalar_t, typename index_t>")
        ap(f"void launch_{kernel_name}_typed(")
        ap("    const scalar_t* w,")
        ap("    const scalar_t* x,")
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
        ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
        ap("        w, x, y, grad_out,")
        ap("        grad_w, grad_x, grad_y,")
        ap("        src_idx, dst_idx, b_list,")
        ap("        B, WB, Iw, Ix, Ky, V, U, S);")
        ap("}")
        ap("")

        ap("template <typename scalar_t>")
        ap(f"void launch_{kernel_name}_auto(")
        ap("    const scalar_t* w,")
        ap("    const scalar_t* x,")
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
        ap(f"    constexpr bool kUseXSrc = {use_x_src_cpp};")
        ap(f"    constexpr bool kUseYSrc = {use_y_src_cpp};")
        ap(f"    constexpr bool kUseScatter = {use_scatter_cpp};")
        ap(f"    constexpr bool kModeScalarY = {mode_scalar_y};")
        ap("    if (should_use_int32_index(B, WB, Iw, Ix, Ky, V, U, S,")
        ap("                               kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
        ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
        ap("            src_idx, dst_idx, b_list,")
        ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
        ap("    } else {")
        ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
        ap("            src_idx, dst_idx, b_list,")
        ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
        ap("    }")
        ap("}")
        ap("")

        ap("template <typename scalar_t>")
        ap(f"void launch_{kernel_name}(")
        ap("    const scalar_t* w,")
        ap("    const scalar_t* x,")
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
        ap(f"    launch_{kernel_name}_auto<scalar_t>(")
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
        ap("        src_idx, dst_idx, b_list,")
        ap("        B, WB, Iw, Ix, Ky, V, U, S, stream);")
        ap("}")
        ap("")

    return '\n'.join(lines)

# ============================================================
# Without GradW API
# ============================================================

def _build_subphases_for_phase_no_gradw(
    candidate_paths,
    *,
    pivot_kind,
    pivot_index,
    reg_budget_inputs: int,
    gx_acc_slot_map,
    gy_acc_slot_map,
):
    """
    最小改动版：
    复用你现有的 _build_subphases_for_phase(...)，
    只是在返回结果时剥掉 gw_acc_slot。

    这样不用重写 operand-pool / x,y,wi,go slot 选择逻辑。
    """

    # 这里构造一个 dummy gw_acc_slot_map，仅用于兼容原 helper 的签名
    uniq_i_local = sorted({int(it[1].i) for it in candidate_paths})
    dummy_gw_acc_slot_map = _stable_slot_map(uniq_i_local)

    raw_subphases = _build_subphases_for_phase(
        candidate_paths,
        pivot_kind=pivot_kind,
        pivot_index=pivot_index,
        reg_budget_inputs=reg_budget_inputs,
        gw_acc_slot_map=dummy_gw_acc_slot_map,
        gx_acc_slot_map=gx_acc_slot_map,
        gy_acc_slot_map=gy_acc_slot_map,
    )

    stripped_subphases = []
    for sp in raw_subphases:
        new_ops = []
        for op in sp.ops:
            new_ops.append(
                SimpleNamespace(
                    i=int(op.i),
                    j=int(op.j),
                    k=int(op.k),
                    v=int(op.v),
                    c=float(op.c),
                    gx_acc_slot=int(op.gx_acc_slot),
                    gy_acc_slot=int(op.gy_acc_slot),
                    use_x_slot=bool(op.use_x_slot),
                    x_slot=int(op.x_slot),
                    use_y_slot=bool(op.use_y_slot),
                    y_slot=int(op.y_slot),
                    use_wi_slot=bool(op.use_wi_slot),
                    wi_slot=int(op.wi_slot),
                    use_go_slot=bool(op.use_go_slot),
                    go_slot=int(op.go_slot),
                )
            )

        stripped_subphases.append(
            SimpleNamespace(
                subphase_id=int(sp.subphase_id),
                x_vals=list(sp.x_vals),
                y_vals=list(sp.y_vals),
                wi_vals=list(sp.wi_vals),
                go_vals=list(sp.go_vals),
                x_slot_map=dict(sp.x_slot_map),
                y_slot_map=dict(sp.y_slot_map),
                wi_slot_map=dict(sp.wi_slot_map),
                go_slot_map=dict(sp.go_slot_map),
                path_ids=list(sp.path_ids),
                reg_usage_inputs=int(sp.reg_usage_inputs),
                ops=new_ops,
            )
        )

    return stripped_subphases

def schedule_gradx_grady_full_acc_subphase(
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
    reg_budget: int,
    *,
    dynamic_iv_order: bool = True,
    dynamic_jk_order: bool = True,
    sanity_check: bool = True,
) -> Dict[str, Any]:
    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

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

    # 仍然保留 uniq_i_all，方便 phase pivot / 排序 / 调试
    uniq_i_all = sorted({p.i for p in paths})
    uniq_j_all = sorted({p.j for p in paths})
    uniq_k_all = sorted({p.k for p in paths})
    uniq_v_all = sorted({p.v for p in paths})

    gx_acc_slot_map = _stable_slot_map(uniq_j_all)
    gy_acc_slot_map = _stable_slot_map(uniq_k_all)

    n_acc_gx = len(uniq_j_all)
    n_acc_gy = len(uniq_k_all)
    n_acc_total = n_acc_gx + n_acc_gy

    r_rem = reg_budget - n_acc_total
    if r_rem <= 0:
        raise ValueError(
            f"reg_budget={reg_budget} is too small: need n_acc_total={n_acc_total}, "
            f"where n_acc_gx={n_acc_gx}, n_acc_gy={n_acc_gy}"
        )

    if dynamic_iv_order:
        iv_order = _choose_pair_order(len(uniq_i_all), len(uniq_v_all), "i", "v")
    else:
        iv_order = ("i", "v")

    if dynamic_jk_order:
        jk_order = _choose_pair_order(len(uniq_j_all), len(uniq_k_all), "j", "k")
    else:
        jk_order = ("j", "k")

    enumerated_paths: List[Tuple[int, Any]] = list(enumerate(paths))
    enumerated_paths.sort(key=lambda it: _path_sort_key_global(it[1], iv_order, jk_order))

    phases: List[PhasePlan] = []
    remaining_paths = list(enumerated_paths)
    phase_id = 0

    while remaining_paths:
        pivot_kind, pivot_index, _ = _choose_pivot_from_remaining(remaining_paths)
        candidate_paths = [it for it in remaining_paths if _path_matches_pivot(it[1], pivot_kind, pivot_index)]
        if not candidate_paths:
            raise RuntimeError("Pivot chosen but no candidate paths found")

        subphases = _build_subphases_for_phase_no_gradw(
            candidate_paths,
            pivot_kind=pivot_kind,
            pivot_index=pivot_index,
            reg_budget_inputs=r_rem,
            gx_acc_slot_map=gx_acc_slot_map,
            gy_acc_slot_map=gy_acc_slot_map,
        )

        path_ids: List[int] = []
        for sp in subphases:
            path_ids.extend(sp.path_ids)

        phases.append(
            PhasePlan(
                phase_id=int(phase_id),
                pivot_kind=str(pivot_kind),
                pivot_index=int(pivot_index),
                path_ids=list(path_ids),
                subphases=subphases,
            )
        )

        done_set = set(path_ids)
        remaining_paths = [it for it in remaining_paths if int(it[0]) not in done_set]
        phase_id += 1

    if sanity_check:
        seen = []
        for ph in phases:
            seen.extend(ph.path_ids)
            for sp in ph.subphases:
                if sp.reg_usage_inputs > r_rem:
                    raise AssertionError(
                        f"subphase exceeds reg budget: {sp.reg_usage_inputs} > {r_rem}"
                    )
        if sorted(seen) != list(range(len(paths))):
            raise AssertionError("phase/subphase scheduler did not cover each path exactly once")

    schedule: Dict[str, Any] = {
        "kernel_mode": "gradx_grady_fullacc_phase_subphase_operand_pool",
        "reg_budget": int(reg_budget),
        "n_acc_gx": int(n_acc_gx),
        "n_acc_gy": int(n_acc_gy),
        "n_acc_total": int(n_acc_total),
        "r_rem": int(r_rem),
        "uniq_i_all": list(uniq_i_all),
        "uniq_j_all": list(uniq_j_all),
        "uniq_k_all": list(uniq_k_all),
        "uniq_v_all": list(uniq_v_all),
        "gx_acc_slot_map": dict(gx_acc_slot_map),
        "gy_acc_slot_map": dict(gy_acc_slot_map),
        "iv_order": tuple(iv_order),
        "jk_order": tuple(jk_order),
        "phases": [],
    }

    for ph in phases:
        ph_dict = {
            "phase_id": int(ph.phase_id),
            "pivot_kind": str(ph.pivot_kind),
            "pivot_index": int(ph.pivot_index),
            "path_ids": list(ph.path_ids),
            "subphases": [],
        }
        for sp in ph.subphases:
            ph_dict["subphases"].append(
                {
                    "subphase_id": int(sp.subphase_id),
                    "x_vals": list(sp.x_vals),
                    "y_vals": list(sp.y_vals),
                    "wi_vals": list(sp.wi_vals),
                    "go_vals": list(sp.go_vals),
                    "x_slot_map": dict(sp.x_slot_map),
                    "y_slot_map": dict(sp.y_slot_map),
                    "wi_slot_map": dict(sp.wi_slot_map),
                    "go_slot_map": dict(sp.go_slot_map),
                    "path_ids": list(sp.path_ids),
                    "reg_usage_inputs": int(sp.reg_usage_inputs),
                    "ops": [
                        {
                            "i": op.i,
                            "j": op.j,
                            "k": op.k,
                            "v": op.v,
                            "c": float(op.c),
                            "gx_acc_slot": int(op.gx_acc_slot),
                            "gy_acc_slot": int(op.gy_acc_slot),
                            "use_x_slot": bool(op.use_x_slot),
                            "x_slot": int(op.x_slot),
                            "use_y_slot": bool(op.use_y_slot),
                            "y_slot": int(op.y_slot),
                            "use_wi_slot": bool(op.use_wi_slot),
                            "wi_slot": int(op.wi_slot),
                            "use_go_slot": bool(op.use_go_slot),
                            "go_slot": int(op.go_slot),
                        }
                        for op in sp.ops
                    ],
                }
            )
        schedule["phases"].append(ph_dict)

    return schedule

def emit_fused_bwd_kernel_no_gradw_decls(
    *,
    kernel_name: str,
) -> str:
    lines: List[str] = []
    ap = lines.append

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    ap("    scalar_t* __restrict__ grad_x,")
    ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S);")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    cudaStream_t stream);")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    cudaStream_t stream);")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    cudaStream_t stream);")
    ap("")

    return "\n".join(lines)

def emit_bundle_launcher_no_gradw(
    *,
    bundle_name: str,
    kernel_name: str,
    use_scatter: bool,
) -> str:
    lines: List[str] = []
    ap = lines.append

    ap(f"std::vector<torch::Tensor> launcher_{bundle_name}(")
    ap("    torch::Tensor w,")
    ap("    torch::Tensor x,")
    ap("    torch::Tensor y,")
    ap("    torch::Tensor grad_out,")
    ap("    torch::Tensor src_idx,")
    ap("    c10::optional<torch::Tensor> dst_idx_opt,")
    ap("    c10::optional<torch::Tensor> b_list_opt,")
    ap("    int64_t V)")
    ap("{")
    ap('    TORCH_CHECK(w.is_cuda(), "w must be CUDA");')
    ap('    TORCH_CHECK(x.is_cuda(), "x must be CUDA");')
    ap('    TORCH_CHECK(y.is_cuda(), "y must be CUDA");')
    ap('    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");')
    ap('    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA");')
    ap("")

    if use_scatter:
        ap('    TORCH_CHECK(dst_idx_opt.has_value(), "dst_idx is required when use_scatter=True");')
        ap('    TORCH_CHECK(dst_idx_opt.value().is_cuda(), "dst_idx must be CUDA");')
    ap("")

    ap("    auto b_list = b_list_opt.has_value() ? b_list_opt.value() : torch::Tensor();")
    ap("    auto dst_idx = dst_idx_opt.has_value() ? dst_idx_opt.value() : torch::Tensor();")
    ap("")

    ap("    const int B  = b_list.defined() ? (int)b_list.numel() : (int)w.size(0);")
    ap("    const int WB = (int)w.size(0);")
    ap("    const int Iw = (int)w.size(1);")
    ap("    const int U  = (int)w.size(2);")
    ap("    const int Ix = (int)x.size(1);")
    ap("    const int S  = (int)x.size(0);")
    ap("")

    ap("    int Ky = 0;")
    ap("    if (y.dim() == 2) Ky = (int)y.size(1);")
    ap("    else if (y.dim() == 3) Ky = (int)y.size(1);")
    ap('    else TORCH_CHECK(false, "y must be rank-2 or rank-3");')
    ap("")

    ap("    auto grad_x = torch::zeros_like(x);")
    ap("    auto grad_y = torch::zeros_like(y);")
    ap("")

    ap("    const int32_t* dst_idx_ptr = nullptr;")
    if use_scatter:
        ap("    dst_idx_ptr = (const int32_t*)dst_idx.data_ptr<int32_t>();")
    ap("    const int32_t* b_list_ptr = b_list.defined() ? (const int32_t*)b_list.data_ptr<int32_t>() : nullptr;")
    ap("")
    ap("    auto stream = at::cuda::getDefaultCUDAStream();")
    ap("")
    ap("    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), \"launcher_no_gradw\", [&] {")
    ap(f"        launch_{kernel_name}<scalar_t>(")
    ap("            (const scalar_t*)w.data_ptr<scalar_t>(),")
    ap("            (const scalar_t*)x.data_ptr<scalar_t>(),")
    ap("            (const scalar_t*)y.data_ptr<scalar_t>(),")
    ap("            (const scalar_t*)grad_out.data_ptr<scalar_t>(),")
    ap("            (scalar_t*)grad_x.data_ptr<scalar_t>(),")
    ap("            (scalar_t*)grad_y.data_ptr<scalar_t>(),")
    ap("            (const int32_t*)src_idx.data_ptr<int32_t>(),")
    ap("            dst_idx_ptr,")
    ap("            b_list_ptr,")
    ap("            B, WB, Iw, Ix, Ky, (int)V, U, S, stream.stream());")
    ap("    });")
    ap("")
    ap("    return {grad_x, grad_y};")
    ap("}")
    ap("")

    ap("TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {")
    ap(f'    m.def("{bundle_name}(Tensor w, Tensor x, Tensor y, Tensor grad_out, Tensor src_idx, Tensor? dst_idx, Tensor? b_list, int V) -> Tensor[]");')
    ap("}")
    ap("")
    ap("TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CUDA, m) {")
    ap(f'    m.impl("{bundle_name}", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(launcher_{bundle_name})));')
    ap("}")

    return "\n".join(lines)

def emit_fused_bwd_kernel_from_subphase_schedule_no_gradw(
    schedule: Dict[str, Any],
    *,
    kernel_name: str,
    scalar_t: str,
    mode: str,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
    u_dim: int,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
    write_grad_x_atomic: bool = True,
    write_grad_y_atomic: bool = True,
    enable_wg_subphase_cache: bool = True,
    wg_min_reuse: int = 2,
    wg_max_slots_per_subphase: int = 2,
    emit_dual_index_launcher: bool = True,
) -> str:
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")
    if not isinstance(u_dim, int) or u_dim <= 0:
        raise ValueError(f"u_dim must be positive int, got {u_dim}")
    if block_size != 32:
        raise ValueError(f"Only block_size=32 is supported, got {block_size}")
    if not emit_dual_index_launcher:
        raise ValueError("no_gradw fused bwd requires emit_dual_index_launcher=True")

    uniq_j_all = [int(x) for x in schedule["uniq_j_all"]]
    uniq_k_all = [int(x) for x in schedule["uniq_k_all"]]
    phases = schedule["phases"]

    gx_u_offsets = {jj: jj * u_dim for jj in uniq_j_all}
    gy_u_offsets = {kk: kk * u_dim for kk in uniq_k_all}

    x_row_stride_const = None if ix_dim is None else int(ix_dim) * u_dim
    y_row_stride_const = None if (mode == "u,u,,u" or ky_dim is None) else int(ky_dim) * u_dim
    go_row_stride_const = None if v_dim is None else int(v_dim) * u_dim

    max_wi_slot = -1
    max_go_slot = -1
    max_x_slot = -1
    max_y_slot = -1

    for ph in phases:
        for sp in ph["subphases"]:
            for _, s in sp["wi_slot_map"].items():
                max_wi_slot = max(max_wi_slot, int(s))
            for _, s in sp["go_slot_map"].items():
                max_go_slot = max(max_go_slot, int(s))
            for _, s in sp["x_slot_map"].items():
                max_x_slot = max(max_x_slot, int(s))
            for _, s in sp["y_slot_map"].items():
                max_y_slot = max(max_y_slot, int(s))

    lines: List[str] = []
    ap = lines.append

    def fmt_float(x: float) -> str:
        return repr(float(x))

    def x_expr(op: Dict[str, Any]) -> str:
        if op["use_x_slot"]:
            return f"x_slot_{int(op['x_slot'])}"
        return f"x[x_base + (index_t){int(op['j']) * u_dim} + (index_t)u]"

    def y_expr(op: Dict[str, Any]) -> str:
        if op["use_y_slot"]:
            return f"y_slot_{int(op['y_slot'])}"
        if mode == "u,u,,u":
            return f"y[y_base + (index_t){int(op['k'])}]"
        return f"y[y_base + (index_t){int(op['k']) * u_dim} + (index_t)u]"

    def wi_expr(op: Dict[str, Any]) -> str:
        if op["use_wi_slot"]:
            return f"wi_slot_{int(op['wi_slot'])}"
        return f"w[w_base + (index_t){int(op['i']) * u_dim} + (index_t)u]"

    def go_expr(op: Dict[str, Any]) -> str:
        if op["use_go_slot"]:
            return f"go_slot_{int(op['go_slot'])}"
        return f"grad_out[go_base + (index_t){int(op['v']) * u_dim} + (index_t)u]"

    def _op_wg_sort_key(op: Dict[str, Any]):
        use_wg_pair = bool(op["use_wi_slot"]) and bool(op["use_go_slot"])
        return (
            0 if use_wg_pair else 1,
            int(op["wi_slot"]) if bool(op["use_wi_slot"]) else -1,
            int(op["go_slot"]) if bool(op["use_go_slot"]) else -1,
            int(op["j"]),
            int(op["k"]),
            int(op["v"]),
            int(op["i"]),
        )

    def _build_wg_plan_for_sorted_ops(sp_ops: List[Dict[str, Any]]) -> Dict[Tuple[int, int], int]:
        if not enable_wg_subphase_cache:
            return {}
        pair_count: Dict[Tuple[int, int], int] = {}
        for op in sp_ops:
            if bool(op["use_wi_slot"]) and bool(op["use_go_slot"]):
                key = (int(op["wi_slot"]), int(op["go_slot"]))
                pair_count[key] = pair_count.get(key, 0) + 1

        cand = [(key, cnt) for key, cnt in pair_count.items() if cnt >= wg_min_reuse]
        if not cand:
            return {}

        cand.sort(key=lambda x: (-x[1], x[0][0], x[0][1]))
        cand = cand[:wg_max_slots_per_subphase]
        return {key: idx for idx, (key, _) in enumerate(cand)}

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
    ap("    const scalar_t* __restrict__ grad_out,")
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

    ap("    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if x_row_stride_const is not None:
        if use_x_src:
            ap(f"    const index_t x_base  = (index_t)src * (index_t){x_row_stride_const};")
            ap(f"    const index_t gx_base = (index_t)src * (index_t){x_row_stride_const};")
        else:
            ap(f"    const index_t x_base  = (index_t)e_local * (index_t){x_row_stride_const};")
            ap(f"    const index_t gx_base = (index_t)e_local * (index_t){x_row_stride_const};")
    else:
        if use_x_src:
            ap("    const index_t x_base  = (index_t)src * (index_t)Ix * (index_t)U;")
            ap("    const index_t gx_base = (index_t)src * (index_t)Ix * (index_t)U;")
        else:
            ap("    const index_t x_base  = (index_t)e_local * (index_t)Ix * (index_t)U;")
            ap("    const index_t gx_base = (index_t)e_local * (index_t)Ix * (index_t)U;")

    if mode == "u,u,,u":
        ap("    const index_t y_base  = (index_t)e_orig * (index_t)Ky;")
        ap("    const index_t gy_base = (index_t)e_orig * (index_t)Ky;")
    else:
        if y_row_stride_const is not None:
            if use_y_src:
                ap(f"    const index_t y_base  = (index_t)src * (index_t){y_row_stride_const};")
                ap(f"    const index_t gy_base = (index_t)src * (index_t){y_row_stride_const};")
            else:
                ap(f"    const index_t y_base  = (index_t)e_orig * (index_t){y_row_stride_const};")
                ap(f"    const index_t gy_base = (index_t)e_orig * (index_t){y_row_stride_const};")
        else:
            if use_y_src:
                ap("    const index_t y_base  = (index_t)src * (index_t)Ky * (index_t)U;")
                ap("    const index_t gy_base = (index_t)src * (index_t)Ky * (index_t)U;")
            else:
                ap("    const index_t y_base  = (index_t)e_orig * (index_t)Ky * (index_t)U;")
                ap("    const index_t gy_base = (index_t)e_orig * (index_t)Ky * (index_t)U;")

    if go_row_stride_const is not None:
        if use_scatter:
            ap(f"    const index_t go_base = (index_t)dst * (index_t){go_row_stride_const};")
        else:
            ap(f"    const index_t go_base = (index_t)e_orig * (index_t){go_row_stride_const};")
    else:
        if use_scatter:
            ap("    const index_t go_base = (index_t)dst * (index_t)V * (index_t)U;")
        else:
            ap("    const index_t go_base = (index_t)e_orig * (index_t)V * (index_t)U;")

    ap("")
    ap("    // full-resident accumulators across all phases/subphases")
    for s in range(len(uniq_j_all)):
        ap(f"    scalar_t gx_acc_j_{s};")
    for s in range(len(uniq_k_all)):
        ap(f"    scalar_t gy_acc_k_{s};")
    ap("")

    ap("    // subphase-local slot names")
    for s in range(max_wi_slot + 1):
        ap(f"    scalar_t wi_slot_{s};")
    for s in range(max_go_slot + 1):
        ap(f"    scalar_t go_slot_{s};")
    for s in range(max_x_slot + 1):
        ap(f"    scalar_t x_slot_{s};")
    for s in range(max_y_slot + 1):
        ap(f"    scalar_t y_slot_{s};")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    ap("")

    for s in range(len(uniq_j_all)):
        ap(f"            gx_acc_j_{s} = scalar_t(0);")
    for s in range(len(uniq_k_all)):
        ap(f"            gy_acc_k_{s} = scalar_t(0);")
    ap("")

    for ph in phases:
        ap(f"            // ===== phase {int(ph['phase_id'])}: pivot={ph['pivot_kind']}[{int(ph['pivot_index'])}] =====")
        for sp in ph["subphases"]:
            sp_id = int(sp["subphase_id"])
            sp_ops = sorted(sp["ops"], key=_op_wg_sort_key)
            wg_plan = _build_wg_plan_for_sorted_ops(sp_ops)

            ap(f"            // ---- subphase {sp_id} ----")
            ap("            {")
            if sp["wi_vals"]:
                for i_val in sp["wi_vals"]:
                    slot = int(sp["wi_slot_map"][i_val])
                    ap(f"                wi_slot_{slot} = w[w_base + (index_t){int(i_val) * u_dim} + (index_t)u];")

            if sp["go_vals"]:
                for v_val in sp["go_vals"]:
                    slot = int(sp["go_slot_map"][v_val])
                    ap(f"                go_slot_{slot} = grad_out[go_base + (index_t){int(v_val) * u_dim} + (index_t)u];")

            if sp["x_vals"]:
                for j_val in sp["x_vals"]:
                    slot = int(sp["x_slot_map"][j_val])
                    ap(f"                x_slot_{slot} = x[x_base + (index_t){int(j_val) * u_dim} + (index_t)u];")

            if sp["y_vals"]:
                for k_val in sp["y_vals"]:
                    slot = int(sp["y_slot_map"][k_val])
                    if mode == "u,u,,u":
                        ap(f"                y_slot_{slot} = y[y_base + (index_t){int(k_val)}];")
                    else:
                        ap(f"                y_slot_{slot} = y[y_base + (index_t){int(k_val) * u_dim} + (index_t)u];")

            if wg_plan:
                for (wi_slot, go_slot), wg_slot in sorted(wg_plan.items(), key=lambda kv: kv[1]):
                    ap(f"                {scalar_t} wg_slot_{wg_slot} = wi_slot_{wi_slot} * go_slot_{go_slot};")

            for op in sp_ops:
                xe = x_expr(op)
                ye = y_expr(op)
                go = go_expr(op)
                wi = wi_expr(op)
                c = fmt_float(op["c"])

                wg_expr = None
                if bool(op["use_wi_slot"]) and bool(op["use_go_slot"]):
                    key = (int(op["wi_slot"]), int(op["go_slot"]))
                    if key in wg_plan:
                        wg_expr = f"wg_slot_{wg_plan[key]}"

                if wg_expr is not None:
                    ap(f"                gx_acc_j_{int(op['gx_acc_slot'])} += scalar_t({c}) * {wg_expr} * {ye};")
                    ap(f"                gy_acc_k_{int(op['gy_acc_slot'])} += scalar_t({c}) * {wg_expr} * {xe};")
                else:
                    ap(f"                gx_acc_j_{int(op['gx_acc_slot'])} += scalar_t({c}) * {wi} * {go} * {ye};")
                    ap(f"                gy_acc_k_{int(op['gy_acc_slot'])} += scalar_t({c}) * {wi} * {go} * {xe};")

            ap("            }")
            ap("")

    ap("            // grad_x writeback")
    for s, jj in enumerate(uniq_j_all):
        if write_grad_x_atomic:
            ap(f"            atomicAdd(&grad_x[gx_base + (index_t){gx_u_offsets[jj]} + (index_t)u], gx_acc_j_{s});")
        else:
            ap(f"            grad_x[gx_base + (index_t){gx_u_offsets[jj]} + (index_t)u] = gx_acc_j_{s};")
    ap("")

    ap("            // grad_y writeback")
    if mode == "u,u,,u":
        for s, kk in enumerate(uniq_k_all):
            ap(f"            scalar_t gy_sum_{s} = warp_sum_xor(gy_acc_k_{s});")
            ap("            if (lane == 0) {")
            if write_grad_y_atomic:
                ap(f"                atomicAdd(&grad_y[gy_base + (index_t){kk}], gy_sum_{s});")
            else:
                ap(f"                grad_y[gy_base + (index_t){kk}] = gy_sum_{s};")
            ap("            }")
    else:
        for s, kk in enumerate(uniq_k_all):
            if write_grad_y_atomic:
                ap(f"            atomicAdd(&grad_y[gy_base + (index_t){gy_u_offsets[kk]} + (index_t)u], gy_acc_k_{s});")
            else:
                ap(f"            grad_y[gy_base + (index_t){gy_u_offsets[kk]} + (index_t)u] = gy_acc_k_{s};")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

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

    ap("static inline bool should_use_int32_index(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    bool y_ok = false;")
    ap("    if (mode_scalar_y) {")
    ap("        y_ok = mul_fits_int32((int64_t)B, (int64_t)Ky);")
    ap("    } else {")
    ap("        int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("        y_ok = mul3_fits_int32(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    }")
    ap("    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool go_ok = mul3_fits_int32(go_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && go_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
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
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
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
    ap("    if (should_use_int32_index(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                               kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, b_list,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, b_list,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    cudaStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S, stream);")
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
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
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
    )

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
        v_tile_size=1,
        i_tile_size=1,
        j_tile_size=1,
        k_tile_size=1,
    )

    if need_grad_w:
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
    else:
        code = emit_fused_bwd_kernel_from_schedule_no_gradw(
            sched,
            kernel_name=bundle_name,
            scalar_t=scalar_t,
            mode=mode,
            use_x_src=use_x_src,
            use_y_src=use_y_src,
            use_scatter=use_scatter,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )

        code += "\n"
        code += emit_fused_bwd_launcher_no_gradw(
            bundle_name,
            mode=mode,
            use_x_src=use_x_src,
            use_y_src=use_y_src,
            use_scatter=use_scatter,
        ) """

    if need_grad_w:
        schedule = schedule_gradw_gradx_grady_full_acc_subphase(
            i_list, j_list, k_list, v_list, coeff_list,
            reg_budget=128,
        )

        print(schedule)
        code = emit_fused_bwd_kernel_from_subphase_schedule_with_gradw(
            schedule,
            kernel_name=bundle_name,
            scalar_t=scalar_t,
            mode=mode,
            use_x_src=use_x_src,
            use_y_src=use_y_src,
            use_scatter=use_scatter,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
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

    else:
        schedule = schedule_gradx_grady_full_acc_subphase(
            i_list=i_list,
            j_list=j_list,
            k_list=k_list,
            v_list=v_list,
            coeff_list=coeff_list,
            reg_budget=64,
            dynamic_iv_order=True,
            dynamic_jk_order=True,
            sanity_check=True,
        )


        code = emit_fused_bwd_kernel_from_subphase_schedule_no_gradw(
            schedule,
            kernel_name=bundle_name,
            scalar_t=scalar_t,
            mode=mode,
            use_x_src=use_x_src,
            use_y_src=use_y_src,
            use_scatter=use_scatter,
            u_dim=u_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
            block_size=32,
            write_grad_x_atomic=False,
            write_grad_y_atomic=True,
            enable_wg_subphase_cache=True,
            wg_min_reuse=2,
            wg_max_slots_per_subphase=2,
            emit_dual_index_launcher=True,
        )

        code += "\n"
        code += emit_fused_bwd_launcher_no_gradw(
            bundle_name=bundle_name,
            mode=mode,
            use_x_src=use_x_src,
            use_y_src=use_y_src,
            use_scatter=use_scatter,
        )

    return code

    