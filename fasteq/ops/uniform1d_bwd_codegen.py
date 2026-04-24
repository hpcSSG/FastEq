from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, OrderedDict, Counter
from typing import Any, Dict, List, Optional, Tuple, Set
from copy import deepcopy
from types import SimpleNamespace
import math
import torch


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
        print(
            f"reg_budget={reg_budget} is too small: need n_acc_total={n_acc_total}, "
            f"where n_acc_gw={n_acc_gw}, n_acc_gx={n_acc_gx}, n_acc_gy={n_acc_gy}"
        )
        r_rem = 128

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