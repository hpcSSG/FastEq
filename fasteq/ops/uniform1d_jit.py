import os
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch._dynamo
from torch.utils.cpp_extension import load

from .uniform1d_scatter_fwd_codegen import generate_code_uniform1d_fwd
from .uniform1d_fwd_codegen import generate_code_uniform1d_fwd_no_scatter
from .uniform1d_scatter_bwd_codegen import (
    build_backward_schedule_from_lists,
    emit_backward_cuda_from_schedule,
    generate_full_uniform1d_bwd_split_cuda,
)

# -----------------------------------------------------------------------------
# JIT cache
# -----------------------------------------------------------------------------

_FWD_JIT_CACHE: Dict[str, object] = {}
_BWD_JIT_CACHE: Dict[str, object] = {}


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _find_fasteq_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if p.name == "fasteq":
            return p
    raise RuntimeError("Cannot find fasteq project root from __file__")

def _default_build_root() -> Path:
    root = _find_fasteq_root(Path(__file__).parent) / "cuda" / "src" / "uniform1d_jit_codegen"
    _ensure_dir(root)
    return root

def _default_src_path() -> Path:
    root = _find_fasteq_root(Path(__file__).parent) / "cuda" / "src"
    _ensure_dir(root)
    return root


def _write_code_file(code: str, build_dir: Path, module_name: str) -> Path:
    cu_path = build_dir / f"{module_name}.cu"
    cu_path.write_text(code, encoding="utf-8")
    return cu_path


def _load_jit_module(
    *,
    module_name: str,
    code: str,
    cache: Dict[str, object],
    extra_cuda_cflags=None,
    extra_cflags=None,
):
    if module_name in cache:
        return cache[module_name]

    build_root = _default_build_root()
    build_dir = build_root / module_name
    _ensure_dir(build_dir)

    src_dir = _default_src_path()

    cu_path = build_dir / f"{module_name}.cu"
    if code is not None:
        cu_path.write_text(code, encoding="utf-8")
    elif not cu_path.exists():
        raise RuntimeError(
            f"JIT source file does not exist for module {module_name}: {cu_path}"
        )

    mod = load(
        name=module_name,
        sources=[str(cu_path)],
        extra_cflags=extra_cflags or ["-O3"],
        extra_cuda_cflags=extra_cuda_cflags or ["-O3", "--use_fast_math", "-lineinfo"],
        extra_include_paths=[str(src_dir)],
        build_directory=str(build_dir),
        verbose=True,
        with_cuda=True,
    )

    cache[module_name] = mod
    return mod


# -----------------------------------------------------------------------------
# codegen -> jit module
# -----------------------------------------------------------------------------

def _tensor_to_cpu_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    raise TypeError(f"Unsupported type for _tensor_to_cpu_list: {type(x)}")

def _make_fwd_module_name(
    *,
    P: int,
    u_dim: int,
    mode,
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
    dtype_str: str,
) -> str:
    sig = repr((
        P, u_dim, dtype_str, mode,
        tuple(i_list), tuple(j_list), tuple(k_list), tuple(v_list),
        #tuple(float(c) for c in coeff_list),
    ))
    h = _sha1_text(sig)
    mode_str = "u_u__u" if mode == "u,u,,u" else "u_u_u_u"
    return f"uniform1d_fwd_{mode_str}_u{u_dim}_path{P}_jit_{dtype_str}_{h}"

def _make_bwd_module_name(
    *,
    P: int,
    u_dim: int,
    mode: str,
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
    dtype_str: str,
    split_mode: bool,
) -> str:
    sig = repr((
        P, u_dim, dtype_str, split_mode, mode,
        tuple(i_list), tuple(j_list), tuple(k_list), tuple(v_list),
        tuple(float(c) for c in coeff_list),
    ))
    h = _sha1_text(sig)
    tag = "split" if split_mode else "combine"
    mode_str = "u_u__u" if mode == "u,u,,u" else "u_u_u_u"
    return f"uniform1d_bwd_{mode_str}_u{u_dim}_path{P}_{tag}_jit_{dtype_str}_{h}"


def _get_scalar_t_str(t: torch.Tensor) -> str:
    if t.dtype == torch.float32:
        return "float"
    if t.dtype == torch.float64:
        return "double"
    raise TypeError(f"Unsupported dtype for JIT codegen: {t.dtype}")


def _build_jit_module_common(
    *,
    module_name: str,
    cache: dict,
    kind: str,
    codegen_fn,
):
    # Step 1: Check the in-memory cache for the current process.
    if module_name in cache:
        print(f"[JIT][{kind}] hit in-memory cache: {module_name}")
        return cache[module_name]

    print(f"[JIT][{kind}] build/check module: {module_name}")

    build_root = _default_build_root()
    build_dir = build_root / module_name
    _ensure_dir(build_dir)

    cu_path = build_dir / f"{module_name}.cu"

    # Step 2: Check the disk cache.
    # If the CUDA source file already exists, assume the module name uniquely
    # identifies the generated code and load it directly.
    if cu_path.exists():
        print(f"[JIT][{kind}] hit disk cache source: {cu_path}")
        return _load_jit_module(
            module_name=module_name,
            code=None,
            cache=cache,
        )

    # Step 3: Generate CUDA source code if no cache is found on disk.
    code = codegen_fn()
    if not isinstance(code, str):
        raise RuntimeError(
            f"{codegen_fn.__name__}(...) must return a complete CUDA source string."
        )

    return _load_jit_module(
        module_name=module_name,
        code=code,
        cache=cache,
    )

def _build_fwd_jit_module(
    *,
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    u_dim: int,
    mode: str,
    dtype_str: str,
    fused_scatter: bool,
):
    # Convert tensors to CPU-side Python lists for hashing and code generation.
    i_cpu = _tensor_to_cpu_list(i_list)
    j_cpu = _tensor_to_cpu_list(j_list)
    k_cpu = _tensor_to_cpu_list(k_list)
    v_cpu = _tensor_to_cpu_list(v_list)
    coeff_cpu = _tensor_to_cpu_list(coeff_list)

    P = len(i_cpu)

    module_name = _make_fwd_module_name(
        P=P,
        u_dim=u_dim,
        mode=mode,
        i_list=i_cpu,
        j_list=j_cpu,
        k_list=k_cpu,
        v_list=v_cpu,
        coeff_list=coeff_cpu,
        dtype_str=dtype_str,
    )

    if fused_scatter:
        codegen_fn = lambda: generate_code_uniform1d_fwd(
            i_list=i_list,
            j_list=j_list,
            k_list=k_list,
            v_list=v_list,
            coeff_list=coeff_list,
            u_dim=u_dim,
            mode=mode,
        )
    else:
        codegen_fn = lambda: generate_code_uniform1d_fwd_no_scatter(
            i_list=i_list,
            j_list=j_list,
            k_list=k_list,
            v_list=v_list,
            coeff_list=coeff_list,
            u_dim=u_dim,
            mode=mode,
        )

    return _build_jit_module_common(
        module_name=module_name,
        cache=_FWD_JIT_CACHE,
        kind="FWD",
        codegen_fn=codegen_fn,
    )

def _build_bwd_jit_module(
    *,
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    u_dim: int,
    mode: str,
    dtype_str: str,
):
    # Convert tensors to CPU-side Python lists for hashing and code generation.
    i_cpu = _tensor_to_cpu_list(i_list)
    j_cpu = _tensor_to_cpu_list(j_list)
    k_cpu = _tensor_to_cpu_list(k_list)
    v_cpu = _tensor_to_cpu_list(v_list)
    coeff_cpu = _tensor_to_cpu_list(coeff_list)

    P = len(i_cpu)
    split_mode = P > 512

    module_name = _make_bwd_module_name(
        P=P,
        u_dim=u_dim,
        mode=mode,
        i_list=i_cpu,
        j_list=j_cpu,
        k_list=k_cpu,
        v_list=v_cpu,
        coeff_list=coeff_cpu,
        dtype_str=dtype_str,
        split_mode=split_mode,
    )

    def _codegen_bwd():
        if split_mode:
            return generate_full_uniform1d_bwd_split_cuda(
                i_list=i_cpu,
                j_list=j_cpu,
                k_list=k_cpu,
                v_list=v_cpu,
                coeff_list=coeff_cpu,
                bundle_name=f"uniform1d_split_u{u_dim}_path{P}_bwd",
            )

        sched = build_backward_schedule_from_lists(
            i_list=i_cpu,
            j_list=j_cpu,
            k_list=k_cpu,
            v_list=v_cpu,
            coeff_list=coeff_cpu,
            U_dim=u_dim,
        )
        return emit_backward_cuda_from_schedule(
            sched,
            kernel_name=f"uniform1d_combine_u{u_dim}_path{P}_bwd",
            scalar_t=dtype_str,
        )

    return _build_jit_module_common(
        module_name=module_name,
        cache=_BWD_JIT_CACHE,
        kind="BWD",
        codegen_fn=_codegen_bwd,
    )


# -----------------------------------------------------------------------------
# runtime dispatch
# -----------------------------------------------------------------------------
def _run_fwd(
    *,
    w,
    x,
    y,
    src_idx,
    dst_idx,
    b_list,
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
    out_seg_num,
    u_dim,
    mode,
    fused_scatter,
):

    dtype_str = _get_scalar_t_str(w)
    mod = _build_fwd_jit_module(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
        u_dim=u_dim,
        dtype_str=dtype_str,
        mode=mode,
        fused_scatter=fused_scatter,
    )

    if fused_scatter:

        return mod.run(
            w, x, y,
            src_idx, dst_idx, b_list, out_seg_num
        )
    else:
        return mod.run(
            w, x, y,
            src_idx, b_list, out_seg_num
        )

        
def _run_bwd(
    *,
    grad_out,
    w,
    x,
    y,
    src_idx,
    dst_idx,
    b_list,
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
    w_seg_num,
    x_seg_num,
    y_seg_num,
    out_seg_num,
    u_dim,
    mode,
):

    dtype_str = _get_scalar_t_str(w)
    mod = _build_bwd_jit_module(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
        u_dim=u_dim,
        dtype_str=dtype_str,
        mode=mode,
    )

    return mod.run(
        grad_out, w, x, y,
        src_idx, dst_idx, b_list,
        w_seg_num, x_seg_num, y_seg_num, out_seg_num
    )
    


# -----------------------------------------------------------------------------
# autograd function
# -----------------------------------------------------------------------------

class FastUniform1dJITFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, y, src_idx, dst_idx, b_list, meta, fused_scatter):
        i_list = meta["i_list"].to(torch.int32)
        j_list = meta["j_list"].to(torch.int32)
        k_list = meta["k_list"].to(torch.int32)
        v_list = meta["v_list"].to(torch.int32)
        coeff_list = meta["coeff_list"]

        out_seg_num = meta["out_seg_num"]
        w_seg_num = meta["w_seg_num"]
        x_seg_num = meta["x_seg_num"]
        y_seg_num = meta["y_seg_num"]
        u_dim = meta["u_dim"]

        edge_num = src_idx.shape[0]

        w = w.view(-1, w_seg_num, u_dim)
        x = x.view(-1, x_seg_num, u_dim) # [node_num, x_seg_num, u_dim]

        y = y.view(edge_num, y_seg_num, -1)

        if y.shape[2] == 1:
            mode = "u,u,,u"
        else:
            mode = "u,u,u,u"

        src_idx = src_idx.to(torch.int32)
        if fused_scatter:
            dst_idx = dst_idx.to(torch.int32)
            b_list = b_list.to(torch.int32)

        P = i_list.numel()
        print(f"[uniform1d][forward] P={P}, u_dim={u_dim}")

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000.0

        out = _run_fwd(
            w=w,
            x=x,
            y=y,
            src_idx=src_idx,
            dst_idx=dst_idx,
            b_list=b_list,
            i_list=i_list,
            j_list=j_list,
            k_list=k_list,
            v_list=v_list,
            coeff_list=coeff_list,
            out_seg_num=out_seg_num,
            u_dim=u_dim,
            mode=mode,
            fused_scatter=fused_scatter,
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000.0
        print(f"<< fasteq uniform1d fused forward cost: {end_time - start_time:.3f} ms >>")

        ctx.save_for_backward(w, x, y)
        ctx.src_idx = src_idx
        ctx.dst_idx = dst_idx
        ctx.b_list = b_list
        ctx.i_list = i_list
        ctx.j_list = j_list
        ctx.k_list = k_list
        ctx.v_list = v_list
        ctx.coeff_list = coeff_list
        ctx.out_seg_num = out_seg_num
        ctx.w_seg_num = w_seg_num
        ctx.x_seg_num = x_seg_num
        ctx.y_seg_num = y_seg_num
        ctx.u_dim = u_dim
        ctx.P = P
        ctx.mode = mode

        return out

    @staticmethod
    def backward(ctx, grad_out):
        w, x, y = ctx.saved_tensors

        grad_out = grad_out.view(-1, ctx.out_seg_num, ctx.u_dim)
        w = w.view(-1, ctx.w_seg_num, ctx.u_dim)
        x = x.view(-1, ctx.x_seg_num, ctx.u_dim)
        y = y.view(-1, ctx.y_seg_num, 1)

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000.0

        grad_w, grad_x, grad_y = _run_bwd(
            grad_out=grad_out,
            w=w,
            x=x,
            y=y,
            src_idx=ctx.src_idx,
            dst_idx=ctx.dst_idx,
            b_list=ctx.b_list,
            i_list=ctx.i_list,
            j_list=ctx.j_list,
            k_list=ctx.k_list,
            v_list=ctx.v_list,
            coeff_list=ctx.coeff_list,
            w_seg_num=ctx.w_seg_num,
            x_seg_num=ctx.x_seg_num,
            y_seg_num=ctx.y_seg_num,
            out_seg_num=ctx.out_seg_num,
            u_dim=ctx.u_dim,
            mode=ctx.mode,
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000.0
        print(f"<< fasteq uniform1d path:{ctx.P} backward cost: {end_time - start_time:.3f} ms >>")

        grad_w = grad_w.view(-1, ctx.w_seg_num * ctx.u_dim)
        grad_x = grad_x.view(-1, ctx.x_seg_num * ctx.u_dim)
        grad_y = grad_y.view(-1, ctx.y_seg_num)

        return grad_w, grad_x, grad_y, None, None, None, None, None

def fast_uniform1d_jit(w, x, y, src_idx, dst_idx, b_list, meta, fused_scatter):
    return FastUniform1dJITFunction.apply(
        w, x, y, src_idx, dst_idx, b_list, meta, fused_scatter
    )