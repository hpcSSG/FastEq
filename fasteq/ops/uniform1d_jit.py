import os
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import torch
import torch._dynamo
from torch.utils.cpp_extension import load

from .uniform1d_fwd_codegen import generate_code_uniform1d_fwd
from .uniform1d_bwd_codegen import generate_code_uniform1d_bwd_fused
from .uniform1d_split_bwd_codegen import (
    #emit_backward_cuda_from_schedule,
    #build_backward_schedule_from_lists,
    #summarize_backward_schedule,
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
    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
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
    grad_w: bool,
) -> str:
    sig = repr((
        P, u_dim, dtype_str, split_mode, mode, grad_w,
        tuple(i_list), tuple(j_list), tuple(k_list), tuple(v_list),
        tuple(float(c) for c in coeff_list),
    ))
    h = _sha1_text(sig)
    tag = "split" if split_mode else "combine"
    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
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
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
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

    # enable tileU for Sevennet
    # disable tileU for Allegro
    codegen_fn = lambda: generate_code_uniform1d_fwd(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
        input_indices=input_indices,
        output_indices=output_indices,
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
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str,
    dtype_str: str,
    grad_w: bool,
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
        grad_w=grad_w,
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
        else:
            return generate_code_uniform1d_bwd_fused(
                    i_list=i_list,
                    j_list=j_list,
                    k_list=k_list,
                    v_list=v_list,
                    coeff_list=coeff_list,
                    input_indices=input_indices,
                    output_indices=output_indices,
                    u_dim=u_dim,
                    iw_dim=iw_dim,
                    ix_dim=ix_dim,
                    ky_dim=ky_dim,
                    v_dim=v_dim,
                    mode=mode,
                    need_grad_w=grad_w,
                )
                
            """ sched = build_backward_schedule_from_lists(
                i_list=i_cpu,
                j_list=j_cpu,
                k_list=k_cpu,
                v_list=v_cpu,
                coeff_list=coeff_cpu,
                U_dim=u_dim,
            )
            print(summarize_backward_schedule(sched))
            return emit_backward_cuda_from_schedule(
                sched,
                kernel_name=f"uniform1d_fused_u{u_dim}_path{P}_bwd",
                scalar_t=dtype_str,
            ) """
    

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
    b_list,
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
    input_indices,
    output_indices,
    out_seg_num,
    u_dim,
    mode,
):

    dtype_str = _get_scalar_t_str(w)
    mod = _build_fwd_jit_module(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
        u_dim=u_dim,
        input_indices=input_indices,
        output_indices=output_indices,
        dtype_str=dtype_str,
        mode=mode,
    )

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    fused_scatter = 0 in output_indices


    if use_x_src:
        src_idx = input_indices[1].to(torch.int32)
    elif use_y_src:
        src_idx = input_indices[2].to(torch.int32)
    else:
        src_idx = None
        raise RuntimeError("Input_indices 1 and 2 all empty")

    
    if fused_scatter:
        b_list = b_list.to(torch.int32)
        dst_idx = output_indices[0].to(torch.int32)
        return mod.run(
            w, x, y,
            src_idx, dst_idx, b_list, out_seg_num
        )
    else:
        return mod.run(
            w, x, y,
            src_idx, out_seg_num
        )

        
def _run_bwd(
    *,
    grad_out,
    w,
    x,
    y,
    b_list,
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
    input_indices,
    output_indices,
    out_seg_num,
    u_dim,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode,
    grad_w,
):

    """ dtype_str = _get_scalar_t_str(w)
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
    ) """

    dtype_str = _get_scalar_t_str(w)
    mod = _build_bwd_jit_module(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        input_indices=input_indices,
        output_indices=output_indices,
        dtype_str=dtype_str,
        mode=mode,
        grad_w=grad_w,
    )

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    fused_scatter = 0 in output_indices


    if use_x_src:
        src_idx = input_indices[1].to(torch.int32)
    elif use_y_src:
        src_idx = input_indices[2].to(torch.int32)
    else:
        src_idx = None
        raise RuntimeError("Input_indices 1 and 2 all empty")

    grad_out = grad_out.view(-1, out_seg_num, u_dim)

    #print(f"grad_out shape:{grad_out.shape}. w shape:{w.shape}, x shape:{x.shape}, y shape:{y.shape}")
    if fused_scatter:
        b_list = b_list.to(torch.int32)
        dst_idx = output_indices[0].to(torch.int32)
        return mod.run(
            w, x, y, grad_out,
            src_idx, dst_idx, b_list, out_seg_num
        )
    else:
        return mod.run(
            w, x, y, grad_out,
            src_idx, out_seg_num
        )
    


# -----------------------------------------------------------------------------
# autograd function
# -----------------------------------------------------------------------------

class FastUniform1dJITFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, y, input_indices, output_indices, meta, b_list):

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

        
        w_irreps =  int(meta["size_list"][0] / w_seg_num)
        x_irreps =  int(meta["size_list"][1] / x_seg_num)
        y_irreps =  int(meta["size_list"][2] / y_seg_num)
        out_irreps = int(meta["size_list"][3] / out_seg_num)

        

        w = w.view(-1, w_seg_num, w_irreps)
        x = x.view(-1, x_seg_num, x_irreps) # [node_num, x_seg_num, u_dim]

        y = y.view(-1, y_seg_num, y_irreps)

        #print(f"fasteq fwd w shape:{w.shape}, x shape:{x.shape}, y shape:{y.shape}")

        if y.shape[2] == 1:
            mode = "u,u,,u"
        else:
            mode = "u,u,u,u"

        P = i_list.numel()
        #print(f"[uniform1d][forward] P={P}, u_dim={u_dim}")

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000.0

        out = _run_fwd(
            w=w,
            x=x,
            y=y,
            i_list=i_list,
            j_list=j_list,
            k_list=k_list,
            v_list=v_list,
            coeff_list=coeff_list,
            input_indices=input_indices,
            output_indices=output_indices,
            b_list=b_list,
            out_seg_num=out_seg_num,
            u_dim=u_dim,
            mode=mode,
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000.0
        print(f"<< fasteq uniform1d fused forward cost: {end_time - start_time:.3f} ms >>")

        ctx.save_for_backward(w, x, y)
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
        ctx.w_irreps = w_irreps
        ctx.x_irreps = x_irreps
        ctx.y_irreps = y_irreps
        ctx.u_dim = u_dim
        ctx.P = P
        ctx.mode = mode
        ctx.input_indices=input_indices
        ctx.output_indices=output_indices

        return out

    @staticmethod
    def backward(ctx, grad_out):
        w, x, y = ctx.saved_tensors

        grad_out = grad_out.view(-1, ctx.out_seg_num, ctx.u_dim)

        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000.0

        if w.requires_grad:
            grad_w, grad_x, grad_y = _run_bwd(
                grad_out=grad_out,
                w=w,
                x=x,
                y=y,
                i_list=ctx.i_list,
                j_list=ctx.j_list,
                k_list=ctx.k_list,
                v_list=ctx.v_list,
                coeff_list=ctx.coeff_list,
                input_indices=ctx.input_indices,
                output_indices=ctx.output_indices,
                b_list=ctx.b_list,
                out_seg_num=ctx.out_seg_num,
                u_dim=ctx.u_dim,
                mode=ctx.mode,
                grad_w=w.requires_grad,
            )

            grad_w = grad_w.view(-1, ctx.w_seg_num * ctx.w_irreps)
            grad_x = grad_x.view(-1, ctx.x_seg_num * ctx.x_irreps)
            grad_y = grad_y.view(-1, ctx.y_seg_num * ctx.y_irreps)
        else:
            grad_x, grad_y = _run_bwd(
                grad_out=grad_out,
                w=w,
                x=x,
                y=y,
                i_list=ctx.i_list,
                j_list=ctx.j_list,
                k_list=ctx.k_list,
                v_list=ctx.v_list,
                coeff_list=ctx.coeff_list,
                input_indices=ctx.input_indices,
                output_indices=ctx.output_indices,
                b_list=ctx.b_list,
                out_seg_num=ctx.out_seg_num,
                u_dim=ctx.u_dim,
                iw_dim=ctx.w_seg_num,
                ix_dim=ctx.x_seg_num,
                ky_dim=ctx.y_seg_num,
                v_dim=ctx.out_seg_num,
                mode=ctx.mode,
                grad_w=w.requires_grad,
            )

            grad_x = grad_x.view(-1, ctx.x_seg_num * ctx.x_irreps)
            grad_y = grad_y.view(-1, ctx.y_seg_num * ctx.y_irreps)
            grad_w = None

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000.0
        print(f"<< fasteq uniform1d path:{ctx.P} backward cost: {end_time - start_time:.3f} ms >>")

        return grad_w, grad_x, grad_y, None, None, None, None

def fast_uniform1d_jit(w, x, y, input_indices, output_indices, meta, b_list):
    return FastUniform1dJITFunction.apply(
        w, x, y, input_indices, output_indices, meta, b_list
    )