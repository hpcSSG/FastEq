import os
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import torch
import torch._dynamo
from torch.utils.cpp_extension import load

from .uniform1d_auto_schedule import (
    generate_code_uniform1d_fwd_with_scheduler,
    generate_code_uniform1d_bwd_with_scheduler,
)

# -----------------------------------------------------------------------------
# JIT cache
# -----------------------------------------------------------------------------

_FWD_JIT_CACHE: Dict[str, object] = {}
_BWD_JIT_CACHE: Dict[str, object] = {}

# For each Uniform1D forward/backward signature, remember the fastest candidate
# module selected by the first runtime microbenchmark in this Python process.
_FWD_BEST_CANDIDATE_CACHE: Dict[str, Tuple[str, object, float]] = {}
_BWD_BEST_CANDIDATE_CACHE: Dict[str, Tuple[str, object, float]] = {}


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _sanitize_module_tag(tag: Any) -> str:
    """Return a short string that is safe to embed in a C++ extension name."""
    s = str(tag)
    out = []
    for ch in s:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s[:48] if s else "cand"


def _normalize_codegen_candidates(codegen_out: Any) -> List[Tuple[str, str]]:
    """
    Normalize generate_code_uniform1d_fwd_with_scheduler output.

    Supported forms:
      - str                              -> [("cand0", code)]
      - [str, str, ...]                  -> [("cand0", code0), ...]
      - {"tag": code, ...}             -> [(tag, code), ...]
      - [{"name": tag, "code": code}, ...]
      - [(tag, code), ...]

    This keeps the current single-code path compatible while allowing the
    scheduler/codegen to return multiple candidate CUDA sources.
    """
    if isinstance(codegen_out, str):
        return [("cand0", codegen_out)]

    if isinstance(codegen_out, dict):
        if "candidates" in codegen_out:
            return _normalize_codegen_candidates(codegen_out["candidates"])

        if "code" in codegen_out:
            tag = codegen_out.get("name", codegen_out.get("tag", "cand0"))
            code = codegen_out["code"]
            if not isinstance(code, str):
                raise RuntimeError("candidate['code'] must be a CUDA source string")
            return [(str(tag), code)]

        items = []
        for tag, code in codegen_out.items():
            if isinstance(code, dict):
                code = code.get("code")
            if not isinstance(code, str):
                raise RuntimeError(f"candidate {tag!r} must be a CUDA source string")
            items.append((str(tag), code))
        return items

    if isinstance(codegen_out, (list, tuple)):
        # A common debug form is (code, schedule). Treat it as one candidate.
        if len(codegen_out) == 2 and isinstance(codegen_out[0], str) and not isinstance(codegen_out[1], str):
            return [("cand0", codegen_out[0])]

        # A single named candidate may be returned as (tag, code).
        if isinstance(codegen_out, tuple) and len(codegen_out) == 2 and all(isinstance(x, str) for x in codegen_out):
            return [(codegen_out[0], codegen_out[1])]

        items = []
        for idx, item in enumerate(codegen_out):
            if isinstance(item, str):
                items.append((f"cand{idx}", item))
            elif isinstance(item, dict):
                tag = item.get("name", item.get("tag", f"cand{idx}"))
                code = item.get("code")
                if not isinstance(code, str):
                    raise RuntimeError(f"candidate {idx} missing string field 'code'")
                items.append((str(tag), code))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                tag, code = item[0], item[1]
                if not isinstance(code, str):
                    raise RuntimeError(f"candidate {idx} second element must be code string")
                items.append((str(tag), code))
            else:
                raise RuntimeError(f"unsupported candidate format at index {idx}: {type(item)}")
        return items

    raise RuntimeError(
        "generate_code_uniform1d_fwd_with_scheduler must return a str, "
        "a list/tuple of candidates, or a dict of candidates."
    )


def _get_fwd_tune_params() -> Tuple[bool, int, int]:
    enabled = os.environ.get("FASTEQ_UNIFORM1D_FWD_TUNE", "1") != "0"
    warmup = int(os.environ.get("FASTEQ_UNIFORM1D_FWD_TUNE_WARMUP", "3"))
    repeat = int(os.environ.get("FASTEQ_UNIFORM1D_FWD_TUNE_REPEAT", "10"))
    return enabled, max(0, warmup), max(1, repeat)


def _get_bwd_tune_params() -> Tuple[bool, int, int]:
    enabled = os.environ.get("FASTEQ_UNIFORM1D_BWD_TUNE", "1") != "0"
    warmup = int(os.environ.get("FASTEQ_UNIFORM1D_BWD_TUNE_WARMUP", "3"))
    repeat = int(os.environ.get("FASTEQ_UNIFORM1D_BWD_TUNE_REPEAT", "10"))
    return enabled, max(0, warmup), max(1, repeat)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _find_fasteq_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if p.name == "fasteq":
            return p
    raise RuntimeError("Cannot find fasteq project root from __file__")



from pathlib import Path
import torch


def _detect_gpu_backend() -> str:
    """
    Return:
        "cuda" for NVIDIA CUDA
        "hip"  for AMD ROCm/HIP
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA/HIP GPU is available.")

    name = torch.cuda.get_device_name(0).lower()
    if "nvidia" in name:
        return "cuda"
    if "amd" in name or "radeon" in name or "instinct" in name or "bw200" in name:
        return "hip"

    raise RuntimeError(f"Cannot determine GPU backend from device name: {name}")


def _default_build_root() -> Path:
    backend = _detect_gpu_backend()

    root = (
        _find_fasteq_root(Path(__file__).parent)
        / backend
        / "src"
        / "uniform1d_jit_codegen"
    )

    _ensure_dir(root)
    return root


def _default_src_path() -> Path:
    backend = _detect_gpu_backend()
    
    root = _find_fasteq_root(Path(__file__).parent) / backend / "src"
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
        extra_cuda_cflags=extra_cuda_cflags or ["-O3", "--ptxas-options=-v"],
        #extra_cuda_cflags=extra_cuda_cflags or ["-O3", "--use_fast_math", "-lineinfo"],
        #extra_cuda_cflags=extra_cuda_cflags or ["-O3", "--offload-arch=gfx936"],
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
    layout_tag: str = "dense",
) -> str:
    # Include all code-affecting fields in the hash.  The previous version did
    # not include layout flags or coeff_list, which can accidentally reuse a
    # stale module when the same path structure is generated with different
    # x/y indirection, scatter mode, or constants.
    sig = repr((
        P, u_dim, dtype_str, mode, layout_tag,
        tuple(i_list), tuple(j_list), tuple(k_list), tuple(v_list),
        tuple(float(c) for c in coeff_list),
    ))
    h = _sha1_text(sig)
    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = _sanitize_module_tag(layout_tag)
    return f"uniform1d_fwd_{mode_str}_u{u_dim}_path{P}_{layout_tag}_jit_{dtype_str}_{h}"

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
    grad_w: bool,
    layout_tag: str = "dense",
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
) -> str:
    # Scheduler backward replaces the previous fused/split backward codegen.
    # Include layout and optional static dimensions in the signature so disk
    # cache cannot accidentally reuse a stale backward module generated by the
    # old split/combine paths or by a different scatter/source layout.
    sig = repr((
        "scheduler_bwd",
        P, u_dim, dtype_str, mode, grad_w, layout_tag,
        iw_dim, ix_dim, ky_dim, v_dim,
        tuple(i_list), tuple(j_list), tuple(k_list), tuple(v_list),
        tuple(float(c) for c in coeff_list),
    ))
    h = _sha1_text(sig)
    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = _sanitize_module_tag(layout_tag)
    gw_tag = "gradw" if grad_w else "nogradw"
    return f"uniform1d_bwd_sched_{mode_str}_u{u_dim}_path{P}_{layout_tag}_{gw_tag}_jit_{dtype_str}_{h}"


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
        #print(f"[JIT][{kind}] hit disk cache source: {cu_path}")
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

def _build_fwd_jit_candidates(
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
) -> Tuple[str, List[Tuple[str, object]]]:
    """
    Generate/build all forward candidate modules for one Uniform1D signature.

    Returns:
        tune_key, [(candidate_tag, loaded_module), ...]

    If a best candidate has already been tuned in this process, only that
    selected module is returned.
    """
    input_indices = {} if input_indices is None else input_indices
    output_indices = {} if output_indices is None else output_indices

    i_cpu = _tensor_to_cpu_list(i_list)
    j_cpu = _tensor_to_cpu_list(j_list)
    k_cpu = _tensor_to_cpu_list(k_list)
    v_cpu = _tensor_to_cpu_list(v_list)
    coeff_cpu = _tensor_to_cpu_list(coeff_list)

    P = len(i_cpu)
    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"

    tune_key = _make_fwd_module_name(
        P=P,
        u_dim=u_dim,
        mode=mode,
        i_list=i_cpu,
        j_list=j_cpu,
        k_list=k_cpu,
        v_list=v_cpu,
        coeff_list=coeff_cpu,
        dtype_str=dtype_str,
        layout_tag=layout_tag,
    )

    if tune_key in _FWD_BEST_CANDIDATE_CACHE:
        best_tag, best_mod, best_ms = _FWD_BEST_CANDIDATE_CACHE[tune_key]
        print(f"[JIT][FWD] hit best candidate cache: {tune_key} -> {best_tag} ({best_ms:.4f} ms)")
        return tune_key, [(best_tag, best_mod)]

    def _codegen_candidates():
        return generate_code_uniform1d_fwd_with_scheduler(
            i_list=i_list,
            j_list=j_list,
            k_list=k_list,
            v_list=v_list,
            coeff_list=coeff_list,
            input_indices=input_indices,
            output_indices=output_indices,
            u_dim=u_dim,
            mode=mode,
            reg_budget=[16, 64],
        )

    print(f"[JIT][FWD] generate candidates for: {tune_key}")
    raw_candidates = _normalize_codegen_candidates(_codegen_candidates())
    if not raw_candidates:
        raise RuntimeError("forward codegen returned zero candidates")

    modules: List[Tuple[str, object]] = []
    errors: List[Tuple[str, BaseException]] = []

    for cand_idx, (cand_tag, code) in enumerate(raw_candidates):
        safe_tag = _sanitize_module_tag(cand_tag or f"cand{cand_idx}")
        code_hash = _sha1_text(code)
        module_name = f"{tune_key}_{safe_tag}_{code_hash}"
        try:
            mod = _build_jit_module_common(
                module_name=module_name,
                cache=_FWD_JIT_CACHE,
                kind=f"FWD:{safe_tag}",
                codegen_fn=lambda code=code: code,
            )
            modules.append((safe_tag, mod))
        except BaseException as exc:
            errors.append((safe_tag, exc))
            print(f"[JIT][FWD] candidate {safe_tag} failed to build: {exc}")

    if not modules:
        details = "; ".join(f"{tag}: {exc}" for tag, exc in errors)
        raise RuntimeError(f"all forward candidates failed to build: {details}")

    return tune_key, modules


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
    # Backward-compatible wrapper: build candidates and return the first one.
    # Runtime auto-tuning is performed in _run_fwd, where real tensor inputs are
    # available for benchmarking.
    _, modules = _build_fwd_jit_candidates(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
        input_indices=input_indices,
        output_indices=output_indices,
        u_dim=u_dim,
        mode=mode,
        dtype_str=dtype_str,
    )
    return modules[0][1]

def _build_bwd_jit_candidates(
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
) -> Tuple[str, List[Tuple[str, object]]]:
    """
    Generate/build all backward scheduler candidates for one Uniform1D signature.

    Returns:
        tune_key, [(candidate_tag, loaded_module), ...]

    If a best candidate has already been tuned in this process, only that
    selected module is returned.
    """
    i_cpu = _tensor_to_cpu_list(i_list)
    j_cpu = _tensor_to_cpu_list(j_list)
    k_cpu = _tensor_to_cpu_list(k_list)
    v_cpu = _tensor_to_cpu_list(v_list)
    coeff_cpu = _tensor_to_cpu_list(coeff_list)

    input_indices = {} if input_indices is None else input_indices
    output_indices = {} if output_indices is None else output_indices

    P = len(i_cpu)
    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"

    tune_key = _make_bwd_module_name(
        P=P,
        u_dim=u_dim,
        mode=mode,
        i_list=i_cpu,
        j_list=j_cpu,
        k_list=k_cpu,
        v_list=v_cpu,
        coeff_list=coeff_cpu,
        dtype_str=dtype_str,
        grad_w=grad_w,
        layout_tag=layout_tag,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
    )

    if tune_key in _BWD_BEST_CANDIDATE_CACHE:
        best_tag, best_mod, best_ms = _BWD_BEST_CANDIDATE_CACHE[tune_key]
        print(f"[JIT][BWD] hit best candidate cache: {tune_key} -> {best_tag} ({best_ms:.4f} ms)")
        return tune_key, [(best_tag, best_mod)]

    reg_budgets = [16, 64, 128]
    def _codegen_candidates():
        return generate_code_uniform1d_bwd_with_scheduler(
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
            reg_budget=reg_budgets,
        )

    print(f"[JIT][BWD] generate candidates for: {tune_key}, reg_budget={reg_budgets}")
    raw_candidates = _normalize_codegen_candidates(_codegen_candidates())
    if not raw_candidates:
        raise RuntimeError("backward codegen returned zero candidates")

    modules: List[Tuple[str, object]] = []
    errors: List[Tuple[str, BaseException]] = []

    for cand_idx, (cand_tag, code) in enumerate(raw_candidates):
        safe_tag = _sanitize_module_tag(cand_tag or f"cand{cand_idx}")
        code_hash = _sha1_text(code)
        module_name = f"{tune_key}_{safe_tag}_{code_hash}"
        try:
            mod = _build_jit_module_common(
                module_name=module_name,
                cache=_BWD_JIT_CACHE,
                kind=f"BWD:{safe_tag}",
                codegen_fn=lambda code=code: code,
            )
            modules.append((safe_tag, mod))
        except BaseException as exc:
            errors.append((safe_tag, exc))
            print(f"[JIT][BWD] candidate {safe_tag} failed to build: {exc}")

    if not modules:
        details = "; ".join(f"{tag}: {exc}" for tag, exc in errors)
        raise RuntimeError(f"all backward candidates failed to build: {details}")

    return tune_key, modules


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
    # Backward-compatible wrapper: build candidates and return the first one.
    # Runtime auto-tuning is performed in _run_bwd, where real tensor inputs are
    # available for benchmarking.
    _, modules = _build_bwd_jit_candidates(
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
        dtype_str=dtype_str,
        grad_w=grad_w,
    )
    return modules[0][1]


# -----------------------------------------------------------------------------
# runtime dispatch
# -----------------------------------------------------------------------------
def _call_fwd_module(
    mod,
    *,
    w,
    x,
    y,
    src_idx,
    dst_idx,
    b_list,
    out_seg_num,
    fused_scatter: bool,
):
    if fused_scatter:
        return mod.run(w, x, y, src_idx, dst_idx, b_list, out_seg_num)
    return mod.run(w, x, y, src_idx, out_seg_num)



def _call_bwd_module(
    mod,
    *,
    w,
    x,
    y,
    grad_out,
    src_idx,
    dst_idx,
    b_list,
    out_seg_num,
    fused_scatter: bool,
):
    if fused_scatter:
        return mod.run(w, x, y, grad_out, src_idx, dst_idx, b_list, out_seg_num)
    return mod.run(w, x, y, grad_out, src_idx, out_seg_num)


def _select_best_fwd_module(
    *,
    tune_key: str,
    candidates: List[Tuple[str, object]],
    w,
    x,
    y,
    src_idx,
    dst_idx,
    b_list,
    out_seg_num,
    fused_scatter: bool,
) -> Tuple[str, object, float]:
    if tune_key in _FWD_BEST_CANDIDATE_CACHE:
        return _FWD_BEST_CANDIDATE_CACHE[tune_key]

    tune_enabled, warmup, repeat = _get_fwd_tune_params()
    if len(candidates) == 1 or not tune_enabled:
        tag, mod = candidates[0]
        best = (tag, mod, float("nan"))
        _FWD_BEST_CANDIDATE_CACHE[tune_key] = best
        if len(candidates) == 1:
            print(f"[JIT][FWD][tune] only one candidate: {tag}")
        else:
            print(f"[JIT][FWD][tune] disabled, use first candidate: {tag}")
        return best

    print(
        f"[JIT][FWD][tune] benchmarking {len(candidates)} candidates "
        f"for {tune_key}, warmup={warmup}, repeat={repeat}"
    )

    timings: List[Tuple[float, str, object]] = []
    for tag, mod in candidates:
        try:
            for _ in range(warmup):
                _call_fwd_module(
                    mod,
                    w=w, x=x, y=y,
                    src_idx=src_idx, dst_idx=dst_idx, b_list=b_list,
                    out_seg_num=out_seg_num, fused_scatter=fused_scatter,
                )
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(repeat):
                _call_fwd_module(
                    mod,
                    w=w, x=x, y=y,
                    src_idx=src_idx, dst_idx=dst_idx, b_list=b_list,
                    out_seg_num=out_seg_num, fused_scatter=fused_scatter,
                )
            torch.cuda.synchronize()
            avg_ms = (time.perf_counter() - t0) * 1000.0 / float(repeat)
            timings.append((avg_ms, tag, mod))
            print(f"[JIT][FWD][tune] candidate={tag:>16s} avg={avg_ms:.4f} ms")
        except BaseException as exc:
            print(f"[JIT][FWD][tune] candidate={tag} failed at runtime: {exc}")

    if not timings:
        raise RuntimeError("all forward candidates failed during runtime benchmark")

    timings.sort(key=lambda x: x[0])
    best_ms, best_tag, best_mod = timings[0]
    _FWD_BEST_CANDIDATE_CACHE[tune_key] = (best_tag, best_mod, best_ms)
    print(f"[JIT][FWD][tune] selected candidate={best_tag} avg={best_ms:.4f} ms")
    return best_tag, best_mod, best_ms



def _select_best_bwd_module(
    *,
    tune_key: str,
    candidates: List[Tuple[str, object]],
    w,
    x,
    y,
    grad_out,
    src_idx,
    dst_idx,
    b_list,
    out_seg_num,
    fused_scatter: bool,
) -> Tuple[str, object, float]:
    if tune_key in _BWD_BEST_CANDIDATE_CACHE:
        return _BWD_BEST_CANDIDATE_CACHE[tune_key]

    tune_enabled, warmup, repeat = _get_bwd_tune_params()
    if len(candidates) == 1 or not tune_enabled:
        tag, mod = candidates[0]
        best = (tag, mod, float("nan"))
        _BWD_BEST_CANDIDATE_CACHE[tune_key] = best
        if len(candidates) == 1:
            print(f"[JIT][BWD][tune] only one candidate: {tag}")
        else:
            print(f"[JIT][BWD][tune] disabled, use first candidate: {tag}")
        return best

    print(
        f"[JIT][BWD][tune] benchmarking {len(candidates)} candidates "
        f"for {tune_key}, warmup={warmup}, repeat={repeat}"
    )

    timings: List[Tuple[float, str, object]] = []
    for tag, mod in candidates:
        try:
            for _ in range(warmup):
                _call_bwd_module(
                    mod,
                    w=w, x=x, y=y, grad_out=grad_out,
                    src_idx=src_idx, dst_idx=dst_idx, b_list=b_list,
                    out_seg_num=out_seg_num, fused_scatter=fused_scatter,
                )
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(repeat):
                _call_bwd_module(
                    mod,
                    w=w, x=x, y=y, grad_out=grad_out,
                    src_idx=src_idx, dst_idx=dst_idx, b_list=b_list,
                    out_seg_num=out_seg_num, fused_scatter=fused_scatter,
                )
            torch.cuda.synchronize()
            avg_ms = (time.perf_counter() - t0) * 1000.0 / float(repeat)
            timings.append((avg_ms, tag, mod))
            print(f"[JIT][BWD][tune] candidate={tag:>16s} avg={avg_ms:.4f} ms")
        except BaseException as exc:
            print(f"[JIT][BWD][tune] candidate={tag} failed at runtime: {exc}")

    if not timings:
        raise RuntimeError("all backward candidates failed during runtime benchmark")

    timings.sort(key=lambda x: x[0])
    best_ms, best_tag, best_mod = timings[0]
    _BWD_BEST_CANDIDATE_CACHE[tune_key] = (best_tag, best_mod, best_ms)
    print(f"[JIT][BWD][tune] selected candidate={best_tag} avg={best_ms:.4f} ms")
    return best_tag, best_mod, best_ms


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
    tune_key, candidates = _build_fwd_jit_candidates(
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
        raise RuntimeError("Input_indices 1 and 2 all empty")

    if fused_scatter:
        b_list = b_list.to(torch.int32)
        dst_idx = output_indices[0].to(torch.int32)
    else:
        dst_idx = None
        b_list = None

    best_tag, best_mod, best_ms = _select_best_fwd_module(
        tune_key=tune_key,
        candidates=candidates,
        w=w, x=x, y=y,
        src_idx=src_idx, dst_idx=dst_idx, b_list=b_list,
        out_seg_num=out_seg_num,
        fused_scatter=fused_scatter,
    )

    return _call_fwd_module(
        best_mod,
        w=w, x=x, y=y,
        src_idx=src_idx, dst_idx=dst_idx, b_list=b_list,
        out_seg_num=out_seg_num,
        fused_scatter=fused_scatter,
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

    dtype_str = _get_scalar_t_str(w)
    tune_key, candidates = _build_bwd_jit_candidates(
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

    if fused_scatter:
        b_list = b_list.to(torch.int32)
        dst_idx = output_indices[0].to(torch.int32)
    else:
        dst_idx = None
        b_list = None

    best_tag, best_mod, best_ms = _select_best_bwd_module(
        tune_key=tune_key,
        candidates=candidates,
        w=w, x=x, y=y, grad_out=grad_out,
        src_idx=src_idx, dst_idx=dst_idx, b_list=b_list,
        out_seg_num=out_seg_num,
        fused_scatter=fused_scatter,
    )

    return _call_bwd_module(
        best_mod,
        w=w, x=x, y=y, grad_out=grad_out,
        src_idx=src_idx, dst_idx=dst_idx, b_list=b_list,
        out_seg_num=out_seg_num,
        fused_scatter=fused_scatter,
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

        #print(f"uniform1d forward output:{out}")

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