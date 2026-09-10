from __future__ import annotations

import hashlib
import os
import time
import re
import io
import json
import contextlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    from torch.utils.cpp_extension import load
except Exception:  # pragma: no cover
    load = None

import importlib.machinery
import importlib.util
import sys

from .uniform1d_auto_schedule import (
    STC_PAD_VALUE,
    generate_code_stc_fwd_with_scheduler,
    generate_code_stc_bwd_with_scheduler,
    generate_code_stc_double_bwd_with_scheduler,
    _normalize_stc_padded_paths,
    infer_stc_path_lens_from_padded,
    _infer_stc_path_lens_tensor_from_padded,
)

from .uniform1d_jit import (
    _build_jit_candidates_from_sources as _u1d_build_jit_candidates_from_sources,
    _discover_prebuilt_jit_candidates as _u1d_discover_prebuilt_jit_candidates,
    _load_persistent_best_candidate as _u1d_load_persistent_best_candidate,
    _benchmark_and_select_best_jit_candidate as _u1d_benchmark_and_select_best_jit_candidate,
)


_MODULE_CACHE: Dict[str, Any] = {}


# -----------------------------------------------------------------------------
# Default auto-warp tuning policy
# -----------------------------------------------------------------------------
_DEFAULT_STC_FWD_TUNE_ENABLED = False
_DEFAULT_STC_BWD_TUNE_ENABLED = False
_DEFAULT_STC_DOUBLE_BWD_TUNE_ENABLED = False
_DEFAULT_STC_TUNE_WARMUP = 3
_DEFAULT_STC_TUNE_REPEAT = 10

# Fallbacks are used only when importing or smoke-testing without a visible GPU.
# Normal runtime tuning obtains these values from torch.cuda.get_device_properties().
_FALLBACK_STC_REGISTER_FILE_REGS_PER_SM = 65536
_FALLBACK_STC_WARP_SIZE = 32
_FALLBACK_STC_MAX_THREADS_PER_BLOCK = 1024


def _env_int(name: str, default: int, *, min_value: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return max(int(min_value), int(default))
    try:
        return max(int(min_value), int(raw))
    except ValueError:
        return max(int(min_value), int(default))


def _runtime_device_properties():
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(torch.cuda.current_device())
    except BaseException:
        pass
    return None


def _runtime_register_file_regs_per_sm() -> int:
    """Return total 32-bit register-file capacity per SM/CU from runtime API."""
    props = _runtime_device_properties()
    if props is not None:
        for attr in ("regs_per_multiprocessor", "regsPerMultiprocessor"):
            value = getattr(props, attr, None)
            if value is not None and int(value) > 0:
                return int(value)
    return _FALLBACK_STC_REGISTER_FILE_REGS_PER_SM


def _runtime_warp_size() -> int:
    """Return backend warp/wavefront size from runtime API."""
    props = _runtime_device_properties()
    if props is not None:
        for attr in ("warp_size", "warpSize"):
            value = getattr(props, attr, None)
            if value is not None and int(value) > 0:
                return int(value)
    return _FALLBACK_STC_WARP_SIZE


def _runtime_max_threads_per_block() -> int:
    """Return max block size from runtime API, used to cap generated warp candidates."""
    props = _runtime_device_properties()
    if props is not None:
        for attr in ("max_threads_per_block", "maxThreadsPerBlock"):
            value = getattr(props, attr, None)
            if value is not None and int(value) > 0:
                return int(value)
    return _FALLBACK_STC_MAX_THREADS_PER_BLOCK


def _autowarp_runtime_policy() -> tuple[int, int, int]:
    """Return (regs_per_sm, warp_size, max_threads_per_block) from runtime API.

    ptxas/hipcc registers/thread already accounts for dtype in 32-bit register
    words, so the auto-warp formula does not multiply by dtype bytes/words.
    """
    regs_per_sm = _runtime_register_file_regs_per_sm()
    warp_size = _runtime_warp_size()
    max_threads_per_block = _runtime_max_threads_per_block()
    return regs_per_sm, warp_size, max_threads_per_block

_FWD_BEST_CANDIDATE_CACHE: Dict[str, Any] = {}
_BWD_BEST_CANDIDATE_CACHE: Dict[str, Any] = {}
_DOUBLE_BWD_BEST_CANDIDATE_CACHE: Dict[str, Any] = {}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _detect_gpu_backend() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA/HIP GPU is available.")
    if getattr(torch.version, "hip", None):
        return "hip"
    return "cuda"


def _find_fasteq_root(start: Path) -> Optional[Path]:
    start = start.resolve()
    for p in [start, *start.parents]:
        if p.name == "fasteq":
            return p
    return None


def _default_build_root() -> Path:
    env_root = os.environ.get("FASTEQ_STC_JIT_CACHE_DIR")
    if env_root:
        root = Path(env_root).expanduser()
    else:
        backend = _detect_gpu_backend()
        env_root = os.environ.get("FASTEQ_JIT_CACHE_DIR", "").strip()
        if env_root:
            cache_root = Path(env_root).expanduser()
        else:
            xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
            cache_root = (
                Path(xdg_cache_home).expanduser()
                if xdg_cache_home
                else Path.home() / ".cache"
            ) / "fasteq"

        runtime_version = (
            getattr(torch.version, "hip", None)
            or getattr(torch.version, "cuda", None)
            or "unknown"
        )
        runtime_version = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(runtime_version))
        abi_key = (
            f"py{sys.version_info.major}{sys.version_info.minor}_"
            f"{backend}{runtime_version}"
        )
        root = cache_root / abi_key / "uniform1d_jit_codegen"
    _ensure_dir(root)
    return root


def _compiled_extension_path_in_dir(build_dir: Path, module_name: str) -> Optional[Path]:
    if not build_dir.exists():
        return None
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        cand = build_dir / f"{module_name}{suffix}"
        if cand.exists():
            return cand
    for cand in build_dir.iterdir():
        if not cand.is_file():
            continue
        name = cand.name
        if name.startswith(module_name) and any(name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
            return cand
    return None



def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "OFF", "off", "no", "No")


def _parse_register_count_from_build_log(log: str) -> Optional[int]:
    if not log:
        return None
    best = None
    for pat in [r"Used\s+(\d+)\s+registers", r"used\s+(\d+)\s+registers", r"vgpr_count\s*[:=]\s*(\d+)"]:
        for m in re.finditer(pat, log, flags=re.IGNORECASE):
            val = int(m.group(1))
            best = val if best is None else max(best, val)
    return best


def _register_meta_path(build_dir: Path, module_name: str) -> Path:
    return build_dir / f"{module_name}.registers.json"


def _store_register_metadata(build_dir: Path, module_name: str, registers: Optional[int], build_log: str = "") -> None:
    if registers is None:
        return
    try:
        _register_meta_path(build_dir, module_name).write_text(
            json.dumps({"registers_per_thread": int(registers)}, indent=2),
            encoding="utf-8",
        )
        if build_log:
            (build_dir / f"{module_name}.build.log").write_text(build_log, encoding="utf-8", errors="ignore")
    except BaseException:
        pass


def _load_register_metadata(build_dir: Path, module_name: str) -> Optional[int]:
    try:
        meta_path = _register_meta_path(build_dir, module_name)
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        value = meta.get("registers_per_thread")
        return None if value is None else int(value)
    except BaseException:
        return None


def _attach_register_metadata(mod: Any, build_dir: Path, module_name: str, registers: Optional[int]) -> None:
    if registers is None:
        registers = _load_register_metadata(build_dir, module_name)
    try:
        setattr(mod, "__fasteq_registers_per_thread__", None if registers is None else int(registers))
    except BaseException:
        pass


def _warp_candidates_from_registers(registers: Optional[int], *, u_dim: int) -> list[int]:
    """Return candidate warps/block using block_size * nvcc_reg <= regs_per_sm.

    block_size = warps_per_block * warp_size.  ``registers`` is the 1-warp
    compiler-reported registers/thread.  More warps than ceil(U / warp_size)
    are skipped because they would not own any U tile.
    """
    regs_per_sm, warp_size, max_threads_per_block = _autowarp_runtime_policy()
    warp_size = max(1, int(warp_size))
    max_warps_by_block = max(1, int(max_threads_per_block) // warp_size)
    max_warps_by_u = max(1, (int(u_dim) + warp_size - 1) // warp_size)

    if registers is None or int(registers) <= 0:
        n = min(max_warps_by_block, max_warps_by_u)
        return list(range(1, max(1, n) + 1))

    nvcc_reg = max(1, int(registers))
    max_warps_by_regs = max(1, int(regs_per_sm) // max(1, warp_size * nvcc_reg))
    n = max(1, min(max_warps_by_regs, max_warps_by_block, max_warps_by_u))
    return list(range(1, n + 1))

def _make_multiwarp_cuda_source(code: str, warps_per_block: int, *, warp_size: Optional[int] = None) -> str:
    warp_size = max(1, int(warp_size if warp_size is not None else _runtime_warp_size()))
    warps = max(1, int(warps_per_block))
    block_size = warp_size * warps
    out = str(code)

    multiwarp_header = (
        f"const int lane = tid % {warp_size};\n"
        f"    const int warp_id = tid / {warp_size};\n"
        f"    const int warp_count = blockDim.x / {warp_size};"
    )
    out = out.replace(
        "const int lane = tid & 31;\n    if (tid >= 32) return;",
        multiwarp_header,
    )
    out = out.replace(
        "const int lane = tid & 31;\n    const int warp_id = tid >> 5;\n    const int warp_count = blockDim.x >> 5;",
        multiwarp_header,
    )
    out = out.replace(
        "for (int u_base = 0; u_base < U; u_base += 32)",
        f"for (int u_base = warp_id * {warp_size}; u_base < U; u_base += warp_count * {warp_size})",
    )
    out = out.replace(
        "for (int u_base = warp_id * 32; u_base < U; u_base += warp_count * 32)",
        f"for (int u_base = warp_id * {warp_size}; u_base < U; u_base += warp_count * {warp_size})",
    )
    out = re.sub(r"dim3 block\(\d+\);", f"dim3 block({block_size});", out)
    return out

def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _get_tune_params(kind: str) -> tuple[bool, int, int]:
    if kind == "fwd":
        prefix = "FASTEQ_STC_FWD"
        default_enabled = _DEFAULT_STC_FWD_TUNE_ENABLED
    elif kind == "bwd":
        prefix = "FASTEQ_STC_BWD"
        default_enabled = _DEFAULT_STC_BWD_TUNE_ENABLED
    elif kind == "double_bwd":
        prefix = "FASTEQ_STC_DOUBLE_BWD"
        default_enabled = _DEFAULT_STC_DOUBLE_BWD_TUNE_ENABLED
    else:
        raise ValueError(f"unknown STC tune kind: {kind}")
    return (
        _env_flag(f"{prefix}_TUNE", "1" if default_enabled else "0"),
        _env_int(f"{prefix}_TUNE_WARMUP", _DEFAULT_STC_TUNE_WARMUP, min_value=0),
        _env_int(f"{prefix}_TUNE_REPEAT", _DEFAULT_STC_TUNE_REPEAT, min_value=1),
    )

def _load_prebuilt_jit_module(module_name: str, build_dir: Path):
    if module_name in _MODULE_CACHE:
        return _MODULE_CACHE[module_name]
    if module_name in sys.modules:
        mod = sys.modules[module_name]
        _MODULE_CACHE[module_name] = mod
        return mod
    ext_path = _compiled_extension_path_in_dir(build_dir, module_name)
    if ext_path is None:
        return None
    spec = importlib.util.spec_from_file_location(module_name, str(ext_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[module_name] = mod
    _attach_register_metadata(mod, build_dir, module_name, None)
    if os.environ.get("FASTEQ_STC_JIT_CACHE_LOG", "0") != "0":
        print(f"[JIT][STC] loaded prebuilt extension: {module_name}")
    return mod


def _stable_meta_hash(
    idx_lists: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    V: int,
    U: int,
    dtype: torch.dtype,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    use_multiwarp_candidates: bool = False,
) -> str:
    h = hashlib.sha1()
    tensors = [idx_lists.detach().cpu().contiguous()]
    if path_lens is not None:
        tensors.append(path_lens.detach().cpu().contiguous())
    tensors.append(coeffs.detach().cpu().reshape(-1).contiguous())

    for t in tensors:
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    h.update(str(int(V)).encode())
    h.update(str(int(U)).encode())
    h.update(str(dtype).encode())
    h.update(str(int(pad_value)).encode())
    h.update(b"autowarp" if bool(use_multiwarp_candidates) else b"w1only")
    # Keep generated-module caches ABI-safe.  v9 changes STC backward from an
    # x1-only result to (grad_x1, grad_x0), and double backward gains ggx0.
    h.update(b"stc-filejit-v9-sentinel-padding-fwd-bwd-x1-x0-double-bwd-ggx0")
    return h.hexdigest()[:16]



def _make_stc_fwd_tune_key(
    idx_lists: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    V: int,
    U: int,
    dtype: torch.dtype,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    use_multiwarp_candidates: bool = False,
) -> str:
    key = _stable_meta_hash(
        idx_lists,
        coeffs,
        V=V,
        U=U,
        dtype=dtype,
        path_lens=path_lens,
        pad_value=pad_value,
        use_multiwarp_candidates=use_multiwarp_candidates,
    )
    if dtype == torch.float32:
        dtype_str = "float"
    if dtype == torch.float64:
        dtype_str = "double"
    warp_tag = "autowarp" if bool(use_multiwarp_candidates) else "w1only"
    return f"stc_u1d_fwd_path_{idx_lists.shape[1]}_{dtype_str}_{warp_tag}_jit_{key}"


def _make_stc_bwd_tune_key(
    idx_lists: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    V: int,
    U: int,
    dtype: torch.dtype,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    use_multiwarp_candidates: bool = False,
    need_grad_x0: bool = True,
) -> str:
    key = _stable_meta_hash(
        idx_lists,
        coeffs,
        V=V,
        U=U,
        dtype=dtype,
        path_lens=path_lens,
        pad_value=pad_value,
        use_multiwarp_candidates=use_multiwarp_candidates,
    )
    if dtype == torch.float32:
        dtype_str = "float"
    if dtype == torch.float64:
        dtype_str = "double"
    warp_tag = "autowarp" if bool(use_multiwarp_candidates) else "w1only"
    grad_tag = "x1_x0" if bool(need_grad_x0) else "x1_only"
    return f"stc_u1d_bwd_{grad_tag}_path_{idx_lists.shape[1]}_{dtype_str}_{warp_tag}_jit_{key}"

def _make_stc_double_bwd_tune_key(
    idx_lists: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    V: int,
    U: int,
    dtype: torch.dtype,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    use_multiwarp_candidates: bool = False,
) -> str:
    key = _stable_meta_hash(
        idx_lists, coeffs, V=V, U=U, dtype=dtype, path_lens=path_lens,
        pad_value=pad_value, use_multiwarp_candidates=use_multiwarp_candidates,
    )
    if dtype == torch.float32:
        dtype_str = "float"
    elif dtype == torch.float64:
        dtype_str = "double"
    else:
        raise TypeError(f"Unsupported dtype for STC double backward JIT: {dtype}")
    warp_tag = "autowarp" if bool(use_multiwarp_candidates) else "w1only"
    return f"stc_u1d_double_bwd_x0gather_v3_ggx0_path_{idx_lists.shape[1]}_{dtype_str}_{warp_tag}_jit_{key}"


def _load_jit_module_file(*, module_name: str, code: Optional[str], build_dir: Path, verbose: bool = False):
    if load is None:
        raise RuntimeError("torch.utils.cpp_extension.load is not available")

    if module_name in _MODULE_CACHE:
        return _MODULE_CACHE[module_name]

    _ensure_dir(build_dir)
    cu_path = build_dir / f"{module_name}.cu"

    if code is None:
        prebuilt = _load_prebuilt_jit_module(module_name, build_dir)
        if prebuilt is not None:
            return prebuilt
        if not cu_path.exists():
            raise RuntimeError(f"JIT source file does not exist for module {module_name}: {cu_path}")
    else:
        cu_path.write_text(code, encoding="utf-8")

    log_buf = io.StringIO()
    with contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
        mod = load(
            name=module_name,
            sources=[str(cu_path)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
            build_directory=str(build_dir),
            verbose=verbose,
            with_cuda=True,
        )
    build_log = log_buf.getvalue()
    if build_log:
        print(build_log, end="")
    registers = _parse_register_count_from_build_log(build_log)
    _store_register_metadata(build_dir, module_name, registers, build_log)
    _attach_register_metadata(mod, build_dir, module_name, registers)
    if registers is not None and _env_flag("FASTEQ_STC_AUTOWARP_LOG", "0"):
        print(f"[JIT][STC] {module_name} registers_per_thread={registers}")
    _MODULE_CACHE[module_name] = mod
    return mod


def _build_stc_candidate_modules(
    *,
    base_name: str,
    base_code: str,
    kind: str,
    dtype: Any,
    u_dim: int,
    use_multiwarp_candidates: bool = False,
    verbose: bool = False,
) -> list[tuple[str, Any]]:
    """Build STC candidates through Uniform1D's shared JIT candidate layer."""
    del dtype, verbose  # kept for backward-compatible callers
    return _u1d_build_jit_candidates_from_sources(
        tune_key=base_name,
        raw_candidates=[("cand0", base_code)],
        cache=_MODULE_CACHE,
        kind=f"STC_{kind.upper()}",
        u_dim=int(u_dim),
        use_multiwarp_candidates=bool(use_multiwarp_candidates),
    )

def _get_or_build_module_candidates(
    idx_lists: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    V: int,
    U: int,
    dtype: torch.dtype,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    use_multiwarp_candidates: bool = False,
    verbose: bool = False,
):
    base_name = _make_stc_fwd_tune_key(
        idx_lists,
        coeffs,
        V=V,
        U=U,
        dtype=dtype,
        path_lens=path_lens,
        pad_value=pad_value,
        use_multiwarp_candidates=use_multiwarp_candidates,
    )
    if base_name in _FWD_BEST_CANDIDATE_CACHE:
        best_tag, best_mod, _best_ms = _FWD_BEST_CANDIDATE_CACHE[base_name]
        return [(best_tag, best_mod)]

    persistent_best = _u1d_load_persistent_best_candidate(
        tune_key=base_name,
        cache=_MODULE_CACHE,
        kind="STC_FWD",
    )
    if persistent_best is not None:
        best_tag, best_mod, best_ms = persistent_best
        _FWD_BEST_CANDIDATE_CACHE[base_name] = (best_tag, best_mod, best_ms)
        return [(best_tag, best_mod)]

    prebuilt_modules = _u1d_discover_prebuilt_jit_candidates(
        tune_key=base_name,
        cache=_MODULE_CACHE,
        kind="STC_FWD",
    )
    if prebuilt_modules:
        return prebuilt_modules

    idx_norm = _normalize_stc_padded_paths(idx_lists, coeff_list=coeffs, path_lens=path_lens)
    if path_lens is None:
        lens = infer_stc_path_lens_from_padded(idx_norm, coeffs, pad_value=pad_value)
    else:
        lens = path_lens.detach().cpu().to(torch.int64).reshape(-1)

    max_path_len = int(lens.max().item())
    num_paths = int(lens.numel())
    _launcher_name = f"{base_name}_u{int(U)}_path{num_paths}_maxlen{max_path_len}"

    code = generate_code_stc_fwd_with_scheduler(
        idx_lists,
        coeffs,
        path_lens=path_lens,
        pad_value=pad_value,
        num_out_segments=int(V),
        u_dim=int(U),
        out_path="",
        kernel_name=base_name,
    )
    return _build_stc_candidate_modules(
        base_name=base_name,
        base_code=code,
        kind="fwd",
        dtype=dtype,
        u_dim=int(U),
        use_multiwarp_candidates=use_multiwarp_candidates,
        verbose=verbose,
    )

def _get_or_build_module(*args, **kwargs):
    # Backward-compatible helper: return the first built candidate.
    return _get_or_build_module_candidates(*args, **kwargs)[0][1]



def _get_or_build_bwd_module_candidates(
    idx_lists: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    V: int,
    U: int,
    dtype: torch.dtype,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    use_multiwarp_candidates: bool = False,
    need_grad_x0: bool = True,
    verbose: bool = False,
):
    base_name = _make_stc_bwd_tune_key(
        idx_lists,
        coeffs,
        V=V,
        U=U,
        dtype=dtype,
        path_lens=path_lens,
        pad_value=pad_value,
        use_multiwarp_candidates=use_multiwarp_candidates,
        need_grad_x0=need_grad_x0,
    )
    if base_name in _BWD_BEST_CANDIDATE_CACHE:
        best_tag, best_mod, _best_ms = _BWD_BEST_CANDIDATE_CACHE[base_name]
        return [(best_tag, best_mod)]

    persistent_best = _u1d_load_persistent_best_candidate(
        tune_key=base_name,
        cache=_MODULE_CACHE,
        kind="STC_BWD",
    )
    if persistent_best is not None:
        best_tag, best_mod, best_ms = persistent_best
        _BWD_BEST_CANDIDATE_CACHE[base_name] = (best_tag, best_mod, best_ms)
        return [(best_tag, best_mod)]

    prebuilt_modules = _u1d_discover_prebuilt_jit_candidates(
        tune_key=base_name,
        cache=_MODULE_CACHE,
        kind="STC_BWD",
    )
    if prebuilt_modules:
        return prebuilt_modules

    code = generate_code_stc_bwd_with_scheduler(
        idx_lists,
        coeffs,
        path_lens=path_lens,
        pad_value=pad_value,
        num_out_segments=int(V),
        u_dim=int(U),
        out_path="",
        kernel_name=base_name,
        tile_u=32,
        need_grad_x0=bool(need_grad_x0),
    )
    return _build_stc_candidate_modules(
        base_name=base_name,
        base_code=code,
        kind="bwd",
        dtype=dtype,
        u_dim=int(U),
        use_multiwarp_candidates=use_multiwarp_candidates,
        verbose=verbose,
    )

def _get_or_build_bwd_module(*args, **kwargs):
    return _get_or_build_bwd_module_candidates(*args, **kwargs)[0][1]


def _get_or_build_double_bwd_module_candidates(
    idx_lists: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    V: int,
    U: int,
    dtype: torch.dtype,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    use_multiwarp_candidates: bool = False,
    verbose: bool = False,
):
    base_name = _make_stc_double_bwd_tune_key(
        idx_lists, coeffs, V=V, U=U, dtype=dtype, path_lens=path_lens,
        pad_value=pad_value, use_multiwarp_candidates=use_multiwarp_candidates,
    )
    if base_name in _DOUBLE_BWD_BEST_CANDIDATE_CACHE:
        tag, mod, _ms = _DOUBLE_BWD_BEST_CANDIDATE_CACHE[base_name]
        return [(tag, mod)]

    persistent_best = _u1d_load_persistent_best_candidate(
        tune_key=base_name, cache=_MODULE_CACHE, kind="STC_DOUBLE_BWD",
    )
    if persistent_best is not None:
        tag, mod, ms = persistent_best
        _DOUBLE_BWD_BEST_CANDIDATE_CACHE[base_name] = (tag, mod, ms)
        return [(tag, mod)]

    prebuilt_modules = _u1d_discover_prebuilt_jit_candidates(
        tune_key=base_name, cache=_MODULE_CACHE, kind="STC_DOUBLE_BWD",
    )
    if prebuilt_modules:
        return prebuilt_modules

    code = generate_code_stc_double_bwd_with_scheduler(
        idx_lists, coeffs, path_lens=path_lens, pad_value=pad_value,
        num_out_segments=int(V), u_dim=int(U), out_path="",
        kernel_name=base_name,
    )
    return _build_stc_candidate_modules(
        base_name=base_name, base_code=code, kind="double_bwd", dtype=dtype,
        u_dim=int(U), use_multiwarp_candidates=use_multiwarp_candidates,
        verbose=verbose,
    )


def _get_or_build_double_bwd_module(*args, **kwargs):
    return _get_or_build_double_bwd_module_candidates(*args, **kwargs)[0][1]


def _select_best_stc_fwd_module(base_key: str, candidates: list[tuple[str, Any]], x1, x0_g, V: int):
    tune_enabled, warmup, repeat = _get_tune_params("fwd")
    return _u1d_benchmark_and_select_best_jit_candidate(
        tune_key=base_key,
        candidates=candidates,
        best_cache=_FWD_BEST_CANDIDATE_CACHE,
        kind="STC_FWD",
        tune_enabled=tune_enabled,
        warmup=warmup,
        repeat=repeat,
        call_fn=lambda mod: mod.run(x1.contiguous(), x0_g.contiguous(), int(V)),
    )

def _select_best_stc_bwd_module(base_key: str, candidates: list[tuple[str, Any]], grad_out, x1, x0_g, V: int):
    tune_enabled, warmup, repeat = _get_tune_params("bwd")
    return _u1d_benchmark_and_select_best_jit_candidate(
        tune_key=base_key,
        candidates=candidates,
        best_cache=_BWD_BEST_CANDIDATE_CACHE,
        kind="STC_BWD",
        tune_enabled=tune_enabled,
        warmup=warmup,
        repeat=repeat,
        call_fn=lambda mod: mod.run(grad_out.contiguous(), x1.contiguous(), x0_g.contiguous(), int(V)),
    )


def _select_best_stc_double_bwd_module(
    base_key: str, candidates: list[tuple[str, Any]],
    grad_out, x1, x0, i0, grad_grad_x0, grad_grad_x1, V: int,
):
    tune_enabled, warmup, repeat = _get_tune_params("double_bwd")
    return _u1d_benchmark_and_select_best_jit_candidate(
        tune_key=base_key, candidates=candidates,
        best_cache=_DOUBLE_BWD_BEST_CANDIDATE_CACHE, kind="STC_DOUBLE_BWD",
        tune_enabled=tune_enabled, warmup=warmup, repeat=repeat,
        call_fn=lambda mod: mod.run(
            grad_out.contiguous(), x1.contiguous(), x0.contiguous(),
            i0.contiguous(), grad_grad_x0.contiguous(),
            grad_grad_x1.contiguous(), int(V),
        ),
    )

# -----------------------------------------------------------------------------
# Differentiable STC backward / training double backward
# -----------------------------------------------------------------------------

class FastSTCBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_out, x1, x0, i0, coeffs_tensor, idx_lists_tensor,
                num_out_segments, pad_value, use_multiwarp_candidates,
                need_grad_x0):
        # The public STC forward returns [B, V * U].  Generated kernels use
        # [B, V, U] internally.  x0 is intentionally kept in its ORIGINAL
        # ungathered shape; the first backward uses x0_g=x0[i0], while the
        # CUDA double backward reduces d_x0_g back to d_x0 through i0.
        B = int(x1.size(0))
        U = int(x1.size(2))
        V = int(num_out_segments)
        expected_grad_out_numel = B * V * U
        if int(grad_out.numel()) != expected_grad_out_numel:
            raise RuntimeError(
                f"STC grad_out numel mismatch: got shape={tuple(grad_out.shape)} "
                f"numel={grad_out.numel()}, expected B*V*U={B}*{V}*{U}="
                f"{expected_grad_out_numel}"
            )
        
        grad_out_shape = tuple(int(v) for v in grad_out.shape)
        grad_out_3d = grad_out.reshape(B, V, U).contiguous()

        # Generated double-bwd ABI uses int64 i0.  Keep this tensor in ctx so
        # tuning and the final launch see exactly the same contiguous metadata.
        i0_i64 = i0.to(device=x1.device, dtype=torch.int64).reshape(-1).contiguous()
        if int(i0_i64.numel()) != B:
            raise RuntimeError(
                f"STC i0 length mismatch: got {i0_i64.numel()}, expected B={B}"
            )

        need_grad_x0 = bool(need_grad_x0)

        # x0_g is always needed to form gx1.  Its gradient is generated only
        # when the original x0 input participates in autograd.
        x0_g = x0[i0_i64].contiguous()

        key = _make_stc_bwd_tune_key(
            idx_lists_tensor, coeffs_tensor, V=V, U=U,
            dtype=x1.dtype, path_lens=None, pad_value=int(pad_value),
            use_multiwarp_candidates=bool(use_multiwarp_candidates),
            need_grad_x0=need_grad_x0)
        candidates = _get_or_build_bwd_module_candidates(
            idx_lists_tensor, coeffs_tensor, path_lens=None, pad_value=int(pad_value),
            V=V, U=U, dtype=x1.dtype,
            use_multiwarp_candidates=bool(use_multiwarp_candidates),
            need_grad_x0=need_grad_x0)
        _tag, mod, _ms = _select_best_stc_bwd_module(
            key, candidates, grad_out_3d, x1.contiguous(), x0_g, V)
        bwd_result = mod.run(grad_out_3d, x1.contiguous(), x0_g, V)
        if need_grad_x0:
            gx1, gx0_g = bwd_result
        else:
            # The x1-only generated module returns a one-element vector.
            gx1 = bwd_result[0]
            gx0_g = None
        if tuple(gx1.shape) != tuple(x1.shape):
            raise RuntimeError(
                f"STC backward gx1 shape mismatch: got {tuple(gx1.shape)}, "
                f"expected {tuple(x1.shape)}"
            )
        if need_grad_x0 and tuple(gx0_g.shape) != tuple(x0_g.shape):
            raise RuntimeError(
                f"STC backward gathered gx0 shape mismatch: got {tuple(gx0_g.shape)}, "
                f"expected {tuple(x0_g.shape)}"
            )
        gx0 = None
        if need_grad_x0:
            gx0 = torch.zeros_like(x0)
            gx0.index_add_(0, i0_i64, gx0_g)

        ctx.save_for_backward(
            grad_out_3d, x1, x0, i0_i64, coeffs_tensor, idx_lists_tensor
        )
        ctx.grad_out_shape = grad_out_shape
        ctx.num_out_segments = V
        ctx.pad_value = int(pad_value)
        ctx.use_multiwarp_candidates = bool(use_multiwarp_candidates)
        ctx.need_grad_x0 = need_grad_x0

        return (gx1, gx0) if need_grad_x0 else gx1

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_grad_x1 = grad_outputs[0]
        grad_grad_x0 = grad_outputs[1] if ctx.need_grad_x0 else None
        (
            grad_out_3d, x1, x0, i0_i64, coeffs_tensor, idx_lists_tensor
        ) = ctx.saved_tensors
        if grad_grad_x1 is None and grad_grad_x0 is None:
            return None, None, None, None, None, None, None, None, None, None

        if grad_grad_x1 is None:
            grad_grad_x1 = torch.zeros_like(x1)
        if grad_grad_x0 is None:
            grad_grad_x0 = torch.zeros_like(x0)

        if tuple(grad_grad_x1.shape) != tuple(x1.shape):
            raise RuntimeError(
                f"STC grad_grad_x1 shape mismatch: got {tuple(grad_grad_x1.shape)}, "
                f"expected {tuple(x1.shape)}"
            )
        if tuple(grad_grad_x0.shape) != tuple(x0.shape):
            raise RuntimeError(
                f"STC grad_grad_x0 shape mismatch: got {tuple(grad_grad_x0.shape)}, "
                f"expected {tuple(x0.shape)}"
            )
        grad_grad_x1_3d = grad_grad_x1.reshape_as(x1).contiguous()
        grad_grad_x0_3d = grad_grad_x0.reshape_as(x0).contiguous()

        key = _make_stc_double_bwd_tune_key(
            idx_lists_tensor, coeffs_tensor,
            V=int(ctx.num_out_segments), U=int(x1.size(2)), dtype=x1.dtype,
            path_lens=None, pad_value=int(ctx.pad_value),
            use_multiwarp_candidates=bool(ctx.use_multiwarp_candidates),
        )
        candidates = _get_or_build_double_bwd_module_candidates(
            idx_lists_tensor, coeffs_tensor, path_lens=None,
            pad_value=int(ctx.pad_value), V=int(ctx.num_out_segments),
            U=int(x1.size(2)), dtype=x1.dtype,
            use_multiwarp_candidates=bool(ctx.use_multiwarp_candidates),
        )
        _tag, mod, _ms = _select_best_stc_double_bwd_module(
            key, candidates, grad_out_3d, x1, x0, i0_i64,
            grad_grad_x0_3d, grad_grad_x1_3d, int(ctx.num_out_segments),
        )
        d_go_3d, d_x1, d_x0 = mod.run(
            grad_out_3d, x1.contiguous(), x0.contiguous(),
            i0_i64, grad_grad_x0_3d, grad_grad_x1_3d,
            int(ctx.num_out_segments),
        )

        # The CUDA kernel already performs gather-backward reduction:
        # d_x0.shape == x0.shape, even when B != x0.size(0).
        if tuple(d_x0.shape) != tuple(x0.shape):
            raise RuntimeError(
                f"STC double backward d_x0 shape mismatch: got {tuple(d_x0.shape)}, "
                f"expected original x0 shape {tuple(x0.shape)}"
            )

        d_go = d_go_3d.reshape(ctx.grad_out_shape)
        d_x1_reshaped = d_x1.reshape(d_x1.shape[0], -1)
        d_x0_reshaped = d_x0.reshape(d_x0.shape[0], -1)

        return d_go, d_x1, d_x0, None, None, None, None, None, None, None


class FastSymmetricTensorContractionUniform1dFunction(torch.autograd.Function):
    """Forward uses generated STC-LARS code with preprocessed STC metadata.

    Public callers pass only sentinel-padded metadata produced during STC
    preprocessing: ``coeffs_tensor``, ``paths_tensor`` and ``idx_lists_tensor``.
    ``path_lens_tensor`` is no longer part of the public call path.  It is
    inferred internally only for the existing backward op, whose ABI still
    expects lengths.
    """

    @staticmethod
    def forward(
        ctx,
        x1,
        x0,
        i0,
        coeffs_tensor,
        paths_tensor,
        idx_lists_tensor,
        num_out_segments: int,
        pad_value: int = STC_PAD_VALUE,
        use_multiwarp_candidates: bool = False,
    ):
        #torch.cuda.synchronize()
        #start_time = time.perf_counter() * 1000

        x0_g = x0[i0]
        U = int(x1.size(2))

        # ``paths_tensor`` is the canonical preprocessed [P, max_len] tensor.
        # It must already use suffix padding with STC_PAD_VALUE.  We infer
        # lengths here only because the reused backward kernel still requires
        # path_lens_tensor.
        path_lens_tensor = _infer_stc_path_lens_tensor_from_padded(
            paths_tensor,
            coeffs_tensor,
            pad_value=int(pad_value),
        )

        if idx_lists_tensor is None:
            # This fallback should not normally be used after preprocessing, but
            # keeps the function robust if a caller only stores paths_tensor.
            idx_lists_tensor = paths_tensor

        fwd_key = _make_stc_fwd_tune_key(
            idx_lists_tensor,
            coeffs_tensor,
            V=int(num_out_segments),
            U=U,
            dtype=x1.dtype,
            path_lens=None,
            pad_value=int(pad_value),
            use_multiwarp_candidates=bool(use_multiwarp_candidates),
        )
        candidates = _get_or_build_module_candidates(
            idx_lists_tensor,
            coeffs_tensor,
            path_lens=None,
            pad_value=int(pad_value),
            V=int(num_out_segments),
            U=U,
            dtype=x1.dtype,
            use_multiwarp_candidates=bool(use_multiwarp_candidates),
        )
        best_tag, mod, _best_ms = _select_best_stc_fwd_module(
            fwd_key, candidates, x1.contiguous(), x0_g.contiguous(), int(num_out_segments)
        )
        out = mod.run(x1.contiguous(), x0_g.contiguous(), int(num_out_segments))

        out = out.view(out.shape[0], -1)

        #torch.cuda.synchronize()
        #end_time = time.perf_counter() * 1000
        #print(f"<< fasteq stc uniform1d-lars forward cost: {end_time - start_time:.3f} ms >>")

        ctx.save_for_backward(x1, x0, i0, coeffs_tensor, paths_tensor, path_lens_tensor, idx_lists_tensor)
        ctx.num_out_segments = int(num_out_segments)
        ctx.pad_value = int(pad_value)
        ctx.use_multiwarp_candidates = bool(use_multiwarp_candidates)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x1, x0, i0, coeffs_tensor, paths_tensor, path_lens_tensor, idx_lists_tensor = ctx.saved_tensors
        need_grad_x0 = bool(ctx.needs_input_grad[1])
        backward_args = (
            grad_out.contiguous(), x1, x0, i0, coeffs_tensor, idx_lists_tensor,
            int(ctx.num_out_segments), int(ctx.pad_value),
            bool(ctx.use_multiwarp_candidates), need_grad_x0,
        )
        if need_grad_x0:
            grad_x1, grad_x0 = FastSTCBackwardFunction.apply(*backward_args)
        else:
            grad_x1 = FastSTCBackwardFunction.apply(*backward_args)
            grad_x0 = None
        return grad_x1, grad_x0, None, None, None, None, None, None, None


def fast_stc_uniform1d_jit(
    x1,
    x0,
    i0,
    coeffs_tensor,
    paths_tensor,
    idx_lists_tensor,
    num_out_segments: int,
    *,
    pad_value: int = STC_PAD_VALUE,
    use_multiwarp_candidates: bool = False,
):
    """Run preprocessed STC Uniform1D JIT.

    Expected metadata is produced by the STC preprocessing stage:
      * ``paths_tensor``: [P, max_len], suffix padded with ``pad_value``.
      * ``idx_lists_tensor``: [max_len, P], same padded values transposed from
        ``paths_tensor``.

    ``path_lens_tensor`` and variable-argument parsing are intentionally not
    accepted here; preprocessing is the single source of metadata layout.

    ``use_multiwarp_candidates=False`` builds and uses only the one-warp
    candidate. Set it to ``True`` to generate, benchmark and select from legal
    multi-warp variants. Forward and backward always use the same setting.
    """
    return FastSymmetricTensorContractionUniform1dFunction.apply(
        x1,
        x0,
        i0,
        coeffs_tensor,
        paths_tensor,
        idx_lists_tensor,
        int(num_out_segments),
        int(pad_value),
        bool(use_multiwarp_candidates),
    )
