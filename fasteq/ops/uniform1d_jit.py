import os
import time
import json
import sys
import shutil
import hashlib
import traceback
import re
import io
import contextlib
import importlib.util
import importlib.machinery
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List, Set

import torch
import torch._dynamo
from torch.utils.cpp_extension import load

from .uniform1d_auto_schedule import (
    generate_code_uniform1d_fwd_with_scheduler,
    generate_code_uniform1d_bwd_with_scheduler,
    generate_code_uniform1d_double_bwd_with_scheduler,
    generate_code_uniform1d_fwd_baseline_unrolled,
    generate_code_uniform1d_bwd_baseline_unrolled,

)

# -----------------------------------------------------------------------------
# JIT cache
# -----------------------------------------------------------------------------

_FWD_JIT_CACHE: Dict[str, object] = {}
_BWD_JIT_CACHE: Dict[str, object] = {}
_DOUBLE_BWD_JIT_CACHE: Dict[str, object] = {}

# For each Uniform1D forward/backward signature, remember the fastest candidate
# module selected by the first runtime microbenchmark in this Python process.
_FWD_BEST_CANDIDATE_CACHE: Dict[str, Tuple[str, object, float]] = {}
_BWD_BEST_CANDIDATE_CACHE: Dict[str, Tuple[str, object, float]] = {}
_DOUBLE_BWD_BEST_CANDIDATE_CACHE: Dict[str, Tuple[str, object, float]] = {}

# Fast hot-path mapping from runtime metadata identity to the final tune_key.
# This avoids rebuilding the expensive content-hash tune_key on every call.
_FWD_TUNE_KEY_FAST_CACHE: Dict[Tuple[Any, ...], str] = {}
_BWD_TUNE_KEY_FAST_CACHE: Dict[Tuple[Any, ...], str] = {}
_DOUBLE_BWD_TUNE_KEY_FAST_CACHE: Dict[Tuple[Any, ...], str] = {}

# Avoid repeating expensive disk cleanup / source pruning on every hot-path call.
_JIT_PRUNE_DONE_CACHE: Set[Tuple[str, str, str]] = set()

# Cache int32 copies of immutable path metadata tensors.  If meta stores int64
# tensors, calling .to(torch.int32) in every forward creates fresh tensors and
# breaks the fast build-cache identity key.
_META_INT32_TENSOR_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}


# -----------------------------------------------------------------------------
# Default auto-warp tuning policy
# -----------------------------------------------------------------------------
# Normal callers do not need to set warp/register-capacity environment variables.
# Warp size and register-file capacity are read from torch CUDA/HIP runtime device
# properties.  The maximum tested warps/block is derived after compiling the
# 1-warp candidate and parsing its compiler-reported registers/thread.
_DEFAULT_UNIFORM1D_FWD_TUNE_ENABLED = True
_DEFAULT_UNIFORM1D_BWD_TUNE_ENABLED = True
_DEFAULT_UNIFORM1D_TUNE_WARMUP = 3
_DEFAULT_UNIFORM1D_TUNE_REPEAT = 10
# Generate and benchmark w2/w3/... candidates in addition to the mandatory w1 candidate.
_DEFAULT_UNIFORM1D_MULTI_WARP_CANDIDATES_ENABLED = True

# Fallbacks are used only when importing or smoke-testing without a visible GPU.
# Normal runtime tuning obtains these values from torch.cuda.get_device_properties().
_FALLBACK_UNIFORM1D_REGISTER_FILE_REGS_PER_SM = 65536
_FALLBACK_UNIFORM1D_WARP_SIZE = 32
_FALLBACK_UNIFORM1D_MAX_THREADS_PER_BLOCK = 1024


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw not in ("0", "false", "False", "OFF", "off", "no", "No")


def _env_int(name: str, default: int, *, min_value: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return max(int(min_value), int(default))
    try:
        return max(int(min_value), int(raw))
    except ValueError:
        return max(int(min_value), int(default))


def _resolve_multiwarp_candidates_enabled(value: Optional[bool]) -> bool:
    """Resolve the per-call multi-warp candidate switch.

    A non-None function argument has priority.  When it is None, the environment
    variable keeps command-line experimentation convenient without changing the
    operator call site.
    """
    if value is not None:
        return bool(value)
    return _env_bool(
        "FASTEQ_UNIFORM1D_MULTI_WARP_CANDIDATES",
        _DEFAULT_UNIFORM1D_MULTI_WARP_CANDIDATES_ENABLED,
    )


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
    return _FALLBACK_UNIFORM1D_REGISTER_FILE_REGS_PER_SM


def _runtime_warp_size() -> int:
    """Return backend warp/wavefront size from runtime API."""
    props = _runtime_device_properties()
    if props is not None:
        for attr in ("warp_size", "warpSize"):
            value = getattr(props, attr, None)
            if value is not None and int(value) > 0:
                return int(value)
    return _FALLBACK_UNIFORM1D_WARP_SIZE


def _runtime_max_threads_per_block() -> int:
    """Return max block size from runtime API, used to cap generated warp candidates."""
    props = _runtime_device_properties()
    if props is not None:
        for attr in ("max_threads_per_block", "maxThreadsPerBlock"):
            value = getattr(props, attr, None)
            if value is not None and int(value) > 0:
                return int(value)
    return _FALLBACK_UNIFORM1D_MAX_THREADS_PER_BLOCK


def _autowarp_runtime_policy() -> Tuple[int, int, int]:
    """Return (regs_per_sm, warp_size, max_threads_per_block) from runtime API.

    The compiler-reported register count parsed from ptxas/hipcc is already the
    per-thread number of 32-bit register words for the current dtype and target.
    Therefore dtype does not appear in the occupancy-style warp candidate formula.
    """
    regs_per_sm = _runtime_register_file_regs_per_sm()
    warp_size = _runtime_warp_size()
    max_threads_per_block = _runtime_max_threads_per_block()
    return regs_per_sm, warp_size, max_threads_per_block


def _format_cuda_arch_list_from_runtime() -> Optional[str]:
    """Return a TORCH_CUDA_ARCH_LIST string using the parent process runtime.

    torch.utils.cpp_extension.load() calls _get_cuda_arch_flags().  When
    TORCH_CUDA_ARCH_LIST is unset, PyTorch queries torch.cuda.get_device_capability()
    inside the process that performs compilation.  That is unsafe in a forked
    worker after the parent has initialized CUDA and causes:

        RuntimeError: Cannot re-initialize CUDA in forked subprocess

    Therefore the parent process should materialize the arch list once and pass
    it to compile workers through the environment.
    """
    try:
        if not torch.cuda.is_available():
            return None
        archs = set()
        ndev = max(1, int(torch.cuda.device_count()))
        for i in range(ndev):
            major, minor = torch.cuda.get_device_capability(i)
            archs.add(f"{int(major)}.{int(minor)}")
        return ";".join(sorted(archs)) if archs else None
    except BaseException:
        return None


def _format_rocm_arch_list_from_runtime() -> Optional[str]:
    """Best-effort ROCm arch list for HIP compile workers."""
    try:
        if not torch.cuda.is_available():
            return None
        archs = set()
        ndev = max(1, int(torch.cuda.device_count()))
        for i in range(ndev):
            props = torch.cuda.get_device_properties(i)
            for attr in ("gcnArchName", "gcn_arch_name"):
                value = getattr(props, attr, None)
                if value:
                    # PyTorch may return gfx90a:sramecc-:xnack-.  HIP compile
                    # flags generally accept the full string, and users can
                    # override with PYTORCH_ROCM_ARCH if their toolchain wants
                    # plain gfx90a/gfx936.
                    archs.add(str(value))
                    break
        return ";".join(sorted(archs)) if archs else None
    except BaseException:
        return None


def _ensure_compile_arch_env(*, allow_runtime_query: bool = True) -> None:
    """Set compile arch env vars before entering subprocess JIT builds.

    This avoids CUDA/HIP runtime probing in forked compile workers.  It also
    makes spawn/forkserver workers cheaper because cpp_extension can skip device
    capability discovery.
    """
    if getattr(torch.version, "hip", None):
        if os.environ.get("PYTORCH_ROCM_ARCH") or os.environ.get("AMDGPU_TARGETS"):
            return
        if allow_runtime_query:
            arch = _format_rocm_arch_list_from_runtime()
            if arch:
                os.environ.setdefault("PYTORCH_ROCM_ARCH", arch)
                os.environ.setdefault("AMDGPU_TARGETS", arch)
        return

    if getattr(torch.version, "cuda", None):
        if os.environ.get("TORCH_CUDA_ARCH_LIST"):
            return
        if allow_runtime_query:
            arch = _format_cuda_arch_list_from_runtime()
            if arch:
                os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
        return

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
    return (
        _env_bool("FASTEQ_UNIFORM1D_FWD_TUNE", _DEFAULT_UNIFORM1D_FWD_TUNE_ENABLED),
        _env_int("FASTEQ_UNIFORM1D_FWD_TUNE_WARMUP", _DEFAULT_UNIFORM1D_TUNE_WARMUP, min_value=0),
        _env_int("FASTEQ_UNIFORM1D_FWD_TUNE_REPEAT", _DEFAULT_UNIFORM1D_TUNE_REPEAT, min_value=1),
    )


def _get_bwd_tune_params() -> Tuple[bool, int, int]:
    return (
        _env_bool("FASTEQ_UNIFORM1D_BWD_TUNE", _DEFAULT_UNIFORM1D_BWD_TUNE_ENABLED),
        _env_int("FASTEQ_UNIFORM1D_BWD_TUNE_WARMUP", _DEFAULT_UNIFORM1D_TUNE_WARMUP, min_value=0),
        _env_int("FASTEQ_UNIFORM1D_BWD_TUNE_REPEAT", _DEFAULT_UNIFORM1D_TUNE_REPEAT, min_value=1),
    )


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
    """Return the build backend without forcing CUDA/HIP runtime init.

    Compile workers may be forked after the parent process has already used the
    GPU.  Calling torch.cuda.is_available()/get_device_name() in those workers
    can trigger CUDA re-initialization errors.  torch.version is enough to decide
    which source tree should be used for a PyTorch build.
    """
    if getattr(torch.version, "hip", None):
        return "hip"
    if getattr(torch.version, "cuda", None):
        return "cuda"

    # Last-resort fallback for unusual builds.  This branch may initialize the
    # runtime, so normal CUDA/HIP builds should not reach it.
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA/HIP GPU is available.")

    name = torch.cuda.get_device_name(0).lower()
    if "amd" in name or "radeon" in name or "instinct" in name or "bw200" in name:
        return "hip"
    if "nvidia" in name:
        return "cuda"
    raise RuntimeError(f"Cannot determine GPU backend from device name: {name}")


def _default_build_root() -> Path:
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


def _default_src_path() -> Path:
    backend = _detect_gpu_backend()
    
    root = _find_fasteq_root(Path(__file__).parent) / backend / "src"
    _ensure_dir(root)
    return root


def _write_code_file(code: str, build_dir: Path, module_name: str) -> Path:
    cu_path = build_dir / f"{module_name}.cu"
    cu_path.write_text(code, encoding="utf-8")
    return cu_path



def _parse_register_count_from_build_log(log: str) -> Optional[int]:
    """Parse ptxas/hipcc register usage from a torch extension build log."""
    if not log:
        return None
    patterns = [
        r"Used\s+(\d+)\s+registers",
        r"used\s+(\d+)\s+registers",
        r"SGPRs?:\s*(\d+).*?VGPRs?:\s*(\d+)",
        r"vgpr_count\s*[:=]\s*(\d+)",
    ]
    best = None
    for pat in patterns:
        for m in re.finditer(pat, log, flags=re.IGNORECASE | re.DOTALL):
            if len(m.groups()) >= 2:
                val = max(int(m.group(1)), int(m.group(2)))
            else:
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


def _attach_register_metadata(mod: object, build_dir: Path, module_name: str, registers: Optional[int]) -> None:
    if registers is None:
        registers = _load_register_metadata(build_dir, module_name)
    try:
        setattr(mod, "__fasteq_registers_per_thread__", None if registers is None else int(registers))
    except BaseException:
        pass


def _warp_candidates_from_registers(registers: Optional[int], *, u_dim: int) -> List[int]:
    """Return candidate warps/block using the agreed register-capacity rule.

    Definitions:
      - block_size = warps_per_block * warp_size
      - nvcc_reg is compiler-reported registers/thread from the 1-warp build
      - regs_per_sm and warp_size come from torch.cuda.get_device_properties()

    The register bound is:

        block_size * nvcc_reg <= regs_per_sm

    equivalently:

        max_warps_by_regs = regs_per_sm // (warp_size * nvcc_reg)

    Because these kernels split work by U tiles, testing more warps than
    ceil(U / warp_size) only creates idle warps, so U also caps candidates.
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
    """Convert a 1-warp U-tile kernel source into a multi-warp U-tile variant."""
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
    if code is None:
        # If the extension binary already exists, import it directly and avoid
        # torch cpp_extension.load(), which may still run ninja.
        ext_path = _compiled_extension_path_in_dir(build_dir, module_name)
        if ext_path is not None:
            return _load_prebuilt_jit_module(
                module_name=module_name,
                cache=cache,
                kind="JIT",
            )

    if code is not None:
        cu_path.write_text(code, encoding="utf-8")
    elif not cu_path.exists():
        raise RuntimeError(
            f"JIT source file does not exist for module {module_name}: {cu_path}"
        )

    log_buf = io.StringIO()
    mod = None
    with contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
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

    build_log = log_buf.getvalue()
    print(build_log, end="")
    registers = _parse_register_count_from_build_log(build_log)
    _store_register_metadata(build_dir, module_name, registers, build_log)
    _attach_register_metadata(mod, build_dir, module_name, registers)
    print(f"[JIT][{module_name}] registers_per_thread={registers}")

    cache[module_name] = mod
    return mod




# -----------------------------------------------------------------------------
# Prebuilt JIT module discovery / direct loading
# -----------------------------------------------------------------------------

def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "OFF", "off", "no", "No")


def _jit_cache_log_enabled() -> bool:
    # Cache-hit/prune prints are useful when debugging the JIT cache, but they
    # easily dominate millisecond-level kernels and become very noisy when
    # subprocesses are used for candidate compilation.
    return _env_flag("FASTEQ_UNIFORM1D_JIT_CACHE_LOG", "0")


def _tensor_runtime_identity(x: Any) -> Tuple[Any, ...]:
    """Cheap, hashable identity for immutable path metadata tensors.

    The full tune_key still hashes the actual i/j/k/v/coeff contents the first
    time a signature is seen.  For steady-state calls we only need to recognize
    that the same metadata object/storage is being reused, so data_ptr/shape/dtype
    is enough and avoids .cpu().tolist().  Include _version when available to
    avoid stale reuse after accidental in-place metadata mutation.
    """
    if isinstance(x, torch.Tensor):
        try:
            return (
                "tensor",
                str(x.device),
                int(x.data_ptr()),
                tuple(int(d) for d in x.shape),
                tuple(int(d) for d in x.stride()),
                int(x.storage_offset()),
                str(x.dtype),
                int(getattr(x, "_version", 0)),
            )
        except BaseException:
            return ("tensor_id", id(x))

    if isinstance(x, (list, tuple)):
        # Path metadata supplied as Python lists is usually small/static.  Use
        # value identity here so a mutated list naturally changes the key.
        try:
            return ("seq", tuple(x))
        except TypeError:
            return ("seq_id", id(x), len(x))

    return ("obj", id(x), type(x).__name__)


def _as_int32_meta_tensor(x: Any) -> Any:
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype == torch.int32:
        return x

    key = ("to_int32",) + _tensor_runtime_identity(x)
    cached = _META_INT32_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached

    out = x.to(torch.int32)
    if _env_flag("FASTEQ_UNIFORM1D_CACHE_INT32_META", "1"):
        _META_INT32_TENSOR_CACHE[key] = out
    return out


def _make_candidate_fast_key(
    *,
    kind: str,
    i_list: Any,
    j_list: Any,
    k_list: Any,
    v_list: Any,
    coeff_list: Any,
    input_indices: Optional[Dict[int, Any]],
    output_indices: Optional[Dict[int, Any]],
    u_dim: int,
    iw_dim: Optional[int],
    ix_dim: Optional[int],
    ky_dim: Optional[int],
    v_dim: Optional[int],
    mode: str,
    dtype_str: str,
    grad_w: Optional[bool] = None,
    grad_x: Optional[bool] = None,
    grad_y: Optional[bool] = None,
    use_multiwarp_candidates: bool = True,
) -> Tuple[Any, ...]:
    input_indices = {} if input_indices is None else input_indices
    output_indices = {} if output_indices is None else output_indices
    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices
    return (
        kind,
        _tensor_runtime_identity(i_list),
        _tensor_runtime_identity(j_list),
        _tensor_runtime_identity(k_list),
        _tensor_runtime_identity(v_list),
        _tensor_runtime_identity(coeff_list),
        int(u_dim),
        None if iw_dim is None else int(iw_dim),
        None if ix_dim is None else int(ix_dim),
        None if ky_dim is None else int(ky_dim),
        None if v_dim is None else int(v_dim),
        str(mode),
        str(dtype_str),
        None if grad_w is None else bool(grad_w),
        None if grad_x is None else bool(grad_x),
        None if grad_y is None else bool(grad_y),
        bool(use_multiwarp_candidates),
        bool(use_x_src),
        bool(use_y_src),
        bool(use_scatter),
    )


def _compiled_extension_path_in_dir(build_dir: Path, module_name: str) -> Optional[Path]:
    """
    Return the compiled extension path for a torch cpp_extension module if it
    already exists.  This is intentionally stricter than checking for a .cu file:
    a .cu file means code was generated, but the kernel may not have been built.
    """
    if not build_dir.exists():
        return None

    # Typical torch extension output: <build_dir>/<module_name>.so or .pyd.
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        cand = build_dir / f"{module_name}{suffix}"
        if cand.exists():
            return cand

    # Be tolerant of platform-specific suffixes or ABI tags.
    for cand in build_dir.iterdir():
        if not cand.is_file():
            continue
        name = cand.name
        if not name.startswith(module_name):
            continue
        if any(name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
            return cand

    return None


def _compiled_extension_path(module_name: str) -> Optional[Path]:
    build_dir = _default_build_root() / module_name
    return _compiled_extension_path_in_dir(build_dir, module_name)


def _has_prebuilt_jit_module(module_name: str) -> bool:
    return _compiled_extension_path(module_name) is not None


def _load_prebuilt_jit_module(
    *,
    module_name: str,
    cache: Dict[str, object],
    kind: str = "JIT",
):
    """
    Load an already-compiled extension directly from its .so/.pyd.

    This bypasses torch.utils.cpp_extension.load(), so it does not regenerate
    ninja files or invoke nvcc/hipcc.  It is used only when a compiled extension
    binary is already present on disk.
    """
    if module_name in cache:
        return cache[module_name]

    if module_name in sys.modules:
        mod = sys.modules[module_name]
        cache[module_name] = mod
        return mod

    ext_path = _compiled_extension_path(module_name)
    if ext_path is None:
        raise FileNotFoundError(f"compiled extension not found for {module_name}")

    spec = importlib.util.spec_from_file_location(module_name, str(ext_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {ext_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    cache[module_name] = mod
    try:
        build_dir = _default_build_root() / module_name
        _attach_register_metadata(mod, build_dir, module_name, None)
    except BaseException:
        pass
    if _jit_cache_log_enabled():
        print(f"[JIT][{kind}] loaded prebuilt extension: {module_name}")
    return mod


def _parse_candidate_from_module_name(tune_key: str, module_name: str) -> Optional[str]:
    prefix = f"{tune_key}_"
    if not module_name.startswith(prefix):
        return None

    suffix = module_name[len(prefix):]
    if "_" not in suffix:
        return None

    safe_tag, code_hash = suffix.rsplit("_", 1)
    if len(code_hash) != 16:
        return None

    try:
        int(code_hash, 16)
    except ValueError:
        return None

    return safe_tag or "cand"


def _discover_prebuilt_jit_candidates(
    *,
    tune_key: str,
    cache: Dict[str, object],
    kind: str,
) -> List[Tuple[str, object]]:
    """
    Discover already-compiled candidate modules for this tune_key.

    This is the key fast path: it avoids calling codegen_fn(), so scheduler
    search/LARS scheduling and CUDA source regeneration are skipped entirely.
    """

    build_root = _default_build_root()
    if not build_root.exists():
        return []

    found: List[Tuple[str, str, Path]] = []
    prefix = f"{tune_key}_"
    for build_dir in sorted(build_root.iterdir(), key=lambda p: p.name):
        if not build_dir.is_dir():
            continue
        module_name = build_dir.name
        if not module_name.startswith(prefix):
            continue

        safe_tag = _parse_candidate_from_module_name(tune_key, module_name)
        if safe_tag is None:
            continue

        ext_path = _compiled_extension_path_in_dir(build_dir, module_name)
        if ext_path is None:
            continue

        found.append((safe_tag, module_name, ext_path))

    if not found:
        return []

    modules: List[Tuple[str, object]] = []
    errors: List[Tuple[str, BaseException]] = []
    for safe_tag, module_name, _ext_path in found:
        try:
            mod = _load_prebuilt_jit_module(
                module_name=module_name,
                cache=cache,
                kind=f"{kind}:{safe_tag}",
            )
            modules.append((safe_tag, mod))
        except BaseException as exc:
            errors.append((safe_tag, exc))
            print(f"[JIT][{kind}] prebuilt candidate {safe_tag} failed to load: {exc}")

    if modules:
        if _jit_cache_log_enabled():
            print(
                f"[JIT][{kind}] hit prebuilt candidate cache for {tune_key}: "
                f"{len(modules)} module(s); skip candidate generation, scheduling and compilation"
            )
    elif errors:
        details = "; ".join(f"{tag}: {exc}" for tag, exc in errors)
        print(f"[JIT][{kind}] found prebuilt candidates but none loaded: {details}")

    return modules


def _best_candidate_meta_path(kind: str, tune_key: str) -> Path:
    meta_dir = _default_build_root() / "_uniform1d_jit_best"
    _ensure_dir(meta_dir)
    return meta_dir / f"uniform1d_{kind.lower()}_{_sha1_text(tune_key)}.json"


def _load_persistent_best_candidate(
    *,
    tune_key: str,
    cache: Dict[str, object],
    kind: str,
) -> Optional[Tuple[str, object, float]]:
    """
    Load the best candidate selected by a previous process.

    When this hits, callers can skip candidate generation/scheduling/compilation
    and also skip runtime benchmarking.
    """

    meta_path = _best_candidate_meta_path(kind, tune_key)
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("tune_key") != tune_key or meta.get("kind") != kind:
            return None

        module_name = str(meta["module_name"])
        tag = str(meta.get("tag", "cand"))
        best_ms = float(meta.get("best_ms", float("nan")))

        if not _has_prebuilt_jit_module(module_name):
            return None

        mod = _load_prebuilt_jit_module(
            module_name=module_name,
            cache=cache,
            kind=f"{kind}:{tag}:best",
        )
        if _jit_cache_log_enabled():
            print(
                f"[JIT][{kind}] hit persistent best candidate: "
                f"{tune_key} -> {tag} ({best_ms:.4f} ms); "
                f"skip candidate generation, scheduling, compilation and tuning"
            )
        return tag, mod, best_ms
    except BaseException as exc:
        print(f"[JIT][{kind}] failed to load persistent best metadata {meta_path}: {exc}")
        return None


def _store_persistent_best_candidate(
    *,
    tune_key: str,
    kind: str,
    tag: str,
    mod: object,
    best_ms: float,
) -> None:

    module_name = getattr(mod, "__name__", None)
    if not module_name:
        return

    try:
        meta_path = _best_candidate_meta_path(kind, tune_key)
        tmp_path = meta_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "kind": kind,
                    "tune_key": tune_key,
                    "tag": tag,
                    "module_name": module_name,
                    "best_ms": float(best_ms),
                    "time": time.time(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(tmp_path, meta_path)
    except BaseException as exc:
        print(f"[JIT][{kind}] failed to store persistent best metadata: {exc}")



def _is_compiled_extension_file(path: Path, module_name: str) -> bool:
    if not path.is_file():
        return False
    name = path.name
    if not name.startswith(module_name):
        return False
    return any(name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)


def _prune_best_candidate_build_dir(
    *,
    build_dir: Path,
    module_name: str,
    kind: str,
    tag: str,
) -> None:
    """Keep only the generated source and compiled extension for the best candidate."""
    if not build_dir.exists() or not build_dir.is_dir():
        return

    keep_names = {f"{module_name}.cu"}
    for child in list(build_dir.iterdir()):
        try:
            if child.name in keep_names:
                continue
            if _is_compiled_extension_file(child, module_name):
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except BaseException as exc:
            print(f"[JIT][{kind}] failed to prune best candidate file {child}: {exc}")

    if _jit_cache_log_enabled():
        print(f"[JIT][{kind}] pruned best candidate build dir: {tag} -> {build_dir.name}")


def _keep_only_best_jit_candidate(
    *,
    tune_key: str,
    kind: str,
    tag: str,
    mod: object,
) -> None:
    """Delete non-best candidate build dirs for this tune_key.

    After runtime tuning, only the selected candidate is useful for future runs.
    Keeping only the best directory also makes the prebuilt fast path deterministic:
    future processes load the persisted best module directly and do not regenerate
    candidates, rerun scheduling, or recompile.
    """

    module_name = getattr(mod, "__name__", None)
    if not module_name:
        return

    prune_key = (str(kind), str(tune_key), str(module_name))
    if prune_key in _JIT_PRUNE_DONE_CACHE:
        return

    build_root = _default_build_root()
    best_dir = build_root / module_name
    prefix = f"{tune_key}_"

    removed = 0
    failed = 0
    if build_root.exists():
        for build_dir in list(build_root.iterdir()):
            if not build_dir.is_dir():
                continue
            if not build_dir.name.startswith(prefix):
                continue
            if build_dir.name == module_name:
                continue
            if _parse_candidate_from_module_name(tune_key, build_dir.name) is None:
                continue

            try:
                shutil.rmtree(build_dir, ignore_errors=False)
                removed += 1
            except BaseException as exc:
                failed += 1
                print(f"[JIT][{kind}] failed to remove non-best candidate {build_dir}: {exc}")

    _prune_best_candidate_build_dir(
        build_dir=best_dir,
        module_name=module_name,
        kind=kind,
        tag=tag,
    )

    _JIT_PRUNE_DONE_CACHE.add(prune_key)
    if _jit_cache_log_enabled():
        print(
            f"[JIT][{kind}] keep only best candidate for {tune_key}: "
            f"best={tag}/{module_name}, removed={removed}, failed={failed}"
        )


# -----------------------------------------------------------------------------
# Parallel candidate compilation
# -----------------------------------------------------------------------------

def _default_compile_mp_context_name() -> str:
    if os.name == "posix":
        return "fork"
    return "spawn"

def _get_candidate_compile_workers(kind: str, candidate_count: int) -> int:
    """Return process count for parallel JIT candidate compilation.

    ``candidate_count`` is the number of non-w1 warp variants waiting to be
    compiled.  The default intentionally caps by CPU count, while
    FASTEQ_UNIFORM1D_COMPILE_WORKERS / FASTEQ_JIT_COMPILE_WORKERS can override
    it for large build machines.
    """
    candidate_count = max(1, int(candidate_count))
    raw = os.environ.get("FASTEQ_UNIFORM1D_COMPILE_WORKERS") or os.environ.get("FASTEQ_JIT_COMPILE_WORKERS")
    if raw not in (None, ""):
        try:
            return max(1, min(candidate_count, int(raw)))
        except ValueError:
            pass

    cpu_count = os.cpu_count() or 1
    return max(1, min(candidate_count, int(cpu_count)))

def _compile_jit_candidate_worker(payload: Tuple[int, str, str, str, str]) -> Dict[str, Any]:
    """
    Child-process build entry.

    It returns only serializable metadata.  The compiled Python extension module
    object is intentionally not returned across process boundaries; the parent
    process loads the already-built module from the same build directory.
    """
    cand_idx, safe_tag, module_name, code, kind = payload
    try:
        # Parent should already set TORCH_CUDA_ARCH_LIST/PYTORCH_ROCM_ARCH.
        # Do not query runtime here: forked workers cannot safely reinitialize CUDA.
        _ensure_compile_arch_env(allow_runtime_query=False)

        # Prevent oversubscription: multiple candidate processes are already
        # running concurrently, so each ninja invocation should use few jobs by
        # default. Users can override this when the machine has enough RAM/cores.
        os.environ.setdefault("MAX_JOBS", os.environ.get("FASTEQ_UNIFORM1D_NINJA_JOBS", "1"))
        _load_jit_module(
            module_name=module_name,
            code=code,
            cache={},
        )
        return {
            "ok": True,
            "idx": int(cand_idx),
            "tag": safe_tag,
            "module_name": module_name,
            "kind": kind,
            "error": "",
        }
    except BaseException:
        return {
            "ok": False,
            "idx": int(cand_idx),
            "tag": safe_tag,
            "module_name": module_name,
            "kind": kind,
            "error": traceback.format_exc(),
        }


def _candidate_infos_from_sources(
    *,
    tune_key: str,
    raw_candidates: List[Tuple[str, str]],
) -> List[Tuple[int, str, str, str]]:
    infos: List[Tuple[int, str, str, str]] = []
    for cand_idx, (cand_tag, code) in enumerate(raw_candidates):
        safe_tag = _sanitize_module_tag(cand_tag or f"cand{cand_idx}")
        code_hash = _sha1_text(code)
        module_name = f"{tune_key}_{safe_tag}_{code_hash}"
        infos.append((cand_idx, safe_tag, module_name, code))
    return infos


def _build_jit_candidates_from_sources(
    *,
    tune_key: str,
    raw_candidates: List[Tuple[str, str]],
    cache: Dict[str, object],
    kind: str,
    u_dim: int,
    use_multiwarp_candidates: bool = True,
) -> List[Tuple[str, object]]:
    """Build scheduler candidates plus optional auto-warp variants.

    The build order is intentionally two-stage:
      1. Compile each scheduler candidate's w1 variant in the parent process.
         The w1 build gives compiler-reported registers/thread.
      2. When ``use_multiwarp_candidates`` is true, derive legal warps/block
         from the w1 register count and compile the remaining warp variants
         concurrently in child processes.  When false, return only w1.

    This function is shared by Uniform1D and STC.  STC passes one raw CUDA
    source, while Uniform1D may pass multiple scheduler/codegen candidates.
    """
    use_multiwarp_candidates = bool(use_multiwarp_candidates)
    base_infos = _candidate_infos_from_sources(
        tune_key=tune_key,
        raw_candidates=raw_candidates,
    )
    if not base_infos:
        return []

    modules: List[Tuple[str, object]] = []
    errors: List[Tuple[str, BaseException]] = []
    parallel_payloads: List[Tuple[int, str, str, str, str]] = []
    warp_size = _runtime_warp_size()

    # First pass: compile w1 variants serially so each source gets an accurate
    # register count.  This avoids guessing occupancy from source-level features.
    for cand_idx, safe_tag, _base_module_name, code in base_infos:
        try:
            code_w1 = _make_multiwarp_cuda_source(code, 1, warp_size=warp_size)
            tag_w1 = f"{safe_tag}_w1"
            module_name_w1 = f"{tune_key}_{tag_w1}_{_sha1_text(code_w1)}"
            mod_w1 = _build_jit_module_common(
                module_name=module_name_w1,
                cache=cache,
                kind=f"{kind}:{tag_w1}",
                codegen_fn=lambda code_w1=code_w1: code_w1,
            )
            modules.append((tag_w1, mod_w1))

            registers = getattr(mod_w1, "__fasteq_registers_per_thread__", None)
            if use_multiwarp_candidates:
                warp_candidates = _warp_candidates_from_registers(registers, u_dim=int(u_dim))
            else:
                warp_candidates = [1]
            print(
                f"[JIT][{kind}] candidate={safe_tag} regs32/thread={registers} "
                f"warp_size={warp_size} regs_per_sm={_runtime_register_file_regs_per_sm()} "
                f"u_tiles={(int(u_dim) + max(1, int(warp_size)) - 1) // max(1, int(warp_size))} "
                f"multiwarp={int(use_multiwarp_candidates)} "
                f"warp_candidates={warp_candidates}"
            )

            for warps in warp_candidates:
                if int(warps) == 1:
                    continue
                code_w = _make_multiwarp_cuda_source(code, int(warps), warp_size=warp_size)
                tag_w = f"{safe_tag}_w{int(warps)}"
                module_name_w = f"{tune_key}_{tag_w}_{_sha1_text(code_w)}"

                # Fast path for variants that were compiled in an earlier run.
                if module_name_w in cache:
                    modules.append((tag_w, cache[module_name_w]))
                    continue
                if _has_prebuilt_jit_module(module_name_w):
                    mod_w = _load_prebuilt_jit_module(
                        module_name=module_name_w,
                        cache=cache,
                        kind=f"{kind}:{tag_w}",
                    )
                    modules.append((tag_w, mod_w))
                    continue

                parallel_payloads.append((
                    len(parallel_payloads),
                    tag_w,
                    module_name_w,
                    code_w,
                    f"{kind}:{tag_w}",
                ))
        except BaseException as exc:
            errors.append((safe_tag, exc))
            print(f"[JIT][{kind}] candidate {safe_tag} failed to build w1/autowarp metadata: {exc}")

    # Second pass: compile non-w1 variants in parallel.  The worker writes the
    # extension to disk; the parent then imports the built module into its cache.
    if parallel_payloads:
        # Critical for forked compile workers: materialize CUDA/HIP arch flags in
        # the parent so torch cpp_extension does not call torch.cuda capability
        # APIs in a forked child process.
        _ensure_compile_arch_env(allow_runtime_query=True)

        parallel_enabled = _env_bool("FASTEQ_UNIFORM1D_PARALLEL_COMPILE", True)
        workers = _get_candidate_compile_workers(kind, len(parallel_payloads))
        print(
            f"[JIT][{kind}] parallel compile {len(parallel_payloads)} non-w1 "
            f"warp candidates with workers={workers}, enabled={int(parallel_enabled)}"
        )

        if not parallel_enabled or workers <= 1:
            results = [_compile_jit_candidate_worker(payload) for payload in parallel_payloads]
        else:
            ctx_name = _default_compile_mp_context_name()
            try:
                mp_ctx = mp.get_context(ctx_name)
            except ValueError:
                mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
                results = list(ex.map(_compile_jit_candidate_worker, parallel_payloads))

        for result in sorted(results, key=lambda item: int(item.get("idx", 0))):
            tag = str(result.get("tag", "cand"))
            module_name = str(result.get("module_name", ""))
            result_kind = str(result.get("kind", kind))
            if not result.get("ok", False):
                err = RuntimeError(str(result.get("error", "unknown compile failure")))
                errors.append((tag, err))
                print(f"[JIT][{kind}] candidate {tag} failed to build:\n{result.get('error', '')}")
                continue
            try:
                mod = _load_prebuilt_jit_module(
                    module_name=module_name,
                    cache=cache,
                    kind=result_kind,
                )
                modules.append((tag, mod))
            except BaseException as exc:
                errors.append((tag, exc))
                print(f"[JIT][{kind}] candidate {tag} built but failed to load: {exc}")

    if not modules:
        details = "; ".join(f"{tag}: {exc}" for tag, exc in errors)
        raise RuntimeError(f"all {kind.lower()} autowarp candidates failed to build: {details}")
    return modules

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
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    use_multiwarp_candidates: bool = True,
) -> str:
    # Include all code-affecting fields in the hash.  The previous version did
    # not include layout flags or coeff_list, which can accidentally reuse a
    # stale module when the same path structure is generated with different
    # x/y indirection, scatter mode, or constants.
    sig = repr((
        P, u_dim, dtype_str, mode, layout_tag,
        bool(use_multiwarp_candidates),
        iw_dim, ix_dim, ky_dim, v_dim,
        tuple(i_list), tuple(j_list), tuple(k_list), tuple(v_list),
        tuple(float(c) for c in coeff_list),
    ))
    h = _sha1_text(sig)
    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = _sanitize_module_tag(layout_tag)
    warp_tag = "autowarp" if use_multiwarp_candidates else "w1only"
    return f"uniform1d_fwd_{mode_str}_u{u_dim}_path{P}_{layout_tag}_{warp_tag}_jit_{dtype_str}_{h}"

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
    grad_x: bool,
    grad_y: bool,
    layout_tag: str = "dense",
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    use_multiwarp_candidates: bool = True,
) -> str:
    # Scheduler backward replaces the previous fused/split backward codegen.
    # Include layout and optional static dimensions in the signature so disk
    # cache cannot accidentally reuse a stale backward module generated by the
    # old split/combine paths or by a different scatter/source layout.
    sig = repr((
        "scheduler_bwd",
        P, u_dim, dtype_str, mode, grad_w, grad_x, grad_y, layout_tag,
        bool(use_multiwarp_candidates),
        iw_dim, ix_dim, ky_dim, v_dim,
        tuple(i_list), tuple(j_list), tuple(k_list), tuple(v_list),
        tuple(float(c) for c in coeff_list),
    ))
    h = _sha1_text(sig)
    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = _sanitize_module_tag(layout_tag)
    gw_tag = "gradw" if grad_w else "nogradw"
    gx_tag = "gradx" if grad_x else "nogradx"
    gy_tag = "grady" if grad_y else "nogrady"
    warp_tag = "autowarp" if use_multiwarp_candidates else "w1only"
    return f"uniform1d_bwd_sched_{mode_str}_u{u_dim}_path{P}_{layout_tag}_{gw_tag}_{gx_tag}_{gy_tag}_{warp_tag}_jit_{dtype_str}_{h}"


def _make_double_bwd_module_name(
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
    grad_x: bool,
    grad_y: bool,
    layout_tag: str = "dense",
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    use_multiwarp_candidates: bool = True,
) -> str:
    sig = repr((
        "scheduler_double_bwd_v1",
        P, u_dim, dtype_str, mode, grad_w, grad_x, grad_y, layout_tag,
        bool(use_multiwarp_candidates),
        iw_dim, ix_dim, ky_dim, v_dim,
        tuple(i_list), tuple(j_list), tuple(k_list), tuple(v_list),
        tuple(float(c) for c in coeff_list),
    ))
    h = _sha1_text(sig)
    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = _sanitize_module_tag(layout_tag)
    gw_tag = "gradw" if grad_w else "nogradw"
    gx_tag = "gradx" if grad_x else "nogradx"
    gy_tag = "grady" if grad_y else "nogrady"
    warp_tag = "autowarp" if use_multiwarp_candidates else "w1only"
    return (
        f"uniform1d_double_bwd_sched_{mode_str}_u{u_dim}_path{P}_"
        f"{layout_tag}_{gw_tag}_{gx_tag}_{gy_tag}_{warp_tag}_jit_{dtype_str}_{h}"
    )


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
        if _jit_cache_log_enabled():
            print(f"[JIT][{kind}] hit in-memory cache: {module_name}")
        return cache[module_name]

    if _jit_cache_log_enabled():
        print(f"[JIT][{kind}] build/check module: {module_name}")

    build_root = _default_build_root()
    build_dir = build_root / module_name
    _ensure_dir(build_dir)

    cu_path = build_dir / f"{module_name}.cu"

    # Step 2: Check the disk cache.
    # If the CUDA source file already exists, assume the module name uniquely
    # identifies the generated code and load it directly.
    if cu_path.exists():
        # If both source and compiled extension exist, this becomes a pure
        # import.  If only source exists, _load_jit_module may invoke the
        # compiler to finish an incomplete build.
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
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str,
    dtype_str: str,
    use_multiwarp_candidates: Optional[bool] = None,
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
    use_multiwarp_candidates = _resolve_multiwarp_candidates_enabled(
        use_multiwarp_candidates
    )

    fast_key = _make_candidate_fast_key(
        kind="FWD",
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
        use_multiwarp_candidates=use_multiwarp_candidates,
    )
    cached_tune_key = _FWD_TUNE_KEY_FAST_CACHE.get(fast_key)
    if cached_tune_key is not None:
        cached_best = _FWD_BEST_CANDIDATE_CACHE.get(cached_tune_key)
        if cached_best is not None:
            best_tag, best_mod, _best_ms = cached_best
            return cached_tune_key, [(best_tag, best_mod)]

        persistent_best = _load_persistent_best_candidate(
            tune_key=cached_tune_key,
            cache=_FWD_JIT_CACHE,
            kind="FWD",
        )
        if persistent_best is not None:
            best_tag, best_mod, best_ms = persistent_best
            _FWD_BEST_CANDIDATE_CACHE[cached_tune_key] = (best_tag, best_mod, best_ms)
            return cached_tune_key, [(best_tag, best_mod)]

        prebuilt_modules = _discover_prebuilt_jit_candidates(
            tune_key=cached_tune_key,
            cache=_FWD_JIT_CACHE,
            kind="FWD",
        )
        if prebuilt_modules:
            return cached_tune_key, prebuilt_modules

        # Fast key is known but no in-memory/disk module was found.  Reuse the
        # cached tune_key and fall through to codegen without rebuilding it from
        # CPU lists.
        tune_key = cached_tune_key
    else:
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
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
            use_multiwarp_candidates=use_multiwarp_candidates,
        )
        _FWD_TUNE_KEY_FAST_CACHE[fast_key] = tune_key

    if tune_key in _FWD_BEST_CANDIDATE_CACHE:
        best_tag, best_mod, best_ms = _FWD_BEST_CANDIDATE_CACHE[tune_key]
        if _jit_cache_log_enabled():
            print(f"[JIT][FWD] hit best candidate cache: {tune_key} -> {best_tag} ({best_ms:.4f} ms)")
        return tune_key, [(best_tag, best_mod)]

    persistent_best = _load_persistent_best_candidate(
        tune_key=tune_key,
        cache=_FWD_JIT_CACHE,
        kind="FWD",
    )
    if persistent_best is not None:
        best_tag, best_mod, best_ms = persistent_best
        _FWD_BEST_CANDIDATE_CACHE[tune_key] = (best_tag, best_mod, best_ms)
        return tune_key, [(best_tag, best_mod)]

    prebuilt_modules = _discover_prebuilt_jit_candidates(
        tune_key=tune_key,
        cache=_FWD_JIT_CACHE,
        kind="FWD",
    )
    if prebuilt_modules:
        return tune_key, prebuilt_modules

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
        )

        """ return generate_code_uniform1d_fwd_baseline_unrolled(
            i_list, j_list, k_list, v_list, coeff_list,
            input_indices=input_indices,
            output_indices=output_indices,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
            mode=mode,
            unroll_size=min(256,len(i_list)),
            out_path="generated_uniform1d_fwd_baseline_unrolled.cu",
            path_semantics="wxy",
        ) """
    print(f"[JIT][FWD] generate candidates for: {tune_key}")
    codegen_out = _codegen_candidates()
    raw_candidates = _normalize_codegen_candidates(codegen_out)
    if not raw_candidates:
        raise RuntimeError("forward codegen returned zero candidates")


    modules = _build_jit_candidates_from_sources(
        tune_key=tune_key,
        raw_candidates=raw_candidates,
        cache=_FWD_JIT_CACHE,
        kind="FWD",
        u_dim=int(u_dim),
        use_multiwarp_candidates=use_multiwarp_candidates,
    )

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
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str,
    dtype_str: str,
    use_multiwarp_candidates: Optional[bool] = None,
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
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        mode=mode,
        dtype_str=dtype_str,
        use_multiwarp_candidates=use_multiwarp_candidates,
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
    grad_x: bool,
    grad_y: bool,
    use_multiwarp_candidates: Optional[bool] = None,
) -> Tuple[str, List[Tuple[str, object]]]:
    """
    Generate/build all backward scheduler candidates for one Uniform1D signature.

    Returns:
        tune_key, [(candidate_tag, loaded_module), ...]

    If a best candidate has already been tuned in this process, only that
    selected module is returned.
    """
    input_indices = {} if input_indices is None else input_indices
    output_indices = {} if output_indices is None else output_indices
    use_multiwarp_candidates = _resolve_multiwarp_candidates_enabled(
        use_multiwarp_candidates
    )

    fast_key = _make_candidate_fast_key(
        kind="BWD",
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
        grad_x=grad_x,
        grad_y=grad_y,
        use_multiwarp_candidates=use_multiwarp_candidates,
    )
    cached_tune_key = _BWD_TUNE_KEY_FAST_CACHE.get(fast_key)
    if cached_tune_key is not None:
        cached_best = _BWD_BEST_CANDIDATE_CACHE.get(cached_tune_key)
        if cached_best is not None:
            best_tag, best_mod, _best_ms = cached_best
            return cached_tune_key, [(best_tag, best_mod)]

        persistent_best = _load_persistent_best_candidate(
            tune_key=cached_tune_key,
            cache=_BWD_JIT_CACHE,
            kind="BWD",
        )
        if persistent_best is not None:
            best_tag, best_mod, best_ms = persistent_best
            _BWD_BEST_CANDIDATE_CACHE[cached_tune_key] = (best_tag, best_mod, best_ms)
            return cached_tune_key, [(best_tag, best_mod)]

        prebuilt_modules = _discover_prebuilt_jit_candidates(
            tune_key=cached_tune_key,
            cache=_BWD_JIT_CACHE,
            kind="BWD",
        )
        if prebuilt_modules:
            return cached_tune_key, prebuilt_modules

        tune_key = cached_tune_key
    else:
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
            grad_x=grad_x,
            grad_y=grad_y,
            layout_tag=layout_tag,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
            use_multiwarp_candidates=use_multiwarp_candidates,
        )
        _BWD_TUNE_KEY_FAST_CACHE[fast_key] = tune_key

    if tune_key in _BWD_BEST_CANDIDATE_CACHE:
        best_tag, best_mod, best_ms = _BWD_BEST_CANDIDATE_CACHE[tune_key]
        if _jit_cache_log_enabled():
            print(f"[JIT][BWD] hit best candidate cache: {tune_key} -> {best_tag} ({best_ms:.4f} ms)")
        return tune_key, [(best_tag, best_mod)]

    persistent_best = _load_persistent_best_candidate(
        tune_key=tune_key,
        cache=_BWD_JIT_CACHE,
        kind="BWD",
    )
    if persistent_best is not None:
        best_tag, best_mod, best_ms = persistent_best
        _BWD_BEST_CANDIDATE_CACHE[tune_key] = (best_tag, best_mod, best_ms)
        return tune_key, [(best_tag, best_mod)]

    prebuilt_modules = _discover_prebuilt_jit_candidates(
        tune_key=tune_key,
        cache=_BWD_JIT_CACHE,
        kind="BWD",
    )
    if prebuilt_modules:
        return tune_key, prebuilt_modules

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
            need_grad_x=grad_x,
            need_grad_y=grad_y,
        )

        """ return generate_code_uniform1d_bwd_baseline_unrolled(
            i_list, j_list, k_list, v_list, coeff_list,
            input_indices=input_indices,
            output_indices=output_indices,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
            mode=mode,
            need_grad_w=grad_w,
            unroll_size=min(256,len(i_list)),
            out_path="generated_uniform1d_bwd_baseline_unrolled.cu",
            path_semantics="wxy",
        ) """

    print(f"[JIT][BWD] generate candidates for: {tune_key}")
    raw_candidates = _normalize_codegen_candidates(_codegen_candidates())
    if not raw_candidates:
        raise RuntimeError("backward codegen returned zero candidates")

    modules = _build_jit_candidates_from_sources(
        tune_key=tune_key,
        raw_candidates=raw_candidates,
        cache=_BWD_JIT_CACHE,
        kind="BWD",
        u_dim=int(u_dim),
        use_multiwarp_candidates=use_multiwarp_candidates,
    )

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
    grad_x: bool,
    grad_y: bool,
    use_multiwarp_candidates: Optional[bool] = None,
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
        grad_x=grad_x,
        grad_y=grad_y,
        use_multiwarp_candidates=use_multiwarp_candidates,
    )
    return modules[0][1]


def _build_double_bwd_jit_candidates(
    *,
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int,
    iw_dim: Optional[int],
    ix_dim: Optional[int],
    ky_dim: Optional[int],
    v_dim: Optional[int],
    mode: str,
    dtype_str: str,
    grad_w: bool,
    grad_x: bool,
    grad_y: bool,
    use_multiwarp_candidates: Optional[bool] = None,
) -> Tuple[str, List[Tuple[str, object]]]:
    input_indices = {} if input_indices is None else input_indices
    output_indices = {} if output_indices is None else output_indices
    use_multiwarp_candidates = _resolve_multiwarp_candidates_enabled(
        use_multiwarp_candidates
    )

    fast_key = _make_candidate_fast_key(
        kind="DOUBLE_BWD",
        i_list=i_list, j_list=j_list, k_list=k_list, v_list=v_list,
        coeff_list=coeff_list,
        input_indices=input_indices, output_indices=output_indices,
        u_dim=u_dim, iw_dim=iw_dim, ix_dim=ix_dim, ky_dim=ky_dim, v_dim=v_dim,
        mode=mode, dtype_str=dtype_str, grad_w=grad_w, grad_x=grad_x,
        grad_y=grad_y,
        use_multiwarp_candidates=use_multiwarp_candidates,
    )
    tune_key = _DOUBLE_BWD_TUNE_KEY_FAST_CACHE.get(fast_key)
    if tune_key is None:
        i_cpu = _tensor_to_cpu_list(i_list)
        j_cpu = _tensor_to_cpu_list(j_list)
        k_cpu = _tensor_to_cpu_list(k_list)
        v_cpu = _tensor_to_cpu_list(v_list)
        coeff_cpu = _tensor_to_cpu_list(coeff_list)
        use_x_src = 1 in input_indices
        use_y_src = 2 in input_indices
        use_scatter = 0 in output_indices
        layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"
        tune_key = _make_double_bwd_module_name(
            P=len(i_cpu), u_dim=u_dim, mode=mode,
            i_list=i_cpu, j_list=j_cpu, k_list=k_cpu, v_list=v_cpu,
            coeff_list=coeff_cpu, dtype_str=dtype_str, grad_w=grad_w, grad_x=grad_x,
            grad_y=grad_y,
            layout_tag=layout_tag, iw_dim=iw_dim, ix_dim=ix_dim,
            ky_dim=ky_dim, v_dim=v_dim,
            use_multiwarp_candidates=use_multiwarp_candidates,
        )
        _DOUBLE_BWD_TUNE_KEY_FAST_CACHE[fast_key] = tune_key

    cached_best = _DOUBLE_BWD_BEST_CANDIDATE_CACHE.get(tune_key)
    if cached_best is not None:
        return tune_key, [(cached_best[0], cached_best[1])]

    persistent_best = _load_persistent_best_candidate(
        tune_key=tune_key,
        cache=_DOUBLE_BWD_JIT_CACHE,
        kind="DOUBLE_BWD",
    )
    if persistent_best is not None:
        tag, mod, ms = persistent_best
        _DOUBLE_BWD_BEST_CANDIDATE_CACHE[tune_key] = (tag, mod, ms)
        return tune_key, [(tag, mod)]

    prebuilt = _discover_prebuilt_jit_candidates(
        tune_key=tune_key,
        cache=_DOUBLE_BWD_JIT_CACHE,
        kind="DOUBLE_BWD",
    )
    if prebuilt:
        return tune_key, prebuilt

    print(f"[JIT][DOUBLE_BWD] generate candidates for: {tune_key}")
    codegen_out = generate_code_uniform1d_double_bwd_with_scheduler(
        i_list=i_list, j_list=j_list, k_list=k_list, v_list=v_list,
        coeff_list=coeff_list,
        input_indices=input_indices, output_indices=output_indices,
        u_dim=u_dim, iw_dim=iw_dim, ix_dim=ix_dim, ky_dim=ky_dim, v_dim=v_dim,
        mode=mode, need_grad_w=grad_w, need_grad_x=grad_x, need_grad_y=grad_y,
        out_path="", profile=False, profile_print=False,
    )
    raw_candidates = _normalize_codegen_candidates(codegen_out)
    if not raw_candidates:
        raise RuntimeError("double backward codegen returned zero candidates")
    modules = _build_jit_candidates_from_sources(
        tune_key=tune_key,
        raw_candidates=raw_candidates,
        cache=_DOUBLE_BWD_JIT_CACHE,
        kind="DOUBLE_BWD",
        u_dim=int(u_dim),
        use_multiwarp_candidates=use_multiwarp_candidates,
    )
    return tune_key, modules


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
    out_seg_num,
    fused_scatter: bool,
):
    if fused_scatter:
        return mod.run(w, x, y, src_idx, dst_idx, out_seg_num)
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
    out_seg_num,
    fused_scatter: bool,
):
    if fused_scatter:
        return mod.run(w, x, y, grad_out, src_idx, dst_idx, out_seg_num)
    return mod.run(w, x, y, grad_out, src_idx, out_seg_num)





def _call_double_bwd_module(
    mod,
    *,
    w,
    x,
    y,
    grad_out,
    grad_grad_w,
    grad_grad_x,
    grad_grad_y,
    src_idx,
    dst_idx,
    out_seg_num,
    use_src: bool,
    fused_scatter: bool,
    need_grad_w: bool,
    need_grad_x: bool,
    need_grad_y: bool,
):
    args = [w, x, y, grad_out]
    if need_grad_w:
        args.append(grad_grad_w)
    if need_grad_x:
        args.append(grad_grad_x)
    if need_grad_y:
        args.append(grad_grad_y)
    if use_src:
        args.append(src_idx)
    if fused_scatter:
        args.append(dst_idx)
    args.append(int(out_seg_num))
    return mod.run(*args)


def _benchmark_and_select_best_jit_candidate(
    *,
    tune_key: str,
    candidates: List[Tuple[str, object]],
    best_cache: Dict[str, Tuple[str, object, float]],
    kind: str,
    tune_enabled: bool,
    warmup: int,
    repeat: int,
    call_fn,
) -> Tuple[str, object, float]:
    """Benchmark compiled candidates and persist/prune the selected one.

    ``call_fn`` receives a module and launches its ``run`` entry with the
    operator-specific ABI.  Everything else is operator-agnostic and is reused
    by Uniform1D FWD/BWD and STC FWD/BWD.
    """
    if tune_key in best_cache:
        return best_cache[tune_key]
    if not candidates:
        raise RuntimeError(f"no {kind.lower()} candidates to benchmark")

    if len(candidates) == 1 or not tune_enabled:
        tag, mod = candidates[0]
        best = (tag, mod, float("nan"))
        best_cache[tune_key] = best
        _store_persistent_best_candidate(
            tune_key=tune_key,
            kind=kind,
            tag=tag,
            mod=mod,
            best_ms=float("nan"),
        )
        _keep_only_best_jit_candidate(
            tune_key=tune_key,
            kind=kind,
            tag=tag,
            mod=mod,
        )
        if len(candidates) == 1:
            print(f"[JIT][{kind}][tune] only one candidate: {tag}")
        else:
            print(f"[JIT][{kind}][tune] disabled, use first candidate: {tag}")
        return best

    print(
        f"[JIT][{kind}][tune] benchmarking {len(candidates)} candidates "
        f"for {tune_key}, warmup={warmup}, repeat={repeat}"
    )

    timings: List[Tuple[float, str, object]] = []
    for tag, mod in candidates:
        try:
            for _ in range(warmup):
                call_fn(mod)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(repeat):
                call_fn(mod)
            torch.cuda.synchronize()
            avg_ms = (time.perf_counter() - t0) * 1000.0 / float(repeat)
            timings.append((avg_ms, tag, mod))
            regs = getattr(mod, "__fasteq_registers_per_thread__", None)
            print(f"[JIT][{kind}][tune] candidate={tag:>16s} regs={regs} avg={avg_ms:.4f} ms")
        except BaseException as exc:
            print(f"[JIT][{kind}][tune] candidate={tag} failed at runtime: {exc}")

    if not timings:
        raise RuntimeError(f"all {kind.lower()} candidates failed during runtime benchmark")

    timings.sort(key=lambda x: x[0])
    best_ms, best_tag, best_mod = timings[0]
    best = (best_tag, best_mod, best_ms)
    best_cache[tune_key] = best
    _store_persistent_best_candidate(
        tune_key=tune_key,
        kind=kind,
        tag=best_tag,
        mod=best_mod,
        best_ms=best_ms,
    )
    _keep_only_best_jit_candidate(
        tune_key=tune_key,
        kind=kind,
        tag=best_tag,
        mod=best_mod,
    )
    print(f"[JIT][{kind}][tune] selected candidate={best_tag} avg={best_ms:.4f} ms")
    return best

def _select_best_fwd_module(
    *,
    tune_key: str,
    candidates: List[Tuple[str, object]],
    w,
    x,
    y,
    src_idx,
    dst_idx,
    out_seg_num,
    fused_scatter: bool,
) -> Tuple[str, object, float]:
    tune_enabled, warmup, repeat = _get_fwd_tune_params()
    return _benchmark_and_select_best_jit_candidate(
        tune_key=tune_key,
        candidates=candidates,
        best_cache=_FWD_BEST_CANDIDATE_CACHE,
        kind="FWD",
        tune_enabled=tune_enabled,
        warmup=warmup,
        repeat=repeat,
        call_fn=lambda mod: _call_fwd_module(
            mod,
            w=w,
            x=x,
            y=y,
            src_idx=src_idx,
            dst_idx=dst_idx,
            out_seg_num=out_seg_num,
            fused_scatter=fused_scatter,
        ),
    )

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
    out_seg_num,
    fused_scatter: bool,
) -> Tuple[str, object, float]:
    tune_enabled, warmup, repeat = _get_bwd_tune_params()
    return _benchmark_and_select_best_jit_candidate(
        tune_key=tune_key,
        candidates=candidates,
        best_cache=_BWD_BEST_CANDIDATE_CACHE,
        kind="BWD",
        tune_enabled=tune_enabled,
        warmup=warmup,
        repeat=repeat,
        call_fn=lambda mod: _call_bwd_module(
            mod,
            w=w,
            x=x,
            y=y,
            grad_out=grad_out,
            src_idx=src_idx,
            dst_idx=dst_idx,
            out_seg_num=out_seg_num,
            fused_scatter=fused_scatter,
        ),
    )


def _select_best_double_bwd_module(
    *,
    tune_key: str,
    candidates: List[Tuple[str, object]],
    w, x, y, grad_out,
    grad_grad_w, grad_grad_x, grad_grad_y,
    src_idx, dst_idx,
    out_seg_num,
    use_src: bool,
    fused_scatter: bool,
    need_grad_w: bool,
    need_grad_x: bool,
    need_grad_y: bool,
) -> Tuple[str, object, float]:
    # Double backward uses its own env controls when present and otherwise
    # inherits the regular backward tuning policy.
    tune_enabled = _env_bool(
        "FASTEQ_UNIFORM1D_DOUBLE_BWD_TUNE",
        _DEFAULT_UNIFORM1D_BWD_TUNE_ENABLED,
    )
    warmup = _env_int(
        "FASTEQ_UNIFORM1D_DOUBLE_BWD_TUNE_WARMUP",
        _DEFAULT_UNIFORM1D_TUNE_WARMUP,
        min_value=0,
    )
    repeat = _env_int(
        "FASTEQ_UNIFORM1D_DOUBLE_BWD_TUNE_REPEAT",
        _DEFAULT_UNIFORM1D_TUNE_REPEAT,
        min_value=1,
    )
    return _benchmark_and_select_best_jit_candidate(
        tune_key=tune_key,
        candidates=candidates,
        best_cache=_DOUBLE_BWD_BEST_CANDIDATE_CACHE,
        kind="DOUBLE_BWD",
        tune_enabled=tune_enabled,
        warmup=warmup,
        repeat=repeat,
        call_fn=lambda mod: _call_double_bwd_module(
            mod,
            w=w, x=x, y=y, grad_out=grad_out,
            grad_grad_w=grad_grad_w,
            grad_grad_x=grad_grad_x,
            grad_grad_y=grad_grad_y,
            src_idx=src_idx, dst_idx=dst_idx,
            out_seg_num=out_seg_num,
            use_src=use_src,
            fused_scatter=fused_scatter,
            need_grad_w=need_grad_w,
            need_grad_x=need_grad_x,
            need_grad_y=need_grad_y,
        ),
    )

def _run_fwd(
    *,
    w,
    x,
    y,
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
    use_multiwarp_candidates: Optional[bool] = None,
):

    use_multiwarp_candidates = _resolve_multiwarp_candidates_enabled(
        use_multiwarp_candidates
    )
    dtype_str = _get_scalar_t_str(w)
    iw_dim = int(w.size(1))
    ix_dim = int(x.size(1))
    ky_dim = int(y.size(1))
    v_dim = int(out_seg_num)

    tune_key, candidates = _build_fwd_jit_candidates(
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
        use_multiwarp_candidates=use_multiwarp_candidates,
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
        dst_idx = output_indices[0].to(torch.int32)
    else:
        dst_idx = None

    best_tag, best_mod, best_ms = _select_best_fwd_module(
        tune_key=tune_key,
        candidates=candidates,
        w=w, x=x, y=y,
        src_idx=src_idx, dst_idx=dst_idx,
        out_seg_num=out_seg_num,
        fused_scatter=fused_scatter,
    )

    out = _call_fwd_module(
        best_mod,
        w=w, x=x, y=y,
        src_idx=src_idx, dst_idx=dst_idx,
        out_seg_num=out_seg_num,
        fused_scatter=fused_scatter,
    )
    return out

        

def _run_bwd(
    *,
    grad_out,
    w,
    x,
    y,
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
    grad_x,
    grad_y,
    use_multiwarp_candidates: Optional[bool] = None,
):

    use_multiwarp_candidates = _resolve_multiwarp_candidates_enabled(
        use_multiwarp_candidates
    )
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
        grad_x=grad_x,
        grad_y=grad_y,
        use_multiwarp_candidates=use_multiwarp_candidates,
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
        dst_idx = output_indices[0].to(torch.int32)
    else:
        dst_idx = None

    best_tag, best_mod, best_ms = _select_best_bwd_module(
        tune_key=tune_key,
        candidates=candidates,
        w=w, x=x, y=y, grad_out=grad_out,
        src_idx=src_idx, dst_idx=dst_idx,
        out_seg_num=out_seg_num,
        fused_scatter=fused_scatter,
    )

    out =  _call_bwd_module(
        best_mod,
        w=w, x=x, y=y, grad_out=grad_out,
        src_idx=src_idx, dst_idx=dst_idx,
        out_seg_num=out_seg_num,
        fused_scatter=fused_scatter,
    )
    
    return out


def _run_double_bwd(
    *,
    grad_out,
    w,
    x,
    y,
    grad_grad_w,
    grad_grad_x,
    grad_grad_y,
    i_list,
    j_list,
    k_list,
    v_list,
    coeff_list,
    input_indices,
    output_indices,
    out_seg_num,
    u_dim,
    iw_dim,
    ix_dim,
    ky_dim,
    v_dim,
    mode,
    grad_w,
    grad_x,
    grad_y,
    use_multiwarp_candidates: Optional[bool] = None,
):
    use_multiwarp_candidates = _resolve_multiwarp_candidates_enabled(
        use_multiwarp_candidates
    )
    dtype_str = _get_scalar_t_str(w)
    tune_key, candidates = _build_double_bwd_jit_candidates(
        i_list=i_list, j_list=j_list, k_list=k_list, v_list=v_list,
        coeff_list=coeff_list,
        input_indices=input_indices, output_indices=output_indices,
        u_dim=u_dim, iw_dim=iw_dim, ix_dim=ix_dim, ky_dim=ky_dim, v_dim=v_dim,
        mode=mode, dtype_str=dtype_str, grad_w=bool(grad_w),
        grad_x=bool(grad_x), grad_y=bool(grad_y),
        use_multiwarp_candidates=use_multiwarp_candidates,
    )

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_src = use_x_src or use_y_src
    fused_scatter = 0 in output_indices
    if use_x_src:
        src_idx = _as_int32_meta_tensor(input_indices[1]).contiguous()
    elif use_y_src:
        src_idx = _as_int32_meta_tensor(input_indices[2]).contiguous()
    else:
        src_idx = None
    dst_idx = (
        _as_int32_meta_tensor(output_indices[0]).contiguous()
        if fused_scatter else None
    )

    grad_out = grad_out.view(-1, out_seg_num, u_dim).contiguous()
    # None means that the outer loss did not consume this first-gradient output.
    # Materializing zero cotangents preserves the exact VJP while keeping all
    # mathematical double-backward work inside the generated CUDA/HIP kernel.
    if bool(grad_w):
        if grad_grad_w is None:
            grad_grad_w = torch.zeros_like(w)
        else:
            grad_grad_w = grad_grad_w.contiguous()
    if bool(grad_x):
        if grad_grad_x is None:
            grad_grad_x = torch.zeros_like(x)
        else:
            grad_grad_x = grad_grad_x.contiguous()
    if bool(grad_y):
        if grad_grad_y is None:
            grad_grad_y = torch.zeros_like(y)
        else:
            grad_grad_y = grad_grad_y.contiguous()

    _tag, mod, _ms = _select_best_double_bwd_module(
        tune_key=tune_key, candidates=candidates,
        w=w, x=x, y=y, grad_out=grad_out,
        grad_grad_w=grad_grad_w,
        grad_grad_x=grad_grad_x,
        grad_grad_y=grad_grad_y,
        src_idx=src_idx, dst_idx=dst_idx,
        out_seg_num=out_seg_num,
        use_src=use_src,
        fused_scatter=fused_scatter,
        need_grad_w=bool(grad_w),
        need_grad_x=bool(grad_x),
        need_grad_y=bool(grad_y),
    )
    return _call_double_bwd_module(
        mod,
        w=w, x=x, y=y, grad_out=grad_out,
        grad_grad_w=grad_grad_w,
        grad_grad_x=grad_grad_x,
        grad_grad_y=grad_grad_y,
        src_idx=src_idx, dst_idx=dst_idx,
        out_seg_num=out_seg_num,
        use_src=use_src,
        fused_scatter=fused_scatter,
        need_grad_w=bool(grad_w),
        need_grad_x=bool(grad_x),
        need_grad_y=bool(grad_y),
    )

@torch.no_grad()
def _print_index_reuse_stats(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    print_details: bool = True,
) -> None:
    """
    统计路径索引的重复情况。

    指标定义：
      total:
          索引总出现次数，即路径数量 P。

      unique:
          不同索引值的数量。

      repeated_values:
          出现次数大于 1 的不同索引值数量。

      repeated_occurrences:
          所有重复索引值对应的总出现次数。
          例如 [0, 0, 0, 1, 2, 2] 中为 3 + 2 = 5。

      reusable:
          除第一次出现之外的重复出现次数，即理论可复用次数：
              sum(count - 1) = total - unique

      reuse_ratio:
          reusable / total。

      adjacent_reuse:
          当前调度顺序下，相邻路径使用相同索引的次数。
          该指标比全局 reusable 更能反映直接连续复用。
    """

    index_lists = {
        "i": i_list,
        "j": j_list,
        "k": k_list,
        "v": v_list,
    }

    print("\n[index reuse statistics]")
    print(
        f"{'index':<8}"
        f"{'total':>10}"
        f"{'unique':>10}"
        f"{'repeat val':>12}"
        f"{'repeat occ':>12}"
        f"{'reusable':>12}"
        f"{'reuse ratio':>14}"
        f"{'adjacent':>12}"
    )

    for name, tensor in index_lists.items():
        # 调试统计放到 CPU 上执行，避免后续 Python 格式化处理 GPU Tensor。
        values_cpu = tensor.detach().reshape(-1).to(
            device="cpu",
            dtype=torch.int64,
        )

        total = values_cpu.numel()

        if total == 0:
            print(
                f"{name:<8}"
                f"{0:>10}"
                f"{0:>10}"
                f"{0:>12}"
                f"{0:>12}"
                f"{0:>12}"
                f"{0.0:>13.2%}"
                f"{0:>12}"
            )
            continue

        unique_values, counts = torch.unique(
            values_cpu,
            sorted=True,
            return_counts=True,
        )

        repeated_mask = counts > 1

        unique_count = unique_values.numel()
        repeated_value_count = int(repeated_mask.sum().item())

        repeated_occurrences = int(
            counts[repeated_mask].sum().item()
        )

        # 每个索引第一次出现不算复用，后续出现均算潜在复用。
        reusable_counts = torch.clamp(counts - 1, min=0)
        reusable_count = int(reusable_counts.sum().item())

        reuse_ratio = reusable_count / total

        # 当前路径顺序中，相邻两条路径使用同一索引的次数。
        adjacent_reuse_count = int(
            (values_cpu[1:] == values_cpu[:-1]).sum().item()
        )

        print(
            f"{name:<8}"
            f"{total:>10}"
            f"{unique_count:>10}"
            f"{repeated_value_count:>12}"
            f"{repeated_occurrences:>12}"
            f"{reusable_count:>12}"
            f"{reuse_ratio:>13.2%}"
            f"{adjacent_reuse_count:>12}"
        )

        if print_details and repeated_value_count > 0:
            repeated_values = unique_values[repeated_mask]
            repeated_counts = counts[repeated_mask]

            details = [
                (
                    int(value.item()),
                    int(count.item()),
                    int(count.item()) - 1,
                )
                for value, count in zip(
                    repeated_values,
                    repeated_counts,
                )
            ]

            # 优先展示出现次数最多的索引。
            details.sort(key=lambda item: (-item[1], item[0]))

            detail_text = ", ".join(
                f"{name}[{value}]: count={count}, reuse={reuse}"
                for value, count, reuse in details
            )
            print(f"  repeated {name}: {detail_text}")

    print()

from datetime import datetime
from pathlib import Path
def dump_tensor(
    tensor: torch.Tensor,
    file_path: str,
    *,
    to_cpu: bool = True,
) -> None:
    """
    将 Tensor 保存到磁盘。

    Args:
        tensor: 要保存的 Tensor。
        file_path: 输出路径，例如 "./dump/stc_d_x0.pt"。
        to_cpu: 是否先转移到 CPU，建议开启。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(file_path) / timestamp
    path.parent.mkdir(parents=True, exist_ok=True)

    value = tensor.detach()
    if to_cpu:
        value = value.cpu()

    # clone 避免底层存储包含无关数据
    value = value.contiguous().clone()
    torch.save(value, path)

    print(
        f"Dumped tensor: {path}, "
        f"shape={tuple(value.shape)}, "
        f"dtype={value.dtype}"
    )

# -----------------------------------------------------------------------------
# Differentiable first backward / generated CUDA double backward
# -----------------------------------------------------------------------------

class FastUniform1dBackwardFunction(torch.autograd.Function):
    """First backward uses JIT BWD; its backward uses JIT DOUBLE_BWD."""

    @staticmethod
    def forward(ctx, grad_out, w, x, y, i_list, j_list, k_list, v_list,
                coeff_list, input_indices, output_indices, out_seg_num, u_dim,
                iw_dim, ix_dim, ky_dim, v_dim, mode, need_grad_w,
                need_grad_x,
                need_grad_y,
                use_multiwarp_candidates):
        result = _run_bwd(
            grad_out=grad_out, w=w, x=x, y=y,
            i_list=i_list, j_list=j_list, k_list=k_list, v_list=v_list,
            coeff_list=coeff_list, input_indices=input_indices,
            output_indices=output_indices, out_seg_num=int(out_seg_num),
            u_dim=int(u_dim), iw_dim=int(iw_dim), ix_dim=int(ix_dim),
            ky_dim=int(ky_dim), v_dim=int(v_dim), mode=mode,
            grad_w=bool(need_grad_w),
            grad_x=bool(need_grad_x),
            grad_y=bool(need_grad_y),
            use_multiwarp_candidates=bool(use_multiwarp_candidates),
        )
        result_iter = iter(result)
        gw = next(result_iter) if bool(need_grad_w) else w.new_empty((0,))
        gx = next(result_iter) if bool(need_grad_x) else x.new_empty((0,))
        gy = next(result_iter) if bool(need_grad_y) else y.new_empty((0,))

        ctx.save_for_backward(
            grad_out, w, x, y, i_list, j_list, k_list, v_list, coeff_list
        )
        ctx.input_indices = input_indices
        ctx.output_indices = output_indices
        ctx.out_seg_num = int(out_seg_num)
        ctx.u_dim = int(u_dim)
        ctx.iw_dim = int(iw_dim)
        ctx.ix_dim = int(ix_dim)
        ctx.ky_dim = int(ky_dim)
        ctx.v_dim = int(v_dim)
        ctx.mode = mode
        ctx.need_grad_w = bool(need_grad_w)
        ctx.need_grad_x = bool(need_grad_x)
        ctx.need_grad_y = bool(need_grad_y)
        ctx.use_multiwarp_candidates = bool(use_multiwarp_candidates)
        print(f"fasteq uniform1d_jit gw shape:{gw.shape}: {gw.sum()}")
        if ctx.need_grad_x:
            print(f"fasteq uniform1d_jit gx shape:{gx.shape}: {gx.sum()}")
        if ctx.need_grad_y:
            print(f"fasteq uniform1d_jit gy shape:{gy.shape}: {gy.sum()}")
        return gw, gx, gy

    @staticmethod
    def backward(ctx, grad_grad_w, grad_grad_x, grad_grad_y):
        grad_out, w, x, y, i_list, j_list, k_list, v_list, coeff_list = ctx.saved_tensors
        result = _run_double_bwd(
            grad_out=grad_out,
            w=w, x=x, y=y,
            grad_grad_w=grad_grad_w if ctx.need_grad_w else None,
            grad_grad_x=grad_grad_x,
            grad_grad_y=grad_grad_y,
            i_list=i_list, j_list=j_list, k_list=k_list, v_list=v_list,
            coeff_list=coeff_list,
            input_indices=ctx.input_indices,
            output_indices=ctx.output_indices,
            out_seg_num=ctx.out_seg_num,
            u_dim=ctx.u_dim,
            iw_dim=ctx.iw_dim,
            ix_dim=ctx.ix_dim,
            ky_dim=ctx.ky_dim,
            v_dim=ctx.v_dim,
            mode=ctx.mode,
            grad_w=ctx.need_grad_w,
            grad_x=ctx.need_grad_x,
            grad_y=ctx.need_grad_y,
            use_multiwarp_candidates=ctx.use_multiwarp_candidates,
        )
        """ print(f"uniform1d_jit d_go shape:{d_go.shape}: {d_go.sum()}")
        print(f"uniform1d_jit d_w shape:{d_w.shape}: {d_w.sum()}")
        print(f"uniform1d_jit d_x shape:{d_x.shape}: {d_x.sum()}")
        print(f"uniform1d_jit d_y shape:{d_y.shape}: {d_y.sum()}")
        dump_tensor(d_go, f"/home/malixian/repos/FastEq/test/dump/uniform1d_jit_{d_go.shape[0]}_{d_go.shape[1]}.pt")
        dump_tensor(d_w, f"/home/malixian/repos/FastEq/test/dump/uniform1d_jit_{d_w.shape[0]}_{d_w.shape[1]}.pt")
        dump_tensor(d_x, f"/home/malixian/repos/FastEq/test/dump/uniform1d_jit_{d_x.shape[0]}_{d_x.shape[1]}.pt")
        dump_tensor(d_y, f"/home/malixian/repos/FastEq/test/dump/uniform1d_jit_{d_y.shape[0]}_{d_y.shape[1]}.pt") """
        result_iter = iter(result)
        d_go = next(result_iter)
        d_w = next(result_iter) if ctx.need_grad_w else None
        d_x = next(result_iter) if ctx.need_grad_x else None
        d_y = next(result_iter) if ctx.need_grad_y else None
        return (
            d_go, d_w, d_x, d_y,
            None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None,
        )

# -----------------------------------------------------------------------------
# autograd function
# -----------------------------------------------------------------------------

class FastUniform1dJITFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w,
        x,
        y,
        input_indices,
        output_indices,
        meta,
        use_multiwarp_candidates,
    ):
        # Capture the public input's autograd requirement before reshaping.
        # This flag is part of both backward and double-backward JIT signatures.
        need_grad_w = bool(ctx.needs_input_grad[0])
        need_grad_x = bool(ctx.needs_input_grad[1])
        need_grad_y = bool(ctx.needs_input_grad[2])

        i_list = _as_int32_meta_tensor(meta["i_list"])
        j_list = _as_int32_meta_tensor(meta["j_list"])
        k_list = _as_int32_meta_tensor(meta["k_list"])
        v_list = _as_int32_meta_tensor(meta["v_list"])
        coeff_list = meta["coeff_list"]

        out_seg_num = meta["out_seg_num"]
        w_seg_num = meta["w_seg_num"]
        x_seg_num = meta["x_seg_num"]
        y_seg_num = meta["y_seg_num"]
        u_dim = meta["u_dim"]
        # The public fast_uniform1d_jit() argument is the source of truth.
        # Previously this value was read only from meta, while the wrapper did
        # not pass its use_multiwarp_candidates argument into autograd.apply().
        # Consequently False was ignored and None fell back to the global
        # multi-warp default.
        use_multiwarp_candidates = _resolve_multiwarp_candidates_enabled(
            use_multiwarp_candidates
        )

        """ _print_index_reuse_stats(
            i_list=i_list,
            j_list=j_list,
            k_list=k_list,
            v_list=v_list,
            print_details=True,
        ) """
        
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
            out_seg_num=out_seg_num,
            u_dim=u_dim,
            mode=mode,
            use_multiwarp_candidates=use_multiwarp_candidates,
        )

        out = out.view(-1, out_seg_num * u_dim)

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000.0
        print(f"<< fasteq uniform1d fused forward cost: {end_time - start_time:.3f} ms >>")

        ctx.save_for_backward(w, x, y)
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
        ctx.use_multiwarp_candidates = use_multiwarp_candidates
        ctx.input_indices=input_indices
        ctx.output_indices=output_indices
        ctx.need_grad_w = need_grad_w
        ctx.need_grad_x = need_grad_x
        ctx.need_grad_y = need_grad_y

        return out

    @staticmethod
    def backward(ctx, grad_out):
        w, x, y = ctx.saved_tensors
        grad_out = grad_out.view(-1, ctx.out_seg_num, ctx.u_dim)

        gw3, gx3, gy3 = FastUniform1dBackwardFunction.apply(
            grad_out, w, x, y,
            ctx.i_list, ctx.j_list, ctx.k_list, ctx.v_list, ctx.coeff_list,
            ctx.input_indices, ctx.output_indices,
            ctx.out_seg_num, ctx.u_dim,
            ctx.w_seg_num, ctx.x_seg_num, ctx.y_seg_num, ctx.out_seg_num,
            ctx.mode, ctx.need_grad_w, ctx.need_grad_x, ctx.need_grad_y,
            bool(ctx.use_multiwarp_candidates),
        )
        grad_w = gw3.view(-1, ctx.w_seg_num * ctx.w_irreps) if ctx.need_grad_w else None
        grad_x = gx3.view(-1, ctx.x_seg_num * ctx.x_irreps) if ctx.need_grad_x else None
        grad_y = gy3.view(-1, ctx.y_seg_num * ctx.y_irreps) if ctx.need_grad_y else None
        return grad_w, grad_x, grad_y, None, None, None, None

def fast_uniform1d_jit(
    w,
    x,
    y,
    input_indices,
    output_indices,
    meta,
    use_multiwarp_candidates: Optional[bool] = False,
):
    """Run Uniform1D JIT with optional multi-warp candidate generation.
    """
    return FastUniform1dJITFunction.apply(
        w,
        x,
        y,
        input_indices,
        output_indices,
        meta,
        use_multiwarp_candidates,
    )
