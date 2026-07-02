from __future__ import annotations

import hashlib
import os
import tempfile
import time
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

try:
    from .uniform1d_auto_schedule import (
        STC_PAD_VALUE,
        generate_code_stc_fwd_with_scheduler,
        generate_code_stc_bwd_with_scheduler,
        _normalize_stc_padded_paths,
        infer_stc_path_lens_from_padded,
        _infer_stc_path_lens_tensor_from_padded,
    )
except ImportError:  # Allows direct local testing when this file is not imported as a package module.
    from uniform1d_auto_schedule import (
        STC_PAD_VALUE,
        generate_code_stc_fwd_with_scheduler,
        generate_code_stc_bwd_with_scheduler,
        _normalize_stc_padded_paths,
        infer_stc_path_lens_from_padded,
        _infer_stc_path_lens_tensor_from_padded,
    )

_MODULE_CACHE: Dict[str, Any] = {}


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
        root = Path(env_root)
    else:
        fasteq_root = _find_fasteq_root(Path(__file__).parent)
        if fasteq_root is not None:
            root = fasteq_root / _detect_gpu_backend() / "src" / "uniform1d_jit_codegen"
        else:
            root = Path(tempfile.gettempdir()) / "fasteq_uniform1d_jit_codegen"
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
    # v5 drops external path_lens_tensor when paths are sentinel-padded.
    h.update(b"stc-filejit-v6-sentinel-padding-fwd-bwd-x1-only")
    return h.hexdigest()[:16]


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

    mod = load(
        name=module_name,
        sources=[str(cu_path)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--ptxas-options=-v"],
        build_directory=str(build_dir),
        verbose=verbose,
        with_cuda=True,
    )
    _MODULE_CACHE[module_name] = mod
    return mod


def _get_or_build_module(
    idx_lists: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    V: int,
    U: int,
    dtype: torch.dtype,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    verbose: bool = False,
):
    key = _stable_meta_hash(
        idx_lists,
        coeffs,
        V=V,
        U=U,
        dtype=dtype,
        path_lens=path_lens,
        pad_value=pad_value,
    )
    name = f"stc_lars_u1d_fwd_filejit_{key}"
    build_root = _default_build_root()
    build_dir = build_root / name

    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]

    cu_path = build_dir / f"{name}.cu"
    if cu_path.exists():
        return _load_jit_module_file(module_name=name, code=None, build_dir=build_dir, verbose=verbose)

    idx_norm = _normalize_stc_padded_paths(idx_lists, coeff_list=coeffs, path_lens=path_lens)
    if path_lens is None:
        lens = infer_stc_path_lens_from_padded(idx_norm, coeffs, pad_value=pad_value)
    else:
        lens = path_lens.detach().cpu().to(torch.int64).reshape(-1)

    max_path_len = int(lens.max().item())
    num_paths = int(lens.numel())
    launcher_name = f"{name}_u{int(U)}_path{num_paths}_maxlen{max_path_len}"

    code = generate_code_stc_fwd_with_scheduler(
        idx_lists,
        coeffs,
        path_lens=path_lens,
        pad_value=pad_value,
        num_out_segments=int(V),
        u_dim=int(U),
        out_path="",
        kernel_name=name,
    )

    # The generated CUDA source contains both launcher_<launcher_name>() and
    # PYBIND11_MODULE. It is compiled as a single source file, matching the
    # file-based Uniform1D JIT style.
    return _load_jit_module_file(module_name=name, code=code, build_dir=build_dir, verbose=verbose)



def _get_or_build_bwd_module(
    idx_lists: torch.Tensor,
    coeffs: torch.Tensor,
    *,
    V: int,
    U: int,
    dtype: torch.dtype,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
    verbose: bool = False,
):
    key = _stable_meta_hash(
        idx_lists,
        coeffs,
        V=V,
        U=U,
        dtype=dtype,
        path_lens=path_lens,
        pad_value=pad_value,
    )
    name = f"stc_lars_u1d_bwd_x1_filejit_{key}"
    build_root = _default_build_root()
    build_dir = build_root / name

    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]

    cu_path = build_dir / f"{name}.cu"
    if cu_path.exists():
        return _load_jit_module_file(module_name=name, code=None, build_dir=build_dir, verbose=verbose)

    code = generate_code_stc_bwd_with_scheduler(
        idx_lists,
        coeffs,
        path_lens=path_lens,
        pad_value=pad_value,
        num_out_segments=int(V),
        u_dim=int(U),
        out_path="",
        kernel_name=name,
        tile_u=32,
    )

    # Separate file-based extension for backward. The generated CUDA source
    # exports a single run(grad_out, x1, x0_g, V) function that returns grad_x1.
    return _load_jit_module_file(module_name=name, code=code, build_dir=build_dir, verbose=verbose)


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
    ):
        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

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

        mod = _get_or_build_module(
            idx_lists_tensor,
            coeffs_tensor,
            path_lens=None,
            pad_value=int(pad_value),
            V=int(num_out_segments),
            U=U,
            dtype=x1.dtype,
        )
        out = mod.run(x1.contiguous(), x0_g.contiguous(), int(num_out_segments))

        out = out.view(out.shape[0], -1)

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        print(f"<< fasteq stc uniform1d-lars forward cost: {end_time - start_time:.3f} ms >>")

        ctx.save_for_backward(x1, x0_g, coeffs_tensor, paths_tensor, path_lens_tensor, idx_lists_tensor)
        ctx.num_out_segments = int(num_out_segments)
        ctx.pad_value = int(pad_value)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        torch.cuda.synchronize()
        start_time = time.perf_counter() * 1000

        x1, x0_g, coeffs_tensor, paths_tensor, path_lens_tensor, idx_lists_tensor = ctx.saved_tensors
        mod = _get_or_build_bwd_module(
            idx_lists_tensor,
            coeffs_tensor,
            path_lens=None,
            pad_value=int(ctx.pad_value),
            V=int(ctx.num_out_segments),
            U=int(x1.size(2)),
            dtype=x1.dtype,
        )
        grad_x1 = mod.run(
            grad_out.contiguous(),
            x1.contiguous(),
            x0_g.contiguous(),
            int(ctx.num_out_segments),
        )

        torch.cuda.synchronize()
        end_time = time.perf_counter() * 1000
        print(f"<< fasteq stc uniform1d-lars backward cost: {end_time - start_time:.3f} ms >>")
        return grad_x1, None, None, None, None, None, None, None


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
):
    """Run preprocessed STC Uniform1D JIT.

    Expected metadata is produced by the STC preprocessing stage:
      * ``paths_tensor``: [P, max_len], suffix padded with ``pad_value``.
      * ``idx_lists_tensor``: [max_len, P], same padded values transposed from
        ``paths_tensor``.

    ``path_lens_tensor`` and variable-argument parsing are intentionally not
    accepted here; preprocessing is the single source of metadata layout.
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
    )
