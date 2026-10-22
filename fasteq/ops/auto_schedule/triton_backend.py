"""FastEq scheduling: triton backend."""
from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Union, Iterable, Set, Sequence
from time import perf_counter
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import torch

from .cuda_common import (
    _parse_bwd_lars_label_ref,
    _parse_lars_label_ref,
    _parse_stc_placement_ref,
    _to_float_list,
    _to_int_list,
)
from .metadata import (
    _make_lars_paths_from_uniform1d_lists,
    make_stc_paths_from_padded_lists,
)
from .models import (
    LARSPlacementConfig,
    STC_PAD_VALUE,
    ScheduleResult,
)
from .placement import (
    apply_lars_lifetime_placement,
)
from .schedulers import (
    LARSSTCDoubleBwdScheduler,
    LARSUniform1DBwdSplitScheduler,
    LARSUniform1DDoubleBwdScheduler,
    LARSUniform1DScheduler,
)
from .stc_backward import (
    _make_stc_bwd_logical_schedule,
)
from .uniform1d_backward import (
    _prepare_uniform1d_bwd_codegen_context,
)


_TRITON_RUNTIME_MODULE_CACHE: Dict[str, Any] = {}


def _triton_register_only_placement_config() -> LARSPlacementConfig:
    """Placement policy used by the Triton backend: every live state is register-resident."""
    return LARSPlacementConfig(
        enabled=True,
        direct_use_threshold=0,
        input_low_freq_threshold=-1,
        input_long_lifetime_threshold=1,
        accumulator_low_freq_threshold=-1,
        accumulator_long_lifetime_threshold=1,
        max_shared_slots=0,
        stats_print=False,
        stats_path=None,
    )


def _triton_place_register_only(schedule_result: ScheduleResult, *, schedule_kind: str) -> ScheduleResult:
    placed = apply_lars_lifetime_placement(
        schedule_result,
        schedule_kind=schedule_kind,
        config=_triton_register_only_placement_config(),
    )
    if int(getattr(placed, "shared_slots", 0) or 0) != 0:
        raise RuntimeError("Triton register-only placement unexpectedly produced shared slots")
    for inst in placed.instructions:
        for arg in inst.args:
            if isinstance(arg, str) and (arg.startswith("s") or arg.startswith("d:")):
                raise RuntimeError(f"Triton register-only placement produced non-register token: {arg}")
    return placed


def _triton_source_preamble() -> List[str]:
    return [
        "import torch",
        "import triton",
        "import triton.language as tl",
        "",
    ]


def _triton_validate_num_warps(num_warps: int) -> int:
    """Validate the launch configuration supplied by a caller or tuner."""
    if isinstance(num_warps, bool) or not isinstance(num_warps, int):
        raise TypeError("num_warps must be an integer")
    if num_warps <= 0 or num_warps & (num_warps - 1):
        raise ValueError("num_warps must be a positive power of two")
    return num_warps


def _triton_emit_zero(lines: List[str], name: str, indent: str = "    ") -> None:
    lines.append(f"{indent}if IS_FP64:")
    lines.append(f"{indent}    {name} = tl.zeros((BLOCK_U,), dtype=tl.float64)")
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    {name} = tl.zeros((BLOCK_U,), dtype=tl.float32)")


def _triton_float(x: float) -> str:
    return repr(float(x))


def _triton_mul(terms: Sequence[str]) -> str:
    terms = [str(t) for t in terms]
    if not terms:
        return "1.0"
    expr = terms[0]
    for term in terms[1:]:
        expr = f"({expr} * {term})"
    return expr


def _triton_bind_run_entry(module):
    """Expose the frontend run ABI, including for older generated sources."""
    if callable(getattr(module, "run", None)):
        return module
    generated = getattr(module, "generated_module", None)
    run = getattr(generated, "run", None)
    if not callable(run):
        raise RuntimeError(
            f"Generated Triton module {module.__name__!r} has no callable "
            f"run or generated_module.run entry; source: {module.__file__}"
        )
    module.run = run
    return module


def load_triton_module_from_source(source, *, module_prefix="fasteq_triton"):
    """Load source with a backend-specific cache filename and module name."""
    import importlib.util
    import hashlib
    import pathlib
    import sys
    import tempfile
    from importlib._bootstrap import _ModuleLockManager

    key = hashlib.sha1(source.encode("utf-8")).hexdigest()
    cache_root = pathlib.Path(
        os.environ.get("FASTEQ_TRITON_CACHE", pathlib.Path.home() / ".cache/fasteq/triton")
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    module_name = f"{module_prefix}_{key[:16]}"
    py_file = cache_root / (module_name + ".py")

    # Serialize imports of this name so callers never see a partial module.
    with _ModuleLockManager(module_name):
        cached = sys.modules.get(module_name)
        if cached is not None:
            try:
                return _triton_bind_run_entry(cached)
            except RuntimeError:
                # A previous failed manual import may have left an empty module.
                sys.modules.pop(module_name, None)

        # Publish a complete file atomically, also repairing truncated caches.
        if not py_file.exists() or py_file.read_text(encoding="utf-8") != source:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=cache_root,
                    prefix=module_name + ".", suffix=".tmp", delete=False,
                ) as handle:
                    temporary = pathlib.Path(handle.name)
                    handle.write(source)
                os.replace(temporary, py_file)
                bytecode = pathlib.Path(importlib.util.cache_from_source(str(py_file)))
                bytecode.unlink(missing_ok=True)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

        spec = importlib.util.spec_from_file_location(module_name, str(py_file))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load generated Triton source: {py_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            return _triton_bind_run_entry(module)
        except BaseException:
            if sys.modules.get(module_name) is module:
                del sys.modules[module_name]
            raise


class TritonEmitter:
    """Register-only Triton source emitter for FastEq ``ScheduleResult`` objects.

    The emitter does not reproduce CUDA threadIdx/lane/shared-memory semantics.
    One Triton program owns one ``BLOCK_U`` tile of the U dimension, while LARS
    virtual registers map to Triton SSA tensors/scalars. The default tile is
    128 channels (one tile for U=128). Callers can pass num_warps from their
    launch configuration or autotuner; it defaults to four warps per program.
    block_u and num_warps are independent launch parameters.
    """

    def __init__(self, schedule_result: ScheduleResult, *, kernel_name: str):
        self.schedule_result = schedule_result
        self.kernel_name = str(kernel_name)

    @staticmethod
    def _append_module_footer(lines: List[str]) -> str:
        lines.append("")
        lines.append("generated_module = _GeneratedTritonModule()")
        lines.append("run = generated_module.run")
        lines.append("")
        return "\n".join(lines)

    def emit_uniform1d_fwd(
        self,
        *,
        mode: str,
        use_x_src: bool,
        use_y_src: bool,
        use_scatter: bool,
        block_u: int = 128, num_warps: int = 4,
    ) -> str:
        if mode not in ("u,u,,u", "u,u,u,u"):
            raise ValueError(f"Unsupported Uniform1D Triton mode: {mode}")
        mode_scalar_y = mode == "u,u,,u"
        lines = _triton_source_preamble()
        k = self.kernel_name
        lines += [
            "@triton.jit",
            f"def {k}(w, x, y, out, src_idx, dst_idx, B: tl.constexpr, WB: tl.constexpr, Iw: tl.constexpr, Ix: tl.constexpr, Ky: tl.constexpr, V: tl.constexpr, U: tl.constexpr, BLOCK_U: tl.constexpr, IS_FP64: tl.constexpr):",
            "    e = tl.program_id(0)",
            "    pid_u = tl.program_id(1)",
            "    offs_u = pid_u * BLOCK_U + tl.arange(0, BLOCK_U)",
            "    mask_u = offs_u < U",
            "    w_row = 0 if WB == 1 else e",
        ]
        if use_x_src or use_y_src:
            lines.append("    src = tl.load(src_idx + e)")
        else:
            lines.append("    src = e")
        if use_scatter:
            lines.append("    out_row = tl.load(dst_idx + e)")
        else:
            lines.append("    out_row = e")
        lines.append(f"    x_row = {'src' if use_x_src else 'e'}")
        lines.append(f"    y_row = {'src' if use_y_src else 'e'}")

        def load_expr(ref: str) -> str:
            kind, idx = _parse_lars_label_ref(str(ref))
            if kind == "w":
                return f"tl.load(w + (w_row * Iw + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            if kind == "x":
                return f"tl.load(x + (x_row * Ix + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            if kind == "y":
                if mode_scalar_y:
                    return f"tl.load(y + y_row * Ky + {idx})"
                return f"tl.load(y + (y_row * Ky + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            raise ValueError(f"Bad Uniform1D forward Triton ref: {ref}")

        for inst in self.schedule_result.instructions:
            if inst.op == "load":
                token, ref = inst.args
                lines.append(f"    {token} = {load_expr(str(ref))}")
            elif inst.op == "init_acc":
                token, ref = inst.args
                kind, _ = _parse_lars_label_ref(str(ref))
                if kind != "o":
                    raise ValueError("Uniform1D Triton fwd init_acc expects out[]")
                _triton_emit_zero(lines, str(token))
            elif inst.op == "fma_u1d_placed":
                out_idx, out_token, rx, ry, rw, coeff = inst.args
                c = _triton_float(float(coeff))
                lines.append(f"    {out_token} += {c} * ({rw} * {rx}) * {ry}")
            elif inst.op == "store_acc_placed":
                ref, token = inst.args
                kind, idx = _parse_lars_label_ref(str(ref))
                if kind != "o":
                    raise ValueError("Uniform1D Triton fwd store expects out[]")
                ptr = f"out + (out_row * V + {idx}) * U + offs_u"
                if use_scatter:
                    lines.append(f"    tl.atomic_add({ptr}, {token}, mask=mask_u)")
                else:
                    lines.append(f"    tl.store({ptr}, {token}, mask=mask_u)")
            elif inst.op in ("release",):
                pass
            else:
                raise ValueError(f"Unsupported Uniform1D Triton fwd op: {inst.op}")

        lines += [
            "",
            "class _GeneratedTritonModule:",
            "    def run(self, *args):",
            "        w, x, y = args[0], args[1], args[2]",
            "        p = 3",
        ]
        if use_x_src or use_y_src:
            lines += ["        src_idx = args[p].contiguous(); p += 1"]
        else:
            lines += ["        src_idx = w"]
        if use_scatter:
            lines += ["        dst_idx = args[p].contiguous(); p += 1"]
        else:
            lines += ["        dst_idx = w"]
        lines += [
            "        V = int(args[p])",
            "        B = " + ("int(src_idx.numel())" if (use_x_src or use_y_src) else ("int(dst_idx.numel())" if use_scatter else "int(x.size(0))")),
            "        WB, Iw, U = int(w.size(0)), int(w.size(1)), int(w.size(2))",
            "        Ix, Ky = int(x.size(1)), int(y.size(1))",
            ("        out_rows = int(x.size(0))" if use_scatter else "        out_rows = B"),
            "        out = torch.zeros((out_rows, V, U), device=w.device, dtype=w.dtype)",
            f"        BLOCK_U = {int(block_u)}",
            f"        NUM_WARPS = {_triton_validate_num_warps(num_warps)}",
            "        grid = (B, triton.cdiv(U, BLOCK_U))",
            f"        {k}[grid](w, x, y, out, src_idx, dst_idx, B=B, WB=WB, Iw=Iw, Ix=Ix, Ky=Ky, V=V, U=U, BLOCK_U=BLOCK_U, num_warps=NUM_WARPS, IS_FP64=(w.dtype == torch.float64))",
            "        return out",
        ]
        return self._append_module_footer(lines)

    def emit_uniform1d_bwd_split_bundle(
        self,
        *,
        schedules: Dict[str, ScheduleResult],
        mode: str,
        need_grad_w: bool,
        need_grad_x: bool,
        need_grad_y: bool,
        use_x_src: bool,
        use_y_src: bool,
        use_scatter: bool,
        block_u: int = 128, num_warps: int = 4,
    ) -> str:
        mode_scalar_y = mode == "u,u,,u"
        lines = _triton_source_preamble()
        kernel_names: Dict[str, str] = {}

        for gkind in ("gw", "gx", "gy"):
            if gkind not in schedules:
                continue
            sched = schedules[gkind]
            kn = f"{self.kernel_name}_{gkind}"
            kernel_names[gkind] = kn
            lines += [
                "@triton.jit",
                f"def {kn}(w, x, y, grad_out, grad, src_idx, dst_idx, B: tl.constexpr, WB: tl.constexpr, Iw: tl.constexpr, Ix: tl.constexpr, Ky: tl.constexpr, V: tl.constexpr, U: tl.constexpr, BLOCK_U: tl.constexpr, IS_FP64: tl.constexpr):",
                "    e = tl.program_id(0)",
                "    pid_u = tl.program_id(1)",
                "    offs_u = pid_u * BLOCK_U + tl.arange(0, BLOCK_U)",
                "    mask_u = offs_u < U",
                "    w_row = 0 if WB == 1 else e",
            ]
            if use_x_src or use_y_src:
                lines.append("    src = tl.load(src_idx + e)")
            else:
                lines.append("    src = e")
            if use_scatter:
                lines.append("    go_row = tl.load(dst_idx + e)")
            else:
                lines.append("    go_row = e")
            lines.append(f"    x_row = {'src' if use_x_src else 'e'}")
            lines.append(f"    y_row = {'src' if use_y_src else 'e'}")

            def load_expr(ref: str) -> str:
                kind, idx = _parse_bwd_lars_label_ref(str(ref))
                if kind == "w":
                    return f"tl.load(w + (w_row * Iw + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
                if kind == "x":
                    return f"tl.load(x + (x_row * Ix + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
                if kind == "y":
                    if mode_scalar_y:
                        return f"tl.load(y + y_row * Ky + {idx})"
                    return f"tl.load(y + (y_row * Ky + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
                if kind == "go":
                    return f"tl.load(grad_out + (go_row * V + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
                raise ValueError(f"Bad Uniform1D Triton bwd ref: {ref}")

            for inst in sched.instructions:
                if inst.op == "load":
                    token, ref = inst.args
                    lines.append(f"    {token} = {load_expr(str(ref))}")
                elif inst.op == "init_acc":
                    token, _ref = inst.args
                    _triton_emit_zero(lines, str(token))
                elif inst.op == "bwd_split_fma_placed":
                    ikind, target_idx, acc_token, rw, rx, ry, rgo, coeff = inst.args
                    if str(ikind) != gkind:
                        raise ValueError(f"split kind mismatch {ikind} != {gkind}")
                    c = _triton_float(float(coeff))
                    if gkind == "gw":
                        value = f"{c} * {rgo} * {rx} * {ry}"
                    elif gkind == "gx":
                        value = f"{c} * {rw} * {rgo} * {ry}"
                    else:
                        value = f"{c} * {rw} * {rgo} * {rx}"
                    lines.append(f"    {acc_token} += {value}")
                elif inst.op == "store_acc_placed":
                    ref, token = inst.args
                    kind, idx = _parse_bwd_lars_label_ref(str(ref))
                    expected = {"gw": "gw", "gx": "gx", "gy": "gy"}[gkind]
                    if kind != expected:
                        raise ValueError(f"split store kind mismatch {kind} != {expected}")
                    if gkind == "gw":
                        ptr = f"grad + (w_row * Iw + {idx}) * U + offs_u"
                        lines.append(f"    tl.atomic_add({ptr}, {token}, mask=mask_u)")
                    elif gkind == "gx":
                        ptr = f"grad + (x_row * Ix + {idx}) * U + offs_u"
                        lines.append(f"    tl.atomic_add({ptr}, {token}, mask=mask_u)")
                    elif mode_scalar_y:
                        lines.append(f"    _red_{idx} = tl.sum(tl.where(mask_u, {token}, 0.0), axis=0)")
                        lines.append(f"    tl.atomic_add(grad + y_row * Ky + {idx}, _red_{idx})")
                    else:
                        ptr = f"grad + (y_row * Ky + {idx}) * U + offs_u"
                        lines.append(f"    tl.atomic_add({ptr}, {token}, mask=mask_u)")
                elif inst.op == "release":
                    pass
                else:
                    raise ValueError(f"Unsupported Uniform1D Triton split-bwd op: {inst.op}")
            lines.append("")

        lines += [
            "class _GeneratedTritonModule:",
            "    def run(self, *args):",
            "        w, x, y, grad_out = args[0], args[1], args[2], args[3]",
            "        p = 4",
        ]
        if use_x_src or use_y_src:
            lines.append("        src_idx = args[p].contiguous(); p += 1")
        else:
            lines.append("        src_idx = w")
        if use_scatter:
            lines.append("        dst_idx = args[p].contiguous(); p += 1")
        else:
            lines.append("        dst_idx = w")
        lines += [
            "        V = int(args[p])",
            "        B = " + ("int(src_idx.numel())" if (use_x_src or use_y_src) else ("int(dst_idx.numel())" if use_scatter else "int(grad_out.size(0))")),
            "        WB, Iw, U = int(w.size(0)), int(w.size(1)), int(w.size(2))",
            "        Ix, Ky = int(x.size(1)), int(y.size(1))",
            f"        BLOCK_U = {int(block_u)}",
            f"        NUM_WARPS = {_triton_validate_num_warps(num_warps)}",
            "        grid = (B, triton.cdiv(U, BLOCK_U))",
            "        outs = []",
        ]
        if need_grad_w:
            lines += [
                "        gw = torch.zeros_like(w)",
                f"        {kernel_names['gw']}[grid](w, x, y, grad_out, gw, src_idx, dst_idx, B=B, WB=WB, Iw=Iw, Ix=Ix, Ky=Ky, V=V, U=U, BLOCK_U=BLOCK_U, num_warps=NUM_WARPS, IS_FP64=(w.dtype == torch.float64))",
                "        outs.append(gw)",
            ]
        if need_grad_x:
            lines += [
                "        gx = torch.zeros_like(x)",
                f"        {kernel_names['gx']}[grid](w, x, y, grad_out, gx, src_idx, dst_idx, B=B, WB=WB, Iw=Iw, Ix=Ix, Ky=Ky, V=V, U=U, BLOCK_U=BLOCK_U, num_warps=NUM_WARPS, IS_FP64=(w.dtype == torch.float64))",
                "        outs.append(gx)",
            ]
        if need_grad_y:
            lines += [
                "        gy = torch.zeros_like(y)",
                f"        {kernel_names['gy']}[grid](w, x, y, grad_out, gy, src_idx, dst_idx, B=B, WB=WB, Iw=Iw, Ix=Ix, Ky=Ky, V=V, U=U, BLOCK_U=BLOCK_U, num_warps=NUM_WARPS, IS_FP64=(w.dtype == torch.float64))",
                "        outs.append(gy)",
            ]
        lines.append("        return outs")
        return self._append_module_footer(lines)

    def emit_uniform1d_double_bwd(
        self,
        *,
        mode: str,
        need_grad_w: bool,
        need_grad_x: bool,
        need_grad_y: bool,
        use_x_src: bool,
        use_y_src: bool,
        use_scatter: bool,
        block_u: int = 128, num_warps: int = 4,
    ) -> str:
        mode_scalar_y = mode == "u,u,,u"
        lines = _triton_source_preamble()
        k = self.kernel_name
        lines += [
            "@triton.jit",
            f"def {k}(w, x, y, grad_out, ggw, ggx, ggy, dgo, dw, dx, dy, src_idx, dst_idx, B: tl.constexpr, WB: tl.constexpr, Iw: tl.constexpr, Ix: tl.constexpr, Ky: tl.constexpr, V: tl.constexpr, U: tl.constexpr, BLOCK_U: tl.constexpr, IS_FP64: tl.constexpr):",
            "    e = tl.program_id(0)",
            "    pid_u = tl.program_id(1)",
            "    offs_u = pid_u * BLOCK_U + tl.arange(0, BLOCK_U)",
            "    mask_u = offs_u < U",
            "    w_row = 0 if WB == 1 else e",
        ]
        if use_x_src or use_y_src:
            lines.append("    src = tl.load(src_idx + e)")
        else:
            lines.append("    src = e")
        if use_scatter:
            lines.append("    go_row = tl.load(dst_idx + e)")
        else:
            lines.append("    go_row = e")
        lines.append(f"    x_row = {'src' if use_x_src else 'e'}")
        lines.append(f"    y_row = {'src' if use_y_src else 'e'}")

        def parse_ref(ref: str) -> Tuple[str, int]:
            kind, tail = str(ref).split("[", 1)
            idx = int(tail[:-1])
            return {"grad_out":"go","grad_grad_w":"ggw","grad_grad_x":"ggx","grad_grad_y":"ggy"}.get(kind, kind), idx

        def load_expr(ref: str) -> str:
            kind, idx = parse_ref(ref)
            if kind == "w": return f"tl.load(w + (w_row * Iw + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            if kind == "x": return f"tl.load(x + (x_row * Ix + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            if kind == "y":
                if mode_scalar_y: return f"tl.load(y + y_row * Ky + {idx})"
                return f"tl.load(y + (y_row * Ky + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            if kind == "go": return f"tl.load(grad_out + (go_row * V + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            if kind == "ggw": return f"tl.load(ggw + (w_row * Iw + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            if kind == "ggx": return f"tl.load(ggx + (x_row * Ix + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            if kind == "ggy":
                if mode_scalar_y: return f"tl.load(ggy + y_row * Ky + {idx})"
                return f"tl.load(ggy + (y_row * Ky + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            raise ValueError(ref)

        for inst_id, inst in enumerate(self.schedule_result.instructions):
            if inst.op == "load":
                reg, ref = inst.args
                lines.append(f"    {reg} = {load_expr(str(ref))}")
            elif inst.op == "double_bwd_fma_resident":
                wi, xj, yk, ov, rw, rx, ry, rgo, rggw, rggx, rggy, coeff, igw, igx, igy = inst.args
                c = _triton_float(float(coeff))
                dgo_terms = []
                if bool(igw): dgo_terms.append(f"({rggw} * {rx} * {ry})")
                if bool(igx): dgo_terms.append(f"({rggx} * {rw} * {ry})")
                if bool(igy): dgo_terms.append(f"({rggy} * {rw} * {rx})")
                lines.append(f"    tl.atomic_add(dgo + (go_row * V + {int(ov)}) * U + offs_u, {c} * ({' + '.join(dgo_terms)}), mask=mask_u)")
                if bool(igw):
                    terms=[]
                    if bool(igx): terms.append(f"({rggx} * {rgo} * {ry})")
                    if bool(igy): terms.append(f"({rggy} * {rgo} * {rx})")
                    if terms:
                        lines.append(f"    tl.atomic_add(dw + (w_row * Iw + {int(wi)}) * U + offs_u, {c} * ({' + '.join(terms)}), mask=mask_u)")
                if bool(igx):
                    terms=[]
                    if bool(igw): terms.append(f"({rggw} * {rgo} * {ry})")
                    if bool(igy): terms.append(f"({rggy} * {rgo} * {rw})")
                    if terms:
                        lines.append(f"    tl.atomic_add(dx + (x_row * Ix + {int(xj)}) * U + offs_u, {c} * ({' + '.join(terms)}), mask=mask_u)")
                if bool(igy):
                    terms=[]
                    if bool(igw): terms.append(f"({rggw} * {rgo} * {rx})")
                    if bool(igx): terms.append(f"({rggx} * {rgo} * {rw})")
                    if terms:
                        value=f"{c} * ({' + '.join(terms)})"
                        if mode_scalar_y:
                            lines.append(f"    _dy_{inst_id} = tl.sum(tl.where(mask_u, {value}, 0.0), axis=0)")
                            lines.append(f"    tl.atomic_add(dy + y_row * Ky + {int(yk)}, _dy_{inst_id})")
                        else:
                            lines.append(f"    tl.atomic_add(dy + (y_row * Ky + {int(yk)}) * U + offs_u, {value}, mask=mask_u)")
            elif inst.op == "release":
                pass
            else:
                raise ValueError(f"Unsupported Uniform1D Triton double-bwd op: {inst.op}")

        lines += [
            "",
            "class _GeneratedTritonModule:",
            "    def run(self, *args):",
            "        w, x, y, grad_out = args[0], args[1], args[2], args[3]",
            "        p = 4",
        ]
        if need_grad_w:
            lines += ["        ggw = args[p]; p += 1"]
        else:
            lines += ["        ggw = w"]
        if need_grad_x:
            lines += ["        ggx = args[p]; p += 1"]
        else:
            lines += ["        ggx = x"]
        if need_grad_y:
            lines += ["        ggy = args[p]; p += 1"]
        else:
            lines += ["        ggy = y"]
        if use_x_src or use_y_src:
            lines += ["        src_idx = args[p].contiguous(); p += 1"]
        else:
            lines += ["        src_idx = w"]
        if use_scatter:
            lines += ["        dst_idx = args[p].contiguous(); p += 1"]
        else:
            lines += ["        dst_idx = w"]
        lines += [
            "        V = int(args[p])",
            "        B = " + ("int(src_idx.numel())" if (use_x_src or use_y_src) else ("int(dst_idx.numel())" if use_scatter else "int(x.size(0))")),
            "        WB, Iw, U = int(w.size(0)), int(w.size(1)), int(w.size(2))",
            "        Ix, Ky = int(x.size(1)), int(y.size(1))",
            "        dgo = torch.zeros_like(grad_out)",
            "        dw = torch.zeros_like(w) if " + ("True" if need_grad_w else "False") + " else dgo",
            "        dx = torch.zeros_like(x) if " + ("True" if need_grad_x else "False") + " else dgo",
            "        dy = torch.zeros_like(y) if " + ("True" if need_grad_y else "False") + " else dgo",
            f"        BLOCK_U = {int(block_u)}",
            f"        NUM_WARPS = {_triton_validate_num_warps(num_warps)}",
            "        grid = (B, triton.cdiv(U, BLOCK_U))",
            f"        {k}[grid](w, x, y, grad_out, ggw, ggx, ggy, dgo, dw, dx, dy, src_idx, dst_idx, B=B, WB=WB, Iw=Iw, Ix=Ix, Ky=Ky, V=V, U=U, BLOCK_U=BLOCK_U, num_warps=NUM_WARPS, IS_FP64=(w.dtype == torch.float64))",
            "        outs = [dgo]",
        ]
        if need_grad_w: lines.append("        outs.append(dw)")
        if need_grad_x: lines.append("        outs.append(dx)")
        if need_grad_y: lines.append("        outs.append(dy)")
        lines.append("        return outs")
        return self._append_module_footer(lines)

    def emit_stc_fwd(self, *, block_u: int = 128, num_warps: int = 4) -> str:
        lines = _triton_source_preamble()
        k = self.kernel_name
        lines += [
            "@triton.jit",
            f"def {k}(x1, x0, out, B: tl.constexpr, X1: tl.constexpr, X0: tl.constexpr, V: tl.constexpr, U: tl.constexpr, BLOCK_U: tl.constexpr, IS_FP64: tl.constexpr):",
            "    b = tl.program_id(0)",
            "    pid_u = tl.program_id(1)",
            "    offs_u = pid_u * BLOCK_U + tl.arange(0, BLOCK_U)",
            "    mask_u = offs_u < U",
        ]
        def load_expr(ref: str) -> str:
            kind, idx = _parse_stc_placement_ref(str(ref))
            if kind == "x1": return f"tl.load(x1 + (b * X1 + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            if kind == "x0": return f"tl.load(x0 + (b * X0 + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            raise ValueError(ref)
        for inst in self.schedule_result.instructions:
            if inst.op == "load":
                token, ref = inst.args; lines.append(f"    {token} = {load_expr(str(ref))}")
            elif inst.op == "init_acc":
                token, _ = inst.args; _triton_emit_zero(lines, str(token))
            elif inst.op == "mul_stc_placed":
                _out_idx, out_token, operands, coeff = inst.args
                lines.append(f"    {out_token} += {_triton_float(float(coeff))} * {_triton_mul(tuple(str(v) for v in operands))}")
            elif inst.op == "store_acc_placed":
                ref, token = inst.args
                kind, idx = _parse_stc_placement_ref(str(ref))
                if kind != "o": raise ValueError("STC Triton fwd store expects out")
                lines.append(f"    tl.store(out + (b * V + {idx}) * U + offs_u, {token}, mask=mask_u)")
            elif inst.op == "release": pass
            else: raise ValueError(f"Unsupported STC Triton fwd op: {inst.op}")
        lines += [
            "",
            "class _GeneratedTritonModule:",
            "    def run(self, x1, x0, V):",
            "        B, X1, U = int(x1.size(0)), int(x1.size(1)), int(x1.size(2))",
            "        X0, V = int(x0.size(1)), int(V)",
            "        out = torch.zeros((B, V, U), device=x1.device, dtype=x1.dtype)",
            f"        BLOCK_U = {int(block_u)}",
            f"        NUM_WARPS = {_triton_validate_num_warps(num_warps)}",
            "        grid = (B, triton.cdiv(U, BLOCK_U))",
            f"        {k}[grid](x1, x0, out, B=B, X1=X1, X0=X0, V=V, U=U, BLOCK_U=BLOCK_U, num_warps=NUM_WARPS, IS_FP64=(x1.dtype == torch.float64))",
            "        return out",
        ]
        return self._append_module_footer(lines)

    def emit_stc_bwd(self, *, need_grad_x0: bool, block_u: int = 128, num_warps: int = 4) -> str:
        lines = _triton_source_preamble(); k=self.kernel_name
        lines += [
            "@triton.jit",
            f"def {k}(grad_out, x1, x0, gx1, gx0, B: tl.constexpr, X1: tl.constexpr, X0: tl.constexpr, V: tl.constexpr, U: tl.constexpr, BLOCK_U: tl.constexpr, IS_FP64: tl.constexpr):",
            "    b = tl.program_id(0)",
            "    pid_u = tl.program_id(1)",
            "    offs_u = pid_u * BLOCK_U + tl.arange(0, BLOCK_U)",
            "    mask_u = offs_u < U",
        ]
        def load_expr(ref: str) -> str:
            kind, idx = _parse_stc_placement_ref(str(ref))
            if kind == "x1": return f"tl.load(x1 + (b * X1 + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            if kind == "x0": return f"tl.load(x0 + (b * X0 + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            if kind == "go": return f"tl.load(grad_out + (b * V + {idx}) * U + offs_u, mask=mask_u, other=0.0)"
            raise ValueError(ref)
        for inst in self.schedule_result.instructions:
            if inst.op == "load":
                token, ref=inst.args; lines.append(f"    {token} = {load_expr(str(ref))}")
            elif inst.op == "init_acc":
                token,_=inst.args; _triton_emit_zero(lines,str(token))
            elif inst.op == "stc_bwd_placed":
                x1_indices, target_tokens, x0_target_token, go_token, x0_token, x1_tokens, coeff = inst.args
                xs=tuple(int(v) for v in x1_indices); toks=tuple(str(v) for v in x1_tokens)
                c=_triton_float(float(coeff))
                for pos,_target in enumerate(xs):
                    prod=_triton_mul([toks[q] for q in range(len(toks)) if q != pos])
                    lines.append(f"    {target_tokens[pos]} += {c} * {go_token} * {x0_token} * {prod}")
                if need_grad_x0:
                    lines.append(f"    {x0_target_token} += {c} * {go_token} * {_triton_mul(toks)}")
            elif inst.op == "store_acc_placed":
                ref, token=inst.args; kind,idx=_parse_stc_placement_ref(str(ref))
                if kind == "gx1": lines.append(f"    tl.store(gx1 + (b * X1 + {idx}) * U + offs_u, {token}, mask=mask_u)")
                elif kind == "gx0": lines.append(f"    tl.store(gx0 + (b * X0 + {idx}) * U + offs_u, {token}, mask=mask_u)")
                else: raise ValueError(ref)
            elif inst.op == "release": pass
            else: raise ValueError(f"Unsupported STC Triton bwd op: {inst.op}")
        lines += [
            "",
            "class _GeneratedTritonModule:",
            "    def run(self, grad_out, x1, x0, V):",
            "        B, X1, U = int(x1.size(0)), int(x1.size(1)), int(x1.size(2))",
            "        X0, V = int(x0.size(1)), int(V)",
            "        gx1 = torch.zeros_like(x1)",
            ("        gx0 = torch.zeros_like(x0)" if need_grad_x0 else "        gx0 = gx1"),
            f"        BLOCK_U = {int(block_u)}",
            f"        NUM_WARPS = {_triton_validate_num_warps(num_warps)}",
            "        grid = (B, triton.cdiv(U, BLOCK_U))",
            f"        {k}[grid](grad_out, x1, x0, gx1, gx0, B=B, X1=X1, X0=X0, V=V, U=U, BLOCK_U=BLOCK_U, num_warps=NUM_WARPS, IS_FP64=(x1.dtype == torch.float64))",
            ("        return [gx1, gx0]" if need_grad_x0 else "        return [gx1]"),
        ]
        return self._append_module_footer(lines)

    def emit_stc_double_bwd(self, *, block_u: int = 128, num_warps: int = 4) -> str:
        lines=_triton_source_preamble(); k=self.kernel_name
        lines += [
            "@triton.jit",
            f"def {k}(grad_out, x1, x0, i0, ggx0, ggx1, dgo, dx1, dx0, B: tl.constexpr, B0: tl.constexpr, X1: tl.constexpr, X0: tl.constexpr, V: tl.constexpr, U: tl.constexpr, BLOCK_U: tl.constexpr, IS_FP64: tl.constexpr):",
            "    b = tl.program_id(0)",
            "    pid_u = tl.program_id(1)",
            "    offs_u = pid_u * BLOCK_U + tl.arange(0, BLOCK_U)",
            "    mask_u = offs_u < U",
            "    x0_b = tl.load(i0 + b)",
            "    valid_b = (x0_b >= 0) & (x0_b < B0)",
            "    mask = mask_u & valid_b",
        ]
        def parse_ref(ref: str) -> Tuple[str,int]:
            kind,tail=str(ref).split("[",1); idx=int(tail[:-1])
            return {"grad_out":"go","grad_grad_x0":"ggx0","grad_grad_x1":"ggx1","d_grad_out":"dgo","d_x0":"dx0","d_x1":"dx1"}.get(kind,kind),idx
        def load_expr(ref: str) -> str:
            kind,idx=parse_ref(ref)
            if kind=="go": return f"tl.load(grad_out + (b * V + {idx}) * U + offs_u, mask=mask, other=0.0)"
            if kind=="x1": return f"tl.load(x1 + (b * X1 + {idx}) * U + offs_u, mask=mask, other=0.0)"
            if kind=="x0": return f"tl.load(x0 + (x0_b * X0 + {idx}) * U + offs_u, mask=mask, other=0.0)"
            if kind=="ggx1": return f"tl.load(ggx1 + (b * X1 + {idx}) * U + offs_u, mask=mask, other=0.0)"
            if kind=="ggx0": return f"tl.load(ggx0 + (x0_b * X0 + {idx}) * U + offs_u, mask=mask, other=0.0)"
            raise ValueError(ref)
        for inst_id,inst in enumerate(self.schedule_result.instructions):
            if inst.op=="load":
                reg,ref=inst.args; lines.append(f"    {reg} = {load_expr(str(ref))}")
            elif inst.op=="init_acc":
                reg,_=inst.args; _triton_emit_zero(lines,str(reg))
            elif inst.op=="stc_double_bwd_path":
                x1_indices,x0_idx,ov,coeff,rgo,rx0,rggx0,rx1,rgg,rdgo,rdx0,rdx1=inst.args
                xs=tuple(int(v) for v in x1_indices); rx1=tuple(str(v) for v in rx1); rgg=tuple(str(v) for v in rgg); rdx1=tuple(str(v) for v in rdx1); c=_triton_float(float(coeff))
                lines.append(f"    {rdgo} += {c} * {rggx0} * {_triton_mul(rx1)}")
                for r in range(len(xs)):
                    prod=_triton_mul([rx1[q] for q in range(len(xs)) if q != r])
                    lines.append(f"    {rdx1[r]} += {c} * {rggx0} * {rgo} * {prod}")
                for t in range(len(xs)):
                    prod_t=_triton_mul([rx1[q] for q in range(len(xs)) if q != t])
                    lines.append(f"    {rdgo} += {c} * {rgg[t]} * {rx0} * {prod_t}")
                    lines.append(f"    {rdx0} += {c} * {rgg[t]} * {rgo} * {prod_t}")
                    for r in range(len(xs)):
                        if r==t: continue
                        prod_tr=_triton_mul([rx1[q] for q in range(len(xs)) if q != t and q != r])
                        lines.append(f"    {rdx1[r]} += {c} * {rgg[t]} * {rgo} * {rx0} * {prod_tr}")
            elif inst.op=="store_acc":
                ref,reg=inst.args; kind,idx=parse_ref(str(ref))
                if kind=="dgo": lines.append(f"    tl.store(dgo + (b * V + {idx}) * U + offs_u, {reg}, mask=mask)")
                elif kind=="dx1": lines.append(f"    tl.store(dx1 + (b * X1 + {idx}) * U + offs_u, {reg}, mask=mask)")
                elif kind=="dx0": lines.append(f"    tl.atomic_add(dx0 + (x0_b * X0 + {idx}) * U + offs_u, {reg}, mask=mask)")
                else: raise ValueError(ref)
            elif inst.op=="release": pass
            else: raise ValueError(f"Unsupported STC Triton double-bwd op: {inst.op}")
        lines += [
            "",
            "class _GeneratedTritonModule:",
            "    def run(self, grad_out, x1, x0, i0, grad_grad_x0, grad_grad_x1, V):",
            "        B, B0 = int(x1.size(0)), int(x0.size(0))",
            "        X1, X0, U, V = int(x1.size(1)), int(x0.size(1)), int(x1.size(2)), int(V)",
            "        dgo = torch.zeros_like(grad_out)",
            "        dx1 = torch.zeros_like(x1)",
            "        dx0 = torch.zeros_like(x0)",
            f"        BLOCK_U = {int(block_u)}",
            f"        NUM_WARPS = {_triton_validate_num_warps(num_warps)}",
            "        grid = (B, triton.cdiv(U, BLOCK_U))",
            f"        {k}[grid](grad_out, x1, x0, i0, grad_grad_x0, grad_grad_x1, dgo, dx1, dx0, B=B, B0=B0, X1=X1, X0=X0, V=V, U=U, BLOCK_U=BLOCK_U, num_warps=NUM_WARPS, IS_FP64=(x1.dtype == torch.float64))",
            "        return [dgo, dx1, dx0]",
        ]
        return self._append_module_footer(lines)


def build_triton_uniform1d_fwd_module_source(
    i_list: torch.Tensor, j_list: torch.Tensor, k_list: torch.Tensor,
    v_list: torch.Tensor, coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    *, u_dim: int = 1, mode: str = "u,u,,u", path_semantics: str = "wxy",
    kernel_name: str = "uniform1d_triton_fwd", block_u: int = 128, num_warps: int = 4,
) -> str:
    input_indices={} if input_indices is None else dict(input_indices); output_indices={} if output_indices is None else dict(output_indices)
    paths=_make_lars_paths_from_uniform1d_lists(_to_int_list(i_list),_to_int_list(j_list),_to_int_list(k_list),_to_int_list(v_list),_to_float_list(coeff_list),path_semantics=path_semantics)
    sched=LARSUniform1DScheduler(paths=paths,enable_secondary_affinity=False,topk_candidates=128,profile=False,profile_print=False).schedule()
    placed=_triton_place_register_only(sched,schedule_kind="fwd")
    return TritonEmitter(placed,kernel_name=f"{kernel_name}_u{int(u_dim)}_p{len(paths)}").emit_uniform1d_fwd(mode=mode,use_x_src=1 in input_indices,use_y_src=2 in input_indices,use_scatter=0 in output_indices,block_u=block_u,num_warps=num_warps)


def build_triton_uniform1d_bwd_module_source(
    i_list: torch.Tensor, j_list: torch.Tensor, k_list: torch.Tensor,
    v_list: torch.Tensor, coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    *, u_dim: int = 1, iw_dim: Optional[int] = None, ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None, v_dim: Optional[int] = None,
    mode: str = "u,u,,u", need_grad_w: bool = True, need_grad_x: bool = True,
    need_grad_y: bool = True, path_semantics: str = "wxy",
    kernel_name: str = "uniform1d_triton_bwd", block_u: int = 128, num_warps: int = 4,
) -> str:
    del iw_dim,ix_dim,ky_dim,v_dim
    ctx=_prepare_uniform1d_bwd_codegen_context(i_list=i_list,j_list=j_list,k_list=k_list,v_list=v_list,coeff_list=coeff_list,input_indices=input_indices,output_indices=output_indices,mode=mode,path_semantics=path_semantics)
    schedules: Dict[str,ScheduleResult]={}
    for gkind,needed in (("gw",need_grad_w),("gx",need_grad_x),("gy",need_grad_y)):
        if not needed: continue
        s=LARSUniform1DBwdSplitScheduler(ctx.bwd_paths,grad_kind=gkind,enable_secondary_affinity=False,topk_candidates=128,profile=False,profile_print=False).schedule()
        schedules[gkind]=_triton_place_register_only(s,schedule_kind=f"bwd_split:{gkind}")
    if not schedules: raise ValueError("at least one Uniform1D gradient must be requested")
    dummy=next(iter(schedules.values()))
    return TritonEmitter(dummy,kernel_name=f"{kernel_name}_u{int(u_dim)}_p{ctx.P}").emit_uniform1d_bwd_split_bundle(schedules=schedules,mode=mode,need_grad_w=bool(need_grad_w),need_grad_x=bool(need_grad_x),need_grad_y=bool(need_grad_y),use_x_src=ctx.use_x_src,use_y_src=ctx.use_y_src,use_scatter=ctx.use_scatter,block_u=block_u,num_warps=num_warps)


def build_triton_uniform1d_double_bwd_module_source(
    i_list: torch.Tensor, j_list: torch.Tensor, k_list: torch.Tensor,
    v_list: torch.Tensor, coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    *, u_dim: int = 1, iw_dim: Optional[int] = None, ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None, v_dim: Optional[int] = None,
    mode: str = "u,u,,u", need_grad_w: bool = True, need_grad_x: bool = True,
    need_grad_y: bool = True, path_semantics: str = "wxy",
    kernel_name: str = "uniform1d_triton_double_bwd", block_u: int = 128, num_warps: int = 4,
) -> str:
    del iw_dim,ix_dim,ky_dim,v_dim
    ctx=_prepare_uniform1d_bwd_codegen_context(i_list=i_list,j_list=j_list,k_list=k_list,v_list=v_list,coeff_list=coeff_list,input_indices=input_indices,output_indices=output_indices,mode=mode,path_semantics=path_semantics)
    s=LARSUniform1DDoubleBwdScheduler(paths=ctx.bwd_paths,need_grad_w=bool(need_grad_w),need_grad_x=bool(need_grad_x),need_grad_y=bool(need_grad_y),enable_secondary_affinity=False,topk_candidates=128,profile=False,profile_print=False).schedule()
    return TritonEmitter(s,kernel_name=f"{kernel_name}_u{int(u_dim)}_p{ctx.P}").emit_uniform1d_double_bwd(mode=mode,need_grad_w=bool(need_grad_w),need_grad_x=bool(need_grad_x),need_grad_y=bool(need_grad_y),use_x_src=ctx.use_x_src,use_y_src=ctx.use_y_src,use_scatter=ctx.use_scatter,block_u=block_u,num_warps=num_warps)


def build_triton_stc_fwd_module_source(idx_lists, coeff_list, *, path_lens=None, pad_value: int = STC_PAD_VALUE, num_out_segments: int, u_dim: int, kernel_name: str = "stc_triton_fwd", block_u: int = 128, num_warps: int = 4) -> str:
    paths=make_stc_paths_from_padded_lists(idx_lists,coeff_list,path_lens=path_lens,pad_value=pad_value)
    s=LARSUniform1DScheduler(paths,path_kind="stc",enable_secondary_affinity=False,topk_candidates=128,profile=False,profile_print=False).schedule()
    placed=_triton_place_register_only(s,schedule_kind="stc_fwd")
    return TritonEmitter(placed,kernel_name=f"{kernel_name}_u{int(u_dim)}_p{len(paths)}").emit_stc_fwd(block_u=block_u,num_warps=num_warps)


def build_triton_stc_bwd_module_source(idx_lists, coeff_list, *, path_lens=None, pad_value: int = STC_PAD_VALUE, num_out_segments: int, u_dim: int, need_grad_x0: bool = True, kernel_name: str = "stc_triton_bwd", block_u: int = 128, num_warps: int = 4) -> str:
    paths=make_stc_paths_from_padded_lists(idx_lists,coeff_list,path_lens=path_lens,pad_value=pad_value)
    base=LARSUniform1DScheduler(paths,path_kind="stc",enable_secondary_affinity=False,topk_candidates=128,profile=False,profile_print=False).schedule()
    logical=_make_stc_bwd_logical_schedule(paths,base,need_grad_x0=bool(need_grad_x0))
    placed=_triton_place_register_only(logical,schedule_kind="stc_bwd")
    return TritonEmitter(placed,kernel_name=f"{kernel_name}_u{int(u_dim)}_p{len(paths)}").emit_stc_bwd(need_grad_x0=bool(need_grad_x0),block_u=block_u,num_warps=num_warps)


def build_triton_stc_double_bwd_module_source(idx_lists, coeff_list, *, path_lens=None, pad_value: int = STC_PAD_VALUE, num_out_segments: int, u_dim: int, kernel_name: str = "stc_triton_double_bwd", block_u: int = 128, num_warps: int = 4) -> str:
    paths=make_stc_paths_from_padded_lists(idx_lists,coeff_list,path_lens=path_lens,pad_value=pad_value)
    s=LARSSTCDoubleBwdScheduler(paths,enable_secondary_affinity=False,topk_candidates=128,profile=False,profile_print=False).schedule()
    return TritonEmitter(s,kernel_name=f"{kernel_name}_u{int(u_dim)}_p{len(paths)}").emit_stc_double_bwd(block_u=block_u,num_warps=num_warps)


def build_triton_uniform1d_fwd_module(*args, **kwargs):
    src=build_triton_uniform1d_fwd_module_source(*args, **kwargs); return load_triton_module_from_source(src)


def build_triton_uniform1d_bwd_module(*args, **kwargs):
    src=build_triton_uniform1d_bwd_module_source(*args, **kwargs); return load_triton_module_from_source(src)


def build_triton_uniform1d_double_bwd_module(*args, **kwargs):
    src=build_triton_uniform1d_double_bwd_module_source(*args, **kwargs); return load_triton_module_from_source(src)


def build_triton_stc_fwd_module(*args, **kwargs):
    src=build_triton_stc_fwd_module_source(*args, **kwargs); return load_triton_module_from_source(src)


def build_triton_stc_bwd_module(*args, **kwargs):
    src=build_triton_stc_bwd_module_source(*args, **kwargs); return load_triton_module_from_source(src)


def build_triton_stc_double_bwd_module(*args, **kwargs):
    src=build_triton_stc_double_bwd_module_source(*args, **kwargs); return load_triton_module_from_source(src)
