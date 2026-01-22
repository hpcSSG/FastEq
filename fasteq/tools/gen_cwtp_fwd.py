# -*- coding: utf-8 -*-
"""
Torch-friendly CWTP/Ell eval_pid codegen.

Provides:
- parse_patterns_text(text) -> patterns
- generate_eval_pid_cuh(patterns, header_guard=...) -> str  (C++ header content)
- build_pid_table_from_torch(patterns, meta2_4, ell_ij, num_paths=16, max_k_dim=8) -> torch.ByteTensor [P,K]
- generate_pid_table_cuh(pid_table, header_guard=...) -> str  (C++ header content)

Notes
-----
- patterns: list of Pattern(pid, E, ij_tuple) where ij elements are packed uint16 (i + (j<<8)).
- meta2_4: torch.Tensor on CPU, dtype int32, shape [P,4] (or [P,>=3]):
    meta2_4[p] = (k_dim, E, base, _)
- ell_ij: torch.Tensor on CPU, dtype uint16/int32, shape [N]
    It is the same flat storage as c_ell_ij (host copy).
- pid_table: uint8, 255 means invalid.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch



# ----------------------------- data model -----------------------------

@dataclass(frozen=True)
class Pattern:
    pid: int
    E: int
    ij: Tuple[int, ...]  # packed uint16 list length E



def count_patterns_for_paths(meta2, c_ell_ij):
    """
    meta2: torch.int32 [num_paths, 4] 或 numpy，至少包含 base(E rows起点) 和 E
    c_ell_ij: torch.uint16 [ell_total]
    统计所有 path 覆盖到的 row 的 pattern 频次
    """
    if hasattr(c_ell_ij, "cpu"):
        ij = c_ell_ij.cpu().numpy()
    else:
        ij = c_ell_ij  # numpy

    counter = Counter()

    for p in range(len(meta2)):
        k_dim = int(meta2[p, 0])
        E     = int(meta2[p, 1])
        base  = int(meta2[p, 2])

        for k_local in range(k_dim):
            row = base + k_local * E
            key = (E, *ij[row:row+E].tolist())
            counter[key] += 1

    return counter

def analyze_topN(counter: Counter, N=48):
    total = sum(counter.values())
    topN = counter.most_common(N)
    topN_count = sum(c for _, c in topN)
    cover = topN_count / total if total else 0.0

    out_str = ""

    i_max = -1
    j_max = -1
    for key, _ in topN:
        E = key[0]
        ijs = key[1:]
        assert len(ijs) == E, (key, "E mismatch")
        for ij in ijs:
            i = int(ij) & 0xFF
            j = (int(ij) >> 8) & 0xFF
            if i > i_max: i_max = i
            if j > j_max: j_max = j

    print(f"total_rows={total}")
    print(f"Top{N}_rows={topN_count}, cover={cover*100:.2f}%")
    print(f"Top{N} i_max={i_max}, j_max={j_max}")
    print("\nTop patterns:")
    for rank, (key, cnt) in enumerate(topN, 1):
        E = key[0]
        ijs = key[1:]
        pairs = [((ij & 0xFF), (ij >> 8)) for ij in ijs]  # (i,j)
        line_str = f"{rank:2d}. cnt={cnt:4d}  E={E}  ij={ijs}  (i,j)={pairs}\n"
        #print(line_str)

        out_str += line_str
    print(out_str)
    return out_str

# ----------------------------- parse patterns text -----------------------------

_PAT_RE = re.compile(
    r"E\s*=\s*(\d+)\s+ij\s*=\s*\(([^)]*)\)",
    flags=re.IGNORECASE,
)

def _parse_ij_list(ij_str: str) -> List[int]:
    toks = [t.strip() for t in ij_str.split(",")]
    out: List[int] = []
    for t in toks:
        if not t:
            continue
        out.append(int(t))
    return out

def parse_patterns_text(text: str, keep_order: bool = True) -> List[Pattern]:
    """
    Parse patterns from a pasted text block; assign pid by encounter order.
    Deduplicate exact (E, ij_tuple).
    """
    found: List[Tuple[int, Tuple[int, ...]]] = []
    for m in _PAT_RE.finditer(text):
        E = int(m.group(1))
        ij_list = _parse_ij_list(m.group(2))
        if len(ij_list) != E:
            raise ValueError(f"Pattern parse error: E={E} but len(ij)={len(ij_list)}; ij={ij_list}")
        found.append((E, tuple(ij_list)))

    if not found:
        raise ValueError("No patterns found. Ensure text contains 'E=.. ij=(...)'.")

    if keep_order:
        seen = set()
        uniq: List[Tuple[int, Tuple[int, ...]]] = []
        for item in found:
            if item in seen:
                continue
            seen.add(item)
            uniq.append(item)
    else:
        uniq = sorted(set(found), key=lambda x: (x[0], x[1]))

    patterns: List[Pattern] = []
    for pid, (E, ij_t) in enumerate(uniq):
        patterns.append(Pattern(pid=pid, E=E, ij=ij_t))
    return patterns


# ----------------------------- pid_table building (torch tensors) -----------------------------

def _ensure_cpu(t: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(t)}")
    if t.is_cuda:
        raise ValueError(f"{name} must be on CPU (meta is tiny; do this offline). Got CUDA tensor.")
    return t

def build_pid_table_from_torch(
    patterns: List[Pattern],
    meta2_4: torch.Tensor,
    ell_ij: torch.Tensor,
    max_k_dim: int = 8,
    invalid_pid: int = 255,
) -> torch.Tensor:
    """
    Build pid_table[p,k] by matching (E, ij_tuple) read from ell_ij at row=base+k*E.
    Returns torch.uint8 tensor [num_paths, max_k_dim].
    """
    meta2_4 = _ensure_cpu(meta2_4, "meta2_4").contiguous()
    ell_ij = _ensure_cpu(ell_ij, "ell_ij").contiguous()
    num_paths = int(meta2_4.size(0))

    if meta2_4.ndim != 2 or meta2_4.size(1) < 3:
        raise ValueError(f"meta2_4 expected shape [P,4] (or [P,>=3]), got {tuple(meta2_4.shape)}")
    if ell_ij.ndim != 1:
        raise ValueError(f"ell_ij expected 1D tensor, got {tuple(ell_ij.shape)}")

    # Types
    meta2_i32 = meta2_4.to(dtype=torch.int32)
    # ell_ij should be uint16 values; keep in int32 for safe indexing/tuple conversion
    ell_i32 = ell_ij.to(dtype=torch.int32)

    # Map (E, ij_tuple) -> pid
    pat_map: Dict[Tuple[int, Tuple[int, ...]], int] = {(p.E, p.ij): p.pid for p in patterns}

    pid_table = torch.full((num_paths, max_k_dim), int(invalid_pid), dtype=torch.uint8)

    ell_len = int(ell_i32.numel())
    for p in range(num_paths):
        k_dim = int(meta2_i32[p, 0].item())
        E = int(meta2_i32[p, 1].item())
        base = int(meta2_i32[p, 2].item())

        if k_dim < 0 or k_dim > max_k_dim:
            raise ValueError(f"Bad k_dim={k_dim} at p={p} (max_k_dim={max_k_dim})")
        if E <= 0:
            raise ValueError(f"Bad E={E} at p={p}")
        if base < 0:
            raise ValueError(f"Bad base={base} at p={p}")

        for k_local in range(k_dim):
            row = base + k_local * E
            start = row
            end = row + E
            if end > ell_len:
                raise IndexError(
                    f"ell_ij OOB: p={p} k={k_local} row={row} E={E} ell_len={ell_len}"
                )
            ij_slice = ell_i32[start:end].tolist()
            ij_t = tuple(int(x) for x in ij_slice)
            key = (E, ij_t)
            if key not in pat_map:
                raise KeyError(f"Pattern not found: p={p} k={k_local} E={E} ij={ij_t}")
            pid_table[p, k_local] = int(pat_map[key])

    return pid_table


# ----------------------------- C++ header generation -----------------------------

def _emit_header_prelude(header_guard: str) -> str:
    return f"""#pragma once
#ifndef {header_guard}
#define {header_guard}

#include <stdint.h>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#error "This header must be compiled by nvcc (device code)."
#endif

"""

def _emit_header_epilogue(header_guard: str) -> str:
    return f"""

#endif // {header_guard}
"""

def _emit_eval_pid_function(p: Pattern) -> str:
    pid, E, ij = p.pid, p.E, p.ij
    lines: List[str] = []
    lines.append("template <typename T>")
    lines.append(f"__device__ __forceinline__ T eval_pid_{pid}(")
    lines.append("    const T* __restrict__ v_row,")
    lines.append("    const T* __restrict__ s_iu, int iu_base, int U, int u,")
    lines.append("    const T* __restrict__ s_jv, int jv_base")
    lines.append(") {")
    lines.append("    T acc = (T)0;")
    for e in range(E):
        lines.append("    {")
        lines.append(f"        constexpr uint16_t ij = (uint16_t){ij[e]};")
        lines.append("        constexpr int i = (int)(ij & 0xFF);")
        lines.append("        constexpr int j = (int)(ij >> 8);")
        lines.append(f"        T c   = v_row[{e}];")
        lines.append("        T xiu = s_iu[iu_base + i * U + u];")
        lines.append("        T xjv = s_jv[jv_base + j];")
        lines.append("        acc = fma(c, xiu * xjv, acc);")
        lines.append("    }")
    lines.append("    return acc;")
    lines.append("}")
    return "\n".join(lines)

def _emit_dispatch(patterns: List[Pattern]) -> str:
    lines: List[str] = []
    lines.append("template <typename T>")
    lines.append("__device__ __forceinline__ T eval_pid_dispatch(")
    lines.append("    int pid,")
    lines.append("    const T* __restrict__ v_row,")
    lines.append("    const T* __restrict__ s_iu, int iu_base, int U, int u,")
    lines.append("    const T* __restrict__ s_jv, int jv_base")
    lines.append(") {")
    lines.append("    switch (pid) {")
    for p in patterns:
        lines.append(f"      case {p.pid}: return eval_pid_{p.pid}<T>(v_row, s_iu, iu_base, U, u, s_jv, jv_base);")
    lines.append("      default: return (T)0;")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines)

def generate_eval_pid_cuh(
    patterns: List[Pattern],
    header_guard: str = "EVAL_PID_GENERATED_CUH",
) -> str:
    """
    Return C++ header content as a string.
    """
    out: List[str] = []
    out.append(_emit_header_prelude(header_guard))
    out.append("// ---- generated eval_pid_<pid> ----")
    for p in patterns:
        out.append(_emit_eval_pid_function(p))
        out.append("")
    out.append("// ---- generated eval_pid_dispatch ----")
    out.append(_emit_dispatch(patterns))
    out.append(_emit_header_epilogue(header_guard))
    return "\n".join(out)

def generate_pid_table_cuh(
    pid_table: torch.Tensor,
    header_guard: str = "CWTP_PID_TABLE_GENERATED_CUH",
    symbol_name: str = "CWTP_PID_TABLE",
) -> str:
    """
    Emit a constexpr uint8_t table [P][K] in a header.
    pid_table must be CPU uint8.
    """
    pid_table = _ensure_cpu(pid_table, "pid_table").contiguous()
    if pid_table.dtype != torch.uint8:
        pid_table = pid_table.to(dtype=torch.uint8)
    if pid_table.ndim != 2:
        raise ValueError(f"pid_table expected [P,K], got {tuple(pid_table.shape)}")

    P, K = pid_table.shape
    rows = pid_table.tolist()

    out: List[str] = []
    out.append(_emit_header_prelude(header_guard))
    out.append(f"// pid_table shape: [{P}][{K}] ; 255 means invalid/unset")
    out.append(f"static constexpr uint8_t {symbol_name}[{P}][{K}] = {{")
    for p in range(P):
        row = ", ".join(str(int(x)) for x in rows[p])
        out.append(f"  {{{row}}},")
    out.append("};")
    out.append(_emit_header_epilogue(header_guard))
    return "\n".join(out)
