"""FastEq scheduling: metadata."""
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

from .models import (
    STCPath,
    STC_PAD_VALUE,
)


def _normalize_stc_padded_paths(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: Optional[torch.Tensor] = None,
    path_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return padded STC paths as a CPU int64 tensor with shape [max_len, P].

    Accepted inputs:
      * ``idx_lists_tensor`` style: [max_len, P]
      * ``paths_tensor`` style:    [P, max_len]
      * Python sequence of max_len tensors, each [P]

    Without ``path_lens``, the path dimension is inferred from
    ``coeff_list.numel()``.  This allows dropping ``path_lens_tensor`` from the
    public API as long as padding uses a sentinel value such as -1.
    """
    if path_lens is not None:
        P = int(path_lens.detach().cpu().reshape(-1).numel())
    elif coeff_list is not None:
        P = int(coeff_list.detach().cpu().reshape(-1).numel())
    else:
        raise ValueError("Either coeff_list or path_lens is required to infer the path dimension")

    if isinstance(idx_lists, torch.Tensor):
        idx = idx_lists.detach().cpu().to(torch.int64)
    else:
        idx = torch.stack([t.detach().cpu().to(torch.int64).reshape(-1) for t in idx_lists], dim=0)

    if idx.dim() != 2:
        raise ValueError(f"STC padded paths must be 2D, got shape={tuple(idx.shape)}")

    # Already [max_len, P], as in stc_meta['idx_lists_tensor'].
    if idx.shape[1] == P and idx.shape[0] != P:
        return idx.contiguous()

    # Row-major [P, max_len], as in stc_meta['paths'].
    if idx.shape[0] == P and idx.shape[1] != P:
        return idx.t().contiguous()

    # Ambiguous square case. Prefer the STC convention max_len <= 5 when possible.
    if idx.shape[0] == P and idx.shape[1] == P:
        if idx.shape[0] <= 8:
            raise ValueError(
                f"Ambiguous square STC padded path shape={tuple(idx.shape)}. "
                "Pass a non-square [P,max_len]/[max_len,P] tensor or keep path_lens."
            )
        return idx.t().contiguous()

    raise ValueError(
        f"Cannot infer STC path layout from shape={tuple(idx.shape)} and "
        f"P={P}; expected [max_len, P] or [P, max_len]"
    )


def infer_stc_path_lens_from_padded(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    pad_value: int = STC_PAD_VALUE,
) -> torch.Tensor:
    """Infer [P] int64 path lengths from sentinel-padded STC paths.

    Padding must be a suffix and must use ``pad_value``.  Valid indices are
    expected to be non-negative, so -1 is the recommended sentinel.
    """
    idx = _normalize_stc_padded_paths(idx_lists, coeff_list=coeff_list)
    valid = idx.ne(int(pad_value))
    lens = valid.to(torch.int64).sum(dim=0)
    max_len, P = idx.shape

    for p in range(P):
        L = int(lens[p].item())
        if L < 3 or L > max_len:
            raise ValueError(
                f"Bad inferred STC path length {L} for path {p}; "
                f"expected 3..{max_len}. Did you use pad_value={pad_value}?"
            )
        # Enforce suffix padding: valid entries must be exactly [:L].
        if not bool(valid[:L, p].all().item()) or bool(valid[L:, p].any().item()):
            vals = [int(v) for v in idx[:, p].tolist()]
            raise ValueError(
                f"Non-suffix padding in STC path {p}: {vals}. "
                f"Use valid entries first, then pad with {pad_value}."
            )
    return lens.contiguous()


def _infer_stc_path_lens_tensor_from_padded(
    padded_paths: torch.Tensor,
    coeff_list: torch.Tensor,
    *,
    pad_value: int = STC_PAD_VALUE,
) -> torch.Tensor:
    """Infer CUDA/CPU [P] int32 path lengths from sentinel-padded paths.

    This is used only to call the existing baseline backward kernel, whose API
    still expects path_lens_tensor.
    """
    if not isinstance(padded_paths, torch.Tensor) or padded_paths.dim() != 2:
        raise ValueError(f"padded_paths must be a 2D tensor, got {type(padded_paths)}")
    P = int(coeff_list.detach().reshape(-1).numel())

    if int(padded_paths.size(0)) == P and int(padded_paths.size(1)) != P:
        lens = padded_paths.ne(int(pad_value)).to(torch.int32).sum(dim=1)
    elif int(padded_paths.size(1)) == P and int(padded_paths.size(0)) != P:
        lens = padded_paths.ne(int(pad_value)).to(torch.int32).sum(dim=0)
    elif int(padded_paths.size(0)) == P and int(padded_paths.size(1)) == P:
        raise ValueError(
            f"Ambiguous square STC padded path shape={tuple(padded_paths.shape)}; "
            "cannot infer path dimension without path_lens_tensor."
        )
    else:
        raise ValueError(
            f"Cannot infer STC path layout from shape={tuple(padded_paths.shape)} and P={P}"
        )

    return lens.to(device=padded_paths.device, dtype=torch.int32).contiguous()


def make_stc_paths_from_padded_lists(
    idx_lists: torch.Tensor | Sequence[torch.Tensor],
    coeff_list: torch.Tensor,
    *,
    path_lens: Optional[torch.Tensor] = None,
    pad_value: int = STC_PAD_VALUE,
) -> List[STCPath]:
    """Convert padded STC metadata into scheduler-native paths.

    If ``path_lens`` is omitted, lengths are inferred from sentinel padding.
    Recommended padded format is:

        len == 3: [x1_a,                 x0_d, out_v, -1, -1]
        len == 4: [x1_a, x1_b,          x0_d, out_v, -1]
        len == 5: [x1_a, x1_b, x1_c,   x0_d, out_v]

    Baseline-compatible semantics are:

        x1_indices = idx[0 : L-2, p]
        x0_index   = idx[L-2, p]
        v          = idx[L-1, p]
    """
    idx = _normalize_stc_padded_paths(idx_lists, coeff_list=coeff_list, path_lens=path_lens)
    if path_lens is None:
        lens = infer_stc_path_lens_from_padded(idx, coeff_list, pad_value=pad_value)
    else:
        lens = path_lens.detach().cpu().to(torch.int64).reshape(-1).contiguous()

    coeff = coeff_list.detach().cpu().reshape(-1)
    max_len, P = idx.shape

    if lens.numel() != P:
        raise ValueError(f"path_lens numel {lens.numel()} does not match P={P}")
    if coeff.numel() != P:
        raise ValueError(f"coeff_list numel {coeff.numel()} does not match P={P}")

    paths: List[STCPath] = []
    for p in range(P):
        L = int(lens[p].item())
        if L < 3 or L > max_len:
            raise ValueError(f"Bad STC path length {L} for path {p}; expected 3..{max_len}")

        vals = [int(idx[t, p].item()) for t in range(L)]
        x1_indices = tuple(vals[: L - 2])
        x0_index = vals[L - 2]
        v = vals[L - 1]

        paths.append(
            STCPath(
                pid=p,
                x1_indices=x1_indices,
                x0_index=x0_index,
                v=v,
                c=float(coeff[p].item()),
            )
        )
    return paths


def _check_stc_path_bounds_for_bwd(
    paths: Sequence[STCPath],
    *,
    x0_dim: Optional[int],
    x1_dim: Optional[int],
    v_dim: Optional[int],
) -> None:
    for p in paths:
        if x0_dim is not None and int(p.x0_index) >= int(x0_dim):
            raise ValueError(f"x0 index {p.x0_index} out of x0_dim={x0_dim} in STC path {p.pid}")
        if v_dim is not None and int(p.v) >= int(v_dim):
            raise ValueError(f"out index {p.v} out of v_dim={v_dim} in STC path {p.pid}")
        if int(p.x0_index) < 0 or int(p.v) < 0:
            raise ValueError(f"negative x0/out index in STC path {p.pid}: x0={p.x0_index}, out={p.v}")
        for x1_idx in p.x1_indices:
            if int(x1_idx) < 0:
                raise ValueError(f"negative x1 index {x1_idx} in STC path {p.pid}")
            if x1_dim is not None and int(x1_idx) >= int(x1_dim):
                raise ValueError(f"x1 index {x1_idx} out of x1_dim={x1_dim} in STC path {p.pid}")


def _make_lars_paths_from_uniform1d_lists(
    i_list: List[int],
    j_list: List[int],
    k_list: List[int],
    v_list: List[int],
    coeff_list: List[float],
    *,
    path_semantics: str,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Convert external Uniform1D path indices to LARS-native tuples.

    path_semantics="wxy" keeps compatibility with the previous emitter in this
    file, where external (i,j,k) means w[i], x[j], y[k]. Since LARS expects
    x[i], y[j], w[k], we feed it as (j,k,i,v,c).

    path_semantics="xyw" uses LARS-native semantics directly: x[i], y[j], w[k].
    """
    if path_semantics not in ("wxy", "xyw"):
        raise ValueError("path_semantics must be 'wxy' or 'xyw'")
    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

    paths: List[Tuple[int, int, int, int, float]] = []
    for i, j, k, v, c in zip(i_list, j_list, k_list, v_list, coeff_list):
        i = int(i); j = int(j); k = int(k); v = int(v); c = float(c)
        if path_semantics == "wxy":
            paths.append((j, k, i, v, c))  # LARS native: x[j], y[k], w[i]
        else:
            paths.append((i, j, k, v, c))  # LARS native: x[i], y[j], w[k]
    return paths


def _make_lars_bwd_paths_from_uniform1d_lists(
    i_list: List[int],
    j_list: List[int],
    k_list: List[int],
    v_list: List[int],
    coeff_list: List[float],
    *,
    path_semantics: str,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Convert external Uniform1D path indices to backward-LARS tuples.

    Backward LARS native path tuple is:
        (w_index, x_index, y_index, out_index, coeff)

    path_semantics="wxy":
        external (i,j,k) means w[i], x[j], y[k].
    path_semantics="xyw":
        external (i,j,k) means x[i], y[j], w[k].
    """
    if path_semantics not in ("wxy", "xyw"):
        raise ValueError("path_semantics must be 'wxy' or 'xyw'")
    if not (len(i_list) == len(j_list) == len(k_list) == len(v_list) == len(coeff_list)):
        raise ValueError("i/j/k/v/coeff list lengths must match")

    paths: List[Tuple[int, int, int, int, float]] = []
    for i, j, k, v, c in zip(i_list, j_list, k_list, v_list, coeff_list):
        i = int(i); j = int(j); k = int(k); v = int(v); c = float(c)
        if path_semantics == "wxy":
            paths.append((i, j, k, v, c))
        else:
            paths.append((k, i, j, v, c))
    return paths
