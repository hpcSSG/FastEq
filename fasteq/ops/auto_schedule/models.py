"""FastEq scheduling: models."""
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



Label = Tuple[str, int]  # ('x0', i), ('x1', j), ('o', v)


STC_PAD_VALUE = -1


@dataclass
class Inst:
    op: str
    args: Tuple[Any, ...]
    comment: str = ""

    def __str__(self) -> str:
        body = f"{self.op} " + ", ".join(map(str, self.args))
        return f"{body:<48} # {self.comment}" if self.comment else body


@dataclass
class ScheduleResult:
    instructions: List[Inst]
    path_order: List[int]
    max_live: int
    spills: int
    reloads: int
    final_reg_map: Dict[Label, str]
    profile: Dict[str, Any] = field(default_factory=dict)
    variable_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    placement_summary: Dict[str, Any] = field(default_factory=dict)
    shared_slots: int = 0


@dataclass(frozen=True)
class STCPath:
    """One STC contraction path.

    Padded STC metadata uses baseline-compatible row semantics:

        len == 3: [x1_a,                 x0_d, out_v, pad, ...]
        len == 4: [x1_a, x1_b,          x0_d, out_v, pad, ...]
        len == 5: [x1_a, x1_b, x1_c,   x0_d, out_v, pad, ...]

    Only the first ``path_lens[p]`` entries are semantic; right-side padding is
    ignored.  The generated expression is therefore:

        out[v] += c * prod_i x1[x1_indices[i]] * x0[x0_index]

    Duplicate indices are intentionally preserved in ``product_labels`` so that
    x1[i] * x1[i] is emitted as a square/cube instead of being collapsed.
    """

    pid: int
    x1_indices: Tuple[int, ...]
    x0_index: int
    v: int
    c: float

    @property
    def product_labels(self) -> Tuple[Label, ...]:
        labels: List[Label] = [("x1", int(i)) for i in self.x1_indices]
        labels.append(("x0", int(self.x0_index)))
        return tuple(labels)

    @property
    def labels(self) -> Tuple[Label, ...]:
        # Remove duplicates only for liveness/register allocation.  The product
        # expression still uses product_labels, so x1[i] * x1[i] remains squared.
        seen: Set[Label] = set()
        out: List[Label] = []
        for lab in self.product_labels:
            if lab not in seen:
                seen.add(lab)
                out.append(lab)
        return tuple(out)


@dataclass(frozen=True)
class U1DPath:
    pid: int
    i: int
    j: int
    k: int
    v: int
    c: float

    @property
    def labels(self) -> Tuple[Label, Label, Label]:
        # LARS schedules only input operands.  Output accumulators are emitted
        # as full-resident local variables and are not reload-managed candidates.
        return (("x", self.i), ("y", self.j), ("w", self.k))

    @property
    def product_labels(self) -> Tuple[Label, Label, Label]:
        # Uniform1D has no duplicate-input multiplicity beyond x/y/w.  This
        # property lets the common scheduler handle both fixed-arity Uniform1D
        # and variable-arity STC paths through the same emission path.
        return self.labels


@dataclass(frozen=True)
class LARSPlacementConfig:
    """Post-schedule storage placement policy.

    The defaults intentionally target only clearly low-frequency, long-lived
    values.  They are conservative starting points and are exposed so JIT
    candidate generation can tune them later.
    """

    enabled: bool = True
    direct_use_threshold: int = 1
    input_low_freq_threshold: int = 1000
    input_long_lifetime_threshold: int = 10
    accumulator_low_freq_threshold: int = 2
    accumulator_long_lifetime_threshold: int = 32
    max_shared_slots: Optional[int] = 16
    stats_print: bool = False
    stats_path: Optional[str] = None


PairKey = Tuple[Label, Label]  # (w-label, x-label)


BWD_OUTPUT_KINDS = {"gw", "gx", "gy"}


@dataclass(frozen=True)
class U1DBwdPath:
    """
    Backward path for forward expression:

        out[v] += w[i] * x[j] * y[k] * c

    Per path, backward contributes:
        grad_w[i] += c * grad_out[v] * x[j] * y[k]
        grad_x[j] += c * grad_out[v] * w[i] * y[k]
        grad_y[k] += c * grad_out[v] * w[i] * x[j]
    """
    pid: int
    i: int  # w index
    j: int  # x index
    k: int  # y index
    v: int  # grad_out/out index
    c: float
    need_grad_w: bool = True

    @property
    def labels(self) -> Tuple[Label, ...]:
        # LARS schedules only input operands.  Backward accumulators
        # (gw/gx/gy) are full-resident emitter variables, so they are never
        # reload-managed candidates.
        return (
            ("w", self.i),
            ("x", self.j),
            ("y", self.k),
            ("go", self.v),
        )


BwdPairKey = Tuple[Label, Label]  # (w-label, grad_out-label)


BWD_SPLIT_KINDS = {"gw", "gx", "gy"}


@dataclass(frozen=True)
class U1DBwdSplitPath:
    """
    Backward path used by split-gradient LARS scheduling.

    grad_kind="gw": grad_w[i] += c * grad_out[v] * x[j] * y[k]
    grad_kind="gx": grad_x[j] += c * w[i]        * grad_out[v] * y[k]
    grad_kind="gy": grad_y[k] += c * w[i]        * grad_out[v] * x[j]
    """
    pid: int
    i: int
    j: int
    k: int
    v: int
    c: float
    grad_kind: str

    @property
    def target_index(self) -> int:
        if self.grad_kind == "gw":
            return int(self.i)
        if self.grad_kind == "gx":
            return int(self.j)
        if self.grad_kind == "gy":
            return int(self.k)
        raise ValueError(f"Bad grad_kind: {self.grad_kind}")

    @property
    def labels(self) -> Tuple[Label, ...]:
        if self.grad_kind == "gw":
            return (("x", self.j), ("y", self.k), ("go", self.v))
        if self.grad_kind == "gx":
            return (("w", self.i), ("y", self.k), ("go", self.v))
        if self.grad_kind == "gy":
            return (("w", self.i), ("x", self.j), ("go", self.v))
        raise ValueError(f"Bad grad_kind: {self.grad_kind}")


@dataclass(frozen=True)
class BwdCodegenContext:
    """Pre-parsed Uniform1D backward codegen inputs used by fused/split paths."""
    input_indices: Dict[int, Any]
    output_indices: Dict[int, Any]
    use_x_src: bool
    use_y_src: bool
    use_scatter: bool
    P: int
    i_cpu: List[int]
    j_cpu: List[int]
    k_cpu: List[int]
    v_cpu: List[int]
    c_cpu: List[float]
    bwd_paths: List[Tuple[int, int, int, int, float]]
    mode_str: str
    layout_tag: str


@dataclass(frozen=True)
class U1DDoubleBwdPath:
    """One path in the VJP of the Uniform1D first backward.

    Forward path:
        out[v] += c * w[i] * x[j] * y[k]

    First backward upstream is ``go = grad_out``.  Double backward receives
    upstream gradients of (grad_w, grad_x, grad_y), denoted ggw/ggx/ggy, and
    returns gradients with respect to (go, w, x, y).
    """
    pid: int
    i: int
    j: int
    k: int
    v: int
    c: float
    need_grad_w: bool = True
    need_grad_x: bool = True
    need_grad_y: bool = True

    @property
    def labels(self) -> Tuple[Label, ...]:
        labs: List[Label] = [
            ("w", self.i),
            ("x", self.j),
            ("y", self.k),
            ("go", self.v),
        ]
        if self.need_grad_w:
            labs.append(("ggw", self.i))
        if self.need_grad_x:
            labs.append(("ggx", self.j))
        if self.need_grad_y:
            labs.append(("ggy", self.k))
        return tuple(labs)


@dataclass(frozen=True)
class STCDoubleBwdPath:
    """One logical STC double-backward path.

    First backward computes grad_x1 and grad_x0.  Given upstream gradients
    grad_grad_x1 and grad_grad_x0, double backward returns gradients with
    respect to (grad_out, x1, x0).

    Accumulator labels (dgo/dx1/dx0) are part of the LARS live set, so their
    lifetimes are scheduled together with read-only operands instead of being
    emitted as per-path global atomics.
    """
    pid: int
    x1_indices: Tuple[int, ...]
    x0_index: int
    v: int
    c: float

    @property
    def labels(self) -> Tuple[Label, ...]:
        labs: List[Label] = [
            ("go", int(self.v)),
            ("x0", int(self.x0_index)),
            ("ggx0", int(self.x0_index)),
        ]
        seen: Set[Label] = set(labs)
        for idx in self.x1_indices:
            lab = ("x1", int(idx))
            if lab not in seen:
                labs.append(lab); seen.add(lab)
        for idx in self.x1_indices:
            lab = ("ggx1", int(idx))
            if lab not in seen:
                labs.append(lab); seen.add(lab)
        for lab in (("dgo", int(self.v)), ("dx0", int(self.x0_index))):
            if lab not in seen:
                labs.append(lab); seen.add(lab)
        # grad_x0 depends on every x1 factor, so its upstream gradient creates
        # a d_x1 contribution even for degree-1 paths.
        for idx in self.x1_indices:
            lab = ("dx1", int(idx))
            if lab not in seen:
                labs.append(lab); seen.add(lab)
        return tuple(labs)
