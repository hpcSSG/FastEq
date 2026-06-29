from __future__ import annotations

from collections import defaultdict, OrderedDict, Counter
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple, Optional, Union, Iterable, Set
from time import perf_counter
import math, re
from pathlib import Path
import struct
import numpy as np
import torch


# =============================================================================
# LARS scheduler integration
# =============================================================================

Label = Tuple[str, int]  # ('x', i), ('y', j), ('w', k), ('o', v)

def _to_int_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().tolist()]


def _to_float_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]

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
        # as full-resident local variables and are not spill/reload candidates.
        return (("x", self.i), ("y", self.j), ("w", self.k))


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


class LARSUniform1DScheduler:
    """
    Low-register-budget stable LARS scheduler for Uniform1D paths:

        out[v] += x[i] * y[j] * w[k] * c

    Compared with the simple LARSUniform1DScheduler, this version fixes the
    low-reg thrashing problem:

      old behavior:
          select label -> _alloc_reg() blindly spills one live label

      optimized behavior:
          jointly select (label, spill_victim), and when no progress is made,
          fall back to path-directed scheduling. This avoids repeatedly loading
          a label while spilling another label required by the same target path.

    Notes:
      - reg_budget is a virtual scheduling budget, not ptxas max register count.
      - load_acc/store_acc are emitted as logical accumulator operations. The
        CUDA emitter should implement load_acc as local acc = 0 and store_acc as
        out += acc / atomicAdd(out, acc).
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        reg_budget: int = 16,
        *,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        path_fallback_after: Optional[int] = None,
        prefer_path_fallback_when_full: bool = True,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        self.paths: List[U1DPath] = [
            U1DPath(pid=p, i=i, j=j, k=k, v=v, c=c)
            for p, (i, j, k, v, c) in enumerate(paths)
        ]

        self.reg_budget = int(reg_budget)
        if self.reg_budget < 4:
            raise ValueError("reg_budget must be at least 4.")

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
        self.prefer_path_fallback_when_full = bool(prefer_path_fallback_when_full)
        self.path_fallback_after = (
            int(path_fallback_after)
            if path_fallback_after is not None
            else max(8, 2 * self.reg_budget)
        )
        self.debug = bool(debug)

        self.label_to_paths: Dict[Label, Set[int]] = defaultdict(set)
        for path in self.paths:
            for lab in path.labels:
                self.label_to_paths[lab].add(path.pid)

        self.path_label_sets: List[Set[Label]] = [set(p.labels) for p in self.paths]
        self.unscheduled: Set[int] = set(p.pid for p in self.paths)

        self.remaining_uses: Counter[Label] = Counter()
        for p in self.paths:
            for lab in p.labels:
                self.remaining_uses[lab] += 1

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = [f"r{r}" for r in range(self.reg_budget)]

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        # max_live is kept for backward compatibility.  In the base scheduler
        # it is the maximum number of live normal labels.  In the CSE scheduler
        # it is the maximum total live virtual registers, i.e. labels + CSE
        # pair temporaries.  The explicit fields below make the distinction
        # visible to profiling and budget folding.
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}
        self._no_progress_iters = 0
        self._last_done = 0

        # Optional profile / progress reporting.  Kept disabled by default so
        # existing code generation remains silent unless requested.
        self.profile_enabled = bool(profile)
        self.profile_name = str(profile_name or self.__class__.__name__)
        self.profile_interval = int(profile_interval)
        self.profile_seconds = float(profile_seconds)
        self.profile_print = bool(profile_print)
        self._profile_t0 = perf_counter()
        self._profile_last_print_t = self._profile_t0
        self._profile_last_print_done = 0
        self._profile_loop_iter = 0
        self._profile_select_rounds = 0
        self._profile_fire_rounds = 0
        self._profile_load_rounds = 0
        self._profile_fallback_rounds = 0
        self._profile_records: List[Dict[str, Any]] = []
        self._profile_last_event = ""
        self._profile_last_reason = ""

    # ------------------------------------------------------------------
    # Profile helpers
    # ------------------------------------------------------------------
    def _profile_total_live_regs(self) -> int:
        if hasattr(self, "_total_live_regs"):
            try:
                return int(self._total_live_regs())
            except Exception:
                pass
        return len(self.live)

    def _fmt_profile_eta(self, seconds: float) -> str:
        if seconds == float("inf"):
            return "inf"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"

    def _profile_candidate_count_safe(self) -> int:
        try:
            return len(self._candidate_labels())
        except Exception:
            return 0

    def _profile_pair_stats(self) -> Dict[str, int]:
        return {
            "pair_creates": int(getattr(self, "cse_pair_creates", 0)),
            "pair_hits": int(getattr(self, "cse_pair_hits", 0)),
            "pair_releases": int(getattr(self, "cse_pair_releases", 0)),
            "pair_drops_for_label_release": int(getattr(self, "cse_pair_drops_for_label_release", 0)),
            "pair_drops_for_reg_pressure": int(getattr(self, "cse_pair_drops_for_reg_pressure", 0)),
            "live_pairs": int(len(getattr(self, "pair_reg_of", {}))),
            "max_live_pairs": int(getattr(self, "max_live_pairs", 0)),
        }

    def _profile_snapshot(
        self,
        *,
        event: str,
        reason: str = "",
        fireable_n: Optional[int] = None,
        candidate_n: Optional[int] = None,
        select_ms: Optional[float] = None,
        lab: Optional[Label] = None,
        victim: Optional[Label] = None,
    ) -> Dict[str, Any]:
        done = len(self.path_order)
        remain = len(self.unscheduled)
        total = len(self.paths)
        elapsed = max(perf_counter() - self._profile_t0, 1e-12)
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (remain / rate) if rate > 0 else float("inf")
        snap: Dict[str, Any] = {
            "name": self.profile_name,
            "event": event,
            "iter": int(self._profile_loop_iter),
            "done_paths": int(done),
            "remaining_paths": int(remain),
            "total_paths": int(total),
            "progress_pct": float(100.0 * done / max(1, total)),
            "live_labels": int(len(self.live)),
            "live_regs_total": int(self._profile_total_live_regs()),
            "free_regs": int(len(self.free_regs)),
            "reg_budget": int(self.reg_budget),
            "max_live": int(self.max_live),
            "max_live_labels": int(getattr(self, "max_live_labels", self.max_live)),
            "max_live_total": int(getattr(self, "max_live_total", self.max_live)),
            "fireable_paths": int(fireable_n if fireable_n is not None else 0),
            "candidate_labels": int(candidate_n if candidate_n is not None else 0),
            "select_ms": float(select_ms or 0.0),
            "spills": int(self.spills),
            "reloads": int(self.reloads),
            "instructions": int(len(self.instructions)),
            "select_rounds": int(self._profile_select_rounds),
            "fire_rounds": int(self._profile_fire_rounds),
            "load_rounds": int(self._profile_load_rounds),
            "fallback_rounds": int(self._profile_fallback_rounds),
            "elapsed_s": float(elapsed),
            "rate_paths_per_s": float(rate),
            "eta_s": float(eta),
            "reason": str(reason),
            "label": None if lab is None else self._label_name(lab),
            "victim": None if victim is None else self._label_name(victim),
        }
        snap.update(self._profile_pair_stats())
        return snap

    def _profile_emit(
        self,
        *,
        event: str,
        force: bool = False,
        reason: str = "",
        fireable_n: Optional[int] = None,
        candidate_n: Optional[int] = None,
        select_ms: Optional[float] = None,
        lab: Optional[Label] = None,
        victim: Optional[Label] = None,
    ) -> None:
        if not self.profile_enabled:
            return

        now = perf_counter()
        done = len(self.path_order)
        enough_paths = (
            self.profile_interval > 0
            and done > 0
            and done != self._profile_last_print_done
            and done % self.profile_interval == 0
        )
        enough_time = (now - self._profile_last_print_t) >= self.profile_seconds
        if not (force or enough_paths or enough_time):
            return

        snap = self._profile_snapshot(
            event=event,
            reason=reason,
            fireable_n=fireable_n,
            candidate_n=candidate_n,
            select_ms=select_ms,
            lab=lab,
            victim=victim,
        )
        self._profile_records.append(snap)
        self._profile_last_event = event
        self._profile_last_reason = reason

        if self.profile_print:
            eta = self._fmt_profile_eta(snap["eta_s"])
            elapsed = self._fmt_profile_eta(snap["elapsed_s"])
            cse_part = (
                f" pairs=create/hit/live "
                f"{snap['pair_creates']}/{snap['pair_hits']}/{snap['live_pairs']}"
                if hasattr(self, "pair_reg_of") else ""
            )
            print(
                f"[LARS][profile] {snap['name']} {event:>8s} "
                f"iter={snap['iter']} "
                f"done={snap['done_paths']}/{snap['total_paths']} "
                f"remain={snap['remaining_paths']} "
                f"live={snap['live_regs_total']}/{snap['reg_budget']} "
                f"max=label/pair/total "
                f"{snap['max_live_labels']}/{snap['max_live_pairs']}/{snap['max_live_total']} "
                f"free={snap['free_regs']} "
                f"fireable={snap['fireable_paths']} "
                f"candidates={snap['candidate_labels']} "
                f"label={snap['label']} victim={snap['victim']} "
                f"select={snap['select_ms']:.2f}ms "
                f"spills={snap['spills']} reloads={snap['reloads']} "
                f"insts={snap['instructions']}"
                f"{cse_part} "
                f"rate={snap['rate_paths_per_s']:.1f} path/s "
                f"elapsed={elapsed} eta={eta} "
                f"reason={snap['reason']}",
                flush=True,
            )

        self._profile_last_print_t = now
        self._profile_last_print_done = done

    def _build_profile_summary(self) -> Dict[str, Any]:
        elapsed = max(perf_counter() - self._profile_t0, 1e-12)
        total = len(self.paths)
        done = len(self.path_order)
        remain = len(self.unscheduled)
        summary: Dict[str, Any] = {
            "name": self.profile_name,
            "total_paths": int(total),
            "done_paths": int(done),
            "remaining_paths": int(remain),
            "reg_budget": int(self.reg_budget),
            "max_live": int(self.max_live),
            "max_live_labels": int(getattr(self, "max_live_labels", self.max_live)),
            "max_live_total": int(getattr(self, "max_live_total", self.max_live)),
            "spills": int(self.spills),
            "reloads": int(self.reloads),
            "instructions": int(len(self.instructions)),
            "select_rounds": int(self._profile_select_rounds),
            "fire_rounds": int(self._profile_fire_rounds),
            "load_rounds": int(self._profile_load_rounds),
            "fallback_rounds": int(self._profile_fallback_rounds),
            "elapsed_s": float(elapsed),
            "rate_paths_per_s": float(done / elapsed if elapsed > 0 else 0.0),
            "records": list(self._profile_records),
        }
        summary.update(self._profile_pair_stats())
        return summary

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _path(self, pid: int) -> U1DPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        if kind == "o":
            return f"out[{idx}]"
        return f"{kind}[{idx}]"

    def _invalidate_score_cache(self) -> None:
        self._score_cache.clear()

    def _alloc_reg_no_spill(self, lab: Label) -> str:
        if not self.free_regs:
            raise RuntimeError("_alloc_reg_no_spill called with no free registers")

        reg = self.free_regs.pop(0)
        self.reg_of[lab] = reg
        self.live.add(lab)
        self.max_live_labels = max(self.max_live_labels, len(self.live))
        self.max_live_total = max(self.max_live_total, len(self.live))
        self.max_live = max(self.max_live, len(self.live))
        self._invalidate_score_cache()
        return reg

    def _emit_load_inst(self, lab: Label, reg: str, reason: str) -> None:
        if lab[0] == "o":
            self.instructions.append(
                Inst("load_acc", (reg, self._label_name(lab)), reason)
            )
        else:
            self.instructions.append(
                Inst("load", (reg, self._label_name(lab)), reason)
            )
        self.reloads += 1

    def _store_and_release(self, lab: Label, reason: str = "") -> None:
        if lab not in self.live:
            return

        reg = self.reg_of[lab]

        if lab[0] == "o":
            # If the output accumulator was never modified, a store_acc is not
            # needed. Emit a plain release comment instead. The emitter can
            # ignore release instructions.
            if lab in self.dirty_outputs:
                self.instructions.append(
                    Inst("store_acc", (self._label_name(lab), reg), reason)
                )
                self.dirty_outputs.discard(lab)
            else:
                self.instructions.append(
                    Inst("release", (reg, self._label_name(lab)), reason)
                )
        else:
            self.instructions.append(
                Inst("release", (reg, self._label_name(lab)), reason)
            )

        self.live.remove(lab)
        del self.reg_of[lab]
        self.free_regs.insert(0, reg)
        self._invalidate_score_cache()

    # ------------------------------------------------------------------
    # Fireability
    # ------------------------------------------------------------------
    def _is_fireable(self, pid: int) -> bool:
        return self.path_label_sets[pid] <= self.live

    def _fireable_paths(self) -> List[int]:
        return [pid for pid in self.unscheduled if self.path_label_sets[pid] <= self.live]

    def _fireable_count_after_load_with_victim(
        self,
        lab: Label,
        victim: Optional[Label],
    ) -> int:
        if victim is None:
            tmp_live = self.live | {lab}
        else:
            tmp_live = (self.live - {victim}) | {lab}

        count = 0
        for pid in self.unscheduled:
            if self.path_label_sets[pid] <= tmp_live:
                count += 1
        return count

    def _best_fireable_release_after_load_with_victim(
        self,
        lab: Label,
        victim: Optional[Label],
    ) -> int:
        if victim is None:
            tmp_live = self.live | {lab}
        else:
            tmp_live = (self.live - {victim}) | {lab}

        best = 0
        for pid in self.unscheduled:
            labels = self.path_label_sets[pid]
            if labels <= tmp_live:
                release_now = sum(1 for l in labels if self.remaining_uses[l] == 1)
                best = max(best, release_now)
        return best

    # ------------------------------------------------------------------
    # LARS score
    # ------------------------------------------------------------------
    def _release_potential_if_add(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
        score = 0

        for pid in self.unscheduled:
            labels = self.path_label_sets[pid]
            if labels <= tmp_live:
                for l in labels:
                    if self.remaining_uses[l] == 1:
                        score += 2 if l in self.live else 1

        return score

    def _fire_potential_if_add(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
        count = 0

        for pid in self.label_to_paths[lab] & self.unscheduled:
            if self.path_label_sets[pid] <= tmp_live:
                count += 1

        return count

    def _primary_affinity(self, lab: Label) -> int:
        score = 0
        for pid in self.label_to_paths[lab] & self.unscheduled:
            score += len(self.path_label_sets[pid] & self.live)
        return score

    def _secondary_affinity(self, lab: Label) -> int:
        if not self.enable_secondary_affinity:
            return 0

        neighbor_labels: Set[Label] = set()
        for pid in self.unscheduled:
            labels = self.path_label_sets[pid]
            if labels & self.live:
                neighbor_labels |= labels

        score = 0
        for pid in self.label_to_paths[lab] & self.unscheduled:
            labels = self.path_label_sets[pid]
            score += len((labels & neighbor_labels) - self.live - {lab})
        return score

    def _non_live_affinity_penalty(self, lab: Label) -> int:
        neighbors: Set[Label] = set()
        for pid in self.label_to_paths[lab] & self.unscheduled:
            neighbors |= self.path_label_sets[pid]
        return len(neighbors - self.live - {lab})

    def _priority(self, lab: Label) -> int:
        acc_bonus = 1 if lab[0] == "o" else 0
        return self.remaining_uses[lab] + acc_bonus

    def _label_score(self, lab: Label) -> Tuple[int, int, int, int, int, int]:
        cached = self._score_cache.get(lab)
        if cached is not None:
            return cached

        score = (
            self._release_potential_if_add(lab),
            self._fire_potential_if_add(lab),
            self._primary_affinity(lab),
            self._secondary_affinity(lab),
            -self._non_live_affinity_penalty(lab),
            self._priority(lab),
        )
        self._score_cache[lab] = score
        return score

    def _candidate_labels(self) -> List[Label]:
        candidates = {
            lab
            for pid in self.unscheduled
            for lab in self.path_label_sets[pid]
            if lab not in self.live
        }

        if not candidates:
            raise RuntimeError("No candidate label but no path is fireable.")

        # Optional cheap prefilter. This is useful for large path lists, because
        # full LARS score repeatedly scans unscheduled paths.
        if self.topk_candidates is not None and len(candidates) > self.topk_candidates:
            ranked = sorted(
                candidates,
                key=lambda lab: (
                    self.remaining_uses[lab],
                    len(self.label_to_paths[lab] & self.unscheduled),
                    1 if lab[0] == "o" else 0,
                    str(lab),
                ),
                reverse=True,
            )
            return ranked[: int(self.topk_candidates)]

        return list(candidates)

    # ------------------------------------------------------------------
    # Spill victim selection
    # ------------------------------------------------------------------
    def _choose_spill_victim_avoid(self, avoid: Set[Label]) -> Label:
        candidates = [lab for lab in self.live if lab not in avoid]
        if not candidates:
            candidates = list(self.live)

        if not candidates:
            raise RuntimeError("No live label to spill.")

        def victim_key(lab: Label):
            remain = self.remaining_uses[lab]
            dirty_output_penalty = 1 if lab[0] == "o" and lab in self.dirty_outputs else 0
            output_penalty = 1 if lab[0] == "o" else 0
            return (
                remain,
                dirty_output_penalty,
                output_penalty,
                str(lab),
            )

        return min(candidates, key=victim_key)

    def _select_next_label_and_victim_global(self) -> Tuple[Label, Optional[Label], str]:
        candidates = self._candidate_labels()
        has_free_reg = bool(self.free_regs)

        best_lab: Optional[Label] = None
        best_victim: Optional[Label] = None
        best_key = None

        for lab in candidates:
            lab_score = self._label_score(lab)

            if has_free_reg:
                fire_after = self._fireable_count_after_load_with_victim(lab, None)
                release_after = self._best_fireable_release_after_load_with_victim(lab, None)
                key = (
                    fire_after,
                    release_after,
                    lab_score,
                    str(lab),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_lab = lab
                    best_victim = None
                continue

            # Register file is full: explicitly evaluate spill victims.
            for victim in list(self.live):
                if victim == lab:
                    continue

                fire_after = self._fireable_count_after_load_with_victim(lab, victim)
                if fire_after <= 0:
                    # Loading this label while spilling this victim makes no
                    # immediate progress, so skip to avoid thrashing.
                    continue

                release_after = self._best_fireable_release_after_load_with_victim(lab, victim)
                victim_remaining = self.remaining_uses[victim]
                victim_dirty = 1 if victim[0] == "o" and victim in self.dirty_outputs else 0
                victim_is_output = 1 if victim[0] == "o" else 0

                key = (
                    fire_after,
                    release_after,
                    lab_score,
                    -victim_remaining,
                    -victim_dirty,
                    -victim_is_output,
                    str(lab),
                    str(victim),
                )

                if best_key is None or key > best_key:
                    best_key = key
                    best_lab = lab
                    best_victim = victim

        if best_lab is not None:
            return best_lab, best_victim, f"global key={best_key}"

        # No (lab, victim) pair can immediately create a fireable path.
        # Use path-directed fallback to avoid load/spill oscillation.
        lab, victim, reason = self._select_label_and_victim_by_target_path()
        return lab, victim, "global-no-fire -> " + reason

    def _select_label_and_victim_by_target_path(self) -> Tuple[Label, Optional[Label], str]:
        """
        Low-reg fallback:
          1. Pick an unscheduled path closest to fireable.
          2. Load one missing label from that path.
          3. If spilling is required, avoid spilling labels already live and
             required by that same target path.
        """
        best_pid: Optional[int] = None
        best_key = None

        for pid in self.unscheduled:
            labels = self.path_label_sets[pid]
            live_hits = len(labels & self.live)
            missing = len(labels - self.live)
            p = self._path(pid)

            out_lab = ("o", p.v)
            out_dirty = 1 if out_lab in self.dirty_outputs else 0
            release_now = sum(1 for l in labels if self.remaining_uses[l] == 1)

            key = (
                live_hits,
                -missing,
                out_dirty,
                release_now,
                -pid,
            )

            if best_key is None or key > best_key:
                best_key = key
                best_pid = pid

        if best_pid is None:
            raise RuntimeError("No target path found.")

        target_labels = self.path_label_sets[best_pid]
        missing_labels = list(target_labels - self.live)
        if not missing_labels:
            raise RuntimeError("Target path is already fireable; schedule loop should have fired it first.")

        lab = max(
            missing_labels,
            key=lambda l: (self._label_score(l), str(l)),
        )

        if self.free_regs:
            return lab, None, f"path-fallback pid={best_pid}"

        # Do not spill labels that are already part of the target path.
        avoid = target_labels & self.live
        victim = self._choose_spill_victim_avoid(avoid=avoid)
        return lab, victim, f"path-fallback pid={best_pid} avoid={sorted(avoid)}"

    def _load_label_with_victim(
        self,
        lab: Label,
        victim: Optional[Label],
        reason: str = "",
    ) -> None:
        if lab in self.live:
            return

        if not self.free_regs:
            if victim is None:
                victim = self._choose_spill_victim_avoid(avoid=set())
            self.spills += 1
            self._store_and_release(
                victim,
                reason=f"spill before loading {self._label_name(lab)}; {reason}",
            )

        reg = self._alloc_reg_no_spill(lab)
        self._emit_load_inst(lab, reg, reason)

    # ------------------------------------------------------------------
    # Path firing
    # ------------------------------------------------------------------
    def _choose_fireable_path(self, fireable: List[int]) -> int:
        def key(pid: int):
            labels = self.path_label_sets[pid]
            p = self._path(pid)
            release_now = sum(1 for l in labels if self.remaining_uses[l] == 1)
            out_dirty = 1 if ("o", p.v) in self.dirty_outputs else 0
            reuse_score = sum(self.remaining_uses[l] for l in labels)
            return (release_now, out_dirty, reuse_score, -pid)

        return max(fireable, key=key)

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)
        lx, ly, lw = p.labels

        rx = self.reg_of[lx]
        ry = self.reg_of[ly]
        rw = self.reg_of[lw]

        self.instructions.append(
            Inst(
                "fma_u1d_resident",
                (p.v, rx, ry, rw, p.c),
                f"path#{pid}: out[{p.v}] += x[{p.i}] * y[{p.j}] * w[{p.k}] * {p.c}",
            )
        )

        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        for lab in p.labels:
            self.remaining_uses[lab] -= 1

        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")

    # ------------------------------------------------------------------
    # Public scheduling entry
    # ------------------------------------------------------------------
    def schedule(self) -> ScheduleResult:
        self._profile_emit(event="start", force=True, reason="schedule start")

        while self.unscheduled:
            self._profile_loop_iter += 1
            done = len(self.path_order)
            if done == self._last_done:
                self._no_progress_iters += 1
            else:
                self._no_progress_iters = 0
                self._last_done = done

            # Deadlock/thrash guard: if no progress for too long, force
            # path-directed selection.
            force_path_fallback = self._no_progress_iters > self.path_fallback_after

            fireable = self._fireable_paths()
            fireable_n = len(fireable)
            if fireable:
                pid = self._choose_fireable_path(fireable)
                self._emit_path_compute(pid)
                self._profile_fire_rounds += 1
                self._profile_emit(
                    event="fire",
                    reason=f"path#{pid}",
                    fireable_n=fireable_n,
                    candidate_n=0,
                )
                continue

            select_t0 = perf_counter()
            candidate_n = self._profile_candidate_count_safe()
            if force_path_fallback or (self.prefer_path_fallback_when_full and not self.free_regs):
                lab, victim, reason = self._select_label_and_victim_by_target_path()
                self._no_progress_iters = 0
                self._profile_fallback_rounds += 1
            else:
                lab, victim, reason = self._select_next_label_and_victim_global()

            select_ms = (perf_counter() - select_t0) * 1000.0
            self._profile_select_rounds += 1

            self._load_label_with_victim(
                lab,
                victim,
                reason=reason,
            )
            self._profile_load_rounds += 1
            self._profile_emit(
                event="load",
                reason=reason,
                fireable_n=fireable_n,
                candidate_n=candidate_n,
                select_ms=select_ms,
                lab=lab,
                victim=victim,
            )

            if self.debug and len(self.path_order) == done and self._no_progress_iters > self.path_fallback_after:
                print(
                    "[OptimizedLARS] no progress guard active: "
                    f"done={done}, live={len(self.live)}/{self.reg_budget}, "
                    f"remain={len(self.unscheduled)}, "
                    f"spills={self.spills}, reloads={self.reloads}",
                    flush=True,
                )

        for lab in list(self.live):
            self._store_and_release(lab, reason="end of schedule")

        self._profile_emit(event="end", force=True, reason="schedule end")
        profile_summary = self._build_profile_summary()

        return ScheduleResult(
            instructions=self.instructions,
            path_order=self.path_order,
            max_live=self.max_live,
            spills=self.spills,
            reloads=self.reloads,
            final_reg_map=dict(self.reg_of),
            profile=profile_summary,
        )


PairKey = Tuple[Label, Label]  # (w-label, x-label)


class CSELARSUniform1DScheduler(LARSUniform1DScheduler):
    """
    LARS scheduler with schedule-level common subexpression elimination (CSE)
    for repeated pair products:

        pair = w[k] * x[i]
        out[v] += coeff * pair * y[j]

    Compared with emitter-only CSE, this version makes pair reuse explicit in the
    instruction stream. It emits extra logical instructions:

        pair_cse      (pair_reg, w_reg, x_reg, pair_name)
        fma_u1d_pair  (out_reg, pair_reg, y_reg, coeff)
        release_pair  (pair_reg, pair_name)

    Important behavior:
      - Pair temporaries consume the same virtual register budget as normal
        x/y/w/out labels.
      - CSE is opportunistic only: a pair is cached only when the current
        schedule has at least one free virtual register. It never forces a spill
        and it has no cse_min/cse_max tuning knobs.
      - New pair caches are created only when the pair has at least two
        remaining uses; this is a fixed correctness/performance guard, not a
        tunable search parameter.
      - If a label used by a live pair is released/spilled, the pair is released
        first. It can be recomputed later if useful.
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        reg_budget: int = 16,
        *,
        enable_cse: bool = True,
        cse_release_pair_before_label_spill: bool = True,
        **kwargs,
    ):
        super().__init__(paths=paths, reg_budget=reg_budget, **kwargs)

        self.enable_cse = bool(enable_cse)
        self.cse_release_pair_before_label_spill = bool(
            cse_release_pair_before_label_spill
        )

        # Count initial and remaining uses of every (w, x) pair.  There are no
        # tunable cse_min/cse_max knobs: every repeated pair is a potential CSE
        # candidate, and _ensure_pair_cached() will only materialize it when a
        # virtual register is actually free.
        self.initial_pair_uses: Counter[PairKey] = Counter()
        for p in self.paths:
            self.initial_pair_uses[self._pair_key_for_path_obj(p)] += 1

        self.remaining_pair_uses: Counter[PairKey] = Counter(self.initial_pair_uses)
        self.cse_candidate_pairs: Set[PairKey] = {
            key for key, cnt in self.initial_pair_uses.items()
            if cnt >= 2
        }

        # Live pair temporaries.
        self.pair_reg_of: Dict[PairKey, str] = {}
        self.pair_key_of_reg: Dict[str, PairKey] = {}

        # Debug counters.
        self.cse_pair_creates = 0
        self.cse_pair_hits = 0
        self.cse_pair_releases = 0
        self.cse_pair_drops_for_label_release = 0
        self.cse_pair_drops_for_reg_pressure = 0

    # ------------------------------------------------------------------
    # Pair helpers
    # ------------------------------------------------------------------
    def _pair_key_for_path_obj(self, p) -> PairKey:
        # U1DPath labels are now input-only: (x, y, w). We cache (w * x).
        lx, _ly, lw = p.labels
        return (lw, lx)

    def _pair_name(self, key: PairKey) -> str:
        lw, lx = key
        return f"pair[{self._label_name(lw)}*{self._label_name(lx)}]"

    def _total_live_regs(self) -> int:
        return len(self.live) + len(self.pair_reg_of)

    def _update_max_live_total(self) -> None:
        self.max_live_labels = max(self.max_live_labels, len(self.live))
        self.max_live_pairs = max(self.max_live_pairs, len(self.pair_reg_of))
        self.max_live_total = max(self.max_live_total, self._total_live_regs())
        # For CSE schedules, expose total virtual-register pressure through
        # the legacy max_live field.
        self.max_live = max(self.max_live, self._total_live_regs())

    def _alloc_reg_no_spill(self, lab: Label) -> str:
        reg = super()._alloc_reg_no_spill(lab)
        self._update_max_live_total()
        return reg

    def _choose_pair_victim(self) -> Optional[PairKey]:
        if not self.pair_reg_of:
            return None

        def key_fn(pair_key: PairKey):
            # Prefer dropping pairs with fewer future uses. If equal, drop the
            # one that was originally less reusable.
            return (
                self.remaining_pair_uses[pair_key],
                self.initial_pair_uses[pair_key],
                self._pair_name(pair_key),
            )

        return min(self.pair_reg_of.keys(), key=key_fn)

    def _release_pair(self, pair_key: PairKey, reason: str = "") -> None:
        reg = self.pair_reg_of.pop(pair_key, None)
        if reg is None:
            return

        self.pair_key_of_reg.pop(reg, None)
        self.instructions.append(
            Inst("release_pair", (reg, self._pair_name(pair_key)), reason)
        )
        self.free_regs.insert(0, reg)
        self.cse_pair_releases += 1
        self._invalidate_score_cache()

    def _release_pairs_touching_label(self, lab: Label, reason: str = "") -> None:
        # Conservative rule: pair value is only kept while both operands remain
        # live. If either operand is released/spilled, drop the pair; it can be
        # recomputed later if useful.
        to_release = [key for key in self.pair_reg_of if lab in key]
        for key in to_release:
            self.cse_pair_drops_for_label_release += 1
            self._release_pair(
                key,
                reason=f"drop pair before releasing {self._label_name(lab)}; {reason}",
            )

    def _store_and_release(self, lab: Label, reason: str = "") -> None:
        # Pair regs must be released before their operand label is released.
        self._release_pairs_touching_label(lab, reason=reason)
        super()._store_and_release(lab, reason=reason)
        self._update_max_live_total()

    def _load_label_with_victim(
        self,
        lab: Label,
        victim: Optional[Label],
        reason: str = "",
    ) -> None:
        if lab in self.live:
            return

        # A live pair is cheaper to drop than spilling a normal label/output.
        # This prevents CSE temporaries from causing low-reg thrashing.
        if (
            not self.free_regs
            and self.cse_release_pair_before_label_spill
            and self.pair_reg_of
        ):
            pair_victim = self._choose_pair_victim()
            if pair_victim is not None:
                self.cse_pair_drops_for_reg_pressure += 1
                self._release_pair(
                    pair_victim,
                    reason=f"drop pair under reg pressure before loading {self._label_name(lab)}",
                )

        super()._load_label_with_victim(lab, victim, reason=reason)
        self._update_max_live_total()

    def _should_cache_pair(self, pair_key: PairKey) -> bool:
        if not self.enable_cse:
            return False

        # Only repeated pairs are worth considering. This is fixed policy, not a
        # user/config search knob.
        if pair_key not in self.cse_candidate_pairs:
            return False

        # Already-live pairs do not consume another register.
        if pair_key in self.pair_reg_of:
            return True

        # New CSE temporaries are allowed only when the current schedule has
        # spare virtual registers. Larger reg_budget values therefore help only
        # when they expose actual slack; CSE never spills normal operands.
        if not self.free_regs:
            return False

        # Creating a pair for its last use would add pair_cse without reuse.
        if self.remaining_pair_uses[pair_key] < 2:
            return False

        return True

    def _ensure_pair_cached(self, pair_key: PairKey) -> Optional[str]:
        if not self._should_cache_pair(pair_key):
            return None

        reg = self.pair_reg_of.get(pair_key)
        if reg is not None:
            self.cse_pair_hits += 1
            return reg

        # Both operands must be live at the point of creating the pair.
        lw, lx = pair_key
        if lw not in self.live or lx not in self.live:
            return None

        # Opportunistic CSE: do not force a spill just to create a pair cache.
        # If no free register exists, emit normal fma_u1d for this use.
        if not self.free_regs:
            return None

        pair_reg = self.free_regs.pop(0)
        self.pair_reg_of[pair_key] = pair_reg
        self.pair_key_of_reg[pair_reg] = pair_key
        self._update_max_live_total()
        self._invalidate_score_cache()

        rw = self.reg_of[lw]
        rx = self.reg_of[lx]
        self.instructions.append(
            Inst(
                "pair_cse",
                (pair_reg, rw, rx, self._pair_name(pair_key)),
                f"create {self._pair_name(pair_key)} reuse_left={self.remaining_pair_uses[pair_key]}",
            )
        )
        self.cse_pair_creates += 1
        return pair_reg

    # ------------------------------------------------------------------
    # Path firing with pair CSE
    # ------------------------------------------------------------------
    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)
        lx, ly, lw = p.labels

        rx = self.reg_of[lx]
        ry = self.reg_of[ly]
        rw = self.reg_of[lw]

        pair_key = (lw, lx)
        pair_reg = self._ensure_pair_cached(pair_key)

        if pair_reg is not None:
            self.instructions.append(
                Inst(
                    "fma_u1d_pair_resident",
                    (p.v, pair_reg, ry, p.c),
                    f"path#{pid}: out[{p.v}] += {self._pair_name(pair_key)} * y[{p.j}] * {p.c}",
                )
            )
        else:
            self.instructions.append(
                Inst(
                    "fma_u1d_resident",
                    (p.v, rx, ry, rw, p.c),
                    f"path#{pid}: out[{p.v}] += x[{p.i}] * y[{p.j}] * w[{p.k}] * {p.c}",
                )
            )

        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        # Consume one pair use after emitting the path.
        self.remaining_pair_uses[pair_key] -= 1
        if self.remaining_pair_uses[pair_key] <= 0 and pair_key in self.pair_reg_of:
            self._release_pair(pair_key, reason=f"last pair use after path#{pid}")

        for lab in p.labels:
            self.remaining_uses[lab] -= 1

        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")


class ProgressLARSUniform1DScheduler(LARSUniform1DScheduler):
    """
    Progress-printing wrapper for LARSUniform1DScheduler.

    Compatible with the optimized scheduler whose methods are:
      - _select_next_label_and_victim_global() -> (lab, victim, reason)
      - _select_label_and_victim_by_target_path() -> (lab, victim, reason)
      - _load_label_with_victim(lab, victim, reason)

    Key fix:
      Do not call the old _load_label(lab), because it may trigger blind spill.
      Instead, select label and spill victim together, then call
      _load_label_with_victim(lab, victim).
    """

    def __init__(
        self,
        *args,
        progress_interval: int = 1000,
        progress_seconds: float = 2.0,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.progress_interval = int(progress_interval)
        self.progress_seconds = float(progress_seconds)
        self.verbose = bool(verbose)

        self._total_paths = len(self.paths)
        self._loop_iter = 0

        self._select_rounds = 0
        self._fire_rounds = 0
        self._load_rounds = 0
        self._fallback_rounds = 0

        self._t0 = perf_counter()
        self._last_print_t = self._t0
        self._last_print_done = 0

        self._last_fireable_n = 0
        self._last_candidate_n = 0
        self._last_best_lab = None
        self._last_victim = None
        self._last_best_score = None
        self._last_select_ms = 0.0
        self._last_select_mode = ""
        self._last_select_reason = ""

        # Local progress guard for printing wrapper.
        # The optimized base class may also have its own guard, but this wrapper
        # does not call base schedule(), so we maintain one here.
        self._progress_last_done = 0
        self._progress_no_progress_iters = 0

    def _fmt_lab(self, lab) -> str:
        if lab is None:
            return "None"
        kind, idx = lab
        return f"{kind}[{idx}]"

    def _fmt_eta(self, seconds: float) -> str:
        if seconds == float("inf"):
            return "inf"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"

    def _candidate_count(self) -> int:
        # Prefer optimized scheduler's candidate filter if it exists, because it
        # may apply topk_candidates and other pruning rules.
        if hasattr(self, "_candidate_labels"):
            try:
                return len(self._candidate_labels())
            except Exception:
                pass

        return sum(
            1
            for pid in self.unscheduled
            for lab in self._path(pid).labels
            if lab not in self.live
        )

    def _update_progress_guard(self) -> None:
        done = len(self.path_order)
        if done == self._progress_last_done:
            self._progress_no_progress_iters += 1
        else:
            self._progress_no_progress_iters = 0
            self._progress_last_done = done

    def _should_force_path_fallback(self) -> bool:
        # If the base optimized scheduler exposes path_fallback_after, use it.
        threshold = int(getattr(self, "path_fallback_after", max(8, 2 * int(self.reg_budget))))
        return self._progress_no_progress_iters > threshold

    def _print_progress(self, *, force: bool = False, event: str = "") -> None:
        if not self.verbose:
            return

        now = perf_counter()
        done = len(self.path_order)
        remain = len(self.unscheduled)

        enough_paths = (
            self.progress_interval > 0
            and done > 0
            and done != self._last_print_done
            and done % self.progress_interval == 0
        )
        enough_time = (now - self._last_print_t) >= self.progress_seconds

        if not (force or enough_paths or enough_time):
            return

        elapsed = max(now - self._t0, 1e-12)
        rate = done / elapsed
        eta = (remain / rate) if rate > 0 else float("inf")
        pct = 100.0 * done / max(1, self._total_paths)

        print(
            f"[LARS] {event:>8s} "
            f"iter={self._loop_iter} "
            f"done={done}/{self._total_paths} ({pct:.2f}%) "
            f"remain={remain} "
            f"live={len(self.live)}/{self.reg_budget} "
            f"fireable={self._last_fireable_n} "
            f"candidates={self._last_candidate_n} "
            f"mode={self._last_select_mode} "
            f"best={self._fmt_lab(self._last_best_lab)} "
            f"victim={self._fmt_lab(self._last_victim)} "
            f"score={self._last_best_score} "
            f"select={self._last_select_ms:.2f}ms "
            f"spills={self.spills} reloads={self.reloads} "
            f"insts={len(self.instructions)} "
            f"rate={rate:.1f} path/s "
            f"elapsed={self._fmt_eta(elapsed)} eta={self._fmt_eta(eta)} "
            f"reason={self._last_select_reason}",
            flush=True,
        )

        self._last_print_t = now
        self._last_print_done = done

    def _select_label_victim_with_progress(self):
        """
        Select (lab, victim) using the optimized scheduler API and collect
        progress/debug stats.

        The optimized base class returns:
            lab, victim, reason
        """
        t0 = perf_counter()

        self._last_candidate_n = self._candidate_count()
        force_fallback = self._should_force_path_fallback()
        prefer_fallback_when_full = bool(
            getattr(self, "prefer_path_fallback_when_full", True)
        )

        if force_fallback:
            lab, victim, reason = self._select_label_and_victim_by_target_path()
            self._last_select_mode = "thrash_path"
            self._fallback_rounds += 1
            self._progress_no_progress_iters = 0

        elif (not self.free_regs) and prefer_fallback_when_full:
            lab, victim, reason = self._select_label_and_victim_by_target_path()
            self._last_select_mode = "path"
            self._fallback_rounds += 1

        else:
            # This is the method name used by the optimized scheduler you loaded.
            lab, victim, reason = self._select_next_label_and_victim_global()
            self._last_select_mode = "global"

        self._select_rounds += 1
        self._last_select_ms = (perf_counter() - t0) * 1000.0
        self._last_best_lab = lab
        self._last_victim = victim
        self._last_select_reason = str(reason)

        # For logging only. Do not use this for choosing after victim selection.
        try:
            self._last_best_score = self._label_score(lab)
        except Exception:
            self._last_best_score = None

        return lab, victim, reason

    def schedule(self) -> ScheduleResult:
        self._print_progress(force=True, event="start")

        while self.unscheduled:
            self._loop_iter += 1
            self._update_progress_guard()

            fireable = self._fireable_paths()
            self._last_fireable_n = len(fireable)

            if fireable:
                pid = self._choose_fireable_path(fireable)
                self._emit_path_compute(pid)
                self._fire_rounds += 1

                self._last_select_mode = "fire"
                self._last_best_lab = None
                self._last_victim = None
                self._last_best_score = None
                self._last_select_reason = f"path#{pid}"

                self._print_progress(event="fire")
                continue

            lab, victim, reason = self._select_label_victim_with_progress()

            # Critical fix:
            # Do not call old _load_label(lab). It may call _alloc_reg() and
            # blind-spill an arbitrary victim. The optimized scheduler has already
            # selected an explicit victim that preserves/creates progress.
            self._load_rounds += 1
            self._load_label_with_victim(
                lab,
                victim,
                reason=(
                    f"LARS-score={self._last_best_score}, "
                    f"mode={self._last_select_mode}, "
                    f"victim={victim}, "
                    f"select_reason={reason}"
                ),
            )

            self._print_progress(event="load")

        # Flush live labels at the end.
        for lab in list(self.live):
            self._store_and_release(lab, reason="end of schedule")

        self._print_progress(force=True, event="end")

        if self.verbose:
            print(
                f"[LARS] summary "
                f"select_rounds={self._select_rounds} "
                f"fire_rounds={self._fire_rounds} "
                f"load_rounds={self._load_rounds} "
                f"fallback_rounds={self._fallback_rounds} "
                f"max_live={self.max_live} "
                f"spills={self.spills} "
                f"reloads={self.reloads}",
                flush=True,
            )

        return ScheduleResult(
            instructions=self.instructions,
            path_order=self.path_order,
            max_live=self.max_live,
            spills=self.spills,
            reloads=self.reloads,
            final_reg_map=dict(self.reg_of),
            profile=self._build_profile_summary(),
        )


# =============================================================================
# CUDA codegen from LARSUniform1DScheduler ScheduleResult
# =============================================================================

def _lars_reg_id(reg: str) -> int:
    if not isinstance(reg, str) or not reg.startswith("r"):
        raise ValueError(f"Bad LARS register name: {reg!r}")
    return int(reg[1:])


def _parse_lars_label_ref(ref: str) -> Tuple[str, int]:
    """Parse refs emitted by LARSUniform1DScheduler: x[0], y[1], w[2], out[3]."""
    if not isinstance(ref, str) or "[" not in ref or not ref.endswith("]"):
        raise ValueError(f"Bad LARS label ref: {ref!r}")
    kind, rest = ref.split("[", 1)
    idx = int(rest[:-1])
    if kind == "out":
        kind = "o"
    if kind not in ("x", "y", "w", "o"):
        raise ValueError(f"Bad LARS label kind in ref: {ref!r}")
    return kind, idx


def _fmt_lars_float(x: float) -> str:
    # repr(float) is accepted by CUDA for normal finite constants.
    return repr(float(x))


def _sanitize_cuda_comment(s: str) -> str:
    return str(s).replace("\n", " ").replace("\r", " ").replace("*/", "* /")


def _max_lars_reg_count(schedule_result: ScheduleResult) -> int:
    max_id = -1
    for inst in schedule_result.instructions:
        if inst.op in ("load", "load_acc"):
            max_id = max(max_id, _lars_reg_id(inst.args[0]))
        elif inst.op == "release":
            max_id = max(max_id, _lars_reg_id(inst.args[0]))
        elif inst.op == "store_acc":
            max_id = max(max_id, _lars_reg_id(inst.args[1]))
        elif inst.op == "fma_u1d":
            ro, rx, ry, rw, _ = inst.args
            max_id = max(max_id, _lars_reg_id(ro), _lars_reg_id(rx), _lars_reg_id(ry), _lars_reg_id(rw))
        else:
            raise ValueError(f"Unsupported LARS instruction op: {inst.op}")
    return max_id + 1


def _lars_has_output_reload_after_store(schedule_result: ScheduleResult) -> bool:
    """
    Scatter mode cannot safely use out[] as an accumulator-spill buffer, because
    another block may atomically update the same out element between a spill store
    and a later reload. Dense mode is safe because one block owns one output row.
    """
    stored = set()
    for inst in schedule_result.instructions:
        if inst.op == "store_acc":
            lab = _parse_lars_label_ref(inst.args[0])
            stored.add(lab)
        elif inst.op == "load_acc":
            lab = _parse_lars_label_ref(inst.args[1])
            if lab in stored:
                return True
    return False


def _lars_label_index_expr(
    kind: str,
    idx: int,
    *,
    mode_scalar_y: bool,
    u_dim: Optional[int],
    x_dim: Optional[int],
    y_dim: Optional[int],
    w_dim: Optional[int],
    v_dim: Optional[int],
) -> str:
    if kind == "x":
        if x_dim is not None and idx >= x_dim:
            raise ValueError(f"x index {idx} out of x_dim={x_dim}")
        if u_dim is not None:
            return f"x_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"x_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "w":
        if w_dim is not None and idx >= w_dim:
            raise ValueError(f"w index {idx} out of w_dim={w_dim}")
        if u_dim is not None:
            return f"w_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"w_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "y":
        if y_dim is not None and idx >= y_dim:
            raise ValueError(f"y index {idx} out of y_dim={y_dim}")
        if mode_scalar_y:
            return f"y_base + (index_t){idx}"
        if u_dim is not None:
            return f"y_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"y_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "o":
        if v_dim is not None and idx >= v_dim:
            raise ValueError(f"out index {idx} out of v_dim={v_dim}")
        if u_dim is not None:
            return f"out_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"out_base + (index_t){idx} * (index_t)U + (index_t)u"

    raise ValueError(f"Bad label kind: {kind}")



def emit_launcher(
    bundle_name: str,
    mode: str = "u,u,,u",
    *,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
    if mode == "u,u,,u":
        y_comment = "[B,Ky,1] or [S,Ky,1]"
        y_check_u = (
            'TORCH_CHECK((int)y.size(2) == 1, '
            '"y.size(2) must be 1 for mode u,u,,u");'
        )
    elif mode == "u,u,u,u":
        y_comment = "[B,Ky,U] or [S,Ky,U]"
        y_check_u = (
            'TORCH_CHECK((int)y.size(2) == U, '
            '"y.size(2) must equal U for mode u,u,u,u");'
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    param_lines = [
        "    torch::Tensor w",
        "    torch::Tensor x_all",
        "    torch::Tensor y",
    ]

    if use_x_src or use_y_src:
        param_lines.append("    torch::Tensor src_idx")

    if use_scatter:
        param_lines.append("    torch::Tensor dst_idx")
        param_lines.append("    torch::Tensor b_list")

    param_lines.append("    int64_t V64")

    params = ",\n".join(param_lines)

    src_check = ""
    if use_x_src or use_y_src:
        src_check = r'''
    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''

    dst_check = ""
    if use_scatter:
        dst_check = r'''
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''

    if use_scatter:
        out_alloc = "    auto out = torch::zeros({S, V, U}, w.options());"
    else:
        out_alloc = "    auto out = torch::zeros({B, V, U}, w.options());"

    blist_logic = ""
    if use_scatter:
        blist_logic = r'''
    const int32_t* b_list_ptr = nullptr;
    if (b_list.defined() && b_list.numel() > 0) {
        TORCH_CHECK(b_list.is_cuda(), "b_list must be CUDA/HIP");
        TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list must be int32");
        TORCH_CHECK((int)b_list.numel() == B, "b_list must be [B]");
        b_list_ptr = (const int32_t*)b_list.data_ptr<int32_t>();
    }
'''
    else:
        blist_logic = "    const int32_t* b_list_ptr = nullptr;\n"

    src_numel_check = ""
    if use_x_src or use_y_src:
        src_numel_check = (
            '    TORCH_CHECK(src_idx.numel() >= B, '
            '"src_idx numel must be >= B");'
        )

    dst_numel_check = ""
    if use_scatter:
        dst_numel_check = (
            '    TORCH_CHECK(dst_idx.numel() >= B, '
            '"dst_idx numel must be >= B");'
        )

    launch_src_arg = (
        "(const int32_t*)src_idx.data_ptr<int32_t>(),"
        if (use_x_src or use_y_src)
        else "nullptr,"
    )

    launch_dst_arg = (
        "(const int32_t*)dst_idx.data_ptr<int32_t>(),"
        if use_scatter
        else "nullptr,"
    )

    # ------------------------------------------------------------------
    # Select B source according to execution mode.
    #
    # scatter mode:
    #   B is number of destination/scatter entries.
    #
    # src-index mode:
    #   B is number of source-index entries.
    #
    # dense mode:
    #   B comes from w.size(0), where w is either [1,Iw,U] or [B,Iw,U].
    # ------------------------------------------------------------------
    if use_scatter:
        b_expr = "dst_idx.size(0)"
    elif use_x_src or use_y_src:
        b_expr = "src_idx.size(0)"
    else:
        b_expr = "w.size(0)"

    return rf'''

torch::Tensor launcher_{bundle_name}(
{params})
{{
    // Expected tensors:
    //   w      : [WB, Iw, U], WB can be 1 or B
    //   x_all  : [S, Ix, U] or [B, Ix, U]
    //   y      : {y_comment}
    // Optional:
    //   src_idx: [?] int32, enabled when x/y source indirection is used
    //   dst_idx: [?] int32, enabled when scatter is used
    //   b_list : [B] int32 optional, enabled when scatter is used

    TORCH_CHECK(w.is_cuda() && x_all.is_cuda() && y.is_cuda(),
                "w/x_all/y must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x_all.is_contiguous() && y.is_contiguous(),
                "w/x_all/y must be contiguous");
{src_check}{dst_check}

    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x_all.dim() == 3, "x_all must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");

    int B  = (int){b_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x_all.size(0);
    int Ix = (int)x_all.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");

    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(1) > 0, "Iw must be > 0");

    TORCH_CHECK((int)x_all.size(1) > 0, "Ix must be > 0");
    TORCH_CHECK((int)x_all.size(2) == U, "x_all U mismatch");

    {y_check_u}

    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
{src_numel_check}
{dst_numel_check}

{blist_logic}
{out_alloc}

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{

        TORCH_CHECK(x_all.scalar_type() == w.scalar_type(),
                    "x_all dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(),
                    "y dtype must match w");

        launch_{bundle_name}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x_all.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (scalar_t*)out.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                b_list_ptr,
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    }});

    GPU_KERNEL_LAUNCH_CHECK();

    return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} forward jit impl");
}}
'''


def _max_lars_reg_count_cse_aware(schedule_result) -> int:
    """
    Return max virtual register id + 1 by scanning every string argument in
    every LARS instruction. This is required after scheduler-level CSE because
    pair_cse / fma_u1d_pair / release_pair may introduce additional rN names.
    """
    max_id = -1
    pat = re.compile(r"^r(\d+)$")
    for inst in schedule_result.instructions:
        for arg in inst.args:
            if isinstance(arg, str):
                m = pat.match(arg)
                if m:
                    max_id = max(max_id, int(m.group(1)))
    return max_id + 1





def emit_fused_fwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    w_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """
    Emit CUDA/HIP-compatible forward code directly from LARS instruction order.

    This version is scheduler-CSE-aware. In addition to the original ops
    load/load_acc/fma_u1d/store_acc/release, it supports:
      - pair_cse:      pair_reg = w_reg * x_reg
      - fma_u1d_pair:  out_reg += coeff * pair_reg * y_reg
      - release_pair:  logical lifetime marker only

    LARS-native label semantics:
        x[i], y[j], w[k], out[v]

    The generated kernel keeps the previous emitter's runtime ABI so it can reuse
    emit_launcher(...):
        w:   [WB, Iw, U]
        x:   [S or B, Ix, U]
        y:   [B, Ky, 1] for mode="u,u,,u" or [B, Ky, U] for mode="u,u,u,u"
        out: [B or S, V, U]
    """
    if block_size != 32:
        raise ValueError("this emitter currently assumes block_size=32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")
    # Important semantics fix:
    # LARS output registers are treated as block-local delta accumulators.
    # load_acc initializes a fresh local delta accumulator to zero; store_acc
    # adds that delta to global out. Therefore output reload after store is safe
    # even in scatter mode: we never use global out[] as spill memory.

    mode_scalar_y = mode == "u,u,,u"
    reg_count = _max_lars_reg_count_cse_aware(schedule_result)
    resident_out_indices = sorted({
        int(inst.args[0])
        for inst in schedule_result.instructions
        if inst.op in ("fma_u1d_resident", "fma_u1d_pair_resident")
    })
    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    scalar_t* __restrict__ out,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
        ap("")
    ap("    const int e_orig = b_list ? b_list[e_local] : e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    if use_x_src:
        ap("    const int x_row = src_idx[e_orig];")
    else:
        ap("    const int x_row = e_local;")
    if use_y_src:
        ap("    const int y_row = src_idx[e_orig];")
    else:
        ap("    const int y_row = e_orig;")
    if use_scatter:
        ap("    const int out_row = dst_idx[e_orig];")
    else:
        ap("    const int out_row = e_orig;")
    ap("")
    ap("    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;")
    ap("    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;")
    if mode_scalar_y:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky;")
    else:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky * (index_t)U;")
    ap("    const index_t out_base = (index_t)out_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    for out_idx in resident_out_indices:
        ap(f"            scalar_t out_acc_v_{out_idx} = scalar_t(0);")
    if resident_out_indices:
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op == "load":
            reg, ref = inst.args
            kind, idx = _parse_lars_label_ref(ref)
            if kind == "o":
                raise ValueError("load must not be used for output labels; expected load_acc")
            expr = _lars_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                w_dim=w_dim,
                v_dim=v_dim,
            )
            ap(f"            {reg} = {kind}[{expr}];")

        elif inst.op == "load_acc":
            reg, ref = inst.args
            kind, idx = _parse_lars_label_ref(ref)
            if kind != "o":
                raise ValueError("load_acc expects an output label")
            expr = _lars_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                w_dim=w_dim,
                v_dim=v_dim,
            )
            # Do NOT read global out here.
            # The logical output accumulator is a local delta initialized to 0.
            ap(f"            {reg} = scalar_t(0);")

        elif inst.op == "fma_u1d_resident":
            out_idx, rx, ry, rw, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            # Output accumulators are full-resident local variables.
            ap(f"            out_acc_v_{int(out_idx)} += scalar_t({c}) * ({rw} * {rx}) * {ry};")

        elif inst.op == "fma_u1d":
            # Backward-compatible support for older schedules that kept output
            # accumulators inside the LARS register file.
            ro, rx, ry, rw, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            # Match the baseline emitter's multiplication association as closely
            # as possible: pair = w * x, then coeff * pair * y.
            ap(f"            {ro} += scalar_t({c}) * ({rw} * {rx}) * {ry};")

        elif inst.op == "store_acc":
            ref, reg = inst.args
            kind, idx = _parse_lars_label_ref(ref)
            if kind != "o":
                raise ValueError("store_acc expects an output label")
            expr = _lars_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                x_dim=x_dim,
                y_dim=y_dim,
                w_dim=w_dim,
                v_dim=v_dim,
            )
            if use_scatter:
                ap(f"            atomicAdd(&out[{expr}], {reg});")
            else:
                # LARS may flush the same output accumulator multiple times
                # when the register budget is tight, so non-scatter must also
                # accumulate partial deltas instead of overwriting.
                ap(f"            out[{expr}] += {reg};")

        elif inst.op == "pair_cse":
            # Scheduler-level CSE instruction:
            #   pair_reg = rw * rx
            # The scheduler owns the pair_reg lifetime and counts it against
            # the virtual register budget.
            pair_reg, rw, rx, pair_name = inst.args
            ap(f"            {pair_reg} = {rw} * {rx};")

        elif inst.op == "fma_u1d_pair_resident":
            # FMA using a scheduler-created pair register and a full-resident
            # output accumulator.
            out_idx, pair_reg, ry, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            ap(f"            out_acc_v_{int(out_idx)} += scalar_t({c}) * {pair_reg} * {ry};")

        elif inst.op == "fma_u1d_pair":
            # Backward-compatible support for older schedules.
            ro, pair_reg, ry, coeff = inst.args
            c = _fmt_lars_float(float(coeff))
            ap(f"            {ro} += scalar_t({c}) * {pair_reg} * {ry};")

        elif inst.op == "release_pair":
            # Logical lifetime marker only. No CUDA statement is needed because
            # virtual register reuse is already represented by later load/pair_cse
            # instructions assigning the same rN variable.
            pass

        elif inst.op == "release":
            # Non-output registers are simply allowed to die; the virtual register
            # name may be reused by later generated instructions.
            pass
        else:
            raise ValueError(f"Unsupported LARS instruction op: {inst.op}")

    if resident_out_indices:
        ap("")
        ap("            // resident output accumulator writeback")
    for out_idx in resident_out_indices:
        expr = _lars_label_index_expr(
            "o", int(out_idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=x_dim,
            y_dim=y_dim,
            w_dim=w_dim,
            v_dim=v_dim,
        )
        if use_scatter:
            ap(f"            atomicAdd(&out[{expr}], out_acc_v_{out_idx});")
        else:
            ap(f"            out[{expr}] += out_acc_v_{out_idx};")
    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32(a, b)) return false;")
    ap("    return mul_fits_int32(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_fwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t out_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool out_ok = mul3_fits_int32(out_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && out_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        w, x, y, out,")
    ap("        src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_fwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    ap("            w, x, y, out, src_idx, dst_idx, b_list,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    ap("            w, x, y, out, src_idx, dst_idx, b_list,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    ap("        w, x, y, out, src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


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



def _normalize_reg_budget_candidates(
    reg_budget: Union[int, Iterable[int]],
) -> Tuple[List[int], bool]:
    """
    Normalize a single register budget or a list of budgets.

    Returns:
      budgets: unique positive budgets in input order
      is_multi_budget: True if input describes multiple budgets
    """
    if isinstance(reg_budget, int):
        if reg_budget < 4:
            raise ValueError(f"reg_budget must be at least 4, got {reg_budget}")
        return [int(reg_budget)], False

    budgets: List[int] = []
    seen = set()
    for rb in reg_budget:
        rb = int(rb)
        if rb < 4:
            raise ValueError(f"reg_budget must be at least 4, got {rb}")
        if rb not in seen:
            seen.add(rb)
            budgets.append(rb)

    if not budgets:
        raise ValueError("reg_budget candidate list must not be empty")

    return budgets, len(budgets) > 1


def _fold_reg_budget_candidates_by_max_live(
    reg_budget: Union[int, Iterable[int]],
    *,
    lars_paths: List[Tuple[int, int, int, int, float]],
    consider_cse: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = True,
) -> Tuple[List[int], Dict[str, int]]:
    """
    Remove register-budget candidates that cannot change the generated schedule.

    The old folding rule used only no-CSE max_live.  That is too aggressive for
    auto-CSE: if no-CSE max_live=6, then reg_budget=8 has only 2 spare registers
    for pair CSE, while reg_budget=12 has 6 spare registers and may generate a
    different CSE schedule.  Therefore the equivalence threshold is:

        effective_live_need = no_cse_max_live_labels + probed_max_cse_live_pairs

    where probed_max_cse_live_pairs is measured by running an auto-CSE probe at
    the largest candidate budget.  We keep every budget <= effective_live_need
    and only fold budgets larger than that to the first over-threshold budget.

    Example:
        reg_budget=[8, 12, 16, 20, 24, 28]
        no_cse_max_live=6, max_cse_live_pairs=8
        effective_live_need=14
        -> keep [8, 12, 16]
    """
    budgets, is_multi_budget = _normalize_reg_budget_candidates(reg_budget)
    if not is_multi_budget:
        return budgets, {
            "no_cse_max_live": 0,
            "max_cse_live_pairs": 0,
            "effective_live_need": 0,
            "probe_budget": int(budgets[0]),
            "auto_fallback_threshold": 0,
        }

    probe_budget = max(budgets)

    # Probe normal label pressure without CSE.
    no_cse_probe = LARSUniform1DScheduler(
        paths=lars_paths,
        reg_budget=probe_budget,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        # Probe the natural high-budget schedule. Fallback is disabled here so
        # the probe measures max_live + max CSE capacity rather than forcing a
        # low-budget path-directed order.
        prefer_path_fallback_when_full=False,
        profile=profile,
        profile_name=f"budget_probe_nocse_r{probe_budget}",
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    no_cse_result = no_cse_probe.schedule()
    no_cse_max_live = int(getattr(no_cse_probe, "max_live_labels", no_cse_result.max_live))

    max_cse_live_pairs = 0
    if consider_cse:
        # Probe opportunistic pair-CSE pressure using the largest candidate.  This
        # estimates how many pair temporaries can be simultaneously useful when
        # the budget is not the limiting factor within the user's candidate set.
        cse_probe = CSELARSUniform1DScheduler(
            paths=lars_paths,
            reg_budget=probe_budget,
            enable_cse=True,
            enable_secondary_affinity=enable_secondary_affinity,
            topk_candidates=topk_candidates,
            prefer_path_fallback_when_full=False,
            profile=profile,
            profile_name=f"budget_probe_cse_r{probe_budget}",
            profile_interval=profile_interval,
            profile_seconds=profile_seconds,
            profile_print=profile_print,
        )
        cse_probe.schedule()
        max_cse_live_pairs = int(getattr(cse_probe, "max_live_pairs", 0))

    effective_live_need = int(no_cse_max_live + max_cse_live_pairs)

    kept: List[int] = []
    added_first_over = False
    for rb in budgets:
        if rb <= effective_live_need:
            kept.append(rb)
        elif not added_first_over:
            kept.append(rb)
            added_first_over = True

    return kept, {
        "no_cse_max_live": int(no_cse_max_live),
        "max_cse_live_pairs": int(max_cse_live_pairs),
        "effective_live_need": int(effective_live_need),
        "probe_budget": int(probe_budget),
        "auto_fallback_threshold": int(effective_live_need),
    }

def _fold_lars_configs_by_reg_budget(
    configs: List[Dict[str, Any]],
    *,
    effective_live_need: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Apply budget folding to explicit config lists while preserving distinct
    non-budget variants for the kept budgets.
    """
    if effective_live_need is None:
        return configs

    kept_budgets: Set[int] = set()
    first_over: Optional[int] = None
    ordered_budgets: List[int] = []
    for cfg in configs:
        rb = int(cfg["reg_budget"])
        if rb not in ordered_budgets:
            ordered_budgets.append(rb)

    for rb in ordered_budgets:
        if rb <= effective_live_need:
            kept_budgets.add(rb)
        elif first_over is None:
            first_over = rb
            kept_budgets.add(rb)

    return [cfg for cfg in configs if int(cfg["reg_budget"]) in kept_budgets]

def default_lars_cse_candidate_configs(
    reg_budget: Union[int, Iterable[int]],
) -> List[Dict[str, Any]]:
    """
    Build the automatic LARS+CSE candidate configs.

    There is no explicit config list and no no-CSE baseline generation here:
    each kept register budget produces exactly one candidate, using auto CSE.
    Pair CSE itself is opportunistic: a repeated (w, x) pair is materialized
    only when the current schedule has a free virtual register.
    """
    budgets, _ = _normalize_reg_budget_candidates(reg_budget)
    return [
        {
            "name": f"lars_r{rb}_cse_auto",
            "reg_budget": int(rb),
            "enable_cse": False,
        }
        for rb in budgets
    ]


def generate_code_uniform1d_fwd_with_scheduler(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    mode: str = "u,u,,u",
    out_path: str = "generated_uniform1d_fwd_lars_cse.cu",
    kernel_name: str = "stp_codegen_lars",
    scalar_t: str = "float",
    reg_budget: Union[int, Iterable[int]] = 16,
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    fold_reg_budgets_by_max_live: bool = True,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = False,
    consider_cse: bool = True,

    # Shared scheduler defaults.
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
):
    """
    Generate one or more LARS forward CUDA implementations with scheduler-level
    pair CSE candidates.

    Return forms:
      - if a single automatic candidate is produced and return_schedule=False:
            code: str
      - if multiple automatic candidates are produced and return_schedule=False:
            [(candidate_name, code), ...]
      - if return_schedule=True:
            [{"name", "code", "schedule", "config", "profile"}, ...]

    Candidate generation is automatic only: one auto-CSE candidate is produced
    for each kept reg_budget. No explicit candidate-list parameter and no no-CSE
    baseline are exposed/generated by this entry.

    Profile options:
      - profile=True prints scheduling progress for each candidate.
      - each profile line includes done/remaining path counts, live registers,
        free registers, fireable/candidate counts, spills/reloads, instruction
        count, and CSE pair statistics.
      - return_schedule=True also returns the structured profile summary in
        result["profile"].
    """
    del scalar_t

    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    lars_paths = _make_lars_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )

    budget_fold_info: Dict[str, int] = {}

    # Always probe when a budget list is supplied so fallback can be decided
    # from effective_live_need = max_live_labels + max_live_pairs.  Folding then
    # optionally removes budgets that cannot expose more label/CSE liveness.
    probed_budgets, budget_fold_info = _fold_reg_budget_candidates_by_max_live(
        reg_budget,
        lars_paths=lars_paths,
        consider_cse=consider_cse,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )

    if fold_reg_budgets_by_max_live:
        effective_reg_budget = probed_budgets
    else:
        effective_reg_budget, _ = _normalize_reg_budget_candidates(reg_budget)

    effective_live_need = int(budget_fold_info.get("effective_live_need", 0))
    auto_cse_configs = default_lars_cse_candidate_configs(effective_reg_budget)

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"

    candidates = []

    for cfg_in in auto_cse_configs:
        cfg = dict(cfg_in)
        rb = int(cfg.get("reg_budget", reg_budget if isinstance(reg_budget, int) else list(reg_budget)[0]))
        if rb < 4:
            raise ValueError(f"reg_budget must be at least 4, got {rb}")

        enable_cse = bool(cfg.get("enable_cse", True))

        cand_name = str(cfg.get(
            "name",
            f"lars_r{rb}_" + ("cse_auto" if enable_cse else "nocse")
        ))

        # Automatic fallback policy:
        #   rb < effective_live_need  -> path-directed fallback when full
        #   rb >= effective_live_need -> global LARS only, no full-reg fallback
        # If there is only one budget and no meaningful probe threshold, keep
        # fallback disabled; the global selector still has its no-fire deadlock
        # fallback for correctness.
        auto_fallback_when_full = bool(effective_live_need > 0 and rb < effective_live_need)
        cfg["auto_fallback_when_full"] = auto_fallback_when_full
        cfg["effective_live_need"] = effective_live_need

        scheduler = CSELARSUniform1DScheduler(
            paths=lars_paths,
            reg_budget=rb,
            enable_cse=enable_cse,
            enable_secondary_affinity=bool(cfg.get("enable_secondary_affinity", enable_secondary_affinity)),
            topk_candidates=cfg.get("topk_candidates", topk_candidates),
            prefer_path_fallback_when_full=auto_fallback_when_full,
            profile=bool(cfg.get("profile", profile)),
            profile_name=str(cfg.get("profile_name", cand_name)),
            profile_interval=int(cfg.get("profile_interval", profile_interval)),
            profile_seconds=float(cfg.get("profile_seconds", profile_seconds)),
            profile_print=bool(cfg.get("profile_print", profile_print)),
        )
        schedule_result = scheduler.schedule()

        if kernel_name and kernel_name != "stp_codegen_lars":
            base_kernel_name = f"{kernel_name}_{cand_name}"
        else:
            safe_cand = cand_name.replace("-", "_").replace(".", "_")
            base_kernel_name = (
                f"uniform1d_{safe_cand}_u{u_dim}_path{P}_"
                f"{mode_str}_{layout_tag}_fwd"
            )

        code = emit_fused_fwd_kernel_from_lars_schedule(
            schedule_result,
            kernel_name=base_kernel_name,
            mode=mode,
            use_x_src=use_x_src,
            use_y_src=use_y_src,
            use_scatter=use_scatter,
            u_dim=u_dim,
            x_dim=None,
            y_dim=None,
            w_dim=None,
            v_dim=None,
            block_size=32,
        )

        code = code + "\n" + emit_launcher(
            bundle_name=base_kernel_name,
            mode=mode,
            use_x_src=use_x_src,
            use_y_src=use_y_src,
            use_scatter=use_scatter,
        )

        if out_path:
            out_p = Path(out_path)
            if len(auto_cse_configs) > 1:
                stem = out_p.stem
                suffix = out_p.suffix or ".cu"
                candidate_path = out_p.with_name(f"{stem}_{cand_name}{suffix}")
                candidate_path.write_text(code, encoding="utf-8")
            else:
                out_p.write_text(code, encoding="utf-8")

        if return_schedule:
            candidates.append({
                "name": cand_name,
                "code": code,
                "schedule": schedule_result,
                "config": cfg,
                "budget_fold_info": dict(budget_fold_info),
                "budget_fold_max_live": budget_fold_info.get("no_cse_max_live"),
                "budget_fold_max_cse": budget_fold_info.get("max_cse_live_pairs"),
                "budget_fold_effective_live_need": budget_fold_info.get("effective_live_need"),
                "auto_fallback_when_full": auto_fallback_when_full,
                "profile": schedule_result.profile,
                "cse_stats": {
                    "pair_creates": getattr(scheduler, "cse_pair_creates", 0),
                    "pair_hits": getattr(scheduler, "cse_pair_hits", 0),
                    "pair_releases": getattr(scheduler, "cse_pair_releases", 0),
                    "pair_drops_for_label_release": getattr(scheduler, "cse_pair_drops_for_label_release", 0),
                    "pair_drops_for_reg_pressure": getattr(scheduler, "cse_pair_drops_for_reg_pressure", 0),
                },
            })
        else:
            candidates.append((cand_name, code))

    if len(candidates) == 1:
        item = candidates[0]
        if return_schedule:
            return item
        return item[1]

    return candidates

# =============================================================================
# LARS backward scheduler/codegen
# =============================================================================

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
        # spill/reload candidates.
        return (
            ("w", self.i),
            ("x", self.j),
            ("y", self.k),
            ("go", self.v),
        )


class LARSUniform1DBwdScheduler(LARSUniform1DScheduler):
    """
    LARS scheduler for fused Uniform1D backward.

    It reuses the forward LARS machinery, but a path now has four input labels
    (w, x, y, grad_out) and two or three output accumulators
    (grad_x, grad_y, and optionally grad_w).
    """

    output_kinds = BWD_OUTPUT_KINDS

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        reg_budget: int = 24,
        *,
        need_grad_w: bool = True,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        path_fallback_after: Optional[int] = None,
        prefer_path_fallback_when_full: bool = True,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        self.need_grad_w = bool(need_grad_w)
        self.paths: List[U1DBwdPath] = [
            U1DBwdPath(pid=p, i=i, j=j, k=k, v=v, c=c, need_grad_w=self.need_grad_w)
            for p, (i, j, k, v, c) in enumerate(paths)
        ]

        self.reg_budget = int(reg_budget)
        min_budget = 4
        if self.reg_budget < min_budget:
            raise ValueError(f"backward reg_budget must be at least {min_budget}, got {self.reg_budget}")

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
        self.prefer_path_fallback_when_full = bool(prefer_path_fallback_when_full)
        self.path_fallback_after = (
            int(path_fallback_after)
            if path_fallback_after is not None
            else max(8, 2 * self.reg_budget)
        )
        self.debug = bool(debug)

        self.label_to_paths: Dict[Label, Set[int]] = defaultdict(set)
        for path in self.paths:
            for lab in path.labels:
                self.label_to_paths[lab].add(path.pid)

        self.path_label_sets: List[Set[Label]] = [set(p.labels) for p in self.paths]
        self.unscheduled: Set[int] = set(p.pid for p in self.paths)

        self.remaining_uses: Counter[Label] = Counter()
        for p in self.paths:
            for lab in p.labels:
                self.remaining_uses[lab] += 1

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = [f"r{r}" for r in range(self.reg_budget)]

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}
        self._no_progress_iters = 0
        self._last_done = 0

        self.profile_enabled = bool(profile)
        self.profile_name = str(profile_name or self.__class__.__name__)
        self.profile_interval = int(profile_interval)
        self.profile_seconds = float(profile_seconds)
        self.profile_print = bool(profile_print)
        self._profile_t0 = perf_counter()
        self._profile_last_print_t = self._profile_t0
        self._profile_last_print_done = 0
        self._profile_loop_iter = 0
        self._profile_select_rounds = 0
        self._profile_fire_rounds = 0
        self._profile_load_rounds = 0
        self._profile_fallback_rounds = 0
        self._profile_records: List[Dict[str, Any]] = []
        self._profile_last_event = ""
        self._profile_last_reason = ""

    def _path(self, pid: int) -> U1DBwdPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        if kind == "go":
            return f"grad_out[{idx}]"
        if kind == "gw":
            return f"grad_w[{idx}]"
        if kind == "gx":
            return f"grad_x[{idx}]"
        if kind == "gy":
            return f"grad_y[{idx}]"
        return f"{kind}[{idx}]"

    def _is_output_label(self, lab: Label) -> bool:
        return lab[0] in self.output_kinds

    def _emit_load_inst(self, lab: Label, reg: str, reason: str) -> None:
        if self._is_output_label(lab):
            self.instructions.append(
                Inst("load_acc", (reg, self._label_name(lab)), reason)
            )
        else:
            self.instructions.append(
                Inst("load", (reg, self._label_name(lab)), reason)
            )
        self.reloads += 1

    def _store_and_release(self, lab: Label, reason: str = "") -> None:
        if lab not in self.live:
            return

        reg = self.reg_of[lab]
        if self._is_output_label(lab):
            if lab in self.dirty_outputs:
                self.instructions.append(
                    Inst("store_acc", (self._label_name(lab), reg), reason)
                )
                self.dirty_outputs.discard(lab)
            else:
                self.instructions.append(
                    Inst("release", (reg, self._label_name(lab)), reason)
                )
        else:
            self.instructions.append(
                Inst("release", (reg, self._label_name(lab)), reason)
            )

        self.live.remove(lab)
        del self.reg_of[lab]
        self.free_regs.insert(0, reg)
        self._invalidate_score_cache()

    def _choose_spill_victim_avoid(self, avoid: Set[Label]) -> Label:
        candidates = [lab for lab in self.live if lab not in avoid]
        if not candidates:
            candidates = list(self.live)
        if not candidates:
            raise RuntimeError("No live label to spill.")

        def victim_key(lab: Label):
            remain = self.remaining_uses[lab]
            dirty_output_penalty = 1 if self._is_output_label(lab) and lab in self.dirty_outputs else 0
            output_penalty = 1 if self._is_output_label(lab) else 0
            return (remain, dirty_output_penalty, output_penalty, str(lab))

        return min(candidates, key=victim_key)

    def _priority(self, lab: Label) -> int:
        acc_bonus = 1 if self._is_output_label(lab) else 0
        return self.remaining_uses[lab] + acc_bonus

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)

        lw = ("w", p.i)
        lx = ("x", p.j)
        ly = ("y", p.k)
        lgo = ("go", p.v)

        rw = self.reg_of[lw]
        rx = self.reg_of[lx]
        ry = self.reg_of[ly]
        rgo = self.reg_of[lgo]

        self.instructions.append(
            Inst(
                "bwd_fma_resident",
                (p.i, p.j, p.k, rw, rx, ry, rgo, p.c, self.need_grad_w),
                (
                    f"path#{pid}: "
                    f"gw[{p.i}] += go[{p.v}]*x[{p.j}]*y[{p.k}], "
                    f"gx[{p.j}] += w[{p.i}]*go[{p.v}]*y[{p.k}], "
                    f"gy[{p.k}] += w[{p.i}]*go[{p.v}]*x[{p.j}]"
                ),
            )
        )

        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        for lab in p.labels:
            self.remaining_uses[lab] -= 1

        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")


BwdPairKey = Tuple[Label, Label]  # (w-label, grad_out-label)


class CSELARSUniform1DBwdScheduler(LARSUniform1DBwdScheduler):
    """
    Backward LARS scheduler with opportunistic pair CSE.

    The default backward pair is:
        wg = w[i] * grad_out[v]

    This pair is used by both grad_x and grad_y for the same path and can also
    be reused across paths sharing the same (w, grad_out) pair.
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        reg_budget: int = 24,
        *,
        enable_cse: bool = True,
        cse_release_pair_before_label_spill: bool = True,
        **kwargs,
    ):
        super().__init__(paths=paths, reg_budget=reg_budget, **kwargs)
        self.enable_cse = bool(enable_cse)
        self.cse_release_pair_before_label_spill = bool(cse_release_pair_before_label_spill)

        self.initial_pair_uses: Counter[BwdPairKey] = Counter()
        for p in self.paths:
            self.initial_pair_uses[self._pair_key_for_path_obj(p)] += 1

        self.remaining_pair_uses: Counter[BwdPairKey] = Counter(self.initial_pair_uses)
        self.cse_candidate_pairs: Set[BwdPairKey] = {
            key for key, cnt in self.initial_pair_uses.items()
            if cnt >= 2
        }

        self.pair_reg_of: Dict[BwdPairKey, str] = {}
        self.pair_key_of_reg: Dict[str, BwdPairKey] = {}

        self.cse_pair_creates = 0
        self.cse_pair_hits = 0
        self.cse_pair_releases = 0
        self.cse_pair_drops_for_label_release = 0
        self.cse_pair_drops_for_reg_pressure = 0

    def _pair_key_for_path_obj(self, p: U1DBwdPath) -> BwdPairKey:
        return (("w", p.i), ("go", p.v))

    def _pair_name(self, key: BwdPairKey) -> str:
        lw, lgo = key
        return f"pair[{self._label_name(lw)}*{self._label_name(lgo)}]"

    def _total_live_regs(self) -> int:
        return len(self.live) + len(self.pair_reg_of)

    def _update_max_live_total(self) -> None:
        self.max_live_labels = max(self.max_live_labels, len(self.live))
        self.max_live_pairs = max(self.max_live_pairs, len(self.pair_reg_of))
        self.max_live_total = max(self.max_live_total, self._total_live_regs())
        self.max_live = max(self.max_live, self._total_live_regs())

    def _alloc_reg_no_spill(self, lab: Label) -> str:
        reg = super()._alloc_reg_no_spill(lab)
        self._update_max_live_total()
        return reg

    def _choose_pair_victim(self) -> Optional[BwdPairKey]:
        if not self.pair_reg_of:
            return None

        def key_fn(pair_key: BwdPairKey):
            return (
                self.remaining_pair_uses[pair_key],
                self.initial_pair_uses[pair_key],
                self._pair_name(pair_key),
            )

        return min(self.pair_reg_of.keys(), key=key_fn)

    def _release_pair(self, pair_key: BwdPairKey, reason: str = "") -> None:
        reg = self.pair_reg_of.pop(pair_key, None)
        if reg is None:
            return
        self.pair_key_of_reg.pop(reg, None)
        self.instructions.append(
            Inst("release_pair", (reg, self._pair_name(pair_key)), reason)
        )
        self.free_regs.insert(0, reg)
        self.cse_pair_releases += 1
        self._invalidate_score_cache()

    def _release_pairs_touching_label(self, lab: Label, reason: str = "") -> None:
        to_release = [key for key in self.pair_reg_of if lab in key]
        for key in to_release:
            self.cse_pair_drops_for_label_release += 1
            self._release_pair(
                key,
                reason=f"drop pair before releasing {self._label_name(lab)}; {reason}",
            )

    def _store_and_release(self, lab: Label, reason: str = "") -> None:
        self._release_pairs_touching_label(lab, reason=reason)
        super()._store_and_release(lab, reason=reason)
        self._update_max_live_total()

    def _load_label_with_victim(
        self,
        lab: Label,
        victim: Optional[Label],
        reason: str = "",
    ) -> None:
        if lab in self.live:
            return

        if (
            not self.free_regs
            and self.cse_release_pair_before_label_spill
            and self.pair_reg_of
        ):
            pair_victim = self._choose_pair_victim()
            if pair_victim is not None:
                self.cse_pair_drops_for_reg_pressure += 1
                self._release_pair(
                    pair_victim,
                    reason=f"drop pair under reg pressure before loading {self._label_name(lab)}",
                )

        super()._load_label_with_victim(lab, victim, reason=reason)
        self._update_max_live_total()

    def _should_cache_pair(self, pair_key: BwdPairKey) -> bool:
        if not self.enable_cse:
            return False
        if pair_key not in self.cse_candidate_pairs:
            return False
        if pair_key in self.pair_reg_of:
            return True
        if not self.free_regs:
            return False
        if self.remaining_pair_uses[pair_key] < 2:
            return False
        return True

    def _ensure_pair_cached(self, pair_key: BwdPairKey) -> Optional[str]:
        if not self._should_cache_pair(pair_key):
            return None

        reg = self.pair_reg_of.get(pair_key)
        if reg is not None:
            self.cse_pair_hits += 1
            return reg

        lw, lgo = pair_key
        if lw not in self.live or lgo not in self.live:
            return None
        if not self.free_regs:
            return None

        pair_reg = self.free_regs.pop(0)
        self.pair_reg_of[pair_key] = pair_reg
        self.pair_key_of_reg[pair_reg] = pair_key
        self._update_max_live_total()
        self._invalidate_score_cache()

        rw = self.reg_of[lw]
        rgo = self.reg_of[lgo]
        self.instructions.append(
            Inst(
                "pair_cse",
                (pair_reg, rw, rgo, self._pair_name(pair_key)),
                f"create {self._pair_name(pair_key)} reuse_left={self.remaining_pair_uses[pair_key]}",
            )
        )
        self.cse_pair_creates += 1
        return pair_reg

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)

        lw = ("w", p.i)
        lx = ("x", p.j)
        ly = ("y", p.k)
        lgo = ("go", p.v)

        rw = self.reg_of[lw]
        rx = self.reg_of[lx]
        ry = self.reg_of[ly]
        rgo = self.reg_of[lgo]

        pair_key = (lw, lgo)
        pair_reg = self._ensure_pair_cached(pair_key)

        if pair_reg is not None:
            self.instructions.append(
                Inst(
                    "bwd_fma_wg_pair_resident",
                    (p.i, p.j, p.k, pair_reg, rx, ry, rgo, p.c, self.need_grad_w),
                    (
                        f"path#{pid}: use {self._pair_name(pair_key)}; "
                        f"gx[{p.j}] += wg*y[{p.k}], gy[{p.k}] += wg*x[{p.j}]"
                    ),
                )
            )
        else:
            self.instructions.append(
                Inst(
                    "bwd_fma_resident",
                    (p.i, p.j, p.k, rw, rx, ry, rgo, p.c, self.need_grad_w),
                    (
                        f"path#{pid}: "
                        f"gw[{p.i}] += go[{p.v}]*x[{p.j}]*y[{p.k}], "
                        f"gx[{p.j}] += w[{p.i}]*go[{p.v}]*y[{p.k}], "
                        f"gy[{p.k}] += w[{p.i}]*go[{p.v}]*x[{p.j}]"
                    ),
                )
            )

        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        self.remaining_pair_uses[pair_key] -= 1
        if self.remaining_pair_uses[pair_key] <= 0 and pair_key in self.pair_reg_of:
            self._release_pair(pair_key, reason=f"last pair use after path#{pid}")

        for lab in p.labels:
            self.remaining_uses[lab] -= 1

        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")


def _parse_bwd_lars_label_ref(ref: str) -> Tuple[str, int]:
    if not isinstance(ref, str) or "[" not in ref or not ref.endswith("]"):
        raise ValueError(f"Bad LARS backward label ref: {ref!r}")
    kind, rest = ref.split("[", 1)
    idx = int(rest[:-1])
    aliases = {
        "grad_out": "go",
        "grad_w": "gw",
        "grad_x": "gx",
        "grad_y": "gy",
    }
    kind = aliases.get(kind, kind)
    if kind not in ("w", "x", "y", "go", "gw", "gx", "gy"):
        raise ValueError(f"Bad backward label kind in ref: {ref!r}")
    return kind, idx


def _bwd_label_index_expr(
    kind: str,
    idx: int,
    *,
    mode_scalar_y: bool,
    u_dim: Optional[int],
    iw_dim: Optional[int],
    ix_dim: Optional[int],
    ky_dim: Optional[int],
    v_dim: Optional[int],
) -> str:
    if kind == "w":
        if iw_dim is not None and idx >= iw_dim:
            raise ValueError(f"w index {idx} out of iw_dim={iw_dim}")
        if u_dim is not None:
            return f"w_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"w_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "gw":
        if iw_dim is not None and idx >= iw_dim:
            raise ValueError(f"grad_w index {idx} out of iw_dim={iw_dim}")
        if u_dim is not None:
            return f"gw_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"gw_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "x":
        if ix_dim is not None and idx >= ix_dim:
            raise ValueError(f"x index {idx} out of ix_dim={ix_dim}")
        if u_dim is not None:
            return f"x_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"x_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "gx":
        if ix_dim is not None and idx >= ix_dim:
            raise ValueError(f"grad_x index {idx} out of ix_dim={ix_dim}")
        if u_dim is not None:
            return f"gx_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"gx_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "y":
        if ky_dim is not None and idx >= ky_dim:
            raise ValueError(f"y index {idx} out of ky_dim={ky_dim}")
        if mode_scalar_y:
            return f"y_base + (index_t){idx}"
        if u_dim is not None:
            return f"y_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"y_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "gy":
        if ky_dim is not None and idx >= ky_dim:
            raise ValueError(f"grad_y index {idx} out of ky_dim={ky_dim}")
        if mode_scalar_y:
            return f"gy_base + (index_t){idx}"
        if u_dim is not None:
            return f"gy_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"gy_base + (index_t){idx} * (index_t)U + (index_t)u"

    if kind == "go":
        if v_dim is not None and idx >= v_dim:
            raise ValueError(f"grad_out index {idx} out of v_dim={v_dim}")
        if u_dim is not None:
            return f"go_base + (index_t){int(idx) * int(u_dim)} + (index_t)u"
        return f"go_base + (index_t){idx} * (index_t)U + (index_t)u"

    raise ValueError(f"Bad backward label kind: {kind}")


def emit_fused_bwd_kernel_from_lars_schedule(
    schedule_result: ScheduleResult,
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
    acc_reg_budget: Optional[int] = None,
    smem_acc_volatile: bool = False,
) -> str:
    """
    Emit CUDA/HIP-compatible fused backward code from a backward LARS schedule.

    Supported instruction ops:
      - load/load_acc/release/store_acc
      - pair_cse/release_pair
      - bwd_fma
      - bwd_fma_wg_pair
    """
    if block_size != 32:
        raise ValueError("this emitter currently assumes block_size=32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    reg_count = _max_lars_reg_count_cse_aware(schedule_result)

    # ------------------------------------------------------------------
    # Mixed accumulator placement for backward gradients.
    #   acc_reg_budget=None: old behavior, keep every gw/gx/gy accumulator
    #                        as a scalar local variable/register.
    #   acc_reg_budget=K:    keep only the top-K hottest accumulators in
    #                        registers and demote the rest to per-thread
    #                        shared-memory accumulators.  Demoted gx/gy still
    #                        write back once at the end, preserving atomic
    #                        coalescing compared with path-level atomicAdd.
    # ------------------------------------------------------------------
    gw_use: Counter[int] = Counter()
    gx_use: Counter[int] = Counter()
    gy_use: Counter[int] = Counter()
    first_seen: Dict[Tuple[str, int], int] = {}

    def _note_acc(kind: str, idx: int, inst_id: int) -> None:
        first_seen.setdefault((kind, int(idx)), int(inst_id))

    for inst_id, inst in enumerate(schedule_result.instructions):
        if inst.op in ("bwd_fma_resident", "bwd_fma_wg_pair_resident"):
            wi, xj, yk = map(int, inst.args[:3])
            if need_grad_w:
                gw_use[wi] += 1
                _note_acc("gw", wi, inst_id)
            gx_use[xj] += 1
            gy_use[yk] += 1
            _note_acc("gx", xj, inst_id)
            _note_acc("gy", yk, inst_id)

    all_accs: Set[Tuple[str, int]] = set()
    if need_grad_w:
        all_accs |= {("gw", int(i)) for i in gw_use}
    all_accs |= {("gx", int(j)) for j in gx_use}
    all_accs |= {("gy", int(k)) for k in gy_use}

    def _acc_use_count(acc: Tuple[str, int]) -> int:
        kind, idx = acc
        if kind == "gw":
            return int(gw_use.get(idx, 0))
        if kind == "gx":
            return int(gx_use.get(idx, 0))
        if kind == "gy":
            return int(gy_use.get(idx, 0))
        return 0

    def _acc_score(acc: Tuple[str, int]) -> Tuple[float, int, int, int, str, int]:
        # gx/gy have high value because register/shared accumulation reduces
        # global atomicAdd count.  gy is slightly favored in scalar-y mode
        # because writeback also needs a warp reduction.
        kind, idx = acc
        cnt = _acc_use_count(acc)
        if kind == "gy":
            weight = 5.0 if mode_scalar_y else 4.0
            kind_rank = 3
        elif kind == "gx":
            weight = 4.0
            kind_rank = 2
        else:  # gw uses non-atomic writeback, so its register value is lower.
            weight = 2.0
            kind_rank = 1
        return (float(cnt) * weight, cnt, -int(first_seen.get(acc, 10**9)), kind_rank, kind, -idx)

    if acc_reg_budget is None:
        reg_accs: Set[Tuple[str, int]] = set(all_accs)
    else:
        budget = max(0, int(acc_reg_budget))
        ranked = sorted(all_accs, key=_acc_score, reverse=True)
        reg_accs = set(ranked[:budget])

    smem_accs: Set[Tuple[str, int]] = set(all_accs) - set(reg_accs)

    resident_gw_indices: Set[int] = {idx for kind, idx in reg_accs if kind == "gw"}
    resident_gx_indices: Set[int] = {idx for kind, idx in reg_accs if kind == "gx"}
    resident_gy_indices: Set[int] = {idx for kind, idx in reg_accs if kind == "gy"}
    resident_gw_indices_sorted = sorted(resident_gw_indices)
    resident_gx_indices_sorted = sorted(resident_gx_indices)
    resident_gy_indices_sorted = sorted(resident_gy_indices)

    # Stable shared-memory layout: all demoted accumulators are laid out as
    # [acc_id][lane].  Each lane owns its own scalar slot for its current u.
    smem_accs_sorted: List[Tuple[str, int]] = sorted(
        smem_accs,
        key=lambda acc: ({"gw": 0, "gx": 1, "gy": 2}[acc[0]], acc[1]),
    )
    smem_acc_id: Dict[Tuple[str, int], int] = {
        acc: n for n, acc in enumerate(smem_accs_sorted)
    }

    def _is_reg_acc(kind: str, idx: int) -> bool:
        return (kind, int(idx)) in reg_accs

    def _is_smem_acc(kind: str, idx: int) -> bool:
        return (kind, int(idx)) in smem_acc_id

    def _reg_acc_name(kind: str, idx: int) -> str:
        if kind == "gw":
            return f"gw_acc_i_{int(idx)}"
        if kind == "gx":
            return f"gx_acc_j_{int(idx)}"
        if kind == "gy":
            return f"gy_acc_k_{int(idx)}"
        raise ValueError(f"Bad accumulator kind: {kind}")

    def _smem_acc_expr(kind: str, idx: int) -> str:
        sid = smem_acc_id[(kind, int(idx))]
        return f"smem_acc[{sid} * 32 + lane]"

    def _emit_bwd_acc_update(kind: str, idx: int, value_expr: str) -> None:
        if _is_reg_acc(kind, idx):
            ap(f"            {_reg_acc_name(kind, idx)} += {value_expr};")
        elif _is_smem_acc(kind, idx):
            ap(f"            {_smem_acc_expr(kind, idx)} += {value_expr};")
        else:
            # This path is reachable only if need_grad_w=False and kind==gw.
            pass

    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")
    ap("template <typename scalar_t>")
    ap("__device__ __forceinline__ scalar_t warp_sum_xor_lars_bwd(scalar_t v) {")
    ap("    for (int offset = 16; offset > 0; offset >>= 1) {")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("        v += __shfl_xor(v, offset);")
    ap("#else")
    ap("        v += __shfl_xor_sync(0xffffffff, v, offset);")
    ap("#endif")
    ap("    }")
    ap("    return v;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    if need_grad_w:
        ap("    scalar_t* __restrict__ grad_w,")
    ap("    scalar_t* __restrict__ grad_x,")
    ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    if smem_accs_sorted:
        ap("    extern __shared__ __align__(16) unsigned char smem_raw[];")
        if smem_acc_volatile:
            ap("    volatile scalar_t* smem_acc = reinterpret_cast<volatile scalar_t*>(smem_raw);")
        else:
            ap("    scalar_t* smem_acc = reinterpret_cast<scalar_t*>(smem_raw);")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
    ap("    const int e_orig = b_list ? b_list[e_local] : e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e_orig];")
    else:
        ap("    const int src = e_orig;")
    if use_scatter:
        ap("    const int dst = dst_idx[e_orig];")
    else:
        ap("    const int dst = e_orig;")
    ap("")
    ap("    const int x_row = " + ("src;" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src;" if use_y_src else "e_orig;"))
    ap("    const int go_row = " + ("dst;" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base  = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
        if need_grad_w:
            ap(f"    const index_t gw_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base  = (index_t)w_row * (index_t)Iw * (index_t)U;")
        if need_grad_w:
            ap("    const index_t gw_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base  = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
        ap(f"    const index_t gx_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base  = (index_t)x_row * (index_t)Ix * (index_t)U;")
        ap("    const index_t gx_base = (index_t)x_row * (index_t)Ix * (index_t)U;")

    if mode_scalar_y:
        ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky;")
        ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base  = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
            ap(f"    const index_t gy_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky * (index_t)U;")
            ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky * (index_t)U;")

    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t go_base = (index_t)go_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t go_base = (index_t)go_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for sid, (akind, aidx) in enumerate(smem_accs_sorted):
        ap(f"            smem_acc[{sid} * 32 + lane] = scalar_t(0);  // smem {akind}[{aidx}]")
    if smem_accs_sorted:
        ap("")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    for gw_idx in resident_gw_indices_sorted:
        ap(f"            scalar_t gw_acc_i_{gw_idx} = scalar_t(0);")
    for gx_idx in resident_gx_indices_sorted:
        ap(f"            scalar_t gx_acc_j_{gx_idx} = scalar_t(0);")
    for gy_idx in resident_gy_indices_sorted:
        ap(f"            scalar_t gy_acc_k_{gy_idx} = scalar_t(0);")
    if resident_gw_indices_sorted or resident_gx_indices_sorted or resident_gy_indices_sorted:
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op == "load":
            reg, ref = inst.args
            kind, idx = _parse_bwd_lars_label_ref(ref)
            if kind in BWD_OUTPUT_KINDS:
                raise ValueError("load must not be used for backward output labels")
            expr = _bwd_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            arr = "grad_out" if kind == "go" else kind
            ap(f"            {reg} = {arr}[{expr}];")

        elif inst.op == "load_acc":
            reg, ref = inst.args
            kind, _idx = _parse_bwd_lars_label_ref(ref)
            if kind not in BWD_OUTPUT_KINDS:
                raise ValueError("load_acc expects a backward output label")
            ap(f"            {reg} = scalar_t(0);")

        elif inst.op == "bwd_fma_resident":
            wi, xj, yk, rw, rx, ry, rgo, coeff, inst_need_grad_w = inst.args
            c = _fmt_lars_float(float(coeff))
            if bool(inst_need_grad_w):
                _emit_bwd_acc_update("gw", int(wi), f"scalar_t({c}) * {rgo} * {rx} * {ry}")
            _emit_bwd_acc_update("gx", int(xj), f"scalar_t({c}) * ({rw} * {rgo}) * {ry}")
            _emit_bwd_acc_update("gy", int(yk), f"scalar_t({c}) * ({rw} * {rgo}) * {rx}")

        elif inst.op == "bwd_fma":
            # Backward-compatible support for older schedules that kept
            # output accumulators inside the LARS register file.
            rgw, rgx, rgy, rw, rx, ry, rgo, coeff, inst_need_grad_w = inst.args
            c = _fmt_lars_float(float(coeff))
            if bool(inst_need_grad_w):
                ap(f"            {rgw} += scalar_t({c}) * {rgo} * {rx} * {ry};")
            ap(f"            {rgx} += scalar_t({c}) * ({rw} * {rgo}) * {ry};")
            ap(f"            {rgy} += scalar_t({c}) * ({rw} * {rgo}) * {rx};")

        elif inst.op == "pair_cse":
            pair_reg, ra, rb, pair_name = inst.args
            ap(f"            {pair_reg} = {ra} * {rb};")

        elif inst.op == "bwd_fma_wg_pair_resident":
            wi, xj, yk, pair_reg, rx, ry, rgo, coeff, inst_need_grad_w = inst.args
            c = _fmt_lars_float(float(coeff))
            if bool(inst_need_grad_w):
                _emit_bwd_acc_update("gw", int(wi), f"scalar_t({c}) * {rgo} * {rx} * {ry}")
            _emit_bwd_acc_update("gx", int(xj), f"scalar_t({c}) * {pair_reg} * {ry}")
            _emit_bwd_acc_update("gy", int(yk), f"scalar_t({c}) * {pair_reg} * {rx}")

        elif inst.op == "bwd_fma_wg_pair":
            # Backward-compatible support for older schedules.
            rgw, rgx, rgy, pair_reg, rx, ry, rgo, coeff, inst_need_grad_w = inst.args
            c = _fmt_lars_float(float(coeff))
            if bool(inst_need_grad_w):
                ap(f"            {rgw} += scalar_t({c}) * {rgo} * {rx} * {ry};")
            ap(f"            {rgx} += scalar_t({c}) * {pair_reg} * {ry};")
            ap(f"            {rgy} += scalar_t({c}) * {pair_reg} * {rx};")

        elif inst.op == "store_acc":
            ref, reg = inst.args
            kind, idx = _parse_bwd_lars_label_ref(ref)
            expr = _bwd_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            if kind == "gw":
                if not need_grad_w:
                    raise ValueError("schedule stores grad_w but need_grad_w=False")
                # grad_w is owned by the current w_row in this generated
                # backward path, matching the baseline fused implementation:
                # use a plain store rather than atomicAdd.
                ap(f"            grad_w[{expr}] = {reg};")
            elif kind == "gx":
                ap(f"            atomicAdd(&grad_x[{expr}], {reg});")
            elif kind == "gy":
                if mode_scalar_y:
                    ap(f"            scalar_t gy_sum_{inst_id} = warp_sum_xor_lars_bwd({reg});")
                    ap("            if (lane == 0) {")
                    ap(f"                atomicAdd(&grad_y[{expr}], gy_sum_{inst_id});")
                    ap("            }")
                else:
                    ap(f"            atomicAdd(&grad_y[{expr}], {reg});")
            else:
                raise ValueError("store_acc expects a backward output label")

        elif inst.op in ("release", "release_pair"):
            pass
        else:
            raise ValueError(f"Unsupported backward LARS instruction op: {inst.op}")

    if (resident_gw_indices_sorted or resident_gx_indices_sorted or resident_gy_indices_sorted
            or smem_accs_sorted):
        ap("")
        ap("            // mixed register/shared-memory backward accumulator writeback")

    # Register-resident writeback.
    for gw_idx in resident_gw_indices_sorted:
        expr = _bwd_label_index_expr(
            "gw", int(gw_idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        ap(f"            grad_w[{expr}] = gw_acc_i_{gw_idx};")
    for gx_idx in resident_gx_indices_sorted:
        expr = _bwd_label_index_expr(
            "gx", int(gx_idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        ap(f"            atomicAdd(&grad_x[{expr}], gx_acc_j_{gx_idx});")
    for gy_idx in resident_gy_indices_sorted:
        expr = _bwd_label_index_expr(
            "gy", int(gy_idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        if mode_scalar_y:
            ap(f"            scalar_t gy_sum_resident_{gy_idx} = warp_sum_xor_lars_bwd(gy_acc_k_{gy_idx});")
            ap("            if (lane == 0) {")
            ap(f"                atomicAdd(&grad_y[{expr}], gy_sum_resident_{gy_idx});")
            ap("            }")
        else:
            ap(f"            atomicAdd(&grad_y[{expr}], gy_acc_k_{gy_idx});")

    # Shared-memory-resident writeback.
    for kind, idx in smem_accs_sorted:
        expr = _bwd_label_index_expr(
            kind, int(idx),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        smem_expr = _smem_acc_expr(kind, int(idx))
        if kind == "gw":
            if not need_grad_w:
                continue
            ap(f"            grad_w[{expr}] = {smem_expr};")
        elif kind == "gx":
            ap(f"            atomicAdd(&grad_x[{expr}], {smem_expr});")
        elif kind == "gy":
            if mode_scalar_y:
                ap(f"            scalar_t gy_sum_smem_{idx} = warp_sum_xor_lars_bwd({smem_expr});")
                ap("            if (lane == 0) {")
                ap(f"                atomicAdd(&grad_y[{expr}], gy_sum_smem_{idx});")
                ap("            }")
            else:
                ap(f"            atomicAdd(&grad_y[{expr}], {smem_expr});")
        else:
            raise ValueError(f"Bad smem accumulator kind: {kind}")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32_bwd(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32_bwd(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32_bwd(a, b)) return false;")
    ap("    return mul_fits_int32_bwd(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_bwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32_bwd((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32_bwd(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32_bwd(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32_bwd(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool go_ok = mul3_fits_int32_bwd(go_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && go_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    constexpr int kSmemAccCount = {len(smem_accs_sorted)};")
    ap("    size_t smem_bytes = (size_t)kSmemAccCount * 32u * sizeof(scalar_t);")
    ap("#if !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)")
    ap("    if (smem_bytes > 49152) {")
    ap(f"        cudaFuncSetAttribute({kernel_name}<scalar_t, index_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);")
    ap("    }")
    ap("#endif")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, smem_bytes, stream>>>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_bwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                   kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, b_list, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, b_list, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def emit_lars_bwd_launcher(
    bundle_name: str,
    mode: str,
    *,
    need_grad_w: bool,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
) -> str:
    if mode == "u,u,,u":
        y_comment = "[B,Ky,1] or [S,Ky,1]"
        gy_alloc = "    auto grad_y = torch::zeros({y.size(0), Ky, 1}, y.options());"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");'
    elif mode == "u,u,u,u":
        y_comment = "[B,Ky,U] or [S,Ky,U]"
        gy_alloc = "    auto grad_y = torch::zeros_like(y);"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == U, "y.size(2) must equal U for mode u,u,u,u");'
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    param_lines = [
        "    torch::Tensor w",
        "    torch::Tensor x",
        "    torch::Tensor y",
        "    torch::Tensor grad_out",
    ]
    if use_x_src or use_y_src:
        param_lines.append("    torch::Tensor src_idx")
    if use_scatter:
        param_lines.append("    torch::Tensor dst_idx")
        param_lines.append("    torch::Tensor b_list")
    param_lines.append("    int64_t V64")
    params = ",\n".join(param_lines)

    src_check = ""
    if use_x_src or use_y_src:
        src_check = r'''
    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''
    dst_check = ""
    if use_scatter:
        dst_check = r'''
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''

    if use_x_src or use_y_src:
        b_expr = "src_idx.size(0)"
        src_numel_check = '    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");'
    elif use_scatter:
        b_expr = "dst_idx.size(0)"
        src_numel_check = ""
    else:
        b_expr = "grad_out.size(0)"
        src_numel_check = ""

    dst_numel_check = '    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");' if use_scatter else ""

    blist_logic = ""
    if use_scatter:
        blist_logic = r'''
    const int32_t* b_list_ptr = nullptr;
    if (b_list.defined() && b_list.numel() > 0) {
        TORCH_CHECK(b_list.is_cuda(), "b_list must be CUDA/HIP");
        TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list must be int32");
        TORCH_CHECK((int)b_list.numel() == B, "b_list must be [B]");
        b_list_ptr = (const int32_t*)b_list.data_ptr<int32_t>();
    }
'''
    else:
        blist_logic = "    const int32_t* b_list_ptr = nullptr;\n"

    launch_src_arg = "(const int32_t*)src_idx.data_ptr<int32_t>()," if (use_x_src or use_y_src) else "nullptr,"
    launch_dst_arg = "(const int32_t*)dst_idx.data_ptr<int32_t>()," if use_scatter else "nullptr,"

    grad_w_alloc = "    auto grad_w = torch::zeros_like(w);\n" if need_grad_w else ""
    grad_w_launch_arg = "                (scalar_t*)grad_w.data_ptr<scalar_t>(),\n" if need_grad_w else ""
    ret_expr = "return {grad_w, grad_x, grad_y};" if need_grad_w else "return {grad_x, grad_y};"

    return rf'''

std::vector<torch::Tensor> launcher_{bundle_name}(
{params})
{{
    // Expected tensors:
    //   w        : [WB,Iw,U], WB can be 1 or B
    //   x        : [S,Ix,U] or [B,Ix,U]
    //   y        : {y_comment}
    //   grad_out : [B,V,U] or [S,V,U], matching forward output layout

    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x/y/grad_out must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");
{src_check}{dst_check}

    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x.dim() == 3, "x must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");

    int B  = (int){b_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x.size(0);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
    {y_check_u}
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
{src_numel_check}
{dst_numel_check}

{blist_logic}
{grad_w_alloc}    auto grad_x = torch::zeros_like(x);
{gy_alloc}

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{

        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");

        launch_{bundle_name}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
{grad_w_launch_arg}                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                b_list_ptr,
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    }});

    GPU_KERNEL_LAUNCH_CHECK();

    {ret_expr}
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} backward LARS fused jit impl");
}}
'''


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


def _fold_bwd_reg_budget_candidates_by_max_live(
    reg_budget: Union[int, Iterable[int]],
    *,
    bwd_paths: List[Tuple[int, int, int, int, float]],
    need_grad_w: bool = True,
    consider_cse: bool = False,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    profile: bool = False,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = True,
) -> Tuple[List[int], Dict[str, int]]:
    budgets, is_multi_budget = _normalize_reg_budget_candidates(reg_budget)
    if not is_multi_budget:
        return budgets, {
            "no_cse_max_live": 0,
            "max_cse_live_pairs": 0,
            "effective_live_need": 0,
            "probe_budget": int(budgets[0]),
            "auto_fallback_threshold": 0,
        }

    probe_budget = max(budgets)

    no_cse_probe = LARSUniform1DBwdScheduler(
        paths=bwd_paths,
        reg_budget=probe_budget,
        need_grad_w=need_grad_w,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        prefer_path_fallback_when_full=False,
        profile=profile,
        profile_name=f"bwd_budget_probe_nocse_r{probe_budget}",
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )
    no_cse_result = no_cse_probe.schedule()
    no_cse_max_live = int(getattr(no_cse_probe, "max_live_labels", no_cse_result.max_live))

    max_cse_live_pairs = 0
    if consider_cse:
        cse_probe = CSELARSUniform1DBwdScheduler(
            paths=bwd_paths,
            reg_budget=probe_budget,
            need_grad_w=need_grad_w,
            enable_cse=True,
            enable_secondary_affinity=enable_secondary_affinity,
            topk_candidates=topk_candidates,
            prefer_path_fallback_when_full=False,
            profile=profile,
            profile_name=f"bwd_budget_probe_cse_r{probe_budget}",
            profile_interval=profile_interval,
            profile_seconds=profile_seconds,
            profile_print=profile_print,
        )
        cse_probe.schedule()
        max_cse_live_pairs = int(getattr(cse_probe, "max_live_pairs", 0))

    effective_live_need = int(no_cse_max_live + max_cse_live_pairs)
    kept: List[int] = []
    added_first_over = False
    for rb in budgets:
        if rb <= effective_live_need:
            kept.append(rb)
        elif not added_first_over:
            kept.append(rb)
            added_first_over = True

    return kept, {
        "no_cse_max_live": int(no_cse_max_live),
        "max_cse_live_pairs": int(max_cse_live_pairs),
        "effective_live_need": int(effective_live_need),
        "probe_budget": int(probe_budget),
        "auto_fallback_threshold": int(effective_live_need),
    }


def _normalize_acc_reg_budget_candidates(
    acc_reg_budget: Optional[Union[int, Iterable[Optional[int]]]],
) -> List[Optional[int]]:
    """
    Normalize accumulator register-budget candidates.

    None means old behavior: all gw/gx/gy accumulators stay in scalar local
    variables/registers.  An integer K means keep top-K hot accumulators in
    registers and demote all remaining accumulators to shared memory.
    """
    if acc_reg_budget is None:
        return [None]
    if isinstance(acc_reg_budget, int):
        if int(acc_reg_budget) < 0:
            raise ValueError(f"acc_reg_budget must be >= 0 or None, got {acc_reg_budget}")
        return [int(acc_reg_budget)]

    out: List[Optional[int]] = []
    seen = set()
    for item in acc_reg_budget:
        val = None if item is None else int(item)
        if val is not None and val < 0:
            raise ValueError(f"acc_reg_budget must be >= 0 or None, got {val}")
        key = "all" if val is None else val
        if key not in seen:
            seen.add(key)
            out.append(val)
    if not out:
        raise ValueError("acc_reg_budget candidate list must not be empty")
    return out


def default_lars_bwd_cse_candidate_configs(
    reg_budget: Union[int, Iterable[int]],
    acc_reg_budget: Optional[Union[int, Iterable[Optional[int]]]] = None,
    *,
    enable_cse: bool = False,
    smem_acc_volatile: bool = True,
) -> List[Dict[str, Any]]:
    budgets, _ = _normalize_reg_budget_candidates(reg_budget)
    acc_budgets = _normalize_acc_reg_budget_candidates(acc_reg_budget)
    configs: List[Dict[str, Any]] = []
    cse_tag = "cse_auto" if bool(enable_cse) else "nocse"
    for rb in budgets:
        for ab in acc_budgets:
            acc_tag = "accall" if ab is None else f"acc{int(ab)}_smem"
            volatile_tag = "_volatile" if (ab is not None and bool(smem_acc_volatile)) else ""
            configs.append({
                "name": f"lars_bwd_r{rb}_{acc_tag}_{cse_tag}{volatile_tag}",
                "reg_budget": int(rb),
                "acc_reg_budget": ab,
                "enable_cse": bool(enable_cse),
                "smem_acc_volatile": bool(smem_acc_volatile),
            })
    return configs


# =============================================================================
# Split backward LARS codegen: grad_w / grad_x / grad_y in separate kernels
# =============================================================================

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


class LARSUniform1DBwdSplitScheduler(LARSUniform1DScheduler):
    """
    Per-gradient backward scheduler.  It keeps code2's LARS label selection,
    spill-victim selection and path-fallback behavior, but builds labels for a
    single gradient target only.
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        reg_budget: int = 24,
        *,
        grad_kind: str,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        path_fallback_after: Optional[int] = None,
        prefer_path_fallback_when_full: bool = True,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        if grad_kind not in BWD_SPLIT_KINDS:
            raise ValueError(f"grad_kind must be one of {sorted(BWD_SPLIT_KINDS)}, got {grad_kind!r}")
        self.grad_kind = str(grad_kind)
        self.paths: List[U1DBwdSplitPath] = [
            U1DBwdSplitPath(pid=p, i=i, j=j, k=k, v=v, c=c, grad_kind=self.grad_kind)
            for p, (i, j, k, v, c) in enumerate(paths)
        ]

        self.reg_budget = int(reg_budget)
        if self.reg_budget < 3:
            raise ValueError(f"split backward reg_budget must be at least 3, got {self.reg_budget}")

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
        self.prefer_path_fallback_when_full = bool(prefer_path_fallback_when_full)
        self.path_fallback_after = (
            int(path_fallback_after)
            if path_fallback_after is not None
            else max(8, 2 * self.reg_budget)
        )
        self.debug = bool(debug)

        self.label_to_paths: Dict[Label, Set[int]] = defaultdict(set)
        for path in self.paths:
            for lab in path.labels:
                self.label_to_paths[lab].add(path.pid)

        self.path_label_sets: List[Set[Label]] = [set(p.labels) for p in self.paths]
        self.unscheduled: Set[int] = set(p.pid for p in self.paths)

        self.remaining_uses: Counter[Label] = Counter()
        for p in self.paths:
            for lab in p.labels:
                self.remaining_uses[lab] += 1

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = [f"r{r}" for r in range(self.reg_budget)]

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}
        self._no_progress_iters = 0
        self._last_done = 0

        self.profile_enabled = bool(profile)
        self.profile_name = str(profile_name or f"{self.__class__.__name__}_{self.grad_kind}")
        self.profile_interval = int(profile_interval)
        self.profile_seconds = float(profile_seconds)
        self.profile_print = bool(profile_print)
        self._profile_t0 = perf_counter()
        self._profile_last_print_t = self._profile_t0
        self._profile_last_print_done = 0
        self._profile_loop_iter = 0
        self._profile_select_rounds = 0
        self._profile_fire_rounds = 0
        self._profile_load_rounds = 0
        self._profile_fallback_rounds = 0
        self._profile_records: List[Dict[str, Any]] = []
        self._profile_last_event = ""
        self._profile_last_reason = ""

    def _path(self, pid: int) -> U1DBwdSplitPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        if kind == "go":
            return f"grad_out[{idx}]"
        return f"{kind}[{idx}]"

    def _priority(self, lab: Label) -> int:
        return self.remaining_uses[lab]

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)
        regs: Dict[str, str] = {}
        for kind, idx in p.labels:
            regs[kind] = self.reg_of[(kind, idx)]

        self.instructions.append(
            Inst(
                "bwd_split_fma_resident",
                (
                    self.grad_kind,
                    p.target_index,
                    regs.get("w", ""),
                    regs.get("x", ""),
                    regs.get("y", ""),
                    regs.get("go", ""),
                    p.c,
                ),
                (
                    f"path#{pid}: split {self.grad_kind}; "
                    f"w[{p.i}], x[{p.j}], y[{p.k}], go[{p.v}], c={p.c}"
                ),
            )
        )

        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        for lab in p.labels:
            self.remaining_uses[lab] -= 1

        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")


def _split_grad_acc_name(grad_kind: str, idx: int) -> str:
    if grad_kind == "gw":
        return f"gw_acc_i_{int(idx)}"
    if grad_kind == "gx":
        return f"gx_acc_j_{int(idx)}"
    if grad_kind == "gy":
        return f"gy_acc_k_{int(idx)}"
    raise ValueError(f"Bad grad_kind: {grad_kind}")


def emit_lars_bwd_split_preamble() -> str:
    return r'''
#include <stdint.h>
#include <torch/extension.h>
#include <vector>
#include <cstdint>

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #include <ATen/hip/HIPContext.h>
  #include <c10/hip/HIPGuard.h>
  using gpuStream_t = hipStream_t;
  #define getCurrentGPUStream at::hip::getCurrentHIPStream
  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()
#else
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <ATen/cuda/CUDAContext.h>
  using gpuStream_t = cudaStream_t;
  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream
  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

using GPU_Guard = c10::DeviceGuard;

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_sum_xor_lars_bwd_split(scalar_t v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
        v += __shfl_xor(v, offset);
#else
        v += __shfl_xor_sync(0xffffffff, v, offset);
#endif
    }
    return v;
}

static inline bool mul_fits_int32_bwd_split(int64_t a, int64_t b) {
    if (a < 0 || b < 0) return false;
    constexpr int64_t LIM = 2147483647LL;
    if (a == 0 || b == 0) return true;
    return a <= LIM / b;
}

static inline bool mul3_fits_int32_bwd_split(int64_t a, int64_t b, int64_t c) {
    if (!mul_fits_int32_bwd_split(a, b)) return false;
    return mul_fits_int32_bwd_split(a * b, c);
}

static inline bool should_use_int32_index_bwd_split(
    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,
    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)
{
    bool w_ok = mul3_fits_int32_bwd_split((int64_t)WB, (int64_t)Iw, (int64_t)U);
    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;
    bool x_ok = mul3_fits_int32_bwd_split(x_dim0, (int64_t)Ix, (int64_t)U);
    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;
    bool y_ok = mode_scalar_y ? mul_fits_int32_bwd_split(y_dim0, (int64_t)Ky)
                              : mul3_fits_int32_bwd_split(y_dim0, (int64_t)Ky, (int64_t)U);
    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;
    bool go_ok = mul3_fits_int32_bwd_split(go_dim0, (int64_t)V, (int64_t)U);
    return w_ok && x_ok && y_ok && go_ok;
}
'''


def emit_lars_bwd_split_kernel_from_schedule(
    schedule_result: ScheduleResult,
    *,
    grad_kind: str,
    kernel_name: str,
    mode: str = "u,u,,u",
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
    acc_reg_budget: Optional[int] = None,
    smem_acc_volatile: bool = False,
) -> str:
    if grad_kind not in BWD_SPLIT_KINDS:
        raise ValueError(f"grad_kind must be one of {sorted(BWD_SPLIT_KINDS)}, got {grad_kind!r}")
    if block_size != 32:
        raise ValueError("this emitter currently assumes block_size=32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    reg_count = _max_lars_reg_count_cse_aware(schedule_result)

    target_use: Counter[int] = Counter()
    first_seen: Dict[int, int] = {}
    for inst_id, inst in enumerate(schedule_result.instructions):
        if inst.op == "bwd_split_fma_resident":
            ikind, target_idx = inst.args[0], int(inst.args[1])
            if ikind != grad_kind:
                raise ValueError(f"schedule contains {ikind}, but emitter grad_kind is {grad_kind}")
            target_use[target_idx] += 1
            first_seen.setdefault(target_idx, inst_id)

    all_targets: Set[int] = set(target_use)
    if acc_reg_budget is None:
        reg_targets: Set[int] = set(all_targets)
    else:
        budget = max(0, int(acc_reg_budget))
        ranked = sorted(
            all_targets,
            key=lambda idx: (int(target_use[idx]), -int(first_seen.get(idx, 10**9)), -int(idx)),
            reverse=True,
        )
        reg_targets = set(ranked[:budget])

    smem_targets_sorted: List[int] = sorted(all_targets - reg_targets)
    smem_id: Dict[int, int] = {idx: n for n, idx in enumerate(smem_targets_sorted)}
    reg_targets_sorted = sorted(reg_targets)

    def _is_reg_target(idx: int) -> bool:
        return int(idx) in reg_targets

    def _target_acc_name(idx: int) -> str:
        return _split_grad_acc_name(grad_kind, int(idx))

    def _smem_expr(idx: int) -> str:
        return f"smem_acc[{smem_id[int(idx)]} * 32 + lane]"

    def _emit_acc_update(idx: int, value_expr: str) -> None:
        if _is_reg_target(idx):
            ap(f"            {_target_acc_name(idx)} += {value_expr};")
        else:
            ap(f"            {_smem_expr(idx)} += {value_expr};")

    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* __restrict__ grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* __restrict__ grad_x,")
    else:
        ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    if smem_targets_sorted:
        ap("    extern __shared__ __align__(16) unsigned char smem_raw[];")
        if smem_acc_volatile:
            ap("    volatile scalar_t* smem_acc = reinterpret_cast<volatile scalar_t*>(smem_raw);")
        else:
            ap("    scalar_t* smem_acc = reinterpret_cast<scalar_t*>(smem_raw);")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
    ap("    const int e_orig = b_list ? b_list[e_local] : e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e_orig];")
    else:
        ap("    const int src = e_orig;")
    if use_scatter:
        ap("    const int dst = dst_idx[e_orig];")
    else:
        ap("    const int dst = e_orig;")
    ap("")
    ap("    const int x_row = " + ("src;" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src;" if use_y_src else "e_orig;"))
    ap("    const int go_row = " + ("dst;" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base  = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
        if grad_kind == "gw":
            ap(f"    const index_t gw_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base  = (index_t)w_row * (index_t)Iw * (index_t)U;")
        if grad_kind == "gw":
            ap("    const index_t gw_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base  = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
        if grad_kind == "gx":
            ap(f"    const index_t gx_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base  = (index_t)x_row * (index_t)Ix * (index_t)U;")
        if grad_kind == "gx":
            ap("    const index_t gx_base = (index_t)x_row * (index_t)Ix * (index_t)U;")

    if mode_scalar_y:
        ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky;")
        if grad_kind == "gy":
            ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base  = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
            if grad_kind == "gy":
                ap(f"    const index_t gy_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky * (index_t)U;")
            if grad_kind == "gy":
                ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky * (index_t)U;")

    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t go_base = (index_t)go_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t go_base = (index_t)go_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")
    for sid, idx in enumerate(smem_targets_sorted):
        ap(f"            smem_acc[{sid} * 32 + lane] = scalar_t(0);  // smem {grad_kind}[{idx}]")
    if smem_targets_sorted:
        ap("")
    for rid in range(reg_count):
        ap(f"            scalar_t r{rid};")
    if reg_count:
        ap("")
    for idx in reg_targets_sorted:
        ap(f"            scalar_t {_target_acc_name(idx)} = scalar_t(0);")
    if reg_targets_sorted:
        ap("")

    for inst_id, inst in enumerate(schedule_result.instructions):
        comment = _sanitize_cuda_comment(inst.comment)
        prefix = f"            // inst {inst_id}: {inst.op}"
        if comment:
            prefix += f" | {comment}"
        ap(prefix)

        if inst.op == "load":
            reg, ref = inst.args
            kind, idx = _parse_bwd_lars_label_ref(ref)
            if kind in BWD_OUTPUT_KINDS:
                raise ValueError("load must not be used for backward output labels")
            expr = _bwd_label_index_expr(
                kind, idx,
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            arr = "grad_out" if kind == "go" else kind
            ap(f"            {reg} = {arr}[{expr}];")

        elif inst.op == "bwd_split_fma_resident":
            ikind, target_idx, rw, rx, ry, rgo, coeff = inst.args
            if ikind != grad_kind:
                raise ValueError(f"schedule contains {ikind}, but emitter grad_kind is {grad_kind}")
            c = _fmt_lars_float(float(coeff))
            target_idx = int(target_idx)
            if grad_kind == "gw":
                _emit_acc_update(target_idx, f"scalar_t({c}) * {rgo} * {rx} * {ry}")
            elif grad_kind == "gx":
                _emit_acc_update(target_idx, f"scalar_t({c}) * ({rw} * {rgo}) * {ry}")
            elif grad_kind == "gy":
                _emit_acc_update(target_idx, f"scalar_t({c}) * ({rw} * {rgo}) * {rx}")
            else:
                raise ValueError(f"Bad grad_kind: {grad_kind}")

        elif inst.op == "release":
            pass
        else:
            raise ValueError(f"Unsupported split backward LARS instruction op: {inst.op}")

    if reg_targets_sorted or smem_targets_sorted:
        ap("")
        ap("            // split backward accumulator writeback")

    def _emit_writeback(idx: int, acc_expr: str) -> None:
        if grad_kind == "gw":
            expr = _bwd_label_index_expr(
                "gw", int(idx),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            ap(f"            grad_w[{expr}] = {acc_expr};")
        elif grad_kind == "gx":
            expr = _bwd_label_index_expr(
                "gx", int(idx),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            ap(f"            atomicAdd(&grad_x[{expr}], {acc_expr});")
        elif grad_kind == "gy":
            expr = _bwd_label_index_expr(
                "gy", int(idx),
                mode_scalar_y=mode_scalar_y,
                u_dim=u_dim,
                iw_dim=iw_dim,
                ix_dim=ix_dim,
                ky_dim=ky_dim,
                v_dim=v_dim,
            )
            if mode_scalar_y:
                safe_idx = str(int(idx)).replace("-", "m")
                ap(f"            scalar_t gy_sum_{safe_idx} = warp_sum_xor_lars_bwd_split({acc_expr});")
                ap("            if (lane == 0) {")
                ap(f"                atomicAdd(&grad_y[{expr}], gy_sum_{safe_idx});")
                ap("            }")
            else:
                ap(f"            atomicAdd(&grad_y[{expr}], {acc_expr});")
        else:
            raise ValueError(f"Bad grad_kind: {grad_kind}")

    for idx in reg_targets_sorted:
        _emit_writeback(int(idx), _target_acc_name(int(idx)))
    for idx in smem_targets_sorted:
        _emit_writeback(int(idx), _smem_expr(int(idx)))

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* grad_x,")
    else:
        ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    constexpr int kSmemAccCount = {len(smem_targets_sorted)};")
    ap("    size_t smem_bytes = (size_t)kSmemAccCount * 32u * sizeof(scalar_t);")
    ap("#if !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)")
    ap("    if (smem_bytes > 49152) {")
    ap(f"        cudaFuncSetAttribute({kernel_name}<scalar_t, index_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);")
    ap("    }")
    ap("#endif")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, smem_bytes, stream>>>(")
    if grad_kind == "gw":
        ap("        w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("        w, x, y, grad_out, grad_x,")
    else:
        ap("        w, x, y, grad_out, grad_y,")
    ap("        src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* grad_x,")
    else:
        ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_bwd_split(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                       kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    if grad_kind == "gw":
        ap("            w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("            w, x, y, grad_out, grad_x,")
    else:
        ap("            w, x, y, grad_out, grad_y,")
    ap("            src_idx, dst_idx, b_list, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    if grad_kind == "gw":
        ap("            w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("            w, x, y, grad_out, grad_x,")
    else:
        ap("            w, x, y, grad_out, grad_y,")
    ap("            src_idx, dst_idx, b_list, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if grad_kind == "gw":
        ap("    scalar_t* grad_w,")
    elif grad_kind == "gx":
        ap("    scalar_t* grad_x,")
    else:
        ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    if grad_kind == "gw":
        ap("        w, x, y, grad_out, grad_w,")
    elif grad_kind == "gx":
        ap("        w, x, y, grad_out, grad_x,")
    else:
        ap("        w, x, y, grad_out, grad_y,")
    ap("        src_idx, dst_idx, b_list, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def emit_lars_bwd_split_launcher(
    bundle_name: str,
    mode: str,
    *,
    need_grad_w: bool,
    use_x_src: bool,
    use_y_src: bool,
    use_scatter: bool,
    gradw_kernel: Optional[str],
    gradx_kernel: str,
    grady_kernel: str,
) -> str:
    if mode == "u,u,,u":
        y_comment = "[B,Ky,1] or [S,Ky,1]"
        gy_alloc = "    auto grad_y = torch::zeros({y.size(0), Ky, 1}, y.options());"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == 1, "y.size(2) must be 1 for mode u,u,,u");'
    elif mode == "u,u,u,u":
        y_comment = "[B,Ky,U] or [S,Ky,U]"
        gy_alloc = "    auto grad_y = torch::zeros_like(y);"
        y_check_u = 'TORCH_CHECK((int)y.size(2) == U, "y.size(2) must equal U for mode u,u,u,u");'
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    param_lines = [
        "    torch::Tensor w",
        "    torch::Tensor x",
        "    torch::Tensor y",
        "    torch::Tensor grad_out",
    ]
    if use_x_src or use_y_src:
        param_lines.append("    torch::Tensor src_idx")
    if use_scatter:
        param_lines.append("    torch::Tensor dst_idx")
        param_lines.append("    torch::Tensor b_list")
    param_lines.append("    int64_t V64")
    params = ",\n".join(param_lines)

    src_check = ""
    if use_x_src or use_y_src:
        src_check = r'''
    TORCH_CHECK(src_idx.is_cuda(), "src_idx must be CUDA/HIP");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
'''
    dst_check = ""
    if use_scatter:
        dst_check = r'''
    TORCH_CHECK(dst_idx.is_cuda(), "dst_idx must be CUDA/HIP");
    TORCH_CHECK(dst_idx.scalar_type() == torch::kInt32, "dst_idx must be int32");
'''

    if use_x_src or use_y_src:
        b_expr = "src_idx.size(0)"
        src_numel_check = '    TORCH_CHECK(src_idx.numel() >= B, "src_idx numel must be >= B");'
    elif use_scatter:
        b_expr = "dst_idx.size(0)"
        src_numel_check = ""
    else:
        b_expr = "grad_out.size(0)"
        src_numel_check = ""

    dst_numel_check = '    TORCH_CHECK(dst_idx.numel() >= B, "dst_idx numel must be >= B");' if use_scatter else ""

    blist_logic = ""
    if use_scatter:
        blist_logic = r'''
    const int32_t* b_list_ptr = nullptr;
    if (b_list.defined() && b_list.numel() > 0) {
        TORCH_CHECK(b_list.is_cuda(), "b_list must be CUDA/HIP");
        TORCH_CHECK(b_list.scalar_type() == torch::kInt32, "b_list must be int32");
        TORCH_CHECK((int)b_list.numel() == B, "b_list must be [B]");
        b_list_ptr = (const int32_t*)b_list.data_ptr<int32_t>();
    }
'''
    else:
        blist_logic = "    const int32_t* b_list_ptr = nullptr;\n"

    launch_src_arg = "(const int32_t*)src_idx.data_ptr<int32_t>()," if (use_x_src or use_y_src) else "nullptr,"
    launch_dst_arg = "(const int32_t*)dst_idx.data_ptr<int32_t>()," if use_scatter else "nullptr,"

    grad_w_alloc = "    auto grad_w = torch::zeros_like(w);\n" if need_grad_w else ""
    grad_w_launch = ""
    if need_grad_w:
        assert gradw_kernel is not None
        grad_w_launch = f'''
        launch_{gradw_kernel}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_w.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                b_list_ptr,
                B, WB, Iw, Ix, Ky, V, U, S, stream);
'''
    ret_expr = "return {grad_w, grad_x, grad_y};" if need_grad_w else "return {grad_x, grad_y};"

    return rf'''

std::vector<torch::Tensor> launcher_{bundle_name}(
{params})
{{
    // Split backward: grad_w / grad_x / grad_y are computed by independent
    // LARS-scheduled kernels to reduce register pressure for large path lists.
    // Expected tensors:
    //   w        : [WB,Iw,U], WB can be 1 or B
    //   x        : [S,Ix,U] or [B,Ix,U]
    //   y        : {y_comment}
    //   grad_out : [B,V,U] or [S,V,U], matching forward output layout

    TORCH_CHECK(w.is_cuda() && x.is_cuda() && y.is_cuda() && grad_out.is_cuda(),
                "w/x/y/grad_out must be CUDA/HIP");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && y.is_contiguous() && grad_out.is_contiguous(),
                "w/x/y/grad_out must be contiguous");
{src_check}{dst_check}

    TORCH_CHECK(w.dim() == 3, "w must be [WB,Iw,U]");
    TORCH_CHECK(x.dim() == 3, "x must be [S,Ix,U] or [B,Ix,U]");
    TORCH_CHECK(y.dim() == 3, "y must be 3D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");

    int B  = (int){b_expr};
    int WB = (int)w.size(0);
    int Iw = (int)w.size(1);
    int U  = (int)w.size(2);

    int S  = (int)x.size(0);
    int Ix = (int)x.size(1);
    int Ky = (int)y.size(1);
    int V  = (int)V64;

    TORCH_CHECK(V > 0, "V must be > 0");
    TORCH_CHECK(B > 0, "B must be > 0");
    TORCH_CHECK((int)w.size(0) == 1 || (int)w.size(0) == B,
                "w.size(0) must be 1 or B");
    TORCH_CHECK((int)w.size(2) == U, "w U mismatch");
    TORCH_CHECK((int)x.size(2) == U, "x U mismatch");
    {y_check_u}
    TORCH_CHECK((int)grad_out.size(1) == V, "grad_out V mismatch");
    TORCH_CHECK((int)grad_out.size(2) == U, "grad_out U mismatch");
    TORCH_CHECK((U % 32) == 0, "U must be a multiple of 32");
{src_numel_check}
{dst_numel_check}

{blist_logic}
{grad_w_alloc}    auto grad_x = torch::zeros_like(x);
{gy_alloc}

    GPU_Guard device_guard(w.device());
    gpuStream_t stream = getCurrentGPUStream(w.device().index());

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "{bundle_name}", [&] {{

        TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x dtype must match w");
        TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y dtype must match w");
        TORCH_CHECK(grad_out.scalar_type() == w.scalar_type(), "grad_out dtype must match w");
{grad_w_launch}
        launch_{gradx_kernel}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_x.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                b_list_ptr,
                B, WB, Iw, Ix, Ky, V, U, S, stream);

        launch_{grady_kernel}<scalar_t>(
                (const scalar_t*)w.data_ptr<scalar_t>(),
                (const scalar_t*)x.data_ptr<scalar_t>(),
                (const scalar_t*)y.data_ptr<scalar_t>(),
                (const scalar_t*)grad_out.data_ptr<scalar_t>(),
                (scalar_t*)grad_y.data_ptr<scalar_t>(),
                {launch_src_arg}
                {launch_dst_arg}
                b_list_ptr,
                B, WB, Iw, Ix, Ky, V, U, S, stream);
    }});

    GPU_KERNEL_LAUNCH_CHECK();

    {ret_expr}
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run", &launcher_{bundle_name}, "{bundle_name} backward LARS split jit impl");
}}
'''



@dataclass(frozen=True)
class BwdCodegenContext:
    """Pre-parsed Uniform1D backward codegen inputs shared by fused/split paths."""
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


def _prepare_uniform1d_bwd_codegen_context(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]],
    output_indices: Optional[Dict[int, Any]],
    *,
    mode: str,
    path_semantics: str,
) -> BwdCodegenContext:
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = int(i_list.numel())
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    bwd_paths = _make_lars_bwd_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )

    return BwdCodegenContext(
        input_indices=input_indices,
        output_indices=output_indices,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        P=P,
        i_cpu=i_cpu,
        j_cpu=j_cpu,
        k_cpu=k_cpu,
        v_cpu=v_cpu,
        c_cpu=c_cpu,
        bwd_paths=bwd_paths,
        mode_str="uu_u" if mode == "u,u,,u" else "uuuu",
        layout_tag=f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}",
    )


def _resolve_split_backward(
    split_backward: Union[bool, str],
    *,
    path_count: int,
    split_path_threshold: int,
) -> bool:
    if isinstance(split_backward, str):
        split_mode = split_backward.lower()
        if split_mode in ("auto",):
            return int(path_count) > int(split_path_threshold)
        if split_mode in ("split", "true", "always", "on", "yes"):
            return True
        if split_mode in ("fused", "false", "never", "off", "no"):
            return False
        raise ValueError(
            "split_backward must be bool or one of "
            "'auto', 'split'/'always'/'on', 'fused'/'never'/'off'"
        )
    return bool(split_backward)


def _write_codegen_candidate(
    *,
    out_path: str,
    candidate_count: int,
    cand_name: str,
    code: str,
) -> None:
    if not out_path:
        return
    out_p = Path(out_path)
    if candidate_count > 1:
        stem = out_p.stem
        suffix = out_p.suffix or ".cu"
        out_p.with_name(f"{stem}_{cand_name}{suffix}").write_text(code, encoding="utf-8")
    else:
        out_p.write_text(code, encoding="utf-8")


def _finalize_codegen_candidates(candidates: List[Any], *, return_schedule: bool):
    if return_schedule:
        return candidates if len(candidates) != 1 else candidates[0]
    if len(candidates) == 1:
        return candidates[0][1]
    return candidates


def _bwd_base_kernel_name(
    *,
    kernel_name: str,
    cand_name: str,
    u_dim: int,
    path_count: int,
    mode_str: str,
    layout_tag: str,
    grad_tag: str,
) -> str:
    safe_cand = cand_name.replace("-", "_").replace(".", "_")
    if kernel_name and kernel_name != "uniform1d_bwd_lars":
        return f"{kernel_name}_{safe_cand}"
    return (
        f"uniform1d_{safe_cand}_u{u_dim}_path{path_count}_"
        f"{mode_str}_{layout_tag}_{grad_tag}_bwd"
    )


def _generate_code_uniform1d_bwd_split_from_context(
    ctx: BwdCodegenContext,
    *,
    u_dim: int,
    iw_dim: Optional[int],
    ix_dim: Optional[int],
    ky_dim: Optional[int],
    v_dim: Optional[int],
    mode: str,
    need_grad_w: bool,
    out_path: str,
    kernel_name: str,
    reg_budget: Union[int, Iterable[int]],
    acc_reg_budget: Optional[Union[int, Iterable[Optional[int]]]],
    smem_acc_volatile: bool,
    return_schedule: bool,
    fold_reg_budgets_by_max_live: bool,
    profile: bool,
    profile_interval: int,
    profile_seconds: float,
    profile_print: bool,
    enable_secondary_affinity: bool,
    topk_candidates: Optional[int],
):
    """Internal split-backward implementation.  The public entry is the unified wrapper."""
    budgets, _ = _normalize_reg_budget_candidates(reg_budget)
    acc_budgets = _normalize_acc_reg_budget_candidates(acc_reg_budget)
    configs: List[Dict[str, Any]] = []
    for rb in budgets:
        for ab in acc_budgets:
            acc_tag = "accall" if ab is None else f"acc{int(ab)}_smem"
            volatile_tag = "_volatile" if (ab is not None and bool(smem_acc_volatile)) else ""
            configs.append({
                "name": f"lars_bwd_split_r{int(rb)}_{acc_tag}{volatile_tag}",
                "reg_budget": int(rb),
                "acc_reg_budget": ab,
                "smem_acc_volatile": bool(smem_acc_volatile),
            })

    if fold_reg_budgets_by_max_live and len(budgets) > 1:
        max_effective_need = 0
        for gkind in (["gw"] if need_grad_w else []) + ["gx", "gy"]:
            probe = LARSUniform1DBwdSplitScheduler(
                paths=ctx.bwd_paths,
                reg_budget=max(budgets),
                grad_kind=gkind,
                enable_secondary_affinity=enable_secondary_affinity,
                topk_candidates=topk_candidates,
                prefer_path_fallback_when_full=False,
                profile=profile,
                profile_name=f"bwd_split_budget_probe_{gkind}_r{max(budgets)}",
                profile_interval=profile_interval,
                profile_seconds=profile_seconds,
                profile_print=profile_print,
            )
            probe_result = probe.schedule()
            max_effective_need = max(
                max_effective_need,
                int(getattr(probe, "max_live_labels", probe_result.max_live)),
            )
        configs = _fold_lars_configs_by_reg_budget(configs, effective_live_need=max_effective_need)
    else:
        max_effective_need = 0

    grad_tag = "split" if need_grad_w else "split_nogradw"
    candidates = []
    split_kinds = (["gw"] if need_grad_w else []) + ["gx", "gy"]

    for cfg_in in configs:
        cfg = dict(cfg_in)
        rb = int(cfg["reg_budget"])
        acc_rb = cfg.get("acc_reg_budget", None)
        acc_rb = None if acc_rb is None else int(acc_rb)
        cfg_smem_acc_volatile = bool(cfg.get("smem_acc_volatile", smem_acc_volatile))
        cand_name = str(cfg["name"])
        auto_fallback_when_full = bool(max_effective_need > 0 and rb < max_effective_need)
        cfg["auto_fallback_when_full"] = auto_fallback_when_full
        cfg["effective_live_need"] = int(max_effective_need)
        cfg["split_backward"] = True

        base_kernel_name = _bwd_base_kernel_name(
            kernel_name=kernel_name,
            cand_name=cand_name,
            u_dim=u_dim,
            path_count=ctx.P,
            mode_str=ctx.mode_str,
            layout_tag=ctx.layout_tag,
            grad_tag=grad_tag,
        )

        schedules: Dict[str, ScheduleResult] = {}
        profiles: Dict[str, Any] = {}
        code_parts: List[str] = [emit_lars_bwd_split_preamble()]
        kernel_names: Dict[str, str] = {}

        for gkind in split_kinds:
            scheduler = LARSUniform1DBwdSplitScheduler(
                paths=ctx.bwd_paths,
                reg_budget=rb,
                grad_kind=gkind,
                enable_secondary_affinity=bool(cfg.get("enable_secondary_affinity", enable_secondary_affinity)),
                topk_candidates=cfg.get("topk_candidates", topk_candidates),
                prefer_path_fallback_when_full=auto_fallback_when_full,
                profile=bool(cfg.get("profile", profile)),
                profile_name=str(cfg.get("profile_name", f"{cand_name}_{gkind}")),
                profile_interval=int(cfg.get("profile_interval", profile_interval)),
                profile_seconds=float(cfg.get("profile_seconds", profile_seconds)),
                profile_print=bool(cfg.get("profile_print", profile_print)),
            )
            schedule_result = scheduler.schedule()
            schedules[gkind] = schedule_result
            profiles[gkind] = schedule_result.profile
            split_kernel_name = f"{base_kernel_name}_{gkind}"
            kernel_names[gkind] = split_kernel_name
            code_parts.append(
                emit_lars_bwd_split_kernel_from_schedule(
                    schedule_result,
                    grad_kind=gkind,
                    kernel_name=split_kernel_name,
                    mode=mode,
                    use_x_src=ctx.use_x_src,
                    use_y_src=ctx.use_y_src,
                    use_scatter=ctx.use_scatter,
                    u_dim=u_dim,
                    iw_dim=iw_dim,
                    ix_dim=ix_dim,
                    ky_dim=ky_dim,
                    v_dim=v_dim,
                    block_size=32,
                    acc_reg_budget=acc_rb,
                    smem_acc_volatile=cfg_smem_acc_volatile,
                )
            )

        code_parts.append(
            emit_lars_bwd_split_launcher(
                bundle_name=base_kernel_name,
                mode=mode,
                need_grad_w=need_grad_w,
                use_x_src=ctx.use_x_src,
                use_y_src=ctx.use_y_src,
                use_scatter=ctx.use_scatter,
                gradw_kernel=kernel_names.get("gw"),
                gradx_kernel=kernel_names["gx"],
                grady_kernel=kernel_names["gy"],
            )
        )
        code = "\n".join(code_parts)

        _write_codegen_candidate(
            out_path=out_path,
            candidate_count=len(configs),
            cand_name=cand_name,
            code=code,
        )

        if return_schedule:
            candidates.append({
                "name": cand_name,
                "code": code,
                "schedule": schedules,
                "config": cfg,
                "profiles": profiles,
                "budget_fold_effective_live_need": int(max_effective_need),
                "auto_fallback_when_full": auto_fallback_when_full,
            })
        else:
            candidates.append((cand_name, code))

    return _finalize_codegen_candidates(candidates, return_schedule=return_schedule)


def _generate_code_uniform1d_bwd_fused_from_context(
    ctx: BwdCodegenContext,
    *,
    u_dim: int,
    iw_dim: Optional[int],
    ix_dim: Optional[int],
    ky_dim: Optional[int],
    v_dim: Optional[int],
    mode: str,
    need_grad_w: bool,
    out_path: str,
    kernel_name: str,
    reg_budget: Union[int, Iterable[int]],
    acc_reg_budget: Optional[Union[int, Iterable[Optional[int]]]],
    enable_cse: bool,
    smem_acc_volatile: bool,
    consider_cse: bool,
    return_schedule: bool,
    fold_reg_budgets_by_max_live: bool,
    profile: bool,
    profile_interval: int,
    profile_seconds: float,
    profile_print: bool,
    enable_secondary_affinity: bool,
    topk_candidates: Optional[int],
):
    probed_budgets, budget_fold_info = _fold_bwd_reg_budget_candidates_by_max_live(
        reg_budget,
        bwd_paths=ctx.bwd_paths,
        need_grad_w=need_grad_w,
        consider_cse=consider_cse,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        profile=profile,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
    )

    if fold_reg_budgets_by_max_live:
        effective_reg_budget = probed_budgets
    else:
        effective_reg_budget, _ = _normalize_reg_budget_candidates(reg_budget)

    effective_live_need = int(budget_fold_info.get("effective_live_need", 0))
    auto_cse_configs = default_lars_bwd_cse_candidate_configs(
        effective_reg_budget,
        acc_reg_budget=acc_reg_budget,
        enable_cse=enable_cse,
        smem_acc_volatile=smem_acc_volatile,
    )

    grad_tag = "full" if need_grad_w else "nogradw"
    candidates = []

    for cfg_in in auto_cse_configs:
        cfg = dict(cfg_in)
        rb = int(cfg.get("reg_budget", reg_budget if isinstance(reg_budget, int) else list(reg_budget)[0]))
        acc_rb = cfg.get("acc_reg_budget", None)
        acc_rb = None if acc_rb is None else int(acc_rb)
        cfg_enable_cse = bool(cfg.get("enable_cse", enable_cse))
        cfg_smem_acc_volatile = bool(cfg.get("smem_acc_volatile", smem_acc_volatile))

        acc_tag = "accall" if acc_rb is None else f"acc{acc_rb}_smem"
        cse_tag = "cse_auto" if cfg_enable_cse else "nocse"
        volatile_tag = "_volatile" if (acc_rb is not None and cfg_smem_acc_volatile) else ""
        cand_name = str(cfg.get("name", f"lars_bwd_r{rb}_{acc_tag}_{cse_tag}{volatile_tag}"))

        auto_fallback_when_full = bool(effective_live_need > 0 and rb < effective_live_need)
        cfg["auto_fallback_when_full"] = auto_fallback_when_full
        cfg["effective_live_need"] = effective_live_need
        cfg["need_grad_w"] = bool(need_grad_w)
        cfg["acc_reg_budget"] = acc_rb
        cfg["enable_cse"] = bool(cfg_enable_cse)
        cfg["smem_acc_volatile"] = bool(cfg_smem_acc_volatile)
        cfg["split_backward"] = False

        scheduler = CSELARSUniform1DBwdScheduler(
            paths=ctx.bwd_paths,
            reg_budget=rb,
            need_grad_w=need_grad_w,
            enable_cse=cfg_enable_cse,
            enable_secondary_affinity=bool(cfg.get("enable_secondary_affinity", enable_secondary_affinity)),
            topk_candidates=cfg.get("topk_candidates", topk_candidates),
            prefer_path_fallback_when_full=auto_fallback_when_full,
            profile=bool(cfg.get("profile", profile)),
            profile_name=str(cfg.get("profile_name", cand_name)),
            profile_interval=int(cfg.get("profile_interval", profile_interval)),
            profile_seconds=float(cfg.get("profile_seconds", profile_seconds)),
            profile_print=bool(cfg.get("profile_print", profile_print)),
        )
        schedule_result = scheduler.schedule()

        base_kernel_name = _bwd_base_kernel_name(
            kernel_name=kernel_name,
            cand_name=cand_name,
            u_dim=u_dim,
            path_count=ctx.P,
            mode_str=ctx.mode_str,
            layout_tag=ctx.layout_tag,
            grad_tag=grad_tag,
        )

        code = emit_fused_bwd_kernel_from_lars_schedule(
            schedule_result,
            kernel_name=base_kernel_name,
            mode=mode,
            need_grad_w=need_grad_w,
            use_x_src=ctx.use_x_src,
            use_y_src=ctx.use_y_src,
            use_scatter=ctx.use_scatter,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
            block_size=32,
            acc_reg_budget=acc_rb,
            smem_acc_volatile=cfg_smem_acc_volatile,
        )

        code = code + "\n" + emit_lars_bwd_launcher(
            bundle_name=base_kernel_name,
            mode=mode,
            need_grad_w=need_grad_w,
            use_x_src=ctx.use_x_src,
            use_y_src=ctx.use_y_src,
            use_scatter=ctx.use_scatter,
        )

        _write_codegen_candidate(
            out_path=out_path,
            candidate_count=len(auto_cse_configs),
            cand_name=cand_name,
            code=code,
        )

        if return_schedule:
            candidates.append({
                "name": cand_name,
                "code": code,
                "schedule": schedule_result,
                "config": cfg,
                "budget_fold_info": dict(budget_fold_info),
                "budget_fold_max_live": budget_fold_info.get("no_cse_max_live"),
                "budget_fold_max_cse": budget_fold_info.get("max_cse_live_pairs"),
                "budget_fold_effective_live_need": budget_fold_info.get("effective_live_need"),
                "auto_fallback_when_full": auto_fallback_when_full,
                "profile": schedule_result.profile,
                "cse_stats": {
                    "pair_creates": getattr(scheduler, "cse_pair_creates", 0),
                    "pair_hits": getattr(scheduler, "cse_pair_hits", 0),
                    "pair_releases": getattr(scheduler, "cse_pair_releases", 0),
                    "pair_drops_for_label_release": getattr(scheduler, "cse_pair_drops_for_label_release", 0),
                    "pair_drops_for_reg_pressure": getattr(scheduler, "cse_pair_drops_for_reg_pressure", 0),
                },
            })
        else:
            candidates.append((cand_name, code))

    return _finalize_codegen_candidates(candidates, return_schedule=return_schedule)


def generate_code_uniform1d_bwd_split_with_scheduler(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    out_path: str = "generated_uniform1d_bwd_lars_split.cu",
    kernel_name: str = "uniform1d_bwd_lars",
    reg_budget: Union[int, Iterable[int]] = 24,
    acc_reg_budget: Optional[Union[int, Iterable[Optional[int]]]] = None,
    smem_acc_volatile: bool = True,
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    fold_reg_budgets_by_max_live: bool = True,
    profile: bool = True,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = True,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
):
    """
    Backward-compatible wrapper.  New code should call
    generate_code_uniform1d_bwd_with_scheduler(..., split_backward=True).
    """
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
        need_grad_w=need_grad_w,
        out_path=out_path,
        kernel_name=kernel_name,
        reg_budget=reg_budget,
        acc_reg_budget=acc_reg_budget,
        smem_acc_volatile=smem_acc_volatile,
        path_semantics=path_semantics,
        return_schedule=return_schedule,
        fold_reg_budgets_by_max_live=fold_reg_budgets_by_max_live,
        profile=profile,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
        split_backward=True,
    )


def generate_code_uniform1d_bwd_with_scheduler(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    out_path: str = "generated_uniform1d_bwd_lars_cse.cu",
    kernel_name: str = "uniform1d_bwd_lars",
    scalar_t: str = "float",
    reg_budget: Union[int, Iterable[int]] = 24,
    acc_reg_budget: Optional[Union[int, Iterable[Optional[int]]]] = None,
    enable_cse: bool = False,
    smem_acc_volatile: bool = True,
    consider_cse: bool = False,
    path_semantics: str = "wxy",
    return_schedule: bool = False,
    fold_reg_budgets_by_max_live: bool = True,
    profile: bool = True,
    profile_interval: int = 1000,
    profile_seconds: float = 2.0,
    profile_print: bool = True,
    enable_secondary_affinity: bool = False,
    topk_candidates: Optional[int] = 128,
    split_backward: Union[bool, str] = "auto",
    split_path_threshold: int = 256,
):
    """
    Unified LARS backward code generator.

    split_backward controls implementation strategy:
      - False / "fused": one fused LARS backward kernel.
      - True / "split": independent grad_w, grad_x and grad_y kernels.
      - "auto": split when path_count > split_path_threshold.
    """
    del scalar_t

    ctx = _prepare_uniform1d_bwd_codegen_context(
        i_list=i_list,
        j_list=j_list,
        k_list=k_list,
        v_list=v_list,
        coeff_list=coeff_list,
        input_indices=input_indices,
        output_indices=output_indices,
        mode=mode,
        path_semantics=path_semantics,
    )

    use_split_backward = _resolve_split_backward(
        split_backward,
        path_count=ctx.P,
        split_path_threshold=split_path_threshold,
    )

    common_kwargs = dict(
        ctx=ctx,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        mode=mode,
        need_grad_w=need_grad_w,
        out_path=out_path,
        kernel_name=kernel_name,
        reg_budget=reg_budget,
        acc_reg_budget=acc_reg_budget,
        smem_acc_volatile=smem_acc_volatile,
        return_schedule=return_schedule,
        fold_reg_budgets_by_max_live=fold_reg_budgets_by_max_live,
        profile=profile,
        profile_interval=profile_interval,
        profile_seconds=profile_seconds,
        profile_print=profile_print,
        enable_secondary_affinity=enable_secondary_affinity,
        topk_candidates=topk_candidates,
    )

    if use_split_backward:
        return _generate_code_uniform1d_bwd_split_from_context(**common_kwargs)

    return _generate_code_uniform1d_bwd_fused_from_context(
        **common_kwargs,
        enable_cse=enable_cse,
        consider_cse=consider_cse,
    )


# =============================================================================
# Baseline full-unrolled codegen: no LARS, no CSE, no resident accumulator reuse
# =============================================================================

def _make_baseline_wxy_paths_from_uniform1d_lists(
    i_list: List[int],
    j_list: List[int],
    k_list: List[int],
    v_list: List[int],
    coeff_list: List[float],
    *,
    path_semantics: str,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Convert external Uniform1D path indices to baseline-native tuples:
        (w_index, x_index, y_index, out_index, coeff)

    path_semantics="wxy": external (i,j,k) means w[i], x[j], y[k].
    path_semantics="xyw": external (i,j,k) means x[i], y[j], w[k].

    This baseline intentionally does not perform path scheduling, operand reuse,
    pair CSE, or register-budget management.  The Python code generator simply
    emits one straight-line block per cg path, in the original path order.
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


def emit_fused_fwd_kernel_baseline_unrolled(
    paths_wxy: List[Tuple[int, int, int, int, float]],
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """
    Emit a forward baseline kernel with every cg path fully unrolled.

    Baseline policy:
      - no LARS schedule_result;
      - no pair CSE;
      - no resident output accumulator grouping;
      - every path reloads w/x/y from global memory and immediately updates out.

    This is intended as a diagnostic lower-level baseline for comparing the
    benefit/cost of LARS scheduling, pair CSE, and accumulator residency.
    """
    if block_size != 32:
        raise ValueError("this baseline emitter currently assumes block_size=32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    scalar_t* __restrict__ out,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
        ap("")
    ap("    const int e_orig = b_list ? b_list[e_local] : e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    ap("    const int x_row = " + ("src_idx[e_orig];" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src_idx[e_orig];" if use_y_src else "e_orig;"))
    ap("    const int out_row = " + ("dst_idx[e_orig];" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base = (index_t)w_row * (index_t)Iw * (index_t)U;")
    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base = (index_t)x_row * (index_t)Ix * (index_t)U;")
    if mode_scalar_y:
        ap("    const index_t y_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base = (index_t)y_row * (index_t)Ky * (index_t)U;")
    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t out_base = (index_t)out_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t out_base = (index_t)out_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")

    for pid, (wi, xj, yk, ov, coeff) in enumerate(paths_wxy):
        w_expr = _lars_label_index_expr(
            "w", int(wi),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=ix_dim,
            y_dim=ky_dim,
            w_dim=iw_dim,
            v_dim=v_dim,
        )
        x_expr = _lars_label_index_expr(
            "x", int(xj),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=ix_dim,
            y_dim=ky_dim,
            w_dim=iw_dim,
            v_dim=v_dim,
        )
        y_expr = _lars_label_index_expr(
            "y", int(yk),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=ix_dim,
            y_dim=ky_dim,
            w_dim=iw_dim,
            v_dim=v_dim,
        )
        out_expr = _lars_label_index_expr(
            "o", int(ov),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            x_dim=ix_dim,
            y_dim=ky_dim,
            w_dim=iw_dim,
            v_dim=v_dim,
        )
        c = _fmt_lars_float(float(coeff))
        ap(f"            // baseline path#{pid}: out[{int(ov)}] += w[{int(wi)}] * x[{int(xj)}] * y[{int(yk)}] * {c}")
        ap("            {")
        ap(f"                const scalar_t wv = w[{w_expr}];")
        ap(f"                const scalar_t xv = x[{x_expr}];")
        ap(f"                const scalar_t yv = y[{y_expr}];")
        ap(f"                const scalar_t delta = scalar_t({c}) * (wv * xv) * yv;")
        if use_scatter:
            ap(f"                atomicAdd(&out[{out_expr}], delta);")
        else:
            ap(f"                out[{out_expr}] += delta;")
        ap("            }")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32_baseline_fwd(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32_baseline_fwd(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32_baseline_fwd(a, b)) return false;")
    ap("    return mul_fits_int32_baseline_fwd(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_baseline_fwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32_baseline_fwd((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32_baseline_fwd(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32_baseline_fwd(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32_baseline_fwd(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t out_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool out_ok = mul3_fits_int32_baseline_fwd(out_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && out_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    ap("        w, x, y, out,")
    ap("        src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_baseline_fwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                            kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    ap("            w, x, y, out, src_idx, dst_idx, b_list,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    ap("            w, x, y, out, src_idx, dst_idx, b_list,")
    ap("            B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    scalar_t* out,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    ap("        w, x, y, out, src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def emit_fused_bwd_kernel_baseline_unrolled(
    paths_wxy: List[Tuple[int, int, int, int, float]],
    *,
    kernel_name: str,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    use_x_src: bool = False,
    use_y_src: bool = False,
    use_scatter: bool = False,
    u_dim: Optional[int] = None,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    block_size: int = 32,
) -> str:
    """
    Emit a backward baseline kernel with every cg path fully unrolled.

    Baseline policy:
      - no LARS input-register scheduling;
      - no pair CSE, e.g. no cached w*grad_out;
      - no resident gw/gx/gy accumulators;
      - every path reloads w/x/y/grad_out and immediately writes gradients.

    grad_x/grad_y use atomicAdd, matching the fused backward semantics.  grad_w
    intentionally uses a plain += path update to stay consistent with the
    existing non-atomic grad_w assumption in this codebase.
    """
    if block_size != 32:
        raise ValueError("this baseline emitter currently assumes block_size=32")
    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    mode_scalar_y = mode == "u,u,,u"
    lines: List[str] = []

    def ap(line: str = ""):
        lines.append(line)

    ap("#include <stdint.h>")
    ap("#include <torch/extension.h>")
    ap("#include <vector>")
    ap("#include <cstdint>")
    ap("")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("  #include <hip/hip_runtime.h>")
    ap("  #include <ATen/hip/HIPContext.h>")
    ap("  #include <c10/hip/HIPGuard.h>")
    ap("  using gpuStream_t = hipStream_t;")
    ap("  #define getCurrentGPUStream at::hip::getCurrentHIPStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_HIP_KERNEL_LAUNCH_CHECK()")
    ap("#else")
    ap("  #include <cuda.h>")
    ap("  #include <cuda_runtime.h>")
    ap("  #include <ATen/cuda/CUDAContext.h>")
    ap("  using gpuStream_t = cudaStream_t;")
    ap("  #define getCurrentGPUStream at::cuda::getCurrentCUDAStream")
    ap("  #define GPU_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()")
    ap("#endif")
    ap("")
    ap("using GPU_Guard = c10::DeviceGuard;")
    ap("")
    ap("template <typename scalar_t>")
    ap("__device__ __forceinline__ scalar_t warp_sum_xor_baseline_bwd(scalar_t v) {")
    ap("    for (int offset = 16; offset > 0; offset >>= 1) {")
    ap("#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)")
    ap("        v += __shfl_xor(v, offset);")
    ap("#else")
    ap("        v += __shfl_xor_sync(0xffffffff, v, offset);")
    ap("#endif")
    ap("    }")
    ap("    return v;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"__global__ void {kernel_name}(")
    ap("    const scalar_t* __restrict__ w,")
    ap("    const scalar_t* __restrict__ x,")
    ap("    const scalar_t* __restrict__ y,")
    ap("    const scalar_t* __restrict__ grad_out,")
    if need_grad_w:
        ap("    scalar_t* __restrict__ grad_w,")
    ap("    scalar_t* __restrict__ grad_x,")
    ap("    scalar_t* __restrict__ grad_y,")
    ap("    const int32_t* __restrict__ src_idx,")
    ap("    const int32_t* __restrict__ dst_idx,")
    ap("    const int32_t* __restrict__ b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S)")
    ap("{")
    ap("    const int e_local = (int)blockIdx.x;")
    ap("    if (e_local >= B) return;")
    ap("")
    ap("    const int tid  = (int)threadIdx.x;")
    ap("    const int lane = tid & 31;")
    ap("    if (tid >= 32) return;")
    ap("")
    if u_dim is not None:
        ap(f"    constexpr int U_CONST = {int(u_dim)};")
        ap("    (void)U_CONST;")
    ap("    const int e_orig = b_list ? b_list[e_local] : e_local;")
    ap("    const int w_row  = (WB == 1 ? 0 : e_orig);")
    ap("")
    if use_x_src or use_y_src:
        ap("    const int src = src_idx[e_orig];")
    else:
        ap("    const int src = e_orig;")
    if use_scatter:
        ap("    const int dst = dst_idx[e_orig];")
    else:
        ap("    const int dst = e_orig;")
    ap("")
    ap("    const int x_row = " + ("src;" if use_x_src else "e_local;"))
    ap("    const int y_row = " + ("src;" if use_y_src else "e_orig;"))
    ap("    const int go_row = " + ("dst;" if use_scatter else "e_orig;"))
    ap("")

    if iw_dim is not None and u_dim is not None:
        ap(f"    const index_t w_base  = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
        if need_grad_w:
            ap(f"    const index_t gw_base = (index_t)w_row * (index_t){int(iw_dim) * int(u_dim)};")
    else:
        ap("    const index_t w_base  = (index_t)w_row * (index_t)Iw * (index_t)U;")
        if need_grad_w:
            ap("    const index_t gw_base = (index_t)w_row * (index_t)Iw * (index_t)U;")

    if ix_dim is not None and u_dim is not None:
        ap(f"    const index_t x_base  = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
        ap(f"    const index_t gx_base = (index_t)x_row * (index_t){int(ix_dim) * int(u_dim)};")
    else:
        ap("    const index_t x_base  = (index_t)x_row * (index_t)Ix * (index_t)U;")
        ap("    const index_t gx_base = (index_t)x_row * (index_t)Ix * (index_t)U;")

    if mode_scalar_y:
        ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky;")
        ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky;")
    else:
        if ky_dim is not None and u_dim is not None:
            ap(f"    const index_t y_base  = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
            ap(f"    const index_t gy_base = (index_t)y_row * (index_t){int(ky_dim) * int(u_dim)};")
        else:
            ap("    const index_t y_base  = (index_t)y_row * (index_t)Ky * (index_t)U;")
            ap("    const index_t gy_base = (index_t)y_row * (index_t)Ky * (index_t)U;")

    if v_dim is not None and u_dim is not None:
        ap(f"    const index_t go_base = (index_t)go_row * (index_t){int(v_dim) * int(u_dim)};")
    else:
        ap("    const index_t go_base = (index_t)go_row * (index_t)V * (index_t)U;")
    ap("")

    ap("    for (int u_base = 0; u_base < U; u_base += 32) {")
    ap("        const int u = u_base + lane;")
    ap("        if (u < U) {")

    for pid, (wi, xj, yk, ov, coeff) in enumerate(paths_wxy):
        w_expr = _bwd_label_index_expr(
            "w", int(wi),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        gw_expr = _bwd_label_index_expr(
            "gw", int(wi),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        x_expr = _bwd_label_index_expr(
            "x", int(xj),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        gx_expr = _bwd_label_index_expr(
            "gx", int(xj),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        y_expr = _bwd_label_index_expr(
            "y", int(yk),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        gy_expr = _bwd_label_index_expr(
            "gy", int(yk),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        go_expr = _bwd_label_index_expr(
            "go", int(ov),
            mode_scalar_y=mode_scalar_y,
            u_dim=u_dim,
            iw_dim=iw_dim,
            ix_dim=ix_dim,
            ky_dim=ky_dim,
            v_dim=v_dim,
        )
        c = _fmt_lars_float(float(coeff))
        ap(f"            // baseline path#{pid}: immediate gradient writes for w[{int(wi)}], x[{int(xj)}], y[{int(yk)}], go[{int(ov)}]")
        ap("            {")
        ap(f"                const scalar_t wv = w[{w_expr}];")
        ap(f"                const scalar_t xv = x[{x_expr}];")
        ap(f"                const scalar_t yv = y[{y_expr}];")
        ap(f"                const scalar_t gov = grad_out[{go_expr}];")
        ap(f"                const scalar_t cg = scalar_t({c});")
        if need_grad_w:
            ap("                const scalar_t dgw = cg * gov * xv * yv;")
            ap(f"                grad_w[{gw_expr}] += dgw;")
        ap("                const scalar_t wg = wv * gov;")
        ap("                const scalar_t dgx = cg * wg * yv;")
        ap(f"                atomicAdd(&grad_x[{gx_expr}], dgx);")
        ap("                const scalar_t dgy = cg * wg * xv;")
        if mode_scalar_y:
            ap("                const scalar_t dgy_sum = warp_sum_xor_baseline_bwd(dgy);")
            ap("                if (lane == 0) {")
            ap(f"                    atomicAdd(&grad_y[{gy_expr}], dgy_sum);")
            ap("                }")
        else:
            ap(f"                atomicAdd(&grad_y[{gy_expr}], dgy);")
        ap("            }")

    ap("        }")
    ap("    }")
    ap("}")
    ap("")

    ap("static inline bool mul_fits_int32_baseline_bwd(int64_t a, int64_t b) {")
    ap("    if (a < 0 || b < 0) return false;")
    ap("    constexpr int64_t LIM = 2147483647LL;")
    ap("    if (a == 0 || b == 0) return true;")
    ap("    return a <= LIM / b;")
    ap("}")
    ap("")
    ap("static inline bool mul3_fits_int32_baseline_bwd(int64_t a, int64_t b, int64_t c) {")
    ap("    if (!mul_fits_int32_baseline_bwd(a, b)) return false;")
    ap("    return mul_fits_int32_baseline_bwd(a * b, c);")
    ap("}")
    ap("")
    ap("static inline bool should_use_int32_index_baseline_bwd(")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    bool use_x_src, bool use_y_src, bool use_scatter, bool mode_scalar_y)")
    ap("{")
    ap("    bool w_ok = mul3_fits_int32_baseline_bwd((int64_t)WB, (int64_t)Iw, (int64_t)U);")
    ap("    int64_t x_dim0 = use_x_src ? (int64_t)S : (int64_t)B;")
    ap("    bool x_ok = mul3_fits_int32_baseline_bwd(x_dim0, (int64_t)Ix, (int64_t)U);")
    ap("    int64_t y_dim0 = use_y_src ? (int64_t)S : (int64_t)B;")
    ap("    bool y_ok = mode_scalar_y ? mul_fits_int32_baseline_bwd(y_dim0, (int64_t)Ky)")
    ap("                              : mul3_fits_int32_baseline_bwd(y_dim0, (int64_t)Ky, (int64_t)U);")
    ap("    int64_t go_dim0 = use_scatter ? (int64_t)S : (int64_t)B;")
    ap("    bool go_ok = mul3_fits_int32_baseline_bwd(go_dim0, (int64_t)V, (int64_t)U);")
    ap("    return w_ok && x_ok && y_ok && go_ok;")
    ap("}")
    ap("")

    ap("template <typename scalar_t, typename index_t>")
    ap(f"void launch_{kernel_name}_typed(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    dim3 block({block_size});")
    ap("    dim3 grid(B);")
    ap(f"    {kernel_name}<scalar_t, index_t><<<grid, block, 0, stream>>>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list,")
    ap("        B, WB, Iw, Ix, Ky, V, U, S);")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}_auto(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    constexpr bool kUseXSrc = {'true' if use_x_src else 'false'};")
    ap(f"    constexpr bool kUseYSrc = {'true' if use_y_src else 'false'};")
    ap(f"    constexpr bool kUseScatter = {'true' if use_scatter else 'false'};")
    ap(f"    constexpr bool kModeScalarY = {'true' if mode_scalar_y else 'false'};")
    ap("    if (should_use_int32_index_baseline_bwd(B, WB, Iw, Ix, Ky, V, U, S,")
    ap("                                            kUseXSrc, kUseYSrc, kUseScatter, kModeScalarY)) {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int32_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, b_list, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    } else {")
    ap(f"        launch_{kernel_name}_typed<scalar_t, int64_t>(")
    if need_grad_w:
        ap("            w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("            w, x, y, grad_out, grad_x, grad_y,")
    ap("            src_idx, dst_idx, b_list, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("    }")
    ap("}")
    ap("")

    ap("template <typename scalar_t>")
    ap(f"void launch_{kernel_name}(")
    ap("    const scalar_t* w,")
    ap("    const scalar_t* x,")
    ap("    const scalar_t* y,")
    ap("    const scalar_t* grad_out,")
    if need_grad_w:
        ap("    scalar_t* grad_w,")
    ap("    scalar_t* grad_x,")
    ap("    scalar_t* grad_y,")
    ap("    const int32_t* src_idx,")
    ap("    const int32_t* dst_idx,")
    ap("    const int32_t* b_list,")
    ap("    int B, int WB, int Iw, int Ix, int Ky, int V, int U, int S,")
    ap("    gpuStream_t stream)")
    ap("{")
    ap(f"    launch_{kernel_name}_auto<scalar_t>(")
    if need_grad_w:
        ap("        w, x, y, grad_out, grad_w, grad_x, grad_y,")
    else:
        ap("        w, x, y, grad_out, grad_x, grad_y,")
    ap("        src_idx, dst_idx, b_list, B, WB, Iw, Ix, Ky, V, U, S, stream);")
    ap("}")
    ap("")

    return "\n".join(lines)


def generate_code_uniform1d_fwd_baseline_unrolled(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    out_path: str = "generated_uniform1d_fwd_baseline_unrolled.cu",
    kernel_name: str = "uniform1d_fwd_baseline_unrolled",
    scalar_t: str = "float",
    path_semantics: str = "wxy",
    return_metadata: bool = False,
):
    """
    Generate a forward baseline CUDA implementation by fully unrolling cg paths.

    This entry deliberately bypasses LARS and all reuse-aware scheduling.  It is
    meant to answer: what happens if we just emit the cg path list as straight-
    line code?
    """
    del scalar_t

    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    paths_wxy = _make_baseline_wxy_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"
    if kernel_name and kernel_name != "uniform1d_fwd_baseline_unrolled":
        base_kernel_name = kernel_name
    else:
        base_kernel_name = f"uniform1d_baseline_unrolled_u{u_dim}_path{P}_{mode_str}_{layout_tag}_fwd"

    code = emit_fused_fwd_kernel_baseline_unrolled(
        paths_wxy,
        kernel_name=base_kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        block_size=32,
    )

    code = code + "\n" + emit_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")

    if return_metadata:
        return {
            "name": "baseline_unrolled_fwd",
            "code": code,
            "kernel_name": base_kernel_name,
            "num_paths": int(P),
            "config": {
                "mode": mode,
                "path_semantics": path_semantics,
                "use_x_src": use_x_src,
                "use_y_src": use_y_src,
                "use_scatter": use_scatter,
                "manual_register_management": False,
                "manual_data_reuse": False,
                "fully_unrolled_paths": True,
            },
        }
    return code


def generate_code_uniform1d_bwd_baseline_unrolled(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    k_list: torch.Tensor,
    v_list: torch.Tensor,
    coeff_list: torch.Tensor,
    input_indices: Optional[Dict[int, Any]] = None,
    output_indices: Optional[Dict[int, Any]] = None,
    u_dim: int = 1,
    iw_dim: Optional[int] = None,
    ix_dim: Optional[int] = None,
    ky_dim: Optional[int] = None,
    v_dim: Optional[int] = None,
    mode: str = "u,u,,u",
    need_grad_w: bool = True,
    out_path: str = "generated_uniform1d_bwd_baseline_unrolled.cu",
    kernel_name: str = "uniform1d_bwd_baseline_unrolled",
    scalar_t: str = "float",
    path_semantics: str = "wxy",
    return_metadata: bool = False,
):
    """
    Generate a backward baseline CUDA implementation by fully unrolling cg paths.

    This entry deliberately bypasses LARS, pair CSE, and resident gradient
    accumulators.  Each path reloads w/x/y/grad_out and immediately writes
    grad_w/grad_x/grad_y.
    """
    del scalar_t

    if mode not in ("u,u,,u", "u,u,u,u"):
        raise ValueError(f"Unsupported mode: {mode}")

    input_indices = {} if input_indices is None else dict(input_indices)
    output_indices = {} if output_indices is None else dict(output_indices)

    use_x_src = 1 in input_indices
    use_y_src = 2 in input_indices
    use_scatter = 0 in output_indices

    assert i_list.ndim == j_list.ndim == k_list.ndim == v_list.ndim == coeff_list.ndim == 1
    P = i_list.numel()
    assert j_list.numel() == P and k_list.numel() == P and v_list.numel() == P and coeff_list.numel() == P

    i_cpu = _to_int_list(i_list)
    j_cpu = _to_int_list(j_list)
    k_cpu = _to_int_list(k_list)
    v_cpu = _to_int_list(v_list)
    c_cpu = _to_float_list(coeff_list)

    paths_wxy = _make_baseline_wxy_paths_from_uniform1d_lists(
        i_cpu, j_cpu, k_cpu, v_cpu, c_cpu,
        path_semantics=path_semantics,
    )

    mode_str = "uu_u" if mode == "u,u,,u" else "uuuu"
    layout_tag = f"xsrc{int(use_x_src)}_ysrc{int(use_y_src)}_scatter{int(use_scatter)}"
    grad_tag = "full" if need_grad_w else "nogradw"
    if kernel_name and kernel_name != "uniform1d_bwd_baseline_unrolled":
        base_kernel_name = kernel_name
    else:
        base_kernel_name = f"uniform1d_baseline_unrolled_u{u_dim}_path{P}_{mode_str}_{layout_tag}_{grad_tag}_bwd"

    code = emit_fused_bwd_kernel_baseline_unrolled(
        paths_wxy,
        kernel_name=base_kernel_name,
        mode=mode,
        need_grad_w=need_grad_w,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
        u_dim=u_dim,
        iw_dim=iw_dim,
        ix_dim=ix_dim,
        ky_dim=ky_dim,
        v_dim=v_dim,
        block_size=32,
    )

    code = code + "\n" + emit_lars_bwd_launcher(
        bundle_name=base_kernel_name,
        mode=mode,
        need_grad_w=need_grad_w,
        use_x_src=use_x_src,
        use_y_src=use_y_src,
        use_scatter=use_scatter,
    )

    if out_path:
        Path(out_path).write_text(code, encoding="utf-8")

    if return_metadata:
        return {
            "name": "baseline_unrolled_bwd",
            "code": code,
            "kernel_name": base_kernel_name,
            "num_paths": int(P),
            "config": {
                "mode": mode,
                "path_semantics": path_semantics,
                "need_grad_w": need_grad_w,
                "use_x_src": use_x_src,
                "use_y_src": use_y_src,
                "use_scatter": use_scatter,
                "manual_register_management": False,
                "manual_data_reuse": False,
                "resident_gradient_accumulators": False,
                "fully_unrolled_paths": True,
            },
        }
    return code
