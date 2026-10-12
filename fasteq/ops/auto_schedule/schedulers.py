"""FastEq scheduling: schedulers."""
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
    _lars_reg_id,
)
from .models import (
    BWD_OUTPUT_KINDS,
    BWD_SPLIT_KINDS,
    Inst,
    Label,
    STCDoubleBwdPath,
    STCPath,
    ScheduleResult,
    U1DBwdPath,
    U1DBwdSplitPath,
    U1DDoubleBwdPath,
    U1DPath,
)


class LARSUniform1DScheduler:
    """
    All-input-resident LARS-style scheduler for Uniform1D paths:

        out[v] += x[i] * y[j] * w[k] * c

    This scheduler no longer implements spill/victim selection.  It allocates
    one virtual register for every distinct input label, chooses labels by the
    LARS score, and fires paths as soon as their input labels are live.

    Notes:
      - ScheduleResult.spills is kept for compatibility and is always zero.
      - Output accumulators are handled by the emitter, not by the LARS label
        allocator.
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        *,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        debug: bool = False,
        path_kind: str = "u1d",
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        self.path_kind = str(path_kind)
        if self.path_kind == "u1d":
            self.paths: List[Any] = [
                U1DPath(pid=p, i=i, j=j, k=k, v=v, c=c)
                for p, (i, j, k, v, c) in enumerate(paths)
            ]
        elif self.path_kind == "stc":
            # STC paths are already materialized as STCPath objects.  They expose
            # labels for live-set management and product_labels for multiplicity-
            # preserving emission, e.g. x1[i] * x1[i].
            self.paths = list(paths)
            for expected_pid, path in enumerate(self.paths):
                if int(path.pid) != expected_pid:
                    raise ValueError(
                        f"STC paths must be dense pid-ordered; "
                        f"got path.pid={path.pid} at position {expected_pid}"
                    )
        else:
            raise ValueError(f"Unsupported LARSUniform1DScheduler path_kind={self.path_kind!r}")

        # Input labels are unbounded: virtual registers are allocated on demand.

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
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

        # Forward output accumulator use counts.  out[v] with exactly one path
        # does not need a full-resident accumulator: emit one direct writeback
        # immediately after its computation instead of keeping out_acc_v alive
        # until the end of the kernel.
        self.output_uses: Counter[int] = Counter(int(p.v) for p in self.paths)

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = []
        self._next_reg_id = 0

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        # max_live is kept for backward compatibility. It records the maximum
        # number of live normal input labels in the logical schedule.
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}

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


    def _profile_snapshot(
        self,
        *,
        event: str,
        reason: str = "",
        fireable_n: Optional[int] = None,
        candidate_n: Optional[int] = None,
        select_ms: Optional[float] = None,
        lab: Optional[Label] = None,
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
            "elapsed_s": float(elapsed),
            "rate_paths_per_s": float(rate),
            "eta_s": float(eta),
            "reason": str(reason),
            "label": None if lab is None else self._label_name(lab),
        }
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
        )
        self._profile_records.append(snap)
        self._profile_last_event = event
        self._profile_last_reason = reason

        if self.profile_print:
            eta = self._fmt_profile_eta(snap["eta_s"])
            elapsed = self._fmt_profile_eta(snap["elapsed_s"])
            print(
                f"[LARS][profile] {snap['name']} {event:>8s} "
                f"iter={snap['iter']} "
                f"done={snap['done_paths']}/{snap['total_paths']} "
                f"remain={snap['remaining_paths']} "
                f"max=label/total "
                f"{snap['max_live_labels']}/{snap['max_live_total']} "
                f"fireable={snap['fireable_paths']} "
                f"candidates={snap['candidate_labels']} "
                f"label={snap['label']} "
                f"select={snap['select_ms']:.2f}ms "
                f"spills={snap['spills']} reloads={snap['reloads']} "
                f"insts={snap['instructions']} "
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
            "max_live": int(self.max_live),
            "max_live_labels": int(getattr(self, "max_live_labels", self.max_live)),
            "max_live_total": int(getattr(self, "max_live_total", self.max_live)),
            "spills": int(self.spills),
            "reloads": int(self.reloads),
            "instructions": int(len(self.instructions)),
            "select_rounds": int(self._profile_select_rounds),
            "fire_rounds": int(self._profile_fire_rounds),
            "load_rounds": int(self._profile_load_rounds),
            "elapsed_s": float(elapsed),
            "rate_paths_per_s": float(done / elapsed if elapsed > 0 else 0.0),
            "records": list(self._profile_records),
        }
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
        if self.free_regs:
            reg = self.free_regs.pop(0)
        else:
            reg = f"r{self._next_reg_id}"
            self._next_reg_id += 1

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
        self.reg_of.pop(lab, None)
        self.free_regs.insert(0, reg)
        self._invalidate_score_cache()

    # ------------------------------------------------------------------
    # Fireability
    # ------------------------------------------------------------------
    def _is_fireable(self, pid: int) -> bool:
        return self.path_label_sets[pid] <= self.live

    def _fireable_paths(self) -> List[int]:
        return [pid for pid in self.unscheduled if self.path_label_sets[pid] <= self.live]

    def _fireable_count_after_load(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
        count = 0
        for pid in self.unscheduled:
            if self.path_label_sets[pid] <= tmp_live:
                count += 1
        return count

    def _best_fireable_release_after_load(self, lab: Label) -> int:
        tmp_live = self.live | {lab}
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
    # Label selection / no-spill load
    # ------------------------------------------------------------------
    def _select_next_label_global(self) -> Tuple[Label, str]:
        candidates = self._candidate_labels()

        best_lab: Optional[Label] = None
        best_key = None

        for lab in candidates:
            lab_score = self._label_score(lab)
            fire_after = self._fireable_count_after_load(lab)
            release_after = self._best_fireable_release_after_load(lab)
            key = (
                fire_after,
                lab_score,
                #release_after,
                str(lab),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_lab = lab

        if best_lab is None:
            raise RuntimeError("No candidate label but no path is fireable.")
        return best_lab, f"global key={best_key}"

    def _load_label_no_spill(self, lab: Label, reason: str = "") -> None:
        if lab in self.live:
            return
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
            reuse_score = sum(self.remaining_uses[l] for l in labels)

            if getattr(self, "path_kind", "u1d") == "stc":
                # Prefer shorter products as a final tie-breaker; this keeps the
                # emitted STC expression compact when all other reuse signals tie.
                arity = len(getattr(p, "product_labels", p.labels))
                return (release_now, reuse_score, -arity, -pid)

            out_dirty = 1 if ("o", p.v) in self.dirty_outputs else 0
            return (release_now, out_dirty, reuse_score, -pid)

        return max(fireable, key=key)

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)

        if getattr(self, "path_kind", "u1d") == "stc":
            product_labels = tuple(getattr(p, "product_labels", p.labels))
            regs = tuple(self.reg_of[lab] for lab in product_labels)
            if self.output_uses[int(p.v)] <= 1:
                self.instructions.append(
                    Inst(
                        "mul_stc_direct",
                        (int(p.v), regs, float(p.c)),
                        f"path#{pid}: direct out[{p.v}] += c * product",
                    )
                )
            else:
                self.instructions.append(
                    Inst(
                        "mul_stc_resident",
                        (int(p.v), regs, float(p.c)),
                        f"path#{pid}: out[{p.v}] += c * product",
                    )
                )
        else:
            lx, ly, lw = p.labels

            rx = self.reg_of[lx]
            ry = self.reg_of[ly]
            rw = self.reg_of[lw]

            if self.output_uses[int(p.v)] <= 1:
                self.instructions.append(
                    Inst(
                        "fma_u1d_direct",
                        (p.v, rx, ry, rw, p.c),
                        (
                            f"path#{pid}: direct out[{p.v}] += "
                            f"x[{p.i}] * y[{p.j}] * w[{p.k}] * {p.c}"
                        ),
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
            lab, reason = self._select_next_label_global()
            select_ms = (perf_counter() - select_t0) * 1000.0
            self._profile_select_rounds += 1

            self._load_label_no_spill(lab, reason=reason)
            self._profile_load_rounds += 1
            self._profile_emit(
                event="load",
                reason=reason,
                fireable_n=fireable_n,
                candidate_n=candidate_n,
                select_ms=select_ms,
                lab=lab,
            )

        for lab in list(self.live):
            self._store_and_release(lab, reason="end of schedule")

        self._profile_emit(event="end", force=True, reason="schedule end")
        profile_summary = self._build_profile_summary()

        return ScheduleResult(
            instructions=self.instructions,
            path_order=self.path_order,
            max_live=self.max_live,
            spills=0,
            reloads=self.reloads,
            final_reg_map=dict(self.reg_of),
            profile=profile_summary,
        )


class ProgressLARSUniform1DScheduler(LARSUniform1DScheduler):
    """Progress-printing wrapper for the no-spill all-input-resident scheduler."""

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

        self._t0 = perf_counter()
        self._last_print_t = self._t0
        self._last_print_done = 0

        self._last_fireable_n = 0
        self._last_candidate_n = 0
        self._last_best_lab = None
        self._last_best_score = None
        self._last_select_ms = 0.0
        self._last_select_mode = ""
        self._last_select_reason = ""

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
        try:
            return len(self._candidate_labels())
        except Exception:
            return 0

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
            f"live={len(self.live)} "
            f"fireable={self._last_fireable_n} "
            f"candidates={self._last_candidate_n} "
            f"mode={self._last_select_mode} "
            f"best={self._fmt_lab(self._last_best_lab)} "
            f"score={self._last_best_score} "
            f"select={self._last_select_ms:.2f}ms "
            f"reloads={self.reloads} "
            f"insts={len(self.instructions)} "
            f"rate={rate:.1f} path/s "
            f"elapsed={self._fmt_eta(elapsed)} eta={self._fmt_eta(eta)} "
            f"reason={self._last_select_reason}",
            flush=True,
        )

        self._last_print_t = now
        self._last_print_done = done

    def _select_label_with_progress(self):
        t0 = perf_counter()
        self._last_candidate_n = self._candidate_count()
        lab, reason = self._select_next_label_global()
        self._select_rounds += 1
        self._last_select_ms = (perf_counter() - t0) * 1000.0
        self._last_best_lab = lab
        self._last_select_mode = "global"
        self._last_select_reason = str(reason)
        try:
            self._last_best_score = self._label_score(lab)
        except Exception:
            self._last_best_score = None
        return lab, reason

    def schedule(self) -> ScheduleResult:
        self._print_progress(force=True, event="start")

        while self.unscheduled:
            self._loop_iter += 1
            fireable = self._fireable_paths()
            self._last_fireable_n = len(fireable)

            if fireable:
                pid = self._choose_fireable_path(fireable)
                self._emit_path_compute(pid)
                self._fire_rounds += 1
                self._last_select_mode = "fire"
                self._last_best_lab = None
                self._last_best_score = None
                self._last_select_reason = f"path#{pid}"
                self._print_progress(event="fire")
                continue

            lab, reason = self._select_label_with_progress()
            self._load_rounds += 1
            self._load_label_no_spill(
                lab,
                reason=(
                    f"LARS-score={self._last_best_score}, "
                    f"mode={self._last_select_mode}, "
                    f"select_reason={reason}"
                ),
            )
            self._print_progress(event="load")

        for lab in list(self.live):
            self._store_and_release(lab, reason="end of schedule")

        self._print_progress(force=True, event="end")

        if self.verbose:
            print(
                f"[LARS] summary "
                f"select_rounds={self._select_rounds} "
                f"fire_rounds={self._fire_rounds} "
                f"load_rounds={self._load_rounds} "
                f"max_live={self.max_live} "
                f"spills=0 "
                f"reloads={self.reloads}",
                flush=True,
            )

        return ScheduleResult(
            instructions=self.instructions,
            path_order=self.path_order,
            max_live=self.max_live,
            spills=0,
            reloads=self.reloads,
            final_reg_map=dict(self.reg_of),
            profile=self._build_profile_summary(),
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
        *,
        need_grad_w: bool = True,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        # Backward reuses the generic forward scheduler helpers such as
        # _choose_fireable_path().  Those helpers now branch on path_kind for
        # STC, so make the backward kind explicit and keep it on the normal
        # Uniform1D branch.
        self.path_kind = "u1d_bwd"
        self.need_grad_w = bool(need_grad_w)
        self.paths: List[U1DBwdPath] = [
            U1DBwdPath(pid=p, i=i, j=j, k=k, v=v, c=c, need_grad_w=self.need_grad_w)
            for p, (i, j, k, v, c) in enumerate(paths)
        ]

        # Backward input labels are unbounded: virtual registers are allocated on demand.

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
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
        self.free_regs: List[str] = []
        self._next_reg_id = 0

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}

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
        self.reg_of.pop(lab, None)
        self.free_regs.insert(0, reg)
        self._invalidate_score_cache()


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


class LARSUniform1DBwdSplitScheduler(LARSUniform1DScheduler):
    """
    Per-gradient backward scheduler.  It keeps the no-spill LARS label
    selection policy, but builds labels for a single gradient target only.
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        *,
        grad_kind: str,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
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

        # Split-backward input labels are unbounded: virtual registers are allocated on demand.

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
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
        self.free_regs: List[str] = []
        self._next_reg_id = 0

        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0

        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}

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


class LARSUniform1DDoubleBwdScheduler(LARSUniform1DScheduler):
    """LARS scheduler for the fully-unrolled Uniform1D double backward.

    Only read-only operands participate in the LARS live set.  The four output
    gradients are emitted as atomic accumulations, which is required for all
    supported layouts (source indirection, scatter, WB==1, and scalar-y).
    """

    def __init__(
        self,
        paths: Iterable[Tuple[int, int, int, int, float]],
        *,
        need_grad_w: bool = True,
        need_grad_x: bool = True,
        need_grad_y: bool = True,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = True,
    ):
        self.path_kind = "u1d_double_bwd"
        self.need_grad_w = bool(need_grad_w)
        self.need_grad_x = bool(need_grad_x)
        self.need_grad_y = bool(need_grad_y)
        self.paths: List[U1DDoubleBwdPath] = [
            U1DDoubleBwdPath(
                pid=p, i=i, j=j, k=k, v=v, c=c,
                need_grad_w=self.need_grad_w,
                need_grad_x=self.need_grad_x,
                need_grad_y=self.need_grad_y,
            )
            for p, (i, j, k, v, c) in enumerate(paths)
        ]

        self.enable_secondary_affinity = bool(enable_secondary_affinity)
        self.topk_candidates = topk_candidates
        self.debug = bool(debug)

        self.label_to_paths: Dict[Label, Set[int]] = defaultdict(set)
        for path in self.paths:
            for lab in path.labels:
                self.label_to_paths[lab].add(path.pid)

        self.path_label_sets: List[Set[Label]] = [set(p.labels) for p in self.paths]
        self.unscheduled: Set[int] = set(p.pid for p in self.paths)
        self.remaining_uses: Counter[Label] = Counter()
        for path in self.paths:
            for lab in path.labels:
                self.remaining_uses[lab] += 1

        # The generic scheduler expects this attribute for forward bookkeeping.
        self.output_uses: Counter[int] = Counter(int(p.v) for p in self.paths)

        self.live: Set[Label] = set()
        self.dirty_outputs: Set[Label] = set()
        self.reg_of: Dict[Label, str] = {}
        self.free_regs: List[str] = []
        self._next_reg_id = 0
        self.instructions: List[Inst] = []
        self.path_order: List[int] = []
        self.spills = 0
        self.reloads = 0
        self.max_live = 0
        self.max_live_labels = 0
        self.max_live_pairs = 0
        self.max_live_total = 0
        self._score_cache: Dict[Label, Tuple[int, int, int, int, int, int]] = {}

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
        self._profile_records: List[Dict[str, Any]] = []
        self._profile_last_event = ""
        self._profile_last_reason = ""

    def _path(self, pid: int) -> U1DDoubleBwdPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        names = {
            "go": "grad_out",
            "ggw": "grad_grad_w",
            "ggx": "grad_grad_x",
            "ggy": "grad_grad_y",
        }
        return f"{names.get(kind, kind)}[{idx}]"

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)
        reg = self.reg_of
        rw = reg[("w", p.i)]
        rx = reg[("x", p.j)]
        ry = reg[("y", p.k)]
        rgo = reg[("go", p.v)]
        rggw = reg[("ggw", p.i)] if self.need_grad_w else ""
        rggx = reg[("ggx", p.j)] if self.need_grad_x else ""
        rggy = reg[("ggy", p.k)] if self.need_grad_y else ""

        self.instructions.append(Inst(
            "double_bwd_fma_resident",
            (
                p.i, p.j, p.k, p.v,
                rw, rx, ry, rgo, rggw, rggx, rggy,
                p.c, self.need_grad_w, self.need_grad_x, self.need_grad_y,
            ),
            (
                f"path#{pid}: double backward; "
                f"w[{p.i}], x[{p.j}], y[{p.k}], go[{p.v}], c={p.c}"
            ),
        ))
        self.path_order.append(pid)
        self.unscheduled.remove(pid)

        for lab in p.labels:
            self.remaining_uses[lab] -= 1
        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")


class LARSSTCDoubleBwdScheduler(LARSUniform1DScheduler):
    """LARS scheduler for STC double backward, including output accumulators."""

    _ACC_KINDS = {"dgo", "dx0", "dx1"}

    def __init__(
        self,
        paths: Sequence[STCPath],
        *,
        enable_secondary_affinity: bool = True,
        topk_candidates: Optional[int] = None,
        debug: bool = False,
        profile: bool = False,
        profile_name: str = "stc_double_bwd_lars",
        profile_interval: int = 1000,
        profile_seconds: float = 2.0,
        profile_print: bool = False,
    ):
        double_paths = [
            STCDoubleBwdPath(
                pid=int(p.pid),
                x1_indices=tuple(int(v) for v in p.x1_indices),
                x0_index=int(p.x0_index),
                v=int(p.v),
                c=float(p.c),
            )
            for p in paths
        ]
        # Reuse the mature STC LARS selection logic.  We override path emission,
        # accumulator initialization, and accumulator writeback below.
        super().__init__(
            double_paths,
            path_kind="stc",
            enable_secondary_affinity=enable_secondary_affinity,
            topk_candidates=topk_candidates,
            debug=debug,
            profile=profile,
            profile_name=profile_name,
            profile_interval=profile_interval,
            profile_seconds=profile_seconds,
            profile_print=profile_print,
        )

    def _path(self, pid: int) -> STCDoubleBwdPath:
        return self.paths[pid]

    def _label_name(self, lab: Label) -> str:
        kind, idx = lab
        names = {
            "go": "grad_out",
            "ggx0": "grad_grad_x0",
            "ggx1": "grad_grad_x1",
            "dgo": "d_grad_out",
            "dx0": "d_x0",
            "dx1": "d_x1",
        }
        return f"{names.get(kind, kind)}[{idx}]"

    def _priority(self, lab: Label) -> int:
        # Accumulator labels benefit from staying resident across neighboring
        # paths just like forward output accumulators.
        return int(self.remaining_uses[lab]) + (1 if lab[0] in self._ACC_KINDS else 0)

    def _candidate_labels(self) -> List[Label]:
        candidates = {
            lab for pid in self.unscheduled
            for lab in self.path_label_sets[pid]
            if lab not in self.live
        }
        if not candidates:
            raise RuntimeError("No candidate label but no path is fireable.")
        if self.topk_candidates is not None and len(candidates) > self.topk_candidates:
            ranked = sorted(
                candidates,
                key=lambda lab: (
                    int(self.remaining_uses[lab]),
                    len(self.label_to_paths[lab] & self.unscheduled),
                    1 if lab[0] in self._ACC_KINDS else 0,
                    str(lab),
                ),
                reverse=True,
            )
            return ranked[: int(self.topk_candidates)]
        return list(candidates)

    def _emit_load_inst(self, lab: Label, reg: str, reason: str) -> None:
        if lab[0] in self._ACC_KINDS:
            self.instructions.append(
                Inst("init_acc", (reg, self._label_name(lab)), reason)
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
        if lab[0] in self._ACC_KINDS:
            # All accumulator labels are initialized to zero and every path that
            # references one contributes before its final remaining_use reaches 0.
            self.instructions.append(
                Inst("store_acc", (self._label_name(lab), reg), reason)
            )
            self.dirty_outputs.discard(lab)
        else:
            self.instructions.append(
                Inst("release", (reg, self._label_name(lab)), reason)
            )
        self.live.remove(lab)
        self.reg_of.pop(lab, None)
        self.free_regs.append(reg)
        self.free_regs.sort(key=_lars_reg_id)
        self._invalidate_score_cache()

    def _emit_path_compute(self, pid: int) -> None:
        p = self._path(pid)
        reg = self.reg_of
        rgo = reg[("go", int(p.v))]
        rx0 = reg[("x0", int(p.x0_index))]
        rggx0 = reg[("ggx0", int(p.x0_index))]
        rx1 = tuple(reg[("x1", int(idx))] for idx in p.x1_indices)
        rgg = tuple(reg[("ggx1", int(idx))] for idx in p.x1_indices)
        rdgo = reg[("dgo", int(p.v))]
        rdx0 = reg[("dx0", int(p.x0_index))]
        rdx1 = tuple(reg[("dx1", int(idx))] for idx in p.x1_indices)
        self.instructions.append(Inst(
            "stc_double_bwd_path",
            (
                tuple(int(v) for v in p.x1_indices),
                int(p.x0_index), int(p.v), float(p.c),
                rgo, rx0, rggx0, rx1, rgg, rdgo, rdx0, rdx1,
            ),
            f"path#{pid}: STC double backward",
        ))
        for lab in p.labels:
            if lab[0] in self._ACC_KINDS:
                self.dirty_outputs.add(lab)
        self.path_order.append(pid)
        self.unscheduled.remove(pid)
        for lab in p.labels:
            self.remaining_uses[lab] -= 1
        for lab in p.labels:
            if self.remaining_uses[lab] == 0 and lab in self.live:
                self._store_and_release(lab, reason=f"last use after path#{pid}")
