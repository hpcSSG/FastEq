"""Compatibility entry point for the split FastEq scheduler/code generators.

Existing relative imports keep working. Implementation lives in auto_schedule/.
"""
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

from .auto_schedule.baseline_codegen import (
    _baseline_path_write_targets,
    _baseline_unroll_block_parallelism,
    _dump_baseline_unroll_parallelism_stats,
    _make_baseline_wxy_paths_from_uniform1d_lists,
    _print_baseline_unroll_parallelism_stats,
    _resolve_baseline_unroll_size,
    analyze_baseline_unroll_parallelism,
    emit_fused_bwd_kernel_baseline_unrolled,
    emit_fused_fwd_kernel_baseline_unrolled,
    generate_code_uniform1d_bwd_baseline_unrolled,
    generate_code_uniform1d_fwd_baseline_unrolled,
)

from .auto_schedule.cuda_common import (
    _bwd_label_index_expr,
    _cuda_scalar_warp_sum_preamble,
    _emit_backend_gradient_atomic,
    _fmt_float,
    _fmt_lars_float,
    _index_expr,
    _lars_has_output_reload_after_store,
    _lars_label_index_expr,
    _lars_reg_id,
    _max_lars_reg_count_any,
    _max_reg_count_any,
    _parse_bwd_lars_label_ref,
    _parse_label_ref,
    _parse_lars_label_ref,
    _parse_stc_placement_ref,
    _product_expr,
    _sanitize_cuda_comment,
    _split_grad_acc_name,
    _to_float_list,
    _to_int_list,
)

from .auto_schedule.metadata import (
    _check_stc_path_bounds_for_bwd,
    _infer_stc_path_lens_tensor_from_padded,
    _make_lars_bwd_paths_from_uniform1d_lists,
    _make_lars_paths_from_uniform1d_lists,
    _normalize_stc_padded_paths,
    infer_stc_path_lens_from_padded,
    make_stc_paths_from_padded_lists,
)

from .auto_schedule.models import (
    BWD_OUTPUT_KINDS,
    BWD_SPLIT_KINDS,
    BwdCodegenContext,
    BwdPairKey,
    Inst,
    LARSPlacementConfig,
    Label,
    PairKey,
    STCDoubleBwdPath,
    STCPath,
    STC_PAD_VALUE,
    ScheduleResult,
    U1DBwdPath,
    U1DBwdSplitPath,
    U1DDoubleBwdPath,
    U1DPath,
)

from .auto_schedule.placement import (
    _extract_lars_compute_records,
    _placement_assign_tokens,
    _placement_peak_live,
    _placement_ref_parts,
    _resolve_lars_placement_config,
    apply_lars_lifetime_placement,
    dump_lars_variable_stats,
)

from .auto_schedule.schedule_stats import (
    _DEFAULT_SCHEDULE_STATS_ENABLED,
    _SCHEDULE_STATS_DUMPED,
    _build_schedule_label_stats,
    _dump_and_print_schedule_label_stats,
    _dump_schedule_stats_dir,
    _schedule_stats_enabled,
    _schedule_stats_safe_tag,
    _schedule_stats_sha1,
    _stats_labels_stc_bwd,
    _stats_labels_stc_double_bwd,
    _stats_labels_stc_fwd,
    _stats_labels_uniform1d_bwd_fused,
    _stats_labels_uniform1d_bwd_split,
    _stats_labels_uniform1d_double_bwd,
    _stats_labels_uniform1d_fwd,
)

from .auto_schedule.schedulers import (
    LARSSTCDoubleBwdScheduler,
    LARSUniform1DBwdScheduler,
    LARSUniform1DBwdSplitScheduler,
    LARSUniform1DDoubleBwdScheduler,
    LARSUniform1DScheduler,
    ProgressLARSUniform1DScheduler,
)

from .auto_schedule.stc_backward import (
    _make_stc_bwd_logical_schedule,
    _stc_bwd_grad_expr,
    _stc_grad_x0_acc_name,
    _stc_grad_x1_acc_name,
    _stc_x1_global_expr,
    emit_stc_bwd_kernel_from_paths,
    generate_code_stc_bwd_with_scheduler,
)

from .auto_schedule.stc_double_backward import (
    emit_stc_double_bwd_kernel_from_lars_schedule,
    emit_stc_double_bwd_launcher,
    generate_code_stc_double_bwd_with_scheduler,
)

from .auto_schedule.stc_forward import (
    emit_stc_fwd_kernel_from_lars_schedule,
    generate_code_stc_fwd_with_scheduler,
)

from .auto_schedule.triton_backend import (
    TritonEmitter,
    _TRITON_RUNTIME_MODULE_CACHE,
    _triton_bind_run_entry,
    _triton_emit_zero,
    _triton_float,
    _triton_mul,
    _triton_place_register_only,
    _triton_register_only_placement_config,
    _triton_source_preamble,
    build_triton_stc_bwd_module,
    build_triton_stc_bwd_module_source,
    build_triton_stc_double_bwd_module,
    build_triton_stc_double_bwd_module_source,
    build_triton_stc_fwd_module,
    build_triton_stc_fwd_module_source,
    build_triton_uniform1d_bwd_module,
    build_triton_uniform1d_bwd_module_source,
    build_triton_uniform1d_double_bwd_module,
    build_triton_uniform1d_double_bwd_module_source,
    build_triton_uniform1d_fwd_module,
    build_triton_uniform1d_fwd_module_source,
    load_triton_module_from_source,
)

from .auto_schedule.uniform1d_backward import (
    _bwd_base_kernel_name,
    _finalize_codegen_candidates,
    _generate_code_uniform1d_bwd_fused_from_context,
    _generate_code_uniform1d_bwd_split_from_context,
    _prepare_uniform1d_bwd_codegen_context,
    _write_codegen_candidate,
    emit_fused_bwd_kernel_from_lars_schedule,
    emit_lars_bwd_launcher,
    emit_lars_bwd_split_kernel_from_schedule,
    emit_lars_bwd_split_launcher,
    emit_lars_bwd_split_preamble,
    generate_code_uniform1d_bwd_split_with_scheduler,
    generate_code_uniform1d_bwd_with_scheduler,
)

from .auto_schedule.uniform1d_double_backward import (
    emit_uniform1d_double_bwd_kernel_from_lars_schedule,
    emit_uniform1d_double_bwd_launcher,
    generate_code_uniform1d_double_bwd_with_scheduler,
)

from .auto_schedule.uniform1d_forward import (
    emit_fused_fwd_kernel_from_lars_schedule,
    emit_launcher,
    generate_code_uniform1d_fwd_with_scheduler,
)

# Forward assignments to existing symbols to their defining module and importers.
# This preserves legacy configuration assignments and monkeypatch-based callers.
import sys as _sys
import types as _types
from . import auto_schedule as _implementation

_SYMBOL_OWNERS = {'BWD_OUTPUT_KINDS': 'models',
 'BWD_SPLIT_KINDS': 'models',
 'BwdCodegenContext': 'models',
 'BwdPairKey': 'models',
 'Inst': 'models',
 'LARSPlacementConfig': 'models',
 'LARSSTCDoubleBwdScheduler': 'schedulers',
 'LARSUniform1DBwdScheduler': 'schedulers',
 'LARSUniform1DBwdSplitScheduler': 'schedulers',
 'LARSUniform1DDoubleBwdScheduler': 'schedulers',
 'LARSUniform1DScheduler': 'schedulers',
 'Label': 'models',
 'PairKey': 'models',
 'ProgressLARSUniform1DScheduler': 'schedulers',
 'STCDoubleBwdPath': 'models',
 'STCPath': 'models',
 'STC_PAD_VALUE': 'models',
 'ScheduleResult': 'models',
 'TritonEmitter': 'triton_backend',
 'U1DBwdPath': 'models',
 'U1DBwdSplitPath': 'models',
 'U1DDoubleBwdPath': 'models',
 'U1DPath': 'models',
 '_DEFAULT_SCHEDULE_STATS_ENABLED': 'schedule_stats',
 '_SCHEDULE_STATS_DUMPED': 'schedule_stats',
 '_TRITON_RUNTIME_MODULE_CACHE': 'triton_backend',
 '_baseline_path_write_targets': 'baseline_codegen',
 '_baseline_unroll_block_parallelism': 'baseline_codegen',
 '_build_schedule_label_stats': 'schedule_stats',
 '_bwd_base_kernel_name': 'uniform1d_backward',
 '_bwd_label_index_expr': 'cuda_common',
 '_check_stc_path_bounds_for_bwd': 'metadata',
 '_cuda_scalar_warp_sum_preamble': 'cuda_common',
 '_dump_and_print_schedule_label_stats': 'schedule_stats',
 '_dump_baseline_unroll_parallelism_stats': 'baseline_codegen',
 '_dump_schedule_stats_dir': 'schedule_stats',
 '_emit_backend_gradient_atomic': 'cuda_common',
 '_extract_lars_compute_records': 'placement',
 '_finalize_codegen_candidates': 'uniform1d_backward',
 '_fmt_float': 'cuda_common',
 '_fmt_lars_float': 'cuda_common',
 '_generate_code_uniform1d_bwd_fused_from_context': 'uniform1d_backward',
 '_generate_code_uniform1d_bwd_split_from_context': 'uniform1d_backward',
 '_index_expr': 'cuda_common',
 '_infer_stc_path_lens_tensor_from_padded': 'metadata',
 '_lars_has_output_reload_after_store': 'cuda_common',
 '_lars_label_index_expr': 'cuda_common',
 '_lars_reg_id': 'cuda_common',
 '_make_baseline_wxy_paths_from_uniform1d_lists': 'baseline_codegen',
 '_make_lars_bwd_paths_from_uniform1d_lists': 'metadata',
 '_make_lars_paths_from_uniform1d_lists': 'metadata',
 '_make_stc_bwd_logical_schedule': 'stc_backward',
 '_max_lars_reg_count_any': 'cuda_common',
 '_max_reg_count_any': 'cuda_common',
 '_normalize_stc_padded_paths': 'metadata',
 '_parse_bwd_lars_label_ref': 'cuda_common',
 '_parse_label_ref': 'cuda_common',
 '_parse_lars_label_ref': 'cuda_common',
 '_parse_stc_placement_ref': 'cuda_common',
 '_placement_assign_tokens': 'placement',
 '_placement_peak_live': 'placement',
 '_placement_ref_parts': 'placement',
 '_prepare_uniform1d_bwd_codegen_context': 'uniform1d_backward',
 '_print_baseline_unroll_parallelism_stats': 'baseline_codegen',
 '_product_expr': 'cuda_common',
 '_resolve_baseline_unroll_size': 'baseline_codegen',
 '_resolve_lars_placement_config': 'placement',
 '_sanitize_cuda_comment': 'cuda_common',
 '_schedule_stats_enabled': 'schedule_stats',
 '_schedule_stats_safe_tag': 'schedule_stats',
 '_schedule_stats_sha1': 'schedule_stats',
 '_split_grad_acc_name': 'cuda_common',
 '_stats_labels_stc_bwd': 'schedule_stats',
 '_stats_labels_stc_double_bwd': 'schedule_stats',
 '_stats_labels_stc_fwd': 'schedule_stats',
 '_stats_labels_uniform1d_bwd_fused': 'schedule_stats',
 '_stats_labels_uniform1d_bwd_split': 'schedule_stats',
 '_stats_labels_uniform1d_double_bwd': 'schedule_stats',
 '_stats_labels_uniform1d_fwd': 'schedule_stats',
 '_stc_bwd_grad_expr': 'stc_backward',
 '_stc_grad_x0_acc_name': 'stc_backward',
 '_stc_grad_x1_acc_name': 'stc_backward',
 '_stc_x1_global_expr': 'stc_backward',
 '_to_float_list': 'cuda_common',
 '_to_int_list': 'cuda_common',
 '_triton_bind_run_entry': 'triton_backend',
 '_triton_emit_zero': 'triton_backend',
 '_triton_float': 'triton_backend',
 '_triton_mul': 'triton_backend',
 '_triton_place_register_only': 'triton_backend',
 '_triton_register_only_placement_config': 'triton_backend',
 '_triton_source_preamble': 'triton_backend',
 '_write_codegen_candidate': 'uniform1d_backward',
 'analyze_baseline_unroll_parallelism': 'baseline_codegen',
 'apply_lars_lifetime_placement': 'placement',
 'build_triton_stc_bwd_module': 'triton_backend',
 'build_triton_stc_bwd_module_source': 'triton_backend',
 'build_triton_stc_double_bwd_module': 'triton_backend',
 'build_triton_stc_double_bwd_module_source': 'triton_backend',
 'build_triton_stc_fwd_module': 'triton_backend',
 'build_triton_stc_fwd_module_source': 'triton_backend',
 'build_triton_uniform1d_bwd_module': 'triton_backend',
 'build_triton_uniform1d_bwd_module_source': 'triton_backend',
 'build_triton_uniform1d_double_bwd_module': 'triton_backend',
 'build_triton_uniform1d_double_bwd_module_source': 'triton_backend',
 'build_triton_uniform1d_fwd_module': 'triton_backend',
 'build_triton_uniform1d_fwd_module_source': 'triton_backend',
 'dump_lars_variable_stats': 'placement',
 'emit_fused_bwd_kernel_baseline_unrolled': 'baseline_codegen',
 'emit_fused_bwd_kernel_from_lars_schedule': 'uniform1d_backward',
 'emit_fused_fwd_kernel_baseline_unrolled': 'baseline_codegen',
 'emit_fused_fwd_kernel_from_lars_schedule': 'uniform1d_forward',
 'emit_lars_bwd_launcher': 'uniform1d_backward',
 'emit_lars_bwd_split_kernel_from_schedule': 'uniform1d_backward',
 'emit_lars_bwd_split_launcher': 'uniform1d_backward',
 'emit_lars_bwd_split_preamble': 'uniform1d_backward',
 'emit_launcher': 'uniform1d_forward',
 'emit_stc_bwd_kernel_from_paths': 'stc_backward',
 'emit_stc_double_bwd_kernel_from_lars_schedule': 'stc_double_backward',
 'emit_stc_double_bwd_launcher': 'stc_double_backward',
 'emit_stc_fwd_kernel_from_lars_schedule': 'stc_forward',
 'emit_uniform1d_double_bwd_kernel_from_lars_schedule': 'uniform1d_double_backward',
 'emit_uniform1d_double_bwd_launcher': 'uniform1d_double_backward',
 'generate_code_stc_bwd_with_scheduler': 'stc_backward',
 'generate_code_stc_double_bwd_with_scheduler': 'stc_double_backward',
 'generate_code_stc_fwd_with_scheduler': 'stc_forward',
 'generate_code_uniform1d_bwd_baseline_unrolled': 'baseline_codegen',
 'generate_code_uniform1d_bwd_split_with_scheduler': 'uniform1d_backward',
 'generate_code_uniform1d_bwd_with_scheduler': 'uniform1d_backward',
 'generate_code_uniform1d_double_bwd_with_scheduler': 'uniform1d_double_backward',
 'generate_code_uniform1d_fwd_baseline_unrolled': 'baseline_codegen',
 'generate_code_uniform1d_fwd_with_scheduler': 'uniform1d_forward',
 'infer_stc_path_lens_from_padded': 'metadata',
 'load_triton_module_from_source': 'triton_backend',
 'make_stc_paths_from_padded_lists': 'metadata'}

class _CompatibilityModule(_types.ModuleType):
    def __setattr__(self, name, value):
        owner = _SYMBOL_OWNERS.get(name)
        if owner is not None:
            target = _sys.modules[f"{_implementation.__name__}.{owner}"]
            old = getattr(target, name)
            for module_name in set(_SYMBOL_OWNERS.values()):
                module = _sys.modules.get(f"{_implementation.__name__}.{module_name}")
                if module is not None and getattr(module, name, object()) is old:
                    setattr(module, name, value)
        super().__setattr__(name, value)

_sys.modules[__name__].__class__ = _CompatibilityModule
