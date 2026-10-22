"""FastEq Gluon code generation with scheduler-controlled shared placement.

Install beside triton_backend.py, schedulers.py, placement.py and models.py.
Arithmetic/host ABI are reused from TritonEmitter via a checked AST lowering.
Generated modules are standalone (they do not import this code generator).
"""
from __future__ import annotations

import ast
import copy
import importlib
import keyword
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


def _base():
    return importlib.import_module('.triton_backend', __package__)


def _parse_expr(source):
    return ast.parse(source, mode='eval').body


def _power_of_two(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value & (value - 1):
        raise ValueError(f'{name} must be a positive power of two')
    return value


@dataclass(frozen=True)
class _Instruction:
    op: str
    args: tuple
    comment: str = ''


def _adapt_schedule(schedule, shared_inputs=()):
    """Expose placed instructions to the existing arithmetic emitter.

    Reserved temporary names carry shared/direct semantics into the AST pass;
    they never become long-lived register variables in generated Gluon code.
    """
    instructions = []
    direct_inputs = {}
    shared_inputs = set(shared_inputs)
    slot_ids = []
    def find_slots(value):
        if isinstance(value, (tuple, list)):
            for item in value: find_slots(item)
        elif isinstance(value, str) and re.fullmatch(r's\d+', value):
            slot_ids.append(int(value[1:]))
    for inst in schedule.instructions: find_slots(inst.args)
    next_slot = max(slot_ids, default=-1) + 1
    manual_direct = {}

    def token(value):
        nonlocal next_slot
        if isinstance(value, (tuple, list)):
            return tuple(token(v) for v in value)
        if not isinstance(value, str):
            return value
        if re.fullmatch(r's\d+', value):
            return '_shared_' + value[1:]
        if value.startswith('d:'):
            ref = value[2:]
            match = re.fullmatch(r'([a-zA-Z_]\w*)\[(\d+)\]', ref)
            if match is None:
                raise ValueError(f'Invalid direct reference: {value}')
            kind, index = match.groups()
            output = kind in {'out', 'grad_w', 'grad_x', 'grad_y', 'grad_x0', 'grad_x1'}
            if not output and ref in shared_inputs:
                if ref not in manual_direct:
                    manual_direct[ref] = f'_shared_{next_slot}'
                    next_slot += 1
                    instructions.append(_Instruction('load', (manual_direct[ref], ref)))
                return manual_direct[ref]
            name = f'_direct_{"acc" if output else "input"}_{kind}_{index}'
            if not output:
                direct_inputs[name] = ref
            return name
        return value

    aliases = {}
    def rename(value):
        if isinstance(value, (tuple, list)):
            return tuple(rename(v) for v in value)
        return aliases.get(value, value) if isinstance(value, str) else value

    for inst in schedule.instructions:
        original_args = inst.args
        original_op = inst.op
        if original_op in ('load', 'load_shared', 'init_acc'):
            aliases.pop(str(original_args[0]), None)
        if original_op == 'load' and str(original_args[1]) in shared_inputs:
            aliases[str(original_args[0])] = f's{next_slot}'
            next_slot += 1
            original_op = 'load_shared'
        args = token(rename(original_args))
        op = {'load_shared': 'load', 'release_shared': 'release'}.get(original_op, original_op)
        # Direct input reads are inserted as bindings, then inlined by the AST
        # pass at each use. Direct accumulators collect only this instruction.
        used = set()
        def collect(v):
            if isinstance(v, (tuple, list)):
                for x in v: collect(x)
            elif isinstance(v, str) and v.startswith('_direct_'): used.add(v)
        collect(args)
        for name in sorted(used):
            if name.startswith('_direct_input_'):
                instructions.append(_Instruction('load', (name, direct_inputs[name])))
        outputs = {}
        for name in sorted(used):
            if name.startswith('_direct_acc_'):
                kind, index = name[len('_direct_acc_'):].rsplit('_', 1)
                ref = f'{kind}[{index}]'
                outputs[name] = ref
                instructions.append(_Instruction('init_acc', (name, ref)))
        instructions.append(_Instruction(op, args, getattr(inst, 'comment', '')))
        for name, ref in outputs.items():
            instructions.append(_Instruction('store_acc_placed', (ref, name)))
        if original_op in ('release', 'release_shared'):
            aliases.pop(str(original_args[0]), None)
    return SimpleNamespace(instructions=instructions)


class _LowerKernel(ast.NodeTransformer):
    def __init__(self, num_warps, preserve_shared):
        self.num_warps = num_warps
        self.preserve_shared = preserve_shared
        self.direct_reads = {}

    def _read(self, name):
        if self.preserve_shared:
            return _parse_expr(f'{name}.gather(_shared_indices, 0)')
        return _parse_expr(f'{name}.load(_register_layout)')

    def _store(self, name, value):
        # Scalar y operands are broadcast to the channel tile before storage.
        broadcast = ast.BinOp(value, ast.Add(), ast.Name('_zero', ast.Load()))
        return ast.Expr(ast.Call(ast.Attribute(ast.Name(name, ast.Load()), 'store', ast.Load()), [broadcast], []))

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id.startswith('_shared_') and re.fullmatch(r'_shared_\d+', node.id):
                return ast.copy_location(self._read(node.id), node)
            if node.id in self.direct_reads:
                return copy.deepcopy(self.direct_reads[node.id])
        return node

    def visit_Assign(self, node):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return self.generic_visit(node)
        name = node.targets[0].id
        value = self.visit(node.value)
        if name.startswith('_direct_input_'):
            self.direct_reads[name] = value
            return None
        if re.fullmatch(r'_shared_\d+', name):
            return ast.copy_location(self._store(name, value), node)
        # Widen rows before multiplication, not after an overflowing offset.
        if name in {'e', 'b', 'src', 'out_row', 'go_row', 'x0_b'}:
            value = ast.Call(ast.Attribute(value, 'to', ast.Load()), [_parse_expr('gl.int64')], [])
        node.value = value
        return node

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name) and re.fullmatch(r'_shared_\d+', node.target.id):
            value = ast.BinOp(self._read(node.target.id), node.op, self.visit(node.value))
            return ast.copy_location(self._store(node.target.id, value), node)
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'tl':
            node.value.id = 'gl'
        return node

    def visit_Call(self, node):
        direct_store = (
            isinstance(node.func, ast.Attribute) and node.func.attr == 'store'
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'tl'
            and len(node.args) > 1 and isinstance(node.args[1], ast.Name)
            and node.args[1].id.startswith('_direct_acc_')
        )
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'gl':
            if node.func.attr in {'arange', 'zeros'}:
                if node.func.attr == 'zeros':
                    node.func.attr = 'full'
                    node.args.insert(1, ast.Constant(0))
                node.keywords.append(ast.keyword(arg='layout', value=ast.Name('_register_layout', ast.Load())))
            if direct_store:
                node.func.attr = 'atomic_add'
        return node

    def lower(self, function):
        shared = sorted({n.id for n in ast.walk(function) if isinstance(n, ast.Name)
                         and re.fullmatch(r'_shared_\d+', n.id)}, key=lambda n: int(n[8:]))
        function = self.visit(function)
        function.decorator_list = [_parse_expr('gluon.jit')]
        setup = [
            f'_register_layout: gl.constexpr = gl.BlockedLayout([1], [32], [{self.num_warps}], [0])',
            '_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])',
            '_dtype: gl.constexpr = gl.float64 if IS_FP64 else gl.float32',
            '_zero = gl.full((BLOCK_U,), 0, _dtype, _register_layout)',
        ]
        if shared and self.preserve_shared:
            # Opaque identity indices prevent trivial store/load forwarding.
            # The identity is exact; native Gluon shared-gather tracks effects
            # and lets the compiler insert required shared-memory barriers.
            setup += [
                '_shared_indices = gl.inline_asm_elementwise("mov.u32 $0, $1;", constraints="=r,r", args=[gl.arange(0, BLOCK_U, layout=_register_layout)], dtype=gl.int32, is_pure=True, pack=1)'
            ]
        setup += [f'{name} = gl.allocate_shared_memory(_dtype, [BLOCK_U], _shared_layout)' for name in shared]
        function.body = ast.parse('\n'.join(setup)).body + function.body
        return function


def _validate_host_run(function):
    """Keep the original run ABI while checking its implicit assumptions."""
    function.body.insert(0, ast.Import(names=[ast.alias('torch')]))
    body = []
    for node in function.body:
        # Uniform1D uses the destination count when scattering; require aligned
        # one-dimensional index arrays instead of silently processing a prefix.
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'B' for t in node.targets):
            variables = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
            if 'dst_idx' in variables and 'src_idx' in variables:
                body += ast.parse('''
if src_idx is not w:
    if src_idx.ndim != 1:
        raise ValueError("src_idx must be one-dimensional")
if dst_idx is not w:
    if dst_idx.ndim != 1:
        raise ValueError("dst_idx must be one-dimensional")
    if src_idx is not w and src_idx.numel() != dst_idx.numel():
        raise ValueError("src_idx and dst_idx must have equal lengths")
''').body
        body.append(node)
    function.body = body
    return function


def _to_gluon(source, num_warps, preserve_shared):
    tree = ast.parse(source)
    body = ast.parse('''import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
''').body
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.FunctionDef):
            node = _LowerKernel(num_warps, preserve_shared).lower(node)
        elif isinstance(node, ast.ClassDef):
            for function in node.body:
                if isinstance(function, ast.FunctionDef) and function.name == 'run':
                    _validate_host_run(function)
        body.append(node)
    tree.body = body
    ast.fix_missing_locations(tree)
    result = ast.unparse(tree) + '\n'
    compile(result, '<generated-gluon>', 'exec')
    return result


class GluonEmitter:
    """Emit placed schedules, including shared inputs and accumulators.

    preserve_shared=True uses shared gather with opaque identity indices to
    discourage promotion back into registers. False uses native descriptor.load.
    Final register pressure/performance must still be checked after compilation.
    """
    def __init__(self, schedule_result, *, kernel_name, preserve_shared=False, shared_inputs=()):
        if not isinstance(kernel_name, str) or not kernel_name.isidentifier() or keyword.iskeyword(kernel_name):
            raise ValueError('kernel_name must be a Python identifier')
        self.schedule_result = schedule_result
        self.kernel_name = kernel_name
        self.preserve_shared = bool(preserve_shared)
        self.shared_inputs = tuple(shared_inputs or ())
        for ref in self.shared_inputs:
            if not re.fullmatch(r'(?:w|x|y|grad_out|x0|x1|grad_grad_w|grad_grad_x|grad_grad_y|grad_grad_x0|grad_grad_x1)\[\d+\]', ref):
                raise ValueError(f'Invalid shared input reference: {ref!r}')

    def _emit(self, method, **kwargs):
        num_warps = _power_of_two(kwargs.get('num_warps', 4), 'num_warps')
        _power_of_two(kwargs.get('block_u', 128), 'block_u')
        if kwargs.get('mode', 'u,u,,u') not in ('u,u,,u', 'u,u,u,u'):
            raise ValueError('Unsupported Uniform1D mode')
        source_schedule = self.schedule_result
        if method == 'emit_uniform1d_bwd_fused':
            # gy is an outer accumulator, independent of placement-token reuse.
            instructions = []
            for inst in source_schedule.instructions:
                if inst.op in ('init_acc', 'store_acc_placed'):
                    ref = inst.args[1] if inst.op == 'init_acc' else inst.args[0]
                    if _base()._parse_bwd_lars_label_ref(str(ref))[0] == 'gy':
                        continue
                if inst.op == 'bwd_fma_placed':
                    args = list(inst.args); args[5] = ''
                    inst = _Instruction(inst.op, tuple(args), getattr(inst, 'comment', ''))
                instructions.append(inst)
            source_schedule = SimpleNamespace(instructions=instructions)
        schedule = _adapt_schedule(source_schedule, self.shared_inputs)
        if 'schedules' in kwargs:
            schedules = kwargs['schedules']
            for kind, needed in (('gw', kwargs['need_grad_w']), ('gx', kwargs['need_grad_x']), ('gy', kwargs['need_grad_y'])):
                if needed and kind not in schedules:
                    raise ValueError(f'Missing requested {kind} schedule')
            kwargs['schedules'] = {k: _adapt_schedule(s, self.shared_inputs) for k, s in schedules.items()}
        emitter = _base().TritonEmitter(schedule, kernel_name=self.kernel_name)
        return _to_gluon(getattr(emitter, method)(**kwargs), num_warps, self.preserve_shared)

    def emit_uniform1d_fwd(self, **kwargs):
        return self._emit('emit_uniform1d_fwd', **kwargs)

    def emit_uniform1d_bwd_fused(self, **kwargs):
        return self._emit('emit_uniform1d_bwd_fused', **kwargs)

    def emit_uniform1d_bwd_split_bundle(self, **kwargs):
        return self._emit('emit_uniform1d_bwd_split_bundle', **kwargs)

    def emit_uniform1d_double_bwd(self, **kwargs):
        return self._emit('emit_uniform1d_double_bwd', **kwargs)

    def emit_stc_fwd(self, **kwargs):
        return self._emit('emit_stc_fwd', **kwargs)

    def emit_stc_bwd(self, **kwargs):
        return self._emit('emit_stc_bwd', **kwargs)

    def emit_stc_double_bwd(self, **kwargs):
        return self._emit('emit_stc_double_bwd', **kwargs)


def _place(schedule, kind, config, *, already_placed=False, exclusive_outputs=False):
    if already_placed:
        if config is not None:
            raise ValueError('placement_config cannot be combined with already_placed=True')
        return schedule
    enabled = config.get('enabled', True) if isinstance(config, dict) else getattr(config, 'enabled', True)
    if not enabled:
        return _base()._triton_place_register_only(schedule, schedule_kind=kind)
    return _base().apply_lars_lifetime_placement(schedule, schedule_kind=kind, config=config, exclusive_outputs=exclusive_outputs)


def _scheduler_options(options):
    result = dict(enable_secondary_affinity=False, topk_candidates=128, profile=False, profile_print=False)
    result.update(options or {})
    return result


def build_gluon_uniform1d_fwd_module_source(
    i_list, j_list, k_list, v_list, coeff_list, input_indices=None, output_indices=None,
    *, u_dim=1, mode='u,u,,u', path_semantics='wxy', kernel_name='uniform1d_gluon_fwd',
    block_u=128, num_warps=4, placement_config=None, schedule_result=None,
    already_placed=False, scheduler_options=None, preserve_shared=False, shared_inputs=(),
):
    b = _base()
    inputs, outputs = dict(input_indices or {}), dict(output_indices or {})
    if schedule_result is None:
        if already_placed:
            raise ValueError('already_placed requires schedule_result')
        paths = b._make_lars_paths_from_uniform1d_lists(b._to_int_list(i_list), b._to_int_list(j_list), b._to_int_list(k_list), b._to_int_list(v_list), b._to_float_list(coeff_list), path_semantics=path_semantics)
        schedule_result = b.LARSUniform1DScheduler(paths=paths, **_scheduler_options(scheduler_options)).schedule()
    placed = _place(schedule_result, 'fwd', placement_config, already_placed=already_placed)
    return GluonEmitter(placed, kernel_name=f'{kernel_name}_u{int(u_dim)}', preserve_shared=preserve_shared, shared_inputs=shared_inputs).emit_uniform1d_fwd(mode=mode, use_x_src=1 in inputs, use_y_src=2 in inputs, use_scatter=0 in outputs, block_u=block_u, num_warps=num_warps)


def build_gluon_uniform1d_bwd_module_source(
    i_list, j_list, k_list, v_list, coeff_list, input_indices=None, output_indices=None,
    *, u_dim=1, iw_dim=None, ix_dim=None, ky_dim=None, v_dim=None,
    mode='u,u,,u', need_grad_w=True, need_grad_x=True, need_grad_y=True,
    path_semantics='wxy', kernel_name='uniform1d_gluon_bwd', block_u=128, num_warps=4,
    placement_config=None, schedules=None, already_placed=False,
    scheduler_options=None, preserve_shared=False, shared_inputs=(),
    backward_mode="split", channel_chunk=None,
):
    b = _base()
    if not any((need_grad_w, need_grad_x, need_grad_y)):
        raise ValueError('At least one gradient must be requested')
    if backward_mode not in ('split', 'fused'):
        raise ValueError('backward_mode must be split or fused')
    if backward_mode == 'fused':
        if not (need_grad_w and need_grad_x and need_grad_y):
            raise ValueError('fused backward requires all three gradients; use split for partial backward')
        if schedules is None:
            if already_placed: raise ValueError('already_placed requires schedules')
            ctx = b._prepare_uniform1d_bwd_codegen_context(i_list=i_list,j_list=j_list,k_list=k_list,v_list=v_list,coeff_list=coeff_list,input_indices=input_indices,output_indices=output_indices,mode=mode,path_semantics=path_semantics)
            schedule = b.LARSUniform1DBwdScheduler(ctx.bwd_paths,need_grad_w=True,**_scheduler_options(scheduler_options)).schedule()
        else:
            schedule = schedules['fused']
        placed = _place(schedule, 'bwd_fused', placement_config, already_placed=already_placed, exclusive_outputs=True)
        return GluonEmitter(placed,kernel_name=f'{kernel_name}_u{int(u_dim)}_fused',preserve_shared=preserve_shared,shared_inputs=shared_inputs).emit_uniform1d_bwd_fused(mode=mode,use_x_src=1 in (input_indices or {}),use_y_src=2 in (input_indices or {}),use_scatter=0 in (output_indices or {}),block_u=channel_chunk or block_u,num_warps=num_warps)
    if schedules is None:
        if already_placed:
            raise ValueError('already_placed requires schedules')
        ctx = b._prepare_uniform1d_bwd_codegen_context(i_list=i_list, j_list=j_list, k_list=k_list, v_list=v_list, coeff_list=coeff_list, input_indices=input_indices, output_indices=output_indices, mode=mode, path_semantics=path_semantics)
        schedules = {kind: b.LARSUniform1DBwdSplitScheduler(ctx.bwd_paths, grad_kind=kind, **_scheduler_options(scheduler_options)).schedule()
                     for kind, needed in (('gw', need_grad_w), ('gx', need_grad_x), ('gy', need_grad_y)) if needed}
    placed = {k: _place(s, f'bwd_split:{k}', placement_config, already_placed=already_placed) for k, s in schedules.items()}
    for kind, needed in (('gw', need_grad_w), ('gx', need_grad_x), ('gy', need_grad_y)):
        if needed and kind not in placed:
            raise ValueError(f'Missing requested {kind} schedule')
    first = next(iter(placed.values()))
    return GluonEmitter(first, kernel_name=f'{kernel_name}_u{int(u_dim)}', preserve_shared=preserve_shared, shared_inputs=shared_inputs).emit_uniform1d_bwd_split_bundle(schedules=placed, mode=mode, need_grad_w=need_grad_w, need_grad_x=need_grad_x, need_grad_y=need_grad_y, use_x_src=1 in (input_indices or {}), use_y_src=2 in (input_indices or {}), use_scatter=0 in (output_indices or {}), block_u=block_u, num_warps=num_warps)


def _build_stc_source(phase, idx_lists, coeff_list, *, num_out_segments, u_dim,
                      path_lens=None, pad_value=None, need_grad_x0=True, kernel_name=None,
                      block_u=128, num_warps=4, placement_config=None, schedule_result=None,
                      already_placed=False, scheduler_options=None, preserve_shared=False, shared_inputs=()):
    b = _base()
    if schedule_result is None:
        if already_placed:
            raise ValueError('already_placed requires schedule_result')
        paths = b.make_stc_paths_from_padded_lists(idx_lists, coeff_list, path_lens=path_lens, pad_value=b.STC_PAD_VALUE if pad_value is None else pad_value)
        schedule_result = b.LARSUniform1DScheduler(paths, path_kind='stc', **_scheduler_options(scheduler_options)).schedule()
        if phase == 'bwd':
            schedule_result = b._make_stc_bwd_logical_schedule(paths, schedule_result, need_grad_x0=need_grad_x0)
    placed = _place(schedule_result, f'stc_{phase}', placement_config, already_placed=already_placed)
    emitter = GluonEmitter(placed, kernel_name=f'{kernel_name or "stc_gluon_" + phase}_u{int(u_dim)}', preserve_shared=preserve_shared, shared_inputs=shared_inputs)
    kwargs = dict(block_u=block_u, num_warps=num_warps)
    if phase == 'bwd': kwargs['need_grad_x0'] = need_grad_x0
    return getattr(emitter, f'emit_stc_{phase}')(**kwargs)


def build_gluon_stc_fwd_module_source(idx_lists, coeff_list, **kwargs):
    return _build_stc_source('fwd', idx_lists, coeff_list, **kwargs)


def build_gluon_stc_bwd_module_source(idx_lists, coeff_list, **kwargs):
    return _build_stc_source('bwd', idx_lists, coeff_list, **kwargs)


def build_gluon_uniform1d_double_bwd_module_source(
    i_list, j_list, k_list, v_list, coeff_list, input_indices=None, output_indices=None,
    *, u_dim=1, iw_dim=None, ix_dim=None, ky_dim=None, v_dim=None,
    mode='u,u,,u', need_grad_w=True, need_grad_x=True, need_grad_y=True,
    path_semantics='wxy', kernel_name='uniform1d_gluon_double_bwd', block_u=128,
    num_warps=4, schedule_result=None, scheduler_options=None,
    preserve_shared=False, shared_inputs=(),
):
    """Reuse double-backward math; shared_inputs explicitly selects state.

    The supplied lifetime placement pass does not support double-backward
    records, so its automatic classification is intentionally not applied here.
    """
    b = _base()
    if not any((need_grad_w, need_grad_x, need_grad_y)):
        raise ValueError('At least one gradient must be requested')
    if schedule_result is None:
        ctx = b._prepare_uniform1d_bwd_codegen_context(i_list=i_list, j_list=j_list, k_list=k_list, v_list=v_list, coeff_list=coeff_list, input_indices=input_indices, output_indices=output_indices, mode=mode, path_semantics=path_semantics)
        schedule_result = b.LARSUniform1DDoubleBwdScheduler(ctx.bwd_paths, need_grad_w=need_grad_w, need_grad_x=need_grad_x, need_grad_y=need_grad_y, **_scheduler_options(scheduler_options)).schedule()
    return GluonEmitter(schedule_result, kernel_name=f'{kernel_name}_u{int(u_dim)}', preserve_shared=preserve_shared, shared_inputs=shared_inputs).emit_uniform1d_double_bwd(mode=mode, need_grad_w=need_grad_w, need_grad_x=need_grad_x, need_grad_y=need_grad_y, use_x_src=1 in (input_indices or {}), use_y_src=2 in (input_indices or {}), use_scatter=0 in (output_indices or {}), block_u=block_u, num_warps=num_warps)


def build_gluon_stc_double_bwd_module_source(
    idx_lists, coeff_list, *, path_lens=None, pad_value=None, num_out_segments,
    u_dim, kernel_name='stc_gluon_double_bwd', block_u=128, num_warps=4,
    schedule_result=None, scheduler_options=None, preserve_shared=False, shared_inputs=(),
):
    b = _base()
    if schedule_result is None:
        paths = b.make_stc_paths_from_padded_lists(idx_lists, coeff_list, path_lens=path_lens, pad_value=b.STC_PAD_VALUE if pad_value is None else pad_value)
        schedule_result = b.LARSSTCDoubleBwdScheduler(paths, **_scheduler_options(scheduler_options)).schedule()
    return GluonEmitter(schedule_result, kernel_name=f'{kernel_name}_u{int(u_dim)}', preserve_shared=preserve_shared, shared_inputs=shared_inputs).emit_stc_double_bwd(block_u=block_u, num_warps=num_warps)


def load_gluon_module_from_source(source):
    # Reuse the existing atomic, file-backed import/cache implementation.
    return _base().load_triton_module_from_source(source, module_prefix="fasteq_gluon")


def _load(builder, *args, **kwargs):
    return load_gluon_module_from_source(builder(*args, **kwargs))


def build_gluon_uniform1d_fwd_module(*args, **kwargs):
    return _load(build_gluon_uniform1d_fwd_module_source, *args, **kwargs)


def build_gluon_uniform1d_bwd_module(*args, **kwargs):
    return _load(build_gluon_uniform1d_bwd_module_source, *args, **kwargs)


def build_gluon_uniform1d_double_bwd_module(*args, **kwargs):
    return _load(build_gluon_uniform1d_double_bwd_module_source, *args, **kwargs)


def build_gluon_stc_fwd_module(*args, **kwargs):
    return _load(build_gluon_stc_fwd_module_source, *args, **kwargs)


def build_gluon_stc_bwd_module(*args, **kwargs):
    return _load(build_gluon_stc_bwd_module_source, *args, **kwargs)


def build_gluon_stc_double_bwd_module(*args, **kwargs):
    return _load(build_gluon_stc_double_bwd_module_source, *args, **kwargs)
