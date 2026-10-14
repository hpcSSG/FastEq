"""Large-workload candidate selection shared by Uniform1D and STC frontends."""
from __future__ import annotations
import hashlib
import json
import math
import os
import statistics

DEFAULT_SIZES = (256, 1024, 4096, 16384, 65536, 262144, 1048576)

def tuning_sizes(actual_size):
    actual_size = int(actual_size)
    if actual_size <= 0:
        raise ValueError('actual tuning batch size must be positive')
    raw = os.environ.get('FASTEQ_JIT_TUNE_SIZES', '').strip()
    sizes = [int(v.strip()) for v in raw.split(',')] if raw else list(DEFAULT_SIZES)
    if not raw and actual_size > max(DEFAULT_SIZES):
        sizes.append(1 << (2 * actual_size - 1).bit_length())
    if not sizes or any(v <= 0 for v in sizes):
        raise ValueError('FASTEQ_JIT_TUNE_SIZES must contain positive comma-separated integers')
    return sorted(set(sizes + [actual_size]))

def policy_signature():
    config = {'version': 1, 'selection': 'largest_successful_size',
              'sizes': os.environ.get('FASTEQ_JIT_TUNE_SIZES', 'default'),
              'samples': os.environ.get('FASTEQ_JIT_TUNE_SAMPLES', '3'),
              'memory_fraction': os.environ.get('FASTEQ_JIT_TUNE_MEMORY_FRACTION', '0.75'),
              'forward_memory': os.environ.get('FASTEQ_DENSE_FWD_WRITEBACK', '1'),
              'backward_memory': os.environ.get('FASTEQ_DENSE_BWD_SKIP_ZERO', '1')}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]

def is_oom(exc):
    return isinstance(exc, MemoryError) or 'out of memory' in str(exc).lower()

def benchmark_sizes(candidates, *, sizes, actual_size, prepare_case, measure_case, cleanup=None):
    """Measure all candidates before pruning; require success on actual input size.

    Largest successful size is primary. Smaller-size latency breaks exact ties.
    OOM sizes may be skipped. Fatal device-context errors are propagated.
    """
    rows, measured = [], {}
    sizes = sorted(set(map(int, sizes)) | {int(actual_size)})
    for size in sizes:
        case = None
        try:
            case = prepare_case(size)
            for tag, mod in candidates:
                try:
                    ms = float(measure_case(mod, case))
                    if not math.isfinite(ms) or ms < 0:
                        raise RuntimeError('invalid benchmark latency')
                    measured[(tag, size)] = ms
                    rows.append({'size': size, 'tag': tag, 'ms': ms, 'status': 'ok'})
                    print(f'[JIT][multi-size] size={size} candidate={tag} median_ms={ms:.6f}')
                except Exception as exc:
                    message = str(exc)
                    if any(v in message.lower() for v in ('illegal memory access', 'device-side assert', 'misaligned address', 'unspecified launch failure')):
                        raise
                    rows.append({'size': size, 'tag': tag, 'status': 'oom' if is_oom(exc) else 'failed', 'error': message})
                    print(f'[JIT][multi-size] size={size} candidate={tag} skipped: {message}')
                    if is_oom(exc) and cleanup is not None: cleanup()
        except Exception as exc:
            if not is_oom(exc): raise
            rows.append({'size': size, 'status': 'input_oom', 'error': str(exc)})
            print(f'[JIT][multi-size] size={size} skipped: input allocation budget/OOM')
        finally:
            del case
            if cleanup is not None: cleanup()
    valid = [(tag, mod) for tag, mod in candidates if (tag, int(actual_size)) in measured]
    if not valid:
        raise RuntimeError('all tuning candidates failed at the actual runtime input size')
    successful_sizes = [size for size in sizes if any((tag, size) in measured for tag, _ in valid)]
    selected_size = max(successful_sizes)
    def rank(candidate):
        tag, _ = candidate
        return (measured.get((tag, selected_size), float('inf')),
                statistics.mean(measured[(tag, s)] / s for s in sizes if (tag, s) in measured))
    tag, mod = min(valid, key=rank)
    return (tag, mod, measured[(tag, selected_size)]), {
        'policy': policy_signature(), 'sizes_requested': sizes,
        'selected_size': selected_size, 'selected_tag': tag, 'results': rows}

def make_case_factory(data, *, batch_size, per_row, index_groups=(), broadcast_rows=()):
    """Replicate independent graph batches, offsetting source/destination indices.

    Connected source/destination domains use one padded replication stride.
    Uniform1D channel/segment dimensions and per-node graph structure stay fixed.
    """
    batch_size = int(batch_size)
    def repeat_rows(tensor, rows):
        tensor = tensor.detach()
        count = int(tensor.size(0))
        if count <= 0: raise ValueError('cannot expand an empty tuning tensor')
        if rows == count: return tensor.contiguous()
        repeats = (math.ceil(rows / count),) + (1,) * (tensor.dim() - 1)
        return tensor.repeat(repeats)[:rows].contiguous()
    def prepare(size):
        import torch
        size = int(size)
        if size == batch_size:
            return {key: value.detach().contiguous() if value is not None else None for key, value in data.items()}
        fraction = float(os.environ.get('FASTEQ_JIT_TUNE_MEMORY_FRACTION', '0.75'))
        if not 0 < fraction <= 1: raise ValueError('memory fraction must be in (0, 1]')
        first = next(v for v in data.values() if v is not None)
        input_bytes = sum(v.numel() * v.element_size() for v in data.values() if v is not None)
        # Reserve space for copied inputs and outputs, not only the input tensors.
        estimate = input_bytes * max(1, math.ceil(size / batch_size)) * 2
        free, _ = torch.cuda.mem_get_info(first.device)
        if estimate > int(free * fraction):
            raise MemoryError(f'tuning allocation budget exceeded at size={size}')
        result, grouped = dict(data), set()
        blocks = math.ceil(size / batch_size)
        groups = [(name, [k for k in targets if data.get(k) is not None])
                  for name, targets in index_groups if data.get(name) is not None]
        components = []
        for _, targets in groups:
            if not targets: raise ValueError('an index vector needs at least one referenced tensor')
            merged = set(targets)
            remaining = []
            for component in components:
                if component & merged: merged |= component
                else: remaining.append(component)
            components = remaining + [merged]
        strides = {key: max(int(data[k].size(0)) for k in component)
                   for component in components for key in component}
        for index_name, targets in groups:
            index = data[index_name]
            stride = strides[targets[0]]
            if index.dtype == torch.int32 and stride * blocks > 2**31 - 1:
                raise ValueError('resized tuning indices exceed int32 capacity')
            for key in targets:
                padded = repeat_rows(data[key], stride)
                result[key] = repeat_rows(padded, stride * blocks)
                grouped.add(key)
            resized = repeat_rows(index, size).to(torch.int64)
            offsets = (torch.arange(size, device=index.device, dtype=torch.int64) // batch_size) * stride
            result[index_name] = (resized + offsets).to(index.dtype).contiguous()
            grouped.add(index_name)
        for key in per_row:
            tensor = data.get(key)
            if tensor is not None and key not in grouped:
                if key in broadcast_rows and int(tensor.size(0)) == 1:
                    result[key] = tensor.detach().contiguous()
                else:
                    if int(tensor.size(0)) != batch_size:
                        raise ValueError(f'{key} must have {batch_size} rows for dense expansion')
                    result[key] = repeat_rows(tensor, size)
        return result
    return prepare
