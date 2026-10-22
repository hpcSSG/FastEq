"""Candidate spaces and persistent best configurations for Python GPU backends.

No CUDA extension loading. Builders return modules exposing the existing run ABI.
The caller supplies its benchmark policy, so native and Python backends can share
multi-size selection without sharing code artifacts or best-candidate records.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

_CACHE = {}
_VERSION = 'python-uniform1d-autotune-v2-operator-cache'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _powers(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{name} must be a nonempty list')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 or v&(v-1) for v in values):
        raise ValueError(f'{name} values must be positive powers of two')
    return list(dict.fromkeys(values))


def candidate_configs(backend, phase, options, *, all_gradients=True, enabled=True, space=None):
    if backend not in ('triton','gluon') or phase not in ('fwd','bwd','double_bwd'):
        raise ValueError('unsupported Python backend or phase')
    options=dict(options)
    if not enabled:
        result=dict(block_u=128,num_warps=4,**{k:v for k,v in options.items() if k not in ('block_u','num_warps')})
        result.update({k:options[k] for k in ('block_u','num_warps') if k in options})
        if phase=='bwd':result.setdefault('backward_mode','split')
        if phase=='bwd' and not all_gradients and result['backward_mode']=='fused':
            raise ValueError('partial backward requires split')
        return [result]
    if space is None:
        try: space=json.loads(os.environ.get('FASTEQ_PYTHON_TUNE_OPTIONS','{}') or '{}')
        except ValueError as exc: raise ValueError('FASTEQ_PYTHON_TUNE_OPTIONS must be a JSON object') from exc
    if not isinstance(space,dict):raise ValueError('FASTEQ_PYTHON_TUNE_OPTIONS must be a JSON object')
    unknown=set(space)-{'num_warps','block_u','channel_chunks','backward_modes','placements'}
    if unknown:raise ValueError(f'Unknown Python tuning keys: {sorted(unknown)}')
    warps=_powers(space.get('num_warps',[1,2,4]),'num_warps')
    if any(w>32 for w in warps):raise ValueError('num_warps cannot exceed 32')
    blocks=_powers(space.get('block_u',[128]),'block_u')
    chunks=_powers(space.get('channel_chunks',[32,128]),'channel_chunks')
    modes=space.get('backward_modes',['split','fused'])
    if not isinstance(modes,list) or not modes or any(m not in ('split','fused') for m in modes):
        raise ValueError('backward_modes must contain split and/or fused')
    placements=space.get('placements',[{'enabled':False},{}])
    if not isinstance(placements,list) or not placements or any(not isinstance(p,dict) for p in placements):
        raise ValueError('placements must be a nonempty list of objects')
    warps=[options['num_warps']] if 'num_warps' in options else warps
    blocks=[options['block_u']] if 'block_u' in options else blocks
    if phase=='bwd':
        modes=[options['backward_mode']] if 'backward_mode' in options else modes
        if not all_gradients:
            if options.get('backward_mode')=='fused':raise ValueError('partial backward requires split')
            modes=['split']
    else:modes=[None]
    # Triton has register-only placement. Double backward does not consume the
    # first-order placement pass; manual shared_inputs still work in Gluon.
    if backend=='triton' or phase=='double_bwd':placements=[None]
    elif 'placement_config' in options:placements=[options['placement_config']]
    result={}
    for mode,warp,block,placement in itertools.product(modes,warps,blocks,placements):
        chunk_values=([options['channel_chunk']] if 'channel_chunk' in options else chunks) if mode=='fused' else [None]
        for chunk in chunk_values:
            config=dict(options,block_u=block,num_warps=warp)
            if mode is not None:config['backward_mode']=mode
            if chunk is not None:config['channel_chunk']=chunk
            else:config.pop('channel_chunk',None)
            if placement is not None:config['placement_config']=placement
            result[_json(config)]=config
    if not result:raise ValueError('Python tuning generated no candidates')
    return list(result.values())


def _codegen_fingerprint():
    h=hashlib.sha256(_VERSION.encode())
    for name in ('python_tuning.py','triton_backend.py','gluon_backend.py','schedulers.py','placement.py','models.py','metadata.py','uniform1d_backward.py','cuda_common.py','multi_size_tuning.py','../uniform1d_jit.py'):
        p=Path(__file__).parent/name
        if p.is_file():h.update(p.read_bytes())
    return h.hexdigest()


def _tag(config):
    mode=config.get('backward_mode','kernel')
    return f'{mode}_w{config.get("num_warps",4)}_u{config.get("channel_chunk",config.get("block_u",128))}_'+hashlib.sha256(_json(config).encode()).hexdigest()[:8]


def select_best(*, backend, phase, signature, configs, builder, benchmark,
                enabled=True, memory_cache=None, cache_dir=None):
    cache=_CACHE if memory_cache is None else memory_cache
    payload=dict(version=_VERSION,backend=backend,phase=phase,signature=signature,
                 configs=configs,codegen=_codegen_fingerprint(),enabled=enabled)
    key=hashlib.sha256(_json(payload).encode()).hexdigest()
    if key in cache:return cache[key]
    if not enabled:
        mod=builder(configs[0]);cache[key]=mod;return mod
    root=Path(cache_dir or os.environ.get('FASTEQ_PYTHON_TUNE_CACHE',str(Path.home()/'.cache/fasteq/python_tuning')))
    record=root/f'{backend}_{phase}_{key}.json'
    try:
        saved=json.loads(record.read_text())
        config=saved['config']
        if saved.get('key')==key and config in configs:
            mod=builder(config);cache[key]=mod
            print(f'[JIT][{backend.upper()}][{phase.upper()}][tune] cache selected={saved["tag"]}')
            return mod
    except (OSError,ValueError,KeyError,ImportError,RuntimeError):pass
    candidates=[];by_tag={};errors=[]
    for config in configs:
        tag=_tag(config)
        try:mod=builder(config)
        except Exception as exc:
            errors.append(f'{tag}: {exc}');print(f'[JIT][{backend.upper()}][tune] skip {tag}: {exc}');continue
        candidates.append((tag,mod));by_tag[tag]=config
    if not candidates:raise RuntimeError('All Python candidates failed to build: '+'; '.join(errors))
    best,report=benchmark(candidates)
    tag,mod,ms=best
    if tag not in by_tag:raise RuntimeError('Benchmark selected an unknown candidate')
    cache[key]=mod
    saved=dict(key=key,backend=backend,phase=phase,tag=tag,config=by_tag[tag],milliseconds=ms,report=report)
    tmp=None
    try:
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile('w',dir=root,suffix='.tmp',delete=False) as f:
            tmp=Path(f.name);json.dump(saved,f,sort_keys=True,allow_nan=False)
        os.replace(tmp,record)
    except (OSError,ValueError) as exc:
        print(f'[JIT][{backend.upper()}][tune] could not persist best config: {exc}')
    finally:
        if tmp is not None:tmp.unlink(missing_ok=True)
    print(f'[JIT][{backend.upper()}][{phase.upper()}][tune] selected={tag} median={ms:.6f} ms')
    return mod
