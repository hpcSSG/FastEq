"""Fresh-process standalone operator scaling; the controller has no GPU context."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(os.environ.get('FASTEQ_VALIDATION_OUTPUT_DIR', 'validation_output')).resolve()
REPO=Path(__file__).resolve().parents[2]
OPS=('gate','norm','separable','dropout')


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def boundary(op,n,backend):
    if backend=='torch': return None
    if op in ('norm','separable') and n*16*128>2**31-1:
        return 'LIMIT','LayerNorm explicit 32-bit element indexing bound'
    if op=='gate' and n*(16+3)*128>2**31-1:
        return 'INDEX_GUARD','Harness stops before signed 32-bit row*stride overflow; source has no runtime guard'
    if op=='alpha' and n*32*8*32>2**32:
        return 'FALLBACK','Public adapter falls back to Torch above 2**32 input elements'
    if op=='dropout' and n*4*128>2**32:
        return 'LIMIT','Explicit uint32 RNG index bound'
    return None


def worker(args):
    import gc
    import torch
    import triton
    from cases import Case,check,sample
    torch.set_num_threads(8)
    row=dict(operator=args.op,atoms=args.n,mode=args.mode,backend=args.backend,
             task=args.task,stage='initialization',warmup=args.warmup,samples=args.samples)
    try:
        props=torch.cuda.get_device_properties(0)
        row['environment']=dict(torch=torch.__version__,triton=triton.__version__,
            cuda=torch.version.cuda,hip=torch.version.hip,gpu=props.name,uuid=str(props.uuid),
            total_memory=props.total_memory,free_before=torch.cuda.mem_get_info()[0],
            visible_device=os.environ['CUDA_VISIBLE_DEVICES'])
        limit=boundary(args.op,args.n,args.backend)
        if limit:
            row.update(status=limit[0],error=limit[1])
            write(args.output,row);print(json.dumps({k:row[k] for k in ('operator','atoms','mode','backend','status')}),flush=True)
            return
        row['stage']='setup'
        case=Case(args.op,args.n,args.mode,args.backend if args.task=='timing' else 'both',layout=args.layout)
        row.update(case.meta)
        row['setup']=case.setup
        row['input_samples']={name:sample(t) for name,t in zip(case.names,case.inputs)}
        if args.task=='check':
            row['stage']='full_tensor_comparison'
            row.update(check(case))
        else:
            def invoke(cold=False): return case.invoke(args.backend,cold=cold)

            def measure(cold=False):
                for _ in range(args.warmup):
                    result=invoke(cold);del result
                gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
                baseline=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                result=invoke(cold);torch.cuda.synchronize()
                peak=torch.cuda.max_memory_allocated()
                reserved=torch.cuda.max_memory_reserved()
                values={name:sample(t) for name,t in zip(('output',*case.names),result)}
                del result
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();end.record();end.synchronize()
                gpu,wall=[],[]
                for _ in range(args.samples):
                    torch.cuda.synchronize()
                    t0=time.perf_counter_ns();start.record()
                    result=invoke(cold)
                    end.record();end.synchronize()
                    wall.append((time.perf_counter_ns()-t0)/1e6)
                    gpu.append(start.elapsed_time(end))
                    del result
                return dict(gpu_ms=gpu,wall_ms=wall,gpu_ms_median=statistics.median(gpu),
                    wall_ms_median=statistics.median(wall),peak_allocated_bytes=peak,
                    peak_reserved_bytes=reserved,baseline_allocated_bytes=baseline,
                    incremental_peak_bytes=peak-baseline,output_values=values)

            row['stage']='warmup_memory_timing'
            row.update(measure())
            row['status']='OK'
            if args.op=='softmax' and args.backend=='fused' and args.n<=65536:
                row['stage']='uncached_graph_timing'
                case.csr=None
                row['including_csr_preprocessing']=measure(cold=True)
        row['stage']='complete'
    except Exception as exc:
        if isinstance(exc,torch.OutOfMemoryError) or 'out of memory' in str(exc).lower():
            status='OOM'
        elif isinstance(exc,(NotImplementedError,ValueError)) and any(x in str(exc).lower() for x in ('index','uint32','2**32')):
            status='LIMIT'
        else:
            status='ERROR'
        row.update(status=status,error=str(exc),traceback=traceback.format_exc())
    write(args.output,row)
    keys=('operator','atoms','mode','backend','task','status','gpu_ms_median','peak_allocated_bytes')
    print(json.dumps({k:row[k] for k in keys if k in row}),flush=True)


def launch(args,op,n,mode,backend,task='timing',layout='NKC'):
    path=ROOT/('points' if task=='timing' else 'checks')/f'{op}_{n}_{mode}_{backend}_{layout}.json'
    if path.exists():
        return json.loads(path.read_text())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),FASTEQ_BACKEND='cpu',
        PYTHONPATH=str(REPO)+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='8',PYTHONWARNINGS='ignore',
        TRITON_CACHE_DIR=os.environ['TRITON_CACHE_DIR'])
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--op',op,'--n',str(n),
         '--mode',mode,'--backend',backend,'--task',task,'--layout',layout,
         '--output',str(path),'--warmup',str(args.warmup),'--samples',str(args.samples)]
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.log').open('w') as stream:
        proc=subprocess.run(cmd,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if not path.exists():
        write(path,dict(operator=op,atoms=n,mode=mode,backend=backend,task=task,
                        status='PROCESS_ERROR',returncode=proc.returncode))
    result=json.loads(path.read_text())
    print(json.dumps({k:result[k] for k in ('operator','atoms','mode','backend','task','status','gpu_ms_median') if k in result}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--task',choices=['timing','check'],default='timing')
    p.add_argument('--op',choices=OPS)
    p.add_argument('--n',type=int)
    p.add_argument('--mode',choices=['fwd','fwd_bwd'])
    p.add_argument('--backend',choices=['torch','fused'],default='fused')
    p.add_argument('--layout',choices=['NKC','KNC'],default='NKC')
    p.add_argument('--output',type=Path)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--samples',type=int,default=20)
    p.add_argument('--ops',nargs='+',choices=OPS,default=list(OPS))
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--start-n',type=int,default=256)
    args=p.parse_args()
    if args.worker:
        worker(args);return
    if args.smoke:
        for op in args.ops:
            for n in (1,17,257,4096):
                launch(args,op,n,'fwd_bwd','fused','check')
        return
    for op in args.ops:
        active={(mode,backend) for mode in ('fwd','fwd_bwd') for backend in ('torch','fused')}
        n=args.start_n
        while active:
            rows={}
            backends=('torch','fused') if n.bit_length()%2 else ('fused','torch')
            for mode in ('fwd','fwd_bwd'):
                for backend in backends:
                    if (mode,backend) not in active:continue
                    row=launch(args,op,n,mode,backend)
                    rows[mode,backend]=row
                    if row['status']!='OK':active.remove((mode,backend))
            # Independent process compares every output/gradient element; it retains
            # one implementation's results on CPU, preserving each GPU memory ceiling.
            for mode in ('fwd_bwd','fwd'):
                if all(rows.get((mode,b),{}).get('status')=='OK' for b in ('torch','fused')):
                    launch(args,op,n,mode,'fused','check')
                    break
            n*=2
        write(ROOT/'boundaries'/f'{op}.json',dict(operator=op,finished=True))
    write(ROOT/'scaling_complete.json',dict(complete=True,operators=args.ops))


if __name__=='__main__':main()
