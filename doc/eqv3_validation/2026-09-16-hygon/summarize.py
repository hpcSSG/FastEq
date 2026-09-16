"""Aggregate existing measurements only; never launches GPU work."""
import collections
import csv
import gzip
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[3]
ORDER=('softmax','alpha','gate','norm','separable','dropout')
LABELS=dict(softmax='GraphSoftmax',alpha='AttentionAlpha',gate='e3nn Gate',
           norm='LayerNorm',separable='SeparableLayerNorm',dropout='EquivariantDropout')


def read(path): return json.loads(path.read_text())
def write(path,data): path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def csv_write(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as out:
        writer=csv.DictWriter(out,fieldnames=keys,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def main():
    points=[read(p) for p in sorted((ROOT/'points').glob('*.json'))]
    checks=[read(p) for p in sorted((ROOT/'checks').glob('*.json'))]
    if not points and (ROOT/'raw_results.json.gz').exists():
        with gzip.open(ROOT/'raw_results.json.gz', 'rt') as stream:
            raw = json.load(stream)
        points = list(raw['points'].values())
        checks = list(raw['checks'].values())
    points = [r for r in points if r['operator'] in ORDER]
    checks = [r for r in checks if r['operator'] in ORDER]
    suites = ET.parse(ROOT/'layernorm.xml').getroot().findall('testsuite')
    suite = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
             for key in ('tests', 'failures', 'errors', 'skipped')}
    suite['passed'] = suite['tests'] - suite['failures'] - suite['errors'] - suite['skipped']
    extras=read(ROOT/'extra_checks.json')
    failed_ops={r['operator'] for r in checks if r['status']!='PASS'}
    if any(r['name']=='graph_softmax_builtin' and r['status']!='PASS' for r in extras):
        failed_ops.add('softmax')
    native_path=ROOT/'softmax_native_checks.json'
    native=read(native_path) if native_path.exists() else {}
    if native.get('status') in ('FAIL','ERROR'):
        failed_ops.add('softmax')
    runtime_error_ops={r['operator'] for r in points if r['status'] in ('ERROR','PROCESS_ERROR')}
    failures=[]
    for r in checks:
        for name,value in r.get('checks',{}).items():
            if value['failures']:
                failures.append(dict(operator=r['operator'],atoms=r['atoms'],tensor=name,
                    failing_elements=value['failures'],max_abs=value['max_abs'],
                    max_tolerance_ratio=value['max_tolerance_ratio'],examples=value['examples']))
    csv_write(ROOT/'failures.csv',failures)
    lookup={(r['operator'],r['atoms'],r['mode'],r['backend']):r for r in points}
    checkmap={(r['operator'],r['atoms'],r['mode']):r for r in checks}
    pairs=[]
    for op in ORDER:
        for n in sorted({r['atoms'] for r in points if r['operator']==op}):
            check=checkmap.get((op,n,'fwd_bwd'))
            for mode in ('fwd','fwd_bwd'):
                t=lookup.get((op,n,mode,'torch'),{})
                f=lookup.get((op,n,mode,'fused'),{})
                if t.get('status')!='OK' or f.get('status')!='OK': continue
                c=check or checkmap.get((op,n,mode),{})
                metrics=c.get('checks',{})
                needed=list(metrics.values()) if mode=='fwd_bwd' else [metrics.get('output',{})]
                valid=bool(metrics) and all(x and x['failures']==0 for x in needed)
                row=dict(operator=op,atoms=n,edges=t.get('edges',''),mode=mode,
                    correctness='PASS' if valid else 'DIAGNOSTIC',
                    operator_accuracy_status='FAILED_OTHER_CONFIG' if op in failed_ops else 'PASSED_TESTED_CONFIGS',
                    operator_runtime_status='ERROR_AT_ANOTHER_SIZE' if op in runtime_error_ops else 'NO_UNEXPECTED_ERRORS_RECORDED',
                    check_file=str(Path('checks')/f"{op}_{n}_{c.get('mode',mode)}_fused_NKC.json"),
                    input_samples_match=t.get('input_samples')==f.get('input_samples'),
                    torch_gpu_ms=t['gpu_ms_median'],fused_gpu_ms=f['gpu_ms_median'],
                    gpu_speedup=t['gpu_ms_median']/f['gpu_ms_median'],
                    torch_wall_ms=t['wall_ms_median'],fused_wall_ms=f['wall_ms_median'],
                    wall_speedup=t['wall_ms_median']/f['wall_ms_median'],
                    torch_peak_gib=t['peak_allocated_bytes']/2**30,
                    fused_peak_gib=f['peak_allocated_bytes']/2**30,
                    peak_reduction_percent=100*(1-f['peak_allocated_bytes']/t['peak_allocated_bytes']),
                    scope='cached CSR' if op=='softmax' else 'public operator invocation')
                cold=f.get('including_csr_preprocessing')
                if cold:
                    row.update(fused_with_csr_gpu_ms=cold['gpu_ms_median'],
                        fused_with_csr_wall_ms=cold['wall_ms_median'],
                        fused_with_csr_peak_gib=cold['peak_allocated_bytes']/2**30,
                        with_csr_wall_speedup=t['wall_ms_median']/cold['wall_ms_median'])
                pairs.append(row)
    csv_write(ROOT/'paired.csv',pairs)
    flat=[]
    for r in points:
        fields=('operator','atoms','edges','mode','backend','status','stage','error',
                'gpu_ms_median','wall_ms_median','peak_allocated_bytes','baseline_allocated_bytes',
                'incremental_peak_bytes','peak_reserved_bytes')
        flat.append({k:r[k] for k in fields if k in r})
    csv_write(ROOT/'scaling.csv',flat)
    bounds=[]
    for op in ORDER:
        for mode in ('fwd','fwd_bwd'):
            for backend in ('torch','fused'):
                rows=sorted((r for r in points if (r['operator'],r['mode'],r['backend'])==(op,mode,backend)),key=lambda x:x['atoms'])
                if not rows:continue
                ok=[r for r in rows if r['status']=='OK']
                stop=next((r for r in rows if r['status']!='OK'),{})
                bounds.append(dict(operator=op,mode=mode,backend=backend,last_success_n=max((r['atoms'] for r in ok),default=0),
                    next_n=stop.get('atoms',''),stop=stop.get('status','RUNNING'),stage=stop.get('stage',''),reason=stop.get('error','')))
    csv_write(ROOT/'boundaries.csv',bounds)
    correctness=[]
    for op in ORDER:
        rows=[r for r in checks if r['operator']==op]
        components=collections.defaultdict(list)
        for r in rows:
            for name,value in r.get('checks',{}).items():
                components[name].append(dict(atoms=r['atoms'],mode=r['mode'],**value))
        worst={name:dict(max_abs=max(x['max_abs'] for x in values),
                        max_tolerance_ratio=max(x['max_tolerance_ratio'] for x in values),
                        failing_elements=sum(x['failures'] for x in values),
                        worst_ratio_n=max(values,key=lambda x:x['max_tolerance_ratio'])['atoms']) for name,values in components.items()}
        correctness.append(dict(operator=op,checks=len(rows),statuses=dict(collections.Counter(r['status'] for r in rows)),
            atom_counts=sorted({r['atoms'] for r in rows}),components=worst,
            largest_passing_forward_n=max((r['atoms'] for r in rows if r.get('checks',{}).get('output',{}).get('failures',1)==0),default=0),
            largest_passing_fwd_bwd_n=max((r['atoms'] for r in rows if r['mode']=='fwd_bwd' and r['status']=='PASS'),default=0)))
    write(ROOT/'correctness_summary.json',dict(operators=correctness,
        native_graph_softmax=dict(status=native.get('status','NOT_RUN'),
            assertions=native.get('assertions',0),failed_assertions=len(native.get('failed_assertions',[]))),
        extras=[{k:r[k] for k in ('name','status','error') if k in r} for r in extras]))
    representative=[r for r in pairs if r['atoms']==4096]
    largest=[]
    for op in ORDER:
        for mode in ('fwd','fwd_bwd'):
            rows=[r for r in pairs if r['operator']==op and r['mode']==mode]
            if rows:largest.append(max(rows,key=lambda r:r['atoms']))
    csv_write(ROOT/'representative.csv',representative)
    csv_write(ROOT/'largest_common.csv',largest)
    native_path=ROOT/'softmax_native_checks.json'
    native=read(native_path) if native_path.exists() else {}
    if native.get('status') in ('FAIL','ERROR'):
        failed_ops.add('softmax')
    manifest=read(ROOT/'manifest.json')
    complete=(ROOT/'scaling_complete.json').exists()
    summary=dict(complete=complete,commit=manifest['commit'],
        point_statuses=dict(collections.Counter(r['status'] for r in points)),
        check_statuses=dict(collections.Counter(r['status'] for r in checks)),
        paired_measurements=len(pairs),diagnostic_pairs=sum(r['correctness']!='PASS' for r in pairs),
        all_input_samples_match=all(r['input_samples_match'] for r in pairs),
        accuracy_failed_operators=sorted(failed_ops),runtime_error_operators=sorted(runtime_error_ops),
        failures=failures,layernorm_tests=suite,
        representative=representative,largest_common=largest,boundaries=bounds,
        softmax_native_assertions=native.get('assertions',0),
        softmax_native_failed_assertions=len(native.get('failed_assertions',[])))
    write(ROOT/'summary.json',summary)
    # Preserve the published archive when summarizing without unpacked records.
    if list((ROOT/'points').glob('*.json')):
        raw={'points':{p.name:read(p) for p in sorted((ROOT/'points').glob('*.json'))},
             'checks':{p.name:read(p) for p in sorted((ROOT/'checks').glob('*.json'))}}
        with gzip.open(ROOT/'raw_results.json.gz','wt') as stream:
            json.dump(raw,stream)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('representative','largest_common','boundaries')},indent=2))


if __name__=='__main__':main()
