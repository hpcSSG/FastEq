"""Verify completed records, summaries and links before publication."""
from pathlib import Path
import json,csv,gzip,hashlib,statistics,math,re,subprocess
repo=Path(__file__).resolve().parents[3]
r=repo/'doc/eqv3_validation/2026-09-16-hygon'
s=json.loads((r/'summary.json').read_text());assert s['complete']
m=json.loads((r/'manifest.json').read_text())
with gzip.open(r/'raw_results.json.gz','rt') as f:raw=json.load(f)
points=list(raw['points'].values());checks=list(raw['checks'].values())
lookup={(x['operator'],x['atoms'],x['mode'],x['backend']):x for x in points}
checkmap={(x['operator'],x['atoms'],x['mode']):x for x in checks}
for p,h in m['runtime_sha256'].items():assert hashlib.sha256((repo/p).read_bytes()).hexdigest()==h,p
for p,h in m['test_module_sha256'].items():assert hashlib.sha256((repo/p).read_bytes()).hexdigest()==h,p
for x in points:
 assert x['environment']['hip']=='6.3.26045'
 assert x['environment']['torch']=='2.7.1'
 if x['status']=='OK':
  assert len(x['gpu_ms'])==20 and len(x['wall_ms'])==20 and x['warmup']==5
  assert x['gpu_ms_median']==statistics.median(x['gpu_ms'])
  assert x['wall_ms_median']==statistics.median(x['wall_ms'])
for x in checks:
 if x['status'] in ('PASS','FAIL'):
  assert x['full_tensor_comparison']
  assert (x['status']=='PASS')==all(y['failures']==0 for y in x['checks'].values())
  for name,y in x['checks'].items():
   expected=(0.,0.) if x['operator']=='dropout' else ((3e-6,3e-5) if name=='output' else (3e-5,3e-4)) if x['operator']=='softmax' else (5e-5,5e-4)
   assert (y['atol'],y['rtol'])==expected
bounds=list(csv.DictReader((r/'boundaries.csv').open()));assert len(bounds)==24
for b in bounds:
 seq=sorted((x for x in points if (x['operator'],x['mode'],x['backend'])==(b['operator'],b['mode'],b['backend'])),key=lambda x:x['atoms'])
 assert [x['atoms'] for x in seq]==[256*2**i for i in range(len(seq))],b
 assert all(x['status']=='OK' for x in seq[:-1])
 assert seq[-1]['status'] in ('OOM','LIMIT','INDEX_GUARD','FALLBACK','ERROR')
 assert int(b['last_success_n'])==seq[-2]['atoms']
 assert int(b['next_n'])==seq[-1]['atoms'] and b['stop']==seq[-1]['status']
for x in csv.DictReader((r/'paired.csv').open()):
 key=x['operator'],int(x['atoms']),x['mode']
 t=lookup[(*key,'torch')];f=lookup[(*key,'fused')]
 assert t['status']==f['status']=='OK'
 assert float(x['torch_gpu_ms'])==t['gpu_ms_median']
 assert float(x['fused_gpu_ms'])==f['gpu_ms_median']
 assert math.isclose(float(x['gpu_speedup']),t['gpu_ms_median']/f['gpu_ms_median'],rel_tol=1e-14)
 assert x['input_samples_match']=='True'
 c=checkmap.get((key[0],key[1],'fwd_bwd')) or checkmap.get(key)
 metrics=c.get('checks',{}) if c else {}
 required=metrics.values() if key[2]=='fwd_bwd' else [metrics.get('output',{})]
 valid=bool(metrics) and all(v and v['failures']==0 for v in required)
 assert (x['correctness']=='PASS')==valid
for name in ['layernorm','graph_softmax','attention_alpha','equivariant_gate','equivariant_dropout','eqv3_validation']:
 p=repo/'doc'/f'{name}.md'
 for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' in link or link.startswith('#'):continue
  assert (p.parent/link.split('#')[0]).exists(),(p,link)
 for line in p.read_text().splitlines():
  if line.startswith('| Hygon BW |') and ' | Pass |' in line:
   a=[x.strip() for x in line.strip('|').split('|')]
   names={'GraphSoftmax':'softmax','AttentionAlpha':'alpha','e3nn Gate':'gate','LayerNorm':'norm','SeparableLayerNorm':'separable','EquivariantDropout':'dropout'}
   if len(a)==9 and a[1] in names:
    q=checkmap.get((names[a[1]],int(a[2].replace(',','')),'fwd_bwd')) or checkmap.get((names[a[1]],int(a[2].replace(',','')),'fwd'))
    assert q
    expected=q['checks'].values() if a[3]=='Forward + backward' else [q['checks']['output']]
    assert all(y['failures']==0 for y in expected)
# No computation or tests changed during this documentation task.
changed=subprocess.check_output(['git','diff','--name-only'],cwd=repo).decode().splitlines()
assert all(p.startswith('doc/') for p in changed),changed
print('Verified',len(points),'execution points,',len(checks),'full comparisons, 24 stopping boundaries, all medians/tolerances, production/test hashes and documentation links.')
