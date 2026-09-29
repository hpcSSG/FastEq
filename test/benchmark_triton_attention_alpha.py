import argparse, gc, hashlib, importlib.util, json, os, time
from pathlib import Path
import torch
import torch.nn.functional as F
import triton

ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=4096);ap.add_argument('--suite',action='store_true');ap.add_argument('--bench',action='store_true');ap.add_argument('--module',default=str(Path(__file__).resolve().parents[1] / 'fasteq/triton/fused_attention_alpha.py'));ap.add_argument('--out',required=True);args=ap.parse_args()
spec=importlib.util.spec_from_file_location('alpha_candidate',args.module);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(8)
results={'module_sha256':hashlib.sha256(Path(args.module).read_bytes()).hexdigest(),'device':torch.cuda.get_device_name(),'torch':torch.__version__,'triton':triton.__version__,'cases':[]}
def metric(a,b):
 a=a.detach().reshape(-1);b=b.detach().reshape(-1);bad=0;max_ratio=0.;max_abs=0.;sse=0.;ssb=0.;worst=[]
 for start in range(0,a.numel(),1048576):
  aa=a[start:start+1048576];bb=b[start:start+1048576];d=(aa-bb).abs();ratio=d/(5e-5+5e-4*bb.abs());bad+=int(((ratio>1)|~torch.isfinite(ratio)).sum());idx=int(ratio.argmax())
  rr=float(ratio[idx]);max_abs=max(max_abs,float(d.max()));sse+=float(d.double().square().sum());ssb+=float(bb.double().square().sum())
  if rr>max_ratio:max_ratio=rr;worst=[start+idx,float(aa[idx]),float(bb[idx])]
 return dict(bad=bad,max_ratio=max_ratio,max_abs=max_abs,relative_l2=(sse/max(ssb,1e-300))**.5,worst=worst)
def save():Path(args.out).write_text(json.dumps(results,indent=2)+'\n')
def run(E,H,C,ln=True,affine='both',act='smooth_leaky_relu',p=0.,training=True,layout=False,subset=None,seed=20260914,stages=False):
 torch.manual_seed(seed)
 w=torch.randn(H,C,device='cuda',requires_grad=True)
 g=torch.randn(C,device='cuda',requires_grad=True) if ln and affine in ('both','weight') else None
 b=torch.randn(C,device='cuda',requires_grad=True) if ln and affine in ('both','bias') else None
 x=torch.randn(E,H,C,device='cuda')
 if layout:x=x.transpose(0,1).contiguous().transpose(0,1)
 x.requires_grad_(True)
 dy=torch.randn(E,H,device='cuda')
 if layout:dy=dy.T.contiguous().T
 items=[('x',x),('alpha_dot',w)]+([('norm_weight',g)] if g is not None else [])+([('norm_bias',b)] if b is not None else [])
 if subset is not None:
  for name,t in items:t.requires_grad_(name in subset)
  items=[(name,t) for name,t in items if t.requires_grad]
 # Reuse the exact Triton dropout mask in the independent Torch reference.
 kwargs=dict(use_layer_norm=ln,activation=act,dropout_p=p,training=training,seed=torch.tensor(123,device='cuda',dtype=torch.int64))
 y=m.fused_attention_alpha(x,w,g,b,**kwargs)
 saved=y.grad_fn.saved_tensors
 mask=saved[-1]
 z=F.layer_norm(x,(C,),g,b,1e-5) if ln else x
 if act=='silu':a=F.silu(z)
 else:
  x1=.6*z
  x2=.4*z*(2*torch.sigmoid(z)-1)
  a=x1+x2
 pe=p if training else 0
 if pe:a=a*(mask.to(x.dtype)/(1-pe) if pe<1 else 0.)
 ref=torch.einsum('bik,ik->bi',a,w)
 ref_grads=torch.autograd.grad(ref,[t for _,t in items],dy)
 grads=torch.autograd.grad(y,[t for _,t in items],dy)
 checks={'output':metric(y,ref),**{name:metric(v,r) for (name,_),v,r in zip(items,grads,ref_grads)}}
 rec=dict(E=E,H=H,C=C,ln=ln,affine=affine,activation=act,p=p,training=training,layout=layout,subset=subset,seed=seed,checks=checks)
 if stages and hasattr(m, '_partial_norm'):
  with torch.no_grad():
   _,_,u,zz,aa,rs,_=saved
   un,mn,rn=torch.native_layer_norm(x,(C,),None,None,1e-5)
   rec['stages']={'normalized':metric(u,un),'affine':metric(zz,z),'activation':metric(aa,a),'rstd':metric(rs,rn)}
 if args.bench:
  import statistics
  def bench(fn):
   for _ in range(5):fn()
   torch.cuda.synchronize();ts=[]
   for _ in range(20):
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();fn();e.record();e.synchronize();ts.append(s.elapsed_time(e))
   return statistics.median(ts)
  def native():
   z=F.layer_norm(x,(C,),g,b,1e-5) if ln else x
   a=F.silu(z) if act=='silu' else .6*z+.4*z*(2*torch.sigmoid(z)-1)
   return torch.einsum('bik,ik->bi',a,w)
  fast=lambda:m.fused_attention_alpha(x,w,g,b,**kwargs)
  rec['timing']={}
  for name,fn in [('native',native),('triton',fast)]:
   with torch.no_grad():forward=bench(fn)
   rec['timing'][name]=dict(forward_ms=forward,forward_backward_ms=bench(lambda:torch.autograd.grad(fn(),[t for _,t in items],dy)))
 results['cases'].append(rec);save();print(json.dumps({k:v for k,v in rec.items() if k!='stages'}),flush=True)
 del x,w,g,b,dy,y,ref,grads,ref_grads,saved;gc.collect();torch.cuda.empty_cache()
if args.suite:
 for c in [1,3,13,32,127,256,1025,8192]:
  for ln in [False,True]:
   for act in ['silu','smooth_leaky_relu']:
    run(17,3,c,ln=ln,affine='both' if ln else 'none',act=act)
 for affine in ['none','weight','bias','both']:
  for p in [0.,.25,1.]:run(19,5,33,affine=affine,p=p,layout=True)
 for subset in [['x'],['alpha_dot'],['norm_weight'],['norm_bias'],['alpha_dot','norm_bias']]:run(11,2,13,subset=subset)
 run(0,3,32);run(1,1,1);run(19,3,32,p=.4,training=False)
else:run(args.n*32,8,32,stages=True)
save();bad=sum(c['bad'] for r in results['cases'] for c in r['checks'].values());print('TOTAL_BAD',bad,flush=True)
raise SystemExit(bool(bad))
