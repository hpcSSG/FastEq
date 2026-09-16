"""Stage-wise diagnosis of unchanged GraphSoftmax and AttentionAlpha operators."""
import argparse,contextlib,gc,hashlib,json,os,socket,statistics,time
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import fasteq.triton.fused_attention_alpha as alpha
import fasteq.triton.graph_softmax as soft
from cases import Case,compare_cpu,load_file,ORIGINAL

parser=argparse.ArgumentParser();parser.add_argument('task',choices=['alpha','graph','graph_perf','profile']);parser.add_argument('--n',type=int,default=4096);parser.add_argument('--out',required=True);parser.add_argument('--fixtures',default='fixtures');args=parser.parse_args()
torch.set_num_threads(8)
OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
result={'task':args.task,'N':args.n,'environment':{'host':socket.gethostname(),'torch':torch.__version__,'hip':torch.version.hip,'triton':triton.__version__,'device':torch.cuda.get_device_name(0),'target':str(triton.runtime.driver.active.get_current_target())}}
result['source_sha256']={str(Path(m.__file__).name):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (alpha,soft)}
result['reference_sha256']={name:hashlib.sha256((ORIGINAL/name).read_bytes()).hexdigest() for name in ('softmax.py','activation.py')}
def write(): (OUT/(args.task+'_'+str(args.n)+'.json')).write_text(json.dumps(result,indent=2)+'\n')
def mark(name,values):
 result[name]=values;write();print(name,json.dumps(values)[:1200],flush=True)
def metric(a,b,atol=5e-5,rtol=5e-4):return compare_cpu(a.detach().cpu(),b,atol,rtol)
def bench(fn,warmup=5,reps=20):
 for _ in range(warmup):fn()
 torch.cuda.synchronize();gpu=[];wall=[]
 for _ in range(reps):
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  t=time.perf_counter_ns();start.record();fn();end.record();end.synchronize()
  gpu.append(start.elapsed_time(end));wall.append((time.perf_counter_ns()-t)/1e6)
 return {'gpu_ms':statistics.median(gpu),'wall_ms':statistics.median(wall),'gpu_samples':gpu,'wall_samples':wall}

def native_recompute(x,w,gamma,beta,ln,eps,silu,slope,p,mask):
 z=F.layer_norm(x,(x.shape[-1],),gamma,beta,eps) if ln else x
 s=torch.sigmoid(z)
 a=z*s if silu else ((1+slope)/2)*z+((1-slope)/2)*z*(2*s-1)
 if p:a=a*(mask.to(x.dtype)/(1-p) if p<1 else 0.)
 return torch.einsum('bik,ik->bi',a,w)
@contextlib.contextmanager
def patched_reference():
 old=alpha._reference;alpha._reference=native_recompute
 try:yield
 finally:alpha._reference=old

if args.task=='alpha':
 c=Case('alpha',args.n,'fwd_bwd');inputs=dict(zip(c.names,c.inputs));x,w,g,b=[inputs[k] for k in ('x','alpha_dot','norm_weight','norm_bias')];dy=c.dy;eps=c.norm.eps
 def build(ln,dot):
  if ln=='native':z=c.norm(x)
  else:
   centered=x-x.mean(-1,keepdim=True)
   z=centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+eps)
   z=z*g+b
  a=c.act(z)
  return torch.einsum('bik,ik->bi',a,w) if dot=='einsum' else (a*w).sum(-1)
 with torch.no_grad():
  zn,mn,rn=torch.native_layer_norm(x,(x.shape[-1],),g,b,eps)
  mm=x.mean(-1,keepdim=True);centered=x-mm;rm=torch.rsqrt(centered.square().mean(-1,keepdim=True)+eps);zm=centered*rm*g+b
  mark('statistics',{'mean':metric(mm,mn),'rstd':metric(rm,rn),'normalized_affine':metric(zm,zn),'activation':metric(c.act(zm),c.act(zn))})
  del zn,mn,rn,mm,centered,rm,zm
 y=build('native','einsum');ref=[y.detach().cpu()]+[v.detach().cpu() for v in torch.autograd.grad(y,c.inputs,dy)];del y
 variants={}
 for ln,dot in [('manual','sum'),('manual','einsum'),('native','sum'),('native','einsum')]:
  y=build(ln,dot);gg=torch.autograd.grad(y,c.inputs,dy)
  variants[ln+'_'+dot]={name:metric(a,e) for name,a,e in zip(['output']+c.names,[y]+list(gg),ref)}
  del y,gg;gc.collect();torch.cuda.empty_cache()
  mark('ablation',variants)
 y=c.fast();gg=torch.autograd.grad(y,c.inputs,dy)
 mark('public_fused',{name:metric(a,e) for name,a,e in zip(['output']+c.names,[y]+list(gg),ref)})
 del y,gg
 with patched_reference():
  y=c.fast();gg=torch.autograd.grad(y,c.inputs,dy)
  mark('native_recompute_candidate',{name:metric(a,e) for name,a,e in zip(['output']+c.names,[y]+list(gg),ref)})
 del y,gg,ref;gc.collect();torch.cuda.empty_cache()
 timing={}
 for name,fn in [('native_fwd',lambda:build('native','einsum')),('manual_fwd',lambda:build('manual','sum')),('triton_fwd',c.fast)]:
  with torch.no_grad():timing[name]=bench(fn)
 for name,fn in [('native_fwd_bwd',lambda:build('native','einsum')),('manual_fwd_bwd',lambda:build('manual','sum')),('triton_fwd_bwd',c.fast)]:
  timing[name]=bench(lambda:torch.autograd.grad(fn(),c.inputs,dy))
 with patched_reference():timing['triton_native_recompute_fwd_bwd']=bench(lambda:torch.autograd.grad(c.fast(),c.inputs,dy))
 for name,fn in [('native_norm',lambda:c.norm(x)),('manual_norm',lambda:((x-x.mean(-1,keepdim=True))*torch.rsqrt((x-x.mean(-1,keepdim=True)).square().mean(-1,keepdim=True)+eps))*g+b)]:
  with torch.no_grad():timing[name]=bench(fn)
 mark('timing',timing)

if args.task=='graph':
 source=load_file(ORIGINAL/'softmax.py','native_graph_diagnosis')
 failures={}
 for path in sorted(Path(args.fixtures).glob('*.pt')):
  fixture=torch.load(path,map_location='cpu',weights_only=True)
  # One fixture from each device. Replay identical stored tensors here.
  cpu={k:fixture[k] for k in ['x','index','r','g']}
  hashes={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in cpu.items()}
  x=cpu['x'].cuda().requires_grad_();idx=cpu['index'].cuda();r=cpu['r'].cuda().requires_grad_();up=cpu['g'].cuda();n=17
  csr=soft.prepare_graph_softmax(idx,num_nodes=n)
  native=source.GraphSoftmax(eps=1e-16,softcap=3.,exp_dropout=0.).cuda().train()
  def run(which):
   if which=='native':y=native(x,idx,num_nodes=n,exp_rescale=r)
   elif which=='fused':y=soft.fused_graph_softmax(x,idx,num_nodes=n,exp_rescale=r,csr=csr,softcap=3.)
   else:
    z=x.double().requires_grad_();rr=r.double().requires_grad_()
    y=native(z,idx,num_nodes=n,exp_rescale=rr)
    grads=torch.autograd.grad(y,[z,rr],up.double());return y.detach(),[v.detach() for v in grads]
   grads=torch.autograd.grad(y,[x,r],up);return y.detach(),[v.detach() for v in grads]
  repeats={k:[] for k in ['native','fused','fp64']}
  for _ in range(20):
   for name in repeats:
    y,grads=run(name);repeats[name].append(grads[1].cpu().tolist())
  yf,gf=run('fused');yn,gn=run('native');yd,gd=run('fp64')
  details={'hashes':hashes,'grad_r_repeats':repeats,'fused_vs_native':{'output':metric(yf,yn,3e-6,3e-5),'dx':metric(gf[0],gn[0],3e-5,3e-4),'dr':metric(gf[1],gn[1],3e-5,3e-4)},'fused_vs_fp64_dr':metric(gf[1].double(),gd[1],3e-5,3e-4),'native_vs_fp64_dr':metric(gn[1].double(),gd[1],3e-5,3e-4)}
  # Source-ordered Torch division backward versus compact softmax identity.
  with torch.no_grad():
   z=3*torch.tanh(x/3);e=torch.exp(z-soft.scatter(z,idx,0,dim_size=n,reduce='max')[idx]);a=e*r
   den=soft.scatter(a,idx,0,dim_size=n,reduce='sum')[idx]+1e-16;y=a/den
   dot=soft.scatter(up*y,idx,0,dim_size=n,reduce='sum')[idx]
   compact=((e/den)*(up-dot)).sum_to_size(r.shape)
   native_order=((up/den+soft.scatter(-(up*a)/(den*den),idx,0,dim_size=n,reduce='sum')[idx])*e).sum_to_size(r.shape)
   details['torch_arithmetic_dr']={'compact':compact.cpu().tolist(),'source_order':native_order.cpu().tolist()}
  failures[path.name]=details;mark('fixtures',failures)

if args.task=='graph_perf':
 c=Case('softmax',args.n,'fwd_bwd');x,r=c.inputs;up=c.dy;csr=c.csr
 # Keep the exact graph and edge order; vary only the launch geometry.
 index=csr.source;timings={}
 for name,fn in [('native',c.reference),('current',c.fast),('current_with_csr',c.cold)]:
  with torch.no_grad():fw=bench(fn)
  fb=bench(lambda:torch.autograd.grad(fn(),c.inputs,up))
  y=fn();bw=bench(lambda:torch.autograd.grad(y,c.inputs,up,retain_graph=True));del y
  timings[name]={'fwd':fw,'fwd_bwd':fb,'bwd_retain_graph':bw};mark('timings',timings)
 # Check every launch configuration against the same native output and gradients.
 y=c.reference();ref=[y.detach().cpu()]+[t.detach().cpu() for t in torch.autograd.grad(y,c.inputs,up)];del y
 configurations={}
 for heads,warps in [(8,1),(8,2),(8,4),(4,1),(4,2),(2,1),(1,1)]:
  fn=lambda:soft.fused_graph_softmax(x,index,num_nodes=args.n,exp_rescale=r,csr=csr,softcap=3.,block_heads=heads,num_warps=warps)
  try:
   y=fn();gg=torch.autograd.grad(y,c.inputs,up)
   checks={name:metric(a,b,3e-6 if name=='output' else 3e-5,3e-5 if name=='output' else 3e-4) for name,a,b in zip(['output','dx','dr'],[y]+list(gg),ref)}
   del y,gg
   with torch.no_grad():fw=bench(fn)
   configurations[f'h{heads}_w{warps}']={'checks':checks,'fwd':fw,'fwd_bwd':bench(lambda:torch.autograd.grad(fn(),c.inputs,up))}
  except Exception as exc:configurations[f'h{heads}_w{warps}']={'error':str(exc)}
  mark('launch_configurations',configurations)
 with torch.no_grad():mark('csr_only',bench(lambda:soft.prepare_graph_softmax(index,num_nodes=args.n)))

if args.task=='profile':
 # Same N, warm operator first, then record a single first-order invocation.
 profiles={}
 for op in ['alpha','softmax']:
  c=Case(op,args.n,'fwd_bwd')
  for backend in ['torch','fused']:
   for _ in range(3):c.invoke(backend)
   torch.cuda.synchronize()
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=False) as p:
    c.invoke(backend);torch.cuda.synchronize()
   events=p.key_averages();rows=[]
   for e in events:
    rows.append({'name':e.key,'calls':e.count,'self_cpu_us':e.self_cpu_time_total,'self_device_us':getattr(e,'self_device_time_total',0),'device_us':getattr(e,'device_time_total',0)})
   profiles[op+'_'+backend]=sorted(rows,key=lambda e:e['self_device_us'],reverse=True)
   (OUT/(op+'_'+backend+'_profile.txt')).write_text(events.table(sort_by='self_cuda_time_total',row_limit=40))
   mark('profiles',profiles)
write()
