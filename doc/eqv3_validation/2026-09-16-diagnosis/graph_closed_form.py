from pathlib import Path
import json,torch
import fasteq.triton.graph_softmax as mod
out={}
for path in sorted(Path('fixtures').glob('*.pt')):
 data=torch.load(path,map_location='cpu',weights_only=True)
 x=data['x'].double();r=data['r'].double();idx=data['index'];g=data['g'].double();eps=1e-16
 z=3*torch.tanh(x/3);e=torch.exp(z-mod.scatter(z,idx,0,dim_size=17,reduce='max')[idx])
 sums=mod.scatter(e,idx,0,dim_size=17,reduce='sum')
 gs=mod.scatter(g*e,idx,0,dim_size=17,reduce='sum')
 # For broadcast-per-head r, each node's denominator is r*sum(e)+eps.
 closed=(eps*gs/(r*sums+eps).square()).sum(0,keepdim=True)
 rr=r.detach().requires_grad_();a=e*rr;y=a/(mod.scatter(a,idx,0,dim_size=17,reduce='sum')[idx]+eps)
 actual,=torch.autograd.grad(y,rr,g)
 out[path.name]={'fp64_cpu_closed_form':closed.tolist(),'fp64_cpu_autograd':actual.tolist(),'rescale':r.tolist()}
Path('h100/graph_closed_form.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
