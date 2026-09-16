"""Additional existing checks and source API coverage; no kernel changes."""
import gc
import importlib
import json
from pathlib import Path
import time
import traceback

import torch
from cases import Case, check, source_class, ORIGINAL
from run import write, ROOT

torch.set_num_threads(8)
results=[]
for name,module,function in (
    ('graph_softmax_builtin','graph_softmax','check'),
    ('attention_alpha_builtin','fused_attention_alpha','self_test'),
    ('equivariant_dropout_builtin','fused_equivariant_dropout','self_test')):
    t=time.monotonic()
    try:
        mod=importlib.import_module('fasteq.triton.'+module)
        getattr(mod,function)()
        row=dict(name=name,status='PASS')
    except Exception as e:
        row=dict(name=name,status='ERROR',error=str(e),traceback=traceback.format_exc())
    row['seconds']=time.monotonic()-t
    results.append(row);write(ROOT/'extra_checks.json',results)
    print(json.dumps(row),flush=True)
    gc.collect();torch.cuda.empty_cache()

for op in ('alpha','gate','norm','separable','dropout'):
    try:
        case=Case(op,17,'fwd_bwd',channels=13,alpha_channels=13,
                  layout='KNC')
        row=check(case);row['name']=op+'_irregular_strided'
        del case
    except Exception as e:
        row=dict(name=op+'_irregular_strided',status='ERROR',error=str(e),traceback=traceback.format_exc())
    results.append(row);write(ROOT/'extra_checks.json',results)
    print(json.dumps({k:row[k] for k in ('name','status')}),flush=True)
    gc.collect();torch.cuda.empty_cache()

# The catalog's e3nn Gate is a different API from the actual EQv3 GateActivation.
try:
    mod=importlib.import_module('fasteq.triton.fused_gate_act')
    original=source_class(ORIGINAL/'activation.py','GateActivation')(3,3).cuda()
    candidate=mod.fused_gate_activation(3,3,13).cuda()
    x=torch.randn(17,16,13,device='cuda')
    g=torch.randn(17,39,device='cuda')
    expected=original(x,g)
    actual=candidate(g,x)
    torch.testing.assert_close(actual,expected,atol=5e-5,rtol=5e-4)
    row=dict(name='eqv3_gate_activation_public_module',status='PASS')
except Exception as e:
    row=dict(name='eqv3_gate_activation_public_module',status='ERROR',
             error=str(e),traceback=traceback.format_exc())
results.append(row);write(ROOT/'extra_checks.json',results)
print(json.dumps(row),flush=True)
