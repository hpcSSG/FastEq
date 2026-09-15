"""Locate existing self-test failures and compare the same tensors to EQv3."""
import inspect
import json
import traceback
import contextlib
import io
import argparse

import torch
from cases import ORIGINAL, compare_cpu, load_file
from run import ROOT, write
import fasteq.triton.graph_softmax as module

torch.set_num_threads(8)
source = load_file(ORIGINAL / 'softmax.py', 'original_softmax_diagnosis')
original_assert = torch.testing.assert_close
p=argparse.ArgumentParser()
p.add_argument('--native',action='store_true')
p.add_argument('--repeats',type=int,default=1)
args=p.parse_args()
output=ROOT/('softmax_native_checks.json' if args.native else 'softmax_diagnosis.json')
results = dict(assertions=0, failed_assertions=[], source=str(ORIGINAL / 'softmax.py'),
    method='Original self-test, recording each failed assertion to continue coverage; no tolerance changes',
    reference='original EQv3 class' if args.native else 'repository reference',repeats=args.repeats)
if args.native:
    def native_reference(src,index=None,ptr=None,num_nodes=None,dim=0,exp_rescale=None,
                         eps=1e-16,exp_dropout=0.,softcap=None,training=False):
        native=source.GraphSoftmax(eps=eps,exp_dropout=exp_dropout,softcap=softcap).cuda().train(training)
        return native(src,index,ptr,num_nodes,dim,exp_rescale)
    module.reference_graph_softmax=native_reference


def recording_assert(a, b, **kwargs):
    results['assertions'] += 1
    try:
        original_assert(a, b, **kwargs)
    except AssertionError as exc:
        frame = inspect.currentframe().f_back
        state = frame.f_locals
        row = dict(assertion=results['assertions'], line=frame.f_lineno,
                   repeat=results['current_repeat'],error=str(exc), comparison=compare_cpu(a.detach().cpu(), b, **kwargs))
        row['configuration'] = {k: state[k] for k in ('E', 'N', 'H', 'cap', 'eps', 'rshape', 'p') if k in state}
        if frame.f_lineno == 54:
            gradient = next(i for i, value in enumerate(state['ga']) if value is a)
            row['gradient'] = ('x', 'exp_rescale')[gradient]
            if args.native:
                row['against_original_eqv3']=row['comparison']
            else:
                native = source.GraphSoftmax(eps=state['eps'], exp_dropout=state['p'],
                                             softcap=state['cap']).cuda().train()
                y = native(state['x'], state['index'], num_nodes=state['N'], exp_rescale=state['r'])
                native_grads = torch.autograd.grad(y, state['args'], state['g'])
                row['against_original_eqv3'] = compare_cpu(a.detach().cpu(), native_grads[gradient], **kwargs)
                row['repository_reference_against_original'] = compare_cpu(b.detach().cpu(), native_grads[gradient], **kwargs)
            directory=ROOT/'softmax_failures';directory.mkdir(exist_ok=True)
            saved={k:state[k].detach().cpu() if state[k] is not None else None for k in ('x','index','r','g')}
            saved.update(actual_gradient=a.detach().cpu(),reference_gradient=b.detach().cpu())
            saved_path=directory/f"{'native' if args.native else 'builtin'}_{row['repeat']}_{row['assertion']}.pt"
            torch.save(saved,saved_path)
            row['saved_tensors']=str(saved_path.relative_to(ROOT))
        results['failed_assertions'].append(row)
        write(output, results)


torch.testing.assert_close = recording_assert
try:
    # The original check's final print assumes every assertion raises on error.
    # This recorder continues after failures, so suppress that unconditional text.
    for repeat in range(args.repeats):
        results['current_repeat']=repeat
        with contextlib.redirect_stdout(io.StringIO()):
            module.check()
    results['completed_all_assertions'] = True
except Exception:
    results['completed_all_assertions'] = False
    results['execution_error'] = traceback.format_exc()
finally:
    torch.testing.assert_close = original_assert
results['status'] = 'FAIL' if results['failed_assertions'] else 'PASS'
if not results.get('completed_all_assertions'):
    results['status'] = 'ERROR'
write(output, results)
print(json.dumps({k: v for k, v in results.items() if k != 'failed_assertions'}))
for row in results['failed_assertions']:
    print(json.dumps(dict(configuration=row['configuration'], gradient=row.get('gradient'),
                         max_abs=row['comparison']['max_abs'],
                         failures=row['comparison']['failures'],
                         source_max_abs=row.get('against_original_eqv3', {}).get('max_abs'))))
