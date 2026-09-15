"""Close two concrete harness coverage gaps, without rerunning the sweep."""
import json
import torch
from cases import Case, check
from run import ROOT, write

torch.set_num_threads(8)
case=Case('dropout',17,'fwd_bwd',channels=13)
with torch.no_grad():
    case.inputs[0].zero_()
row=check(case)
row['name']='dropout_exact_zero_input_mask_replay'
write(ROOT/'dropout_zero_check.json',row)
print(row['name'],row['status'],flush=True)
del case

case=Case('gate',17,'fwd_bwd',channels=13,layout='KNC')
assert not case.inputs[0].is_contiguous()
row=check(case)
row['name']='gate_irregular_strided'
row['input_stride']=list(case.inputs[0].stride())
write(ROOT/'gate_strided_check.json',row)
print(row['name'],row['status'],row['input_stride'],flush=True)

# Preserve the initial record, whose Gate check actually used contiguous input.
path=ROOT/'extra_checks.json'
initial=ROOT/'extra_checks_initial.json'
if not initial.exists(): initial.write_bytes(path.read_bytes())
rows=json.loads(initial.read_text())
for entry in rows:
    if entry['name']=='gate_irregular_strided' and entry.get('layout')=='NKC':
        entry['name']='gate_irregular_contiguous'
rows=[entry for entry in rows if entry['name'] not in ('gate_irregular_strided','dropout_exact_zero_input_mask_replay')]
rows.append(row)
rows.append(json.loads((ROOT/'dropout_zero_check.json').read_text()))
write(path,rows)
