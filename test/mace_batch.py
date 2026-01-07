import torch
from torch.profiler import profile, record_function, ProfilerActivity

from ase import units
import ase.io
import numpy as np
import time, os
from tqdm import tqdm

from mace.calculators import MACECalculator

import sys

if torch.cuda.is_available():
    device = 'cuda'
elif torch.xpu.is_available():
    device = 'xpu'
else:
    print('Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices')
    sys.exit(0)

mace_scale = "small"
dtype = "float64"

mace_scale = str(sys.argv[1])
dtype = str(sys.argv[2])

calculator = MACECalculator(model_paths='../../mace_bench/models/MACE-OFF23_' + mace_scale + '.model', device=device, default_dtype=dtype, compile_mode=None, enable_cueq=True, use_batch_size=16)

file_list = []
atoms_list = []
data_path = "../../mace_bench/test_data"
# args.configs now should be a directory.
for file in tqdm(os.listdir(data_path), desc="Reading files", unit="file"):
    if file.endswith(".cif"):
        file_list.append(file)
file_list.sort()
print(file_list)

for file in file_list:
    atoms_list.append(ase.io.read(os.path.join(data_path, file), index=0))


activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
sort_by_keyword = "self_" + device + "_time_total"

'''
with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
    calculator.batch_calculate(atoms_list=atoms_list)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=10))


prof.export_chrome_trace("cueq-trace.json")

'''
'''
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/mace'),
        activities=activities,
        #record_shapes=True,
        #profile_memory=True,
        with_modules=True,
        with_stack=True
) as prof:
    for step in range(0, 10):
        prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        calculator.batch_calculate(atoms_list=atoms_list)
'''


for i in range(0, 1):
    calculator.batch_calculate(atoms_list=atoms_list)
'''
with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
    calculator.batch_calculate(atoms_list=atoms_list)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
prof.export_chrome_trace("mace-70-trace.json")
'''
