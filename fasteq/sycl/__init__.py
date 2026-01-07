import os, glob
import torch

_this_dir = os.path.dirname(__file__)
cands = glob.glob(os.path.join(_this_dir, "_sycl*.so"))
if not cands:
    raise ImportError(f"Cannot find _sycl*.so under: {_this_dir}")

so_path = max(cands, key=os.path.getmtime)
torch.ops.load_library(so_path)
