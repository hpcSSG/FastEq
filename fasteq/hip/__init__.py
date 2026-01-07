import os, glob
import torch

_this_dir = os.path.dirname(__file__)
cands = glob.glob(os.path.join(_this_dir, "_hip*.so"))
if not cands:
    raise ImportError(f"Cannot find _hip*.so under: {_this_dir}")

so_path = max(cands, key=os.path.getmtime)
torch.ops.load_library(so_path)
