import json
import contextlib
import importlib
import io
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

versions = {}
for name in ("torch", "e3nn", "mace-torch", "ninja", "fasteq", "numpy", "scipy",
             "ase", "torch-geometric", "matscipy", "joblib", "psutil", "setuptools", "wheel"):
    try:
        versions[name] = version(name)
    except PackageNotFoundError:
        versions[name] = None

# Import paths matter: the historical venv also used packages from a base module.
modules = {}
torch_state = None
for name in ("torch", "e3nn", "mace", "numpy", "scipy", "ase", "torch_geometric",
             "matscipy", "joblib", "psutil", "fasteq"):
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            module = importlib.import_module(name)
            modules[name] = {"path": getattr(module, "__file__", None),
                             "version": str(getattr(module, "__version__", "unknown"))}
            if name == "torch":
                torch_state = {"cuda_available": module.cuda.is_available(),
                               "num_threads": module.get_num_threads(),
                               "interop_threads": module.get_num_interop_threads(),
                               "build_config": module.__config__.show()}
    except Exception as exc:
        modules[name] = {"import_error": str(exc)}
    if captured.getvalue():
        modules[name]["import_output"] = captured.getvalue()

allowed = {}
status = Path('/proc/self/status')
if status.exists():
    for line in status.read_text().splitlines():
        if line.startswith(('Cpus_allowed_list:', 'Mems_allowed_list:')):
            key, value = line.split(':', 1)
            allowed[key] = value.strip()
print(json.dumps({
    "python": sys.version, "python_executable": sys.executable,
    "platform": platform.platform(), "hostname": platform.node(),
    "machine": platform.machine(), "packages": versions,
    "modules": modules, "torch": torch_state,
    "logical_cpu_count": os.cpu_count(), "process_allowed_lists": allowed,
    "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
    "environment": {k: os.environ.get(k) for k in (
        "CC", "CXX", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NATIVE_CPU_BLAS_INCLUDE", "NATIVE_CPU_BLAS_LIB",
        "NATIVE_CPU_BLAS_NAME", "FASTEQ_CPU_FAST_MATH", "MAX_JOBS",
        "TORCH_EXTENSIONS_DIR", "LD_LIBRARY_PATH", "PYTHONPATH",
        "CCS_JOB_ID", "CCS_JOB_STEP_RANK", "CCS_JOB_EXEC_NODES",
        "BATCHASE_TORCH_THREADS", "NATIVE_EVD_THREADS", "OMP_PROC_BIND", "OMP_PLACES")},
}, indent=2))
