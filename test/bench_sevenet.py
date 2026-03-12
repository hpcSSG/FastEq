# bench_cueq_vs_flashtp.py
import copy
import numpy as np
import torch
from ase.build import bulk

import sevenn
import sevenn.train.dataload as dl
from sevenn.calculator import SevenNetCalculator

from torch.profiler import profile, record_function, ProfilerActivity

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
sort_by_keyword = "self_cuda_time_total"

cutoff = 4.0

#_atoms = bulk("NaCl", "rocksalt", a=4.00) * (2, 2, 2)
_atoms = bulk("NaCl", "rocksalt", a=4.00) * (16, 16, 16) 
_atoms.rattle()

print(f"atoms:{_atoms}")


def assert_atoms(atoms1, atoms2, rtol=1e-5, atol=1e-6):
    def acl(a, b, rtol=rtol, atol=atol):
        return np.allclose(a, b, rtol=rtol, atol=atol)

    assert len(atoms1) == len(atoms2)
    assert acl(atoms1.get_cell(), atoms2.get_cell())
    assert acl(atoms1.get_potential_energy(), atoms2.get_potential_energy())
    assert acl(atoms1.get_forces(), atoms2.get_forces(), rtol * 10, atol * 10)
    assert acl(
        atoms1.get_stress(voigt=False),
        atoms2.get_stress(voigt=False),
        rtol * 10,
        atol * 10,
    )


def clear_ase_cache(atoms):
    """
    ASE 会缓存 calc.results；不清缓存会导致后续迭代不重新算。
    """
    calc = atoms.calc
    if hasattr(calc, "reset"):
        try:
            calc.reset()
        except TypeError:
            pass
    if hasattr(calc, "results") and isinstance(calc.results, dict):
        calc.results.clear()


def singlepoint(atoms):
    """
    模拟真实 calculator 使用：energy -> forces -> stress
    通常 energy 触发一次计算，forces/stress 读缓存。
    """
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    #s = atoms.get_stress(voigt=False)
    #return e, f, s
    return e, f


def bench_atoms_cuda_event(atoms, warmup=20, iters=200, label=""):
    """
    用 CUDA Event 计时每次 singlepoint 的总耗时（energy+forces+stress）。
    """
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(warmup):
        clear_ase_cache(atoms)
        _ = singlepoint(atoms)
    torch.cuda.synchronize()

    # measure (per-iter)
    times_ms = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        clear_ase_cache(atoms)
        torch.cuda.synchronize()

        starter.record()
        _ = singlepoint(atoms)
        ender.record()

        torch.cuda.synchronize()
        times_ms[i] = starter.elapsed_time(ender)
    '''
    print(
        f"{label:>10s} | mean {times_ms.mean():8.3f} ms  "
        f"p50 {np.percentile(times_ms,50):8.3f}  "
        f"p90 {np.percentile(times_ms,90):8.3f}  "
        f"min {times_ms.min():8.3f}  max {times_ms.max():8.3f}"
    )

    clear_ase_cache(atoms)
    with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
        singlepoint(atoms)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("cueq-sevenet-trace.json")
    '''


    return times_ms


def make_calc_flashtp(path, modal="mpa"):
    """
    如果 SevenNetCalculator 的参数名确实是 enable_flash，则优先用正确拼写。
    """
    try:
        return SevenNetCalculator(path, modal=modal, enable_flash=True)
    except TypeError:
        # fallback to the typo name, if that is what the codebase uses
        return SevenNetCalculator(path, modal=modal, enbale_flash=False)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Need GPU for this benchmark.")
    print("torch  =", torch.__version__)
    print("cuda   =", torch.version.cuda)
    print("gpu    =", torch.cuda.get_device_name(0))
    print("sevenn =", sevenn.__version__)

    # 你给的 checkpoint
    path = "/home/malixian/repos/SevenNet/models/checkpoint_sevennet_omni.pth"
    modal = "mpa"

    torch.set_grad_enabled(True)

    # Build calculators
    # flashtp_calc = make_calc_flashtp(path, modal=modal)
    cueq_calc = SevenNetCalculator(path, modal=modal, enable_cueq=True)

    #atoms_flash = copy.deepcopy(_atoms)
    atoms_cueq = copy.deepcopy(_atoms)
    #atoms_flash.calc = flashtp_calc
    atoms_cueq.calc = cueq_calc
    
    '''
    # -------------------------
    # 1) correctness check
    # -------------------------
    print("\n[1] Correctness check (cueq vs flashtp)")
    # 先各算一次把结果填出来
    clear_ase_cache(atoms_flash)
    clear_ase_cache(atoms_cueq)
    _ = singlepoint(atoms_flash)
    _ = singlepoint(atoms_cueq)

    assert_atoms(atoms_flash, atoms_cueq)
    print("OK: results match within tolerances.")
    '''

    # -------------------------
    # 2) performance benchmark
    # -------------------------
    warmup = 1
    iters = 0
    t_cueq = bench_atoms_cuda_event(atoms_cueq, warmup=warmup, iters=iters, label="cueq")

    '''
    print(f"\n[2] Benchmark: singlepoint (energy->forces->stress), warmup={warmup}, iters={iters}")

    t_flash = bench_atoms_cuda_event(atoms_flash, warmup=warmup, iters=iters, label="flashtp")
    t_cueq = bench_atoms_cuda_event(atoms_cueq, warmup=warmup, iters=iters, label="cueq")

    speedup = t_cueq.mean() / t_flash.mean()
    print(f"\nSpeedup (cueq / flashtp) = {speedup:.3f}x")
    print(f"Speedup (flashtp / cueq) = {(t_flash.mean() / t_cueq.mean()):.3f}x")
    '''


if __name__ == "__main__":
    main()
