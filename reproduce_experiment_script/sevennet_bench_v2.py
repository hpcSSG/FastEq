#!/usr/bin/env python3
# bench_e3nn_cueq_flashtp_cu.py

import argparse
import csv
import gc
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

import sevenn
import sevenn.train.dataload as dl  # noqa: F401
from sevenn.calculator import SevenNetCalculator


CSV_FIELDS = [
    "status",
    "backend",
    "model",
    "modal",
    "replica",
    "num_atoms",
    "warmup",
    "iters",
    "wall_mean_ms",
    "wall_p50_ms",
    "wall_p90_ms",
    "wall_min_ms",
    "wall_max_ms",
    "gpu_mean_ms",
    "gpu_p50_ms",
    "gpu_p90_ms",
    "atoms_per_second",
    "peak_allocated_gib",
    "peak_reserved_gib",
]


# ============================================================
# Cu system
# ============================================================

def build_cu_supercell(
    replica: int,
    lattice_constant: float = 3.615,
    rattle_std: float = 0.01,
    seed: int = 1234,
) -> Atoms:
    """
    构造 FCC Cu 常规晶胞超胞。

    cubic=True 的 FCC 常规晶胞包含 4 个原子，因此：

        num_atoms = 4 * replica^3

    例如：
        replica=10 -> 4,000 atoms
        replica=19 -> 27,436 atoms，约等于论文中的 28K
        replica=20 -> 32,000 atoms
    """
    if replica <= 0:
        raise ValueError("replica must be positive")

    unit_cell = bulk(
        "Cu",
        crystalstructure="fcc",
        a=lattice_constant,
        cubic=True,
    )

    atoms = unit_cell * (replica, replica, replica)

    # 打破完美晶格对称性，避免所有力接近零。
    atoms.rattle(
        stdev=rattle_std,
        seed=seed,
    )

    atoms.wrap()
    return atoms


# ============================================================
# ASE and CUDA helpers
# ============================================================

def clear_ase_cache(atoms: Atoms) -> None:
    """
    ASE 会缓存 calculator results。

    对相同结构反复调用 get_forces() 时，如果不清缓存，
    ASE 可能直接返回上一次结果而不执行模型。
    """
    calculator = atoms.calc

    if calculator is None:
        return

    if hasattr(calculator, "reset"):
        try:
            calculator.reset()
        except TypeError:
            pass

    if hasattr(calculator, "results"):
        results = calculator.results
        if isinstance(results, dict):
            results.clear()


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def clear_cuda_memory(device: str) -> None:
    gc.collect()

    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def reset_peak_memory(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory(device: str) -> tuple[float, float]:
    if not device.startswith("cuda"):
        return float("nan"), float("nan")

    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3

    return allocated, reserved


# ============================================================
# Forward + backward
# ============================================================

def singlepoint_forward_backward(atoms: Atoms) -> np.ndarray:
    """
    执行一次 energy forward + position backward。

    SevenNet 首先预测能量，然后通过

        forces = -dE / dR

    得到原子力。因此 get_forces() 对应模型前向和坐标反向。

    这里不调用 get_potential_energy()，因为单独调用 energy
    不能明确保证执行 force backward。
    """
    return atoms.get_forces()


# ============================================================
# Calculator
# ============================================================

def make_calculator(
    model: str,
    modal: Optional[str],
    backend: str,
    device: str,
) -> SevenNetCalculator:
    """
    使用同一 SevenNet checkpoint 创建不同 TP 后端。

    e3nn:
        关闭 cuEquivariance、FlashTP 和 OpenEquivariance，
        使用 SevenNet 默认的 e3nn TensorProduct 实现。

    cueq:
        启用 cuEquivariance。

    flashtp:
        启用 FlashTP。
    """
    if backend not in {"e3nn", "cueq", "flashtp"}:
        raise ValueError(f"Unsupported backend: {backend}")

    kwargs = {
        "model": model,
        "device": device,
        "enable_cueq": backend == "cueq",
        "enable_flash": backend == "flashtp",
        "enable_oeq": False,
    }

    if modal is not None:
        kwargs["modal"] = modal

    try:
        return SevenNetCalculator(**kwargs)
    except TypeError as error:
        if backend == "flashtp":
            raise RuntimeError(
                "当前 SevenNet 版本不接受 enable_flash 参数。"
                "请安装支持 FlashTP 的新版 SevenNet；"
                "不要使用 enbale_flash 这一错误拼写。"
            ) from error
        raise


# ============================================================
# Benchmark
# ============================================================

def summarize_times(
    wall_times_ms: np.ndarray,
    gpu_times_ms: np.ndarray,
    num_atoms: int,
) -> Dict[str, float]:
    wall_mean_ms = float(np.mean(wall_times_ms))
    wall_p50_ms = float(np.percentile(wall_times_ms, 50))
    wall_p90_ms = float(np.percentile(wall_times_ms, 90))

    return {
        "wall_mean_ms": wall_mean_ms,
        "wall_p50_ms": wall_p50_ms,
        "wall_p90_ms": wall_p90_ms,
        "wall_min_ms": float(np.min(wall_times_ms)),
        "wall_max_ms": float(np.max(wall_times_ms)),
        "gpu_mean_ms": float(np.mean(gpu_times_ms)),
        "gpu_p50_ms": float(np.percentile(gpu_times_ms, 50)),
        "gpu_p90_ms": float(np.percentile(gpu_times_ms, 90)),
        "atoms_per_second": (
            num_atoms / (wall_mean_ms / 1000.0)
        ),
    }


def benchmark_atoms(
    atoms: Atoms,
    warmup: int,
    iters: int,
    device: str,
    label: str,
) -> Dict[str, float]:
    """
    测量 forces inference，即 energy forward + position backward。

    同时给出：

    wall time:
        包含 ASE 调用、邻居图准备、CPU->GPU 数据准备和 GPU 计算，
        更接近端到端推理时间。

    CUDA-event time:
        主要反映 GPU 时间。

    推荐论文绘图使用 wall_mean_ms。
    """
    if atoms.calc is None:
        raise RuntimeError("atoms.calc has not been assigned")

    # --------------------------------------------------------
    # Per-size warmup
    # --------------------------------------------------------
    print(f"  warming up: {warmup} iterations")

    for _ in range(warmup):
        clear_ase_cache(atoms)
        _ = singlepoint_forward_backward(atoms)

    synchronize(device)
    reset_peak_memory(device)

    wall_times_ms = np.empty(iters, dtype=np.float64)
    gpu_times_ms = np.empty(iters, dtype=np.float64)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    # --------------------------------------------------------
    # Timed iterations
    # --------------------------------------------------------
    for iteration in range(iters):
        clear_ase_cache(atoms)
        synchronize(device)

        wall_start = time.perf_counter()
        starter.record()

        forces = singlepoint_forward_backward(atoms)

        ender.record()
        synchronize(device)
        wall_end = time.perf_counter()

        # 防止结果在计时完成前被释放或优化掉。
        if forces.shape != (len(atoms), 3):
            raise RuntimeError(
                f"Unexpected force shape: {forces.shape}"
            )

        wall_times_ms[iteration] = (
            wall_end - wall_start
        ) * 1000.0

        gpu_times_ms[iteration] = starter.elapsed_time(ender)

    stats = summarize_times(
        wall_times_ms=wall_times_ms,
        gpu_times_ms=gpu_times_ms,
        num_atoms=len(atoms),
    )

    peak_allocated, peak_reserved = get_peak_memory(device)

    stats["peak_allocated_gib"] = peak_allocated
    stats["peak_reserved_gib"] = peak_reserved

    print(
        f"{label:>8s} | atoms {len(atoms):7d} | "
        f"wall mean {stats['wall_mean_ms']:9.3f} ms | "
        f"p50 {stats['wall_p50_ms']:9.3f} ms | "
        f"p90 {stats['wall_p90_ms']:9.3f} ms | "
        f"GPU {stats['gpu_mean_ms']:9.3f} ms | "
        f"{stats['atoms_per_second']:11,.0f} atom/s | "
        f"memory {peak_allocated:6.2f} GiB"
    )

    return stats


# ============================================================
# Output
# ============================================================

def write_csv(
    output_path: Path,
    rows: List[Dict],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def print_speedup_table(rows: List[Dict]) -> None:
    """
    打印三个后端的端到端平均时间，并以 e3nn 为基线计算加速比。

    speedup = e3nn_time / optimized_backend_time

    speedup > 1 表示对应优化后端快于 e3nn。
    """
    valid_rows = [
        row for row in rows
        if row["status"] == "OK"
    ]

    by_atom_count: Dict[int, Dict[str, Dict]] = {}

    for row in valid_rows:
        num_atoms = int(row["num_atoms"])
        backend = str(row["backend"])

        by_atom_count.setdefault(num_atoms, {})
        by_atom_count[num_atoms][backend] = row

    print()
    print("=" * 112)
    print("e3nn versus cuEq versus FlashTP")
    print("=" * 112)
    print(
        f"{'atoms':>10s} "
        f"{'e3nn/ms':>12s} "
        f"{'cuEq/ms':>12s} "
        f"{'FlashTP/ms':>12s} "
        f"{'e3nn/cuEq':>13s} "
        f"{'e3nn/FlashTP':>16s} "
        f"{'fastest':>12s}"
    )

    for num_atoms in sorted(by_atom_count):
        item = by_atom_count[num_atoms]

        def get_time(backend: str) -> Optional[float]:
            if backend not in item:
                return None
            return float(item[backend]["wall_mean_ms"])

        e3nn_ms = get_time("e3nn")
        cueq_ms = get_time("cueq")
        flash_ms = get_time("flashtp")

        available = {
            backend: value
            for backend, value in {
                "e3nn": e3nn_ms,
                "cuEq": cueq_ms,
                "FlashTP": flash_ms,
            }.items()
            if value is not None
        }

        if not available:
            continue

        fastest = min(available, key=available.get)

        e3nn_text = (
            f"{e3nn_ms:.3f}" if e3nn_ms is not None else "N/A"
        )
        cueq_text = (
            f"{cueq_ms:.3f}" if cueq_ms is not None else "N/A"
        )
        flash_text = (
            f"{flash_ms:.3f}" if flash_ms is not None else "N/A"
        )

        cueq_speedup = (
            f"{e3nn_ms / cueq_ms:.3f}x"
            if e3nn_ms is not None and cueq_ms is not None
            else "N/A"
        )
        flash_speedup = (
            f"{e3nn_ms / flash_ms:.3f}x"
            if e3nn_ms is not None and flash_ms is not None
            else "N/A"
        )

        print(
            f"{num_atoms:10d} "
            f"{e3nn_text:>12s} "
            f"{cueq_text:>12s} "
            f"{flash_text:>12s} "
            f"{cueq_speedup:>13s} "
            f"{flash_speedup:>16s} "
            f"{fastest:>12s}"
        )

def plot_results(
    rows: List[Dict],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skip plotting.")
        return

    plt.figure(figsize=(6.2, 4.4))

    for backend in ("e3nn", "cueq", "flashtp"):
        backend_rows = [
            row
            for row in rows
            if (
                row["status"] == "OK"
                and row["backend"] == backend
            )
        ]

        backend_rows.sort(
            key=lambda row: int(row["num_atoms"])
        )

        if not backend_rows:
            continue

        atom_counts = [
            int(row["num_atoms"])
            for row in backend_rows
        ]
        times_ms = [
            float(row["wall_mean_ms"])
            for row in backend_rows
        ]

        plt.plot(
            atom_counts,
            times_ms,
            marker="o",
            label=backend,
        )

    plt.xlabel("Number of Cu atoms")
    plt.ylabel("Energy + force time (ms)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot saved to: {output_path}")


# ============================================================
# Backend runner
# ============================================================

def run_backend(
    backend: str,
    args: argparse.Namespace,
    rows: List[Dict],
) -> None:
    print()
    print("=" * 92)
    print(f"Backend: {backend}")
    print("=" * 92)

    calculator = make_calculator(
        model=args.model,
        modal=args.modal,
        backend=backend,
        device=args.device,
    )

    # --------------------------------------------------------
    # Global warmup
    #
    # 用小体系排除 CUDA context、模型初始化、后端初始化和
    # 首次 kernel/JIT 开销。
    # --------------------------------------------------------
    global_atoms = build_cu_supercell(
        replica=2,
        lattice_constant=args.lattice_constant,
        rattle_std=args.rattle_std,
        seed=args.seed,
    )
    global_atoms.calc = calculator

    print(
        f"Global warmup: {args.global_warmup} iterations, "
        f"{len(global_atoms)} Cu atoms"
    )

    for _ in range(args.global_warmup):
        clear_ase_cache(global_atoms)
        _ = singlepoint_forward_backward(global_atoms)

    synchronize(args.device)

    del global_atoms
    clear_cuda_memory(args.device)

    # --------------------------------------------------------
    # Atom-count scan
    # --------------------------------------------------------
    for replica in args.replicas:
        atoms = build_cu_supercell(
            replica=replica,
            lattice_constant=args.lattice_constant,
            rattle_std=args.rattle_std,
            seed=args.seed + replica,
        )
        atoms.calc = calculator

        num_atoms = len(atoms)

        print()
        print(
            f"replica={replica:2d} | "
            f"Cu atoms={num_atoms:,d}"
        )

        try:
            stats = benchmark_atoms(
                atoms=atoms,
                warmup=args.warmup,
                iters=args.iters,
                device=args.device,
                label=backend,
            )

            row = {
                "status": "OK",
                "backend": backend,
                "model": args.model,
                "modal": (
                    args.modal
                    if args.modal is not None
                    else ""
                ),
                "replica": replica,
                "num_atoms": num_atoms,
                "warmup": args.warmup,
                "iters": args.iters,
                **stats,
            }

            rows.append(row)
            write_csv(Path(args.output), rows)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            message = str(error)
            is_oom = (
                isinstance(
                    error,
                    torch.cuda.OutOfMemoryError,
                )
                or "out of memory" in message.lower()
            )

            if not is_oom:
                raise

            print()
            print(
                f"OOM: backend={backend}, "
                f"atoms={num_atoms:,d}"
            )
            print(message)

            row = {
                "status": "OOM",
                "backend": backend,
                "model": args.model,
                "modal": (
                    args.modal
                    if args.modal is not None
                    else ""
                ),
                "replica": replica,
                "num_atoms": num_atoms,
                "warmup": args.warmup,
                "iters": args.iters,
                "wall_mean_ms": "",
                "wall_p50_ms": "",
                "wall_p90_ms": "",
                "wall_min_ms": "",
                "wall_max_ms": "",
                "gpu_mean_ms": "",
                "gpu_p50_ms": "",
                "gpu_p90_ms": "",
                "atoms_per_second": "",
                "peak_allocated_gib": "",
                "peak_reserved_gib": "",
            }

            rows.append(row)
            write_csv(Path(args.output), rows)

            del atoms
            clear_cuda_memory(args.device)

            # 固定密度下，更大的体系通常仍会 OOM。
            break

        del atoms
        clear_cuda_memory(args.device)

    del calculator
    clear_cuda_memory(args.device)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SevenNet e3nn, cuEquivariance, and FlashTP "
            "on FCC Cu systems."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=(
            "/home/malixian/repos/SevenNet/models/"
            "checkpoint_sevennet_omni.pth"
        ),
        help=(
            "SevenNet checkpoint path or pretrained model name. "
            "For the FlashTP paper setting, use 7net-l3i5."
        ),
    )

    parser.add_argument(
        "--modal",
        type=str,
        default="mpa",
        help=(
            "Model modality. Use 'none' for a non-multimodal "
            "model such as 7net-l3i5."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        choices=["e3nn", "cueq", "flashtp"],
        default=["e3nn", "cueq", "flashtp"],
    )

    parser.add_argument(
        "--replicas",
        type=int,
        nargs="+",
        default=[
            5,   # 500
            6,   # 864
            7,   # 1,372
            8,   # 2,048
            9,   # 2,916
            10,  # 4,000
            12,  # 6,912
            14,  # 10,976
            16,  # 16,384
            18,  # 23,328
            19,  # 27,436 ~= 28K
            20,  # 32,000
            21,
            22,

        ],
        help="FCC cubic supercell replication factors.",
    )

    parser.add_argument(
        "--lattice-constant",
        type=float,
        default=3.615,
        help="FCC Cu lattice constant in Angstrom.",
    )

    parser.add_argument(
        "--rattle-std",
        type=float,
        default=0.01,
        help="Position perturbation standard deviation in Angstrom.",
    )

    parser.add_argument(
        "--global-warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="e3nn_cueq_flashtp_cu.csv",
    )

    parser.add_argument(
        "--plot",
        type=str,
        default="e3nn_cueq_flashtp_cu.png",
    )

    args = parser.parse_args()

    if args.modal.lower() in {
        "none",
        "null",
        "",
    }:
        args.modal = None

    return args


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This benchmark requires a GPU."
        )

    torch.set_grad_enabled(True)

    print("=" * 92)
    print("SevenNet e3nn versus cuEq versus FlashTP: FCC Cu benchmark")
    print("=" * 92)
    print(f"torch       : {torch.__version__}")
    print(f"torch CUDA  : {torch.version.cuda}")
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"SevenNet    : {sevenn.__version__}")
    print(f"model       : {args.model}")
    print(f"modal       : {args.modal}")
    print(f"backends    : {args.backends}")
    print(f"replicas    : {args.replicas}")
    print(f"warmup      : {args.warmup}")
    print(f"iterations  : {args.iters}")
    print()

    print("Atom counts:")
    for replica in args.replicas:
        print(
            f"  replica={replica:2d}: "
            f"{4 * replica**3:,d} Cu atoms"
        )

    rows: List[Dict] = []

    for backend in args.backends:
        run_backend(
            backend=backend,
            args=args,
            rows=rows,
        )

    output_path = Path(args.output)
    write_csv(output_path, rows)

    print_speedup_table(rows)

    if args.plot:
        plot_results(
            rows=rows,
            output_path=Path(args.plot),
        )

    print()
    print(f"CSV saved to:  {output_path.resolve()}")


if __name__ == "__main__":
    main()

