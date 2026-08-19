#!/usr/bin/env python3
import argparse
import csv
import gc
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator

from torch.profiler import profile, record_function, ProfilerActivity

from torch.profiler import profile, record_function, ProfilerActivity
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
# ============================================================
# H2O box generation
# ============================================================

AVOGADRO = 6.02214076e23
WATER_MOLAR_MASS = 18.01528  # g/mol


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a uniformly distributed 3D rotation matrix."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)

    w, x, y, z = q

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def water_lattice_spacing(density_g_cm3: float) -> float:
    """
    Compute the cubic volume assigned to each H2O molecule at a given density.

    Returns
    -------
    spacing : float
        Cubic lattice spacing in Angstrom.
    """
    volume_cm3_per_molecule = (
        WATER_MOLAR_MASS / density_g_cm3 / AVOGADRO
    )

    # 1 cm^3 = 1e24 Angstrom^3
    volume_ang3_per_molecule = volume_cm3_per_molecule * 1.0e24

    return volume_ang3_per_molecule ** (1.0 / 3.0)


def build_water_box(
    n_side: int,
    density_g_cm3: float = 1.0,
    random_orientation: bool = True,
    seed: int = 1234,
) -> Atoms:
    """
    Construct an n_side x n_side x n_side periodic H2O box.

    The number of atoms is exactly:

        N_atoms = 3 * n_side^3

    Each lattice site contains one H2O molecule.
    """
    if n_side <= 0:
        raise ValueError("n_side must be positive")

    water = molecule("H2O")

    # Place oxygen at the local origin.
    relative_positions = (
        water.get_positions() - water.get_positions()[0]
    )
    water_symbols = water.get_chemical_symbols()

    spacing = water_lattice_spacing(density_g_cm3)
    box_length = n_side * spacing

    rng = np.random.default_rng(seed)

    all_positions: List[np.ndarray] = []
    all_symbols: List[str] = []

    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                oxygen_position = np.array(
                    [
                        (ix + 0.5) * spacing,
                        (iy + 0.5) * spacing,
                        (iz + 0.5) * spacing,
                    ],
                    dtype=np.float64,
                )

                local_positions = relative_positions.copy()

                if random_orientation:
                    rotation = random_rotation_matrix(rng)
                    local_positions = local_positions @ rotation.T

                all_positions.append(
                    local_positions + oxygen_position
                )
                all_symbols.extend(water_symbols)

    positions = np.concatenate(all_positions, axis=0)

    atoms = Atoms(
        symbols=all_symbols,
        positions=positions,
        cell=[box_length, box_length, box_length],
        pbc=True,
    )
    atoms.wrap()

    return atoms


# ============================================================
# MACE helpers
# ============================================================

def cuda_synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_memory_stats(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_cuda_memory_gb(device: str) -> float:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return float("nan")

    return torch.cuda.max_memory_allocated() / (1024**3)


def clear_memory(device: str) -> None:
    gc.collect()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_cutoff(calculator: MACECalculator) -> Optional[float]:
    """Try to obtain r_max from the loaded MACE model."""
    try:
        model = calculator.models[0]
        r_max = model.r_max

        if isinstance(r_max, torch.Tensor):
            return float(r_max.detach().cpu().item())

        return float(r_max)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def count_directed_edges(
    atoms: Atoms,
    cutoff: Optional[float],
) -> Optional[int]:
    """
    Count ASE neighbor-list entries.

    The value is reported as directed edges because neighbor-list entries
    generally include both i->j and j->i.
    """
    if cutoff is None:
        return None

    i, _ = neighbor_list(
        "ij",
        atoms,
        cutoff=cutoff,
        self_interaction=False,
    )
    return int(len(i))


def run_force_inference(
    atoms: Atoms,
    calculator: MACECalculator,
) -> np.ndarray:
    """
    Run one actual MACE calculation.

    calculator.reset() is essential. Without it, ASE may return cached forces
    because the atomic positions have not changed.
    """
    calculator.reset()
    atoms.calc = calculator

    return atoms.get_forces()


# ============================================================
# Benchmark
# ============================================================

def benchmark_one_size(
    atoms: Atoms,
    calculator: MACECalculator,
    device: str,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    if repeats <= 0:
        raise ValueError("repeats must be positive")

    # --------------------------------------------------------
    # Per-size warmup
    # --------------------------------------------------------
    for _ in range(warmup):
        _ = run_force_inference(atoms, calculator)

    cuda_synchronize(device)

    # Start memory statistics after warmup.
    reset_cuda_memory_stats(device)

    times_ms: List[float] = []

    # --------------------------------------------------------
    # Timed iterations
    # --------------------------------------------------------
    for  idx in range(repeats):
        calculator.reset()
        atoms.calc = calculator

        cuda_synchronize(device)
        start = time.perf_counter()
        
        """
        if idx == repeats-1:
            with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
                _ = atoms.get_forces()
            prof.export_chrome_trace("mace-bench-trace.json")
        else:
            _ = atoms.get_forces()
        """
        _ = atoms.get_forces()

        cuda_synchronize(device)
        end = time.perf_counter()

        times_ms.append((end - start) * 1000.0)

    

    times = np.asarray(times_ms, dtype=np.float64)

    mean_ms = float(np.mean(times))
    median_ms = float(np.median(times))
    std_ms = float(np.std(times))
    p10_ms = float(np.percentile(times, 10))
    p90_ms = float(np.percentile(times, 90))

    num_atoms = len(atoms)

    atoms_per_second = (
        num_atoms / (median_ms / 1000.0)
        if median_ms > 0.0
        else float("inf")
    )

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "p10_ms": p10_ms,
        "p90_ms": p90_ms,
        "atoms_per_second": atoms_per_second,
        "peak_memory_gb": peak_cuda_memory_gb(device),
    }


def write_results(
    output_path: Path,
    results: List[Dict],
) -> None:
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MACE-OFF23 on periodic H2O boxes."
    )

    parser.add_argument(
        "--mace-scale",
        type=str,
        default="large",
        choices=["small", "medium", "large"],
        help="MACE-OFF23 model scale.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="../../mace_bench/models",
        help="Directory containing MACE-OFF23 model files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations for every atom count.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Timed iterations for every atom count.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Nominal water density in g/cm^3.",
    )
    parser.add_argument(
        "--box-sides",
        type=int,
        nargs="+",
        #default=[4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40],
        default=[10],
        help=(
            "Numbers of water molecules along each box dimension. "
            "N_atoms = 3 * n_side^3."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mace_off23_h2o_benchmark.csv",
    )
    parser.add_argument(
        "--disable-random-orientation",
        action="store_true",
        help="Keep all water molecules in the same orientation.",
    )
    parser.add_argument(
        "--disable-edge-count",
        action="store_true",
        help="Skip the ASE neighbor-list edge count.",
    )

    args = parser.parse_args()

    model_path = (
        Path(args.model_dir)
        / f"MACE-OFF23_{args.mace_scale}.model"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}"
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested, but torch.cuda.is_available() is False"
        )

    print("=" * 80)
    print("MACE-OFF23 H2O benchmark")
    print("=" * 80)
    print(f"Model:      {model_path}")
    print(f"Scale:      {args.mace_scale}")
    print(f"Device:     {args.device}")
    print(f"Dtype:      {args.dtype}")
    print(f"Warmup:     {args.warmup}")
    print(f"Repeats:    {args.repeats}")
    print(f"Density:    {args.density:.3f} g/cm^3")
    print(f"Box sides:  {args.box_sides}")
    print()

    # ========================================================
    # Calculator initialization
    # ========================================================

    calculator = MACECalculator(
        model_paths=str(model_path),
        device=args.device,
        default_dtype=args.dtype,
        compile_mode=None,
        enable_cueq=True,
    )

    cutoff = get_model_cutoff(calculator)

    if cutoff is not None:
        print(f"Model cutoff: {cutoff:.3f} Angstrom")
    else:
        print("Model cutoff: unavailable")

    # ========================================================
    # Global warmup
    #
    # This removes one-time initialization costs, including
    # lazy CUDA setup and backend initialization, from the
    # per-size benchmark.
    # ========================================================

    print("\nRunning global model warmup...")

    global_warmup_atoms = build_water_box(
        n_side=3,
        density_g_cm3=args.density,
        random_orientation=not args.disable_random_orientation,
        seed=1234,
    )

    for _ in range(max(3, args.warmup)):
        _ = run_force_inference(
            global_warmup_atoms,
            calculator,
        )

    cuda_synchronize(args.device)

    del global_warmup_atoms
    clear_memory(args.device)

    print("Global warmup finished.\n")

    all_results: List[Dict] = []
    output_path = Path(args.output)

    # ========================================================
    # Different atom counts
    # ========================================================

    for n_side in args.box_sides:
        atoms = build_water_box(
            n_side=n_side,
            density_g_cm3=args.density,
            random_orientation=not args.disable_random_orientation,
            seed=1234 + n_side,
        )

        num_molecules = n_side**3
        num_atoms = len(atoms)
        box_length = float(atoms.cell.lengths()[0])

        if args.disable_edge_count:
            num_edges = None
        else:
            num_edges = count_directed_edges(atoms, cutoff)

        edge_text = (
            str(num_edges)
            if num_edges is not None
            else "N/A"
        )

        print("-" * 80)
        print(
            f"n_side={n_side}, "
            f"molecules={num_molecules}, "
            f"atoms={num_atoms}, "
            f"edges={edge_text}, "
            f"box={box_length:.2f} Angstrom"
        )

        try:
            stats = benchmark_one_size(
                atoms=atoms,
                calculator=calculator,
                device=args.device,
                warmup=args.warmup,
                repeats=args.repeats,
            )

            result = {
                "status": "OK",
                "mace_scale": args.mace_scale,
                "device": args.device,
                "dtype": args.dtype,
                "n_side": n_side,
                "num_molecules": num_molecules,
                "num_atoms": num_atoms,
                "num_edges": (
                    num_edges if num_edges is not None else ""
                ),
                "box_length_angstrom": box_length,
                "density_g_cm3": args.density,
                "warmup": args.warmup,
                "repeats": args.repeats,
                **stats,
            }

            all_results.append(result)
            write_results(output_path, all_results)

            print(
                f"Mean:       {stats['mean_ms']:.3f} ms"
            )
            print(
                f"Median:     {stats['median_ms']:.3f} ms"
            )
            print(
                f"P10--P90:   "
                f"{stats['p10_ms']:.3f}--"
                f"{stats['p90_ms']:.3f} ms"
            )
            print(
                f"Throughput: "
                f"{stats['atoms_per_second']:,.0f} atom/s"
            )
            print(
                f"Peak memory:"
                f" {stats['peak_memory_gb']:.3f} GiB"
            )

        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            error_text = str(error)

            is_oom = (
                isinstance(error, torch.cuda.OutOfMemoryError)
                or "out of memory" in error_text.lower()
            )

            if not is_oom:
                raise

            print(f"OOM at {num_atoms} atoms: {error_text}")

            result = {
                "status": "OOM",
                "mace_scale": args.mace_scale,
                "device": args.device,
                "dtype": args.dtype,
                "n_side": n_side,
                "num_molecules": num_molecules,
                "num_atoms": num_atoms,
                "num_edges": (
                    num_edges if num_edges is not None else ""
                ),
                "box_length_angstrom": box_length,
                "density_g_cm3": args.density,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "mean_ms": "",
                "median_ms": "",
                "std_ms": "",
                "p10_ms": "",
                "p90_ms": "",
                "atoms_per_second": "",
                "peak_memory_gb": "",
            }

            all_results.append(result)
            write_results(output_path, all_results)

            del atoms
            clear_memory(args.device)

            # At fixed density, larger systems will generally require
            # more memory, so stop after the first OOM.
            break

        del atoms
        clear_memory(args.device)

    print("\n" + "=" * 80)
    print(f"Results written to: {output_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
