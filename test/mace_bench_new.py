import argparse
import csv
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator
from torch.profiler import ProfilerActivity, profile, record_function


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
    """Return the cubic lattice spacing per H2O molecule in Angstrom."""
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

    Number of atoms:
        N_atoms = 3 * n_side^3
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
# CUDA and MACE helpers
# ============================================================


def is_cuda_device(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def cuda_synchronize(device: str) -> None:
    if is_cuda_device(device):
        torch.cuda.synchronize()


def reset_cuda_memory_stats(device: str) -> None:
    if is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats()


def peak_cuda_memory_gb(device: str) -> float:
    if not is_cuda_device(device):
        return float("nan")

    return torch.cuda.max_memory_allocated() / (1024**3)


def clear_memory(device: str) -> None:
    gc.collect()

    if is_cuda_device(device):
        torch.cuda.empty_cache()


def get_model_cutoff(
    calculator: MACECalculator,
) -> Optional[float]:
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
    """Count ASE neighbor-list entries as directed edges."""
    if cutoff is None:
        return None

    i, _ = neighbor_list(
        "ij",
        atoms,
        cutoff=cutoff,
        self_interaction=False,
    )
    return int(len(i))


def get_batch_keys(batch: Any) -> List[str]:
    """
    Support both torch_geometric variants:
        batch.keys
        batch.keys()
    """
    keys = batch.keys
    if callable(keys):
        keys = keys()
    return list(keys)


def clone_batch(
    calculator: MACECalculator,
    batch_base: Any,
) -> Any:
    """
    Match MACECalculator.calculate() as closely as possible.

    Recent MACE versions provide calculator._clone_batch(). The fallback keeps
    compatibility with versions where that helper is absent.
    """
    clone_fn = getattr(calculator, "_clone_batch", None)

    if callable(clone_fn):
        return clone_fn(batch_base)

    return batch_base.clone()


def convert_batch_dtype(
    batch: Any,
    model: torch.nn.Module,
) -> Any:
    """Convert floating-point batch tensors to the model parameter dtype."""
    model_dtype = next(model.parameters()).dtype

    for key in get_batch_keys(batch):
        value = batch[key]

        if torch.is_tensor(value) and torch.is_floating_point(value):
            if value.dtype != model_dtype:
                batch[key] = value.to(dtype=model_dtype)

    return batch


def prepare_model_input(
    atoms: Atoms,
    calculator: MACECalculator,
) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor], int]:
    """
    Prepare the model input outside the timed region.

    Excluded from model timing:
        - Atoms -> Config
        - neighbor-list/graph construction
        - Batch construction
        - host-to-device transfer
        - batch cloning
        - dtype conversion
        - Batch.to_dict()

    Returns
    -------
    model
        calculator.models[0]
    batch_dict
        GPU-resident dictionary passed directly to model.forward()
    num_graph_edges
        Number of directed graph edges in the prepared MACE batch
    """
    if len(calculator.models) != 1:
        raise ValueError(
            "This model-only benchmark currently requires exactly one model. "
            f"Found {len(calculator.models)} models."
        )

    model = calculator.models[0]
    model.eval()

    # Private MACE helper intentionally used to reproduce Calculator input
    # construction while moving it outside the timed region.
    batch_base = calculator._atoms_to_batch(atoms)
    batch = clone_batch(calculator, batch_base)
    batch = convert_batch_dtype(batch, model)
    batch_dict = batch.to_dict()

    edge_index = batch_dict.get("edge_index")
    num_graph_edges = (
        int(edge_index.shape[1])
        if torch.is_tensor(edge_index) and edge_index.ndim == 2
        else -1
    )

    # Keep batch alive through the tensors stored in batch_dict. Tensors in the
    # dictionary reference their own storage, so batch_base and batch need not
    # be returned.
    return model, batch_dict, num_graph_edges


def run_model_forward(
    model: torch.nn.Module,
    batch_dict: Dict[str, torch.Tensor],
    compute_stress: bool,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Direct MACE model.forward() call.

    Do not wrap this function with torch.no_grad() or inference_mode(), because
    force computation differentiates energy with respect to atomic positions.
    """
    return model(
        batch_dict,
        training=False,
        compute_force=True,
        compute_virials=compute_stress,
        compute_stress=compute_stress,
    )


def validate_model_output(
    output: Dict[str, Optional[torch.Tensor]],
) -> None:
    forces = output.get("forces")

    if forces is None:
        raise RuntimeError(
            "model.forward() did not return forces. "
            "Do not use torch.no_grad() or torch.inference_mode()."
        )


# ============================================================
# Model-only benchmark
# ============================================================


def summarize_times(
    values_ms: List[float],
    prefix: str,
) -> Dict[str, float]:
    values = np.asarray(values_ms, dtype=np.float64)

    return {
        f"{prefix}_mean_ms": float(np.mean(values)),
        f"{prefix}_median_ms": float(np.median(values)),
        f"{prefix}_std_ms": float(np.std(values)),
        f"{prefix}_p10_ms": float(np.percentile(values, 10)),
        f"{prefix}_p90_ms": float(np.percentile(values, 90)),
    }


def benchmark_model_forward(
    atoms: Atoms,
    calculator: MACECalculator,
    device: str,
    warmup: int,
    repeats: int,
    compute_stress: bool,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Benchmark only model.forward().

    The input graph and all GPU-resident input tensors are prepared once before
    warmup and reused for all timed iterations.
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    if repeats <= 0:
        raise ValueError("repeats must be positive")

    # --------------------------------------------------------
    # Input preparation: outside timed region
    # --------------------------------------------------------
    prepare_start = time.perf_counter()

    model, batch_dict, num_graph_edges = prepare_model_input(
        atoms=atoms,
        calculator=calculator,
    )

    cuda_synchronize(device)
    prepare_end = time.perf_counter()

    input_prepare_ms = (prepare_end - prepare_start) * 1000.0

    # --------------------------------------------------------
    # Model-forward warmup
    # --------------------------------------------------------
    for _ in range(warmup):
        output = run_model_forward(
            model=model,
            batch_dict=batch_dict,
            compute_stress=compute_stress,
        )
        validate_model_output(output)
        del output

    cuda_synchronize(device)
    reset_cuda_memory_stats(device)

    wall_times_ms: List[float] = []
    cuda_times_ms: List[float] = []

    # --------------------------------------------------------
    # Timed model.forward iterations
    # --------------------------------------------------------
    for _ in range(repeats):
        cuda_synchronize(device)

        if is_cuda_device(device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            wall_start = time.perf_counter()
            start_event.record()

            output = run_model_forward(
                model=model,
                batch_dict=batch_dict,
                compute_stress=compute_stress,
            )

            end_event.record()
            end_event.synchronize()
            wall_end = time.perf_counter()

            validate_model_output(output)

            cuda_times_ms.append(
                float(start_event.elapsed_time(end_event))
            )
            wall_times_ms.append(
                (wall_end - wall_start) * 1000.0
            )
        else:
            wall_start = time.perf_counter()

            output = run_model_forward(
                model=model,
                batch_dict=batch_dict,
                compute_stress=compute_stress,
            )

            wall_end = time.perf_counter()

            validate_model_output(output)

            wall_ms = (wall_end - wall_start) * 1000.0
            wall_times_ms.append(wall_ms)

            # On CPU there is no CUDA Event time. Use wall time so the CSV
            # remains numeric and directly comparable within a CPU run.
            cuda_times_ms.append(wall_ms)

        del output

    wall_stats = summarize_times(
        wall_times_ms,
        prefix="model_wall",
    )
    cuda_stats = summarize_times(
        cuda_times_ms,
        prefix="model_cuda",
    )

    median_model_ms = cuda_stats["model_cuda_median_ms"]
    num_atoms = len(atoms)

    stats: Dict[str, float] = {
        "input_prepare_ms": input_prepare_ms,
        "num_graph_edges": float(num_graph_edges),
        **cuda_stats,
        **wall_stats,
        "atoms_per_second": (
            num_atoms / (median_model_ms / 1000.0)
            if median_model_ms > 0.0
            else float("inf")
        ),
        "peak_memory_gb": peak_cuda_memory_gb(device),
    }

    return stats, batch_dict


# ============================================================
# Optional model-only profiler
# ============================================================


def profile_model_forward(
    model: torch.nn.Module,
    batch_dict: Dict[str, torch.Tensor],
    device: str,
    compute_stress: bool,
    output_path: Path,
    row_limit: int,
    with_stack: bool,
) -> None:
    """
    Profile one model.forward() call outside benchmark timing.

    A long cudaDeviceSynchronize/cuda_synchronize CPU event means the CPU is
    waiting for previously submitted GPU kernels. Inspect CUDA kernel durations
    and self_cuda_time_total to find the actual GPU bottleneck.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]

    if is_cuda_device(device):
        activities.append(ProfilerActivity.CUDA)

    cuda_synchronize(device)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
    ) as prof:
        with record_function("mace_model_forward_only"):
            output = run_model_forward(
                model=model,
                batch_dict=batch_dict,
                compute_stress=compute_stress,
            )
            validate_model_output(output)

        # Keep the GPU work within the profiler interval.
        cuda_synchronize(device)

    del output

    prof.export_chrome_trace(str(output_path))

    sort_key = (
        "self_cuda_time_total"
        if is_cuda_device(device)
        else "self_cpu_time_total"
    )

    print()
    print(f"Profiler table sorted by {sort_key}:")
    print(
        prof.key_averages().table(
            sort_by=sort_key,
            row_limit=row_limit,
        )
    )
    print(f"Chrome trace written to: {output_path.resolve()}")


# ============================================================
# Results
# ============================================================


def write_results(
    output_path: Path,
    results: List[Dict[str, Any]],
) -> None:
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def make_oom_result(
    args: argparse.Namespace,
    n_side: int,
    num_molecules: int,
    num_atoms: int,
    num_edges: Optional[int],
    box_length: float,
) -> Dict[str, Any]:
    return {
        "status": "OOM",
        "mace_scale": args.mace_scale,
        "device": args.device,
        "dtype": args.dtype,
        "compute_stress": args.compute_stress,
        "n_side": n_side,
        "num_molecules": num_molecules,
        "num_atoms": num_atoms,
        "num_edges_ase": (
            num_edges if num_edges is not None else ""
        ),
        "num_graph_edges": "",
        "box_length_angstrom": box_length,
        "density_g_cm3": args.density,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "input_prepare_ms": "",
        "model_cuda_mean_ms": "",
        "model_cuda_median_ms": "",
        "model_cuda_std_ms": "",
        "model_cuda_p10_ms": "",
        "model_cuda_p90_ms": "",
        "model_wall_mean_ms": "",
        "model_wall_median_ms": "",
        "model_wall_std_ms": "",
        "model_wall_p10_ms": "",
        "model_wall_p90_ms": "",
        "atoms_per_second": "",
        "peak_memory_gb": "",
    }


# ============================================================
# Main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark only MACE-OFF23 model.forward() on periodic H2O boxes. "
            "Atoms-to-batch preprocessing and output conversion are excluded."
        )
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
        default=5,
        help="Model-forward warmup iterations for every atom count.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Timed model-forward iterations for every atom count.",
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
        #default=[4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 42, 44],
        default=[8],
        help=(
            "Numbers of water molecules along each box dimension. "
            "N_atoms = 3 * n_side^3."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mace_off23_model_forward_benchmark.csv",
    )
    parser.add_argument(
        "--disable-random-orientation",
        action="store_true",
        help="Keep all water molecules in the same orientation.",
    )
    parser.add_argument(
        "--disable-edge-count",
        action="store_true",
        help="Skip the independent ASE neighbor-list edge count.",
    )
    parser.add_argument(
        "--compute-stress",
        action="store_true",
        help=(
            "Include stress/virial computation in model.forward(). "
            "By default only energy and forces are computed."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Profile one extra model.forward() call after each benchmark. "
            "The profiled call is not included in timing statistics."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="mace_traces",
        help="Directory for optional Chrome trace JSON files.",
    )
    parser.add_argument(
        "--profile-row-limit",
        type=int,
        default=50,
        help="Number of rows in the profiler summary table.",
    )
    parser.add_argument(
        "--profile-with-stack",
        action="store_true",
        help=(
            "Collect Python stacks in profiler. This adds substantial overhead "
            "but does not affect benchmark timing because profiling is separate."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
            "CUDA was requested, but torch.cuda.is_available() is False"
        )

    print("=" * 88)
    print("MACE-OFF23 model.forward-only benchmark")
    print("=" * 88)
    print(f"Model:           {model_path}")
    print(f"Scale:           {args.mace_scale}")
    print(f"Device:          {args.device}")
    print(f"Dtype:           {args.dtype}")
    print(f"Warmup:          {args.warmup}")
    print(f"Repeats:         {args.repeats}")
    print(f"Density:         {args.density:.3f} g/cm^3")
    print(f"Box sides:       {args.box_sides}")
    print(f"Compute stress:  {args.compute_stress}")
    print(f"Profile:         {args.profile}")
    print()
    print("Timed region:")
    print("  calculator.models[0](batch_dict, compute_force=True, ...)")
    print()
    print("Excluded:")
    print("  atoms_to_batch, graph construction, H2D transfer, dtype conversion,")
    print("  ASE dispatch, detach/cpu/numpy output conversion")
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
    # ========================================================

    print("\nRunning global model warmup...")

    global_warmup_atoms = build_water_box(
        n_side=3,
        density_g_cm3=args.density,
        random_orientation=not args.disable_random_orientation,
        seed=1234,
    )

    global_model, global_batch_dict, _ = prepare_model_input(
        atoms=global_warmup_atoms,
        calculator=calculator,
    )

    for _ in range(max(3, args.warmup)):
        global_output = run_model_forward(
            model=global_model,
            batch_dict=global_batch_dict,
            compute_stress=args.compute_stress,
        )
        validate_model_output(global_output)
        del global_output

    cuda_synchronize(args.device)

    del global_batch_dict
    del global_model
    del global_warmup_atoms
    clear_memory(args.device)

    print("Global warmup finished.\n")

    all_results: List[Dict[str, Any]] = []
    output_path = Path(args.output)
    trace_dir = Path(args.trace_dir)

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

        print("-" * 88)
        print(
            f"n_side={n_side}, "
            f"molecules={num_molecules}, "
            f"atoms={num_atoms}, "
            f"ASE edges={edge_text}, "
            f"box={box_length:.2f} Angstrom"
        )

        try:
            stats, batch_dict = benchmark_model_forward(
                atoms=atoms,
                calculator=calculator,
                device=args.device,
                warmup=args.warmup,
                repeats=args.repeats,
                compute_stress=args.compute_stress,
            )

            graph_edges = int(stats["num_graph_edges"])

            result: Dict[str, Any] = {
                "status": "OK",
                "mace_scale": args.mace_scale,
                "device": args.device,
                "dtype": args.dtype,
                "compute_stress": args.compute_stress,
                "n_side": n_side,
                "num_molecules": num_molecules,
                "num_atoms": num_atoms,
                "num_edges_ase": (
                    num_edges if num_edges is not None else ""
                ),
                "num_graph_edges": graph_edges,
                "box_length_angstrom": box_length,
                "density_g_cm3": args.density,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "input_prepare_ms": stats["input_prepare_ms"],
                "model_cuda_mean_ms": stats["model_cuda_mean_ms"],
                "model_cuda_median_ms": stats["model_cuda_median_ms"],
                "model_cuda_std_ms": stats["model_cuda_std_ms"],
                "model_cuda_p10_ms": stats["model_cuda_p10_ms"],
                "model_cuda_p90_ms": stats["model_cuda_p90_ms"],
                "model_wall_mean_ms": stats["model_wall_mean_ms"],
                "model_wall_median_ms": stats["model_wall_median_ms"],
                "model_wall_std_ms": stats["model_wall_std_ms"],
                "model_wall_p10_ms": stats["model_wall_p10_ms"],
                "model_wall_p90_ms": stats["model_wall_p90_ms"],
                "atoms_per_second": stats["atoms_per_second"],
                "peak_memory_gb": stats["peak_memory_gb"],
            }

            all_results.append(result)
            write_results(output_path, all_results)

            print(
                f"Input preparation:   "
                f"{stats['input_prepare_ms']:.3f} ms "
                f"(excluded from model timing)"
            )
            print(
                f"MACE graph edges:     {graph_edges}"
            )
            print(
                f"Model CUDA mean:      "
                f"{stats['model_cuda_mean_ms']:.3f} ms"
            )
            print(
                f"Model CUDA median:    "
                f"{stats['model_cuda_median_ms']:.3f} ms"
            )
            print(
                f"Model CUDA P10--P90:  "
                f"{stats['model_cuda_p10_ms']:.3f}--"
                f"{stats['model_cuda_p90_ms']:.3f} ms"
            )
            print(
                f"Model wall median:    "
                f"{stats['model_wall_median_ms']:.3f} ms"
            )
            print(
                f"Throughput:           "
                f"{stats['atoms_per_second']:,.0f} atom/s"
            )
            print(
                f"Peak model memory:    "
                f"{stats['peak_memory_gb']:.3f} GiB"
            )

            if args.profile:
                model = calculator.models[0]
                trace_path = (
                    trace_dir
                    / f"mace_model_forward_n{n_side}.json"
                )

                profile_model_forward(
                    model=model,
                    batch_dict=batch_dict,
                    device=args.device,
                    compute_stress=args.compute_stress,
                    output_path=trace_path,
                    row_limit=args.profile_row_limit,
                    with_stack=args.profile_with_stack,
                )

            del batch_dict

        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            error_text = str(error)

            is_oom = (
                isinstance(error, torch.cuda.OutOfMemoryError)
                or "out of memory" in error_text.lower()
            )

            if not is_oom:
                raise

            print(f"OOM at {num_atoms} atoms: {error_text}")

            result = make_oom_result(
                args=args,
                n_side=n_side,
                num_molecules=num_molecules,
                num_atoms=num_atoms,
                num_edges=num_edges,
                box_length=box_length,
            )

            all_results.append(result)
            write_results(output_path, all_results)

            del atoms
            clear_memory(args.device)

            # At fixed density, larger systems generally need more memory.
            break

        del atoms
        clear_memory(args.device)

    print("\n" + "=" * 88)
    print(f"Results written to: {output_path.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()

