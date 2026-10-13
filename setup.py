import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from enum import Enum, auto
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent


class Backend(Enum):
    """Supported compute backends."""

    NONE = auto()
    CUDA = auto()
    HIP = auto()
    SYCL = auto()


def detect_backend() -> Backend:
    """Default to a pure Python build; enable detection explicitly with auto."""
    value = os.environ.get("FASTEQ_BACKEND", "none").strip().lower()
    aliases = {
        "": Backend.NONE,
        "none": Backend.NONE,
        "cuda": Backend.CUDA,
        "hip": Backend.HIP,
        "rocm": Backend.HIP,
        "sycl": Backend.SYCL,
        "oneapi": Backend.SYCL,
    }
    if value in aliases:
        backend = aliases[value]
    elif value == "auto":
        backend = Backend.NONE
        checks = (
            (Backend.CUDA, ("nvcc",), ("CUDA_HOME", "CUDA_PATH")),
            (Backend.HIP, ("hipcc",), ("ROCM_PATH", "ROCM_HOME", "HIP_PATH")),
            (Backend.SYCL, ("icpx", "dpcpp"), ("ONEAPI_ROOT", "CMPLR_ROOT")),
        )
        # Prefer available compiler executables over environment hints.
        for candidate, compilers, _ in checks:
            if any(shutil.which(compiler) for compiler in compilers):
                backend = candidate
                break
        else:
            for candidate, _, hints in checks:
                if any(os.environ.get(hint, "").strip() for hint in hints):
                    backend = candidate
                    break
    else:
        raise ValueError(
            f"Unsupported FASTEQ_BACKEND={value!r}. "
            "Expected none, auto, cuda, hip, rocm, sycl, or oneapi."
        )

    print(f"[FastEq] Backend: {backend.name.lower()}")
    if backend == Backend.NONE:
        print("[FastEq] Skipping extension compilation.")
    return backend


def get_backend_config(backend: Backend) -> dict:
    if backend == Backend.NONE:
        return {"description": "FastEq (Pure Python, no extensions compiled)"}
    name = backend.name.lower()
    return {
        "name": name,
        "source_dir": f"fasteq/{name}",
        "lib_prefix": f"_{name}",
        "description": f"FastEq {backend.name} extensions",
        "arch_env": f"FASTEQ_{backend.name}_ARCH",
        "arch_cmake_var": {
            Backend.CUDA: "CMAKE_CUDA_ARCHITECTURES",
            Backend.HIP: "CMAKE_HIP_ARCHITECTURES",
            Backend.SYCL: "SYCL_TARGETS",
        }[backend],
    }


def default_jobs() -> int:
    """Use process CPU affinity when available, with a conservative cap."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return max(1, min(cpus, 8))


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: Path, backend: Backend):
        super().__init__(name, sources=[])
        self.sourcedir = str(sourcedir.resolve())
        self.backend = backend


class CMakeBuild(build_ext):
    def run(self):
        self.cmake = shutil.which("cmake")
        if not self.cmake:
            raise RuntimeError("CMake is required to build FastEq extensions.")
        super().run()

    def build_extension(self, ext: CMakeExtension):
        config = get_backend_config(ext.backend)
        cfg = "Debug" if self.debug else "Release"
        expected = Path(self.get_ext_fullpath(ext.name)).resolve()
        expected.parent.mkdir(parents=True, exist_ok=True)

        # Separate build caches by extension, Python ABI, and configuration.
        abi = sys.implementation.cache_tag or "python"
        build_dir = (
            Path(self.build_temp)
            / f"{ext.name.replace('.', '_')}-{abi}-{cfg.lower()}"
        ).resolve()
        output_dir = build_dir / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)

        extra_args = shlex.split(os.environ.get("CMAKE_ARGS", ""))

        def has_definition(name):
            return any(
                arg.startswith(f"-D{name}=") or arg.startswith(f"-D{name}:")
                for arg in extra_args
            )

        defaults = []
        if ext.backend == Backend.HIP:
            rocm_value = next(
                (os.environ[key].strip() for key in
                 ("ROCM_PATH", "ROCM_HOME", "HIP_PATH")
                 if os.environ.get(key, "").strip()),
                "",
            )
            if rocm_value:
                rocm = Path(rocm_value).expanduser().resolve()
                if not rocm.is_dir():
                    raise RuntimeError(f"ROCm/DTK directory does not exist: {rocm}")
                defaults.append(f"-DFASTEQ_ROCM_ROOT={rocm}")
                # Native CMake HIP language requires Clang, not hipcc.
                # Respect user compiler and toolchain selections.
                if (not has_definition("CMAKE_TOOLCHAIN_FILE")
                        and not os.environ.get("CMAKE_TOOLCHAIN_FILE")):
                    clang = next(
                        (path for path in (rocm / "llvm/bin/clang++",
                                          rocm / "bin/clang++")
                         if path.is_file() and os.access(path, os.X_OK)),
                        None,
                    )
                    if clang:
                        if (not has_definition("CMAKE_HIP_COMPILER")
                                and not os.environ.get("HIPCXX")):
                            defaults.append(f"-DCMAKE_HIP_COMPILER={clang}")
                        if (not has_definition("CMAKE_CXX_COMPILER")
                                and not os.environ.get("CXX")):
                            defaults.append(f"-DCMAKE_CXX_COMPILER={clang}")
                print(f"[FastEq] ROCm/DTK root: {rocm}")
        elif ext.backend == Backend.SYCL:
            if not has_definition("CMAKE_CXX_COMPILER") and not os.environ.get("CXX"):
                compiler = shutil.which("icpx") or shutil.which("dpcpp")
                if not compiler:
                    raise RuntimeError("SYCL requires icpx or dpcpp, or an explicit compiler.")
                defaults.append(f"-DCMAKE_CXX_COMPILER={compiler}")

        generator = []
        if (not (build_dir / "CMakeCache.txt").exists()
                and not os.environ.get("CMAKE_GENERATOR", "").strip()
                and not any(arg == "-G" or arg.startswith("-G") for arg in extra_args)
                and shutil.which("ninja")):
            generator = ["-G", "Ninja"]

        cmake_args = [*generator, f"-DCMAKE_BUILD_TYPE={cfg}", *defaults, *extra_args]
        arch = os.environ.get(config["arch_env"], "").strip()
        if arch:
            # FASTEQ_*_ARCH takes precedence over CMAKE_ARGS.
            cmake_args.append(f"-D{config['arch_cmake_var']}={arch}")

        # These settings must match setuptools and the active Python interpreter.
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if not suffix:
            raise RuntimeError("Python EXT_SUFFIX is unavailable.")
        cmake_args += [
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={output_dir}",
        ]
        if ext.backend == Backend.HIP:
            cmake_args += [
                f"-DFASTEQ_EXTENSION_NAME={config['lib_prefix']}",
                f"-DFASTEQ_EXTENSION_SUFFIX={suffix}",
            ]

        cmake = getattr(self, "cmake", None) or shutil.which("cmake")
        if not cmake:
            raise RuntimeError("CMake is required to build FastEq extensions.")
        print(f"[FastEq] Configuring {config['name'].upper()} ({cfg})")
        subprocess.check_call([
            cmake, "-S", ext.sourcedir, "-B", str(build_dir), *cmake_args
        ])

        # Remove only known output files to prevent accepting a stale artifact.
        names = [expected.name]
        if ext.backend != Backend.HIP:
            # Compatibility with existing CUDA/SYCL CMake files.
            names.extend(f"{config['lib_prefix']}{ending}"
                         for ending in (".so", ".pyd", ".dylib"))
        candidates = list(dict.fromkeys(output_dir / name for name in names))
        for path in candidates:
            if path.is_file():
                path.unlink()

        build_args = [cmake, "--build", str(build_dir), "--config", cfg]
        if ext.backend == Backend.HIP:
            build_args += ["--target", "_hip"]
        # Leave CMake's environment-controlled parallelism intact.
        if not os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "").strip():
            build_args += ["--parallel", str(self.parallel or default_jobs())]
        print(f"[FastEq] Building {config['name'].upper()} extension")
        subprocess.check_call(build_args)

        built = next((path for path in candidates if path.is_file()), None)
        if built is None:
            raise RuntimeError(
                "Expected extension output was not produced. Checked: "
                + ", ".join(str(path) for path in candidates)
            )
        # Copy the artifact produced by this build regardless of timestamps.
        shutil.copy2(built, expected)


def get_ext_modules():
    backend = detect_backend()
    config = get_backend_config(backend)
    if backend == Backend.NONE:
        return [], config
    source_dir = ROOT / config["source_dir"]
    if not (source_dir / "CMakeLists.txt").is_file():
        raise RuntimeError(f"Backend CMakeLists.txt not found: {source_dir}")
    name = f"fasteq.{config['name']}.{config['lib_prefix']}"
    return [CMakeExtension(name, source_dir, backend)], config


def main():
    extensions, config = get_ext_modules()
    kwargs = dict(
        name="fasteq",
        version="0.3.0",
        description=config["description"],
        python_requires=">=3.10",
        packages=find_packages(where="."),
        include_package_data=True,
        zip_safe=False,
    )
    if extensions:
        kwargs.update(ext_modules=extensions, cmdclass={"build_ext": CMakeBuild})
    setup(**kwargs)


if __name__ == "__main__":
    main()
