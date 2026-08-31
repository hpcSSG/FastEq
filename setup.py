import os
import sys
import sysconfig
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from enum import Enum, auto

class Backend(Enum):
    """Supported compute backends."""
    CUDA = auto()
    HIP = auto()
    SYCL = auto()


def detect_backend() -> Backend:
    """
    Detect the available compute backend.
    Priority: Environment variable > Auto-detection
    
    Environment variable FASTEQ_BACKEND can be set to:
    - 'cuda' : Force CUDA backend
    - 'hip'  : Force HIP/ROCm backend
    - 'sycl' : Force SYCL/oneAPI backend
    """
    # Check environment variable first
    env_backend = os.environ.get("FASTEQ_BACKEND", "").strip().lower()
    if env_backend:
        backend_map = {
            "cuda": Backend.CUDA,
            "hip": Backend.HIP,
            "rocm": Backend.HIP,
            "sycl": Backend.SYCL,
            "oneapi": Backend.SYCL,
        }
        if env_backend in backend_map:
            print(f"[FastEq] Using backend from environment: {env_backend.upper()}")
            return backend_map[env_backend]
        else:
            print(f"[FastEq] Warning: Unknown backend '{env_backend}', auto-detecting...")
    
    # Auto-detection
    # 1. Check for CUDA (nvcc)
    if shutil.which("nvcc"):
        print("[FastEq] Detected CUDA (nvcc found)")
        return Backend.CUDA
    
    # 2. Check for HIP/ROCm (hipcc)
    if shutil.which("hipcc"):
        print("[FastEq] Detected HIP/ROCm (hipcc found)")
        return Backend.HIP
    
    # 3. Check for SYCL/oneAPI (icpx or dpcpp)
    if shutil.which("icpx") or shutil.which("dpcpp"):
        print("[FastEq] Detected SYCL/oneAPI (icpx/dpcpp found)")
        return Backend.SYCL
    
    # 4. Check environment hints
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        print("[FastEq] Detected CUDA from environment variables")
        return Backend.CUDA
    
    if os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"):
        print("[FastEq] Detected HIP/ROCm from environment variables")
        return Backend.HIP
    
    if os.environ.get("ONEAPI_ROOT") or os.environ.get("CMPLR_ROOT"):
        print("[FastEq] Detected SYCL/oneAPI from environment variables")
        return Backend.SYCL
    
    # Default to CUDA
    print("[FastEq] No backend detected, defaulting to CUDA")
    return Backend.CUDA


def get_backend_config(backend: Backend) -> dict:
    """Get configuration for the specified backend."""

    rocm_path = ""
    if backend == Backend.HIP:
        # Ensure ROCM_PATH is set
        if "ROCM_PATH" in os.environ:
            rocm_path = os.environ["ROCM_PATH"]
        else :
            rocm_path = "/opt/rocm"
        print(f"[FastEq] Using ROCm path: {rocm_path}")

        
    configs = {
        Backend.CUDA: {
            "name": "cuda",
            "source_dir": "fasteq/cuda",
            "lib_prefix": "_cuda",
            "description": "FastEq CUDA extensions",
            "arch_env": "FASTEQ_CUDA_ARCH",
            "arch_cmake_var": "CMAKE_CUDA_ARCHITECTURES",
            "extra_cmake_args": [],
        },
        Backend.HIP: {
            "name": "hip",
            "source_dir": "fasteq/hip",
            "lib_prefix": "_hip",
            "description": "FastEq HIP/ROCm extensions",
            "arch_env": "FASTEQ_HIP_ARCH",
            "arch_cmake_var": "CMAKE_HIP_ARCHITECTURES",
            "extra_cmake_args": [
                "-DCMAKE_CXX_COMPILER=hipcc",
                # f"-DROCM_PATH = {rocm_path}",
            ],
        },
        Backend.SYCL: {
            "name": "sycl",
            "source_dir": "fasteq/sycl",
            "lib_prefix": "_sycl",
            "description": "FastEq SYCL/oneAPI extensions",
            "arch_env": "FASTEQ_SYCL_ARCH",
            "arch_cmake_var": "SYCL_TARGETS",
            "extra_cmake_args": [
                "-DCMAKE_CXX_COMPILER=icpx",
            ],
        },
    }
    return configs[backend]


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str, backend: Backend):
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())
        self.backend = backend


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except Exception as e:
            raise RuntimeError("CMake is required to build this project.") from e
        super().run()

    def build_extension(self, ext: CMakeExtension):
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent
        extdir.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"
        build_temp = Path(self.build_temp) / ext.name.replace(".", "_")
        build_temp.mkdir(parents=True, exist_ok=True)

        python_exe = sys.executable
        backend_config = get_backend_config(ext.backend)

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython3_EXECUTABLE={python_exe}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
        ]

        # Add backend-specific cmake args
        cmake_args.extend(backend_config["extra_cmake_args"])

        # Handle architecture settings
        arch_env = backend_config["arch_env"]
        arch_cmake_var = backend_config["arch_cmake_var"]
        arch = os.environ.get(arch_env, "").strip()
        if arch:
            cmake_args.append(f"-D{arch_cmake_var}={arch}")

        # Generator
        if os.environ.get("CMAKE_GENERATOR", "").strip() == "":
            # Use Ninja if available
            if shutil.which("ninja"):
                cmake_args += ["-GNinja"]

        build_args = ["--config", cfg]
        
        # Parallel build
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "")
        if not jobs:
            if hasattr(self, "parallel") and self.parallel:
                jobs = str(self.parallel)
        if jobs:
            build_args += ["-j", jobs]
        else:
            build_args += ["-j", "255"]

        # Configure
        print(f"[FastEq] Configuring {backend_config['name'].upper()} build...")
        print("CMake args:", " ".join(cmake_args))
        subprocess.check_call(
            ["cmake", ext.sourcedir,  # 显式指定使用系统的错误输出
              *cmake_args],
            cwd=str(build_temp),
        )

        
        # Build
        print(f"[FastEq] Building {backend_config['name'].upper()} extension...")
        subprocess.check_call(
            ["cmake", "--build", ".", *build_args],
            cwd=str(build_temp),
        )

        # Find and copy built library
        expected = ext_fullpath
        expected_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

        lib_prefix = backend_config["lib_prefix"]
        candidates = []
        for pat in [f"{lib_prefix}.so", f"{lib_prefix}.pyd", f"{lib_prefix}.dylib"]:
            p = extdir / pat
            if p.exists():
                candidates.append(p)
        if not candidates:
            candidates = list(extdir.glob(f"{lib_prefix}*"))

        if not candidates:
            raise RuntimeError(f"Cannot find built library in {extdir}")

        built = max(candidates, key=lambda p: p.stat().st_mtime)

        expected.parent.mkdir(parents=True, exist_ok=True)
        if built.resolve() != expected.resolve():
            self.copy_file(str(built), str(expected))

        if expected.suffix != Path(expected_suffix).suffix and not expected.exists():
            alt = expected.with_suffix(Path(expected_suffix).suffix)
            if built.exists():
                self.copy_file(str(built), str(alt))


def get_ext_modules():
    """Get extension modules based on detected backend."""
    backend = detect_backend()
    config = get_backend_config(backend)
    
    ROOT = Path(__file__).resolve().parent
    source_dir = ROOT / config["source_dir"]
    
    if not source_dir.exists():
        raise RuntimeError(
            f"Source directory for {config['name'].upper()} backend not found: {source_dir}\n"
            f"Please ensure the {config['name']} implementation exists."
        )
    
    ext_name = f"fasteq.{config['name']}.{config['lib_prefix']}"
    
    return [
        CMakeExtension(ext_name, sourcedir=str(source_dir), backend=backend)
    ], config


def main():
    ext_modules, config = get_ext_modules()
    
    setup(
        name="fasteq",
        version="0.3.0",
        description=config["description"],
        python_requires=">=3.10",
        packages=find_packages(where="."),
        include_package_data=True,
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
