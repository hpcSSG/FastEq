import os
import sys
import sysconfig
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str):
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


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

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython3_EXECUTABLE={python_exe}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
        ]

        cuda_arch = os.environ.get("FASTEQ_CUDA_ARCH", "").strip()
        if cuda_arch:
            cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")

        if os.environ.get("CMAKE_GENERATOR", "").strip() == "":
            cmake_args += ["-GNinja"]

        build_args = ["--config", cfg, "-j", "10"]
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "")
        if not jobs:
            # setuptools 的 -j
            if hasattr(self, "parallel") and self.parallel:
                jobs = str(self.parallel)
        if jobs:
            build_args += ["-j", jobs]

        # Configure
        subprocess.check_call(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=str(build_temp),
        )
        # Build
        subprocess.check_call(
            ["cmake", "--build", ".", *build_args],
            cwd=str(build_temp),
        )

        expected = ext_fullpath
        expected_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

        candidates = []
        for pat in ["_cuda.so", "_cuda.pyd", "_cuda.dylib"]:
            p = extdir / pat
            if p.exists():
                candidates.append(p)
        if not candidates:
            candidates = list(extdir.glob("_cuda*"))

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


ROOT = Path(__file__).resolve().parent
CMAKE_SOURCE_DIR = ROOT / "fasteq" / "cuda"

setup(
    name="fasteq",
    version="0.1.0",
    description="FastEq CUDA extensions",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    include_package_data=True,
    ext_modules=[
        CMakeExtension("fasteq.cuda._cuda", sourcedir=str(CMAKE_SOURCE_DIR)),
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
