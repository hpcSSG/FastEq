"""Version-isolated Luna linear adapter for the shared H20 fixture.

Native layouts and prepared launches are distinct boundaries. The generated
launch allocates its output in C++; it is never labelled allocation-free GEMM.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import shutil
import sys
import tomllib

from common import Evidence


class LunaSource:
    """Load exactly one source tree under Luna's canonical package name."""

    def __init__(self, root: Path | str, version: str, manifest_root: Path | None = None):
        self.root = Path(root).resolve()
        self.version = version
        project = tomllib.loads((self.root / "pyproject.toml").read_text())
        if project["project"]["version"] != version:
            raise ValueError("Luna source package version does not match the requested variant")
        self.package = self.root / "src"
        origin = self.package / "__init__.py"
        loaded = sys.modules.get("LunaEquivariance")
        if loaded is not None:
            if Path(loaded.__file__).resolve() != origin:
                raise RuntimeError("Different Luna sources require separate processes")
        else:
            spec = importlib.util.spec_from_file_location(
                "LunaEquivariance", origin, submodule_search_locations=[str(self.package)])
            if spec is None or spec.loader is None:
                raise ImportError(origin)
            module = importlib.util.module_from_spec(spec)
            sys.modules["LunaEquivariance"] = module
            spec.loader.exec_module(module)
        self.interface = importlib.import_module("LunaEquivariance.o3.linear.interface")
        self.metadata = importlib.import_module("LunaEquivariance.o3.linear.metadata")
        self.kernels = importlib.import_module("LunaEquivariance.o3.linear.kernels.utils")
        self.manifest_root = manifest_root
        if manifest_root is not None:
            self.capture_jit(manifest_root)

    def capture_jit(self, root: Path):
        """Record generated sources/flags at build time without changing their bytes."""
        extension = importlib.import_module("LunaEquivariance.utils.extension")
        original = extension.load_extension
        previous = getattr(original, "_h20_manifest_root", None)
        if previous is not None:
            if previous != root.resolve():
                raise RuntimeError("A Luna process cannot mix JIT evidence roots")
            return
        root.mkdir(parents=True, exist_ok=True)

        def recorded_load(name, **kwargs):
            record = {"name": name, "arguments": kwargs, "source_root": str(self.root),
                      "version": self.version}
            key = Evidence.fingerprint(record)
            directory = root / key
            directory.mkdir(parents=True, exist_ok=True)
            for label, suffix in (("cpp_sources", "cpp"), ("cuda_sources", "cu")):
                source = kwargs.get(label, "")
                if source:
                    target = directory / ("generated." + suffix)
                    if target.exists() and target.read_text() != source:
                        raise RuntimeError("Content addressed Luna JIT source changed")
                    target.write_text(source)
            # The immutable request proves flags and original generator output even
            # if upstream removes its temporary Ninja directory after compilation.
            request = directory / "request.json"
            if not request.exists():
                Evidence.write(request, record)
            module = original(name, **kwargs)
            library = Path(module.__file__).resolve()
            loaded = {"path": str(library), "sha256": Evidence.sha256(library),
                      "request_sha256": Evidence.sha256(request)}
            history = directory / "loads" / (Evidence.fingerprint(loaded) + ".json")
            if not history.exists():
                Evidence.write(history, loaded)
            if not (directory / "loaded.json").exists():
                Evidence.write(directory / "loaded.json", loaded)
            return module

        recorded_load._h20_manifest_root = root.resolve()
        extension.load_extension = recorded_load
        # Auxiliary streams import this entry lazily, outside extension.py.
        jit = importlib.import_module("LunaEquivariance.utils.jit")
        jit.load_extension = recorded_load
        build = jit.cpp_extension._write_ninja_file_and_build_library
        if getattr(build, "_h20_manifest_root", None) is None:
            def recorded_build(*args, **kwargs):
                import inspect

                arguments = inspect.signature(build).bind(*args, **kwargs).arguments
                sources = [Path(path) for path in arguments["sources"]]
                record = {"arguments": dict(arguments),
                          "source_hashes": {path.name: Evidence.sha256(path) for path in sources}}
                directory = root / "builds" / Evidence.fingerprint(record)
                directory.mkdir(parents=True, exist_ok=True)
                for path in sources:
                    shutil.copy2(path, directory / path.name)
                Evidence.write(directory / "arguments.json", record)
                try:
                    return build(*args, **kwargs)
                finally:
                    recipe = Path(arguments["build_directory"]) / "build.ninja"
                    if recipe.is_file():
                        shutil.copy2(recipe, directory / "build.ninja")

            recorded_build._h20_manifest_root = root.resolve()
            jit.cpp_extension._write_ninja_file_and_build_library = recorded_build

    def identity(self) -> dict:
        files = {str(path.relative_to(self.root)): Evidence.sha256(path)
                 for path in sorted(self.package.rglob("*.py"))}
        digest = Evidence.fingerprint(files)
        jit = importlib.import_module("LunaEquivariance.utils.jit")
        libraries = {}
        for module in jit.MODULES.values():
            filename = getattr(module, "__file__", None)
            if filename is not None:
                path = Path(filename).resolve()
                libraries[str(path)] = Evidence.sha256(path)
        mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                         if ".so" in line and any(name in line for name in libraries)})
        return {"version": self.version, "source_root": str(self.root),
                "pyproject_sha256": Evidence.sha256(self.root / "pyproject.toml"),
                "package_origin": str(Path(sys.modules["LunaEquivariance"].__file__).resolve()),
                "python_source_hash": digest, "python_files": files,
                "jit_libraries": libraries, "mapped_jit_libraries": mapped,
                "jit_manifest_root": str(self.manifest_root) if self.manifest_root is not None else None,
                "adapter_sha256": Evidence.sha256(Path(__file__))}


class LunaAdapter:
    """One fixture with exact path instructions, normalization and layout mapping."""

    BOUNDARIES = ("native_public", "adapted_public_e2e", "prepared_native_launch_with_alloc")
    OPERATIONS = ("forward", "grad_x", "grad_w", "grad_x_w")

    def __init__(self, case, source_root: Path | str, version: str, manifest_root: Path | None = None):
        import torch

        self.case = case
        self.source = LunaSource(source_root, version, manifest_root)
        self.dtype = case.x.dtype
        self.device = case.x.device
        self.batch = case.point.batch
        self.u = case.u
        self.v = case.w3.shape[-1]
        self.dims = tuple(case.dims)
        self.in_offsets, self.out_offsets, self.group_dims, group_of = [], [], [], []
        in_offset = 0
        out_offset = 0
        for index, dim in enumerate(self.dims):
            self.in_offsets.append(in_offset)
            in_offset += dim
            if index == 0 or dim != self.dims[index - 1]:
                self.out_offsets.append(out_offset)
                self.group_dims.append(dim)
                out_offset += dim
            group_of.append(len(self.group_dims) - 1)
        self.instructions = tuple((index, group) for index, group in enumerate(group_of))
        irreps_in = "+".join(f"{self.u}x{self.irrep(dim)}" for dim in self.dims)
        irreps_out = "+".join(f"{self.v}x{self.irrep(dim)}" for dim in self.group_dims)
        self.config = self.source.metadata.linear_config(
            irreps_in, irreps_out, None, None, True, self.instructions,
            False, "element", self.dtype, 32)
        if self.config.weight_numel != len(self.dims) * self.u * self.v:
            raise RuntimeError("Luna changed path or weight ordering")
        actual = tuple((item.i_in, item.i_out) for item in self.config.instructions)
        if actual != self.instructions:
            raise RuntimeError("Luna instruction order differs from the shared fixture")
        self.alpha = tuple(float(item.path_weight) for item in self.config.instructions)
        self.scale = (torch.tensor(case.cg, device=self.device, dtype=self.dtype)
                      / torch.tensor(self.alpha, device=self.device, dtype=self.dtype))[:, None, None]
        with torch.no_grad():
            self.x_native = self.pack_x(case.x3)
            self.w_native = self.prepare_weight(case.w3)
            self.dy_native = self.pack_grad(case.grad3)
        self._graphs = {}
        self._prepared = {}
        self._prepared_keys = {}

    @staticmethod
    def irrep(dim: int) -> str:
        degree = (dim - 1) // 2
        if 2 * degree + 1 != dim:
            raise ValueError("Luna irreps require positive odd component counts")
        return f"{degree}{'e' if degree % 2 == 0 else 'o'}"

    def pack_x(self, x):
        import torch

        value = x.reshape(self.batch, sum(self.dims), self.u)
        return torch.cat([value[:, offset:offset + dim, :].transpose(1, 2).reshape(self.batch, -1)
                          for offset, dim in zip(self.in_offsets, self.dims)], dim=1).contiguous()

    def pack_grad(self, value):
        import torch

        value = value.reshape(self.batch, sum(self.group_dims), self.v)
        return torch.cat([value[:, offset:offset + dim, :].transpose(1, 2).reshape(self.batch, -1)
                          for offset, dim in zip(self.out_offsets, self.group_dims)], dim=1).contiguous()

    def unpack(self, value):
        import torch

        return torch.cat([value[:, self.v * offset:self.v * (offset + dim)]
                          .reshape(self.batch, self.v, dim).transpose(1, 2)
                          for offset, dim in zip(self.out_offsets, self.group_dims)], dim=1)

    def unpack_x(self, value):
        import torch

        return torch.cat([value[:, self.u * offset:self.u * (offset + dim)]
                          .reshape(self.batch, self.u, dim).transpose(1, 2)
                          for offset, dim in zip(self.in_offsets, self.dims)], dim=1)

    def prepare_weight(self, weight):
        return (weight.reshape(len(self.dims), self.u, self.v) * self.scale).reshape(-1).contiguous()

    def native_forward(self, x=None, weight=None):
        return self.source.interface.linear(self.x_native if x is None else x,
                                            weight=self.w_native if weight is None else weight,
                                            config=self.config)

    def adapted_forward(self, x=None, weight=None):
        x = self.case.x3 if x is None else x
        native_weight = self.w_native if weight is None else self.prepare_weight(weight)
        return self.unpack(self.native_forward(self.pack_x(x), native_weight)).reshape(self.batch, -1)

    @staticmethod
    def needs(operation: str) -> tuple[bool, bool, bool]:
        return operation in ("grad_x", "grad_x_w"), operation in ("grad_w", "grad_x_w"), False

    def _graph(self, operation: str, boundary: str):
        import torch

        key = operation, boundary
        if key not in self._graphs:
            nx, nw, _ = self.needs(operation)
            with torch.enable_grad():
                if boundary == "adapted_public_e2e":
                    x = self.case.x.detach().requires_grad_(nx)
                    weight = self.case.w.detach().requires_grad_(nw)
                    output = self.adapted_forward(x, weight if nw else None)
                    dy = self.case.grad
                else:
                    x = self.x_native.detach().requires_grad_(nx)
                    weight = self.w_native.detach().requires_grad_(nw)
                    output = self.native_forward(x, weight)
                    dy = self.dy_native
                leaves = tuple(value for value in (x, weight) if value.requires_grad)
                self._graphs[key] = output, leaves, dy
        return self._graphs[key]

    def prepare(self, operation: str, boundary: str):
        """Finish JIT and graph preparation before accuracy/timing windows."""
        if operation not in self.OPERATIONS or boundary not in self.BOUNDARIES:
            raise ValueError((operation, boundary))
        if boundary == "prepared_native_launch_with_alloc":
            self._prepare_launch(operation)
        elif operation != "forward":
            self._graph(operation, boundary)
        return self.call(operation, boundary)

    def _prepare_launch(self, operation: str):
        if operation in self._prepared:
            return
        nx, nw, _ = self.needs(operation)
        order = 0 if operation == "forward" else 1
        values = ({"x": self.x_native, "weight": self.w_native}
                  if order == 0 else {"x": self.x_native if nw else None,
                                      "weight": self.w_native if nx else None,
                                      "grad_out": self.dy_native})
        roles = ("x", "weight", "bias") if order == 0 else ("x", "weight", "bias", "grad_out")
        arguments = tuple(values.get(role) for role in roles)
        needs = None if order == 0 else (nx, nw, False)
        self.source.kernels.run_kernel(self.config, order, *arguments, needs=needs, num_items=self.batch)
        available = tuple(role for role in roles if values.get(role) is not None)
        program = self.source.kernels.select_program(self.config, order, needs=needs, available=available)
        bound = max(self.batch * self.config.index_batch_span + self.config.meta["bias_numel"]
                    + 3 * self.source.kernels.TASK_ALIGNMENT,
                    program["weight_terms"] * self.config.index_workspace_span)
        bits = 64 if bound + 65535 * 256 > (1 << 31) - 1 else self.config.index_bits
        key = (order, needs, available, self.source.kernels.get_cuda_arch(self.device), bits, False, self.dtype)
        if order == 0:
            key += (None,)
        if key not in self.config.kernels:
            raise RuntimeError(f"Warmed native kernel exact key is missing: {key}")
        launch = self.config.kernels[key]
        launch_arguments = tuple(values[role] for role in program["inputs"])
        self._prepared[operation] = launch, launch_arguments, order
        self._prepared_keys[operation] = repr(key)

    def call(self, operation: str, boundary: str):
        import torch

        if boundary == "prepared_native_launch_with_alloc":
            launch, arguments, order = self._prepared[operation]
            value = launch(*arguments, self.batch)
            if order == 0:
                return value[0]
            nx, nw, _ = self.needs(operation)
            selected = tuple(result for result, wanted in zip(value[:2], (nx, nw)) if wanted)
            return selected[0] if len(selected) == 1 else selected
        if operation == "forward":
            with torch.no_grad():
                return self.adapted_forward() if boundary == "adapted_public_e2e" else self.native_forward()
        output, leaves, dy = self._graphs[(operation, boundary)]
        result = torch.autograd.grad(output, leaves, dy, retain_graph=True)
        return result[0] if len(result) == 1 else result

    def common_output(self, operation: str, boundary: str, value):
        """Map outputs for accuracy outside the native/prepared timed windows."""
        if boundary == "adapted_public_e2e":
            return value
        if operation == "forward":
            return self.unpack(value).reshape(self.batch, -1)
        if operation == "grad_x":
            return self.unpack_x(value).reshape_as(self.case.x)
        if operation == "grad_w":
            return (value.reshape_as(self.case.w3) * self.scale).reshape_as(self.case.w)
        dx, dw = value
        return (self.unpack_x(dx).reshape_as(self.case.x),
                (dw.reshape_as(self.case.w3) * self.scale).reshape_as(self.case.w))

    def identity(self) -> dict:
        value = self.source.identity()
        value.update({"backend": self.config.last_execution_backend,
                      "execution_dtype": str(getattr(self.config, "last_execution_dtype", None)),
                      "instructions": self.instructions, "alpha": self.alpha,
                      "weight_numel": self.config.weight_numel,
                      "kernel_keys": [repr(key) for key in self.config.kernels],
                      "raw_layout_kernel_keys": [repr(key) for key in getattr(self.config, "raw_layout_kernels", {})],
                      "prepared_kernel_keys": self._prepared_keys,
                      "prepared_bindings": {operation: {"module": getattr(launch, "__module__", None),
                                                          "name": getattr(launch, "__name__", None)}
                                            for operation, (launch, _, _) in self._prepared.items()},
                      "prepared_allocation": "generated C++ torch::empty; output allocation is inside launch"})
        return value

    def close(self):
        self._graphs.clear()
        self._prepared.clear()
