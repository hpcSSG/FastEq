"""Runtime integration of native CPU operators into a MACE model."""
from __future__ import annotations

import atexit
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch

from fasteq.cpu import load_extension
from fasteq.ops._autograd import NativeSHFunction, NativeFCTPFunction, NativeTPUFunction, NativeLinearFunction, PackedFCTPFunction, PackedLinearFunction, NativeSymmetricContractionFunction, NativeSymmetricContractionFusedFunction, NativeRadialMLPFunction
from fasteq.ops.metadata import prod, irreps_metadata, make_symmetric_contraction_metadata, make_radial_mlp_metadata, make_fctp_metadata, make_linear_metadata, make_tpu_metadata, make_packed_fctp_metadata, make_packed_linear_metadata
from fasteq.ops._profiling import set_profile_callback


_EXTENSION = None
_PROFILE_ENABLED = os.environ.get("NATIVE_CPU_MACE_PROFILE", "0") == "1"
_AUTOGRAD_PROFILE_ENABLED = (
    os.environ.get("NATIVE_CPU_MACE_AUTOGRAD_PROFILE", "0") == "1"
)
_PROFILE_HOOKS_ENABLED = (
    os.environ.get("NATIVE_CPU_MACE_PROFILE_HOOKS", "1") == "1"
)
_PROFILE_ACC = defaultdict(float)
_PROFILE_COUNT = defaultdict(int)
_PROFILE_PRINTED = False
_PROFILE_FORWARD_COUNT = 0
_AUTOGRAD_PROFILE_COUNT = 0
_ONES_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}


def _cached_ones_like(tensor: torch.Tensor) -> torch.Tensor:
    key = (tensor.device, tensor.dtype, tuple(tensor.shape))
    value = _ONES_CACHE.get(key)
    if value is None:
        value = torch.ones_like(tensor, requires_grad=False)
        _ONES_CACHE[key] = value
    return value


def _fast_get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    displacement: Optional[torch.Tensor],
    vectors: Optional[torch.Tensor] = None,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
    compute_hessian: bool = False,
    compute_edge_forces: bool = False,
):
    if compute_hessian or compute_edge_forces:
        from mace.modules.utils import get_outputs

        return get_outputs(
            energy=energy,
            positions=positions,
            cell=cell,
            displacement=displacement,
            vectors=vectors,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

    grad_outputs = [_cached_ones_like(energy)]
    retain_graph = training
    forces = None
    virials = None
    stress = None
    if (compute_virials or compute_stress) and displacement is not None:
        forces, virials = torch.autograd.grad(
            outputs=[energy],
            inputs=[positions, displacement],
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=training,
            allow_unused=True,
            materialize_grads=False,
        )
        if compute_stress and virials is not None:
            cell_3x3 = cell.view(-1, 3, 3)
            volume = torch.linalg.det(cell_3x3).abs().unsqueeze(-1)
            stress = virials / volume.view(-1, 1, 1)
            stress = torch.where(
                torch.abs(stress) < 1e10,
                stress,
                torch.zeros_like(stress),
            )
        if forces is None:
            forces = torch.zeros_like(positions)
        if virials is None:
            virials = torch.zeros(
                (1, 3, 3),
                dtype=positions.dtype,
                device=positions.device,
            )
    elif compute_force:
        forces = torch.autograd.grad(
            outputs=[energy],
            inputs=[positions],
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=training,
            allow_unused=True,
            materialize_grads=False,
        )[0]
        if forces is None:
            forces = torch.zeros_like(positions)

    if forces is not None:
        forces = -1 * forces
    if virials is not None:
        virials = -1 * virials
    return forces, virials, stress, None, None


def _record(name: str, elapsed: float) -> None:
    if _PROFILE_ENABLED:
        _PROFILE_ACC[name] += elapsed
        _PROFILE_COUNT[name] += 1


def _reset_profile() -> None:
    global _PROFILE_PRINTED
    _PROFILE_ACC.clear()
    _PROFILE_COUNT.clear()
    _PROFILE_PRINTED = False


set_profile_callback(_record)


def _print_profile() -> None:
    global _PROFILE_PRINTED
    if not _PROFILE_ENABLED or not _PROFILE_ACC or _PROFILE_PRINTED:
        return
    _PROFILE_PRINTED = True
    total = _PROFILE_ACC.get("forward_total", 0.0)
    backward_total = _PROFILE_ACC.get("get_outputs_autograd", 0.0)
    backward_attributed = sum(
        elapsed
        for name, elapsed in _PROFILE_ACC.items()
        if name.startswith("backward.")
    )
    forward_attributed = sum(
        elapsed
        for name, elapsed in _PROFILE_ACC.items()
        if name not in {"forward_total", "get_outputs_autograd"}
        and not name.startswith("backward.")
    )
    print(f"[native-profile pid={os.getpid()}]", flush=True)
    for name, elapsed in sorted(
        _PROFILE_ACC.items(), key=lambda item: -item[1]
    ):
        if name == "forward_total":
            continue
        share = 100.0 * elapsed / total if total > 0 else 0.0
        print(
            f"  {name:<24} {elapsed:10.6f} s "
            f"{_PROFILE_COUNT[name]:8d} calls {share:7.2f}%",
            flush=True,
        )
    def summary(name: str, elapsed: float) -> None:
        share = 100.0 * elapsed / total if total > 0 else 0.0
        print(f"  {name:<24} {elapsed:10.6f} s"
              f"{'':>8} calls {share:7.2f}%", flush=True)

    print("  -- complete accounting --", flush=True)
    summary("backward_total", backward_total)
    summary("backward_attributed", backward_attributed)
    summary("backward_other", backward_total - backward_attributed)
    summary("forward_attributed", forward_attributed)
    summary(
        "forward_other",
        total - backward_total - forward_attributed,
    )


atexit.register(_print_profile)


def get_extension():
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load_extension()
        print(f"[native-mace] extension loaded: {_EXTENSION.__file__}",
              flush=True)
    return _EXTENSION


class NativeSHComponentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, extension, scale):
        ctx.extension = extension
        ctx.scale = float(scale)
        ctx.save_for_backward(x)
        return extension.sh3_forward(x, True) * ctx.scale

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        started = time.perf_counter()
        try:
            return (
                ctx.extension.sh3_backward(
                    x, grad_out.contiguous() * ctx.scale, True
                ),
                None,
                None,
            )
        finally:
            _record(
                "backward.spherical_harmonics",
                time.perf_counter() - started,
            )


class NativeScatterSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, index, dim_size, extension):
        squeeze = src.dim() == 1
        value = src.unsqueeze(-1) if squeeze else src
        out = extension.scatter_sum_forward(
            value.contiguous(), index.contiguous(), int(dim_size)
        )
        ctx.extension = extension
        ctx.num_edges = src.shape[0]
        ctx.squeeze = squeeze
        ctx.save_for_backward(index)
        return out.squeeze(-1) if squeeze else out

    @staticmethod
    def backward(ctx, grad_out):
        (index,) = ctx.saved_tensors
        value = grad_out.unsqueeze(-1) if ctx.squeeze else grad_out
        grad_src = ctx.extension.scatter_sum_backward(
            value.contiguous(), index, ctx.num_edges
        )
        return (
            grad_src.squeeze(-1) if ctx.squeeze else grad_src,
            None,
            None,
            None,
        )


class NativeRadialEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lengths, bessel_weights, r_max, cutoff_p, extension):
        ctx.extension = extension
        ctx.cutoff_p = int(cutoff_p)
        ctx.save_for_backward(lengths, bessel_weights, r_max)
        return extension.radial_embedding_forward(
            lengths, bessel_weights, r_max, int(cutoff_p)
        )

    @staticmethod
    def backward(ctx, grad_out):
        lengths, bessel_weights, r_max = ctx.saved_tensors
        grad = ctx.extension.radial_embedding_backward(
            grad_out,
            lengths,
            bessel_weights,
            r_max,
            ctx.cutoff_p,
        )
        return grad, None, None, None, None


class NativeEdgeGeometryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, positions, edge_index, shifts, extension, normalize, eps
    ):
        vectors, lengths = extension.edge_geometry_forward(
            positions,
            edge_index,
            shifts,
            bool(normalize),
            float(eps),
        )
        ctx.extension = extension
        ctx.num_nodes = positions.shape[0]
        ctx.normalize = bool(normalize)
        ctx.eps = float(eps)
        ctx.save_for_backward(vectors, lengths, edge_index)
        return vectors, lengths

    @staticmethod
    def backward(ctx, grad_vectors, grad_lengths):
        vectors, lengths, edge_index = ctx.saved_tensors
        if grad_vectors is None:
            grad_vectors = torch.zeros_like(vectors)
        if grad_lengths is None:
            grad_lengths = torch.zeros_like(lengths)
        grad_positions, grad_shifts = ctx.extension.edge_geometry_backward(
            grad_vectors.contiguous(),
            grad_lengths.contiguous(),
            vectors,
            lengths,
            edge_index,
            ctx.num_nodes,
            ctx.normalize,
            ctx.eps,
        )
        return (
            grad_positions,
            None,
            grad_shifts,
            None,
            None,
            None,
        )


class NativeMaceModel(torch.nn.Module):
    """Python forward that replaces selected e3nn calls inside MACE."""

    def __init__(self, inner):
        super().__init__()
        object.__setattr__(self, "_native_inner", inner)
        self.add_module("_native_inner_module", inner)
        if os.environ.get("NATIVE_CPU_MACE_FREEZE_WEIGHTS", "1") == "1":
            inner.requires_grad_(False)
        self.extension = get_extension()
        self.enabled = False
        self.use_sh = os.environ.get("NATIVE_CPU_MACE_SH", "1") == "1"
        self.use_fctp = os.environ.get("NATIVE_CPU_MACE_FCTP", "1") == "1"
        self.use_linear = os.environ.get("NATIVE_CPU_MACE_LINEAR", "1") == "1"
        self.use_linear_up = (
            os.environ.get("NATIVE_CPU_MACE_LINEAR_UP", "1") == "1"
        )
        self.fold_norm = (
            os.environ.get("NATIVE_CPU_MACE_FOLD_NORM", "1") == "1"
        )
        self.use_conv_tp = os.environ.get("NATIVE_CPU_MACE_CONV_TP", "1") == "1"
        self.use_radial_mlp = (
            os.environ.get("NATIVE_CPU_MACE_RADIAL_MLP", "1") == "1"
        )
        self.use_radial_embedding = (
            os.environ.get("NATIVE_CPU_MACE_RADIAL_EMBEDDING", "1") == "1"
        )
        self.use_sc = os.environ.get("NATIVE_CPU_MACE_SC", "1") == "1"
        self.use_scatter = (
            os.environ.get("NATIVE_CPU_MACE_SCATTER", "1") == "1"
        )
        self.use_native_geometry = (
            os.environ.get("NATIVE_CPU_MACE_GEOMETRY", "0") == "1"
        )
        self.minimal_output = (
            os.environ.get("NATIVE_CPU_MACE_MINIMAL_OUTPUT", "0") == "1"
        )
        self.capture = None
        self.sh_scale = 1.0
        self.interaction_plans = []
        self.product_plans = []
        self.readout_plans = []
        self.node_embedding_plan = None
        self.radial_embedding_weights = None
        self.radial_embedding_r_max = None
        self.radial_embedding_p = None
        self._build_plans()

    def _install_backward_timer(self, module, name):
        if not _PROFILE_ENABLED or not _PROFILE_HOOKS_ENABLED:
            return
        if not hasattr(module, "register_full_backward_pre_hook"):
            return

        def pre_hook(module, grad_output):
            module._native_profile_backward_start = time.perf_counter()

        def post_hook(module, grad_input, grad_output):
            started = getattr(
                module, "_native_profile_backward_start", None
            )
            if started is not None:
                _record(
                    f"backward.{name}",
                    time.perf_counter() - started,
                )

        module.register_full_backward_pre_hook(pre_hook)
        module.register_full_backward_hook(post_hook)

    def _build_plans(self):
        inner = self._native_inner
        try:
            from mace.modules.utils import (  # noqa: F401
                get_atomic_virials_stresses,
                get_outputs,
                prepare_graph,
            )
            from mace.tools.scatter import scatter_sum  # noqa: F401
        except Exception as exc:
            print(f"[native-mace] disabled, MACE utilities unavailable: {exc}",
                  flush=True)
            return

        sh = inner.spherical_harmonics
        if list(getattr(sh, "_ls_list", [])) != [0, 1, 2, 3]:
            print("[native-mace] SH disabled: expected l=0..3", flush=True)
            return
        normalization = str(getattr(sh, "normalization", "component"))
        if normalization == "component":
            self.sh_scale = math.sqrt(4.0 * math.pi)
        elif normalization == "integral":
            self.sh_scale = 1.0
        else:
            print(f"[native-mace] SH disabled: normalization={normalization}",
                  flush=True)
            return

        for interaction in inner.interactions:
            is_residual = "Residual" in type(interaction).__name__
            try:
                linear_up = make_packed_linear_metadata(interaction.linear_up)
                linear = make_packed_linear_metadata(interaction.linear)
                if self.fold_norm:
                    linear["packed_weight"] = (
                        linear["packed_weight"]
                        * (1.0 / float(interaction.avg_num_neighbors))
                    ).contiguous()
                skip_tp = make_packed_fctp_metadata(
                    interaction.skip_tp, interaction.skip_tp.weight.detach()
                )
                conv_tp = make_tpu_metadata(interaction.conv_tp)
            except Exception as exc:
                print(
                    f"[native-mace] interaction disabled "
                    f"({type(interaction).__name__}): {exc}",
                    flush=True,
                )
                self.interaction_plans.append(None)
                continue
            try:
                radial_mlp = make_radial_mlp_metadata(
                    interaction.conv_tp_weights
                )
            except Exception as exc:
                radial_mlp = None
                print(
                    f"[native-mace] radial MLP disabled for "
                    f"{type(interaction).__name__}: {exc}",
                    flush=True,
                )
            self.interaction_plans.append({
                "module": interaction,
                "residual": is_residual,
                "linear_up": linear_up,
                "linear": linear,
                "skip_tp": skip_tp,
                "conv_tp": conv_tp,
                "radial_mlp": radial_mlp,
            })

        for product in inner.products:
            try:
                linear = make_packed_linear_metadata(product.linear)
                contractions = getattr(
                    product.symmetric_contractions, "contractions", None
                )
                if contractions is None:
                    raise ValueError("no Contraction modules found")
                sc_plans = []
                for contraction in contractions:
                    plan = {
                        "module": contraction,
                        "correlation": int(contraction.correlation),
                    }
                    try:
                        plan["fused"] = make_symmetric_contraction_metadata(
                            contraction
                        )
                    except Exception as exc:
                        plan["fused"] = None
                        print(
                            f"[native-mace] fused SC disabled for "
                            f"{type(product).__name__}: {exc}",
                            flush=True,
                        )
                    sc_plans.append(plan)
            except Exception as exc:
                print(f"[native-mace] product linear disabled: {exc}",
                      flush=True)
                self.product_plans.append(None)
                continue
            self.product_plans.append({
                "module": product,
                "linear": linear,
                "sc": sc_plans,
            })

        for readout in inner.readouts:
            try:
                if hasattr(readout, "linear_1") and hasattr(readout, "linear_2"):
                    plan = (
                        make_packed_linear_metadata(readout.linear_1),
                        make_packed_linear_metadata(readout.linear_2),
                    )
                elif hasattr(readout, "linear"):
                    plan = (make_packed_linear_metadata(readout.linear),)
                else:
                    raise NotImplementedError("unsupported readout layout")
            except Exception as exc:
                plan = None
                print(
                    f"[native-mace] readout not replaced "
                    f"({type(readout).__name__}): {exc}",
                    flush=True,
                )
            self.readout_plans.append(plan)

        try:
            self.node_embedding_plan = make_packed_linear_metadata(
                inner.node_embedding.linear
            )
        except Exception as exc:
            print(
                f"[native-mace] node embedding not replaced: {exc}",
                flush=True,
            )

        try:
            radial_embedding = inner.radial_embedding
            if hasattr(radial_embedding, "distance_transform"):
                raise NotImplementedError("distance_transform is not supported")
            bessel = radial_embedding.bessel_fn
            cutoff = radial_embedding.cutoff_fn
            self.radial_embedding_weights = (
                bessel.bessel_weights.detach().to(torch.float64).contiguous()
            )
            self.radial_embedding_r_max = (
                bessel.r_max.detach().to(torch.float64).contiguous()
            )
            self.radial_embedding_p = int(cutoff.p.detach().cpu().item())
        except Exception as exc:
            self.use_radial_embedding = False
            print(
                f"[native-mace] radial embedding not replaced: {exc}",
                flush=True,
            )

        self.enabled = (
            all(plan is not None for plan in self.interaction_plans)
            and all(plan is not None for plan in self.product_plans)
        )
        self._install_backward_timer(
            inner.node_embedding, "node_embedding"
        )
        self._install_backward_timer(
            inner.radial_embedding, "radial_embedding"
        )
        for index, interaction in enumerate(inner.interactions):
            self._install_backward_timer(
                interaction.conv_tp_weights,
                f"conv_tp_weights_mlp[{index}]",
            )
        for index, readout in enumerate(inner.readouts):
            self._install_backward_timer(
                readout, f"readout[{index}]"
            )
        if hasattr(inner, "scale_shift"):
            self._install_backward_timer(
                inner.scale_shift, "scale_shift"
            )
        print(
            f"[native-mace] enabled={self.enabled} "
            f"interactions={len(self.interaction_plans)} "
            f"products={len(self.product_plans)} "
            f"sh_scale={self.sh_scale:.6f}",
            flush=True,
        )
    def __getattr__(self, name: str) -> Any:
        if name == "_native_inner":
            raise AttributeError(name)
        inner = self.__dict__.get("_native_inner")
        if inner is not None:
            return getattr(inner, name)
        raise AttributeError(name)

    @staticmethod
    def _timed(name, function, *args, **kwargs):
        if not _PROFILE_ENABLED:
            return function(*args, **kwargs)
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            _record(name, time.perf_counter() - started)

    def _linear(self, module, plan, value):
        if not self.use_linear:
            return module(value)
        started = time.perf_counter()
        out = PackedLinearFunction.apply(
            value, plan["packed_weight"], self.extension, plan
        )
        _record("linear", time.perf_counter() - started)
        return out

    def _skip_tp(self, module, plan, value, node_attrs):
        if not self.use_fctp:
            return module(value, node_attrs)
        vstar = node_attrs.argmax(dim=-1).to(torch.int64)
        started = time.perf_counter()
        out = PackedFCTPFunction.apply(
            value, plan["packed_weight"], vstar, self.extension, plan
        )
        _record("fctp_skip_tp", time.perf_counter() - started)
        return out

    def _conv_tp(self, module, plan, x1, x2, weight):
        if not self.use_conv_tp:
            return module(x1, x2, weight)
        started = time.perf_counter()
        out = NativeTPUFunction.apply(
            x1, x2, weight, self.extension, plan
        )
        _record("conv_tp", time.perf_counter() - started)
        return out

    def _radial_mlp(self, module, plan, value):
        if not self.use_radial_mlp or plan is None:
            return module(value)
        started = time.perf_counter()
        out = NativeRadialMLPFunction.apply(value, self.extension, plan)
        _record("conv_tp_weights_mlp", time.perf_counter() - started)
        return out

    def _symmetric_contraction(self, plan, module, x, y):
        if not self.use_sc:
            started = time.perf_counter()
            out = module(x, y)
            _record("sc_original", time.perf_counter() - started)
            return out
        element = y.argmax(dim=-1).to(torch.int64)
        outs = []
        for contraction_plan in plan:
            contraction = contraction_plan["module"]
            if contraction_plan.get("fused") is not None:
                started = time.perf_counter()
                out = NativeSymmetricContractionFusedFunction.apply(
                    x,
                    element,
                    self.extension,
                    contraction_plan["fused"],
                )
                _record("sc_fused", time.perf_counter() - started)
                outs.append(out)
                continue
            correlation = contraction_plan["correlation"]
            started = time.perf_counter()
            out = NativeSymmetricContractionFunction.apply(
                contraction.U_tensors(correlation),
                contraction.weights_max,
                x,
                element,
                self.extension,
            )
            _record("sc_main", time.perf_counter() - started)
            lower_started = time.perf_counter()
            for i, (weight, cw, cf) in enumerate(zip(
                contraction.weights,
                contraction.contractions_weighting,
                contraction.contractions_features,
            )):
                c_tensor = cw(
                    contraction.U_tensors(correlation - i - 1),
                    weight,
                    y,
                )
                c_tensor = c_tensor + out
                out = cf(c_tensor, x)
            _record("sc_lower", time.perf_counter() - lower_started)
            outs.append(out.view(out.shape[0], -1))
        if len(outs) == 1:
            return outs[0]
        return torch.cat(outs, dim=-1)

    def _readout(self, plan, module, node_feats, node_heads):
        if plan is None:
            return module(node_feats, node_heads)
        if len(plan) == 2:
            hidden = self._linear(module.linear_1, plan[0], node_feats)
            hidden = module.non_linearity(hidden)
            return self._linear(module.linear_2, plan[1], hidden)
        return self._linear(module.linear, plan[0], node_feats)

    def _scatter_sum(self, src, index, dim_size):
        if not self.use_scatter:
            from mace.tools.scatter import scatter_sum

            return scatter_sum(src, index, dim=0, dim_size=dim_size)
        return NativeScatterSumFunction.apply(
            src, index, dim_size, self.extension
        )

    def _interaction(self, plan, node_attrs, node_feats, edge_attrs,
                     edge_feats, edge_index):
        from mace.tools.scatter import scatter_sum

        module = plan["module"]
        capture = self.capture
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        if plan["residual"]:
            sc = self._skip_tp(
                module.skip_tp, plan["skip_tp"], node_feats, node_attrs
            )
        else:
            sc = None

        if self.use_linear_up:
            node_feats = self._linear(
                module.linear_up, plan["linear_up"], node_feats
            )
        else:
            node_feats = module.linear_up(node_feats)
        if capture is not None:
            capture["up"] = node_feats
        tp_weights = self._radial_mlp(
            module.conv_tp_weights, plan["radial_mlp"], edge_feats
        )
        message = self._conv_tp(
            module.conv_tp,
            plan["conv_tp"],
            node_feats[sender],
            edge_attrs,
            tp_weights,
        )
        if capture is not None:
            capture["raw"] = message
        message = self._timed(
            "scatter",
            self._scatter_sum,
            message,
            receiver,
            num_nodes,
        )
        if capture is not None:
            capture["scatter"] = message
        message = self._linear(module.linear, plan["linear"], message)
        if not self.use_linear or not self.fold_norm:
            message = message / module.avg_num_neighbors
        if capture is not None:
            capture["linear"] = message
        if capture is not None:
            capture["norm"] = message
        if not plan["residual"]:
            message = self._skip_tp(
                module.skip_tp, plan["skip_tp"], message, node_attrs
            )
            if capture is not None:
                capture["postskip"] = message
            sc = None
        return module.reshape(message), sc

    def forward(
        self,
        data,
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ):
        forward_started = time.perf_counter()
        if not self.enabled or lammps_mliap:
            return self._native_inner(
                data,
                training=training,
                compute_force=compute_force,
                compute_virials=compute_virials,
                compute_stress=compute_stress,
                compute_displacement=compute_displacement,
                compute_hessian=compute_hessian,
                compute_edge_forces=compute_edge_forces,
                compute_atomic_stresses=compute_atomic_stresses,
                lammps_mliap=lammps_mliap,
            )

        from mace.modules.utils import (
            get_atomic_virials_stresses,
            get_outputs,
            prepare_graph,
        )
        from mace.tools.scatter import scatter_sum

        geometry_restore = None
        if self.use_native_geometry:
            mace_utils = sys.modules["mace.modules.utils"]
            geometry_restore = mace_utils.get_edge_vectors_and_lengths
            mace_utils.get_edge_vectors_and_lengths = (
                self._native_edge_vectors_and_lengths
            )
        try:
            ctx = self._timed(
                "prepare_graph",
                prepare_graph,
                data,
                compute_virials=compute_virials,
                compute_stress=compute_stress,
                compute_displacement=compute_displacement,
                lammps_mliap=False,
            )
        finally:
            if geometry_restore is not None:
                sys.modules[
                    "mace.modules.utils"
                ].get_edge_vectors_and_lengths = geometry_restore
        num_atoms_arange = ctx.num_atoms_arange
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        cell = ctx.cell
        node_heads = ctx.node_heads

        node_e0 = self._timed(
            "atomic_energies",
            self._native_inner.atomic_energies_fn,
            data["node_attrs"],
        )[num_atoms_arange, node_heads]
        e0 = self._timed(
            "node_energy_scatter",
            self._scatter_sum,
            node_e0,
            data["batch"],
            num_graphs,
        )
        if self.node_embedding_plan is not None:
            node_feats = self._timed(
                "node_embedding",
                self._linear,
                self._native_inner.node_embedding.linear,
                self.node_embedding_plan,
                data["node_attrs"],
            )
        else:
            node_feats = self._timed(
                "node_embedding",
                self._native_inner.node_embedding,
                data["node_attrs"],
            )
        if self.use_sh:
            started = time.perf_counter()
            edge_attrs = NativeSHComponentFunction.apply(
                vectors, self.extension, self.sh_scale
            )
            _record("spherical_harmonics", time.perf_counter() - started)
        else:
            edge_attrs = self._native_inner.spherical_harmonics(vectors)
        if self.use_radial_embedding:
            edge_feats = self._timed(
                "radial_embedding",
                NativeRadialEmbeddingFunction.apply,
                ctx.lengths,
                self.radial_embedding_weights,
                self.radial_embedding_r_max,
                self.radial_embedding_p,
                self.extension,
            )
        else:
            edge_feats = self._timed(
                "radial_embedding",
                self._native_inner.radial_embedding,
                ctx.lengths,
                data["node_attrs"],
                data["edge_index"],
                self._native_inner.atomic_numbers,
            )

        if hasattr(self._native_inner, "pair_repulsion"):
            pair_node_energy = self._timed(
                "pair_repulsion",
                self._native_inner.pair_repulsion_fn,
                ctx.lengths,
                data["node_attrs"],
                data["edge_index"],
                self._native_inner.atomic_numbers,
            )
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        node_es_list = [pair_node_energy]
        node_feats_list = []
        for interaction_index, (interaction_plan, product_plan) in enumerate(
            zip(self.interaction_plans, self.product_plans)
        ):
            node_feats, sc = self._interaction(
                interaction_plan,
                data["node_attrs"],
                node_feats,
                edge_attrs,
                edge_feats,
                data["edge_index"],
            )
            product = product_plan["module"]
            node_feats = self._symmetric_contraction(
                product_plan["sc"],
                product.symmetric_contractions,
                node_feats,
                data["node_attrs"],
            )
            node_feats = self._linear(
                product.linear, product_plan["linear"], node_feats
            )
            if product.use_sc and sc is not None:
                node_feats = node_feats + sc
            if not self.minimal_output:
                node_feats_list.append(node_feats)
            readout_index = interaction_index
            readout = self._native_inner.readouts[readout_index]
            node_es_list.append(
                self._timed(
                    "readout",
                    self._readout,
                    self.readout_plans[readout_index],
                    readout,
                    node_feats,
                    node_heads,
                )[
                    num_atoms_arange, node_heads
                ]
            )

        node_feats_out = (
            None
            if self.minimal_output
            else torch.cat(node_feats_list, dim=-1)
        )
        node_inter_es = node_es_list[0]
        for extra_node_energy in node_es_list[1:]:
            node_inter_es = node_inter_es + extra_node_energy
        node_inter_es = self._timed(
            "scale_shift",
            self._native_inner.scale_shift,
            node_inter_es,
            node_heads,
        )
        inter_e = self._timed(
            "graph_energy_scatter",
            self._scatter_sum,
            node_inter_es,
            data["batch"],
            num_graphs,
        )
        total_energy = e0 + inter_e
        node_energy = (
            None
            if self.minimal_output
            else node_e0.clone().double() + node_inter_es.clone().double()
        )

        outputs_kwargs = {
            "energy": inter_e,
            "positions": positions,
            "displacement": displacement,
            "vectors": vectors,
            "cell": cell,
            "training": training,
            "compute_force": compute_force,
            "compute_virials": compute_virials,
            "compute_stress": compute_stress,
            "compute_hessian": compute_hessian,
            "compute_edge_forces": compute_edge_forces,
        }
        global _AUTOGRAD_PROFILE_COUNT
        if _AUTOGRAD_PROFILE_ENABLED and _AUTOGRAD_PROFILE_COUNT < 2:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=(
                    os.environ.get(
                        "NATIVE_CPU_MACE_AUTOGRAD_PROFILE_STACK", "0"
                    ) == "1"
                ),
                profile_memory=False,
                with_stack=(
                    os.environ.get(
                        "NATIVE_CPU_MACE_AUTOGRAD_PROFILE_STACK", "0"
                    ) == "1"
                ),
            ) as prof:
                outputs = self._timed(
                    "get_outputs_autograd",
                    _fast_get_outputs,
                    **outputs_kwargs,
                )
            key_averages = prof.key_averages()
            if os.environ.get(
                "NATIVE_CPU_MACE_AUTOGRAD_PROFILE_STACK", "0"
            ) == "1":
                key_averages = prof.key_averages(group_by_stack_n=8)
            print(
                key_averages.table(
                    sort_by="self_cpu_time_total", row_limit=40
                ),
                flush=True,
            )
            if os.environ.get(
                "NATIVE_CPU_MACE_AUTOGRAD_PROFILE_STACK", "0"
            ) == "1":
                seen = set()
                for event in prof.events():
                    if event.name not in {
                        "aten::fill_",
                        "aten::zeros",
                        "aten::copy_",
                        "aten::add_",
                        "aten::cat",
                        "aten::slice_backward",
                        "aten::slice",
                        "SliceBackward0",
                        "CatBackward0",
                    }:
                        continue
                    stack = event.stack
                    stack_key = str(stack)
                    if stack_key not in seen:
                        seen.add(stack_key)
                        parents = []
                        parent = event.cpu_parent
                        while parent is not None and len(parents) < 12:
                            parents.append(parent.name)
                            parent = parent.cpu_parent
                        print(
                            f"[autograd-stack] {event.name} "
                            f"shapes={event.input_shapes} "
                            f"parents={' <- '.join(parents)}\n{stack}",
                            flush=True,
                        )
            _AUTOGRAD_PROFILE_COUNT += 1
        else:
            outputs = self._timed(
                "get_outputs_autograd", _fast_get_outputs, **outputs_kwargs
            )
        forces, virials, stress, hessian, edge_forces = outputs
        atomic_virials = None
        atomic_stresses = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )
        _record("forward_total", time.perf_counter() - forward_started)
        global _PROFILE_FORWARD_COUNT
        if _PROFILE_ENABLED:
            _PROFILE_FORWARD_COUNT += 1
            print_after = int(
                os.environ.get("NATIVE_CPU_MACE_PROFILE_PRINT_AFTER", "0")
            )
            if print_after > 0 and _PROFILE_FORWARD_COUNT >= print_after:
                _print_profile()
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }

    def _native_edge_vectors_and_lengths(
        self,
        positions,
        edge_index,
        shifts,
        normalize: bool = False,
        eps: float = 1e-9,
    ):
        if not self.use_native_geometry or normalize:
            raise RuntimeError("native edge geometry fallback is unavailable")
        return NativeEdgeGeometryFunction.apply(
            positions,
            edge_index,
            shifts,
            self.extension,
            normalize,
            eps,
        )


def install_native_mace(calculator):
    """Wrap every model in a MACECalculator when explicitly enabled."""
    if os.environ.get("NATIVE_CPU_MACE", "0") != "1":
        return calculator
    calculator.models = [
        model if isinstance(model, NativeMaceModel) else NativeMaceModel(model)
        for model in calculator.models
    ]
    return calculator
