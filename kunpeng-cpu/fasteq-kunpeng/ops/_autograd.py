"""Input-gradient wrappers extracted from the original CPU implementation."""
import torch
from ._profiling import _profile_backward

class NativeSHFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ext):
        ctx.ext = ext
        ctx.save_for_backward(x)
        return ext.sh3_forward(x, True)

    @staticmethod
    @_profile_backward("spherical_harmonics")
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return ctx.ext.sh3_backward(x, grad_out, True), None


class NativeFCTPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, weight, ext, metadata, one_hot):
        ctx.ext = ext
        ctx.native_meta = metadata
        ctx.one_hot = one_hot
        vstar = metadata["vstar"] if one_hot else torch.empty(0, dtype=torch.int64)
        ctx.save_for_backward(x1, x2, weight, vstar)
        return ext.fctp_forward(
            x1,
            x2,
            weight,
            vstar,
            *metadata["args"],
            metadata["w3js"],
            metadata["shared_weights"],
            one_hot,
        )

    @staticmethod
    @_profile_backward("fctp")
    def backward(ctx, grad_out):
        x1, x2, weight, vstar = ctx.saved_tensors
        grad_x1 = ctx.ext.fctp_backward_x(
            grad_out,
            x1,
            x2,
            weight,
            vstar,
            *ctx.native_meta["args"],
            ctx.native_meta["w3js"],
            ctx.native_meta["shared_weights"],
            ctx.one_hot,
        )
        return grad_x1, None, None, None, None, None


class NativeTPUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, weight, ext, metadata):
        ctx.ext = ext
        ctx.native_meta = metadata
        ctx.save_for_backward(x1, x2, weight)
        return ext.tpu_forward(
            x1,
            x2,
            weight,
            *metadata["args"],
            metadata["w3js"],
            metadata["shared_weights"],
            metadata.get("disjoint_output", False),
        )

    @staticmethod
    @_profile_backward("conv_tp")
    def backward(ctx, grad_out):
        x1, x2, weight = ctx.saved_tensors
        grad_x1, grad_x2, grad_weight = ctx.ext.tpu_backward(
            grad_out,
            x1,
            x2,
            weight,
            *ctx.native_meta["args"],
            ctx.native_meta["w3js"],
            ctx.native_meta["shared_weights"],
        )
        return grad_x1, grad_x2, grad_weight, None, None


class NativeLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, ext, metadata):
        ctx.ext = ext
        ctx.native_meta = metadata
        ctx.save_for_backward(x, weight)
        return ext.linear_forward(
            x, weight, *metadata["args"], metadata["shared_weights"]
        )

    @staticmethod
    @_profile_backward("linear")
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_x = ctx.ext.linear_backward_x(
            grad_out,
            weight,
            *ctx.native_meta["args"],
            ctx.native_meta["shared_weights"],
        )
        return grad_x, None, None, None


class PackedFCTPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_weight, vstar, ext, metadata):
        ctx.ext = ext
        ctx.native_meta = metadata
        ctx.save_for_backward(packed_weight, vstar)
        return ext.fctp_onehot_packed_forward(
            x,
            packed_weight,
            vstar,
            *metadata["args"],
            metadata["out_total"],
        )

    @staticmethod
    @_profile_backward("fctp_skip_tp")
    def backward(ctx, grad_out):
        packed_weight, vstar = ctx.saved_tensors
        grad_x = ctx.ext.fctp_onehot_packed_backward_x(
            grad_out,
            packed_weight,
            vstar,
            *ctx.native_meta["args"],
            ctx.native_meta["in_total"],
        )
        return grad_x, None, None, None, None


class PackedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_weight, ext, metadata):
        ctx.ext = ext
        ctx.native_meta = metadata
        ctx.save_for_backward(packed_weight)
        return ext.linear_packed_forward(
            x,
            packed_weight,
            *metadata["args"],
            metadata["out_total"],
        )

    @staticmethod
    @_profile_backward("linear")
    def backward(ctx, grad_out):
        (packed_weight,) = ctx.saved_tensors
        grad_x = ctx.ext.linear_packed_backward_x(
            grad_out,
            packed_weight,
            *ctx.native_meta["args"],
            ctx.native_meta["in_total"],
        )
        return grad_x, None, None, None


class NativeSymmetricContractionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, weight, x, element, ext):
        ctx.ext = ext
        ctx.save_for_backward(U, weight, element)
        return ext.symmetric_contraction3_forward(U, weight, x, element)

    @staticmethod
    @_profile_backward("sc_main")
    def backward(ctx, grad_out):
        U, weight, element = ctx.saved_tensors
        grad_x = ctx.ext.symmetric_contraction3_backward_x(
            grad_out, U, weight, element
        )
        return None, None, grad_x, None, None


class NativeSymmetricContractionFusedFunction(torch.autograd.Function):
    """Fused corr=3 scalar symmetric contraction with packed weight layout."""

    @staticmethod
    def forward(ctx, x, element, ext, metadata):
        outputs = ext.symmetric_contraction3_scalar_fused_forward(
            x,
            element,
            metadata["effective"],
            metadata["C0"],
            metadata["C1"],
        )
        out, saved_ct, saved_ct1 = outputs
        ctx.ext = ext
        ctx.native_meta = metadata
        ctx.save_for_backward(
            x,
            element,
            metadata["effective"],
            saved_ct,
            saved_ct1,
        )
        return out

    @staticmethod
    @_profile_backward("sc_fused")
    def backward(ctx, grad_out):
        x, element, effective, saved_ct, saved_ct1 = ctx.saved_tensors
        grad_x = ctx.ext.symmetric_contraction3_scalar_fused_backward_x(
            grad_out,
            x,
            element,
            effective,
            saved_ct,
            saved_ct1,
        )
        return grad_x, None, None, None


class NativeRadialMLPFunction(torch.autograd.Function):
    """Manual forward/backward for the fixed-weight edge radial MLP."""

    @staticmethod
    def forward(ctx, x, ext, metadata):
        weights = list(metadata["weights"])
        outputs = ext.radial_mlp_forward(
            x,
            weights,
            metadata["native_scales"],
            metadata["native_active"],
        )
        ctx.ext = ext
        ctx.native_meta = metadata
        ctx.layer_count = len(weights)
        ctx.save_for_backward(*outputs[1:], *weights)
        return outputs[0]

    @staticmethod
    @_profile_backward("radial_mlp")
    def backward(ctx, grad_out):
        values = list(ctx.saved_tensors)
        pre_activations = values[:ctx.layer_count - 1]
        weights = values[ctx.layer_count - 1:]
        grad_x = ctx.ext.radial_mlp_backward(
            grad_out,
            weights,
            ctx.native_meta["native_scales"],
            ctx.native_meta["native_active"],
            pre_activations,
        )
        return grad_x, None, None
