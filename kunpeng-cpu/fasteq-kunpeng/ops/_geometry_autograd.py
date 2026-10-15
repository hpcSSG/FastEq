import torch

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
