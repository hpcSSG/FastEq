"""Reusable node-major edge layout for GraphSoftmax.

Keep all per-edge tensors in this order across neighboring operations when
possible. Packing and unpacking at every softmax call also works, but their
memory traffic must be included in end-to-end performance measurements.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .graph_softmax import GraphCSR, _version, fused_graph_softmax, prepare_graph_softmax


@triton.jit
def _permute_rows(X, Y, PERM, ELEMENTS: tl.constexpr, WIDTH: tl.constexpr,
                  BLOCK: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = off // WIDTH, off % WIDTH
    source = tl.load(PERM + row, off < ELEMENTS, other=0).to(tl.int64)
    value = tl.load(X + source * WIDTH + col, off < ELEMENTS, other=0)
    tl.store(Y + off, value, off < ELEMENTS)


def _copy_permutation(x, permutation):
    x = x.contiguous()
    y = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            _permute_rows[(triton.cdiv(x.numel(), 1024),)](
                x, y, permutation, x.numel(), x.numel() // x.shape[0],
                1024, num_warps=4)
    return y


class _PermuteEdges(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, permutation, inverse):
        ctx.save_for_backward(inverse)
        return _copy_permutation(x, permutation)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        inverse, = ctx.saved_tensors
        # A bijection needs one copy per element, with no atomic accumulation.
        return _copy_permutation(grad, inverse), None, None


@dataclass(frozen=True)
class GraphSoftmaxLayout:
    """Stable destination-node ordering; pack/unpack operate on dimension zero.

    ``softmax`` consumes and returns already packed tensors. ``pack_rescale``
    reorders [E,1]/[E,H] weights while preserving scalar/shared-head weights.
    All other per-edge features must use the same permutation too. No implicit
    tensor cache is used; rebuild the layout when graph topology changes.
    """
    csr: GraphCSR
    order: torch.Tensor
    inverse: torch.Tensor
    source: torch.Tensor
    source_version: int

    def _check(self):
        if _version(self.source) != self.source_version:
            raise ValueError('Graph topology changed: rebuild the edge layout')

    def _permute(self, tensor, permutation, inverse):
        self._check()
        if tensor.ndim == 0 or tensor.shape[0] != self.csr.edges:
            raise ValueError('Expected a tensor with edges on dimension zero')
        if tensor.device != permutation.device:
            raise ValueError('Edge features and layout must be on the same device')
        if not tensor.is_cuda:
            return tensor.index_select(0, permutation)
        return _PermuteEdges.apply(tensor, permutation, inverse)

    def pack(self, tensor):
        """Copy edge features from original order to contiguous node order."""
        return self._permute(tensor, self.order, self.inverse)

    def unpack(self, tensor):
        """Copy node-ordered edge features back to original order."""
        return self._permute(tensor, self.inverse, self.order)

    def pack_rescale(self, rescale):
        self._check()
        if rescale is not None and rescale.ndim == 2 and rescale.shape[0] != 1:
            return self.pack(rescale)
        return rescale

    def softmax(self, packed_src, *, exp_rescale=None, **kwargs):
        """Run the common kernel on packed features and packed rescale weights."""
        self._check()
        return fused_graph_softmax(packed_src, ptr=self.csr.ptr, csr=self.csr,
                                   exp_rescale=exp_rescale, **kwargs)


@torch.no_grad()
def prepare_graph_softmax_layout(index, num_nodes=None):
    """Sort topology once; feature permutation is explicit and differentiable.

    Example: ``xs = layout.pack(x); rs = layout.pack_rescale(r)`` followed by
    ``ys = layout.softmax(xs, exp_rescale=rs)``. Use ``layout.unpack(ys)`` only
    when a consumer needs original edge order. Seeded Triton dropout keeps the
    same realized mask as the original-order GraphSoftmax fast path.
    """
    original = prepare_graph_softmax(index=index, num_nodes=num_nodes)
    order = original.order
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    packed = GraphCSR(original.ptr, None, original.edges, original.nodes,
                      original.max_degree, original.ptr, _version(original.ptr),
                      rng_order=order)
    return GraphSoftmaxLayout(packed, order, inverse, index, _version(index))
