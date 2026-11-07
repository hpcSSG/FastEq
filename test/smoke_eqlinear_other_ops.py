"""Check unaffected compiled operators against direct PyTorch equations."""
import torch
import fasteq  # noqa: F401
from fasteq.ops.atombit.fused_bmm_sym import fused_bmm_sym
from fasteq.ops.atombit.fused_commutator import fused_commutator
from fasteq.ops.fctp import fast_fctp


class OtherOpsSmoke:
    @staticmethod
    def check(actual, reference, inputs):
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
        gradient = torch.randn_like(actual)
        got = torch.autograd.grad(actual, inputs, gradient, retain_graph=True)
        expected = torch.autograd.grad(reference, inputs, gradient)
        for a, b in zip(got, expected):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)

    def run(self):
        torch.manual_seed(23)
        edges, features = 4, 8
        a = torch.randn(edges * features, 3, 3, device='cuda', requires_grad=True)
        b = torch.randn_like(a, requires_grad=True)
        comm = a @ b - b @ a
        reference = torch.stack((comm[:, 2, 1], comm[:, 0, 2], comm[:, 1, 0]), dim=1)
        reference = reference.reshape(edges, features, 3).transpose(1, 2)
        self.check(fused_commutator(a, b, features), reference, (a, b))

        h = torch.randn(edges, 3, 3, features, device='cuda', requires_grad=True)
        geom = torch.randn_like(h, requires_grad=True)
        raw = torch.einsum('eikf,ekjf->eijf', h, geom)
        sym = (raw + raw.transpose(1, 2)) * .5
        trace = torch.einsum('eiif->ef', sym)
        identity = torch.eye(3, device='cuda').reshape(1, 3, 3, 1)
        reference = sym - identity * (trace[:, None, None, :] / 3)
        self.check(fused_bmm_sym(h, geom), reference, (h, geom))

        batch, channels, species, width, paths = 8, 16, 4, 16, 2
        x = torch.randn(batch, paths * channels, device='cuda', requires_grad=True)
        y = torch.randn(batch, species, device='cuda')
        weights = torch.randn(paths, channels, species, width, device='cuda')
        index = torch.arange(paths, device='cuda', dtype=torch.int32)
        values = torch.tensor([.5, .75], device='cuda')
        meta = {key: None for key in ('cg_indices', 'cg_values', 'cg_i_all', 'cg_j_all',
                                      'cg_k_all', 'cg_val_all', 'nnz_per_path',
                                      'K_per_path', 'path_offset')}
        meta.update(i_for_k=index, p_for_k=index, val_for_k=values, cg_val_0=.8,
                    P=paths, nnz0=2, U=channels, V=species, W=width,
                    K_total=paths, I_total=paths, can_use_empty_grad_x=True)
        selected = y.argmax(dim=1)
        x3 = x.reshape(batch, paths, channels)
        reference = torch.stack([
            torch.einsum('bu,buw->bw', x3[:, i], weights[i, :, selected, :].permute(1, 0, 2))
            * values[i] * .8 for i in range(paths)
        ], dim=1).reshape(batch, -1)
        self.check(fast_fctp(weights, x, y, meta), reference, (x,))
        torch.cuda.synchronize()
        print('fused_commutator, fused_bmm_sym and fast_fctp: forward/backward PASS')


if __name__ == '__main__':
    OtherOpsSmoke().run()
