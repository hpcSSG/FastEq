"""Original microbenchmarks, now using the installed FastEq CPU package."""
import argparse
import time
import torch
from e3nn import o3
from fasteq.cpu import load_extension
from fasteq.ops._autograd import NativeSHFunction, NativeFCTPFunction, NativeTPUFunction, NativeLinearFunction, PackedFCTPFunction, PackedLinearFunction, NativeSymmetricContractionFunction, NativeSymmetricContractionFusedFunction, NativeRadialMLPFunction
from fasteq.ops.metadata import prod, irreps_metadata, make_symmetric_contraction_metadata, make_radial_mlp_metadata, make_fctp_metadata, make_linear_metadata, make_tpu_metadata, make_packed_fctp_metadata, make_packed_linear_metadata
from fasteq.ops._profiling import set_profile_callback

def benchmark(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - start) * 1e3 / repeat


def error_metrics(reference, actual):
    delta = (reference - actual).abs()
    scale = reference.abs().max().clamp_min(1e-12)
    return delta.max().item(), (delta.max() / scale).item()


def benchmark_forward_backward(forward, x, grad_out, warmup, repeat):
    def run():
        leaf = x.detach().clone().requires_grad_(True)
        forward(leaf).backward(grad_out)

    return benchmark(run, warmup, repeat)


def print_benchmark_row(
    name,
    batch,
    threads,
    reference_ms,
    native_ms,
    reference_total_ms,
    native_total_ms,
    forward_error,
    forward_relative,
    gradient_error,
    gradient_relative,
):
    print(
        f"{name:<26} B={batch:5d} T={threads} "
        f"| fwd e3nn/native={reference_ms:8.4f}/{native_ms:8.4f} ms "
        f"({reference_ms / native_ms:5.2f}x) "
        f"| fwd+bwd={reference_total_ms:8.4f}/{native_total_ms:8.4f} ms "
        f"({reference_total_ms / native_total_ms:5.2f}x) "
        f"| err abs/rel={forward_error:.2e}/{forward_relative:.2e} "
        f"grad abs/rel={gradient_error:.2e}/{gradient_relative:.2e}"
    )


def print_benchmark_header():
    print("=" * 172)
    print(
        "All timings are ms/call. fwd is forward only; fwd+bwd includes "
        "autograd graph construction and grad_x."
    )
    print("=" * 172)


def benchmark_sh(ext, batch, threads, warmup, repeat):
    torch.manual_seed(0)
    x = torch.randn(batch, 3, dtype=torch.float32)
    reference = o3.SphericalHarmonics(
        [0, 1, 2, 3], normalize=True, normalization="integral"
    )

    x_ref = x.clone().requires_grad_(True)
    x_native = x.clone().requires_grad_(True)
    ref_out = reference(x_ref)
    native_out = NativeSHFunction.apply(x_native, ext)
    max_err, max_rel = error_metrics(ref_out, native_out)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    native_out.backward(grad_out)
    grad_err, grad_rel = error_metrics(x_ref.grad, x_native.grad)

    torch.set_num_threads(threads)
    ref_ms = benchmark(lambda: reference(x), warmup, repeat)

    def native_call():
        return ext.sh3_forward(x, True)

    native_ms = benchmark(native_call, warmup, repeat)
    ref_total_ms = benchmark_forward_backward(
        reference, x, grad_out, warmup, repeat
    )
    native_total_ms = benchmark_forward_backward(
        lambda value: NativeSHFunction.apply(value, ext),
        x,
        grad_out,
        warmup,
        repeat,
    )
    print_benchmark_row(
        "SH lmax=3",
        batch,
        threads,
        ref_ms,
        native_ms,
        ref_total_ms,
        native_total_ms,
        max_err,
        max_rel,
        grad_err,
        grad_rel,
    )


def benchmark_fctp(ext, batch, channels, elements, threads, warmup, repeat):
    torch.manual_seed(1)
    irreps_in = o3.Irreps(f"{channels}x0e + {channels}x1o + {channels}x2e")
    irreps_attr = o3.Irreps(f"{elements}x0e")
    module = o3.FullyConnectedTensorProduct(
        irreps_in,
        irreps_attr,
        irreps_in,
        internal_weights=True,
        shared_weights=True,
    ).to(dtype=torch.float32)
    x1 = torch.randn(batch, irreps_in.dim, dtype=torch.float32)
    element = torch.randint(0, elements, (batch,))
    x2 = torch.nn.functional.one_hot(element, elements).to(torch.float32)
    weight = module.weight.detach().clone()
    vstar = element.to(torch.int64)
    metadata = make_packed_fctp_metadata(module, weight)
    packed_weight = metadata["packed_weight"]

    x1_ref = x1.clone().requires_grad_(True)
    x1_native = x1.clone().requires_grad_(True)
    ref_out = module(x1_ref, x2, weight)
    native_out = PackedFCTPFunction.apply(
        x1_native, packed_weight, vstar, ext, metadata
    )
    max_err, max_rel = error_metrics(ref_out, native_out)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    native_out.backward(grad_out)
    grad_err, grad_rel = error_metrics(x1_ref.grad, x1_native.grad)

    torch.set_num_threads(threads)
    ref_ms = benchmark(lambda: module(x1, x2, weight), warmup, repeat)

    def native_call():
        return ext.fctp_onehot_packed_forward(
            x1,
            packed_weight,
            vstar,
            *metadata["args"],
            metadata["out_total"],
        )

    native_ms = benchmark(native_call, warmup, repeat)
    ref_total_ms = benchmark_forward_backward(
        lambda value: module(value, x2, weight), x1, grad_out, warmup, repeat
    )
    native_total_ms = benchmark_forward_backward(
        lambda value: PackedFCTPFunction.apply(
            value, packed_weight, vstar, ext, metadata
        ),
        x1,
        grad_out,
        warmup,
        repeat,
    )
    print_benchmark_row(
        f"FCTP C={channels} E={elements}",
        batch,
        threads,
        ref_ms,
        native_ms,
        ref_total_ms,
        native_total_ms,
        max_err,
        max_rel,
        grad_err,
        grad_rel,
    )


def benchmark_linear(ext, batch, channels, threads, warmup, repeat):
    torch.manual_seed(2)
    irreps = o3.Irreps(f"{channels}x0e + {channels}x1o + {channels}x2e")
    module = o3.Linear(
        irreps,
        irreps,
        instructions=[(0, 0), (1, 1), (2, 2)],
    ).to(dtype=torch.float32)
    x = torch.randn(batch, irreps.dim, dtype=torch.float32)
    weight = module.weight.detach().clone()
    metadata = make_packed_linear_metadata(module)
    packed_weight = metadata["packed_weight"]

    x_ref = x.clone().requires_grad_(True)
    x_native = x.clone().requires_grad_(True)
    ref_out = module(x_ref)
    native_out = PackedLinearFunction.apply(
        x_native, packed_weight, ext, metadata
    )
    max_err, max_rel = error_metrics(ref_out, native_out)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    native_out.backward(grad_out)
    grad_err, grad_rel = error_metrics(x_ref.grad, x_native.grad)

    torch.set_num_threads(threads)
    ref_ms = benchmark(lambda: module(x), warmup, repeat)

    def native_call():
        return ext.linear_packed_forward(
            x, packed_weight, *metadata["args"], metadata["out_total"]
        )

    native_ms = benchmark(native_call, warmup, repeat)
    ref_total_ms = benchmark_forward_backward(
        module, x, grad_out, warmup, repeat
    )
    native_total_ms = benchmark_forward_backward(
        lambda value: PackedLinearFunction.apply(
            value, packed_weight, ext, metadata
        ),
        x,
        grad_out,
        warmup,
        repeat,
    )
    print_benchmark_row(
        f"Linear C={channels}",
        batch,
        threads,
        ref_ms,
        native_ms,
        ref_total_ms,
        native_total_ms,
        max_err,
        max_rel,
        grad_err,
        grad_rel,
    )


def torch_symmetric_contraction3(U, weight, x, element):
    batch = x.shape[0]
    one_hot = torch.nn.functional.one_hot(
        element, weight.shape[0]
    ).to(x.dtype)
    return torch.einsum("wxyik,ekc,bci,be->bcwxy", U, weight, x, one_hot)


def benchmark_symmetric_contraction(
    ext, batch, channels, irreps, num_params, elements, threads, warmup, repeat
):
    torch.manual_seed(3)
    U = torch.randn(
        3, 3, 3, irreps, num_params, dtype=torch.float32
    )
    weight = torch.randn(elements, num_params, channels, dtype=torch.float32)
    x = torch.randn(batch, channels, irreps, dtype=torch.float32)
    element = torch.randint(0, elements, (batch,))

    x_ref = x.clone().requires_grad_(True)
    x_native = x.clone().requires_grad_(True)
    ref_out = torch_symmetric_contraction3(U, weight, x_ref, element)
    native_out = NativeSymmetricContractionFunction.apply(
        U, weight, x_native, element, ext
    )
    max_err, max_rel = error_metrics(ref_out, native_out)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    native_out.backward(grad_out)
    grad_err, grad_rel = error_metrics(x_ref.grad, x_native.grad)

    torch.set_num_threads(threads)
    ref_ms = benchmark(
        lambda: torch_symmetric_contraction3(U, weight, x, element),
        warmup,
        repeat,
    )

    def native_call():
        return ext.symmetric_contraction3_forward(U, weight, x, element)

    native_ms = benchmark(native_call, warmup, repeat)
    ref_total_ms = benchmark_forward_backward(
        lambda value: torch_symmetric_contraction3(
            U, weight, value, element
        ),
        x,
        grad_out,
        warmup,
        repeat,
    )
    native_total_ms = benchmark_forward_backward(
        lambda value: NativeSymmetricContractionFunction.apply(
            U, weight, value, element, ext
        ),
        x,
        grad_out,
        warmup,
        repeat,
    )
    print_benchmark_row(
        f"SC corr=3 C={channels} I={irreps} K={num_params}",
        batch,
        threads,
        ref_ms,
        native_ms,
        ref_total_ms,
        native_total_ms,
        max_err,
        max_rel,
        grad_err,
        grad_rel,
    )


def torch_fused_scalar_sc_reference(x, element, effective, C0, C1):
    wdim = C1.shape[-1]
    channels = x.shape[1]
    xw = x[..., :wdim]
    selected = effective[element]
    main = torch.einsum("bcim,bci->bcm", selected, x)
    ct = main + C0[element]
    ct = ct.reshape(x.shape[0], channels, wdim, wdim)
    out1 = torch.einsum("bcwx,bcw->bcx", ct, xw)
    ct1 = out1 + C1[element]
    return (ct1 * xw).sum(dim=-1)


def benchmark_fused_symmetric_contraction(
    ext, batch, channels, irreps, elements, threads, warmup, repeat
):
    torch.manual_seed(4)
    wdim = 3
    params = 23
    U3 = torch.randn(wdim, wdim, irreps, params, dtype=torch.float32)
    WmaxT = torch.randn(elements, channels, params, dtype=torch.float32)
    U3T = (
        U3.permute(2, 3, 0, 1)
        .reshape(irreps * params, wdim * wdim)
        .contiguous()
    )
    effective = torch.einsum(
        "eck,ikm->ecim",
        WmaxT,
        U3T.reshape(irreps, params, wdim * wdim),
    ).contiguous()
    C0 = torch.randn(elements, channels, wdim * wdim)
    C1 = torch.randn(elements, channels, wdim)
    metadata = {"effective": effective, "C0": C0, "C1": C1}
    x = torch.randn(batch, channels, irreps, dtype=torch.float32)
    element = torch.randint(0, elements, (batch,))

    x_ref = x.clone().requires_grad_(True)
    x_native = x.clone().requires_grad_(True)
    ref_out = torch_fused_scalar_sc_reference(
        x_ref, element, effective, C0, C1
    )
    native_out = NativeSymmetricContractionFusedFunction.apply(
        x_native, element, ext, metadata
    )
    max_err, max_rel = error_metrics(ref_out, native_out)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    native_out.backward(grad_out)
    grad_err, grad_rel = error_metrics(x_ref.grad, x_native.grad)

    ref_ms = benchmark(
        lambda: torch_fused_scalar_sc_reference(
            x, element, effective, C0, C1
        ),
        warmup,
        repeat,
    )

    def native_call():
        return ext.symmetric_contraction3_scalar_fused_forward(
            x, element, effective, C0, C1
        )[0]

    native_ms = benchmark(native_call, warmup, repeat)
    ref_total_ms = benchmark_forward_backward(
        lambda value: torch_fused_scalar_sc_reference(
            value, element, effective, C0, C1
        ),
        x,
        grad_out,
        warmup,
        repeat,
    )
    native_total_ms = benchmark_forward_backward(
        lambda value: NativeSymmetricContractionFusedFunction.apply(
            value, element, ext, metadata
        ),
        x,
        grad_out,
        warmup,
        repeat,
    )
    print_benchmark_row(
        f"SC fused C={channels} I={irreps} K={params}",
        batch,
        threads,
        ref_ms,
        native_ms,
        ref_total_ms,
        native_total_ms,
        max_err,
        max_rel,
        grad_err,
        grad_rel,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--irreps", type=int, default=16)
    parser.add_argument("--elements", type=int, default=10)
    parser.add_argument("--batches", nargs="+", type=int, default=[19, 172])
    parser.add_argument("--fused-sc", action="store_true")
    args = parser.parse_args()

    ext = load_extension()
    print(f"native extension: {ext.__file__}")
    print_benchmark_header()
    for threads in args.threads:
        torch.set_num_threads(threads)
        for batch in args.batches:
            benchmark_sh(
                ext, batch, threads, args.warmup, args.repeat
            )
            benchmark_fctp(
                ext,
                batch,
                args.channels,
                args.elements,
                threads,
                args.warmup,
                args.repeat,
            )
            benchmark_linear(
                ext,
                batch,
                args.channels,
                threads,
                args.warmup,
                args.repeat,
            )
            benchmark_symmetric_contraction(
                ext,
                batch,
                args.channels,
                args.irreps,
                23,
                args.elements,
                threads,
                args.warmup,
                args.repeat,
            )
            if args.fused_sc:
                benchmark_fused_symmetric_contraction(
                    ext,
                    batch,
                    args.channels,
                    args.irreps,
                    args.elements,
                    threads,
                    args.warmup,
                    args.repeat,
                )


if __name__ == "__main__":
    main()
