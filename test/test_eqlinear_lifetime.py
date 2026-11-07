"""Prepared-plan lifetime regression; reference work finishes before measured enqueues.

Run: timeout --signal=TERM --kill-after=5s 120s /app/work/fasteq_venv/bin/python \
    -m pytest test/test_eqlinear_lifetime.py -q
"""
from dataclasses import dataclass

import pytest
import torch
import fasteq  # noqa: F401; registers torch.ops.eqlinear

from test_eqlinear import _out_rows, _reference, _rel_fro


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@dataclass
class LifetimeCase:
    precision: int
    direction: str
    dims: list
    batch: int = 512
    width: int = 96
    repeats: int = 6

    def prepare(self):
        dtype = torch.float64 if self.precision == 2 else torch.float32
        cg = [.01 + .001 * p for p in range(len(self.dims))]
        w = torch.randn(len(self.dims), self.width, self.width, device="cuda", dtype=dtype)
        input_rows = sum(self.dims) if self.direction == "forward" else _out_rows(self.dims)
        output_rows = _out_rows(self.dims) if self.direction == "forward" else sum(self.dims)
        inputs = [torch.randn(self.batch, input_rows, self.width, device="cuda", dtype=dtype)
                  for _ in range(self.repeats)]
        outputs = [torch.empty(self.batch, output_rows, self.width, device="cuda", dtype=dtype)
                   for _ in inputs]
        if self.direction == "forward":
            expected = [_reference(x, w, self.dims, cg, torch.float64) for x in inputs]
        else:
            x = torch.zeros(self.batch, sum(self.dims), self.width, device="cuda",
                            dtype=torch.float64, requires_grad=True)
            y = _reference(x, w, self.dims, cg, torch.float64)
            expected = [torch.autograd.grad(y, x, dy.double(), retain_graph=True)[0]
                        for dy in inputs]
        return w, cg, inputs, outputs, expected

    def create(self, w, cg):
        create = getattr(torch.ops.eqlinear, self.direction + "_create")
        return create(w, self.dims, cg, self.batch, self.width, len(self.dims), self.width,
                      self.precision)

    def enqueue(self, handle, inputs, outputs, delay=True):
        # A bounded delay keeps work pending while Python queues all six calls. No
        # reference, allocation, item(), event wait or synchronization follows here.
        if delay:
            torch.cuda._sleep(20_000_000)
        run = getattr(torch.ops.eqlinear, self.direction + "_run")
        for x, out in zip(inputs, outputs):
            run(handle, x, out)

    def destroy(self, handle):
        getattr(torch.ops.eqlinear, self.direction + "_destroy")(handle)


@pytest.mark.parametrize("precision,dims", [
    pytest.param(0, [1, 3, 5, 7], id="tf32-cutlass2"),
    pytest.param(0, [1, 1, 3, 3, 5], id="tf32-sm90"),
    pytest.param(1, [1, 3, 5, 7], id="fp32"),
    pytest.param(2, [1, 1, 3, 3, 5], id="fp64"),
])
@pytest.mark.parametrize("direction", ["forward", "dgrad"])
@pytest.mark.parametrize("stream_mode", ["default", "nondefault", "independent"])
def test_enqueue_destroy_then_synchronize(precision, dims, direction, stream_mode):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("eqlinear requires Hopper")
    torch.manual_seed(9023)
    case = LifetimeCase(precision, direction, dims)
    streams = ([torch.cuda.default_stream()] if stream_mode == "default" else
               [torch.cuda.Stream() for _ in range(2 if stream_mode == "independent" else 1)])
    prepared = [case.prepare() for _ in streams]
    # References, random input generation and output allocation all finish first.
    torch.cuda.synchronize()
    handles = []
    try:
        # Create every plan before enqueueing anything: later cudaMalloc calls must
        # not accidentally complete a previously enqueued plan's work.
        for stream, (w, cg, _, _, _) in zip(streams, prepared):
            with torch.cuda.stream(stream):
                handles.append(case.create(w, cg))
        for stream, handle, (_, _, inputs, outputs, _) in zip(streams, handles, prepared):
            with torch.cuda.stream(stream):
                case.enqueue(handle, inputs, outputs)
        # Destroy outside the nondefault contexts; current-stream fencing is insufficient.
        while handles:
            case.destroy(handles.pop())
        torch.cuda.synchronize()
        tolerance = 1e-12 if precision == 2 else 2e-5
        for _, _, _, outputs, expected in prepared:
            for out, ref in zip(outputs, expected):
                assert _rel_fro(out.double(), ref) <= tolerance
    finally:
        for handle in handles:
            case.destroy(handle)


def run_cross_device_smoke(plan_device, current_device):
    """Explicit two-GPU smoke; never collected/skipped by the default pytest suite."""
    assert plan_device != current_device, "select two distinct devices"
    assert max(plan_device, current_device) < torch.cuda.device_count()
    assert torch.cuda.get_device_capability(plan_device) == (9, 0), "plan device must be Hopper"
    for precision in (0, 1, 2):
        for direction in ("forward", "dgrad"):
            case = LifetimeCase(precision, direction, [1, 3, 5, 7])
            with torch.cuda.device(plan_device):
                w, cg, inputs, outputs, expected = case.prepare()
                stream = torch.cuda.Stream()
                torch.cuda.synchronize(plan_device)
            # Allocate wrong-device arguments before any test work is enqueued.
            with torch.cuda.device(current_device):
                wrong_input = torch.empty_like(inputs[0], device=f"cuda:{current_device}")
                wrong_output = torch.empty_like(outputs[0], device=f"cuda:{current_device}")
            handle = None
            try:
                with torch.cuda.device(plan_device), torch.cuda.stream(stream):
                    # The plan device's nondefault stream remains selected while the caller's
                    # current device differs. Every dispatcher guard must restore that caller.
                    with torch.cuda.device(current_device):
                        handle = case.create(w, cg)
                        assert torch.cuda.current_device() == current_device
                        run = getattr(torch.ops.eqlinear, direction + "_run")
                        with pytest.raises(RuntimeError, match="plan device"):
                            run(handle, wrong_input, outputs[0])
                        with pytest.raises(RuntimeError, match="plan device"):
                            run(handle, inputs[0], wrong_output)
                    torch.cuda._sleep(20_000_000)
                    with torch.cuda.device(current_device):
                        case.enqueue(handle, inputs, outputs, delay=False)
                        assert torch.cuda.current_device() == current_device
                        case.destroy(handle)
                        handle = None
                        assert torch.cuda.current_device() == current_device
                torch.cuda.synchronize(plan_device)
                tolerance = 1e-12 if precision == 2 else 2e-5
                for out, ref in zip(outputs, expected):
                    assert _rel_fro(out.double(), ref) <= tolerance
                # Cover one-shot entry points, which also select their allocation device.
                with torch.cuda.device(current_device):
                    one_shot = getattr(torch.ops.eqlinear, direction)
                    result = one_shot(inputs[0], w, case.dims, cg, precision)
                    assert torch.cuda.current_device() == current_device
                torch.cuda.synchronize(plan_device)
                assert _rel_fro(result.double(), expected[0]) <= tolerance
                if direction == "forward":
                    with torch.cuda.device(current_device):
                        handle = case.create(w, cg)
                        torch.ops.eqlinear.forward_run_flat(
                            handle, inputs[0].reshape(case.batch, -1),
                            outputs[0].reshape(case.batch, -1))
                        assert torch.cuda.current_device() == current_device
                        case.destroy(handle)
                        handle = None
                        assert torch.cuda.current_device() == current_device
                    torch.cuda.synchronize(plan_device)
                    assert _rel_fro(outputs[0].double(), expected[0]) <= tolerance
                print(f"precision={precision} {direction}: wrong-current create/run/destroy PASS",
                      flush=True)
            finally:
                if handle is not None:
                    case.destroy(handle)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run explicit two-GPU lifetime smoke")
    parser.add_argument("--plan-device", type=int, required=True)
    parser.add_argument("--current-device", type=int, required=True)
    args = parser.parse_args()
    run_cross_device_smoke(args.plan_device, args.current_device)
