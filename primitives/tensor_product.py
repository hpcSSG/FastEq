# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math, time
import warnings
from functools import partial
from typing import List, Optional, OrderedDict, Tuple

import torch
import torch.fx

import cuequivariance as cue

logger = logging.getLogger(__name__)


def prod(numbers: List[int]):
    """
    This method is a workaround for script() not recognizing math.prod()
    """
    if torch.jit.is_scripting():
        product = 1
        for num in numbers:
            product *= num
        return product
    else:
        return math.prod(numbers)

def _my_tensor_product_fx(
    inputs: List[torch.Tensor],
    descriptor: cue.SegmentedTensorProduct,
    device: Optional[torch.device],
    math_dtype: Optional[torch.dtype],
) -> torch.nn.Module:
    """
    batch support of this function:
    - at least one input operand should have a batch dimension (ndim=2)
    - the output operand will have a batch dimension (ndim=2)
    """

    #descriptor = descriptor.remove_zero_paths()
    #descriptor = descriptor.remove_empty_segments()

    num_inputs = descriptor.num_operands - 1

    #operand_segment_offsets=[ [s.start for s in ope.segment_slices()] for ope in descriptor.operands]
    #operand_segment_shapes=[ope.segments for ope in descriptor.operands]
    coefficients_shape = []
    for _, path in enumerate(descriptor.paths):
        coefficients_shape.append(path.coefficients.shape)

    if path.coefficients.shape:
        dim = path.coefficients.shape[0]
    else:
        dim = 1

    if num_inputs > 0 and descriptor.num_paths > 0:

        constants = OrderedDict()

        operand_subscripts = [f"Z{ss}" for ss in descriptor.subscripts.operands]

        formula = (
            ",".join([descriptor.coefficient_subscripts] + operand_subscripts[:-1])
            + "->"
            + operand_subscripts[-1]
        )
        slices = [ope.segment_slices() for ope in descriptor.operands]

        outputs = []

        for path_idx, path in enumerate(descriptor.paths):
            segments = []
            for oid in range(num_inputs):
                seg_shape = descriptor.get_segment_shape(oid, path)
                inp = inputs[oid][..., slices[oid][path.indices[oid]]]
                if len(seg_shape) > 0:
                    inp = inp.reshape(inputs[oid].shape[:-1] + seg_shape)
                else:
                    inp = inp.reshape(inputs[oid].shape[:-1])
                segments.append(inp.to(dtype=math_dtype))
            

            c_tensor = disable_type_conv(
                torch.tensor(path.coefficients, dtype=math_dtype, device=device)
            )
            #out = torch.einsum(formula, c_tensor, *segments)
            segment0 = segments[0].squeeze(0)
            segment1 = segments[1].squeeze(1)
            out = torch.matmul(segment1, segment0) * c_tensor
            out.unsqueeze_(1)
            #print(f"out.shape:{out.shape}")   

            seg_shape = descriptor.get_segment_shape(-1, path)
            outputs += [
                out.reshape(out.shape[: out.ndim - len(seg_shape)] + (prod(seg_shape),))
            ]
        
        if len(outputs) == 0:
            raise NotImplementedError("No FX implementation for empty paths")

        def _sum(tensors, *, shape=None, like=None):
            if len(tensors) == 0:
                return like.new_zeros(shape)
            out = tensors[0]
            for t in tensors[1:]:
                out = torch.add(out, t)
            return out

        batch_shape = outputs[0].shape[:-1]
        output = torch.cat(
            [
                _sum(
                    [
                        out
                        for out, path in zip(outputs, descriptor.paths)
                        if path.indices[-1] == i
                    ],
                    shape=batch_shape + (prod(descriptor.operands[-1][i]),),
                    like=outputs[0],
                )
                for i in range(descriptor.operands[-1].num_segments)
            ],
            dim=-1,
        )


    else:
        raise NotImplementedError(
            "No FX implementation for empty paths and non-empty inputs"
        )
    return output

class TensorProduct(torch.nn.Module):
    """
    PyTorch module that computes the last operand of the segmented tensor product defined by the descriptor.

    Args:
        descriptor (SegmentedTensorProduct): The descriptor of the segmented tensor product.
        math_dtype (torch.dtype, optional): The data type of the coefficients and calculations.
        device (torch.device, optional): The device on which the calculations are performed.
        use_fallback (bool, optional):  Determines the computation method. If `None` (default), a CUDA kernel will be used if available. If `False`, a CUDA kernel will be used, and an exception is raised if it's not available. If `True`, a PyTorch fallback method is used regardless of CUDA kernel availability.

        Raises:
            RuntimeError: If `use_fallback` is `False` and no CUDA kernel is available.

    """

    def __init__(
        self,
        descriptor: cue.SegmentedTensorProduct,
        *,
        device: Optional[torch.device] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
        use_fasteq: bool = False,
        op_name: Optional[str] = "",
    ):
        super().__init__()
        self.descriptor = descriptor
        print(f"op_name:{op_name}, math_dtype:{math_dtype}")
        if math_dtype is None:
            math_dtype = torch.get_default_dtype()

        self.has_cuda = False
        self.f = None
        self.num_operands = descriptor.num_operands

        # ================= FastEq need =================
        from .utils import make_FastFullyConnectedTensorProductFunction, make_FastEquiLinearFunction, make_FastChannelWiseTensorProductFunction, make_FastFullyConnectedTensorProductPathFused
        self.use_fasteq = use_fasteq
        self.op_name = op_name
        
        if self.op_name == "tp_fully_connected" or self.op_name == "equi_linear":
                
            self.cg_indices: list[torch.Tensor] = []
            self.cg_values:  list[torch.Tensor] = []
            self.c_tensors:  list[torch.Tensor] = []
            dim_list = []
            device = "cuda"

            with torch.no_grad():
                for i, path in enumerate(descriptor.paths):
                    if getattr(path, "coefficients", None) is None or path.coefficients.ndim < 3:
                        continue
                    coeffs = torch.from_numpy(path.coefficients).to(device=device, dtype=math_dtype)

                    # idx: [nnz, 3] (i,j,k)；vals: [nnz]
                    idx = coeffs.nonzero(as_tuple=False)     
                    vals = coeffs[idx[:, 0], idx[:, 1], idx[:, 2]]
                    idx = idx.to(device=device, dtype=torch.int)
                    vals = vals.to(device=device, dtype=math_dtype)

                    dim_list.append(len(vals))

                    name_idx = f"cg_indices_{i}"
                    name_val = f"cg_values_{i}"
                    name_c   = f"c_tensors_{i}"

                    self.register_buffer(name_idx, idx, persistent=True)
                    self.register_buffer(name_val, vals, persistent=True)
                    self.register_buffer(name_c,  disable_type_conv(coeffs), persistent=True)

                    self.cg_indices.append(getattr(self, f"cg_indices_{i}"))
                    self.cg_values.append(getattr(self, f"cg_values_{i}"))
                    self.c_tensors.append(getattr(self, f"c_tensors_{i}"))
                
                if self.op_name == "tp_fully_connected":

                    dimensions_dict = self.descriptor.get_dimensions_dict()
                    self.U, self.V, self.W = sum(dimensions_dict['u']), sum(dimensions_dict['v']), sum(dimensions_dict['w'])
                
                    P = len(self.cg_indices)
                    assert P == len(dim_list)

                    self.K_per_path = torch.tensor(dim_list, device=device, dtype=torch.int32)
                    self.path_offset = torch.empty(P, device=device, dtype=torch.int32)
                    self.path_offset[0] = 0
                    if P > 1:
                        self.path_offset[1:] = torch.cumsum(self.K_per_path[:-1], dim=0)
                    self.K_total = int(self.K_per_path.sum().item())

                    self.nnz_list = [ci.shape[0] for ci in self.cg_indices]
                    self.nnz_max = max(self.nnz_list)
                    self.nnz_per_path = torch.tensor(self.nnz_list, device=device, dtype=torch.int32)

                    self.cg_i_all   = torch.zeros((P, self.nnz_max), device=device, dtype=torch.int32)
                    self.cg_j_all   = torch.zeros((P, self.nnz_max), device=device, dtype=torch.int32)
                    self.cg_k_all   = torch.zeros((P, self.nnz_max), device=device, dtype=torch.int32)
                    self.cg_val_all = torch.zeros((P, self.nnz_max), device=device, dtype=math_dtype)

                    for p in range(P):
                        ci_local = self.cg_indices[p]  # [nnz_p, 3], (i_local, j_local, k_local)
                        cv       = self.cg_values[p]         # [nnz_p]
                        nnz_p    = self.nnz_list[p]
                        offset_p = self.path_offset[p].item()

                        i_local = ci_local[:, 0]
                        j_local = ci_local[:, 1]
                        k_local = ci_local[:, 2]

                        # --- local -> global ---
                        i_global = i_local + offset_p
                        j_global = j_local             # TODO: only support j_local = j_global = 0
                        k_global = k_local + offset_p

                        self.cg_i_all[p, :nnz_p]   = i_global
                        self.cg_j_all[p, :nnz_p]   = j_global
                        self.cg_k_all[p, :nnz_p]   = k_global
                        self.cg_val_all[p, :nnz_p] = cv
                        
                        print(f"op_name:{self.op_name}")
                        self.FastFCTPFused = make_FastFullyConnectedTensorProductPathFused(
                            self.cg_i_all, self.cg_j_all, self.cg_k_all, self.cg_val_all,
                            self.nnz_per_path, self.K_per_path, self.path_offset, 
                            self.U, self.V, self.W, self.K_total
                        )
                        self.FastFCTPFunc = make_FastFullyConnectedTensorProductFunction()
        
        if self.op_name == "equi_linear":
            self.FastEquiLinearFunction = make_FastEquiLinearFunction()
        if self.op_name == "tp_channel_wise":
            self.FastCWTPFunc = make_FastChannelWiseTensorProductFunction()
        # ================================================

        if use_fallback is False:
            self.f = _tensor_product_cuda(descriptor, device, math_dtype)
            self.has_cuda = True
        elif use_fallback is None:
            try:
                self.f = _tensor_product_cuda(descriptor, device, math_dtype)
                self.has_cuda = True
            except NotImplementedError as e:
                logger.info(f"CUDA implementation not available: {e}")
            except ImportError as e:
                logger.warning(f"CUDA implementation not available: {e}")
                logger.warning(
                    "Did you forget to install the CUDA version of cuequivariance-ops-torch?\n"
                    "Install it with one of the following commands:\n"
                    "pip install cuequivariance-ops-torch-cu11\n"
                    "pip install cuequivariance-ops-torch-cu12"
                )

        if self.f is None:
            self.f = _tensor_product_fx(descriptor, device, math_dtype, True)

        self.f = _Wrapper(self.f, descriptor)

        self.operands_dims = [ope.size for ope in descriptor.operands]

    @torch.jit.ignore
    def __repr__(self):
        has_cuda_kernel = (
            "(with CUDA kernel)" if self.has_cuda else "(without CUDA kernel)"
        )
        return f"TensorProduct({self.descriptor} {has_cuda_kernel})"

    def forward(
        self,
        x0: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        x2: Optional[torch.Tensor] = None,
        x3: Optional[torch.Tensor] = None,
        x4: Optional[torch.Tensor] = None,
        x5: Optional[torch.Tensor] = None,
        x6: Optional[torch.Tensor] = None,
    ):
        r"""
        Perform the tensor product based on the specified descriptor.

        Args:
            x0, x1[, x2, x3, x4, x5, x6]: The input tensors. The number of input tensors should match the number of operands in the descriptor minus one.
                Each input tensor should have a shape of (batch, operand_size) or (1, operand_size)
                where `operand_size` corresponds to the size of each operand as defined in
                the tensor product descriptor.

        Returns:
            torch.Tensor:
                The output tensor resulting from the tensor product.
                It has a shape of (batch, last_operand_size), where
                `last_operand_size` is the size of the last operand in the descriptor.
        """

        if (
            x6 is not None
            and x5 is not None
            and x4 is not None
            and x3 is not None
            and x2 is not None
            and x1 is not None
        ):
            inputs = [x0, x1, x2, x3, x4, x5, x6]
        elif (
            x5 is not None
            and x4 is not None
            and x3 is not None
            and x2 is not None
            and x1 is not None
        ):
            inputs = [x0, x1, x2, x3, x4, x5]
        elif x4 is not None and x3 is not None and x2 is not None and x1 is not None:
            inputs = [x0, x1, x2, x3, x4]
        elif x3 is not None and x2 is not None and x1 is not None:
            inputs = [x0, x1, x2, x3]
        elif x2 is not None and x1 is not None:
            inputs = [x0, x1, x2]
        elif x1 is not None:
            inputs = [x0, x1]
        else:
            inputs = [x0]

        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            if len(inputs) != self.num_operands - 1:
                raise ValueError(
                    f"Expected {self.num_operands - 1} input tensors, got {len(inputs)}"
                )
            for oid, input in enumerate(inputs):
                torch._assert(
                    input.ndim == 2,
                    f"input {oid} should have ndim=2",
                )
                torch._assert(
                    input.shape[1] == self.operands_dims[oid],
                    f"input {oid} should have shape (batch, {self.operands_dims[oid]}), got {input.shape}",
                )
        
        if self.use_fasteq:
            if self.op_name == "tp_fully_connected":
                #print("== call fasteq fully connect tensor product ==")
                
                '''
                torch.cuda.synchronize()
                start_time = time.perf_counter() * 1000
                
                out = self.FastFCTPFunc.apply(
                    inputs[0], inputs[1], inputs[2],
                    self.descriptor,
                    self.cg_indices,
                    self.cg_values,
                    inputs[0].dtype,
                )

                torch.cuda.synchronize()
                end_time = time.perf_counter() * 1000
                execution_time_ms = end_time - start_time
                print(f"<< fasteq fctp cost: {execution_time_ms:.3f} ms >>")
                '''

                #print(f"inputs[0].shape:{inputs[0].shape}, inputs[1].shape:{inputs[1].shape}, inputs[2].shape:{inputs[2].shape}, self.K_total:{self.K_total}")
                #print(f"self.U={self.U},  self.V={self.V}, self.W={self.W}, descriptor={self.descriptor.get_dimensions_dict()}")
                #print(f"inputs[0]:{inputs[0].dtype}, inputs[1]:{inputs[1].dtype}, inputs[2]:{inputs[2].dtype}, cg_val_all.dtype:{self.cg_val_all.dtype}")

                out = self.FastFCTPFused.apply(
                     inputs[0], inputs[1], inputs[2], 
                )


            elif self.op_name == "tp_channel_wise":
                #print("== call fasteq channel-wise tensor product ==")
                #print(f"inputs[0].shape:{inputs[0].shape}, inputs[1].shape:{inputs[1].shape}, inputs[2].shape:{inputs[2].shape}")
                out = self.FastCWTPFunc.apply(inputs[0], inputs[1], inputs[2])
            # TODO fix 
            elif self.op_name == "equi_linear" and (tuple(inputs[0].shape) == (1, 36864)):
                #print("== call fasteq equi-linear tensor product ==")
                dtype = inputs[0].dtype
                w = inputs[0].to(torch.float64)
                x = inputs[1].to(torch.float64)

                out = self.FastEquiLinearFunction.apply(w, x, self.descriptor)
                out = out.to(dtype)

            elif self.op_name == "equi_linear" and (tuple(inputs[0].shape) == (1, 9216)) and inputs[1].shape[1] == 96:
                #print(f"==== call my matmul linear ====")
                weight = inputs[0].reshape(96, 96)
                out = torch.matmul(inputs[1], weight) * 0.10206207261596577
                #out = _my_tensor_product_fx(inputs, self.descriptor, "cuda", torch.float64)
            else:
                out = self.f(inputs)
        else:
            out = self.f(inputs)
        
        return out


def to_notypeconv(t, *args, **kwargs):
    new_kwargs = kwargs.copy()
    new_kwargs.pop("dtype", None)
    new_args = [None if isinstance(a, torch.dtype) else a for a in args]
    result = t.__original_to(*new_args, **new_kwargs)
    return result


def disable_type_conv(t):
    """
    This modifier can be used on Tensors or whole Modules
    to prevent them from being modified during to(dtype=x) calls
    """
    t.__original_to = t.to
    t.to = partial(to_notypeconv, t)
    return t


def _tensor_product_fx(
    descriptor: cue.SegmentedTensorProduct,
    device: Optional[torch.device],
    math_dtype: torch.dtype,
    optimize_einsums: bool,
) -> torch.nn.Module:
    """
    batch support of this function:
    - at least one input operand should have a batch dimension (ndim=2)
    - the output operand will have a batch dimension (ndim=2)
    """
    descriptor = descriptor.remove_zero_paths()
    descriptor = descriptor.remove_empty_segments()

    num_inputs = descriptor.num_operands - 1

    if num_inputs > 0 and descriptor.num_paths > 0:
        graph = torch.fx.Graph()
        tracer = torch.fx.proxy.GraphAppendingTracer(graph)
        constants = OrderedDict()

        inputs = [
            torch.fx.Proxy(graph.placeholder(f"input_{i}"), tracer)
            for i in range(num_inputs)
        ]

        operand_subscripts = [f"Z{ss}" for ss in descriptor.subscripts.operands]

        formula = (
            ",".join([descriptor.coefficient_subscripts] + operand_subscripts[:-1])
            + "->"
            + operand_subscripts[-1]
        )
        slices = [ope.segment_slices() for ope in descriptor.operands]

        outputs = []
        for path_idx, path in enumerate(descriptor.paths):
            segments = []
            for oid in range(num_inputs):
                seg_shape = descriptor.get_segment_shape(oid, path)
                inp = inputs[oid][..., slices[oid][path.indices[oid]]]
                if len(seg_shape) > 0:
                    inp = inp.reshape(inputs[oid].shape[:-1] + seg_shape)
                else:
                    inp = inp.reshape(inputs[oid].shape[:-1])

                segments.append(inp.to(dtype=math_dtype))

            c_tensor = disable_type_conv(
                torch.tensor(path.coefficients, dtype=math_dtype, device=device)
            )
            constants[f"c{path_idx}"] = c_tensor

            c = torch.fx.Proxy(graph.get_attr(f"c{path_idx}"), tracer=tracer).clone()
            out = torch.einsum(formula, c, *segments)
            out = out.to(dtype=inputs[0].dtype)

            seg_shape = descriptor.get_segment_shape(-1, path)
            outputs += [
                out.reshape(out.shape[: out.ndim - len(seg_shape)] + (prod(seg_shape),))
            ]

        if len(outputs) == 0:
            raise NotImplementedError("No FX implementation for empty paths")

        def _sum(tensors, *, shape=None, like=None):
            if len(tensors) == 0:
                return like.new_zeros(shape)
            out = tensors[0]
            for t in tensors[1:]:
                out = torch.add(out, t)
            return out

        batch_shape = outputs[0].shape[:-1]
        output = torch.cat(
            [
                _sum(
                    [
                        out
                        for out, path in zip(outputs, descriptor.paths)
                        if path.indices[-1] == i
                    ],
                    shape=batch_shape + (prod(descriptor.operands[-1][i]),),
                    like=outputs[0],
                )
                for i in range(descriptor.operands[-1].num_segments)
            ],
            dim=-1,
        )

        graph.output(output.node)

        graph.lint()
        constants_root = torch.nn.Module()
        for key, value in constants.items():
            constants_root.register_buffer(key, value)
        graphmod = torch.fx.GraphModule(constants_root, graph)

        if optimize_einsums:
            try:
                import opt_einsum_fx
            except ImportError:
                logger.warning(
                    "opt_einsum_fx not available.\n"
                    "To use the optimization, please install opt_einsum_fx.\n"
                    "pip install opt_einsum_fx"
                )
            else:
                example_inputs = [
                    torch.zeros((10, operand.size))
                    for operand in descriptor.operands[:num_inputs]
                ]
                graphmod = opt_einsum_fx.optimize_einsums_full(graphmod, example_inputs)
    elif num_inputs == 0:

        class _no_input(torch.nn.Module):
            def __init__(self, descriptor: cue.SegmentedTensorProduct):
                super().__init__()

                for pid, path in enumerate(descriptor.paths):
                    self.register_buffer(
                        f"c{pid}",
                        torch.tensor(
                            path.coefficients, dtype=math_dtype, device=device
                        ),
                    )

            def forward(self):
                output = torch.zeros(
                    (descriptor.operands[-1].size,), device=device, dtype=math_dtype
                )
                for pid in range(descriptor.num_paths):
                    output += torch.einsum(
                        descriptor.coefficient_subscripts
                        + "->"
                        + descriptor.operands[0].subscripts,
                        getattr(self, f"c{pid}"),
                    )
                return output

        graphmod = _no_input(descriptor)

    else:
        raise NotImplementedError(
            "No FX implementation for empty paths and non-empty inputs"
        )

    return graphmod


class _Caller(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module


class _NoArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module()


class _OneArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0])


class _TwoArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0], args[1])


class _ThreeArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0], args[1], args[2])


class _FourArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0], args[1], args[2], args[3])


class _FiveArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0], args[1], args[2], args[3], args[4])


class _SixArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(args[0], args[1], args[2], args[3], args[4], args[5])


class _SevenArgCaller(_Caller):
    def forward(self, args: List[torch.Tensor]):
        return self.module(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )


CALL_DISPATCHERS = [
    _NoArgCaller,
    _OneArgCaller,
    _TwoArgCaller,
    _ThreeArgCaller,
    _FourArgCaller,
    _FiveArgCaller,
    _SixArgCaller,
    _SevenArgCaller,
]


class _Wrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, descriptor: cue.SegmentedTensorProduct):
        super().__init__()
        self.module = CALL_DISPATCHERS[descriptor.num_operands - 1](module)
        self.descriptor = descriptor

    def forward(self, args: List[torch.Tensor]):
        return self.module(args)


def _tensor_product_cuda(
    descriptor: cue.SegmentedTensorProduct,
    device: Optional[torch.device],
    math_dtype: torch.dtype,
) -> torch.nn.Module:
    logger.debug(f"Starting search for a cuda kernel for {descriptor}")

    if descriptor.num_paths == 0:
        raise NotImplementedError("No cuda kernel for empty paths.")

    if descriptor.num_operands not in (3, 4):
        raise NotImplementedError(
            "Only descriptors with 3 or 4 operands are supported."
            f" Got {descriptor.subscripts}."
        )

    if not torch.cuda.is_available():
        raise NotImplementedError("CUDA is not available.")

    # Dispatch strategy:
    # 1. try to use TensorProductUniform4x1d
    # 2. try to use FusedTensorProductOp3 or FusedTensorProductOp4

    if math_dtype in [torch.float32, torch.float64]:
        d = descriptor
        d = d.flatten_coefficient_modes(force=True)
        d = d.squeeze_modes()
        if len(d.subscripts.modes()) == 1:
            d = d.canonicalize_subscripts()
            dims = d.get_dims("u")
            d = d.split_mode("u", math.gcd(*dims))
            u = next(iter(d.get_dims("u")))

            import cuequivariance_ops_torch as ops

            if ops.TensorProductUniform1d.is_supported(
                operand_dim=[o.ndim for o in d.operands],
                operand_extent=u,
                operand_num_segments=[o.num_segments for o in d.operands],
            ):
                if descriptor.num_operands == 3:
                    return TensorProductUniform3x1d(d, device, math_dtype)
                else:
                    return TensorProductUniform4x1d(d, device, math_dtype)

    supported_targets = [
        cue.segmented_polynomials.Subscripts(subscripts)
        for subscripts in [
            "u__uw_w",
            "_v_vw_w",
            "u_v_uv_u",
            "u_v_uv_v",
            "u_u_uw_w",
            "u_v_uvw_w",
            "u_u_u",
            "u_v_uv",
            "u_uv_v",
            "u__u",
            "_v_v",
        ]
    ]

    try:
        descriptor, perm = next(
            cue.segmented_polynomials.dispatch(
                descriptor, supported_targets, "permute_all_but_last"
            )
        )
    except StopIteration:
        raise NotImplementedError(
            f"No cuda kernel found for {descriptor}."
            " Supported targets are: " + ", ".join(str(t) for t in supported_targets)
        )

    if descriptor.num_operands == 3:
        return FusedTensorProductOp3(descriptor, perm[:2], device, math_dtype)
    elif descriptor.num_operands == 4:
        return FusedTensorProductOp4(descriptor, perm[:3], device, math_dtype)


def _permutation_module(permutation: Tuple[int, ...]):
    graph = torch.fx.Graph()
    inputs = [graph.placeholder(f"input_{i}") for i in range(len(permutation))]
    graph.output([inputs[i] for i in permutation])
    return torch.fx.GraphModule(dict(), graph, class_name="perm")


class FusedTensorProductOp3(torch.nn.Module):
    def __init__(
        self,
        descriptor: cue.SegmentedTensorProduct,
        perm: Tuple[int, int],
        device: Optional[torch.device],
        math_dtype: torch.dtype,
    ):
        super().__init__()

        self._perm = _permutation_module(perm)
        self.descriptor = descriptor.permute_operands(
            [perm.index(i) for i in range(2)] + [2]
        )

        if math_dtype not in [torch.float32, torch.float64]:
            warnings.warn(
                "cuequivariance_ops_torch.FusedTensorProductOp3 only supports math_dtype==float32 or math_dtype==float64"
            )

        import cuequivariance_ops_torch as ops

        self._f = ops.FusedTensorProductOp3(
            operand_segment_modes=descriptor.subscripts.operands,
            operand_segment_offsets=[
                [s.start for s in ope.segment_slices()] for ope in descriptor.operands
            ],
            operand_segment_shapes=[ope.segments for ope in descriptor.operands],
            path_indices=descriptor.indices,
            path_coefficients=descriptor.stacked_coefficients,
            math_dtype=math_dtype,
        ).to(device=device)

    @torch.jit.ignore
    def __repr__(self) -> str:
        return f"FusedTensorProductOp3({self.descriptor} (output last operand))"

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        x0, x1 = self._perm(x0, x1)

        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            logger.debug(
                f"Calling FusedTensorProductOp3: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}"
            )

        torch._assert(x0.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x1.ndim == 2, "input should be (batch, dim) or (1, dim)")

        return self._f(x0, x1)


class FusedTensorProductOp4(torch.nn.Module):
    def __init__(
        self,
        descriptor: cue.SegmentedTensorProduct,
        perm: Tuple[int, int, int],
        device: Optional[torch.device],
        math_dtype: torch.dtype,
    ):
        super().__init__()

        self._perm = _permutation_module(perm)
        self.descriptor = descriptor.permute_operands(
            [perm.index(i) for i in range(3)] + [3]
        )

        if math_dtype not in [torch.float32, torch.float64]:
            warnings.warn(
                "cuequivariance_ops_torch.FusedTensorProductOp4 only supports math_dtype==float32 or math_dtype==float64"
            )

        import cuequivariance_ops_torch as ops

        self._f = ops.FusedTensorProductOp4(
            operand_segment_modes=descriptor.subscripts.operands,
            operand_segment_offsets=[
                [s.start for s in ope.segment_slices()] for ope in descriptor.operands
            ],
            operand_segment_shapes=[ope.segments for ope in descriptor.operands],
            path_indices=descriptor.indices,
            path_coefficients=descriptor.stacked_coefficients,
            math_dtype=math_dtype,
        ).to(device=device)

    @torch.jit.ignore
    def __repr__(self) -> str:
        return f"FusedTensorProductOp4({self.descriptor} (output last operand))"

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        x0, x1, x2 = self._perm(x0, x1, x2)

        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            logger.debug(
                f"Calling FusedTensorProductOp4: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}, {x2.shape}"
            )

        torch._assert(x0.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x1.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x2.ndim == 2, "input should be (batch, dim) or (1, dim)")

        return self._f(x0, x1, x2)


class TensorProductUniform1d(torch.nn.Module):
    def __init__(
        self,
        descriptor: cue.SegmentedTensorProduct,
        device: Optional[torch.device],
        math_dtype: torch.dtype,
    ):
        super().__init__()
        import cuequivariance_ops_torch as ops

        self.descriptor = descriptor

        assert len(descriptor.subscripts.modes()) == 1
        assert descriptor.all_same_segment_shape()
        assert descriptor.coefficient_subscripts == ""
        u = next(iter(descriptor.get_dims(descriptor.subscripts.modes()[0])))

        self._f = ops.TensorProductUniform1d(
            operand_dim=[ope.ndim for ope in descriptor.operands],
            operand_extent=u,
            operand_num_segments=[ope.num_segments for ope in descriptor.operands],
            path_indices=[path.indices for path in descriptor.paths],
            path_coefficients=[float(path.coefficients) for path in descriptor.paths],
            math_dtype=math_dtype,
        ).to(device=device)


class TensorProductUniform3x1d(TensorProductUniform1d):
    @torch.jit.ignore
    def __repr__(self):
        return f"TensorProductUniform3x1d({self.descriptor} (output last operand))"

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            logger.debug(
                f"Calling TensorProductUniform3x1d: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}"
            )
        torch._assert(x0.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x1.ndim == 2, "input should be (batch, dim) or (1, dim)")

        # ops.TensorProductUniform1d expects inputs
        # of shape (Z, dim) or (1, dim)
        return self._f(x0, x1)


class TensorProductUniform4x1d(TensorProductUniform1d):
    @torch.jit.ignore
    def __repr__(self):
        return f"TensorProductUniform4x1d({self.descriptor} (output last operand))"

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            logger.debug(
                f"Calling TensorProductUniform4x1d: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}, {x2.shape}"
            )
        torch._assert(x0.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x1.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x2.ndim == 2, "input should be (batch, dim) or (1, dim)")

        # ops.TensorProductUniform1d expects inputs
        # of shape (Z, dim) or (1, dim)
        return self._f(x0, x1, x2)


class TensorProductUniform3x1dIndexed(torch.nn.Module):
    def __init__(
        self,
        descriptor: cue.SegmentedTensorProduct,
        device: Optional[torch.device],
        math_dtype: torch.dtype,
    ):
        super().__init__()
        import cuequivariance_ops_torch as ops

        self.descriptor = descriptor

        assert len(descriptor.subscripts.modes()) == 1
        assert descriptor.all_same_segment_shape()
        assert descriptor.coefficient_subscripts == ""
        u = next(iter(descriptor.get_dims(descriptor.subscripts.modes()[0])))

        self._f = ops.TensorProductUniform3x1dIndexed(
            operand_dim=[ope.ndim for ope in descriptor.operands],
            operand_extent=u,
            operand_num_segments=[ope.num_segments for ope in descriptor.operands],
            path_indices=[path.indices for path in descriptor.paths],
            path_coefficients=[float(path.coefficients) for path in descriptor.paths],
            math_dtype=math_dtype,
        ).to(device=device)

    @torch.jit.ignore
    def __repr__(self):
        return (
            f"TensorProductUniform3x1dIndexed({self.descriptor} (output last operand))"
        )

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        op_idx0: Optional[torch.Tensor],
        op_idx1: Optional[torch.Tensor],
        op_idx_out: Optional[torch.Tensor],
        num_output_rows: int,
    ) -> torch.Tensor:
        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            logger.debug(
                f"Calling TensorProductUniform3x1d: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}"
            )
        torch._assert(x0.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x1.ndim == 2, "input should be (batch, dim) or (1, dim)")

        # ops.TensorProductUniform1d expects inputs
        # of shape (Z, dim) or (1, dim)
        return self._f(x0, x1, op_idx0, op_idx1, op_idx_out, num_output_rows)


class TensorProductUniform4x1dIndexed(torch.nn.Module):
    def __init__(
        self,
        descriptor: cue.SegmentedTensorProduct,
        device: Optional[torch.device],
        math_dtype: torch.dtype,
    ):
        super().__init__()
        import cuequivariance_ops_torch as ops

        self.descriptor = descriptor

        assert len(descriptor.subscripts.modes()) == 1
        assert descriptor.all_same_segment_shape()
        assert descriptor.coefficient_subscripts == ""
        u = next(iter(descriptor.get_dims(descriptor.subscripts.modes()[0])))

        self._f = ops.TensorProductUniform4x1dIndexed(
            operand_dim=[ope.ndim for ope in descriptor.operands],
            operand_extent=u,
            operand_num_segments=[ope.num_segments for ope in descriptor.operands],
            path_indices=[path.indices for path in descriptor.paths],
            path_coefficients=[float(path.coefficients) for path in descriptor.paths],
            math_dtype=math_dtype,
        ).to(device=device)

    @torch.jit.ignore
    def __repr__(self):
        return (
            f"TensorProductUniform4x1dIndexed({self.descriptor} (output last operand))"
        )

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        op_idx0: Optional[torch.Tensor],
        op_idx1: Optional[torch.Tensor],
        op_idx2: Optional[torch.Tensor],
        op_idx_out: Optional[torch.Tensor],
        num_output_rows,
    ) -> torch.Tensor:
        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and not torch.compiler.is_compiling()
        ):
            logger.debug(
                f"Calling TensorProductUniform4x1d: {self.descriptor}, input shapes: {x0.shape}, {x1.shape}"
            )
        torch._assert(x0.ndim == 2, "input should be (batch, dim) or (1, dim)")
        torch._assert(x1.ndim == 2, "input should be (batch, dim) or (1, dim)")

        # ops.TensorProductUniform1d expects inputs
        # of shape (Z, dim) or (1, dim)
        return self._f(
            x0, x1, x2, op_idx0, op_idx1, op_idx2, op_idx_out, num_output_rows
        )
