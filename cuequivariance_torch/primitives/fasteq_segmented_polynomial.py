# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by mlx in 2025
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

import warnings
from typing import Dict, List, Optional, OrderedDict, Tuple, Any

import torch
import torch.nn as nn
from cuequivariance_torch.primitives.segmented_polynomial_fused_tp import (
    SegmentedPolynomialFusedTP,
)
from cuequivariance_torch.primitives.segmented_polynomial_indexed_linear import (
    SegmentedPolynomialIndexedLinear,
)
from cuequivariance_torch.primitives.segmented_polynomial_naive import (
    SegmentedPolynomialNaive,
)
from cuequivariance_torch.primitives.segmented_polynomial_uniform_1d import (
    SegmentedPolynomialFromUniform1dJit,
)

import cuequivariance as cue

try:
    import cuequivariance_ops_torch  # noqa: F401

    HAS_CUE_OPS = True
except ImportError:
    HAS_CUE_OPS = False

from fasteq.ops.equi_linear import fast_equi_linear
from fasteq.ops.stc import fast_stc
from fasteq.ops.cwtp import fast_cwtp
from fasteq.ops.mptp import fast_mptp
from fasteq.ops.fctp import fast_fctp

@torch.no_grad()
def infer_cwtp_meta(
    descriptor,
    math_dtype,
    device,
    max_k_dim_for_kernel: int = 8, # TODO: make it flexible
) -> Dict[str, Any]:
    """
    生成 ChannelWise TP 的所有 meta 信息：
      - 分段信息：uv / iu / jv / kv offsets + slices
      - dense c_tensors + (i/j/k_dims, c_offsets, c_all)
      - sparse CG 信息：
        * cg_i_all, cg_j_all, cg_k_all, cg_val_all
        * nnz_per_path, nnz_offsets
      - 按 k 分组的 sparse meta：
        * nnz_k_offsets: [P, MAX_K_DIM]
        * nnz_k_counts:  [P, MAX_K_DIM]
    """

    # -------------------- 1) paths & dense c_tensors --------------------
    path_indices: List[Tuple[int, int, int, int]] = []
    c_tensors: List[torch.Tensor] = []

    for path_idx, path in enumerate(descriptor.paths):
        path_indices.append(tuple(path.indices))

        c_tensor = torch.tensor(path.coefficients, dtype=math_dtype, device=device).contiguous()
        c_tensors.append(c_tensor)

    # U/V
    u = list(descriptor.get_dims("u"))[0]
    v = list(descriptor.get_dims("v"))[0]

    UV_TOTAL = int(descriptor.operands[0].size)
    IU_TOTAL = int(descriptor.operands[1].size)
    JV_TOTAL = int(descriptor.operands[2].size)

    print(f"UV_TOTAL:{UV_TOTAL}, IU_TOTAL:{IU_TOTAL}, JV_TOTAL:{JV_TOTAL}")
    P = len(path_indices)

    # segment counts
    uv_seg_count = max(p[0] for p in path_indices) + 1
    iu_seg_count = max(p[1] for p in path_indices) + 1
    jv_seg_count = max(p[2] for p in path_indices) + 1
    kv_seg_count = max(p[3] for p in path_indices) + 1

    # -------------------- 2) i/j/k dims + c_offsets + c_all --------------------
    i_dims, j_dims, k_dims = [], [], []
    c_offsets = [0]
    c_flat_list = []
    running = 0

    for c in c_tensors:
        i_dim, j_dim, k_dim = map(int, c.shape)
        i_dims.append(i_dim)
        j_dims.append(j_dim)
        k_dims.append(k_dim)

        c_flat = c.reshape(-1).contiguous()          # original layout (i,j,k) flatten => ((i*J + j)*K + k)
        c_flat_list.append(c_flat)

        running += i_dim * j_dim * k_dim
        c_offsets.append(running)

    c_all = (
        torch.cat(c_flat_list, dim=0)
        if len(c_flat_list) > 0
        else torch.empty(0, dtype=math_dtype, device=device)
    )

    max_k_dim = max(k_dims) if len(k_dims) > 0 else 0
    assert max_k_dim <= max_k_dim_for_kernel, \
        f"max k_dim {max_k_dim} > kernel MAX_K_DIM {max_k_dim_for_kernel}"

    # -------------------- 3) uv slices + uv_seg_offsets --------------------
    uv_slices = []
    uv_seg_offsets = []
    start = 0
    uv_stride = int(u * v)
    for _ in range(uv_seg_count):
        uv_seg_offsets.append(start)
        end = start + uv_stride
        uv_slices.append(slice(start, end))
        start = end
    assert start == UV_TOTAL, f"UV total {start} != {UV_TOTAL}"

    # -------------------- 4) iu slices + iu_seg_offsets --------------------
    iu_slices = []
    iu_seg_offsets = []
    start = 0
    for s in range(iu_seg_count):
        iu_seg_offsets.append(start)
        idx = next(idx for idx, p in enumerate(path_indices) if p[1] == s)
        i_dim = i_dims[idx]
        length = int(i_dim * u)
        end = start + length
        iu_slices.append(slice(start, end))
        start = end
    assert start == IU_TOTAL, f"IU total {start} != {IU_TOTAL}"

    # -------------------- 5) jv slices + jv_seg_offsets --------------------
    jv_slices = []
    jv_seg_offsets = []
    start = 0
    for s in range(jv_seg_count):
        jv_seg_offsets.append(start)
        idx = next(idx for idx, p in enumerate(path_indices) if p[2] == s)
        j_dim = j_dims[idx]
        length = int(j_dim * v)
        end = start + length
        jv_slices.append(slice(start, end))
        start = end
    assert start == JV_TOTAL, f"JV total {start} != {JV_TOTAL}"

    # -------------------- 6) K offsets: kv_k_offsets --------------------
    kv_k_offsets = []
    start = 0
    for s in range(kv_seg_count):
        kv_k_offsets.append(start)
        idx = next(idx for idx, p in enumerate(path_indices) if p[3] == s)
        k_dim = k_dims[idx]
        start += int(k_dim)
    K_TOTAL = int(start)

    # -------------------- 7) sparse CG meta + per-k grouping --------------------
    cg_i_list, cg_j_list, cg_k_list, cg_val_list = [], [], [], []
    nnz_per_path = []
    nnz_offsets = []  # NOTE: this is "start offset per path", length P (kept as your original)

    nnz_k_offsets_list = []  # [P, MAX_K_DIM]
    nnz_k_counts_list  = []  # [P, MAX_K_DIM]

    nnz_running = 0
    for path_id, c in enumerate(c_tensors):
        i_dim, j_dim, k_dim = map(int, c.shape)

        nz_idx = torch.nonzero(c != 0, as_tuple=False)  # [nnz,3] (i,j,k)
        nnz = int(nz_idx.size(0))

        nnz_per_path.append(nnz)
        nnz_offsets.append(nnz_running)
        nnz_running += nnz

        local_k_offsets = torch.zeros(max_k_dim_for_kernel, dtype=torch.int32, device=device)
        local_k_counts  = torch.zeros(max_k_dim_for_kernel, dtype=torch.int32, device=device)

        if nnz > 0:
            sort_idx = torch.argsort(nz_idx[:, 2])  # sort by k
            nz_sorted = nz_idx[sort_idx]
            i_idx = nz_sorted[:, 0]
            j_idx = nz_sorted[:, 1]
            k_idx = nz_sorted[:, 2]

            vals = c[i_idx, j_idx, k_idx]

            cg_i_list.append(i_idx.to(torch.uint8))
            cg_j_list.append(j_idx.to(torch.uint8))
            cg_k_list.append(k_idx.to(torch.uint8))
            cg_val_list.append(vals)

            prev_k = int(k_idx[0].item())
            local_k_offsets[prev_k] = 0

            for t in range(1, nnz):
                curr_k = int(k_idx[t].item())
                if curr_k != prev_k:
                    local_k_counts[prev_k] = t - int(local_k_offsets[prev_k].item())
                    local_k_offsets[curr_k] = t
                    prev_k = curr_k

            local_k_counts[prev_k] = nnz - int(local_k_offsets[prev_k].item())

        nnz_k_offsets_list.append(local_k_offsets)
        nnz_k_counts_list.append(local_k_counts)

    if len(cg_i_list) > 0:
        cg_i_all = torch.cat(cg_i_list, dim=0).contiguous()
        cg_j_all = torch.cat(cg_j_list, dim=0).contiguous()
        cg_k_all = torch.cat(cg_k_list, dim=0).contiguous()
        cg_val_all = torch.cat(cg_val_list, dim=0).contiguous()
    else:
        cg_i_all = torch.empty(0, dtype=torch.uint8, device=device)
        cg_j_all = torch.empty(0, dtype=torch.uint8, device=device)
        cg_k_all = torch.empty(0, dtype=torch.uint8, device=device)
        cg_val_all = torch.empty(0, dtype=math_dtype, device=device)

    nnz_per_path_t = torch.tensor(nnz_per_path, dtype=torch.int32, device=device)
    nnz_offsets_t  = torch.tensor(nnz_offsets,  dtype=torch.int32, device=device)

    nnz_k_offsets = torch.stack(nnz_k_offsets_list, dim=0).contiguous()  # [P, MAX_K_DIM]
    nnz_k_counts  = torch.stack(nnz_k_counts_list,  dim=0).contiguous()  # [P, MAX_K_DIM]
    nnz_k_offsets_flat = nnz_k_offsets.reshape(-1).contiguous()
    nnz_k_counts_flat  = nnz_k_counts.reshape(-1).contiguous()

    # -------------------- 8) pack tensors for kernels --------------------
    path_indices_tensor = torch.tensor(path_indices, dtype=torch.int32, device=device).contiguous()
    i_dims_t = torch.tensor(i_dims, dtype=torch.int32, device=device).contiguous()
    j_dims_t = torch.tensor(j_dims, dtype=torch.int32, device=device).contiguous()
    k_dims_t = torch.tensor(k_dims, dtype=torch.int32, device=device).contiguous()
    c_offsets_t = torch.tensor(c_offsets, dtype=torch.int32, device=device).contiguous()  # [P+1]

    meta = {
        # dense
        "c_tensors": c_tensors,
        "path_indices": path_indices,

        # dense packed (original i-j-k flatten)
        "path_indices_tensor": path_indices_tensor,  # [P,4] int32
        "c_all": c_all,                              # [sum(i*j*k)] layout ((i*J+j)*K+k)
        "c_offsets": c_offsets_t,                    # [P+1]

        # slices (reference)
        "uv_slices": uv_slices,
        "iu_slices": iu_slices,
        "jv_slices": jv_slices,

        # offsets
        "uv_seg_offsets": torch.tensor(uv_seg_offsets, dtype=torch.int32, device=device).contiguous(),
        "iu_seg_offsets": torch.tensor(iu_seg_offsets, dtype=torch.int32, device=device).contiguous(),
        "jv_seg_offsets": torch.tensor(jv_seg_offsets, dtype=torch.int32, device=device).contiguous(),
        "kv_k_offsets": torch.tensor(kv_k_offsets, dtype=torch.int32, device=device).contiguous(),

        # dims
        "i_dims": i_dims_t,
        "j_dims": j_dims_t,
        "k_dims": k_dims_t,

        # sizes
        "U": int(u),
        "V": int(v),
        "UV_TOTAL": UV_TOTAL,
        "IU_TOTAL": IU_TOTAL,
        "JV_TOTAL": JV_TOTAL,
        "K_TOTAL": K_TOTAL,

        # sparse CG info
        "cg_i_all": cg_i_all,
        "cg_j_all": cg_j_all,
        "cg_k_all": cg_k_all,
        "cg_val_all": cg_val_all,
        "nnz_per_path": nnz_per_path_t,
        "nnz_offsets": nnz_offsets_t,  # length P (start offset per path)

        # sparse per-k grouping
        "nnz_k_offsets": nnz_k_offsets_flat,  # [P*MAX_K_DIM]
        "nnz_k_counts": nnz_k_counts_flat,    # [P*MAX_K_DIM]
        "MAX_K_DIM": int(max_k_dim_for_kernel),
    }
    return meta

@torch.no_grad()
def infer_fctp_meta(descriptor, math_dtype, device):
    # per-path tensors
    cg_indices = []
    cg_values  = []
    c_tensors  = []
    dim_list   = []

    # 1) build per-path (idx, val, coeffs)
    for i, path in enumerate(descriptor.paths):
        if getattr(path, "coefficients", None) is None or path.coefficients.ndim < 3:
            raise ValueError("FCTP only supports paths with explicit 3D coefficient tensors.")

        coeffs = torch.from_numpy(path.coefficients).to(device=device, dtype=math_dtype)

        # idx: [nnz, 3] (i,j,k) ; vals: [nnz]
        idx = coeffs.nonzero(as_tuple=False).to(device=device, dtype=torch.int32)
        vals = coeffs[idx[:, 0], idx[:, 1], idx[:, 2]].to(device=device, dtype=math_dtype)

        dim_list.append(int(vals.numel()))

        cg_indices.append(idx)
        cg_values.append(vals)
        c_tensors.append(coeffs)

    # 2) global dimensions
    dimensions_dict = descriptor.get_dimensions_dict()
    U = sum(dimensions_dict["u"])
    V = sum(dimensions_dict["v"])
    W = sum(dimensions_dict["w"])

    # 3) K_per_path / offsets / totals
    P = len(cg_indices)
    assert P == len(dim_list)

    K_per_path = torch.tensor(dim_list, device=device, dtype=torch.int32)
    path_offset = torch.empty(P, device=device, dtype=torch.int32)
    path_offset[0] = 0
    if P > 1:
        path_offset[1:] = torch.cumsum(K_per_path[:-1], dim=0)
    K_total = int(K_per_path.sum().item())

    # 4) pack nnz info
    nnz_list = [int(ci.shape[0]) for ci in cg_indices]
    nnz_max = max(nnz_list) if nnz_list else 0
    nnz_per_path = torch.tensor(nnz_list, device=device, dtype=torch.int32)

    # 5) pack cg_*_all
    cg_i_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_j_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_k_all   = torch.zeros((P, nnz_max), device=device, dtype=torch.int32)
    cg_val_all = torch.zeros((P, nnz_max), device=device, dtype=math_dtype)

    for p in range(P):
        ci_local = cg_indices[p]   # [nnz_p, 3]
        cv       = cg_values[p]    # [nnz_p]
        nnz_p    = nnz_list[p]
        offset_p = int(path_offset[p].item())

        i_local = ci_local[:, 0]
        j_local = ci_local[:, 1]
        k_local = ci_local[:, 2]

        # --- local -> global ---
        i_global = i_local + offset_p
        j_global = j_local              # TODO: only support j_local = j_global = 0
        k_global = k_local + offset_p

        cg_i_all[p, :nnz_p]   = i_global
        cg_j_all[p, :nnz_p]   = j_global
        cg_k_all[p, :nnz_p]   = k_global
        cg_val_all[p, :nnz_p] = cv

    return {
        "cg_indices": cg_indices,
        "cg_values": cg_values,
        "c_tensors": c_tensors,

        "U": U, "V": V, "W": W,

        "P": P,
        "K_per_path": K_per_path,
        "path_offset": path_offset,
        "K_total": K_total,

        "nnz_list": nnz_list,
        "nnz_max": nnz_max,
        "nnz_per_path": nnz_per_path,

        "cg_i_all": cg_i_all,
        "cg_j_all": cg_j_all,
        "cg_k_all": cg_k_all,
        "cg_val_all": cg_val_all,
    }

class FastEqSegmentedPolynomial(nn.Module):
    """PyTorch module that computes a segmented polynomial.

    Args:
        polynomial: The segmented polynomial to compute, an instance of
            `cue.SegmentedPolynomial <cuequivariance.SegmentedPolynomial>`.
        method: Specifies the implementation method to use. Options are:

            - ``"naive"``: Uses a naive PyTorch implementation. It always works but is not optimized.
            - ``"uniform_1d"``: Uses a CUDA implementation for polynomials with a single uniform mode.
            - ``"fused_tp"``: Uses a CUDA implementation for polynomials with 3- and 4-operand contractions.
            - ``"indexed_linear"``: Uses a CUDA implementation for linear layers with indexed weights.

        math_dtype: Optional data type for computational operations.
            If specified, internal buffers will be of this dtype,
            and operands will be converted to this type for all computations.

            Values can be specified as a string corresponding to a torch.dtype,
            or as a torch.dtype.
            For some methods, special values can be used:

            - For method ``"naive"``: Any torch.dtype or corresponding string.
            - For method ``"uniform_1d"``: ``torch.float32`` or ``torch.float64`` or corresponding strings.
            - For method ``"fused_tp"``: ``torch.float32`` or ``torch.float64`` or corresponding strings.
            - For method ``"indexed_linear"``: this is not supported and will be ignored.

            .. note::
               This will not be affected by changes to the module dtype,
               and not all methods support all dtypes.

            If ``math_dtype`` is not specified:

            - For method ``"naive"``, the dtype of the input tensors will be used.
            - For method ``"uniform_1d"``, the dtype of the input tensors will be used if allowed
              (FP32 or FP64), otherwise float32 will be used.
            - For method ``"fused_tp"``, the default dtype (FP32) will be used.
            - For method ``"indexed_linear"``, the dtype of the input tensors will be used.

        output_dtype_map: Optional list that, for each output buffer, specifies
            the index of the input buffer from which it inherits its data type.
            -1 means the math_dtype is used.
            Default is 0 if there are input tensors, otherwise -1.
        name: Optional name for the operation. Defaults to "segmented_polynomial".

    Examples:
        Basic usage with spherical harmonics:

        >>> import torch
        >>> import cuequivariance as cue
        >>> from cuequivariance_torch import SegmentedPolynomial
        >>>
        >>> # Create spherical harmonics polynomial
        >>> poly = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2]).polynomial
        >>> sp = SegmentedPolynomial(poly, method="naive")
        >>>
        >>> # Compute spherical harmonics for unit vector along y-axis
        >>> x = torch.tensor([[0.0, 1.0, 0.0]])
        >>> result = sp([x])
        >>> print(result[0].shape)
        torch.Size([1, 9])

        Example with a linear layer:

        >>> # Create a linear transformation
        >>> input_irreps = cue.Irreps(cue.O3, "5x0e + 3x1o")
        >>> output_irreps = cue.Irreps(cue.O3, "4x0e + 2x1o")
        >>> poly = cue.descriptors.linear(input_irreps, output_irreps).polynomial
        >>>
        >>> # Create the module
        >>> linear = SegmentedPolynomial(poly, method="naive")
        >>>
        >>> # Create random weights and input
        >>> weights = torch.randn(1, poly.inputs[0].size)
        >>> x = torch.randn(10, poly.inputs[1].size)
        >>>
        >>> # Forward pass
        >>> result = linear([weights, x])
        >>> print(result[0].shape)
        torch.Size([10, 10])

        Example with indexed operations:

        >>> # Create indexed weights for different elements
        >>> weights = torch.randn(3, poly.inputs[0].size)  # 3 different weight sets
        >>> x = torch.randn(5, poly.inputs[1].size)        # 5 input vectors
        >>>
        >>> # Index tensor specifying which weights to use for each input
        >>> weight_indices = torch.tensor([0, 1, 0, 2, 1])  # Use weights 0,1,0,2,1
        >>>
        >>> result = linear([weights, x],
        ...                input_indices={0: weight_indices})
        >>> print(result[0].shape)
        torch.Size([5, 10])
    """

    def __init__(
        self,
        polynomial: cue.SegmentedPolynomial,
        method: str = "",
        math_dtype: str | torch.dtype = None,
        output_dtype_map: List[int] = None,
        name: str = "segmented_polynomial",
        op_name: str = "",
        use_fasteq: Optional[bool] = None,
    ):
        super().__init__()

        self.num_inputs = polynomial.num_inputs
        self.num_outputs = polynomial.num_outputs
        self.method = method
        self.repr = polynomial.__repr__()
        self.op_name = op_name
        self.descriptor = polynomial.operations[0][1]
        self.use_fasteq = use_fasteq
        
        if method == "":
            warnings.warn(
                "Hello! It looks like you're using code that was written for an older version of this library.\n"
                "Starting in v0.6.0, the `method` argument is suggested when using `SegmentedPolynomial()`.\n"
                "This change helps ensure you get optimal performance by explicitly choosing the computation method.\n"
                "For the moment, we will default to the 'uniform_1d' method.\n\n"
                "To remove this warning, add a `method` parameter to your function call. Here are the available options:\n"
                "• 'naive' - Works everywhere but not optimized (good for testing)\n"
                "• 'uniform_1d' - Fast CUDA implementation for single uniform mode polynomials\n"
                "• 'fused_tp' - A more general CUDA implementation, supporting many 3 and 4 operands contractions.\n"
                "• 'indexed_linear' - A CUDA implementation for linear layers with indexed weights.\n"
            )
            method = "naive"

        if not isinstance(polynomial, cue.SegmentedPolynomial):
            raise ValueError(
                f"The polynomial is not a cue.SegmentedPolynomial, but a {type(polynomial)}",
                "Did you forget to call `.polynomial` on the descriptor?",
            )

        if method != "naive" and not HAS_CUE_OPS:
            method = "naive"
            warnings.warn(
                "cuequivariance_ops_torch is not available. Falling back to naive implementation."
            )

        if method == "uniform_1d":
            self.m = SegmentedPolynomialFromUniform1dJit(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        elif method == "naive":
            self.m = SegmentedPolynomialNaive(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        elif method == "fused_tp":
            self.m = SegmentedPolynomialFusedTP(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = SegmentedPolynomialNaive(
                polynomial, math_dtype, output_dtype_map, name
            )
        elif method == "indexed_linear":
            self.m = SegmentedPolynomialIndexedLinear(
                polynomial, math_dtype, output_dtype_map, name
            )
            self.fallback = self.m
        else:
            raise ValueError(f"Invalid method: {method}")

        if use_fasteq and op_name == "stc":
            import math
            from torch.nn.utils.rnn import pad_sequence
            def f(d: cue.SegmentedTensorProduct) -> cue.SegmentedTensorProduct:
                

                d = d.move_operand(0, -2)
                d = d.flatten_coefficient_modes(force=True)
                d = d.flatten_modes(
                    [
                        m
                        for m in d.subscripts.modes()
                        if not all(m in ss for ss in d.subscripts.operands)
                    ]
                )
                d = d.consolidate_modes()
                if d.subscripts.modes() == []:
                    d = d.append_modes_to_all_operands("u", dict(u=1))
                '''
                for oid in range(0, d.num_operands - 2):
                    print(f"oid:{oid}, len d.operands[oid].num_segments:{d.operands[oid].num_segments}")
                '''

                # ops.SymmetricTensorContraction will "symmetrize" for the derivatives so we can sort for the forward pass
                d = d.sort_indices_for_identical_operands(range(0, d.num_operands - 2))

                if len(d.subscripts.modes()) != 1:
                    raise NotImplementedError("Different modes are not supported.")

                m = d.subscripts.modes()[0]

                if not all(ss == m for ss in d.subscripts.operands):
                    raise NotImplementedError("Different subscripts are not supported.")

                d = d.split_mode(m, math.gcd(*d.get_dims(m)))

                return d

            ds_ = [f(d) for _, d in polynomial.operations]
            d_max = max(ds_, key=lambda d: d.num_operands)
            self.num_out_segments = d_max.operands[-1].num_segments
            self.u = d_max.operands[0].size // d_max.operands[0].num_segments

            path_segment_indices = sum((d.indices.tolist() for  d in ds_), [])
            path_coefficients = sum((d.stacked_coefficients.tolist() for d in ds_), [])

            device = "cuda" # TODO: make it general
            self.coeffs_tensor = torch.as_tensor(path_coefficients, dtype=math_dtype).to(device)
            self.path_lens_tensor = torch.as_tensor([len(p) for p in path_segment_indices], dtype=torch.int32).to(device)
            self.paths_tensor = pad_sequence(
                [torch.as_tensor(p, dtype=torch.int32) for p in path_segment_indices],
                batch_first=True, padding_value=0
            ).to(device)
        
        if use_fasteq and (op_name == "cwtp"):
            self.meta = infer_cwtp_meta(self.descriptor, math_dtype=math_dtype, device="cuda") # device hardcoded for now
        
        if use_fasteq and (op_name == "fctp"):
            self.meta = infer_fctp_meta(self.descriptor, math_dtype=math_dtype, device="cuda") # device hardcoded for now

    def __repr__(self):
        return self.repr + f"\n{super().__repr__()}"

    # For torch.jit.trace, we cannot pass explicit optionals,
    # so these must be passed as kwargs then.
    # List[Optional[Tensor]] does not work for similar reasons, hence, Dict
    # is the only option.
    # Also, shapes cannot be passed as integers, so they are passed via a
    # (potentially small-strided) tensor with the right shape.
    def forward(
        self,
        inputs: List[torch.Tensor],
        input_indices: Optional[Dict[int, torch.Tensor]] = None,
        output_shapes: Optional[Dict[int, torch.Tensor]] = None,
        output_indices: Optional[Dict[int, torch.Tensor]] = None,
    ):
        """Compute the segmented polynomial based on the specified descriptor.

        Args:
            inputs: The input tensors. The number of input tensors must match
                the number of input buffers in the descriptor.
                Each input tensor should have a shape of ``(batch, operand_size)`` or
                ``(1, operand_size)`` or ``(index, operand_size)`` in the indexed case.
                Here, ``operand_size`` is the size of each operand as defined in
                the descriptor.
            input_indices: A dictionary that contains an optional indexing tensor
                for each input tensor. The key is the index into the inputs.
                If a key is not present, no indexing takes place.
                The contents of the index tensor must be suitable to index the
                input tensor (i.e., ``0 <= index_tensor[i] < input.shape[0]``).

                .. note::
                   Method ``"indexed_linear"`` requires the indices to be sorted.

            output_shapes: A dictionary specifying the size of the output batch
                dimensions using Tensors. We only read ``shape_tensor.shape[0]``.
                This is mandatory if the output tensor is indexed. Otherwise,
                the default shape is ``(batch, operand_size)``.
            output_indices: A dictionary that contains an optional indexing tensor
                for each output tensor. See ``input_indices`` for details.

        Returns:
            The output tensors resulting from the segmented polynomial.
            Their shapes are specified just like the inputs.
        """

        # General checks
        empty_dict: Dict[int, torch.Tensor] = {}
        if input_indices is None:
            input_indices = dict(empty_dict)
        if output_shapes is None:
            output_shapes = dict(empty_dict)
        if output_indices is None:
            output_indices = dict(empty_dict)

        inputs = list(inputs)

        if not torch.jit.is_scripting():
            if (
                not torch.jit.is_tracing()
                and not torch.compiler.is_compiling()
                and not torch.fx._symbolic_trace.is_fx_tracing()
            ):
                torch._assert(
                    len(inputs) == self.num_inputs,
                    "the number of inputs must match the number of inputs of the polynomial",
                )

                for k, v in input_indices.items():
                    torch._assert(
                        0 <= k < self.num_inputs, "input index must be in range"
                    )
                    torch._assert(v.ndim == 1, "input index must be one-dimensional")
                    torch._assert(
                        v.dtype in [torch.int32, torch.int64],
                        "input index must be integral",
                    )
                for k, v in output_indices.items():
                    torch._assert(
                        0 <= k < self.num_outputs, "output index must be in range"
                    )
                    torch._assert(v.ndim == 1, "input index must be one-dimensional")
                    torch._assert(
                        v.dtype in [torch.int32, torch.int64],
                        "input index must be integral",
                    )
                for k, v in output_shapes.items():
                    torch._assert(
                        0 <= k < self.num_outputs, "output index must be in range"
                    )
                    torch._assert(v.ndim == 2, "output shape must be two-dimensional")

                # If the input is on the CPU and we're using fused_tp, we need to fall back to naive
                if (
                    inputs[0].device == torch.device("cpu")
                    and self.method == "fused_tp"
                ):
                    warnings.warn(
                        "Fused TP is not supported on CPU. Falling back to naive implementation."
                    )
                    return self.fallback(
                        inputs, input_indices, output_shapes, output_indices
                    )

        if self.use_fasteq:
            out = [torch.empty(0) for _ in range(self.num_outputs)]
            if self.num_outputs != 1:
                    raise ValueError("equi_linear should have exactly one output")
            
            if self.op_name == "equi_linear":
                if tuple(inputs[0].shape) == (1, 36864):
                    ref = fast_equi_linear(self.descriptor, inputs[0], inputs[1])
                    out[0] = ref
                else:
                    return self.m(inputs, input_indices, output_shapes, output_indices)
            elif self.op_name == "stc":
                i0 = input_indices[0].to(torch.int32)
                x0 = inputs[0]
                x1 = inputs[1]
                
                x0 = x0.reshape(x0.shape[0], x0.shape[1] // self.u, self.u)
                x1 = x1.reshape(x1.shape[0], x1.shape[1] // self.u, self.u)

                #print(f"x1 shape:{x1.shape}, x0 shape:{x0.shape}, i0 shape:{i0.shape}")

                ref = fast_stc(
                    x1, x0, i0, 
                    self.coeffs_tensor, 
                    self.paths_tensor, 
                    self.path_lens_tensor, 
                    self.num_out_segments,
                )
                out[0] = ref
            elif self.op_name == "cwtp":
                # mptp case use input and output indices
                if input_indices.get(1) is not None and output_indices.get(0) is not None:
                    '''
                    for k, v in input_indices.items():
                        print(f"input_indices key:{k}, value:{v}")
                    for k, v in output_indices.items():
                        print(f"output_indices key:{k}, value:{v}")
                    for inp in inputs:
                        print(f"input shape:{inp.shape}")
                    '''
                    w, x, y = inputs[0], inputs[1], inputs[2]
                    sender = input_indices[1].to(torch.int32)
                    receiver= output_indices[0].to(torch.int32)
                    ref = fast_mptp(
                        w, x, y,sender, receiver,
                        self.meta,
                    )
                    out[0] = ref
                # cwtp case use only inputs
                else:
                    w, x, y = inputs[0], inputs[1], inputs[2]
                    ref = fast_cwtp(
                        w, x, y,
                        self.meta,
                    )
                    out[0] = ref
            elif self.op_name == "fctp":
                w, x, y = inputs[0], inputs[1], inputs[2]
                ref = fast_fctp(
                    w, x, y,
                    self.meta,
                )
                out[0] = ref
            else:
                out = self.m(inputs, input_indices, output_shapes, output_indices)
                
        else:
            out = self.m(inputs, input_indices, output_shapes, output_indices)
        return out