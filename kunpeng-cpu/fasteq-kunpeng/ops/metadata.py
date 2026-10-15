"""Build native metadata from e3nn/MACE modules (CPU float32/float64)."""
import math
import torch
import torch.nn.functional as F
from e3nn import o3

def prod(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def irreps_metadata(irreps):
    offsets, muls, dims = [], [], []
    offset = 0
    for mul, ir in irreps:
        offsets.append(offset)
        muls.append(mul)
        dims.append(ir.dim)
        offset += mul * ir.dim
    return (
        torch.tensor(offsets, dtype=torch.int64),
        torch.tensor(muls, dtype=torch.int64),
        torch.tensor(dims, dtype=torch.int64),
    )


def make_symmetric_contraction_metadata(contraction):
    """Pack the exact corr=3 scalar contraction used by MACE-OFF."""
    correlation = int(contraction.correlation)
    if correlation != 3:
        raise NotImplementedError(
            f"the fused CPU SC path requires correlation=3, got {correlation}"
        )
    U3 = contraction.U_tensors(3).detach().contiguous()
    U2 = contraction.U_tensors(2).detach().contiguous()
    U1 = contraction.U_tensors(1).detach().contiguous()
    if U3.dim() != 4 or U2.dim() != 3 or U1.dim() != 2:
        raise NotImplementedError("the fused CPU SC path only handles scalar output")
    if tuple(U3.shape[:2]) != tuple(U2.shape[:2]):
        raise NotImplementedError("U2/U3 spatial dimensions do not match")
    if U3.size(2) != U1.size(0):
        raise NotImplementedError("U1/U3 irrep dimensions do not match")
    if len(contraction.weights) != 2:
        raise NotImplementedError("expected exactly two lower-order weights")

    Wmax = contraction.weights_max.detach().contiguous()
    W0 = contraction.weights[0].detach().contiguous()
    W1 = contraction.weights[1].detach().contiguous()
    if Wmax.dim() != 3 or W0.dim() != 3 or W1.dim() != 3:
        raise NotImplementedError("unexpected packed weight rank")
    if Wmax.size(1) != U3.size(3) or W0.size(1) != U2.size(2):
        raise NotImplementedError("packed weight/U parameter mismatch")
    if W1.size(1) != U1.size(1):
        raise NotImplementedError("U1/W1 parameter mismatch")
    if Wmax.size(0) != W0.size(0) or Wmax.size(0) != W1.size(0):
        raise NotImplementedError("element dimension mismatch")
    if Wmax.size(2) != W0.size(2) or Wmax.size(2) != W1.size(2):
        raise NotImplementedError("channel dimension mismatch")

    U2T = (
        U2.permute(2, 0, 1)
        .reshape(U2.size(2), U2.size(0) * U2.size(1))
        .contiguous()
    )
    C0 = torch.matmul(
        W0.permute(0, 2, 1).contiguous(), U2T
    ).contiguous()
    C1 = torch.matmul(
        U1, W1.permute(0, 2, 1).transpose(1, 2)
    ).permute(0, 2, 1).contiguous()
    U3T = (
        U3.permute(2, 3, 0, 1)
        .reshape(U3.size(2) * U3.size(3), U3.size(0) * U3.size(1))
        .contiguous()
    )
    WmaxT = Wmax.permute(0, 2, 1).contiguous()
    effective = torch.einsum(
        "eck,ikm->ecim",
        WmaxT,
        U3T.reshape(U3.size(2), U3.size(3), U3.size(0) * U3.size(1)),
    ).contiguous()

    return {
        "kind": "fused_scalar_corr3",
        "U3": U3,
        "U3T": U3T,
        "WmaxT": WmaxT,
        "effective": effective,
        "C0": C0,
        "C1": C1,
    }


def make_radial_mlp_metadata(mlp):
    weights = []
    activation_scales = []
    for layer in mlp:
        activation = layer.act
        if activation is None:
            scale = math.sqrt(
                layer.h_in * layer.var_in / layer.var_out
            )
            weights.append((layer.weight / scale).detach().contiguous())
            activation_scales.append(None)
            continue
        if not hasattr(activation, "f") or activation.f is not F.silu:
            raise NotImplementedError(
                "radial MLP expects normalize2mom(SiLU)"
            )
        scale = math.sqrt(layer.h_in * layer.var_in)
        weights.append((layer.weight / scale).detach().contiguous())
        activation_scales.append(
            math.sqrt(layer.var_out) * float(activation.cst)
        )
    return {
        "weights": tuple(weights),
        "activation_scales": tuple(activation_scales),
        "native_scales": torch.tensor(
            [
                float("nan") if scale is None else float(scale)
                for scale in activation_scales
            ],
            dtype=torch.float64,
        ),
        "native_active": torch.tensor(
            [0 if scale is None else 1 for scale in activation_scales],
            dtype=torch.int64,
        ),
    }


def make_fctp_metadata(module, vstar):
    in1_meta = irreps_metadata(module.irreps_in1)
    # Wigner-3j coefficients must follow the module weight dtype; hard-coding
    # float32 broke float64 models (mixed dtype inside the native kernel).
    w3j_dtype = getattr(getattr(module, "weight", None), "dtype", torch.float64)
    in2_meta = irreps_metadata(module.irreps_in2)
    out_meta = irreps_metadata(module.irreps_out)
    i1, i2, io, offsets, path_weights = [], [], [], [], []
    w3js = []
    offset = 0
    for instruction in module.instructions:
        if instruction.i_in1 < 0:
            continue
        mul1, ir1 = module.irreps_in1[instruction.i_in1]
        mul2, ir2 = module.irreps_in2[instruction.i_in2]
        mul_out, ir_out = module.irreps_out[instruction.i_out]
        i1.append(instruction.i_in1)
        i2.append(instruction.i_in2)
        io.append(instruction.i_out)
        offsets.append(offset)
        path_weights.append(instruction.path_weight)
        w3js.append(
            o3.wigner_3j(ir1.l, ir2.l, ir_out.l)
            .to(dtype=w3j_dtype)
            .contiguous()
        )
        offset += prod((mul1, mul2, mul_out))
    return {
        "args": (
            *in1_meta,
            *in2_meta,
            *out_meta,
            torch.tensor(i1, dtype=torch.int64),
            torch.tensor(i2, dtype=torch.int64),
            torch.tensor(io, dtype=torch.int64),
            torch.tensor(offsets, dtype=torch.int64),
            torch.tensor(path_weights, dtype=torch.float64),
        ),
        "w3js": w3js,
        "shared_weights": module.shared_weights,
        "vstar": vstar,
    }


def make_linear_metadata(module):
    in_meta = irreps_metadata(module.irreps_in)
    out_meta = irreps_metadata(module.irreps_out)
    ins_in, ins_out, offsets, path_weights = [], [], [], []
    offset = 0
    for instruction in module.instructions:
        if instruction.i_in < 0:
            raise NotImplementedError("bias instructions are not included in this test")
        ins_in.append(instruction.i_in)
        ins_out.append(instruction.i_out)
        offsets.append(offset)
        path_weights.append(instruction.path_weight)
        offset += prod(instruction.path_shape)
    return {
        "args": (
            *in_meta,
            *out_meta,
            torch.tensor(ins_in, dtype=torch.int64),
            torch.tensor(ins_out, dtype=torch.int64),
            torch.tensor(offsets, dtype=torch.int64),
            torch.tensor(path_weights, dtype=torch.float64),
        ),
        "shared_weights": module.shared_weights,
    }


def make_tpu_metadata(module):
    in1_meta = irreps_metadata(module.irreps_in1)
    # See make_fctp_metadata: keep the coefficient dtype aligned with weights.
    w3j_dtype = getattr(getattr(module, "weight", None), "dtype", torch.float64)
    in2_meta = irreps_metadata(module.irreps_in2)
    out_meta = irreps_metadata(module.irreps_out)
    i1, i2, io, offsets, path_weights = [], [], [], [], []
    w3js = []
    offset = 0
    for instruction in module.instructions:
        if instruction.i_in1 < 0:
            continue
        if instruction.connection_mode != "uvu":
            raise NotImplementedError(
                f"unsupported tensor product mode: {instruction.connection_mode}"
            )
        mul1, ir1 = module.irreps_in1[instruction.i_in1]
        mul2, ir2 = module.irreps_in2[instruction.i_in2]
        mul_out, ir_out = module.irreps_out[instruction.i_out]
        if mul1 != mul_out:
            raise ValueError("uvu output multiplicity mismatch")
        i1.append(instruction.i_in1)
        i2.append(instruction.i_in2)
        io.append(instruction.i_out)
        offsets.append(offset)
        path_weights.append(instruction.path_weight)
        w3js.append(
            o3.wigner_3j(ir1.l, ir2.l, ir_out.l)
            .to(dtype=w3j_dtype)
            .contiguous()
        )
        offset += mul1 * mul2
    return {
        "args": (
            *in1_meta,
            *in2_meta,
            *out_meta,
            torch.tensor(i1, dtype=torch.int64),
            torch.tensor(i2, dtype=torch.int64),
            torch.tensor(io, dtype=torch.int64),
            torch.tensor(offsets, dtype=torch.int64),
            torch.tensor(path_weights, dtype=torch.float64),
        ),
        "w3js": w3js,
        "shared_weights": module.shared_weights,
        "disjoint_output": len(io) == len(set(io)),
    }


def make_packed_fctp_metadata(module, weight):
    if not module.shared_weights:
        raise ValueError("packed FCTP requires shared weights")
    in_meta = irreps_metadata(module.irreps_in1)
    out_meta = irreps_metadata(module.irreps_out)
    if len(module.irreps_in2) != 1 or module.irreps_in2[0].ir.dim != 1:
        raise ValueError("packed one-hot FCTP requires a scalar second input")
    elements = module.irreps_in2[0].mul
    packed_size = 0
    coeff_size = 0
    packed_offsets, coeff_offsets = [], []
    ins_in_offset, ins_out_offset = [], []
    ins_mul_in, ins_mul_out, ins_dim = [], [], []
    coeffs = []
    for instruction in module.instructions:
        if instruction.i_in1 < 0:
            raise NotImplementedError("bias instructions are unsupported")
        mul_in, ir_in = module.irreps_in1[instruction.i_in1]
        mul_attr, ir_attr = module.irreps_in2[instruction.i_in2]
        mul_out, ir_out = module.irreps_out[instruction.i_out]
        if ir_in != ir_out or mul_attr != elements:
            raise NotImplementedError("packed FCTP requires matching scalar-input paths")
        w3j = o3.wigner_3j(ir_in.l, ir_attr.l, ir_out.l).to(torch.float64)
        if w3j.shape != (ir_in.dim, 1, ir_out.dim):
            raise NotImplementedError("unexpected scalar-input Wigner shape")
        diagonal = w3j[:, 0, :].diagonal()
        off_diagonal = w3j[:, 0, :] - torch.diag(diagonal)
        if off_diagonal.abs().max().item() != 0.0:
            raise ValueError("expected diagonal scalar-input coupling")
        packed_offsets.append(packed_size)
        coeff_offsets.append(coeff_size)
        ins_in_offset.append(in_meta[0][instruction.i_in1].item())
        ins_out_offset.append(out_meta[0][instruction.i_out].item())
        ins_mul_in.append(mul_in)
        ins_mul_out.append(mul_out)
        ins_dim.append(ir_in.dim)
        coeffs.extend(
            (diagonal * instruction.path_weight).to(weight.dtype).tolist()
        )
        packed_size += mul_in * mul_out
        coeff_size += ir_in.dim

    packed = torch.zeros(elements, packed_size, dtype=weight.dtype)
    weight_flat = weight.reshape(-1)
    weight_offset = 0
    for instruction, instruction_pack_offset in zip(
        module.instructions, packed_offsets
    ):
        mul_in, ir_in = module.irreps_in1[instruction.i_in1]
        mul_out, ir_out = module.irreps_out[instruction.i_out]
        mul_attr = module.irreps_in2[instruction.i_in2].mul
        del ir_in, ir_out
        for element in range(elements):
            for u in range(mul_in):
                for wo in range(mul_out):
                    source = weight_offset + (
                        (u * mul_attr + element) * mul_out + wo
                    )
                    destination = instruction_pack_offset + wo * mul_in + u
                    packed[element, destination] = weight_flat[source]
        weight_offset += mul_in * mul_attr * mul_out

    return {
        "packed_weight": packed,
        "args": (
            torch.tensor(ins_in_offset, dtype=torch.int64),
            torch.tensor(ins_out_offset, dtype=torch.int64),
            torch.tensor(ins_mul_in, dtype=torch.int64),
            torch.tensor(ins_mul_out, dtype=torch.int64),
            torch.tensor(ins_dim, dtype=torch.int64),
            torch.tensor(packed_offsets, dtype=torch.int64),
            torch.tensor(coeff_offsets, dtype=torch.int64),
            torch.tensor(coeffs, dtype=weight.dtype),
        ),
        "in_total": module.irreps_in1.dim,
        "out_total": module.irreps_out.dim,
    }


def make_packed_linear_metadata(module):
    if not module.shared_weights:
        raise ValueError("packed linear requires shared weights")
    in_meta = irreps_metadata(module.irreps_in)
    out_meta = irreps_metadata(module.irreps_out)
    ins_in_offset, ins_out_offset = [], []
    ins_mul_in, ins_mul_out, ins_dim = [], [], []
    weight_offsets = []
    packed_size = 0
    weight_offset = 0
    for instruction in module.instructions:
        if instruction.i_in < 0:
            raise NotImplementedError("bias instructions are unsupported")
        mul_in, ir_in = module.irreps_in[instruction.i_in]
        mul_out, ir_out = module.irreps_out[instruction.i_out]
        if ir_in != ir_out:
            raise ValueError("linear path irrep mismatch")
        weight_offsets.append(packed_size)
        ins_in_offset.append(in_meta[0][instruction.i_in].item())
        ins_out_offset.append(out_meta[0][instruction.i_out].item())
        ins_mul_in.append(mul_in)
        ins_mul_out.append(mul_out)
        ins_dim.append(ir_in.dim)
        packed_size += mul_in * mul_out
        weight_offset += mul_in * mul_out

    packed = torch.zeros(packed_size, dtype=module.weight.dtype)
    weight_flat = module.weight.detach().reshape(-1)
    source_offset = 0
    for instruction, destination_offset in zip(
        module.instructions, weight_offsets
    ):
        mul_in = module.irreps_in[instruction.i_in].mul
        mul_out = module.irreps_out[instruction.i_out].mul
        source = weight_flat[source_offset : source_offset + mul_in * mul_out]
        source = source.reshape(mul_in, mul_out).transpose(0, 1)
        source = source * instruction.path_weight
        packed[
            destination_offset : destination_offset + mul_in * mul_out
        ] = source.reshape(-1)
        source_offset += mul_in * mul_out

    return {
        "packed_weight": packed.contiguous(),
        "args": (
            torch.tensor(ins_in_offset, dtype=torch.int64),
            torch.tensor(ins_out_offset, dtype=torch.int64),
            torch.tensor(ins_mul_in, dtype=torch.int64),
            torch.tensor(ins_mul_out, dtype=torch.int64),
            torch.tensor(ins_dim, dtype=torch.int64),
            torch.tensor(weight_offsets, dtype=torch.int64),
        ),
        "in_total": module.irreps_in.dim,
        "out_total": module.irreps_out.dim,
    }
