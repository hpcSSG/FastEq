# CPU API 与上游差异

所有函数从 `fasteq.ops` 导入，元数据构造函数从 `fasteq.ops.metadata` 导入。
`load_extension()` 从 `fasteq.cpu` 导入，返回 pybind 扩展。无需自行传入扩展实例。

| 调用 | 元数据/限制 |
|---|---|
| `fast_cwtp(x1, x2, weight, metadata)` | `make_tpu_metadata(e3nn_tp)`；仅 `uvu`，非任意 cuEquivariance CWTP |
| `fast_fctp(x1, x2, weight, metadata, one_hot=False)` | `make_fctp_metadata(e3nn_fctp, element)`；one_hot=True 时 element 与 x2 必须一致 |
| `fast_fctp_packed(x, element, metadata)` | `make_packed_fctp_metadata(module, module.weight.detach())`；共享权重、标量 one-hot 第二输入 |
| `fast_equi_linear(x, weight, metadata)` | `make_linear_metadata(module)`；无 bias |
| `fast_equi_linear_packed(x, metadata)` | `make_packed_linear_metadata(module)`；共享权重、无 bias |
| `fast_stc(U, weight, x, element)` | U=[*S,I,K]，weight=[E,K,C]，x=[B,C,I]，返回 [B,C,*S]；仅收缩第一步 |
| `fast_stc_scalar_fused(x, element, metadata)` | `make_symmetric_contraction_metadata(mace_contraction)`；corr=3、标量输出 |
| `fast_mptp(node_features, edge_attrs, weight, sender, receiver, metadata, num_nodes=None)` | `make_tpu_metadata`；显式生成边消息再聚合，未融合 |
| `fast_spherical_harmonics(x, normalization='integral')` | x=[B,3]，输出 [B,16]；component 也支持 |
| `fast_scatter_sum(src, index, dim_size)` | src=[E] 或 [E,D]；index=[E]，int64，0 <= index < dim_size |
| `fast_radial_mlp(x, metadata)` | `make_radial_mlp_metadata(mlp)`；固定权重、normalize2mom(SiLU) |
| `fast_radial_embedding(lengths, bessel_weights, r_max, cutoff_p)` | lengths=[E,1]，weights=[K]；r_max 为正数/标量张量，非可训练参数；距离应在模型 cutoff 域内 |
| `fast_edge_geometry(positions, edge_index, shifts, normalize=False, eps=1e-9)` | positions=[N,3]，edge_index=[2,E]，shifts=[E,3]；返回 (vectors,lengths) |

除 uvu 外，大多数 Autograd 封装只返回输入梯度；禁止据此推断支持参数训练。原生扩展反向不构建可微计算图，二阶导数不属于接口承诺。

与上游 FastEq 的区别：

- 未复制 CUDA/HIP/SYCL 分支；这是 CPU 专用整理版本。
- 没有 `use_fasteq=True` 的 cuEquivariance 补丁或 descriptor 翻译器。
- Python 函数参数顺序是此版本的接口，不与上游同名函数保证兼容。
- 构建采用原有 PyTorch JIT，未接入上游的 CMake 后端选择。
- 权重打包和元数据来自 e3nn/MACE，不应直接传入 cuEquivariance descriptor。
