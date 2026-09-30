# e3nn math / o3 / nn Triton 实现清单（更新）

统计日期：2026-09-30。原生基线：`2aa7f58440a06b15352a2cbce01fa4c26f824969`。实现按当前工作区核对，包含新增 orthonormalize、complete_basis、reduce_permutation 和 standard_representation。

## 统计口径

- 计数单位为原始顶层 def 和显式类方法；排除类 __init__、__repr__。仍保留 forward、其他方法、属性、私有函数。
- 不计新增 kernel/辅助函数、嵌套函数、导入符号。沿用排除 o3/_tensor_product、experimental、_irreps.py 和 nn/models 的范围；__init__.py 导出表不计。
- Triton 数值路径与混合接入均计为已接入；只要受支持调用链能调用 Triton，即使 forward 本身未重写，也标明间接接入。
- 沿用原生数值路径与无需 Triton 分开统计，避免把控制、元数据、无计算入口视为缺失数值算子。
- 初始化接入不等于 forward 执行 Triton；外部任意 callable 的运行行为不能凭静态源码一概算为已接入。
- 存在设备、dtype、形状和梯度回退；统计不是 GPU 通过率、训练支持率或性能结论。

## 汇总

| 模块 | 纳入 def | Triton 数值路径 | 混合接入 | 已接入合计 | 沿用原生数值 | 无需 Triton | 排除 init/repr |
|---|---:|---:|---:|---:|---:|---:|---:|
| math | 28 | 10 | 2 | 12 | 1 | 15 | 1 |
| o3 | 71 | 41 | 5 | 46 | 13 | 12 | 15 |
| nn | 14 | 5 | 4 | 9 | 2 | 3 | 22 |
| 合计 | 113 | 56 | 11 | 67 | 16 | 30 | 38 |

新增 Legendre.forward 已调用 Triton，但原生由 FX 动态生成，无显式 def，单独注明，不加入原始函数分母。继承方法不按类重复计数。

## 按文件统计

| 文件 | 纳入 def | Triton 数值路径 | 混合接入 | 沿用原生数值 | 无需 Triton | 排除 init/repr |
|---|---:|---:|---:|---:|---:|---:|
| `math/_bessel.py` | 1 | 1 | 0 | 0 | 0 | 0 |
| `math/_linalg.py` | 4 | 3 | 0 | 0 | 1 | 0 |
| `math/_normalize_activation.py` | 3 | 0 | 1 | 1 | 1 | 1 |
| `math/_reduce.py` | 2 | 0 | 1 | 0 | 1 | 0 |
| `math/_soft_one_hot_linspace.py` | 1 | 1 | 0 | 0 | 0 | 0 |
| `math/_soft_unit_step.py` | 3 | 3 | 0 | 0 | 0 | 0 |
| `math/perm.py` | 14 | 2 | 0 | 0 | 12 | 0 |
| `o3/_angular_spherical_harmonics.py` | 7 | 4 | 0 | 0 | 3 | 2 |
| `o3/_linear.py` | 4 | 1 | 0 | 0 | 3 | 2 |
| `o3/_norm.py` | 1 | 1 | 0 | 0 | 0 | 2 |
| `o3/_reduce.py` | 3 | 0 | 0 | 2 | 1 | 2 |
| `o3/_rotation.py` | 28 | 24 | 4 | 0 | 0 | 0 |
| `o3/_s2grid.py` | 13 | 9 | 1 | 0 | 3 | 4 |
| `o3/_so3grid.py` | 3 | 2 | 0 | 1 | 0 | 2 |
| `o3/_spherical_harmonics.py` | 3 | 0 | 0 | 3 | 0 | 1 |
| `o3/_spherical_harmonics_generator.py` | 1 | 0 | 0 | 0 | 1 | 0 |
| `o3/_wigner.py` | 8 | 0 | 0 | 7 | 1 | 0 |
| `nn/_activation.py` | 1 | 1 | 0 | 0 | 0 | 2 |
| `nn/_batchnorm.py` | 2 | 1 | 0 | 1 | 0 | 2 |
| `nn/_dropout.py` | 1 | 0 | 1 | 0 | 0 | 2 |
| `nn/_extract.py` | 1 | 1 | 0 | 0 | 0 | 2 |
| `nn/_fc.py` | 1 | 0 | 0 | 1 | 0 | 4 |
| `nn/_gate.py` | 4 | 2 | 0 | 0 | 2 | 3 |
| `nn/_identity.py` | 1 | 0 | 0 | 0 | 1 | 2 |
| `nn/_normact.py` | 1 | 0 | 1 | 0 | 0 | 1 |
| `nn/_s2act.py` | 1 | 0 | 1 | 0 | 0 | 2 |
| `nn/_so3act.py` | 1 | 0 | 1 | 0 | 0 | 2 |

## math/_bessel.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `bessel` | Triton 数值路径 | 直接 kernel 或组合调用 | Bessel 基函数前向调用 _bessel_kernel。 |

## math/_linalg.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `_conditional_script` | 无需 Triton（控制/元数据/无计算） | — | JIT 装饰器分派，Python 控制，无需 Triton。 |
| `direct_sum` | Triton 数值路径 | 直接 kernel 或组合调用 | Triton kernel 组装块对角矩阵。 |
| `orthonormalize` | Triton 数值路径 | 直接 kernel 或组合调用 | CUDA FP32/FP64 无梯度路径逐候选调用 Gram–Schmidt kernel，投影、范数、归一化、阈值及符号在 Triton 中完成；Python 读取最终秩并整理输出。尺寸及 eps 有限制，其他配置回退。 |
| `complete_basis` | Triton 数值路径 | 直接 kernel 或组合调用 | 输入基由 Triton 逐行归一化，单位向量候选复用 Gram–Schmidt kernel；保留原生输入假设、顺序及符号。支持范围同 orthonormalize。 |

## math/_normalize_activation.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `moment` | 混合接入 | 当前入口接入 | torch 保留随机采样及任意 f(z)，Triton 实现幂与均值归约；不支持的设备、幂次或梯度条件回退。 |
| `normalize2mom.forward` | 原生前向封装 | — | 原生前向封装：执行 self.f(x) 及可选 cst 缩放，并不调用 moment。moment 仅在 __init__ 估计常数；当前重导出类仍绑定原生 moment。若 f 本身是 Triton callable，可能运行时间接接入，但不能据此把默认原生类计为已迁移。nn.Activation 的支持路径可另行融合激活及 cst。 |
| `normalize2mom._make_tracing_inputs` | 无需 Triton（控制/元数据/无计算） | — | 生成 JIT tracing 样例输入，无需 Triton。 |

## math/_reduce.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `germinate_formulas` | 无需 Triton（控制/元数据/无计算） | — | 字符串公式及置换群规则生成，主要是 Python 控制和元数据，无需 GPU kernel。 |
| `reduce_permutation` | 混合接入 | 当前入口接入 | Python 保留维度检查、轨道枚举、筛选、排序、符号及 ret；CUDA FP32/FP64 由 Triton 清零并写入 Q；其他配置回退原生。 |

## math/_soft_one_hot_linspace.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `soft_one_hot_linspace` | Triton 数值路径 | 直接 kernel 或组合调用 | 五种 basis 及 cutoff 的前向由 _basis_kernel 实现。 |

## math/_soft_unit_step.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `_SoftUnitStep.forward` | Triton 数值路径 | 直接 kernel 或组合调用 | 函数入口及自定义 forward/backward 均调用 Triton；不据此承诺二阶梯度。 |
| `_SoftUnitStep.backward` | Triton 数值路径 | 直接 kernel 或组合调用 | 函数入口及自定义 forward/backward 均调用 Triton；不据此承诺二阶梯度。 |
| `soft_unit_step` | Triton 数值路径 | 直接 kernel 或组合调用 | 函数入口及自定义 forward/backward 均调用 Triton；不据此承诺二阶梯度。 |

## math/perm.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `is_perm` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `identity` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `compose` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `inverse` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `rand` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `from_int` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `to_int` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `group` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `germinate` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `is_group` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `to_cycles` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `sign` | 无需 Triton（控制/元数据/无计算） | — | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `standard_representation` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用本包 Triton complete_basis，再以索引 gather 与归约融合 A @ P @ A.T。支持 CUDA FP32/FP64、1≤n≤4096；其他配置回退。 |
| `natural_representation` | Triton 数值路径 | 直接 kernel 或组合调用 | Triton 填充自然置换表示矩阵。 |

## o3/_angular_spherical_harmonics.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `_conditional_script` | 无需 Triton（控制/元数据/无计算） | — | JIT 装饰器分派，Python 控制，无需 Triton。 |
| `SphericalHarmonicsAlphaBeta.forward` | Triton 数值路径 | 直接 kernel 或组合调用 | 受支持的 GPU 前向使用 alpha、乘法或融合球谐 kernel。 |
| `spherical_harmonics_alpha_beta` | 间接接入 Triton | 通过子模块/辅助函数 | 受支持的 GPU 前向使用 alpha、乘法或融合球谐 kernel。 |
| `spherical_harmonics_alpha` | Triton 数值路径 | 直接 kernel 或组合调用 | 受支持的 GPU 前向使用 alpha、乘法或融合球谐 kernel。 |
| `_poly_legendre` | 无需 Triton（控制/元数据/无计算） | — | SymPy 多项式系数生成，在初始化阶段由 CPU 完成，无需 Triton。 |
| `_sympy_legendre` | 无需 Triton（控制/元数据/无计算） | — | SymPy 符号微分及阶乘表达式，CPU 符号预处理，无需 Triton。 |
| `_mul_m_lm` | Triton 数值路径 | 直接 kernel 或组合调用 | 受支持的 GPU 前向使用 alpha、乘法或融合球谐 kernel。 |
| `Legendre.forward`（新增，分母外） | Triton 数值路径 | 直接 kernel | Legendre 多项式求值；替代原生 FX 动态生成的 forward。 |

## o3/_linear.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `Linear.forward` | Triton 数值路径 | 直接 kernel 或组合调用 | 仅无梯度、单输入/输出 irrep、单 instruction、共享权重且无 bias 路径使用 Triton；一般 Linear 回退 e3nn。 |
| `Linear.weight_view_for_instruction` | 无需 Triton（控制/元数据/无计算） | — | 权重视图、切片和迭代元数据，不需要独立 GPU 数值 kernel。 |
| `Linear.weight_views` | 无需 Triton（控制/元数据/无计算） | — | 权重视图、切片和迭代元数据，不需要独立 GPU 数值 kernel。 |
| `_codegen_linear` | 无需 Triton（控制/元数据/无计算） | — | FX/代码生成及指令构造，保留 CPU 元编程；运行计算由 Linear.forward 分派。 |

## o3/_norm.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `Norm.forward` | Triton 数值路径 | 直接 kernel 或组合调用 | CUDA FP32/FP64 无输入梯度的 irrep 范数计算使用 Triton；梯度路径回退。 |

## o3/_reduce.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `_wigner_nj` | 沿用原生数值路径 | — | 递归 CG/Wigner 耦合及约化基构造，尚未迁移，保留原生系数构造流程。 |
| `_get_ops` | 无需 Triton（控制/元数据/无计算） | — | 操作图遍历与元数据收集，保留 Python/FX。 |
| `ReducedTensorProducts.forward` | 沿用原生数值路径 | — | 直接导入原生类，依赖生成的 TensorProduct/FX 图和 autograd；TensorProduct 在本次范围外。 |

## o3/_rotation.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `rand_matrix` | 间接混合接入 | 通过子模块/辅助函数 | 保留 torch.rand 的调用形状、次序与 RNG 状态，后续转换调用 Triton。 |
| `identity_angles` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `rand_angles` | 混合接入 | 当前入口接入 | 保留 torch.rand 的调用形状、次序与 RNG 状态，后续转换调用 Triton。 |
| `compose_angles` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `inverse_angles` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `identity_quaternion` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `rand_quaternion` | 间接混合接入 | 通过子模块/辅助函数 | 保留 torch.rand 的调用形状、次序与 RNG 状态，后续转换调用 Triton。 |
| `compose_quaternion` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `inverse_quaternion` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `rand_axis_angle` | 间接混合接入 | 通过子模块/辅助函数 | 保留 torch.rand 的调用形状、次序与 RNG 状态，后续转换调用 Triton。 |
| `compose_axis_angle` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_x` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_y` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_z` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `angles_to_matrix` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_to_angles` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `angles_to_quaternion` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_to_quaternion` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `axis_angle_to_quaternion` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `quaternion_to_axis_angle` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_to_axis_angle` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `angles_to_axis_angle` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `axis_angle_to_matrix` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `quaternion_to_matrix` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `quaternion_to_angles` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `axis_angle_to_angles` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `angles_to_xyz` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `xyz_to_angles` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |

## o3/_s2grid.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `_quadrature_weights` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `s2_grid` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `spherical_harmonics_s2_grid` | 混合接入 | 当前入口接入 | 网格、Legendre 和 alpha 因子已调用 Triton；beta.cos()/sin()/abs() 仍用 torch。 |
| `_complete_lmax_res` | 无需 Triton（控制/元数据/无计算） | — | 整数分辨率/阶数推断和合法性判断，Python 元数据逻辑，无需 Triton。 |
| `_expand_matrix` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `rfft` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `irfft` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `ToS2Grid.grid` | 间接接入 Triton | 通过子模块/辅助函数 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `ToS2Grid.forward` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `ToS2Grid._make_tracing_inputs` | 无需 Triton（控制/元数据/无计算） | — | 生成 JIT tracing 样例输入，无需 Triton。 |
| `FromS2Grid.grid` | 间接接入 Triton | 通过子模块/辅助函数 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `FromS2Grid.forward` | Triton 数值路径 | 直接 kernel 或组合调用 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `FromS2Grid._make_tracing_inputs` | 无需 Triton（控制/元数据/无计算） | — | 生成 JIT tracing 样例输入，无需 Triton。 |

## o3/_so3grid.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `flat_wigner` | 沿用原生数值路径 | — | 保留原生 Wigner 矩阵与各阶缩放/拼接；用于初始化，Wigner 数值路径尚未迁移。 |
| `SO3Grid.to_grid` | Triton 数值路径 | 直接 kernel 或组合调用 | 两个方向的网格收缩各调用 Triton kernel；构造仍为原生。 |
| `SO3Grid.from_grid` | Triton 数值路径 | 直接 kernel 或组合调用 | 两个方向的网格收缩各调用 Triton kernel；构造仍为原生。 |

## o3/_spherical_harmonics.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `SphericalHarmonics.forward` | 沿用原生数值路径 | — | 保留原生各阶笛卡尔多项式及 autograd；当前只有角度球谐有 Triton 实现。 |
| `spherical_harmonics` | 沿用原生数值路径 | — | 原生类的函数入口；仍走笛卡尔球谐 torch 多项式计算。 |
| `_spherical_harmonics` | 沿用原生数值路径 | — | 支持至 l=12 的生成多项式尚未迁移；需保留完整阶数及梯度能力。并非算法不能由 Triton 实现。 |

## o3/_spherical_harmonics_generator.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `_generate_spherical_harmonics` | 无需 Triton（控制/元数据/无计算） | — | CPU 符号推导及源码生成，不是运行时 GPU 数值算子。 |

## o3/_wigner.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `su2_generators` | 沿用原生数值路径 | — | 复数升降算符和小矩阵构造尚未迁移，保留原生 torch。 |
| `change_basis_real_to_complex` | 沿用原生数值路径 | — | 复数基变换矩阵构造尚未迁移，主要用于系数预处理。 |
| `so3_generators` | 沿用原生数值路径 | — | 复数基变换和小矩阵乘法尚未迁移，保留原生 torch。 |
| `wigner_D` | 沿用原生数值路径 | — | 矩阵指数及三段矩阵乘法尚未迁移；当前上游还存在 CPU 生成矩阵与 CUDA 角度设备不匹配限制。 |
| `wigner_3j` | 沿用原生数值路径 | — | CG 系数构造、缓存及设备复制仍为原生路径，适合先预计算再迁移运行计算。 |
| `_so3_clebsch_gordan` | 沿用原生数值路径 | — | CG 系数与复数/实数基变换仍为原生构造，尚未迁移。 |
| `_su2_clebsch_gordan` | 沿用原生数值路径 | — | 离散量子数组合遍历及系数填充在 CPU 构造，尚未迁移。 |
| `_su2_clebsch_gordan_coeff` | 无需 Triton（控制/元数据/无计算） | — | 阶乘与求和的标量系数公式，Python 预计算，不需独立 GPU kernel。 |

## nn/_activation.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `Activation.forward` | Triton 数值路径 | 直接 kernel 或组合调用 | CUDA 推理的 abs/tanh/sigmoid/relu/silu 及二阶矩缩放融合到 Triton；任意 callable、非末维或梯度条件回退。 |

## nn/_batchnorm.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `BatchNorm._roll_avg` | 沿用原生数值路径 | — | 训练态 running statistics 更新仍用 torch；当前仅推理归一化接入 Triton。 |
| `BatchNorm.forward` | Triton 数值路径 | 直接 kernel 或组合调用 | 仅 eval、非 instance、无梯度路径调用 Triton；训练/instance/梯度路径仍为 e3nn。 |

## nn/_dropout.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `Dropout.forward` | 混合接入 | 当前入口接入 | 保留 torch.bernoulli_、缩放和掩码拼接；Triton 广播掩码并乘输入。仅受支持的训练前向路径接入，梯度等条件回退。 |

## nn/_extract.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `Extract.forward` | Triton 数值路径 | 直接 kernel 或组合调用 | CUDA 无梯度的 gather 使用 Triton；ExtractIr 继承此 forward，未重复计数。 |

## nn/_fc.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `_Layer.forward` | 沿用原生数值路径 | — | 保留 torch GEMM、任意激活及训练 autograd；当前未开发 Triton GEMM/激活融合。 |

## nn/_gate.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `_Sortcut.forward` | 间接接入 Triton | 通过子模块/辅助函数 | 排序截取通过替换的 Extract 调用 Triton；支持的激活与门控乘法调用 Triton，存在 e3nn 回退路径。 |
| `Gate.forward` | Triton 数值路径 | 直接 kernel 或组合调用 | 排序截取通过替换的 Extract 调用 Triton；支持的激活与门控乘法调用 Triton，存在 e3nn 回退路径。 |
| `Gate.irreps_in` | 无需 Triton（控制/元数据/无计算） | — | 返回 Irreps 元数据的属性，无需 Triton。 |
| `Gate.irreps_out` | 无需 Triton（控制/元数据/无计算） | — | 返回 Irreps 元数据的属性，无需 Triton。 |

## nn/_identity.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `Identity.forward` | 无需 Triton（控制/元数据/无计算） | — | 直接返回输入，无数值计算或拷贝，不需要 Triton kernel。 |

## nn/_normact.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `NormActivation.forward` | 间接混合接入 | 通过子模块/辅助函数 | 方法继承 e3nn，self.norm 被替换为 Triton Norm；截断、sqrt、bias、任意激活、除法及 ElementwiseTensorProduct 仍用 torch/e3nn。 |

## nn/_s2act.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `S2Activation.forward` | 间接混合接入 | 通过子模块/辅助函数 | CUDA 推理的球面投影调用 Triton；任意激活及随机旋转的 einsum/Wigner 仍用 torch/e3nn。CPU/梯度路径回退。 |

## nn/_so3act.py

| 原始函数/方法 | 状态 | 接入方式 | 范围或保留原因 |
|---|---|---|---|
| `SO3Activation.forward` | 间接混合接入 | 通过子模块/辅助函数 | CUDA 推理的 to_grid/from_grid 调用 Triton；任意激活仍用 torch。CPU/梯度路径回退。 |