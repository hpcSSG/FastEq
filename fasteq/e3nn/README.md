# e3nn math / o3 / nn Triton 实现清单

统计日期：2026-09-30。源代码基线提交：`2aa7f58440a06b15352a2cbce01fa4c26f824969`。

## 统计口径

- 以原始 e3nn 文件中的顶层 `def` 和显式类方法为单位，包含 `__init__`、`__repr__`、property getter、私有函数。
- 不计嵌套局部函数、导入符号、Triton 新增 kernel/辅助函数，避免扩大迁移数量。
- 范围是 math 的 7 个文件、o3 的 10 个文件、nn 的 10 个文件；沿用此前排除 `_tensor_product`、experimental、irrep/`_irreps.py` 和 nn/models 的范围。`__init__.py` 导出表不单独计数。
- Triton计算路径：指受支持路径的主要数值计算由 Triton 完成，可以调用一个或多个 kernel，允许 Python 分派、shape/view 和初始化。存在回退分支的函数不会被宣称所有配置已迁移。
- 混合接入： 指已调用 Triton，但 RNG、任意激活或其他主要张量运算仍使用 torch/e3nn。
- 未接入：同时包含尚未迁移的数值计算和不需要迁移的构造、元数据、符号代码；逐函数说明区分二者。

## 汇总

| 模块 | 原始 def | Triton计算路径 | 混合接入 | 已接入合计 | 未接入 |
|---|---:|---:|---:|---:|---:|
| math | 29 | 7 | 1 | 8 | 21 |
| o3 | 86 | 41 | 5 | 46 | 40 |
| nn | 36 | 5 | 4 | 9 | 27 |
| 合计 | 151 | 53 | 10 | 63 | 88 |

## 按文件统计

| 文件 | 原始def | Triton计算路径 | 混合接入 | 未接入 |
|---|---:|---:|---:|---:|
| `math/_bessel.py` | 1 | 1 | 0 | 0 |
| `math/_linalg.py` | 4 | 1 | 0 | 3 |
| `math/_normalize_activation.py` | 4 | 0 | 1 | 3 |
| `math/_reduce.py` | 2 | 0 | 0 | 2 |
| `math/_soft_one_hot_linspace.py` | 1 | 1 | 0 | 0 |
| `math/_soft_unit_step.py` | 3 | 3 | 0 | 0 |
| `math/perm.py` | 14 | 1 | 0 | 13 |
| `o3/_angular_spherical_harmonics.py` | 9 | 4 | 0 | 5 |
| `o3/_linear.py` | 6 | 1 | 0 | 5 |
| `o3/_norm.py` | 3 | 1 | 0 | 2 |
| `o3/_reduce.py` | 5 | 0 | 0 | 5 |
| `o3/_rotation.py` | 28 | 24 | 4 | 0 |
| `o3/_s2grid.py` | 17 | 9 | 1 | 7 |
| `o3/_so3grid.py` | 5 | 2 | 0 | 3 |
| `o3/_spherical_harmonics.py` | 4 | 0 | 0 | 4 |
| `o3/_spherical_harmonics_generator.py` | 1 | 0 | 0 | 1 |
| `o3/_wigner.py` | 8 | 0 | 0 | 8 |
| `nn/_activation.py` | 3 | 1 | 0 | 2 |
| `nn/_batchnorm.py` | 4 | 1 | 0 | 3 |
| `nn/_dropout.py` | 3 | 0 | 1 | 2 |
| `nn/_extract.py` | 3 | 1 | 0 | 2 |
| `nn/_fc.py` | 5 | 0 | 0 | 5 |
| `nn/_gate.py` | 7 | 2 | 0 | 5 |
| `nn/_identity.py` | 3 | 0 | 0 | 3 |
| `nn/_normact.py` | 2 | 0 | 1 | 1 |
| `nn/_s2act.py` | 3 | 0 | 1 | 2 |
| `nn/_so3act.py` | 3 | 0 | 1 | 2 |

## math/_bessel.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `bessel` | Triton计算路径 | Bessel 基函数前向调用 _bessel_kernel。 |

## math/_linalg.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `_conditional_script` | 未接入 | JIT 装饰器分派，Python 控制，无需 Triton。 |
| `direct_sum` | Triton计算路径 | 块对角矩阵组装调用 _direct_sum_kernel。 |
| `orthonormalize` | 未接入 | 尚未迁移：顺序 Gram–Schmidt、秩判定、阈值清零及动态输出长度需要专门设计；并非不能用 Triton。 |
| `complete_basis` | 未接入 | 尚未迁移：基向量逐步扩展、投影依赖和数据相关输出长度需要多阶段或小矩阵专用实现。 |

## math/_normalize_activation.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `moment` | 混合接入 | torch 保留随机采样及任意 f(z)，Triton 实现幂与均值归约；不支持的设备、幂次或梯度条件回退。 |
| `normalize2mom.__init__` | 未接入 | 保留原生一次性二阶矩估计与标量初始化；内部调用的是原生 moment，并非此包的 Triton moment。 |
| `normalize2mom.forward` | 未接入 | 直接导入原生 normalize2mom；任意 f 与缩放仍用 torch。单独实现 moment 不会改变原生类内部的全局函数绑定。 |
| `normalize2mom._make_tracing_inputs` | 未接入 | 生成 JIT tracing 样例输入，无需 Triton。 |

## math/_reduce.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `germinate_formulas` | 未接入 | 字符串公式及置换群规则生成，主要是 Python 控制和元数据，无需 GPU kernel。 |
| `reduce_permutation` | 未接入 | 置换群遍历、张量索引、约束空间与正交基构造尚未迁移；动态流程与线性代数不适合直接通用单 kernel。 |

## math/_soft_one_hot_linspace.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `soft_one_hot_linspace` | Triton计算路径 | 五种 basis 及 cutoff 的前向由 _basis_kernel 实现。 |

## math/_soft_unit_step.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `_SoftUnitStep.forward` | Triton计算路径 | 函数入口及自定义 forward/backward 均调用 Triton；不据此承诺二阶梯度。 |
| `_SoftUnitStep.backward` | Triton计算路径 | 函数入口及自定义 forward/backward 均调用 Triton；不据此承诺二阶梯度。 |
| `soft_unit_step` | Triton计算路径 | 函数入口及自定义 forward/backward 均调用 Triton；不据此承诺二阶梯度。 |

## math/perm.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `is_perm` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `identity` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `compose` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `inverse` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `rand` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `from_int` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `to_int` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `group` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `germinate` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `is_group` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `to_cycles` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `sign` | 未接入 | 置换元组/整数转换、群枚举、循环与符号判断在 Python 处理，无需 GPU kernel。 |
| `standard_representation` | 未接入 | 仍调用原生自然表示与正交基构造；原生函数的内部绑定不会自动使用本包 natural_representation。依赖 complete_basis 尚未迁移。 |
| `natural_representation` | Triton计算路径 | 自然置换表示矩阵由 _natural_kernel 填充。 |

## o3/_angular_spherical_harmonics.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `_conditional_script` | 未接入 | JIT 装饰器分派，Python 控制，无需 Triton。 |
| `SphericalHarmonicsAlphaBeta.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `SphericalHarmonicsAlphaBeta.forward` | Triton计算路径 | 受支持的 GPU 前向使用 alpha、乘法或融合球谐 kernel。 |
| `spherical_harmonics_alpha_beta` | Triton计算路径 | 受支持的 GPU 前向使用 alpha、乘法或融合球谐 kernel。 |
| `spherical_harmonics_alpha` | Triton计算路径 | 受支持的 GPU 前向使用 alpha、乘法或融合球谐 kernel。 |
| `Legendre.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `_poly_legendre` | 未接入 | SymPy 多项式系数生成，在初始化阶段由 CPU 完成，无需 Triton。 |
| `_sympy_legendre` | 未接入 | SymPy 符号微分及阶乘表达式，CPU 符号预处理，无需 Triton。 |
| `_mul_m_lm` | Triton计算路径 | 受支持的 GPU 前向使用 alpha、乘法或融合球谐 kernel。 |
| `Legendre.forward`（新增，不进原始def分母） | Triton计算路径 | `_legendre_kernel` 求值；原生 FX 生成方法被显式 forward 替代。 |

## o3/_linear.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `Linear.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `Linear.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `Linear.forward` | Triton计算路径 | 仅无梯度、单输入/输出 irrep、单 instruction、共享权重且无 bias 路径使用 Triton；一般 Linear 回退 e3nn。 |
| `Linear.weight_view_for_instruction` | 未接入 | 权重视图、切片和迭代元数据，不需要独立 GPU 数值 kernel。 |
| `Linear.weight_views` | 未接入 | 权重视图、切片和迭代元数据，不需要独立 GPU 数值 kernel。 |
| `_codegen_linear` | 未接入 | FX/代码生成及指令构造，保留 CPU 元编程；运行计算由 Linear.forward 分派。 |

## o3/_norm.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `Norm.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `Norm.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `Norm.forward` | Triton计算路径 | CUDA FP32/FP64 无输入梯度的 irrep 范数计算使用 Triton；梯度路径回退。 |

## o3/_reduce.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `_wigner_nj` | 未接入 | 递归 CG/Wigner 耦合及约化基构造，尚未迁移，保留原生系数构造流程。 |
| `_get_ops` | 未接入 | 操作图遍历与元数据收集，保留 Python/FX。 |
| `ReducedTensorProducts.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `ReducedTensorProducts.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `ReducedTensorProducts.forward` | 未接入 | 直接导入原生类，依赖生成的 TensorProduct/FX 图和 autograd；TensorProduct 在本次范围外。 |

## o3/_rotation.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `rand_matrix` | 混合接入 | 保留 torch.rand 的调用形状、次序与 RNG 状态，后续转换调用 Triton。 |
| `identity_angles` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `rand_angles` | 混合接入 | 保留 torch.rand 的调用形状、次序与 RNG 状态，后续转换调用 Triton。 |
| `compose_angles` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `inverse_angles` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `identity_quaternion` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `rand_quaternion` | 混合接入 | 保留 torch.rand 的调用形状、次序与 RNG 状态，后续转换调用 Triton。 |
| `compose_quaternion` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `inverse_quaternion` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `rand_axis_angle` | 混合接入 | 保留 torch.rand 的调用形状、次序与 RNG 状态，后续转换调用 Triton。 |
| `compose_axis_angle` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_x` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_y` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_z` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `angles_to_matrix` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_to_angles` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `angles_to_quaternion` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_to_quaternion` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `axis_angle_to_quaternion` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `quaternion_to_axis_angle` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `matrix_to_axis_angle` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `angles_to_axis_angle` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `axis_angle_to_matrix` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `quaternion_to_matrix` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `quaternion_to_angles` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `axis_angle_to_angles` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `angles_to_xyz` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |
| `xyz_to_angles` | Triton计算路径 | 调用旋转 kernel 或组合多个 Triton 转换；不表示该 def 对应单 kernel。 |

## o3/_s2grid.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `_quadrature_weights` | Triton计算路径 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `s2_grid` | Triton计算路径 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `spherical_harmonics_s2_grid` | 混合接入 | 网格、Legendre 和 alpha 因子已调用 Triton；beta.cos()/sin()/abs() 仍用 torch。 |
| `_complete_lmax_res` | 未接入 | 整数分辨率/阶数推断和合法性判断，Python 元数据逻辑，无需 Triton。 |
| `_expand_matrix` | Triton计算路径 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `rfft` | Triton计算路径 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `irfft` | Triton计算路径 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `ToS2Grid.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `ToS2Grid.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `ToS2Grid.grid` | Triton计算路径 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `ToS2Grid.forward` | Triton计算路径 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `ToS2Grid._make_tracing_inputs` | 未接入 | 生成 JIT tracing 样例输入，无需 Triton。 |
| `FromS2Grid.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `FromS2Grid.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `FromS2Grid.grid` | Triton计算路径 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `FromS2Grid.forward` | Triton计算路径 | 调用网格、矩阵准备、Fourier 或球面投影 kernel；grid 属性通过 Triton angles_to_xyz 计算。 |
| `FromS2Grid._make_tracing_inputs` | 未接入 | 生成 JIT tracing 样例输入，无需 Triton。 |

## o3/_so3grid.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `flat_wigner` | 未接入 | 保留原生 Wigner 矩阵与各阶缩放/拼接；用于初始化，Wigner 数值路径尚未迁移。 |
| `SO3Grid.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `SO3Grid.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `SO3Grid.to_grid` | Triton计算路径 | 两个方向的网格收缩各调用 Triton kernel；构造仍为原生。 |
| `SO3Grid.from_grid` | Triton计算路径 | 两个方向的网格收缩各调用 Triton kernel；构造仍为原生。 |

## o3/_spherical_harmonics.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `SphericalHarmonics.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `SphericalHarmonics.forward` | 未接入 | 保留原生各阶笛卡尔多项式及 autograd；当前只有角度球谐有 Triton 实现。 |
| `spherical_harmonics` | 未接入 | 原生类的函数入口；仍走笛卡尔球谐 torch 多项式计算。 |
| `_spherical_harmonics` | 未接入 | 支持至 l=12 的生成多项式尚未迁移；需保留完整阶数及梯度能力。并非算法不能由 Triton 实现。 |

## o3/_spherical_harmonics_generator.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `_generate_spherical_harmonics` | 未接入 | CPU 符号推导及源码生成，不是运行时 GPU 数值算子。 |

## o3/_wigner.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `su2_generators` | 未接入 | 复数升降算符和小矩阵构造尚未迁移，保留原生 torch。 |
| `change_basis_real_to_complex` | 未接入 | 复数基变换矩阵构造尚未迁移，主要用于系数预处理。 |
| `so3_generators` | 未接入 | 复数基变换和小矩阵乘法尚未迁移，保留原生 torch。 |
| `wigner_D` | 未接入 | 矩阵指数及三段矩阵乘法尚未迁移；当前上游还存在 CPU 生成矩阵与 CUDA 角度设备不匹配限制。 |
| `wigner_3j` | 未接入 | CG 系数构造、缓存及设备复制仍为原生路径，适合先预计算再迁移运行计算。 |
| `_so3_clebsch_gordan` | 未接入 | CG 系数与复数/实数基变换仍为原生构造，尚未迁移。 |
| `_su2_clebsch_gordan` | 未接入 | 离散量子数组合遍历及系数填充在 CPU 构造，尚未迁移。 |
| `_su2_clebsch_gordan_coeff` | 未接入 | 阶乘与求和的标量系数公式，Python 预计算，不需独立 GPU kernel。 |

## nn/_activation.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `Activation.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `Activation.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `Activation.forward` | Triton计算路径 | CUDA 推理的 abs/tanh/sigmoid/relu/silu 及二阶矩缩放融合到 Triton；任意 callable、非末维或梯度条件回退。 |

## nn/_batchnorm.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `BatchNorm.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `BatchNorm.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `BatchNorm._roll_avg` | 未接入 | 训练态 running statistics 更新仍用 torch；当前仅推理归一化接入 Triton。 |
| `BatchNorm.forward` | Triton计算路径 | 仅 eval、非 instance、无梯度路径调用 Triton；训练/instance/梯度路径仍为 e3nn。 |

## nn/_dropout.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `Dropout.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `Dropout.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `Dropout.forward` | 混合接入 | 保留 torch.bernoulli_、缩放和掩码拼接；Triton 广播掩码并乘输入。仅受支持的训练前向路径接入，梯度等条件回退。 |

## nn/_extract.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `Extract.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `Extract.forward` | Triton计算路径 | CUDA 无梯度的 gather 使用 Triton；ExtractIr 继承此 forward，未重复计数。 |
| `ExtractIr.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |

## nn/_fc.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `_Layer.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `_Layer.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `_Layer.forward` | 未接入 | 保留 torch GEMM、任意激活及训练 autograd；当前未开发 Triton GEMM/激活融合。 |
| `FullyConnectedNet.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `FullyConnectedNet.__repr__` | 未接入 | 字符串展示，无需 Triton。 |

## nn/_gate.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `_Sortcut.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `_Sortcut.forward` | Triton计算路径 | 排序截取通过替换的 Extract 调用 Triton；支持的激活与门控乘法调用 Triton，存在 e3nn 回退路径。 |
| `Gate.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `Gate.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `Gate.forward` | Triton计算路径 | 排序截取通过替换的 Extract 调用 Triton；支持的激活与门控乘法调用 Triton，存在 e3nn 回退路径。 |
| `Gate.irreps_in` | 未接入 | 返回 Irreps 元数据的属性，无需 Triton。 |
| `Gate.irreps_out` | 未接入 | 返回 Irreps 元数据的属性，无需 Triton。 |

## nn/_identity.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `Identity.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `Identity.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `Identity.forward` | 未接入 | 直接返回输入，无数值计算或拷贝，不需要 Triton kernel。 |

## nn/_normact.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `NormActivation.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `NormActivation.forward` | 混合接入 | 方法继承 e3nn，self.norm 被替换为 Triton Norm；截断、sqrt、bias、任意激活、除法及 ElementwiseTensorProduct 仍用 torch/e3nn。 |

## nn/_s2act.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `S2Activation.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `S2Activation.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `S2Activation.forward` | 混合接入 | CUDA 推理的球面投影调用 Triton；任意激活及随机旋转的 einsum/Wigner 仍用 torch/e3nn。CPU/梯度路径回退。 |

## nn/_so3act.py

| 原始函数/方法 | 状态 | 实现范围或未接入原因 |
|---|---|---|
| `SO3Activation.__init__` | 未接入 | 类构造、参数/Irreps 校验、buffer/子模块或指令初始化保留 Python/e3nn；不作为运行计算迁移。 |
| `SO3Activation.__repr__` | 未接入 | 字符串展示，无需 Triton。 |
| `SO3Activation.forward` | 混合接入 | CUDA 推理的 to_grid/from_grid 调用 Triton；任意激活仍用 torch。CPU/梯度路径回退。 |

## 需要特别注意的连接关系

1. `normalize2mom` 直接重导出原生类，单独的 Triton `moment` 并未替换该类内部调用；但 `nn.Activation` 支持的激活路径会把 `normalize2mom` 的 cst 缩放融合到自己的 kernel。
2. `NormActivation.forward` 未重写，但 init 替换 `self.norm`，因此是混合接入，不能记为整个 forward 已由 Triton 实现。
3. `_Sortcut.forward` 未重写，但 init 替换 `self.cut` 为 Triton Extract，因此运行时确实能调用 Triton。
4. `rand_*` 和 Dropout 的 torch RNG 是按用户要求保留，旨在维持随机序列兼容，不属于忘记迁移。
5. `orthonormalize`、`complete_basis`、笛卡尔球谐及 Wigner 尚未迁移的原因是当前设计与实现取舍，不代表 Triton 无法实现。
6. S2/SO3 activation 的投影 kernel 已接入，但任意激活仍是 torch；S2 random_rot 的 GPU Wigner 设备限制仍存在。
