# EQv3 算子正确性与性能汇总

FastEq `3bc9a82d40ef`；H100 80GB，FP32，Torch 2.11.0 / Triton 3.6.0。
本次测量独立算子的公开调用；没有计时完整 EQv3 模型、优化器或纯反向。

**当前存在未通过项，不能将整组实现标记为已通过正确性。** AttentionAlpha 的大 N 参数梯度超差；GraphSoftmax 的内置广播 rescale 梯度测试超差；EQv3 GateActivation 公开类缺少 forward。逐档失败见下表与 failures.csv。

## 对照范围

| 项目 | Torch 对照 | 当前实现与验证边界 |
|---|---|---|
| GraphSoftmax | EQv3 原始 `GraphSoftmax` | 主表复用 CSR；另记每次建 CSR 的完整调用。内置广播 rescale 反向测试存在超差，见问题记录。 |
| AttentionAlpha | EQv3 attention 的 LayerNorm → SmoothLeakyReLU → Dropout → einsum | 前向融合；反向通过 Torch 重算。主测试 dropout=0。 |
| e3nn Gate | 原始 `e3nn.nn.Gate` | 标量激活、门控激活、广播乘法融合；不是 EQv3 `GateActivation`。 |
| LayerNorm / Separable | EQv3 原始两个类 | 分别按阶、标量与高阶分组；共享融合实现。默认反向含 Torch 运算与参数归约。 |
| EquivariantDropout | EQv3 原始 `EquivariantDropout` | 融合随机掩码、广播与乘法；p=0.3、training=True，前向在 no_grad 下仍实际执行 dropout。 |
| EQv3 GateActivation | EQv3 原始 `GateActivation` | 当前公开类缺少 `forward()`，直接调用失败，不能给出有效加速比。 |

## 正确性

删减适配范围后的 LayerNorm 回归：**218 passed，0 failed，0 skipped**。配置了 EQv2、EQv3 的原始 Torch 源码。性能数据沿用上述旧提交，不是本次重新计时。
独立全张量输出/梯度对照：{'PASS': 93, 'FAIL': 2}。所有共同规模均执行独立精度检查；每个张量同时保存 32 个数值样本。
逐元素条件为 `abs(fused - torch) <= atol + rtol * abs(torch)`：普通项目 atol=5e-5、rtol=5e-4；GraphSoftmax 前向 3e-6/3e-5、反向 3e-5/3e-4；Dropout 对同一掩码要求完全相等。容差未放宽。
Dropout 比较时将相同已采样掩码送入原始类，保留原始 forward 的分组、广播、乘法逻辑；计时仍各自正常生成随机掩码。非连续布局、C=13 等补充检查记录在 extra_checks.json。

| 项目 | 对照组数 | 状态 | 最大通过 N（前向 / 前向+反向） |
|---|---:|---|---:|
| GraphSoftmax | 19 | {'PASS': 19} | 8,388,608 / 4,194,304 |
| AttentionAlpha | 14 | {'PASS': 12, 'FAIL': 2} | 262,144 / 65,536 |
| e3nn Gate | 15 | {'PASS': 15} | 524,288 / 524,288 |
| LayerNorm | 15 | {'PASS': 15} | 524,288 / 524,288 |
| SeparableLayerNorm | 15 | {'PASS': 15} | 524,288 / 524,288 |
| EquivariantDropout | 17 | {'PASS': 17} | 2,097,152 / 1,048,576 |

| 超差项目 | N | 梯度/输出 | 超差元素数 | 最大容差比 |
|---|---:|---|---:|---:|
| AttentionAlpha | 131,072 | alpha_dot | 2 | 7.681 |
| AttentionAlpha | 32,768 | alpha_dot | 1 | 2.626 |

## N=4096 对比

LayerNorm/Dropout 输入 [N,16,128]；Gate 输入 [N,2432]、输出 [N,2048]；图算子 E=32N，H=8，AttentionAlpha 通道=32。

| 项目 | N | 模式 | Torch ms | FastEq ms | 倍率 | 精度 |
|---|---:|---|---:|---:|---:|---|
| GraphSoftmax | 4,096 | fwd | 0.2967 | 0.0616 | 4.82× | PASS |
| GraphSoftmax | 4,096 | fwd_bwd | 0.4704 | 0.3810 | 1.23× | PASS |
| AttentionAlpha | 4,096 | fwd | 2.3449 | 0.6840 | 3.43× | PASS |
| AttentionAlpha | 4,096 | fwd_bwd | 4.5485 | 6.0283 | 0.75× | PASS |
| e3nn Gate | 4,096 | fwd | 0.2278 | 0.1048 | 2.17× | PASS |
| e3nn Gate | 4,096 | fwd_bwd | 1.2874 | 0.4136 | 3.11× | PASS |
| LayerNorm | 4,096 | fwd | 0.3847 | 0.1317 | 2.92× | PASS |
| LayerNorm | 4,096 | fwd_bwd | 1.6003 | 0.8641 | 1.85× | PASS |
| SeparableLayerNorm | 4,096 | fwd | 0.1900 | 0.1228 | 1.55× | PASS |
| SeparableLayerNorm | 4,096 | fwd_bwd | 0.7303 | 0.5221 | 1.40× | PASS |
| EquivariantDropout | 4,096 | fwd | 0.1083 | 0.1767 | 0.61× | PASS |
| EquivariantDropout | 4,096 | fwd_bwd | 0.2162 | 0.2557 | 0.85× | PASS |

## 最大共同规模

每个模式分别列最后一个双方可运行的规模。表内倍率仅针对该配置；GraphSoftmax 的其他配置超差仍需修复。

| 项目 | N | 模式 | Torch ms | FastEq ms | 倍率 | 精度 | 峰值 GiB（Torch / FastEq） |
|---|---:|---|---:|---:|---:|---|---:|
| GraphSoftmax | 8,388,608 | fwd | 400.0083 | 18.0312 | 22.18× | PASS | 43.250 / 21.063 |
| GraphSoftmax | 4,194,304 | fwd_bwd | 342.3557 | 34.3892 | 9.96× | PASS | 45.500 / 27.031 |
| AttentionAlpha | 262,144 | fwd | 152.8908 | 40.3675 | 3.79× | PASS | 48.031 / 8.250 |
| AttentionAlpha | 131,072 | fwd_bwd | 142.3553 | 180.3597 | 0.79× | DIAGNOSTIC | 32.438 / 40.500 |
| e3nn Gate | 524,288 | fwd | 16.5866 | 5.1104 | 3.25× | PASS | 17.000 / 8.750 |
| e3nn Gate | 524,288 | fwd_bwd | 102.0716 | 34.3886 | 2.97× | PASS | 28.500 / 17.500 |
| LayerNorm | 524,288 | fwd | 13.9120 | 7.1032 | 1.96× | PASS | 12.252 / 8.000 |
| LayerNorm | 524,288 | fwd_bwd | 67.7502 | 45.8218 | 1.48× | PASS | 23.008 / 24.270 |
| SeparableLayerNorm | 524,288 | fwd | 15.7124 | 6.1454 | 2.56× | PASS | 15.781 / 8.000 |
| SeparableLayerNorm | 524,288 | fwd_bwd | 61.9116 | 28.9391 | 2.14× | PASS | 30.816 / 31.758 |
| EquivariantDropout | 2,097,152 | fwd | 39.0111 | 14.6830 | 2.66× | PASS | 48.000 / 32.000 |
| EquivariantDropout | 1,048,576 | fwd_bwd | 27.8281 | 14.7071 | 1.89× | PASS | 40.000 / 32.000 |

## 扩容边界

| 项目 | 模式 | 后端 | 最后可运行 N | 下一档 N | 停止原因 |
|---|---|---|---:|---:|---|
| GraphSoftmax | fwd | torch | 8,388,608 | 16777216 | OOM |
| GraphSoftmax | fwd | fused | 16,777,216 | 33554432 | OOM |
| GraphSoftmax | fwd_bwd | torch | 4,194,304 | 8388608 | OOM |
| GraphSoftmax | fwd_bwd | fused | 8,388,608 | 16777216 | OOM |
| AttentionAlpha | fwd | torch | 262,144 | 524288 | OOM |
| AttentionAlpha | fwd | fused | 524,288 | 1048576 | FALLBACK |
| AttentionAlpha | fwd_bwd | torch | 262,144 | 524288 | OOM |
| AttentionAlpha | fwd_bwd | fused | 131,072 | 262144 | OOM |
| e3nn Gate | fwd | torch | 1,048,576 | 2097152 | OOM |
| e3nn Gate | fwd | fused | 524,288 | 1048576 | INDEX_GUARD |
| e3nn Gate | fwd_bwd | torch | 1,048,576 | 2097152 | OOM |
| e3nn Gate | fwd_bwd | fused | 524,288 | 1048576 | INDEX_GUARD |
| LayerNorm | fwd | torch | 2,097,152 | 4194304 | OOM |
| LayerNorm | fwd | fused | 524,288 | 1048576 | LIMIT |
| LayerNorm | fwd_bwd | torch | 1,048,576 | 2097152 | OOM |
| LayerNorm | fwd_bwd | fused | 524,288 | 1048576 | LIMIT |
| SeparableLayerNorm | fwd | torch | 1,048,576 | 2097152 | OOM |
| SeparableLayerNorm | fwd | fused | 524,288 | 1048576 | LIMIT |
| SeparableLayerNorm | fwd_bwd | torch | 1,048,576 | 2097152 | OOM |
| SeparableLayerNorm | fwd_bwd | fused | 524,288 | 1048576 | LIMIT |
| EquivariantDropout | fwd | torch | 2,097,152 | 4194304 | OOM |
| EquivariantDropout | fwd | fused | 4,194,304 | 8388608 | OOM |
| EquivariantDropout | fwd_bwd | torch | 1,048,576 | 2097152 | OOM |
| EquivariantDropout | fwd_bwd | fused | 2,097,152 | 4194304 | OOM |

`OOM` 是实际分配失败；`LIMIT` 是公开接口索引限制；`INDEX_GUARD` 是测试在源码缺少检查的有符号 32 位地址溢出前停止；`FALLBACK` 表示公开接口会退回 Torch，不继续把它计成 Triton。完整错误、失败阶段和内存占用见 boundaries.csv / points。
`LIMIT`、`INDEX_GUARD`、`FALLBACK` 档由测试脚本按已核实的源码条件预检停止，未分配超限输入；它们不是实测 OOM。双方都能运行的规模才有完整逐元素对照。

## 计时与内存口径

N 从 256 连续翻倍。每个后端、模式、规模使用新的子进程；调度进程不建立 GPU 上下文。预热 5 次，测量 20 次，报告中位数；JIT、输入初始化不计时。GPU event 区间包含公开算子调用期间的 GPU 工作与 CPU 提交间隙，并非单个 Triton kernel 延迟；同时保存同步墙钟时间。
`fwd` 为 no_grad；`fwd_bwd` 包含前向、输入梯度、全部可学习参数梯度，没有优化器。峰值为 Torch allocator allocated bytes，含输入、输出、梯度与临时张量，不能当作整卡占用。
GraphSoftmax 的 CSR 构建包含排序和主机同步。主表复用同一拓扑的 CSR；N≤65536 另测每次重建 CSR 的完整调用，其墙钟倍率见 paired.csv 的 with_csr_wall_speedup。
主表 GraphSoftmax 峰值对应复用 CSR 阶段，包含常驻 CSR，但不包含一次性构建时的临时峰值；含构建测量的峰值另存 fused_with_csr_peak_gib。

## HIP 已有结果复核

复用仓库 2026-09-15 的 BW gfx936 记录：当时的 LayerNorm 源码、测试、参考与结果哈希均已核对；旧完整套件为 305 项通过，不代表本次删改后的 HIP 验证。以下为 N=4096 代表点，本次没有在 HIP 重新扩容，也没有为其他四类补做 HIP 验证。

| LayerNorm 变体 | 模式 | Torch ms | FastEq ms | 倍率 |
|---|---|---:|---:|---:|
| EquivariantLayerNorm | forward | 0.8011 | 0.3758 | 2.13× |
| EquivariantLayerNorm | forward_backward | 2.7340 | 1.8670 | 1.46× |
| EquivariantSeparableLayerNorm | forward | 0.7174 | 0.3911 | 1.83× |
| EquivariantSeparableLayerNorm | forward_backward | 2.1048 | 1.2849 | 1.64× |

## GraphSoftmax 重复性复核

首次内置自测超差，随后一次 344 断言的复跑通过。再将自测参考替换为 EQv3 原始类，重复 5 轮、1720 个断言，出现 1 项失败。该问题表现为偶发超差；没有通过放宽容差消除它。
配置 {'E': 73, 'N': 17, 'H': 3, 'cap': 3.0, 'eps': 1e-16, 'rshape': [1, 3], 'p': 0.0}；梯度 exp_rescale；最大容差比 1.016865。失败输入与梯度保存于 `softmax_failures/native_4_1563.pt`，后续可直接复现诊断。

## 文件

- `paired.csv`：全部共同规模、GPU/墙钟计时、峰值内存、加速比与精度状态。
- `points/*.json`：每次独立计时的 20 个样本、输出/梯度数值、输入样本、环境和 OOM 错误。
- `checks/*.json`、`correctness_summary.json`：完整逐元素对照、最坏误差、失败位置。
- `extra_checks.json`、`softmax_diagnosis.json`、`softmax_native_checks.json`：额外测试、重复检查与原始 EQv3 参照下的失败配置。
- `layernorm.xml`、`layernorm.log`：删减适配范围后的 218 项回归；源码与测试哈希见 regression.json。
- `manifest.json`：源码与验证范围；`cases.py`、`run.py`、`extras.py` 为复现脚本。

测试数据不修改生产算子实现。
