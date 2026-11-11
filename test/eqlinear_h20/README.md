# H20 eqlinear 测试工具

本目录是 H20 新测试工具，生产数学和默认选择策略未修改。交接包、旧 H100 producer、raw 和 receipt 均保持原字节。

所有运行从 session 的 `config/paths.yaml`、`config/scope.yaml` 读取实际路径；这两个文件采用 JSON 语法（合法 YAML），不依赖额外 YAML 包。只有用户指定的 uv 环境可安装依赖，所有缓存、临时输出和构建目录在工作盘。

## 模块

| 模块 | 职责 |
|---|---|
| `session.py` / `common.py` | 配置、缓存路径、连续两次空闲 GPU 采样、有总超时的进程组与退出 receipt |
| `preflight.py` | 交接包只读校验、H100 CPU 重算、环境/工具盘点；不运行目标 kernel |
| `registry.py` | 全版本清单、用户排除、P1–P30 源码/构建/精度/边界语义映射 |
| `artifacts.py` | 当前源码/新构建指纹、保留旧 DSO 后安装四个新产物 |
| `cases.py` | 唯一 fixture、真实 canonical descriptor、后乘 cg 的 FP64 forward/dx/dw 参考 |
| `protocol.py` | 统一精度 gate、wall/event/host timer、增量 raw journal、独立覆盖审计、trace kernel 分类 |
| `current.py` / `luna.py` / `standalone.py` | 各后端 prepare/call/identity/close，不各自生成随机输入或数学 gold |
| `*_probe.py` | 单点 JIT/backend/精度/重复/独立 buffer 探测，支持时采集 trace/NCU |
| `current_matrix.py` | 每 config 15 个 canonical 点的有界精度 chunk，逐点独立子进程、冻结 producer |
| `selfcheck.py` | 纯 CPU 反例测试；输出明确标识 synthetic，不能当 GPU 性能 |
| `stage_report.py` | 从 receipt/probe 生成明确 INCOMPLETE 的阶段报告和 CSV |

## 本机会话

已确认环境为 `/work-ict/venv/luna_env/bin/python`，当前会话为
`/work-ict/tmp/eqlinear_h20/20261107T095000Z_54b151f`。目录名是 session ID；各命令实际 UTC 起止时间见 receipt。
工作目录为 FastEq 根。示例：

```bash
/work-ict/venv/luna_env/bin/python -B test/eqlinear_h20/session.py run \
  --root /work-ict/tmp/eqlinear_h20/20261107T095000Z_54b151f \
  --name current_accuracy_LARGE_P3_chunk02 --deadline 300 -- \
  /work-ict/venv/luna_env/bin/python -B test/eqlinear_h20/current_matrix.py \
  --session /work-ict/tmp/eqlinear_h20/20261107T095000Z_54b151f \
  --config LARGE_P3 --gpu 4 --attempt cycle02
```

`session.py run --gpu N` 会先两次确认物理卡空闲，再以 UUID 设置逻辑设备；不指定 `--gpu` 的命令默认隐藏 CUDA 设备。
`current_matrix.py` 自行选择 GPU，外层不传 `--gpu`，避免父协调器持有 CUDA context。
NVML 在前一进程退出后可能短暂残留利用率；选择器最多采样6次，仍严格要求连续两次利用率0、低显存占用且无活跃进程。不改共享机器时钟/MIG/MPS，不终止其他用户任务。

所有输出名称/目录必须是新 attempt。观察工具超时后先检查原 session/PID；只有确认 terminal 或 handle 缺失才恢复。
`*.running.json` 只是定位线索，不能单独证明进程仍运行。不能把 partial cycle 与旧 cycle 混成完整轮。

## 已知边界

- 当前用户明确排除 TF32x1、FP16/BF16、TileLang 和 CUDA Graph；排除项仍在 registry 留痕。
- 不同家族 standalone 必须独立进程；namespace 不同本身不足以证明底层 C++ 符号隔离。
- 当前 FastEq Torch API 的 run 包含 activation preparation；不能标成 pure GEMM。
- Luna direct launch 仍分配输出，使用 `prepared_native_launch_with_alloc`，不能标成无分配 GEMM。
- `dx`、`dw`、`dx+dw` 分开记录；Luna 对原始权重的梯度包含布局与 cg/alpha 链式法则。
- 精度失败的安全 raw 诊断应保留，并排除合格排名；非法访问后停止该进程并调查。
- H100 冻结 P14 限制仅在全部 tensor/descriptor hash 匹配时适用。输入不匹配保持 UNVERIFIED，不能重签旧 hash。
- 旧 compare 是迁移回归证据，使用独立 seed、简化 descriptor 和 event 计时，不进入统一三轮 wall 主表。
- 单点 probe 的 setup/JIT 秒数、trace/NCU duration 都不是正式未插桩性能。全矩阵三轮、ring、Roofline 和最终封包仍需独立完成和审计。

所有新工具都提供 `--help`；构建/探测命令、完整环境覆盖、stdout/stderr、退出码与 SHA 位于 session 的 `logs/` 和 `manifests/`。
