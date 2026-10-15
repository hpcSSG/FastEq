# 鲲鹏 CPU 分支部署环境

更新日期：2026-10-15。历史软件配置依据 [env-kunpeng.md](env-kunpeng.md)；硬件与编译器信息已由用户提供的 [cn19793 实机报告](../../test/results/environment/kunpeng_env_cn19793_1886300.txt) 补充。报告采集时间为 2026-10-15 10:45:53，作业号 1886300。以下实测结论限定于该节点和采集时刻，本次未远程连接服务器。

## 实机确认：cn19793

| 项目 | 采集结果 |
|---|---|
| 架构 / 厂商 | aarch64 / HiSilicon；CPU implementer `0x48`，part `0xd22` |
| CPU 系列 | 鲲鹏 920 系列（用户于 2026-10-15 确认；具体子型号未提供）|
| 操作系统 | 麒麟系统（用户于 2026-10-15 确认；具体版本/版本号未提供）|
| CPU 拓扑 | 2 个插槽，每插槽 304 核，每核 1 线程；共 608 个在线逻辑 CPU，编号 0–607 |
| 插槽映射 | socket0：CPU 0–303；socket1：CPU 304–607 |
| 频率范围 | lscpu 报告最小 1200 MHz、最大 2000 MHz，boost enabled；不是实时运行频率 |
| NUMA | 32 个节点，编号 0–31 |
| 有 CPU 的 NUMA 节点 | node0–15，每节点 38 个 CPU；node k 的 CPU 范围为 `[38k,38k+37]` |
| 无 CPU 的 NUMA 节点 | node16–31，cpulist 为空，每节点 MemTotal=4194304 kB，即 4 GiB；报告未说明其内存介质类型 |
| 内存 | 32 个节点 MemTotal 合计 592446084 kB，约 565.001 GiB；这是内核报告的 NUMA 内存总量，不是 DIMM 标称安装量或作业额度 |
| Linux 内核 | `5.10.0-285.5.v2403.ky10p.aarch64` |
| 默认 gcc | `/usr/bin/gcc`，GCC 10.3.1 |
| 指定路径的 gcc / g++ | HPCKit 路径下均为 `gcc for openEuler 3.0.1`，版本 12.3.1 |
| 采集时 CC / CXX | 均为空；不能仅凭目录存在就认定某次 JIT 已使用该编译器 |

`lscpu` 和逐 CPU 映射共同确认了上述插槽、核心和 NUMA 关系。CPU 系列“鲲鹏 920”和操作系统“麒麟”来自用户确认，不是从报告中的 `Model: 0`、part 编号或内核后缀推断。具体 CPU 子型号和麒麟版本号尚未提供。`gcc for openEuler` 是编译器版本字符串，不表示服务器操作系统是 openEuler。

采集进程的 CPU/内存许可范围为：

```text
Cpus_allowed_list: 37,75,113,151,189,227,265,303,341,379,417,455,493,531,569,607
Mems_allowed_list: 0-31
```

因此采集进程当时只允许使用 16 个 CPU 编号，恰好是历史 worker 布局排除的那组编号。该进程快照不能代表所有 worker 的亲和性，也不能说明调度系统实际授予的完整资源。运行 37 线程测试之前，应在实际 worker 内检查允许 CPU 集合与资源申请是否一致；仅设置 `OMP_NUM_THREADS=37` 不会扩大 CPU 许可范围。本仓库环境脚本不自动修改 affinity。

## 软件基线

| 项目 | 历史记录 |
|---|---|
| 平台 | 深圳超算 PAC 鲲鹏 920 系列，麒麟系统，aarch64（系列及系统名称由用户确认）|
| Python | CPython 3.9.25 |
| PyTorch / e3nn | 2.2.2（CPU）/ 0.4.4 |
| MACE | mace-torch 0.3.13，模型 MACE-OFF23_small.model |
| NumPy / SciPy / ASE | 1.26.3 / 1.13.1 / 3.26.0 |
| torch-geometric / matscipy | 2.6.1 / 1.1.1（历史安装记录）|
| ninja | 包版本 1.13.2；命令历史输出带 git.kitware.jobserver-pipe-1 后缀 |
| 数值精度 | 正式端到端作业 float64；现有微基准脚本主要为 float32 |
| Python 环境 | `$HOME/venvs/mace-cpu` |
| OpenBLAS | `/work_ssd/software/soft/lib/openblas/0.3.18-hpckit25.1.0.SPC001-gcc12.3.1` |
| 编译器目录 | `/work_ssd/software/soft/tool/hpckit/2025.0.0/HPCKit/25.0.0/compiler/gcc`；cn19793 实测 GCC/G++ 12.3.1 |

依赖最低声明已调整到 Python 3.9、torch 2.2.2、e3nn 0.4.4，以允许在原环境进行复验；这不是已经证明所有版本组合兼容。`test/constraints-kunpeng.txt` 固定已知版本，未记录的依赖仍由安装工具解析，不能称为完整锁文件。

## 在现有环境中部署

先激活原虚拟环境，确认 `python -V`、包版本和导入路径。历史上使用过 `pytorch/1.9.0-gcc10.3.1` 基础 module；模块名不代表虚拟环境实际导入的 torch 版本。原 NumPy/SciPy 依赖基础 conda 安装，不能仅复制 venv。

```bash
source "$HOME/venvs/mace-cpu/bin/activate"
cd /actual/path/FastEq-kunpeng-cpu
python test/scripts/collect_environment.py > environment-before.json
python -m pip check
python -m pip install -r requirements.txt -c test/constraints-kunpeng.txt
python -m pip install -e '.[test,mace]' --no-build-isolation -c test/constraints-kunpeng.txt
```

先保证环境具备 setuptools>=68 和 wheel。离线部署需准备匹配 Python 3.9 / aarch64 的依赖；C++ JIT 扩展在计算节点本地重新编译，不复制旧账户的二进制。

在调度分配的计算节点内执行：

```bash
source test/scripts/kunpeng_env.sh
python -c 'from fasteq.cpu import load_extension; print(load_extension().__file__)'
python test/scripts/collect_environment.py > environment-after.json
python -m pytest -q
python test/benchmark_native_vs_e3nn.py --threads 1 37 --batches 19 172 --fused-sc
```

`kunpeng_env.sh` 仅在目录存在时采用历史编译器/BLAS 路径，保留用户显式设置。没有设置 BLAS 路径时，内核可使用内部计算路径；检查采集输出确认实际启用的配置。脚本按用户、job、rank 隔离 JIT 缓存，不设置全节点 CPU affinity，不自动提交作业。

## 核布局、调度和线程

- 调度：Donau，历史队列 `q_pacapp`，提交/查询使用 `dsub`、`djob`、`dqueue`；启动器 `/opt/batch/agent/tools/drun`。
- 实测在线逻辑 CPU 编号 `0-607`。项目历史使用 16 组、每组 37 个计算核，共 592 个编号；第 k 组为 `[38k,38k+36]`，排除 `38k+37`。它对应 node0–15 各 38 核中使用 37 核的配置，不是本次采集进程的实际 CPU 许可范围。
- NUMA 总数已确认是 32，其中 16 个节点有 CPU，另 16 个无 CPU 但有内存；不能把 16 个 worker 表述为系统只有 16 个 NUMA 节点。
- `mem=256G` 是历史作业申请量。内核可见 NUMA 内存合计约 565.001 GiB；物理安装规格、鲲鹏 920 的具体子型号、麒麟具体版本、队列资源上限仍需补充。
- 编译 `MAX_JOBS=2`、计算 OpenMP 线程、BLAS 线程、PyTorch 线程及 EVD 线程互相独立。

| 配置 | 本仓库单算子测试默认 | 原交接资料中的优化版端到端 |
|---|---|---|
| OpenMP | 37（可覆盖）| 37 |
| OpenBLAS | 1（可覆盖）| 37 |
| PyTorch | 微基准按 `--threads` 设置；回归测试设为 1 | `BATCHASE_TORCH_THREADS=1` |
| worker | 不由环境模板创建 | 每节点 16 个 |
| EVD | 未包含 | `NATIVE_EVD_THREADS=8`，每矩阵线程配置 1 |

不能把两套设置视为同一实验。环境模板保留单算子 BLAS=1 的默认策略，端到端历史 BLAS=37 作为独立配置记录，不静默替换。

## 验证边界

原 50 节点实验使用 `--use_fasteq false --cueq false`，实际走 native MACE + batchASE + native EVD。它能作为原始实现的运行背景，不能作为本分支 `fasteq` 包或 cuEquivariance 接口已验证的证据。

原 baseline 每节点申请 128 核，优化版申请 592 核；二者耗时差不能直接称为同资源纯算子加速。模型同名且历史交接确认哈希一致，但本文未提供具体模型哈希和新分支误差报告。仍需提交本分支安装/JIT/pytest 输出，以及能量、力、应力对齐结果。

EVD 的 cp39 `.so`、`torch.libs/libomp` 链接及 batchASE 启动脚本属于原端到端包，不加入本仓库算子依赖，也不将其 OpenMP 动态库链接方法直接套用到本 JIT 扩展。
