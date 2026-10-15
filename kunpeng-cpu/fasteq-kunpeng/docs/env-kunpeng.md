# 鲲鹏服务器：MACE 项目运行环境与配置

整理日期：2026-10-14。

本文依据项目现有交接脚本、迁移手册和服务器实际运行记录整理，供其他账户或机器部署参考。脚本依据为 `MACE_perf184_handoff/baseline/` 与 `MACE_perf184_handoff/optimized/scripts/`。未重新连接服务器采集硬件信息；历史验证版本、脚本配置与待确认信息分别标注。

## 1. 服务器与 CPU 布局

| 项目 | 已知配置/运行记录 |
| --- | --- |
| 平台 | 深圳超算 PAC 鲲鹏 CPU 集群 |
| 操作系统架构 | Linux，`aarch64` |
| 登录节点 | 历史使用过 `login02`、`login04`、`login05`、`login06`、`login08` 等 |
| 计算节点 | `cn007xx`、`cn194xx`、`cn198xx` 等，实际由调度系统分配 |
| 单计算节点在线 CPU 编号 | `0-607`，共 608 个逻辑 CPU 编号 |
| 项目使用的计算核布局 | 16 组，每组 37 个计算核，共 592 核 |
| 预留/系统核 | `37,75,113,151,189,227,265,303,341,379,417,455,493,531,569,607`，共 16 个，项目 worker 绑定时排除 |
| NUMA 绑定 | 优化脚本配置 16 个 worker，按上述 16 组核绑定；历史 worker 日志记录 NUMA 内存绑定，例如 worker15 → NUMA15 |
| 可见内存节点范围 | 历史 `Mems_allowed_list` 为 `0-31`；不能据此简单断定系统只有 16 个内存 NUMA 域 |
| GPU | 本项目使用 CPU；已验证环境 `torch.cuda.is_available()` 为 `False` |
| 内存申请示例 | 正式 50 节点测试每节点申请 `mem=256G`；这是作业申请量，不代表机器物理内存容量或队列上限 |
| 尚待实机确认 | CPU 精确型号、插槽数、SMT/物理核对应关系、完整 NUMA 拓扑、物理内存、Linux 发行版/内核、队列资源上限 |

项目所用第 k 组（k=0…15）计算核为 `[38k, 38k+36]`，跳过 `38k+37`。完整绑定字符串：

```text
0-36,38-74,76-112,114-150,152-188,190-226,228-264,266-302,304-340,342-378,380-416,418-454,456-492,494-530,532-568,570-606
```

启动器曾默认只允许使用上述 16 个系统核，表现为 `nproc=16`。即使申请 `cpu=128`，启动器亲和性也未自动扩大。因此申请核数、进程可运行核集合、实际线程数需要分别确认。当前优化脚本先设置 `MACE_CPU_PRE_AFFINITY=0-607`，再由 worker 做计算核和 NUMA 绑定。换机器时必须核对调度授权及拓扑，不能直接照搬 CPU 编号。

## 2. 调度与共享目录

| 项目 | 配置 |
| --- | --- |
| 调度系统 | Donau |
| 使用队列 | `q_pacapp` |
| 提交/查询 | `dsub` / `djob` / `dqueue` |
| 分布式启动器 | `/opt/batch/agent/tools/drun` |
| 项目目录 | `$HOME/mace_cpu_test` |
| 原账户主目录示例 | `/home/share/kongbaizhe` |
| Python 环境 | `$HOME/venvs/mace-cpu` |
| 共享数据 | `$HOME/mace_cpu_test/data_12_20261012/perf_184` |
| 节点本地编译缓存 | `${TMPDIR:-/tmp}/mace_torch_extensions_${CCS_JOB_ID}_rank_${CCS_JOB_STEP_RANK}` |

登录节点用于准备文件和提交任务，正式计算通过队列在计算节点执行。共享目录供各节点读取代码、模型和输入，也保存结果；native MACE JIT 编译缓存按 job/rank 放到节点本地，避免跨节点竞争同一编译锁。

重要调度变量：`CCS_JOB_ID`（作业号）、`CCS_JOB_EXEC_NODES`（节点列表）、`CCS_JOB_STEP_RANK`（作业步 rank）。多副本实验中 `CCS_TASK_INDEX` 曾在同一节点内重复，正式分片优先用 `CCS_JOB_STEP_RANK`。

`dsub` 负责资源申请，`drun` 负责在获分配节点上启动程序。当前 50 节点启动命令为：

```bash
/opt/batch/agent/tools/drun -N 50 -nn 50 -rpn 1 -v bash <worker脚本> run
```

历史 `drun -R ...` 在普通模式报错，提示只可与 `-mpi pmix` 配合；当前普通 Bash/Python 分布式启动不传该参数。最初直接 `mpirun` 曾受主机列表和 SSH 主机密钥影响，现行流程使用 `drun`。

## 3. Python 与软件依赖

以下为原账户历史成功运行或安装记录，不能代替接收方环境检查。

| 软件 | 版本/说明 |
| --- | --- |
| Python | CPython 3.9.25，AArch64 |
| PyTorch | 2.2.2，CPU 运行 |
| NumPy | 1.26.3 |
| SciPy | 1.13.1 |
| ASE | 3.26.0 |
| MACE / mace-torch | 0.3.13 |
| e3nn | 0.4.4 |
| torch-geometric | 2.6.1 |
| matscipy | 历史安装记录 1.1.1；优化版邻居图使用 |
| ninja | 安装包 1.13.2；命令输出 `1.13.2.git.kitware.jobserver-pipe-1` |
| batchASE | 初始基于 v0.7.1，当前优化版以 `kunpeng_cpu_mace_evd` 包内的 `batchase_native` 为准，含本地修改 |
| joblib | baseline 必需，当前版本需采集 |
| psutil | 环境/进程依赖，当前版本需采集 |
| 其他 | h5py、torch-ema、torchmetrics、opt-einsum、python-hostlist 等，以实际环境清单为准 |

原环境建立时使用过以下模块：

```bash
module use /work_ssd/software/modulefile/modules/
module load pytorch/1.9.0-gcc10.3.1
source "$HOME/venvs/mace-cpu/bin/activate"
```

模块名中的 PyTorch 1.9.0 是基础环境名称，项目实际应导入虚拟环境中的 PyTorch 2.2.2。历史 NumPy/SciPy 路径来自该基础 conda 环境，说明原环境依赖基础安装；不能只复制 venv 就认为完成迁移。接收方应同时验证版本和导入路径。

离线 wheel 应匹配 `aarch64`、Python 3.9、`cp39` 以及目标系统支持的 manylinux 标签。交接包不包含完整 wheel 仓库或完整 Python 环境。

## 4. native 编译与动态库

| 项目 | 当前运行脚本配置 |
| --- | --- |
| 优化源码根目录 | `$HOME/mace_cpu_test/kunpeng_cpu_mace_evd`，可由 `KUNPENG_PACKAGE_ROOT` 覆盖 |
| native EVD 扩展 | `batchase_native/third_party/kunpeng_evd_openblas/_kunpeng_native_evd.cpython-39-aarch64-linux-gnu.so` |
| native MACE 源码 | `native_cpu_ops/native_equiv_cpu.cpp` |
| native MACE 加载入口 | `native_cpu_ops/mace_runtime.py`，使用 PyTorch C++ 扩展 JIT 编译 |
| OpenBLAS 路径 | `/work_ssd/software/soft/lib/openblas/0.3.18-hpckit25.1.0.SPC001-gcc12.3.1` |
| 编译器候选目录 | `/work_ssd/software/soft/tool/hpckit/2025.0.0/HPCKit/25.0.0/compiler/gcc` |
| OpenMP 库 | `$HOME/venvs/mace-cpu/lib/python3.9/site-packages/torch.libs/libomp-b8e5bcfb.so` |
| 编译并发 | `MAX_JOBS=2`，仅控制编译，不是计算 worker 数 |

脚本在上述编译器目录存在时设置 `CC`、`CXX`；OpenBLAS 目录存在时设置 `NATIVE_CPU_BLAS_INCLUDE`、`NATIVE_CPU_BLAS_LIB`、`NATIVE_CPU_BLAS_NAME=openblas`。这些是集群安装路径，目录名不等于已核实的编译器运行时版本。

native EVD 预编译文件依赖 CPython 3.9/AArch64 ABI 和相关动态库。源码包的 `.llvm_openmp_link/libomp.so` 可能指向原作者账户，迁移后需要重建为本账户有效路径。Linux tar 解压可保留符号链接，优先在服务器解压原始包。

检查示例：

```bash
PROJECT="$HOME/mace_cpu_test"
EVD_DIR="$PROJECT/kunpeng_cpu_mace_evd/batchase_native/third_party/kunpeng_evd_openblas"
TORCH_LIBS="$HOME/venvs/mace-cpu/lib/python3.9/site-packages/torch.libs"
readlink -f "$EVD_DIR/.llvm_openmp_link/libomp.so"
LD_LIBRARY_PATH="$EVD_DIR/.llvm_openmp_link:$TORCH_LIBS:${LD_LIBRARY_PATH:-}" \
  ldd "$EVD_DIR/_kunpeng_native_evd.cpython-39-aarch64-linux-gnu.so"
"$HOME/venvs/mace-cpu/bin/ninja" --version
```

## 5. 当前两套正式测试配置

| 参数 | baseline | 优化版 |
| --- | --- | --- |
| 计算实现 | ASE + MACECalculator，单结构优化 | batchASE + native CPU MACE + native EVD |
| 节点数 | 50 | 50 |
| 每节点进程/worker | 16 | 16 |
| 每节点申请核数 | 128 | 592 |
| 每节点申请内存 | 256G | 256G |
| worker CPU 配置 | 每进程 8 个 PyTorch 线程 | 每 worker 绑定一组 37 核；PyTorch 线程单独设为 1 |
| batch size | 单结构处理 | 8，正式分片初始 batch=4 |
| 输入 | perf_184，共 3221 CIF | 同左 |
| 节点间分配 | 静态等数量分片 | 同左 |
| 分片数量 | 21 节点各 65 个，29 节点各 64 个 | 同左 |
| 模型 | MACE-OFF23_small.model | 同名模型，交接时已确认文件哈希一致 |
| 数值精度 | float64 | float64 |
| 优化阶段 | 仅第一阶段，UnitCellFilter + BFGS | 同左 |
| 优化步数 | 每结构最多 500 步，收敛可提前结束 | 同左 |
| fmax | 0.01 | 0.01 |
| 第二阶段 | 禁用 | `MACE_SKIP_SECOND_STAGE=true` |

baseline 的线程环境为 `OMP_NUM_THREADS=8`、`MKL_NUM_THREADS=8`、`OPENBLAS_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`，并清除 `PYTHONPATH`。CPU 池为 `0-36,38-74,76-112,114-130`，共 128 核。

优化版其他关键配置：

```text
BATCHASE_NATIVE_EVD=1
NATIVE_CPU_MACE=1
BATCHASE_NATIVE_CPU=1
BATCHASE_CPU_NEIGHBOR=matscipy
BATCHASE_NUMA_MEMBIND=1
BATCHASE_MP_CONTEXT=fork
BATCHASE_TORCH_THREADS=1
OMP_NUM_THREADS=37
OPENBLAS_NUM_THREADS=37
MKL_NUM_THREADS=37
NATIVE_EVD_THREADS=8
NATIVE_EVD_THREADS_PER_MATRIX=1
NATIVE_EVD_INNER_PARALLEL=1
NATIVE_EVD_STAGE2_MODE=bench2l
NATIVE_EVD_LEAF_SIZE=32
NATIVE_EVD_CPU_STAGE3=precomputed_wy
NATIVE_CPU_MACE_RADIAL_MANUAL=1
BATCHASE_SKIP_GC=1
BATCHASE_DYNAMIC_SCHEDULING=1
BATCHASE_DYNAMIC_INITIAL_BATCH=4
BATCHASE_DYNAMIC_SLOW_WORKER_INDEX=-1
BATCHASE_DYNAMIC_SLOW_WORKER_FRACTION=0
OMP_WAIT_POLICY=PASSIVE
KMP_BLOCKTIME=0
GOMP_SPINCOUNT=0
OMP_PROC_BIND=close
OMP_PLACES=cores
```

优化版传入 `--num_threads 37 --bfgs_cpu_thread 1 --use_fasteq false --cueq false`。当前配置未启用 FastEq 或 cuequivariance。37 个绑定核表示允许执行范围，不代表每个计算阶段必定同时使用 37 核；PyTorch、BLAS、native EVD 的线程配置分别控制不同计算路径。

正式 worker 覆盖 `MACE_MAX_STEPS=500`；直接调用底层 runner 的 `full` 模式默认步数仍为 5，所以正式任务需走 launch → worker → runner。smoke 模式使用 1 worker、batch2、2 个结构、2 步，不能作为正式性能数据。

提交示例（需先部署交接包并通过 smoke）：

```bash
dsub -q q_pacapp -n perf184-baseline-500 \
  -N 50 -nn 50 -rpn 1 -R "cpu=128;mem=256G" -wo \
  bash "$HOME/mace_cpu_test/worksteal_baseline/mn_job.sh"

dsub -q q_pacapp -n perf184-native-500 \
  -N 50 -nn 50 -rpn 1 -R "cpu=592;mem=256G" -wo \
  bash "$HOME/mace_cpu_test/launch_50node_perf184_stage1_500.sh" run
```

两套正式配置的资源量不同；耗时对比应同时报告核数，不能将其直接认作同资源下的纯算法加速。

## 6. 计时与结果口径

- 节点墙钟时间：该节点启动到完成的实际经过时间；全作业耗时另受启动、编译及最慢节点影响。
- worker/NUMA 计时：对应进程的计算范围；当前一个 worker 绑定一组核，不等于硬件 NUMA 域总忙碌时间。
- Dashboard 的 `Total Worker Time`：多个 worker 的分项时间累计，不能当作作业实际等待时间。
- 各 worker `Tot Time`：该 worker 计时区间的经过时间；进程间可重叠，求和不是节点墙钟。
- `max_steps`：达到步数上限，未必是程序异常；调度状态 `SUCCEEDED` 不代表全部结构收敛。
- 历史统计曾排除 W04 后对其余 15 个 worker 求平均；这是人为筛选口径，报告时需注明，不能据此替代完整节点耗时。

baseline 输出位于 `worksteal_baseline_results/job_<JOB_ID>/`；优化版位于 `kunpeng_evd_results/perf184_50node_<JOB_ID>/`，后者包含各 rank 的 `results_scheduler.csv`、运行日志与 `metrics/worker_*.json`。

## 7. 在新账户/机器采集核对信息

硬件命令应放在申请到的计算节点任务里执行，登录节点信息不代表计算节点。

```bash
hostname
uname -a
cat /etc/os-release
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
numactl --hardware
free -h
taskset -pc $$
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
```

`numactl` 若未安装，可查看 `/sys/devices/system/node/node*/cpulist`。内存配额和可申请上限需以队列配置/管理员说明为准，`free -h` 仅显示当前可见内存。

软件环境检查：

```bash
source "$HOME/venvs/mace-cpu/bin/activate"
python -V
python -m pip freeze
python -m pip check
python - <<'PY'
import importlib
import platform
import sys
print('python:', sys.executable)
print('machine:', platform.machine())
for name in ['torch', 'numpy', 'scipy', 'ase', 'mace', 'e3nn',
             'torch_geometric', 'matscipy', 'joblib', 'psutil']:
    module = importlib.import_module(name)
    print(name, getattr(module, '__version__', 'unknown'), module.__file__)
PY
```

接收方先确认依赖导入、动态库、计算核绑定与小规模真实计算，再扩大节点数。完整部署步骤见 `MACE_perf184_handoff/README.md`。
