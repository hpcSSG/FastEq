# 上传前文件与代码检查

日期：2026-10-15。本轮检查当前本地 `kunpeng-cpu` 初始分支；没有提交或远程地址。

结论：源码目录、接口文件、依赖声明、安装配置、说明文档和原始环境报告已齐全；打包成功不表示算子已可正式发布。以下代码问题与目标服务器验证仍未完成。当前可作为明确标注限制的开发源码，不能据此宣称完整兼容或性能达标。

## 已完成检查

- 审核报告生成前遍历读取 47 个文件（排除 Git 元数据、缓存和构建产物），24 个 Python 文件通过 Python 3.9 语法模式解析。
- `VERSION`、顶层包版本和 pyproject 版本一致，均为 `0.2.0+kunpeng.1`。
- 检查所有声明包的目录映射及 Markdown 相对链接，没有缺失目标。
- C++ 注册 25 个 pybind 方法；Python 源码静态识别的 24 个原生调用都有对应导出。该检查不验证参数类型、运行时签名和算法正确性。
- C++ 内核 SHA-256 与原始实现一致。
- 重新构建 sdist 和 wheel，核对 CPU 源码、MACE 集成、约束文件、环境脚本及实机报告进入对应包；wheel 顶层导入与 CPU 加载入口导入通过。
- 新增 `.gitattributes`，让 shell、Python、C++ 等文本在 Git 检出时使用 LF，避免 Windows 换行影响 Linux 脚本。
- 本轮未修改 C++ 算法、Python 算子行为或线程控制逻辑；下列缺陷仍待修复。

## 优先处理的代码问题

### 1. CWTP 某些无路径输出段可能未初始化

位置：[metadata.py](../ops/metadata.py) 的 `make_tpu_metadata`，以及 [native_equiv_cpu.cpp](../cpu/src/native_equiv_cpu.cpp) 的 `tpu_forward`。

`disjoint_output` 目前只检查输出指令编号是否重复，没有检查是否覆盖全部输出段。C++ 在它为真时使用 `torch::empty`；当走 separable 分支时，仅对被指令覆盖的输出段赋值。

因此“输出段编号不重复，但存在没有指令的输出段”时，未覆盖段可能返回未初始化数据。非 separable 分支会写入全部累加缓冲，不属于同一触发条件。应补充完整覆盖条件或统一零初始化，并增加含无路径输出段的前向/梯度回归。

### 2. 公开接口缺少完整的输入边界防护

位置：[native_equiv_cpu.cpp](../cpu/src/native_equiv_cpu.cpp) 的 `tpu_forward`、`fctp_onehot_packed_forward`、`edge_geometry_forward`，及对应 Python 包装层。

- `tpu_forward` 在非共享权重时计算 `weight.numel() / batch`，batch=0 没有提前处理；空边图经 `fast_mptp` 可到达该路径。
- packed FCTP 的 `element` 直接计算权重指针偏移，未校验元素编号是否在权重表范围内。
- edge geometry 直接用 sender/receiver 访问 positions，没有节点索引范围检查。
- 元数据段偏移、部分形状/设备条件也需要系统核对，不能仅依赖文档要求调用者提供正确数据。

这些触发条件可能导致原生进程异常或错误结果，应在进入并行循环前检查。需要增加空输入、负索引、越界索引、错误形状/设备及非连续输入测试。本轮没有在当前机器执行这些危险输入来复现。

### 3. 基准线程标签尚未验证原生线程配置

位置：[benchmark_native_vs_e3nn.py](../../test/benchmark_native_vs_e3nn.py) 与 C++ `native_parallel_for`。

基准循环仅调用 `torch.set_num_threads(threads)`；原生并行读取 `omp_get_max_threads()`，两者可能来自不同 OpenMP 运行时。现有日志只标记传入的线程数，未打印/检查扩展的 `native_thread_info()`。

因此在是否共用运行时尚未确认时，不能保证 `--threads 1 37` 同时切换了原生线程。建议不同线程配置使用独立进程，在导入 torch 前设置 OpenMP 环境，并记录 PyTorch 线程、原生 OpenMP 上限、CPU affinity 和 BLAS 配置。实际参与线程数还受任务大小限制，不等同于线程上限。

## 需要补充的测试与集成证据

- 新分支在 Python 3.9.25 / torch 2.2.2 / e3nn 0.4.4 的实际安装、JIT 编译和 pytest 输出。本地语法模式检查不是这些版本的运行验证。
- 现有测试覆盖基本算子、float32/float64 和一阶输入梯度，但主要是小规模正常输入。仍缺上述边界条件、共享权重 uvu、一般 FCTP 的非标量第二输入、更多 irreps 组合、并行重复执行及一致性检查。
- 测试中的融合 STC/径向 MLP 直接构造 metadata，未覆盖从真实 MACE 模块生成这些 metadata 的完整路径。
- MACE 集成缺可复现的最小模型/结构输入、模型哈希及原版/新包能量、力、应力对齐结果。集成中的回退路径应记录哪些算子真正走了 native，不能仅看程序运行成功。
- 当前微基准主要为 float32，且打印误差而不因误差超限失败；不能替代 pytest，也不能代表正式 float64 模型性能。
- 完整参数梯度、高阶梯度、cuEquivariance descriptor 接入、真正融合 MPTP、任意阶融合 STC 尚未实现；首个推理版本可明确限定范围，无需把所有功能都列为上传前必须完成。

## 发布与复现信息

- 尚缺远程仓库 URL、上游基准提交/目标分支、Git 作者姓名和邮箱，当前没有提交历史。
- 新增代码的作者/单位与许可仍需确认。保留原 FastEq 的 MIT 文件不等于新增代码已获同样许可。
- 历史版本 constraints 已有，但不是完整锁文件；需要当前服务器 `pip freeze`、`pip check`、实际导入路径和编译/动态库记录。
- 环境报告已足以描述已知平台。CPU 子型号、麒麟版本等可进一步补齐，不应与算子运行测试混为一项。
- 默认 JIT 编译依赖现场编译器；当前 wheel 包含源码，并非已编译的 aarch64 wheel。若发布预编译 wheel，还需要独立的 ABI/架构与构建流程。

上传一个开发仓库所需的最小资料是远程地址、作者身份和许可选择；若目标是可合并、可复现的正式 CPU 支持，应先修复上述触发路径并取得目标服务器测试结果。
