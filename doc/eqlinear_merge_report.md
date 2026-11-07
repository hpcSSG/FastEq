# eqlinear 合入与最终验收

## 结论

按用户2026-11-07确认的决定，**接受±3%性能容差及已知稳定性风险**，以干净文件集合合入
`xjw/dev_eqlinear`。代码合入点为`a9feaa1309be7a499b35bde6138050e08cc4c58f`，从`229a1ce`
fast-forward而来；没有把prep的64个实验提交带入目标历史。push实际结果在本报告末尾记录。

生产代码与P14/P27最终实现保持字节一致，**代码未改，性能沿用P29**，没有伪造新的稳定扫描。
P29严格spread/跨轮门禁仍失败；本次是用户接受容差及风险后的合入，不是宣称旧严格门禁全部通过。

## 代码范围

带入：`fasteq/ops/eqlinear.py`、`fasteq/cuda/src/eqlinear/**`的必要变更、`.gitignore`、
`doc/eqlinear.md`、本报告，以及以下测试：

- `test/test_eqlinear.py`、`test/test_eqlinear_real_descriptor.py`
- `test/test_eqlinear_descriptor_mutation.py`、`test/test_eqlinear_lifetime.py`
- `test/test_eqlinear_weight_layout.py`、`test/test_eqlinear_benchmark_io.py`
- `test/bench_fasteq_compare.py`、`test/smoke_eqlinear_other_ops.py`

整个89文件生产树及8个测试/基准文件与prep逐字核对一致；实际净差异31文件。
补全目标原来缺失的、现已跟踪的`fasteq/cuda/src/eqlinear/CMakeLists.txt`，gitignore增加
对应例外及`/tmp_dev/`。`setup.py`、CUDA顶层CMake/loader与prep相比本就未改，不额外带入。
没有其他operator、`3rdparty`、`origin/main`改动；`git diff --check`通过。

六个pytest文件不依赖`test/eqlinear_acceptance`，所以该目录及profile/生成工具均未带入。
未带入P1–P30逐轮报告、evolution、历史FINAL、GOAL、MERGE_READY、证据目录、raw CSV/trace、
临时脚本、DSO或缓存。这样保持目标分支可维护，同时保留独立开发线和完整过程证据。

## 正确性与构建

| 阶段/项目 | 实际结果 |
|---|---|
| staging干净构建/import | PASS，editable构建7m15s |
| staging六文件pytest | 144 passed，0 failed/error/skip，7.84s |
| 目标分支重新构建/import | PASS，editable构建7m01s |
| 目标分支六文件pytest | 144 passed，0 failed/error/skip，7.95s |
| 目标分支compare | 75/75 supported，精度全部PASS |
| 其他operator smoke | fused_commutator / fused_bmm_sym / fast_fctp forward/backward PASS |
| 目标fresh DSO额外同点精度 | 600/600 PASS，TF32冻结P14逐点1.10倍限制PASS |

| 精度 | 75点compare最大rel_fro | P29同点新库最大rel_fro | P29原记录最大rel_fro | 同点结论 |
|---|---:|---:|---:|---|
| TF32 | 7.2181738189e-06 | 7.2105521211e-06 | 7.2105521211e-06 | 一致，未退化 |
| FP32 | 2.7412662366e-07 | 2.7426368313e-07 | 2.7426368313e-07 | 一致，未退化 |
| FP64 | 1.1363721902e-15 | 1.1349518313e-15 | 1.1349518313e-15 | 一致，未退化 |

compare脚本与P29采用不同seed；冻结P14/P29的逐点约束和“最大误差不劣于P29”由额外同点600项验证，
不将不同输入的两个全局最大值当作逐点对照。FP32门禁逐点相对torch检查，未只使用全局上限。


验收使用唯一`/app/work/fasteq_venv`，editable指向`/app/work/fasteq-pub`。
先在干净staging checkout构建、import及测试，fast-forward后又在目标分支重新执行editable
构建、外部cwd import、全部测试、compare和其他算子smoke。临时构建和日志全部位于tmp_dev。
六个pytest覆盖生命周期排队销毁、多stream独立handle、非连续权重、descriptor替换/结构变化。

额外使用保留prep中的独立验收工具，在目标fresh DSO上执行600项同点精度检查：
两版×25点×3精度×public/run forward/grad_x；不是新的性能扫描。
其参考数学和seed与P29一致，验证TF32≤2e-5且≤冻结P14同点同路径1.10倍，
FP32≤torch误差1.05倍，FP64≤1e-12，finite、不同out指针和同步后bitwise repeat。
这份工具只用于外部审计，目标pytest不依赖它，因此不复制到目标分支。

## 性能：实测与容差判定


用户接受改善率差3个百分点 **或** 相对目标延迟/开销3%的容差，并接受下述已知稳定性风险。raw未改；adjusted仅指判定。

小/全分别为10点/25点 public forward 改善率，目标20%/5%。

| run/轮 | TF32 小/全% | FP32 小/全% | FP64 小/全% | 严格收益目标 | OR容差后 |
| --- | --- | --- | --- | --- | --- |
| initial R0 | 29.75/18.18 | 19.12/11.08 | 19.85/10.99 | 未全过 | 全部通过 |
| initial R1 | 29.40/17.95 | 18.89/10.69 | 20.56/11.22 | 未全过 | 全部通过 |
| initial R2 | 29.36/18.00 | 19.11/10.88 | 20.26/11.29 | 未全过 | 全部通过 |
| replica R0 | 29.88/18.00 | 18.76/10.67 | 20.03/11.07 | 未全过 | 全部通过 |
| replica R1 | 29.92/18.09 | 18.68/10.78 | 20.11/11.13 | 未全过 | 全部通过 |
| replica R2 | 29.64/18.44 | 18.80/10.61 | 19.75/11.01 | 未全过 | 全部通过 |

全点18/18严格达到5%；小点10/18严格达到20%，其余8项均在两条容差内。

开销定义：`O_B=P_B−R_B`、`O_C=P_C−R_C`，下降率 `1−O_C/O_B`；分母是P0开销，不是P0 public。
严格半开销要求 `O_C≤0.5O_B`；3pp容差要求 `O_C≤0.53O_B`；3%相对严格目标要求 `O_C≤0.515O_B`。

| run/轮 | TF32开销下降% | 严格50% | 3pp后 | 3%相对目标后 |
| --- | --- | --- | --- | --- |
| initial R0 | 49.6018 | 未达 | 通过 | 通过 |
| initial R1 | 47.7656 | 未达 | 通过 | 未达 |
| initial R2 | 49.6063 | 未达 | 通过 | 通过 |
| replica R0 | 52.3535 | 通过 | 通过 | 通过 |
| replica R1 | 49.3487 | 未达 | 通过 | 通过 |
| replica R2 | 47.0583 | 未达 | 通过 | 未达 |

FP32/FP64开销12/12轮严格通过。TF32所有六轮在3pp内，但initial R1、replica R2不在3%相对目标内，依OR规则接受。
replica R2开销下降47.058315%，距47%容差边界仅0.058315个百分点（0.023783µs）。public绝对值减半替代项未通过；接受来自开销分支。

收益与开销共54项：严格41/54，3pp后54/54，单独3%相对目标52/54，OR后54/54。

稳定性仍未严格通过：initial/replica有86/79个任一轮spread>10%的聚合条件，跨轮>5%有14/29项；最大spread127.01%/154.82%。
正式wall key CSV的每版本/轮spread中位2.36–3.75%，但不能以此掩盖尾部超限。initial候选独有反转记录2项、replica为0；两组均无逐点/单轮>2%回归。
用户已接受上述风险；不能写成“已稳定通过”。

原始 CUDA为1-pass TF32，新TF32为3-pass tf32x3；下表非同精度加速比。

| 范围 | 新入口 | 原始raw ms | 新3-pass ms | 新实现更慢% |
| --- | --- | --- | --- | --- |
| small_10 | public_forward | 0.550939 | 0.966971 | 75.51 |
| small_10 | prepared_3d | 0.550939 | 0.603512 | 9.54 |
| all_25 | public_forward | 2.925141 | 3.938306 | 34.64 |
| all_25 | prepared_3d | 2.925141 | 3.024535 | 3.40 |

按全部forward public/run条件重算的spread中位数为2.36–3.75%；旧goal中“14–20%”概述未给出
同一统计口径，这里以定义明确的正式wall组数据为准。尾部超限与跨轮失败完整保留，不将较低
中位数或±3%收益容差用来宣称稳定通过。统计来源和复核脚本在外部performance_review目录。

## 已知caveat

- P0历史queued pinned-host pointer顺序问题：原诊断P0仅2/8通过，6项首输出保留NaN；
  候选8/8通过。候选的输出指针更新和生命周期保护修复此路径，P0本体没有被改写。
  同步精度/性能不能推导P0任意queued模式正确；本次依据用户接受该历史风险的决定合入。
- `DeviceFence`在plan销毁/缓存驱逐前执行device-wide synchronize，会等待该设备其他stream，
  销毁窗口可能变长，不能把host快路径收益描述成无同步代价。异常清理无法建立安全fence时
  记录错误并保留资源至进程退出；析构不抛异常。
- 同handle跨stream无同步并发仍不支持；独立stream使用独立handle。创建/运行选择plan设备并
  恢复调用者设备，输入输出必须位于plan设备。
- flat快路径要求x/w都连续；非连续w走3D reshape/contiguous，每call可能复制，
  历史同值实验额外约21–23µs。没有把fallback描述成零成本。
- plan cache以真实weight identity/version/shape/stride/dtype、问题形状和stream等区分；
  修改权重会失效。`eqlinear_clear_plan_cache()`清理，`eqlinear_set_plan_cache(0)`禁用；
  清理可能触发上述device fence。descriptor只对可跟踪canonical结构缓存，未知类型或
  结构变化回退完整解析，系数每次重读，不依赖运行时FASTEQ_EQLINEAR环境开关。
- 原始1-pass TF32的严格FP32精度门禁失败且存在重复输出/旧launch故障记录；raw仍被计时。
  新public/prepared的实际差距与精度档同时呈现，不当作同精度加速比。

## 合入、构建与复现

代码提交`a9feaa1309be7a499b35bde6138050e08cc4c58f`由`xjw/eqlinear-merge-clean`挑选文件后创建，
在staging验收通过后执行：

```bash
git switch xjw/dev_eqlinear
git merge --ff-only xjw/eqlinear-merge-clean
```

目标分支重建，始终复用现有环境，不新建venv；保留`/app/work/luna_venv`：

```bash
cd /app/work/fasteq-pub
export TMPDIR=/app/work/fasteq-pub/tmp_dev
export CUDA_CACHE_PATH=/app/work/fasteq-pub/tmp_dev/cuda_cache
FASTEQ_BACKEND=cuda CMAKE_BUILD_PARALLEL_LEVEL=2 \
  timeout --signal=INT --kill-after=10s 600s \
  uv pip install --python /app/work/fasteq_venv/bin/python \
  --no-build-isolation --no-deps --reinstall -e .
```

先两次GPU采样选择空闲卡。本次测试使用物理GPU7
`GPU-f070b1c1-83b3-cc1e-3962-136b69446838`，逻辑cuda:0：

```bash
CUDA_VISIBLE_DEVICES=7 timeout --signal=INT --kill-after=5s 120s \
  /app/work/fasteq_venv/bin/python -m pytest -q \
  test/test_eqlinear.py test/test_eqlinear_real_descriptor.py \
  test/test_eqlinear_descriptor_mutation.py test/test_eqlinear_lifetime.py \
  test/test_eqlinear_weight_layout.py test/test_eqlinear_benchmark_io.py
CUDA_VISIBLE_DEVICES=7 timeout --signal=INT --kill-after=5s 180s \
  /app/work/fasteq_venv/bin/python test/bench_fasteq_compare.py --mode after \
  --out /app/work/fasteq-pub/tmp_dev/merge_final_compare.csv
CUDA_VISIBLE_DEVICES=7 timeout --signal=INT --kill-after=5s 60s \
  /app/work/fasteq_venv/bin/python test/smoke_eqlinear_other_ops.py
```

同点额外精度使用保留prep的`test/eqlinear_acceptance/acceptance.py --mode accuracy`，
`--root /app/work/fasteq-pub/tmp_dev/eqlinear_fix --prep-root /app/work/fasteq-pub`，
`--frozen-errors frozen_p14_accuracy/frozen_accuracy.csv --out-dir merge_target_accuracy_20261107`，
timeout120s。实际日志、原始compare、精度receipt、源码/DSO manifest和最终远端核验在
`/app/work/fasteq-pub/tmp_dev/eqlinear_merge`及上述fresh accuracy目录。

正常push命令为`git push origin xjw/dev_eqlinear`；此版已完成本地合入与验收，
远端push实绩在执行后补记，不把计划视为成功。

## 未合入的历史与过程数据

- 本地`xjw/eqlinear-prep-opt`保留`229a1ce..2635512`的64个提交，HEAD未移动，未rebase/squash，
  未删除分支或其`/app/work/fasteq-eqlinear-prep` worktree。用户本次更新的goal文件也未改写。
- P1–P30报告、`doc/EVOLUTION_EQLINEAR_PREP.md`、现存`doc/eqlinear_prep_evidence/`、
  `FINAL_EQLINEAR_PREP_FIX_REPORT.md`等仍在该worktree/分支，原始报告链接可追溯历史Git对象。
- 全部原始日志/CSV/trace保留在`/app/work/fasteq-pub/tmp_dev/eqlinear_fix`；旧P1–P22的287文件
  已有外置`archive/p1_p22_evidence.tar.gz`，SHA256
  `d727be3bcea748750d5e48522a13bfad3f1247c22898da47379c5e68dab32e7e`。
  此前P30已归档的路径仍能从prep历史和tar恢复；本次没有再移动或删除任何过程数据。
- 合入范围/性能复核及新日志单独追加到`tmp_dev/eqlinear_merge`，不覆盖旧证据。
  省略这些材料只影响目标分支的文件集合，不删除开发线证据；59个现存报告/关键文件的
  merge前后SHA核对及prep HEAD检查保存在preservation manifest。
