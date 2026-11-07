# eqlinear 合入与验收报告

当前为staging验收版本：仅带入P14/P27生产代码、六个必要pytest文件、compare/smoke和操作文档。
用户已接受±3%容差及P29已知稳定性风险；严格spread失败不重新作为阻止合入的条件。
生产源码与prep保持字节一致；性能沿用P29，不伪造新扫描。

目标分支：xjw/dev_eqlinear；基线229a1ce。staging构建/验收及后续目标分支构建/验收、
正常push完成后，本报告将补充实际commit、结果和完整raw/adjusted表。
当前尚未合入或push，不能把计划当成已完成结果。

历史开发分支xjw/eqlinear-prep-opt保留2635512及其64个commit、全部P1–P30报告、
doc/eqlinear_prep_evidence和FINAL_EQLINEAR_PREP_FIX_REPORT.md。
外部原始日志/CSV/trace/归档位于/app/work/fasteq-pub/tmp_dev/eqlinear_fix，未删除或改写。
这些过程材料和GOAL文件不复制到目标分支，因为目标仅保留生产代码、必要测试与最终文档。
