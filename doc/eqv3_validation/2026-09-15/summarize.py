"""Aggregate existing measurements only; never launches GPU work."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
REPO=Path('/public-data/zhouxibo/zxb/FastEq-fork-layernorm')
ORDER=('softmax','alpha','gate','norm','separable','dropout')
LABELS=dict(softmax='GraphSoftmax',alpha='AttentionAlpha',gate='e3nn Gate',
           norm='LayerNorm',separable='SeparableLayerNorm',dropout='EquivariantDropout')


def read(path): return json.loads(path.read_text())
def write(path,data): path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def csv_write(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as out:
        writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def main():
    points=[read(p) for p in sorted((ROOT/'points').glob('*.json'))]
    checks=[read(p) for p in sorted((ROOT/'checks').glob('*.json'))]
    if not points and (ROOT/'raw_results.json.gz').exists():
        with gzip.open(ROOT/'raw_results.json.gz', 'rt') as stream:
            raw = json.load(stream)
        points = list(raw['points'].values())
        checks = list(raw['checks'].values())
    points = [r for r in points if r['operator'] in ORDER]
    checks = [r for r in checks if r['operator'] in ORDER]
    suites = ET.parse(ROOT/'layernorm.xml').getroot().findall('testsuite')
    suite = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
             for key in ('tests', 'failures', 'errors', 'skipped')}
    suite['passed'] = suite['tests'] - suite['failures'] - suite['errors'] - suite['skipped']
    extras=read(ROOT/'extra_checks.json')
    failed_ops={r['operator'] for r in checks if r['status']!='PASS'}
    if any(r['name']=='graph_softmax_builtin' and r['status']!='PASS' for r in extras):
        failed_ops.add('softmax')
    failures=[]
    for r in checks:
        for name,value in r.get('checks',{}).items():
            if value['failures']:
                failures.append(dict(operator=r['operator'],atoms=r['atoms'],tensor=name,
                    failing_elements=value['failures'],max_abs=value['max_abs'],
                    max_tolerance_ratio=value['max_tolerance_ratio'],examples=value['examples']))
    csv_write(ROOT/'failures.csv',failures)
    lookup={(r['operator'],r['atoms'],r['mode'],r['backend']):r for r in points}
    checkmap={(r['operator'],r['atoms'],r['mode']):r for r in checks}
    pairs=[]
    for op in ORDER:
        for n in sorted({r['atoms'] for r in points if r['operator']==op}):
            check=checkmap.get((op,n,'fwd_bwd'))
            for mode in ('fwd','fwd_bwd'):
                t=lookup.get((op,n,mode,'torch'),{})
                f=lookup.get((op,n,mode,'fused'),{})
                if t.get('status')!='OK' or f.get('status')!='OK': continue
                c=check or checkmap.get((op,n,mode),{})
                metrics=c.get('checks',{})
                needed=list(metrics.values()) if mode=='fwd_bwd' else [metrics.get('output',{})]
                valid=bool(metrics) and all(x and x['failures']==0 for x in needed)
                row=dict(operator=op,atoms=n,edges=t.get('edges',''),mode=mode,
                    correctness='PASS' if valid else 'DIAGNOSTIC',
                    operator_status='FAILED_OTHER_CONFIG' if op in failed_ops else 'PASSED_TESTED_CONFIGS',
                    check_file=str(Path('checks')/f"{op}_{n}_{c.get('mode',mode)}_fused_NKC.json"),
                    input_samples_match=t.get('input_samples')==f.get('input_samples'),
                    torch_gpu_ms=t['gpu_ms_median'],fused_gpu_ms=f['gpu_ms_median'],
                    gpu_speedup=t['gpu_ms_median']/f['gpu_ms_median'],
                    torch_wall_ms=t['wall_ms_median'],fused_wall_ms=f['wall_ms_median'],
                    wall_speedup=t['wall_ms_median']/f['wall_ms_median'],
                    torch_peak_gib=t['peak_allocated_bytes']/2**30,
                    fused_peak_gib=f['peak_allocated_bytes']/2**30,
                    peak_reduction_percent=100*(1-f['peak_allocated_bytes']/t['peak_allocated_bytes']),
                    scope='cached CSR' if op=='softmax' else 'public operator invocation')
                cold=f.get('including_csr_preprocessing')
                if cold:
                    row.update(fused_with_csr_gpu_ms=cold['gpu_ms_median'],
                        fused_with_csr_wall_ms=cold['wall_ms_median'],
                        fused_with_csr_peak_gib=cold['peak_allocated_bytes']/2**30,
                        with_csr_wall_speedup=t['wall_ms_median']/cold['wall_ms_median'])
                pairs.append(row)
    csv_write(ROOT/'paired.csv',pairs)
    flat=[]
    for r in points:
        fields=('operator','atoms','edges','mode','backend','status','stage','error',
                'gpu_ms_median','wall_ms_median','peak_allocated_bytes','baseline_allocated_bytes',
                'incremental_peak_bytes','peak_reserved_bytes')
        flat.append({k:r[k] for k in fields if k in r})
    csv_write(ROOT/'scaling.csv',flat)
    bounds=[]
    for op in ORDER:
        for mode in ('fwd','fwd_bwd'):
            for backend in ('torch','fused'):
                rows=sorted((r for r in points if (r['operator'],r['mode'],r['backend'])==(op,mode,backend)),key=lambda x:x['atoms'])
                if not rows:continue
                ok=[r for r in rows if r['status']=='OK']
                stop=next((r for r in rows if r['status']!='OK'),{})
                bounds.append(dict(operator=op,mode=mode,backend=backend,last_success_n=max((r['atoms'] for r in ok),default=0),
                    next_n=stop.get('atoms',''),stop=stop.get('status','RUNNING'),stage=stop.get('stage',''),reason=stop.get('error','')))
    csv_write(ROOT/'boundaries.csv',bounds)
    correctness=[]
    for op in ORDER:
        rows=[r for r in checks if r['operator']==op]
        components=collections.defaultdict(list)
        for r in rows:
            for name,value in r.get('checks',{}).items():
                components[name].append(dict(atoms=r['atoms'],mode=r['mode'],**value))
        worst={name:dict(max_abs=max(x['max_abs'] for x in values),
                        max_tolerance_ratio=max(x['max_tolerance_ratio'] for x in values),
                        failing_elements=sum(x['failures'] for x in values),
                        worst_ratio_n=max(values,key=lambda x:x['max_tolerance_ratio'])['atoms']) for name,values in components.items()}
        correctness.append(dict(operator=op,checks=len(rows),statuses=dict(collections.Counter(r['status'] for r in rows)),
            atom_counts=sorted({r['atoms'] for r in rows}),components=worst,
            largest_passing_forward_n=max((r['atoms'] for r in rows if r.get('checks',{}).get('output',{}).get('failures',1)==0),default=0),
            largest_passing_fwd_bwd_n=max((r['atoms'] for r in rows if r['mode']=='fwd_bwd' and r['status']=='PASS'),default=0)))
    write(ROOT/'correctness_summary.json',dict(operators=correctness,extras=[{k:r[k] for k in ('name','status','error') if k in r} for r in extras]))
    representative=[r for r in pairs if r['atoms']==4096]
    largest=[]
    for op in ORDER:
        for mode in ('fwd','fwd_bwd'):
            rows=[r for r in pairs if r['operator']==op and r['mode']==mode]
            if rows:largest.append(max(rows,key=lambda r:r['atoms']))
    csv_write(ROOT/'representative.csv',representative)
    csv_write(ROOT/'largest_common.csv',largest)
    old=REPO/'doc/layernorm_validation/2026-09-15'
    hip=[]
    rows=read(old/'perf_hip.json')
    for t in rows:
        if t['class'] not in ('EquivariantLayerNorm', 'EquivariantSeparableLayerNorm') or t['family']!='V3' or t['backend']!='torch':continue
        f=next(r for r in rows if (r['family'],r['class'],r['mode'],r['backend'])==(t['family'],t['class'],t['mode'],'unified'))
        hip.append(dict(operator=t['class'],mode=t['mode'],atoms=4096,torch_ms=t['median_ms'],
                        fused_ms=f['median_ms'],speedup=t['median_ms']/f['median_ms'],reused=True))
    csv_write(ROOT/'hip_layernorm_reused.csv',hip)
    summary=dict(complete=read(ROOT/'manifest.json')['completion']['finished'],commit=read(ROOT/'manifest.json')['commit'],
        point_statuses=dict(collections.Counter(r['status'] for r in points)),
        check_statuses=dict(collections.Counter(r['status'] for r in checks)),
        paired_measurements=len(pairs),diagnostic_pairs=sum(r['correctness']!='PASS' for r in pairs),
        all_input_samples_match=all(r['input_samples_match'] for r in pairs),
        failed_operators=sorted(failed_ops),failures=failures,
        representative=representative,largest_common=largest,boundaries=bounds,hip_reused=hip)
    write(ROOT/'summary.json',summary)
    report=['# EQv3 算子正确性与性能汇总','',f"FastEq `{summary['commit'][:12]}`；H100 80GB，FP32，Torch 2.11.0 / Triton 3.6.0。",
            '本次测量独立算子的公开调用；没有计时完整 EQv3 模型、优化器或纯反向。','',
            '**当前存在未通过项，不能将整组实现标记为已通过正确性。** AttentionAlpha 的大 N 参数梯度超差；GraphSoftmax 的内置广播 rescale 梯度测试超差；EQv3 GateActivation 公开类缺少 forward。逐档失败见下表与 failures.csv。','',
            '## 对照范围','',
            '| 项目 | Torch 对照 | 当前实现与验证边界 |',
            '|---|---|---|',
            '| GraphSoftmax | EQv3 原始 `GraphSoftmax` | 主表复用 CSR；另记每次建 CSR 的完整调用。内置广播 rescale 反向测试存在超差，见问题记录。 |',
            '| AttentionAlpha | EQv3 attention 的 LayerNorm → SmoothLeakyReLU → Dropout → einsum | 前向融合；反向通过 Torch 重算。主测试 dropout=0。 |',
            '| e3nn Gate | 原始 `e3nn.nn.Gate` | 标量激活、门控激活、广播乘法融合；不是 EQv3 `GateActivation`。 |',
            '| LayerNorm / Separable | EQv3 原始两个类 | 分别按阶、标量与高阶分组；共享融合实现。默认反向含 Torch 运算与参数归约。 |',
            '| EquivariantDropout | EQv3 原始 `EquivariantDropout` | 融合随机掩码、广播与乘法；p=0.3、training=True，前向在 no_grad 下仍实际执行 dropout。 |',
            '| EQv3 GateActivation | EQv3 原始 `GateActivation` | 当前公开类缺少 `forward()`，直接调用失败，不能给出有效加速比。 |','',
            '## 正确性','',
            f"删减适配范围后的 LayerNorm 回归：**{suite['passed']} passed，{suite['failures']} failed，{suite['skipped']} skipped**。配置了 EQv2、EQv3 的原始 Torch 源码。性能数据沿用上述旧提交，不是本次重新计时。",
            f"独立全张量输出/梯度对照：{dict(collections.Counter(r['status'] for r in checks))}。所有共同规模均执行独立精度检查；每个张量同时保存 32 个数值样本。",
            '逐元素条件为 `abs(fused - torch) <= atol + rtol * abs(torch)`：普通项目 atol=5e-5、rtol=5e-4；GraphSoftmax 前向 3e-6/3e-5、反向 3e-5/3e-4；Dropout 对同一掩码要求完全相等。容差未放宽。',
            'Dropout 比较时将相同已采样掩码送入原始类，保留原始 forward 的分组、广播、乘法逻辑；计时仍各自正常生成随机掩码。非连续布局、C=13 等补充检查记录在 extra_checks.json。','',
            '| 项目 | 对照组数 | 状态 | 最大通过 N（前向 / 前向+反向） |', '|---|---:|---|---:|']
    for r in correctness:
        report.append(f"| {LABELS[r['operator']]} | {r['checks']} | {r['statuses']} | {r['largest_passing_forward_n']:,} / {r['largest_passing_fwd_bwd_n']:,} |")
    if failures:
        report.extend(['','| 超差项目 | N | 梯度/输出 | 超差元素数 | 最大容差比 |', '|---|---:|---|---:|---:|'])
        for r in failures:
            report.append(f"| {LABELS[r['operator']]} | {r['atoms']:,} | {r['tensor']} | {int(r['failing_elements'])} | {r['max_tolerance_ratio']:.3f} |")
    def timing_table(rows,largest=False):
        tail=' 峰值 GiB（Torch / FastEq） |' if largest else ''
        report.extend(['','| 项目 | N | 模式 | Torch ms | FastEq ms | 倍率 | 精度 |'+tail, '|---|---:|---|---:|---:|---:|---|'+('---:|' if largest else '')])
        for r in rows:
            memory=f" {r['torch_peak_gib']:.3f} / {r['fused_peak_gib']:.3f} |" if largest else ''
            report.append(f"| {LABELS[r['operator']]} | {r['atoms']:,} | {r['mode']} | {r['torch_gpu_ms']:.4f} | {r['fused_gpu_ms']:.4f} | {r['gpu_speedup']:.2f}× | {r['correctness']} |"+memory)
    report.extend(['','## N=4096 对比','',
        'LayerNorm/Dropout 输入 [N,16,128]；Gate 输入 [N,2432]、输出 [N,2048]；图算子 E=32N，H=8，AttentionAlpha 通道=32。'])
    timing_table(representative)
    report.extend(['','## 最大共同规模','',
        '每个模式分别列最后一个双方可运行的规模。表内倍率仅针对该配置；GraphSoftmax 的其他配置超差仍需修复。'])
    timing_table(largest,True)
    report.extend(['','## 扩容边界','','| 项目 | 模式 | 后端 | 最后可运行 N | 下一档 N | 停止原因 |','|---|---|---|---:|---:|---|'])
    for r in bounds:
        report.append(f"| {LABELS[r['operator']]} | {r['mode']} | {r['backend']} | {r['last_success_n']:,} | {r['next_n']} | {r['stop']} |")
    report.extend(['',
        '`OOM` 是实际分配失败；`LIMIT` 是公开接口索引限制；`INDEX_GUARD` 是测试在源码缺少检查的有符号 32 位地址溢出前停止；`FALLBACK` 表示公开接口会退回 Torch，不继续把它计成 Triton。完整错误、失败阶段和内存占用见 boundaries.csv / points。',
        '`LIMIT`、`INDEX_GUARD`、`FALLBACK` 档由测试脚本按已核实的源码条件预检停止，未分配超限输入；它们不是实测 OOM。双方都能运行的规模才有完整逐元素对照。',
        '', '## 计时与内存口径','',
        'N 从 256 连续翻倍。每个后端、模式、规模使用新的子进程；调度进程不建立 GPU 上下文。预热 5 次，测量 20 次，报告中位数；JIT、输入初始化不计时。GPU event 区间包含公开算子调用期间的 GPU 工作与 CPU 提交间隙，并非单个 Triton kernel 延迟；同时保存同步墙钟时间。',
        '`fwd` 为 no_grad；`fwd_bwd` 包含前向、输入梯度、全部可学习参数梯度，没有优化器。峰值为 Torch allocator allocated bytes，含输入、输出、梯度与临时张量，不能当作整卡占用。',
        'GraphSoftmax 的 CSR 构建包含排序和主机同步。主表复用同一拓扑的 CSR；N≤65536 另测每次重建 CSR 的完整调用，其墙钟倍率见 paired.csv 的 with_csr_wall_speedup。',
        '主表 GraphSoftmax 峰值对应复用 CSR 阶段，包含常驻 CSR，但不包含一次性构建时的临时峰值；含构建测量的峰值另存 fused_with_csr_peak_gib。',
        '', '## HIP 已有结果复核','',
        '复用仓库 2026-09-15 的 BW gfx936 记录：当时的 LayerNorm 源码、测试、参考与结果哈希均已核对；旧完整套件为 305 项通过，不代表本次删改后的 HIP 验证。以下为 N=4096 代表点，本次没有在 HIP 重新扩容，也没有为其他四类补做 HIP 验证。',
        '', '| LayerNorm 变体 | 模式 | Torch ms | FastEq ms | 倍率 |', '|---|---|---:|---:|---:|'])
    for r in hip:
        report.append(f"| {r['operator']} | {r['mode']} | {r['torch_ms']:.4f} | {r['fused_ms']:.4f} | {r['speedup']:.2f}× |")
    native_path=ROOT/'softmax_native_checks.json'
    if native_path.exists():
        native=read(native_path)
        report.extend(['','## GraphSoftmax 重复性复核','',
            f"首次内置自测超差，随后一次 344 断言的复跑通过。再将自测参考替换为 EQv3 原始类，重复 {native['repeats']} 轮、{native['assertions']} 个断言，出现 {len(native['failed_assertions'])} 项失败。该问题表现为偶发超差；没有通过放宽容差消除它。"])
        for r in native['failed_assertions']:
            report.append(f"配置 {r['configuration']}；梯度 {r.get('gradient')}；最大容差比 {r['comparison']['max_tolerance_ratio']:.6f}。失败输入与梯度保存于 `{r.get('saved_tensors')}`，后续可直接复现诊断。")
    report.extend(['','## 文件','',
        '- `paired.csv`：全部共同规模、GPU/墙钟计时、峰值内存、加速比与精度状态。',
        '- `points/*.json`：每次独立计时的 20 个样本、输出/梯度数值、输入样本、环境和 OOM 错误。',
        '- `checks/*.json`、`correctness_summary.json`：完整逐元素对照、最坏误差、失败位置。',
        '- `extra_checks.json`、`softmax_diagnosis.json`、`softmax_native_checks.json`：额外测试、重复检查与原始 EQv3 参照下的失败配置。',
        f"- `layernorm.xml`、`layernorm.log`：删减适配范围后的 {suite['passed']} 项回归；源码与测试哈希见 regression.json。",
        '- `manifest.json`：源码与验证范围；`cases.py`、`run.py`、`extras.py` 为复现脚本。',
        '', '测试数据不修改生产算子实现。'])
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('representative','largest_common','boundaries','hip_reused')},indent=2))


if __name__=='__main__':main()
