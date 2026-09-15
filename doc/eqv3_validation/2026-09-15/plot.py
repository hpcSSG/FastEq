"""Static plots of measured public-call speedup (not a roofline estimate)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
ORDER=('softmax','alpha','gate','norm','separable','dropout')
LABELS=('GraphSoftmax (cached CSR)*','AttentionAlpha','e3nn Gate**','LayerNorm',
        'SeparableLayerNorm','EquivariantDropout')
rows=list(csv.DictReader((ROOT/'paired.csv').open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'fwd':'#3066a5','fwd_bwd':'#cc7129'}

fig,axes=plt.subplots(1,2,figsize=(12,5.6),sharey=True)
for ax,mode,title in zip(axes,('fwd','fwd_bwd'),('Forward (no_grad)','Forward + first backward')):
    for i,op in enumerate(ORDER):
        r=next((r for r in rows if r['operator']==op and r['mode']==mode and int(r['atoms'])==4096),None)
        if r is None: continue
        speed=float(r['gpu_speedup'])
        valid=r['correctness']=='PASS'
        ax.barh(i,speed,color=colors[mode],alpha=.85 if valid else .3,
                hatch=None if valid else '//')
        ax.text(speed+.025,i,f'{speed:.2f}x'+(' [accuracy failed]' if not valid else ''),va='center',fontsize=9)
    ax.axvline(1,color='black',linewidth=.8,linestyle='--')
    ax.set_title(title);ax.set_xlabel('Torch time / FastEq time')
    ax.set_yticks(range(len(ORDER)),LABELS);ax.set_xlim(0,max(2,ax.get_xlim()[1]*1.17))
axes[0].invert_yaxis()
fig.suptitle('H100, FP32, N=4096 — median public-call GPU interval',fontsize=13)
fig.text(.02,.02,'* Regular benchmark configuration only; GraphSoftmax has a separate broadcast-rescale gradient failure.\n'
    '** e3nn Gate is a different API from EQv3 GateActivation; the latter public module cannot run.\n'
    'AttentionAlpha also has large-N weight-gradient failures; see failures.csv.\n'
    'Alpha backward recomputes in Torch; LayerNorm uses native Torch reductions. Graph inputs E=32N; no full-model timing.',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.95))
fig.savefig(ROOT/'speedup_4096.png',dpi=180)
fig.savefig(ROOT/'speedup_4096.svg')
plt.close(fig)

fig,axes=plt.subplots(2,4,figsize=(14,7))
for ax,op,label in zip(axes.flat,ORDER,LABELS):
    for mode in ('fwd','fwd_bwd'):
        selected=sorted([r for r in rows if r['operator']==op and r['mode']==mode],key=lambda r:int(r['atoms']))
        x=[int(r['atoms']) for r in selected]
        y=[float(r['gpu_speedup']) for r in selected]
        ax.plot(x,y,color=colors[mode],linewidth=1.4,marker='.',markersize=5)
        bad=[r for r in selected if r['correctness']!='PASS']
        ax.scatter([int(r['atoms']) for r in bad],[float(r['gpu_speedup']) for r in bad],marker='x',s=45,color='red',zorder=5)
    ax.axhline(1,color='black',linewidth=.7,linestyle='--')
    if not any(r['operator']==op for r in rows):
        ax.set_xlim(256,524288)
    ax.set_xscale('log',base=2);ax.set_title(label,fontsize=10)
    ax.set_xlabel('Atoms N');ax.set_ylabel('Torch / FastEq')
    ax.tick_params(axis='x',labelrotation=35,labelsize=8)
for unused in axes.flat[len(ORDER):]:
    unused.axis('off')
axes.flat[-1].legend([Line2D([0],[0],color=colors[m]) for m in ('fwd','fwd_bwd')],
    ['Forward','Forward + backward'],loc='upper left',frameon=False)
axes.flat[-1].text(0,.6,'Red x: accuracy failed or unavailable.\n\nEnd points are implementation limits\nor memory boundaries; see boundaries.csv.\n\nAlpha and LayerNorm backward include\nTorch operations. Alpha has large-N\nweight-gradient failures.\n\n* Cached topology; broadcast case fails.\n** e3nn Gate, not EQv3 GateActivation.',va='top',fontsize=9)
fig.suptitle('H100, FP32 — doubling N; independent processes, 5 warmups + 20 samples',fontsize=13)
fig.tight_layout(rect=(0,0,1,.95))
fig.savefig(ROOT/'speedup_scaling.png',dpi=180)
fig.savefig(ROOT/'speedup_scaling.svg')
plt.close(fig)
