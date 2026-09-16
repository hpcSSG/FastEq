"""Plot H100 archive and fresh Hygon measurements on separate panels."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
GROUPS={
 'layernorm':(('norm','LayerNorm'),('separable','Separable')),
 'graph_softmax':(('softmax','GraphSoftmax (cached CSR)'),),
 'attention_alpha':(('alpha','AttentionAlpha'),),
 'equivariant_gate':(('gate','e3nn Gate'),),
 'equivariant_dropout':(('dropout','EquivariantDropout'),),
}
NOTES={
 'layernorm':'Public calls include Torch reductions in backward. The two records have different source hashes; see the manifest.',
 'graph_softmax':'Cached CSR; preprocessing is reported separately. These curves do not cover all broadcast-rescale configurations.',
 'attention_alpha':'Backward recomputes Torch expressions. Red crosses denote failed accuracy checks; those timings are diagnostic.',
 'equivariant_gate':'Reference: e3nn 0.4.4 Gate. The separate EQv3 GateActivation interface is not validated by these curves.',
 'equivariant_dropout':'Active dropout: p=0.3, training=True. Correctness replays a common mask; timing uses normal RNG.',
}

def main():
 sources=[('H100 / CUDA (2026-09-15)',ROOT.parent/'2026-09-15/paired.csv'),
          ('Hygon BW / HIP (2026-09-16)',ROOT/'paired.csv')]
 rows=[list(csv.DictReader(p.open())) for _,p in sources]
 output=ROOT/'operator_plots';output.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 for name,ops in GROUPS.items():
  fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),sharey=True)
  atom_range=[int(r['atoms']) for data in rows for r in data if r['operator'] in {x[0] for x in ops}]
  for ax,(title,_),data in zip(axes,sources,rows):
   for i,(op,label) in enumerate(ops):
    for mode,mode_label,color in [('fwd','Forward','#245f9a'),('fwd_bwd','Forward + backward','#c66b20')]:
     selected=sorted([r for r in data if r['operator']==op and r['mode']==mode],key=lambda r:int(r['atoms']))
     legend=(label+' / ' if len(ops)>1 else '')+mode_label
     ax.plot([int(r['atoms']) for r in selected],[float(r['gpu_speedup']) for r in selected],color=color,
             linestyle='-' if i==0 else '--',label=legend,marker='.',linewidth=1.5)
     bad=[r for r in selected if r['correctness']!='PASS']
     if bad:ax.scatter([int(r['atoms']) for r in bad],[float(r['gpu_speedup']) for r in bad],marker='x',color='#bc2025',s=48,zorder=5)
   ax.axhline(1,color='#555',linestyle=':',linewidth=.8)
   ax.set_xscale('log',base=2);ax.set_xlim(min(atom_range)/1.2,max(atom_range)*1.2)
   ax.set_title(title);ax.set_xlabel('Atoms N')
   ax.grid(alpha=.16);ax.legend(frameon=False,fontsize=8,loc='best')
  axes[0].set_ylabel('Native Torch time / FastEq time')
  fig.suptitle('Standalone FP32 calls: '+('LayerNorm variants' if name=='layernorm' else ops[0][1]),fontsize=13)
  fig.text(.03,.047,NOTES[name],fontsize=8)
  fig.text(.03,.012,'Red x: failed accuracy check (diagnostic timing). Ratios compare implementations within a device, not hardware throughput.',fontsize=8,color='#555')
  fig.tight_layout(rect=(0,.085,1,.94))
  for ext in ('png','svg'):
   destination=output/(name+'.'+ext)
   fig.savefig(destination,dpi=180)
   if ext=='svg':destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
  plt.close(fig)
if __name__=='__main__':main()
