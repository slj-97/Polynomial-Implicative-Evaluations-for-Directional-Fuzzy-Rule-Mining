"""Build table bodies for the affine, selection, temporal, and computational analyses."""
from __future__ import annotations
import csv
import json
from _common import ROOT
import pandas as pd

OUT=ROOT/'experiments/outputs/revision_r2'
NAMES=dict(iris='Iris',wdbc='WDBC',vehicle='Vehicle',abalone='Abalone',magic='MAGIC',online_news='Online News')
def sci(v: float) -> str:
    if v==0:return '$0$'
    a,b=f'{v:.3e}'.split('e');return '$'+a+r'\times10^{'+str(int(b))+'}$'
def fmt(v: float) -> str:return f'{v:g}'
def table(label: str,caption: str,header: list[str],rows: list[list[str]],note: str,wide: bool=True) -> str:
    env='table*' if wide else 'table'
    return '\n'.join([r'\begin{'+env+'}[t]',r'\centering\color{red}',r'\caption{'+caption+'}',
          r'\label{tab:'+label+'}',r'\begin{tabular}{'+'l'+'r'*(len(header)-1)+'}',r'\toprule',
          ' & '.join(header)+r' \\',r'\midrule',*[' & '.join(r)+r' \\' for r in rows],r'\bottomrule',
          r'\end{tabular}',r'\par\vspace{0.4ex}\footnotesize\emph{Note:} '+note,r'\end{'+env+'}'])

def main() -> None:
    target=OUT/'tables';target.mkdir(exist_ok=True)
    rows={};snippets={}
    static=pd.read_csv(ROOT/'experiments/outputs/tabular/summary.csv');static=static[static.label_count==3]
    data=[]
    for op,power,label in [('product',None,'Product'),('minimum',None,'Minimum'),('power',.01,r'$E_{0.01}$'),('power',2,r'$E_2$')]:
        part=static[static.evaluation==op]
        if power is not None:part=part[part.power==power]
        data.append([label,sci(part.groupby('dataset').absolute_delta_median.median().median())])
    proposed=static[static.evaluation=='proposed']
    data.append([r'$E_{\sigma,0}$ (symmetric)',sci(0)])
    for sigma,d in [(.25,.125),(.5,.25),(1,.5),(1,1)]:
        part=proposed[(proposed.sigma==sigma)&(proposed.delta==d)]
        data.append([f'$E_{{{sigma:g},{d:g}}}$',sci(part.absolute_delta_median.median())])
    affine=pd.read_csv(OUT/'affine_summary.csv')
    data.append([r'Broader affine, $d=0.25$',sci(affine[(affine.family=='broader')&(affine.d==.25)].absolute_gap_median.median())])
    rows['static-operator-comparison']=data
    snippets['static-operator-comparison']=table('static-operator-comparison','Static support comparison with three labels.',
          ['Evaluation',r'Median $|\Delta_E|$'],data,
          r'Medians across six datasets. Parameter-specific rows identify $(\sigma,\delta)$. Broader affine coefficients are defined in Table~\ref{tab:affine-matched}.',False)
    counts=[]
    for ds,label in NAMES.items():
        b=affine[affine.dataset==ds].set_index(['family','d'])
        counts.append([label]+[fmt(b.loc[(family,d),'retained']) for family in ['subclass','broader'] for d in [-.25,0,.25]])
    rows['affine-matched']=counts
    snippets['affine-matched']=table('affine-matched','Retained rules under matched admissible affine evaluations.',
          ['Dataset',r'$N_{\mathrm S,-}$',r'$N_{\mathrm S,0}$',r'$N_{\mathrm S,+}$',r'$N_{\mathrm B,-}$',r'$N_{\mathrm B,0}$',r'$N_{\mathrm B,+}$'],counts,
          r'S: original subclass with $\sigma=0.5$; B: $Q=1/2+(x+y)/8+d(x-y)/2$. Signs correspond to $d=\pm0.25$; zero is symmetric. All use three labels and thresholds $(0.01,0.01,0.5)$. Matched $d$ gives the same support difference and $\mathfrak D=|d|/140$.')
    choices=pd.read_csv(OUT/'selection_choices.csv');outer=pd.read_csv(OUT/'selection_outer.csv')
    sel=[]
    for ds,label in NAMES.items():
        c=choices[choices.dataset==ds];b=outer[outer.dataset==ds]
        frequencies='('+', '.join(str(int((c.chosen_delta==d).sum())) for d in [-1,-.5,0,.5,1])+')'
        sel.append([label,frequencies]+[f'{b[b.strategy==strategy].jaccard.median():.4f}' for strategy in ['symmetric','selected']]
                   +[fmt(b[b.strategy==strategy].evaluation_retained.median()) for strategy in ['symmetric','selected']])
    rows['validation-selection']=sel
    snippets['validation-selection']=table('validation-selection','Validation-selected parameters and outer evaluation agreement.',
       ['Dataset',r'Chosen-$\delta$ counts',r'$J_0$',r'$J_{\mathrm{sel}}$',r'$N_0$',r'$N_{\mathrm{sel}}$'],sel,
       r'Five outer 70/30 splits. Counts follow $\delta=(-1,-0.5,0,0.5,1)$ at $\sigma=1$. $J$ compares development and outer-evaluation retained sets; $N$ is the outer retained count. Both report medians. Parameter choice uses only three inner 80/20 splits of development data.')
    multi=pd.read_csv(OUT/'multiitem_summary.csv')
    rows['multiitem-worked']=[[fmt(r.delta)]+[f'{getattr(r,k):.8f}' for k in ['support_forward','support_reverse','baseline_difference','increment','difference']] for r in multi.itertuples()]
    snippets['multiitem-worked']=table('multiitem-worked','Worked multi-item comparison at fixed symmetric parameter.',
       [r'$\delta$',r'$S_\delta(AB\Rightarrow C)$',r'$S_\delta(CB\Rightarrow A)$',r'$D_0$',r'$D_\delta-D_0$',r'$D_\delta$'],rows['multiitem-worked'],
       r'$\sigma=1$, $T=T_{\mathrm{LK}}$. Three records $(a,b,c)$ are $(0.9,0.8,0.4)$, $(0.3,0.9,0.7)$ and $(0.8,0.6,0.5)$. Supports are empirical means. Exact rational arithmetic verifies the decomposition.')
    temporal=pd.read_csv(OUT/'temporal_summary.csv');tr=[]
    def interval(r):return f'{r["median"]*1000:.4f} [{r.q10*1000:.4f}, {r.q90*1000:.4f}]'
    for null in [False,True]:
        for i in range(3):
            for j in range(3):
                b=temporal[(temporal['null']==null)&(temporal.label_x==i)&(temporal.label_y==j)].set_index('metric')
                tr.append(['AR null' if null else 'Coupled',f'{i}/{j}',interval(b.loc['Psi']),interval(b.loc['Omega']),
                           f'{int(b.loc["Omega","positive"])}/{int(b.loc["Omega","negative"])}'])
    rows['temporal-variability']=tr
    snippets['temporal-variability']=table('temporal-variability','Whole-trajectory variation at a fixed dynamic configuration.',
       ['Process',r'Labels $i/j$',r'$\Psi$: median $[Q_{10},Q_{90}]$',r'$\Omega$: median $[Q_{10},Q_{90}]$',r'$N_+/N_-$'],tr,
       r'All numerical gaps are in $10^{-3}$. Each row summarizes 50 independent complete trajectories; labels 0/1/2 are low/middle/high. $\Omega=\delta\Lambda$, $(\sigma,\delta)=(1,0.5)$, true and evaluation lag 1, linear mechanism, coupling 0.5, noise 0.1. The null removes cross-driving; signs count $\Omega$ using tolerance $10^{-12}$.')
    comp=pd.read_csv(OUT/'computation/summary.csv');cr=[]
    cases=['synthetic-base','synthetic-n1000','synthetic-n4000','synthetic-label5','synthetic-K2','iris-K2','wdbc-K2']
    names=dict(zip(cases,['Base','1000 records','4000 records','Five labels','$K=2$','Iris','WDBC']))
    for case in cases:
        for threshold in [.01,.05]:
            for mode,label in [('exhaustive','E'),('coverage_only','C'),('coverage_support','J')]:
                r=comp[(comp['case']==case)&(comp.support_threshold==threshold)&(comp['mode']==mode)].iloc[0]
                cr.append([names[case],fmt(threshold),label,fmt(r.generated_median),fmt(r.scored_median),fmt(r.retained_median),
                           f'{r.elapsed_seconds_median:.3f} [{r.elapsed_seconds_q25:.3f}, {r.elapsed_seconds_q75:.3f}]',
                           f'{r.peak_mib_median:.1f} [{r.peak_mib_q25:.1f}, {r.peak_mib_q75:.1f}]'])
    rows['computational-axes']=cr
    snippets['computational-axes']=table('computational-axes','Record, label, antecedent-length, and real-data computational comparisons.',
       ['Case',r'$\tau_{\rm supp}$','Mode','Generated','Scored','Retained','Time (s)','Peak (MiB)'],cr,
       r'Medians [quartiles] over five seeds, 20260903--20260907. E/C/J: exhaustive/coverage-only/joint. Base: five variables, 240 evaluation records, three labels, $K=4$; one factor varies per synthetic case. Iris/WDBC use all numeric predictors, three labels, $K=2$ and a training-only 70/30 split. All searches in a matched comparison return identical rule keys. Time excludes preprocessing; peak is whole-worker memory.')
    phases=pd.read_csv(OUT/'computation/phase_diagnostics.csv');pr=[]
    for case in cases:
        b=phases[(phases['case']==case)&(phases.support_threshold==.01)].set_index('mode')
        j=b.loc['coverage_support']
        pr.append([names[case]]+[f'{b.loc[m,"candidate"]*1000:.3f}' for m in ['exhaustive','coverage_only','coverage_support']]
                 +[f'{j[k]*1000:.3f}' for k in ['aggregation','coverage','support','confidence','other']])
    rows['candidate-phases']=pr
    snippets['candidate-phases']=table('candidate-phases','Separate diagnostic timing of candidate construction and evaluation phases.',
       ['Case',r'Cand. E',r'Cand. C',r'Cand. J',r'Agg. J',r'Cov. J',r'Supp. J',r'Conf. J',r'Other J'],pr,
       r'Milliseconds in separate instrumented passes at seed 20260903 and support threshold 0.01. Candidate time covers consequent filtering, used-variable checks and prefix creation. Other includes remaining traversal, bookkeeping, output sorting and instrumentation. These diagnostic times are not the ordinary wall times in Table~\ref{tab:computational-axes}. All diagnostic counts and retained sets match the unchanged miner. All 42 diagnostic records, including threshold 0.05, are supplied.')
    for label,data in rows.items():
        with (target/(label+'.csv')).open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerows(data)
        (target/(label+'.tex')).write_text(snippets[label]+'\n',encoding='utf-8')
    (target/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(f'Built {len(rows)} table bodies from saved results.')

if __name__=='__main__':main()
