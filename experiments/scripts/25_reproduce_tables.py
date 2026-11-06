"""Regenerate the 24 current empirical table bodies from saved results.

Writes standalone LaTeX tabular/CSV files and checks displayed numerical cells
against the manuscript. Does not edit the manuscript or rerun experiments.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import importlib
import re
import pandas as pd
from _common import ROOT

NAMES=dict(iris='Iris',wdbc='WDBC',vehicle='Vehicle',abalone='Abalone',magic='MAGIC',online_news='Online News')
SOURCES=[]
def read(name):
    p=ROOT/'experiments/outputs'/name
    SOURCES.append(p)
    return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
def g(v): return f'{v:g}'
def sci(v):
    if v==0: return '$0$'
    mantissa,exponent=f'{v:.3e}'.split('e')
    return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
def nums(text):
    text=re.sub(r'([-+]?\d+(?:\.\d+)?)\\times10\^\{([-+]?\d+)\}',r'\1e\2',text)
    return [float(x) for x in re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?',text)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/outputs/revision_r1_q6/tables')
    parser.add_argument('--no-manuscript-check',action='store_true',
                        help='Generate from fresh results without comparing their numbers to the saved manuscript (e.g. new hardware timings).')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    tables={}
    m=read('raw_results/synthetic_membership_raw_v2.parquet')
    p=m[m.evaluation=='proposed']; gen=p[p.control=='generated']
    seed=gen.groupby(['kappa','eta','n_records','sigma','delta']).absolute_delta.median()
    null=read('revision_r1_q7/summary.csv').set_index('n_records')
    def null_interval(n):
        r=null.loc[n]; scale=1e-4 if n in [500,2000] else 1e-5
        exponent=-4 if n in [500,2000] else -5
        return f'$[{r.q10/scale:.3f},{r.q90/scale:.3f}]'+r'\times10^{'+str(exponent)+'}$'
    combined='$'+f'{null.loc[2000,"median"]/1e-4:.5f}'+r'\,'+null_interval(2000)[1:]
    fixed=gen[(gen.sigma==1)&(gen.delta==.5)].copy()
    fixed['uncancelled']=fixed.absolute_delta / fixed.empirical_absolute_asymmetry.where(fixed.empirical_absolute_asymmetry!=0)
    cm=fixed.groupby(['kappa','eta','n_records'])[['absolute_delta','empirical_absolute_asymmetry','uncancelled']].median().groupby('eta').median()
    tables['controlled']=[[g(eta),f'{cm.loc[eta,"absolute_delta"]*1000:.3f}',f'{cm.loc[eta,"empirical_absolute_asymmetry"]*1000:.3f}',f'{cm.loc[eta,"uncancelled"]:.4f}'] for eta in [0,.02,.05,.1]]
    tables['exchangeable-fixed']=[[g(n)]+[sci(null.loc[n,k]) for k in ['mean','std','median']]+[null_interval(n),g(null.loc[n,'positive']),g(null.loc[n,'negative'])] for n in [500,2000,10000]]
    static=read('tabular/summary.csv'); inv=read('tabular/datasets.csv').set_index('dataset')
    static=static[static.label_count==3]
    tables['static-benchmarks']=[]
    for ds,name in NAMES.items():
        v=static[(static.dataset==ds)&(static.evaluation=='proposed')&(static.sigma==1)&(static.delta==1)].absolute_delta_median.iloc[0]*1000
        tables['static-benchmarks'].append([name,g(inv.loc[ds,'rows']),g(inv.loc[ds,'predictors']),f'{v:.3g}'])
    ops=[('product',None,'Product'),('minimum',None,'Minimum'),('power',.01,r'$E_{0.01}$'),('power',2,r'$E_2$'),('proposed',None,r'$E_{\sigma,\delta}$')]
    operator_blocks=[]; tables['static-operator-comparison']=[]
    for op,power,label in ops:
        block=static[static.evaluation==op]
        if power is not None: block=block[block.power==power]
        grouped=block.groupby('dataset').median(numeric_only=True)
        operator_blocks.append(grouped)
        tables['static-operator-comparison'].append([label,sci(grouped.absolute_delta_median.median())])
    primary=read('revision_r1_q3/primary_summary.csv'); contrasts=read('revision_r1_q3/matched_contrasts.csv')
    tables['matched-retention']=[]
    for r in primary.itertuples():
        tables['matched-retention'].append([NAMES[r.dataset],g(r.sigma),g(r.n_zero),g(r.n_minus),g(r.n_plus),f'({r.zero_to_minus_added}, {r.zero_to_minus_removed})',f'({r.zero_to_plus_added}, {r.zero_to_plus_removed})',f'{r.zero_to_minus_jaccard:.4f}',f'{r.zero_to_plus_jaccard:.4f}',g(r.minus_to_plus_orientation_flips)])
    q4=read('revision_r1_q4/dataset_summary.csv').set_index(['dataset','metric'])
    def med(ds,k): return q4.loc[(ds,k),'median']
    def cell(ds,k,places=None):
        r=q4.loc[(ds,k)]; f=g if places is None else lambda x:f'{x:.{places}f}'
        return f"{f(r['median'])} [{f(r.q25)}, {f(r.q75)}]"
    tables['split-stability']=[[NAMES[ds],cell(ds,'zero_to_minus_jaccard',4),cell(ds,'zero_to_plus_jaccard',4)]+[f'({g(med(ds,"zero_to_"+s+"_added"))}, {g(med(ds,"zero_to_"+s+"_removed"))})' for s in ['minus','plus']] for ds in NAMES]
    prune=read('raw_results/pruning_multivariate.csv')
    tables['pruning']=[]
    for k in [2,4]:
        for t in [.01,.05]:
            block=prune[(prune.K==k)&(prune.support_threshold==t)]
            counts=[block[block['mode']==mode].scored.median() for mode in ['exhaustive','coverage_only','coverage_support']]
            wide=block.pivot(index='seed',columns='mode',values='scored')
            ratio=(1-wide.coverage_support/wide.exhaustive).median()
            incremental=(1-wide.coverage_support/wide.coverage_only).median()
            tables['pruning'].append([g(k),g(t)]+list(map(g,counts))+[f'{ratio:.3f}',f'{incremental:.3f}'])
    q5=read('computational_followup/controlled/measurements.csv')
    assert len(q5)==300 and q5.retained_set_equal.all() and q5.counts_match_original.all()
    modes=['exhaustive','coverage_only','coverage_support']; display=['Exhaustive','Coverage only','Joint']
    def block(v,d,t,mode): return q5[(q5.variables==v)&(q5.delta==d)&(q5.support_threshold==t)&(q5['mode']==mode)]
    tables['computational-extension']=[]
    for v in [5,8,10]:
        for t in [.01,.05]:
            bs=[block(v,.375,t,mode) for mode in modes]
            tables['computational-extension'].append([g(v),g(t)]+[g(b.scored.median()) for b in bs]+[f'{b.elapsed_seconds.median():.3f}' for b in bs])
    dynamic=read('revision_r1_q6/dynamic_null_completed.csv')
    tables['dynamic']=[]
    for filt,metric,value,label in [('all delta','Psi','median','Signed median'),('all delta','Psi','iqr','IQR'),('all delta','abs_Psi','median','Absolute median'),('delta != 0','abs_Omega','median','Increment median'),('delta != 0','abs_Omega','q95','Increment Q95')]:
        rows=dynamic[(dynamic['filter']==filt)&(dynamic.metric==metric)].set_index('condition')
        tables['dynamic'].append([label]+[sci(rows.loc[c,value]) for c in ['lambda=0.25','lambda=0.5','lambda=1','independent_AR_null']])
    thresholds=read('summaries/selection_asymmetry_thresholds_v2.csv').set_index(['label_count','threshold'])
    tables['selection-thresholds']=[]
    for s in [.01,.025,.05]:
        for c in [.3,.5,.7]:
            key=f'cov0p01_supp{g(s).replace(".","p")}_conf{g(c).replace(".","p")}'
            vals=[b['SA_'+key].median() for b in operator_blocks]+[thresholds.loc[(l,key),'median_SA'] for l in [3,5]]
            tables['selection-thresholds'].append([f'{s:.3f}',g(c)]+[f'{x:.3f}' for x in vals])
    selected=gen[(gen.kappa==2)&(gen.eta==.05)&(gen.n_records==10000)&(gen.sigma==1)].copy()
    selected['delta_gamma']=selected.delta*selected.gamma
    tables['parameter']=[[g(d),sci(abs(d)/140),sci(b.support_delta.median()),sci(b.delta_gamma.median())] for d,b in selected.groupby('delta')]
    flips=contrasts[contrasts.minus_to_plus_orientation_flips>0]
    tables['matched-flips']=[[NAMES[r.dataset],g(r.label_count),g(r.sigma),g(r.magnitude),g(r.tau_support),g(r.tau_confidence),g(r.n_zero),g(r.n_minus),g(r.n_plus),g(r.minus_to_plus_orientation_flips)] for r in flips.itertuples()]
    reps=read('revision_r1_q3/representative_rules.csv')
    tables['matched-examples']=[]
    for r in reps.itertuples():
        state='Both' if r.selected_forward and r.selected_reverse else (r'$A\Rightarrow C$' if r.selected_forward else (r'$C\Rightarrow A$' if r.selected_reverse else 'Neither'))
        tables['matched-examples'].append([g(r.example),g(r.delta)]+[f'{getattr(r,k):.6f}' for k in ['support_forward','support_reverse','confidence_forward','confidence_reverse']]+[state])
    for r in reps.drop_duplicates('example').itertuples():
        def item(s):
            name,index=s.rsplit('::',1); labels=['low','middle','high'] if r.label_count==3 else ['lowest','lower','middle','higher','highest']
            return r'\texttt{'+name.replace('_',r'\_')+'} ('+labels[int(index)]+')'
        tables['matched-examples'].append([g(r.example),NAMES[r.dataset],item(r.item_a),item(r.item_c),f'{r.coverage_forward:.6f}',f'{r.coverage_reverse:.6f}'])
    tables['split-counts']=[[NAMES[ds]]+[cell(ds,k) for k in ['n_zero','n_minus','n_plus']] for ds in NAMES]
    tables['split-counts'] += [[NAMES[ds]]+[cell(ds,k) for k in ['zero_to_minus_added','zero_to_minus_removed','zero_to_plus_added','zero_to_plus_removed']] for ds in NAMES]
    reps4=read('revision_r1_q4/representative_summary.csv')
    tables['split-gaps']=[[g(r.example),g(r.delta),f'{r.gap_median*1000:.4f}',f'[{r.gap_q10*1000:.4f}, {r.gap_q90*1000:.4f}]',f'{r.sign_agrees}/{r.available}'] for r in reps4[reps4.delta>0].itertuples()]
    tables['split-retention']=[[g(r.example),g(r.delta)]+[f'{getattr(r,k)}/{r.available}' for k in ['selected_forward_count','selected_reverse_count','state_changed_count']] for r in reps4.itertuples()]
    for key,configs in [('computational-scale',[(v,.375) for v in [5,8,10]]),('computational-parameters',[(8,d) for d in [-.375,0,.375]])]:
        tables[key]=[]
        for v,d in configs:
            for t in [.01,.05]:
                for mode,label in zip(modes,display):
                    b=block(v,d,t,mode)
                    vals=[g(v if key=='computational-scale' else d),g(t),label,g(b.scored.median())]
                    for metric,places in [('elapsed_seconds',3),('peak_mib',1)]:
                        a=b[metric]; vals.append(f'{a.median():.{places}f} [{a.quantile(.25):.{places}f}, {a.quantile(.75):.{places}f}]')
                    tables[key].append(vals)
    importlib.import_module('32_build_reviewer2_tables').main()
    r2=ROOT/'experiments/outputs/revision_r2'
    tables.update(json.loads((r2/'tables/rows.json').read_text(encoding='utf-8')))
    SOURCES.extend([r2/n for n in ['affine_summary.csv','selection_choices.csv','selection_outer.csv',
                                  'multiitem_summary.csv','temporal_summary.csv',
                                  'computation/summary.csv','computation/phase_diagnostics.csv','tables/rows.json']])
    texts=[(ROOT/'table_reference'/n).read_text(encoding='utf-8') for n in ['main_submission.tex','supplementary_appendix.tex']]
    checks=[]
    for label,rows in tables.items():
        text=next(t for t in texts if r'\label{tab:'+label+'}' in t)
        part=text.split(r'\label{tab:'+label+'}',1)[1].split(r'\end{table',1)[0]
        actual=[]
        for segment in re.findall(r'\\midrule(.*?)\\bottomrule',part,re.S):
            for line in segment.splitlines():
                if '&' in line and r'\\' in line: actual.append(line.split(r'\\')[0].strip().split('&'))
        assert len(actual)==len(rows),(label,len(actual),len(rows))
        if not args.no_manuscript_check:
            for i,(expected,saved) in enumerate(zip(rows,actual)):
                # The first column contains row labels, not numerical outcomes.
                assert len(expected)==len(saved),(label,i)
                if re.fullmatch(r'-?\d+(?:\.\d+)?',expected[0]):
                    assert nums(expected[0])==nums(saved[0]),(label,i,0)
                for j,(a,b) in enumerate(zip(expected[1:],saved[1:]),1):
                    assert nums(a)==nums(b),(label,i,j,a,b)
        with (args.output/(label+'.csv')).open('w',encoding='utf-8',newline='') as f: csv.writer(f).writerows(rows)
        chunks=[]
        for row in rows:
            if not chunks or len(chunks[-1][0])!=len(row): chunks.append([])
            chunks[-1].append(row)
        tex=[]
        for chunk in chunks:
            tex += [r'\begin{tabular}{'+'l'*len(chunk[0])+'}']+[' & '.join(r)+r' \\' for r in chunk]+[r'\end{tabular}']
        (args.output/(label+'.tex')).write_text('\n'.join(tex)+'\n',encoding='utf-8')
        checks.append(dict(label=label,rows=len(rows),displayed_numeric_cells_match=None if args.no_manuscript_check else True))
    manifest=dict(tables=checks,manuscript_numbers_checked=not args.no_manuscript_check,input_files=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dict.fromkeys(SOURCES)])
    (args.output/'verification.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if args.no_manuscript_check:
        print(f'Generated {len(tables)} table bodies from saved result files; manuscript-number comparison disabled.')
    else:
        print(f'Generated and numerically verified {len(tables)} table bodies against the current manuscript.')

if __name__=='__main__': main()
