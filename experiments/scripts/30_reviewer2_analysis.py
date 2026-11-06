"""Compute exact examples, matched affine comparisons, nested parameter
selection, and variation across complete trajectories.
"""
from __future__ import annotations
import argparse
import importlib
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

from _common import ROOT
from experiments.src.affine_admissibility import is_admissible
from experiments.src.fuzzification import fuzzify_our_main
from experiments.src.matched_rules import retention_flags
from experiments.src.operators import boundary_factor, eval_poly
from experiments.src.repeated_splits import pair_moments, score_pairs
from experiments.src.reproducibility import load_yaml
from experiments.src.synthetic import dynamic_process

OUT = ROOT / 'experiments/outputs/revision_r2'
CONFIG = ROOT / 'experiments/configs/revision_r2.yaml'

def save_json(name: str, value: object) -> None:
    (OUT/name).write_text(json.dumps(value, indent=2), encoding='utf-8')

def numeric_data(dataset: str, tabular: dict) -> pd.DataFrame:
    provenance=json.loads((ROOT/'data/raw/tabular_firm/manifest.json').read_text())
    entry=next(r for r in provenance if r['dataset']==dataset)
    path=ROOT/entry['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    frame=pd.read_csv(path); frame.columns=frame.columns.str.strip()
    return frame.drop(columns=[tabular['targets'][dataset]]).select_dtypes(include=[np.number])

def flags(frame: pd.DataFrame, config: dict) -> np.ndarray:
    t=config['thresholds']
    return retention_flags(frame,t['coverage'],t['support'],t['confidence'])

def overlap(a: np.ndarray,b: np.ndarray) -> float:
    union=np.logical_or(a,b).sum()
    return float(np.logical_and(a,b).sum()/union) if union else 1.0

def keyed_flags(frame: pd.DataFrame, config: dict) -> set[tuple[str,str]]:
    f=flags(frame,config); result=set()
    for side in range(2):
        for a,c in frame.loc[f[:,side],['item_a','item_c']].itertuples(index=False,name=None):
            result.add((a,c) if side==0 else (c,a))
    return result

def set_overlap(a: set,b: set) -> float:
    return len(a&b)/len(a|b) if a|b else 1.0

def examples() -> None:
    x,y=sp.symbols('x y', real=True)
    B=lambda a,c:a*c*(1-a)*(1-c)
    H=lambda a,c:B(a,c)*(a-c)
    E=lambda a,c,d:a*c-B(a,c)*(a+c)/2+d*H(a,c)/2
    records=[(sp.Rational(9,10),sp.Rational(8,10),sp.Rational(4,10)),
             (sp.Rational(3,10),sp.Rational(9,10),sp.Rational(7,10)),
             (sp.Rational(8,10),sp.Rational(6,10),sp.Rational(5,10))]
    rows=[]
    for i,(a,b,c) in enumerate(records,1):
        u=max(0,a+b-1); v=max(0,c+b-1)
        base=E(u,c,0)-E(v,a,0)
        coefficient=(H(u,c)-H(v,a))/2
        for d in [sp.Rational(-1,2),sp.Integer(0),sp.Rational(1,2)]:
            difference=E(u,c,d)-E(v,a,d)
            assert sp.simplify(difference-base-d*coefficient)==0
            values=dict(record=i,a=a,b=b,c=c,u=u,v=v,delta=d,
                        support_forward=E(u,c,d),support_reverse=E(v,a,d),
                        baseline_difference=base,increment=d*coefficient,difference=difference)
            rows.append({k:float(v) for k,v in values.items()})
    pd.DataFrame(rows).to_csv(OUT/'multiitem_records.csv',index=False)
    means=pd.DataFrame(rows).groupby('delta').mean(numeric_only=True).reset_index()
    means.to_csv(OUT/'multiitem_summary.csv',index=False)
    e=x*y**2; induced=1-x+e
    assert sp.diff(e,x)==y**2 and sp.diff(induced,x)==-1+y**2
    save_json('examples.json',dict(relaxed_evaluation=str(e),induced=str(induced),
              degree=3,Ex=str(sp.diff(e,x)),Ix=str(sp.diff(induced,x)),
              Iy=str(sp.diff(induced,y)),NP_at_one=str(induced.subs(x,1)),
              multiitem_records='multiitem_records.csv',sigma=1,
              formula='D_delta=D_0+delta*mean(H(T(a,b),c)-H(T(c,b),a))/2'))

def affine(config: dict,tabular: dict) -> None:
    summary=[]; contrasts=[]; changed=[]; diagnostics=[]
    for dataset in tabular['datasets']:
        data=numeric_data(dataset,tabular)
        split=np.load(ROOT/f'experiments/outputs/tabular/{dataset}_split.npz')
        _,memberships=fuzzify_our_main(data.iloc[split['training']],data.iloc[split['test']],3)
        moments=pair_moments(memberships)
        z=np.column_stack(list(memberships.values()))
        v=z*(1-z); boundary=(v.T@v)/len(z)
        names=list(memberships); loc={n:i for i,n in enumerate(names)}
        i=np.array([loc[n] for n in moments.item_a]); j=np.array([loc[n] for n in moments.item_c])
        moment_b=boundary[i,j]
        groups={}
        for family in ['subclass','broader']:
            scores={}
            for d in config['affine']['differences']:
                a=0.0 if family=='subclass' else config['affine']['broader_a']
                s=-config['affine']['subclass_sigma'] if family=='subclass' else config['affine']['broader_sum']
                b=(s+d)/2; c=(s-d)/2
                assert is_admissible(*(sp.Rational(str(q)) for q in [a,b,c]))
                base=moments['product'].to_numpy()+a*moment_b+s*moments.symmetric.to_numpy()/2
                f=moments.copy()
                for side,sign in [('forward',1),('reverse',-1)]:
                    f['support_'+side]=base+sign*d*f.gamma/2
                    f['confidence_'+side]=np.divide(f['support_'+side],f['coverage_'+side],
                        out=np.full(len(f),np.nan),where=f['coverage_'+side]>0)
                scores[d]=f
                summary.append(dict(dataset=dataset,family=family,a=a,b=b,c=c,d=d,D=abs(d)/140,
                                    retained=int(flags(f,config).sum()),
                                    absolute_gap_median=float(np.median(abs(d*f.gamma))),
                                    support_median=float(np.median(base))))
                # Independent recordwise check, fixed first/middle/last pairs.
                for k in sorted({0,len(f)//2,len(f)-1}):
                    r=f.iloc[k]; xa=memberships[r.item_a]; yc=memberships[r.item_c]
                    for side,left,right in [('forward',xa,yc),('reverse',yc,xa)]:
                        direct=np.mean(left*right+boundary_factor(left,right)*(a+b*left+c*right))
                        diagnostics.append(abs(direct-r['support_'+side]))
            groups[family]=scores
            reference=flags(scores[0],config)
            for d in [-.25,.25]:
                target=flags(scores[d],config)
                contrasts.append(dict(dataset=dataset,comparison=family,d=d,
                    n_reference=int(reference.sum()),n_target=int(target.sum()),
                    added=int((~reference&target).sum()),removed=int((reference&~target).sum()),
                    jaccard=overlap(reference,target)))
        for d in config['affine']['differences']:
            left,right=groups['subclass'][d],groups['broader'][d]
            gap_error=np.max(abs((left.support_forward-left.support_reverse)-(right.support_forward-right.support_reverse)))
            assert gap_error<1e-12
            f0,f1=flags(left,config),flags(right,config)
            contrasts.append(dict(dataset=dataset,comparison='broader-vs-subclass',d=d,
                n_reference=int(f0.sum()),n_target=int(f1.sum()),added=int((~f0&f1).sum()),
                removed=int((f0&~f1).sum()),jaccard=overlap(f0,f1)))
            for k,side in zip(*np.where(f0!=f1)):
                r=left.iloc[k]; s=right.iloc[k]; suffix=['forward','reverse'][side]
                changed.append(dict(dataset=dataset,d=d,item_a=r.item_a,item_c=r.item_c,direction=suffix,
                    support_subclass=r['support_'+suffix],support_broader=s['support_'+suffix],
                    confidence_subclass=r['confidence_'+suffix],confidence_broader=s['confidence_'+suffix],
                    selected_subclass=bool(f0[k,side]),selected_broader=bool(f1[k,side])))
        print('affine',dataset,flush=True)
    assert max(diagnostics)<1e-12
    pd.DataFrame(summary).to_csv(OUT/'affine_summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'affine_contrasts.csv',index=False)
    pd.DataFrame(changed).to_csv(OUT/'affine_changed_rules.csv',index=False)
    save_json('affine_validation.json',dict(scalar_checks=len(diagnostics),max_error=max(diagnostics),
             coefficients_verified_exactly=True,matched_gap_tolerance=1e-12))

def selection(config: dict,tabular: dict) -> None:
    params=config['selection']; records=[]; decisions=[]; outer_records=[]
    for dataset in tabular['datasets']:
        data=numeric_data(dataset,tabular)
        for seed in config['seeds']:
            order=np.random.default_rng(seed).permutation(len(data)); cut=int(.7*len(data))
            dev,test=data.iloc[order[:cut]],data.iloc[order[cut:]]
            inner=[]
            for repeat in range(params['inner_repeats']):
                inner_seed=seed+10_000*(repeat+1)
                shuffled=np.random.default_rng(inner_seed).permutation(len(dev))
                k=int(params['inner_training_fraction']*len(dev))
                fit,validation=dev.iloc[shuffled[:k]],dev.iloc[shuffled[k:]]
                _,mt=fuzzify_our_main(fit,fit,3); _,mv=fuzzify_our_main(fit,validation,3)
                a,b=pair_moments(mt),pair_moments(mv)
                sets_a={d:keyed_flags(score_pairs(a,1,d),config) for d in params['deltas']}
                sets_b={d:keyed_flags(score_pairs(b,1,d),config) for d in params['deltas']}
                minimum=max(1,int(np.ceil(params['minimum_rules_fraction_of_symmetric']*len(sets_a[0]))))
                for d in params['deltas']:
                    valid=len(sets_a[d])>=minimum and len(sets_b[d])>0
                    r=dict(dataset=dataset,seed=seed,repeat=repeat,inner_seed=inner_seed,delta=d,
                           training_retained=len(sets_a[d]),validation_retained=len(sets_b[d]),
                           minimum_training_rules=minimum,eligible=valid,
                           jaccard=set_overlap(sets_a[d],sets_b[d]))
                    records.append(r); inner.append(r)
            grouped=pd.DataFrame(inner).groupby('delta').agg(objective=('jaccard','mean'),eligible=('eligible','all'))
            candidates=grouped[grouped.eligible]
            if candidates.empty:
                chosen=0.; fallback=True
            else:
                best=candidates.objective.max()
                near=candidates[candidates.objective>=best-params['tie_tolerance']]
                chosen=min(near.index,key=lambda d:(abs(d),d)); fallback=False
            decisions.append(dict(dataset=dataset,seed=seed,chosen_delta=chosen,
                objective=float(grouped.loc[chosen,'objective']),symmetric_objective=float(grouped.loc[0,'objective']),fallback=fallback))
            _,md=fuzzify_our_main(dev,dev,3); _,me=fuzzify_our_main(dev,test,3)
            ad,ae=pair_moments(md),pair_moments(me)
            for name,d in [('selected',chosen),('symmetric',0.)]:
                rd=keyed_flags(score_pairs(ad,1,d),config); re=keyed_flags(score_pairs(ae,1,d),config)
                outer_records.append(dict(dataset=dataset,seed=seed,strategy=name,delta=d,
                    development_retained=len(rd),evaluation_retained=len(re),intersection=len(rd&re),
                    jaccard=set_overlap(rd,re),reproduction_fraction=len(rd&re)/len(rd) if rd else np.nan))
            np.savez(OUT/f'selection_{dataset}_{seed}_split.npz',development=order[:cut],evaluation=order[cut:])
            print('selection',dataset,seed,chosen,flush=True)
    pd.DataFrame(records).to_csv(OUT/'selection_inner.csv',index=False)
    pd.DataFrame(decisions).to_csv(OUT/'selection_choices.csv',index=False)
    pd.DataFrame(outer_records).to_csv(OUT/'selection_outer.csv',index=False)

def temporal(config: dict) -> None:
    cfg=load_yaml(ROOT/'experiments/configs/synthetic_dynamic.yaml'); dcfg=config['dynamic']
    module=importlib.import_module('13_dynamic_operator_increment_audit')
    raw=pd.read_parquet(ROOT/'experiments/outputs/raw_results/synthetic_dynamic_raw.parquet')
    rows=[]; errors=[]
    for null in [False,True]:
        selected=raw[(raw.protocol=='process_order')&(raw.operator=='proposed')
                     &(raw.independent_null==null)&(raw.mechanism==dcfg['mechanism'])
                     &(raw.h_true==dcfg['true_lag'])&(raw.h_eval==dcfg['evaluation_lag'])
                     &(raw['lambda']==dcfg['coupling'])&(raw.noise==dcfg['noise'])
                     &(raw.sigma==dcfg['sigma'])&(raw.delta==dcfg['delta'])]
        assert len(selected)==selected.seed.nunique()==50
        for r in selected.itertuples():
            data=dynamic_process(np.random.default_rng(int(r.seed)),phi=cfg['phi'],rho=cfg['rho'],
                lag=dcfg['true_lag'],coupling=dcfg['coupling'],noise=dcfg['noise'],
                length=cfg['length'],burn_in=cfg['burn_in'],function=dcfg['mechanism'],independent_null=null)
            x,y=module._memberships(data)
            base,symmetric,lam=module._process_terms(x,y,dcfg['evaluation_lag'])
            psi0=base-dcfg['sigma']*symmetric/2; omega=dcfg['delta']*lam
            errors.append(abs((psi0+omega).mean()-r.Psi))
            for i in range(3):
                for j in range(3):
                    rows.append(dict(null=null,seed=r.seed,label_x=i,label_y=j,
                                     Psi0=psi0[i,j],Omega=omega[i,j],Psi=(psi0+omega)[i,j]))
    assert max(errors)<1e-12
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'temporal_trajectories.csv',index=False)
    summaries=[]
    for (null,i,j),b in frame.groupby(['null','label_x','label_y']):
        for metric in ['Psi0','Omega','Psi']:
            a=b[metric]; summaries.append(dict(null=null,label_x=i,label_y=j,metric=metric,n=len(a),
              median=a.median(),q10=a.quantile(.1),q90=a.quantile(.9),mean=a.mean(),std=a.std(),
              positive=int((a>1e-12).sum()),negative=int((a< -1e-12).sum())))
    pd.DataFrame(summaries).to_csv(OUT/'temporal_summary.csv',index=False)
    save_json('temporal_validation.json',dict(independent_trajectories=100,label_pair_records=len(frame),
             max_saved_aggregate_error=max(errors),statistical_unit='one complete trajectory',configuration=dcfg))

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--part',choices=['examples','affine','selection','temporal','all'],default='all')
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    cfg=load_yaml(CONFIG); tabular=load_yaml(ROOT/'experiments/configs/tabular.yaml')
    save_json('protocol.json',dict(configuration=cfg,config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
              selection_goal='development/validation retained-set Jaccard subject to rule-count constraint',
              outer_evaluation_used_for_selection=False))
    for part in ['examples','affine','selection','temporal']:
        if args.part in [part,'all']:
            if part=='examples': examples()
            elif part=='affine': affine(cfg,tabular)
            elif part=='selection': selection(cfg,tabular)
            else: temporal(cfg)

if __name__=='__main__': main()
