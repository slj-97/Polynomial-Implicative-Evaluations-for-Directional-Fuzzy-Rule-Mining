"""Reviewer 2 Q8: record/label/length axes and real data in isolated workers.

Ordinary wall times use the unchanged miner. A separate diagnostic pass adds
phase timers and must reproduce every count and retained key of that miner.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

from _common import ROOT
from experiments.src.fuzzification import RobustScaler, triangular_memberships, fuzzify_our_main
from experiments.src.operators import eval_poly,t_lukasiewicz
from experiments.src.reproducibility import load_yaml
from experiments.src.rule_mining import mine_rules,retained_rule_set

OUT=ROOT/'experiments/outputs/revision_r2/computation'
CONFIG=ROOT/'experiments/configs/revision_r2.yaml'
MODES=['exhaustive','coverage_only','coverage_support']

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def name(case: str,seed: int,support: float,mode: str) -> str:
    return f'{case}_{seed}_{support:g}_{mode}.json'

def profile_miner(memberships: dict,kwargs: dict):
    import experiments.src.rule_mining as module
    clocks=dict(candidate=0.,aggregation=0.,coverage=0.,support=0.,confidence=0.)
    ns=dict(vars(module)); ns.update(_phase=clocks,_clock=time.perf_counter)
    for function,key in [('aggregate_memberships','aggregation'),('fuzzy_coverage','coverage'),
                         ('fuzzy_support','support'),('fuzzy_confidence','confidence')]:
        original=ns[function]
        def timed(*args,_function=original,_key=key,**kw):
            started=time.perf_counter(); result=_function(*args,**kw)
            clocks[_key]+=time.perf_counter()-started
            return result
        ns[function]=timed
    source=inspect.getsource(mine_rules).replace('def mine_rules(','def profiled_mine_rules(',1)
    old='''            used_variables = {item_variable(item) for item in prefix}
            for index in range(start, len(candidates)):
                item = candidates[index]
                if item_variable(item) in used_variables:
                    continue
                visit(prefix + (item,), index + 1)'''
    new='''            _started = _clock()
            used_variables = {item_variable(item) for item in prefix}
            _phase['candidate'] += _clock() - _started
            for index in range(start, len(candidates)):
                _started = _clock()
                item = candidates[index]
                if item_variable(item) in used_variables:
                    _phase['candidate'] += _clock() - _started
                    continue
                _next = prefix + (item,)
                _phase['candidate'] += _clock() - _started
                visit(_next, index + 1)'''
    assert source.count(old)==1
    source=source.replace(old,new)
    oldline='        candidates = [item for item in items if item_variable(item) != item_variable(consequent)]'
    assert source.count(oldline)==1
    source=source.replace(oldline,'        _started = _clock()\n'+oldline+"\n        _phase['candidate'] += _clock() - _started")
    exec(compile(source,'<diagnostic-miner>','exec'),ns)
    started=time.perf_counter(); result=ns['profiled_mine_rules'](memberships,**kwargs)
    total=time.perf_counter()-started
    clocks['instrumented_total']=total
    clocks['other']=total-sum(clocks[k] for k in ['candidate','aggregation','coverage','support','confidence'])
    return result,clocks

def worker(case: dict,seed: int,support: float,mode: str,cfg: dict) -> None:
    if case['dataset']=='synthetic':
        # Every record-count case shares one training prefix and nested test prefixes.
        rng=np.random.default_rng(seed); rows=4360
        shared=rng.normal(size=(rows,1))
        data=pd.DataFrame(.5*shared+rng.normal(size=(rows,5)),columns=list('ABCDE'))
        train=data.iloc[:360]; test=data.iloc[360:360+case['evaluation_rows']]
        memberships=triangular_memberships(RobustScaler.fit(train).transform(test),case['labels'])
    else:
        module=__import__('30_reviewer2_analysis')
        data=module.numeric_data(case['dataset'],load_yaml(ROOT/'experiments/configs/tabular.yaml'))
        order=np.random.default_rng(seed).permutation(len(data)); cut=int(.7*len(data))
        train,test=data.iloc[order[:cut]],data.iloc[order[cut:]]
        _,memberships=fuzzify_our_main(train,test,case['labels'])
    process=psutil.Process(); before=process.memory_info()
    setting=cfg['computational']
    kwargs=dict(tnorm=t_lukasiewicz,evaluation=lambda a,c:eval_poly(a,c,setting['sigma'],setting['delta']),
                thresholds=dict(coverage=.01,support=support,confidence=.5),max_antecedent_size=case['K'],
                pruning_mode=mode,mtc_valid=True,protocol='reviewer2-computational-axes')
    started=time.perf_counter(); outcome=mine_rules(memberships,**kwargs)
    elapsed=time.perf_counter()-started
    peak=process.memory_info().peak_wset
    keys=sorted(retained_rule_set(outcome))
    diagnostic={}
    if seed==cfg['seeds'][0]:
        diagnostic_outcome,diagnostic=profile_miner(memberships,kwargs)
        assert sorted(retained_rule_set(diagnostic_outcome))==keys
        for field in ['generated_candidates','scored_candidates','coverage_pruned_branches','support_pruned_branches']:
            assert getattr(diagnostic_outcome,field)==getattr(outcome,field)
    result=dict(case=case['name'],dataset=case['dataset'],seed=seed,support_threshold=support,mode=mode,
                labels=case['labels'],K=case['K'],variables=len(train.columns),training_rows=len(train),evaluation_rows=len(test),
                generated=outcome.generated_candidates,scored=outcome.scored_candidates,retained=len(keys),
                elapsed_seconds=elapsed,peak_mib=peak/2**20,pre_search_peak_mib=before.peak_wset/2**20,
                memberships_sha256=hashlib.sha256(np.column_stack(list(memberships.values())).tobytes()).hexdigest(),
                source_sha256=sha(ROOT/'experiments/src/rule_mining.py'),config_sha256=sha(CONFIG),
                retained_keys=keys,diagnostic=diagnostic)
    (OUT/'workers'/name(case['name'],seed,support,mode)).write_text(json.dumps(result),encoding='utf-8')

def summarize(cfg: dict) -> None:
    rows=[]; profiles=[]
    for case in cfg['computational']['cases']:
        for support in cfg['computational']['support_thresholds']:
            for seed in cfg['seeds']:
                group=[json.loads((OUT/'workers'/name(case['name'],seed,support,mode)).read_text()) for mode in MODES]
                base,coverage=group[:2]
                assert len({r['memberships_sha256'] for r in group})==1
                for original in group:
                    assert original['retained_keys']==base['retained_keys']
                    row={k:v for k,v in original.items() if k not in ['retained_keys','diagnostic']}
                    row.update(retained_set_equal=True,incremental_reduction=1-row['scored']/coverage['scored'],
                               exhaustive_to_joint_ratio=base['elapsed_seconds']/group[2]['elapsed_seconds'])
                    rows.append(row)
                    if original['diagnostic']:
                        profiles.append(dict(case=case['name'],support_threshold=support,mode=row['mode'],
                                             **original['diagnostic']))
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'measurements.csv',index=False)
    pd.DataFrame(profiles).to_csv(OUT/'phase_diagnostics.csv',index=False)
    records=[]
    for (case,support,mode),b in frame.groupby(['case','support_threshold','mode'],sort=False):
        r=dict(case=case,support_threshold=support,mode=mode,seeds=len(b))
        for metric in ['generated','scored','retained','elapsed_seconds','peak_mib','incremental_reduction','exhaustive_to_joint_ratio']:
            for suffix,value in [('median',b[metric].median()),('q25',b[metric].quantile(.25)),('q75',b[metric].quantile(.75))]:
                r[metric+'_'+suffix]=value
        records.append(r)
    pd.DataFrame(records).to_csv(OUT/'summary.csv',index=False)
    (OUT/'validation.json').write_text(json.dumps(dict(searches=len(frame),matched_comparisons=len(frame)//3,
        all_retained_sets_equal=True,diagnostic_passes=len(profiles),diagnostics_counts_and_keys_equal=True,
        source_sha256=sha(ROOT/'experiments/src/rule_mining.py'),timing='unchanged miner incl. output sorting; excludes preprocessing',
        memory='lifetime worker peak through ordinary search; before diagnostics and serialization',
        profiling='separate timer-instrumented pass at first seed; candidate timer covers consequent filtering, variable checks and prefix construction; other includes bookkeeping, output and instrumentation',
        execution='fresh sequential processes; one BLAS thread; cyclic mode order'),indent=2),encoding='utf-8')
    print(f'Verified {len(frame)//3} matched comparisons / {len(frame)} searches; {len(profiles)} diagnostic passes.',flush=True)

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--case');parser.add_argument('--seed',type=int)
    parser.add_argument('--support',type=float);parser.add_argument('--mode');parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args();cfg=load_yaml(CONFIG);(OUT/'workers').mkdir(parents=True,exist_ok=True)
    if args.case:
        worker(next(c for c in cfg['computational']['cases'] if c['name']==args.case),args.seed,args.support,args.mode,cfg);return
    if args.summarize_only: summarize(cfg);return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    completed=0
    for i,case in enumerate(cfg['computational']['cases']):
        for support in cfg['computational']['support_thresholds']:
            for j,seed in enumerate(cfg['seeds']):
                offset=(i+j)%3
                for mode in MODES[offset:]+MODES[:offset]:
                    path=OUT/'workers'/name(case['name'],seed,support,mode)
                    if path.exists():
                        previous=json.loads(path.read_text());assert previous['config_sha256']==sha(CONFIG)
                        assert previous['source_sha256']==sha(ROOT/'experiments/src/rule_mining.py')
                    else:
                        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--case',case['name'],'--seed',str(seed),
                                        '--support',str(support),'--mode',mode],cwd=ROOT,env=env,check=True)
                    completed+=1
                    if completed%15==0:print(f'{completed}/210 searches completed',flush=True)
    summarize(cfg)

if __name__=='__main__': main()
