"""Reconstruct the missing nonzero-delta AR-null summary from frozen runs.

Uses the original audit's record-level calculation and checks reconstruction
against saved supports. No new seeds or experimental configurations are added.
"""
import importlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from _common import ROOT
from experiments.src.reproducibility import load_yaml

audit=importlib.import_module('13_dynamic_operator_increment_audit')

def main():
    out=ROOT/'experiments/outputs/revision_r1_q6'
    out.mkdir(exist_ok=True,parents=True)
    raw=pd.read_parquet(ROOT/'experiments/outputs/raw_results/synthetic_dynamic_raw.parquet')
    p=raw[(raw.protocol=='process_order')&(raw.operator=='proposed')&raw.independent_null].copy()
    cols=['independent_null','seed','mechanism','h_true','lambda','noise']
    p=p.sort_values(cols+['h_eval','sigma','delta']).reset_index(drop=True)
    cfg=load_yaml(ROOT/'experiments/configs/synthetic_dynamic.yaml')
    tasks=[]
    for _,rows in p.groupby(cols,sort=False,dropna=False):
        specs=[]
        for lag,lr in rows.groupby('h_eval',sort=False):
            specs.append((int(lag),[(r.index.to_numpy(dtype=int),float(s),r.delta.to_numpy(),r.Psi.to_numpy()) for s,r in lr.groupby('sigma',sort=False)]))
        tasks.append((rows.iloc[0][cols].to_dict(),specs,cfg))
    print(f'Reconstructing {len(tasks)} existing AR-null trajectories',flush=True)
    omega=np.empty(len(p)); error=np.empty(len(p))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(audit._reconstruct_dataset,tasks,chunksize=8),1):
            for positions,_,o,_,e in result: omega[positions]=o.mean(axis=1); error[positions]=e
            if i%500==0: print(f'{i}/{len(tasks)}',flush=True)
    assert error.max()<1e-12
    p['abs_Omega']=np.abs(omega)
    keys=['independent_null','mechanism','h_true','h_eval','lambda','noise','sigma','delta']
    seed=p.groupby(keys,dropna=False).abs_Omega.median().reset_index()
    seed.to_csv(out/'dynamic_null_seed_medians.csv',index=False)
    values=seed[seed.delta!=0].abs_Omega
    original=pd.read_csv(ROOT/'experiments/outputs/summaries/dynamic_null_audit.csv')
    mask=(original.condition=='independent_AR_null')&(original['filter']=='delta != 0')&(original.metric=='abs_Omega')
    for k,v in audit._summary(values).items(): original.loc[mask,k]=v
    original.to_csv(out/'dynamic_null_completed.csv',index=False)
    print(f'Null median={values.median():.9e}, Q95={values.quantile(.95):.9e}; reconstruction error={error.max():.3e}')

if __name__=='__main__': main()
