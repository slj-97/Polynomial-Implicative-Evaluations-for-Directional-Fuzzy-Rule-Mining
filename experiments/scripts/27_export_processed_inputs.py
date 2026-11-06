"""Rebuild exact fixed-split inputs from the six pinned CSVs.

--fetch retrieves missing CSVs from their recorded upstream commit and checks
their hashes. --expected verifies the derived array hashes against a manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
from _common import ROOT
from experiments.src.fuzzification import fuzzify_our_main
from experiments.src.reproducibility import load_yaml

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/outputs/revision_r1_q6/processed')
    parser.add_argument('--expected',type=Path)
    args=parser.parse_args(); args.output.mkdir(exist_ok=True,parents=True)
    cfg=load_yaml(ROOT/'experiments/configs/tabular.yaml')
    sources=json.loads((ROOT/'data/raw/tabular_firm/manifest.json').read_text(encoding='utf-8'))
    records=[]
    for ds in cfg['datasets']:
        source=next(r for r in sources if r['dataset']==ds)
        path=ROOT/source['path']
        if not path.exists() and args.fetch:
            url=f"https://raw.githubusercontent.com/rferper/FIRM-paper/{source['commit']}/assets/{ds}.csv"
            payload=urllib.request.urlopen(url,timeout=60).read()
            assert hashlib.sha256(payload).hexdigest()==source['sha256'],ds
            path.parent.mkdir(exist_ok=True,parents=True); path.write_bytes(payload)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256'],ds
        raw=pd.read_csv(path); raw.columns=raw.columns.str.strip()
        numeric=raw.drop(columns=[cfg['targets'][ds]]).select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy()).all()
        ids=np.load(ROOT/f'experiments/outputs/tabular/{ds}_split.npz')
        order=np.random.default_rng(cfg['seed']).permutation(len(raw)); cut=int(len(raw)*cfg['training_fraction'])
        assert np.array_equal(order[:cut],ids['training']) and np.array_equal(order[cut:],ids['test'])
        train=numeric.iloc[ids['training']]; test=numeric.iloc[ids['test']]
        meta=dict(dataset=ds,source_commit=source['commit'],source_sha256=source['sha256'],
            rows=len(raw),columns=list(numeric.columns),target_removed=cfg['targets'][ds],
            excluded_columns=[c for c in raw if c not in numeric],seed=cfg['seed'],
            training_index_sha256=digest(ids['training']),evaluation_index_sha256=digest(ids['test']),
            training_array_sha256=digest(train.to_numpy(dtype=np.float64)),evaluation_array_sha256=digest(test.to_numpy(dtype=np.float64)),
            missing_value_policy='reject nonfinite predictors; no imputation',partitions=[])
        np.savez_compressed(args.output/(ds+'_numeric.npz'),training=train.to_numpy(dtype=np.float64),evaluation=test.to_numpy(dtype=np.float64),training_indices=ids['training'],evaluation_indices=ids['test'],columns=np.asarray(numeric.columns,dtype=str))
        for labels in cfg['label_counts']:
            scaler,members=fuzzify_our_main(train,test,labels)
            array=np.column_stack(list(members.values()))
            np.savez_compressed(args.output/f'{ds}_{labels}_memberships.npz',values=array,items=np.asarray(list(members),dtype=str))
            meta['partitions'].append(dict(labels=labels,shape=list(array.shape),items=list(members),array_sha256=digest(array),
                lower=scaler.lower.to_dict(),upper=scaler.upper.to_dict(),active_continuous_columns=list(scaler.active_columns),
                binary_columns=[c for c in train if np.isin(train[c].unique(),[0,1]).all()],dtype='float64'))
        records.append(meta)
    if args.expected:
        assert records==json.loads(args.expected.read_text(encoding='utf-8')),'Processed inputs differ from frozen manifest'
    (args.output/'processed_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Verified six pinned datasets, saved split indices, and 12 deterministic membership matrices.')

if __name__=='__main__': main()
