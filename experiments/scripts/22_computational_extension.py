"""Q5 computational extension: isolated sequential workers, frozen search code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from math import comb

import numpy as np
import pandas as pd
import psutil

from _common import ROOT
from experiments.src.fuzzification import RobustScaler, triangular_memberships
from experiments.src.operators import eval_poly, t_lukasiewicz
from experiments.src.reproducibility import load_yaml
from experiments.src.rule_mining import mine_rules, retained_rule_set

OUT = ROOT / "experiments/outputs/revision_r1_q5"
CONFIG_PATH = ROOT / "experiments/configs/revision_r1_q5.yaml"
ENGINE = "OUR-MAIN"
PROTOCOL = "computational-extension-isolated-workers"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stem(seed: int, variables: int, delta: float, support: float, mode: str) -> str:
    return f"s{seed}_v{variables}_d{delta:g}_t{support:g}_{mode}"


def worker(args: argparse.Namespace, config: dict) -> None:
    rng = np.random.default_rng(args.seed)
    shared = rng.normal(size=(config["rows"], 1))
    data = pd.DataFrame(.5*shared+rng.normal(size=(config["rows"], args.variables)),
                        columns=[chr(ord("A")+i) for i in range(args.variables)])
    cut=config["training_rows"]
    scaler=RobustScaler.fit(data.iloc[:cut])
    memberships=triangular_memberships(scaler.transform(data.iloc[cut:]), labels=config["labels"])
    process=psutil.Process()
    memory_before=process.memory_info()
    started=time.perf_counter()
    outcome=mine_rules(memberships, tnorm=t_lukasiewicz,
                       evaluation=lambda a,c:eval_poly(a,c,config["sigma"],args.delta),
                       thresholds=dict(coverage=config["coverage_threshold"], support=args.support,
                                       confidence=config["confidence_threshold"]),
                       max_antecedent_size=config["max_antecedent_size"], pruning_mode=args.mode,
                       mtc_valid=True, protocol=PROTOCOL)
    elapsed=time.perf_counter()-started
    memory_after=process.memory_info()
    # Snapshot before constructing comparison sets or serializing anything.
    peak=int(memory_after.peak_wset) if hasattr(memory_after,"peak_wset") else None
    if peak is None:
        raise RuntimeError("This protocol requires the Windows peak working-set counter.")
    rules=sorted(retained_rule_set(outcome))
    expected=config["labels"]*args.variables*sum(comb(args.variables-1,k)*config["labels"]**k
              for k in range(1, config["max_antecedent_size"]+1))
    if args.mode=="exhaustive":
        assert outcome.scored_candidates==expected and outcome.generated_candidates==expected
    result=dict(engine=ENGINE,protocol=PROTOCOL,seed=args.seed,variables=args.variables,
                sigma=config["sigma"],delta=args.delta,support_threshold=args.support,mode=args.mode,
                K=config["max_antecedent_size"],training_rows=cut,evaluation_rows=len(data)-cut,
                generated=outcome.generated_candidates,scored=outcome.scored_candidates,
                coverage_rejections=outcome.coverage_pruned_branches,support_rejections=outcome.support_pruned_branches,
                retained=len(rules),expected_exhaustive=expected,elapsed_seconds=elapsed,
                peak_working_set_bytes=peak,rss_before_bytes=int(memory_before.rss),
                peak_before_bytes=int(memory_before.peak_wset),peak_after_search_bytes=peak,
                data_sha256=hashlib.sha256(data.to_numpy().tobytes()).hexdigest(),
                memberships_sha256=hashlib.sha256(np.column_stack(list(memberships.values())).tobytes()).hexdigest(),
                source_sha256=sha(ROOT/"experiments/src/rule_mining.py"),config_sha256=sha(CONFIG_PATH),
                retained_keys=rules)
    name=stem(args.seed,args.variables,args.delta,args.support,args.mode)
    (OUT/"workers"/f"{name}.json").write_text(json.dumps(result),encoding="utf-8")


def summarize(config: dict) -> None:
    measurements=[]
    for variables,delta in configurations(config):
        for support in config["support_thresholds"]:
            for seed in config["seeds"]:
                group=[]
                for mode in config["modes"]:
                    path=OUT/"workers"/(stem(seed,variables,delta,support,mode)+".json")
                    group.append(json.loads(path.read_text(encoding="utf-8")))
                reference=next(r for r in group if r["mode"]=="exhaustive")
                coverage=next(r for r in group if r["mode"]=="coverage_only")
                assert len({r["memberships_sha256"] for r in group})==1
                for original in group:
                    row=original.copy()
                    assert row["retained_keys"]==reference["retained_keys"]
                    row["retained_set_equal"]=True
                    row["pruning_ratio"]=1-row["scored"]/reference["scored"]
                    row["time_ratio_to_exhaustive"]=reference["elapsed_seconds"]/row["elapsed_seconds"]
                    row["incremental_scored_reduction_vs_coverage"]=1-row["scored"]/coverage["scored"]
                    row["peak_mib"]=row["peak_working_set_bytes"]/2**20
                    row["rss_before_mib"]=row["rss_before_bytes"]/2**20
                    row.pop("retained_keys")
                    measurements.append(row)
    frame=pd.DataFrame(measurements)
    frame.to_csv(OUT/"measurements.csv",index=False)
    summary=[]
    metrics=["generated","scored","retained","elapsed_seconds","peak_mib","rss_before_mib",
             "pruning_ratio","time_ratio_to_exhaustive","incremental_scored_reduction_vs_coverage"]
    for (variables,delta,support,mode),block in frame.groupby(["variables","delta","support_threshold","mode"],sort=True):
        for metric in metrics:
            values=block[metric]
            summary.append(dict(engine=ENGINE,protocol=PROTOCOL,variables=variables,delta=delta,
                                support_threshold=support,mode=mode,metric=metric,n=len(values),
                                median=float(values.median()),q25=float(values.quantile(.25)),
                                q75=float(values.quantile(.75)),minimum=float(values.min()),maximum=float(values.max())))
    summary=pd.DataFrame(summary)
    summary.to_csv(OUT/"summary.csv",index=False)
    manifest=dict(status="complete",engine=ENGINE,protocol=PROTOCOL,configuration_count=len(configurations(config))*2,
                  matched_comparisons=len(frame)//3,search_runs=len(frame),all_retained_sets_equal=bool(frame.retained_set_equal.all()),
                  seeds=config["seeds"],config_sha256=sha(CONFIG_PATH),source_sha256=sha(ROOT/"experiments/src/rule_mining.py"),
                  script_sha256=sha(Path(__file__)),python=sys.version,numpy=np.__version__,pandas=pd.__version__,
                  psutil=psutil.__version__,platform=platform.platform(),processor=platform.processor(),
                  physical_cores=psutil.cpu_count(logical=False),logical_cores=psutil.cpu_count(),
                  timing=config["timing"],memory=config["memory"],execution="sequential isolated processes; one BLAS thread",
                  scale_variables=config["variable_counts"],scale_delta=config["scale_delta"],
                  parameter_variables=config["parameter_variables"],parameter_deltas=config["parameter_deltas"])
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2),flush=True)


def configurations(config: dict) -> list[tuple[int,float]]:
    return list(dict.fromkeys([(v,config["scale_delta"]) for v in config["variable_counts"]]+
                             [(config["parameter_variables"],d) for d in config["parameter_deltas"]]))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--seed",type=int)
    parser.add_argument("--variables",type=int)
    parser.add_argument("--delta",type=float)
    parser.add_argument("--support",type=float)
    parser.add_argument("--mode")
    parser.add_argument("--summarize-only",action="store_true")
    args=parser.parse_args()
    config=load_yaml(CONFIG_PATH)
    (OUT/"workers").mkdir(parents=True,exist_ok=True)
    if args.worker:
        worker(args,config)
        return
    if args.summarize_only:
        summarize(config)
        return
    (OUT/"protocol.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    child_env=os.environ.copy()
    child_env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    total=len(configurations(config))*len(config["support_thresholds"])*len(config["seeds"])*len(config["modes"])
    completed=0
    for config_index,(variables,delta) in enumerate(configurations(config)):
        for support_index,support in enumerate(config["support_thresholds"]):
            for seed_index,seed in enumerate(config["seeds"]):
                modes=config["modes"]
                offset=(config_index+support_index+seed_index)%len(modes)
                for mode in modes[offset:]+modes[:offset]:
                    name=stem(seed,variables,delta,support,mode)
                    path=OUT/"workers"/(name+".json")
                    if path.exists():
                        previous=json.loads(path.read_text(encoding="utf-8"))
                        assert previous["source_sha256"]==sha(ROOT/"experiments/src/rule_mining.py")
                        assert previous["config_sha256"]==sha(CONFIG_PATH)
                    else:
                        subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--worker", "--seed",str(seed),
                                        "--variables",str(variables),"--delta",str(delta),"--support",str(support),
                                        "--mode",mode],cwd=ROOT,env=child_env,check=True)
                    completed+=1
                    value=json.loads(path.read_text(encoding="utf-8"))
                    print(f"{completed}/{total} {name}: {value['elapsed_seconds']:.3f}s; peak {value['peak_working_set_bytes']/2**20:.2f} MiB",flush=True)
    summarize(config)


if __name__=="__main__":
    main()
