"""Audit saved computational results and collect interleaved timing follow-ups.

The original worker measurements and miner remain unchanged. All new ordinary
runs use the same input generation and measurement boundary as script 22.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

from _common import ROOT
from experiments.src.fuzzification import RobustScaler, triangular_memberships
from experiments.src.operators import eval_poly, t_lukasiewicz
from experiments.src.reproducibility import load_yaml
from experiments.src.rule_mining import mine_rules, retained_rule_set

OUT = ROOT / "experiments/outputs/computational_followup"
OLD = ROOT / "experiments/outputs/revision_r1_q5"
CONFIG = ROOT / "experiments/configs/revision_r1_q5.yaml"
MODES = ["exhaustive", "coverage_only", "coverage_support"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token(seed: int, variables: int, delta: float, support: float, mode: str) -> str:
    return f"s{seed}_v{variables}_d{delta:g}_t{support:g}_{mode}"


def audit() -> None:
    frame = pd.read_csv(OLD / "measurements.csv")
    checks = []
    for keys, block in frame.groupby(["seed", "variables", "delta", "support_threshold"]):
        workers = [json.loads((OLD / "workers" / (token(*keys, mode) + ".json")).read_text())
                   for mode in MODES]
        assert len({w["memberships_sha256"] for w in workers}) == 1
        assert all(w["retained_keys"] == workers[0]["retained_keys"] for w in workers)
        assert all(w["source_sha256"] == sha(ROOT / "experiments/src/rule_mining.py") for w in workers)
        checks.append(dict(seed=int(keys[0]), variables=int(keys[1]), delta=float(keys[2]),
                           support_threshold=float(keys[3]), keys_equal=True, inputs_equal=True))
    pd.DataFrame(checks).to_csv(OUT / "original_checks.csv", index=False)
    summaries = []
    for (v, d, t, mode), b in frame.groupby(["variables", "delta", "support_threshold", "mode"]):
        summaries.append(dict(variables=int(v), delta=d, support_threshold=t, mode=mode,
                              seeds=len(b), generated_unique=b.generated.nunique(),
                              scored_unique=b.scored.nunique(), median=b.elapsed_seconds.median(),
                              q25=b.elapsed_seconds.quantile(.25), q75=b.elapsed_seconds.quantile(.75),
                              minimum=b.elapsed_seconds.min(), maximum=b.elapsed_seconds.max()))
    pd.DataFrame(summaries).to_csv(OUT / "original_summary.csv", index=False)
    print(f"Verified original inputs and retained keys for {len(checks)} matched comparisons.", flush=True)


def worker(args: argparse.Namespace, config: dict) -> None:
    process = psutil.Process()
    if args.batch in {"controlled_pilot", "controlled"}:
        process.cpu_affinity([0])
    rng = np.random.default_rng(args.seed)
    shared = rng.normal(size=(config["rows"], 1))
    data = pd.DataFrame(.5 * shared + rng.normal(size=(config["rows"], args.variables)),
                        columns=[chr(ord("A") + i) for i in range(args.variables)])
    cut = config["training_rows"]
    memberships = triangular_memberships(
        RobustScaler.fit(data.iloc[:cut]).transform(data.iloc[cut:]), config["labels"])
    memory_before = process.memory_info()
    cpu_before = process.cpu_times()
    started = time.perf_counter()
    outcome = mine_rules(
        memberships, tnorm=t_lukasiewicz,
        evaluation=lambda a, c: eval_poly(a, c, config["sigma"], args.delta),
        thresholds=dict(coverage=config["coverage_threshold"], support=args.support,
                        confidence=config["confidence_threshold"]),
        max_antecedent_size=config["max_antecedent_size"], pruning_mode=args.mode,
        mtc_valid=True, protocol="computational-followup-interleaved")
    elapsed = time.perf_counter() - started
    cpu_after = process.cpu_times()
    memory_after = process.memory_info()
    keys = sorted(retained_rule_set(outcome))
    name = token(args.seed, args.variables, args.delta, args.support, args.mode)
    original = json.loads((OLD / "workers" / (name + ".json")).read_text())
    membership_hash = hashlib.sha256(np.column_stack(list(memberships.values())).tobytes()).hexdigest()
    assert membership_hash == original["memberships_sha256"]
    assert keys == [tuple(k) for k in original["retained_keys"]]
    assert outcome.generated_candidates == original["generated"]
    assert outcome.scored_candidates == original["scored"]
    assert outcome.coverage_pruned_branches == original["coverage_rejections"]
    assert outcome.support_pruned_branches == original["support_rejections"]
    result = {k: v for k, v in original.items() if k != "retained_keys"}
    result.update(protocol="computational-followup-interleaved", batch=args.batch,
                  repetition=args.repetition, elapsed_seconds=elapsed,
                  cpu_seconds=(cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system),
                  peak_working_set_bytes=int(memory_after.peak_wset),
                  peak_after_search_bytes=int(memory_after.peak_wset),
                  rss_before_bytes=int(memory_before.rss), peak_before_bytes=int(memory_before.peak_wset),
                  retained_keys=keys, retained_set_equal=True, counts_match_original=True,
                  original_elapsed_seconds=original["elapsed_seconds"],
                  measured_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    result["cpu_affinity"] = process.cpu_affinity()
    result["worker_script_sha256"] = sha(Path(__file__))
    if args.profile:
        diagnostic = importlib.import_module("31_reviewer2_computation")
        kwargs = dict(tnorm=t_lukasiewicz,
                      evaluation=lambda a, c: eval_poly(a, c, config["sigma"], args.delta),
                      thresholds=dict(coverage=config["coverage_threshold"], support=args.support,
                                      confidence=config["confidence_threshold"]),
                      max_antecedent_size=config["max_antecedent_size"], pruning_mode=args.mode,
                      mtc_valid=True, protocol="computational-followup-profile")
        profiled, phases = diagnostic.profile_miner(memberships, kwargs)
        assert sorted(retained_rule_set(profiled)) == keys
        for field in ["generated_candidates", "scored_candidates", "coverage_pruned_branches", "support_pruned_branches"]:
            assert getattr(profiled, field) == getattr(outcome, field)
        result["separate_diagnostic"] = phases
    dest = OUT / args.batch / "workers"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / (name + f"_r{args.repetition}.json")).write_text(json.dumps(result), encoding="utf-8")


def summarize(batch: str) -> None:
    workers = sorted((OUT / batch / "workers").glob("*.json"))
    rows = []
    for path in workers:
        r = json.loads(path.read_text())
        r.pop("retained_keys")
        r.pop("separate_diagnostic", None)
        r["peak_mib"] = r["peak_working_set_bytes"] / 2**20
        r["rss_before_mib"] = r["rss_before_bytes"] / 2**20
        rows.append(r)
    frame = pd.DataFrame(rows)
    for _, block in frame.groupby(["seed", "variables", "delta", "support_threshold", "repetition"]):
        base = block[block["mode"] == "exhaustive"].iloc[0]
        coverage = block[block["mode"] == "coverage_only"].iloc[0]
        frame.loc[block.index, "time_ratio_to_exhaustive"] = base.elapsed_seconds / block.elapsed_seconds
        frame.loc[block.index, "incremental_scored_reduction_vs_coverage"] = 1 - block.scored / coverage.scored
    frame.to_csv(OUT / batch / "measurements.csv", index=False)
    results = []
    for (v, d, t, mode), b in frame.groupby(["variables", "delta", "support_threshold", "mode"]):
        r = dict(variables=int(v), delta=d, support_threshold=t, mode=mode, runs=len(b),
                 seeds=int(b.seed.nunique()))
        for metric in ["elapsed_seconds", "cpu_seconds", "peak_mib", "time_ratio_to_exhaustive"]:
            for suffix, value in [("median", b[metric].median()), ("q25", b[metric].quantile(.25)),
                                  ("q75", b[metric].quantile(.75)), ("minimum", b[metric].min())]:
                r[metric + "_" + suffix] = float(value)
        results.append(r)
    pd.DataFrame(results).to_csv(OUT / batch / "summary.csv", index=False)
    (OUT / batch / "manifest.json").write_text(json.dumps(dict(
        batch=batch, searches=len(frame), all_counts_and_keys_match_original=True,
        source_sha256=sha(ROOT / "experiments/src/rule_mining.py"), config_sha256=sha(CONFIG),
        script_sha256=sha(Path(__file__)), python=sys.version, numpy=np.__version__, pandas=pd.__version__,
        platform=platform.platform(), psutil=psutil.__version__,
        timing="unchanged miner including output sort; imports, preprocessing, comparison and JSON excluded",
        memory="lifetime worker peak through ordinary search; diagnostics occur afterwards",
        execution="fresh sequential workers; one BLAS thread; seeded shuffle of configurations per round",
        fixed_logical_processor=0 if batch in {"controlled_pilot", "controlled"} else None,
        seeds=sorted(int(s) for s in frame.seed.unique()), shuffle_seed=20261024), indent=2), encoding="utf-8")
    print(f"Summarized {len(frame)} {batch} searches; all counts and keys match original.", flush=True)


def run_batch(batch: str, config: dict) -> None:
    cases = importlib.import_module("22_computational_extension").configurations(config)
    if batch == "controlled_pilot":
        cases = [(8, 0.)]
        rounds = [(config["seeds"][2], rep) for rep in range(3)]
    elif batch == "pilot":
        cases = [(8, -.375), (8, 0.), (8, .375), (10, .375)]
        rounds = [(config["seeds"][0], rep) for rep in range(3)]
    else:
        rounds = [(seed, 0) for seed in config["seeds"]]
    jobs = []
    for seed, rep in rounds:
        thresholds = [.01] if batch == "controlled_pilot" else config["support_thresholds"]
        group = [(seed, v, d, t, mode, rep) for v, d in cases
                 for t in thresholds for mode in MODES
                 if not (batch == "pilot" and v == 10 and t == .01)]
        order = np.random.default_rng(20261024 + seed + rep).permutation(len(group))
        jobs.extend(group[i] for i in order)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    for i, (seed, v, d, t, mode, rep) in enumerate(jobs, 1):
        path = OUT / batch / "workers" / (token(seed, v, d, t, mode) + f"_r{rep}.json")
        if not path.exists():
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--batch", batch,
                       "--seed", str(seed), "--variables", str(v), "--delta", str(d), "--support", str(t),
                       "--mode", mode, "--repetition", str(rep)]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        result = json.loads(path.read_text())
        assert result["source_sha256"] == sha(ROOT / "experiments/src/rule_mining.py")
        assert result["config_sha256"] == sha(CONFIG)
        if batch in {"controlled_pilot", "controlled"}:
            assert result["cpu_affinity"] == [0]
        print(f"{i}/{len(jobs)} {path.stem}: {result['elapsed_seconds']:.3f}s wall, "
              f"{result['cpu_seconds']:.3f}s CPU", flush=True)
    summarize(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--batch", choices=["pilot", "refresh", "controlled_pilot", "controlled"], default="pilot")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variables", type=int)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--support", type=float)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_yaml(CONFIG)
    if args.worker:
        worker(args, config)
    elif args.audit:
        audit()
    elif args.summarize_only:
        summarize(args.batch)
    else:
        run_batch(args.batch, config)


if __name__ == "__main__":
    main()
