"""Frozen 20-split protocol for Reviewer 1 Q4; original outputs are read only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.fuzzification import fuzzify_our_main
from experiments.src.matched_rules import pair_states, retention_contrast, retention_flags
from experiments.src.operators import eval_poly
from experiments.src.repeated_splits import pair_moments, score_pairs
from experiments.src.reproducibility import load_yaml

OUT = ROOT / "experiments/outputs/revision_r1_q4"
NAMES = dict(iris="Iris", wdbc="WDBC", vehicle="Vehicle", abalone="Abalone",
             magic="MAGIC", online_news="Online News")
STATES = ["neither", "forward_only", "reverse_only", "both"]
ENGINE = "OUR-MAIN"
PROTOCOL = "static-20-split-fixed-sigma-stability"


def tagged(frame: pd.DataFrame) -> pd.DataFrame:
    frame["engine"], frame["protocol"] = ENGINE, PROTOCOL
    return frame


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: pd.Series) -> dict[str, float]:
    return dict(median=float(values.median()), q25=float(values.quantile(.25)),
                q75=float(values.quantile(.75)), minimum=float(values.min()), maximum=float(values.max()))


def md(frame: pd.DataFrame) -> str:
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)])


def figures(runs: pd.DataFrame, examples: pd.DataFrame, datasets: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, sign in zip(axes, ["minus", "plus"]):
        arrays = [runs.loc[runs.dataset == d, f"zero_to_{sign}_jaccard"].to_numpy() for d in datasets]
        ax.boxplot(arrays, tick_labels=[NAMES[d] for d in datasets], whis=(0, 100), showfliers=False)
        for i, values in enumerate(arrays):
            ax.scatter(i+1+np.linspace(-.13,.13,len(values)), values, s=11, alpha=.55, color="#286b9c")
        ax.set_title(f"delta = {'-' if sign == 'minus' else '+'}0.5 versus delta = 0")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Within-split Jaccard overlap")
        ax.set_ylim(0.90, 1.005)
    fig.suptitle("Twenty 70/30 splits; sigma = 1; three labels")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"figure-01-split-overlap.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for number in range(1, 5):
        block = examples[(examples.example == number) & (examples.delta > 0) & examples.available]
        axes[0].scatter(number+np.linspace(-.12,.12,len(block)), block.support_gap*1000, s=18)
        q = block.support_gap.quantile([.1,.5,.9]).to_numpy()*1000
        axes[0].plot([number,number], [q[0],q[2]], color="black", lw=1)
        axes[0].scatter([number], [q[1]], marker="_", color="black", s=100)
    axes[0].axhline(0, color="gray", lw=.7)
    axes[0].set_xticks(range(1,5), ["1: addition", "2: removal", "3: unchanged", "4: flip"], rotation=20)
    axes[0].set_ylabel("Support difference at positive delta (x 0.001)")
    axes[0].set_title("Fixed Q3 pairs; median and 10th–90th percentiles")
    freq = examples[examples.available].copy()
    for offset, (setting, label, color) in enumerate([(-1,"negative","#286b9c"),(0,"zero","#666666"),(1,"positive","#ba5b3b")]):
        values=[]
        for number in range(1,5):
            block=freq[(freq.example==number) & (np.sign(freq.delta)==setting)]
            values.append(block.state_changed.mean())
        axes[1].bar(np.arange(4)+(offset-1)*.24, values, .24, label=label, color=color)
    axes[1].set_xticks(range(4), ["Pair 1", "Pair 2", "Pair 3", "Pair 4"])
    axes[1].set_ylabel("Fraction of available splits")
    axes[1].set_title("Pair retention state differs from symmetric member")
    axes[1].set_ylim(0,1)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(OUT / "figures" / f"figure-02-representative-stability.{ext}", dpi=180)
    plt.close(fig)


def main() -> None:
    config_path = ROOT / "experiments/configs/revision_r1_q4.yaml"
    config = load_yaml(config_path)
    tabular = load_yaml(ROOT / "experiments/configs/tabular.yaml")
    datasets = tabular["datasets"]
    seeds = config["seeds"]
    assert len(seeds) == len(set(seeds)) == 20 and seeds[0] == 20260903
    delta_values = config["deltas"]
    thresholds = tuple(config["thresholds"][key] for key in ("coverage","support","confidence"))
    tolerance = float(config["sign_zero_tolerance"])
    original_examples = pd.read_csv(ROOT / config["representatives"])
    specs = original_examples.drop_duplicates("example").copy()
    provenance = json.loads((ROOT / "data/raw/tabular_firm/manifest.json").read_text())
    for folder in (OUT, OUT / "splits", OUT / "preprocessing", OUT / "pairs", OUT / "figures"):
        folder.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(dict(config=config, datasets=datasets,
        representative_specifications=specs.fillna("").to_dict("records"),
        empty_jaccard=1, summaries="linear empirical quantiles; no confidence intervals or tests",
        rule_identity="variable name and label index; unavailable is not unretained"), indent=2), encoding="utf-8")
    runs, transitions, example_rows, inventories = [], [], [], []
    verification = dict(original_seed_checked_datasets=[], max_original_score_error=0.,
                        max_direct_error=0., direct_scalar_checks=0, original_flag_disagreements=0,
                        effective_threshold_disagreements=0, max_support_identity_error=0.)
    for dataset in datasets:
        source = next(p for p in provenance if p["dataset"] == dataset)
        data_path = ROOT / source["path"]
        assert sha(data_path) == source["sha256"]
        frame = pd.read_csv(data_path)
        frame.columns = frame.columns.str.strip()
        numeric = frame.drop(columns=[tabular["targets"][dataset]]).select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy()).all()
        dataset_specs = specs[specs.dataset == dataset]
        original = pd.read_parquet(ROOT / f"experiments/outputs/tabular/{dataset}_3_pairs.parquet")
        original = original[(original.evaluation == "proposed") & (original.sigma == 1)]
        for seed in seeds:
            permutation = np.random.default_rng(seed).permutation(len(frame))
            cut = int(len(frame)*config["training_fraction"])
            train_index, eval_index = permutation[:cut], permutation[cut:]
            np.savez_compressed(OUT / "splits" / f"{dataset}_{seed}.npz", training=train_index, test=eval_index)
            if seed == seeds[0]:
                frozen_split = np.load(ROOT / f"experiments/outputs/tabular/{dataset}_split.npz")
                assert np.array_equal(frozen_split["training"], train_index)
                assert np.array_equal(frozen_split["test"], eval_index)
            training, evaluation = numeric.iloc[train_index], numeric.iloc[eval_index]
            scaler, memberships = fuzzify_our_main(training, evaluation, labels=3)
            (OUT / "preprocessing" / f"{dataset}_{seed}.json").write_text(json.dumps(dict(
                lower=scaler.lower.to_dict(), upper=scaler.upper.to_dict(),
                active_continuous_columns=scaler.active_columns, item_names=list(memberships)), indent=2), encoding="utf-8")
            moments = pair_moments(memberships)
            inventories.append(dict(dataset=dataset, seed=seed, training_rows=len(training),
                                    evaluation_rows=len(evaluation), items=len(memberships), pairs=len(moments),
                                    data_sha256=source["sha256"], split_sha256=sha(OUT / "splits" / f"{dataset}_{seed}.npz")))
            scores, flags = {}, {}
            saved_scores = []
            for delta in delta_values:
                scored = score_pairs(moments, config["sigma"], delta)
                selected = retention_flags(scored, *thresholds)
                coverage = scored[["coverage_forward","coverage_reverse"]].to_numpy()
                supports = scored[["support_forward","support_reverse"]].to_numpy()
                effective = np.maximum(thresholds[1], thresholds[2]*coverage)
                alternate = (coverage > 0) & (coverage >= thresholds[0]) & (supports >= effective)
                verification["effective_threshold_disagreements"] += int((selected != alternate).sum())
                error = np.max(np.abs(supports[:,0]-supports[:,1]-delta*moments.gamma.to_numpy()))
                verification["max_support_identity_error"] = max(verification["max_support_identity_error"], float(error))
                scores[delta], flags[delta] = scored, selected
                scored["selected_forward"], scored["selected_reverse"] = selected[:,0], selected[:,1]
                scored["delta"], scored["seed"], scored["dataset"] = delta, seed, dataset
                saved_scores.append(scored)
                # Independent record-level check spread across the candidate universe.
                for idx in np.unique(np.linspace(0,len(scored)-1,4,dtype=int)):
                    row = scored.iloc[idx]
                    a,c = memberships[row.item_a],memberships[row.item_c]
                    direct = [eval_poly(a,c,1,delta).mean(), eval_poly(c,a,1,delta).mean()]
                    errors = np.abs(np.asarray(direct)-np.array([row.support_forward,row.support_reverse]))
                    verification["max_direct_error"] = max(verification["max_direct_error"], float(errors.max()))
                    verification["direct_scalar_checks"] += 2
                if seed == seeds[0]:
                    old = original[original.delta == delta].set_index(["item_a","item_c"]).sort_index()
                    new = scored.set_index(["item_a","item_c"]).sort_index()
                    assert old.index.equals(new.index)
                    cols=["coverage_forward","coverage_reverse","support_forward","support_reverse",
                          "confidence_forward","confidence_reverse","gamma"]
                    error = np.nanmax(np.abs(old[cols].to_numpy()-new[cols].to_numpy()))
                    verification["max_original_score_error"] = max(verification["max_original_score_error"], float(error))
                    verification["original_flag_disagreements"] += int((retention_flags(old,*thresholds)!=retention_flags(new,*thresholds)).sum())
            tagged(pd.concat(saved_scores,ignore_index=True)).to_parquet(OUT / "pairs" / f"{dataset}_{seed}.parquet", index=False)
            meta=dict(dataset=dataset,seed=seed,n_zero=int(flags[0.].sum()),n_minus=int(flags[-.5].sum()),
                      n_plus=int(flags[.5].sum()),unordered_pairs=len(moments))
            for label, before, after in [("zero_to_minus",0.,-.5),("zero_to_plus",0.,.5),("minus_to_plus",-.5,.5)]:
                contrast, matrix = retention_contrast(flags[before], flags[after])
                meta.update({f"{label}_{key}":value for key,value in contrast.items()})
                for i,j in np.ndindex(4,4):
                    transitions.append(dict(dataset=dataset,seed=seed,contrast=label,from_state=STATES[i],
                                            to_state=STATES[j],count=int(matrix[i,j])))
            runs.append(meta)
            for spec in dataset_specs.itertuples():
                labels=int(spec.label_count)
                example_memberships=memberships if labels==3 else fuzzify_our_main(training,evaluation,labels=labels)[1]
                available=spec.item_a in example_memberships and spec.item_c in example_memberships
                frozen=original_examples[original_examples.example==spec.example]
                ds=sorted(frozen.delta.unique())
                if not available:
                    for delta in ds:
                        example_rows.append(dict(example=int(spec.example), dataset=dataset, seed=seed,
                                                 delta=delta, available=False))
                    continue
                a,c=example_memberships[spec.item_a],example_memberships[spec.item_c]
                cov_a,cov_c=float(a.mean()),float(c.mean())
                computed=[]
                for delta in ds:
                    sf,sr=float(eval_poly(a,c,float(spec.sigma),delta).mean()),float(eval_poly(c,a,float(spec.sigma),delta).mean())
                    cf,cr=sf/cov_a if cov_a>0 else np.nan,sr/cov_c if cov_c>0 else np.nan
                    selected_a=bool(cov_a>0 and cov_a>=spec.tau_coverage and sf>=spec.tau_support and cf>=spec.tau_confidence)
                    selected_c=bool(cov_c>0 and cov_c>=spec.tau_coverage and sr>=spec.tau_support and cr>=spec.tau_confidence)
                    gap=sf-sr
                    frozen_row=frozen[frozen.delta==delta].iloc[0]
                    reference_gap=float(frozen_row.support_forward-frozen_row.support_reverse)
                    sign=lambda value: 0 if abs(value)<=tolerance else int(np.sign(value))
                    row=dict(example=int(spec.example),category=spec.category,dataset=dataset,seed=seed,
                             label_count=labels,sigma=float(spec.sigma),delta=delta,item_a=spec.item_a,item_c=spec.item_c,
                             available=True,coverage_forward=cov_a,coverage_reverse=cov_c,
                             support_forward=sf,support_reverse=sr,confidence_forward=cf,confidence_reverse=cr,
                             support_gap=gap,sign=sign(gap),reference_sign=sign(reference_gap),
                             sign_agrees=sign(gap)==sign(reference_gap),selected_forward=selected_a,selected_reverse=selected_c,
                             state=int(selected_a)+2*int(selected_c),
                             margin_forward=sf-max(spec.tau_support,spec.tau_confidence*cov_a),
                             margin_reverse=sr-max(spec.tau_support,spec.tau_confidence*cov_c))
                    if seed==seeds[0]:
                        assert abs(sf-frozen_row.support_forward)<1e-12 and abs(sr-frozen_row.support_reverse)<1e-12
                        assert selected_a==frozen_row.selected_forward and selected_c==frozen_row.selected_reverse
                    computed.append(row)
                zero=next(row for row in computed if row["delta"]==0)
                low,high=computed[0],computed[-1]
                flip=bool((low["state"],high["state"]) in [(1,2),(2,1)])
                for row in computed:
                    row.update(state_changed=row["state"]!=zero["state"],
                               forward_added=row["selected_forward"] and not zero["selected_forward"],
                               forward_removed=zero["selected_forward"] and not row["selected_forward"],
                               reverse_added=row["selected_reverse"] and not zero["selected_reverse"],
                               reverse_removed=zero["selected_reverse"] and not row["selected_reverse"],
                               matched_sign_flip=flip)
                example_rows.extend(computed)
            print(f"{dataset} seed={seed}: retained {meta['n_minus']}/{meta['n_zero']}/{meta['n_plus']}",flush=True)
        verification["original_seed_checked_datasets"].append(dataset)
    assert verification["original_flag_disagreements"]==0
    assert verification["effective_threshold_disagreements"]==0
    assert verification["max_direct_error"]<1e-12 and verification["max_original_score_error"]<1e-12
    runs=tagged(pd.DataFrame(runs))
    examples=tagged(pd.DataFrame(example_rows))
    runs.to_csv(OUT / "split_metrics.csv",index=False)
    tagged(pd.DataFrame(transitions)).to_csv(OUT / "state_transitions.csv",index=False)
    examples.to_csv(OUT / "representative_runs.csv",index=False)
    tagged(pd.DataFrame(inventories)).to_csv(OUT / "inventory.csv",index=False)
    summary=[]
    for dataset in datasets:
        block=runs[runs.dataset==dataset]
        for column in runs.select_dtypes(include=[np.number]).columns:
            if column=="seed": continue
            summary.append(dict(dataset=dataset,metric=column,n=len(block),**quantiles(block[column])))
    summary=tagged(pd.DataFrame(summary))
    summary.to_csv(OUT / "dataset_summary.csv",index=False)
    example_summary=[]
    for (example,delta),block in examples.groupby(["example","delta"],sort=True):
        valid=block[block.available]
        row=dict(example=int(example),delta=float(delta),available=int(len(valid)),unavailable=int(len(block)-len(valid)))
        if len(valid):
            row.update(gap_median=float(valid.support_gap.median()),gap_q10=float(valid.support_gap.quantile(.1)),
                       gap_q90=float(valid.support_gap.quantile(.9)),sign_agrees=int(valid.sign_agrees.sum()),
                       positive=int((valid.sign>0).sum()),negative=int((valid.sign<0).sum()),near_zero=int((valid.sign==0).sum()))
            for key in ["selected_forward","selected_reverse","state_changed","forward_added","forward_removed",
                        "reverse_added","reverse_removed","matched_sign_flip"]:
                row[key+"_count"]=int(valid[key].sum())
        example_summary.append(row)
    example_summary=tagged(pd.DataFrame(example_summary))
    example_summary.to_csv(OUT / "representative_summary.csv",index=False)
    manifest=dict(status="complete",engine=ENGINE,protocol=PROTOCOL,datasets=datasets,seeds=seeds,main_evaluations=len(runs)*3,
                  main_split_comparisons=len(runs),main_sigma=1,main_labels=3,thresholds=thresholds,
                  source_config_sha256=sha(config_path),example_specification_sha256=sha(ROOT/config["representatives"]),
                  analysis_script_sha256=sha(Path(__file__)),moment_module_sha256=sha(ROOT/"experiments/src/repeated_splits.py"),
                  verification=verification,unavailable_example_rows=int((~examples.available).sum()),
                  interpretation="Descriptive repeated-split sensitivity on shared source data; no inferential tests or confidence intervals.")
    (OUT / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    figures(runs,examples,datasets)
    display=[]
    for dataset in datasets:
        row={"dataset":NAMES[dataset]}
        for key in ["n_zero","n_minus","n_plus","zero_to_minus_jaccard","zero_to_plus_jaccard",
                    "zero_to_minus_added","zero_to_minus_removed","zero_to_plus_added","zero_to_plus_removed"]:
            q=quantiles(runs.loc[runs.dataset==dataset,key])
            row[key]=f"{q['median']:.4f} [{q['q25']:.4f}, {q['q75']:.4f}]"
        display.append(row)
    report="# Reviewer 1 Q4: repeated-split stability\n\n"
    report+="20 frozen seeds 20260903–20260922; 70/30 splits; training-only original preprocessing.\n"
    report+="Main analysis: six datasets, three labels, sigma=1, delta=-0.5/0/0.5, thresholds 0.01/0.01/0.5.\n"
    report+="120 split triplets and 360 evaluations. Original seed and all four Q3 examples reproduced.\n\n"
    report+=md(pd.DataFrame(display))+"\n\n## Fixed Q3 representatives\n\n"+md(example_summary)+"\n\n"
    report+="The examples retain their Q3 partitions, parameters and thresholds. Sign agreement is with the original seed, separately at each fixed delta; |gap|<=1e-12 is near zero. Unavailable splits are excluded from frequencies and recorded separately.\n\n"
    report+="## Claim candidates\n\nUse the per-dataset overlap distributions to characterize within-split parameter effects and their variability across splits. Use the four frozen examples to distinguish sign consistency from selection consistency. Do not infer stable retention merely from a stable sign, treat opposite signs as independent evidence, or infer new-population/causal/predictive utility. Empirical intervals describe shared-data split variation.\n\n"
    report+="## Validation\n\n"+json.dumps(verification,indent=2)+"\n"
    (OUT / "analysis-report.md").write_text(report,encoding="utf-8")
    (OUT / "stats-appendix.md").write_text("""# Statistical definitions

Units: each dataset has 20 repeated splits of shared original records. Within a
split the two directional settings are paired with the symmetric reference.
The main table uses median and [25th,75th] empirical percentiles (NumPy/pandas
linear interpolation), not a confidence interval. Counts may have fractional
quantiles. Representative gaps use median and [10th,90th] percentiles.
Sign agreement uses the Q3 original-seed sign at each fixed nonzero delta;
absolute gaps <=1e-12 are near zero. An unavailable rule is not unretained:
report available and unavailable denominators separately. Empty/empty Jaccard
is 1 with both counts and union saved. Addition/loss and state changes are
computed within each split against delta=0. Four states are neither, forward
only, reverse only, both. Exclusive flips use forward-only <-> reverse-only
between negative and positive delta. No hypothesis tests, confidence intervals,
multiple-comparison corrections or independent-run claims: these are overlapping
resplits, not independent datasets. Rule pairs also share observations.
""",encoding="utf-8")
    (OUT / "figure-catalog.md").write_text("""# Figure catalog

1. figure-01-split-overlap: split_metrics.csv; within-split Jaccard to symmetric
reference by dataset and parameter sign. Boxes median/IQR; whiskers min/max;
dots are the 20 dependent splits. Read the distribution rather than a single
original split. No confidence interval or cross-split set Jaccard is shown.
2. figure-02-representative-stability: representative_runs.csv; gaps for the
four frozen Q3 pairs at positive delta, with median and 10th/90th percentiles,
and pair-state-change frequencies relative to symmetric within each split.
Each pair keeps its original parameters. This separates sign from retention
stability; examples are descriptive and fixed before resplitting.
""",encoding="utf-8")
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":
    main()
