"""Descriptive checks of saved parameter choices and fixed-parameter split gaps.

No experiment is rerun and no item pairs are used as independent replicates.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.reproducibility import load_yaml

OUT = ROOT / "experiments/outputs/precision_followup"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_yaml(ROOT / "experiments/configs/revision_r2.yaml")
    inputs = [dict(path=name, sha256=sha(ROOT / name)) for name in [
        "experiments/configs/revision_r2.yaml", "experiments/configs/tabular.yaml",
        "data/raw/tabular_firm/manifest.json"]]

    def read(path: Path) -> pd.DataFrame:
        inputs.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path)))
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    folder = ROOT / "experiments/outputs/revision_r2"
    inner = read(folder / "selection_inner.csv")
    choices = read(folder / "selection_choices.csv")
    outer = read(folder / "selection_outer.csv")
    rows = []
    for (dataset, seed), b in inner.groupby(["dataset", "seed"], sort=False):
        grouped = b.groupby("delta").agg(objective=("jaccard", "mean"), eligible=("eligible", "all"))
        eligible = grouped[grouped.eligible]
        if eligible.empty:
            negative = positive = 0.
            near = []
        else:
            near = eligible[eligible.objective >= eligible.objective.max() - config["selection"]["tie_tolerance"]].index.tolist()
            negative = min(near, key=lambda d: (abs(d), d))
            positive = min(near, key=lambda d: (abs(d), -d))
        saved = choices[(choices.dataset == dataset) & (choices.seed == seed)].iloc[0]
        assert negative == saved.chosen_delta
        rows.append(dict(dataset=dataset, seed=int(seed), chosen_negative_tie=negative,
                         chosen_positive_tie=positive, sign_tie_used=negative != positive,
                         eligible_near_best=json.dumps(near)))
    ties = pd.DataFrame(rows)
    ties.to_csv(OUT / "sign_tie_audit.csv", index=False)
    paired = outer.pivot(index=["dataset", "seed"], columns="strategy", values="jaccard")
    diff = paired.selected - paired.symmetric
    assert (int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())) == (11, 13, 6)
    alternative_rows = []
    if ties.sign_tie_used.any():
        module = importlib.import_module("30_reviewer2_analysis")
        from experiments.src.fuzzification import fuzzify_our_main
        from experiments.src.repeated_splits import pair_moments, score_pairs
        tabular = load_yaml(ROOT / "experiments/configs/tabular.yaml")
        for r in ties[ties.sign_tie_used].itertuples():
            source = folder / f"selection_{r.dataset}_{r.seed}_split.npz"
            inputs.append(dict(path=source.relative_to(ROOT).as_posix(), sha256=sha(source)))
            split = np.load(source)
            data = module.numeric_data(r.dataset, tabular)
            dev, test = data.iloc[split["development"]], data.iloc[split["evaluation"]]
            _, md = fuzzify_our_main(dev, dev, 3)
            _, me = fuzzify_our_main(dev, test, 3)
            ad, ae = pair_moments(md), pair_moments(me)
            for policy, delta in [("negative", r.chosen_negative_tie), ("positive", r.chosen_positive_tie), ("symmetric", 0.)]:
                rd = module.keyed_flags(score_pairs(ad, 1, delta), config)
                re = module.keyed_flags(score_pairs(ae, 1, delta), config)
                value = module.set_overlap(rd, re)
                if policy in ["negative", "symmetric"]:
                    label = "selected" if policy == "negative" else "symmetric"
                    assert abs(value - paired.loc[(r.dataset, r.seed), label]) < 1e-14
                alternative_rows.append(dict(dataset=r.dataset, seed=r.seed, tie_policy=policy, delta=delta,
                                             development_retained=len(rd), evaluation_retained=len(re),
                                             jaccard=value))
        pd.DataFrame(alternative_rows).to_csv(OUT / "sign_tie_outer_check.csv", index=False)

    static = read(ROOT / "experiments/outputs/tabular/summary.csv")
    static = static[(static.label_count == 3) & (static.evaluation == "proposed") & (static.delta != 0)]
    ranges = []
    for dataset, block in static.groupby("dataset", sort=False):
        scaled = block.absolute_delta_median / block.delta.abs()
        assert np.allclose(scaled, scaled.iloc[0], rtol=1e-10, atol=1e-14)
        ranges.append(dict(dataset=dataset, settings=len(block), min_abs_delta=block.delta.abs().min(),
                           max_abs_delta=block.delta.abs().max(),
                           minimum=block.absolute_delta_median.min(), maximum=block.absolute_delta_median.max(),
                           maximum_to_minimum=block.absolute_delta_median.max() / block.absolute_delta_median.min(),
                           gamma_median=scaled.median()))
    pd.DataFrame(ranges).to_csv(OUT / "parameter_range_audit.csv", index=False)

    split_rows = []
    pair_folder = ROOT / "experiments/outputs/revision_r1_q4/pairs"
    for path in sorted(pair_folder.glob("*.parquet")):
        frame = read(path)
        fixed = frame[frame.delta == .5]
        assert len(fixed) > 0
        gap = (fixed.support_forward - fixed.support_reverse).abs()
        assert np.allclose(gap, .5 * fixed.gamma.abs(), rtol=1e-9, atol=1e-14)
        split_rows.append(dict(dataset=fixed.dataset.iloc[0], seed=int(fixed.seed.iloc[0]),
                               sigma=1., delta=.5, item_pairs=len(fixed), median_absolute_gap=gap.median()))
    splits = pd.DataFrame(split_rows)
    assert len(splits) == 120 and splits.groupby("dataset").seed.nunique().eq(20).all()
    splits.to_csv(OUT / "fixed_gap_by_split.csv", index=False)
    summary = []
    for dataset, b in splits.groupby("dataset", sort=False):
        v = b.median_absolute_gap
        summary.append(dict(dataset=dataset, splits=len(b), median=v.median(), q10=v.quantile(.1),
                            q90=v.quantile(.9), minimum=v.min(), maximum=v.max()))
    pd.DataFrame(summary).to_csv(OUT / "fixed_gap_summary.csv", index=False)
    manifest = dict(inputs=inputs, script_sha256=sha(Path(__file__)), python=sys.version,
                    numpy=np.__version__, pandas=pd.__version__, choices_checked=len(ties),
                    sign_tie_used=int(ties.sign_tie_used.sum()),
                    alternative_sign_changes=ties.loc[ties.sign_tie_used].to_dict("records"),
                    alternative_outer_check=alternative_rows,
                    outer_better=11, outer_equal=13, outer_worse=6, median_outer_change=float(diff.median()),
                    fixed_gap_split_summaries=len(splits), independent_unit="descriptive split-level summaries; shared source records",
                    parameter_range_cause="absolute delta ranges from 0.125 to 1; all dataset median ratios equal eight")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "inputs"}, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
