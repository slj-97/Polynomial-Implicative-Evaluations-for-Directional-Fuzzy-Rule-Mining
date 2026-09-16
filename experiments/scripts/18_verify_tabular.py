"""Cross-check moment-based exports against direct record-level evaluations."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.fuzzification import fuzzify_our_main
from experiments.src.metrics import direct_swap_metrics
from experiments.src.operators import configured_evaluations
from experiments.src.reproducibility import load_yaml


def main() -> None:
    config = load_yaml(ROOT / "experiments/configs/tabular.yaml")
    common = load_yaml(ROOT / "experiments/configs/common.yaml")
    output = ROOT / "experiments/outputs/tabular"
    errors = []
    for dataset in config["datasets"]:
        frame = pd.read_csv(ROOT / f"data/raw/tabular_firm/{dataset}/{dataset}.csv")
        frame.columns = frame.columns.str.strip()
        numeric = frame.drop(columns=[config["targets"][dataset]]).select_dtypes(include=[np.number])
        split = np.load(output / f"{dataset}_split.npz")
        assert len(set(split["training"]) & set(split["test"])) == 0
        assert len(split["training"]) + len(split["test"]) == len(frame)
        for labels in config["label_counts"]:
            _, memberships = fuzzify_our_main(numeric.iloc[split["training"]], numeric.iloc[split["test"]], labels=labels)
            results = pd.read_parquet(output / f"{dataset}_{labels}_pairs.parquet")
            assert len([c for c in results if c.startswith("SA_")]) == 9
            assert not results.duplicated(["item_a", "item_c", "evaluation", "power", "sigma", "delta"]).any()
            for name, power, sigma, delta, function in configured_evaluations(common):
                mask = results.evaluation.eq(name)
                for column, value in [("power", power), ("sigma", sigma), ("delta", delta)]:
                    mask &= results[column].isna() if value is None else results[column].eq(value)
                group = results[mask]
                for row in group.sample(min(5, len(group)), random_state=43).itertuples():
                    measured = direct_swap_metrics(memberships[row.item_a], memberships[row.item_c], function)
                    for metric in ["support_forward", "support_reverse", "support_delta", "absolute_delta", "empirical_absolute_asymmetry"]:
                        errors.append(abs(measured[metric] - getattr(row, metric)))
    maximum = max(errors)
    assert maximum <= common["float_tolerance"], maximum
    report = dict(status="passed", checked_scalar_values=len(errors), max_error=maximum,
                  method="Five seeded label-pair rows per evaluation, dataset and partition; direct record-level computation.")
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
