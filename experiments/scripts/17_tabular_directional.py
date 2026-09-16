"""Fixed-split static benchmark analysis, with every pair and threshold retained."""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.fuzzification import fuzzify_our_main
from experiments.src.metrics import threshold_selection_asymmetry
from experiments.src.operators import configured_evaluations, operator_asymmetry_index
from experiments.src.reproducibility import load_yaml


def main() -> None:
    config_path = ROOT / "experiments/configs/tabular.yaml"
    config = load_yaml(config_path)
    common = load_yaml(ROOT / "experiments/configs/common.yaml")
    provenance = json.loads((ROOT / "data/raw/tabular_firm/manifest.json").read_text())
    outputs = ROOT / "experiments/outputs/tabular"
    outputs.mkdir(parents=True, exist_ok=True)
    thresholds = [dict(coverage=common["thresholds"]["coverage"], support=s, confidence=c)
                  for s in common["thresholds"]["support"] for c in common["thresholds"]["confidence"]]
    threshold_names = list(threshold_selection_asymmetry([1], [1], [1], [1], [1], [1], thresholds))
    summaries, inventory = [], []
    for dataset in config["datasets"]:
        entry = next(row for row in provenance if row["dataset"] == dataset)
        path = ROOT / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Data hash mismatch: {dataset}")
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        numeric = frame.drop(columns=[config["targets"][dataset]]).select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"Missing/nonfinite predictors in {dataset}; preprocessing decision required.")
        indices = np.random.default_rng(config["seed"]).permutation(len(frame))
        split = int(len(frame) * config["training_fraction"])
        train, test = numeric.iloc[indices[:split]], numeric.iloc[indices[split:]]
        np.savez(outputs / f"{dataset}_split.npz", training=indices[:split], test=indices[split:])
        inventory.append(dict(dataset=dataset, rows=len(frame), predictors=len(numeric.columns),
                              training_rows=len(train), test_rows=len(test), sha256=entry["sha256"]))
        for labels in config["label_counts"]:
            _, memberships = fuzzify_our_main(train, test, labels=labels)
            rows = []
            for (left, a), (right, c) in itertools.combinations(memberships.items(), 2):
                if left.rsplit("::", 1)[0] == right.rsplit("::", 1)[0]:
                    continue
                cov_a, cov_c = float(a.mean()), float(c.mean())
                gamma = float((a*c*(1-a)*(1-c)*(a-c)).mean())
                product_mean = float((a*c).mean())
                boundary = a*c*(1-a)*(1-c)
                symmetric_mean = float((boundary*(a+c)).mean())
                absolute_mean = float(np.abs(boundary*(a-c)).mean())
                for name, power, sigma, delta, evaluation in configured_evaluations(common):
                    if name == "proposed":
                        baseline = product_mean - float(sigma)*symmetric_mean/2
                        sf, sr = baseline + float(delta)*gamma/2, baseline - float(delta)*gamma/2
                        absolute_asymmetry = abs(float(delta))*absolute_mean
                    else:
                        forward, reverse = evaluation(a, c), evaluation(c, a)
                        sf, sr = float(forward.mean()), float(reverse.mean())
                        absolute_asymmetry = float(np.abs(forward-reverse).mean())
                    cf, cr = sf/cov_a if cov_a > 0 else np.nan, sr/cov_c if cov_c > 0 else np.nan
                    gap = sf-sr
                    row = dict(engine="OUR-MAIN", protocol="static-fixed-split",
                               dataset=dataset, label_count=labels, item_a=left, item_c=right,
                               evaluation=name, power=power, sigma=sigma, delta=delta,
                               coverage_forward=cov_a, coverage_reverse=cov_c,
                               support_forward=sf, support_reverse=sr,
                               confidence_forward=cf, confidence_reverse=cr,
                               support_delta=gap, absolute_delta=abs(gap),
                               normalized_gap=abs(gap)/(sf+sr+1e-12), gamma=gamma,
                               empirical_absolute_asymmetry=absolute_asymmetry,
                               operator_D=operator_asymmetry_index(float(delta)) if name == "proposed" else np.nan,
                               identity_error=gap-float(delta)*gamma if name == "proposed" else np.nan)
                    for threshold_name, threshold in zip(threshold_names, thresholds):
                        selected_forward = cov_a > 0 and cov_a >= threshold["coverage"] and sf >= threshold["support"] and cf >= threshold["confidence"]
                        selected_reverse = cov_c > 0 and cov_c >= threshold["coverage"] and sr >= threshold["support"] and cr >= threshold["confidence"]
                        row[threshold_name] = float(selected_forward != selected_reverse)
                    rows.append(row)
            results = pd.DataFrame(rows)
            errors = results.identity_error.dropna().abs()
            if errors.max() > common["float_tolerance"]:
                raise AssertionError(f"Identity verification failed: {dataset}")
            controls = results[results.evaluation.isin(["product", "minimum"])]
            if not controls.empty and controls.absolute_delta.max() > common["float_tolerance"]:
                raise AssertionError("Symmetric control failed")
            results.to_parquet(outputs / f"{dataset}_{labels}_pairs.parquet", index=False)
            keys = ["engine", "protocol", "dataset", "label_count", "evaluation", "power", "sigma", "delta"]
            measures = ["absolute_delta", "normalized_gap", "empirical_absolute_asymmetry"]
            summary = results.groupby(keys, dropna=False)[measures].agg(["median", "min", "max"])
            summary.columns = ["_".join(column) for column in summary.columns]
            summary = summary.reset_index()
            selection_columns = [column for column in results if column.startswith("SA")]
            if selection_columns:
                selection = results.groupby(keys, dropna=False)[selection_columns].mean().reset_index()
                summary = summary.merge(selection, on=keys)
            summaries.append(summary)
            print(f"{dataset}: labels={labels}, rows={len(results)}, max identity error={errors.max():.3g}", flush=True)
    pd.concat(summaries, ignore_index=True).to_csv(outputs / "summary.csv", index=False)
    pd.DataFrame(inventory).to_csv(outputs / "datasets.csv", index=False)
    manifest = dict(status="complete", protocol="static-fixed-split", smoke=False,
                    config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    common_sha256=hashlib.sha256((ROOT / "experiments/configs/common.yaml").read_bytes()).hexdigest(),
                    datasets=inventory,
                    scope="Direct swaps and threshold selection only; no process-order or pruning results.")
    (outputs / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
