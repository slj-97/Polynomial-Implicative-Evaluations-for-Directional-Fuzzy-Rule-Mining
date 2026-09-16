"""Validate the revised protocol and export fuzzification-sensitivity evidence."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.reproducibility import write_dataframe, write_json


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_first(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    groups = [
        "label_count", "independent_null", "mechanism", "h_true", "h_eval",
        "lambda", "noise", "operator", "sigma", "delta", "p", "protocol",
    ]
    return frame.groupby(groups, as_index=False, dropna=False)[metrics].median()


def main() -> None:
    raw = ROOT / "experiments/outputs/raw_results"
    logs = ROOT / "experiments/outputs/logs"
    membership_path = raw / "synthetic_membership_raw_v2.parquet"
    dynamic_path = raw / "synthetic_dynamic_raw_v2.parquet"
    membership = pd.read_parquet(membership_path)
    dynamic = pd.read_parquet(dynamic_path)
    membership_manifest = json.loads((logs / "synthetic_membership_manifest_v2.json").read_text())
    dynamic_manifest = json.loads((logs / "synthetic_dynamic_manifest_v2.json").read_text())
    sa_columns = [column for column in dynamic if column.startswith("SA_")]

    expected_membership_rows = 129_600
    expected_dynamic_rows = 2_332_800
    checks = {
        "membership_full": not membership_manifest["smoke"] and len(membership) == expected_membership_rows,
        "dynamic_full": not dynamic_manifest["smoke"] and len(dynamic) == expected_dynamic_rows,
        "replications_50": int(dynamic["replication"].nunique()) == 50,
        "label_counts_3_and_5": sorted(dynamic["label_count"].unique().tolist()) == [3, 5],
        "nine_selection_thresholds": len(sa_columns) == 9,
        "dynamic_identity_within_tolerance": float(dynamic["identity_error"].max(skipna=True)) <= 1.0e-12,
    }

    old_dynamic = pd.read_parquet(raw / "synthetic_dynamic_raw.parquet")
    revised_three = dynamic[dynamic["label_count"].eq(3)].reset_index(drop=True)
    shared = [column for column in old_dynamic if column in revised_three and column != "selection_asymmetry"]
    maximum_difference = 0.0
    for column in shared:
        if pd.api.types.is_numeric_dtype(old_dynamic[column]):
            left = old_dynamic[column].to_numpy(dtype=np.float64)
            right = revised_three[column].to_numpy(dtype=np.float64)
            finite = np.isfinite(left) | np.isfinite(right)
            if finite.any():
                maximum_difference = max(maximum_difference, float(np.nanmax(np.abs(left - right))))
        elif not old_dynamic[column].reset_index(drop=True).equals(revised_three[column].reset_index(drop=True)):
            raise RuntimeError(f"Three-label regression in nonnumeric column {column!r}.")
    checks["frozen_three_label_values_unchanged"] = len(old_dynamic) == len(revised_three) and maximum_difference == 0.0
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Protocol-alignment checks failed: {failed}")

    proposed = dynamic[dynamic["operator"].eq("proposed")].copy()
    proposed["absolute_directional_quantity"] = np.where(
        proposed["protocol"].eq("direct_swap"), proposed["Delta"].abs(), proposed["Psi"].abs()
    )
    metrics = ["absolute_directional_quantity", "normalized_gap", "empirical_absolute_asymmetry", *sa_columns]
    seed_first = _seed_first(proposed, metrics)
    summary = seed_first.groupby(["label_count", "protocol"], as_index=False)[metrics].median()
    summary["minimum_median_SA"] = summary[sa_columns].min(axis=1)
    summary["maximum_median_SA"] = summary[sa_columns].max(axis=1)
    write_dataframe(summary, "experiments/outputs/summaries/fuzzification_sensitivity_v2.csv")

    threshold_rows: list[dict[str, object]] = []
    for label_count, part in seed_first[seed_first["protocol"].eq("direct_swap")].groupby("label_count"):
        for column in sa_columns:
            threshold_rows.append(
                {
                    "label_count": int(label_count),
                    "threshold": column.removeprefix("SA_"),
                    "median_SA": float(part[column].median()),
                    "q25_SA": float(part[column].quantile(0.25)),
                    "q75_SA": float(part[column].quantile(0.75)),
                }
            )
    write_dataframe(pd.DataFrame(threshold_rows), "experiments/outputs/summaries/selection_asymmetry_thresholds_v2.csv")

    direct_summary = summary[summary["protocol"].eq("direct_swap")].set_index("label_count")
    process_summary = summary[summary["protocol"].eq("process_order")].set_index("label_count")
    digest = "\n".join(
        [
            "# Protocol-alignment results digest", "",
            "- Revised membership and dynamic manifests are full runs (`smoke=false`).",
            f"- The revised dynamic file contains {len(dynamic):,} rows, 50 replications, and label counts 3 and 5.",
            f"- Nine threshold-specific selection-asymmetry fields are retained.",
            f"- Maximum proposed-family direct-swap identity error: {dynamic['identity_error'].max(skipna=True):.3e}.",
            f"- The frozen three-label dynamic values are unchanged over {len(shared)} shared fields; maximum numeric difference: {maximum_difference:.1f}.",
            f"- Seed-first median empirical absolute asymmetry for proposed direct-swap rows: three labels {direct_summary.loc[3, 'empirical_absolute_asymmetry']:.3e}, five labels {direct_summary.loc[5, 'empirical_absolute_asymmetry']:.3e}.",
            f"- Across the nine thresholds, median direct-swap SA ranges from {direct_summary.loc[3, 'minimum_median_SA']:.3f} to {direct_summary.loc[3, 'maximum_median_SA']:.3f} for three labels and from {direct_summary.loc[5, 'minimum_median_SA']:.3f} to {direct_summary.loc[5, 'maximum_median_SA']:.3f} for five labels.",
            f"- Seed-first median absolute process-order gap: three labels {process_summary.loc[3, 'absolute_directional_quantity']:.3e}, five labels {process_summary.loc[5, 'absolute_directional_quantity']:.3e}.",
            "- These are descriptive sensitivity summaries, not evidence of numerical invariance or causality.",
            "- SWaT and TEP remain blocked because no real measurements or approved process pairs are configured.",
        ]
    ) + "\n"
    (ROOT / "experiments/outputs/paper/results_digest_v3.md").write_text(digest, encoding="utf-8")
    write_json(
        {
            "checks": checks,
            "rows": {"membership_v2": len(membership), "dynamic_v2": len(dynamic)},
            "hashes": {"membership_v2": _sha256(membership_path), "dynamic_v2": _sha256(dynamic_path)},
            "maximum_three_label_regression_difference": maximum_difference,
            "maximum_identity_error": float(dynamic["identity_error"].max(skipna=True)),
            "real_data_completed": False,
        },
        "experiments/outputs/paper/results_manifest_v3.json",
    )
    print("Protocol-alignment summaries and v3 evidence manifest written.")


if __name__ == "__main__":
    main()
