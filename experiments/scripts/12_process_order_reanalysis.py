"""Recover the affine process-order decomposition from the completed delta grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.plotting import save_line_figure
from experiments.src.reproducibility import write_dataframe


def summary(frame: pd.DataFrame, groups: list[str], columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(groups, dropna=False):
        row = dict(zip(groups, key if isinstance(key, tuple) else (key,), strict=True))
        for column in columns:
            values = part[column].to_numpy(float)
            row[f"median_{column}"] = float(np.median(values))
            row[f"iqr_{column}"] = float(np.subtract(*np.percentile(values, [75, 25])))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    raw = pd.read_parquet(ROOT / "experiments/outputs/raw_results/synthetic_dynamic_raw.parquet")
    process = raw[(raw["protocol"] == "process_order") & (raw["operator"] == "proposed")].copy()
    base = ["seed", "independent_null", "mechanism", "h_true", "h_eval", "lambda", "noise", "sigma"]
    pieces = []
    for _, part in process.groupby(base, dropna=False):
        part = part.sort_values("delta").copy()
        delta = part["delta"].to_numpy(float); psi = part["Psi"].to_numpy(float)
        slope, intercept = np.polyfit(delta, psi, 1)
        predicted = intercept + slope * delta
        ss_total = float(np.sum((psi - psi.mean()) ** 2))
        r2 = 1.0 if ss_total == 0.0 else float(1.0 - np.sum((psi-predicted)**2) / ss_total)
        part["symmetric_process_order_gap"] = intercept
        part["process_lambda_statistic"] = slope
        part["process_operator_increment"] = psi - intercept
        part["process_decomposition_error"] = psi - intercept - delta * slope
        part["slope_error"] = slope - (part["process_operator_increment"] / part["delta"].replace(0.0, np.nan)).median(skipna=True)
        part["delta_linearity_r2"] = r2
        pieces.append(part)
    enriched = pd.concat(pieces, ignore_index=True)
    max_error = float(enriched["process_decomposition_error"].abs().max())
    if max_error > 1e-10:
        raise RuntimeError(f"Process decomposition error is too large: {max_error}")
    write_dataframe(enriched, "experiments/outputs/summaries/dynamic_process_decomposition.csv")
    nonnull = enriched[~enriched["independent_null"]]
    strength = summary(nonnull, ["lambda"], ["Psi", "symmetric_process_order_gap", "process_operator_increment", "process_lambda_statistic", "normalized_gap"])
    noise = summary(nonnull, ["noise"], ["Psi", "process_operator_increment", "process_lambda_statistic"])
    lag = summary(nonnull.assign(lag_match=nonnull["h_true"] == nonnull["h_eval"]), ["h_true", "h_eval", "lag_match"], ["Psi", "process_operator_increment"])
    null_part = enriched[enriched["independent_null"]]
    null = pd.DataFrame([{**{f"median_{column}": float(null_part[column].median()) for column in ["Psi", "symmetric_process_order_gap", "process_operator_increment", "process_lambda_statistic"]}, **{f"iqr_{column}": float(null_part[column].quantile(.75)-null_part[column].quantile(.25)) for column in ["Psi", "symmetric_process_order_gap", "process_operator_increment", "process_lambda_statistic"]}}])
    write_dataframe(strength, "experiments/outputs/summaries/dynamic_strength_summary.csv")
    write_dataframe(noise, "experiments/outputs/summaries/dynamic_noise_summary.csv")
    write_dataframe(lag, "experiments/outputs/summaries/dynamic_lag_summary.csv")
    write_dataframe(null, "experiments/outputs/summaries/dynamic_null_summary.csv")
    plot = enriched[(~enriched["independent_null"]) & (enriched["sigma"] == 1.0) & (enriched["delta"] == 1.0)].groupby("lambda", as_index=False)[["Psi", "symmetric_process_order_gap", "process_operator_increment"]].median().melt("lambda", var_name="component", value_name="value")
    save_line_figure(plot, x="lambda", y="value", group="component", title="Process-order decomposition", output_stem=ROOT / "experiments/outputs/figures/Fig_dynamic_process_decomposition")
    increment = enriched[(~enriched["independent_null"]) & (enriched["sigma"] == 1.0)].assign(abs_increment=lambda x: x["process_operator_increment"].abs()).groupby(["lambda", "noise"], as_index=False)["abs_increment"].median()
    save_line_figure(increment, x="lambda", y="abs_increment", group="noise", title="Absolute operator-induced process increment", output_stem=ROOT / "experiments/outputs/figures/Fig_dynamic_operator_increment")
    linear = enriched[(~enriched["independent_null"]) & (enriched["sigma"] == 1.0)].groupby("delta", as_index=False)["Psi"].median()
    save_line_figure(linear, x="delta", y="Psi", group="delta", title="Affine dependence of process-order gap on delta", output_stem=ROOT / "experiments/outputs/figures/Fig_dynamic_delta_linearity")
    digest = f"# Dynamic process-order reanalysis\n\n- The maximum affine decomposition residual is {max_error:.3e}. [source: dynamic_process_decomposition.csv, all proposed process-order rows]\n- Legacy signed GDA is retained only for backward compatibility and is descriptive, not a success criterion. [source: synthetic_dynamic_raw.parquet, legacy Psi sign aggregation]\n"
    (ROOT / "experiments/outputs/paper/dynamic_reanalysis_digest.md").write_text(digest, encoding="utf-8")
    print(f"Process decomposition passed; max error={max_error:.3e}.")


if __name__ == "__main__": main()
