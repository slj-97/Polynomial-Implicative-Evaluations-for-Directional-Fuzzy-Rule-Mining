"""Exploratory known-order directional ranking pilot with a frozen protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from _common import ROOT
from experiments.src.fuzzification import RobustScaler, chronological_split, triangular_memberships
from experiments.src.operators import eval_minimum, eval_poly, eval_power, eval_product
from experiments.src.reproducibility import load_yaml, runtime_manifest, write_dataframe, write_json
from experiments.src.statistics import holm_adjust


def _transform(values: np.ndarray, mechanism: str) -> np.ndarray:
    if mechanism == "linear":
        return values
    if mechanism == "tanh":
        return np.tanh(values)
    if mechanism == "saturation":
        return values / (1.0 + np.abs(values))
    raise ValueError(f"Unknown mechanism: {mechanism}")


def generate_known_order_process(
    rng: np.random.Generator,
    *,
    phi: float,
    rho: float,
    lag: int,
    coupling: float,
    response_noise_to_signal: float,
    length: int,
    burn_in: int,
    mechanism: str,
    independent_null: bool,
) -> tuple[pd.DataFrame, float]:
    """Generate X before Y while holding source innovation variance fixed."""
    total = length + burn_in + lag
    x = np.zeros(total, dtype=np.float64)
    y = np.zeros(total, dtype=np.float64)
    source_innovation = rng.normal(0.0, 1.0, total)
    response_standard_normal = rng.normal(0.0, 1.0, total)
    for index in range(1, total):
        x[index] = phi * x[index - 1] + source_innovation[index]
    driving = np.zeros(total, dtype=np.float64)
    driving[lag:] = coupling * _transform(x[:-lag], mechanism)
    active = driving[burn_in + lag :]
    signal_sd = float(np.std(active, ddof=1))
    if not np.isfinite(signal_sd) or signal_sd <= 0.0:
        raise RuntimeError("The generated driving signal has zero or invalid variance.")
    response_noise_sd = response_noise_to_signal * signal_sd
    for index in range(1, total):
        signal = 0.0 if independent_null or index < lag else driving[index]
        y[index] = rho * y[index - 1] + signal + response_noise_sd * response_standard_normal[index]
    start = burn_in + lag
    return pd.DataFrame({"X": x[start:], "Y": y[start:]}), response_noise_sd


def _test_memberships(data: pd.DataFrame, labels: int) -> dict[str, np.ndarray]:
    training, _, test = chronological_split(data, (0.6, 0.2, 0.2))
    scaled_test = RobustScaler.fit(training).transform(test)
    return triangular_memberships(scaled_test, labels=labels)


def _evaluations(config: dict[str, object]):
    sigma = float(config["sigma"])
    result = [
        ("product", np.nan, eval_product),
        ("minimum", np.nan, eval_minimum),
        ("power-0.01", np.nan, lambda x, y: eval_power(x, y, 0.01)),
        ("power-2", np.nan, lambda x, y: eval_power(x, y, 2.0)),
    ]
    for value in config["deltas"]:
        delta = float(value)
        result.append((f"proposed-delta-{delta:g}", delta, lambda x, y, d=delta: eval_poly(x, y, sigma, d)))
    return result


def _support(evaluation, antecedent: np.ndarray, consequent: np.ndarray, lag: int) -> float:
    return float(np.mean(evaluation(antecedent[:-lag], consequent[lag:])))


def _score_aggregations(
    memberships: dict[str, np.ndarray],
    evaluation,
    lag: int,
    swapped: bool,
    labels: int,
) -> dict[str, float]:
    presented_u = "Y" if swapped else "X"
    presented_v = "X" if swapped else "Y"
    true_source, true_target = (presented_v, presented_u) if swapped else (presented_u, presented_v)

    margins: list[float] = []
    all_margins: list[float] = []
    for source_label in range(labels):
        for target_label in range(labels):
            source = memberships[f"{true_source}::{source_label}"]
            target = memberships[f"{true_target}::{target_label}"]
            forward = _support(evaluation, source, target, lag)
            reverse = _support(evaluation, target, source, lag)
            margin = forward - reverse
            all_margins.append(margin)
            if source_label == target_label:
                margins.append(margin)
    return {
        "margin_high_high": margins[-1],
        "margin_same_label": float(np.mean(margins)),
        "margin_all_pairs": float(np.mean(all_margins)),
    }


def _ranking_value(margin: pd.Series | np.ndarray, tolerance: float) -> np.ndarray:
    values = np.asarray(margin, dtype=np.float64)
    return np.where(values > tolerance, 1.0, np.where(values < -tolerance, 0.0, 0.5))


def _bootstrap_mean_ci(values: np.ndarray, seed: int, repetitions: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repetitions, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _accuracy_summary(frame: pd.DataFrame, config: dict[str, object], seed: int) -> pd.DataFrame:
    tolerance = float(config["tie_tolerance"])
    repetitions = int(config["bootstrap_repetitions"])
    rows: list[dict[str, object]] = []
    columns = ["independent_null", "mechanism", "lag", "response_noise_to_signal", "method", "delta"]
    for key, subset in frame.groupby(columns, dropna=False):
        accuracy = _ranking_value(subset["margin_high_high"], tolerance)
        low, high = _bootstrap_mean_ci(accuracy, seed + len(rows), repetitions)
        row = dict(zip(columns, key, strict=True))
        row.update({
            "n": int(accuracy.size),
            "accuracy": float(accuracy.mean()),
            "accuracy_ci_low": low,
            "accuracy_ci_high": high,
            "tie_rate": float(np.mean(accuracy == 0.5)),
            "margin_median": float(subset["margin_high_high"].median()),
            "margin_iqr": float(subset["margin_high_high"].quantile(0.75) - subset["margin_high_high"].quantile(0.25)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_contrasts(frame: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    tolerance = float(config["tie_tolerance"])
    positive = frame[(~frame["independent_null"]) & (frame["method"].str.startswith("proposed"))].copy()
    positive["accuracy_value"] = _ranking_value(positive["margin_high_high"], tolerance)
    keys = ["mechanism", "lag", "response_noise_to_signal", "replication", "seed", "swapped"]
    baseline = positive[positive["delta"] == 0.0][keys + ["accuracy_value"]].rename(columns={"accuracy_value": "baseline_accuracy"})
    rows: list[dict[str, object]] = []
    for delta in sorted(value for value in positive["delta"].dropna().unique() if value != 0.0):
        current = positive[positive["delta"] == delta].merge(baseline, on=keys, validate="one_to_one")
        current["gain"] = current["accuracy_value"] - current["baseline_accuracy"]
        non_tied = current[(current["accuracy_value"] != 0.5) & (current["baseline_accuracy"] != 0.5)]
        improved = int(((non_tied["accuracy_value"] == 1.0) & (non_tied["baseline_accuracy"] == 0.0)).sum())
        worsened = int(((non_tied["accuracy_value"] == 0.0) & (non_tied["baseline_accuracy"] == 1.0)).sum())
        discordant = improved + worsened
        p_value = float(binomtest(improved, discordant, 0.5).pvalue) if discordant else 1.0
        row: dict[str, object] = {
            "delta": delta,
            "n": len(current),
            "accuracy": float(current["accuracy_value"].mean()),
            "baseline_accuracy": float(current["baseline_accuracy"].mean()),
            "paired_accuracy_gain": float(current["gain"].mean()),
            "improved_discordant": improved,
            "worsened_discordant": worsened,
            "exact_mcnemar_p": p_value,
        }
        for mechanism in ["tanh", "saturation", "linear"]:
            part = current[current["mechanism"] == mechanism]
            row[f"gain_{mechanism}"] = float(part["gain"].mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    result["holm_p"] = holm_adjust(result["exact_mcnemar_p"].to_numpy())
    null = frame[(frame["independent_null"]) & (frame["method"].str.startswith("proposed"))].copy()
    null["accuracy_value"] = _ranking_value(null["margin_high_high"], tolerance)
    null_accuracy = null.groupby("delta", dropna=False)["accuracy_value"].mean()
    result["null_accuracy"] = result["delta"].map(null_accuracy)
    result["null_deviation"] = np.abs(result["null_accuracy"] - 0.5)
    rule = config["decision_rule"]
    result["passes_frozen_rule"] = (
        (result["paired_accuracy_gain"] >= float(rule["minimum_overall_accuracy_gain"]))
        & (result["gain_tanh"] > 0.0)
        & (result["gain_saturation"] > 0.0)
        & (result["holm_p"] < float(rule["holm_alpha"]))
        & (result["null_deviation"] <= float(rule["maximum_null_deviation_from_chance"]))
    )
    return result


def _make_figures(frame: pd.DataFrame, contrasts: pd.DataFrame, output: Path, tolerance: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    positive = frame[(~frame["independent_null"]) & (frame["mechanism"].isin(["tanh", "saturation"]))].copy()
    positive["accuracy_value"] = _ranking_value(positive["margin_high_high"], tolerance)
    selected = positive[positive["method"].isin(["proposed-delta--1", "proposed-delta-0", "proposed-delta-1", "product"])]
    plot = selected.groupby(["response_noise_to_signal", "method"])["accuracy_value"].agg(["mean", "sem"]).reset_index()
    colors = {"proposed-delta--1": "#0072B2", "proposed-delta-0": "#000000", "proposed-delta-1": "#D55E00", "product": "#009E73"}
    labels = {"proposed-delta--1": r"$E_{1,-1}$", "proposed-delta-0": r"$E_{1,0}$", "proposed-delta-1": r"$E_{1,1}$", "product": "Product"}
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for method, part in plot.groupby("method"):
        part = part.sort_values("response_noise_to_signal")
        ci = 1.96 * part["sem"]
        ax.plot(part["response_noise_to_signal"], part["mean"], marker="o", linewidth=1.7, color=colors[method], label=labels[method])
        ax.fill_between(part["response_noise_to_signal"], part["mean"] - ci, part["mean"] + ci, color=colors[method], alpha=0.16)
    ax.axhline(0.5, color="0.55", linestyle=":", linewidth=1.2)
    ax.set(xlabel="Response noise / driving-signal SD", ylabel="Directional ranking accuracy", ylim=(0.0, 1.0))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "figure-01-ranking-vs-noise.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(len(contrasts))
    ax.axhline(0.0, color="0.45", linewidth=1.1)
    ax.bar(x - 0.2, contrasts["gain_tanh"], width=0.2, color="#0072B2", label="tanh")
    ax.bar(x, contrasts["gain_saturation"], width=0.2, color="#E69F00", label="saturation")
    ax.bar(x + 0.2, contrasts["gain_linear"], width=0.2, color="#009E73", label="linear")
    ax.set_xticks(x, [f"{value:g}" for value in contrasts["delta"]])
    ax.set(xlabel=r"Directional parameter $\delta$", ylabel=r"Accuracy gain over $E_{1,0}$")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output / "figure-02-paired-ablation.pdf")
    plt.close(fig)


def _write_reports(frame: pd.DataFrame, summary: pd.DataFrame, contrasts: pd.DataFrame, output: Path) -> None:
    passed = contrasts[contrasts["passes_frozen_rule"]]
    proposed = frame[frame["method"].str.startswith("proposed")]
    run_keys = [
        "independent_null", "mechanism", "lag", "response_noise_to_signal",
        "replication", "seed", "swapped",
    ]
    symmetric = proposed[proposed["delta"] == 0.0][run_keys + ["margin_all_pairs"]].rename(
        columns={"margin_all_pairs": "symmetric_all_pair_margin"}
    )
    nonsymmetric = proposed[proposed["delta"] != 0.0].merge(
        symmetric, on=run_keys, validate="many_to_one"
    )
    all_pair_increment_max = float(
        (nonsymmetric["margin_all_pairs"] - nonsymmetric["symmetric_all_pair_margin"]).abs().max()
    )
    decision = "advance to independent confirmation" if not passed.empty else "retain the current manuscript positioning"
    report = f"""# Ground-truth directional pilot: analysis report

## Analysis question

Does a fixed nonsymmetric member of the proposed polynomial family improve held-out
High--High directional ranking relative to the matched symmetric member $E_{{1,0}}$
when the data-generating process has a documented positive-lag order?

## Evidence status

- Development realizations: {int((~frame['independent_null']).sum() / frame[~frame['independent_null']]['method'].nunique())} per method across the complete positive-order grid.
- Unit of analysis: independently generated realization; operator comparisons are paired within realization.
- Primary metric: High--High directional ranking accuracy; ties count as one half.
- The maximum absolute operator-dependent change in the complete-label-pair mean margin was {all_pair_increment_max:.3e}, confirming cancellation to numerical precision.
- Frozen-rule decision: **{decision}**.

## Primary contrasts

{contrasts.to_markdown(index=False, floatfmt='.4g')}

## Claim candidates

- Claim:
  - Source evidence: `paired-contrasts.csv` and `figure-02-paired-ablation.pdf`.
  - Allowed wording: This exploratory known-order pilot {'identified a parameter candidate for independent confirmation' if not passed.empty else 'did not identify a fixed nonsymmetric parameter satisfying the pre-specified advancement rule'}.
  - Forbidden stronger wording: the evaluation recovers causal direction; the best inspected parameter is validated; nonsymmetry universally improves direction recovery.
  - Uncertainty: this is a development experiment on simulated autoregressive systems with a known lag.
  - Next check: {'run a fresh-seed confirmation with a held-out mechanism' if not passed.empty else 'do not add a direction-recovery claim; retain the operator-sensitivity interpretation'}.
  - Decision: {'revise' if not passed.empty else 'keep'}.

## Limitations

The experiment supplies the true lag and uses a pre-specified High--High semantic
pair. It does not establish causal identifiability, generalize to arbitrary fuzzy
labels, or substitute for validation on a documented physical process.
"""
    (output / "analysis-report.md").write_text(report, encoding="utf-8")
    stats = f"""# Statistical appendix

- Independent unit: one simulated realization.
- Repetitions: 50 per mechanism, lag, and response-noise level.
- Accuracy intervals: percentile bootstrap of the mean with 2000 resamples.
- Paired test: exact McNemar/binomial test on discordant success decisions between
  each nonzero-delta member and $E_{{1,0}}$.
- Multiplicity: Holm correction across the four pre-specified nonzero deltas.
- Effect size: paired change in directional ranking accuracy.
- Ties: margins within the configured numerical tolerance count as 0.5.

## Accuracy summaries

{summary.to_markdown(index=False, floatfmt='.4g')}

## Paired contrasts

{contrasts.to_markdown(index=False, floatfmt='.4g')}
"""
    (output / "stats-appendix.md").write_text(stats, encoding="utf-8")
    catalog = """# Figure catalog

## figure-01-ranking-vs-noise.pdf

- Purpose: determine whether ranking accuracy remains above chance as response noise increases.
- Data: nonlinear positive-order systems, pooled over true lags and independent realizations.
- Display: mean accuracy with approximate 95% normal intervals; the dotted line is chance.
- Interpretation check: compare each nonsymmetric curve with $E_{1,0}$, not only with chance.
- Caveat: intervals are descriptive across the pooled development grid.

## figure-02-paired-ablation.pdf

- Purpose: show whether a fixed delta improves accuracy over the matched symmetric member.
- Data: paired realization-level accuracy differences, summarized separately by mechanism.
- Observation to inspect: sign consistency across tanh and saturation, not the largest single bar.
- Decision consequence: only a delta satisfying the frozen rule can proceed to confirmation.
- Caveat: this figure is exploratory and is not manuscript-ready confirmatory evidence.
"""
    (output / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    common = load_yaml(ROOT / "experiments/configs/common.yaml")
    config = load_yaml(ROOT / "experiments/configs/ground_truth_directional_pilot.yaml")
    replications = 2 if args.smoke else int(config["replications"])
    mechanisms = config["mechanisms"][:1] if args.smoke else config["mechanisms"]
    noises = config["response_noise_to_signal"][:1] if args.smoke else config["response_noise_to_signal"]
    lags = config["lags"][:1] if args.smoke else config["lags"]
    rows: list[dict[str, object]] = []
    for independent_null in ([False, True] if config["include_independent_null"] else [False]):
        for mechanism_index, mechanism in enumerate(mechanisms):
            for lag in lags:
                for noise_index, noise_ratio in enumerate(noises):
                    for replication in range(replications):
                        seed = int(common["seed"] + 10_000_000 * independent_null + 100_000 * mechanism_index + 10_000 * lag + 100 * noise_index + replication)
                        rng = np.random.default_rng(seed)
                        data, response_noise_sd = generate_known_order_process(
                            rng,
                            phi=float(config["phi"]), rho=float(config["rho"]), lag=int(lag),
                            coupling=float(config["coupling"]), response_noise_to_signal=float(noise_ratio),
                            length=int(config["length"]), burn_in=int(config["burn_in"]),
                            mechanism=str(mechanism), independent_null=independent_null,
                        )
                        swapped = bool(rng.integers(0, 2))
                        memberships = _test_memberships(data, int(config["label_count"]))
                        for method, delta, evaluation in _evaluations(config):
                            scores = _score_aggregations(memberships, evaluation, int(lag), swapped, int(config["label_count"]))
                            rows.append({
                                "seed": seed, "replication": replication, "independent_null": independent_null,
                                "mechanism": mechanism, "lag": lag, "response_noise_to_signal": noise_ratio,
                                "response_noise_sd": response_noise_sd, "swapped": swapped, "method": method,
                                "sigma": float(config["sigma"]) if method.startswith("proposed") else np.nan,
                                "delta": delta, **scores,
                            })
    frame = pd.DataFrame(rows)
    output = ROOT / "experiments/outputs/ground_truth_pilot"
    output.mkdir(parents=True, exist_ok=True)
    write_dataframe(frame, "experiments/outputs/ground_truth_pilot/raw_results.parquet")
    summary = _accuracy_summary(frame, config, int(common["seed"]))
    contrasts = _paired_contrasts(frame, config)
    write_dataframe(summary, "experiments/outputs/ground_truth_pilot/accuracy-summary.csv")
    write_dataframe(contrasts, "experiments/outputs/ground_truth_pilot/paired-contrasts.csv")
    _make_figures(frame, contrasts, output / "figures", float(config["tie_tolerance"]))
    _write_reports(frame, summary, contrasts, output)
    manifest = runtime_manifest(int(common["seed"]))
    manifest.update({
        "smoke": args.smoke, "rows": len(frame), "data_realizations": int(frame.groupby(["independent_null", "mechanism", "lag", "response_noise_to_signal", "replication"]).ngroups),
        "frozen_protocol": "experiments/ground_truth_pilot_protocol.md",
        "passed_deltas": contrasts.loc[contrasts["passes_frozen_rule"], "delta"].tolist(),
    })
    write_json(manifest, "experiments/outputs/ground_truth_pilot/manifest.json")
    print(json.dumps({"rows": len(frame), "passed_deltas": manifest["passed_deltas"], "smoke": args.smoke}))


if __name__ == "__main__":
    main()
