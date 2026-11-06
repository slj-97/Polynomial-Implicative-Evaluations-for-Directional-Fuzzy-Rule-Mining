"""Audit the label-pair origin of the dynamic process-order increment.

This script regenerates only deterministic intermediate memberships from the
frozen dynamic-synthetic seeds.  It does not add experiment conditions or
replace the raw experiment file.  The saved raw process-order supports are
checked against the regenerated label-pair mean before any summaries are
written.
"""
from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import ROOT
from experiments.src.fuzzification import RobustScaler, chronological_split, triangular_memberships
from experiments.src.reproducibility import load_yaml, write_dataframe
from experiments.src.synthetic import dynamic_process


LABEL_COUNT = 3
FLOAT_GATE = 1.0e-10


def _memberships(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train, _, test = chronological_split(data, (0.6, 0.2, 0.2))
    memberships = triangular_memberships(RobustScaler.fit(train).transform(test), labels=LABEL_COUNT)
    return (
        np.stack([memberships[f"X::{index}"] for index in range(LABEL_COUNT)]),
        np.stack([memberships[f"Y::{index}"] for index in range(LABEL_COUNT)]),
    )


def _process_terms(x_labels: np.ndarray, y_labels: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the 3x3 pairwise coefficients for Psi_0 and Lambda."""
    forward_x = x_labels[:, None, :-lag]
    forward_y = y_labels[None, :, lag:]
    reverse_x = y_labels[None, :, :-lag]
    reverse_y = x_labels[:, None, lag:]

    def means(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boundary = left * right * (1.0 - left) * (1.0 - right)
        return (
            np.mean(left * right, axis=2),
            np.mean(boundary * (left + right), axis=2),
            np.mean(boundary * (left - right), axis=2),
        )

    forward_base, forward_symmetric, forward_antisymmetric = means(forward_x, forward_y)
    reverse_base, reverse_symmetric, reverse_antisymmetric = means(reverse_x, reverse_y)
    return (
        forward_base - reverse_base,
        forward_symmetric - reverse_symmetric,
        0.5 * (forward_antisymmetric - reverse_antisymmetric),
    )


def _reconstruct_dataset(task: tuple[object, object, object]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Regenerate one process and return compact per-row audit coefficients."""
    first, lag_specs, config = task
    data = dynamic_process(
        np.random.default_rng(int(first["seed"])), phi=float(config["phi"]), rho=float(config["rho"]),
        lag=int(first["h_true"]), coupling=float(first["lambda"]), noise=float(first["noise"]),
        length=int(config["length"]), burn_in=int(config["burn_in"]), function=str(first["mechanism"]),
        independent_null=bool(first["independent_null"]),
    )
    x_labels, y_labels = _memberships(data)
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for h_eval, sigma_specs in lag_specs:
        base_difference, symmetric_difference, lambda_pair_matrix = _process_terms(x_labels, y_labels, h_eval)
        lambda_flat = lambda_pair_matrix.reshape(-1)
        for positions, sigma, deltas, saved in sigma_specs:
            psi0_flat = (base_difference - 0.5 * sigma * symmetric_difference).reshape(-1)
            pair_omega = deltas[:, None] * lambda_flat[None, :]
            reconstructed = psi0_flat.mean() + pair_omega.mean(axis=1)
            results.append((positions, np.broadcast_to(psi0_flat, pair_omega.shape).copy(), pair_omega, np.broadcast_to(lambda_flat, pair_omega.shape).copy(), np.abs(saved - reconstructed)))
    return results


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    finite = np.asarray(list(values), dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "median": np.nan, "iqr": np.nan, "q75": np.nan, "q90": np.nan, "q95": np.nan, "q999": np.nan, "max": np.nan, "mean": np.nan}
    quantiles = np.quantile(finite, [0.25, 0.5, 0.75, 0.9, 0.95, 0.999])
    return {
        "n": int(finite.size),
        "median": float(quantiles[1]),
        "iqr": float(quantiles[2] - quantiles[0]),
        "q75": float(quantiles[2]),
        "q90": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "q999": float(quantiles[5]),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _append_summary(rows: list[dict[str, object]], *, analysis: str, level: str, filter_name: str, metric: str, values: Iterable[float], **extra: object) -> None:
    rows.append({"analysis": analysis, "level": level, "filter": filter_name, "metric": metric, **extra, **_summary(values)})


def _seed_first(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["independent_null", "mechanism", "h_true", "h_eval", "lambda", "noise", "sigma", "delta"]
    return frame.groupby(group_columns, as_index=False, dropna=False)[["Psi", "Psi0", "Omega", "Lambda", "abs_Psi", "abs_Psi0", "abs_Omega", "abs_Lambda"]].median()


def _write_membership_and_pruning_checks() -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    membership = pd.read_parquet(ROOT / "experiments/outputs/raw_results/synthetic_membership_raw.parquet")
    proposed = membership[membership["evaluation"] == "proposed"].copy()
    generated = proposed[proposed["control"] == "generated"].copy()
    table_one = {
        "max_identity_error": float(proposed["identity_error"].max()),
        "diagonal_max_absolute_delta": float(proposed[proposed["control"] == "diagonal"]["absolute_delta"].max()),
        "exchangeable_median_delta": float(proposed[proposed["control"] == "exchangeable"]["support_delta"].median()),
    }
    eta_seed_first = generated.groupby(["kappa", "eta", "n_records", "sigma", "delta"], as_index=False)["absolute_delta"].median()
    table_one.update({f"eta_{eta:g}": float(part["absolute_delta"].median()) for eta, part in eta_seed_first.groupby("eta")})

    selected = generated[(generated["kappa"] == 2.0) & (generated["eta"] == 0.05) & (generated["n_records"] == 10000) & (generated["sigma"] == 1.0)].copy()
    selected["delta_Gamma"] = selected["delta"] * selected["gamma"]
    table_three = selected.groupby("delta", as_index=False).agg(median_Delta=("support_delta", "median"), median_delta_Gamma=("delta_Gamma", "median"), max_identity_error=("identity_error", "max"))
    spearman_frame = generated.groupby(["kappa", "eta", "n_records", "sigma", "delta"], as_index=False).agg(operator_D=("operator_D", "first"), absolute_delta=("absolute_delta", "median"))
    table_three_check = {
        "max_seed_identity_error": float(selected["identity_error"].max()),
        "max_median_difference": float(np.max(np.abs(table_three["median_Delta"] - table_three["median_delta_Gamma"]))),
        "spearman_seed_first": float(spearmanr(spearman_frame["operator_D"], spearman_frame["absolute_delta"]).statistic),
        "spearman_units": int(len(spearman_frame)),
    }

    pruning = pd.read_csv(ROOT / "experiments/outputs/raw_results/pruning_our.csv")
    joint = pruning[pruning["pruning_mode"] == "coverage_support"]
    table_four = {
        "all_retained_sets_equal": bool(pruning["retained_rules_match_exhaustive"].all()),
        "max_pruning_ratio": float(pruning["pruning_ratio"].max()),
        "joint_scored_candidates": sorted(int(value) for value in joint["scored_candidates"].unique()),
        "support_pruned_branches": sorted(int(value) for value in joint["support_pruned_branches"].unique()),
        "maximum_depth_values": sorted(int(value) for value in joint["max_antecedent_size"].unique()),
        "retained_rules": sorted(int(value) for value in joint["retained_rules"].unique()),
    }
    return table_one, table_three_check, {"table_three": table_three, "table_four": table_four}


def main() -> None:
    config = load_yaml(ROOT / "experiments/configs/synthetic_dynamic.yaml")
    raw = pd.read_parquet(ROOT / "experiments/outputs/raw_results/synthetic_dynamic_raw.parquet")
    process = raw[(raw["protocol"] == "process_order") & (raw["operator"] == "proposed")].copy()
    del raw
    data_columns = ["independent_null", "seed", "mechanism", "h_true", "lambda", "noise"]
    process = process.sort_values(data_columns + ["h_eval", "sigma", "delta"]).reset_index(drop=True)
    n_rows = len(process)
    psi0_pairs = np.empty((n_rows, LABEL_COUNT * LABEL_COUNT), dtype=np.float64)
    omega_pairs = np.empty_like(psi0_pairs)
    lambda_pairs = np.empty_like(psi0_pairs)
    decomposition_abs_error = np.empty(n_rows, dtype=np.float64)
    reconstruction_abs_error = np.empty(n_rows, dtype=np.float64)

    tasks = []
    for _, dataset_rows in process.groupby(data_columns, sort=False, dropna=False):
        lag_specs = []
        for h_eval, lag_rows in dataset_rows.groupby("h_eval", sort=False):
            sigma_specs = []
            for sigma, sigma_rows in lag_rows.groupby("sigma", sort=False):
                sigma_specs.append((sigma_rows.index.to_numpy(dtype=int), float(sigma), sigma_rows["delta"].to_numpy(dtype=np.float64), sigma_rows["Psi"].to_numpy(dtype=np.float64)))
            lag_specs.append((int(h_eval), sigma_specs))
        tasks.append((dataset_rows.iloc[0][data_columns].to_dict(), lag_specs, config))
    worker_count = min(24, os.cpu_count() or 1)
    print(f"Reconstructing {len(tasks)} frozen dynamic processes with {worker_count} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_reconstruct_dataset, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            for positions, psi0_values, omega_values, lambda_values, error in future.result():
                psi0_pairs[positions] = psi0_values
                omega_pairs[positions] = omega_values
                lambda_pairs[positions] = lambda_values
                decomposition_abs_error[positions] = error
                reconstruction_abs_error[positions] = error
            if completed % 1000 == 0:
                print(f"Reconstructed {completed}/{len(tasks)} processes.", flush=True)

    if float(np.max(decomposition_abs_error)) > FLOAT_GATE:
        raise RuntimeError(f"[IMPLEMENTATION ISSUE] process-order decomposition error exceeds {FLOAT_GATE:.1e}: {np.max(decomposition_abs_error):.3e}")

    derived = process[["independent_null", "mechanism", "h_true", "h_eval", "lambda", "noise", "sigma", "delta", "Psi"]].copy()
    derived["Psi0"] = psi0_pairs.mean(axis=1)
    derived["Omega"] = omega_pairs.mean(axis=1)
    derived["Lambda"] = lambda_pairs.mean(axis=1)
    derived["abs_Psi"] = derived["Psi"].abs()
    derived["abs_Psi0"] = derived["Psi0"].abs()
    derived["abs_Omega"] = derived["Omega"].abs()
    derived["abs_Lambda"] = derived["Lambda"].abs()
    derived["abs_delta"] = derived["delta"].abs()
    derived["decomposition_abs_error"] = decomposition_abs_error
    seed_first = _seed_first(derived)
    seed_first["abs_delta"] = seed_first["delta"].abs()

    nonzero = derived[(derived["delta"] != 0.0) & (~derived["independent_null"])].copy()
    seed_first_nonzero = seed_first[(seed_first["delta"] != 0.0) & (~seed_first["independent_null"])].copy()
    label_mask = nonzero.index.to_numpy(dtype=int)
    label_abs_omega = np.abs(omega_pairs[label_mask]).reshape(-1)
    label_abs_lambda = np.abs(lambda_pairs[label_mask]).reshape(-1)
    label_omega = omega_pairs[label_mask].reshape(-1)

    operator_rows: list[dict[str, object]] = []
    _append_summary(operator_rows, analysis="decomposition", level="aggregated", filter_name="all proposed process-order rows", metric="absolute error", values=decomposition_abs_error)
    _append_summary(operator_rows, analysis="increment", level="A individual label pair", filter_name="delta != 0", metric="abs_Omega", values=label_abs_omega)
    _append_summary(operator_rows, analysis="increment", level="A individual label pair", filter_name="delta != 0", metric="abs_Lambda", values=label_abs_lambda)
    for level, frame in [("B within-seed 3x3 mean", nonzero), ("C seed-first configuration median", seed_first_nonzero)]:
        _append_summary(operator_rows, analysis="increment", level=level, filter_name="delta != 0", metric="abs_Omega", values=frame["abs_Omega"])
        _append_summary(operator_rows, analysis="increment", level=level, filter_name="delta != 0", metric="abs_Lambda", values=frame["abs_Lambda"])

    factor_columns = ["lambda", "noise", "mechanism", "h_true", "h_eval", "abs_delta", "sigma"]
    for factor in factor_columns:
        for value, part in seed_first_nonzero.groupby(factor, dropna=False):
            row: dict[str, object] = {
                "analysis": "factor", "level": "C seed-first configuration median", "filter": "non-null; delta != 0", "metric": "multiple", "factor": factor, "factor_value": value, "n": int(len(part)),
                "median_abs_Psi": float(part["abs_Psi"].median()), "median_abs_Psi0": float(part["abs_Psi0"].median()), "median_abs_Omega": float(part["abs_Omega"].median()), "median_abs_Lambda": float(part["abs_Lambda"].median()), "q95_abs_Omega": float(part["abs_Omega"].quantile(0.95)),
            }
            operator_rows.append(row)
    write_dataframe(pd.DataFrame(operator_rows), "experiments/outputs/summaries/dynamic_operator_increment_audit.csv")

    label_rows: list[dict[str, object]] = []
    pair_indices = [(left, right) for left in range(LABEL_COUNT) for right in range(LABEL_COUNT)]
    for index, (antecedent_label, consequent_label) in enumerate(pair_indices):
        values = omega_pairs[label_mask, index]
        row = {
            "analysis": "label_pair", "level": "A individual label pair", "filter": "delta != 0", "antecedent_label": antecedent_label, "consequent_label": consequent_label,
            "positive_count": int(np.count_nonzero(values > 0.0)), "negative_count": int(np.count_nonzero(values < 0.0)), "zero_count": int(np.count_nonzero(values == 0.0)),
            "signed_sum": float(np.sum(values)), "signed_mean": float(np.mean(values)), "signed_median": float(np.median(values)),
            **{f"abs_Omega_{key}": value for key, value in _summary(np.abs(values)).items()},
            **{f"abs_Lambda_{key}": value for key, value in _summary(np.abs(lambda_pairs[label_mask, index])).items()},
        }
        label_rows.append(row)
    for level, values in [("A individual label pair", label_omega), ("B within-seed 3x3 mean", nonzero["Omega"].to_numpy()), ("C seed-first configuration median", seed_first_nonzero["Omega"].to_numpy())]:
        label_rows.append({
            "analysis": "sign_diagnostic", "level": level, "filter": "delta != 0", "antecedent_label": "all", "consequent_label": "all",
            "positive_count": int(np.count_nonzero(values > 0.0)), "negative_count": int(np.count_nonzero(values < 0.0)), "zero_count": int(np.count_nonzero(values == 0.0)),
            "signed_sum": float(np.sum(values)), "signed_mean": float(np.mean(values)), "signed_median": float(np.median(values)),
            **{f"abs_Omega_{key}": value for key, value in _summary(np.abs(values)).items()},
        })
    write_dataframe(pd.DataFrame(label_rows), "experiments/outputs/summaries/dynamic_labelpair_audit.csv")

    delta_rows: list[dict[str, object]] = []
    for abs_delta, part in seed_first_nonzero.groupby("abs_delta", dropna=False):
        scale_error = np.abs(part["abs_Omega"] - float(abs_delta) * part["abs_Lambda"])
        for metric, values in [("abs_Omega", part["abs_Omega"]), ("abs_Lambda", part["abs_Lambda"]), ("abs_Omega - abs_delta*abs_Lambda", scale_error)]:
            _append_summary(delta_rows, analysis="delta_scaling", level="C seed-first configuration median", filter_name="non-null; delta != 0", metric=metric, values=values, abs_delta=float(abs_delta))
    write_dataframe(pd.DataFrame(delta_rows), "experiments/outputs/summaries/dynamic_delta_audit.csv")

    null_rows: list[dict[str, object]] = []
    for scope_name, scope in [("all delta", seed_first), ("delta != 0", seed_first[seed_first["delta"] != 0.0])]:
        nonnull_scope = scope[~scope["independent_null"]]
        for strength, part in nonnull_scope.groupby("lambda", dropna=False):
            for metric, values in [("Psi", part["Psi"]), ("abs_Psi", part["abs_Psi"]), ("abs_Omega", part["abs_Omega"])]:
                _append_summary(null_rows, analysis="null_comparison", level="C seed-first configuration median", filter_name=scope_name, metric=metric, values=values, condition=f"lambda={strength:g}")
        null_scope = scope[scope["independent_null"]]
        for metric, values in [("Psi", null_scope["Psi"]), ("abs_Psi", null_scope["abs_Psi"]), ("abs_Omega", null_scope["abs_Omega"])]:
            _append_summary(null_rows, analysis="null_comparison", level="C seed-first configuration median", filter_name=scope_name, metric=metric, values=values, condition="independent_AR_null")
    write_dataframe(pd.DataFrame(null_rows), "experiments/outputs/summaries/dynamic_null_audit.csv")

    table_one, table_three, additional = _write_membership_and_pruning_checks()
    q95_label = float(np.quantile(label_abs_omega, 0.95))
    q95_aggregate = float(seed_first_nonzero["abs_Omega"].quantile(0.95))
    if q95_label > 1.0e-12 and q95_aggregate <= 1.0e-12:
        case = "A. Label-pair aggregation suppresses the net operator-dependent process-order increment through cancellation."
    elif q95_label <= 1.0e-12:
        case = "B. The individual label-pair operator response is itself close to floating-point precision."
    else:
        case = "D. The aggregate operator-dependent increment is not uniformly near zero under the audited delta != 0 scope."

    all_table_two = seed_first.groupby("independent_null", as_index=False)[["Psi", "abs_Psi", "abs_Omega", "abs_Lambda"]].median()
    audit_markdown = "\n".join([
        "# Dynamic Table-II audit", "",
        "## Provenance", "",
        "- Source file: `experiments/outputs/raw_results/synthetic_dynamic_raw.parquet`.",
        "- Filters: `protocol=process_order`, `operator=proposed`; each saved row is the prescribed mean over the 3x3 triangular fuzzy-label pairs within one seed/configuration.",
        "- Reconstruction: the frozen seed, process parameters, training-only robust scaling, and three-label fuzzification were replayed without changing the grid.",
        "", "## Decomposition checks", "",
        f"- Maximum `|Psi - Psi0 - delta Lambda|`: `{np.max(decomposition_abs_error):.3e}`.",
        f"- Median absolute decomposition error: `{np.median(decomposition_abs_error):.3e}`.",
        f"- 99.9th percentile absolute decomposition error: `{np.quantile(decomposition_abs_error, 0.999):.3e}`.",
        f"- Maximum saved-versus-reconstructed 3x3-mean support error: `{np.max(reconstruction_abs_error):.3e}`.",
        "", "## Why the reported Table-II increment is near machine zero", "",
        f"**Classification: {case}**", "",
        f"- Individual label pairs, `delta != 0`: median `|Omega|={np.median(label_abs_omega):.3e}`, Q95 `{q95_label:.3e}`, maximum `{np.max(label_abs_omega):.3e}`.",
        f"- Prescribed within-seed 3x3 mean followed by seed-first aggregation, `delta != 0`: median `|Omega|={seed_first_nonzero['abs_Omega'].median():.3e}`, Q95 `{q95_aggregate:.3e}`, maximum `{seed_first_nonzero['abs_Omega'].max():.3e}`.",
        f"- Individual-pair signed counts: positive `{np.count_nonzero(label_omega > 0.0)}`, negative `{np.count_nonzero(label_omega < 0.0)}`, zero `{np.count_nonzero(label_omega == 0.0)}`.",
        f"- Individual-pair signed mean/median: `{np.mean(label_omega):.3e}` / `{np.median(label_omega):.3e}`; within-seed 3x3-mean signed mean/median: `{nonzero['Omega'].mean():.3e}` / `{nonzero['Omega'].median():.3e}`.",
        "", "## Direct-swap and pruning cross-checks", "",
        f"- Table I seed-first eta medians: eta=0 `{table_one['eta_0']:.3e}`, eta=0.02 `{table_one['eta_0.02']:.3e}`, eta=0.05 `{table_one['eta_0.05']:.3e}`, eta=0.1 `{table_one['eta_0.1']:.3e}`.",
        f"- Table III maximum seed identity error: `{table_three['max_seed_identity_error']:.3e}`; maximum difference between reported medians: `{table_three['max_median_difference']:.3e}`.",
        f"- Seed-first descriptive Spearman correlation: `{table_three['spearman_seed_first']:.6f}` over `{table_three['spearman_units']}` prespecified configuration medians.",
        f"- Table IV: retained sets equal `{additional['table_four']['all_retained_sets_equal']}`, maximum pruning ratio `{additional['table_four']['max_pruning_ratio']:.0f}`, support-pruned branches `{additional['table_four']['support_pruned_branches']}` at maximum depths `{additional['table_four']['maximum_depth_values']}`.",
        "", "## Scope", "",
        "- This audit does not run SWaT or TEP and does not infer causal or generating-direction accuracy from Psi.",
    ]) + "\n"
    (ROOT / "experiments/outputs/paper/dynamic_tableII_audit.md").write_text(audit_markdown, encoding="utf-8")
    print(f"Dynamic audit complete; max decomposition error={np.max(decomposition_abs_error):.3e}; {case}")


if __name__ == "__main__":
    main()
