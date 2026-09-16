"""Data-gated real-process directional analyses.

This module accepts only supplied measurements and pre-registered process
pairs. It never creates industrial-process substitutes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from experiments.src.directional_pairs import ProcessPair, lagged_process_order
from experiments.src.fuzzification import fuzzify_our_main
from experiments.src.metrics import (
    direct_swap_metrics,
    fuzzy_confidence,
    fuzzy_coverage,
    gamma_statistic,
    process_orientation_gap,
    threshold_selection_asymmetry,
)
from experiments.src.operators import configured_evaluations, eval_poly, operator_asymmetry_index


def _membership_vectors(memberships: dict[str, np.ndarray], column: str) -> list[tuple[int, np.ndarray]]:
    prefix = f"{column}::"
    labels = sorted(
        (int(key.removeprefix(prefix)), values)
        for key, values in memberships.items()
        if key.startswith(prefix)
    )
    if not labels:
        raise ValueError(f"No fuzzy memberships are available for {column!r}.")
    return labels


def _threshold_grid(common: Mapping[str, Any]) -> list[dict[str, float]]:
    return [
        {
            "coverage": float(common["thresholds"]["coverage"]),
            "support": float(support),
            "confidence": float(confidence),
        }
        for support in common["thresholds"]["support"]
        for confidence in common["thresholds"]["confidence"]
    ]


def analyse_pairs(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    pairs: list[ProcessPair],
    *,
    common: Mapping[str, Any],
    label_counts: Sequence[int] = (3, 5),
) -> pd.DataFrame:
    """Compute direct-swap and process-order quantities on the common grid."""
    rows: list[dict[str, object]] = []
    thresholds = _threshold_grid(common)
    evaluations = configured_evaluations(common)
    for label_count in label_counts:
        _, memberships = fuzzify_our_main(training, evaluation, labels=int(label_count))
        for pair in pairs:
            if pair.upstream_variable not in evaluation or pair.downstream_variable not in evaluation:
                raise ValueError(f"Configured pair is absent from evaluation data: {pair}")
            x_labels = _membership_vectors(memberships, pair.upstream_variable)
            y_labels = _membership_vectors(memberships, pair.downstream_variable)
            for antecedent_label, x_all in x_labels:
                for consequent_label, y_all in y_labels:
                    indexed = pd.DataFrame(
                        {pair.upstream_variable: x_all, pair.downstream_variable: y_all}
                    )
                    for lag in pair.lags:
                        x, y, y_now, x_future = lagged_process_order(
                            indexed, pair.upstream_variable, pair.downstream_variable, lag
                        )
                        x_array, y_array = x.to_numpy(), y.to_numpy()
                        y_now_array, x_future_array = y_now.to_numpy(), x_future.to_numpy()
                        lambda_statistic = 0.5 * (
                            gamma_statistic(x_array, y_array)
                            - gamma_statistic(y_now_array, x_future_array)
                        )
                        coverage_forward = fuzzy_coverage(x_array)
                        coverage_reverse = fuzzy_coverage(y_array)
                        for name, power, sigma, delta, function in evaluations:
                            direct = direct_swap_metrics(x_array, y_array, function)
                            confidence_forward = fuzzy_confidence(x_array, y_array, function)
                            confidence_reverse = fuzzy_confidence(y_array, x_array, function)
                            selection = threshold_selection_asymmetry(
                                [coverage_forward], [direct["support_forward"]], [confidence_forward],
                                [coverage_reverse], [direct["support_reverse"]], [confidence_reverse],
                                thresholds,
                            )
                            process = process_orientation_gap(
                                x_array, y_array, y_now_array, x_future_array, function
                            )
                            if name == "proposed":
                                zero_evaluation = lambda u, v, s=float(sigma): eval_poly(u, v, s, 0.0)
                                psi_zero = process_orientation_gap(
                                    x_array, y_array, y_now_array, x_future_array, zero_evaluation
                                )["process_gap"]
                                identity_error = direct["support_delta"] - float(delta) * direct["gamma"]
                                decomposition_error = (
                                    process["process_gap"] - psi_zero - float(delta) * lambda_statistic
                                )
                                operator_d = operator_asymmetry_index(float(delta))
                            else:
                                psi_zero = np.nan
                                identity_error = np.nan
                                decomposition_error = np.nan
                                operator_d = np.nan
                            rows.append(
                                {
                                    "engine": "OUR-MAIN",
                                    "protocol": "real-data",
                                    "dataset": pair.dataset,
                                    "upstream_variable": pair.upstream_variable,
                                    "downstream_variable": pair.downstream_variable,
                                    "antecedent_label": antecedent_label,
                                    "consequent_label": consequent_label,
                                    "label_count": int(label_count),
                                    "lag": lag,
                                    "evaluation": name,
                                    "power": power,
                                    "sigma": sigma,
                                    "delta": delta,
                                    "operator_D": operator_d,
                                    "coverage_forward": coverage_forward,
                                    "coverage_reverse": coverage_reverse,
                                    "confidence_forward": confidence_forward,
                                    "confidence_reverse": confidence_reverse,
                                    "direct_delta": direct["support_delta"],
                                    "direct_gamma": direct["gamma"],
                                    "direct_abs_delta": direct["absolute_delta"],
                                    "direct_G": direct["normalized_gap"],
                                    "empirical_absolute_asymmetry": direct["empirical_absolute_asymmetry"],
                                    "identity_error": identity_error,
                                    "process_gap": process["process_gap"],
                                    "process_psi_zero": psi_zero,
                                    "process_increment": process["process_gap"] - psi_zero,
                                    "process_lambda": lambda_statistic,
                                    "process_decomposition_error": decomposition_error,
                                    "source_reference": pair.source_reference,
                                    "allow_feedback": pair.allow_feedback,
                                    **selection,
                                }
                            )
    return pd.DataFrame(rows)
