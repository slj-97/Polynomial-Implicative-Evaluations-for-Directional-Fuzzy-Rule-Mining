"""Metrics for direct swaps, process ordering, and rule-quality decisions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .operators import boundary_factor

Evaluation = Callable[[Any, Any], np.ndarray]


def _unit_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector.")
    if array.min() < 0.0 or array.max() > 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return array


def fuzzy_coverage(antecedent: Any) -> float:
    return float(_unit_vector(antecedent, "antecedent").mean())


def fuzzy_support(antecedent: Any, consequent: Any, evaluation: Evaluation) -> float:
    a = _unit_vector(antecedent, "antecedent")
    c = _unit_vector(consequent, "consequent")
    if a.shape != c.shape:
        raise ValueError("Antecedent and consequent vectors must have the same length.")
    return float(np.asarray(evaluation(a, c), dtype=np.float64).mean())


def fuzzy_confidence(antecedent: Any, consequent: Any, evaluation: Evaluation) -> float:
    a = _unit_vector(antecedent, "antecedent")
    denominator = float(a.sum())
    if denominator == 0.0:
        return float("nan")
    return float(np.asarray(evaluation(a, consequent), dtype=np.float64).sum() / denominator)


def gamma_statistic(antecedent: Any, consequent: Any) -> float:
    a = _unit_vector(antecedent, "antecedent")
    c = _unit_vector(consequent, "consequent")
    return float((boundary_factor(a, c) * (a - c)).mean())


def direct_swap_metrics(antecedent: Any, consequent: Any, evaluation: Evaluation) -> dict[str, float]:
    a = _unit_vector(antecedent, "antecedent")
    c = _unit_vector(consequent, "consequent")
    forward = fuzzy_support(a, c, evaluation)
    reverse = fuzzy_support(c, a, evaluation)
    delta = forward - reverse
    empirical_absolute = float(np.abs(np.asarray(evaluation(a, c)) - np.asarray(evaluation(c, a))).mean())
    return {
        "support_forward": forward,
        "support_reverse": reverse,
        "support_delta": delta,
        "absolute_delta": abs(delta),
        "normalized_gap": abs(delta) / (forward + reverse + 1e-12),
        "empirical_absolute_asymmetry": empirical_absolute,
        "gamma": gamma_statistic(a, c),
    }


def process_orientation_gap(
    x_now: Any,
    y_future: Any,
    y_now: Any,
    x_future: Any,
    evaluation: Evaluation,
) -> dict[str, float]:
    """Compute Psi for process order; this is intentionally not a direct swap."""
    forward = fuzzy_support(x_now, y_future, evaluation)
    reverse = fuzzy_support(y_now, x_future, evaluation)
    psi = forward - reverse
    return {
        "support_process_forward": forward,
        "support_process_reverse": reverse,
        "process_gap": psi,
        "absolute_process_gap": abs(psi),
        "normalized_psi": abs(psi) / (forward + reverse + 1e-12),
    }


def selection_indicator(coverage: float, support: float, confidence: float, thresholds: dict[str, float]) -> int:
    return int(
        coverage > 0.0
        and np.isfinite(confidence)
        and coverage >= thresholds["coverage"]
        and support >= thresholds["support"]
        and confidence >= thresholds["confidence"]
    )


def threshold_selection_asymmetry(
    coverage_forward: Any,
    support_forward: Any,
    confidence_forward: Any,
    coverage_reverse: Any,
    support_reverse: Any,
    confidence_reverse: Any,
    threshold_grid: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Return threshold-specific selection asymmetry over paired rules."""

    arrays = np.broadcast_arrays(
        np.asarray(coverage_forward, dtype=np.float64),
        np.asarray(support_forward, dtype=np.float64),
        np.asarray(confidence_forward, dtype=np.float64),
        np.asarray(coverage_reverse, dtype=np.float64),
        np.asarray(support_reverse, dtype=np.float64),
        np.asarray(confidence_reverse, dtype=np.float64),
    )
    if arrays[0].size == 0:
        raise ValueError("Selection inputs must be nonempty.")

    def token(value: float) -> str:
        return f"{float(value):g}".replace("-", "m").replace(".", "p")

    results: dict[str, float] = {}
    cov_f, supp_f, conf_f, cov_r, supp_r, conf_r = arrays
    for thresholds in threshold_grid:
        required = {"coverage", "support", "confidence"}
        if required - set(thresholds):
            raise ValueError("Each threshold set must define coverage, support, and confidence.")
        forward = (
            (cov_f > 0.0)
            & np.isfinite(conf_f)
            & (cov_f >= float(thresholds["coverage"]))
            & (supp_f >= float(thresholds["support"]))
            & (conf_f >= float(thresholds["confidence"]))
        )
        reverse = (
            (cov_r > 0.0)
            & np.isfinite(conf_r)
            & (cov_r >= float(thresholds["coverage"]))
            & (supp_r >= float(thresholds["support"]))
            & (conf_r >= float(thresholds["confidence"]))
        )
        suffix = (
            f"cov{token(thresholds['coverage'])}_"
            f"supp{token(thresholds['support'])}_"
            f"conf{token(thresholds['confidence'])}"
        )
        results[f"SA_{suffix}"] = float(np.mean(forward != reverse))
    return results


def selection_asymmetry(forward: Sequence[int], reverse: Sequence[int]) -> float:
    forward_arr = np.asarray(forward, dtype=np.int8)
    reverse_arr = np.asarray(reverse, dtype=np.int8)
    if forward_arr.shape != reverse_arr.shape or forward_arr.size == 0:
        raise ValueError("Selection vectors must be nonempty and have identical shape.")
    return float(np.mean(forward_arr != reverse_arr))


def rank_biserial_from_differences(differences: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation, ignoring exact zero differences."""
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values) & (values != 0.0)]
    if values.size == 0:
        return 0.0
    from scipy.stats import rankdata

    ranks = rankdata(np.abs(values))
    return float((ranks[values > 0].sum() - ranks[values < 0].sum()) / ranks.sum())
