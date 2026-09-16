"""Pre-specified summaries and paired nonparametric comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .metrics import rank_biserial_from_differences


def bootstrap_median_ci(values: Sequence[float], rng: np.random.Generator, repetitions: int = 2000) -> tuple[float, float]:
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.nan, np.nan
    medians = np.median(rng.choice(sample, size=(repetitions, sample.size), replace=True), axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def summarize_groups(
    frame: pd.DataFrame, group_columns: list[str], value_column: str, *, seed: int, bootstrap_repetitions: int = 2000
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for group, subset in frame.groupby(group_columns, dropna=False):
        values = subset[value_column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        low, high = bootstrap_median_ci(finite, rng, bootstrap_repetitions)
        keys = group if isinstance(group, tuple) else (group,)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "n": int(finite.size),
                "median": float(np.median(finite)) if finite.size else np.nan,
                "iqr": float(np.subtract(*np.percentile(finite, [75, 25]))) if finite.size else np.nan,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_wilcoxon(left: Sequence[float], right: Sequence[float]) -> dict[str, float | int]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    differences = x[valid] - y[valid]
    if differences.size == 0 or np.allclose(differences, 0.0):
        return {"n": int(differences.size), "statistic": 0.0, "p_value": 1.0, "rank_biserial": 0.0}
    statistic, p_value = wilcoxon(differences, alternative="two-sided", method="auto")
    return {
        "n": int(differences.size),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "rank_biserial": rank_biserial_from_differences(differences),
    }
