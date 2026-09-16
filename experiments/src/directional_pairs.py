"""Pre-registered process-pair registry and lagged comparison construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "dataset",
    "upstream_variable",
    "downstream_variable",
    "process_stage_up",
    "process_stage_down",
    "relation_description",
    "source_reference",
    "allow_feedback",
    "lags",
    "status",
}


@dataclass(frozen=True)
class ProcessPair:
    dataset: str
    upstream_variable: str
    downstream_variable: str
    process_stage_up: str
    process_stage_down: str
    relation_description: str
    source_reference: str
    allow_feedback: bool
    lags: tuple[int, ...]


def _parse_lags(value: object) -> tuple[int, ...]:
    tokens = str(value).replace(";", ",").split(",")
    lags = tuple(int(token.strip()) for token in tokens if token.strip())
    if not lags or any(lag < 1 for lag in lags):
        raise ValueError("Approved process-pair rows need positive comma-separated lags.")
    return lags


def load_approved_process_pairs(path: str | Path, dataset: str) -> list[ProcessPair]:
    registry = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"Process-pair registry misses required columns: {sorted(missing)}")
    approved = registry[(registry["dataset"].str.lower() == dataset.lower()) & (registry["status"].str.lower() == "approved")]
    if approved.empty:
        raise ValueError(f"No approved {dataset} process pairs are available; real-data analysis remains blocked.")
    pairs: list[ProcessPair] = []
    for row in approved.to_dict(orient="records"):
        pairs.append(
            ProcessPair(
                dataset=str(row["dataset"]),
                upstream_variable=str(row["upstream_variable"]),
                downstream_variable=str(row["downstream_variable"]),
                process_stage_up=str(row["process_stage_up"]),
                process_stage_down=str(row["process_stage_down"]),
                relation_description=str(row["relation_description"]),
                source_reference=str(row["source_reference"]),
                allow_feedback=str(row["allow_feedback"]).strip().lower() in {"true", "1", "yes"},
                lags=_parse_lags(row["lags"]),
            )
        )
    return pairs


def lagged_direct_swap(data: pd.DataFrame, upstream: str, downstream: str, lag: int) -> tuple[pd.Series, pd.Series]:
    """Return X_t and Y_{t+h}; swapping these two vectors is the direct-swap protocol."""
    if lag < 1 or upstream not in data or downstream not in data:
        raise ValueError("Invalid direct-swap variables or lag.")
    return data[upstream].iloc[:-lag].reset_index(drop=True), data[downstream].iloc[lag:].reset_index(drop=True)


def lagged_process_order(
    data: pd.DataFrame, upstream: str, downstream: str, lag: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return X_t, Y_{t+h}, Y_t, X_{t+h} for the separate process-order protocol."""
    x_now, y_future = lagged_direct_swap(data, upstream, downstream, lag)
    y_now, x_future = lagged_direct_swap(data, downstream, upstream, lag)
    return x_now, y_future, y_now, x_future
