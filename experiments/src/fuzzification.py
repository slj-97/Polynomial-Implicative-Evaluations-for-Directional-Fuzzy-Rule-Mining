"""OUR-MAIN training-only scaling and fixed triangular memberships."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustScaler:
    lower: pd.Series
    upper: pd.Series
    active_columns: tuple[str, ...]

    @classmethod
    def fit(cls, training: pd.DataFrame) -> "RobustScaler":
        numeric = training.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            return cls(
                lower=pd.Series(dtype=np.float64),
                upper=pd.Series(dtype=np.float64),
                active_columns=(),
            )
        if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            raise ValueError("Training data contains NaN or infinite values.")
        lower = numeric.quantile(0.01)
        upper = numeric.quantile(0.99)
        active = tuple(column for column in numeric.columns if upper[column] > lower[column])
        return cls(lower=lower, upper=upper, active_columns=active)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.active_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Data is missing fitted columns: {sorted(missing)}")
        transformed = pd.DataFrame(index=data.index)
        for column in self.active_columns:
            values = data[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Column {column!r} contains NaN or infinite values.")
            transformed[column] = np.clip(
                (values - self.lower[column]) / (self.upper[column] - self.lower[column]), 0.0, 1.0
            )
        return transformed


def chronological_split(data: pd.DataFrame, proportions: tuple[float, float, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(proportions) != 3 or not np.isclose(sum(proportions), 1.0):
        raise ValueError("Chronological split proportions must contain three values summing to one.")
    if len(data) < 3:
        raise ValueError("At least three records are required for a chronological split.")
    first = int(len(data) * proportions[0])
    second = first + int(len(data) * proportions[1])
    if first == 0 or second == first or second >= len(data):
        raise ValueError("Chronological split yields an empty partition.")
    return data.iloc[:first].copy(), data.iloc[first:second].copy(), data.iloc[second:].copy()


def triangular_memberships(scaled: pd.DataFrame, labels: int = 3) -> dict[str, np.ndarray]:
    if labels < 2:
        raise ValueError("At least two fuzzy labels are required.")
    result: dict[str, np.ndarray] = {}
    centers = np.linspace(0.0, 1.0, labels)
    step = centers[1] - centers[0]
    for column in scaled.columns:
        values = scaled[column].to_numpy(dtype=np.float64)
        for index, center in enumerate(centers):
            membership = np.maximum(1.0 - np.abs(values - center) / step, 0.0)
            result[f"{column}::{index}"] = membership
    return result


def natural_binary_memberships(data: pd.DataFrame) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for column in data.columns:
        values = data[column].to_numpy(dtype=np.float64)
        unique = np.unique(values)
        if np.all(np.isin(unique, [0.0, 1.0])):
            result[f"{column}::0"] = 1.0 - values
            result[f"{column}::1"] = values
    return result


def fuzzify_our_main(training: pd.DataFrame, evaluation: pd.DataFrame, labels: int = 3) -> tuple[RobustScaler, dict[str, np.ndarray]]:
    """Fit scaling only on training and create memberships for an evaluation partition."""
    numeric_training = training.select_dtypes(include=[np.number]).copy()
    binary_columns = tuple(
        column
        for column in numeric_training.columns
        if np.all(np.isin(np.unique(numeric_training[column].to_numpy(dtype=np.float64)), [0.0, 1.0]))
    )
    continuous_columns = [column for column in numeric_training.columns if column not in binary_columns]
    missing = set(numeric_training.columns) - set(evaluation.columns)
    if missing:
        raise ValueError(f"Evaluation data is missing training columns: {sorted(missing)}")
    for column in binary_columns:
        values = evaluation[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.all(np.isin(np.unique(values), [0.0, 1.0])):
            raise ValueError(f"Binary column {column!r} contains values outside {{0, 1}}.")
    scaler = RobustScaler.fit(training[continuous_columns])
    scaled = scaler.transform(evaluation)
    memberships = triangular_memberships(scaled, labels=labels)
    if binary_columns:
        memberships.update(natural_binary_memberships(evaluation[list(binary_columns)]))
    return scaler, memberships
