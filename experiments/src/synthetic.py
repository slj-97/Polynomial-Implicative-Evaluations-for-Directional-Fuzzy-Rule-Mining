"""Synthetic data generators specified by the manuscript protocol."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def membership_pairs(
    rng: np.random.Generator,
    n_records: int,
    kappa: float,
    eta: float,
    control: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_records <= 0 or kappa <= 0.0 or eta < 0.0:
        raise ValueError("Invalid synthetic membership parameters.")
    antecedent = rng.uniform(0.0, 1.0, n_records)
    if control == "diagonal":
        consequent = antecedent.copy()
    elif control == "exchangeable":
        consequent = rng.uniform(0.0, 1.0, n_records)
    elif control is None:
        consequent = np.clip(np.power(antecedent, kappa) + rng.normal(0.0, eta, n_records), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown control: {control}")
    return antecedent, consequent


def _signal_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    functions: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "linear": lambda x: x,
        "tanh": np.tanh,
        "saturation": lambda x: x / (1.0 + np.abs(x)),
    }
    if name not in functions:
        raise ValueError(f"Unknown dynamic signal function: {name}")
    return functions[name]


def dynamic_process(
    rng: np.random.Generator,
    *,
    phi: float,
    rho: float,
    lag: int,
    coupling: float,
    noise: float,
    length: int,
    burn_in: int,
    function: str,
    independent_null: bool = False,
) -> pd.DataFrame:
    if lag < 1 or length <= 0 or burn_in < 0 or noise < 0.0:
        raise ValueError("Invalid dynamic synthetic parameters.")
    transform = _signal_function(function)
    total = length + burn_in + lag
    x = np.zeros(total, dtype=np.float64)
    y = np.zeros(total, dtype=np.float64)
    eps_x = rng.normal(0.0, noise, total)
    eps_y = rng.normal(0.0, noise, total)
    for time in range(1, total):
        x[time] = phi * x[time - 1] + eps_x[time]
        driving = 0.0 if independent_null or time < lag else coupling * transform(np.array([x[time - lag]]))[0]
        y[time] = rho * y[time - 1] + driving + eps_y[time]
    start = burn_in + lag
    return pd.DataFrame({"X": x[start:], "Y": y[start:]})
