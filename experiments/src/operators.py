"""Independent fuzzy evaluations and numerical admissibility checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

Array = np.ndarray
Evaluation = Callable[[Any, Any], Array]


def _arrays(x: Any, y: Any) -> tuple[Array, Array]:
    """Return broadcast float64 arrays and reject values outside the unit interval."""
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    if not (np.isfinite(x_arr).all() and np.isfinite(y_arr).all()):
        raise ValueError("Fuzzy operator inputs must be finite.")
    if x_arr.min(initial=0.0) < 0.0 or x_arr.max(initial=1.0) > 1.0:
        raise ValueError("First fuzzy operator input is outside [0, 1].")
    if y_arr.min(initial=0.0) < 0.0 or y_arr.max(initial=1.0) > 1.0:
        raise ValueError("Second fuzzy operator input is outside [0, 1].")
    return x_arr, y_arr


def t_lukasiewicz(x: Any, y: Any) -> Array:
    x_arr, y_arr = _arrays(x, y)
    return np.maximum(x_arr + y_arr - 1.0, 0.0)


def eval_product(x: Any, y: Any) -> Array:
    x_arr, y_arr = _arrays(x, y)
    return x_arr * y_arr


def eval_minimum(x: Any, y: Any) -> Array:
    x_arr, y_arr = _arrays(x, y)
    return np.minimum(x_arr, y_arr)


def eval_power(x: Any, y: Any, p: float) -> Array:
    if p <= 0.0:
        raise ValueError("Power parameter p must be strictly positive.")
    x_arr, y_arr = _arrays(x, y)
    return x_arr * np.power(y_arr, p)


def implication_power(x: Any, y: Any, p: float) -> Array:
    x_arr, y_arr = _arrays(x, y)
    return 1.0 - x_arr + eval_power(x_arr, y_arr, p)


def boundary_factor(x: Any, y: Any) -> Array:
    x_arr, y_arr = _arrays(x, y)
    return x_arr * y_arr * (1.0 - x_arr) * (1.0 - y_arr)


def eval_poly(x: Any, y: Any, sigma: float, delta: float) -> Array:
    """Polynomial evaluation in Eq. (sigma-delta-evaluation)."""
    if not (0.0 <= sigma <= 1.0 and abs(delta) <= sigma):
        raise ValueError("Require 0 <= sigma <= 1 and |delta| <= sigma.")
    x_arr, y_arr = _arrays(x, y)
    b = boundary_factor(x_arr, y_arr)
    return x_arr * y_arr - 0.5 * sigma * b * (x_arr + y_arr) + 0.5 * delta * b * (x_arr - y_arr)


def implication_poly(x: Any, y: Any, sigma: float, delta: float) -> Array:
    x_arr, y_arr = _arrays(x, y)
    return 1.0 - x_arr + eval_poly(x_arr, y_arr, sigma, delta)


def asymmetry_poly(x: Any, y: Any, delta: float) -> Array:
    x_arr, y_arr = _arrays(x, y)
    return delta * x_arr * y_arr * (1.0 - x_arr) * (1.0 - y_arr) * (x_arr - y_arr)


def operator_asymmetry_index(delta: float) -> float:
    """Exact index D(E_{sigma,delta}) = |delta| / 140."""
    return abs(float(delta)) / 140.0


def configured_evaluations(
    common: Mapping[str, Any],
) -> list[tuple[str, float | None, float | None, float | None, Evaluation]]:
    """Build the common comparison set and complete proposed parameter grid."""

    result: list[tuple[str, float | None, float | None, float | None, Evaluation]] = [
        ("product", None, None, None, eval_product),
        ("minimum", None, None, None, eval_minimum),
        ("power", 0.01, None, None, lambda x, y: eval_power(x, y, 0.01)),
        ("power", 2.0, None, None, lambda x, y: eval_power(x, y, 2.0)),
    ]
    grid = common["polynomial_grid"]
    for sigma_value in grid["sigma"]:
        sigma = float(sigma_value)
        for ratio_value in grid["relative_delta"]:
            delta = sigma * float(ratio_value)
            result.append(
                (
                    "proposed",
                    None,
                    sigma,
                    delta,
                    lambda x, y, s=sigma, d=delta: eval_poly(x, y, s, d),
                )
            )
    return result


def partial_x_poly(x: Any, y: Any, sigma: float, delta: float) -> Array:
    x_arr, y_arr = _arrays(x, y)
    b = boundary_factor(x_arr, y_arr)
    bx = y_arr * (1.0 - y_arr) * (1.0 - 2.0 * x_arr)
    return y_arr - 0.5 * sigma * (bx * (x_arr + y_arr) + b) + 0.5 * delta * (bx * (x_arr - y_arr) + b)


def partial_y_poly(x: Any, y: Any, sigma: float, delta: float) -> Array:
    x_arr, y_arr = _arrays(x, y)
    b = boundary_factor(x_arr, y_arr)
    by = x_arr * (1.0 - x_arr) * (1.0 - 2.0 * y_arr)
    return x_arr - 0.5 * sigma * (by * (x_arr + y_arr) + b) + 0.5 * delta * (by * (x_arr - y_arr) - b)


@dataclass(frozen=True)
class OperatorVerification:
    sigma: float
    delta: float
    boundary_error: float
    range_violation: float
    derivative_x_lower_violation: float
    derivative_x_upper_violation: float
    derivative_y_violation: float
    tc_violation: float
    mtc_violation: float
    np_error: float
    lukasiewicz_recovery_error: float
    asymmetry_error: float
    d_index_error: float
    passed: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _max_abs(values: Array) -> float:
    return float(np.max(np.abs(values)))


def verify_polynomial_operator(
    sigma: float,
    delta: float,
    *,
    grid_size: int = 401,
    tolerance: float = 1e-12,
) -> OperatorVerification:
    """Numerically check the sufficient conditions and exact formulas on a grid."""
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    e = eval_poly(x, y, sigma, delta)
    dx = partial_x_poly(x, y, sigma, delta)
    dy = partial_y_poly(x, y, sigma, delta)
    implication = implication_poly(x, y, sigma, delta)

    boundary_error = max(
        _max_abs(eval_poly(0.0, grid, sigma, delta)),
        _max_abs(eval_poly(grid, 0.0, sigma, delta)),
        _max_abs(eval_poly(1.0, grid, sigma, delta) - grid),
        _max_abs(eval_poly(grid, 1.0, sigma, delta) - grid),
    )
    range_violation = max(0.0, float(-e.min()), float((e - np.minimum(x, y)).max()))
    dx_lower = max(0.0, float(-dx.min()))
    dx_upper = max(0.0, float((dx - 1.0).max()))
    dy_lower = max(0.0, float(-dy.min()))
    tc_violation = max(0.0, float((e - y).max()))
    mtc_violation = dx_lower
    np_error = _max_abs(implication_poly(1.0, grid, sigma, delta) - grid)
    recovery_error = _max_abs(t_lukasiewicz(x, implication) - e)
    asymmetry_error = _max_abs((e - e.T) - asymmetry_poly(x, y, delta))
    d_index_error = abs(operator_asymmetry_index(delta) - abs(delta) / 140.0)
    checks = [
        boundary_error,
        range_violation,
        dx_lower,
        dx_upper,
        dy_lower,
        tc_violation,
        mtc_violation,
        np_error,
        recovery_error,
        asymmetry_error,
        d_index_error,
    ]
    return OperatorVerification(
        sigma=sigma,
        delta=delta,
        boundary_error=boundary_error,
        range_violation=range_violation,
        derivative_x_lower_violation=dx_lower,
        derivative_x_upper_violation=dx_upper,
        derivative_y_violation=dy_lower,
        tc_violation=tc_violation,
        mtc_violation=mtc_violation,
        np_error=np_error,
        lukasiewicz_recovery_error=recovery_error,
        asymmetry_error=asymmetry_error,
        d_index_error=d_index_error,
        passed=all(value <= tolerance for value in checks),
    )


def polynomial_grid(sigmas: list[float], relative_deltas: list[float]) -> list[tuple[float, float]]:
    return [(sigma, sigma * ratio) for sigma in sigmas for ratio in relative_deltas]


def verify_polynomial_grid(
    sigmas: list[float], relative_deltas: list[float], *, grid_size: int = 401, tolerance: float = 1e-12
) -> pd.DataFrame:
    results = [
        verify_polynomial_operator(sigma, delta, grid_size=grid_size, tolerance=tolerance).to_dict()
        for sigma, delta in polynomial_grid(sigmas, relative_deltas)
    ]
    return pd.DataFrame(results)
