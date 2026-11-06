"""Pair moments and fixed-parameter scores for repeated static evaluations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pair_moments(memberships: dict[str, np.ndarray]) -> pd.DataFrame:
    """All distinct-variable pairs in the original membership insertion order.

    Matrix products evaluate the same empirical moments as the original
    recordwise expressions. U=x^2(1-x), V=x(1-x), so U.T@V gives the
    first contribution to both the symmetric correction and Gamma.
    """
    names = list(memberships)
    x = np.column_stack([memberships[name] for name in names])
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Memberships must be finite and in [0,1].")
    variables = np.array([name.rsplit("::", 1)[0] for name in names])
    i, j = np.triu_indices(len(names), 1)
    keep = variables[i] != variables[j]
    i, j = i[keep], j[keep]
    v = x * (1 - x)
    u = x * v
    cross = (u.T @ v) / len(x)
    product = (x.T @ x) / len(x)
    coverage = x.mean(axis=0)
    return pd.DataFrame(dict(
        item_a=np.array(names)[i], item_c=np.array(names)[j],
        coverage_forward=coverage[i], coverage_reverse=coverage[j],
        product=product[i, j], symmetric=cross[i, j] + cross[j, i],
        gamma=cross[i, j] - cross[j, i],
    ))


def score_pairs(moments: pd.DataFrame, sigma: float, delta: float) -> pd.DataFrame:
    if not 0 <= sigma <= 1 or abs(delta) > sigma:
        raise ValueError("Require 0<=sigma<=1 and |delta|<=sigma.")
    frame = moments.copy()
    baseline = frame["product"].to_numpy() - sigma * frame["symmetric"].to_numpy() / 2
    for side, sign in (("forward", 1), ("reverse", -1)):
        support = baseline + sign * delta * frame["gamma"].to_numpy() / 2
        cov = frame[f"coverage_{side}"].to_numpy()
        frame[f"support_{side}"] = support
        frame[f"confidence_{side}"] = np.divide(support, cov,
                                                out=np.full_like(support, np.nan), where=cov > 0)
    return frame
