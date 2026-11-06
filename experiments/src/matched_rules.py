"""Paired retention contrasts on a fixed universe of singleton rule pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def retention_flags(frame: pd.DataFrame, coverage: float, support: float,
                    confidence: float) -> np.ndarray:
    """Reconstruct the original retention predicate, without numerical slack."""
    flags = []
    for side in ("forward", "reverse"):
        cov = frame[f"coverage_{side}"].to_numpy(dtype=float)
        supp = frame[f"support_{side}"].to_numpy(dtype=float)
        conf = frame[f"confidence_{side}"].to_numpy(dtype=float)
        flags.append((cov > 0) & (cov >= coverage) & (supp >= support)
                     & np.isfinite(conf) & (conf >= confidence))
    return np.column_stack(flags)


def pair_states(flags: np.ndarray) -> np.ndarray:
    """Encode neither=0, forward only=1, reverse only=2, both=3."""
    values = np.asarray(flags, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError("Expected a nonempty (pair, 2) retention array.")
    return values[:, 0].astype(np.int8) + 2 * values[:, 1].astype(np.int8)


def retention_contrast(reference: np.ndarray, treatment: np.ndarray
                       ) -> tuple[dict[str, int | float], np.ndarray]:
    """Compare directed-rule membership and the four paired selection states."""
    before, after = np.asarray(reference, dtype=bool), np.asarray(treatment, dtype=bool)
    state_before, state_after = pair_states(before), pair_states(after)
    if before.shape != after.shape:
        raise ValueError("Both settings must use the same rule-pair universe.")
    transitions = np.bincount(4 * state_before + state_after, minlength=16).reshape(4, 4)
    intersection = int((before & after).sum())
    union = int((before | after).sum())
    metrics: dict[str, int | float] = {
        "reference_count": int(before.sum()),
        "treatment_count": int(after.sum()),
        "intersection": intersection,
        "union": union,
        # Two empty sets agree completely; their counts are also reported.
        "jaccard": intersection / union if union else 1.0,
        "added": int((after & ~before).sum()),
        "removed": int((before & ~after).sum()),
        "changed_pairs": int((state_before != state_after).sum()),
        "orientation_flips": int(transitions[1, 2] + transitions[2, 1]),
    }
    assert metrics["treatment_count"] == metrics["reference_count"] + metrics["added"] - metrics["removed"]
    assert transitions.sum() == len(before)
    return metrics, transitions
