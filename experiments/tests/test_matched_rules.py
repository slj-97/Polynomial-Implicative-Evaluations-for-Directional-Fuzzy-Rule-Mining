import numpy as np
import pandas as pd
import pytest

from experiments.src.matched_rules import pair_states, retention_contrast, retention_flags


def test_equal_counts_can_hide_orientation_flips():
    before = np.array([[True, False], [False, True], [True, True], [False, False]])
    after = np.array([[False, True], [True, False], [False, False], [True, True]])
    metrics, transitions = retention_contrast(before, after)
    assert metrics["reference_count"] == metrics["treatment_count"] == 4
    assert metrics["orientation_flips"] == 2
    assert metrics["jaccard"] == 0
    assert metrics["added"] == metrics["removed"] == 4
    assert transitions[3, 0] == transitions[0, 3] == 1


def test_empty_retained_sets_and_mismatched_universes():
    flags = np.zeros((3, 2), dtype=bool)
    metrics, _ = retention_contrast(flags, flags)
    assert metrics["jaccard"] == 1
    assert metrics["reference_count"] == metrics["treatment_count"] == 0
    with pytest.raises(ValueError):
        retention_contrast(flags, flags[:2])


def test_symmetric_support_can_have_asymmetric_retention():
    frame = pd.DataFrame({
        "coverage_forward": [0.1, 0.0], "coverage_reverse": [0.4, 0.0],
        "support_forward": [0.06, 0.0], "support_reverse": [0.06, 0.0],
        "confidence_forward": [0.6, np.nan], "confidence_reverse": [0.15, np.nan],
    })
    flags = retention_flags(frame, coverage=0, support=0, confidence=0.5)
    assert pair_states(flags).tolist() == [1, 0]


def test_threshold_equality_is_retained_without_tolerance():
    frame = pd.DataFrame({
        "coverage_forward": [0.02], "coverage_reverse": [0.02],
        "support_forward": [0.01], "support_reverse": [np.nextafter(0.01, 0)],
        "confidence_forward": [0.5], "confidence_reverse": [0.5],
    })
    assert retention_flags(frame, .01, .01, .5).tolist() == [[True, False]]
