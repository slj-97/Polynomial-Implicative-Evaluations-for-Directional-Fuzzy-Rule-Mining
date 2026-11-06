import numpy as np
import pytest

from experiments.src.operators import eval_poly
from experiments.src.repeated_splits import pair_moments, score_pairs


def test_matrix_moments_and_both_directions_match_recordwise_evaluation():
    rng = np.random.default_rng(123)
    memberships = {"A::0": rng.random(73), "A::1": rng.random(73),
                   "B::0": rng.random(73), "C::0": rng.integers(0, 2, 73).astype(float)}
    moments = pair_moments(memberships)
    assert len(moments) == 5
    assert not ((moments.item_a == "A::0") & (moments.item_c == "A::1")).any()
    for sigma, delta in [(1, -.5), (1, 0), (1, .5), (.5, .25)]:
        frame = score_pairs(moments, sigma, delta)
        for row in frame.itertuples():
            a, c = memberships[row.item_a], memberships[row.item_c]
            assert row.gamma == pytest.approx(np.mean(a*c*(1-a)*(1-c)*(a-c)), abs=1e-14)
            assert row.support_forward == pytest.approx(eval_poly(a, c, sigma, delta).mean(), abs=1e-14)
            assert row.support_reverse == pytest.approx(eval_poly(c, a, sigma, delta).mean(), abs=1e-14)


def test_zero_coverage_and_membership_validation():
    moments = pair_moments({"A::0": np.zeros(5), "B::0": np.ones(5)})
    frame = score_pairs(moments, 1, .5)
    assert frame.support_forward.iloc[0] == 0
    assert np.isnan(frame.confidence_forward.iloc[0])
    assert frame.confidence_reverse.iloc[0] == 0
    with pytest.raises(ValueError):
        score_pairs(moments, .5, 1)
    with pytest.raises(ValueError):
        pair_moments({"A::0": np.array([2.]), "B::0": np.array([0.])})
