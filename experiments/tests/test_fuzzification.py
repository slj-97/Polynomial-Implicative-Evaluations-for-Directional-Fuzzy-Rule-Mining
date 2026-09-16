import pandas as pd
import numpy as np
import pytest
from experiments.src.fuzzification import fuzzify_our_main, triangular_memberships

def test_scaler_uses_training_only_bounds():
    train = pd.DataFrame({"x": list(range(100))})
    test = pd.DataFrame({"x": [10000.]})
    _, memberships = fuzzify_our_main(train, test)
    assert memberships["x::2"][0] == 1.0


def test_binary_columns_receive_exactly_two_natural_labels():
    train = pd.DataFrame({"continuous": [0.0, 0.3, 0.8, 1.0], "state": [0, 1, 0, 1]})
    evaluation = pd.DataFrame({"continuous": [0.2, 0.7], "state": [1, 0]})
    _, memberships = fuzzify_our_main(train, evaluation, labels=5)
    assert sorted(key for key in memberships if key.startswith("state::")) == ["state::0", "state::1"]
    np.testing.assert_array_equal(memberships["state::0"], np.array([0.0, 1.0]))
    np.testing.assert_array_equal(memberships["state::1"], np.array([1.0, 0.0]))
    assert len([key for key in memberships if key.startswith("continuous::")]) == 5


def test_binary_schema_is_fitted_on_training_partition():
    train = pd.DataFrame({"state": [0, 1, 0, 1]})
    evaluation = pd.DataFrame({"state": [0.0, 0.5]})
    with pytest.raises(ValueError, match="Binary column"):
        fuzzify_our_main(train, evaluation)


def test_five_label_partition_has_unit_sum():
    memberships = triangular_memberships(pd.DataFrame({"x": np.linspace(0.0, 1.0, 21)}), labels=5)
    total = np.sum([memberships[f"x::{index}"] for index in range(5)], axis=0)
    np.testing.assert_allclose(total, np.ones(21), atol=1e-12)
