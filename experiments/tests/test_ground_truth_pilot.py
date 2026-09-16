import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "16_ground_truth_directional_pilot.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("ground_truth_pilot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_source_scale_is_fixed_when_response_noise_changes():
    first, first_sd = MODULE.generate_known_order_process(
        np.random.default_rng(17), phi=0.7, rho=0.5, lag=1, coupling=1.0,
        response_noise_to_signal=0.25, length=100, burn_in=20,
        mechanism="tanh", independent_null=False,
    )
    second, second_sd = MODULE.generate_known_order_process(
        np.random.default_rng(17), phi=0.7, rho=0.5, lag=1, coupling=1.0,
        response_noise_to_signal=1.0, length=100, burn_in=20,
        mechanism="tanh", independent_null=False,
    )
    np.testing.assert_allclose(first["X"], second["X"])
    assert np.isclose(second_sd / first_sd, 4.0)


def test_ranking_value_counts_ties_as_half():
    values = MODULE._ranking_value(np.array([-1.0, 0.0, 1.0]), 1e-12)
    np.testing.assert_allclose(values, [0.0, 0.5, 1.0])
