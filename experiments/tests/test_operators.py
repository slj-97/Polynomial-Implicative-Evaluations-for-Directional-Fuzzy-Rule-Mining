import numpy as np
from experiments.src.metrics import fuzzy_confidence, selection_indicator, threshold_selection_asymmetry
from experiments.src.operators import configured_evaluations, eval_poly, eval_product, verify_polynomial_operator
from experiments.src.reproducibility import load_yaml

def test_polynomial_grid_verification():
    result = verify_polynomial_operator(0.5, 0.5, grid_size=101)
    assert result.passed

def test_lukasiewicz_recovery():
    x = np.array([0.2, 0.8])
    y = np.array([0.7, 0.1])
    # The family is not the Lukasiewicz t-norm; its associated implication
    # recovers E through the Lukasiewicz composition in the admissible region.
    result = verify_polynomial_operator(0.0, 0.0, grid_size=101)
    assert result.lukasiewicz_recovery_error < 1e-12


def test_zero_coverage_makes_confidence_undefined_and_rule_unselected():
    confidence = fuzzy_confidence(np.zeros(3), np.ones(3), eval_product)
    assert np.isnan(confidence)
    assert selection_indicator(0.0, 0.0, confidence, {"coverage": 0.0, "support": 0.0, "confidence": 0.0}) == 0


def test_common_evaluation_registry_contains_controls_and_full_grid():
    common = load_yaml("experiments/configs/common.yaml")
    evaluations = configured_evaluations(common)
    assert len(evaluations) == 24
    assert sum(name == "proposed" for name, *_ in evaluations) == 20


def test_selection_asymmetry_is_threshold_specific():
    thresholds = [
        {"coverage": 0.01, "support": 0.03, "confidence": 0.3},
        {"coverage": 0.01, "support": 0.05, "confidence": 0.7},
    ]
    values = threshold_selection_asymmetry(
        [0.5], [0.04], [0.8], [0.5], [0.02], [0.4], thresholds
    )
    assert values["SA_cov0p01_supp0p03_conf0p3"] == 1.0
    assert values["SA_cov0p01_supp0p05_conf0p7"] == 0.0
