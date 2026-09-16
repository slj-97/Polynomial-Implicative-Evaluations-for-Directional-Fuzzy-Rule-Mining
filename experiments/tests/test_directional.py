import numpy as np
from experiments.src.directional_pairs import ProcessPair, lagged_direct_swap
from experiments.src.metrics import direct_swap_metrics
from experiments.src.operators import eval_poly
from experiments.src.real_analysis import analyse_pairs
from experiments.src.reproducibility import load_yaml
import pandas as pd

def test_direct_swap_identity():
    x = np.array([0.1, 0.4, 0.9]); y = np.array([0.8, 0.2, 0.3])
    metrics = direct_swap_metrics(x, y, lambda a, b: eval_poly(a, b, 0.5, 0.5))
    assert abs(metrics["support_delta"] - 0.5 * metrics["gamma"]) < 1e-12

def test_lagged_pairs_have_equal_length():
    x, y = lagged_direct_swap(pd.DataFrame({"x": [1,2,3], "y": [4,5,6]}), "x", "y", 1)
    assert len(x) == len(y) == 2


def test_real_analysis_uses_common_grid_and_both_partitions():
    training = pd.DataFrame({"X": np.linspace(0.0, 1.0, 30), "Y": np.linspace(0.2, 0.9, 30)})
    evaluation = pd.DataFrame({"X": np.linspace(0.1, 0.9, 20), "Y": np.linspace(0.8, 0.2, 20)})
    pair = ProcessPair("toy", "X", "Y", "up", "down", "documented", "test", False, (1,))
    result = analyse_pairs(training, evaluation, [pair], common=load_yaml("experiments/configs/common.yaml"))
    assert len(result) == 24 * (3**2 + 5**2)
    assert sorted(result["label_count"].unique().tolist()) == [3, 5]
    assert len([column for column in result if column.startswith("SA_")]) == 9
