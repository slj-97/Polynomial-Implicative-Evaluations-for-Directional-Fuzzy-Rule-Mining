from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ROOT

from experiments.src.firm_adapter import firm_available, firm_operator_functions
from experiments.src.operators import eval_minimum, eval_power, eval_product, t_lukasiewicz
from experiments.src.reproducibility import load_yaml, write_dataframe


def main() -> None:
    config = load_yaml(ROOT / "experiments/configs/common.yaml")
    available, message = firm_available()
    if not available:
        status = pd.DataFrame([{"status": "blocked", "reason": message, "engine": "firm"}])
        write_dataframe(status, "experiments/outputs/logs/firm_operator_crosscheck.csv")
        print(message)
        return

    rng = np.random.default_rng(config["seed"])
    x = rng.uniform(0.0, 1.0, 10_000)
    y = rng.uniform(0.0, 1.0, 10_000)
    firm = firm_operator_functions()
    comparisons = [
        ("product", eval_product(x, y), firm["product"](x, y), "FIRM product t-norm"),
        ("minimum", eval_minimum(x, y), firm["minimum"](x, y), "FIRM minimum t-norm"),
        ("lukasiewicz", t_lukasiewicz(x, y), firm["lukasiewicz"](x, y), "FIRM Lukasiewicz t-norm"),
        ("power_p_0.01", eval_power(x, y, 0.01), firm["power_notebook"](x, y, 0.01), "FIRM notebook power formula"),
    ]
    rows = []
    tolerance = float(config["float_tolerance"])
    for name, ours, reference, source in comparisons:
        maximum = float(np.max(np.abs(ours - reference)))
        rows.append(
            {
                "operator": name,
                "our_formula": name,
                "firm_formula": source,
                "max_difference": maximum,
                "tolerance": tolerance,
                "passed": maximum < tolerance,
                "engine": "firm",
            }
        )
    result = pd.DataFrame(rows)
    write_dataframe(result, "experiments/outputs/logs/firm_operator_crosscheck.csv")
    if not bool(result["passed"].all()):
        raise RuntimeError("A matching FIRM operator differs from the independent implementation.")
    print("FIRM operator cross-check passed.")


if __name__ == "__main__":
    main()
