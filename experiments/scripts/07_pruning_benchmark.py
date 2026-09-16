from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.fuzzification import RobustScaler, triangular_memberships
from experiments.src.operators import eval_poly, t_lukasiewicz
from experiments.src.reproducibility import load_yaml, write_dataframe
from experiments.src.rule_mining import mine_rules, pruning_ratio, retained_rule_set
from experiments.src.synthetic import dynamic_process


def main() -> None:
    config = load_yaml(ROOT / "experiments/configs/mining.yaml")
    rng = np.random.default_rng(20260903)
    data = dynamic_process(rng, phi=0.7, rho=0.5, lag=3, coupling=0.5, noise=0.25, length=600, burn_in=100, function="tanh")
    scaler = RobustScaler.fit(data.iloc[:360])
    memberships = triangular_memberships(scaler.transform(data.iloc[360:]), labels=3)
    thresholds = config["thresholds"]
    rows: list[dict[str, object]] = []
    for maximum in config["max_antecedent_sizes"]:
        outcomes = {}
        for mode in config["pruning_modes"]:
            outcomes[mode] = mine_rules(
                memberships, tnorm=t_lukasiewicz,
                evaluation=lambda a, c: eval_poly(a, c, 0.75, 0.375), thresholds=thresholds,
                max_antecedent_size=maximum, pruning_mode=mode, mtc_valid=True, protocol="our_main",
            )
        baseline = retained_rule_set(outcomes["exhaustive"])
        for mode, outcome in outcomes.items():
            if retained_rule_set(outcome) != baseline:
                raise RuntimeError(f"Pruned retained rules differ from exhaustive rules for {mode}, K={maximum}.")
            rows.append({
                "engine": "our", "protocol": "our_main", "max_antecedent_size": maximum,
                "pruning_mode": mode, "generated_candidates": outcome.generated_candidates,
                "scored_candidates": outcome.scored_candidates,
                "coverage_pruned_branches": outcome.coverage_pruned_branches,
                "support_pruned_branches": outcome.support_pruned_branches,
                "retained_rules": len(retained_rule_set(outcome)),
                "pruning_ratio": pruning_ratio(outcomes["exhaustive"], outcome),
                "retained_rules_match_exhaustive": True,
            })
    write_dataframe(pd.DataFrame(rows), "experiments/outputs/raw_results/pruning_our.csv")
    reference = pd.DataFrame([{
        "engine": "firm", "status": "inspected_only",
        "finding": "FIRM coverage Apriori plus confidence redundancy pruning is not Proposition 10 MTC support pruning.",
    }])
    write_dataframe(reference, "experiments/outputs/raw_results/pruning_firm_reference.csv")
    print("Independent pruning benchmark completed with exact retained-rule agreement.")


if __name__ == "__main__":
    main()
