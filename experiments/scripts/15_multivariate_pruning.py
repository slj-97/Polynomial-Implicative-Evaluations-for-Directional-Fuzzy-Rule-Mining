"""Reproducible multi-item pruning benchmark; preserves the original smoke data."""
from __future__ import annotations

from math import comb
import numpy as np
import pandas as pd
from _common import ROOT
from experiments.src.fuzzification import RobustScaler, triangular_memberships
from experiments.src.operators import eval_poly, t_lukasiewicz
from experiments.src.rule_mining import mine_rules, retained_rule_set
from experiments.src.reproducibility import write_dataframe, write_json, runtime_manifest


def main() -> None:
    rows = []
    for replication in range(50):
        seed = 20260903 + replication
        rng = np.random.default_rng(seed)
        shared = rng.normal(size=(600, 1))
        data = pd.DataFrame(0.5 * shared + rng.normal(size=(600, 5)), columns=list('ABCDE'))
        memberships = triangular_memberships(RobustScaler.fit(data.iloc[:360]).transform(data.iloc[360:]), labels=3)
        for maximum in (2, 4):
            for support in (0.01, 0.05):
                outcomes = {}
                for mode in ('exhaustive', 'coverage_only', 'coverage_support'):
                    outcomes[mode] = mine_rules(
                        memberships, tnorm=t_lukasiewicz,
                        evaluation=lambda a, c: eval_poly(a, c, 0.75, 0.375),
                        thresholds={'coverage': 0.01, 'support': support, 'confidence': 0.5},
                        max_antecedent_size=maximum, pruning_mode=mode, mtc_valid=True,
                        protocol='multivariate_pruning',
                    )
                baseline = outcomes['exhaustive']
                expected = 15 * sum(comb(4, k) * 3**k for k in range(1, maximum + 1))
                assert baseline.scored_candidates == expected
                for mode, outcome in outcomes.items():
                    assert retained_rule_set(outcome) == retained_rule_set(baseline)
                    rows.append(dict(engine='our', protocol='multivariate_pruning', seed=seed,
                        replication=replication, K=maximum, support_threshold=support, mode=mode,
                        generated=outcome.generated_candidates, scored=outcome.scored_candidates,
                        coverage_rejections=outcome.coverage_pruned_branches,
                        support_rejections=outcome.support_pruned_branches,
                        retained=len(retained_rule_set(outcome)), equal=True,
                        pruning_ratio=1-outcome.scored_candidates/baseline.scored_candidates))
        print(f'Completed seed {replication + 1}/50', flush=True)
    frame = pd.DataFrame(rows)
    write_dataframe(frame, 'experiments/outputs/raw_results/pruning_multivariate.csv')
    summary = frame.groupby(['K', 'support_threshold', 'mode'], as_index=False).agg(
        generated=('generated', 'median'), scored=('scored', 'median'),
        retained=('retained', 'median'), pruning_ratio=('pruning_ratio', 'median'))
    write_dataframe(summary, 'experiments/outputs/summaries/pruning_multivariate.csv')
    manifest = runtime_manifest(20260903)
    manifest.update(rows=len(frame), replications=50, variables=5, train=360, evaluation=240,
        sigma=0.75, delta=0.375, all_retained_sets_equal=bool(frame.equal.all()))
    write_json(manifest, 'experiments/outputs/logs/pruning_multivariate_manifest.json')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
