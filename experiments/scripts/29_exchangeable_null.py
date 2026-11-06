"""Summarize the prespecified fixed-parameter exchangeable-null seed group.

Reconstructs all 150 membership samples from the original seed formula and
checks the saved supports before exporting seed-level and summary records.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.metrics import direct_swap_metrics
from experiments.src.operators import eval_poly
from experiments.src.reproducibility import load_yaml
from experiments.src.synthetic import membership_pairs


def main() -> None:
    source = ROOT / 'experiments/outputs/raw_results/synthetic_membership_raw_v2.parquet'
    output = ROOT / 'experiments/outputs/revision_r1_q7'
    output.mkdir(parents=True, exist_ok=True)
    common = load_yaml(ROOT / 'experiments/configs/common.yaml')
    config = load_yaml(ROOT / 'experiments/configs/synthetic_membership.yaml')
    sigma, delta = 1.0, 0.5
    kappa, eta = float(config['kappa'][0]), float(config['eta'][0])
    raw = pd.read_parquet(source)
    selected = raw[
        raw.evaluation.eq('proposed') & raw.control.eq('exchangeable')
        & raw.sigma.eq(sigma) & raw.delta.eq(delta)
        & raw.kappa.eq(kappa) & raw.eta.eq(eta)
    ].copy().sort_values(['n_records', 'replication'])
    assert len(selected) == len(config['n_records']) * config['replications']
    tolerance = float(common['float_tolerance'])
    errors, rows = [], []
    for n, block in selected.groupby('n_records', sort=True):
        assert len(block) == block.seed.nunique() == config['replications'] == 50
        assert sorted(block.replication.tolist()) == list(range(50))
        for r in block.itertuples():
            expected_seed = int(common['seed'] + r.replication
                                + 1_000_000 * int(kappa * 100)
                                + 10_000 * int(eta * 1000) + n + 100)
            assert r.seed == expected_seed
            a, c = membership_pairs(np.random.default_rng(expected_seed), int(n), kappa, eta,
                                    control='exchangeable')
            fresh = direct_swap_metrics(a, c, lambda x, y: eval_poly(x, y, sigma, delta))
            errors.extend(abs(fresh[key] - getattr(r, key))
                          for key in ['support_forward', 'support_reverse', 'support_delta', 'gamma'])
        values = block.support_delta
        assert np.isfinite(values).all()
        assert np.max(np.abs(values - delta * block.gamma)) <= tolerance
        rows.append(dict(n_records=int(n), seeds=len(block), sigma=sigma, delta=delta,
                         mean=values.mean(), std=values.std(ddof=1), median=values.median(),
                         q10=values.quantile(.1), q90=values.quantile(.9),
                         positive=int((values > tolerance).sum()),
                         negative=int((values < -tolerance).sum()),
                         numerical_zero=int((values.abs() <= tolerance).sum())))
    assert max(errors) <= tolerance
    columns = ['n_records', 'replication', 'seed', 'sigma', 'delta', 'kappa', 'eta',
               'support_forward', 'support_reverse', 'support_delta', 'gamma', 'identity_error']
    selected[columns].to_csv(output / 'seed_results.csv', index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'summary.csv', index=False)
    protocol = dict(source=str(source.relative_to(ROOT)),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    sigma=sigma, delta=delta, kappa=kappa, eta=eta,
                    selection='First configured kappa/eta seed group, fixed before inspection.',
                    null='Independent Uniform(0,1) coordinate draws; kappa/eta do not affect their law.',
                    n_records=config['n_records'], replications=config['replications'],
                    std_ddof=1, quantiles='linear empirical quantiles', sign_zero_tolerance=tolerance,
                    reconstructed_samples=len(selected), max_reconstruction_error=max(errors),
                    max_identity_error=float(np.max(np.abs(selected.support_delta-delta*selected.gamma))))
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
    print(f'Checked {len(selected)} reconstructed samples; maximum discrepancy {max(errors):.3e}.')


if __name__ == '__main__':
    main()
