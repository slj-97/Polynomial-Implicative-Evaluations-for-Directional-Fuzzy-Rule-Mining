"""Export descriptive benchmark plots and a source-linked result digest."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ROOT


def main() -> None:
    output = ROOT / "experiments/outputs/tabular"
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    verification = json.loads((output / "verification.json").read_text())
    assert verification["status"] == "passed"
    data = pd.read_csv(output / "summary.csv")
    datasets = list(pd.read_csv(output / "datasets.csv").dataset)
    data["operator"] = data.apply(lambda r: f"poly({r.sigma:g},{r.delta:g})" if r.evaluation == "proposed" else f"power({r.power:g})" if r.evaluation == "power" else r.evaluation, axis=1)
    sa_columns = [column for column in data if column.startswith("SA_")]
    for labels in [3, 5]:
        subset = data[data.label_count.eq(labels)].copy()
        gap = subset.pivot(index="operator", columns="dataset", values="absolute_delta_median").reindex(columns=datasets)
        fig, ax = plt.subplots(figsize=(7.2, 7))
        im = ax.imshow(gap.to_numpy(), aspect="auto", cmap="cividis")
        ax.set_xticks(range(len(datasets)), datasets, rotation=30, ha="right")
        ax.set_yticks(range(len(gap)), gap.index, fontsize=8)
        fig.colorbar(im, ax=ax, label="Median absolute direct-swap support difference")
        fig.tight_layout()
        for extension in ["pdf", "png"]:
            fig.savefig(output / f"support_gap_{labels}.{extension}", dpi=200)
        plt.close(fig)
        # All nine thresholds remain distinct; panels share the same [0,1] scale.
        fig, axes = plt.subplots(2, 3, figsize=(12, 10), sharex=True, sharey=True)
        for ax, dataset in zip(axes.flat, datasets):
            block = subset[subset.dataset.eq(dataset)].set_index("operator").reindex(gap.index)
            im = ax.imshow(block[sa_columns].to_numpy(), vmin=0, vmax=1, aspect="auto", cmap="cividis")
            ax.set_title(dataset)
            ax.set_xticks(range(9), [f"{s}/{c}" for s in [0.01, 0.025, 0.05] for c in [0.3, 0.5, 0.7]], rotation=90, fontsize=7)
            ax.set_yticks(range(len(gap)), gap.index, fontsize=7)
        fig.supxlabel("Support / confidence threshold (coverage = 0.01)")
        fig.tight_layout(rect=(0, 0.02, 0.9, 1))
        color_axis = fig.add_axes([0.93, 0.2, 0.015, 0.6])
        fig.colorbar(im, cax=color_axis, label="Selection asymmetry")
        for extension in ["pdf", "png"]:
            fig.savefig(output / f"selection_asymmetry_{labels}.{extension}", dpi=200)
        plt.close(fig)
    count = sum(len(pd.read_parquet(path, columns=["evaluation"])) for path in output.glob("*_pairs.parquet"))
    text = f"""# Static benchmark result digest

Six FIRM-distributed datasets; {count:,} exported pair/operator rows across
three- and five-label partitions. All 24 evaluations and nine thresholds are
retained. Training/evaluation split: 70/30, seed 20260903. Numerical predictors
only, excluding targets. These are processed reference-repository versions.

Independent direct-evaluation verification: {verification['checked_scalar_values']}
scalar comparisons; maximum absolute discrepancy {verification['max_error']:.3g}.
This is an implementation check, not a test of statistical significance.

`datasets.csv` gives sizes; `summary.csv` gives all operator summaries and
threshold-specific SA. `*_pairs.parquet` preserves pair-level evidence.
`support_gap_3.pdf` and `support_gap_5.pdf` show median absolute support gaps.
`selection_asymmetry_3.pdf` and `selection_asymmetry_5.pdf` show every threshold.
Figures are descriptive draft exports; medians are over dependent label pairs,
not independent repetitions. A zero median does not establish zero magnitude
for every pair; inspect the stored maxima and full distributions as well.
SA also reflects antecedent coverage and confidence normalization.

The current manuscript scope is controlled synthetic experiments plus these
six static benchmarks. Global Housing has been removed; TEP is not included
following the claim/evidence audit in `dataset_scope_decision.md`. Downloaded
TEP files remain unused. No new tabular pruning experiment has been completed.
Section V-C reports this descriptive static protocol and its limitations.
"""
    (output / "results_digest.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
