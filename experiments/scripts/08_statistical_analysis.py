from __future__ import annotations

import pandas as pd

from _common import ROOT, require_clean_real_outputs
from experiments.src.reproducibility import load_yaml, write_dataframe
from experiments.src.statistics import summarize_groups


def main() -> None:
    common = load_yaml(ROOT / "experiments/configs/common.yaml")
    source = ROOT / "experiments/outputs/raw_results/synthetic_membership_raw.parquet"
    require_clean_real_outputs([source])
    raw = pd.read_parquet(source)
    summary = summarize_groups(
        raw, ["control", "evaluation", "power", "sigma", "delta", "kappa", "eta", "n_records"],
        "support_delta", seed=common["seed"],
    )
    summary["engine"] = "our"
    summary["protocol"] = "direct_swap"
    write_dataframe(summary, "experiments/outputs/summaries/statistics.csv")
    print(f"Wrote {len(summary)} grouped synthetic summaries.")


if __name__ == "__main__":
    main()
