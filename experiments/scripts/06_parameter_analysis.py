"""Summarise parameter effects from completed synthetic or real-data runs."""
from __future__ import annotations
import pandas as pd
from _common import ROOT
from experiments.src.reproducibility import write_dataframe

def main() -> None:
    raw = ROOT / "experiments/outputs/raw_results"
    inputs = [path for path in raw.glob("*.csv") if "synthetic" in path.name or "directional" in path.name]
    if not inputs:
        print("Parameter analysis blocked: no completed raw result files.")
        return
    frame = pd.concat([pd.read_csv(path) for path in inputs], ignore_index=True)
    measures = [name for name in ("absolute_delta", "direct_abs_delta", "absolute_process_gap") if name in frame and frame[name].notna().any()]
    groups = [name for name in ("engine", "protocol", "evaluation", "kappa", "eta", "coupling", "noise", "lag") if name in frame]
    if not measures:
        print("Parameter analysis blocked: completed files contain no directional measure.")
        return
    summary = frame.groupby(groups, dropna=False)[measures].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(filter(None, map(str, col))).rstrip("_") for col in summary.columns.to_flat_index()]
    write_dataframe(summary, ROOT / "experiments/outputs/summaries/parameter_analysis.csv")
    print(f"Parameter analysis written from {len(inputs)} measured result files.")
if __name__ == "__main__": main()
