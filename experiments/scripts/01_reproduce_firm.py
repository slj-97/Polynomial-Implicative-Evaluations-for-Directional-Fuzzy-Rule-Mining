from __future__ import annotations

import json

import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.firm_adapter import FIRM_ROOT, firm_available, firm_fuzzify, run_firm_aarfi_f
from experiments.src.operators import eval_power, implication_power, t_lukasiewicz
from experiments.src.reproducibility import load_yaml, write_dataframe


def main() -> None:
    available, message = firm_available()
    if not available:
        raise RuntimeError(message)
    common = load_yaml(ROOT / "experiments/configs/common.yaml")
    rows: list[dict[str, object]] = []
    for dataset_name in ("iris.csv", "wdbc.csv"):
        data = pd.read_csv(FIRM_ROOT / "assets" / dataset_name)
        numeric = data.select_dtypes(include=[np.number]).copy()
        fuzzy = firm_fuzzify(numeric, name=dataset_name)
        measures = run_firm_aarfi_f(
            numeric, fuzzy, evaluation=lambda a, c: eval_power(a, c, 0.01),
            tnorm=t_lukasiewicz, implication=lambda a, c: implication_power(a, c, 0.01),
            min_cov=0.2, min_supp=0.2, min_conf=0.7, max_feat=3,
            protocol="firm_reproduction", separable_evaluation=True,
        )
        rows.append({
            "engine": "firm", "protocol": "firm_reproduction", "dataset": dataset_name,
            "n_records": len(numeric), "n_numeric_features": numeric.shape[1],
            "n_rules": len(measures), "mean_coverage": measures["coverage"].mean() if not measures.empty else np.nan,
            "mean_support": measures["support"].mean() if not measures.empty else np.nan,
            "mean_confidence": measures["confidence"].mean() if not measures.empty else np.nan,
            "status": "smoke_passed",
        })
    frame = pd.DataFrame(rows)
    write_dataframe(frame, "experiments/outputs/reproduction/firm_reproduction.csv")
    summary = {
        "status": "FIRM smoke test completed; this is not a claim of exact notebook-table reproduction.",
        "datasets": frame.to_dict(orient="records"),
        "test4": "The original test4 notebook is mapped but not auto-executed because its multi-item replacement-swap estimand is not a main-paper protocol.",
    }
    (ROOT / "experiments/outputs/reproduction/firm_test4_summary.md").write_text(
        "# FIRM test4 reproduction status\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print("FIRM iris and wdbc smoke tests completed.")


if __name__ == "__main__":
    main()
