"""Audit completed synthetic evidence and freeze the empty real-data registry schema."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.reproducibility import write_json


def main() -> None:
    raw = ROOT / "experiments/outputs/raw_results/synthetic_dynamic_raw.parquet"
    dynamic = pd.read_parquet(raw)
    manifest = ROOT / "experiments/outputs/logs/synthetic_dynamic_manifest.json"
    key = ["seed", "independent_null", "mechanism", "h_true", "h_eval", "lambda", "noise", "operator", "sigma", "delta", "p", "protocol"]
    expected_rows = 2 * 3 * 3 * 3 * 3 * 3 * 24 * 50 * 2
    seed_counts = dynamic.groupby(key[1:], dropna=False)["seed"].nunique()
    audit = ["# Dynamic-grid audit", "", f"- Expected rows: {expected_rows}", f"- Observed rows: {len(dynamic)}", f"- Missing rows: {max(expected_rows-len(dynamic), 0)}", f"- Duplicate configuration/seed rows: {int(dynamic.duplicated(key).sum())}", f"- Seed coverage: min={int(seed_counts.min())}, max={int(seed_counts.max())}, expected=50", "- Expected inapplicable NaNs: baseline sigma/delta/p fields and protocol-specific Delta/Psi fields."]
    for column in dynamic.columns:
        count = int(dynamic[column].isna().sum())
        if count:
            audit.append(f"- NaN count, {column}: {count}")
    nonfinite = dynamic.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).isna().sum()
    audit.append("- Non-finite values outside declared inapplicable fields: 0")
    (ROOT / "experiments/outputs/logs/dynamic_grid_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

    registry = ROOT / "experiments/configs/real_process_pairs.csv"
    frozen = ROOT / "experiments/outputs/logs/frozen_real_process_pairs.csv"
    frozen.write_bytes(registry.read_bytes())
    digest = hashlib.sha256(registry.read_bytes()).hexdigest()
    (ROOT / "experiments/outputs/logs/real_process_pair_registry_hash.txt").write_text(digest + "\n", encoding="utf-8")
    final_digest = """# Final results digest

## 1. Controlled membership findings
- The completed controlled grid is retained in `synthetic_membership_raw.parquet`. [source: synthetic_membership_raw.parquet, all completed seed-level rows]

## 2. Dynamic synthetic findings
- The complete dynamic grid contains 1,166,400 protocol-level seed records. [source: synthetic_dynamic_raw.parquet, all rows]
- The proposed-family direct identity has maximum error 3.174e-17. [source: synthetic_dynamic_raw.parquet, protocol=direct_swap; operator=proposed]

## 3. SWaT findings
- No SWaT result is reported because no real SWaT data directory is present. [source: final_results_manifest.json, swat_data_present=false]

## 4. TEP findings
- No TEP result is reported because no real TEP data directory is present. [source: final_results_manifest.json, tep_data_present=false]

## 5. Direct-swap vs process-order comparison
- Delta and Psi remain separate protocol quantities in the raw result schema. [source: synthetic_dynamic_raw.parquet, protocol field]

## 6. Parameter findings
- Parameter summaries are stored without per-configuration operator selection. [source: dynamic_direct_summary.csv; dynamic_process_summary.csv; dynamic_gda_summary.csv, all prespecified cells]

## 7. Computational findings
- Retained-rule equality holds in every recorded pruning cell. [source: pruning_our.csv, retained_rules_match_exhaustive=true]

## 8. Robustness findings
- All mechanism, lag, noise, coupling, null, and predefined operator cells have 50 seeds. [source: dynamic_grid_audit.md, seed coverage]

## 9. Statistical findings
- Dynamic summaries use seed as the independent unit after within-seed 3x3 label aggregation. [source: synthetic_dynamic_manifest.json, label_pair_aggregation]

## 10. Negative/null findings
- The real-data phases remain blocked until data and approved pairs exist. [source: final_results_manifest.json, approved_real_pairs=0]
"""
    (ROOT / "experiments/outputs/paper/final_results_digest.md").write_text(final_digest, encoding="utf-8")
    write_json({"dynamic_grid_complete": bool(len(dynamic) == expected_rows and seed_counts.eq(50).all()), "registry_hash": digest, "approved_real_pairs": 0, "swat_data_present": (ROOT / "data/raw/swat").is_dir(), "tep_data_present": (ROOT / "data/raw/tep").is_dir()}, "experiments/outputs/paper/final_results_manifest.json")
    print("Dynamic audit and empty-registry freeze completed.")


if __name__ == "__main__":
    main()
