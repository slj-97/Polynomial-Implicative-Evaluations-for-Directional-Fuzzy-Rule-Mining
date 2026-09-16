"""Stage byte-identical FIRM benchmark assets and record their provenance."""
from __future__ import annotations

import hashlib
import json
import shutil

import pandas as pd

from _common import ROOT
from experiments.src.reproducibility import load_yaml


def main() -> None:
    rows = []
    config = load_yaml(ROOT / "experiments/configs/tabular.yaml")
    for dataset in config["datasets"]:
        source = ROOT / "external/FIRM-paper/assets" / f"{dataset}.csv"
        destination = ROOT / "data/raw/tabular_firm" / source.stem / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Existing dataset differs; refusing overwrite: {destination}")
        else:
            shutil.copy2(source, destination)
        frame = pd.read_csv(destination)
        rows.append({
            "dataset": source.stem, "path": str(destination.relative_to(ROOT)),
            "sha256": digest, "rows": len(frame), "columns": list(frame.columns),
            "missing_cells": int(frame.isna().sum().sum()),
            "source": "https://github.com/rferper/FIRM-paper",
            "commit": "51706e395ae8902d826475ae7f230fe5bfd39af0",
            "version": "FIRM distributed CSV; not asserted byte-identical to UCI",
            "primary_eligible": True,
            "note": {
                "wdbc": "10 predictors; official UCI has 30 predictors.",
                "abalone": "4174 rows; official UCI has 4177 rows.",
            }.get(source.stem, "Original-source version comparison pending."),
        })
    output = ROOT / "data/raw/tabular_firm/manifest.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Verified {len(rows)} datasets; manifest: {output}")


if __name__ == "__main__":
    main()
