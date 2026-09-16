"""Configuration, provenance, and guarded output helpers."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def output_path(relative: str | Path) -> Path:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_dataframe(frame: pd.DataFrame, relative: str | Path) -> Path:
    path = output_path(relative)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError("Result tables must use .csv or .parquet.")
    return path


def write_json(payload: dict[str, Any], relative: str | Path) -> Path:
    path = output_path(relative)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def runtime_manifest(seed: int) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
