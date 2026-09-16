"""Real-data discovery and guarded loaders; no industrial data are synthesized."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_tabular_data(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required real dataset file does not exist: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(resolved)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    raise ValueError(f"Unsupported tabular data format: {resolved.suffix}")


def inventory_real_datasets(root: str | Path) -> pd.DataFrame:
    """Report candidate real files without inferring their semantic role."""
    base = Path(root)
    patterns = ("*.csv", "*.parquet", "*.pq", "*.dat")
    rows: list[dict[str, str]] = []
    for pattern in patterns:
        for path in sorted(base.rglob(pattern)):
            lower = str(path).lower()
            dataset = "swat" if "swat" in lower else "tep" if any(token in lower for token in ("tep", "tennessee", "eastman")) else "unknown"
            rows.append({"path": str(path), "dataset_hint": dataset, "suffix": path.suffix.lower()})
    return pd.DataFrame(rows, columns=["path", "dataset_hint", "suffix"])


def require_configured_path(value: str | None, dataset_name: str) -> Path:
    if not value:
        raise FileNotFoundError(
            f"{dataset_name} execution is blocked: configure an actual dataset path; no synthetic substitute is allowed."
        )
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{dataset_name} execution is blocked: file not found: {path}")
    return path
