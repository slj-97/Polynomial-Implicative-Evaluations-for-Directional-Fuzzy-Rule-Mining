"""Thin, licence-aware adapter for the read-only FIRM reference checkout."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIRM_ROOT = ROOT / "external" / "FIRM-paper"


def _load_firm() -> tuple[Any, Any, Any]:
    if not FIRM_ROOT.is_dir():
        raise FileNotFoundError(f"FIRM reference checkout is missing: {FIRM_ROOT}")
    if str(FIRM_ROOT) not in sys.path:
        sys.path.insert(0, str(FIRM_ROOT))
    try:
        fuzzy_data = importlib.import_module("FIRM.base.fuzzy_data")
        aarfi = importlib.import_module("FIRM.methods.AARFI")
        tnorms = importlib.import_module("FIRM.base.operators.tnorms")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "FIRM is unavailable because one of its optional dependencies is missing. "
            "Install the project extra needed for FIRM (notably pyFTS) in an isolated environment."
        ) from error
    return fuzzy_data, aarfi, tnorms


def firm_available() -> tuple[bool, str]:
    try:
        _load_firm()
    except (FileNotFoundError, RuntimeError) as error:
        return False, str(error)
    return True, "FIRM adapter imports succeeded."


def firm_fuzzify(data: pd.DataFrame, name: str = "firm_reference", labels: int = 3) -> Any:
    fuzzy_data, _, _ = _load_firm()
    label_names = ["L", "M", "H"] if labels == 3 else [f"Label {index}" for index in range(labels)]
    return fuzzy_data.FuzzyDataQuantiles(name, data, labels, label_names)


def firm_rule_table_to_tidy(frame: pd.DataFrame, *, protocol: str) -> pd.DataFrame:
    """Normalize upstream AARFI_F measure columns while preserving source values."""
    tidy = frame.copy()
    rename = {"fcoverage": "coverage", "fsupport": "support", "fconfidence": "confidence"}
    tidy = tidy.rename(columns=rename)
    required = {"coverage", "support", "confidence"}
    missing = required - set(tidy.columns)
    if missing:
        raise ValueError(f"FIRM result table misses expected quality columns: {sorted(missing)}")
    tidy["engine"] = "firm"
    tidy["protocol"] = protocol
    return tidy


def run_firm_aarfi(
    data: pd.DataFrame,
    fuzzy_dataset: Any,
    *,
    tnorm: Callable[[Any, Any], Any],
    implication: Callable[[Any, Any], Any],
    min_cov: float,
    min_supp: float,
    min_conf: float,
    max_feat: int,
    protocol: str,
) -> pd.DataFrame:
    _, aarfi, _ = _load_firm()
    rules = aarfi.AARFI(
        data, fuzzy_dataset, T=tnorm, I=implication, min_cov=min_cov,
        min_supp=min_supp, min_conf=min_conf, max_feat=max_feat,
    )
    rows = [
        {
            "rule": rule.sentence_rule(fuzzy_dataset),
            "lrule": tuple(rule.lrule),
            "coverage": rule.fcoverage(),
            "support": rule.fsupport(),
            "confidence": rule.fconfidence(),
            "engine": "firm",
            "protocol": protocol,
        }
        for rule in rules.rule_list
    ]
    return pd.DataFrame(rows)


def run_firm_aarfi_f(
    data: pd.DataFrame,
    fuzzy_dataset: Any,
    *,
    evaluation: Callable[[Any, Any], Any],
    tnorm: Callable[[Any, Any], Any],
    implication: Callable[[Any, Any], Any],
    min_cov: float,
    min_supp: float,
    min_conf: float,
    max_feat: int,
    protocol: str,
    separable_evaluation: bool = False,
) -> pd.DataFrame:
    _, aarfi, _ = _load_firm()
    _, measures = aarfi.AARFI_F(
        data,
        fuzzy_dataset,
        T=tnorm,
        I=implication,
        F=evaluation,
        min_cov=min_cov,
        min_supp=min_supp,
        min_conf=min_conf,
        max_feat=max_feat,
        separable_F=separable_evaluation,
    )
    return firm_rule_table_to_tidy(measures, protocol=protocol)


def _scalar_vectorized(function: Callable[[float, float], float]) -> Callable[[Any, Any], np.ndarray]:
    return lambda x, y: np.vectorize(function, otypes=[np.float64])(x, y)


def firm_operator_functions() -> dict[str, Callable[[Any, Any], np.ndarray]]:
    """Return matching FIRM definitions for numerical cross-checks."""
    _, _, tnorms = _load_firm()
    product = tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.PRODUCT)
    minimum = tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.MINIMUM)
    lukasiewicz = tnorms.TnormsExamples.get_tnorm(tnorms.TnormsExamples.LUKASIEWICZ)
    return {
        "product": _scalar_vectorized(product),
        "minimum": _scalar_vectorized(minimum),
        "lukasiewicz": _scalar_vectorized(lukasiewicz),
        "power_notebook": lambda x, y, p=0.01: np.asarray(x, dtype=np.float64) * np.power(np.asarray(y, dtype=np.float64), p),
    }
