"""Independent depth-first fuzzy rule miner with explicit pruning modes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .metrics import fuzzy_confidence, fuzzy_coverage, fuzzy_support

Evaluation = Callable[[Any, Any], np.ndarray]
TNorm = Callable[[Any, Any], np.ndarray]


def item_variable(item: str) -> str:
    if "::" not in item:
        raise ValueError(f"Fuzzy item name must use 'variable::label': {item}")
    return item.rsplit("::", 1)[0]


def aggregate_memberships(items: tuple[str, ...], memberships: Mapping[str, np.ndarray], tnorm: TNorm) -> np.ndarray:
    if not items:
        raise ValueError("An antecedent must contain at least one item.")
    value = np.asarray(memberships[items[0]], dtype=np.float64)
    for item in items[1:]:
        value = np.asarray(tnorm(value, memberships[item]), dtype=np.float64)
    return value


@dataclass(frozen=True)
class MiningOutcome:
    rules: pd.DataFrame
    generated_candidates: int
    scored_candidates: int
    coverage_pruned_branches: int
    support_pruned_branches: int


def mine_rules(
    memberships: Mapping[str, np.ndarray],
    *,
    tnorm: TNorm,
    evaluation: Evaluation,
    thresholds: Mapping[str, float],
    max_antecedent_size: int,
    pruning_mode: str,
    mtc_valid: bool,
    protocol: str,
) -> MiningOutcome:
    """Mine singleton-consequent rules without changing FIRM's implementation.

    `coverage_support` is permitted only when MTC has been independently
    verified. Each branch has a fixed consequent, matching Proposition 10.
    """
    if pruning_mode not in {"exhaustive", "coverage_only", "coverage_support"}:
        raise ValueError(f"Unknown pruning mode: {pruning_mode}")
    if pruning_mode == "coverage_support" and not mtc_valid:
        raise ValueError("MTC support pruning requires an MTC-valid evaluation.")
    if max_antecedent_size < 1:
        raise ValueError("max_antecedent_size must be positive.")
    required = {"coverage", "support", "confidence"}
    if required - set(thresholds):
        raise ValueError("Thresholds must include coverage, support, and confidence.")

    items = sorted(memberships)
    if not items:
        raise ValueError("No fuzzy membership items were supplied.")
    lengths = {len(np.asarray(values)) for values in memberships.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("All membership vectors must have the same positive length.")

    rows: list[dict[str, object]] = []
    counters = {"generated": 0, "scored": 0, "coverage_branches": 0, "support_branches": 0}

    for consequent in items:
        candidates = [item for item in items if item_variable(item) != item_variable(consequent)]

        def visit(prefix: tuple[str, ...], start: int) -> None:
            if prefix:
                counters["generated"] += 1
                antecedent = aggregate_memberships(prefix, memberships, tnorm)
                coverage = fuzzy_coverage(antecedent)
                coverage_ok = coverage > 0.0 and coverage >= float(thresholds["coverage"])
                if pruning_mode != "exhaustive" and not coverage_ok:
                    counters["coverage_branches"] += 1
                    return

                counters["scored"] += 1
                support = fuzzy_support(antecedent, memberships[consequent], evaluation)
                support_ok = support >= float(thresholds["support"])
                if pruning_mode == "coverage_support" and not support_ok:
                    counters["support_branches"] += 1
                    return

                confidence = fuzzy_confidence(antecedent, memberships[consequent], evaluation)
                if coverage_ok and support_ok and confidence >= float(thresholds["confidence"]):
                    rows.append(
                        {
                            "antecedent": " & ".join(prefix),
                            "consequent": consequent,
                            "antecedent_size": len(prefix),
                            "coverage": coverage,
                            "support": support,
                            "confidence": confidence,
                            "engine": "our",
                            "protocol": protocol,
                            "pruning_mode": pruning_mode,
                        }
                    )
            if len(prefix) == max_antecedent_size:
                return
            used_variables = {item_variable(item) for item in prefix}
            for index in range(start, len(candidates)):
                item = candidates[index]
                if item_variable(item) in used_variables:
                    continue
                visit(prefix + (item,), index + 1)

        visit((), 0)

    rules = pd.DataFrame(rows)
    if not rules.empty:
        rules = rules.sort_values(["confidence", "support", "coverage"], ascending=False, kind="mergesort").reset_index(drop=True)
    return MiningOutcome(
        rules=rules,
        generated_candidates=counters["generated"],
        scored_candidates=counters["scored"],
        coverage_pruned_branches=counters["coverage_branches"],
        support_pruned_branches=counters["support_branches"],
    )


def retained_rule_set(outcome: MiningOutcome) -> set[tuple[str, str]]:
    if outcome.rules.empty:
        return set()
    return set(zip(outcome.rules["antecedent"], outcome.rules["consequent"], strict=True))


def pruning_ratio(exhaustive: MiningOutcome, pruned: MiningOutcome) -> float:
    if exhaustive.scored_candidates == 0:
        return 0.0
    return 1.0 - pruned.scored_candidates / exhaustive.scored_candidates
