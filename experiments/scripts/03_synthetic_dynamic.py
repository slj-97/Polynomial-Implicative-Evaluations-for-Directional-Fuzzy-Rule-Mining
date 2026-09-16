"""Full dynamic-synthetic protocol with separate direct and process-order estimands."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from _common import ROOT
from experiments.src.fuzzification import RobustScaler, chronological_split, triangular_memberships
from experiments.src.metrics import direct_swap_metrics, process_orientation_gap, threshold_selection_asymmetry
from experiments.src.operators import boundary_factor, configured_evaluations, operator_asymmetry_index
from experiments.src.reproducibility import load_yaml, runtime_manifest, write_dataframe, write_json
from experiments.src.synthetic import dynamic_process

def _membership_labels(data: pd.DataFrame, label_count: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    train, _, test = chronological_split(data, (0.6, 0.2, 0.2))
    memberships = triangular_memberships(RobustScaler.fit(train).transform(test), labels=label_count)
    return (
        [memberships[f"X::{i}"] for i in range(label_count)],
        [memberships[f"Y::{i}"] for i in range(label_count)],
    )


def _thresholds(common: dict[str, object]) -> list[dict[str, float]]:
    return [{"coverage": float(common["thresholds"]["coverage"]), "support": float(s), "confidence": float(c)} for s in common["thresholds"]["support"] for c in common["thresholds"]["confidence"]]


def _aggregate(x_labels: list[np.ndarray], y_labels: list[np.ndarray], lag: int, evaluation, thresholds: list[dict[str, float]], protocol: str) -> dict[str, float]:
    label_count = len(x_labels)
    if label_count != len(y_labels):
        raise ValueError("The two variables must use the same number of fuzzy labels.")
    x_items = np.repeat(np.stack(x_labels), label_count, axis=0)
    y_items = np.tile(np.stack(y_labels), (label_count, 1))
    x_now, y_future = x_items[:, :-lag], y_items[:, lag:]
    if protocol == "direct_swap":
        reverse_input, reverse_output = y_future, x_now
    else:
        reverse_input, reverse_output = y_items[:, :-lag], x_items[:, lag:]
    forward_values = evaluation(x_now, y_future)
    reverse_values = evaluation(reverse_input, reverse_output)
    forward_supports, reverse_supports = forward_values.mean(axis=1), reverse_values.mean(axis=1)
    forward_sums = x_now.sum(axis=1)
    reverse_sums = reverse_input.sum(axis=1)
    forward_confidences = np.divide(
        forward_values.sum(axis=1), forward_sums,
        out=np.full_like(forward_sums, np.nan), where=forward_sums > 0.0,
    )
    reverse_confidences = np.divide(
        reverse_values.sum(axis=1), reverse_sums,
        out=np.full_like(reverse_sums, np.nan), where=reverse_sums > 0.0,
    )
    forward_coverages, reverse_coverages = x_now.mean(axis=1), reverse_input.mean(axis=1)
    selection_metrics = threshold_selection_asymmetry(
        forward_coverages, forward_supports, forward_confidences,
        reverse_coverages, reverse_supports, reverse_confidences, thresholds,
    )
    if protocol == "direct_swap":
        delta_values = forward_supports - reverse_supports
        return {
            "support_forward": float(forward_supports.mean()), "support_reverse": float(reverse_supports.mean()),
            "support_delta": float(delta_values.mean()), "absolute_delta": float(np.abs(delta_values).mean()),
            "normalized_gap": float(np.mean(np.abs(delta_values) / (forward_supports + reverse_supports + 1e-12))),
            "empirical_absolute_asymmetry": float(np.abs(forward_values - reverse_values).mean()),
            "gamma": float((boundary_factor(x_now, y_future) * (x_now - y_future)).mean()),
            **selection_metrics,
        }
    psi_values = forward_supports - reverse_supports
    return {
        "support_process_forward": float(forward_supports.mean()), "support_process_reverse": float(reverse_supports.mean()),
        "process_gap": float(psi_values.mean()), "absolute_process_gap": float(np.abs(psi_values).mean()),
        "normalized_psi": float(np.mean(np.abs(psi_values) / (forward_supports + reverse_supports + 1e-12))),
        **selection_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a reduced protocol only for execution verification.")
    arguments = parser.parse_args()
    common = load_yaml(ROOT / "experiments/configs/common.yaml")
    config = load_yaml(ROOT / "experiments/configs/synthetic_dynamic.yaml")
    true_lags = [config["lags"][0]] if arguments.smoke else config["lags"]
    eval_lags = [config["lags"][0]] if arguments.smoke else config["lags"]
    couplings = [config["lambda"][0]] if arguments.smoke else config["lambda"]
    noises = [config["noise"][0]] if arguments.smoke else config["noise"]
    mechanisms = [config["functions"][0]] if arguments.smoke else config["functions"]
    replications = 2 if arguments.smoke else int(config["replications"])
    rows: list[dict[str, object]] = []
    thresholds = _thresholds(common)
    label_counts = [int(value) for value in config.get("label_counts", [3])]
    for independent_null in ([False, True] if config["include_independent_null"] else [False]):
        for h_true in true_lags:
            for coupling in couplings:
                for noise in noises:
                    for mechanism in mechanisms:
                        for replication in range(replications):
                            seed = int(common["seed"] + replication + 10_000 * h_true + 100 * int(coupling * 100) + int(noise * 1000) + 1_000_000 * independent_null)
                            data = dynamic_process(np.random.default_rng(seed), phi=config["phi"], rho=config["rho"], lag=h_true, coupling=coupling, noise=noise, length=config["length"], burn_in=config["burn_in"], function=mechanism, independent_null=independent_null)
                            for label_count in label_counts:
                                x_labels, y_labels = _membership_labels(data, label_count)
                                for h_eval in eval_lags:
                                    for operator, power, sigma, delta, evaluation in configured_evaluations(common):
                                        common_row = {"engine": "OUR-MAIN", "seed": seed, "replication": replication, "independent_null": independent_null, "mechanism": mechanism, "h_true": h_true, "h_eval": h_eval, "lambda": coupling, "noise": noise, "label_count": label_count, "operator": operator, "sigma": sigma, "delta": delta, "p": power, "operator_D": np.nan if delta is None else operator_asymmetry_index(delta)}
                                        direct = _aggregate(x_labels, y_labels, h_eval, evaluation, thresholds, "direct_swap")
                                        rows.append({**common_row, "protocol": "direct_swap", "support_forward": direct["support_forward"], "support_reverse": direct["support_reverse"], "Delta": direct["support_delta"], "Gamma": direct["gamma"], "identity_error": np.nan if operator != "proposed" else abs(direct["support_delta"] - float(delta) * direct["gamma"]), "Psi": np.nan, "normalized_gap": direct["normalized_gap"], "empirical_absolute_asymmetry": direct["empirical_absolute_asymmetry"], **{key: value for key, value in direct.items() if key.startswith("SA_")}})
                                        process = _aggregate(x_labels, y_labels, h_eval, evaluation, thresholds, "process_order")
                                        rows.append({**common_row, "protocol": "process_order", "support_forward": process["support_process_forward"], "support_reverse": process["support_process_reverse"], "Delta": np.nan, "Gamma": np.nan, "identity_error": np.nan, "Psi": process["process_gap"], "normalized_gap": process["normalized_psi"], "empirical_absolute_asymmetry": np.nan, **{key: value for key, value in process.items() if key.startswith("SA_")}})
    frame = pd.DataFrame(rows)
    write_dataframe(frame, "experiments/outputs/raw_results/synthetic_dynamic_raw_v2.parquet")
    proposed = frame[(frame["protocol"] == "direct_swap") & (frame["operator"] == "proposed")]
    maximum_error = float(proposed["identity_error"].max())
    if maximum_error > float(common["float_tolerance"]):
        raise RuntimeError("Direct-swap identity failed during the dynamic experiment.")
    manifest = runtime_manifest(common["seed"])
    manifest.update({"smoke": arguments.smoke, "rows": len(frame), "label_counts": label_counts, "threshold_specific_selection_columns": [column for column in frame if column.startswith("SA_")], "label_pair_aggregation": "mean over all ordered triangular-label pairs within seed/configuration", "max_identity_error": maximum_error})
    write_json(manifest, "experiments/outputs/logs/synthetic_dynamic_manifest_v2.json")
    print(f"Wrote {len(frame)} seed-level dynamic rows; maximum identity error={maximum_error:.3e}.")


if __name__ == "__main__":
    main()
