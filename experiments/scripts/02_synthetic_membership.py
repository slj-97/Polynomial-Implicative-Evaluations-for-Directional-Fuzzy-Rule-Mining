from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from _common import ROOT

from experiments.src.metrics import direct_swap_metrics, fuzzy_confidence, fuzzy_coverage, threshold_selection_asymmetry
from experiments.src.operators import configured_evaluations, operator_asymmetry_index
from experiments.src.reproducibility import load_yaml, runtime_manifest, write_dataframe, write_json
from experiments.src.synthetic import membership_pairs


def threshold_grid(common: dict[str, object]) -> list[dict[str, float]]:
    return [
        {
            "coverage": float(common["thresholds"]["coverage"]),
            "support": float(support),
            "confidence": float(confidence),
        }
        for support in common["thresholds"]["support"]
        for confidence in common["thresholds"]["confidence"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run one reduced condition for a fast execution check.")
    arguments = parser.parse_args()
    common = load_yaml(ROOT / "experiments/configs/common.yaml")
    config = load_yaml(ROOT / "experiments/configs/synthetic_membership.yaml")
    thresholds = threshold_grid(common)
    kappas = [config["kappa"][0]] if arguments.smoke else config["kappa"]
    etas = [config["eta"][0]] if arguments.smoke else config["eta"]
    counts = [config["n_records"][0]] if arguments.smoke else config["n_records"]
    replications = 2 if arguments.smoke else int(config["replications"])
    controls = [None] + config["controls"]
    rows: list[dict[str, object]] = []
    for control in controls:
        for kappa in kappas:
            for eta in etas:
                for n_records in counts:
                    for replication in range(replications):
                        seed = int(
                            common["seed"]
                            + replication
                            + 1_000_000 * int(kappa * 100)
                            + 10_000 * int(eta * 1000)
                            + n_records
                            + (0 if control is None else 100)
                        )
                        antecedent, consequent = membership_pairs(
                            np.random.default_rng(seed), n_records, kappa, eta, control=control
                        )
                        for name, power, sigma, delta, evaluation in configured_evaluations(common):
                            metrics = direct_swap_metrics(antecedent, consequent, evaluation)
                            cov_forward = fuzzy_coverage(antecedent)
                            cov_reverse = fuzzy_coverage(consequent)
                            conf_forward = fuzzy_confidence(antecedent, consequent, evaluation)
                            conf_reverse = fuzzy_confidence(consequent, antecedent, evaluation)
                            selection_metrics = threshold_selection_asymmetry(
                                [cov_forward], [metrics["support_forward"]], [conf_forward],
                                [cov_reverse], [metrics["support_reverse"]], [conf_reverse], thresholds,
                            )
                            identity_error = np.nan if name != "proposed" else abs(metrics["support_delta"] - float(delta) * metrics["gamma"])
                            rows.append(
                                {
                                    "engine": "our",
                                    "protocol": "direct_swap",
                                    "control": control or "generated",
                                    "seed": seed,
                                    "replication": replication,
                                    "kappa": kappa,
                                    "eta": eta,
                                    "n_records": n_records,
                                    "evaluation": name,
                                    "power": power,
                                    "sigma": sigma,
                                    "delta": delta,
                                    "operator_D": np.nan if delta is None else operator_asymmetry_index(delta),
                                    **metrics,
                                    "identity_error": identity_error,
                                    "coverage_forward": cov_forward,
                                    "coverage_reverse": cov_reverse,
                                    "confidence_forward": conf_forward,
                                    "confidence_reverse": conf_reverse,
                                    **selection_metrics,
                                }
                            )
    frame = pd.DataFrame(rows)
    write_dataframe(frame, "experiments/outputs/raw_results/synthetic_membership_raw_v2.parquet")
    manifest = runtime_manifest(common["seed"])
    manifest.update({
        "smoke": arguments.smoke,
        "rows": len(frame),
        "threshold_specific_selection_columns": [column for column in frame if column.startswith("SA_")],
        "identity_max_error": float(frame["identity_error"].max(skipna=True)),
    })
    write_json(manifest, "experiments/outputs/logs/synthetic_membership_manifest_v2.json")
    if (frame.loc[frame["evaluation"] == "proposed", "identity_error"] > float(common["float_tolerance"])).any():
        raise RuntimeError("Exact Delta = delta Gamma verification failed.")
    print(f"Wrote {len(frame)} measured synthetic-membership rows.")


if __name__ == "__main__":
    main()
