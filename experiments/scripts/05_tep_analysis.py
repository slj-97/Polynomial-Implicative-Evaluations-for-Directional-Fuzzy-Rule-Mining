"""Run TEP only when files and approved pairs are configured."""
from __future__ import annotations
import json
import numpy as np
import yaml
from _common import ROOT
from experiments.src.datasets import inventory_real_datasets, load_tabular_data, require_configured_path
from experiments.src.directional_pairs import load_approved_process_pairs
from experiments.src.real_analysis import analyse_pairs
from experiments.src.reproducibility import load_yaml, write_dataframe

def main() -> None:
    config = yaml.safe_load((ROOT / "experiments/configs/tep.yaml").read_text())
    common = load_yaml(ROOT / "experiments/configs/common.yaml")
    (ROOT / "experiments/outputs/logs/tep_inventory.json").write_text(json.dumps(inventory_real_datasets(ROOT).to_dict(orient="records"), indent=2), encoding="utf-8")
    try:
        training = load_tabular_data(require_configured_path(config.get("training_path"), "TEP training"))
        test = load_tabular_data(require_configured_path(config.get("test_path"), "TEP test"))
        pairs = load_approved_process_pairs(ROOT / "experiments/configs/real_process_pairs.csv", "tep")
    except (FileNotFoundError, ValueError) as error:
        print(f"TEP blocked: {error}")
        return
    proportions = tuple(float(value) for value in config["normal_test_split"])
    if len(proportions) != 2 or not np.isclose(sum(proportions), 1.0):
        raise ValueError("TEP normal_test_split must contain two values summing to one.")
    split = int(len(test) * proportions[0])
    if split == 0 or split >= len(test):
        raise ValueError("TEP normal_test_split yields an empty partition.")
    final_evaluation = test.iloc[split:].copy()
    write_dataframe(
        analyse_pairs(training, final_evaluation, pairs, common=common),
        ROOT / "experiments/outputs/raw_results/tep_directional_v2.parquet",
    )
    print("TEP analysis completed from supplied measurements.")
if __name__ == "__main__": main()
