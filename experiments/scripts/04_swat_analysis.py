"""Run SWaT only when files and approved pairs are configured."""
from __future__ import annotations
import json
from pathlib import Path
import yaml
from _common import ROOT
from experiments.src.datasets import inventory_real_datasets, load_tabular_data, require_configured_path
from experiments.src.directional_pairs import load_approved_process_pairs
from experiments.src.fuzzification import chronological_split
from experiments.src.real_analysis import analyse_pairs
from experiments.src.reproducibility import load_yaml, write_dataframe

def main() -> None:
    config = yaml.safe_load((ROOT / "experiments/configs/swat.yaml").read_text())
    common = load_yaml(ROOT / "experiments/configs/common.yaml")
    log = ROOT / "experiments/outputs/logs/swat_inventory.json"
    log.write_text(json.dumps(inventory_real_datasets(ROOT).to_dict(orient="records"), indent=2), encoding="utf-8")
    try:
        data = load_tabular_data(require_configured_path(config.get("data_path"), "SWaT"))
        pairs = load_approved_process_pairs(ROOT / "experiments/configs/real_process_pairs.csv", "swat")
    except (FileNotFoundError, ValueError) as error:
        print(f"SWaT blocked: {error}")
        return
    train, _, test = chronological_split(data, tuple(config["chronological_split"]))
    write_dataframe(
        analyse_pairs(train, test, pairs, common=common),
        ROOT / "experiments/outputs/raw_results/swat_directional_v2.parquet",
    )
    print("SWaT analysis completed from supplied measurements.")
if __name__ == "__main__": main()
