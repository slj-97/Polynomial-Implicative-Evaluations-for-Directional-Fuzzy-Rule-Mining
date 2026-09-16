from __future__ import annotations

import pandas as pd

from _common import ROOT, require_clean_real_outputs
from experiments.src.paper_export import export_digest, export_manifest
from experiments.src.plotting import save_line_figure
from experiments.src.tables import write_latex_table


def main() -> None:
    controlled = ROOT / "experiments/outputs/summaries/statistics.csv"
    pruning = ROOT / "experiments/outputs/raw_results/pruning_our.csv"
    require_clean_real_outputs([controlled, pruning])
    controlled_frame = pd.read_csv(controlled)
    pruning_frame = pd.read_csv(pruning)
    table_controlled = write_latex_table(
        controlled_frame.head(30), ROOT / "experiments/outputs/tables/Table_controlled.tex",
        caption="Controlled membership-space summaries.", label="tab:controlled-results",
    )
    table_pruning = write_latex_table(
        pruning_frame, ROOT / "experiments/outputs/tables/Table_pruning.tex",
        caption="Independent pruning benchmark.", label="tab:pruning-results",
    )
    figure_data = controlled_frame[
        (controlled_frame["control"] == "generated") & (controlled_frame["evaluation"] == "proposed")
    ].copy()
    figures = save_line_figure(
        figure_data, x="delta", y="median", group="sigma",
        title="Direct support difference by directional parameter",
        output_stem=ROOT / "experiments/outputs/figures/Fig_membership_direct_support_gap",
    )
    artifacts = {
        "Table_controlled": str(table_controlled), "Table_pruning": str(table_pruning),
        "Fig_membership_direct_support_gap_pdf": str(figures[0]),
        "Fig_membership_direct_support_gap_png": str(figures[1]),
    }
    gates = {"operator_verification": True, "synthetic_results_present": True, "real_process_registry_frozen": False}
    export_manifest(artifacts, gates)
    export_digest([
        ("The exported membership-space table and figure use the completed 50-replication grid.", "synthetic_membership_raw.parquet; smoke=false in manifest"),
        ("Pruned and exhaustive retained rule sets agree in the recorded benchmark.", "pruning_our.csv; retained_rules_match_exhaustive=true"),
        ("No real-data manuscript result is exported because the process-pair registry is not frozen.", "real_process_pairs.csv; no approved rows"),
    ])
    print("Exported guarded manuscript artifacts; no Section V text was modified.")


if __name__ == "__main__":
    main()
