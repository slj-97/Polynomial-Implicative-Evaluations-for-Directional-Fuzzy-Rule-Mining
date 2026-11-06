# Reproducibility Guide

This repository provides the code, experimental settings, and saved results for
*Polynomial Implicative Evaluations for Directional Fuzzy Rule Mining*. Use this
guide to check the coefficient criterion, rebuild the reported tables, or rerun
the experiments.

Run all commands from the repository root. The reported runs used Python 3.12.7
on Windows 11. Install the pinned dependencies in a Python 3.12 environment.

```sh
python -m pip install -r experiments/requirements-reproducibility.txt
```

## Coefficient criterion

```sh
python -B -m pytest experiments/tests/test_affine_admissibility.py -q -p no:cacheprovider
python -B experiments/scripts/24_coefficient_test.py --a=1/2 --b=1/4 --c=0
python -B experiments/scripts/24_coefficient_test.py --a=-1/2 --b=3/2 --c=-2
```

The script accepts integers and rational strings, including exact decimals
such as `0.25`. It reports the intervals, endpoints, stationary points, and
exact values used in the decision. The tests cover empty intervals, singletons,
zero coefficients, lower degrees, and repeated stationary points.

The first example is admissible. The second passes the six boundary tests but
fails the vertex test. At `(x,y)=(22/25,49/50)`, it gives
`E_x - 1 = 269/3125000 > 0`.

## Rebuild the tables from saved results

```sh
python -B experiments/scripts/25_reproduce_tables.py
```

This command regenerates all 24 empirical table bodies (9 in the paper and 15
in the supplement) without rerunning the experiments or downloading datasets.
It writes CSV and LaTeX files to
`experiments/outputs/revision_r1_q6/tables/` and checks the displayed numbers
against [the reference table bodies](table_reference/). The output
`verification.json` records the inputs and their hashes.

## Reconstruct the benchmark inputs

The six CSV files (Iris, WDBC, Vehicle, Abalone, Magic, and Online News)
come from [FIRM-paper](https://github.com/rferper/FIRM-paper) at commit
`51706e395ae8902d826475ae7f230fe5bfd39af0`. The [source manifest](data/raw/tabular_firm/manifest.json)
lists the filenames, hashes, schemas, and dimensions. These versions include
10 predictors for WDBC and 4,174 records for Abalone.

```sh
python -B experiments/scripts/27_export_processed_inputs.py --fetch --expected experiments/outputs/revision_r1_q6/processed/processed_manifest.json --output rebuilt_inputs
```

The script fetches any missing CSV files, checks their hashes, and reconstructs
the numerical and fuzzy membership arrays from the saved 70/30 splits. The
[processed manifest](experiments/outputs/revision_r1_q6/processed/processed_manifest.json)
records the feature and item order, split hashes, scaling parameters fitted
on the 70% preprocessing portion, and array hashes. The repeated-split study
also provides the indices, fitted preprocessing, and pair scores for all 120
splits under [its results directory](experiments/outputs/revision_r1_q4/).

## Rerun the experiments

Use a separate checkout for reruns because the scripts write to the standard
output paths. Use the supplied configurations in `experiments/configs/` for
the reported settings.

### Static benchmarks and split stability

Reconstruct the benchmark inputs first, then run these commands in order.
The split-stability script uses the representative pairs selected by the
matched-rule analysis.

```sh
python -B experiments/scripts/17_tabular_directional.py
python -B experiments/scripts/18_verify_tabular.py
python -B experiments/scripts/20_matched_rule_analysis.py
python -B experiments/scripts/21_repeated_split_stability.py
```

### Synthetic experiments

```sh
python -B experiments/scripts/02_synthetic_membership.py
python -B experiments/scripts/03_synthetic_dynamic.py
python -B experiments/scripts/14_protocol_alignment_summary.py
python -B experiments/scripts/29_exchangeable_null.py
```

The exchangeable control uses fixed `(sigma, delta) = (1, 0.5)` and 50
seeds per sample size. To regenerate the dynamic null summary in Table SI,
run the two commands below after the synthetic runs. They use the supplied
historical three-label records to verify the reconstructed scores.

```sh
python -B experiments/scripts/13_dynamic_operator_increment_audit.py
python -B experiments/scripts/26_rebuild_dynamic_null_summary.py
```

### Pruning benchmarks and timing

```sh
python -B experiments/scripts/15_multivariate_pruning.py
python -B experiments/scripts/22_computational_extension.py
python -B experiments/scripts/23_summarize_computational_extension.py
python -B experiments/scripts/33_computational_followup.py --audit
python -B experiments/scripts/33_computational_followup.py --batch controlled
```

The reported computational extension uses 300 interleaved measurements from
the `controlled` batch, with each worker pinned to logical processor 0. The
follow-up checks input hashes, candidate counts, rejection counters, and
retained rules against the original runs. The original measurements and pilot
batches are available under `experiments/outputs/revision_r1_q5/` and
`experiments/outputs/computational_followup/`.

The timing scripts reuse matching worker records. To collect fresh timings,
move the relevant `workers/` directory aside in your separate checkout before
running its script.

| Run | Saved workers under `experiments/outputs/` |
|---|---|
| Original computational extension | `revision_r1_q5/workers/` |
| Reported interleaved measurements | `computational_followup/controlled/workers/` |
| Additional computational diagnostics | `revision_r2/computation/workers/` |

The follow-up also accepts `--batch pilot` and `--batch controlled_pilot`.
Move their respective worker directories aside to repeat those timings.
Use `--summarize-only` with a completed batch to regenerate its summary.
The original extension and the additional diagnostics use Windows
`peak_wset` for peak memory. Timing and memory values depend on the host
and execution conditions.

### Affine comparisons, parameter selection, and diagnostics

Reconstruct the benchmark inputs before running these commands. The analysis
script accepts `--part examples`, `affine`, `selection`, or `temporal`
to run one analysis. The computation script saves ordinary searches and
instrumented phase timings separately.

```sh
python -B experiments/scripts/30_reviewer2_analysis.py
python -B experiments/scripts/31_reviewer2_computation.py
python -B experiments/scripts/32_build_reviewer2_tables.py
```

### Checks on parameter choices and split variation

```sh
python -B experiments/scripts/34_choice_gap_audit.py
```

After reconstructing the benchmark inputs, use this script to check the
30 parameter choices against the 450 saved inner evaluations and examine the
one choice resolved by the sign convention. It also summarizes absolute
support gaps at fixed `(sigma, delta) = (1, 0.5)` across the 20 splits
per dataset. The 10th and 90th empirical percentiles describe split
variation. Outputs are saved under
[`experiments/outputs/precision_followup/`](experiments/outputs/precision_followup/).

### Export tables after a rerun

Use a separate output directory for newly computed values, especially for
tables with new timings.

```sh
python -B experiments/scripts/25_reproduce_tables.py --no-manuscript-check --output fresh_tables
```

## Table inputs and scripts

Paths in the saved-input column are relative to `experiments/outputs/`.
Scripts are in `experiments/scripts/`.

| Display | Saved result input | Experiment or reconstruction scripts |
|---|---|---|
| Main I, Supplement SIII | `raw_results/synthetic_membership_raw_v2.parquet` | `02_synthetic_membership.py` |
| Main II | `tabular/datasets.csv`, `tabular/summary.csv` | `17_tabular_directional.py`, `18_verify_tabular.py` |
| Main IV, Supplement SIV–SV | `revision_r1_q3/primary_summary.csv`, `matched_contrasts.csv`, `representative_rules.csv` | `20_matched_rule_analysis.py` |
| Main V, Supplement SVI–SVIII | `revision_r1_q4/dataset_summary.csv`, `representative_summary.csv` | `21_repeated_split_stability.py` |
| Main VI | `raw_results/pruning_multivariate.csv` | `15_multivariate_pruning.py` |
| Main VII, Supplement SIX–SX | `computational_followup/controlled/measurements.csv` | `22_computational_extension.py` for the original runs and `33_computational_followup.py` for reported interleaved, fixed-affinity measurements |
| Supplement SI | `revision_r1_q6/dynamic_null_completed.csv` | `03_synthetic_dynamic.py`, `13_dynamic_operator_increment_audit.py`, `26_rebuild_dynamic_null_summary.py` |
| Supplement SII | `tabular/summary.csv`, `summaries/selection_asymmetry_thresholds_v2.csv` | `17_tabular_directional.py`, `03_synthetic_dynamic.py`, `14_protocol_alignment_summary.py` |
| Main I (fixed-parameter null rows), Supplement SXI | `revision_r1_q7/summary.csv` | `29_exchangeable_null.py` after `02_synthetic_membership.py` |
| Main III and VIII–IX, Supplement SXII--SXIII | `revision_r2/affine_summary.csv`, `selection_*.csv`, `multiitem_summary.csv`, `temporal_summary.csv` | `30_reviewer2_analysis.py`, `32_build_reviewer2_tables.py` |
| Supplement SXIV--SXV | `revision_r2/computation/summary.csv`, `phase_diagnostics.csv` | `31_reviewer2_computation.py`, `32_build_reviewer2_tables.py` |

[SHA256SUMS.json](SHA256SUMS.json) lists the checksums of the supplied files.
