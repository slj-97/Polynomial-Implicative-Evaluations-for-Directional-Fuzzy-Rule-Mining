# Reproducibility archive for the revised TFS manuscript

Run the commands below from the repository root. This repository
contains the authors' experimental implementation, fixed configurations,
saved results used by the tables, split/preprocessing records, exact theory
examples, and checksums. It excludes third-party FIRM code and CSV payloads;
the six input versions are retrieved by pinned commit and verified by SHA-256.
No network is required for the coefficient examples or cached table generation.

## Environment and quick checks

The verified host used Python 3.12.7 on Windows 11. Package versions are pinned
in `experiments/requirements-reproducibility.txt`. Install them in a Python
3.12 environment with:

```text
python -m pip install -r experiments/requirements-reproducibility.txt
python -B -m pytest experiments/tests/test_affine_admissibility.py -q -p no:cacheprovider
python -B experiments/scripts/24_coefficient_test.py --a=1/2 --b=1/4 --c=0
python -B experiments/scripts/24_coefficient_test.py --a=-1/2 --b=3/2 --c=-2
python -B experiments/scripts/25_reproduce_tables.py
```

`24` prints every applicable polynomial, interval, distinct endpoint/interior
stationary point and exact value. Input coefficients are integers or rational
strings, including decimal strings such as `0.25`; Python float inputs to the
API are rejected. The theorem applies to real coefficients; this executable
provides exact rational-input decisions with exact algebraic stationary roots.
Empty intervals pass vacuously; singletons are evaluated once. Zero coefficients,
constants, degree reductions and repeated stationary roots are covered by tests.
The rejection example passes all six boundary polynomials but fails `U(a,b,c)`.
Its strict interior witness is `(x,y)=(22/25,49/50)`, where `E_x-1=269/3125000`.

## Route 1: rebuild tables from saved results

Run `python -B experiments/scripts/25_reproduce_tables.py` to rebuild the
tables without rerunning the experiments. This checks the saved-results-to-table
pipeline; it does not execute data generation, rule evaluation or mining.

`25` creates CSV and LaTeX table bodies in
`experiments/outputs/revision_r1_q6/tables/` and verifies their displayed
numerical cells against the reference table bodies in `table_reference/`. The extraction is
only for verification: table values are recomputed from measured result files,
not copied out of the manuscript. Captions and prose remain in the manuscript.
`verification.json` identifies and hashes every input used.

## Exact data versions and processing

The authoritative source is FIRM-paper commit
`51706e395ae8902d826475ae7f230fe5bfd39af0`, assets `iris.csv`, `wdbc.csv`,
`vehicle.csv`, `abalone.csv`, `magic.csv`, and `online_news.csv`.
`data/raw/tabular_firm/manifest.json` records source file hashes, schemas and
dimensions. These are the distributed processed versions: WDBC has 10
predictors and Abalone has 4174 records. Do not substitute another release.

```text
python -B experiments/scripts/27_export_processed_inputs.py --fetch --expected experiments/outputs/revision_r1_q6/processed/processed_manifest.json --output rebuilt_inputs
```

This retrieves missing source files at the pinned commit, verifies their
hashes, loads the saved 70/30 split indices and checks them against seed
20260903. It exports numeric train/evaluation arrays and all 12 evaluation
membership matrices (six datasets, three/five labels). The frozen processed
manifest records feature/item order, float64 array hashes, target/excluded
columns, training-only 1%/99% scaling values, active continuous columns,
binary columns and split hashes. Nonfinite predictors are rejected without
imputation. Continuous memberships use fixed equally spaced triangles after
clipped scaling; binary memberships are the two natural indicators.
The expected manifest validates exact arrays, not compressed-file timestamps.

Q4 split indices, fitted preprocessing and pair-level scores for all 120
repeated splits are in `experiments/outputs/revision_r1_q4/splits/`,
`preprocessing/` and `pairs/`. The original computational extension stores
input-membership hashes and exact retained-rule keys for all 300 workers.
Current computational-extension tables use the complete interleaved follow-up
in `experiments/outputs/computational_followup/controlled/`; original measurements
remain under `revision_r1_q5/` for comparison.

## Route 2: rerun experiments from the specified inputs

Use a separate checkout for reruns: these scripts write to the standard
result paths. First install the dependencies and run the dataset reconstruction
command above. The static benchmark and revision analyses can then be rerun in
this order, from the archive root:

```text
python -B experiments/scripts/17_tabular_directional.py
python -B experiments/scripts/18_verify_tabular.py
python -B experiments/scripts/20_matched_rule_analysis.py
python -B experiments/scripts/21_repeated_split_stability.py
```

These entry points read `common.yaml`, `tabular.yaml` and, for the repeated
splits, `revision_r1_q4.yaml` from `experiments/configs/`. They compute scores
and retention decisions from the verified CSV inputs. Q4 uses the frozen Q3
representative-rule specifications; retain their identities and settings.

The synthetic experiments generate records from the seeds and parameters in
`common.yaml`, `synthetic_membership.yaml` and `synthetic_dynamic.yaml`:

```text
python -B experiments/scripts/02_synthetic_membership.py
python -B experiments/scripts/03_synthetic_dynamic.py
python -B experiments/scripts/14_protocol_alignment_summary.py
python -B experiments/scripts/29_exchangeable_null.py
```

For the pruning experiments, run the following commands. Script `15` contains
the original fixed protocol; script `22` reads `revision_r1_q5.yaml`. Before
running `22` for fresh measurements, rename the supplied
`experiments/outputs/revision_r1_q5/workers/` directory in the separate copy
to an unused backup name. Otherwise, `22` reuses its saved worker results.

```text
python -B experiments/scripts/15_multivariate_pruning.py
python -B experiments/scripts/22_computational_extension.py
python -B experiments/scripts/23_summarize_computational_extension.py
```

For the current computational-extension table measurements, also run:

```text
python -B experiments/scripts/33_computational_followup.py --audit
python -B experiments/scripts/33_computational_followup.py --batch pilot
python -B experiments/scripts/33_computational_followup.py --batch controlled_pilot
python -B experiments/scripts/33_computational_followup.py --batch controlled
```

Script `33` retains the original inputs, miner, thresholds and measurement
boundaries from `22`. The pilot contains three repetitions at seed 20260903
of all six eight-variable parameter/threshold combinations and the
ten-variable, support-threshold 0.05 case, under all three modes (63 searches).
The `controlled_pilot` contains three repetitions under all three modes of
the eight-variable, delta=0, support-threshold 0.01 case at seed 20260905.
It and `controlled` restrict each worker to logical processor 0. The latter
measures all 300 original configurations, interleaving settings
and modes with a fixed seeded shuffle within each of ten seeds. Wall and
process CPU times are recorded separately. Every run checks input hashes,
candidate counts, rejection counters and retained keys against the original
worker. The ordinary-search peak is recorded before comparison and JSON work.
Pilot repeats and ten-seed controlled measurements are summarized separately.
An interrupted unrestricted-affinity follow-up in `refresh/` preserves its
80 completed worker records and an interruption note; it is not a table input.

Script `33` resumes saved follow-up workers. For fresh timing measurements,
run in a separate archive copy and rename its supplied
`experiments/outputs/computational_followup/pilot/workers/`,
`controlled_pilot/workers/` and `controlled/workers/` directories to unused
backup names first. `--summarize-only`
rebuilds a completed batch summary without timing searches again. Hardware and
execution conditions affect timing; the follow-up does not establish the cause
of the original long runs or a runtime effect of the directional parameter.

After rerunning the relevant experiments and summaries, run `25` again to
build their table values. On a fresh timing run, use
`python -B experiments/scripts/25_reproduce_tables.py --no-manuscript-check --output fresh_tables`:
this exports all table bodies without requiring new wall times to equal the
saved manuscript. The default command in Route 1 checks the supplied results
against that manuscript and remains the submission-consistency check.
The table map below identifies additional
record-level reconstruction steps for the dynamic-null table. Those steps
use the supplied historical three-label records as verification inputs.

The extracted-archive checks cover the coefficient tests, input reconstruction
and table generation. They do not constitute a fresh end-to-end rerun of all
experiments. Runtime and memory measurements vary with hardware and load.

### Table-to-script map

All listed displays are generated by `25_reproduce_tables.py`. Paths below
are relative to `experiments/outputs/`; scripts are under `experiments/scripts/`.

| Display | Saved result input | Full experiment/reconstruction entry points |
|---|---|---|
| Main I, Supplement SIII | `raw_results/synthetic_membership_raw_v2.parquet` | `02_synthetic_membership.py` |
| Main II–III | `tabular/datasets.csv`, `tabular/summary.csv` | `17_tabular_directional.py`, `18_verify_tabular.py` |
| Main IV, Supplement SIV–SV | `revision_r1_q3/primary_summary.csv`, `matched_contrasts.csv`, `representative_rules.csv` | `20_matched_rule_analysis.py` |
| Main V, Supplement SVI–SVIII | `revision_r1_q4/dataset_summary.csv`, `representative_summary.csv` | `21_repeated_split_stability.py` |
| Main VI | `raw_results/pruning_multivariate.csv` | `15_multivariate_pruning.py` |
| Main VII, Supplement SIX–SX | `computational_followup/controlled/measurements.csv` | `22_computational_extension.py` for original verification inputs; `33_computational_followup.py` for reported interleaved, fixed-affinity measurements |
| Supplement SI | `revision_r1_q6/dynamic_null_completed.csv` | `03_synthetic_dynamic.py`, `13_dynamic_operator_increment_audit.py`, `26_rebuild_dynamic_null_summary.py` |
| Supplement SII | `tabular/summary.csv`, `summaries/selection_asymmetry_thresholds_v2.csv` | `17_tabular_directional.py`, `03_synthetic_dynamic.py`, `14_protocol_alignment_summary.py` |
| Main I (fixed-parameter null rows), Supplement SXI | `revision_r1_q7/summary.csv` | `29_exchangeable_null.py` after `02_synthetic_membership.py` |
| Main III (revised), VIII--IX, Supplement SXII--SXIII | `revision_r2/affine_summary.csv`, `selection_*.csv`, `multiitem_summary.csv`, `temporal_summary.csv` | `30_reviewer2_analysis.py`, `32_build_reviewer2_tables.py` |
| Supplement SXIV--SXV | `revision_r2/computation/summary.csv`, `phase_diagnostics.csv` | `31_reviewer2_computation.py`, `32_build_reviewer2_tables.py` |

### Reviewer 2 additions

After reconstructing the pinned CSV inputs, run from the archive root:

```text
python -B experiments/scripts/30_reviewer2_analysis.py
python -B experiments/scripts/31_reviewer2_computation.py
python -B experiments/scripts/32_build_reviewer2_tables.py
python -B experiments/scripts/25_reproduce_tables.py
```

Script `30` accepts `--part examples`, `affine`, `selection` or `temporal`
to execute one analysis. Its frozen protocol is `revision_r2.yaml`.
The affine analysis checks all six rational coefficient triples and 216
record-level scores. The validation study uses five outer 70/30 splits,
three inner 80/20 splits and five delta candidates; its count constraint,
tie-breaks and fallback are configured before outer evaluation. All 450
inner candidate results, 30 choices and 60 outer comparisons are supplied,
with outer indices and deterministic inner seeds. No target labels enter
the parameter-selection objective. The temporal analysis reconstructs 50
original complete trajectories for each of two process settings, checking
their saved aggregates while preserving serial dependence and reporting
all nine label pairs. The exact multi-item example separates symmetric
aggregation effects from the added antisymmetric contribution.

Script `31` resumes matching saved workers. For fresh timing measurements,
use a separate archive copy and rename its supplied
`experiments/outputs/revision_r2/computation/workers/` directory to an
unused backup name. It runs 210 ordinary searches and 42 separate
instrumented diagnostic passes. The original miner is unchanged and all
diagnostic counts and retained keys are checked against it. Phase timings
are explicitly from separate instrumented runs; ordinary timings and
peak-memory snapshots precede diagnostics. Windows is required for
`peak_wset`. Synthetic record-count cases share nested evaluation prefixes;
Iris and WDBC use training-only 70/30 preprocessing. Timing varies with
hardware/load; archived values are measurements on the documented host.

Script `32` rebuilds seven table bodies (one replaces the earlier Table III).
Script `25` calls it before verifying all **24** current empirical tables.
The archive validation checks reconstruction and internal invariants;
the full measured computational worker records remain available separately.

### Saved-choice and fixed-gap checks

After reconstructing the pinned datasets, run:

```text
python -B experiments/scripts/34_choice_gap_audit.py
```

This checks all 30 choices against the 450 saved inner evaluations. Only one
choice uses the sign convention after the smaller-magnitude preference;
`precision_followup/sign_tie_audit.csv` identifies it. A separate outer check
evaluates that single alternative sign using its frozen outer split, without
changing the original selection procedure or the reported 11/13/6 outcomes.
The original and alternative overlaps are in `sign_tie_outer_check.csv`.

The script also summarizes each of the 120 supplied pair-score files at fixed
`(sigma,delta)=(1,0.5)`. It first takes the median absolute gap over all
distinct-variable item pairs within each split, then reports the median and
10th/90th empirical percentiles across 20 splits per dataset. These descriptive
ranges share source records and are not confidence intervals. The outputs
`precision_followup/fixed_gap_by_split.csv` and `fixed_gap_summary.csv` distinguish
this variation from Table II's parameter range: its eightfold span follows
from `abs(delta)` ranging from 0.125 to 1 on fixed data. Input hashes and
package versions are recorded in `precision_followup/manifest.json`.

Script `29` fixes `(sigma,delta)=(1,0.5)` and the first configured kappa/eta
seed group, retaining 50 seeds separately at each of three sample sizes.
It reconstructs all 150 exchangeable samples from their original seeds,
checks the saved support scores, and exports seed records, mean/sample
standard deviation, median, empirical 10th/90th percentiles and sign counts.
It does not pool opposite parameter signs. Its numerical-zero tolerance is
`1e-12`; kappa and eta index seeds but do not affect the exchangeable law.

Full reruns use the matching YAML configurations, not `--smoke`. Rebuild the
pinned datasets before `17` or `21`; run `17` before `20`; retain Q3's frozen
representative pairs when running `21`. Run `03` before `14`. The historical
three-label dynamic record file is included for `13` and `26`; its values
match the three-label slice of the revised file, as recorded in the protocol
alignment manifest. `26` restores the previously unexported nonzero-delta
AR-null summary from the original seeds, checking saved supports and leaving
the original result files intact. The corrected `13` also includes the AR
null in that scope on future full runs.

The original script `22` requires Windows (`psutil.peak_wset`). Its timing is
hardware/load dependent and is not expected to reproduce bitwise. It resumes
matching saved workers; to collect new original-protocol measurements, run in
a separate copy with an empty `revision_r1_q5/workers/` directory. Current
tables use all 300 records from script `33`'s fixed-affinity `controlled/`
batch, as described above; earlier measurements remain archived separately.

`SHA256SUMS.json` at the archive root inventories all supplied files. Historical
scripts are included as implementation context; the commands and inputs above
identify the current manuscript evidence. This archive is prepared for the
revision submission; it does not assert that an external repository is live.
