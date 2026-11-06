# Reproducible Experiment Framework

See the [reproducibility guide](../REPRODUCIBILITY.md) for setup and commands
to rebuild the tables or rerun the experiments.

This directory contains the independent experimental implementation for
*Polynomial Implicative Evaluations for Directional Fuzzy Rule Mining*.

## Reference implementation

The read-only reference checkout is `../external/FIRM-paper/`:

- Repository: https://github.com/rferper/FIRM-paper
- Pinned commit: `51706e395ae8902d826475ae7f230fe5bfd39af0`
- Paper: Fernandez-Peralta, ``Fuzzy Implicative Rules: A Unified Approach,''
  *IEEE Transactions on Fuzzy Systems*, 2026.
- Licence status: no explicit licence was found in the audited checkout; see
  `outputs/logs/firm_license_audit.md`.

No FIRM source file is copied or modified. `src/firm_adapter.py` may import the
local reference checkout for smoke tests and controlled comparisons only.

## Provenance

- `firm_reproduction`: unmodified upstream fuzzification and mining settings.
- `firm_compatible`: the FIRM mining path under a controlled operator comparison.
- `our_main`: training-only robust scaling, fixed triangular memberships, and
  the manuscript's predefined thresholds.
- `direct_swap`: compares `X_t -> Y_{t+h}` with `Y_{t+h} -> X_t`; the
  polynomial identity `Delta = delta Gamma` applies here.
- `process_order`: compares `X_t -> Y_{t+h}` with `Y_t -> X_{t+h}`; its
  process-orientation gap is a distinct quantity and is not interpreted with
  the direct-swap identity.

Every result table must contain `engine` and `protocol` columns. The current
manuscript uses controlled synthetic experiments and six FIRM-distributed
tabular benchmarks. Global Housing is removed and TEP is not included; see
`outputs/tabular/dataset_scope_decision.md`. The historical SWaT/TEP entry
points remain data-gated for possible future work and are outside the active
manuscript protocol.

## Commands

Run scripts from the repository root, for example:

```powershell
python experiments/scripts/00_verify_operators.py
python experiments/scripts/02_synthetic_membership.py
python experiments/scripts/03_synthetic_dynamic.py
python experiments/scripts/14_protocol_alignment_summary.py
```

The revised synthetic runs preserve all nine threshold-specific selection
decisions. Dynamic results include the prespecified three- and five-label
partitions. Revised raw files and manifests use the `_v2` suffix; the frozen
pre-revision artifacts are retained for regression checks.

## Matched retention analysis for Reviewer 1 Q3

Run `python experiments/scripts/20_matched_rule_analysis.py` to compare the
matched negative, zero and positive directional parameters at fixed sigma.
The script reads the frozen `outputs/tabular/` pair exports, validates their
data and split provenance, reconstructs the original threshold decisions,
and writes counts, overlaps, state transitions, representative rules and
descriptive figures to `outputs/revision_r1_q3/`. Original results remain
unchanged. The analysis uses the original single split per dataset.

FIRM-dependent commands additionally require `pyFTS`; install the optional
dependency only in an isolated environment after reviewing its compatibility.

## Repeated-split stability for Reviewer 1 Q4

Run `python -B experiments/scripts/21_repeated_split_stability.py` using the
frozen settings in `configs/revision_r1_q4.yaml`. The script repeats the
original training-only preprocessing over 20 specified 70/30 splits, compares
the three settings at sigma=1, and tracks the four fixed Q3 examples at their
original configurations. Outputs in `outputs/revision_r1_q4/` include split
indices, preprocessing parameters, pair scores, metrics, state transitions,
example frequencies, empirical quantiles, provenance and descriptive figures.
The original seed is checked against the frozen Q3/source results. These
shared-data resplits are summarized descriptively, without inference tests.

## Original computational extension for Reviewer 1 Q5

Run `python -B experiments/scripts/22_computational_extension.py`, then
`python -B experiments/scripts/23_summarize_computational_extension.py`.
The fixed `configs/revision_r1_q5.yaml` specifies ten seeds, 5/8/10 variables,
three labels and maximum antecedent size four. At eight variables, matched
negative, zero and positive directional parameters supplement the scale study.
The positive eight-variable runs are shared, for 100 paired configurations
and 300 searches. Existing worker records are reused after configuration and
miner-source hash checks, allowing an interrupted run to resume.

Exhaustive, coverage-only and joint searches run sequentially in fresh
processes, with cyclic mode order and one BLAS thread. Search time includes
returned-rule construction and sorting but excludes startup and preprocessing.
Peak memory uses the Windows process lifetime peak working set through search
completion; it includes interpreter, preprocessing, input and returned output.
Pre-search memory is recorded separately. Windows and `psutil` are required
for this memory protocol. Outputs in `outputs/revision_r1_q5/` include each
worker's retained keys, exact cross-mode comparisons, historical count checks,
per-search measurements, descriptive summaries, provenance and figures.

The current timing tables use the complete interleaved follow-up collected by
`33_computational_followup.py --batch controlled`; the original measurements above
remain available for verification. The three-repeat diagnostic is generated
with `--batch pilot`; a separate `--batch controlled_pilot` checks fixed affinity
to logical processor 0, used by the reported measurements. See the [reproducibility guide](../REPRODUCIBILITY.md) for the table inputs,
fresh-measurement commands and separate saved-result reconstruction route.
`34_choice_gap_audit.py` checks sign ties and fixed-parameter gap variation
using the saved inner evaluations and 120 repeated-split pair-score files.
