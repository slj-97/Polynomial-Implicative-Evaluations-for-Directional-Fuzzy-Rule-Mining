# Reproducible Experiment Framework

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

FIRM-dependent commands additionally require `pyFTS`; install the optional
dependency only in an isolated environment after reviewing its compatibility.
