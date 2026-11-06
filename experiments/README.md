# Experiments

The [reproducibility guide](../REPRODUCIBILITY.md) lists the commands,
data versions, and inputs for each table. Run scripts from the repository root.

- [scripts/](scripts/) contains the experiment and table generation commands.
- [src/](src/) implements the evaluations, fuzzy memberships, support measures,
  and rule search.
- [configs/](configs/) holds the experiment settings.
- [outputs/](outputs/) contains the saved measurements and summaries.

The static benchmarks use six CSV files from the version of
[FIRM-paper](https://github.com/rferper/FIRM-paper) identified in the
[data manifest](../data/raw/tabular_firm/manifest.json). The optional FIRM
adapter uses a separate upstream checkout and requires `pyFTS`.
The reported analyses use the implementation in this repository.

In the dynamic experiments, `direct_swap` compares
`X_t -> Y_{t+h}` with `Y_{t+h} -> X_t`. The identity
`Delta = delta * Gamma` applies to this comparison.
`process_order` compares `X_t -> Y_{t+h}` with `Y_t -> X_{t+h}`
and has a separate decomposition.

For source hashes recorded in the supplied experiment manifests, see the
[publication snapshot](https://github.com/slj-97/Polynomial-Implicative-Evaluations-for-Directional-Fuzzy-Rule-Mining/tree/410d21277dc3a2f8482d0bed4e3cd6fd048c154b).
