from __future__ import annotations

from _common import ROOT

from experiments.src.operators import verify_polynomial_grid
from experiments.src.reproducibility import load_yaml, runtime_manifest, write_dataframe, write_json


def main() -> None:
    config = load_yaml(ROOT / "experiments/configs/common.yaml")
    grid = config["polynomial_grid"]
    results = verify_polynomial_grid(
        grid["sigma"],
        grid["relative_delta"],
        grid_size=401,
        tolerance=float(config["float_tolerance"]),
    )
    write_dataframe(results, "experiments/outputs/logs/operator_verification.csv")
    manifest = runtime_manifest(config["seed"])
    manifest.update({"all_passed": bool(results["passed"].all()), "grid_rows": len(results)})
    write_json(manifest, "experiments/outputs/logs/operator_verification_manifest.json")
    if not bool(results["passed"].all()):
        failures = results.loc[~results["passed"]]
        raise RuntimeError(f"Polynomial operator verification failed:\n{failures.to_string(index=False)}")
    print(f"Verified {len(results)} polynomial parameter pairs.")


if __name__ == "__main__":
    main()
