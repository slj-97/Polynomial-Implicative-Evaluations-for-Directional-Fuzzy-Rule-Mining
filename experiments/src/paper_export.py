"""Source-trace manifests for manuscript-ready measured artifacts."""

from __future__ import annotations

from pathlib import Path

from .reproducibility import write_json


def export_manifest(artifacts: dict[str, str], gates: dict[str, bool]) -> Path:
    return write_json({"artifacts": artifacts, "gates": gates}, "experiments/outputs/paper/results_manifest.json")


def export_digest(facts: list[tuple[str, str]]) -> Path:
    lines = ["# Results Digest", ""]
    for statement, source in facts:
        lines.extend([f"- {statement}", f"  [source: {source}]"])
    output = Path("experiments/outputs/paper/results_digest.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
